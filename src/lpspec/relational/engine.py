"""What an engine is, and everything an engine does not have to write.

`plan.py` is what an engine consumes and `sinks/tables.py` is what it produces.
This is the third side: given those two, most of an executor's surface is not
engine work at all. Sinking to an LP file, handing the model to HiGHS, and
slicing a solver's answer back onto coordinates are all written against
`ModelTables` and the label frames — never against how either was filled.

So they live here once. An engine supplies four things:

- `build(program, sources)` — bind and construct
- `_tables()` — the four frames plus the scalars
- `_variables` / `_constraints` — `(dims…, var_label)` and `(dims…, row)` per
  declaration, and `_blocks`, the contiguous run of labels each was given —
  which together are what a solution is read back through
- `_program` — the plan it built, for the dims a read-back projects to

and inherits the rest. That split is the actual claim `engines/` makes, and it
is why a second engine is a compiler and an assembler rather than a whole lane.

:func:`needs_aggregate` is here for a different reason from the sinks: it is
not work an engine is spared, it is a rule both must reach the *same* answer
to. Inverting it builds a wrong model rather than a slow one, so it has one
home.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

import polars as pl

from lpspec.errors import LpspecError
from lpspec.relational import sinks
from lpspec.relational.result import Result

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from lpspec.relational import plan


class Fragment(Protocol):
    """The part of a compiled term :func:`needs_aggregate` reads.

    Stated rather than imported: the two compilers' `TermFragment`s hold a
    lazy frame and a duckdb relation respectively, and this decision reads
    neither.
    """

    @property
    def dims(self) -> tuple[str, ...]:
        """The coordinates the fragment still carries."""

    def survives_dropping(self, dropped: set[str]) -> bool:
        """Whether one row per ``(dims…, var_label)`` still holds after *dropped* goes."""


#: Each engine passes its *own* fragment type, and `may_share` reads that
#: engine's dimension tables — so the two travel together or not at all.
F = TypeVar('F', bound=Fragment)


def needs_aggregate(
    terms: Sequence[F],
    may_share: Callable[[F, F], bool],
    *,
    projected: bool = False,
) -> bool:
    """Whether stacking *terms* can put two rows on one solver column.

    Named for the answer, not the condition: an inverted test here is a wrong
    model rather than a slow one.

    Two things can put a label twice into the stack, asked separately. A
    fragment that is not keyed already holds one twice on its own. Whether a
    *pair* can is *may_share* — the engine's own, because it reads a dimension
    table — which answers no for distinct variables and otherwise asks whether
    two fragments of one variable send a label to one **row**: for
    ``sum(f, over=line, group_by=to) - sum(f, over=line, group_by=from)``, only
    where a line's two ends are one bus. That second half is what makes the
    ordinary multi-term constraint free: reading only a fragment count says the
    aggregate is reachable for ``reserve_up + reserve_down <= p_max``, which on
    the `fleet` rungs sorts every nonzero in the model to collapse nothing.

    *projected* is what the two call sites do not share. The matrix keeps a
    fragment's dims, so keyed — one row per ``(dims…, var_label)`` — carries
    straight into ``(row, col)``. The objective keeps only ``var_label``, so it
    asks the stronger question: does the key survive losing *all* dims? It does
    exactly when ``var_label`` determines every dim the fragment still carries.
    ``p * cost`` is keyed on dims that are all the variable's own, so a column
    cannot repeat; ``y * w`` — ``y`` over buses, ``w`` over snapshots — is just
    as keyed, but ``snapshot`` arrived by broadcast, so one column holds a row
    per snapshot and their *sum* is the coefficient.

    Worth 2-4x of build time on the polars engine's matrix and little on its
    objective (#408), but the argument is the same at both call sites, so it is
    written once. What it is worth on the duckdb engine is measured in the PR
    that gave it one.
    """
    if any(not t.survives_dropping(set(t.dims) if projected else set()) for t in terms):
        return True
    return any(may_share(a, b) for i, a in enumerate(terms) for b in terms[i + 1 :])


class Engine(ABC):
    """A relational LP builder: plan in, `ModelTables` out.

    The label registries are declared here rather than in each engine because
    the read-back below is written against them. They are polars frames on both
    engines: a label frame is `(dims…, label)` and nothing about it is engine
    work — an engine that holds its labels elsewhere materialises them here,
    which is the price of not writing this file twice.
    """

    _program: plan.Program | None

    #: ``name -> (first label, how many)``. Every labelling path on either
    #: engine hands a declaration a *contiguous, dense* run of labels, so a
    #: declaration's share of a solver vector is a slice of it — which is what
    #: :meth:`_read_back` relies on instead of a join.
    _blocks: dict[str, tuple[int, int]]

    #: ``name -> rows not built``, because every term they had vanished. Empty
    #: for a model whose every declared row reached the solver. Filled by the
    #: engine, read by :meth:`omissions`.
    _omitted: dict[str, int]

    @property
    @abstractmethod
    def _variables(self) -> Mapping[str, pl.LazyFrame]:
        """Per-variable `(dims…, var_label)`. Read-only here; an engine owns the storage."""

    @property
    @abstractmethod
    def _constraints(self) -> Mapping[str, pl.LazyFrame]:
        """Per-constraint `(dims…, row)`. Read-only here; an engine owns the storage."""

    @abstractmethod
    def build(self, program: plan.Program, sources: Mapping[str, Any]) -> None:
        """Bind *sources* and build every declaration. Raises rather than half-building."""

    @abstractmethod
    def _tables(self) -> sinks.ModelTables:
        """The built model as `cols`, `obj`, `rows`, `matrix` plus its scalars."""

    @abstractmethod
    def close(self) -> None:
        """Drop the built model. Optional for a caller — see `Result`."""

    # -- sinks: written against ModelTables, so neither engine owns them ---

    def omissions(self) -> pl.DataFrame:
        """``(constraint, rows_not_built)`` — every row that lost all its terms.

        A row with no variable terms is not built (SPEC §6), and a build that
        said nothing about it would leave a declared constraint unenforced with
        no way to notice. This is that record: empty for a model whose every
        declared row reached the solver, one line per constraint otherwise.

        Counts rather than coordinates, deliberately. The label of a row that
        was not built does not exist, so naming *which* coordinates went would
        mean holding the pre-drop frame — memory proportional to the omission,
        on the path this package measures hardest. A count is enough to be
        noticed, which is the whole job.
        """
        return pl.DataFrame(
            {'constraint': list(self._omitted), 'rows_not_built': list(self._omitted.values())},
            schema={'constraint': pl.String, 'rows_not_built': pl.UInt32},
        )

    def write(self, path: str | Path) -> None:
        """Sink the built model to a file; the **suffix** picks the writer.

        ``.lp`` today, ``.mps`` planned — an unknown suffix is an error naming
        both sets. The caller names an output rather than a writer, which is
        the one place this differs from :meth:`solve`: a file's format is a
        property of the file, where which solver runs is not a property of
        anything but the call.
        """
        from pathlib import Path as _Path

        out = _Path(path)
        sinks.writer(out.suffix.lower())(self._tables(), out)

    def solve(
        self,
        batch_rows: int | None = None,
        solver_options: Mapping[str, Any] | None = None,
        solver_name: str = 'highs',
    ) -> Result:
        """Sink the built model straight into a solver and solve it.

        ``solver_name`` picks the sink — ``highs``, which ships with the
        package, or ``gurobi``, which needs the ``[gurobi]`` extra. Spelled
        the way linopy spells it, and a *caller's* choice at the call: no YAML
        file can express it, because a model means the same thing whoever
        solves it. Neither is it an *engine's* choice — which is why this lives
        here and not in either executor.

        ``solver_options`` is forwarded verbatim to that solver, the way
        linopy's is — ``{'time_limit': 60, 'mip_rel_gap': 0.01}``, and so
        named in the solver's own vocabulary. ``batch_rows`` is the hand-off
        budget in elements, and defaults to the sink's own — see
        :data:`~lpspec.relational.sinks.solvers.highs.HANDOFF_BUDGET`.
        """
        status, objective, primal, dual = sinks.solver(solver_name)(self._tables(), batch_rows, solver_options)
        _spanning(solver_name, 'primal', primal, self._tables().column_count)
        _spanning(solver_name, 'dual', dual, self._tables().row_count)
        return Result(
            _status=status,
            _objective=objective,
            _executor=self,
            _primal_values=primal,
            _dual_values=dual,
        )

    # -- read-back: a slice, and labels are frames on every engine ---------

    def _solution_frame(self, name: str, values: pl.Series | None) -> pl.LazyFrame:
        """The tidy solution of variable *name*: ``(dims…, value)``.

        A slice, never a dense array and never a join. *values* is the solver's
        column vector, held by the :class:`Result` that asks — the labels are
        the build's and shared, the values are one solve's and are not.

        **Ordered by label**, which is the order the coordinates already have:
        a label *is* row-major position in the coordinate product, so sorting
        on it hands the caller back the model's own order rather than the
        order a hash join happened to finish in. Stated rather than inherited,
        because the labels are not guaranteed to arrive sorted — a mask decides
        which rows of the product survive, not how they arrive.

        And once they *are* in label order there is nothing left to look up.
        The declaration owns a contiguous, dense run of labels (:attr:`_blocks`)
        and the solver's vector is positional in the same index, so its
        coordinates and its values line up by construction. Matching them by
        key instead cost 0.38 s against 0.10 s on `dispatch/l`, for the same
        10M rows.
        """
        assert self._program is not None
        assert values is not None, 'no solve has stored a primal'
        return self._read_back(name, self._variables[name], 'var_label', self._program.variable(name).dims, values)

    def _read_back(
        self,
        name: str,
        labels: pl.LazyFrame,
        label: str,
        dims: tuple[str, ...],
        values: pl.Series,
    ) -> pl.LazyFrame:
        """One declaration's coordinates in label order, beside its values.

        **The order is not re-established here, because it was never lost.**
        Both engines owe this method a label-ascending frame — polars because
        every labelling path produces one and two of them verify it, duckdb
        because its registry asks SQL for it on the way out. Sorting again
        moved a full copy of the coordinates, strings included, at the moment
        the solver's own model is still resident: the worst point in the
        process to allocate one.

        The slice is attached as a column rather than concatenated as a frame
        so that a length that does not match the coordinates raises instead of
        padding with nulls — though :func:`_spanning` has already refused a
        vector that does not span the model, so what is left here is the block
        bookkeeping alone.
        """
        start, height = self._blocks[name]
        return labels.select(*dims).with_columns(values.slice(start, height))

    def _primal(self, name: str, values: pl.Series | None) -> pl.DataFrame:
        return self._solution_frame(name, values).collect(engine='streaming')

    def _dual(self, name: str, values: pl.Series) -> pl.DataFrame:
        """:meth:`_solution_frame` against row labels instead of column ones.

        Ordered and sliced the same way, for the same reason — a constraint
        row's label is its position in that constraint's coordinate product.
        """
        assert self._program is not None
        dims = self._program.constraint(name).dims
        return self._read_back(name, self._constraints[name], 'row', dims, values).collect(engine='streaming')

    def _no_duals_reason(self, termination_condition: str) -> str:
        """Why a solve that *did* leave values still has no duals.

        Integrality is decidable from the program, and naming the variable is
        actionable where "the solver reported none" is not.
        """
        assert self._program is not None
        discrete = sorted(v.name for v in self._program.variables if v.variable_type != 'continuous')
        if discrete:
            names = ', '.join(f"'{n}'" for n in discrete)
            return (
                f'duals are undefined for a mixed-integer model: {names} '
                f'{"is" if len(discrete) == 1 else "are"} not continuous. '
                f'Drop the integrality to price the LP relaxation instead.'
            )
        return (
            f'the solver returned no dual solution, though the solve terminated '
            f'{termination_condition!r}. Duals come from a simplex basis, which a '
            f'run stopped short of one does not have.'
        )

    def _solution_to_parquet(self, directory: Path, values: pl.Series | None) -> dict[str, Path]:
        assert self._program is not None
        directory.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for v in self._program.variables:
            out = directory / f'{v.name}.parquet'
            self._solution_frame(v.name, values).sink_parquet(out)
            written[v.name] = out
        return written

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False


def _spanning(solver: str, quantity: str, values: pl.Series | None, expected: int) -> None:
    """Refuse a solver vector that does not span the model.

    Reading a solution back is positional, so a vector of the wrong length is
    an answer about a *different* model rather than a short answer about this
    one. Checked here, where the solver hands it over, rather than where it is
    read: the objective comes back from the solver directly, so a `Result`
    built on a broken vector would report a plausible number and only fail if
    someone asked for a coordinate.

    Here rather than in an executor because `solve` is here — the check belongs
    to the hand-off, and both engines hand off through this one.

    ``None`` is not a wrong length. A mixed-integer model has no duals at all,
    and neither does a run stopped short of a simplex basis.
    """
    if values is not None and len(values) != expected:
        raise LpspecError(
            f'{solver} returned {len(values)} {quantity} values for a model with {expected}. '
            f'Reading a solution back is positional, so a vector that does not span the model '
            f'describes a different one. This is an engine bug rather than a problem with the '
            f'model — please report it.'
        )
