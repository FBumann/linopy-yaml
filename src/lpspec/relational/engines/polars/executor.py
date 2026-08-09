"""Polars executor: fill the model frames, then hand them to a sink.

Owns the *assembly* — turning each declaration into rows of the four model
frames, and holding them until a sink drains them. Owns none of the three
questions it asks on the way: what the data is
(:mod:`lpspec.relational.engines.polars.binding`), what a query over it looks like
(:mod:`lpspec.relational.engines.polars.compiler`), which coordinate gets which solver index
(:mod:`lpspec.relational.engines.polars.labels`). The lane is described in
docs/ARCHITECTURE.md.

The two registries it does own are the ones that fill *during* assembly — the
variable and constraint frames — because a declaration built later has to see
what earlier ones produced. Everything binding produced is frozen by contrast,
which is what :class:`~lpspec.relational.engines.polars.binding.BoundSources` says.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

import polars as pl

from lpspec.errors import (
    DataError,
    LanguageError,
    LpspecError,
    null_bounds_message,
    sparse_divisor_message,
)
from lpspec.relational import plan, sinks
from lpspec.relational.engines.polars import labels
from lpspec.relational.engines.polars.binding import BoundSources, bind
from lpspec.relational.engines.polars.compiler import PolarsCompiler, TermFragment
from lpspec.relational.result import Result

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


#: The four frames a sink reads, as schemas. Stated here because the executor
#: is what fills them and an empty model still has to have them.
_COLS = ('lb', 'ub', 'vtype')
_OBJ = ('col', 'coeff')
_ROWS = ('row', 'sense', 'rhs')
_MATRIX = ('row', 'col', 'coeff')

#: The dtype of each of those columns. ``vtype`` is an ``Enum`` over the
#: variable types the plan declares, rather than a string: it holds one word
#: per column and the same handful of words for the whole model, so as a string
#: it stores that word once per row — 0.098 GB of the ``cols`` frame's 0.333 at
#: 9.8M columns, against 0.010 as an Enum. The Enum also makes the vocabulary
#: explicit, so a fourth variable type added to
#: :data:`~lpspec.relational.plan.VariableType` and not reaching here fails
#: where the column is built rather than in whichever sink first compares
#: against a name it does not know.
_DTYPES = {
    'col': pl.Int64, 'row': pl.Int64,
    'lb': pl.Float64, 'ub': pl.Float64, 'rhs': pl.Float64, 'coeff': pl.Float64,
    'sense': pl.String, 'vtype': pl.Enum(get_args(plan.VariableType)),
}  # fmt: skip


class PolarsExecutor:
    """Build a :class:`Program` into polars frames, then sink it."""

    def __init__(self) -> None:
        self._program: plan.Program | None = None
        self._compiler: PolarsCompiler | None = None
        self._bound: BoundSources | None = None
        self._variables: dict[str, pl.LazyFrame] = {}
        self._constraints: dict[str, pl.LazyFrame] = {}
        #: ``name -> (first label, how many)``.
        #: :func:`~lpspec.relational.engines.polars.labels.frame` hands a
        #: declaration a *contiguous, dense* run of labels, so a declaration's
        #: share of a solver vector is a slice of it.
        self._blocks: dict[str, tuple[int, int]] = {}
        self._cols: pl.DataFrame | None = None
        self._obj: pl.DataFrame | None = None
        self._rows: pl.DataFrame | None = None
        self._matrix: pl.DataFrame | None = None
        self._n_cols = 0
        self._n_rows = 0
        self._obj_const = 0.0
        self._obj_sense: str = 'min'

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def build(self, program: plan.Program, sources: Mapping[str, Any]) -> None:
        """Bind *sources*, then build every declaration into the model frames.

        The compiler comes after the binding, because two of its answers are
        read off the data — a dim's size and whether a parameter is boolean.
        Declarations are then built one at a time and concatenated at the end:
        their rows are independent, which is what lets the model be four frames
        rather than a graph.
        """

        self._program = program
        self._bound = bind(program, sources)
        # The variable frames are passed apart from the bound data on purpose:
        # they are the one registry still being filled, appearing as each
        # declaration is built so a constraint compiled afterwards can see them.
        self._compiler = PolarsCompiler(program, self._bound, self._variables)

        cols = [self._build_variable(v) for v in program.variables]
        built = [self._build_constraint(c) for c in program.constraints]
        objective = self._build_objective(program.objective)

        self._cols = _stack(cols, _COLS)
        self._rows = _stack([r for r, _ in built], _ROWS)
        # `(row, col)` is what `ModelTables` promises its sinks: one reads the
        # matrix a row range at a time, the other renders a row's terms in
        # column order. Stated here, once, rather than re-established by each.
        self._matrix = _stack([m for _, m in built if m is not None], _MATRIX).sort('row', 'col')
        self._obj = _stack([objective] if objective is not None else [], _OBJ)

    @property
    def _q(self) -> PolarsCompiler:
        assert self._compiler is not None, 'build() has not run'
        return self._compiler

    def _check_no_undefined_divisor(self, name: str, matrix: pl.DataFrame, *expressions: plan.Expression) -> None:
        """A null coefficient means a divisor had no value where the model divided.

        Read off the assembled matrix rather than reasoned about from
        coordinates, because only the matrix knows which divisions *survived*.
        A quotient joins its divisor with a left join (:func:`_join_mul`), so a
        missing value leaves a null; if the row was masked out, or the numerator
        variable does not exist there, the term never reaches this frame and
        there is nothing to report.

        That is what keeps the refusal from becoming a wall. Sparse data is the
        ordinary case, and the question is not whether a divisor is dense — it
        is whether it is defined wherever the model actually divides by it.
        """
        undefined = matrix.get_column('coeff').is_null().sum()
        if undefined:
            params = sorted(plan.divisor_parameters(*expressions))
            raise DataError(f'{name}: {sparse_divisor_message(", ".join(params), int(undefined))}')

    # ------------------------------------------------------------------
    # declarations
    # ------------------------------------------------------------------

    def _build_variable(self, v: plan.VariableDeclaration) -> pl.DataFrame:
        """One variable's labelled frame, and its share of ``cols``."""

        start = self._n_cols
        labelled, self._n_cols = labels.frame(self._q, v.dims, v.where, 'var_label', start)
        self._variables[v.name] = labelled.lazy()
        self._blocks[v.name] = (start, labelled.height)

        # Sorted by the label, because `cols` carries no `col` of its own: a
        # row's *position* is its solver column index, and the bounds joins say
        # nothing about the order they hand rows back in.
        bounded = self._q.bounds(labelled.lazy(), v).sort('var_label')
        cols = bounded.select(
            pl.col('lb').cast(pl.Float64),
            pl.col('ub').cast(pl.Float64),
            pl.lit(v.variable_type, dtype=_DTYPES['vtype']).alias('vtype'),
        ).collect(engine='streaming')

        bad = cols.filter(pl.col('lb').is_null() | pl.col('ub').is_null()).height
        if bad:
            raise DataError(null_bounds_message(v.name, bad))
        return cols

    def _build_constraint(self, c: plan.ConstraintDeclaration) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        """One constraint as its ``rows`` and its share of the matrix.

        Terms normalise to the left, constants to the right. Each constant
        fragment is aggregated to its own coordinates and left-joined, so a
        coordinate it has no row for contributes zero.

        The terminal ``SUM(coeff) GROUP BY row, col`` is where duplicates from
        ``Sum`` and ``GroupSum`` — which project rather than aggregate —
        collapse, and where ``x + 2 * x`` becomes one cell. It runs on every
        constraint: whether a *particular* one can repeat a cell is a question
        about how each fragment was reshaped, and asking it is a great deal of
        machinery to save an aggregate that mostly finds nothing.

        The share leaves ordered by ``(row, col)``, which every sink reads it
        as: a row range at a time, entries ascending within the row.
        """

        lhs = self._q.expression(c.lhs, f"constraint '{c.name}' lhs")
        rhs = self._q.expression(c.rhs, f"constraint '{c.name}' rhs")
        terms = [(p, 1.0) for p in lhs.terms] + [(p, -1.0) for p in rhs.terms]
        consts = [(p, 1.0) for p in rhs.consts] + [(p, -1.0) for p in lhs.consts]
        for p, _ in [*terms, *consts]:
            extra = set(p.dims) - set(c.dims)
            if extra:
                raise LanguageError(
                    f"constraint '{c.name}': expression has dims {sorted(extra)} outside "
                    f'foreach {list(c.dims)} — missing a Sum/GroupSum?'
                )

        restrictions = _absence_restrictions([p for p, _ in terms])
        start = self._n_rows
        labelled, self._n_rows = labels.frame(self._q, c.dims, c.where, 'row', start, restrictions)
        frame = labelled.lazy()
        self._constraints[c.name] = frame  # kept for the dual read-back
        self._blocks[c.name] = (start, labelled.height)

        accumulated = pl.lit(0.0, dtype=pl.Float64)
        carrier = frame
        for i, (p, sign) in enumerate(consts):
            column = f'__const {i}__'
            aggregated = self._q.constant_scalar(p).rename({'cval': column})
            carrier = (
                carrier.join(aggregated, on=list(p.dims), how='left')
                if p.dims
                else carrier.join(aggregated, how='cross')
            )
            accumulated = accumulated + sign * pl.col(column).fill_null(0.0)

        rows = carrier.select(
            'row',
            pl.lit(c.sense, dtype=pl.String).alias('sense'),
            accumulated.cast(pl.Float64).alias('rhs'),
        ).collect(engine='streaming')

        if not terms:
            return rows, None

        pieces = []
        for p, sign in terms:
            placed = frame.join(p.frame, on=list(p.dims), how='inner') if p.dims else frame.join(p.frame, how='cross')
            pieces.append(
                placed.select(
                    'row',
                    pl.col('var_label').alias('col'),
                    (sign * pl.col('coeff')).cast(pl.Float64).alias('coeff'),
                )
            )
        # A null coefficient means an undefined divisor, and `sum` would read it
        # as a zero — so the aggregate happens after the check, not before it.
        stacked = pl.concat(pieces).collect(engine='streaming')
        self._check_no_undefined_divisor(f"constraint '{c.name}'", stacked, c.lhs, c.rhs)
        matrix = stacked.group_by('row', 'col').agg(pl.col('coeff').sum()).sort('row', 'col')
        return rows, matrix

    def _build_objective(self, o: plan.ObjectiveDeclaration) -> pl.DataFrame | None:
        """The objective as ``(col, coeff)``, or ``None`` if it has no terms.

        This projection drops the dims, so a dim that arrived by broadcast puts
        several rows on one column and their **sum** is the coefficient.
        Nothing downstream computes it — the hand-off scatters with
        ``dense[at] = values``, which keeps the *last* write — so the aggregate
        here is what makes the objective the one the file wrote.
        """

        comp = self._q.expression(o.expression, 'objective')
        for p in comp.consts:
            if p.dims:
                raise LanguageError(
                    'objective constant part has dims — wrap parameter terms in '
                    'Mul with a Var, or pre-aggregate to a scalar'
                )
            self._obj_const += p.frame.select(pl.col('cval').sum()).collect().item() or 0.0
        self._obj_sense = o.sense
        if not comp.terms:
            return None
        pieces = [p.frame.select(pl.col('var_label').alias('col'), pl.col('coeff')) for p in comp.terms]
        # A null coefficient means an undefined divisor, and `sum` would read it
        # as a zero — so the aggregate happens after the check, not before it.
        stacked = pl.concat(pieces).collect(engine='streaming')
        self._check_no_undefined_divisor('objective', stacked, o.expression)
        return stacked.group_by('col').agg(pl.col('coeff').sum())

    # ------------------------------------------------------------------
    # sinks — see relational/sinks/; the executor only supplies the frames
    # ------------------------------------------------------------------

    def _tables(self) -> sinks.ModelTables:
        assert self._cols is not None and self._obj is not None
        assert self._rows is not None and self._matrix is not None
        return sinks.ModelTables(
            cols=self._cols,
            obj=self._obj,
            rows=self._rows,
            matrix=self._matrix,
            column_count=self._n_cols,
            row_count=self._n_rows,
            objective_sense=self._obj_sense,
            objective_constant=self._obj_const,
        )

    def write(self, path: str | Path) -> None:
        """Sink the built model to a file; the **suffix** picks the writer.

        ``.lp`` today, ``.mps`` planned — an unknown suffix is an error naming
        both sets. The caller names an output rather than a writer, which is
        the one place this differs from :meth:`solve`: a file's format is a
        property of the file, where which solver runs is not a property of
        anything but the call.
        """
        path = Path(path)
        sinks.writer(path.suffix.lower())(self._tables(), path)

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
        solves it.

        ``solver_options`` is forwarded verbatim to that solver, the way
        linopy's is — ``{'time_limit': 60, 'mip_rel_gap': 0.01}``, and so
        named in the solver's own vocabulary. ``batch_rows`` is the hand-off
        budget in elements, and defaults to the sink's own — see
        :data:`~lpspec.relational.sinks.highs.HANDOFF_BUDGET`.
        """
        status, objective, primal, dual = sinks.solver(solver_name)(self._tables(), batch_rows, solver_options)
        _spanning(solver_name, 'primal', primal, self._n_cols)
        _spanning(solver_name, 'dual', dual, self._n_rows)
        return Result(
            _status=status,
            _objective=objective,
            _executor=self,
            _primal_values=primal,
            _dual_values=dual,
        )

    def _solution_frame(self, name: str, values: pl.Series | None) -> pl.LazyFrame:
        """The tidy solution of variable *name*: ``(dims…, value)``.

        A slice, never a dense array and never a join. *values* is the solver's
        column vector, held by the :class:`Result` that asks — the labels are
        the build's and shared, the values are one solve's and are not.

        **In label order**, which is the order the label frame was built in: a
        label *is* row-major position in the coordinate product, so this hands
        the caller back the model's own order rather than the order a hash join
        happened to finish in.
        """
        assert self._program is not None
        assert values is not None, 'no solve has stored a primal'
        return self._read_back(name, self._variables[name], self._program.variable(name).dims, values)

    def _read_back(
        self,
        name: str,
        coordinates: pl.LazyFrame,
        dims: tuple[str, ...],
        values: pl.Series,
    ) -> pl.LazyFrame:
        """One declaration's coordinates in label order, beside its values.

        **The order is not re-established here, because it was never lost.**
        :func:`~lpspec.relational.engines.polars.labels.frame` numbers a sorted
        frame, so it hands back a label-ascending one and this reads the
        ordering rather than imposing it.

        The declaration owns a contiguous, dense run of labels
        (:attr:`_blocks`) and the solver's vector is positional in the same
        index, so its coordinates and its values line up by construction. The
        slice is attached as a column rather than concatenated as a frame so
        that a length that does not match raises instead of padding with nulls
        — though :func:`_spanning` has already refused a vector that does not
        span the model.
        """
        start, height = self._blocks[name]
        return coordinates.select(*dims).with_columns(values.slice(start, height))

    def _primal(self, name: str, values: pl.Series | None) -> pl.DataFrame:
        return self._solution_frame(name, values).collect(engine='streaming')

    def _dual(self, name: str, values: pl.Series) -> pl.DataFrame:
        """:meth:`_solution_frame` against row labels instead of column ones.

        Ordered and sliced the same way, for the same reason — a constraint
        row's label is its position in that constraint's coordinate product.
        """
        assert self._program is not None
        dims = self._program.constraint(name).dims
        return self._read_back(name, self._constraints[name], dims, values).collect(engine='streaming')

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

    # ------------------------------------------------------------------

    def close(self) -> None:
        """Drop the built model. Optional — see :class:`Result`."""
        self._cols = self._obj = self._rows = self._matrix = None
        self._variables.clear()
        self._constraints.clear()
        self._blocks.clear()
        # `BoundSources` is frozen, so it cannot be emptied in place the way the
        # registries above can: what frees the bound frames is dropping every
        # reference to them, and the compiler holds one.
        self._bound = None
        self._compiler = None

    def __enter__(self) -> PolarsExecutor:
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


def _stack(frames: list[pl.DataFrame], columns: tuple[str, ...]) -> pl.DataFrame:
    """Concatenate *frames*, or an empty frame of *columns* when there are none.

    Named rather than inferred because a model may legitimately have nothing to
    stack, and a sink still has to find what it reads.
    """
    if frames:
        return pl.concat(frames)
    return pl.DataFrame(schema={name: _DTYPES[name] for name in columns})


def _absence_restrictions(terms: Sequence[TermFragment]) -> list[tuple[tuple[str, ...], pl.LazyFrame]]:
    """The presence frames a constraint's rows have to be contained in.

    Absence propagates into a comparison and drops the row there (v1
    ``convention.rst`` §6 and §12): ``x + y >= 10`` is not ``x >= 10`` where
    ``y`` is masked, it is no constraint at all. A term whose variable is absent
    therefore restricts the row set rather than merely contributing nothing.

    Only *variable* absence counts. A sparse parameter is a compressed dense
    array whose missing rows mean a zero coefficient (SPEC §8), which is why the
    fragment carries :attr:`~lpspec.relational.engines.polars.compiler.TermFragment.presence`
    separately from its frame, and why this reads that rather than the frame.

    A fragment with nothing to restrict is skipped: an unmasked variable exists
    at every coordinate of its foreach, so its presence would remove nothing.

    *Having* no dims is not *having nothing to restrict*: a masked scalar
    variable restricts every row of every constraint that names it, all or
    nothing. Reading the empty dims as "skip" is what let one through silently.
    """
    out: list[tuple[tuple[str, ...], pl.LazyFrame]] = []
    for p in terms:
        if p.presence is None:
            continue
        # `presence_dims` is narrower than `dims` for an acyclic shift, whose
        # vacated edge lies along one dimension and is silent about the rest.
        out.append((p.presence_dims or p.dims, p.presence))
    return out
