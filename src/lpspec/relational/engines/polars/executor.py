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
    uncovered_constant_message,
)
from lpspec.relational import plan, sinks
from lpspec.relational.engines.polars import labels
from lpspec.relational.engines.polars.binding import BoundSources, bind
from lpspec.relational.engines.polars.compiler import PolarsCompiler, TermFragment
from lpspec.relational.result import Result

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from polars._typing import MaintainOrderJoin


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
#:
#: ``col`` is ``Int32`` — the solver's own index width, HiGHS and Gurobi both
#: being 32-bit indexed, so a count past 2^31 has no sink that could take it
#: and the strict cast raises there rather than wrapping. The cast sits where
#: the column is *produced*, inside the per-declaration streaming collect, and
#: must stay there: narrowing the stacked frame instead allocates the narrow
#: copy while the wide one is still alive, a transient visible in
#: `dispatch/l`'s peak RSS. A *label* stays ``Int64``: the arithmetic path
#: computes it as a position in the full pre-mask coordinate product, which
#: can pass 2^31 while every survivor fits.
_DTYPES = {
    'col': pl.Int32, 'row': pl.Int64,
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
        #: ``name -> rows not built``, because every term they had vanished.
        #: Empty for a model whose every declared row reached the solver.
        self._omitted: dict[str, int] = {}
        #: ``name -> (first label, how many)``.
        #: :func:`~lpspec.relational.engines.polars.labels.frame` hands a
        #: declaration a *contiguous, dense* run of labels, so a declaration's
        #: share of a solver vector is a slice of it.
        self._blocks: dict[str, tuple[int, int]] = {}
        self._cols: pl.DataFrame | None = None
        self._obj: pl.DataFrame | None = None
        self._rows: pl.DataFrame | None = None
        self._matrix: pl.DataFrame | None = None
        self._matrix_starts: Any = None
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
        The variable registry is handed to it apart from the frozen bound data
        because it is still being filled: a constraint compiled later has to
        see the variables built before it (see :class:`PolarsCompiler`).
        Declarations are then built one at a time and concatenated at the end:
        their rows are independent, which is what lets the model be four frames
        rather than a graph.

        The matrix leaves here in ``(row, col)`` order, which is what
        ``ModelTables`` promises its sinks — one reads it a row range at a
        time, the other renders a row's terms in column order. The stack
        already has that order: each share leaves sorted and owns the next run
        of rows, so ascending shares concatenate into an ascending whole, and
        a ``sort`` here would copy the model's largest frame at the peak of
        the build to reorder nothing. :func:`_in_row_order` therefore *checks*
        the claim with one linear scan rather than re-establishing it.

        That order is what :func:`_row_starts` reads the CSR index off, after
        which the ``row`` column itself is dropped: a label repeated once per
        nonzero is 8 bytes per entry no sink reads.
        """

        self._program = program
        self._bound = bind(program, sources)
        self._compiler = PolarsCompiler(program, self._bound, self._variables)

        cols = [self._build_variable(v) for v in program.variables]
        built = [self._build_constraint(c) for c in program.constraints]
        objective = self._build_objective(program.objective)

        self._cols = _stack(cols, _COLS)
        self._rows = _stack([r for r, _ in built], _ROWS)
        ordered = _in_row_order(_stack([m for _, m in built if m is not None], _MATRIX))
        self._matrix_starts = _row_starts(ordered, self._n_rows)
        self._matrix = ordered.select('col', 'coeff').rechunk()
        self._obj = _stack([objective] if objective is not None else [], _OBJ)

    @property
    def _q(self) -> PolarsCompiler:
        assert self._compiler is not None, 'build() has not run'
        return self._compiler

    def _refuse_undefined_divisors(self, stacked: pl.DataFrame, name: str, *expressions: plan.Expression) -> None:
        """A null coefficient means a divisor had no value where the model divided.

        A quotient joins its divisor with a left join (:func:`_join_mul`), so a
        missing value leaves a null — and if the row was masked out, or the
        numerator variable does not exist there, the term never gets this far
        and there is nothing to report. That is what keeps the refusal from
        becoming a wall: sparse data is the ordinary case, and the question is
        not whether a divisor is dense but whether it is defined wherever the
        model actually divides by it. ``sum`` reads a null as a zero, which is
        why the question is put to the stack before any cell collapses.
        """
        undefined = int(stacked.get_column('coeff').null_count())
        if undefined:
            params = sorted(plan.divisor_parameters(*expressions))
            raise DataError(f'{name}: {sparse_divisor_message(", ".join(params), undefined)}')

    def _matrix_share(self, pieces: list[pl.LazyFrame], name: str, *expressions: plan.Expression) -> pl.DataFrame:
        """One constraint's share: in ``(row, col)`` order, repeated cells summed.

        Nothing runs unconditionally except three linear probes — the null
        count, whether the stack already arrives in order, whether any cell
        repeats. The stack usually is in order (each piece leaves the
        label-ordered row frame through joins that preserve it in practice)
        and usually repeats nothing (a cell repeats only when one variable
        reaches a row twice), so the sort and the aggregate run only when a
        probe says they would change something. At 10M entries they cost 5 ms
        where the unconditional hash aggregate costs 325 ms. The answer is
        read off the data rather than reasoned from the declarations — the
        static machinery that made this call is what #520 removed, and
        nothing here has to know *why* a cell repeats.

        **That 5 ms is why the share is rechunked first.** A streaming collect
        returns its morsels as chunks — 104 of them for `dispatch/l`'s share —
        and ``shift(1)`` pays at every boundary where ``is_sorted`` does not,
        so probing the share as it arrives costs 28 ms against 12 ms for the
        rechunk and the probe together (#576). The rechunk is not an extra
        copy: the assembly needs a contiguous matrix anyway (#550) and this
        moves that one earlier. The objective's stack is deliberately left
        fragmented — measured, and it costs peak for no wall.
        """
        stacked = pl.concat(pieces).collect(engine='streaming').rechunk()
        self._refuse_undefined_divisors(stacked, name, *expressions)
        row, col = pl.col('row'), pl.col('col')
        tied, ahead = row == row.shift(1), row > row.shift(1)
        repeat = tied & (col == col.shift(1))
        probes = stacked.select(
            (ahead | (tied & (col >= col.shift(1)))).all().alias('#ordered'),
            repeat.any().alias('#repeated'),
        )
        ordered, repeated = probes.row(0)
        if not ordered:
            stacked = stacked.sort('row', 'col')
            repeated = stacked.select(repeat.any()).item()
        if not repeated:
            return stacked
        matrix = stacked.lazy().group_by('row', 'col').agg(pl.col('coeff').sum()).sort('row', 'col')
        return matrix.collect(engine='streaming')

    # ------------------------------------------------------------------
    # declarations
    # ------------------------------------------------------------------

    def _build_variable(self, v: plan.VariableDeclaration) -> pl.DataFrame:
        """One variable's labelled frame, and its share of ``cols``.

        The share leaves in label order, because ``cols`` carries no ``col``
        of its own: a row's *position* is its solver column index. The bounds
        joins usually keep the labelled frame's order, so the order is
        verified with one linear scan and re-established only when a join
        actually lost it (:func:`labels.in_position_order`). Only the label
        and the two bounds are collected: projecting before the collect keeps
        the dim columns and the joined bound parameters inside the lazy
        pipeline instead of materialising them to be dropped. The label is
        then projected away too, having been the order's witness and nothing
        else — ``cols`` is the three columns a sink reads.
        """

        start = self._n_cols
        labelled, self._n_cols = labels.frame(self._q, v.dims, v.where, 'var_label', start)
        self._variables[v.name] = labelled.lazy()
        self._blocks[v.name] = (start, labelled.height)

        bounded = labels.in_position_order(
            self._q.bounds(labelled.lazy(), v)
            .select('var_label', pl.col('lb').cast(pl.Float64), pl.col('ub').cast(pl.Float64))
            .collect(engine='streaming'),
            'var_label',
        )
        cols = bounded.select('lb', 'ub', pl.lit(v.variable_type, dtype=_DTYPES['vtype']).alias('vtype'))

        bad = cols.filter(pl.col('lb').is_null() | pl.col('ub').is_null()).height
        if bad:
            raise DataError(null_bounds_message(v.name, bad))
        return cols

    def _build_constraint(self, c: plan.ConstraintDeclaration) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        """One constraint as its ``rows`` and its share of the matrix.

        Terms normalise to the left, constants to the right. Each constant
        fragment is aggregated to its own coordinates and left-joined, so a
        coordinate it has no row for contributes zero.

        Duplicates from ``Sum`` and ``GroupSum`` — which project rather than
        aggregate — and from ``x + 2 * x`` collapse in the terminal
        ``SUM(coeff) GROUP BY row, col`` of :meth:`_matrix_share`, which runs
        only when a linear probe over the stacked share finds a repeated
        cell. Whether a *particular* constraint could repeat one used to be
        reasoned statically from how each fragment was reshaped; the answer
        is read off the data instead, which is what #520 removed.

        The share leaves ordered by ``(row, col)``, which every sink reads it
        as: a row range at a time, entries ascending within the row.

        The labelled frame is kept for the dual read-back, and its block is
        narrowed when rows go termless: a row whose every term vanished is not
        built, so the run of labels this declaration owns is what survived
        rather than what it declared (#561).
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
        self._constraints[c.name] = frame
        self._blocks[c.name] = (start, labelled.height)

        accumulated = pl.lit(0.0, dtype=pl.Float64)
        uncovered: pl.Expr | None = None
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
            gap = pl.col(column).is_null()
            uncovered = gap if uncovered is None else uncovered | gap

        if uncovered is not None:
            gaps = int(carrier.select(uncovered.sum()).collect(engine='streaming').item())
            if gaps:
                names = ', '.join(sorted(plan.parameters_of(c.lhs, c.rhs)))
                raise DataError(uncovered_constant_message(names, gaps, f"constraint '{c.name}'"))

        rows = carrier.select(
            'row',
            pl.lit(c.sense, dtype=pl.String).alias('sense'),
            accumulated.cast(pl.Float64).alias('rhs'),
        ).collect(engine='streaming')

        if not terms:
            self._omitted[c.name] = rows.height
            self._blocks[c.name] = (start, 0)
            self._n_rows = start
            self._constraints[c.name] = self._constraints[c.name].clear()
            return rows.clear(), None

        pieces = []
        carried_order: MaintainOrderJoin | None = 'left_right' if len(terms) == 1 else None
        for p, sign in terms:
            placed = (
                frame.join(p.frame, on=list(p.dims), how='inner', maintain_order=carried_order)
                if p.dims
                else frame.join(p.frame, how='cross')
            )
            pieces.append(
                placed.select(
                    'row',
                    pl.col('var_label').cast(_DTYPES['col']).alias('col'),
                    (sign * pl.col('coeff')).cast(pl.Float64).alias('coeff'),
                )
            )
        matrix = self._matrix_share(pieces, f"constraint '{c.name}'", c.lhs, c.rhs)
        rows, matrix, self._n_rows = self._drop_termless_rows(c.name, rows, matrix, start)
        return rows, matrix

    def _drop_termless_rows(
        self, name: str, rows: pl.DataFrame, matrix: pl.DataFrame, start: int
    ) -> tuple[pl.DataFrame, pl.DataFrame, int]:
        """Rows that kept no variable term are not built, and the block closes up.

        A row with no variables is not a constraint — it asserts something about
        constants, which the solver cannot act on. Three different provenances
        reach that shape (an absent variable, an empty reduction, a missing
        coefficient) and the language used to answer them differently, so the
        same row meant different things depending on how it emptied. This is the
        rule at the level the property lives at.

        Labels are dense, and the dual read-back reads a block by position, so a
        dropped row cannot leave a gap: the survivors are renumbered from
        *start* and the row counter rewinds to match.

        Costs one `n_unique` on a frame already in memory when nothing is
        dropped, which is every correct model.
        """
        kept = matrix.get_column('row').unique()
        if kept.len() == rows.height:
            return rows, matrix, start + rows.height

        surviving = rows.filter(pl.col('row').is_in(kept)).sort('row')
        renumber = surviving.select('row').with_row_index('__new__', offset=start)
        self._omitted[name] = rows.height - surviving.height
        self._blocks[name] = (start, surviving.height)
        remap = dict(zip(renumber.get_column('row'), renumber.get_column('__new__'), strict=True))
        rows = surviving.with_columns(pl.col('row').replace_strict(remap))
        matrix = matrix.with_columns(pl.col('row').replace_strict(remap))
        self._constraints[name] = (
            self._constraints[name].filter(pl.col('row').is_in(kept)).with_columns(pl.col('row').replace_strict(remap))
        )
        return rows, matrix, start + surviving.height

    def _build_objective(self, o: plan.ObjectiveDeclaration) -> pl.DataFrame | None:
        """The objective as ``(col, coeff)``, or ``None`` if it has no terms.

        This projection drops the dims, so a dim that arrived by broadcast puts
        several rows on one column and their **sum** is the coefficient.
        Nothing downstream computes it — the hand-off scatters with
        ``dense[at] = values``, which keeps the *last* write — so the aggregate
        here is what makes the objective the one the file wrote.

        The aggregate runs only when a column actually repeats, and the probe
        is the ``n_unique`` hash count — the *only* sound probe here, because
        the stack arrives unordered and adjacency then proves nothing. Buying
        order to probe linearly was tried and is a dead end, twice over: the
        mul join's ``maintain_order`` held the label order on some shapes and
        lost it on others that differ only in data (`dispatch/l` kept it,
        `nodal` and `profiled` did not — all three the same lone masked
        ``p * cost``), so no static gate can say when the tax will pay; and
        where it was paid for nothing, the objective phase tripled
        (`profiled/l` 40 → 147 ms) while the best case had shrunk to a wash
        (69 + 6 ordered against 32 + 42 plain). ``obj`` carries no order
        contract anyway — the writer sorts it and the solver hand-off
        scatters it.
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
        pieces = [
            p.frame.select(pl.col('var_label').cast(_DTYPES['col']).alias('col'), pl.col('coeff')) for p in comp.terms
        ]
        stacked = pl.concat(pieces).collect(engine='streaming')
        self._refuse_undefined_divisors(stacked, 'objective', o.expression)
        if stacked.get_column('col').n_unique() == stacked.height:
            return stacked
        return stacked.lazy().group_by('col').agg(pl.col('coeff').sum()).collect(engine='streaming')

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
            row_starts=self._matrix_starts,
            column_count=self._n_cols,
            row_count=self._n_rows,
            objective_sense=self._obj_sense,
            objective_constant=self._obj_const,
        )

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
        """Drop the built model. Optional — see :class:`Result`.

        ``BoundSources`` is frozen, so it cannot be emptied in place the way
        the registries can: what frees the bound frames is dropping every
        reference to them, and the compiler holds one.
        """
        self._cols = self._obj = self._rows = self._matrix = self._matrix_starts = None
        self._variables.clear()
        self._constraints.clear()
        self._blocks.clear()
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


def _in_row_order(matrix: pl.DataFrame) -> pl.DataFrame:
    """*matrix* with ``row`` known to ascend — checked, not assumed.

    Every constraint orders its own share and they are stacked in declaration
    order, which is ascending row ranges, so the concatenation is ordered by
    construction. What makes it worth *checking* is :func:`_row_starts`, which
    reads one run per row off this column and scatters the lengths by value: a
    row whose entries arrived in two runs would have the first overwritten and
    the CSR spans would be silently wrong. So this is the correctness floor of
    the layout, not a speed choice.

    ``is_sorted`` is a linear scan over a column the frame already holds; the
    sort behind it is expected never to run.
    """
    if matrix.height and not matrix['row'].is_sorted():
        return matrix.sort('row')
    return matrix


def _row_starts(ordered: pl.DataFrame, row_count: int) -> Any:
    """Each row's first entry in the row-ordered *ordered* — CSR's own index.

    Run-length over the sorted column, then a scatter and a cumulative sum —
    robust to the model's shape where the obvious alternatives are not:
    ``bincount`` pays per entry (26 ms to rle's 7 at 10M entries over 100k
    rows), ``searchsorted`` per row times the log of the entries, and either
    is the wrong one on some ladder case. Computed *here* so the ``row``
    column can then be dropped. A label repeated once per nonzero is 8 bytes
    per entry no sink reads: every consumer either slices by these starts or
    asks :meth:`~lpspec.relational.sinks.tables.ModelTables.matrix_block` to
    spell the labels back out.

    The kept matrix is then **rechunked, once**. A streaming collect leaves
    it in chunks, and a sink slices it per row block — against a chunked
    frame every block's ``to_numpy`` is a gather-copy, where against one
    contiguous buffer it is a view (codspeed caught the difference as -6.9%
    on `profiled-m`, ~150 blocks over 16 chunks).
    """
    import numpy as np

    runs = ordered['row'].rle()
    starts = np.zeros(row_count + 1, dtype=np.int64)
    starts[runs.struct.field('value').to_numpy() + 1] = runs.struct.field('len').to_numpy()
    return np.cumsum(starts, out=starts)


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

    Each restriction is keyed by ``presence_dims`` where the fragment states
    them — narrower than ``dims`` for an acyclic shift, whose vacated edge
    lies along one dimension and is silent about the rest.
    """
    out: list[tuple[tuple[str, ...], pl.LazyFrame]] = []
    for p in terms:
        if p.presence is None:
            continue
        out.append((p.presence_dims or p.dims, p.presence))
    return out
