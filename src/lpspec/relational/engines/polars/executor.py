"""Polars executor: fill the model frames, then hand them to a sink.

Owns the *assembly* — turning each declaration into rows of the four model
frames, and holding them until a sink drains them. Owns none of the three
questions it asks on the way: what the data is
(:mod:`lpspec.relational.binding`), what a query over it looks like
(:mod:`lpspec.relational.engines.polars.compiler`), which coordinate gets which solver index
(:mod:`lpspec.relational.engines.polars.labels`). The lane is described in
docs/ARCHITECTURE.md.

The two registries it does own are the ones that fill *during* assembly — the
variable and constraint frames — because a declaration built later has to see
what earlier ones produced. Everything binding produced is frozen by contrast,
which is what :class:`~lpspec.relational.binding.BoundSources` says.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
from lpspec.relational.binding import BoundSources, bind
from lpspec.relational.engine import Engine, needs_aggregate
from lpspec.relational.engines.polars.compiler import PolarsCompiler, TermFragment
from lpspec.relational.engines.polars.labels import Labeller

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from polars._typing import MaintainOrderJoin


#: The four frames a sink reads, and their dtypes — both stated by
#: `sinks/tables.py`, which is what reads them. An engine fills the schema; it
#: does not get to have one of its own.
_COLS, _OBJ, _ROWS, _MATRIX = sinks.COLS, sinks.OBJ, sinks.ROWS, sinks.MATRIX
_DTYPES = sinks.DTYPES


class PolarsExecutor(Engine):
    """Build a :class:`Program` into polars frames, then sink it."""

    def __init__(self) -> None:
        self._program: plan.Program | None = None
        self._compiler: PolarsCompiler | None = None
        self._labels: Labeller | None = None
        self._bound: BoundSources | None = None
        self._var_frames: dict[str, pl.LazyFrame] = {}
        self._row_frames: dict[str, pl.LazyFrame] = {}
        self._omitted: dict[str, int] = {}
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
        Declarations are then built one at a time and concatenated at the end:
        their rows are independent, which is what lets the model be four frames
        rather than a graph.
        """

        self._program = program
        self._bound = bind(program, sources)
        # The variable frames are passed apart from the bound data on purpose:
        # they are the one registry still being filled, appearing as each
        # declaration is built so a constraint compiled afterwards can see them.
        self._compiler = PolarsCompiler(program, self._bound, self._var_frames)
        self._labels = Labeller(self._compiler, self._bound.cardinality, program)

        cols = [self._build_variable(v) for v in program.variables]
        built = [self._build_constraint(c) for c in program.constraints]
        objective = self._build_objective(program.objective)

        self._cols = _stack(cols, _COLS)
        self._rows = _stack([r for r, _ in built], _ROWS)
        stacked = _stack([m for _, m in built if m is not None], _MATRIX)
        self._matrix, self._matrix_starts = sinks.compress_rows(stacked, self._n_rows)
        self._obj = _stack([objective] if objective is not None else [], _OBJ)

    @property
    def _variables(self) -> Mapping[str, pl.LazyFrame]:
        return self._var_frames

    @property
    def _constraints(self) -> Mapping[str, pl.LazyFrame]:
        return self._row_frames

    @property
    def _q(self) -> PolarsCompiler:
        assert self._compiler is not None, 'build() has not run'
        return self._compiler

    @property
    def _label(self) -> Labeller:
        assert self._labels is not None, 'build() has not run'
        return self._labels

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
        if 'coeff' not in matrix.columns:
            return
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
        labelled, self._n_cols = self._label.frame(v.dims, v.where, 'var_label', start)
        self._var_frames[v.name] = labelled.lazy()
        self._blocks[v.name] = (start, labelled.height)

        bounded = self._q.bounds(labelled.lazy(), v)
        # No `col`: the label frame arrives in label order and the bounds join
        # maintains it, so a row's *position* is its solver column index. The
        # column would be the frame's own row number — 0.32 GB of it at 40M
        # columns, held for as long as the model is.
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

        The terminal aggregate is where duplicates from ``Sum`` and
        ``GroupSum`` — which project rather than aggregate — collapse, and it
        is skipped where nothing can (:func:`~lpspec.relational.engine.needs_aggregate`).

        **Either way the share leaves ordered by ``row``.** Every sink reads
        the matrix a row range at a time, so one that is handed an unordered
        matrix orders it — a second pass over a finished frame, and a second
        copy of it while the solver's own model is resident. Ordering it here
        happens inside the pipeline that is already materialising the rows.

        A lone term gets the order from its *join*: the row frame is in label
        order and ``maintain_order='left'`` keeps it, for less than sorting the
        result costs. Several terms are several ordered runs stacked, and runs
        are not an order, so those sort — which is why only the lone term asks
        the join for anything, a stack having paid for the order regardless.
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
        labelled, self._n_rows = self._label.frame(c.dims, c.where, 'row', start, restrictions)
        frame = labelled.lazy()
        self._row_frames[c.name] = frame  # kept for the dual read-back
        self._blocks[c.name] = (start, labelled.height)  # narrowed if rows go termless

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
            self._row_frames[c.name] = self._row_frames[c.name].clear()
            return rows.clear(), None

        pieces = []
        carried_order: MaintainOrderJoin | None = 'left' if len(terms) == 1 else None
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
        stacked = pl.concat(pieces)
        if not needs_aggregate([fragment for fragment, _ in terms], self._q.may_share_a_column):
            ordered = stacked if len(pieces) == 1 else stacked.sort('row')
            matrix = ordered.collect(engine='streaming')
            self._check_no_undefined_divisor(f"constraint '{c.name}'", matrix, c.lhs, c.rhs)
            rows, matrix, self._n_rows = self._drop_termless_rows(c.name, rows, matrix, start)
            return rows, matrix

        # The aggregate is reachable, but "reachable" is all the fragments can
        # say. Sorting first turns the question into one pass over adjacent
        # pairs, and a hash table sized by the number of groups — which is
        # nearly the number of rows, since a repeated cell is the exception —
        # is only built when there is something to collapse.
        matrix = stacked.sort('row', 'col').collect(engine='streaming')
        self._check_no_undefined_divisor(f"constraint '{c.name}'", matrix, c.lhs, c.rhs)
        if _has_repeated_entry(matrix):
            matrix = matrix.group_by('row', 'col').agg(pl.col('coeff').sum()).sort('row', 'col')
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
        self._row_frames[name] = (
            self._row_frames[name].filter(pl.col('row').is_in(kept)).with_columns(pl.col('row').replace_strict(remap))
        )
        return rows, matrix, start + surviving.height

    def _build_objective(self, o: plan.ObjectiveDeclaration) -> pl.DataFrame | None:
        """The objective as ``(col, coeff)``, or ``None`` if it has no terms.

        **This projection drops the dims, so it asks for the stronger key** —
        ``needs_aggregate(..., projected=True)``. Where the matrix keeps a
        fragment's dims in ``row``, here only ``var_label`` survives, and a dim
        that arrived by broadcast then puts several rows on one column.

        Their sum is the coefficient, and nothing downstream computes it: the
        hand-off scatters with ``dense[at] = values``, which keeps the *last*
        write, and the LP writer emits one term per row for a reader to
        interpret as it likes. So a missed aggregate here is a wrong objective
        that still solves, not a slow one.
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
        stacked = pl.concat(pieces)
        if needs_aggregate(comp.terms, self._q.may_share_a_column, projected=True):
            stacked = stacked.group_by('col').agg(pl.col('coeff').sum())
        return stacked.collect(engine='streaming')

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

    # ------------------------------------------------------------------

    def close(self) -> None:
        """Drop the built model. Optional — see :class:`Result`."""
        self._cols = self._obj = self._rows = self._matrix = self._matrix_starts = None
        self._var_frames.clear()
        self._row_frames.clear()
        self._omitted.clear()
        self._blocks.clear()
        # `BoundSources` is frozen, so it cannot be emptied in place the way the
        # registries above can: what frees the bound frames is dropping every
        # reference to them, and the compiler holds one.
        self._bound = None
        self._compiler = None
        self._labels = None


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


def _has_repeated_entry(matrix: pl.DataFrame) -> bool:
    """Whether a matrix sorted by ``(row, col)`` holds one cell twice.

    :func:`~lpspec.relational.engine.needs_aggregate` answers whether a stack *can* repeat a cell, which
    is all a static reading of the fragments can say. This answers whether it
    *did*, which is one pass over a sorted frame and lets the aggregate be
    skipped in the case the static answer is conservative about.

    That case is not rare. `transport` stacks three fragments, so the static
    answer is yes on every row, and at the `l` rung the aggregate collapses
    exactly nothing out of 12.6M entries.
    """
    if matrix.height < 2:
        return False
    repeated = (pl.col('row') == pl.col('row').shift(1)) & (pl.col('col') == pl.col('col').shift(1))
    return bool(matrix.select(repeated.any()).item())


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

    **A fragment with nothing to restrict is skipped**, and that is load-bearing
    rather than tidy: a restriction is data — which rows survive is unknown until
    the presence frames are read — so it costs ``Labeller.frame`` both of its
    arithmetic paths. An unmasked variable's presence is its whole coordinate
    product and would remove nothing, so it never gets to impose that cost.

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
