"""Polars engine: fill the model frames, then hand them to a sink.

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

import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, get_args

import polars as pl

from lpspec.errors import (
    DataError,
    LanguageError,
    LpspecError,
    no_duals_message,
    null_bounds_message,
    sparse_divisor_message,
    uncovered_constant_message,
)
from lpspec.relational import plan, sinks
from lpspec.relational.engines.polars import labels
from lpspec.relational.engines.polars.binding import BoundSources, bind
from lpspec.relational.engines.polars.compiler import PolarsCompiler, Presence, TermFragment
from lpspec.relational.result import Diagnostics, Result
from lpspec.relational.sinks.tables import SENSE

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from polars._typing import MaintainOrderJoin


#: The four frames a sink reads, as schemas. Stated here because the engine
#: is what fills them and an empty model still has to have them.
_COLS = ('lb', 'ub', 'vtype')
_OBJ = ('col', 'coeff')
_ROWS = ('row', 'sense', 'rhs')
_MATRIX = ('row', 'col', 'coeff')
_SOS = ('set', 'type', 'col', 'weight', 'big_m')

#: The dtype of each of those columns. ``vtype`` is an ``Enum`` over the
#: variable types the plan declares, rather than a string: it holds one word
#: per column and the same handful of words for the whole model, so as a string
#: it stores that word once per row, where an Enum stores a code: on a wide
#: model that is most of the ``cols`` frame (#189). The Enum also makes the
#: vocabulary
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
#:
#: ``set`` and ``weight`` are ``Int32`` for the same reason ``col`` is, one
#: step removed: a set holds at least one column and a weight counts along one
#: dim of one variable, so both are bounded by a count that already has to fit
#: a 32-bit index. The stream is one row per *member*, which on a model whose
#: sets cover it is the largest frame after the matrix, and every pass over it
#: — the digest included — is paid in its width (#687).
_DTYPES = {
    'col': pl.Int32, 'row': pl.Int64,
    'lb': pl.Float64, 'ub': pl.Float64, 'rhs': pl.Float64, 'coeff': pl.Float64,
    'sense': SENSE, 'vtype': pl.Enum(get_args(plan.VariableType)),
    'set': pl.Int32, 'type': pl.UInt8, 'weight': pl.Int32, 'big_m': pl.Float64,
}  # fmt: skip


@dataclass(frozen=True)
class _Block:
    """One declaration's contiguous, dense run of labels: its start, and how many.

    :func:`~lpspec.relational.engines.polars.labels.frame` numbers a
    declaration's survivors contiguously, so its share of a solver vector is
    a slice — which is what :meth:`PolarsEngine._laid_out` spends it on.
    """

    start: int
    height: int

    def share(self, values: pl.Series) -> pl.Series:
        """This declaration's share of a solver vector — a slice, never a join."""
        return values.slice(self.start, self.height)


class PolarsEngine:
    """Build a :class:`Program` into polars frames, then sink it."""

    def __init__(self) -> None:
        #: The solver holding this model, kept between solves — the only thing
        #: a rebuild does *not* throw away. ``None`` until one has been solved.
        self._solver: sinks.Solver | None = None
        #: How many solves this model has been through, and how many of them
        #: had to load the solver from scratch instead of pushing values onto
        #: one that already held it. Read together: one load in many solves is
        #: a driver on the fast path, a load per solve is one that is not.
        self._solves = 0
        self._loads = 0
        #: What the last solve's sink had to add to take the model — nothing,
        #: unless it had no concept of a set the model declares. Kept beside
        #: the counters rather than in :meth:`_reset` for their reason: it is
        #: a fact about a *solve*, so a rebuild does not clear it.
        self._sink_columns = 0
        self._sink_rows = 0
        #: Wall seconds each phase has spent, cumulatively — what
        #: :meth:`diagnostics` reports as ``timings``. Kept beside the
        #: counters for their reason: time spent is a fact about what ran,
        #: so a rebuild adds to it rather than clearing it.
        self._timings: dict[str, float] = {}
        self._reset()

    def _reset(self) -> None:
        """The state one build owns, emptied.

        What a second :meth:`build` over the same object clears, so that what
        survives a rebuild is exactly what is *not* the built model: the
        loaded solver and the counters.
        """
        self._program: plan.Program | None = None
        self._compiler: PolarsCompiler | None = None
        self._bound: BoundSources | None = None
        #: ``name -> deferred plan expression``, one per declared named
        #: expression. Thunks, never plans: a build lowers none of them
        #: (SPEC §3), and a solve turns each into a reader that lowers on its
        #: first call (:meth:`_expression_readers`).
        self._expression_thunks: dict[str, Callable[[], plan.Expression]] = {}
        self._variables: dict[str, pl.LazyFrame] = {}
        self._constraints: dict[str, pl.LazyFrame] = {}
        #: ``name -> rows not built``, because every term they had vanished.
        #: Empty for a model whose every declared row reached the solver.
        self._omitted: dict[str, int] = {}
        #: One :class:`_Block` per declaration, one map per label space.
        #: Columns and rows are numbered independently and a model may name a
        #: variable and a constraint alike, so one map keyed by name would
        #: hand the primal reader a row block.
        self._variable_blocks: dict[str, _Block] = {}
        self._constraint_blocks: dict[str, _Block] = {}
        self._cols: pl.DataFrame | None = None
        self._obj: pl.DataFrame | None = None
        self._rows: pl.DataFrame | None = None
        self._matrix: pl.DataFrame | None = None
        self._sos: pl.DataFrame | None = None
        self._matrix_starts: Any = None
        self._n_cols = 0
        self._n_rows = 0
        #: How many special-ordered sets have been numbered. Sets are dense
        #: ``0..n-1`` across the model, like columns and rows, so a sink names
        #: one by its number and two builds agree on which.
        self._n_sets = 0
        #: Entries in the matrix, kept as a count rather than read off the
        #: frame: the frames go when the model is released and this is what a
        #: caller asking how big it *was* is asking for.
        self._n_entries = 0
        self._obj_const = 0.0
        self._obj_sense: str = 'min'

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def build(
        self,
        program: plan.Program,
        sources: Mapping[str, Any],
        expressions: Mapping[str, Callable[[], plan.Expression]] | None = None,
    ) -> None:
        """Bind *sources*, then build every declaration into the model frames.

        The compiler comes after binding, two of its answers being read off the
        data — a dim's size, whether a parameter is boolean. Declarations build
        one at a time and concatenate at the end; their rows are independent,
        which is what lets the model be four frames rather than a graph.

        The matrix leaves in ``(row, col)`` order, as ``ModelTables`` promises
        its sinks. The stack already has it — each share leaves sorted and owns
        the next run of rows — so :func:`_in_row_order` *checks* with one
        linear scan rather than sorting the model's largest frame at the peak
        of the build. :func:`_row_starts` reads the CSR index off that order,
        after which ``row`` is dropped: 8 bytes per entry no sink reads.

        **What a zero coefficient states, absence already states**, so each
        share arrives already pruned of them (:func:`_without_zeros`) and the
        objective of its own. Sparsity that arrives as absence never becomes a
        term at all; sparsity spelled out as zeros used to become one, and a
        solver's own load is most of what a hand-off costs.

        **``rows`` leaves in row order too**, which a solver reads as its own
        index — so :meth:`~lpspec.relational.sinks.tables.ModelTables.dense_rows`
        takes the frame's own vectors instead of scattering by label on every
        solve, and :attr:`~lpspec.relational.sinks.tables.ModelTables.structure`
        hashes the column two builds now agree on. The order is *checked*
        first, by the same rule labelling uses: a constant's left join is what
        usually loses it, and forcing that join to hold it is a bet on the
        model's shape rather than a fix — free on `fleet` and most of a build
        again on `dispatch`.

        **A second call rebuilds over the same object**, which is what
        ``rebind`` is. The previous build is released *before* this one starts,
        so a driver that re-solves in a loop stays at one model's peak; what
        the loaded solver holds survives as the digest it recorded at its load.

        *expressions* maps each declared named expression to a thunk producing
        its plan expression. None is called here — a build pays nothing for a
        declared expression (SPEC §3) — they become the deferred readers a
        solve's result hands out (:meth:`_expression_readers`).
        """
        self._reset()
        self._expression_thunks = dict(expressions or {})

        self._program = program
        with _clocked(self._timings, 'bind'):
            self._bound = bind(program, sources)
        self._compiler = PolarsCompiler(program, self._bound, self._variables)

        with _clocked(self._timings, 'build'):
            cols = [self._build_variable(v) for v in program.variables]
            sets = [self._build_sos(s, program.variable(s.variable)) for s in program.sos]
            built = [self._build_constraint(c) for c in program.constraints]
            objective = self._build_objective(program.objective)

            self._sos = _stack(sets, _SOS)
            self._cols = _stack(cols, _COLS)
            self._rows = labels.in_position_order(_stack([r for r, _ in built], _ROWS), 'row')
            ordered = _in_row_order(_stack([m for _, m in built if m is not None], _MATRIX))
            self._matrix_starts = _row_starts(ordered, self._n_rows)
            self._matrix = ordered.select('col', 'coeff').rechunk()
            self._n_entries = self._matrix.height
            self._obj = _stack([objective] if objective is not None else [], _OBJ)

    @property
    def _q(self) -> PolarsCompiler:
        assert self._compiler is not None, 'build() has not run'
        return self._compiler

    def _refuse_undefined_divisors(self, stacked: pl.DataFrame, name: str, *expressions: plan.Expression) -> None:
        """A null coefficient means a divisor had no value where the model divided.

        A quotient left-joins its divisor (:func:`_join_mul`), so a missing
        value leaves a null — and a term whose row was masked out, or whose
        numerator variable is absent, never gets this far. That is what keeps
        the refusal from becoming a wall on ordinary sparse data. Asked of the
        stack before any cell collapses, since ``sum`` reads a null as zero.
        """
        undefined = int(stacked.get_column('coeff').null_count())
        if undefined:
            params = sorted(plan.divisor_parameters(*expressions))
            raise DataError(f'{name}: {sparse_divisor_message(", ".join(params), undefined)}')

    def _matrix_share(
        self, pieces: list[pl.LazyFrame], name: str, *expressions: plan.Expression
    ) -> tuple[pl.DataFrame, pl.Series]:
        """One constraint's share: in ``(row, col)`` order, repeated cells summed.

        Nothing runs unconditionally except three linear probes — the null
        count, whether the stack arrives in order, whether any cell repeats. It
        usually is in order and usually repeats nothing (a cell repeats only
        when one variable reaches a row twice), so the sort and the aggregate
        run only when a probe says they would change something, which is far
        cheaper than the unconditional hash aggregate it replaced. Read off the
        data, so nothing here knows *why* a cell repeats (#520).

        **Cheap probes are why the share is rechunked first.** A streaming
        collect returns morsels as chunks, and ``shift(1)`` pays at every
        boundary where ``is_sorted`` does not, so rechunking and probing
        together beats probing as it arrives (#576). Not an extra copy, since
        the assembly needs a contiguous matrix anyway (#550). The objective's
        stack is left fragmented — measured, and it costs peak for no wall.

        **Zeros go before any of that**, so the probes read them and the sort
        orders them no longer — on a share that is mostly zeros they were most
        of each. Pruned behind a probe, like everything else here: the filter
        leaves a chunked frame and ``shift(1)`` pays at every boundary (#576),
        so a share with nothing to drop must not pay a rechunk to find that
        out. A cancelling pair survives to the aggregate and only becomes a
        zero there, so the prune runs again on the path that aggregated, and
        only on it.

        Returns:
            The share, and the rows that had *any* term. Those are read off a
            frame before a prune takes the answer away, because that is the one
            question a pruned share can no longer answer: a row whose every
            coefficient is zero owns no entries and is not thereby a row with
            no terms.
        """
        stacked = pl.concat(pieces).collect(engine='streaming').rechunk()
        self._refuse_undefined_divisors(stacked, name, *expressions)
        pruned = _pruned(stacked)
        term_rows = stacked.get_column('row').unique() if pruned.height != stacked.height else None
        stacked = pruned
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
            return stacked, _rows_of(stacked, term_rows)
        aggregated = (
            stacked.lazy()
            .group_by('row', 'col')
            .agg(pl.col('coeff').sum())
            .sort('row', 'col')
            .collect(engine='streaming')
        )
        return _pruned(aggregated), _rows_of(aggregated, term_rows)

    # ------------------------------------------------------------------
    # declarations
    # ------------------------------------------------------------------

    def _build_variable(self, v: plan.VariableDeclaration) -> pl.DataFrame:
        """One variable's labelled frame, and its share of ``cols``.

        The share leaves in label order, ``cols`` carrying no ``col`` of its
        own: a row's *position* is its solver column index. The bounds joins
        usually keep that order, so it is verified with one linear scan and
        re-established only when a join lost it
        (:func:`labels.in_position_order`).

        Only the label and the two bounds are collected, keeping the dim
        columns and joined bound parameters inside the lazy pipeline rather
        than materialising them to be dropped. The label then goes too, having
        been the order's witness and nothing else.
        """
        start = self._n_cols
        labelled = labels.frame(self._q, v.dims, v.where, 'var_label', start)
        self._n_cols = start + labelled.height
        self._variables[v.name] = labelled.lazy()
        self._variable_blocks[v.name] = _Block(start, labelled.height)

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

    def _build_sos(self, s: plan.SosDeclaration, v: plan.VariableDeclaration) -> pl.DataFrame:
        """One declaration's sets as ``(set, type, col, weight, big_m)``, over *v*.

        Builds no column and no row: a set names columns the variable already
        made. It runs after every variable and before any constraint, which is
        when what it reads exists and before rows a constraint may drop.

        **A set and a weight are the two halves of a coordinate's position in
        the declared product**, split at the ``over`` dim. With :math:`c` that
        dim's cardinality and :math:`s` the product of the dims after it, a
        position reads :math:`hcs + ws + l`, and the split drops the middle
        term: the weight is :math:`w`, and the set is :math:`hs + l` — the
        position the same coordinate would have in the product *without*
        ``over``. That is the row-major rule
        (:func:`~lpspec.relational.engines.polars.labels.row_major`) numbering
        every other index in the lane, by two divisions rather than by reading
        a dim's ordinal per member.

        **A position is the label where the variable dropped nothing.** Labels
        are dense and follow declaration order, so an unmasked variable's
        positions are ``0 … height-1`` and neither its frame nor its dims are
        touched at all; a masked one has to say which coordinate survived, so
        there the ordinals are read and the sets are renumbered densely — the
        same split :func:`~lpspec.relational.engines.polars.labels.frame`
        makes, for the same reason (#520, measured at #687).

        **A member's weight is its coordinate's position in the declared
        order** — what ``shift`` walks, and the only order the language has.
        Only the order matters to a solver, so a masked-out coordinate leaves
        its neighbours adjacent rather than leaving a hole.

        **The stream leaves grouped by set and ascending in weight**, which is
        what lets a sink read a set's edges off the neighbouring row rather
        than aggregating. Within a set that is free — for fixed values of every
        other dim, the ``over`` ordinal ascends with the label — and across
        sets it is free too *when ``over`` is the last dim*, which the common
        shapes have. So the order is verified and the sort runs only where
        members genuinely interleave. It sorts on **both** columns rather than
        through
        :func:`~lpspec.relational.engines.polars.labels.in_position_order`,
        which numbers a frame whose position is unique: here a position is a
        whole set, and a sort that reordered ties would be a set whose members
        arrive out of weight order.

        ``big_m`` rides along per member, at ``inf`` where the block declared
        none — the one thing here no sink taking a set natively reads.
        """
        block = self._variable_blocks[s.variable]
        cardinality = self._q.data.cardinality
        stride = math.prod(cardinality[d] for d in v.dims[v.dims.index(s.over) + 1 :])
        span = cardinality[s.over] * stride

        if v.where is None:
            frame = pl.select(pl.int_range(block.height, dtype=pl.Int64).alias('#position')).lazy()
            place, col = pl.col('#position'), (pl.col('#position') + block.start)
        else:
            frame = self._variables[s.variable]
            place, col = labels.row_major(self._q, v.dims, self._q.ordinal_of), pl.col('var_label')
        placed = frame.select(
            ((place // span) * stride + place % stride).alias('#set position'),
            ((place // stride) % cardinality[s.over] + 1).cast(_DTYPES['weight']).alias('weight'),
            col.cast(_DTYPES['col']).alias('col'),
        ).collect(engine='streaming')

        position = pl.col('#set position')
        grouped = placed if placed.get_column('#set position').is_sorted() else placed.sort('#set position', 'weight')
        dense = position if v.where is None else (position != position.shift(1)).fill_null(True).cum_sum() - 1
        built = grouped.select(
            (dense + self._n_sets).cast(_DTYPES['set']).alias('set'),
            pl.lit(s.sos_type, dtype=_DTYPES['type']).alias('type'),
            'col',
            'weight',
            pl.lit(float('inf') if s.big_m is None else s.big_m, dtype=_DTYPES['big_m']).alias('big_m'),
        )
        if built.height:
            self._n_sets = built.item(-1, 'set') + 1
        return built

    def _build_constraint(self, c: plan.ConstraintDeclaration) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        """One constraint as its ``rows`` and its share of the matrix.

        Terms normalise to the left, constants to the right. Each constant
        fragment is aggregated to its own coordinates and left-joined, so a
        coordinate it has no row for contributes zero.

        **The coverage check rides on the rows pass rather than taking its own.**
        Both read the same joined carrier, so asking separately collects every
        constant's join twice. The flag is a boolean column dropped once counted,
        and the refusal still precedes any use of the rows — ``accumulated``
        fills nulls with zero, so a gap cannot reach ``rhs`` before it is caught.

        Duplicates from ``Sum`` and ``GroupSum`` — which project rather than
        aggregate — and from ``x + 2 * x`` collapse in :meth:`_matrix_share`'s
        terminal ``SUM(coeff) GROUP BY row, col``, read off the data rather
        than reasoned from how the fragments were reshaped (#520). The share
        leaves in ``(row, col)`` order, which is how every sink reads it.

        The labelled frame is kept for the dual read-back, and its block
        narrows when rows go termless: the run of labels a declaration owns is
        what survived, not what it declared (#561).
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
        labelled = labels.frame(self._q, c.dims, c.where, 'row', start, restrictions)
        self._n_rows = start + labelled.height
        frame = labelled.lazy()
        self._constraints[c.name] = frame
        self._constraint_blocks[c.name] = _Block(start, labelled.height)

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

        gap_column = '__uncovered__'
        rows = carrier.select(
            'row',
            pl.lit(c.sense, dtype=SENSE).alias('sense'),
            accumulated.cast(pl.Float64).alias('rhs'),
            *([uncovered.alias(gap_column)] if uncovered is not None else []),
        ).collect(engine='streaming')

        if uncovered is not None:
            gaps = int(rows.get_column(gap_column).sum())
            if gaps:
                names = ', '.join(sorted(plan.parameters_of(c.lhs, c.rhs)))
                raise DataError(uncovered_constant_message(names, gaps, f"constraint '{c.name}'"))
            rows = rows.drop(gap_column)

        if not terms:
            self._omitted[c.name] = rows.height
            self._constraint_blocks[c.name] = _Block(start, 0)
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
        matrix, term_rows = self._matrix_share(pieces, f"constraint '{c.name}'", c.lhs, c.rhs)
        rows, matrix, self._n_rows = self._drop_termless_rows(c.name, rows, matrix, term_rows, start)
        return rows, matrix

    def _drop_termless_rows(
        self, name: str, rows: pl.DataFrame, matrix: pl.DataFrame, kept: pl.Series, start: int
    ) -> tuple[pl.DataFrame, pl.DataFrame, int]:
        """Rows that kept no variable term are not built, and the block closes up.

        A row with no variables is not a constraint — it asserts something
        about constants, which the solver cannot act on. Three provenances
        reach that shape (an absent variable, an empty reduction, a missing
        coefficient) and all three drop the row, so the rule is stated once
        here rather than per provenance.

        *kept* is the row set the share had terms for, which is
        :meth:`_matrix_share`'s to answer and not this one's to re-derive: the
        share it returns has been pruned of zero coefficients, so a row missing
        from it may have had every term and every one of them zero. That row
        stays — it asserts something the solver *can* act on, ``0 >= 10`` being
        infeasible — where a row that never had a term goes.

        Labels are dense and the dual read-back reads a block by position, so a
        dropped row may not leave a gap: survivors renumber from *start* and
        the row counter rewinds. Costs one comparison when nothing is dropped,
        which is every correct model.
        """
        if kept.len() == rows.height:
            return rows, matrix, start + rows.height

        surviving = rows.filter(pl.col('row').is_in(kept)).sort('row')
        renumber = surviving.select('row').with_row_index('__new__', offset=start)
        self._omitted[name] = rows.height - surviving.height
        self._constraint_blocks[name] = _Block(start, surviving.height)
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

        The aggregate runs only when a column repeats, probed by ``n_unique``
        — the only sound probe here, the stack arriving unordered so adjacency
        proves nothing. Buying order to probe linearly is a dead end twice
        over: the mul join's ``maintain_order`` holds the label order on some
        shapes and loses it on others differing only in data (`dispatch` keeps
        it, `nodal` and `profiled` do not, all three the same lone masked
        ``p * cost``), so no static gate can say when the tax will pay; and
        paid for nothing it multiplies the objective phase several times over,
        against a best case that is a wash (#581). ``obj`` carries no order
        contract anyway.
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
        if stacked.get_column('col').n_unique() != stacked.height:
            stacked = stacked.lazy().group_by('col').agg(pl.col('coeff').sum()).collect(engine='streaming')
        return _without_zeros(stacked)

    # ------------------------------------------------------------------
    # sinks — see relational/sinks/; the engine only supplies the frames
    # ------------------------------------------------------------------

    def _tables(self) -> sinks.ModelTables:
        if self._cols is None or self._obj is None or self._rows is None or self._matrix is None or self._sos is None:
            raise LpspecError(
                'there is no built model to hand over: it was closed, or a rebind raised and released '
                'it rather than leaving half of one behind. Build it again — rebind() with data it can '
                'bind, or build() from the start.'
            )
        return sinks.ModelTables(
            cols=self._cols,
            obj=self._obj,
            rows=self._rows,
            matrix=self._matrix,
            sos=self._sos,
            row_starts=self._matrix_starts,
            column_count=self._n_cols,
            row_count=self._n_rows,
            objective_sense=self._obj_sense,
            objective_constant=self._obj_const,
        )

    def write(self, path: str | Path) -> None:
        """Stream the built model to *path*, in the format its suffix names.

        Raises:
            ValueError: A suffix nothing writes.
        """
        path = Path(path)
        write = sinks.writer(path.suffix.lower())
        tables = self._tables()
        with _clocked(self._timings, 'write'):
            write(tables, path)

    def solve(
        self,
        solver_name: str = 'highs',
        *,
        solver_options: Mapping[str, Any] | None = None,
        batch_rows: int | None = None,
        warm: bool = True,
    ) -> Result:
        """Hand the built model to a solver and solve it.

        The solver stays loaded where it can, which is
        :func:`~lpspec.relational.sinks.solvers.loaded`'s decision and not this
        method's: a rebound model has its new numbers pushed onto what the
        solver already holds, and one whose structure moved is loaded again.
        All that is kept here is the solver itself and the two counters
        :meth:`diagnostics` reports, the answer being the same either way.

        **What the solver is handed may be wider than what was built**, and
        that too is the family's decision
        (:func:`~lpspec.relational.sinks.solvers.ingestible`): a sink with no
        SOS concept takes the sets as binaries and rows appended past the
        model's own. The read-back is unaffected — a declaration's share is a
        slice, and nothing was appended before one.

        Args:
            solver_name: ``highs``, which ships with the package, or
                ``gurobi``, which needs the ``[gurobi]`` extra.
            solver_options: Forwarded to the solver verbatim, in its own
                vocabulary (``{'time_limit': 60, 'mip_rel_gap': 0.01}``).
            batch_rows: The hand-off budget in elements, defaulting to the
                sink's own
                (:data:`~lpspec.relational.sinks.solvers.highs.HANDOFF_BUDGET`).
                This method's parameter alone, kept off the public handle:
                nothing outside the chunking tests sets it.
            warm: Whether this solve may start from what the session holds.
                ``True``, the default, keeps it — warm on a kept session, cold
                on a fresh load. ``False`` is deliberately cold, held to
                structurally: the held solver is closed before the load
                decision, so the fresh one has nothing to start from, whatever
                a member squirrels away. :attr:`~lpspec.relational.result.Diagnostics.loads`
                ticks with it, the whole model having been transferred again.

        Returns:
            The solution, holding this engine and the build it answered.
        """
        built = self._tables()
        with _clocked(self._timings, 'handoff'):
            tables = sinks.ingestible(solver_name, built)
            self._sink_columns = tables.column_count - built.column_count
            self._sink_rows = tables.row_count - built.row_count
            if not warm and self._solver is not None:
                self._solver.close()
                self._solver = None
            held = self._solver
            self._solver = sinks.loaded(held, solver_name, tables, batch_rows, solver_options)
            started: Literal['cold', 'session'] = 'session' if self._solver is held else 'cold'
        self._solves += 1
        if self._solver is not held:
            self._loads += 1
        with _clocked(self._timings, 'solve'):
            answer = self._solver.run(tables)
        assert answer.primal is not None or not answer.status.is_readable, (
            'a readable status must come with a primal vector'
        )
        assert (answer.activity is None) == (answer.primal is None), (
            'activity travels with the primal: every sink reads it whenever a solution exists, mixed-integer included'
        )
        primals, duals, activities = self._read_back(answer.primal, answer.dual, answer.activity)
        return Result(
            _status=answer.status,
            _objective=answer.objective,
            _primals=primals,
            _duals=duals,
            _activities=activities,
            _started=started,
            _expressions=self._expression_readers(answer.primal),
            _no_duals=None
            if answer.dual is not None
            else no_duals_message(
                self._discrete(),
                answer.status.termination_condition,
                sets=self._reformulated_sets(tables is not built),
            ),
        )

    def diagnostics(self) -> Diagnostics:
        """What this build and its solves did that the answer does not show.

        Answerable after :meth:`close`: every field is a count, a clock or a
        small frame this keeps, not a read of the model it releases.
        """
        return Diagnostics(
            columns=self._n_cols,
            rows=self._n_rows,
            nonzeros=self._n_entries,
            sink_columns=self._sink_columns,
            sink_rows=self._sink_rows,
            omissions=pl.DataFrame(
                {'constraint': list(self._omitted), 'rows_not_built': list(self._omitted.values())},
                schema={'constraint': pl.String, 'rows_not_built': pl.UInt32},
            ),
            solves=self._solves,
            loads=self._loads,
            timings=dict(self._timings),
        )

    def _read_back(
        self, primal: pl.Series | None, dual: pl.Series | None, activity: pl.Series | None
    ) -> tuple[dict[str, pl.LazyFrame], dict[str, pl.LazyFrame], dict[str, pl.LazyFrame]]:
        """One solve's answer as one frame per declaration — a :class:`Result`'s own.

        References rather than copies: the frames point at this build's label
        frames, and :meth:`build` replacing the registries takes nothing from
        what an earlier result still holds. What a retained result costs is
        exactly those label frames staying alive — never the four model frames,
        never the solver.

        Lazy, so composing every declaration's plan here costs nothing for the
        ones nobody reads: the slices are views on the solver's own vector and
        the collect happens where a caller asks.

        A vector that is ``None`` yields no frames at all rather than empty
        ones, which is the state :class:`Result` reports through the status and
        through ``_no_duals``.
        """
        assert self._program is not None
        program = self._program

        def rows(values: pl.Series | None) -> dict[str, pl.LazyFrame]:
            if values is None:
                return {}
            return {
                c.name: self._laid_out(self._constraint_blocks[c.name], self._constraints[c.name], c.dims, values)
                for c in program.constraints
            }

        return (
            {
                v.name: self._laid_out(self._variable_blocks[v.name], self._variables[v.name], v.dims, primal)
                for v in program.variables
            }
            if primal is not None
            else {},
            rows(dual),
            rows(activity),
        )

    def _expression_readers(self, primal: pl.Series | None) -> dict[str, Callable[[], pl.DataFrame]]:
        """One deferred reader per declared named expression — nothing compiled yet.

        Deferral is the contract (SPEC §3): a closure lowers and compiles its
        expression when it is first called, so a solve over fifty declared
        expressions that reads none pays for a dict of closures. Each captures
        a snapshot the result *owns* — the program, the bound data, a copy of
        this build's variable-frame registry and the solver's primal vector —
        so it keeps answering after a rebind or ``close()`` the way every
        other reader does, at the cost of keeping those frames alive.
        """
        if primal is None:
            return {}
        assert self._program is not None and self._bound is not None
        compiler = PolarsCompiler(self._program, self._bound, dict(self._variables))
        values = pl.DataFrame(
            {'var_label': pl.int_range(primal.len(), dtype=pl.Int64, eager=True), _SOLUTION: primal}
        ).lazy()

        def reader(name: str, thunk: Callable[[], plan.Expression]) -> Callable[[], pl.DataFrame]:
            return lambda: _expression_frame(name, thunk(), compiler, values)

        return {name: reader(name, thunk) for name, thunk in self._expression_thunks.items()}

    def _laid_out(
        self,
        block: _Block,
        coordinates: pl.LazyFrame,
        dims: tuple[str, ...],
        values: pl.Series,
    ) -> pl.LazyFrame:
        """One declaration's coordinates in label order, beside its values.

        **The order is not re-established here, because it was never lost**:
        :func:`labels.frame` numbers a sorted frame and hands back a
        label-ascending one, and the solver's vector is positional in the same
        index, so coordinates and values line up by construction.

        The declaration owns a contiguous, dense run of labels — *block*, out
        of the map for the label space *coordinates* is numbered in, which is
        what makes its share of the vector a slice rather than a join. The
        slice is attached as a column rather than concatenated as a frame, so
        a mismatched length raises instead of padding with nulls.

        **Dim columns leave in ``String``**, where the build holds them as
        ``pl.Enum`` (#541). That encoding is internal and every gram of its win
        is upstream of here, but a returned frame is something a caller *joins
        against their own data* — and polars refuses ``Enum`` against
        ``String`` with a message about dtypes that names nothing about the
        cause. Two frames of one sweep will not even concatenate when their
        slices bound different members.

        The cast sits inside this projection rather than after it, so the
        string column is produced once instead of widened from an Enum that
        also exists, which is cheaper in both wall and peak (#593). Declaration
        order is the *row* order and survives, never having been the dtype's to
        carry.
        """
        labelled = coordinates.select(*dims).with_columns(block.share(values))
        return labelled.with_columns(pl.col(d).cast(pl.String) for d in self._string_dims(dims))

    def _string_dims(self, dims: tuple[str, ...]) -> list[str]:
        """Those of *dims* the binder encoded as ``Enum`` — its string ones."""
        assert self._bound is not None, 'build() has not run'
        return [d for d in dims if self._bound.is_enum_encoded(d)]

    def _discrete(self) -> list[str]:
        """The variables this model declared as anything but continuous."""
        assert self._program is not None
        return sorted(v.name for v in self._program.variables if v.variable_type != 'continuous')

    def _reformulated_sets(self, reformulated: bool) -> list[str]:
        """The sets that reached the solver as binaries, if any did.

        What makes an otherwise continuous model come back without duals, and
        the one such reason no declaration shows: the model declares no
        integrality, and the sink added some.
        """
        assert self._program is not None
        return sorted(s.name for s in self._program.sos) if reformulated else []

    # ------------------------------------------------------------------

    def close(self) -> None:
        """Drop the built model. A :class:`Result` keeps its own frames.

        ``BoundSources`` is frozen, so it cannot be emptied in place the way
        the registries can: what frees the bound frames is dropping every
        reference to them, and the compiler holds one. A loaded solver goes
        too, being the one thing here that is not this process's memory.
        """
        if self._solver is not None:
            self._solver.close()
            self._solver = None
        self._cols = self._obj = self._rows = self._matrix = self._sos = self._matrix_starts = None
        self._variables.clear()
        self._constraints.clear()
        self._variable_blocks.clear()
        self._constraint_blocks.clear()
        self._bound = None
        self._compiler = None
        self._expression_thunks = {}

    def __enter__(self) -> PolarsEngine:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False


@contextmanager
def _clocked(timings: dict[str, float], phase: str) -> Iterator[None]:
    """Add the block's wall time onto ``timings[phase]`` — the diagnostics clocks.

    Cumulative, so a phase that runs again — a rebind's bind and build, every
    solve after the first — adds to its total the way the counters count.
    Recorded on failure too (the ``finally``): a build that died mid-phase
    spent its time there, and the clocks are advisory either way.
    """
    started = perf_counter()
    try:
        yield
    finally:
        timings[phase] = timings.get(phase, 0.0) + perf_counter() - started


#: Scratch columns of the expression reader. The spaces make them
#: unrepresentable as declared names, the same trick the compiler's own
#: scratch columns use.
_SOLUTION = '__solution value__'
_EXPRESSION_ROW = '__expression row__'


def _expression_frame(name: str, expr: plan.Expression, compiler: PolarsCompiler, values: pl.LazyFrame) -> pl.DataFrame:
    """Named expression *expr* evaluated at the primal *values* — ``(dims…, value)``.

    The tier is affine by construction, so a value is ``sum(coeff · value)``
    over the expression's term stream plus its constant part — the existing
    compiler reused wholesale, no second evaluation machinery. Each fragment
    from :meth:`PolarsCompiler.expression` is joined to the solver's primal
    vector (a term) or taken as it is (a constant part), aggregated to its own
    dims, and accumulated over the expression's coordinate product by the same
    carrier-and-left-join shape :meth:`PolarsEngine._build_constraint` gives a
    right-hand side.

    The frame answers the way a constraint over the same expression would:
    a coordinate a parameter does not cover contributes zero (SPEC §8), a
    coordinate where a term's variable is absent has no row (SPEC §7), and a
    variable-free expression is one row of ``value``. Dims come back in
    declaration order — an expression has no ``foreach`` to order them — and
    rows in label order over those dims, :meth:`Result.primal`'s promise.

    Raises:
        DataError: A divisor with no value where the expression divides —
            checked before any sum can read the null as zero.
    """
    context = f"named expression '{name}'"
    compiled = compiler.expression(expr, context)
    fragments = (*compiled.terms, *compiled.consts)
    union = {d for p in fragments for d in p.dims}
    dims = tuple(d.name for d in compiler.program.dimensions if d.name in union)

    divisors = sorted(plan.divisor_parameters(expr))
    if divisors:
        counts = pl.collect_all([p.frame.select(pl.col(p.value_column).null_count()) for p in fragments])
        undefined = sum(count.item() for count in counts)
        if undefined:
            raise DataError(f'{context}: {sparse_divisor_message(", ".join(divisors), undefined)}')

    restrictions = _absence_restrictions(list(compiled.terms))
    carrier = labels.frame(compiler, dims, None, _EXPRESSION_ROW, 0, restrictions).lazy()

    total = pl.lit(0.0, dtype=pl.Float64)
    for i, p in enumerate(fragments):
        column = f'__piece {i}__'
        if p.is_term:
            valued = p.frame.join(values, on='var_label', how='left').select(
                *p.dims, (pl.col('coeff') * pl.col(_SOLUTION)).alias(column)
            )
        else:
            valued = p.frame.select(*p.dims, pl.col('cval').alias(column))
        aggregated = (
            valued.group_by(p.dims).agg(pl.col(column).sum()) if p.dims else valued.select(pl.col(column).sum())
        )
        carrier = (
            carrier.join(aggregated, on=list(p.dims), how='left') if p.dims else carrier.join(aggregated, how='cross')
        )
        total = total + pl.col(column).fill_null(0.0)

    laid_out = carrier.select(_EXPRESSION_ROW, *dims, total.alias('value')).collect(engine='streaming')
    ordered = labels.in_position_order(laid_out, _EXPRESSION_ROW).drop(_EXPRESSION_ROW)
    return ordered.with_columns(pl.col(d).cast(pl.String) for d in dims if compiler.data.is_enum_encoded(d))


def _in_row_order(matrix: pl.DataFrame) -> pl.DataFrame:
    """*matrix* with ``row`` known to ascend — checked, not assumed.

    Ordered by construction, each constraint's share sorted and stacked in
    ascending row ranges. Worth *checking* because :func:`_row_starts` reads one
    run per row off this column and scatters the lengths by value: a row whose
    entries arrived in two runs would have the first overwritten and the CSR
    spans would be silently wrong. The correctness floor of the layout, not a
    speed choice — ``is_sorted`` is a linear scan and the sort never runs.
    """
    if matrix.height and not matrix['row'].is_sorted():
        return matrix.sort('row')
    return matrix


def _without_zeros(matrix: pl.DataFrame) -> pl.DataFrame:
    """*matrix* with the entries that cannot reach the answer removed.

    A coefficient of exactly zero states that a variable is not in a row, which
    is what an absent row already states — so the two say the same thing and
    only one of them costs the solver a nonzero to load and presolve away. A
    sparse parameter reaches here as absence and never builds a term (SPEC §8);
    a parameter that spells its zeros out reaches here as this, and on a table
    that is mostly zeros it is most of the matrix.

    **A pruned share can no longer say which rows had terms**, and a row whose
    every coefficient is zero still asserts something — ``0 >= 10`` is
    infeasible — so it keeps its place and its bounds and simply owns no
    entries. :meth:`~PolarsEngine._matrix_share` reads that row set off each
    frame before pruning it and hands it on; deriving it from the pruned share
    instead would drop such a row and turn an infeasible model into a feasible
    one.

    Nulls cannot be here: a null coefficient is an undefined divisor and
    :meth:`~PolarsEngine._refuse_undefined_divisors` has already refused the
    build over it, which is why the comparison can be a bare ``!= 0``.
    """
    return matrix.filter(pl.col('coeff') != 0)


def _pruned(matrix: pl.DataFrame) -> pl.DataFrame:
    """*matrix* without its zeros — unchanged, and not rechunked, when it has none.

    Behind a probe, like everything else in :meth:`PolarsEngine._matrix_share`:
    filtering leaves a chunked frame and the ``shift(1)`` probes downstream pay
    at every boundary (#576), so a share with no zero to drop must not pay a
    rechunk to discover that. A pruned frame can no longer say which rows had a
    term, so a caller that needs that answer reads it off the frame it passes
    in, before this runs — and only when it must: reading it unconditionally
    costs a hash of the *unaggregated* stack on every model, which is the whole
    of a wide build's regression.
    """
    if not matrix.select(pl.col('coeff').eq(0).any()).item():
        return matrix
    return _without_zeros(matrix).rechunk()


def _rows_of(matrix: pl.DataFrame, before: pl.Series | None) -> pl.Series:
    """The rows that had a term: *before* where a prune took the answer away."""
    return matrix.get_column('row').unique() if before is None else before


def _row_starts(ordered: pl.DataFrame, row_count: int) -> Any:
    """Each row's first entry in the row-ordered *ordered* — CSR's own index.

    Run-length, scatter, cumulative sum — robust to the model's shape where the
    alternatives are not: ``bincount`` pays per entry — several times rle's
    cost on a large matrix (#550) — and ``searchsorted`` per row times log
    entries.
    Computed here so ``row`` can then be dropped, since every consumer either
    slices by these starts or asks
    :meth:`~lpspec.relational.sinks.tables.ModelTables.matrix_block` to spell
    the labels back out.

    The kept matrix is then **rechunked, once**: a sink slices it per row
    block, and against a chunked frame every block's ``to_numpy`` is a
    gather-copy where against one contiguous buffer it is a view (#550).
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


def _absence_restrictions(terms: Sequence[TermFragment]) -> list[Presence]:
    """The presence frames a constraint's rows have to be contained in.

    Absence propagates into a comparison and drops the row (v1
    ``convention.rst`` §6, §12): ``x + y >= 10`` where ``y`` is masked is not
    ``x >= 10``, it is no constraint at all.

    Only *variable* absence counts — a sparse parameter's missing rows mean a
    zero coefficient (SPEC §8) — which is why the fragment carries
    :attr:`TermFragment.presence` separately from its frame, and why this reads
    that. A fragment with nothing to restrict is skipped, an unmasked variable
    existing at every coordinate of its foreach.

    *Having* no dims is not *having nothing to restrict*: a masked scalar
    variable restricts every row of every constraint naming it, all or nothing.
    Each restriction leaves with its key spelled out — the fragment's dims
    where the presence implied them — since labelling cannot know the
    fragment it came from.
    """
    return [Presence(p.presence.frame, p.presence.keys(p.dims)) for p in terms if p.presence is not None]
