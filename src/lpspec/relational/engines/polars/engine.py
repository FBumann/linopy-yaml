"""Polars engine: fill the model frames, then hand them to a sink.

Owns the *assembly* — turning each declaration into rows of the four model
frames, and holding them until a sink drains them. Owns none of the three
questions it asks on the way: what the data is
(:mod:`lpspec.relational.engines.polars.binding`), what a query over it looks like
(:mod:`lpspec.relational.engines.polars.compiler`), which coordinate gets which solver index
(:mod:`lpspec.relational.engines.polars.labels`). The lane is described in
docs/about/architecture.md.

The two registries it does own are the ones that fill *during* assembly — the
variable and constraint frames — because a declaration built later has to see
what earlier ones produced. Everything binding produced is frozen by contrast,
which is what :class:`~lpspec.relational.engines.polars.binding.BoundSources` says.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, get_args

import polars as pl
from math_spec import program

from lpspec.errors import (
    DataError,
    LpspecError,
    null_bounds_message,
    sparse_divisor_message,
    uncovered_constant_message,
    unknown_name_message,
)
from lpspec.relational import sinks
from lpspec.relational.engines.polars import labels
from lpspec.relational.engines.polars.binding import BoundSources, bind
from lpspec.relational.engines.polars.compiler import PolarsCompiler
from lpspec.relational.engines.polars.fragments import Presence, TermFragment, constant_scalar, join_on
from lpspec.relational.result import KEEPS, ConstraintRow, Diagnostics, Keep, Result, unknown_keep_message
from lpspec.relational.sinks.tables import SENSE

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from math_spec.program import ObjectiveSense
    from polars._typing import MaintainOrderJoin


#: The frames a sink reads, as schemas. Stated here because the engine
#: is what fills them and an empty model still has to have them.
_COLS = ('lb', 'ub', 'vtype')
_OBJ = ('col', 'coeff')
_QUAD = ('col_l', 'col_r', 'coeff')
_ROWS = ('row', 'sense', 'rhs')
_MATRIX = ('row', 'col', 'coeff')
_QMATRIX = ('row', 'col_l', 'col_r', 'coeff')
_SOS = ('set', 'type', 'col', 'weight', 'big_m')

#: The dtype of each of those columns. ``vtype`` is an ``Enum`` over the
#: variable types the plan declares, rather than a string: it holds one word
#: per column and the same handful of words for the whole model, so as a string
#: it stores that word once per row, where an Enum stores a code: on a wide
#: model that is most of the ``cols`` frame (#189). The Enum also makes the
#: vocabulary
#: explicit, so a fourth variable type added to
#: :data:`~math_spec.program.VariableType` and not reaching here fails
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
    'sense': SENSE, 'vtype': pl.Enum(get_args(program.VariableType)),
    'set': pl.Int32, 'type': pl.UInt8, 'weight': pl.Int32, 'big_m': pl.Float64,
    'col_l': pl.Int32, 'col_r': pl.Int32,
}  # fmt: skip


@dataclass
class _Measured:
    """What one build measured about itself, and a rebuild replaces wholesale.

    Separate from :class:`BuiltModel` because it outlives it: ``close()``
    releases the frames and :meth:`PolarsEngine.diagnostics` still answers, so
    everything here is a count or a small frame rather than a read of the model.

    Most of it is taken as it is measured, so a build that raises still reports
    what it got to. The three **sizes** are the exception: :class:`_Assembly`
    writes them once it has finished, which leaves them at zero after a raise
    rather than at a partial count of a model no engine holds.
    """

    #: ``name -> (coordinates, rows)`` for each parameter bound short of the
    #: coordinates its dims reach. Summarised at bind rather than read at
    #: :meth:`PolarsEngine.diagnostics`, which answers after the frames are
    #: released.
    sparse: dict[str, tuple[int, int]] = field(default_factory=dict)
    #: ``name -> rows not built``, because every term they had vanished.
    #: Empty for a model whose every declared row reached the solver.
    omitted: dict[str, int] = field(default_factory=dict)
    #: ``name -> (smallest, largest)`` coefficient magnitude, per constraint
    #: block. Taken as each share is built, where the numbers are still in
    #: cache, rather than off the assembled matrix the model releases.
    coefficients: dict[str, tuple[float, float]] = field(default_factory=dict)
    objective_range: tuple[float, float] | None = None
    columns: int = 0
    rows: int = 0
    #: Entries in the matrix, kept as a count rather than read off the frame:
    #: the frames go when the model is released and this is what a caller
    #: asking how big it *was* is asking for.
    nonzeros: int = 0


@dataclass(frozen=True)
class BuiltModel:
    """One build's product: the frames a sink drains, and what reads them back.

    A value, because a build is finished when it exists — which is what makes
    releasing it one assignment (:meth:`PolarsEngine.close`) and makes "has
    this engine got a model" one question rather than seven.

    The two registries are the ones that filled *during* assembly, since a
    declaration built later has to see what earlier ones produced; they are
    frozen here along with everything else, and the compiler holds the same
    ``variables`` dict rather than a copy.
    """

    program: program.Program
    bound: BoundSources
    compiler: PolarsCompiler
    #: ``name -> deferred plan expression``, one per declared named expression.
    #: Thunks, never plans: a build lowers none of them (the rules for named
    #: expressions), and a solve turns each into a reader that lowers on its
    #: first call (:meth:`PolarsEngine._expression_readers`).

    #: One :class:`~lpspec.relational.engines.polars.labels.Labelled` per
    #: declaration, one map per label space. Columns and rows are numbered
    #: independently and a model may name a variable and a constraint alike,
    #: so one map keyed by name would hand the primal reader a row block.
    variables: dict[str, labels.Labelled]
    constraints: dict[str, labels.Labelled]

    cols: pl.DataFrame
    obj: pl.DataFrame
    #: The objective's quadratic part, one row per *unordered pair* of columns:
    #: ``coeff`` is the coefficient of ``x[col_l] · x[col_r]`` in the objective
    #: as written, and never half of it. Each sink converts into its own
    #: spelling of the same form, which is three different spellings
    #: (:class:`~lpspec.relational.sinks.tables.ModelTables`). Empty for every
    #: model with an affine objective, which is most.
    quad: pl.DataFrame
    #: The quadratic part of every quadratic constraint row, one row per
    #: ``(row, unordered pair)``. Its rows are the **tail** of the label space
    #: (:func:`_linear_first`), so a sink can take them as a slice rather than
    #: hunting for them among the linear ones.
    qmatrix: pl.DataFrame
    rows: pl.DataFrame
    matrix: pl.DataFrame
    sos: pl.DataFrame
    matrix_starts: Any

    column_count: int
    row_count: int
    objective_constant: float
    objective_sense: ObjectiveSense | None

    def tables(self) -> sinks.ModelTables:
        """What every sink reads, and no more."""
        return sinks.ModelTables(
            cols=self.cols,
            obj=self.obj,
            quad=self.quad,
            qmatrix=self.qmatrix,
            rows=self.rows,
            matrix=self.matrix,
            sos=self.sos,
            row_starts=self.matrix_starts,
            column_count=self.column_count,
            row_count=self.row_count,
            objective_sense=self.objective_sense,
            objective_constant=self.objective_constant,
        )


class _Assembly:
    """One build in progress: the mutable half, discarded once it has frozen.

    Every counter here is one a declaration *advances* — a variable claims the
    next run of columns, a constraint the next run of rows — so they cannot be
    on the frozen product and cannot be recomputed from it. :meth:`run` turns
    the lot into a :class:`BuiltModel`, and nothing outside this class writes
    to any of it.
    """

    def __init__(self, program: program.Program, bound: BoundSources, measured: _Measured) -> None:
        self.program = program
        self.bound = bound
        self.measured = measured
        self.variables: dict[str, labels.Labelled] = {}
        self.constraints: dict[str, labels.Labelled] = {}
        self.compiler = PolarsCompiler(program, bound, self.variables)
        self.n_cols = 0
        self.n_rows = 0
        #: How many special-ordered sets have been numbered. Sets are dense
        #: ``0..n-1`` across the model, like columns and rows, so a sink names
        #: one by its number and two builds agree on which.
        self.n_sets = 0
        self.quad: pl.DataFrame | None = None
        self.obj_const = 0.0
        self.obj_sense: ObjectiveSense | None = None

    def run(self) -> BuiltModel:
        """Build every declaration, then freeze what they produced.

        Declarations build one at a time and concatenate at the end; their rows
        are independent, which is what lets the model be four frames rather
        than a graph.

        The matrix leaves in ``(row, col)`` order, as ``ModelTables`` promises
        its sinks. The stack already has it — each share leaves sorted and owns
        the next run of rows — so :func:`_in_row_order` *checks* with one
        linear scan rather than sorting the model's largest frame at the peak
        of the build. :func:`_row_starts` reads the CSR index off that order,
        after which ``row`` is dropped: 8 bytes per entry no sink reads.

        **``rows`` leaves in row order too**, which a solver reads as its own
        index — so :meth:`~lpspec.relational.sinks.tables.ModelTables.dense_rows`
        takes the frame's own vectors instead of scattering by label on every
        solve, and :attr:`~lpspec.relational.sinks.tables.ModelTables.structure`
        hashes the column two builds now agree on. The order is *checked*
        first, by the same rule labelling uses: a constant's left join is what
        usually loses it, and forcing that join to hold it is a bet on the
        model's shape rather than a fix — free on `fleet` and most of a build
        again on `dispatch`.
        """
        cols = [self._build_variable(v) for v in self.program.variables]
        sets = [self._build_sos(s, self.program.variable(s.variable)) for s in self.program.sos]
        built = [self._build_constraint(c) for c in _linear_first(self.program.constraints)]
        objective = self._build_objective(self.program.objective)

        ordered = _in_row_order(_stack([m for _, m, _ in built if m is not None], _MATRIX))
        matrix_starts = _row_starts(ordered, self.n_rows)
        matrix = ordered.select('col', 'coeff').rechunk()

        self.measured.columns = self.n_cols
        self.measured.rows = self.n_rows
        self.measured.nonzeros = matrix.height
        return BuiltModel(
            program=self.program,
            bound=self.bound,
            compiler=self.compiler,
            variables=self.variables,
            constraints=self.constraints,
            cols=_stack(cols, _COLS),
            obj=_stack([objective] if objective is not None else [], _OBJ),
            quad=_stack([] if self.quad is None else [self.quad], _QUAD),
            qmatrix=_in_row_order(_stack([q for _, _, q in built if q is not None], _QMATRIX)),
            rows=labels.in_position_order(_stack([r for r, _, _ in built], _ROWS), 'row'),
            matrix=matrix,
            sos=_stack(sets, _SOS),
            matrix_starts=matrix_starts,
            column_count=self.n_cols,
            row_count=self.n_rows,
            objective_constant=self.obj_const,
            objective_sense=self.obj_sense,
        )

    def _refuse_undefined_divisors(
        self, stacked: pl.DataFrame, name: str, *expressions: program.ExpressionNode
    ) -> None:
        """A null coefficient means a divisor had no value where the model divided.

        A quotient left-joins its divisor (:func:`_join_mul`), so a missing
        value leaves a null — and a term whose row was masked out, or whose
        numerator variable is absent, never gets this far. That is what keeps
        the refusal from becoming a wall on ordinary sparse data. Asked of the
        stack before any cell collapses, since ``sum`` reads a null as zero.
        """
        undefined = int(stacked.get_column('coeff').null_count())
        if undefined:
            params = sorted(program.divisor_parameters(*expressions))
            raise DataError(f'{name}: {sparse_divisor_message(", ".join(params), undefined)}')

    def _matrix_share(
        self, pieces: list[pl.LazyFrame], name: str, *expressions: program.ExpressionNode
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

    def _build_variable(self, v: program.VariableDeclaration) -> pl.DataFrame:
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
        start = self.n_cols
        labelled = labels.frame(self.compiler, v.dims, v.where, 'var_label', start)
        self.n_cols = start + labelled.height
        self.variables[v.name] = labels.Labelled(labelled.lazy(), start, labelled.height)

        bounded = labels.in_position_order(
            self.compiler.bounds(labelled.lazy(), v)
            .select('var_label', pl.col('lb').cast(pl.Float64), pl.col('ub').cast(pl.Float64))
            .collect(engine='streaming'),
            'var_label',
        )
        cols = bounded.select('lb', 'ub', pl.lit(v.variable_type, dtype=_DTYPES['vtype']).alias('vtype'))

        bad = cols.filter(pl.col('lb').is_null() | pl.col('ub').is_null()).height
        if bad:
            raise DataError(null_bounds_message(v.name, bad))
        return cols

    def _build_sos(self, s: program.SosDeclaration, v: program.VariableDeclaration) -> pl.DataFrame:
        """One declaration's sets as ``(set, type, col, weight, big_m)``, over *v*.

        Builds no column and no row: a set names columns the variable already
        made. It runs after every variable and before any constraint, which is
        when what it reads exists and before rows a constraint may drop.

        **A set and a weight are the two halves of a coordinate's position in
        the declared product**, split at the ``over`` dim — the row-major rule
        (:func:`~lpspec.relational.engines.polars.labels.row_major`) numbering
        every other index in the lane, by two divisions rather than by reading
        a dim's ordinal per member. A position is the label itself where the
        variable dropped nothing; a masked one reads the ordinals and
        renumbers the sets densely (#520, #687). A member's weight is its
        coordinate's position in the declared order, so a masked-out
        coordinate leaves its neighbours adjacent rather than leaving a hole.

        **The stream leaves grouped by set and ascending in weight**, which is
        what lets a sink read a set's edges off the neighbouring row rather
        than aggregating. The order is verified and the sort runs only where
        members genuinely interleave — on **both** columns, because here a
        position is a whole set and a sort that reordered ties would be a set
        whose members arrive out of weight order.

        ``big_m`` rides along per member, at ``inf`` where the block declared
        none — the one thing here no sink taking a set natively reads.
        """
        held = self.variables[s.variable]
        cardinality = self.compiler.data.cardinality
        stride = math.prod(cardinality[d] for d in v.dims[v.dims.index(s.over) + 1 :])
        span = cardinality[s.over] * stride

        if v.where is None:
            frame = pl.select(pl.int_range(held.height, dtype=pl.Int64).alias('#position')).lazy()
            place, col = pl.col('#position'), (pl.col('#position') + held.start)
        else:
            frame = held.frame
            place, col = labels.row_major(self.compiler, v.dims, self.compiler.ordinal_of), pl.col('var_label')
        placed = frame.select(
            ((place // span) * stride + place % stride).alias('#set position'),
            ((place // stride) % cardinality[s.over] + 1).cast(_DTYPES['weight']).alias('weight'),
            col.cast(_DTYPES['col']).alias('col'),
        ).collect(engine='streaming')

        position = pl.col('#set position')
        grouped = placed if placed.get_column('#set position').is_sorted() else placed.sort('#set position', 'weight')
        dense = position if v.where is None else (position != position.shift(1)).fill_null(True).cum_sum() - 1
        built = grouped.select(
            (dense + self.n_sets).cast(_DTYPES['set']).alias('set'),
            pl.lit(s.sos_type, dtype=_DTYPES['type']).alias('type'),
            'col',
            'weight',
            pl.lit(float('inf') if s.big_m is None else s.big_m, dtype=_DTYPES['big_m']).alias('big_m'),
        )
        if built.height:
            self.n_sets = built.item(-1, 'set') + 1
        return built

    def _build_constraint(
        self, c: program.ConstraintDeclaration
    ) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame | None]:
        """One constraint as its ``rows``, its share of the matrix, and its quadratic share.

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
        what survived, not what it declared (#561). A purely quadratic row has
        no linear entries at all, so an empty share is not a missing one — what
        decides whether a row is built is whether *either* matrix has a term.
        """
        quadratic = program.declares_quadratic(c)
        lhs = self.compiler.expression(c.lhs, f"constraint '{c.name}' lhs", quadratic=quadratic)
        rhs = self.compiler.expression(c.rhs, f"constraint '{c.name}' rhs", quadratic=quadratic)
        terms = [(p, 1.0) for p in lhs.terms] + [(p, -1.0) for p in rhs.terms]
        quads = [(p, 1.0) for p in lhs.quads] + [(p, -1.0) for p in rhs.quads]
        consts = [(p, 1.0) for p in rhs.consts] + [(p, -1.0) for p in lhs.consts]
        restrictions = _absence_restrictions([p for p, _ in (*terms, *quads)])
        start = self.n_rows
        declared = labels.declared_height(self.compiler, c.dims, c.where) if restrictions else None
        labelled = labels.frame(self.compiler, c.dims, c.where, 'row', start, restrictions)
        if declared is not None and declared > labelled.height:
            self.measured.omitted[c.name] = self.measured.omitted.get(c.name, 0) + declared - labelled.height
        self.n_rows = start + labelled.height
        frame = labelled.lazy()
        self.constraints[c.name] = labels.Labelled(frame, start, labelled.height)

        accumulated = pl.lit(0.0, dtype=pl.Float64)
        uncovered: pl.Expr | None = None
        carrier = frame
        for i, (p, sign) in enumerate(consts):
            column = f'__const {i}__'
            aggregated = constant_scalar(p).rename({'cval': column})
            carrier = join_on(carrier, aggregated, p.dims, 'left')
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
                names = ', '.join(sorted(program.parameters_of(c.lhs, c.rhs)))
                raise DataError(uncovered_constant_message(names, gaps, f"constraint '{c.name}'"))
            rows = rows.drop(gap_column)

        if not terms and not quads:
            none = pl.Series('row', [], dtype=_DTYPES['row'])
            rows, _, self.n_rows = self._drop_termless_rows(c.name, rows, _stack([], _MATRIX), none, start)
            return rows, None, None

        pieces = []
        carried_order: MaintainOrderJoin | None = 'left_right' if len(terms) == 1 else None
        for p, sign in terms:
            placed = join_on(frame, p.frame, p.dims, 'inner', maintain_order=carried_order)
            pieces.append(
                placed.select(
                    'row',
                    pl.col('var_label').cast(_DTYPES['col']).alias('col'),
                    (sign * pl.col('coeff')).cast(pl.Float64).alias('coeff'),
                )
            )
        matrix, term_rows = (
            self._matrix_share(pieces, f"constraint '{c.name}'", c.lhs, c.rhs)
            if pieces
            else (_stack([], _MATRIX), pl.Series('row', [], dtype=_DTYPES['row']))
        )
        qmatrix = self._quadratic_share(frame, quads, c)
        if qmatrix is not None:
            term_rows = pl.concat([term_rows, qmatrix.get_column('row').unique()]).unique()
        rows, matrix, self.n_rows = self._drop_termless_rows(c.name, rows, matrix, term_rows, start)
        spread = _magnitude_range(matrix.get_column('coeff'))
        if spread is not None:
            self.measured.coefficients[c.name] = spread
        if qmatrix is not None:
            qmatrix = qmatrix.filter(pl.col('row').is_in(rows.get_column('row')))
        return rows, matrix, qmatrix

    def _quadratic_share(
        self, frame: pl.LazyFrame, quads: list[tuple[TermFragment, float]], c: program.ConstraintDeclaration
    ) -> pl.DataFrame | None:
        """One constraint's quadratic entries as ``(row, col_l, col_r, coeff)``.

        The matrix share's twin, deliberately the *simple* version: it sorts
        and aggregates unconditionally where :meth:`_matrix_share` probes
        first, a model having few quadratic rows and each a handful of entries.
        Pairs are ordered by column index for :meth:`_objective_quadratic`'s
        reason.
        """
        if not quads:
            return None
        pieces = [
            join_on(frame, p.frame, p.dims, 'inner').select(
                'row',
                *_ordered_pair(),
                (sign * pl.col('coeff')).cast(pl.Float64).alias('coeff'),
            )
            for p, sign in quads
        ]
        stacked = pl.concat(pieces).collect(engine='streaming')
        self._refuse_undefined_divisors(stacked, f"constraint '{c.name}'", c.lhs, c.rhs)
        return _without_zeros(
            stacked.lazy()
            .group_by('row', 'col_l', 'col_r')
            .agg(pl.col('coeff').sum())
            .sort('row', 'col_l', 'col_r')
            .collect(engine='streaming')
        )

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
        self.measured.omitted[name] = rows.height - surviving.height
        remap = dict(zip(renumber.get_column('row'), renumber.get_column('__new__'), strict=True))
        rows = surviving.with_columns(pl.col('row').replace_strict(remap))
        matrix = matrix.with_columns(pl.col('row').replace_strict(remap))
        kept_frame = (
            self.constraints[name]
            .frame.filter(pl.col('row').is_in(kept))
            .with_columns(pl.col('row').replace_strict(remap))
        )
        self.constraints[name] = labels.Labelled(kept_frame, start, surviving.height)
        return rows, matrix, start + surviving.height

    def _build_objective(self, o: program.ObjectiveDeclaration | None) -> pl.DataFrame | None:
        """The objective as ``(col, coeff)``, or ``None`` if it has no terms.

        ``None`` in is the file that declares no objective at all, and it takes
        the same path out: the sense stays ``min`` and the constant ``0``, so
        the sink is handed a zero objective and answers whether the constraints
        can be met.

        This projection drops the dims, so a dim that arrived by broadcast puts
        several rows on one column and their **sum** is the coefficient.
        Nothing downstream computes it — the hand-off scatters with
        ``dense[at] = values``, which keeps the *last* write — so the aggregate
        here is what makes the objective the one the file wrote.

        The aggregate runs only when a column repeats, probed by ``n_unique``
        — the only sound probe here, the stack arriving unordered so adjacency
        proves nothing. Buying order to probe linearly is a dead end: the mul
        join's ``maintain_order`` holds the label order on some shapes and
        loses it on others differing only in data, so no static gate can say
        when the tax will pay, and paid for nothing it tripled an objective
        phase (#581). ``obj`` carries no order contract anyway.
        """
        if o is None:
            return None
        comp = self.compiler.expression(o.expression, 'objective', quadratic=True)
        for p in comp.consts:
            assert not p.dims, (
                f'objective constant part has dims {list(p.dims)} — Program.check refuses a '
                f'variable-free part of an objective that carries any'
            )
            self.obj_const += p.frame.select(pl.col('cval').sum()).collect().item() or 0.0
        self.obj_sense = o.sense
        self.quad = self._objective_quadratic(comp.quads, o.expression)
        if not comp.terms:
            return None
        pieces = [
            p.frame.select(pl.col('var_label').cast(_DTYPES['col']).alias('col'), pl.col('coeff')) for p in comp.terms
        ]
        stacked = pl.concat(pieces).collect(engine='streaming')
        self._refuse_undefined_divisors(stacked, 'objective', o.expression)
        if stacked.get_column('col').n_unique() != stacked.height:
            stacked = stacked.lazy().group_by('col').agg(pl.col('coeff').sum()).collect(engine='streaming')
        objective = _without_zeros(stacked)
        self.measured.objective_range = _magnitude_range(objective.get_column('coeff'))
        return objective

    def _objective_quadratic(
        self, quads: tuple[TermFragment, ...], expression: program.ExpressionNode
    ) -> pl.DataFrame | None:
        r"""The objective's quadratic part as ``(col_l, col_r, coeff)``, or ``None``.

        **One row per unordered pair**, at the coefficient the file wrote:
        ``coeff · x[col_l] · x[col_r]``, whole and not halved. Each sink spells
        that differently — a Hessian is :math:`\frac12 x^\top Q x`, the LP
        section is divided by two, Gurobi takes :math:`x^\top Q x` — so what
        leaves here is the algebra and the conversion is theirs. The aggregate
        runs only when a pair repeats, probed like the linear half.

        **It leaves sorted, and that is a contract.** The join hands pairs back
        in whatever order the data made, so two builds disagreed and
        :attr:`~lpspec.relational.sinks.tables.ModelTables.structure` read a
        moved *coefficient* as a moved pattern. Unconditionally, unlike the
        matrix: the frame is one row per pair and nothing says it arrives
        sorted.
        """
        if not quads:
            return None
        pieces = [p.frame.select(*_ordered_pair(), pl.col('coeff')) for p in quads]
        stacked = pl.concat(pieces).collect(engine='streaming')
        self._refuse_undefined_divisors(stacked, 'objective', expression)
        if stacked.select(pl.struct('col_l', 'col_r').n_unique()).item() != stacked.height:
            stacked = stacked.lazy().group_by('col_l', 'col_r').agg(pl.col('coeff').sum()).collect(engine='streaming')
        return _without_zeros(stacked.sort('col_l', 'col_r'))


def _no_built_model(doing: str) -> str:
    """Why there is no model *doing*, in the two ways that happens."""
    return (
        f'there is no built model {doing}: it was closed, or a rebind raised and released '
        f'it rather than leaving half of one behind. Build it again — rebind() with data it can '
        f'bind, or build() from the start.'
    )


class PolarsEngine:
    """Build a :class:`Program` into polars frames, then sink it."""

    def __init__(self) -> None:
        #: The build, or ``None`` where there is not one — closed, released by
        #: a rebind that raised, or never run. One field rather than a frame
        #: each, because a build is finished when it exists.
        self._built: BuiltModel | None = None
        #: What the last build measured about itself. Outlives ``_built``,
        #: since :meth:`diagnostics` answers after :meth:`close`.
        self._measured = _Measured()
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
        #: unless it had no concept of a set the model declares. A fact about a
        #: *solve*, so a rebuild does not clear it.
        self._sink_columns = 0
        self._sink_rows = 0
        #: Wall seconds each phase has spent, cumulatively — what
        #: :meth:`diagnostics` reports as ``timings``. Time spent is a fact
        #: about what ran, so a rebuild adds to it rather than clearing it.
        self._timings: dict[str, float] = {}

    @property
    def _model(self) -> BuiltModel:
        """The built model, or why there is not one."""
        if self._built is None:
            raise LpspecError(_no_built_model('to hand over'))
        return self._built

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def build(
        self,
        program: program.Program,
        sources: Mapping[str, Any],
    ) -> None:
        """Bind *sources*, then build every declaration into the model frames.

        The compiler comes after binding, two of its answers being read off the
        data — a dim's size, whether a parameter is boolean. What each
        declaration contributes, and in what order, is :meth:`_Assembly.run`.

        **What a zero coefficient states, absence already states**, so each
        share arrives already pruned of them (:func:`_without_zeros`) and the
        objective of its own. Sparsity that arrives as absence never becomes a
        term at all; sparsity spelled out as zeros used to become one, and a
        solver's own load is most of what a hand-off costs.

        **A second call rebuilds over the same object**, which is what
        ``rebind`` is. The previous build is released *before* this one starts,
        so a driver that re-solves in a loop stays at one model's peak; what
        the loaded solver holds survives as the digest it recorded at its load.
        A build that raises leaves no model at all rather than half of one, and
        ``diagnostics()`` answers from what :class:`_Measured` had by then —
        the bind's own numbers, and no size.

        *expressions* maps each declared named expression to a thunk producing
        its plan expression. None is called here — a build pays nothing for a
        declared expression (the rules for named expressions) — they become the
        deferred readers a solve's result hands out
        (:meth:`_expression_readers`).
        """
        self._built = None
        self._measured = _Measured()
        with _clocked(self._timings, 'bind'):
            bound = bind(program, sources)
        self._measured.sparse = _short_parameters(program, bound)
        assembly = _Assembly(program, bound, self._measured)
        with _clocked(self._timings, 'build'):
            self._built = assembly.run()

    # ------------------------------------------------------------------
    # sinks — see relational/sinks/; the engine only supplies the frames
    # ------------------------------------------------------------------

    def row(self, name: str, coordinate: Mapping[str, Any]) -> ConstraintRow:
        """One built constraint row, spelled back out. See :meth:`~lpspec.api.BoundModel.row`.

        Three lookups against frames the build already keeps, and no scan of
        the matrix: the constraint's own coordinate frame carries the global
        row index, ``row_starts`` says where that row's entries lie, and each
        variable's frame carries the global column index its terms point at.

        Every one of those is a **positional** take rather than a filter, so
        the cost is the row's own width rather than the model's: ``rows`` is
        dense and in row order, and a declaration's labels are dense from its
        block's start (:class:`~lpspec.relational.engines.polars.labels.Labelled`),
        so a label's position in its frame is the label minus that start.
        """
        if self._built is None:
            raise LpspecError(_no_built_model(f"to read '{name}' out of"))
        model = self._built
        if name not in model.constraints:
            raise KeyError(unknown_name_message('constraint', name, sorted(model.constraints)))

        at, ordered = self._row_index(name, coordinate)
        starts = model.matrix_starts
        entries = model.matrix.slice(int(starts[at]), int(starts[at + 1] - starts[at]))
        stated = model.rows.slice(at, 1)
        return ConstraintRow(
            name=name,
            coordinate=ordered,
            terms=self._named_terms(entries),
            sense=str(stated.item(0, 'sense')),
            rhs=float(stated.item(0, 'rhs')),
        )

    def _row_index(self, name: str, coordinate: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        """The global row index constraint *name* built at *coordinate*, and that coordinate in dim order.

        The coordinate has to name **every** dim of the declaration: a partial
        one matches a set of rows, and a verb that quietly answered about the
        first of them would be reporting one row as if it were the block. It
        comes back ordered by the declaration rather than by the keywords it
        was written with, so one row has one spelling however it was asked
        for.

        Raises:
            LpspecError: The coordinate names dims the declaration does not,
                or holds a label the dimension cannot (a string against an
                integer dim, a stranger against an ``Enum``), or names dims
                correctly and matches no row the build produced — which is a
                row masked out by ``where`` or dropped for having no terms,
                and is itself the answer to why a model says nothing here.
        """
        frame = self._model.constraints[name].frame
        schema = frame.collect_schema()
        dims = tuple(d for d in schema.names() if d != 'row')
        if set(coordinate) != set(dims):
            raise LpspecError(
                f"constraint '{name}' is declared over {list(dims)}, and a row is read at all of them "
                f'— got {sorted(coordinate)}. A row is one coordinate: name every dim once, and no dim '
                'the declaration does not have.'
            )
        ordered = {d: coordinate[d] for d in dims}
        predicates = [pl.col(d) == self._label(name, d, v, schema[d]) for d, v in ordered.items()]
        found = frame.filter(predicates).collect() if predicates else frame.collect()
        if not found.height:
            raise LpspecError(
                f"constraint '{name}' built no row at {ordered}. Either a `where` masked the "
                'coordinate out, every term it had was absent, or the labels are not ones the '
                'dimension holds — and which of those it is, is what diagnostics() reports as an '
                'omission.'
            )
        return int(found.item(0, 'row')), ordered

    @staticmethod
    def _label(name: str, dim: str, value: Any, dtype: pl.DataType) -> pl.Expr:
        """*value* as a literal of *dim*'s own type, or a refusal naming what it is not.

        The cast **is** the check, so the dimension's own type answers both
        questions at once: a string against an integer dim and a stranger
        against an ``Enum`` are one failure, and neither reaches polars as a
        comparison it can only report in its own vocabulary.
        """
        try:
            return pl.lit(pl.Series([value], dtype=dtype).item(0), dtype=dtype)
        except (pl.exceptions.PolarsError, TypeError, OverflowError) as refused:
            raise LpspecError(
                f"constraint '{name}' is declared over '{dim}', which holds {dtype}, and {value!r} is "
                f'not one of its labels. Read the row at a label the dimension has.'
            ) from refused

    def _named_terms(self, entries: pl.DataFrame) -> pl.DataFrame:
        """``(variable, coordinate, coefficient)`` for one row's matrix entries.

        A column index is global and each declaration owns a contiguous, dense
        run of them (:class:`~lpspec.relational.engines.polars.labels.Labelled`),
        so which variable a term belongs to is a range test rather than a join
        against the whole column space, and a
        term's place in its own declaration's frame is its label minus that
        block's start — a positional take, not a search. Only the
        declarations this row actually touches are read, and each of them only
        as wide as the row is.

        ``coordinate`` is rendered rather than spread across dim columns
        because one row's terms may come from variables with *different* dims,
        which no single frame schema holds. It carries the **labels alone**,
        in the declaration's dim order — linopy's ``p[1, wind]`` bracket, so a
        reader arriving from there reads a term the way they already do, and
        so a printed row does not repeat a dim name once per term.

        The terms leave in the order the entries arrived in, which is the
        solver's own column order: the join says so rather than happening to
        hold, since polars promises no order without being asked.
        """
        wanted = entries['col'].to_numpy()
        named = []
        for variable, held in self._model.variables.items():
            inside = wanted[(wanted >= held.start) & (wanted < held.start + held.height)]
            if not inside.size:
                continue
            frame = held.frame
            dims = [d for d in frame.collect_schema().names() if d != 'var_label']
            at = pl.Series('#position', inside - held.start, dtype=pl.UInt32)
            picked = frame.select(pl.col('var_label'), *(pl.col(d) for d in dims)).select(pl.all().gather(at))
            rendered = pl.concat_str([pl.col(d).cast(pl.String) for d in dims], separator=', ') if dims else pl.lit('')
            named.append(
                picked.select(
                    pl.col('var_label').alias('col'),
                    pl.lit(variable).alias('variable'),
                    rendered.alias('coordinate'),
                ).collect()
            )
        labelled = (
            pl.concat(named)
            if named
            else pl.DataFrame(schema={'col': pl.Int64, 'variable': pl.String, 'coordinate': pl.String})
        )
        return (
            entries.with_columns(pl.col('col').cast(pl.Int64))
            .join(labelled.with_columns(pl.col('col').cast(pl.Int64)), on='col', how='left', maintain_order='left')
            .select('variable', 'coordinate', pl.col('coeff').alias('coefficient'))
        )

    def write(self, path: str | Path) -> None:
        """Stream the built model to *path*, in the format its suffix names.

        A construct the format has no section for is refused here, the way the
        solve path refuses one a solver cannot ingest
        (:func:`~lpspec.relational.sinks.ingestible`) and with the sentence
        ``check(model, sink=...)`` would have given. Writing it anyway would
        hand back a file that parses, solves, and is a different model: the
        MPS writer spells no quadratic term, so those rows would arrive empty
        (#942).

        Raises:
            ValueError: A suffix nothing writes.
            LpspecError: A construct this format cannot spell.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        chosen = sinks.writer(suffix)
        tables = self._model.tables()
        if (refused := sinks.refusal(self._model.program, suffix)) is not None:
            raise LpspecError(refused)
        with _clocked(self._timings, 'write'):
            chosen.write(tables, path)

    def solve(
        self,
        solver_name: str = 'highs',
        *,
        solver_options: Mapping[str, Any] | None = None,
        batch_rows: int | None = None,
        keep: Keep = 'solver',
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
        (:func:`~lpspec.relational.sinks.ingestible`): a sink with no
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
            keep: How much of the session this solve may keep — one of
                :data:`~lpspec.relational.result.KEEPS`. A preference, not a
                guarantee: a model whose structure moved is loaded again
                whatever was asked, and
                :attr:`~lpspec.relational.result.Result.kept` reports what
                happened. ``nothing`` is held to structurally, the held solver
                being closed before the load decision, so the fresh one has
                nothing to begin from whatever a member squirrels away.

        Returns:
            The solution, holding this engine and the build it answered.

        Raises:
            LpspecError: A *keep* outside
                :data:`~lpspec.relational.result.KEEPS`.
        """
        if keep not in KEEPS:
            raise LpspecError(unknown_keep_message(keep))
        built = self._model.tables()
        with _clocked(self._timings, 'handoff'):
            tables = sinks.ingestible(solver_name, built, self._model.program)
            self._sink_columns = tables.column_count - built.column_count
            self._sink_rows = tables.row_count - built.row_count
            if keep == 'nothing' and self._solver is not None:
                self._solver.close()
                self._solver = None
            held = self._solver
            self._solver = sinks.loaded(held, solver_name, tables, batch_rows, solver_options)
            kept: Keep = keep if self._solver is held else 'nothing'
            if kept == 'solver':
                self._solver.forget()
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
            _kept=kept,
            _expressions=self._expression_readers(answer.primal),
            _no_duals=None
            if answer.dual is not None
            else _no_duals_message(
                self._discrete(),
                answer.status.termination_condition,
                sets=self._reformulated_sets(tables is not built),
                quadratic_rows=self._quadratic_constraints(),
            ),
        )

    def diagnostics(self) -> Diagnostics:
        """What this build and its solves did that the answer does not show.

        Answerable after :meth:`close`: every field is a count, a clock or a
        small frame this keeps, not a read of the model it releases.
        """
        return Diagnostics(
            columns=self._measured.columns,
            rows=self._measured.rows,
            nonzeros=self._measured.nonzeros,
            sink_columns=self._sink_columns,
            sink_rows=self._sink_rows,
            omissions=pl.DataFrame(
                {'constraint': list(self._measured.omitted), 'rows_not_built': list(self._measured.omitted.values())},
                schema={'constraint': pl.String, 'rows_not_built': pl.UInt32},
            ),
            coefficient_range=pl.DataFrame(
                {
                    'constraint': list(self._measured.coefficients),
                    'smallest': [low for low, _ in self._measured.coefficients.values()],
                    'largest': [high for _, high in self._measured.coefficients.values()],
                },
                schema={'constraint': pl.String, 'smallest': pl.Float64, 'largest': pl.Float64},
            ),
            sparse_parameters=pl.DataFrame(
                {
                    'parameter': list(self._measured.sparse),
                    'coordinates': [reach for reach, _ in self._measured.sparse.values()],
                    'rows': [rows for _, rows in self._measured.sparse.values()],
                    'missing': [reach - rows for reach, rows in self._measured.sparse.values()],
                },
                schema={
                    'parameter': pl.String,
                    'coordinates': pl.UInt64,
                    'rows': pl.UInt64,
                    'missing': pl.UInt64,
                },
            ),
            objective_range=self._measured.objective_range,
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
        model = self._model
        program = model.program

        def rows(values: pl.Series | None) -> dict[str, pl.LazyFrame]:
            if values is None:
                return {}
            return {c.name: self._laid_out(model.constraints[c.name], c.dims, values) for c in program.constraints}

        return (
            {v.name: self._laid_out(model.variables[v.name], v.dims, primal) for v in program.variables}
            if primal is not None
            else {},
            rows(dual),
            rows(activity),
        )

    def _expression_readers(self, primal: pl.Series | None) -> dict[str, Callable[[], pl.DataFrame]]:
        """One deferred reader per declared named expression — nothing compiled yet.

        Deferral is the contract (the rules for named expressions): a closure **compiles** its
        expression when it is first called, so a solve over fifty declared
        expressions that reads none pays for a dict of closures. The lowering
        is already done — it is the plan's, and eager, so a named expression
        outside the language is refused when the file is read rather than when
        somebody happens to read that one. What stays deferred is the
        compilation, which is the part that touches data. Each captures
        a snapshot the result *owns* — the program, the bound data, a copy of
        this build's variable-frame registry and the solver's primal vector —
        so it keeps answering after a rebind or ``close()`` the way every
        other reader does, at the cost of keeping those frames alive.
        """
        if primal is None:
            return {}
        model = self._model
        compiler = PolarsCompiler(model.program, model.bound, dict(model.variables))
        values = pl.DataFrame(
            {'var_label': pl.int_range(primal.len(), dtype=pl.Int64, eager=True), _SOLUTION: primal}
        ).lazy()

        def reader(name: str, expression: program.ExpressionNode) -> Callable[[], pl.DataFrame]:
            return lambda: _expression_frame(name, expression, compiler, values)

        return {name: reader(name, e) for name, e in model.program.expressions.items()}

    def _laid_out(self, held: labels.Labelled, dims: tuple[str, ...], values: pl.Series) -> pl.LazyFrame:
        """One declaration's coordinates in label order, beside its share of *values*.

        The order is not re-established here, because it was never lost:
        :func:`labels.frame` numbers a sorted frame and hands back a
        label-ascending one, and the solver's vector is positional in the same
        index. The share is attached as a column rather than concatenated as a
        frame, so a mismatched length raises instead of padding with nulls.

        **Dim columns leave in ``String``**, where the build holds them as
        ``pl.Enum`` (#541): a returned frame is something a caller joins
        against their own data, and polars refuses ``Enum`` against ``String``
        with a message that names nothing about the cause. The cast sits
        inside this projection so the string column is produced once instead
        of widened from an Enum that also exists (#593).
        """
        labelled = held.frame.select(*dims).with_columns(held.share(values))
        return labelled.with_columns(pl.col(d).cast(pl.String) for d in _string_dims(self._model.bound, dims))

    def _discrete(self) -> list[str]:
        """The variables this model declared as anything but continuous."""
        return sorted(v.name for v in self._model.program.variables if v.variable_type != 'continuous')

    def _quadratic_constraints(self) -> list[str]:
        """The constraints this model declared as quadratic.

        The other reason an otherwise continuous model comes back without
        duals, and like the sets one it is a fact about the *model* rather than
        about the solve — so it is read off the program and not off the answer.
        """
        return sorted(c.name for c in self._model.program.constraints if program.declares_quadratic(c))

    def _reformulated_sets(self, reformulated: bool) -> list[str]:
        """The sets that reached the solver as binaries, if any did.

        What makes an otherwise continuous model come back without duals, and
        the one such reason no declaration shows: the model declares no
        integrality, and the sink added some.
        """
        return sorted(s.name for s in self._model.program.sos) if reformulated else []

    # ------------------------------------------------------------------

    def close(self) -> None:
        """Drop the built model. A :class:`Result` keeps its own frames.

        One assignment, because the build is one value: the four model frames,
        the label frames, ``BoundSources`` and the compiler that holds it all
        become unreachable together. A loaded solver goes first, being the one
        thing here that is not this process's memory.

        :meth:`diagnostics` still answers afterwards — what it reports is
        measurements, kept beside the model rather than inside it.
        """
        if self._solver is not None:
            self._solver.close()
            self._solver = None
        self._built = None

    def __enter__(self) -> PolarsEngine:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False


def _no_duals_message(
    discrete: Sequence[str],
    termination_condition: str,
    sets: Sequence[str],
    quadratic_rows: Sequence[str],
) -> str:
    """Why a solve that *did* leave values still has no duals.

    Integrality is decidable from the model, and naming the variable is
    actionable where "the solver reported none" is not.

    *sets* are the special-ordered sets a sink without the concept turned into
    binaries. They come first because a model that declared none of its own
    integrality would otherwise be told it is mixed-integer with nothing named
    — and because the fix is a different one: another sink, not a different
    model.

    *quadratic_rows* are the quadratic constraints, whose prices are off by
    default: asking for them puts the solve on the convex path, and a nonconvex
    row that solves without them fails with them. The one case here where
    nothing is wrong with the model.
    """
    if quadratic_rows and not discrete:
        names = ', '.join(f"'{n}'" for n in quadratic_rows)
        return (
            f"a quadratic constraint prices only under gurobi's QCPDual, which is off by default: "
            f'{names} {"is" if len(quadratic_rows) == 1 else "are"} quadratic. Asking for those '
            f'prices makes the solver take the convex path, so a nonconvex row that solves without '
            f'them fails with them — which is why this is yours to ask for rather than ours to '
            f"assume. Re-solve with solver_options={{'QCPDual': 1}} if the model is convex."
        )
    if sets:
        names = ', '.join(f"'{n}'" for n in sets)
        return (
            f'duals are undefined for a mixed-integer model, and this sink has no SOS concept, so '
            f'{names} reached it as binaries. Solve with a sink that takes a set natively (gurobi) '
            f'to keep the LP, or drop the set to price the relaxation.'
        )
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


def _string_dims(bound: BoundSources, dims: Sequence[str]) -> list[str]:
    """Those of *dims* the binder encoded as ``Enum`` — its string ones."""
    return [d for d in dims if bound.is_enum_encoded(d)]


def _ordered_pair() -> tuple[pl.Expr, pl.Expr]:
    """A quadratic pair canonicalised by column index, so ``x·y`` and ``y·x`` land in one row.

    Left unordered, a sink loads half the coefficient twice — right by
    accident on a symmetric Hessian, silently wrong in the LP section.
    """
    return (
        pl.min_horizontal('var_label', 'var_label_2').cast(_DTYPES['col_l']).alias('col_l'),
        pl.max_horizontal('var_label', 'var_label_2').cast(_DTYPES['col_r']).alias('col_r'),
    )


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


def _expression_frame(
    name: str, expr: program.ExpressionNode, compiler: PolarsCompiler, values: pl.LazyFrame
) -> pl.DataFrame:
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
    a coordinate a parameter does not cover contributes zero (the data-binding rules), a
    coordinate where a term's variable is absent has no row (the operator rules), and a
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

    divisors = sorted(program.divisor_parameters(expr))
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
        if p.kind != 'const':
            valued = p.frame.join(values, on='var_label', how='left').select(
                *p.dims, (pl.col('coeff') * pl.col(_SOLUTION)).alias(column)
            )
        else:
            valued = p.frame.select(*p.dims, pl.col('cval').alias(column))
        aggregated = (
            valued.group_by(p.dims).agg(pl.col(column).sum()) if p.dims else valued.select(pl.col(column).sum())
        )
        carrier = join_on(carrier, aggregated, p.dims, 'left')
        total = total + pl.col(column).fill_null(0.0)

    laid_out = carrier.select(_EXPRESSION_ROW, *dims, total.alias('value')).collect(engine='streaming')
    ordered = labels.in_position_order(laid_out, _EXPRESSION_ROW).drop(_EXPRESSION_ROW)
    return ordered.with_columns(pl.col(d).cast(pl.String) for d in _string_dims(compiler.data, dims))


def _short_parameters(program: program.Program, bound: BoundSources) -> dict[str, tuple[int, int]]:
    """Which parameters arrived short, and by how much: ``name -> (reach, rows)``.

    Arithmetic over two dicts binding already filled — a dimension's height and
    a parameter's — so it costs no pass over any source, which is what lets it
    run on every build rather than behind a flag. The reach is the product of
    the cardinalities because that is what "spans its dims" means; the check
    that no row is a duplicate or a stranger has already run, so the height
    *is* the number of coordinates covered.
    """
    short: dict[str, tuple[int, int]] = {}
    for p in program.parameters:
        if not p.dims:
            continue
        reach = 1
        for d in p.dims:
            reach *= bound.cardinality[d]
        rows = bound.parameter_rows[p.name]
        if rows < reach:
            short[p.name] = (reach, rows)
    return short


def _linear_first(constraints: tuple[program.ConstraintDeclaration, ...]) -> list[program.ConstraintDeclaration]:
    """*constraints* with the quadratic declarations last, order otherwise kept.

    **Quadratic is a property of a declaration, not of a row** — a constraint
    has one expression — so building those last puts their rows in a contiguous
    *tail* of the label space. Everything downstream stays a slice because of
    it: a solver holding linear rows in one object and quadratic rows in
    another concatenates two runs rather than scattering by label. A stable
    sort, so file order survives inside each half.
    """
    return sorted(constraints, key=program.declares_quadratic)


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


def _magnitude_range(coefficients: pl.Series) -> tuple[float, float] | None:
    """The smallest and largest ``|coefficient|`` in *coefficients*, or ``None`` if empty.

    Magnitudes rather than signed extremes, which is the question a solver's own
    ``Matrix range`` line answers and the one that makes the ratio meaningful:
    a row scaled by ``-1e9`` is as badly scaled as one scaled by ``1e9``, and
    the signed minimum of a matrix carrying both is not a scale at all.

    Exact zeros are gone by the time this runs (:func:`_without_zeros`,
    :meth:`PolarsEngine._drop_termless_rows`), so the smallest is a coefficient
    the solver will actually see.
    """
    if not coefficients.len():
        return None
    magnitudes = coefficients.abs()
    return float(magnitudes.min()), float(magnitudes.max())  # pyrefly: ignore[bad-argument-type]


def _without_zeros(matrix: pl.DataFrame) -> pl.DataFrame:
    """*matrix* with the entries that cannot reach the answer removed.

    A coefficient of exactly zero states that a variable is not in a row, which
    is what an absent row already states — so the two say the same thing and
    only one of them costs the solver a nonzero to load and presolve away. A
    sparse parameter reaches here as absence and never builds a term (the data-binding rules);
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
    alternatives are not: ``bincount`` pays per entry — 26 ms against rle's 7 ms
    at 10M entries over 100k rows (#550) — and ``searchsorted`` per row times
    log entries.
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
    zero coefficient (the data-binding rules) — which is why the fragment carries
    :attr:`TermFragment.presences` separately from its frame, and why this reads
    that. A fragment with nothing to restrict is skipped, an unmasked variable
    existing at every coordinate of its foreach.

    *Having* no dims is not *having nothing to restrict*: a masked scalar
    variable restricts every row of every constraint naming it, all or nothing.
    Each restriction leaves with its key spelled out — the fragment's dims
    where the presence implied them — since labelling cannot know the
    fragment it came from.
    """
    return [Presence(x.frame, x.keys(p.dims)) for p in terms for x in p.presences]
