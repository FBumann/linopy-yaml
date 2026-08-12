"""Build a `Program` into `ModelTables` through duckdb.

The duckdb twin of `relational/executor.py`, and a **drop-in at the sink seam**:
it hands back the same `sinks.ModelTables` the polars executor does, so
`lp_file`, `solver_direct`, the status codes and the result readers are
untouched and unaware. That is what makes the two comparable — the only thing
that differs between a `PolarsExecutor` build and a `DuckExecutor` build is
which engine filled the four frames.

Scope: the affine core — variables with bounds and masks, constraints over
sum/group_sum/translate, one objective. Piecewise expansion happens above this
layer, and the solution read-back below it, in `Engine`: both are written
against the plan and the label frames rather than against either engine.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from functools import reduce
from typing import TYPE_CHECKING, Any

import duckdb
import polars as pl
from duckdb import CoalesceOperator, ConstantExpression, Expression, FunctionExpression, SQLExpression

from lpspec.errors import DataError, LanguageError, null_bounds_message
from lpspec.relational import plan, sinks
from lpspec.relational.binding import BoundSources, bind
from lpspec.relational.engine import Engine
from lpspec.relational.engines.duck.compiler import (
    UNIT,
    DuckCompiler,
    Relation,
    TermFragment,
    _ordinal,
    col,
    matching,
    q,
    restrict_to,
    union_all,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

#: The four frames a sink reads, and their dtypes — both stated by
#: `sinks/tables.py`, which is what reads them. Shared with the polars engine
#: for the reason the frames are: a sink cannot see which engine filled them.
_COLS, _OBJ, _ROWS = sinks.COLS, sinks.OBJ, sinks.ROWS

#: What a `cols` table holds. `vtype` is not stored beside them — see `_vtypes`.
_BOUNDS = ('lb', 'ub')


def _ranked(order: Sequence[str], offset: int) -> Expression:
    """A row's position in *order*, counted from *offset*.

    ``ROW_NUMBER`` is the one construct here with no expression-API form —
    `Expression` has no `over`, and `DuckDBPyRelation.row_number` takes its
    window spec and its projection as SQL anyway. So the window is written out
    and the arithmetic around it is not: the names still go through :func:`q`,
    and the offset is a number rather than an identifier.
    """
    by = ', '.join(q(c) for c in order) or '1'
    return (SQLExpression(f'ROW_NUMBER() OVER (ORDER BY {by})') + ConstantExpression(offset - 1)).cast('BIGINT')


class _Labels(Mapping[str, 'pl.LazyFrame']):
    """Label relations, fetched out of duckdb only if a read-back asks.

    `Engine` reads a solution back by joining the solver's answer onto
    `(dims…, label)` frames, and states those as polars. Materialising every
    one at build time would put a second copy of the labels in this process —
    which is most of what choosing this engine was for. So the name is held
    and the frame is fetched on the first access, which for a caller that only
    writes an LP file never happens.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection, tables: dict[str, str], label: str) -> None:
        self._con = con
        self._tables = tables
        self._label = label
        self._frames: dict[str, pl.LazyFrame] = {}

    def __getitem__(self, name: str) -> pl.LazyFrame:
        """The frame, **in label order and in binding's dtypes** — read, not imposed.

        `Engine._read_back` stopped sorting once every labelling path produced
        an ordered frame. polars' paths do and verify it; a duckdb relation
        promises no order at all, and two of the three here are views over a
        cross join. So the order is asked for on the way out, where it is paid
        only by a caller that reads a solution back.

        The dtypes are binding's own: a string dimension crosses into duckdb as
        ``VARCHAR`` and comes back as ``String``, which is what
        `Engine._read_back` hands a caller from either engine (#541, #593).
        """
        if name not in self._frames:
            self._frames[name] = self._con.table(self._tables[name]).order(q(self._label)).pl().lazy()
        return self._frames[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tables)

    def __len__(self) -> int:
        return len(self._tables)

    def clear(self) -> None:
        """Drop the cached frames; the relations go with the connection."""
        self._frames.clear()


#: What a matrix block carries. No ``row``: which row an entry belongs to is
#: its position between two CSR offsets (`sinks.MatrixBlocks`).
_ENTRY = ('col', 'coeff')


def _stacked(
    con: duckdb.DuckDBPyConnection,
    tables: Sequence[str],
    columns: tuple[str, ...],
    keep: Expression | None = None,
) -> Relation | None:
    """*tables* as one relation of *columns*, in the order they were built.

    The model is a table per declaration, and both things that read it back —
    the three frames a sink takes whole, and one block of the matrix — read it
    the same way: stack, project, fetch. `None` for no tables at all, which is
    a model with nothing of that kind rather than an error.

    *keep* filters **before** the projection, so a block can select on ``row``
    and then not carry it.
    """
    if not tables:
        return None
    parts = [con.table(t) for t in tables]
    if keep is not None:
        parts = [p.filter(keep) for p in parts]
    projected = [p.select(*(col(c) for c in columns)) for p in parts]
    return union_all(projected[0], projected[1:])


def _fetch(rel: Relation | None, columns: tuple[str, ...]) -> pl.DataFrame:
    """*rel* as a frame, or the empty one a sink still has to be able to read.

    Empty is not nothing: a model may declare no objective term, and a sink
    still has to find the columns and dtypes it reads. `sinks.stack` states
    them, for both engines.
    """
    return rel.pl() if rel is not None else sinks.stack([], columns)


class _DuckMatrix:
    """The constraint matrix, left in duckdb and read a row range at a time.

    `sinks.MatrixBlocks` against tables rather than against a frame. Nothing
    above this can tell the difference — a sink asks for rows ``[lo, hi)`` and
    gets ``(col, coeff)`` either way — which is the whole reason the matrix can
    stay where it was built while `cols`, `rows` and `obj` cross.

    **A block reads only the constraints it overlaps.** Each share covers a
    contiguous run of rows, recorded when it was built, so a chunk in the middle
    of a large model scans the one or two tables it lands in rather than the
    model. Within a share the rows are ordered, so the filter is a range over a
    sorted column and duckdb skips the chunks that cannot match.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection, spans: tuple[tuple[str, int, int], ...], starts: Any) -> None:
        self._con = con
        self._spans = spans
        self.starts = starts

    @property
    def nonzeros(self) -> int:
        return int(self.starts[-1])

    def block(self, lo: int, hi: int) -> pl.DataFrame:
        """``(col, coeff)`` for rows ``[lo, hi)``, in row-major order."""
        overlapping = [t for t, first, last in self._spans if first < hi and last > lo]
        keep = (col('row') >= ConstantExpression(lo)) & (col('row') < ConstantExpression(hi))
        return _fetch(_stacked(self._con, overlapping, _ENTRY, keep), _ENTRY)


class DuckExecutor(Engine):
    """The duckdb engine: plan → relations → `sinks.ModelTables`.

    Everything past the four frames — both sinks, the solution read-back, the
    context manager — comes from `Engine` and is shared with the polars
    executor, because none of it is engine work.
    """

    _con: duckdb.DuckDBPyConnection

    def __init__(self) -> None:
        self._con = duckdb.connect()
        # **Asked for, not assumed.** A label is positional here: `_factored`
        # and `_repeating_bounds` read the head product in the order duckdb
        # scans it, and `_stacked` reads the shares in the order they were
        # built. duckdb states that as a setting — "if set to false the system
        # is allowed to re-order any results that do not contain ORDER BY
        # clauses" — and it is *global*, so the default being right is not the
        # same as this connection getting it. Off, the model is silently
        # mislabelled rather than slow.
        self._con.execute('SET preserve_insertion_order = true')
        self._bound: BoundSources | None = None
        self._compiler: DuckCompiler | None = None
        self._program: plan.Program | None = None
        #: The model, one table per declaration, in declaration order — which
        #: is the order `_tables` stacks them in and so the order `col` and
        #: `row` count in. It stays here until a sink asks for it.
        self._col_tables: list[str] = []
        self._row_shares: list[str] = []
        self._obj_tables: list[str] = []
        #: `(table, first row, last row + 1)` per constraint that kept a term.
        #: The range is what lets a block read skip the constraints it does not
        #: overlap instead of scanning the whole matrix for every chunk.
        self._matrix_spans: list[tuple[str, int, int]] = []
        #: How many entries each built row owns, taken per constraint while its
        #: matrix was still in duckdb. `_starts` sums them into the CSR index.
        self._row_lengths: list[pl.Series] = []
        #: `(name, variable type, how many columns)` per variable, in
        #: declaration order — the runs `_vtypes` spells out and
        #: `_refuse_null_bounds` attributes a missing bound to. Kept here rather
        #: than read off `_blocks`, which is keyed by declaration name and so
        #: cannot tell a variable from a constraint that shares its name.
        self._col_runs: list[tuple[str, str, int]] = []
        #: The four frames, once something has drained them.
        self._cached: sinks.ModelTables | None = None
        self._var_tables: dict[str, str] = {}
        self._row_tables: dict[str, str] = {}
        #: the dims each declared name is read through — what `plan.free_prefix`
        #: needs to know which label path a mask allows
        self._name_dims: dict[str, tuple[str, ...]] = {}
        self._var_labels = _Labels(self._con, self._var_tables, 'var_label')
        self._row_labels = _Labels(self._con, self._row_tables, 'row')
        #: the contiguous run of labels each declaration was given — what
        #: `Engine._read_back` slices a solver vector by, on both engines
        self._blocks: dict[str, tuple[int, int]] = {}
        self._omitted: dict[str, int] = {}
        #: `(head, kept, ranked survivors)` when the last label frame
        #: factored — `None` otherwise. Read only by `_bounds_run`.
        self._rectangle: tuple[tuple[str, ...], tuple[str, ...], Relation] | None = None
        self._n_cols = 0
        self._n_rows = 0
        self._obj_const = 0.0
        self._obj_sense = 'min'
        self._registered = 0

    # -- registration ---------------------------------------------------

    def _relation(self, rel: Relation, prefix: str, *, materialise: bool) -> str:
        """Name *rel* as a table or a view, and return the name.

        **Materialise when the derivation is the expensive part, not when the
        result is read often.** A label relation is read three or four times
        downstream, which argues for a table until you price the two sides: a
        counted label needs an ordered window over the whole coordinate product
        and is worth paying for once, while an arithmetic one is a cross join
        of two small relations and a multiply. Writing ten million rows to
        answer that a second time costs more than answering it three times.

        The polars engine collects in the same places, and for a reason that
        does not carry over: a `LazyFrame` read twice is *planned* twice, and
        the plan under a label reaches all the way back to the parquet.
        """
        self._registered += 1
        name = f'{prefix}_{self._registered}'
        if materialise:
            rel.create(name)
        else:
            rel.create_view(name)
        return name

    def _register(self, name: str, frame: pl.DataFrame) -> str:
        """Copy *frame* into a table of its own, and return the table's name.

        **Copied, not scanned in place.** A registered frame stays a Python
        object: duckdb reads it through the buffer protocol from whichever
        worker thread the scan lands on, and those threads need the GIL — which
        the thread that called `execute` is holding while it waits. On a plan
        with several scans of registered frames feeding one pipeline, that
        deadlocks outright: `transport/l` reproduces it as a build that sits at
        0% CPU indefinitely, while the same query over copies returns in 0.3 s.

        Copying is not free — the frame and the table are both resident for the
        length of one statement — but it buys a build that cannot stall, and
        duckdb's own storage for every scan after the first.
        """
        source = f'__source {name}__'
        self._con.register(source, frame)
        try:
            self._con.sql(f'SELECT * FROM {q(source)}').create(name)
        finally:
            self._con.unregister(source)
        return name

    def build(self, program: plan.Program, sources: Mapping[str, Any]) -> None:
        """Bind *sources*, then build every declaration into the model tables.

        Binding is `relational/binding.py`, shared with the polars engine: it
        is `scan_parquet` plus dtype and duplicate-coordinate validation, which
        is what a caller's data has to survive whoever builds the model.

        **Nothing crosses into polars here.** Each declaration's share is a
        duckdb table, and the four frames a sink reads are assembled from them
        once, in :meth:`_tables`. A build that is never sunk never pays for a
        frame at all, and a build that is pays for each exactly once — where
        fetching per declaration paid a Python round trip and a polars
        concatenate per variable and per constraint.

        **It does not pay for itself yet, and that is the point of the step.**
        Writing each share into duckdb rather than streaming it straight out
        costs 1.23-1.26x of build at the `l` rung, and `CREATE TABLE` is 82% of
        what a build now spends (#399). Peak does not move to meet it — 0.95x
        to 1.11x — because `ModelTables` still takes whole polars frames, so at
        the seam the model exists on both sides at once. The copy disappears,
        rather than moving, when a sink can read its blocks from whichever
        engine built them; until then this trades wall for a boundary in the
        right place.
        """
        self._program = program
        self._name_dims = plan.name_dims(program)
        bound = self._bound = bind(program, sources)
        materialised = {n: f.collect() for n, f in bound.dimensions.items()}
        dims = {n: self._register(f'dim_{n}', f) for n, f in materialised.items()}
        params = {n: self._register(f'par_{n}', f.collect()) for n, f in bound.parameters.items()}
        self._compiler = DuckCompiler(
            self._con, program, dims, params, bound.cardinality, bound.boolean_parameters, self._var_tables
        )

        for v in program.variables:
            self._build_variable(v)
        for c in program.constraints:
            self._build_constraint(c)
        self._build_objective(program.objective)

    @property
    def _variables(self) -> Mapping[str, pl.LazyFrame]:
        return self._var_labels

    @property
    def _constraints(self) -> Mapping[str, pl.LazyFrame]:
        return self._row_labels

    @property
    def _q(self) -> DuckCompiler:
        assert self._compiler is not None, 'build() has not run'
        return self._compiler

    # -- labels ---------------------------------------------------------

    def _label_frame(
        self,
        dims: tuple[str, ...],
        where: plan.Predicate | None,
        label: str,
        start: int,
        restrictions: Sequence[tuple[tuple[str, ...], Relation]] = (),
    ) -> tuple[str, int]:
        """The masked coord product of *dims* with a dense *label* from *start*.

        The same three paths the polars engine has, chosen by the same
        function. **Unmasked**, a row's label is its position in the product —
        arithmetic on the ordinals, no sort and nothing to count. **Masked but
        factoring**, the survivors are a rectangle and the label is arithmetic
        again over a ranked survivor set (:meth:`_factored`). **Otherwise** it
        is counted, which costs the ordered window over the whole product.

        `plan.free_prefix` decides between them for both engines, because a
        label is the solver's own column index: two engines choosing routes
        independently is how they would come to build different models.
        """
        if not restrictions:
            if where is None:
                rows = math.prod(self._q.cardinality[d] for d in dims)
                rel = self._q.frame(dims, None).select(
                    *_coordinates(dims), self._row_major(dims, start).cast('BIGINT').alias(label)
                )
                return self._relation(rel, 'lbl', materialise=False), start + rows

            free = plan.free_prefix(dims, plan.predicate_dims(where, self._name_dims))
            if free:
                return self._factored(dims, free, where, label, start)

        carrier = self._q.frame(dims, where)
        for on, presence in restrictions:
            carrier = restrict_to(carrier, on, presence)
        return self._counted(carrier, dims, label, start)

    def _row_major(self, dims: tuple[str, ...], start: int, alias: str = '') -> Expression:
        """Row-major position in *dims*' coordinate product, offset by *start*.

        The trailing dim has stride 1 and every other is the product of the
        cardinalities to its right, so the position is a dot product against
        the ordinals the frame already carries — no ordering imposed, because
        the answer does not depend on the order rows arrive in. The polars twin
        is `Labeller.row_major`, and both arithmetic paths reach a label
        through it for the reason the label itself is written once.
        """
        offset = ConstantExpression(start)
        if not dims:
            return offset
        terms: list[Expression] = []
        stride = 1
        for d in reversed(dims):
            terms.append(col(_ordinal(d), of=alias) * ConstantExpression(stride))
            stride *= self._q.cardinality[d]
        return reduce(operator.add, reversed(terms)) + offset

    def _counted(self, carrier: Relation, dims: tuple[str, ...], label: str, start: int) -> tuple[str, int]:
        """Rank the surviving rows of *carrier*: the ordered window, and a count.

        The general answer and the expensive one — a global sort of the whole
        product, which is what the two arithmetic paths exist to avoid.
        """
        ranked = _ranked([_ordinal(d) for d in dims], start).alias(label)
        name = self._relation(carrier.select(*_coordinates(dims), ranked), 'lbl', materialise=True)
        return name, start + self._height(name)

    def _factored(
        self,
        dims: tuple[str, ...],
        free: int,
        where: plan.Predicate,
        label: str,
        start: int,
    ) -> tuple[str, int]:
        """Labels for a mask that reads none of the first *free* dims.

        A mask that cannot see the leading dims removes the same coordinates
        under every one of their values, so the survivors are a rectangle: the
        full product of the leading dims against one surviving set. Ranking
        that set costs a window over the *set*, not over the product — on
        `dispatch` it ranks 100 generators instead of 10M
        ``(snapshot, generator)`` pairs, and the window is what a global sort
        made the dominant cost of the build.

        The label is then arithmetic again, through the same
        :meth:`_row_major` the unmasked path uses: row-major over the leading
        dims, times the width of the surviving set, plus a survivor's rank
        within it. That is the same number the window would have counted,
        because for each leading coordinate the same survivors appear in the
        same order — which is what `tests/test_engine_parity.py` checks by
        comparing the built model against the other engine's.
        """
        head, kept = dims[:free], dims[free:]
        ranked = self._relation(
            self._q.frame(kept, where).select(
                *(col(d) for d in kept), _ranked([_ordinal(d) for d in kept], 0).alias('__rank')
            ),
            'srv',
            # the one relation here worth writing down: it is small, it is
            # counted, and its window is the work the rectangle exists to avoid
            materialise=True,
        )
        width = self._height(ranked)
        if width == 0:
            # nothing survived anywhere, so there is no rectangle to describe.
            # The counted path returns the right columns and dtypes for free.
            return self._counted(self._q.frame(dims, where), dims, label, start)

        survivors = self._con.table(ranked).set_alias('s')
        position = self._row_major(head, 0, alias='h')
        placed = (position * ConstantExpression(width) + col('__rank', of='s') + ConstantExpression(start)).cast(
            'BIGINT'
        )
        rel = (
            self._q.frame(head, None)
            .set_alias('h')
            .cross(survivors)
            .select(
                *(col(d, of='h') for d in head),
                *(col(d, of='s') for d in kept),
                placed.alias(label),
            )
        )
        rows = math.prod(self._q.cardinality[d] for d in head) * width
        #: the rectangle, kept for `_bounds_run` — a caller that only reads the
        #: trailing dims can be answered once and repeated
        self._rectangle = (head, kept, survivors)
        return self._relation(rel, 'lbl', materialise=False), start + rows

    def _height(self, table: str) -> int:
        return self._con.table(table).shape[0]

    # -- declarations ---------------------------------------------------

    def _build_variable(self, v: plan.VariableDeclaration) -> None:
        """One variable's labelled relation, and its share of ``cols``, as tables."""
        start = self._n_cols
        self._rectangle = None
        name, self._n_cols = self._label_frame(v.dims, v.where, 'var_label', start)
        self._var_tables[v.name] = name
        self._blocks[v.name] = (start, self._n_cols - start)

        # **`cols` has no `col`, so a row's place in this frame is its solver
        # column index** (`sinks/tables.py`). The polars engine gets that order
        # from the emission order of a cross join and only *verifies* it; a
        # duckdb relation promises no order at all, so here it is produced
        # deliberately — cheaply where the bounds repeat, and by sorting where
        # they do not.
        repeating = self._repeating_bounds(v)
        bounds = repeating if repeating is not None else self._sorted_bounds(v, name)
        # `vtype` is not stored: every column of one declaration carries the
        # same word, so it is a run per variable rather than a value per row,
        # and `_vtypes` spells it out from the runs at the sink seam.
        #
        # **Written down only where the sort is.** `_relation`'s rule is that a
        # table pays for an expensive derivation, and these two paths are the
        # two sides of it: a sorted bound is a pass over the whole column and is
        # worth keeping, a repeated one is a short run and an `unnest` and is
        # cheaper to answer again at the drain than to store (#399).
        self._col_tables.append(self._relation(bounds, 'cols', materialise=repeating is None))
        self._col_runs.append((v.name, v.variable_type, self._n_cols - start))

    def _sorted_bounds(self, v: plan.VariableDeclaration, labels: str) -> Relation:
        """``(lb, ub)`` in label order, by sorting — always available.

        The projection is applied *over* the ordered relation rather than
        beside it, because ``var_label`` is what orders the frame and is not
        one of the two columns a sink wants. duckdb keeps a subquery's order,
        and the parity gate is what would notice if it stopped:
        `test_engine_parity.py` compares byte-for-byte LP files, where a column
        out of place moves every bound after it.
        """
        bounds = self._q.bounds(self._con.table(labels), v)
        return bounds.order(q('var_label')).select(
            col('lb').cast('DOUBLE').alias('lb'), col('ub').cast('DOUBLE').alias('ub')
        )

    def _bounds_run(self, v: plan.VariableDeclaration) -> tuple[tuple[str, ...], Relation] | None:
        """`(head, ranked run)` when *v*'s bounds repeat once per head coordinate.

        Two label frames have that shape, and they differ only in who chose the
        split. A **factored** one is a rectangle already: the mask fixed which
        dims lead, and the bounds either read inside the surviving set or they
        do not. An **unmasked** one is a rectangle under *every* suffix split —
        nothing was removed, so each leading coordinate carries the whole
        trailing product — which leaves the split free, and the shortest suffix
        the bounds read is the one that makes the run smallest.

        `None` where nothing repeats: a counted label frame, or bounds reading
        the leading dim, which is where the run is the whole column and there
        is nothing to expand.
        """
        reads = {p for e in (v.lower, v.upper) for p in _parameters(e)}
        read = {d for p in reads for d in self._q.program.parameter(p).dims}
        if self._rectangle is not None:
            head, kept, ranked = self._rectangle
            return (head, ranked) if read <= set(kept) else None
        if v.where is not None:
            return None
        split = min((i for i, d in enumerate(v.dims) if d in read), default=len(v.dims))
        if split == 0:
            return None
        head, kept = v.dims[:split], v.dims[split:]
        # arithmetic, not a window: an unmasked run is the whole product of
        # *kept*, so a row's rank in it is its row-major position
        return head, self._q.frame(kept, None).select(*_coordinates(kept), self._row_major(kept, 0).alias('__rank'))

    def _repeating_bounds(self, v: plan.VariableDeclaration) -> Relation | None:
        """Bounds in label order without a sort, when the same run repeats.

        The whole column is one short run repeated once per head coordinate
        (:meth:`_bounds_run`) — so build the run, and expand it.

        `unnest` of a list is that expansion, and it is not merely cheaper than
        the sort it replaces, it is cheaper than the cross join: 0.12 s against
        0.43 s at 10M columns, where an *unordered* join of the same shape is
        0.11 s. Ordering stops costing anything at all.

        `None` where the shape does not apply; the caller sorts, which is
        always correct and sometimes all that is available.

        Both this and the label arithmetic above it read the head product in
        the order duckdb scans it, which is the order it was written in —
        `preserve_insertion_order`, which `__init__` asks for rather than
        inherits. `test_engine_parity.py` compares byte-identical LP files,
        where a column out of place moves every bound after it.
        """
        found = self._bounds_run(v)
        if found is None:
            return None
        head, ranked = found

        run = self._q.bounds(ranked, v)
        # An ordered aggregate has no expression form; the two names it orders
        # and collects are the compiler's own, not the model's.
        lists = run.aggregate(f'list(lb ORDER BY {q("__rank")}) AS lbs, list(ub ORDER BY {q("__rank")}) AS ubs')
        return (
            self._q.frame(head, None)
            .cross(lists.set_alias('k'))
            .select(
                FunctionExpression('unnest', col('lbs', of='k')).cast('DOUBLE').alias('lb'),
                FunctionExpression('unnest', col('ubs', of='k')).cast('DOUBLE').alias('ub'),
            )
        )

    def _build_constraint(self, c: plan.ConstraintDeclaration) -> None:
        """One constraint's ``rows`` and its share of the matrix, as tables."""
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
        name, self._n_rows = self._label_frame(c.dims, c.where, 'row', start, restrictions)
        self._row_tables[c.name] = name
        self._blocks[c.name] = (start, self._n_rows - start)
        frame = self._con.table(name)

        declared = self._n_rows - start
        if not terms:
            self._omitted[c.name] = declared
            self._blocks[c.name] = (start, 0)
            self._n_rows = start
            self._row_tables[c.name] = self._relation(frame.filter(ConstantExpression(False)), 'lbl', materialise=False)
            return

        rows = self._relation(self._constant_side(frame, consts, c.sense), 'rows', materialise=False)
        matrix = self._relation(self._matrix_side(frame, terms), 'mat', materialise=True)
        # `(row, entries)` for the rows that kept a term, ascending. Small — one
        # row per constraint row, not per nonzero — and it answers both
        # questions this constraint still has: which rows survived, and how many
        # entries each owns, which is the CSR index `_tables` needs.
        lengths = self._con.table(matrix).aggregate(f'{q("row")}, count(*) AS n', q('row')).order(q('row')).pl()
        if lengths.height != declared:
            rows, matrix = self._drop_termless_rows(c.name, name, rows, matrix, start, declared, lengths)
        self._row_shares.append(rows)
        self._matrix_spans.append((matrix, start, start + lengths.height))
        self._row_lengths.append(lengths.get_column('n'))
        self._n_rows = start + lengths.height

    def _constant_side(self, frame: Relation, consts: list[tuple[TermFragment, float]], sense: str) -> Relation:
        """``(row, sense, rhs)`` — every constant fragment folded onto the frame."""
        carrier = frame
        accumulated: list[Expression] = []
        for i, (p, sign) in enumerate(consts):
            column = f'__const {i}__'
            aggregated = self._q.constant_scalar(p).set_alias('r')
            left = carrier.set_alias('l')
            joined = left.join(aggregated, matching(p.dims), how='left') if p.dims else left.cross(aggregated)
            carrier = joined.select(*(col(x, of='l') for x in carrier.columns), col('cval', of='r').alias(column))
            accumulated.append(ConstantExpression(sign) * CoalesceOperator(col(column), ConstantExpression(0.0)))
        total = reduce(operator.add, accumulated) if accumulated else ConstantExpression(0.0)
        return carrier.select(col('row'), ConstantExpression(sense).alias('sense'), total.cast('DOUBLE').alias('rhs'))

    def _matrix_side(self, frame: Relation, terms: list[tuple[TermFragment, float]]) -> Relation:
        """``(row, col, coeff)`` for one constraint, in row order, aggregated only when it must be.

        **The order is a requirement, not a preference.** A sink reads this
        matrix a row range at a time and gets back ``(col, coeff)`` with no
        ``row`` beside it, so which row an entry belongs to is *its position*
        between two CSR offsets. Entries arriving out of order inside a block
        would be attributed to the wrong rows — a different model that still
        solves. `test_engine_parity.py` compares LP files byte for byte, which
        is what would notice.
        """
        pieces = []
        for p, sign in terms:
            left, right = frame.set_alias('l'), p.rel.set_alias('r')
            joined = left.join(right, matching(p.dims)) if p.dims else left.cross(right)
            pieces.append(
                joined.select(
                    col('row', of='l'),
                    col('var_label', of='r').cast('INTEGER').alias('col'),
                    (ConstantExpression(sign) * col('coeff', of='r')).cast('DOUBLE').alias('coeff'),
                )
            )
        stacked = union_all(pieces[0], pieces[1:])
        if _needs_aggregate([f for f, _ in terms], self._q.may_share_a_column):
            # `sum` over `(row, col)` is the terminal aggregate — where duplicates
            # from Sum and GroupSum, which project rather than aggregate, collapse.
            total = FunctionExpression('sum', col('coeff')).alias('coeff')
            stacked = stacked.aggregate([col('row'), col('col'), total], f'{q("row")}, {q("col")}')
        return stacked.order(q('row'))

    def _drop_termless_rows(
        self,
        constraint: str,
        labels: str,
        rows: str,
        matrix: str,
        start: int,
        declared: int,
        lengths: pl.DataFrame,
    ) -> tuple[str, str]:
        """Rows that kept no variable term are not built, and the block closes up.

        A row with no variable terms asserts something about constants, which
        the solver cannot act on, so it is not built (SPEC §6). Labels are dense
        and the dual read-back reads a block by position, so the drop cannot
        leave a gap: the survivors are renumbered from *start*, here and in the
        label relation together.

        *lengths* already names the survivors in ascending order — one row per
        surviving constraint row — so the renumbering is a join against it
        rather than a second pass over the matrix. The three tables it rewrites
        are the constraint's own; nothing built before this one moves.

        The polars engine's twin, and written twice for the reason the engines
        are: the frames are polars there and tables here. The *rule* is the
        language's and may not differ, which `test_engine_parity.py` is what
        checks.
        """
        self._omitted[constraint] = declared - lengths.height
        self._blocks[constraint] = (start, lengths.height)
        remap = self._register(
            f'__remap {constraint}__',
            lengths.select('row').with_row_index('__new__', offset=start).cast({'__new__': pl.Int64}),
        )
        renumbered = self._con.table(remap).set_alias('r')
        rewritten = []
        for table, columns in ((rows, ('sense', 'rhs')), (matrix, ('col', 'coeff'))):
            kept = (
                self._con.table(table)
                .set_alias('l')
                .join(renumbered, matching(('row',)))
                .select(col('__new__', of='r').alias('row'), *(col(x, of='l') for x in columns))
                .order(q('row'))
            )
            rewritten.append(self._relation(kept, 'kept', materialise=True))
        self._row_tables[constraint] = self._relation(
            self._surviving_labels(labels, renumbered), 'lbl', materialise=True
        )
        return rewritten[0], rewritten[1]

    def _surviving_labels(self, labels: str, renumbered: Relation) -> Relation:
        """The label relation with only the surviving rows, renumbered.

        An inner join against the survivors *is* the restriction and the
        renumbering at once — the map has one row per surviving row and none
        for a dropped one, so there is nothing left to filter.
        """
        table = self._con.table(labels)
        coordinates = [c for c in table.columns if c != 'row']
        return (
            table.set_alias('l')
            .join(renumbered, matching(('row',)))
            .select(*(col(c, of='l') for c in coordinates), col('__new__', of='r').alias('row'))
        )

    def _build_objective(self, o: plan.ObjectiveDeclaration) -> None:
        """The objective's ``(col, coeff)`` as a table, or nothing if it has no terms."""
        comp = self._q.expression(o.expression, 'objective')
        for p in comp.consts:
            if p.dims:
                raise LanguageError(
                    'objective constant part has dims — wrap parameter terms in '
                    'Mul with a Var, or pre-aggregate to a scalar'
                )
            row = p.rel.aggregate([FunctionExpression('sum', col('cval')).alias('cval')]).fetchone()
            self._obj_const += (row[0] if row else None) or 0.0
        self._obj_sense = o.sense
        if not comp.terms:
            return
        pieces = [p.rel.select(col('var_label').cast('INTEGER').alias('col'), col('coeff')) for p in comp.terms]
        stacked = union_all(pieces[0], pieces[1:])
        if _needs_aggregate(comp.terms, self._q.may_share_a_column, projected=True):
            total = FunctionExpression('sum', col('coeff')).alias('coeff')
            stacked = stacked.aggregate([col('col'), total], q('col'))
        self._obj_tables.append(self._relation(stacked, 'obj', materialise=False))

    # -- the sink seam --------------------------------------------------

    def _tables(self) -> sinks.ModelTables:
        """The built model as the four frames a sink reads — the one fetch.

        Everything above this holds the model in duckdb. Here each frame is
        drained once, in declaration order, and cached: a caller that writes an
        LP file and then solves pays for the crossing once, and a caller that
        does neither never pays at all.

        **``matrix`` is not drained at all.** It stays in duckdb and answers one
        row range at a time (:class:`_DuckMatrix`), which is the only thing a
        sink asks of it — so the model's one unbounded part never becomes a
        polars frame, and the ``row`` label per nonzero never crosses either.
        The other three are one row per declaration coordinate and are what a
        solver wants densely anyway.
        """
        if self._cached is None:
            self._cached = sinks.ModelTables(
                cols=self._bounds(),
                obj=self._drain(self._obj_tables, _OBJ),
                rows=self._drain(self._row_shares, _ROWS),
                matrix=_DuckMatrix(self._con, tuple(self._matrix_spans), self._starts()),
                column_count=self._n_cols,
                row_count=self._n_rows,
                objective_sense=self._obj_sense,
                objective_constant=self._obj_const,
            )
        return self._cached

    def _drain(self, tables: list[str], columns: tuple[str, ...]) -> pl.DataFrame:
        """*tables* stacked and fetched, in the order they were built."""
        return _fetch(_stacked(self._con, tables, columns), columns)

    def _bounds(self) -> pl.DataFrame:
        """``cols``: every variable's bounds, in label order, with its type.

        The null check happens here rather than per declaration because the
        bounds a repeated run produces are not written down (`_build_variable`)
        — asking a view whether it holds a null would re-derive it. One pass
        over the drained column costs less and says the same thing.
        """
        cols = self._drain(self._col_tables, _BOUNDS).with_columns(self._vtypes())
        self._refuse_null_bounds(cols)
        return cols

    def _refuse_null_bounds(self, cols: pl.DataFrame) -> None:
        """A bound with no value where the model needed one, named for its variable.

        Raises:
            DataError: A bound parameter had no row for some coordinate the
                variable exists at. Attributed by walking the runs, so the
                message names the declaration rather than a row number — which
                costs a slice per variable, and only on the failing path.
        """
        missing = cols.select(pl.col('lb').is_null() | pl.col('ub').is_null()).to_series()
        if not missing.any():
            return
        at = 0
        for name, _, height in self._col_runs:
            bad = int(missing.slice(at, height).sum())
            if bad:
                raise DataError(null_bounds_message(name, bad))
            at += height

    def _vtypes(self) -> pl.Series:
        """``cols``' variable-type column, one run per declaration.

        Not stored beside the bounds: every column of one variable carries the
        same word, so what duckdb would hold is that word once per row, where
        the frame's stated dtype is an `Enum` and :attr:`_col_runs` already says
        how long each run is.
        """
        runs = [pl.repeat(t, n, dtype=sinks.VTYPE, eager=True) for _, t, n in self._col_runs]
        return pl.concat(runs).alias('vtype') if runs else pl.Series('vtype', [], dtype=sinks.VTYPE)

    def _starts(self) -> Any:
        """The CSR row index: where each row's entries begin.

        A cumulative sum over the per-row entry counts each constraint took
        while its matrix was still in duckdb. `compress_rows` is the polars
        engine's answer to the same question and reads the ``row`` column to
        get there; this one never brings that column across.
        """
        import numpy as np

        starts = np.zeros(self._n_rows + 1, dtype=np.int64)
        if self._row_lengths:
            counted = pl.concat(self._row_lengths).to_numpy()
            starts[1 : len(counted) + 1] = counted
        return np.cumsum(starts, out=starts)

    def _count(self, rel: Relation) -> int:
        """How many rows *rel* has, without bringing any of them across."""
        found = rel.aggregate([SQLExpression('count(*)').alias('n')]).fetchone()
        return int(found[0]) if found else 0

    def close(self) -> None:
        """Drop the built model. Calling twice is not an error."""
        self._cached = None
        self._bound = None
        self._var_labels.clear()
        self._row_labels.clear()
        self._blocks.clear()
        self._omitted.clear()
        self._row_lengths.clear()
        self._col_tables.clear()
        self._col_runs.clear()
        self._row_shares.clear()
        self._matrix_spans.clear()
        self._obj_tables.clear()
        self._compiler = None
        self._con.close()


def _coordinates(dims: tuple[str, ...]) -> list[Expression]:
    """The columns a label frame carries beside its label.

    A scalar declaration has none, and a relation with an empty projection is
    not a relation — so it carries the unit column the empty coordinate product
    is made of.
    """
    return [col(d) for d in dims] if dims else [col(UNIT)]


def _parameters(expression: plan.Expression) -> set[str]:
    """Every parameter name under *expression*."""
    found = {expression.name} if isinstance(expression, plan.Parameter) else set()
    for child in plan.children(expression):
        found |= _parameters(child)
    return found


def _needs_aggregate(
    terms: Sequence[TermFragment],
    may_share: Callable[[TermFragment, TermFragment], bool],
    *,
    projected: bool = False,
) -> bool:
    """Whether stacking *terms* can put two rows on one solver column.

    Named for the answer, not the condition: an inverted test here is a wrong
    model rather than a slow one.

    Two things can put a label twice into the stack, asked separately. A
    fragment that is not `keyed` already holds one twice on its own. Whether a
    *pair* can is *may_share*, which answers no for distinct variables and
    otherwise asks whether two fragments of one variable send a label to one
    **row** — see `DuckCompiler.may_share_a_column`. That second half is what
    makes the ordinary multi-term constraint free: reading only a fragment
    count says the aggregate is reachable for ``reserve_up + reserve_down <=
    p_max``, which sorts every nonzero in the model to collapse nothing. Worth
    0.90-0.94x of build on the three ladder cases with multi-term constraints,
    and nothing on the other four (#638).

    *projected* is what the two call sites do not share. The matrix keeps a
    fragment's dims, so `keyed` — one row per ``(dims…, var_label)`` — carries
    straight into ``(row, col)``. The objective keeps only ``var_label``, so it
    asks the stronger question: does the key survive losing *all* dims?
    """
    if any(not t.survives_dropping(set(t.dims) if projected else set()) for t in terms):
        return True
    return any(may_share(a, b) for i, a in enumerate(terms) for b in terms[i + 1 :])


def _absence_restrictions(terms: Sequence[TermFragment]) -> list[tuple[tuple[str, ...], Relation]]:
    """Where a masked variable says a constraint row must not exist."""
    return [(p.presence_dims or p.dims, p.presence) for p in terms if p.presence is not None]
