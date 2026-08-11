"""What every sink reads, and nothing more.

Four frames plus the scalars a writer needs to size its batching. A sink that
needs a fifth thing states it here, where both sides can see it.

Also the *projections* of them more than one sink needs — the dense column and
row vectors, and the matrix a block at a time. They belong to the contract
rather than to either solver, because two sinks computing them separately could
disagree about the model they loaded — the one thing neither may do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_args

import polars as pl

from lpspec.relational import chunking, plan

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    import numpy.typing as npt

    #: What a solver sink is handed: three float vectors and an integrality
    #: mask, each as long as the model has columns.
    DenseColumns = tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]
    ]

    #: A sense code and a right-hand side, each as long as the model has rows.
    DenseRows = tuple[npt.NDArray[np.uint8], npt.NDArray[np.float64]]

    #: One chunk of rows: the half-open label range, the matrix entries those
    #: rows own, and the offset of each row's entries within them.
    RowBlock = tuple[int, int, pl.DataFrame, npt.NDArray[np.int64]]


#: ``sense`` as a number, so a row's comparison crosses into numpy as one byte
#: rather than as a boxed Python string. The order is arbitrary and shared: a
#: solver indexes its own spelling with these, so the two cannot drift.
SENSE_CODES = {'<=': 0, '>=': 1, '==': 2}


#: The columns of each frame, in order.
COLS = ('lb', 'ub', 'vtype')
OBJ = ('col', 'coeff')
ROWS = ('row', 'sense', 'rhs')
MATRIX = ('row', 'col', 'coeff')

#: And the dtype of each column. Here rather than in an executor because it is
#: what a *sink* reads: two engines filling the same four frames with different
#: types is a difference no sink asked for and none can see coming.
#:
#: ``vtype`` is an ``Enum`` over the variable types the plan declares, rather
#: than a string: it holds one word per column and the same handful of words
#: for the whole model, so as a string it stores that word once per row —
#: 0.098 GB of the ``cols`` frame's 0.333 at 9.8M columns, against 0.010 as an
#: Enum. The Enum also makes the vocabulary explicit, so a fourth variable type
#: added to :data:`~lpspec.relational.plan.VariableType` and not reaching here
#: fails where the column is built rather than in whichever sink first compares
#: against a name it does not know.
#:
#: ``col`` is ``Int32`` — the solver's own index width, HiGHS and Gurobi both
#: being 32-bit indexed, so a count past 2^31 has no sink that could take it
#: and the strict cast raises there rather than wrapping. An engine casts where
#: the column is *produced*, not on the stacked frame: narrowing afterwards
#: allocates the narrow copy while the wide one is still alive, a transient
#: visible in `dispatch/l`'s peak RSS. A *label* stays ``Int64``: it is a
#: position in the full pre-mask coordinate product, which can pass 2^31 while
#: every survivor fits.
DTYPES = {
    'col': pl.Int32, 'row': pl.Int64,
    'lb': pl.Float64, 'ub': pl.Float64, 'rhs': pl.Float64, 'coeff': pl.Float64,
    'sense': pl.String, 'vtype': pl.Enum(get_args(plan.VariableType)),
}  # fmt: skip

#: The variable-type column's dtype, which an engine builds a literal against.
VTYPE = DTYPES['vtype']


def compress_rows(matrix: pl.DataFrame, row_count: int) -> tuple[pl.DataFrame, npt.NDArray[np.int64]]:
    """A ``(row, col, coeff)`` matrix as the CSR pair `ModelTables` takes.

    Here rather than in an engine because it is the *contract's* layout: both
    engines stack their constraints' shares in declaration order, which is
    ascending row ranges, and both then owe a sink the same compressed form.
    Two engines compressing separately could disagree about which entries a row
    owns — the one thing neither may do.

    ``row`` is known to ascend, and that is **checked rather than assumed**.
    polars cannot see the ordering through a ``concat``, and a sink that finds
    the flag missing orders the whole matrix again; ``is_sorted`` is a linear
    scan over a column the frame already holds, and the sort behind it is the
    correctness floor, expected never to run.

    The starts are a run-length over that sorted column, then a scatter and a
    cumulative sum — robust to the model's shape where the obvious alternatives
    are not: ``bincount`` pays per entry (26 ms to rle's 7 at 10M entries over
    100k rows), ``searchsorted`` per row times the log of the entries, and
    either is the wrong one on some ladder case. Computed here so ``row`` can
    then be dropped: a label repeated once per nonzero is 8 bytes per entry no
    sink reads, since every consumer either slices by these starts or asks
    :meth:`ModelTables.matrix_block` to spell the labels back out.

    The kept matrix is then **rechunked, once**. A streaming collect leaves it
    in chunks, and a sink slices it per row block — against a chunked frame
    every block's ``to_numpy`` is a gather-copy, where against one contiguous
    buffer it is a view (codspeed caught the difference as -6.9% on
    `profiled-m`, ~150 blocks over 16 chunks).
    """
    import numpy as np

    if not matrix.height:
        ordered = matrix
    elif not matrix['row'].is_sorted():
        ordered = matrix.sort('row')
    else:
        ordered = matrix.with_columns(pl.col('row').set_sorted())

    runs = ordered['row'].rle()
    starts = np.zeros(row_count + 1, dtype=np.int64)
    starts[runs.struct.field('value').to_numpy() + 1] = runs.struct.field('len').to_numpy()
    return ordered.select('col', 'coeff').rechunk(), np.cumsum(starts, out=starts)


@dataclass(frozen=True)
class ModelTables:
    """The built model, as a sink sees it.

    ``cols`` (lb, ub, vtype), ``obj`` (col, coeff), ``rows`` (row, sense,
    rhs) and ``matrix`` in CSR: ``(col, coeff)`` in row-major order, with
    ``row_starts[r] : row_starts[r + 1]`` the half-open span row ``r`` owns.
    The scalars are what a sink cannot cheaply recover; the objective constant
    lives outside the frames because it has no column to attach to.

    ``col`` and ``row`` are dense ``0..n-1``, so they *are* the solver's own
    indices and no sink builds a mapping.

    **``cols`` carries no ``col``, and ``matrix`` carries no ``row``** — for
    the same reason at two granularities. A ``cols`` row's position is its
    index; a matrix entry's row is where it sits between two starts. Both are
    what the solvers' own matrix APIs take, so nothing here is a private
    compression a sink first has to undo — and a row label repeated per
    nonzero would hold 8 more bytes per entry for the model's whole lifetime.
    :meth:`matrix_block` spells the labels back out for the one consumer that
    renders them. ``obj`` stays sparse in the index and keeps its ``col``: it
    only looks dense on models where every variable has a cost (0.71 of
    ``cols`` on `transport`).
    """

    cols: pl.DataFrame
    obj: pl.DataFrame
    rows: pl.DataFrame
    matrix: pl.DataFrame
    row_starts: npt.NDArray[np.int64]
    column_count: int
    row_count: int
    objective_sense: str
    objective_constant: float

    def _row_chunks_by_nonzeros(self, budget: int) -> Iterator[tuple[int, int]]:
        """Row ranges holding roughly ``budget`` *nonzeros* each.

        A reader that walks ``matrix`` a range at a time pays in nonzeros, not
        in rows — a range of 100k rows is 900k entries in one model and 10M in
        another, and only the second is a problem. So the width here is the
        average row, and there is deliberately no row-counted twin to reach
        for by mistake. Private: a consumer takes whole blocks from
        :meth:`row_blocks` or :meth:`labeled_blocks`, so no caller can pair
        spans and entries that disagree.
        """
        return chunking.ranges(self.row_count, budget, self.matrix.height / max(1, self.row_count))

    def _span(self, lo: int, hi: int) -> pl.DataFrame:
        """The matrix entries rows ``[lo, hi)`` own — the CSR arithmetic, once.

        Both block readers slice through here, so how a span is located — and
        the half-open ``hi`` bound — cannot drift between them.
        """
        first = int(self.row_starts[lo])
        return self.matrix.slice(first, int(self.row_starts[hi]) - first)

    def col_chunks(self, budget: int) -> Iterator[tuple[int, int]]:
        """Column ranges of roughly ``budget`` columns each.

        Width 1, because a column *is* one row of the batch a sink hands over —
        stated rather than assumed, which is the bargain
        :mod:`~lpspec.relational.chunking` asks for.
        """
        return chunking.ranges(self.column_count, budget, 1.0)

    def dense_columns(self, infinity: float) -> DenseColumns:
        """``(lb, ub, cost, integral)`` as numpy vectors over the solver's index.

        *infinity* is the solver's own spelling of an absent bound — the one
        thing the two disagree on — so it is asked for rather than assumed and
        the vectors come back ready to hand over unedited.

        ``col`` is dense ``0..n-1``, so it *is* the position a value has to end
        up at — and ``cols`` already arrives in that order, one row per column,
        so its three vectors are the frame's own and nothing is scattered.
        Only ``cost`` still is, because ``obj`` is genuinely sparse: a variable
        in no objective term has no row, and is left free rather than holding
        whatever the allocator returned.

        **Nothing textual crosses into numpy.** A polars ``String`` converts by
        boxing every value as a Python object, so the test against
        ``'continuous'`` is made in polars and only its answer crosses: 0.04 s
        against 0.95 s at 10M columns.
        """
        import numpy as np

        count = self.column_count
        # `cols` is already the solver's index, so its three vectors need no
        # scatter. `copy=True` rather than in place because they are views of
        # the frame now: rewriting an infinity through one would edit the built
        # model to suit whichever solver asked last.
        lb = np.nan_to_num(self.cols['lb'].to_numpy(), copy=True, neginf=-infinity, posinf=infinity)
        ub = np.nan_to_num(self.cols['ub'].to_numpy(), copy=True, neginf=-infinity, posinf=infinity)
        integral = self.cols.select(pl.col('vtype') != 'continuous').to_series().to_numpy()
        cost = _scattered(count, self.obj['col'].to_numpy(), self.obj['coeff'].to_numpy(), 0.0)
        return lb, ub, cost, integral

    def dense_rows(self, infinity: float) -> DenseRows:
        """``(sense, rhs)`` as numpy vectors over the solver's row index.

        The row half of :meth:`dense_columns`, and for the same reason: ``row``
        is dense ``0..n-1``, so it *is* the position a value belongs at, and a
        chunk of rows is then a slice rather than a search. Sorting the row
        frame and filtering it once per chunk read the same 6M rows nine times
        over on `fleet/l`, to hand each of them over once.

        It stops at the sense rather than at bounds because that is where the
        two solvers part: HiGHS wants a row's ``lower``/``upper`` and Gurobi its
        comparison and right-hand side, and both are this pair spelled
        differently. :data:`SENSE_CODES` is the spelling neither owns.

        A row with no entry is left with a comparison nothing can fail —
        ``>=`` against ``-infinity`` — rather than the ``== 0`` that would be a
        real equality the model never stated.
        """
        sided = self.rows.select(
            'row',
            pl.col('sense').replace_strict(SENSE_CODES, return_dtype=pl.UInt8).alias('op'),
            'rhs',
        )
        at = sided['row'].to_numpy()
        sense = _scattered(self.row_count, at, sided['op'].to_numpy(), SENSE_CODES['>='])
        rhs = _scattered(self.row_count, at, sided['rhs'].to_numpy(), -infinity)
        return sense, rhs

    def row_blocks(self, budget: int | None) -> Iterator[RowBlock]:
        """Each chunk of rows with the matrix entries it owns.

        **``budget=None`` is one block, and that is a real answer rather than a
        degenerate one.** Whether splitting pays is a property of the API being
        fed, not of the model: HiGHS takes a chunk at a time and its budget
        bounds the temporary, while Gurobi's ``addMConstr`` charges about 42 ns
        per *model column* per call whatever the block holds — 0.23 s in one
        call against 0.89 s in forty on the same matrix (#434). So the caller
        says, and both answers come out of the same code.

        A chunk is a ``slice``: ``row_starts`` already says where every row's
        entries sit, so nothing is sorted and nothing is searched.

        ``starts`` is each row's offset within the block, which is what both
        solvers' matrix APIs ask for. A row with no entries takes the next
        row's offset, and so occupies no span.
        """
        spans = [(0, self.row_count)] if budget is None else self._row_chunks_by_nonzeros(budget)
        for lo, hi in spans:
            yield lo, hi, self._span(lo, hi), self.row_starts[lo:hi] - self.row_starts[lo]

    def matrix_block(self, lo: int, hi: int) -> pl.DataFrame:
        """Rows ``[lo, hi)`` of the matrix with their ``row`` labels spelled out.

        The adjoint of what the CSR layout compressed: ``np.repeat`` walks the
        start offsets back into one label per entry. For a reader that wants
        COO — and, through :meth:`labeled_blocks`, for the LP writer — at the
        cost of one label column per *block*, not per model.
        """
        import numpy as np

        labels = np.repeat(np.arange(lo, hi, dtype=np.int64), np.diff(self.row_starts[lo : hi + 1]))
        return self._span(lo, hi).with_columns(pl.Series('row', labels))

    def labeled_blocks(self, budget: int | None) -> Iterator[tuple[int, int, pl.DataFrame]]:
        """Each chunk of rows with its entries labeled — the LP writer's reader.

        :meth:`matrix_block`'s budget-iterator form, chunked by the same rule
        every reader spends (:mod:`~lpspec.relational.chunking`, in nonzeros).
        One method per consumer shape — solvers take :meth:`row_blocks`, the
        writer this — so no caller pairs spans and entries that disagree.
        """
        spans = [(0, self.row_count)] if budget is None else self._row_chunks_by_nonzeros(budget)
        for lo, hi in spans:
            yield lo, hi, self.matrix_block(lo, hi)


def _scattered(count: int, at: Any, values: Any, absent: Any) -> Any:
    """*values* written at the label each one belongs to, *absent* elsewhere."""
    import numpy as np

    dense = np.full(count, absent, dtype=values.dtype)
    dense[at] = values
    return dense
