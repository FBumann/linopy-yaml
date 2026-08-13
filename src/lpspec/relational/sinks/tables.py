"""What every sink reads, and nothing more.

Four frames plus the scalars a writer needs to size its batching, and the
projections more than one sink needs — the dense column and row vectors, the
matrix a block at a time. Those belong to the contract rather than to either
solver: two sinks computing them separately could disagree about the model
they loaded, which is the one thing neither may do.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import polars as pl

from lpspec.relational import chunking

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

#: The dtype the ``rows`` frame holds a comparison in — the argument that made
#: ``vtype`` an ``Enum`` (#189), applied to the other one-word-per-row column.
#:
#: **Built from :data:`SENSE_CODES`, so a category's index is its code.** That
#: is what lets :meth:`ModelTables.dense_rows` read the physical column rather
#: than hash every row's string through a lookup, and it is why the two are
#: defined together: spelling the categories out a second time is how the order
#: would come to disagree, and a permuted comparison is a different model that
#: every solver answers confidently.
SENSE = pl.Enum(list(SENSE_CODES))


@dataclass(frozen=True)
class ModelTables:
    """The built model, as a sink sees it.

    ``cols`` (lb, ub, vtype), ``obj`` (col, coeff), ``rows`` (row, sense, rhs)
    and ``matrix`` in CSR: ``(col, coeff)`` in row-major order, with
    ``row_starts[r] : row_starts[r + 1]`` the half-open span row ``r`` owns.
    The objective constant lives outside the frames, having no column to
    attach to.

    ``cols``, ``rows`` and ``matrix`` all arrive in the solver's own order —
    ``cols`` by column, the other two by row — which is what lets every dense
    vector be read positionally rather than keyed.

    ``col`` and ``row`` are dense ``0..n-1``, so they *are* the solver's own
    indices and no sink builds a mapping. **``cols`` carries no ``col`` and
    ``matrix`` no ``row``**: a ``cols`` row's position is its index and a
    matrix entry's row is where it sits between two starts, which is what both
    solvers' matrix APIs take — where a row label per nonzero would hold
    8 bytes an entry for the model's lifetime. :meth:`matrix_block` spells them
    back out for the one consumer that renders them. ``obj`` keeps its ``col``,
    being genuinely sparse (0.71 of ``cols`` on `transport`).
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

    def _spans(self, budget: int | None) -> Iterator[tuple[int, int]]:
        """The row ranges a block reader walks — one rule, for both of them.

        Width is the average row, since a reader pays in nonzeros: 100k rows is
        900k entries in one model and 10M in another. There is deliberately no
        row-counted twin to reach for by mistake.

        ``budget=None`` is one span, and a real answer: whether splitting pays
        is a property of the API being fed, not the model. HiGHS takes a chunk
        at a time and its budget bounds the temporary; Gurobi's ``addMConstr``
        charges per *model column* per call whatever the block holds, so
        splitting a matrix into many blocks costs it dearly (#434).

        Private, so no caller can pair spans and entries that disagree.
        """
        if budget is None:
            return iter([(0, self.row_count)])
        return chunking.ranges(self.row_count, budget, self.matrix.height / max(1, self.row_count))

    def _span(self, lo: int, hi: int) -> pl.DataFrame:
        """The matrix entries rows ``[lo, hi)`` own — the CSR arithmetic, once.

        Both block readers slice through here, so how a span is located, and
        the half-open ``hi`` bound, cannot drift between them.
        """
        first = int(self.row_starts[lo])
        return self.matrix.slice(first, int(self.row_starts[hi]) - first)

    def col_chunks(self, budget: int) -> Iterator[tuple[int, int]]:
        """Column ranges of roughly ``budget`` columns each.

        Width 1: a column *is* one row of the batch a sink hands over, stated
        rather than assumed (:mod:`~lpspec.relational.chunking`).
        """
        return chunking.ranges(self.column_count, budget, 1.0)

    def dense_columns(self, infinity: float) -> DenseColumns:
        """``(lb, ub, cost, integral)`` as numpy vectors over the solver's index.

        *infinity* is the solver's own spelling of an absent bound — the one
        thing the two disagree on — so it is asked for and the vectors come
        back ready to hand over unedited.

        ``cols`` already arrives one row per column in ``col`` order, so its
        three vectors are the frame's own. Only ``cost`` is scattered, ``obj``
        being genuinely sparse: a variable in no objective term is left free
        rather than holding whatever the allocator returned.

        **The three column vectors are prepared in one polars pass**, not three
        numpy ones. Substituting the solver's infinity with ``nan_to_num``
        walks the column once per special value and copies it again to leave
        the built model alone; the same substitution as an expression is one
        threaded pass that *produces* a new column, so the copy the old code
        made defensively is the only materialisation left. Expressions in one
        context evaluate in parallel, so the integrality test rides along for
        free rather than taking a pass of its own.

        Nothing here aliases the built model, which is what the copy was for:
        every vector returned is freshly produced.

        **Nothing textual crosses into numpy**: a polars ``String`` converts by
        boxing every value as a Python object, so the test against
        ``'continuous'`` is made in polars and only its answer crosses — an
        order of magnitude apart at the top of the ladder (#418).
        """
        prepared = self.cols.select(
            _finite(pl.col('lb'), infinity).alias('lb'),
            _finite(pl.col('ub'), infinity).alias('ub'),
            (pl.col('vtype') != 'continuous').alias('integral'),
        )
        cost = _scattered(self.column_count, self.obj['col'].to_numpy(), self.obj['coeff'].to_numpy(), 0.0)
        return (
            prepared['lb'].to_numpy(),
            prepared['ub'].to_numpy(),
            cost,
            prepared['integral'].to_numpy(),
        )

    def dense_rows(self, infinity: float) -> DenseRows:
        """``(sense, rhs)`` as numpy vectors over the solver's row index.

        The row half of :meth:`dense_columns`: a chunk of rows is a slice
        rather than a search. Sorting and filtering the row frame once per
        chunk read the same 6M rows nine times over on `fleet/l`.

        It stops at the sense because that is where the two solvers part —
        HiGHS wants ``lower``/``upper``, Gurobi a comparison and right-hand
        side, both this pair spelled differently. A row with no entry gets a
        comparison nothing can fail (``>=`` against ``-infinity``) rather than
        the ``== 0`` that would be an equality the model never stated.

        **The code is read off the column, not looked up per row.** ``sense``
        is a :data:`SENSE` ``Enum`` built from :data:`SENSE_CODES`, so its
        physical value already *is* the code and the byte a solver wants costs
        a cast rather than a string hash for every row of the model.

        **A frame that spans the model is already the answer.** ``rows`` leaves
        the build in row order, so when it holds a row per label its ``row``
        column is the identity and both vectors are the frame's own — where
        scattering allocates a second vector each and permutes into it, on
        every solve. The scatter stays for the frame that falls short of the
        model, which is what the ``>=`` against ``-infinity`` above is for: it
        answers for the labels no row spoke for.
        """
        sided = self.rows.select(
            'row',
            pl.col('sense').to_physical().cast(pl.UInt8).alias('op'),
            'rhs',
        )
        if sided.height == self.row_count:
            return sided['op'].to_numpy(), sided['rhs'].to_numpy()
        at = sided['row'].to_numpy()
        sense = _scattered(self.row_count, at, sided['op'].to_numpy(), SENSE_CODES['>='])
        rhs = _scattered(self.row_count, at, sided['rhs'].to_numpy(), -infinity)
        return sense, rhs

    @cached_property
    def structure(self) -> bytes:
        """A digest of everything a re-solve may **not** change.

        The question a loaded solver asks of a rebuilt model: may I keep what
        I hold and take the new numbers by value? Bounds, costs and right-hand
        sides go in that way; the counts, the matrix, each row's comparison
        and each column's type do not, so a model whose digest moved has to be
        loaded again.

        Read off the data rather than derived from the declarations, because
        whether a rebind moved a label or a coefficient is a property of the
        data (SPEC §8) and not of the model that declared it. Every vector it
        reads is one with an order contract — the label-ordered columns, the
        row-ordered matrix and rows — so that two builds of one model agree.

        **A digest rather than the frames.** Keeping the previous matrix to
        compare against would hold two models alive across a rebuild, which is
        the memory the rebuild exists not to spend (the trade against a diff
        is argued in `README.md`). The cost is one linear pass over the
        matrix, cached on the instance — the frames are frozen — so the two
        askers in one solve, the keep-or-reload comparison and the load that
        records what it loaded, share one pass.

        Each vector is hashed through its own **buffer**, so the matrix is read
        where it lies rather than copied to bytes to be read once.
        """
        import hashlib

        import numpy as np

        digest = hashlib.blake2b(digest_size=16)
        digest.update(f'{self.column_count} {self.row_count} {self.objective_sense}'.encode())
        for vector in (
            self.cols['vtype'].to_physical().to_numpy(),
            self.rows['sense'].to_physical().to_numpy(),
            self.matrix['col'].to_numpy(),
            self.matrix['coeff'].to_numpy(),
            self.row_starts,
        ):
            digest.update(np.ascontiguousarray(vector).data)
        return digest.digest()

    def row_blocks(self, budget: int | None) -> Iterator[RowBlock]:
        """Each chunk of rows with the matrix entries it owns — a solver's reader.

        A chunk is a ``slice``: ``row_starts`` already says where every row's
        entries sit, so nothing is sorted and nothing is searched.

        Yields:
            ``(lo, hi, entries, starts)`` for rows ``[lo, hi)``, where
            ``starts`` is each row's offset within the block — what both
            solvers' matrix APIs ask for. A row with no entries takes the
            next row's offset, and so occupies no span.
        """
        for lo, hi in self._spans(budget):
            yield lo, hi, self._span(lo, hi), self.row_starts[lo:hi] - self.row_starts[lo]

    def matrix_block(self, lo: int, hi: int) -> pl.DataFrame:
        """Rows ``[lo, hi)`` of the matrix with their ``row`` labels spelled out.

        The adjoint of what CSR compressed — ``np.repeat`` walks the start
        offsets back into one label per entry — at the cost of one label column
        per *block*, not per model.
        """
        import numpy as np

        labels = np.repeat(np.arange(lo, hi, dtype=np.int64), np.diff(self.row_starts[lo : hi + 1]))
        return self._span(lo, hi).with_columns(pl.Series('row', labels))

    def labeled_blocks(self, budget: int | None) -> Iterator[tuple[int, int, pl.DataFrame]]:
        """Each chunk of rows with its entries labeled — the LP writer's reader.

        One method per consumer shape — solvers take :meth:`row_blocks`, the
        writer this — so no caller pairs spans and entries that disagree.

        Yields:
            ``(lo, hi, entries)`` for rows ``[lo, hi)``, the entries labelled
            as :meth:`matrix_block` labels them.
        """
        for lo, hi in self._spans(budget):
            yield lo, hi, self.matrix_block(lo, hi)


def solver_vector(values: Any) -> pl.Series:
    """One quantity a solver produced, in its own index — every sink's read-back.

    A series rather than a ``(label, value)`` frame: the read-back takes a
    declaration's share by slicing, so an index column beside it is an
    ``arange`` nothing reads — 8 bytes a column for as long as the result is
    held. The argument that took ``col`` off ``cols`` (#433).
    """
    import numpy as np

    return pl.Series('value', np.asarray(values, dtype=np.float64))


def _finite(value: pl.Expr, infinity: float) -> pl.Expr:
    """*value* with the solver's spelling of an absent bound, and no ``NaN``.

    The three substitutions ``numpy.nan_to_num(neginf=, posinf=)`` makes, as
    one expression: a ``NaN`` bound is no bound (zero, as it was), and each
    infinity becomes the finite sentinel the solver asking recognises. Kept
    together because a bound that took one substitution and not another would
    reach the solver as a number it reads as real.
    """
    return (
        pl.when(value.is_nan())
        .then(pl.lit(0.0))
        .when(value == float('inf'))
        .then(pl.lit(infinity))
        .when(value == float('-inf'))
        .then(pl.lit(-infinity))
        .otherwise(value)
    )


def _scattered(count: int, at: Any, values: Any, absent: Any) -> Any:
    """*values* written at the label each one belongs to, *absent* elsewhere."""
    import numpy as np

    dense = np.full(count, absent, dtype=values.dtype)
    dense[at] = values
    return dense
