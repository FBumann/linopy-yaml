r"""A set expressed as columns and rows, for a sink with no concept of one.

Which sink needs this and why the family rather than a member decides it:
``README.md``, *the one uneven stream*.

The formulation is linopy's (``linopy/sos_reformulation.py``), member for
member, because the lanes are compared against each other and a differently
relaxed MILP is a different search even where it is the same feasible set. For
members :math:`x_1 … x_k` in weight order, with :math:`M_i` the tighter of the
block's ``big_m`` and the member's own upper bound:

- **SOS1** — a binary :math:`y_i` per member, :math:`x_i \le M_i y_i`, and
  :math:`\sum_i y_i \le 1`.
- **SOS2** — a binary :math:`z_j` per *segment*, :math:`j = 1 … k-1`, with
  :math:`x_1 \le M_1 z_1`, :math:`x_i \le M_i (z_{i-1} + z_i)`,
  :math:`x_k \le M_k z_{k-1}`, and :math:`\sum_j z_j \le 1`.

Everything it adds goes **after** the model, so an appended column moves none
of the model's own and an appended row renumbers none of its rows — the label
contract spent (docs/ARCHITECTURE.md), and why a solve reads its answer back by
the same slice either way.

**Nothing here sorts, groups or joins.** The stream arrives in ``(set, weight)``
order, so every question about a set — where it starts, where it ends, which
binary a member reaches — is a comparison against the neighbouring row, and the
rows it emits are produced in the order CSR wants them.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import polars as pl

from lpspec.errors import DataError
from lpspec.relational.sinks.tables import SENSE, ModelTables

if TYPE_CHECKING:
    from typing import Any

#: Scratch columns of the member frame every projection below reads. The space
#: keeps them unrepresentable as a stream's own column name.
_CARDINALITY = '#cardinality row'
_CLOSES = '#closes'
_ENTRIES = '#entries'
_FIRST = '#first in set'
_HOLDS = '#holds a binary'
_LAST = '#last in set'
_M = '#big m'
_OPENS = '#opens'


def reformulated(model: ModelTables) -> ModelTables:
    """*model* with every SOS set written as binaries and linking rows.

    A pure function of the tables — the members, their columns' bounds and the
    two counts are all it reads — so a sink asks for it without knowing how the
    model was built. The result is **mixed-integer**, so an LP that carried a
    set comes back without duals.

    Args:
        model: A built model whose ``sos`` frame holds at least one member.

    Returns:
        A model declaring no sets, its counts grown by what was added.

    Raises:
        DataError: A member with a negative lower bound or no finite upper
            bound — neither is a set a big-M can stand in for.
    """
    members = _members(model)
    binaries = int(members.get_column(_HOLDS).sum())
    cardinality = members.filter(pl.col(_HOLDS))
    rows = members.height + cardinality.get_column('set').n_unique()

    return replace(
        model,
        cols=pl.concat([model.cols, _binary_columns(binaries, model.cols)]),
        rows=pl.concat([model.rows, _rows(members.height, rows, model)]),
        matrix=pl.concat([model.matrix, _linking_matrix(members), _cardinality_matrix(cardinality)]),
        sos=model.sos.clear(),
        row_starts=_row_starts(model, members, cardinality),
        column_count=model.column_count + binaries,
        row_count=model.row_count + rows,
    )


def _members(model: ModelTables) -> pl.DataFrame:
    """Every member of every set, with the rows and columns it is about to make.

    **A set's shape is read off its edges**, never off a group-by: a member is
    the first of its set when the row above belongs to another and the last
    when the row below does. The rest follows — a last member holds no binary,
    a first closes no segment, and a **SOS2 set of one is both**, so it is
    dropped whole (one nonzero is already at most two, and there is no segment
    to hold a binary; linopy returns early on the same case).

    Which binaries a member reaches is the running count of those assigned
    before it: its own is the next, and the segment closing into it the one
    before that — the previous member of a SOS2 set always holding one.

    Raises:
        DataError: A member a big-M cannot stand in for.
    """
    at = model.sos.get_column('col')
    members = model.sos.with_columns(
        model.cols.get_column('lb').gather(at),
        pl.min_horizontal(model.cols.get_column('ub').gather(at), pl.col('big_m')).alias(_M),
        (pl.col('set') != pl.col('set').shift(1)).fill_null(True).alias(_FIRST),
        (pl.col('set') != pl.col('set').shift(-1)).fill_null(True).alias(_LAST),
    )
    _refuse_unbounded(members)

    members = members.filter((pl.col('type') == 1) | pl.col(_FIRST).not_() | pl.col(_LAST).not_()).with_columns(
        ((pl.col('type') == 1) | pl.col(_LAST).not_()).alias(_HOLDS),
        ((pl.col('type') == 2) & pl.col(_FIRST).not_()).alias(_CLOSES),
    )
    column, row = model.matrix.schema['col'], model.rows.schema['row']
    assigned = pl.col(_HOLDS).cast(pl.Int64)
    before = (assigned.cum_sum() - assigned + model.column_count).cast(column)
    return members.with_columns(
        pl.when(pl.col(_HOLDS)).then(before).alias(_OPENS),
        pl.when(pl.col(_CLOSES)).then(before - 1).alias(_CLOSES),
        (assigned + pl.col(_CLOSES).cast(pl.Int64) + 1).alias(_ENTRIES),
        (
            pl.col(_FIRST).cum_sum().cast(row) + (model.row_count + pl.len() - 1)  # the linking rows come first
        ).alias(_CARDINALITY),
    )


def _refuse_unbounded(members: pl.DataFrame) -> None:
    """Refuse a member no finite big-M can stand in for — linopy's two conditions.

    Asked of the big-M rather than of the bound, in that order for the reason
    the order exists: ``big_m:`` is what a caller declares *because* the bound
    is open, so asking first would refuse the model the answer was given for.

    Raises:
        DataError: Either condition, counted rather than located: a member is
            a column index, and what a caller acts on is the bound.
    """
    for condition, what, fix in (
        (
            pl.col('lb') < 0,
            'a negative lower bound',
            'A set says which members are nonzero, and the big-M form a sink without SOS is handed '
            'can only say that of a non-negative variable. Give the variable `bounds: {lower: 0}`',
        ),
        (
            pl.col(_M).is_infinite(),
            'no upper bound and no big_m',
            'so there is no finite coefficient to link them to a binary with. Bound the variable, or '
            'set `big_m:` on the sos block',
        ),
    ):
        offending = members.select(condition.sum()).item()
        if offending:
            raise DataError(
                f'{offending} SOS member(s) have {what}: {fix} — or solve with a sink that takes '
                f'the set natively (`gurobi`, or an LP file), which needs neither.'
            )


def _linking_matrix(members: pl.DataFrame) -> pl.DataFrame:
    """``x_i - M_i * (the binaries that admit it) <= 0``, one row per member.

    **Each member owns a span, and its entries are written into it**, which is
    what keeps the block in CSR order with nothing sorted: a row holds its own
    column, then the segment closing into it, then the segment it opens, and
    those are already ascending. So every destination is arithmetic on the
    member's own width, and the three writes are scatters into it — the shape
    :func:`~lpspec.relational.sinks.tables._scattered` and the engine's CSR
    index are built with. Sorting the block in polars instead is the largest
    thing here, and a list column per member exploded costs more than the sort.
    """
    import numpy as np

    widths = members.get_column(_ENTRIES).to_numpy()
    at = np.cumsum(widths) - widths
    own = members.get_column('col').to_numpy()
    col = np.empty(int(widths.sum()), dtype=own.dtype)
    coeff = np.empty(len(col), dtype=np.float64)
    magnitude = -members.get_column(_M).to_numpy()

    col[at] = own
    coeff[at] = 1.0
    for slot, present in ((_CLOSES, at + 1), (_OPENS, at + widths - 1)):
        reached = members.get_column(slot).is_not_null().to_numpy()
        col[present[reached]] = members.get_column(slot).drop_nulls().to_numpy()
        coeff[present[reached]] = magnitude[reached]
    return pl.DataFrame({'col': col, 'coeff': coeff})


def _cardinality_matrix(cardinality: pl.DataFrame) -> pl.DataFrame:
    """``sum(a set's binaries) <= 1``, one row per set.

    In order already: sets ascend, and within one, so do the binaries.
    """
    return cardinality.select(pl.col(_OPENS).alias('col'), pl.lit(1.0).alias('coeff'))


def _rows(linking: int, total: int, model: ModelTables) -> pl.DataFrame:
    """The appended ``(row, sense, rhs)`` — every linking row, then every set's.

    Both blocks are ``<=``: a linking row against zero, a cardinality row
    against one.
    """
    return pl.select(
        (pl.int_range(total).cast(model.rows.schema['row']) + model.row_count).alias('row'),
        pl.lit('<=', dtype=SENSE).alias('sense'),
        pl.when(pl.int_range(total) < linking).then(0.0).otherwise(1.0).alias('rhs'),
    )


def _binary_columns(count: int, cols: pl.DataFrame) -> pl.DataFrame:
    """*count* binary columns, in the schema the model's own columns hold."""
    return pl.select(
        pl.repeat(0.0, count).alias('lb'),
        pl.repeat(1.0, count).alias('ub'),
        pl.repeat('binary', count, dtype=cols.schema['vtype']).alias('vtype'),
    )


def _row_starts(model: ModelTables, members: pl.DataFrame, cardinality: pl.DataFrame) -> Any:
    """The CSR index, extended by what each appended row owns.

    Counted where the counting is free rather than off the finished block: a
    linking row's width is the member's own arithmetic, and a cardinality
    row's is the length of a run in a column already sorted by set. Reading it
    back off the entries would mean a pass over every one of them, which is
    the largest thing here.
    """
    import numpy as np

    lengths = np.concatenate(
        [
            members.get_column(_ENTRIES).to_numpy(),
            cardinality.get_column('set').rle().struct.field('len').to_numpy(),
        ]
    )
    return np.concatenate([model.row_starts, model.row_starts[-1] + np.cumsum(lengths)])
