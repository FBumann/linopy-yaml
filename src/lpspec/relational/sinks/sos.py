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
contract spent (docs/about/architecture.md), and why a solve reads its answer back by
the same slice either way.

**Nothing here sorts, groups or joins.** The stream arrives in ``(set, weight)``
order, so every question about a set — where it starts, where it ends, which
binary a member reaches — is a comparison against the neighbouring row, and the
rows it emits are produced in the order CSR wants them.

**It asks them in numpy rather than in polars**, the one place this lane
departs from the rest of the sink (``tables._scattered`` and the engine's own
CSR index are the precedents): every question above is a scan or a scatter over
one contiguous buffer, where an expression frame would carry a dozen columns
across the whole member stream to answer them — a factor of two at 2M members
(#687).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from lpspec.errors import DataError
from lpspec.relational.sinks.tables import SENSE, ModelTables

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt


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
    binaries = len(members.cardinality)
    linking, sets = len(members.entries), len(members.set_widths)

    return replace(
        model,
        cols=pl.concat([model.cols, _binary_columns(binaries, model.cols)]),
        rows=pl.concat([model.rows, _rows(linking, linking + sets, model)]),
        matrix=pl.concat([model.matrix, _linking_matrix(members), _cardinality_matrix(members)]),
        sos=model.sos.clear(),
        row_starts=_row_starts(model, members),
        column_count=model.column_count + binaries,
        row_count=model.row_count + linking + sets,
    )


@dataclass(frozen=True)
class _Members:
    """Every member of every set, with the row and column it is about to make.

    One pass over the stream produces the whole arithmetic, so each frame
    below is a scatter into a buffer rather than a second walk over the sets.

    Attributes:
        col: The model column each member already is, in ``(set, weight)``
            order.
        magnitude: :math:`-M_i`, the coefficient linking a member to the
            binaries that admit it.
        binary: The binary a member opens — or, where it opens none, the next
            one to be assigned, which is one past the segment closing into it.
        closes: Whether a segment closes into a member — every member of a
            SOS2 set but its first.
        entries: How many matrix entries a member's linking row holds: its own
            column, plus the segment it opens and the one closing into it,
            each where there is one.
        cardinality: Every binary, in set order, which is the whole of the
            cardinality block's columns.
        set_widths: How many of them each set owns — one cardinality row's
            width, counted where the counting is free rather than off the
            finished block.
    """

    col: npt.NDArray[Any]
    magnitude: npt.NDArray[np.float64]
    binary: npt.NDArray[Any]
    closes: npt.NDArray[np.bool_]
    entries: npt.NDArray[np.int64]
    cardinality: npt.NDArray[Any]
    set_widths: npt.NDArray[np.int64]


def _members(model: ModelTables) -> _Members:
    """The stream read once into the arithmetic every frame below scatters.

    **A set's shape is read off its edges**, never off a group-by: a member is
    the first of its set when the row above belongs to another and the last
    when the row below does. The rest follows — a last member holds no binary,
    a first closes no segment, and a **SOS2 set of one is both**, so it is
    dropped whole (one nonzero is already at most two, and there is no segment
    to hold a binary; linopy returns early on the same case).

    Which binary a member reaches is the running count of those assigned
    before it, its own being the next to be assigned.

    **Nothing is dropped unless a set is that singleton**, which no common
    shape has, so whether any is is asked before every array is compacted for
    the answer.

    Raises:
        DataError: A member a big-M cannot stand in for.
    """
    col = model.sos.get_column('col').to_numpy()
    magnitude = np.minimum(model.cols.get_column('ub').to_numpy()[col], model.sos.get_column('big_m').to_numpy())
    _refuse_unbounded(model, col, magnitude)

    first, last = _edges(model.sos.get_column('set').to_numpy())
    one = model.sos.get_column('type').to_numpy() == 1
    kept = one | ~first | ~last
    if not kept.all():
        first, last, one, col, magnitude = (v[kept] for v in (first, last, one, col, magnitude))

    opens, closes = one | ~last, ~one & ~first
    binary = (np.cumsum(opens, dtype=np.int64) - opens + model.column_count).astype(col.dtype)
    holders = np.flatnonzero(first[opens])
    return _Members(
        col=col,
        magnitude=-magnitude,
        binary=binary,
        closes=closes,
        entries=opens.astype(np.int64) + closes + 1,
        cardinality=binary[opens],
        set_widths=np.diff(holders, append=int(np.count_nonzero(opens))),
    )


def _edges(sets: npt.NDArray[Any]) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Which members begin and which end a run of one set.

    The one comparison everything else here is derived from, and the reason
    the stream's ``(set, weight)`` order is a contract rather than a
    convenience.
    """
    first = np.empty(len(sets), dtype=np.bool_)
    first[:1] = True
    np.not_equal(sets[1:], sets[:-1], out=first[1:])
    last = np.empty_like(first)
    last[-1:] = True
    last[:-1] = first[1:]
    return first, last


def _refuse_unbounded(model: ModelTables, col: npt.NDArray[Any], magnitude: npt.NDArray[np.float64]) -> None:
    """Refuse a member no finite big-M can stand in for — linopy's two conditions.

    Asked of the big-M rather than of the bound, in that order for the reason
    the order exists: ``big_m:`` is what a caller declares *because* the bound
    is open, so asking first would refuse the model the answer was given for.

    The first is asked of the whole model's bounds before it is asked of the
    members: no member can have a negative lower bound where no column does,
    and that is one comparison against a column already in hand rather than a
    gather of one bound per member.

    Raises:
        DataError: Either condition, counted rather than located: a member is
            a column index, and what a caller acts on is the bound.
    """
    lb = model.cols.get_column('lb')
    for offending, what, fix in (
        (
            int(np.count_nonzero(lb.to_numpy()[col] < 0)) if lb.lt(0).any() else 0,
            'a negative lower bound',
            'A set says which members are nonzero, and the big-M form a sink without SOS is handed '
            'can only say that of a non-negative variable. Give the variable `bounds: {lower: 0}`',
        ),
        (
            int(np.count_nonzero(np.isinf(magnitude))),
            'no upper bound and no big_m',
            'so there is no finite coefficient to link them to a binary with. Bound the variable, or '
            'set `big_m:` on the sos block',
        ),
    ):
        if offending:
            raise DataError(
                f'{offending} SOS member(s) have {what}: {fix} — or solve with a sink that takes '
                f'the set natively (`gurobi`, or an LP file), which needs neither.'
            )


def _linking_matrix(members: _Members) -> pl.DataFrame:
    """``x_i - M_i * (the binaries that admit it) <= 0``, one row per member.

    **Each member owns a span, and its entries are written into it**, which is
    what keeps the block in CSR order with nothing sorted: a row holds its own
    column, then the segment closing into it, then the segment it opens, and
    those are already ascending — a model column comes before any appended
    binary, and a closing segment is the one before the opening. Sorting the
    block in polars instead is the largest thing here.

    So a span is **broadcast and then corrected**, twice, rather than
    scattered entry by entry: every entry of a member's row but the first
    carries :math:`-M_i` and names the binary it opens, and the one entry that
    is neither is its own. What is left over is the segment closing into a
    member, which is the binary before that one — the previous member of a
    SOS2 set always opening one, only a last member not — so it is the same
    broadcast, decremented where it lands.
    """
    at = np.cumsum(members.entries) - members.entries
    col = np.repeat(members.binary, members.entries)
    coeff = np.repeat(members.magnitude, members.entries)

    col[at] = members.col
    coeff[at] = 1.0
    col[(at + 1)[members.closes]] -= 1
    return pl.DataFrame({'col': col, 'coeff': coeff})


def _cardinality_matrix(members: _Members) -> pl.DataFrame:
    """``sum(a set's binaries) <= 1``, one row per set.

    Every binary, keyed by its set's row rather than by the member that opened
    it. In order already: sets ascend, and within one, so do the binaries.
    """
    return pl.DataFrame({'col': members.cardinality, 'coeff': np.ones(len(members.cardinality))})


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


def _row_starts(model: ModelTables, members: _Members) -> Any:
    """The CSR index, extended by what each appended row owns.

    Counted where the counting was free rather than off the finished block:
    reading it back off the entries would mean a pass over every one of them,
    which is the largest thing here.
    """
    lengths = np.concatenate([members.entries, members.set_widths])
    return np.concatenate([model.row_starts, model.row_starts[-1] + np.cumsum(lengths)])
