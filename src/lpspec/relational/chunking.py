"""How every batched pass in the engine picks its chunk size.

One rule: a pass has a *budget* in elements and walks *units* carrying
``width`` of them, so it takes ``budget // width`` at a time. The two passes
that need it — the solver hand-off and the constraint text, both through
:class:`~lpspec.relational.sinks.tables.Tables` — differ only in what a
unit is: a row costs its average nonzeros, a column costs one.

The width is the part that gets forgotten, and forgetting it does not look like
a bug: chunking a matrix by rows with no width reads as bounded and is not,
since a row is nine entries in one model and a hundred in another, so what the
pass holds tracks the model's shape rather than the budget. Requiring a width
at every call site is the point of this module — a pass whose unit really does
cost one element says so, in one character, where a reviewer can see it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


def ranges(total: int, budget: int, width: float) -> Iterator[tuple[int, int]]:
    """Half-open ``[lo, hi)`` ranges covering ``[0, total)``.

    Each holds about ``budget`` elements, given that one unit costs ``width``
    of them. A ``width`` below 1 is read as 1 — a unit cannot cost less than
    itself, and a fractional average (0.4 nonzeros per row, in a model that is
    mostly bounds) would otherwise ask for chunks wider than the budget.

    Empty input yields nothing rather than one empty range: a caller looping
    over ``ranges`` should do no work, not one pass over nothing.
    """
    per_chunk = max(1, int(budget // max(1.0, width)))
    for lo in range(0, total, per_chunk):
        yield lo, min(lo + per_chunk, total)
