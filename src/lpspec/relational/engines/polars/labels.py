"""Dense solver indices for a masked coordinate product.

**Labels are the one place order is load-bearing.** ``var_label`` *is* the
solver's column index and ``row`` its row index, so a label is not a detail of
how the executor happens to number things — it is the model's identity, and two
builds of one model must agree on it integer for integer (docs/ARCHITECTURE.md,
"The relational lane").

Variables and constraint rows are the same operation over different frames, so
:func:`frame` is written once; twice is how the two would come to disagree
about which coordinate gets which index. It is one rule with no special cases:
sort the surviving coordinates into declaration order and number them from
*start*. A mask, a restriction, or neither all take the same path, so a mask
that removes nothing is indistinguishable from no mask — down to the schema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from lpspec.relational.engines.polars.compiler import UNIT, ordinal, restrict_by_presence

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lpspec.relational import plan
    from lpspec.relational.engines.polars.compiler import PolarsCompiler


def frame(
    compiler: PolarsCompiler,
    dims: tuple[str, ...],
    where: plan.Predicate | None,
    label: str,
    start: int,
    restrictions: Sequence[tuple[tuple[str, ...], pl.LazyFrame]] = (),
) -> tuple[pl.DataFrame, int]:
    """The masked coord product of *dims* with a dense *label* from *start*.

    Returns ``(dims…, label)`` in that column order, in label order, together
    with the next free label. A label follows declaration order — row-major
    over the dims' declared ordinals — which is what lets it *be* the solver's
    own index with no remapping.

    *restrictions* are variable-presence frames a constraint row must be
    contained in: absence propagates into a comparison and drops the row (v1
    ``convention.rst`` §6, §12). They are semi-joins, so they can only remove
    rows.

    Being semi-joins is also why nothing deduplicates them: a semi-join asks
    whether a key occurs, and a key occurring twice still occurs.
    """
    surviving = compiler.frame(dims, where)
    for on, presence in restrictions:
        surviving = restrict_by_presence(surviving, presence, on)

    materialised = (
        surviving.sort([ordinal(d) for d in dims])
        # No dims means the carrier is `UNIT`: selecting nothing would drop the
        # one row of the empty coordinate product.
        .select(*(dims or (UNIT,)))
        .with_row_index(label, offset=start)
        .select(*dims, pl.col(label).cast(pl.Int64))
        .collect(engine='streaming')
    )
    return materialised, start + materialised.height
