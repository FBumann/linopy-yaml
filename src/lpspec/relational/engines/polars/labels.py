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

    No dims means the carrier is `UNIT`, which is selected in that case because
    selecting nothing would drop the one row of the empty coordinate product.

    **Nothing here sorts unless the data says it must.** Declaration order is
    row-major over the ordinals, which is one arithmetic expression
    (:func:`_row_major`) — and the product is *produced* in that order, a
    filter keeps it, and a semi-join usually does, so the sort that would
    re-establish it usually permutes nothing. :func:`_in_position_order`
    verifies that linearly and sorts only when the engine emitted another
    order. The unconditional sort this replaces was 0.26 s of a 0.73 s build
    at ``dispatch/l``, against ~0.01 s for the verify.
    """
    surviving = compiler.frame(dims, where)
    for on, presence in restrictions:
        surviving = restrict_by_presence(surviving, presence, on)

    position = '#position'
    materialised = _in_position_order(
        surviving.select(*(dims or (UNIT,)), _row_major(compiler, dims).alias(position)).collect(engine='streaming'),
        position,
    )
    materialised = (
        materialised.with_row_index(label, offset=start)
        .select(*dims, pl.col(label).cast(pl.Int64))
        .with_columns(pl.col(label).set_sorted())
    )
    return materialised, start + materialised.height


def _row_major(compiler: PolarsCompiler, dims: tuple[str, ...]) -> pl.Expr:
    """A coordinate's row-major position in the declared product.

    Horner over the declared ordinals — one multiply and one add per dim
    whatever the arity. With no dims the position is the literal zero: the
    empty product's one row. Dense over the *full* product, not the
    survivors, which is why the caller renumbers with a row index rather
    than reading this as the label: a mask leaves gaps, and a label may not
    have any (a declaration's share of the solver vector is a slice).
    """
    position: pl.Expr = pl.lit(0, dtype=pl.Int64)
    for d in dims:
        position = position * compiler.dimension_cardinality[d] + pl.col(ordinal(d))
    return position.cast(pl.Int64)


def _in_position_order(materialised: pl.DataFrame, position: str) -> pl.DataFrame:
    """The frame in declaration order, verified rather than re-established.

    One linear ``is_sorted`` against a single Int64 column, and the sort —
    also single-key, where sorting the ordinal columns was one key per dim —
    only when the engine emitted another order. The position column leaves
    here dropped either way; it was only ever the order's witness.
    """
    ordered = materialised.get_column(position).is_sorted()
    return (materialised if ordered else materialised.sort(position)).drop(position)
