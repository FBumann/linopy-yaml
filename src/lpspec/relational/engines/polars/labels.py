"""Dense solver indices for a masked coordinate product.

**Labels are the one place order is load-bearing.** ``var_label`` *is* the
solver's column index and ``row`` its row index, so a label is not a detail of
how the executor happens to number things — it is the model's identity, and two
builds of one model must agree on it integer for integer (docs/ARCHITECTURE.md,
"The relational lane").

Variables and constraint rows are the same operation over different frames, so
:func:`frame` is written once; twice is how the two would come to disagree
about which coordinate gets which index. One rule: sort the surviving
coordinates into declaration order and number them from *start*. A mask, a
restriction, or neither all produce the same shape, so a mask that removes
nothing is indistinguishable from no mask — down to the schema.

The one split kept is *how much product is materialised*, not what a label is.
A mask that cannot see the leading dims removes the same coordinates under
every one of their values, so the survivors are a rectangle and only the
masked suffix needs to exist as rows (:func:`_factored`). Measured on
`sector/m`: labelling one time-invariantly-masked variable through the full
product transiently peaked +177 MB and 17 ms where the rectangle costs the
suffix — a few hundred rows — plus the output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from lpspec.relational.engines.polars.compiler import UNIT, ordinal, predicate_dims, restrict_by_presence

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
    rows. Which rows is not known until data is read, which is why a
    restriction takes the counted path below whatever the mask looks like.

    Being semi-joins is also why nothing deduplicates them: a semi-join asks
    whether a key occurs, and a key occurring twice still occurs.

    No dims means the carrier is `UNIT`, which is selected in that case because
    selecting nothing would drop the one row of the empty coordinate product.

    **Nothing here sorts unless the data says it must.** Declaration order is
    row-major over the ordinals, which is one arithmetic expression
    (:func:`_row_major`) — and the product is *produced* in that order, a
    filter keeps it, and a semi-join usually does, so the sort that would
    re-establish it usually permutes nothing. :func:`in_position_order`
    verifies that linearly and sorts only when the engine emitted another
    order. The unconditional sort this replaces was 0.26 s of a 0.73 s build
    at ``dispatch/l``, against ~0.01 s for the verify.
    """
    if where is not None and not restrictions:
        free = _free_prefix(dims, predicate_dims(where, compiler.name_dims))
        if free:
            factored = _factored(compiler, dims, free, where, label, start)
            if factored is not None:
                return factored, start + factored.height

    surviving = compiler.frame(dims, where)
    for on, presence in restrictions:
        surviving = restrict_by_presence(surviving, presence, on)

    position = '#position'
    materialised = in_position_order(
        surviving.select(*(dims or (UNIT,)), _row_major(compiler, dims).alias(position)).collect(engine='streaming'),
        position,
    )
    materialised = (
        materialised.with_row_index(label, offset=start)
        .select(*dims, pl.col(label).cast(pl.Int64))
        .with_columns(pl.col(label).set_sorted())
    )
    return materialised, start + materialised.height


def _factored(
    compiler: PolarsCompiler,
    dims: tuple[str, ...],
    free: int,
    where: plan.Predicate,
    label: str,
    start: int,
) -> pl.DataFrame | None:
    """Labels for a mask that reads none of the first *free* dims.

    The survivors are a rectangle — the full product of the leading dims
    against one surviving suffix set — so only the suffix is materialised and
    ranked (a sort of the set, not of the product: on `dispatch` the
    generators rather than 10M ``(snapshot, generator)`` pairs). The label is
    then arithmetic: row-major over the leading dims, times the width of the
    surviving set, plus a survivor's rank within it — the same number the
    counted path would have counted, because each leading coordinate sees the
    same survivors in the same order.

    The prefix has to be *leading* rather than merely absent from the mask: a
    label follows declaration order, and only a prefix leaves the surviving
    set contiguous within it.

    ``None`` when nothing survives: the rectangle degenerates and the counted
    path already answers the empty case with the right columns and dtypes.

    The head product goes on the *left* of the cross join because the
    streaming engine emits the right side fastest — survivors cycling within
    each head coordinate is label order, so the verify below permutes nothing.
    """
    head, kept = dims[:free], dims[free:]
    rank = '#rank'
    survivors = (
        compiler.frame(kept, where)
        .sort([ordinal(d) for d in kept])
        .select(*kept)
        .with_row_index(rank)
        .collect(engine='streaming')
    )
    width = survivors.height
    if width == 0:
        return None

    position = '#position'
    labelled = (
        compiler.frame(head, None)
        .select(*head, _row_major(compiler, head).alias(position))
        .join(survivors.lazy(), how='cross')
        .select(
            *dims,
            (pl.lit(start, dtype=pl.Int64) + pl.col(position) * width + pl.col(rank)).alias(label),
        )
        .collect(engine='streaming')
    )
    if labelled.height and not labelled.get_column(label).is_sorted():
        labelled = labelled.sort(label)
    return labelled.with_columns(pl.col(label).set_sorted())


def _free_prefix(dims: tuple[str, ...], touched: frozenset[str]) -> int:
    """How many leading dims the mask does not read.

    Leading, not merely absent: a label follows declaration order, so only a
    prefix leaves the surviving set contiguous under each of its coordinates.
    Returns 0 when the mask reads the first dim — the case that has to count
    its survivors the slow way — and 0 again when *no* dim is read, where the
    split would gain nothing over the one-path arithmetic.
    """
    free = 0
    while free < len(dims) and dims[free] not in touched:
        free += 1
    return free if free < len(dims) else 0


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


def in_position_order(materialised: pl.DataFrame, position: str) -> pl.DataFrame:
    """The frame ordered by *position*, verified rather than re-established.

    One linear ``is_sorted`` against a single column, and the sort — also
    single-key, where sorting the ordinal columns was one key per dim — only
    when the engine emitted another order. The position column leaves here
    dropped either way; it was only ever the order's witness. The executor's
    per-variable ``cols`` share leans on the same idiom with ``var_label`` as
    the witness.
    """
    ordered = materialised.get_column(position).is_sorted()
    return (materialised if ordered else materialised.sort(position)).drop(position)
