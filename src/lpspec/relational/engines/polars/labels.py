"""Dense solver indices for a masked coordinate product.

**Labels are the one place order is load-bearing.** ``var_label`` *is* the
solver's column index and ``row`` its row index, so a label is the model's
identity: two builds of one model must agree on it integer for integer
(docs/about/architecture.md, "The relational lane").

Variables and constraint rows are the same operation over different frames, so
:func:`frame` is written once — one rule, sort the survivors into declaration
order and number them from *start*. A mask, a restriction or neither produce
the same shape down to the schema.

The one split kept is *how much product is materialised*. A mask that cannot
see the leading dims removes the same coordinates under every one of their
values, so the survivors are a rectangle and only the masked suffix needs rows
(:func:`_factored`): labelling one time-invariantly-masked variable through the
full product costs a large peak the rectangle avoids entirely, where the
rectangle is a few hundred rows plus the output (#520).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from lpspec.relational.engines.polars.compiler import UNIT, ordinal, predicate_dims

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lpspec.relational import plan
    from lpspec.relational.engines.polars.compiler import PolarsCompiler
    from lpspec.relational.engines.polars.fragments import Presence


def frame(
    compiler: PolarsCompiler,
    dims: tuple[str, ...],
    where: plan.Predicate | None,
    label: str,
    start: int,
    restrictions: Sequence[Presence] = (),
) -> pl.DataFrame:
    """The masked coord product of *dims* with a dense *label* from *start*.

    A label follows declaration order — row-major over the dims' declared
    ordinals — which is what lets it *be* the solver's own index with no
    remapping.

    *restrictions* are variable-presence frames a constraint row must be
    contained in (v1 ``convention.rst`` §6, §12). They are semi-joins, so they
    only remove rows, and nothing deduplicates them — a key occurring twice
    still occurs. Which rows they remove is unknown until data is read, so a
    restriction takes the counted path whatever the mask looks like.

    No dims means the carrier is :data:`UNIT`, selected because selecting
    nothing would drop the one row of the empty coordinate product.

    **Nothing sorts unless the data says it must.** The product is *produced*
    in declaration order, a filter keeps it and a semi-join usually does, so
    :func:`in_position_order` verifies linearly and sorts only when the engine
    emitted another order. The unconditional sort was a third of a build on the
    widest case measured (#520).

    **Nothing renumbers unless a row was dropped**, either. With neither mask
    nor restriction, ``start + position`` *is* the label and the row-index pass
    never runs — milliseconds per declaration, on a model that may carry dozens
    (#520).

    Returns:
        ``(dims…, label)`` in that column order and in label order; the next
        free label is ``start`` plus its height.
    """
    if where is not None and not restrictions:
        free = _free_prefix(dims, predicate_dims(where, compiler.name_dims))
        if free:
            factored = _factored(compiler, dims, free, where, label, start)
            if factored is not None:
                return factored

    surviving = compiler.frame(dims, where)
    for restriction in restrictions:
        surviving = restriction.restrict(surviving, restriction.keyed_by or ())

    dropped = where is not None or bool(restrictions)
    numbering = _row_major(compiler, dims)
    if not dropped:
        numbering = pl.lit(start, dtype=pl.Int64) + numbering
    position = '#position' if dropped else label
    materialised = in_position_order(
        surviving.select(*(dims or (UNIT,)), numbering.alias(position)).collect(engine='streaming'),
        position,
    )
    if dropped:
        materialised = materialised.with_row_index(label, offset=start).with_columns(pl.col(label).cast(pl.Int64))
    return materialised.select(*dims, pl.col(label).set_sorted())


def declared_height(compiler: PolarsCompiler, dims: tuple[str, ...], where: plan.Predicate | None) -> int:
    """How many rows a declaration *asks* for: its coord product under its own mask.

    The count :func:`frame` would return if no variable's absence restricted it,
    so the difference between the two is the rows a propagated absence removed —
    which nothing else records, a restricted row never existing to be counted
    (#944).

    **Unmasked, it is arithmetic**: the product of the dims' own index heights,
    which are thousands of rows where the product they span is millions. That is
    the case worth being cheap in, being exactly the shape the count exists to
    report — a constraint that wrote no ``where`` of its own and lost rows to a
    variable's absence anyway.

    With a mask it costs a pass over the masked product, and is therefore asked
    only where there is a restriction to attribute rows to. Projection pushdown
    leaves it a count rather than a materialisation: no column is read.
    """
    if where is None:
        height = 1
        for d in dims:
            height *= int(compiler.data.dimensions[d].select(pl.len()).collect().item())
        return height
    return int(compiler.frame(dims, where).select(pl.len()).collect(engine='streaming').item())


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
    ranked: on `dispatch`, the generators rather than 10M
    ``(snapshot, generator)`` pairs. The label is then arithmetic, row-major
    over the leading dims times the surviving set's width plus a survivor's
    rank, which is the number the counted path would have counted since each
    leading coordinate sees the same survivors in the same order.

    The prefix must be *leading* rather than merely unread by the mask: only a
    prefix leaves the surviving set contiguous within declaration order.
    ``None`` when nothing survives, the counted path already answering the
    empty case with the right columns and dtypes.

    **The survivors go on the left of the cross join**, the side the streaming
    engine cycles fastest, so survivors turning over within each head
    coordinate is label order and :func:`in_position_order` permutes nothing.
    The other way round sorts the whole variable frame on every build, which
    is close to half again the labelling cost (#520). Which side
    cycles is an implementation detail of a dependency asserted nowhere: the
    verify is what
    makes it safe to exploit, and would turn a change in polars back into a
    sort rather than into wrong labels. Rearranging means re-measuring.
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
        survivors.lazy()
        .join(compiler.frame(head, None).select(*head, _row_major(compiler, head).alias(position)), how='cross')
        .select(
            *dims,
            (pl.lit(start, dtype=pl.Int64) + pl.col(position) * width + pl.col(rank)).alias(label),
        )
        .collect(engine='streaming')
    )
    return in_position_order(labelled, label).with_columns(pl.col(label).set_sorted())


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


def row_major(compiler: PolarsCompiler, dims: tuple[str, ...], ordinals: Callable[[str], pl.Expr]) -> pl.Expr:
    """A coordinate's row-major position in the declared product.

    Horner over the declared ordinals — one multiply and one add per dim
    whatever the arity; with no dims, the literal zero of the empty product's
    one row. Dense over the *full* product rather than the survivors, so a
    caller that dropped rows renumbers with a row index (a label may not have
    gaps, a declaration's share of the solver vector being a slice) and one
    that dropped none has the label already, offset by ``start``.

    *ordinals* says how the frame in hand carries a dim's ordinal, which is
    the one thing the two askers differ on: a compiler frame has the column
    beside the label (:func:`_row_major`), while a *built* variable frame kept
    only the label, so a set numbers its members through
    :meth:`~lpspec.relational.engines.polars.compiler.PolarsCompiler.ordinal_of`.
    The rule itself is the same one, and stays here because a second copy is
    how two builds of one model would come to disagree about an index.
    """
    position: pl.Expr = pl.lit(0, dtype=pl.Int64)
    for d in dims:
        position = position * compiler.data.cardinality[d] + ordinals(d)
    return position.cast(pl.Int64)


def _row_major(compiler: PolarsCompiler, dims: tuple[str, ...]) -> pl.Expr:
    """:func:`row_major` over a compiler frame, which carries its ordinals."""
    return row_major(compiler, dims, lambda d: pl.col(ordinal(d)))


def in_position_order(materialised: pl.DataFrame, position: str) -> pl.DataFrame:
    """The frame ordered by *position*, verified rather than re-established.

    One linear ``is_sorted`` against a single column, and a single-key sort
    only when the engine emitted another order. The witness column stays for a
    caller to project away. All three orderings in the lane go through here: a
    second copy of "check before you sort" is a second thing to get backwards.
    """
    if materialised.get_column(position).is_sorted():
        return materialised
    return materialised.sort(position)
