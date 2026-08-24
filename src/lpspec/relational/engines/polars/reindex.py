"""Re-indexing along one dimension's own order: ``shift`` and ``sum_back``.

The two operators that move a fragment's rows *along* a dimension rather than
across dims. ``shift`` is a pointwise remap of the dim through its ordinal —
one output row per input row — and ``sum_back`` a one-to-many one, a row at
*o* contributing at every ``o + lag`` inside the window. They share the
ordinal arithmetic, the scratch columns below, and the question no other
operator has to answer: what happens at the edge, where the walk runs out of
dimension.

Both take the :class:`~lpspec.relational.engines.polars.compiler.PolarsCompiler`
and hold nothing. They read four things off it — ``data``, ``program``,
``partitioned`` and ``widen`` — and everything else here is their own.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import polars as pl

from lpspec.relational.engines.polars.fragments import (
    Presence,
    TermFragment,
    refuse_a_fragment_without_the_dims,
)

if TYPE_CHECKING:
    from lpspec.relational import plan
    from lpspec.relational.engines.polars.compiler import PolarsCompiler


#: Scratch columns. The spaces make them unrepresentable as declared names, so
#: they cannot collide with a dimension or lookup the model already has.
#: The per-entity offset, joined in beside the ordinal it moves.
_OFFSET = '__offset'
_LAG = '__lag'
_WIDTH = '__width'
_ORD_IN = '__ord in__'
_ORD_OUT = '__ord out__'
_POS = '__pos in group__'
_SPAN = '__group size__'


def window_fragment(compiler: PolarsCompiler, p: TermFragment, s: plan.Window, context: str) -> TermFragment:
    """A one-to-many remap of the dim through its ord.

    A row at *o* contributes at every ``o + lag`` for ``lag`` inside the
    window, so the terms land on each output position that can see them and
    the terminal ``sum(coeff)`` at assembly adds them up — the same trick
    :meth:`_sum_fragment` relies on, which is why this needs no aggregate.

    The lag table is built to the widest window the data asks for; a named
    width then keeps only the lags that entity reaches. Every join is still
    on a dim-table key or the width's own dims, so the reach stays a lookup
    and the locality class is the one :meth:`translate_fragment` has.

    Unlike a shift this vacates nothing: the window at the first position
    is short rather than empty, since it always contains that position
    itself. So an operand with no presence gains none.
    """
    if s.dimension not in p.dims:
        refuse_a_fragment_without_the_dims(p, [s.dimension], context, f'sum_back(over={s.dimension!r})')
    card = compiler.data.cardinality[s.dimension]
    table = compiler.data.dimensions[s.dimension]
    incoming = table.select(pl.col('val').alias(s.dimension), pl.col('ord').alias(_ORD_IN))
    outgoing = table.select(pl.col('val').alias(s.dimension), pl.col('ord').alias(_ORD_OUT))

    width_name = s.width if isinstance(s.width, str) else None
    width_dims: tuple[str, ...] = ()
    if width_name is not None:
        width_dims = tuple(compiler.program.parameter(width_name).dims)
        widest = int(compiler.data.parameters[width_name].select(pl.col('value').max()).collect().item() or 0)
    else:
        assert not isinstance(s.width, str)
        widest = s.width
    lags = pl.LazyFrame({_LAG: pl.Series(range(min(widest, card)), dtype=pl.Int64)})

    moved = pl.col(_ORD_IN) + pl.col(_LAG)
    if s.wrap:
        moved = moved % card

    def remap(source: pl.LazyFrame, carried: list[str], source_dims: tuple[str, ...] | None = None) -> pl.LazyFrame:
        kept = [d for d in (source_dims if source_dims is not None else p.dims) if d != s.dimension]
        walked = source.join(incoming, on=s.dimension, how='inner').drop(s.dimension).join(lags, how='cross')
        if width_name is not None:
            walked = walked.join(
                compiler.data.parameters[width_name].select(*width_dims, pl.col('value').cast(pl.Int64).alias(_WIDTH)),
                on=list(width_dims),
                how='inner',
            ).filter(pl.col(_LAG) < pl.col(_WIDTH))
        return (
            walked.with_columns(moved.alias(_ORD_OUT))
            .join(outgoing, on=_ORD_OUT, how='inner')
            .select(*kept, s.dimension, *carried)
        )

    def travelled(presence: Presence) -> Presence:
        keyed_by, source = presence.keyed_by, presence.frame
        if keyed_by is not None and s.dimension not in keyed_by:
            source, keyed_by = compiler.widen(source, keyed_by, p.dims), None
        return Presence(remap(source, [], keyed_by).unique(), keyed_by)

    frame = remap(p.frame, p.carried)
    return TermFragment(p.dims, frame, p.kind, presences=tuple(travelled(x) for x in p.presences))


def translate_fragment(compiler: PolarsCompiler, p: TermFragment, s: plan.Translate, context: str) -> TermFragment:
    """A pointwise remap of the dim through its ord.

    A row at *o* contributes at ``(o + by) % card``.

    Both joins are on a dim-table key, so the row count is unchanged and an
    out-of-range ordinal does not join. No window function; bounded-halo
    locality. The operand's *presences* are :func:`travelled_presences` below.

    Every fill over a *constant* is written, ``0`` included (#551): the
    arithmetic is unchanged, but the slot now has a value, so asking for
    zero stops being indistinguishable from having nothing. Over a *term*
    there is nothing to write — ``edge=0`` on a variable means the vacated
    slot contributes no term at all (the operator rules), where a zero-coefficient
    entry would be a matrix nonzero standing for a term that is not there.
    Lowering refuses every other numeric edge over a variable.
    """
    if s.dimension not in p.dims:
        refuse_a_fragment_without_the_dims(p, [s.dimension], context, f'shift(over={s.dimension!r})')
    card = compiler.data.cardinality[s.dimension]
    others = [d for d in p.dims if d != s.dimension]
    table = compiler.data.dimensions[s.dimension]
    if s.partition is not None:
        # Inside a group the neighbour is decided by position within *that*
        # group, so both joins read a rank rather than the axis-wide `ord`
        # and a wrap closes on the group's own size. A coordinate the map
        # places nowhere is not in this table at all and joins to nothing,
        # which is what it reaches everywhere else.
        table = compiler.partitioned(s.dimension, s.partition)
        grouped = pl.col(s.partition)
        table = table.with_columns(
            (pl.col('ord').rank('ordinal').over(grouped) - 1).cast(pl.Int64).alias(_POS),
            pl.len().over(grouped).cast(pl.Int64).alias(_SPAN),
        )
    position = _POS if s.partition is not None else 'ord'
    group_cols = [pl.col(s.partition)] if s.partition is not None else []
    incoming = table.select(
        pl.col('val').alias(s.dimension),
        pl.col(position).alias(_ORD_IN),
        *group_cols,
        *([pl.col(_SPAN)] if s.partition is not None else []),
    )
    outgoing = table.select(pl.col('val').alias(s.dimension), pl.col(position).alias(_ORD_OUT), *group_cols)

    named_offset = isinstance(s.offset, str)
    if named_offset:
        moved = pl.col(_ORD_IN) + pl.col(_OFFSET)
    else:
        assert not isinstance(s.offset, str)
        moved = pl.col(_ORD_IN) + s.offset
    if s.wrap:
        span = pl.col(_SPAN) if s.partition is not None else pl.lit(card)
        moved = (moved % span + span) % span

    def remap(source: pl.LazyFrame, carried: list[str], source_dims: tuple[str, ...] | None = None) -> pl.LazyFrame:
        """*source*, with ``s.dimension`` moved by ``s.offset``.

        ``source_dims`` is the caller's, because a presence frame need not
        carry the fragment's: an acyclic shift's presence speaks only about
        the dim it vacated, so projecting the fragment's dims onto it asks
        for columns it never had.
        """
        kept = [d for d in (source_dims if source_dims is not None else p.dims) if d != s.dimension]
        walked = source.join(incoming, on=s.dimension, how='inner').drop(s.dimension)
        if named_offset:
            # A per-entity offset is one more equi-join, on keys the frame
            # already carries — the same locality class as a literal one,
            # since the reach is still a lookup on the dim table.
            offsets, keys = _offsets(compiler, s)
            walked = walked.join(offsets, on=keys, how='inner')
        landing = [_ORD_OUT, s.partition] if s.partition is not None else [_ORD_OUT]
        return (
            walked.with_columns(moved.alias(_ORD_OUT))
            .join(outgoing, on=landing, how='inner')
            .select(*kept, s.dimension, *carried)
        )

    def travelled_presences() -> tuple[Presence, ...]:
        """Where the variable exists after the shift, and what keys it.

        An existing presence **travels**: the coordinate set goes through
        the same map the rows did, and the inner join drops whatever the
        edge vacated. Under a fill the vacated positions go back in
        (:meth:`_vacated`) — a filled slot counts as present. A narrow
        presence is widened first when the shift moves a dim it is silent
        about, since there is no column to remap otherwise and joining the
        row set on a column it never had is #546; :meth:`_widen` changes no
        answer, and what comes back is full-width.

        An operand with **no** presence gets one: nothing was absent before
        and the acyclic edge now is, where without this the vacated slot
        would merely fail to join and the row would survive with its term
        quietly gone (#239, #289). It is keyed by the one dimension it
        speaks about, which is what :attr:`Presence.keyed_by` is
        for — keying it by the fragment's dims would materialise the whole
        coordinate product to name an edge, which costs a fifth again of
        build on a wide ramp (#520), a shape no case in `bench/` covers.
        """
        if not p.presences:
            if s.wrap or s.fill is not None:
                # A policy speaks about a group's edge, and a coordinate in
                # no group has none — it is absent under every policy, there
                # being nothing to come round from or to fill from.
                return () if s.partition is None else (Presence(_grouped(compiler, s), (s.dimension,)),)
            return (Presence(_edge(compiler, s, card, vacated=False), (s.dimension, *offset_dims(compiler, s))),)
        return tuple(travelled(x) for x in p.presences)

    def travelled(presence: Presence) -> Presence:
        source, keyed_by = presence.frame, presence.keyed_by
        if keyed_by is not None and not {s.dimension, *offset_dims(compiler, s)}.issubset(keyed_by):
            source, keyed_by = compiler.widen(source, keyed_by, p.dims), None
        moved_presence = remap(source, [], keyed_by)
        if s.wrap or s.fill is None:
            return Presence(moved_presence, keyed_by)
        vacated = _vacated(compiler, presence, p.dims, s, card, others)
        return Presence(pl.concat([moved_presence, vacated], how='vertical_relaxed').unique())

    frame = remap(p.frame, p.carried)
    if not s.wrap and s.fill is not None and p.kind == 'const':
        frame = pl.concat([frame, _filled_edge(compiler, s, card, others, s.fill)], how='vertical_relaxed')
    return replace(p, frame=frame, presences=travelled_presences())


def _filled_edge(
    compiler: PolarsCompiler, s: plan.Translate, card: int, others: list[str], fill: float
) -> pl.LazyFrame:
    """``(dims…, cval=fill)`` at every coordinate the shift vacated.

    Dense over *others*, not over the rows the operand happened to carry:
    the eager lane shifts an array already reindexed to the master
    coordinates, so a fill appearing only where the parameter was
    non-sparse would be a second answer to the same question.

    Only a *truthy* fill gets here, ``fill=0`` needing no rows at all. A
    nonzero fill reaches a translation only over a variable-free operand,
    so this is always the const branch and never invents a ``var_label``.
    """
    edge = _edge(compiler, s, card, vacated=True)
    keyed = offset_dims(compiler, s)
    for d in others:
        if d in keyed:
            continue
        edge = edge.join(compiler.data.dimensions[d].select(pl.col('val').alias(d)), how='cross')
    return edge.with_columns(pl.lit(fill, dtype=pl.Float64).alias('cval')).select(*others, s.dimension, 'cval')


def offset_dims(compiler: PolarsCompiler, s: plan.Translate) -> tuple[str, ...]:
    """The dims a per-entity offset varies over — empty where it is a number.

    An edge is keyed by them as well as by the translated dimension: how
    far back a row reaches decides which rows have nothing to reach, so
    under a named offset the two entities of one coordinate need not agree
    about whether it is the edge.

    The grouped dimension is not among them: a lag per group varies along
    the translated dimension itself, which the edge is already keyed by.
    """
    if not isinstance(s.offset, str):
        return ()
    grouped = _grouped_into(compiler, s)
    return tuple(d for d in compiler.program.parameter(s.offset).dims if d != grouped)


def _grouped_into(compiler: PolarsCompiler, s: plan.Translate) -> str | None:
    """The dimension the partition groups into, where the offset is over it.

    ``None`` for every other shift — a numeric offset, an unpartitioned
    one, or one over dims the operand carries itself.
    """
    if s.partition is None or not isinstance(s.offset, str):
        return None
    targeted = {lk.name: lk.target for lk in compiler.program.dimension(s.dimension).lookups}
    target = targeted[s.partition]
    return target if target in compiler.program.parameter(s.offset).dims else None


def _offsets(compiler: PolarsCompiler, s: plan.Translate) -> tuple[pl.LazyFrame, list[str]]:
    """A named offset's values and the keys a frame reads them by.

    A **per-group** offset is declared over the dimension the partition
    groups into, and no frame carries a column of it: what travels with a
    coordinate is the lookup's own value, so the offset is read under the
    lookup's name and one equi-join lands each group's own lag (#1161).
    A coordinate the map places nowhere is in no partitioned table and
    joins to nothing, which is what it reaches everywhere else.
    """
    assert isinstance(s.offset, str)
    grouped = _grouped_into(compiler, s)
    dims = compiler.program.parameter(s.offset).dims
    keys = [str(s.partition) if d == grouped else d for d in dims]
    frame = compiler.data.parameters[s.offset].select(
        *(pl.col(d).alias(key) for d, key in zip(dims, keys, strict=True)),
        pl.col('value').cast(pl.Int64).alias(_OFFSET),
    )
    return frame, keys


def _edge(compiler: PolarsCompiler, s: plan.Translate, card: int, *, vacated: bool) -> pl.LazyFrame:
    """The coordinates an acyclic shift vacates, or keeps.

    Exact complements, so one filter negated rather than two conditions to
    keep in step: a fill and the presence set it implies must not disagree
    about which coordinates the edge is. One column wide under a numeric
    offset, an edge then being vacated for *every* combination of the other
    dims; under a named one, one column per :meth:`offset_dims` as well.

    Under a partition the edge is **each group's**, counted along the same
    within-group rank the translation itself walks: a coordinate reaches
    outside its own group exactly where it would have reached outside the
    axis. A coordinate in no group is neither — it is absent, the reading
    :meth:`_grouped` gives it, so it is dropped here rather than counted as
    an edge a policy could speak for (#1061). A per-group offset reaches it
    by the lookup rather than by a cross join, one lag standing for the
    whole group.
    """
    table = compiler.data.dimensions[s.dimension]
    if s.partition is not None:
        grouped = pl.col(s.partition)
        table = compiler.partitioned(s.dimension, s.partition).with_columns(
            (pl.col('ord').rank('ordinal').over(grouped) - 1).cast(pl.Int64).alias(_POS),
            pl.len().over(grouped).cast(pl.Int64).alias(_SPAN),
        )
        position, span = pl.col(_POS), pl.col(_SPAN)
    else:
        position, span = pl.col('ord'), pl.lit(card, dtype=pl.Int64)
    dims = offset_dims(compiler, s)
    if isinstance(s.offset, str):
        offsets, keys = _offsets(compiler, s)
        on = [key for key in keys if key == s.partition]
        table = table.join(offsets, on=on, how='inner') if on else table.join(offsets, how='cross')
        offset = pl.col(_OFFSET)
    else:
        offset = pl.lit(s.offset, dtype=pl.Int64)
    source = position - offset
    reaches = (source % span + span) % span if s.wrap else source
    outside = (reaches < 0) | (reaches >= span)
    return table.filter(outside if vacated else ~outside).select(pl.col('val').alias(s.dimension), *dims)


def _grouped(compiler: PolarsCompiler, s: plan.Translate) -> pl.LazyFrame:
    """The labels the partition lookup actually places in a group.

    The rest belong to none, so a translation reaches nothing for them and
    their rows are not built — the reading ``sum(by=)`` gives a label the
    map has no row for, and the one an edge policy cannot speak about.
    """
    return compiler.partitioned(s.dimension, str(s.partition)).select(pl.col('val').alias(s.dimension))


def _vacated(
    compiler: PolarsCompiler,
    presence: Presence,
    dims: tuple[str, ...],
    s: plan.Translate,
    card: int,
    others: list[str],
) -> pl.LazyFrame:
    """The edge positions ``shift`` leaves with nothing to move in.

    Reached only under ``fill=0``, which is the whole of what ``fill`` does
    here: back in the presence set they are present-with-no-term, a zero
    contribution and a surviving row. Left out, absence propagates and the
    row drops — linopy v1's reading of ``.shift()``.

    Only the ``shift`` edge qualifies; a coordinate the variable's own mask
    removed is genuinely absent and remapping already dropped it. So the
    edge is crossed with the other-dim combinations the variable actually
    has, one vacated row each.

    The incoming presence is widened to *others* first, since a narrowly
    keyed one — a pullback's, an earlier shift's — is silent about the
    columns this reads and asking for them is #546 all over again.
    """
    edge = _edge(compiler, s, card, vacated=True)
    if not others:
        return edge
    have = presence.keys(dims)
    source = presence.frame if all(d in have for d in others) else compiler.widen(presence.frame, have, dims)
    keys = [d for d in offset_dims(compiler, s) if d in others]
    rows = source.select(*others).unique()
    return rows.join(edge, on=keys, how='inner') if keys else rows.join(edge, how='cross')
