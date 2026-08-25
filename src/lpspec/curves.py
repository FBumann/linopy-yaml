"""The data-time guard on a ``piecewise:`` curve.

The language decides a curve's *shape* — which links it has, which method
builds it — and can decide nothing about its numbers, because it has never
seen one. This is the other half: given the tidy sources, is the curve
supplied everywhere the block builds a weight for it, are its breakpoints
strictly increasing, and is its curvature the one the declared method is exact
for.

Called from :func:`~lpspec.sources.tidy_sources`, so both lanes pass through it
by entering the one door, and again by the linopy lane over the
``xr.Dataset`` it built — the same verdict asked of the representation that
lane actually holds.

Separate from ``sources.py`` because the question is different: that module
asks what shape a caller's table is in, this one asks whether the numbers in it
describe a curve the declared method can build.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from math_spec import mask_of

from lpspec.errors import DataError, PiecewiseExpansionError
from lpspec.frames import TidySource, as_frame, scan

if TYPE_CHECKING:
    from collections.abc import Sequence

    from math_spec import Model


def derive_curve_masks(
    schema: Model, sources: dict[str, TidySource], data: Mapping[str, object]
) -> dict[str, TidySource]:
    """Fill in the mask a block asked for by naming one of its own values.

    ``points: bp_x`` says the curve runs as far as ``bp_x`` does. A length is a
    fact of the curve, so this keeps it there rather than asking for a second
    table that repeats it — and the other links are still checked against the
    one named, which is what a second table would have caught.

    The frame is true-only and sparse, since a missing row reads as false in a
    ``where``: that is exactly "no breakpoint here". Called from
    :func:`tidy_sources`, before the loop that asks for a parameter no caller
    can have.

    Returns:
        *sources* with one frame added per block whose ``points:`` names a
        values parameter, under the name the expansion emits.
    """
    for name, pw in schema.piecewise.items():
        mask, nominated = mask_of(name, pw), pw.points
        if mask is None or nominated is None or mask == nominated:
            continue
        rows = _coordinates(data.get(nominated), list(schema.parameters[nominated].dims))
        if rows is not None:
            sources[mask] = rows.with_columns(value=pl.lit(True))
    return sources


def derive_curve_edges(
    schema: Model, sources: dict[str, TidySource], data: Mapping[str, object]
) -> dict[str, TidySource]:
    """Mark where each masked curve starts and ends, for the rows that sit on them.

    ``lp`` states a curve as its segment lines, so it needs two rows holding the
    linked expression inside the curve's *own* range — and under ``points:``
    that range is per curve, where ``position(over) == 0`` and
    ``position(over) == -1`` are the axis'. A ``where:`` takes no operators to
    find it with, so the two edges are marked here instead, from the mask the
    caller supplied: the first and last position it marks, per curve.

    True-only and sparse, since a missing row reads as false in a ``where``.
    Called from :func:`tidy_sources`, which both lanes enter, and only for the
    blocks that need it — an emitted parameter with nothing to fill it would be
    a parameter the caller is asked for and cannot know about.

    Returns:
        *sources* with ``<block>_starts`` and ``<block>_ends`` added per
        ``method: lp`` block that carries a mask.
    """
    for name, pw in schema.piecewise.items():
        mask = mask_of(name, pw)
        if mask is None or pw.points is None or pw.method != 'lp':
            continue
        dims = list(schema.parameters[pw.points].dims)
        table = _coordinates(sources.get(mask, data.get(mask)), dims, keep_value=True)
        if table is None:
            continue
        order = _label_frame(pw.over, sources, table).with_row_index('_ord')
        marked = table.filter(pl.col('value').cast(pl.Boolean)).join(order, on=pw.over, how='inner')
        frame_dims = [d for d in dims if d != pw.over]
        for suffix, edge in (('starts', pl.col('_ord').min()), ('ends', pl.col('_ord').max())):
            at = edge.over(frame_dims) if frame_dims else edge
            sources[f'{name}_{suffix}'] = marked.filter(pl.col('_ord') == at).select([*dims, 'value'])
    return sources


def validate_curve_extent(schema: Model, sources: Mapping[str, TidySource]) -> None:
    """Refuse a ``piecewise:`` curve that is not supplied everywhere it is built.

    A block emits one weight per breakpoint over the whole coordinate product —
    the λ it declares carries no mask — so a values parameter short of a row
    does not build a shorter curve. Both lanes call this, and both reach it with
    what :func:`tidy_sources` returned, or would otherwise read that row as a
    zero coefficient and put a breakpoint at the origin.

    Sequences and single numbers are skipped: each is dense by construction,
    spread over labels the dimension supplies. A parquet path is scanned, since
    the check is worth two columns of I/O against an answer that carries no
    sign of being wrong.

    Args:
        schema: The model as written — ``piecewise:`` blocks name the
            parameters, and the expansion has none.
        sources: What :func:`tidy_sources` returned — parameter and dimension
            names to a frame or a path.

    Raises:
        DataError: A link's values parameter with a hole where the block builds
            a weight, or a ``points:`` mask that is not a prefix of the
            breakpoint axis.
    """
    for block, pw in schema.piecewise.items():
        mask = _prefix_mask(schema, block, pw, sources) if pw.points else None
        for link in pw.links:
            dims = list(schema.parameters[link.values].dims)
            present = _coordinates(sources.get(link.values), dims)
            if present is None:
                continue
            extents = {d: _label_frame(d, sources, present) for d in dims}
            if mask is None:
                expected = 1
                for labels in extents.values():
                    expected *= labels.select(pl.len()).collect().item()
                found = present.select(pl.len()).collect().item()
                if found < expected:
                    grid = _grid(extents, dims, present)
                    raise DataError(
                        _curve_with_a_hole_message(
                            block, link.values, _a_hole(grid, present, dims), expected, found, pw.points
                        )
                    )
                continue
            required = _required_under_mask(mask, extents, dims, present)
            counts = pl.collect_all([required.select(pl.len()), present.select(pl.len())])
            if required.join(present, on=dims, how='anti').head(1).collect().height:
                raise DataError(
                    _curve_with_a_hole_message(
                        block,
                        link.values,
                        _a_hole(required, present, dims),
                        counts[0].item(),
                        counts[1].item(),
                        pw.points,
                    )
                )


def _curve_with_a_hole_message(block: str, name: str, shown: str, expected: int, found: int, points: str | None) -> str:
    """A piecewise curve supplied at some of its coordinates — one wording, both lanes.

    A missing row is the one shape whose two readings are both wrong here. The
    absence rules read it as a zero coefficient, which puts a breakpoint at the
    origin that the file never declared; read as a shorter curve it would need
    the weights to shrink with it, which is what ``points:`` says and a bare
    table cannot.

    Which is why the way out depends on whether the block already has a mask.
    Without one the reader wants to hear that a curve may declare its length;
    with one they have said it, and the disagreement is the news — the mask
    claims a breakpoint the values do not carry.
    """
    remedy = (
        f"  Shorten it    '{points}' claims this breakpoint, so either it is one row too long "
        f'or the value is missing\n'
        f'  Or supply it  a value everywhere the mask says the curve runs'
        if points
        else (
            "  Say how far   points: a mask over the curve, true up to each one's last "
            'breakpoint\n'
            '  Or supply it  a value at every coordinate of the axis\n'
            '  Or write it   where the *arity* is data, the λ formulation states it '
            'directly (issue #1101)'
        )
    )
    return (
        f"piecewise '{block}': parameter '{name}' has no value at {shown} — {found} of the "
        f'{expected} coordinates it needs. Every breakpoint the block builds gets a weight, so '
        f'a missing row is not a shorter curve: read as a zero coefficient it is a breakpoint '
        f'at the origin, and the answer mixes onto it.\n{remedy}'
    )


def _prefix_mask(schema: Model, block: str, pw: Any, sources: Mapping[str, TidySource]) -> pl.LazyFrame | None:
    """The coordinates ``points:`` marks present, checked to be a prefix per curve.

    The emitted rows lean on the shape twice — the chord row on "my predecessor
    is present", the upper domain row on "present here and absent next" — so a
    mask with a gap in it, or one naming no breakpoint at all, builds a
    formulation that says something other than the curve it came from.

    Returns ``None`` where the mask is bound to something this cannot read,
    which binding refuses on its own terms.
    """
    mask = mask_of(block, pw)
    if pw.points is None or mask is None:
        return None
    dims = list(schema.parameters[pw.points].dims)
    table = _coordinates(sources.get(mask), dims, keep_value=True)
    if table is None:
        return None
    order = _label_frame(pw.over, sources, table).with_row_index('_ord')
    frame_dims = [d for d in dims if d != pw.over]
    marked = table.filter(pl.col('value').cast(pl.Boolean)).join(order, on=pw.over, how='inner').select([*dims, '_ord'])
    run_length = (
        pl.len().alias('marked'),
        (pl.col('_ord').max() - pl.col('_ord').min() + 1).alias('span'),
    )
    summary = marked.group_by(frame_dims).agg(*run_length) if frame_dims else marked.select(*run_length)
    broken = summary.filter(pl.col('span') != pl.col('marked')).head(1).collect()
    if not broken.height and frame_dims:
        extents = {d: _label_frame(d, sources, table) for d in frame_dims}
        broken = _grid(extents, frame_dims, table).join(marked, on=frame_dims, how='anti').head(1).collect()
    if broken.height:
        shown = ', '.join(f'{d}={broken.row(0, named=True)[d]!r}' for d in frame_dims)
        raise DataError(_curve_mask_is_not_contiguous_message(block, mask, pw.over, shown))
    return marked.select(dims)


def _curve_mask_is_not_contiguous_message(block: str, points: str, over: str, shown: str) -> str:
    """A curve whose breakpoints are not consecutive — one wording, both lanes.

    Where a curve *starts* does not matter: every row that reads the mask asks
    for a predecessor or for an end, and all of those are the curve's own. A
    gap matters twice over — a chord would join across it, and a domain row
    would sit inside the curve rather than at its edge. Both build, and neither
    says what the file does.
    """
    at = f' at {shown}' if shown else ''
    return (
        f"piecewise '{block}': the breakpoints '{points}' marks along '{over}'{at} are not "
        f'consecutive — there is a gap in them, or nothing is marked at all. A curve may sit '
        f'anywhere along the axis, on breakpoints that follow one another.\n'
        f'  Close the gap   a curve is its points and the ones between them\n'
        f'  A curve of none is not a curve: leave the block to the members that have one'
    )


def _required_under_mask(
    mask: pl.LazyFrame, extents: Mapping[str, pl.LazyFrame], dims: Sequence[str], present: pl.LazyFrame
) -> pl.LazyFrame:
    """The coordinates a link's values must carry: the mask, widened to its dims.

    A mask need not carry every dim the values do — one shared by every
    generator is a curve length along ``over`` alone — so the dims it does not
    name are crossed back in at their full extent.
    """
    dtypes = present.collect_schema()
    shared = [d for d in dims if d in mask.collect_schema().names()]
    required = mask.select(shared).unique()
    for d in dims:
        if d not in shared:
            required = required.join(extents[d].select(pl.col(d).cast(dtypes[d])), how='cross')
    return required.select(dims)


def _coordinates(source: object, dims: Sequence[str], keep_value: bool = False) -> pl.LazyFrame | None:
    """The coordinates *source* carries, or ``None`` where it carries all of them.

    ``None`` covers both "dense by construction" and "not readable here" — a
    source binding refuses is refused there, with the message that knows what
    the declaration wanted. *keep_value* keeps the value column too, which the
    ``points:`` mask is read from rather than merely counted.
    """
    if isinstance(source, (str, Path)):
        table = pl.scan_parquet(source)
    elif isinstance(source, Mapping) and len(dims) == 1:
        keys = pl.LazyFrame({dims[0]: list(source.keys())})
        return keys.with_columns(pl.Series('value', list(source.values())).implode().explode()) if keep_value else keys
    else:
        table = as_frame(source, tuple(dims))
    if table is None or not set(dims) <= set(table.collect_schema().names()):
        return None
    columns = [*dims, 'value'] if keep_value else list(dims)
    if keep_value and 'value' not in table.collect_schema().names():
        return None
    return table.select(columns)


def _label_frame(dim: str, sources: Mapping[str, TidySource], present: pl.LazyFrame) -> pl.LazyFrame:
    """*dim*'s labels as one column, from wherever the model's index comes from.

    *sources* is what :func:`tidy_sources` returned, so a dimension with an
    index — declared by the file or passed by the caller — is already a frame
    here, and which of the two it was stopped mattering at that door.

    Falls back to the labels the curve itself carries, which is what a
    dimension with no index of its own is bound against: there the curve cannot
    be short of a breakpoint nothing else declares.

    **Order is kept.** A count does not care, but the ``points:`` prefix check
    numbers these labels to say where a curve stops, and an unordered unique
    hands it a permutation — which fails, or does not, run to run.
    """
    source = sources.get(dim)
    if source is None:
        return present.select(dim).unique(maintain_order=True)
    return scan(source).select(dim).unique(maintain_order=True)


def _grid(extents: Mapping[str, pl.LazyFrame], dims: Sequence[str], present: pl.LazyFrame) -> pl.LazyFrame:
    """Every coordinate the block builds a weight for, where nothing masks them."""
    dtypes = present.collect_schema()
    grid = extents[dims[0]].select(pl.col(dims[0]).cast(dtypes[dims[0]]))
    for dim in dims[1:]:
        grid = grid.join(extents[dim].select(pl.col(dim).cast(dtypes[dim])), how='cross')
    return grid


def _a_hole(required: pl.LazyFrame, present: pl.LazyFrame, dims: Sequence[str]) -> str:
    """One coordinate the curve does not carry, written as the reader would look for it.

    Built only on the way to raising. There is always such a coordinate: the
    caller reached here by carrying fewer than *required* holds.
    """
    row = required.join(present, on=list(dims), how='anti').head(1).collect().row(0, named=True)
    return '(' + ', '.join(f'{d}={row[d]!r}' for d in dims) + ')'


def validate_piecewise_data(schema: Model, values: Mapping[str, Any] | Any) -> None:
    """Data-time guard for the methods a curve's shape can make wrong.

    ``convex``'s hull relaxation is wrong for mixed curvature and ``lp``'s
    segment lines for the bend opposite their bounded link — see
    :attr:`PiecewiseBlock.curvature_required` — and both are ill-defined when
    the x-breakpoints are not strictly monotone. All of it is checkable once
    the breakpoint values are in hand, which the schema never has. *values*
    maps parameter names to whatever its lane holds — :func:`tidy_sources`'
    frames and paths, or the linopy lane's ``xr.Dataset`` — and a path is
    scanned for its two columns rather than skipped, since a verdict that
    turned on how the numbers were handed over is no verdict at all. A block
    is skipped where its parameters are absent, and where nothing supplies its
    breakpoint dimension's index: with no index there is no order, so there is
    no question here to answer, and binding's own message names what is
    missing rather than this one calling a curve backwards.

    **The breakpoints are walked in the ``over`` dimension's own index order**,
    which is the order the model is built in — what ``shift`` walks and what
    ``position(bp)`` counts along. A values table is a function of its
    coordinates and carries no order of its own, so it is laid out on that index first
    (:func:`_breakpoint_order`); reading it in the row order it happened to
    arrive in judged an order the model never builds, in both directions.

    **The bend is measured against a slope**, because that is what it is: a
    difference of two slopes, in y per x. Judged against a share of ``y`` it
    tracked the unit x happens to be measured in — stretch a curve along x and
    a real bend passed, shrink it and a straight line was refused. The
    tolerance is not zero because an exactly collinear curve need not difference
    to exactly zero: ``[0, 0.1, 0.3]`` over ``[0, 1, 3]`` gives ``-1.4e-17``,
    negative, which is the sign that refuses a convex curve.

    Only the curvature check needs xarray, for the broadcast over dims, so the
    import waits until a block that needs it is found.

    **A curve is its marked breakpoints**, which is what ``points:`` names and
    what every check here reads. The two spellings put the length in different
    places — naming a values parameter leaves the unrun breakpoints out of the
    table, a boolean mask marks them in a table that may be dense — so the rows
    that carry a value are the curve under the first and a superset of it under
    the second. Judged that way, values the curve does not run over are
    differenced as if it did, and a curve masked down to one point counts as
    however many rows its table holds.

    **A count is per curve, not per dimension.** ``method: lp`` needs a segment
    to state a line for, and how many points a curve runs over is its own
    property: a block spans one breakpoint dimension but as many curves as its
    frame has rows, and under ``points:`` the two no longer agree. Asking the
    dimension clears a curve that has no segment — its chord row excluded as
    its own first point, its domain rows pinning only the pinned link, and the
    bounded one left on its bound.

    Raises:
        PiecewiseExpansionError: Breakpoints that are not strictly increasing,
            a ``method: lp`` curve with no segment, or a curve of the curvature
            the method is not exact for.
    """
    import numpy as np

    for name, pw in schema.piecewise.items():
        required = pw.curvature_required
        if required is None:
            continue
        try:
            import xarray as xr
        except ImportError as exc:
            msg = (
                f"piecewise '{name}': method: {pw.method} needs its curve's curvature "
                f'checked, which currently requires xarray — pip install "lpspec[linopy]" '
                f'(see issue #27: make this check numpy-only)'
            )
            raise ModuleNotFoundError(msg) from exc
        ctx = f"piecewise '{name}'"
        x_link, y_link = pw.curve
        if pw.over not in values:
            continue
        try:
            arrays = [_as_dataarray(schema, x_link.values, values), _as_dataarray(schema, y_link.values, values)]
            if pw.points is not None:
                mask = mask_of(name, pw)
                assert mask is not None, 'a block that declares points: has a mask to read it from'
                arrays.append(_as_dataarray(schema, mask, values, schema.parameters[pw.points].dims))
        except KeyError:
            continue
        arrays = list(xr.broadcast(*arrays))
        if (order := _breakpoint_order(pw.over, values)) is not None:
            arrays = [array.reindex({pw.over: order}) for array in arrays]
        other = [d for d in arrays[0].dims if d != pw.over]
        width = arrays[0].sizes[pw.over]
        stacked = [array.transpose(*other, pw.over).values.reshape(-1, width) for array in arrays]
        for curve in zip(*stacked, strict=False):
            xs_all, ys_all = curve[0], curve[1]
            live = ~(np.isnan(xs_all) | np.isnan(ys_all))
            if len(curve) > 2:
                live &= np.equal(curve[2], True)
            xs, ys = xs_all[live], ys_all[live]
            if pw.method == 'lp' and xs.size < 2:
                raise PiecewiseExpansionError(
                    f'{ctx}: method: lp needs at least two breakpoints and this curve carries '
                    f'{xs.size} — the method *is* its segment lines, so a curve with no segment '
                    f'states nothing and leaves the bounded link on its own bound. Use method: '
                    f'adjacency, sos2 or convex, which pin it to the points it does have.'
                )
            dx = np.diff(xs)
            if not (dx > 0).all():
                raise PiecewiseExpansionError(
                    f'{ctx}: method: {pw.method} requires strictly increasing breakpoints in '
                    f"'{x_link.values}' (got {xs.tolist()})"
                )
            slopes = np.diff(ys) / dx
            curvature, tol = np.diff(slopes), 1e-9 * float(np.abs(slopes).max(initial=0.0))
            rises, falls = bool((curvature > tol).any()), bool((curvature < -tol).any())
            wrong_bend = (rises and falls) if required == 'either' else (falls if required == 'convex' else rises)
            if wrong_bend:
                shape = 'mixed-curvature' if required == 'either' else f'not {required}'
                raise PiecewiseExpansionError(
                    f'{ctx}: method: {pw.method} is exact only for a '
                    f'{"single-curvature" if required == "either" else required} curve, and '
                    f"'{y_link.values}' is {shape} ({ys.tolist()}) — the relaxation would cut "
                    f'the curve with no sign of it in the answer. Use method: adjacency or '
                    f'method: sos2 for the exact form.'
                )


def _breakpoint_order(over: str, values: Mapping[str, Any] | Any) -> list[Any] | None:
    """*over*'s labels in the order the model builds them, or ``None``.

    ``None`` where the caller holds no index this can read — the linopy lane's
    ``xr.Dataset``, whose arrays its loader has already laid out on that index,
    so there is nothing left to reorder.

    Deduplicated by *first appearance* rather than sorted, because that is the
    ordinal the engine assigns (``relational/engines/polars/binding.py``) and a
    guard reading a second order would answer for a model nobody builds.
    """
    source = values.get(over)
    if isinstance(source, (str, Path)):
        source = pl.scan_parquet(source)
    if not isinstance(source, (pl.LazyFrame, pl.DataFrame)):
        return None
    labels = source.lazy().select(over).unique(maintain_order=True).collect()
    return labels[over].to_list()


def _as_dataarray(schema: Model, pname: str, values: Mapping[str, Any] | Any, dims: Sequence[str] | None = None) -> Any:
    """One source as a DataArray indexed by its declared dims.

    Three shapes reach here — the linopy lane's ``xr.Dataset`` entries, the
    relational lane's tidy frames, and the parquet paths that lane passes
    through untouched. The path is scanned for the columns the check reads,
    which is what keeps the guard's answer a property of the numbers rather
    than of how they arrived.

    The frame crosses to pandas column by column through numpy: a whole-frame
    conversion would reach for pyarrow, and this check already costs the caller
    xarray without adding a third library. Issue #27 would make it numpy-only
    and retire this function.

    Raises:
        KeyError: If there is nothing to lay out — an absent parameter, or one
            whose table carries no ``value`` column — which the caller reads as
            "skip".
    """
    import xarray as xr

    if pname not in values:
        raise KeyError(pname)
    obj = values[pname]
    if isinstance(obj, xr.DataArray):
        return obj
    dims = list(schema.parameters[pname].dims if dims is None else dims)
    frame = pl.scan_parquet(obj) if isinstance(obj, (str, Path)) else as_frame(obj, tuple(dims))
    if frame is None or not dims or 'value' not in frame.collect_schema().names():
        raise KeyError(pname)
    import pandas as pd

    tidy = frame.select([*dims, 'value']).collect()
    columns = {name: tidy[name].to_numpy() for name in tidy.columns}
    return xr.DataArray.from_series(pd.DataFrame(columns).set_index(dims)['value'])
