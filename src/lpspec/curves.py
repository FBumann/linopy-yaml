"""The data-time guard on a ``piecewise:`` curve.

The language decides a curve's *shape* — which links it has, which method
builds it — and can decide nothing about its numbers, because it has never
seen one. This is the other half: given the tidy sources, is the curve
supplied everywhere the block builds a weight for it, and does it hold what
its method rests on. What those conditions *are* is the language's answer —
a block carries them as :data:`~math_spec.program.Check` values, each naming
its own subjects, and :func:`~math_spec.program.check_message` words the
refusal — so what is decided here is only whether the numbers hold them, and
this module appends what it saw.

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
from typing import TYPE_CHECKING, Any, TypeVar

import polars as pl
from math_spec.program import AtLeastTwo, Contiguous, Curved, FirstOf, Increasing, LastOf, MaskOf, check_message

from lpspec.errors import DataError, PiecewiseExpansionError
from lpspec.frames import as_frame, to_pandas

if TYPE_CHECKING:
    from collections.abc import Sequence

    from math_spec.program import Check, PiecewiseDeclaration, Program

_C = TypeVar('_C', bound='Check')


def derive_curve_sources(
    program: Program, sources: dict[str, pl.LazyFrame], data: Mapping[str, object]
) -> dict[str, pl.LazyFrame]:
    """Fill every parameter a ``piecewise:`` expansion emitted, which no caller can have.

    Three kinds, each named by the parameter's own
    :attr:`~math_spec.program.ParameterDeclaration.derivation`, so which
    parameters these are and how each is filled is read off the program rather
    than inferred from a block:

    :class:`~math_spec.program.MaskOf` is the mask a ``points: bp_x`` asked for
    by naming one of the block's own values — the curve runs as far as ``bp_x``
    does, so its length stays where it already is rather than being asked for a
    second time, and the other links are still checked against the one named.
    :class:`~math_spec.program.FirstOf` and :class:`~math_spec.program.LastOf`
    mark where each masked curve begins and ends: ``lp`` states a curve as its
    segment lines, so it needs two rows holding the linked expression inside
    the curve's *own* range, and a ``where:`` has no operator to find them
    with.

    Every frame is true-only and sparse, since a missing row reads as false in
    a ``where``. **The masks are filled first**, because the two edge flags are
    read off one. Called from :func:`tidy_sources`, before the loop that asks
    for data the caller does have.

    Returns:
        *sources* with a frame under each emitted parameter whose own source
        can be read, and nothing under the rest — attaching refuses those in the
        message that knows what the declaration wanted.
    """
    for name, declared in program.parameters.items():
        if isinstance(mask := declared.derivation, MaskOf):
            rows = _coordinates(data.get(mask.values), list(program.parameters[mask.values].dims))
            if rows is not None:
                sources[name] = rows.with_columns(value=pl.lit(True))

    for name, declared in program.parameters.items():
        match declared.derivation:
            case FirstOf(block, mask):
                edge = pl.col('_ord').min()
            case LastOf(block, mask):
                edge = pl.col('_ord').max()
            case _:
                continue
        over = program.piecewise[block].over
        dims = list(program.parameters[mask].dims)
        table = _coordinates(sources.get(mask, data.get(mask)), dims, keep_value=True)
        if table is None:
            continue
        order = _label_frame(over, sources, table).with_row_index('_ord')
        marked = table.filter(pl.col('value').cast(pl.Boolean)).join(order, on=over, how='inner')
        frame_dims = [d for d in dims if d != over]
        at = edge.over(frame_dims) if frame_dims else edge
        sources[name] = marked.filter(pl.col('_ord') == at).select([*dims, 'value'])
    return sources


def _one(checks: Sequence[Check], kind: type[_C]) -> _C | None:
    """The block's check of *kind*, or ``None`` — a block carries at most one of each."""
    return next((check for check in checks if isinstance(check, kind)), None)


def validate_curve_extent(program: Program, sources: Mapping[str, pl.LazyFrame]) -> None:
    """Refuse a ``piecewise:`` curve that is not supplied everywhere it is built.

    A block emits one weight per breakpoint over the whole coordinate product —
    the λ it declares carries no mask — so a values parameter short of a row
    does not build a shorter curve. Both lanes call this, and both reach it with
    what :func:`tidy_sources` returned, or would otherwise read that row as a
    zero coefficient and put a breakpoint at the origin.

    Args:
        program: The lowered spec — each block's breakpoints, and the
            :class:`~math_spec.program.Contiguous` check naming its mask.
        sources: What :func:`tidy_sources` returned — parameter and dimension
            names to a frame.

    Raises:
        DataError: A link's values parameter with a hole where the block builds
            a weight, or a ``points:`` mask that is not a prefix of the
            breakpoint axis.
    """
    for block, decl in program.piecewise.items():
        run = _one(decl.checks, Contiguous)
        mask = _prefix_mask(block, decl, run, program, sources)
        points = (run.values or run.mask) if run else None
        for values in decl.breakpoints:
            dims = list(program.parameters[values].dims)
            present = _coordinates(sources.get(values), dims)
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
                        _curve_with_a_hole_message(block, values, _a_hole(grid, present, dims), expected, found, points)
                    )
                continue
            required = _required_under_mask(mask, extents, dims, present)
            if required.join(present, on=dims, how='anti').head(1).collect().height:
                counts = pl.collect_all([required.select(pl.len()), present.select(pl.len())])
                raise DataError(
                    _curve_with_a_hole_message(
                        block,
                        values,
                        _a_hole(required, present, dims),
                        counts[0].item(),
                        counts[1].item(),
                        points,
                    )
                )


def _curve_with_a_hole_message(block: str, name: str, shown: str, expected: int, found: int, points: str | None) -> str:
    """A piecewise curve supplied at some of its coordinates; the way out depends on whether the block has a mask."""
    remedy = (
        f"  Shorten it    '{points}' claims this breakpoint, so either it is one row too long "
        f'or the value is missing\n'
        f'  Or supply it  a value everywhere the mask says the curve runs'
        if points
        else (
            "  Say how far   points: a mask over the curve, true up to each one's last "
            'breakpoint\n'
            '  Or supply it  a value at every coordinate of the axis\n'
            '  Or write it   where the *arity* is data, the λ formulation states it directly'
        )
    )
    return (
        f"piecewise '{block}': parameter '{name}' has no value at {shown} — {found} of the "
        f'{expected} coordinates it needs. Every breakpoint the block builds gets a weight, so '
        f'a missing row is not a shorter curve: read as a zero coefficient it is a breakpoint '
        f'at the origin, and the answer mixes onto it.\n{remedy}'
    )


def _prefix_mask(
    block: str,
    pw: PiecewiseDeclaration,
    run: Contiguous | None,
    program: Program,
    sources: Mapping[str, pl.LazyFrame],
) -> pl.LazyFrame | None:
    """The coordinates ``points:`` marks present, checked to be a prefix per curve.

    The emitted rows lean on the shape twice — the chord row on "my predecessor
    is present", the upper domain row on "present here and absent next" — so a
    mask with a gap in it, or one naming no breakpoint at all, builds a
    formulation that says something other than the curve it came from. That is
    the block's :class:`~math_spec.program.Contiguous` check, carried where the
    language decided it rather than inferred from a ``points:`` here.

    Returns ``None`` for a block carrying no such check, and where the mask is
    attached to something this cannot read, which attaching refuses on its own
    terms.
    """
    if run is None:
        return None
    mask = run.mask
    dims = list(program.parameters[mask].dims)
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
        message = check_message(block, pw, run)
        raise DataError(f'{message}\n  Not so at {shown}' if shown else message)
    return marked.select(dims)


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
    source attaching refuses is refused there, with the message that knows what
    the declaration wanted. *keep_value* keeps the value column too, which the
    ``points:`` mask is read from rather than merely counted.
    """
    if isinstance(source, Mapping) and len(dims) == 1:
        keys = pl.LazyFrame({dims[0]: list(source.keys())})
        return keys.with_columns(pl.Series('value', list(source.values())).implode().explode()) if keep_value else keys
    table = as_frame(source, tuple(dims))
    if table is None or not set(dims) <= set(table.collect_schema().names()):
        return None
    columns = [*dims, 'value'] if keep_value else list(dims)
    if keep_value and 'value' not in table.collect_schema().names():
        return None
    return table.select(columns)


def _label_frame(dim: str, sources: Mapping[str, pl.LazyFrame], present: pl.LazyFrame) -> pl.LazyFrame:
    """*dim*'s labels as one column, from wherever the model's index comes from.

    *sources* is what :func:`tidy_sources` returned, so a dimension with an
    index — declared by the file or passed by the caller — is already a frame
    here, and which of the two it was stopped mattering at that door.

    Falls back to the labels the curve itself carries, which is what a
    dimension with no index of its own is attached against: there the curve cannot
    be short of a breakpoint nothing else declares.

    **Order is kept.** A count does not care, but the ``points:`` prefix check
    numbers these labels to say where a curve stops, and an unordered unique
    hands it a permutation — which fails, or does not, run to run.
    """
    source = sources.get(dim)
    if source is None:
        return present.select(dim).unique(maintain_order=True)
    return source.select(dim).unique(maintain_order=True)


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


def validate_piecewise_data(program: Program, values: Mapping[Any, Any]) -> None:
    """Data-time guard for the checks a ``piecewise:`` block makes about its own numbers.

    :class:`~math_spec.program.Curved` is the shape the method is exact for,
    :class:`~math_spec.program.Increasing` the monotone x both are ill-defined
    without, and :class:`~math_spec.program.AtLeastTwo` the segment ``lp`` has
    to state a line for. A block with no ``Curved`` check is skipped, as is one
    whose parameters or breakpoint index *values* does not hold. *values* maps
    parameter and dimension names to what a lane holds: :func:`tidy_sources`'
    frames, or the linopy lane's ``xr.Dataset``.

    A curve is its marked breakpoints, walked in the ``over`` dimension's own
    index order — the order the model is built in. The bend is measured as a
    difference of slopes against ``1e-9`` of the largest slope, because an
    exactly collinear curve need not difference to exactly zero.

    Raises:
        PiecewiseExpansionError: Breakpoints that are not strictly increasing,
            a ``method: lp`` curve with no segment, or a curve of the curvature
            the method is not exact for.
        ModuleNotFoundError: A block needing the curvature check on an install
            without xarray.
    """
    import numpy as np

    for name, decl in program.piecewise.items():
        curved = _one(decl.checks, Curved)
        if curved is None:
            continue
        try:
            import xarray as xr
        except ImportError as exc:
            msg = (
                f"piecewise '{name}': method: {decl.method} needs its curve's curvature "
                f'checked, which requires xarray — pip install "lpspec[linopy]"'
            )
            raise ModuleNotFoundError(msg) from exc
        increasing, segment = _one(decl.checks, Increasing), _one(decl.checks, AtLeastTwo)
        over = curved.over
        if over not in values:
            continue
        run = _one(decl.checks, Contiguous)
        mask = run.mask if run else None
        try:
            arrays = [_as_dataarray(program, curved.x, values), _as_dataarray(program, curved.y, values)]
            if mask is not None:
                arrays.append(_as_dataarray(program, mask, values))
        except KeyError:
            continue
        arrays = list(xr.broadcast(*arrays))
        if (order := _breakpoint_order(over, values)) is not None:
            arrays = [array.reindex({over: order}) for array in arrays]
        other = [d for d in arrays[0].dims if d != over]
        width = arrays[0].sizes[over]
        stacked = [array.transpose(*other, over).values.reshape(-1, width) for array in arrays]
        for curve in zip(*stacked, strict=False):
            xs_all, ys_all = curve[0], curve[1]
            live = ~(np.isnan(xs_all) | np.isnan(ys_all))
            if len(curve) > 2:
                live &= np.equal(curve[2], True)
            xs, ys = xs_all[live], ys_all[live]
            if segment is not None and xs.size < 2:
                raise PiecewiseExpansionError(f'{check_message(name, decl, segment)}\n  This curve carries {xs.size}')
            dx = np.diff(xs)
            if increasing is not None and not (dx > 0).all():
                raise PiecewiseExpansionError(f'{check_message(name, decl, increasing)} (got {xs.tolist()})')
            slopes = np.diff(ys) / dx
            bend, tol = np.diff(slopes), 1e-9 * float(np.abs(slopes).max(initial=0.0))
            rises, falls = bool((bend > tol).any()), bool((bend < -tol).any())
            required = curved.curvature
            wrong_bend = (rises and falls) if required == 'either' else (falls if required == 'convex' else rises)
            if wrong_bend:
                raise PiecewiseExpansionError(f'{check_message(name, decl, curved)} (got {ys.tolist()})')


def _breakpoint_order(over: str, values: Mapping[str, Any]) -> list[Any] | None:
    """*over*'s labels in the order the model builds them, or ``None``.

    ``None`` where the caller holds no index this can read — the linopy lane's
    ``xr.Dataset``, whose arrays its loader has already laid out on that index.

    Deduplicated by *first appearance* rather than sorted, because that is the
    ordinal the engine assigns (``relational/engines/polars/attaching.py``).
    """
    source = values.get(over)
    if not isinstance(source, (pl.LazyFrame, pl.DataFrame)):
        return None
    labels = source.lazy().select(over).unique(maintain_order=True).collect()
    return labels[over].to_list()


def _as_dataarray(program: Program, pname: str, values: Mapping[str, Any]) -> Any:
    """One source as a DataArray indexed by its declared dims.

    The frame crosses to pandas through :func:`lpspec.frames.to_pandas`: a
    whole-frame conversion would reach for pyarrow, and this check already
    costs the caller xarray without adding a third library.

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
    dims = list(program.parameters[pname].dims)
    frame = as_frame(obj, tuple(dims))
    if frame is None or not dims or 'value' not in frame.collect_schema().names():
        raise KeyError(pname)
    tidy = frame.select([*dims, 'value']).collect()
    return xr.DataArray.from_series(to_pandas(tidy).set_index(dims)['value'])
