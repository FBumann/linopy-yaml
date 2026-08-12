"""Bind runtime data to a validated schema.

The language says what a parameter *is* — its dims, its dtype — and never where
its values come from. This is the other half: what the caller passed (parquet
paths, or any table exposing the Arrow PyCapsule protocol) becomes the tidy
frames the engine reads by name. The shapes themselves are recognised in
:mod:`lpspec.relational.frames`, so no dataframe library beyond the engine's
own is a dependency of either lane.

Not lowering, which turns an AST into a plan and touches no data; this touches
only data and knows nothing about expressions.

The ``piecewise:`` curvature guard lives here because convexity is a property
of the breakpoint *values* — the one check a schema cannot answer — and this is
the module that already knows what shape a caller's table is in.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from lpspec.errors import DataError, PiecewiseExpansionError
from lpspec.relational.frames import as_frame, labels_frame

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lpspec.language.model import Model


def tidy_sources(
    schema: Model,
    data: dict[str, object],
    coords: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Adapt the caller's ``data=``/``coords=`` inputs to executor sources.

    Every in-memory source becomes a tidy :class:`polars.LazyFrame` with columns
    ``(dims…, value)``; parquet paths pass through untouched for the engine to
    scan directly. Dimension indexes come from ``data``, ``coords``, declared YAML
    values, or fall back to the executor's inference from parameter tables.

    Normalising here rather than at the executor is what lets the piecewise
    curvature guard see every in-memory shape alike (:mod:`relational.frames`
    is where the shapes are recognised).

    Whether a source carries the columns its declaration needs is *not* asked
    here. Binding asks it of every source, path or frame, so a copy of the
    question on this side would answer only for the in-memory half — a second
    wording for one defect, and the narrower one.

    Raises:
        DataError: A declared parameter with no data, or one bound to
            something no tidy table can be made of.
    """
    sources: dict[str, object] = {}
    for pname, pdef in schema.parameters.items():
        if pname not in data:
            raise DataError(f"no data provided for parameter '{pname}'")
        obj = data[pname]
        if isinstance(obj, (str, Path)):
            sources[pname] = obj
            continue
        table = as_frame(obj, pdef.dims)
        if table is None:
            raise DataError(
                f"parameter '{pname}': cannot adapt {type(obj).__name__} to a tidy "
                f'table — pass any table polars can read with columns '
                f'{[*pdef.dims, "value"]} (polars, pyarrow, pandas), or a parquet path'
            )
        sources[pname] = table

    for dname, ddef in schema.dimensions.items():
        if dname in data:
            src = data[dname]
        elif coords and dname in coords:
            src = coords[dname]
        elif ddef.values is not None:
            src = ddef.values
        else:
            continue
        if isinstance(src, (str, Path)):
            sources[dname] = src
            continue
        table = as_frame(src, (dname,))
        sources[dname] = table if table is not None else labels_frame(dname, src, ddef.dtype)

    validate_piecewise_data(schema, sources)

    return sources


def validate_piecewise_data(schema: Model, values: Mapping[str, Any] | Any) -> None:
    """Data-time guard for ``convex: true`` blocks (SPEC §3.6).

    The hull relaxation is silently wrong for mixed curvature and ill-defined
    when the x-breakpoints are not strictly monotone; both are checkable once
    the breakpoint values are in hand, which the schema never has. *values*
    maps parameter names to whatever its lane holds — :func:`tidy_sources`'
    frames and paths, or the linopy lane's ``xr.Dataset`` — and blocks whose
    parameters are missing or bound to a path are skipped. Called by both
    lanes, which is why it sits beside ``tidy_sources``.

    Only the curvature check needs xarray, for the broadcast over dims, so the
    import waits until a ``convex: true`` block is found. Such a block carries
    exactly two links, which the pair unpack relies on.

    Raises:
        PiecewiseExpansionError: Breakpoints that are not strictly increasing,
            or a curve of the curvature the hull is not exact for.
    """
    import numpy as np

    for name, pw in schema.piecewise.items():
        if not pw.convex:
            continue
        try:
            import xarray as xr
        except ImportError as exc:
            msg = (
                f"piecewise '{name}': convex curvature validation currently "
                f'requires xarray — pip install "lpspec[linopy]" '
                f'(see issue #27: make this check numpy-only)'
            )
            raise ModuleNotFoundError(msg) from exc
        ctx = f"piecewise '{name}'"
        (x_link, y_link) = pw.links
        try:
            xa = _as_dataarray(schema, x_link[1], values)
            ya = _as_dataarray(schema, y_link[1], values)
        except KeyError:
            continue
        xa, ya = xr.broadcast(xa, ya)
        other = [d for d in xa.dims if d != pw.over]
        stacked_x = xa.transpose(*other, pw.over).values.reshape(-1, xa.sizes[pw.over])
        stacked_y = ya.transpose(*other, pw.over).values.reshape(-1, ya.sizes[pw.over])
        for xs, ys in zip(stacked_x, stacked_y, strict=False):
            dx = np.diff(xs)
            if not (dx > 0).all():
                raise PiecewiseExpansionError(
                    f"{ctx}: convex: true requires strictly increasing breakpoints in '{x_link[1]}' (got {xs.tolist()})"
                )
            curvature = np.diff(np.diff(ys) / dx)
            tol = 1e-9 * max(1.0, float(np.abs(ys).max()))
            if (curvature > tol).any() and (curvature < -tol).any():
                raise PiecewiseExpansionError(
                    f'{ctx}: convex: true is not exact for the mixed-curvature '
                    f"curve in '{y_link[1]}' — the hull relaxation would silently "
                    f'cut corners; drop convex: true to use the exact MILP form'
                )


def _as_dataarray(schema: Model, pname: str, values: Mapping[str, Any] | Any) -> Any:
    """One source as a DataArray indexed by its declared dims.

    Two shapes reach here — the linopy lane's ``xr.Dataset`` entries and the
    relational lane's tidy frames.

    The frame crosses to pandas column by column through numpy: a whole-frame
    conversion would reach for pyarrow, and this check already costs the caller
    xarray without adding a third library. Issue #27 would make it numpy-only
    and retire this function.

    Raises:
        KeyError: If there is nothing to lay out in process (a parquet path,
            or no ``value`` column), which the caller reads as "skip".
    """
    import xarray as xr

    if pname not in values:
        raise KeyError(pname)
    obj = values[pname]
    if isinstance(obj, xr.DataArray):
        return obj
    dims = list(schema.parameters[pname].dims)
    frame = as_frame(obj, tuple(dims))
    if frame is None or not dims or 'value' not in frame.collect_schema().names():
        raise KeyError(pname)
    import pandas as pd

    tidy = frame.select([*dims, 'value']).collect()
    columns = {name: tidy[name].to_numpy() for name in tidy.columns}
    return xr.DataArray.from_series(pd.DataFrame(columns).set_index(dims)['value'])
