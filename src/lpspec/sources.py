"""Bind runtime data to a validated schema.

The language says what a parameter *is* — its dims, its dtype — and never where
its values come from. This is the other half: what the caller passed (parquet
paths, or any table exposing the Arrow PyCapsule protocol) becomes the tidy
frames the engine reads by name. The shapes themselves are recognised in
:mod:`lpspec.frames`, so no dataframe library beyond the engine's
own is a dependency of either lane.

Not lowering, which turns an AST into a plan and touches no data; this touches
only data and knows nothing about expressions.

The ``piecewise:`` curvature guard lives here because convexity is a property
of the breakpoint *values* — the one check a schema cannot answer — and this is
the module that already knows what shape a caller's table is in.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from lpspec.errors import (
    DataError,
    PiecewiseExpansionError,
    declared_index_also_supplied_message,
    dense_array_message,
    unknown_source_keys_message,
)
from lpspec.frames import as_frame, is_dense_array, labels_frame

if TYPE_CHECKING:
    from lpspec.language.model import Model


def tidy_sources(
    schema: Model,
    data: dict[str, object],
    coords: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Adapt the caller's ``data=``/``coords=`` inputs to engine sources.

    Every in-memory source becomes a tidy :class:`polars.LazyFrame` with columns
    ``(dims…, value)``; parquet paths pass through untouched for the engine to
    scan directly. A dimension index comes from ``data``, from ``coords``, or
    from what the YAML declares — exactly one of them, which
    :func:`check_index_ownership` settles before anything is read.

    Normalising here rather than at the engine is what lets the piecewise
    curvature guard see every in-memory shape alike (:mod:`frames`
    is where the shapes are recognised).

    **Dimensions are resolved first**, because the plain-Python parameter
    shapes :func:`_spread` accepts — a sequence, one number for every
    coordinate — are meaningless without the labels they are spread over.

    Whether a source carries the columns its declaration needs is *not* asked
    here. Binding asks it of every source, path or frame, so a copy of the
    question on this side would answer only for the in-memory half — a second
    wording for one defect, and the narrower one.

    Raises:
        DataError: A key naming nothing the model declares, a declared parameter
            with no data, or one bound to something neither a tidy table nor
            :func:`_spread` can read.
    """
    known = {**schema.parameters, **schema.dimensions}
    if unknown := set(data) - set(known):
        raise DataError(unknown_source_keys_message(unknown, known))
    check_index_ownership(schema, data, coords)

    sources: dict[str, object] = {}
    for dname, ddef in schema.dimensions.items():
        if dname in data:
            src = data[dname]
        elif coords and dname in coords:
            src = coords[dname]
        elif (declared := schema.declared_index(dname)) is not None:
            sources[dname] = pl.LazyFrame(declared)
            continue
        elif ddef.values is not None:
            src = ddef.values
        else:
            continue
        if isinstance(src, (str, Path)):
            sources[dname] = src
            continue
        table = as_frame(src, (dname,))
        sources[dname] = table if table is not None else labels_frame(dname, src, ddef.dtype)

    for pname, pdef in schema.parameters.items():
        if pname not in data:
            raise DataError(f"no data provided for parameter '{pname}'")
        obj = data[pname]
        if isinstance(obj, (str, Path)):
            sources[pname] = obj
            continue
        if is_dense_array(obj):
            raise DataError(dense_array_message(pname))
        table = as_frame(obj, pdef.dims)
        sources[pname] = table if table is not None else _spread(pname, obj, pdef.dims, sources)

    validate_piecewise_data(schema, sources)

    return sources


def _spread(name: str, obj: object, dims: list[str], sources: Mapping[str, object]) -> pl.LazyFrame:
    """A parameter written as plain Python, spread over the dims it declares.

    Three shapes a hand-written model reaches for and no table library
    produces: a ``{label: value}`` map, a sequence in the dimension's own label
    order, and one number standing for every coordinate. Each is dense by
    construction, so each is materialised here — which is the cost of writing a
    parameter out in Python rather than handing over a table, and the reason a
    value that really is constant is better declared ``dims: []``.

    Raises:
        DataError: A shape that does not fit the declared dims, a sequence
            whose length does not match, or a dimension whose labels nothing
            supplies — a positional shape cannot be placed without them.
    """
    if isinstance(obj, Mapping):
        if len(dims) != 1:
            raise DataError(_wrong_rank(name, 'a dict maps one label to one value', dims))
        return pl.LazyFrame({dims[0]: list(obj.keys()), 'value': list(obj.values())})

    if isinstance(obj, bool):
        return _broadcast(name, pl.lit(obj, dtype=pl.Boolean), dims, sources)
    if isinstance(obj, (int, float)):
        return _broadcast(name, pl.lit(float(obj), dtype=pl.Float64), dims, sources)

    if hasattr(obj, '__len__') and not isinstance(obj, (str, bytes)):
        if len(dims) != 1:
            raise DataError(_wrong_rank(name, 'a sequence runs along one dimension', dims))
        labels = _labels(name, dims[0], sources)
        values = list(obj)  # pyrefly: ignore[bad-argument-type]  — narrowed by the __len__ test
        if len(values) != len(labels):
            raise DataError(
                f"parameter '{name}': {len(values)} values against {len(labels)} "
                f"'{dims[0]}' labels. A sequence is positional, so it must have "
                f'one entry per label, in the order the index declares them.'
            )
        return pl.LazyFrame({dims[0]: labels, 'value': values})

    raise DataError(
        f"parameter '{name}': cannot adapt {type(obj).__name__} to a tidy "
        f'table — pass any table polars can read with columns '
        f'{[*dims, "value"]} (polars, pyarrow, pandas), a parquet path, or the '
        f'plain-Python shapes: a dict, a sequence, or one number.'
    )


def _wrong_rank(name: str, said: str, dims: list[str]) -> str:
    """One wording for a plain-Python shape against the dims it cannot cover."""
    return (
        f"parameter '{name}': {said}, and '{name}' is over {dims}. Pass a table "
        f'with columns {[*dims, "value"]} instead.'
    )


def _broadcast(name: str, value: pl.Expr, dims: list[str], sources: Mapping[str, object]) -> pl.LazyFrame:
    """One number over every coordinate of *dims*.

    The cross join is what makes it a table; nothing downstream can broadcast,
    a parameter frame carrying one row per coordinate by contract.
    """
    frame = pl.LazyFrame({'__one__': [0]})
    for dim in dims:
        frame = frame.join(pl.LazyFrame({dim: _labels(name, dim, sources)}), how='cross')
    return frame.drop('__one__').with_columns(value.alias('value'))


def _labels(name: str, dim: str, sources: Mapping[str, object]) -> list[Any]:
    """*dim*'s labels, in index order, for a shape that has none of its own.

    Read back off the dimension source this call already built, which is why
    the dimensions are resolved first.

    Raises:
        DataError: Nothing supplies the labels. A positional shape carries
            none, and no lane reads them off the parameters.
    """
    source = sources.get(dim)
    if source is None:
        raise DataError(
            f"parameter '{name}' is written positionally over '{dim}', so it says what the "
            f'values are but not what they are labelled — and nothing else supplies an index '
            f"for '{dim}'. Declare dimensions.{dim}.values, pass coords={{'{dim}': [...]}}, "
            f"or pass '{name}' as a table carrying its own '{dim}' column."
        )
    frame = pl.scan_parquet(source) if isinstance(source, (str, Path)) else source
    return frame.select(dim).unique(maintain_order=True).collect()[dim].to_list()  # pyrefly: ignore[missing-attribute]


def check_index_ownership(schema: Model, data: Mapping[str, object], coords: Mapping[str, Any] | None) -> None:
    """Refuse a dimension whose index the file declares and the caller also supplies.

    One dimension, one home. A dimension's ``values:`` and the ``values:`` of any
    lookup over it are the file claiming to own which labels exist and how they
    map; a caller passing the same dimension is claiming the same thing from the
    other side. Resolving that by precedence would let the file describe a model
    the caller does not build, so it is refused at bind, before either lane
    reads a table.

    The unit is the dimension rather than the column, because the two halves are
    not independent: labels are derived from the maps where the dimension
    declares none, so taking half a declaration changes what the other half
    means.

    Raises:
        DataError: Naming the dimension, the declaration, and the key that
            collided with it.
    """
    for dim, ddef in schema.dimensions.items():
        declares = _declaring_index(schema, dim, ddef)
        if declares is None:
            continue
        for where, supplied in ((f"sources['{dim}']", data), (f"coords['{dim}']", coords or {})):
            if dim in supplied:
                raise DataError(declared_index_also_supplied_message(dim, declares, where))


def _declaring_index(schema: Model, dim: str, ddef: Any) -> str | None:
    """How the file claims *dim*'s index, spelled as the keys to delete, or ``None``."""
    if ddef.values is not None:
        return f'dimensions.{dim}.values'
    declared = [f'lookups.{n}.values' for n, lk in schema.lookups.items() if lk.over == dim and lk.values is not None]
    return ' and '.join(declared) if declared else None


def validate_piecewise_data(schema: Model, values: Mapping[str, Any] | Any) -> None:
    """Data-time guard for ``method: convex`` blocks (the piecewise rules).

    The hull relaxation is silently wrong for mixed curvature and ill-defined
    when the x-breakpoints are not strictly monotone; both are checkable once
    the breakpoint values are in hand, which the schema never has. *values*
    maps parameter names to whatever its lane holds — :func:`tidy_sources`'
    frames and paths, or the linopy lane's ``xr.Dataset`` — and blocks whose
    parameters are missing or bound to a path are skipped. Called by both
    lanes, which is why it sits beside ``tidy_sources``.

    Only the curvature check needs xarray, for the broadcast over dims, so the
    import waits until a ``method: convex`` block is found. Such a block
    carries exactly two links, which the pair unpack relies on.

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
            xa = _as_dataarray(schema, x_link.values, values)
            ya = _as_dataarray(schema, y_link.values, values)
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
                    f'{ctx}: method: convex requires strictly increasing breakpoints in '
                    f"'{x_link.values}' (got {xs.tolist()})"
                )
            curvature = np.diff(np.diff(ys) / dx)
            tol = 1e-9 * max(1.0, float(np.abs(ys).max()))
            if (curvature > tol).any() and (curvature < -tol).any():
                raise PiecewiseExpansionError(
                    f'{ctx}: method: convex is not exact for the mixed-curvature '
                    f"curve in '{y_link.values}' — the hull relaxation would silently "
                    f'cut corners; use method: adjacency or method: sos2 for the exact form'
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
