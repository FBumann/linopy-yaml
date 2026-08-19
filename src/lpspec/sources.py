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

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import polars as pl

from lpspec.errors import (
    DataError,
    PiecewiseExpansionError,
    curve_with_a_hole_message,
    declared_index_also_supplied_message,
    declared_map_needs_labels_message,
    dense_array_message,
    index_without_its_label_column_message,
    map_keys_are_not_labels_message,
    multi_indexed_series_message,
    unknown_source_keys_message,
)
from lpspec.frames import as_frame, is_dense_array, is_multi_indexed, labels_frame

if TYPE_CHECKING:
    from lpspec.language.model import Model


#: What a source is once :func:`tidy_sources` has read it: a tidy
#: ``(dims…, value)`` frame, or the parquet path the engine scans for itself. A
#: dimension index is always the frame — :func:`polars_index` reads every shape
#: one supports, a path included, because a declared map may have to be joined
#: onto it.
TidySource: TypeAlias = pl.LazyFrame | str | Path


def tidy_sources(schema: Model, data: Mapping[str, object]) -> dict[str, TidySource]:
    """Adapt the caller's ``sources`` mapping to engine sources.

    Every in-memory source becomes a tidy :class:`polars.LazyFrame` with columns
    ``(dims…, value)``; parquet paths pass through untouched for the engine to
    scan directly. The indices come from :func:`dimension_sources`, which both
    lanes read.

    Normalising here rather than at the engine is what lets the piecewise
    curvature guard see every in-memory shape alike (:mod:`frames`
    is where the shapes are recognised).

    **Dimensions are resolved first**, because the plain-Python parameter
    shapes :func:`_spread` accepts — a sequence, one number for every
    coordinate — are meaningless without the labels they are spread over.

    Whether a *parameter* source carries the columns its declaration needs is
    *not* asked here. Binding asks it of every source, path or frame, so a copy
    of the question on this side would answer only for the in-memory half — a
    second wording for one defect, and the narrower one.

    Raises:
        DataError: A key naming nothing the model declares, a declared parameter
            with no data, or one bound to something neither a tidy table nor
            :func:`_spread` can read.
    """
    known = {**schema.parameters, **schema.dimensions}
    if unknown := set(data) - set(known):
        raise DataError(unknown_source_keys_message(unknown, known))

    sources: dict[str, TidySource] = {
        dname: polars_index(source, dname, schema.dimensions[dname].dtype, schema.declared_maps(dname))
        for dname, source in dimension_sources(schema, data).items()
    }

    for pname, pdef in schema.parameters.items():
        if pname not in data:
            raise DataError(f"no data provided for parameter '{pname}'")
        obj = data[pname]
        if isinstance(obj, (str, Path)):
            sources[pname] = obj
            continue
        if is_dense_array(obj):
            raise DataError(dense_array_message(pname))
        if is_multi_indexed(obj):
            raise DataError(multi_indexed_series_message(pname, pdef.dims))
        table = as_frame(obj, pdef.dims)
        sources[pname] = table if table is not None else _spread(pname, obj, pdef.dims, sources)

    validate_curve_extent(schema, sources)
    validate_piecewise_data(schema, sources)

    return sources


def dimension_sources(schema: Model, data: Mapping[str, object]) -> dict[str, object]:
    """Which object supplies each dimension's index — the rule, in one place.

    A key in *data*, or the ``values:`` the YAML declares, never both
    (:func:`check_index_ownership`). What comes back is what the caller or the
    file passed and nothing more — a path, a table, a sequence of labels —
    because each lane reads it into its own frame library
    (:func:`polars_index` here, ``linopy/loader.py`` there), and a shared frame
    would make one of them convert twice over.

    A dimension nothing supplies an index for is simply absent: which
    dimensions *need* one is the caller's question — the engine asks it of the
    dims its program spans, the eager lane of every dim the file declares.

    A declared map does not travel with the source. It is
    :meth:`~lpspec.language.model.Model.declared_maps`, which both readers ask
    for themselves, so a map the file declares is read against the labels the
    same way whether those came from the file or from the caller.

    Raises:
        DataError: A dimension whose index two authors claim, or whose maps the
            file declares and whose labels nothing does.
    """
    check_index_ownership(schema, data)

    sources: dict[str, object] = {}
    for dname, ddef in schema.dimensions.items():
        if dname in data:
            sources[dname] = data[dname]
        elif ddef.values is not None:
            sources[dname] = ddef.values
        elif maps_only := schema.declared_maps(dname):
            raise DataError(declared_map_needs_labels_message(dname, maps_only))

    return sources


def polars_index(source: object, dim: str, dtype: str, maps: Mapping[str, Mapping[Any, Any]]) -> pl.LazyFrame:
    """One dimension's index, read once — the only read of one in the package.

    :func:`dimension_sources` says which object supplies an index; this turns
    that object into labels. A lane wanting another library **converts this
    frame** rather than reading the source a second time, which is what makes a
    caller's choice of dataframe invisible past this line: two readers gave one
    instant two spellings and needed a guard per place they met.

    A parquet path stays lazy — :func:`polars.scan_parquet` is the scan, not
    the read — so the engine still hands the file to the query it builds.

    Raises:
        DataError: A table with no column named after the dimension, labels no
            frame can be made of, or a map keyed by something they do not hold.
    """
    table = pl.scan_parquet(source) if isinstance(source, (str, Path)) else as_frame(source, (dim,))
    table = table if table is not None else labels_frame(dim, source, dtype)
    available = table.collect_schema().names()
    if dim not in available:
        raise DataError(index_without_its_label_column_message(dim, available))
    return _read_declared_maps_against(table, dim, maps) if maps else table


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
            f"for '{dim}'. Declare dimensions.{dim}.values, pass '{dim}': [...] in sources, "
            f"or pass '{name}' as a table carrying its own '{dim}' column."
        )
    frame = pl.scan_parquet(source) if isinstance(source, (str, Path)) else source
    return frame.select(dim).unique(maintain_order=True).collect()[dim].to_list()  # pyrefly: ignore[missing-attribute]


def check_index_ownership(schema: Model, data: Mapping[str, object]) -> None:
    """Refuse anything about a dimension's index that two authors both claim.

    Two facts, each with one home. **The labels** — which members exist, and in
    what order — come from ``dimensions.<d>.values`` or from the caller's ``d``
    source. **Each lookup's map** comes from ``lookups.<x>.values`` or from a
    column named after it in that same source. Resolving either by precedence
    would let the file describe a model the caller does not build, so both are
    refused here, before either lane reads a table.

    A declared map is deliberately *not* a claim on the labels: it is a partial
    relation over the dimension, free to omit members and written in whatever
    key order someone typed
    (:meth:`~lpspec.language.model.Model.declared_maps`). A sparse map over a
    caller's label set is therefore the one index with two authors — one per
    fact — and it is the shape this check exists to keep honest.

    Raises:
        DataError: Naming the dimension, the declaration, and the key or column
            that collided with it.
    """
    for dim, ddef in schema.dimensions.items():
        if dim not in data:
            continue
        where = f"sources['{dim}']"
        if ddef.values is not None:
            raise DataError(declared_index_also_supplied_message(dim, f'dimensions.{dim}.values', where))
        carried = _column_names(data[dim], dim)
        for name in sorted(schema.declared_maps(dim)):
            if name in carried:
                column = f"the '{name}' column of {where}"
                raise DataError(declared_index_also_supplied_message(dim, f'lookups.{name}.values', column))


def check_declared_map_keys(dim: str, maps: Mapping[str, Mapping[Any, Any]], labels: Sequence[Any]) -> None:
    """Every declared map is keyed by labels *dim* actually has — one wording, both lanes.

    The asymmetry is the point. A label no map mentions is the **partial case**
    and gets a null; a key naming no label is a **typo**, and dropping it would
    place its terms nowhere while the model built and solved. Where the file
    declares the labels too, ``Model._declared_lookup_errors`` decides this at
    load; here the labels are the caller's, so the same law lands at bind.

    Raises:
        DataError: Naming the lookup and the keys that match no label.
    """
    known = set(labels)
    for name, values in maps.items():
        strays = sorted(str(k) for k in values if k not in known)
        if strays:
            raise DataError(map_keys_are_not_labels_message(dim, name, strays, [str(x) for x in labels]))


def _read_declared_maps_against(table: pl.LazyFrame, dim: str, maps: Mapping[str, Mapping[Any, Any]]) -> pl.LazyFrame:
    """*table*'s labels, with each declared map read against them as a column.

    The labels are the caller's and stay exactly as they arrive — order
    included. A label a map omits gets a null, which is what a partial relation
    means; a key naming no label is refused first by
    :func:`check_declared_map_keys`, so the left join can only ever add nulls.
    :func:`check_index_ownership` has already refused a table carrying a column
    a map also declares, so neither author can overwrite the other here.
    """
    labels = table.select(dim).collect()[dim].to_list()
    check_declared_map_keys(dim, maps, labels)
    for name, values in maps.items():
        table = table.join(
            pl.LazyFrame({dim: list(values), name: list(values.values())}),
            on=dim,
            how='left',
        )
    return table


def _column_names(source: Any, dim: str) -> frozenset[str]:
    """What a supplied index carries, or nothing where it is a bare label sequence."""
    if isinstance(source, (str, Path)):
        return frozenset(pl.scan_parquet(source).collect_schema().names())
    table = as_frame(source, (dim,))
    return frozenset(table.collect_schema().names()) if table is not None else frozenset()


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
        DataError: A link's values parameter with a hole in the product of the
            dims it declares.
    """
    for block, pw in schema.piecewise.items():
        for link in pw.links:
            dims = list(schema.parameters[link.values].dims)
            present = _coordinates(sources.get(link.values), dims)
            if present is None:
                continue
            extents = {d: _label_frame(d, sources, present) for d in dims}
            expected = 1
            for labels in extents.values():
                expected *= labels.select(pl.len()).collect().item()
            found = present.select(pl.len()).collect().item()
            if found < expected:
                raise DataError(
                    curve_with_a_hole_message(block, link.values, _a_hole(extents, present, dims), expected, found)
                )


def _coordinates(source: object, dims: Sequence[str]) -> pl.LazyFrame | None:
    """The coordinates *source* carries, or ``None`` where it carries all of them.

    ``None`` covers both "dense by construction" and "not readable here" — a
    source binding refuses is refused there, with the message that knows what
    the declaration wanted.
    """
    if isinstance(source, (str, Path)):
        table = pl.scan_parquet(source)
    elif isinstance(source, Mapping) and len(dims) == 1:
        return pl.LazyFrame({dims[0]: list(source.keys())})
    else:
        table = as_frame(source, tuple(dims))
    if table is None or not set(dims) <= set(table.collect_schema().names()):
        return None
    return table.select(dims)


def _label_frame(dim: str, sources: Mapping[str, TidySource], present: pl.LazyFrame) -> pl.LazyFrame:
    """*dim*'s labels as one column, from wherever the model's index comes from.

    *sources* is what :func:`tidy_sources` returned, so a dimension with an
    index — declared by the file or passed by the caller — is already a frame
    here, and which of the two it was stopped mattering at that door.

    Falls back to the labels the curve itself carries, which is what a
    dimension with no index of its own is bound against: there the curve cannot
    be short of a breakpoint nothing else declares.
    """
    source = sources.get(dim)
    if source is None:
        return present.select(dim).unique()
    table = pl.scan_parquet(source) if isinstance(source, (str, Path)) else source
    return table.select(dim).unique()


def _a_hole(extents: Mapping[str, pl.LazyFrame], present: pl.LazyFrame, dims: Sequence[str]) -> str:
    """One coordinate the curve does not carry, written as the reader would look for it.

    Built only on the way to raising, since the product it crosses is the whole
    grid the curve was meant to cover. There is always such a coordinate: the
    caller has counted fewer rows than that grid holds, and a repeated row can
    only make the count larger.
    """
    dtypes = present.collect_schema()
    grid = extents[dims[0]].select(pl.col(dims[0]).cast(dtypes[dims[0]]))
    for dim in dims[1:]:
        grid = grid.join(extents[dim].select(pl.col(dim).cast(dtypes[dim])), how='cross')
    row = grid.join(present, on=list(dims), how='anti').head(1).collect().row(0, named=True)
    return '(' + ', '.join(f'{d}={row[d]!r}' for d in dims) + ')'


def validate_piecewise_data(schema: Model, values: Mapping[str, Any] | Any) -> None:
    """Data-time guard for the methods a curve's shape can make wrong.

    ``convex``'s hull relaxation is wrong for mixed curvature and ``lp``'s
    segment lines for the bend opposite their bounded link — see
    :attr:`PiecewiseBlock.curvature_required` — and both are ill-defined when
    the x-breakpoints are not strictly monotone. All of it is checkable once
    the breakpoint values are in hand, which the schema never has. *values*
    maps parameter names to whatever its lane holds — :func:`tidy_sources`'
    frames and paths, or the linopy lane's ``xr.Dataset`` — and blocks whose
    parameters are missing or bound to a path are skipped. Called by both
    lanes, which is why it sits beside ``tidy_sources``.

    Only the curvature check needs xarray, for the broadcast over dims, so the
    import waits until a block that needs it is found.

    Raises:
        PiecewiseExpansionError: Breakpoints that are not strictly increasing,
            or a curve of the curvature the method is not exact for.
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
                    f'{ctx}: method: {pw.method} requires strictly increasing breakpoints in '
                    f"'{x_link.values}' (got {xs.tolist()})"
                )
            curvature = np.diff(np.diff(ys) / dx)
            tol = 1e-9 * max(1.0, float(np.abs(ys).max()))
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
