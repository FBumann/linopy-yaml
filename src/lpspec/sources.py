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
    curve_mask_is_not_contiguous_message,
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
from lpspec.language.piecewise import mask_of

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
    coordinate — are meaningless without the labels they are spread over. A
    ``piecewise:`` block's derived flags come next, before the parameter loop
    asks for data no caller can have (:func:`derive_curve_edges`).

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

    sources = derive_curve_edges(schema, derive_curve_masks(schema, sources, data), data)

    for pname, pdef in schema.parameters.items():
        if pname in sources:
            continue
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
    that range is per curve, where ``index(over, 0)`` and ``index(over, -1)``
    are the axis'. A ``where:`` takes no operators to find it with, so the two
    edges are marked here instead, from the mask the caller supplied: the first
    and last position it marks, per curve.

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
                        curve_with_a_hole_message(
                            block, link.values, _a_hole(grid, present, dims), expected, found, pw.points
                        )
                    )
                continue
            required = _required_under_mask(mask, extents, dims, present)
            counts = pl.collect_all([required.select(pl.len()), present.select(pl.len())])
            if required.join(present, on=dims, how='anti').head(1).collect().height:
                raise DataError(
                    curve_with_a_hole_message(
                        block,
                        link.values,
                        _a_hole(required, present, dims),
                        counts[0].item(),
                        counts[1].item(),
                        pw.points,
                    )
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
        raise DataError(curve_mask_is_not_contiguous_message(block, mask, pw.over, shown))
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
    table = pl.scan_parquet(source) if isinstance(source, (str, Path)) else source
    return table.select(dim).unique(maintain_order=True)


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
    turned on how the numbers were handed over is no verdict at all. Only a
    block whose parameters are absent is skipped. Called by both lanes, which
    is why it sits beside ``tidy_sources``.

    **The breakpoints are walked in the ``over`` dimension's own index order**,
    which is the order the model is built in — what ``shift`` walks and what
    ``index(bp, 0)`` names. A values table is a function of its coordinates and
    carries no order of its own, so it is laid out on that index first
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
        try:
            xa = _as_dataarray(schema, x_link.values, values)
            ya = _as_dataarray(schema, y_link.values, values)
        except KeyError:
            continue
        xa, ya = xr.broadcast(xa, ya)
        if (order := _breakpoint_order(pw.over, values)) is not None:
            xa, ya = xa.reindex({pw.over: order}), ya.reindex({pw.over: order})
        if pw.method == 'lp' and xa.sizes[pw.over] < 2:
            raise PiecewiseExpansionError(
                f"{ctx}: method: lp needs at least two breakpoints, and '{pw.over}' has "
                f'{xa.sizes[pw.over]} — the method is its segment lines, and a curve of one '
                f'point has no segment for them to state, so the bounded link would be left '
                f'free. Use method: adjacency, sos2 or convex, which pin it to the one point.'
            )
        other = [d for d in xa.dims if d != pw.over]
        stacked_x = xa.transpose(*other, pw.over).values.reshape(-1, xa.sizes[pw.over])
        stacked_y = ya.transpose(*other, pw.over).values.reshape(-1, ya.sizes[pw.over])
        for xs_all, ys_all in zip(stacked_x, stacked_y, strict=False):
            live = ~(np.isnan(xs_all) | np.isnan(ys_all))
            xs, ys = xs_all[live], ys_all[live]
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


def _as_dataarray(schema: Model, pname: str, values: Mapping[str, Any] | Any) -> Any:
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
    dims = list(schema.parameters[pname].dims)
    frame = pl.scan_parquet(obj) if isinstance(obj, (str, Path)) else as_frame(obj, tuple(dims))
    if frame is None or not dims or 'value' not in frame.collect_schema().names():
        raise KeyError(pname)
    import pandas as pd

    tidy = frame.select([*dims, 'value']).collect()
    columns = {name: tidy[name].to_numpy() for name in tidy.columns}
    return xr.DataArray.from_series(pd.DataFrame(columns).set_index(dims)['value'])
