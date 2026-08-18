"""Data loading, coercion, and validation."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from lpspec.errors import (
    DataError,
    coordinates_shown,
    declared_map_needs_labels_message,
    dense_array_message,
    duplicate_coordinate_message,
    holes_in_values_message,
    lookups_need_an_index_message,
    missing_lookup_columns_message,
    no_index_source_message,
    sparse_divisor_message,
    uncovered_constant_message,
    unknown_source_keys_message,
)
from lpspec.frames import as_frame
from lpspec.language.expression_parser import (
    BinaryOperatorNode,
    ComparisonNode,
    NameNode,
    ParameterNode,
    VariableNode,
    children,
)
from lpspec.sources import check_declared_map_keys, check_index_ownership

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lpspec.language.expression_parser import ExpressionNode
    from lpspec.language.model import Model


def build_master_coords(schema: Model, sources: Mapping[str, Any] | None = None) -> dict[str, pd.Index]:
    """Assemble master coordinate indices for every declared dimension.

    Where the index comes from, per the data-binding rules: a key in
    ``sources``, or the ``values:`` the YAML declares — and never both, which
    :func:`~lpspec.sources.check_index_ownership` refuses first. There is no fourth step: a dimension without an index has no
    way to tell a mistyped label from a new one.

    Raises:
        DataError: A dimension with no index, or one the file and the caller
            both claim.
    """
    master: dict[str, pd.Index] = {}

    sources = sources or {}
    check_index_ownership(schema, sources)
    for dim_name, dim_def in schema.dimensions.items():
        supplied = supplied_index(schema, dim_name, sources)
        if supplied is not None:
            master[dim_name] = dim_index_of(supplied, dim_name)
        elif dim_def.values is not None:
            master[dim_name] = pd.Index(dim_def.values, name=dim_name)
        else:
            declared = schema.declared_maps(dim_name)
            if declared:
                raise DataError(declared_map_needs_labels_message(dim_name, declared))
            carried = sorted({**schema.targeted_of(dim_name), **schema.labels_of(dim_name)})
            if carried:
                raise DataError(lookups_need_an_index_message(dim_name, carried, 'nothing'))
            raise DataError(no_index_source_message(dim_name))

    return master


def supplied_index(schema: Model, dim_name: str, sources: Mapping[str, Any] = MappingProxyType({})) -> Any:
    """The index for *dim_name* the caller passed, or the one the file declares.

    Where the labels come from, which is the relational lane's rule: a key in
    ``sources``, or the dimension's own ``values:`` — never both, refused before
    this runs. A lookup's ``values:`` supplies no
    labels; where the caller brings them, each declared map is read against
    them here, so nothing downstream distinguishes a map that was declared from
    one that arrived as a column.

    Returns:
        A frame, path or label sequence, or ``None`` where none supplies one.
    """
    supplied = sources.get(dim_name)
    maps = schema.declared_maps(dim_name)
    if supplied is None:
        declared = schema.declared_index(dim_name)
        return None if declared is None else pd.DataFrame(declared)
    if not maps:
        return supplied
    frame = _index_frame(supplied, dim_name)
    frame = pd.DataFrame({dim_name: list(supplied)}) if frame is None else frame
    check_declared_map_keys(dim_name, maps, frame[dim_name].tolist())
    for name, values in maps.items():
        right = pd.DataFrame({dim_name: list(values), name: list(values.values())})
        frame = frame.merge(right, on=dim_name, how='left')
    return frame


def dim_index_of(source: Any, dim_name: str) -> pd.Index:
    """A dimension index from a label sequence, or from any table carrying the labels.

    Table first, sequence second: everything :func:`_index_frame` recognises is
    read for its label column, and what it cannot make a table of is taken as
    the labels themselves.
    """
    frame = _index_frame(source, dim_name)
    if frame is not None:
        if dim_name not in frame.columns:
            msg = (
                f"the index for '{dim_name}' is a table without a '{dim_name}' column "
                f'(has {list(frame.columns)}). The label column must be named after '
                f'the dimension.'
            )
            raise DataError(msg)
        return pd.Index(pd.unique(frame[dim_name]), name=dim_name)
    return pd.Index(source, name=dim_name)


def build_dim_coords(
    schema: Model,
    master_coords: dict[str, pd.Index],
    sources: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, xr.DataArray]]:
    """Assemble declared lookups, checked against the dimension they target.

    A lookup is a column of its ``over`` dimension's index source, so it
    arrives under that dimension's key as any table carrying the label column
    plus one column per lookup — the shapes the relational lane takes there,
    read through the same recogniser. The containment check mirrors the relational
    lane's: a value that is not a label of the target dimension would
    otherwise be dropped by xarray's inner-join alignment, silently losing
    the term it carries. A null value passes the check: it means "this label
    belongs to no group" — row absence, not a typo.

    Only a *targeted* lookup is checked. A label-space lookup owns its
    values, so there is no dimension for them to be contained in and nothing
    the check could ask.
    """
    sources = sources or {}
    out: dict[str, dict[str, xr.DataArray]] = {}

    for dim_name in schema.dimensions:
        declared = {**schema.targeted_of(dim_name), **schema.labels_of(dim_name)}
        if not declared:
            continue
        supplied = supplied_index(schema, dim_name, sources)
        source = _index_frame(supplied, dim_name)
        if source is None:
            got = 'nothing' if supplied is None else 'a shape no table can be made of'
            raise DataError(lookups_need_an_index_message(dim_name, list(declared), got))
        missing = [c for c in sorted(declared) if c not in source.columns]
        if missing:
            raise DataError(missing_lookup_columns_message(dim_name, missing, list(source.columns)))

        labels = master_coords[dim_name]
        first = source.drop_duplicates(subset=[dim_name]).set_index(dim_name)
        counts = source.groupby(dim_name, sort=False)[sorted(declared)].nunique()
        out[dim_name] = {}
        targeted = schema.targeted_of(dim_name)
        for cname in declared:
            if (counts[cname] > 1).any():
                offending = sorted(counts.index[counts[cname] > 1].astype(str))[:5]
                msg = (
                    f"Dimension '{dim_name}': label(s) {offending} carry more than one "
                    f"value for lookup '{cname}'. A lookup is single-valued per "
                    f'label.'
                )
                raise DataError(msg)
            series = first[cname].reindex(labels)
            if cname in targeted:
                target = targeted[cname]
                known = set(master_coords[target])
                unknown = sorted({str(v) for v in series if not pd.isna(v) and v not in known})[:5]
                if unknown:
                    msg = (
                        f"Dimension '{dim_name}' lookup '{cname}' has value(s) that are "
                        f"not '{target}' labels: {', '.join(unknown)}. Every value must "
                        f"be a declared '{target}' label — otherwise "
                        f'sum(by={cname}) drops those terms and the '
                        f'model builds and solves without them.'
                    )
                    raise DataError(msg)
            out[dim_name][cname] = xr.DataArray(
                series.to_numpy(),
                dims=[dim_name],
                coords={dim_name: labels},
                name=cname,
            )

    return out


def _index_frame(source: Any, dim: str) -> pd.DataFrame | None:
    """A dimension's index source as the pandas frame the lookups are read off.

    Every table shape a parameter may arrive in, plus a parquet path — the
    dimension side of a caller's ``sources`` should take what the parameter
    side does, and a lookup lives in a column either way.

    Returns:
        The frame, or ``None`` for a source no table can be made of.
    """
    if isinstance(source, pd.DataFrame):
        return source
    if source is None:
        return None
    if isinstance(source, (str, Path)):
        return pl.scan_parquet(source).collect().to_pandas()
    table = as_frame(source, (dim,))
    if table is None:
        return None
    frame = table.collect() if isinstance(table, pl.LazyFrame) else table
    return pd.DataFrame({name: frame[name].to_numpy() for name in frame.columns})


def load_parameters(
    schema: Model,
    data: dict[str, Any] | None,
    master_coords: dict[str, pd.Index],
) -> xr.Dataset:
    """Load, coerce, and validate all declared parameters.

    Dim and coordinate checking happens here rather than per input shape:
    every branch of ``_coerce_to_dataarray`` produces a DataArray, and every
    one of them owes the same two guarantees.

    A key naming a *dimension* is a dimension index, read by
    :func:`build_master_coords` and :func:`build_dim_coords` rather than here,
    and passes through untouched — one ``sources`` mapping carries both, as it
    does on the relational lane. A key naming neither is refused there and here
    alike.

    Returns:
        One DataArray per parameter, aligned to the master coordinates.

    Raises:
        DataError: A parameter missing, or dims or labels other than the ones
            declared.
    """
    data = data or {}
    arrays: dict[str, xr.DataArray] = {}

    known = {**schema.parameters, **schema.dimensions}
    if unknown := set(data) - set(known):
        raise DataError(unknown_source_keys_message(unknown, known))

    for pname in schema.parameters:
        if pname not in data:
            msg = f"Parameter '{pname}' is required but was not provided in data.\nAdd '{pname}' to the data= argument."
            raise DataError(msg)

    for pname, pdef in schema.parameters.items():
        raw = data[pname]
        arr = _coerce_to_dataarray(pname, raw, pdef.dims, master_coords)
        _validate_dims(pname, arr, pdef.dims)
        _validate_coords(pname, arr, master_coords)

        if pdef.dims:
            reindex_coords = {d: master_coords[d] for d in pdef.dims}
            if arr.ndim == 0:
                scalar_val = float(arr.values)
                shape = tuple(len(master_coords[d]) for d in pdef.dims)
                arr = xr.DataArray(
                    np.full(shape, scalar_val),
                    dims=pdef.dims,
                    coords=reindex_coords,
                )
            elif arr.dtype == bool:
                arr = arr.reindex(reindex_coords, fill_value=False)
            else:
                arr = arr.reindex(reindex_coords)

        arrays[pname] = arr

    return xr.Dataset(arrays)


def _refuse_holes(name: str, values: pd.Series, dims: Sequence[str]) -> None:
    """A row carrying no value, asked while the source's own shape survives.

    Of the source and never of the array: ``DataArray.from_series`` unstacks a
    MultiIndex and fills every combination the source did not carry with NaN,
    so a sparse parameter would read as holed the moment it became one. Asked
    per input shape for the same reason — a dict, a sequence and a tidy frame
    each stop being a list of supplied rows at a different line.

    This lane cannot tell a null from a NaN and does not try: pandas spells
    both NaN, which is the reason the refusal covers the pair.
    """
    holes = values.isna()
    total = int(holes.sum())
    if not total:
        return
    keys = [_native(key) for key in values.index[holes][:3]] if dims else ()
    raise DataError(holes_in_values_message(name, total, coordinates_shown(dims, keys)))


def _native(key: Any) -> tuple[Any, ...]:
    """One index key as the python natives the shared message formats.

    A numpy scalar reprs as ``np.str_('b')`` under numpy 2, and the relational
    lane's wording has no numpy in it to match.
    """
    values = key if isinstance(key, tuple) else (key,)
    return tuple(v.item() if hasattr(v, 'item') else v for v in values)


def _refuse_duplicate_index(name: str, index: pd.Index, dims: list[str]) -> None:
    """Two values for one coordinate, before xarray sees it.

    `DataArray.from_series` raises `ValueError: cannot reindex or align along
    dimension ... duplicate values` from its index machinery — which names
    neither the parameter nor the repair. The relational lane already refuses
    this with wording that does both, and `tests/test_data_parity.py` is what
    asserts the two lanes agree (#351).
    """
    duplicated = index[index.duplicated()].unique()
    if len(duplicated) == 0:
        return
    shown = '; '.join(f'{dims[0]}={label!r}' for label in duplicated[:3])
    raise DataError(duplicate_coordinate_message(name, shown, dims))


def _coerce_to_dataarray(
    name: str,
    raw: Any,
    dims: list[str],
    master_coords: dict[str, pd.Index],
) -> xr.DataArray:
    """Coerce a user-provided value into an ``xr.DataArray``.

    Tables in: a ``Series`` keeps its dims in an index and a ``DataFrame`` in
    columns, and each binds **by name**, so a caller hands the same object to
    either lane. A dense ``xarray.DataArray`` is refused rather than taken —
    xarray is what this lane *builds*, not what it reads.

    A dict, a sequence and a bare number are the plain-Python shapes, spread
    over the master coordinates the same way the relational front door spreads
    them.
    """
    if isinstance(raw, xr.DataArray):
        raise DataError(dense_array_message(name))

    if isinstance(raw, (str, Path)):
        return _from_tidy(name, pl.scan_parquet(raw), dims)

    if isinstance(raw, (int, float, np.integer, np.floating)):
        _refuse_holes(name, pd.Series([float(raw)]), ())
        return xr.DataArray(float(raw))

    if isinstance(raw, dict):
        if len(dims) != 1:
            msg = f"Parameter '{name}': dict input is only supported for 1-D parameters, but declared dims are {dims}."
            raise DataError(msg)
        series = pd.Series(raw)
        series.index.name = dims[0]
        raw = series

    if isinstance(raw, pd.DataFrame):
        raw = _tidy_series(name, raw, dims)

    if isinstance(raw, pd.Series):
        if raw.index.nlevels != len(dims):
            msg = (
                f"Parameter '{name}' is over {dims}, and its index has "
                f'{raw.index.nlevels} level(s). Set the index to the dims the '
                f'parameter declares — set_index({dims}) on a tidy frame.'
            )
            raise DataError(msg)
        if any(n is None for n in raw.index.names):
            raw = raw.copy()
            raw.index.names = dims
        _refuse_duplicate_index(name, raw.index, dims)
        _refuse_holes(name, raw, dims)
        return xr.DataArray.from_series(raw)

    if not isinstance(raw, (np.ndarray, list, tuple)):
        table = as_frame(raw, dims)
        if table is not None:
            return _from_tidy(name, table, dims)

    if isinstance(raw, (np.ndarray, list, tuple)):
        arr_np = np.asarray(raw)
        if arr_np.ndim == 0:
            return xr.DataArray(float(arr_np))
        if arr_np.ndim == 1 and len(dims) == 1:
            dim = dims[0]
            if len(arr_np) != len(master_coords[dim]):
                msg = (
                    f"Parameter '{name}': array length {len(arr_np)} does not "
                    f"match master coordinate '{dim}' length "
                    f'{len(master_coords[dim])}.'
                )
                raise DataError(msg)
            _refuse_holes(name, pd.Series(arr_np, index=master_coords[dim]), [dim])
            return xr.DataArray(arr_np, dims=[dim], coords={dim: master_coords[dim]})
        msg = (
            f"Parameter '{name}': a sequence is positional along one dimension, and "
            f"'{name}' is over {dims}. Pass a table carrying {[*dims, 'value']} instead."
        )
        raise DataError(msg)

    type_name = type(raw).__name__
    msg = f"Parameter '{name}': unsupported type '{type_name}'."
    raise TypeError(msg)


def _from_tidy(name: str, table: pl.LazyFrame | pl.DataFrame, dims: list[str]) -> xr.DataArray:
    """A tidy ``(dims…, value)`` frame as the array this lane builds against.

    The seam that lets one object reach either lane: everything
    :func:`~lpspec.frames.as_frame` recognises — polars, pyarrow,
    duckdb — plus a parquet path arrives here already tidy, and only this last
    step differs from what the relational lane does with it.

    Read through numpy rather than ``to_pandas()``, which wants pyarrow: this
    extra ships pandas and xarray, and nothing says it ships that too.

    Raises:
        DataError: The frame does not carry the declared dims and ``value``, or
            holds more than one row for a coordinate.
    """
    frame = table.collect() if isinstance(table, pl.LazyFrame) else table
    wanted = [*dims, 'value']
    if any(column not in frame.columns for column in wanted):
        raise DataError(
            f"Parameter '{name}': a table is read tidy — one row per coordinate, with "
            f'columns {wanted}. Got {frame.columns}.'
        )
    if not dims:
        _refuse_holes(name, pd.Series([frame['value'][0]]), ())
        return xr.DataArray(float(frame['value'][0]))
    columns = [frame[d].to_numpy() for d in dims]
    index = pd.Index(columns[0], name=dims[0]) if len(dims) == 1 else pd.MultiIndex.from_arrays(columns, names=dims)
    _refuse_duplicate_index(name, index, dims)
    series = pd.Series(frame['value'].to_numpy(), index=index)
    _refuse_holes(name, series, dims)
    return xr.DataArray.from_series(series)


def _tidy_series(name: str, frame: pd.DataFrame, dims: list[str]) -> pd.Series:
    """A tidy ``(dims…, value)`` frame as the indexed series the rest reads.

    The one reading of a ``DataFrame``, so the same object means the same thing
    on both lanes. A frame indexed by the dims already — no ``value`` column,
    one data column — is taken as it stands, which is what ``reset_index()``
    round-trips to.
    """
    if 'value' not in frame.columns:
        missing = [d for d in dims if d not in frame.columns and d not in (frame.index.names or [])]
        if missing or frame.shape[1] != 1:
            raise DataError(
                f"Parameter '{name}': a frame is read tidy — one row per coordinate, with "
                f'columns {[*dims, "value"]}. Got columns {list(frame.columns)}.'
            )
        return frame.iloc[:, 0].rename('value')
    named = [d for d in dims if d in frame.columns]
    if len(named) != len(dims):
        raise DataError(
            f"Parameter '{name}': a frame is read tidy — one row per coordinate, with "
            f'columns {[*dims, "value"]}. Got columns {list(frame.columns)}.'
        )
    return frame.set_index(named)['value']


def _validate_dims(
    name: str,
    arr: xr.DataArray,
    declared_dims: list[str],
) -> None:
    """Check that the DataArray's dims are a subset of declared dims."""
    unexpected = set(arr.dims) - set(declared_dims)
    if unexpected:
        msg = (
            f"Parameter '{name}' has unexpected dimensions {unexpected}.\n"
            f'Declared dims: {declared_dims}.\n'
            f'Either update the declaration or reshape your data.'
        )
        raise DataError(msg)


def _validate_coords(
    name: str,
    arr: xr.DataArray,
    master_coords: dict[str, pd.Index],
) -> None:
    """Check that all coordinate values exist in the master coordinate."""
    for dim in arr.dims:
        if dim not in master_coords:
            continue
        arr_vals = set(arr.coords[dim].values)
        master_vals = set(master_coords[dim])
        unknown = arr_vals - master_vals
        if unknown:
            msg = (
                f"Parameter '{name}' has values in dimension '{dim}' "
                f'that are not in the master coordinate: {sorted(unknown)}.\n'
                f"Master '{dim}' coords: {list(master_coords[dim])}"
            )
            raise DataError(msg)


def gaps_under(array: Any, mask: Any) -> int:
    """How many slots of *array* are null where *mask* still admits the row.

    The eager lane's one way of asking "is this parameter defined where it is
    needed" — a bound, a divisor and a constant side all ask it, and a second
    spelling is a second chance to forget the mask and refuse a model whose
    ``where`` had already answered. ``None`` means nothing narrows the question.
    """
    missing = array.isnull()
    if mask is not None:
        missing = missing & mask
    return int(missing.sum())


def check_constant_side_covers(name: str, node: ComparisonNode, schema: Model, dataset: Any, mask: Any) -> None:
    """A comparison's constant side must have values wherever the row is built.

    The divisor argument, one position over. A missing row is read as 0, and on
    a side with no variable that zero *is* the bound — `x <= cap` becomes
    `x <= 0`, which binds rather than vanishing, and the solve reports optimal.

    Keyed to the rows the declaration builds, not to the coordinate product:
    a `where` that removed the coordinate has already answered the question,
    which is what makes masking the escape rather than a workaround.

    The relational lane asks the same thing from the other end — it left-joins
    the constant parts and looks for a null before the fill. Same answer,
    reached by the shape each lane has to hand.
    """
    for side in (node.left, node.right):
        if _names_of(side, schema.variables):
            continue
        params = _names_of(side, schema.parameters)
        if not params:
            continue
        for param in sorted(params):
            missing = gaps_under(dataset[param], mask)
            if missing:
                raise DataError(uncovered_constant_message(param, missing, name))


def check_divisors_cover(name: str, node: ExpressionNode, schema: Model, dataset: Any, mask: Any, model: Any) -> None:
    """A divisor must have a value wherever this declaration divides by it.

    Not "wherever it is indexed": sparse data is the ordinary case, and a check
    keyed to the coordinate product would refuse models that never touch the
    gap. Two things can already have removed a coordinate — the row's own
    ``where``, and the mask on a variable in the numerator — and either is
    enough, so the requirement is their conjunction.

    The relational lane asks the same question from the other end: it left-joins
    the divisor and looks for a null coefficient in the assembled matrix, which
    only survives if the row was built and the numerator existed. Same answer,
    reached by the shape each lane has to hand.

    Reached before ``_eval_ast``, the last moment the gap is visible:
    ``builder._coefficient`` fills an uncovered slot with 0.0 at the parameter
    leaf, and from there the division yields an infinity and the row is masked
    out — silently, and identically on both lanes until #312.
    """
    for quotient in _quotients(node):
        params = _names_of(quotient.right, schema.parameters)
        if not params:
            continue
        needed = mask
        for variable in _names_of(quotient.left, schema.variables):
            present = model.variables[variable].labels != -1
            needed = present if needed is None else (needed & present)
        for param in sorted(params):
            missing = gaps_under(dataset[param], needed)
            if missing:
                raise DataError(f'{name}: {sparse_divisor_message(param, missing)}')


def _quotients(node: ExpressionNode) -> list[BinaryOperatorNode]:
    """Every division node under *node*."""
    out = [node] if isinstance(node, BinaryOperatorNode) and node.op == '/' else []
    for child in children(node):
        out.extend(_quotients(child))
    return out


def _names_of(node: ExpressionNode, declared: Iterable[str]) -> set[str]:
    """Declared names under *node*, whether the AST is resolved or not."""
    found: set[str] = set()
    if isinstance(node, (NameNode, ParameterNode, VariableNode)) and node.name in declared:
        found.add(node.name)
    for child in children(node):
        found |= _names_of(child, declared)
    return found
