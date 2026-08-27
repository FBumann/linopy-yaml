"""Data loading, coercion, and validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl
import xarray as xr

from lpspec.errors import (
    DataError,
    coordinates_shown,
    duplicate_coordinate_message,
    holes_in_values_message,
    no_index_source_message,
    wrong_value_dtype_message,
)
from lpspec.frames import scan, to_pandas

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import lpspec.plan as plan
    from lpspec.frames import TidySource


def dimension_coords(
    program: plan.Program,
    tidy: Mapping[str, TidySource],
) -> tuple[dict[str, pd.Index], dict[str, dict[str, xr.DataArray]]]:
    """Every dimension's labels, and the lookup columns that arrived beside them.

    *tidy* is :func:`~lpspec.sources.tidy_sources`' output — the one front door,
    which every lane enters through and which has already read each index into
    polars. This converts, which is what this lane is: linopy's libraries,
    entered at their boundary. Nothing about the object a caller passed
    survives that crossing, so a `datetime.date` out of pandas and a `pl.Date`
    out of polars are one instant by construction rather than by a guard at
    every place the two used to meet.

    Both halves come back from one call because they come from one read of the
    sources — an index is small, but reading it twice is two chances to
    disagree about what it said.

    Returns:
        The master coordinates by dimension, and by dimension the array each
        declared lookup carries over it. A dimension declaring no lookup is
        absent from the second.

    Raises:
        DataError: A dimension with no index, or a lookup value that is not a
            label of the dimension it targets.
    """
    frames: dict[str, pd.DataFrame] = {}
    for declared in program.dimensions:
        table = tidy.get(declared.name)
        if table is None:
            raise DataError(no_index_source_message(declared.name))
        frames[declared.name] = to_pandas(collected(table))

    master = {d.name: pd.Index(pd.unique(frames[d.name][d.name]), name=d.name) for d in program.dimensions}
    return master, _lookup_arrays(program, tidy, master)


def collected(source: TidySource) -> pl.DataFrame:
    """A bound source in hand, scanning the path the engine would have streamed.

    The engine keeps a parquet parameter lazy so its scan lands inside the
    query; linopy holds the whole model in memory anyway, so this lane reads it
    where the engine would push it down.
    """
    return scan(source).collect()


def _lookup_arrays(
    program: plan.Program,
    tidy: Mapping[str, TidySource],
    master: Mapping[str, pd.Index],
) -> dict[str, dict[str, xr.DataArray]]:
    """Each declared lookup as an array over the dimension it is over.

    A map arrives as its own ``(over, lookup)`` relation, single-valued and
    holding rows only where it is defined
    (:func:`~lpspec.sources.tidy_sources`). **The padding happens here**, and
    only here: an array is dense by construction, and linopy's ``groupby``
    wants one aligned to the dimension's coordinates — so a label the relation
    leaves out becomes a null in it, and every reader on this lane already
    treats that null as "in no group".

    Nothing is checked here. A value naming no label of the target would be
    dropped by xarray's inner-join alignment, losing the term it carries with
    no error anywhere — which is why it is refused at
    :func:`~lpspec.sources.lookup_relations`, before either lane holds a map at
    all.
    """
    out: dict[str, dict[str, xr.DataArray]] = {}
    for declared in program.dimensions:
        dim, labels = declared.name, master[declared.name]
        for name in declared.maps:
            series = to_pandas(collected(tidy[name])).set_index(dim)[name].reindex(labels)
            out.setdefault(dim, {})[name] = xr.DataArray(series.to_numpy(), dims=[dim], coords={dim: labels}, name=name)
    return out


def load_parameters(
    program: plan.Program,
    tidy: Mapping[str, TidySource],
    master_coords: Mapping[str, pd.Index],
) -> xr.Dataset:
    """Every declared parameter as the dataset this lane builds against.

    *tidy* is :func:`~lpspec.sources.tidy_sources`' output, so every shape a
    caller may pass — a frame of any library, a dict, a sequence, a bare
    number, a parquet path — has already been read into one tidy
    ``(dims…, value)`` frame, and what is left here is the crossing into
    xarray. A source that is missing, unreadable or the wrong shape was refused
    at that front door, in the sentence the other lane gives for it.

    Returns:
        One DataArray per parameter, reindexed onto the master coordinates.

    Raises:
        DataError: A parameter whose values are not the dtype it declares, or
            whose labels are not the ones its dimensions hold.
    """
    arrays: dict[str, xr.DataArray] = {}
    for pdef in program.parameters:
        pname = pdef.name
        arr = _from_tidy(pname, collected(tidy[pname]), pdef.dims, pdef.dtype)
        _validate_dims(pname, arr, pdef.dims)
        _validate_coords(pname, arr, master_coords)

        if pdef.dims:
            onto = {d: master_coords[d] for d in pdef.dims}
            arr = arr.reindex(onto, fill_value=False) if arr.dtype == bool else arr.reindex(onto)

        arrays[pname] = arr

    return xr.Dataset(arrays)


#: The numpy kinds each declared dtype *is* — this lane's ``_COLUMNS``, and
#: what names the type in the sentence both lanes share.
_KINDS: dict[str, str] = {'float': 'f', 'int': 'iu', 'bool': 'b', 'str': 'OUS'}

#: What each accepts: the same one widening the relational lane makes, built
#: the same way, so neither lane can gain a second without the other. Pinned to
#: the language's ``PARAMETER_DTYPES`` by ``tests/test_architecture.py``.
_ACCEPTED_KINDS: dict[str, str] = {**_KINDS, 'float': _KINDS['float'] + _KINDS['int']}


def _check_values(name: str, values: pd.Series, dims: Sequence[str], declared: str) -> None:
    """The two questions this lane owes a bound column, while its shape survives.

    The declared dtype is a claim about the values and is checked first: a
    column that is not what the file says makes the file describe a model the
    data does not build, and on this lane it also decides what a bare ``where``
    on the name means.

    The holes are asked first: a column of nothing but them carries no type to
    check, and "no value here" is the sentence that names the repair.

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
    if total:
        keys = [_native(key) for key in values.index[holes][:3]] if dims else ()
        raise DataError(holes_in_values_message(name, total, coordinates_shown(dims, keys)))
    kind = values.dtype.kind
    if kind not in _ACCEPTED_KINDS[declared]:
        arrived = next((name for name, kinds in _KINDS.items() if kind in kinds), str(values.dtype))
        raise DataError(wrong_value_dtype_message(name, declared, arrived))


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


def _from_tidy(name: str, table: pl.LazyFrame | pl.DataFrame, dims: Sequence[str], declared: str) -> xr.DataArray:
    """A tidy ``(dims…, value)`` frame as the array this lane builds against.

    The seam that lets one object reach either lane: everything
    :func:`~lpspec.frames.as_frame` recognises — polars, pyarrow,
    duckdb — plus a parquet path arrives here already tidy, and only this last
    step differs from what the relational lane does with it.

    Read through numpy rather than ``to_pandas()``, which wants pyarrow: this
    extra ships pandas and xarray, and nothing says it ships that too. A
    dims-less value keeps the dtype it arrived with, as a column does — a
    ``bool`` cast to ``0.0`` reads as *defined* under a bare ``where``, which
    is the opposite of what the file said.

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
        value = frame['value'].to_numpy()[0]
        _check_values(name, pd.Series([value]), (), declared)
        return xr.DataArray(value)
    columns = [frame[d].to_numpy() for d in dims]
    index = (
        pd.Index(columns[0], name=dims[0]) if len(dims) == 1 else pd.MultiIndex.from_arrays(columns, names=list(dims))
    )
    _refuse_duplicate_index(name, index, list(dims))
    series = pd.Series(frame['value'].to_numpy(), index=index)
    _check_values(name, series, dims, declared)
    return xr.DataArray.from_series(series)


def _validate_dims(
    name: str,
    arr: xr.DataArray,
    declared_dims: Sequence[str],
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
    master_coords: Mapping[str, pd.Index],
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
