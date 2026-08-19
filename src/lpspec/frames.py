"""The one place that knows what a caller's table library is.

What a caller hands over is learned from the Arrow PyCapsule protocol, not from
an import, so neither pyarrow nor pandas is a dependency. ``pandas.Series`` has
no capsule that carries its index, so it is unwrapped first — and only when
pandas is already in ``sys.modules``.

**Tables in, arrays out.** What is read here is a table: rows under named
columns, an index being a column wearing a hat. An ``xarray.DataArray`` is a
dense n-dimensional array rather than a table, and taking one would be this
package agreeing that a parameter is a rectangle already materialised. xarray
is what a result is handed back *as* (``to_dataarray``) and what the linopy
lane builds internally, never what either lane reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from lpspec.errors import DataError

if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = ['as_frame', 'is_dense_array', 'labels_frame']


def as_frame(obj: object, dims: Sequence[str] = ()) -> pl.LazyFrame | None:
    """Normalise one in-memory source to a tidy lazy frame.

    *dims* names the columns an index becomes. A bool stays boolean rather than
    widening to float: the engine reads a mask's truthiness from the column
    type (#47).

    Returns:
        The tidy frame, or ``None`` for "not table-shaped" — the caller knows
        whether it held a parameter or an index and writes the message.
    """
    import sys

    if isinstance(obj, bool) and not dims:
        return pl.LazyFrame({'value': [obj]}, schema={'value': pl.Boolean})
    if isinstance(obj, (int, float)) and not isinstance(obj, bool) and not dims:
        return pl.LazyFrame({'value': [float(obj)]}, schema={'value': pl.Float64})
    if isinstance(obj, pl.LazyFrame):
        return obj
    if isinstance(obj, pl.DataFrame):
        return obj.lazy()

    pd = sys.modules.get('pandas')
    if pd is not None and isinstance(obj, pd.Series):
        obj = _series_to_frame(obj, dims)
    if pd is not None and isinstance(obj, pd.DataFrame):
        return _from_pandas(obj)

    if hasattr(obj, '__arrow_c_stream__') or hasattr(obj, '__arrow_c_array__'):
        try:
            return pl.DataFrame(obj).lazy()  # pyrefly: ignore[bad-argument-type]  — narrowed by the capsule test
        except (TypeError, ValueError, pl.exceptions.PolarsError):
            return None
    return None


def is_dense_array(obj: object) -> bool:
    """Whether *obj* is the one shape recognised and deliberately not read.

    An ``xarray.DataArray`` has ``__len__``, so left unasked it would fall
    through to a positional read and bind a dense array as a sequence of values
    in index order. Asked, the caller raises
    :func:`~lpspec.errors.dense_array_message` and names the rewrite.
    """
    import sys

    xr = sys.modules.get('xarray')
    return xr is not None and isinstance(obj, xr.DataArray)


def is_multi_indexed(obj: object) -> bool:
    """Whether *obj* is a pandas Series carrying more than one index level.

    A pandas index is the one shape with no counterpart in the frames both
    lanes build — polars has no index at all — so a MultiIndex is read by
    promoting its levels to columns, and its *depth* is then a claim about the
    parameter's arity that nothing downstream re-checks. Asked here, the caller
    raises :func:`~lpspec.errors.multi_indexed_series_message` and names the
    tidy frame that says the same thing in columns.
    """
    import sys

    pd = sys.modules.get('pandas')
    return pd is not None and isinstance(obj, pd.Series) and obj.index.nlevels > 1


def _series_to_frame(series: Any, dims: Sequence[str]) -> Any:
    """A pandas Series with its index promoted to columns.

    One level, so one dim: :func:`is_multi_indexed` has already refused the
    rest. Where the caller named the level it binds by that name — renaming it
    to *dims* would transpose the data when two dims share a label space, which
    nothing downstream can catch.
    """
    if any(n is None for n in series.index.names):
        series = series.rename_axis(dims)
    return series.rename('value').reset_index()


def _from_pandas(frame: Any) -> pl.LazyFrame:
    """A pandas frame, column by column, without reaching for pyarrow.

    A whole-frame conversion needs pyarrow for anything Arrow-backed, which
    strings are by default on pandas 3. Object arrays go through a list so
    numpy's float ``nan`` becomes a null rather than a string.
    """
    columns: dict[str, Any] = {}
    for name in frame.columns:
        values = frame[name].to_numpy()
        if values.dtype == object:
            columns[name] = pl.Series(name, [None if _is_missing(v) else v for v in values], strict=False)
        else:
            columns[name] = values
    return pl.DataFrame(columns).lazy()


def _is_missing(value: Any) -> bool:
    """Whether an object-array entry is pandas' rendering of "no value"."""
    return value is None or (isinstance(value, float) and value != value)


#: The declared dimension dtypes (the declaration rules), as the column an index becomes.
#: Read only when there are no labels to infer from — polars decides the rest,
#: and a cast over labels that exist could change how the where-string rules
#: compare them.
_DECLARED: dict[str, pl.DataType] = {
    'int': pl.Int64(),
    'float': pl.Float64(),
    'str': pl.String(),
    'datetime': pl.Datetime('us'),
}


def labels_frame(dname: str, values: object, dtype: str) -> pl.LazyFrame:
    """A one-column index frame from a plain sequence of labels.

    **An empty index takes the dimension's declared dtype.** polars infers
    ``Null`` from no labels, and a ``Null`` key joins against nothing — so a
    parameter with the right dtype and no rows fails to bind against the
    dimension it belongs to. The declaration is the only thing that knows, and
    it always answers.

    An empty index is not a corner case for a driver that grows one. A Benders
    cut set starts empty, and so does any dimension whose members a caller
    appends to between solves.
    """
    try:
        labels: list[Any] = list(values)  # pyrefly: ignore[bad-argument-type]  — `values` is whatever a caller passed
        if not labels:
            return pl.LazyFrame(schema={dname: _DECLARED[dtype]})
        return pl.LazyFrame({dname: labels})
    except (TypeError, pl.exceptions.PolarsError) as exc:
        raise DataError(
            f"index for dimension '{dname}': cannot read labels out of "
            f'{type(values).__name__} — pass a sequence of labels, a table '
            f'polars can read with a {dname!r} column, or a parquet path'
        ) from exc
