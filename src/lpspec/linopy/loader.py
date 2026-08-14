"""Data loading, coercion, and validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import xarray as xr

from lpspec.errors import (
    DataError,
    duplicate_coordinate_message,
    sparse_divisor_message,
    uncovered_constant_message,
)
from lpspec.language.expression_parser import (
    BinaryOperatorNode,
    ComparisonNode,
    NameNode,
    ParameterNode,
    VariableNode,
    children,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lpspec.language.expression_parser import ExpressionNode
    from lpspec.language.model import Model


def build_master_coords(
    schema: Model,
    coords: dict[str, Any] | None,
) -> dict[str, pd.Index]:
    """Assemble master coordinate indices for every declared dimension.

    ``coords`` takes precedence over the ``values:`` the YAML declares.

    Raises:
        DataError: A dimension whose labels come from neither.
    """
    coords = coords or {}
    master: dict[str, pd.Index] = {}

    for dim_name, dim_def in schema.dimensions.items():
        supplied = supplied_index(schema, coords, dim_name)
        if supplied is not None:
            master[dim_name] = dim_index_of(supplied, dim_name)
        elif dim_def.values is not None:
            master[dim_name] = pd.Index(dim_def.values, name=dim_name)
        else:
            msg = (
                f"Dimension '{dim_name}' has no values.\n"
                f"Declare them under 'dimensions.{dim_name}.values' in the YAML\n"
                f"or pass coords={{'{dim_name}': [...]}}."
            )
            raise DataError(msg)

    return master


def supplied_index(schema: Model, coords: dict[str, Any], dim_name: str) -> Any:
    """The index for *dim_name* the caller passed, or the one the file declares.

    A lookup carrying ``values:`` puts its map in the file, so the dimension it
    is over has an index without ``coords=`` — assembled into the same frame a
    caller would have passed, which is why nothing downstream distinguishes the
    two. ``coords=`` wins where both exist, as it does over ``values:``.

    Returns:
        A frame or label sequence, or ``None`` where neither supplies one.
    """
    if dim_name in coords:
        return coords[dim_name]
    declared = schema.declared_index(dim_name)
    return None if declared is None else pd.DataFrame(declared)


def dim_index_of(source: Any, dim_name: str) -> pd.Index:
    """A dimension index from a label sequence or a frame carrying coordinates."""
    if isinstance(source, pd.DataFrame):
        if dim_name not in source.columns:
            msg = (
                f"coords['{dim_name}'] is a DataFrame without a '{dim_name}' column "
                f'(has {list(source.columns)}). The label column must be named after '
                f'the dimension.'
            )
            raise DataError(msg)
        return pd.Index(pd.unique(source[dim_name]), name=dim_name)
    return pd.Index(source, name=dim_name)


def build_dim_coords(
    schema: Model,
    coords: dict[str, Any] | None,
    master_coords: dict[str, pd.Index],
) -> dict[str, dict[str, xr.DataArray]]:
    """Assemble declared lookups, checked against the dimension they target.

    A lookup is a column of its ``over`` dimension's index source, so it
    arrives through ``coords=`` as a DataFrame carrying the label column plus
    one column per lookup. The containment check mirrors the relational
    lane's: a value that is not a label of the target dimension would
    otherwise be dropped by xarray's inner-join alignment, silently losing
    the term it carries. A null value passes the check: it means "this label
    belongs to no group" — row absence, not a typo.

    Only a *targeted* lookup is checked. A label-space lookup owns its
    values, so there is no dimension for them to be contained in and nothing
    the check could ask.
    """
    coords = coords or {}
    out: dict[str, dict[str, xr.DataArray]] = {}

    for dim_name in schema.dimensions:
        declared = {**schema.targeted_of(dim_name), **schema.labels_of(dim_name)}
        if not declared:
            continue
        source = supplied_index(schema, coords, dim_name)
        if not isinstance(source, pd.DataFrame):
            got = type(source).__name__ if source is not None else 'nothing'
            msg = (
                f"Dimension '{dim_name}' carries lookups {sorted(declared)}, "
                f"so coords['{dim_name}'] must be a DataFrame with a '{dim_name}' column "
                f'plus one column per lookup (got {got}).'
            )
            raise DataError(msg)
        missing = [c for c in sorted(declared) if c not in source.columns]
        if missing:
            msg = (
                f"Dimension '{dim_name}' index is missing declared lookup column(s) "
                f'{missing} (has {list(source.columns)}).'
            )
            raise DataError(msg)

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


def load_parameters(
    schema: Model,
    data: dict[str, Any] | None,
    master_coords: dict[str, pd.Index],
) -> xr.Dataset:
    """Load, coerce, and validate all declared parameters.

    Dim and coordinate checking happens here rather than per input shape:
    every branch of ``_coerce_to_dataarray`` produces a DataArray, and every
    one of them owes the same two guarantees.

    Returns:
        One DataArray per parameter, aligned to the master coordinates.

    Raises:
        DataError: A parameter missing, or dims or labels other than the ones
            declared.
    """
    data = data or {}
    arrays: dict[str, xr.DataArray] = {}

    for pname in schema.parameters:
        if pname not in data:
            msg = f"Parameter '{pname}' is required but was not provided in data.\nAdd '{pname}' to the data= argument."
            raise DataError(msg)

    declared = set(schema.parameters)
    unknown = set(data) - declared
    if unknown:
        msg = (
            f'The following data keys are not declared as parameters: '
            f'{sorted(unknown)}.\n'
            f"Declare them under 'parameters:' in the YAML or remove "
            f'them from data=.'
        )
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

    In the DataFrame branch the two-dims check guarantees flat columns, so
    ``stack()`` yields a ``Series`` — which is what the ``cast`` asserts.
    """
    if isinstance(raw, (int, float, np.integer, np.floating)):
        return xr.DataArray(float(raw))

    if isinstance(raw, dict):
        if len(dims) != 1:
            msg = f"Parameter '{name}': dict input is only supported for 1-D parameters, but declared dims are {dims}."
            raise DataError(msg)
        series = pd.Series(raw)
        series.index.name = dims[0]
        raw = series

    if isinstance(raw, pd.Series):
        if len(dims) != 1:
            msg = (
                f"Parameter '{name}': pd.Series input is only supported for "
                f'1-D parameters, but declared dims are {dims}.'
            )
            raise DataError(msg)
        if raw.index.name is None:
            raw = raw.copy()
            raw.index.name = dims[0]
        _refuse_duplicate_index(name, raw.index, dims)
        return xr.DataArray.from_series(raw)

    if isinstance(raw, pd.DataFrame):
        if len(dims) != 2:
            msg = (
                f"Parameter '{name}': pd.DataFrame input is only supported for "
                f'2-D parameters, but declared dims are {dims}.'
            )
            raise DataError(msg)
        if raw.index.name is None:
            raw = raw.copy()
            raw.index.name = dims[0]
        if raw.columns.name is None:
            raw.columns.name = dims[1]
        stacked = cast('pd.Series', raw.stack())
        stacked.name = name
        return xr.DataArray.from_series(stacked).unstack()

    if isinstance(raw, xr.DataArray):
        return raw

    if isinstance(raw, (np.ndarray, list)):
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
            return xr.DataArray(arr_np, dims=[dim], coords={dim: master_coords[dim]})
        msg = (
            f"Parameter '{name}': unsupported type ndarray.\n"
            f'For multi-dimensional arrays without named axes, provide a '
            f'pandas DataFrame or xr.DataArray with named dimensions.\n'
            f'Declared dims: {dims}.'
        )
        raise DataError(msg)

    type_name = type(raw).__name__
    msg = f"Parameter '{name}': unsupported type '{type_name}'."
    raise TypeError(msg)


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
