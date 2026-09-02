"""A ``where:`` predicate as a boolean array over the coordinates it masks.

The other half of what a declaration says: ``builder.py`` builds the thing,
this decides where it exists. A :class:`~math_spec.program.WhereNode` in, one
``xr.DataArray`` of booleans out, and :func:`as_linopy_mask` puts it in the
shape linopy's ``mask=`` takes. Both lanes read the same node kinds, and
``relational/engines/polars/predicates.py`` answers each with a polars
expression where this one answers with an array.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, assert_never

import numpy as np
import xarray as xr
from math_spec import program

from lpspec.errors import DataError, position_out_of_range_message, short_groups_message
from lpspec.linopy import absence
from lpspec.linopy.operators import _grouped

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import linopy
    import pandas as pd


#: Where-comparison operators, evaluated element-wise on a DataArray.
_PREDICATE_OPS: dict[str, Callable[[Any, Any], Any]] = {
    '==': operator.eq,
    '!=': operator.ne,
    '<': operator.lt,
    '>': operator.gt,
    '<=': operator.le,
    '>=': operator.ge,
}


@dataclass(frozen=True)
class EvaluationContext:
    """Everything evaluating a plan needs beyond the node: the data, the axes, the model, the lookups, the program.

    ``dim_coords`` carries the attached lookup columns, which a predicate on a
    lookup and a grouped operator both read instead of the parameter dataset.

    ``read_solution`` is the reader's switch: a build leaves it false and a
    variable enters an expression as its linopy term; reading a reported-grade
    named expression sets it true and a variable enters as its solved values
    (:func:`~lpspec.linopy.absence.solution`), so a body no term can hold folds
    over the primal and the duals instead.
    """

    dataset: xr.Dataset
    master_coords: Mapping[str, pd.Index]
    model: linopy.Model
    dim_coords: Mapping[str, Mapping[str, xr.DataArray]]
    program: program.Program
    read_solution: bool = False


def evaluate_where(mask: program.Mask | None, ctx: EvaluationContext) -> xr.DataArray:
    """Evaluate a lowered mask against a parameter dataset.

    Always a boolean DataArray. The no-mask case comes back 0-dimensional, so
    callers combine with ``&``/``|`` without case analysis.
    """
    if mask is None:
        return xr.DataArray(True)

    return _eval_node(mask.root, ctx)


def _eval_node(node: program.WhereNode, ctx: EvaluationContext) -> xr.DataArray:
    """One predicate node as a boolean DataArray.

    Two absences read as exclusion rather than as an answer: a variable's
    masked-out coordinate is absent (:func:`absence.present`), and a
    comparison over NaN comes back false.

    **A null lookup value is excluded explicitly rather than by ``fillna``.** A
    partial lookup arrives as an object array holding ``None``, and numpy
    answers ``None != 'north'`` with *True* rather than with null — so a ``!=``
    would keep exactly the labels that map nowhere.
    """
    dataset, master_coords = ctx.dataset, ctx.master_coords

    def evaluate(child: program.WhereNode) -> xr.DataArray:
        return _eval_node(child, ctx)

    if isinstance(node, program.BooleanLiteralNode):
        return xr.DataArray(node.value)

    if isinstance(node, program.ParameterDefinedNode):
        return _defined(dataset[node.name], ctx.program.parameter(node.name).dtype)

    if isinstance(node, program.VariableDefinedNode):
        return absence.present(ctx.model, node.name)

    if isinstance(node, (program.ParameterComparisonNode, program.DimensionComparisonNode)):
        if isinstance(node, program.ParameterComparisonNode):
            arr = dataset[node.name]
        else:
            arr = xr.DataArray(
                master_coords[node.name],
                coords={node.name: master_coords[node.name]},
                dims=[node.name],
            )

        result = _PREDICATE_OPS[node.op](arr, _as_the_axis_spells_it(arr, node.value))
        return result.fillna(False).astype(bool)

    if isinstance(node, program.DimensionPositionNode):
        labels = master_coords[node.name]
        if node.by is not None:
            arr = _group_offsets(node, bound_lookup(node.by, node.name, ctx.dim_coords), np.asarray(labels))
            return (_PREDICATE_OPS[node.op](arr, 0) & arr.notnull()).fillna(value=False).astype(bool)
        at = node.position + len(labels) if node.position < 0 else node.position
        if not 0 <= at < len(labels):
            raise DataError(position_out_of_range_message(node.name, node.op, node.position, at, len(labels)))
        arr = xr.DataArray(np.arange(len(labels)), coords={node.name: labels}, dims=[node.name])
        return _PREDICATE_OPS[node.op](arr, at).astype(bool)

    if isinstance(node, program.LookupComparisonNode):
        arr = bound_lookup(node.name, node.over, ctx.dim_coords)
        return (_PREDICATE_OPS[node.op](arr, node.value) & arr.notnull()).fillna(value=False).astype(bool)

    if isinstance(node, program.LookupPairComparisonNode):
        left = bound_lookup(node.name, node.over, ctx.dim_coords)
        right = bound_lookup(node.other, node.over, ctx.dim_coords)
        defined = left.notnull() & right.notnull()
        return (_PREDICATE_OPS[node.op](left, right) & defined).fillna(value=False).astype(bool)

    if isinstance(node, program.LookupDefinedNode):
        return bound_lookup(node.name, node.over, ctx.dim_coords).notnull()

    if isinstance(node, program.NotNode):
        return ~evaluate(node.operand)

    if isinstance(node, program.AndNode):
        return evaluate(node.left) & evaluate(node.right)

    if isinstance(node, program.OrNode):
        return evaluate(node.left) | evaluate(node.right)

    assert_never(node)


def _defined(arr: xr.DataArray, dtype: str) -> xr.DataArray:
    """What a bare parameter name in a ``where`` asks: the declaration picks the reading.

    A ``bool`` is its own answer — a slot the data has no row for is false,
    the array having widened to float to hold the NaN — a ``str`` is defined
    wherever the data has a row, and a number has to be finite as well.
    """
    if dtype == 'bool':
        return arr.fillna(False).astype(bool)
    if dtype == 'str':
        return arr.notnull()
    return arr.notnull() & np.isfinite(arr)


def _group_offsets(node: program.DimensionPositionNode, groups: xr.DataArray, labels: np.ndarray) -> xr.DataArray:
    """Each coordinate's distance from the boundary of *its own* group.

    Zero marks the coordinate the position names, so every comparator reads the
    same as it does ungrouped. ``nan`` where the lookup sends a coordinate
    nowhere: in no group, so no group's boundary.

    Raises:
        DataError: If any group is shorter than the position names, which would
            leave that group's rows unseeded and the model quietly unanchored.
    """
    partition = _grouped(node.name, labels, groups)
    needed = node.position + 1 if node.position >= 0 else -node.position
    short = sorted(str(g) for g, n in zip(partition.names, partition.counts, strict=True) if n < needed)
    if short:
        raise DataError(short_groups_message(node.name, str(node.by), node.op, node.position, short))
    target = node.position if node.position >= 0 else partition.size + node.position
    return partition.within.where(partition.grouped) - target


def unbound_lookup_message(name: str, over: str) -> str:
    """A declared lookup read with no attached map."""
    return (
        f"lookup '{name}' over dimension '{over}' has no attached values. "
        f"Pass it under key '{name}' as a table with columns ['{over}', '{name}']."
    )


def bound_lookup(name: str, over: str, dim_coords: Mapping[str, Mapping[str, xr.DataArray]]) -> xr.DataArray:
    """A lookup's attached values as an array over the dim it is over."""
    try:
        return dim_coords[over][name]
    except KeyError:
        raise DataError(unbound_lookup_message(name, over)) from None


def as_linopy_mask(mask: xr.DataArray) -> xr.DataArray | None:
    """Convert an evaluated where mask to linopy's ``mask=`` argument.

    linopy expects ``None`` for "no mask"; a 0-d True mask means exactly
    that. Everything else (including 0-d False) passes through.
    """
    if mask.ndim == 0 and bool(mask):
        return None
    return mask


def _as_the_axis_spells_it(arr: Any, value: Any) -> Any:
    """A where literal in the spelling the axis it is compared against uses.

    A quoted ISO date resolves to a ``datetime.date`` (the where rules), and a
    temporal axis arrives as ``datetime64`` — numpy compares the two by
    raising, so the axis decides, a literal carrying no dtype of its own.
    """
    if getattr(arr, 'dtype', None) is not None and arr.dtype.kind == 'M':
        return np.datetime64(value)
    return value
