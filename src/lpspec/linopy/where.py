"""A ``where:`` predicate as a boolean array over the coordinates it masks.

The other half of what a declaration says: ``builder.py`` builds the thing,
this decides where it exists. A :class:`~math_spec.where_parser.WhereNode` in, one
``xr.DataArray`` of booleans out, and :func:`as_linopy_mask` puts it in the
shape linopy's ``mask=`` takes.

The plan is the precondition, and it is a stronger one than a resolved AST was:
a predicate node cannot be unresolved, so there is no branch here for a bare
name that never went through ``where_of``. Both lanes read the same twelve
node kinds, and ``relational/engines/polars/predicates.py`` answers each with a
polars expression where this one answers with an array.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from math_spec import where_parser

from lpspec.errors import DataError, LanguageError, unbound_lookup_message
from lpspec.linopy import absence

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
class WhereContext:
    """What a resolved predicate reads: the data, the axes, the model's masks, the bound lookups.

    ``dim_coords`` carries the bound lookup columns, which a predicate on a
    lookup reads instead of the parameter dataset — the same store the grouped
    sum reads its mapping from. The builder's ``EvaluationContext`` extends
    this with what evaluating a whole expression needs.
    """

    dataset: xr.Dataset
    master_coords: Mapping[str, pd.Index]
    model: linopy.Model | None = None
    dim_coords: Mapping[str, Mapping[str, xr.DataArray]] = field(default_factory=dict)


def evaluate_where(node: where_parser.WhereNode | None, ctx: WhereContext) -> xr.DataArray:
    """Evaluate a lowered predicate against a parameter dataset.

    A plan node, not a string and not an AST: lowering has already decided what
    every name refers to and which kind of atom it is, so this performs no
    lookups and cannot disagree with the relational lane about scoping — the
    two read the same node.

    Always a boolean DataArray. The no-mask case comes back 0-dimensional, so
    callers combine with ``&``/``|`` without case analysis.
    """
    if node is None:
        return xr.DataArray(True)

    return _eval_node(node, ctx)


def _eval_node(node: where_parser.WhereNode, ctx: WhereContext) -> xr.DataArray:
    """One predicate node as a boolean DataArray.

    Two absences read as exclusion rather than as an answer: a variable's
    masked-out coordinate is absent (:func:`absence.present`), and a
    comparison over NaN comes back false.

    **A null lookup value is excluded explicitly rather than by ``fillna``.** A
    partial lookup arrives as an object array holding ``None``, and numpy
    answers ``None != 'north'`` with *True* rather than with null — so a ``!=``
    would keep exactly the labels that map nowhere, which is the reading law 8
    forbids and the relational lane does not give.
    """
    dataset, master_coords = ctx.dataset, ctx.master_coords

    def evaluate(child: where_parser.WhereNode) -> xr.DataArray:
        return _eval_node(child, ctx)

    if isinstance(node, where_parser.BooleanLiteralNode):
        return xr.DataArray(node.value)

    if isinstance(node, where_parser.ParameterDefinedNode):
        arr = dataset[node.name]
        if arr.dtype == bool:
            return arr
        if arr.dtype.kind in 'OUS':
            return arr.notnull()
        return arr.notnull() & np.isfinite(arr)

    if isinstance(node, where_parser.VariableDefinedNode):
        if ctx.model is None:
            msg = (
                f"where references variable '{node.name}', but no model was passed to the "
                f'evaluator — a variable mask can only be read off the model that holds it.'
            )
            raise AssertionError(msg)
        return absence.present(ctx.model, node.name)

    if isinstance(node, (where_parser.ParameterComparisonNode, where_parser.DimensionComparisonNode)):
        if isinstance(node, where_parser.ParameterComparisonNode):
            arr = dataset[node.name]
        else:
            arr = xr.DataArray(
                master_coords[node.name],
                coords={node.name: master_coords[node.name]},
                dims=[node.name],
            )

        result = _PREDICATE_OPS[node.op](arr, _as_the_axis_spells_it(arr, node.value))
        return result.fillna(False).astype(bool)

    if isinstance(node, where_parser.DimensionPositionNode):
        labels = master_coords[node.name]
        if node.by is not None:
            groups = _bound_lookup(node.by, node.name, ctx.dim_coords)
            offsets = _group_offsets(node, groups.values)
            arr = xr.DataArray(offsets, coords={node.name: labels}, dims=[node.name])
            return (_PREDICATE_OPS[node.op](arr, 0) & arr.notnull()).fillna(value=False).astype(bool)
        at = node.position + len(labels) if node.position < 0 else node.position
        if not 0 <= at < len(labels):
            msg = (
                f'where: position({node.name}) {node.op} {node.position} names position {at} of '
                f"'{node.name}', which has {len(labels)} coordinate(s). A boundary that "
                f'names no coordinate leaves the rows it was to seed unseeded.'
            )
            raise DataError(msg)
        arr = xr.DataArray(np.arange(len(labels)), coords={node.name: labels}, dims=[node.name])
        return _PREDICATE_OPS[node.op](arr, at).astype(bool)

    if isinstance(node, where_parser.LookupComparisonNode):
        arr = _bound_lookup(node.name, node.over, ctx.dim_coords)
        return (_PREDICATE_OPS[node.op](arr, node.value) & arr.notnull()).fillna(value=False).astype(bool)

    if isinstance(node, where_parser.LookupPairComparisonNode):
        left = _bound_lookup(node.name, node.over, ctx.dim_coords)
        right = _bound_lookup(node.other, node.over, ctx.dim_coords)
        defined = left.notnull() & right.notnull()
        return (_PREDICATE_OPS[node.op](left, right) & defined).fillna(value=False).astype(bool)

    if isinstance(node, where_parser.LookupDefinedNode):
        return _bound_lookup(node.name, node.over, ctx.dim_coords).notnull()

    if isinstance(node, where_parser.NotNode):
        return ~evaluate(node.operand)

    if isinstance(node, where_parser.AndNode):
        return evaluate(node.left) & evaluate(node.right)

    if isinstance(node, where_parser.OrNode):
        return evaluate(node.left) | evaluate(node.right)

    raise LanguageError(f'unsupported predicate node {type(node).__name__}')


def _group_offsets(node: where_parser.DimensionPositionNode, groups: np.ndarray) -> np.ndarray:
    """Each coordinate's distance from the boundary of *its own* group.

    Zero marks the coordinate the position names, so every comparator reads the
    same as it does ungrouped. ``nan`` where the lookup sends a coordinate
    nowhere: in no group, so no group's boundary. The relational lane computes
    the identical column with a rank over the dim table.

    Raises:
        DataError: If any group is shorter than the position names, which would
            leave that group's rows unseeded and the model quietly unanchored.
    """
    counts: dict[object, int] = {}
    within = np.empty(len(groups), dtype=float)
    for k, g in enumerate(groups):
        if absence.unmapped(g):
            within[k] = np.nan
            continue
        within[k] = counts.get(g, 0)
        counts[g] = int(within[k]) + 1
    needed = node.position + 1 if node.position >= 0 else -node.position
    short = sorted(str(g) for g, n in counts.items() if n < needed)
    if short:
        msg = (
            f'where: position({node.name}, by={node.by}) {node.op} {node.position} names position '
            f'{node.position} within each group, and {len(short)} of them are shorter than '
            f'that: {short[:5]}. A boundary that names no coordinate leaves the rows it '
            f'was to seed unseeded.'
        )
        raise DataError(msg)
    sizes = np.array([0 if absence.unmapped(g) else counts.get(g, 0) for g in groups], dtype=float)
    target = node.position if node.position >= 0 else sizes + node.position
    return within - target


def _bound_lookup(
    name: str,
    over: str,
    dim_coords: Mapping[str, Mapping[str, xr.DataArray]],
) -> xr.DataArray:
    """A lookup's bound values as an array over the dim it is over.

    The where counterpart of :func:`_lookup_arrays`, which reads the same
    store for a grouped sum — and the same sentence, led by who was reading.
    """
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
