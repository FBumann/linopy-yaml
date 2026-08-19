"""Model builder: schema + data → linopy Model.

Also the eager evaluation of every built-in operator. The operator *names* are the
language (``operators.py``, imported by the linopy-free lane); these
xarray/linopy evaluations are this backend's private business, mirrored on the
relational side by lowering cases rather than shared code.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from functools import reduce
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, assert_never

import numpy as np
import xarray as xr

from lpspec._notes import note
from lpspec.errors import DataError, LaneError, LanguageError, null_bounds_message
from lpspec.language import degree
from lpspec.language.expression_parser import (
    ArithmeticNode,
    BinaryOperatorNode,
    ComparisonNode,
    DimensionNode,
    EdgeNode,
    FunctionCallNode,
    KeywordNode,
    LookupNode,
    NameNode,
    NumberNode,
    ParameterNode,
    UnaryOperatorNode,
    VariableNode,
)
from lpspec.language.operators import EDGE_WRAP, unknown_operator_message
from lpspec.language.resolution import Namespace, expression_of, where_of
from lpspec.language.where_parser import (
    AndNode,
    BooleanLiteralNode,
    DimensionComparisonNode,
    DimensionPositionNode,
    LookupComparisonNode,
    LookupDefinedNode,
    LookupPairComparisonNode,
    NotNode,
    OrNode,
    ParameterComparisonNode,
    ParameterDefinedNode,
    UnresolvedComparisonNode,
    UnresolvedNameNode,
    VariableDefinedNode,
    WhereNode,
    _UnresolvedPositionNode,
)
from lpspec.linopy.loader import check_constant_side_covers, check_divisors_cover, gaps_under

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping

    import linopy
    import pandas as pd

    from lpspec.language.model import Model

_SIGN_MAP = {'==': '=', '<=': '<=', '>=': '>='}

#: The language's arithmetic. ``**`` is absent on purpose — see ``_eval_ast``.
_ARITHMETIC_OPS: dict[str, Callable[[Any, Any], Any]] = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
}

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
    """Everything expression evaluation needs to resolve names.

    Extend this rather than adding parameters to ``_eval_ast`` and every
    operator-facing seam.
    """

    model: linopy.Model
    dataset: xr.Dataset
    master_coords: dict[str, pd.Index]
    schema: Model
    ns: Namespace
    #: dim -> {coordinate name: values as a DataArray over that dim}
    dim_coords: dict[str, dict[str, xr.DataArray]] = field(default_factory=dict)


def build_model(
    model: linopy.Model,
    schema: Model,
    dataset: xr.Dataset,
    master_coords: dict[str, pd.Index],
    dim_coords: dict[str, dict[str, xr.DataArray]] | None = None,
) -> None:
    """Populate a linopy Model from a parsed schema and loaded parameters.

    This mutates *model* in-place, adding variables, constraints and
    the objective as declared in *schema*.
    """
    ctx = EvaluationContext(
        model,
        dataset,
        master_coords,
        schema,
        Namespace.of(schema),
        dim_coords or {},
    )
    _build_variables(ctx)
    _build_sos(ctx)
    _build_constraints(ctx)
    _build_objective(ctx)


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------


def _build_variables(ctx: EvaluationContext) -> None:
    for vname, vdef in ctx.schema.variables.items():
        with note(f"while building variable '{vname}'"):
            coords = {d: ctx.master_coords[d] for d in vdef.foreach}

            lower = _resolve_bound(vdef.bounds.lower, ctx.dataset)
            upper = _resolve_bound(vdef.bounds.upper, ctx.dataset)

            where = where_of(vdef.where, ctx.ns, f"variable '{vname}'", self_variable=vname)
            mask = evaluate_where(where, ctx.dataset, ctx.master_coords, ctx.model, ctx.dim_coords)

            _check_bounds_are_defined(vname, vdef, ctx.dataset, mask)

            ctx.model.add_variables(
                lower=lower,
                upper=upper,
                coords=coords,
                name=vname,
                mask=_as_linopy_mask(mask),
                binary=vdef.domain == 'binary',
                integer=vdef.domain == 'integer',
            )


def _check_bounds_are_defined(name: str, vdef: Any, dataset: xr.Dataset, mask: Any) -> None:
    """Refuse a bound with no value, at build, as the native lane does.

    Otherwise the NaN travels into linopy and surfaces two phases later from
    inside its IO layer — ``Continuous Variable x contains nan's in field(s)
    ['upper']``, raised at solve or write, naming neither the YAML nor the fix,
    from a ``build()`` that had already returned.

    Checked against the variable's own mask: a coordinate the variable does not
    occupy needs no bound, and supplying data only where it exists is the
    ordinary idiom.
    """
    missing = sum(
        gaps_under(dataset[bound], mask) for bound in (vdef.bounds.lower, vdef.bounds.upper) if isinstance(bound, str)
    )
    if missing:
        raise DataError(null_bounds_message(name, missing))


def _resolve_bound(
    value: float | str,
    dataset: xr.Dataset,
) -> Any:
    """Resolve a bound value — either a literal number or a parameter name."""
    if isinstance(value, str):
        if value not in dataset:
            msg = (
                f"Bound references parameter '{value}' which is not in the "
                f'loaded dataset. Available: {sorted(map(str, dataset.data_vars))}'
            )
            raise DataError(msg)
        return dataset[value]
    return value


def _variable_term(name: str, ctx: EvaluationContext) -> Any:
    """The variable as it enters an expression, carrying its declared ``absence:``.

    The mask stays on the variable either way — it is what keeps the absent
    coordinates out of the model, and dropping it to pin them at zero instead
    would hand the solver a column per absent coordinate that the relational
    lane never emits.

    What differs is the *arithmetic*. This lane sets
    ``linopy.options['semantics'] = 'v1'`` on import (see the module docstring),
    under which an absent slot propagates and takes its row — the default, and
    ``absence: undefined``. ``fillna(0)`` is linopy's own per-expression escape
    back to the other reading: the slot contributes nothing and the row stands.
    Per use rather than per model, which is the granularity a declaration needs
    and the reason the global option alone could not express this.
    """
    variable = ctx.model.variables[name]
    return variable.fillna(0) if ctx.schema.variables[name].absence == 'zero' else variable


def _as_linopy_mask(mask: xr.DataArray) -> xr.DataArray | None:
    """Convert an evaluated where mask to linopy's ``mask=`` argument.

    linopy expects ``None`` for "no mask"; a 0-d True mask means exactly
    that. Everything else (including 0-d False) passes through.
    """
    if mask.ndim == 0 and bool(mask):
        return None
    return mask


# ---------------------------------------------------------------------------
# Special-ordered sets
# ---------------------------------------------------------------------------


def _build_sos(ctx: EvaluationContext) -> None:
    """Attach every ``sos:`` block to the variable it names.

    linopy holds a set the same way the language declares one — a variable, a
    dimension of it, a type — so this is the block handed over, not a
    formulation rebuilt. Which is the point of copying its decomposition:
    the eager lane is the oracle, and a set it had to *reformulate* to accept
    would be an oracle for a different model.

    It runs before the constraints because a set is a property of the
    variable, so it belongs beside the declaration rather than after
    everything that uses it.
    """
    for name, sos in ctx.schema.sos.items():
        with note(f"while building sos '{name}'"):
            ctx.model.add_sos_constraints(
                ctx.model.variables[sos.variable],
                sos_type=1 if sos.type == 1 else 2,
                sos_dim=sos.over,
                big_m=sos.big_m,
            )


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def _build_constraints(ctx: EvaluationContext) -> None:
    for cname, cdef in ctx.schema.constraints.items():
        with note(f"while building constraint '{cname}'"):
            c_where = where_of(cdef.where, ctx.ns, f"constraint '{cname}'")
            mask = evaluate_where(c_where, ctx.dataset, ctx.master_coords, ctx.model, ctx.dim_coords)

            ast = expression_of(cdef.expression, ctx.schema, ctx.ns, f"constraint '{cname}'")
            if not isinstance(ast, ComparisonNode):
                msg = f'expression must contain exactly one comparison operator (<=, >=, ==).\nGot: {cdef.expression!r}'
                raise LanguageError(msg)

            check_divisors_cover(f"constraint '{cname}'", ast, ctx.schema, ctx.dataset, mask, ctx.model)
            check_constant_side_covers(f"constraint '{cname}'", ast, ctx.schema, ctx.dataset, mask)

            lhs = _eval_ast(ast.left, ctx)
            rhs = _eval_ast(ast.right, ctx)
            sign = _SIGN_MAP[ast.op]

            ctx.model.add_constraints(lhs, sign, rhs, name=cname, mask=_as_linopy_mask(mask))


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------


def _build_objective(ctx: EvaluationContext) -> None:
    """Build the declared objective, if any, onto the model.

    An objective has no ``where``, so its divisor check runs with no row mask
    — the numerator's own presence is the only thing that can excuse a gap.

    The expression is scalar by the time it gets here, ``check_schema`` having
    refused one carrying dims. That is what lets it be evaluated like any
    other: an eager ``+`` broadcasts its operands, so a lane that added the
    terms of ``x[i] * a[i] + y[j] * b[j]`` and summed afterwards would count
    each once per coordinate of the other (#197) — and now nothing has to,
    because the file said where each sum ends.
    """
    odef = ctx.schema.objective
    if odef is None:
        return
    with note('while building the objective'):
        ast = expression_of(odef.expression, ctx.schema, ctx.ns, 'the objective')

        if isinstance(ast, ComparisonNode):
            msg = f'Expression must not contain a comparison operator. Got: {odef.expression!r}'
            raise LanguageError(msg)

        check_divisors_cover('the objective', ast, ctx.schema, ctx.dataset, None, ctx.model)

        expr = _eval_ast(ast, ctx)
        _refuse_an_objective_constant(expr)

        sense = 'min' if odef.sense == 'minimize' else 'max'
        ctx.model.add_objective(expr, overwrite=True, sense=sense)


# ---------------------------------------------------------------------------
# AST evaluation
# ---------------------------------------------------------------------------


def _eval_ast(
    node: ArithmeticNode,
    ctx: EvaluationContext,
) -> Any:
    """Evaluate an expression AST node against the model namespace.

    Binary nodes go through ``degree.check_binary`` first: ``**``, a quadratic
    product and a variable divisor are all refused by ``language/degree.py``,
    the same verdict the relational lane asks for and in the same sentence.

    Unknown operator names were rejected by ``validation.py`` at load time; the
    guard on ``_OPERATORS`` covers hand-built calls that skipped it.
    """
    if isinstance(node, NumberNode):
        return node.value

    if isinstance(node, VariableNode):
        return _variable_term(node.name, ctx)

    if isinstance(node, ParameterNode):
        return _coefficient(ctx.dataset[node.name])

    if isinstance(node, EdgeNode):
        msg = f'EdgeNode({node.policy!r}) reached the evaluator: an edge policy is a shift() kwarg, not a value.'
        raise AssertionError(msg)

    if isinstance(node, KeywordNode):
        msg = (
            f'KeywordNode({node.value!r}) reached the evaluator. A quoted keyword is '
            f'consumed by its kwarg during resolution — reaching here means it was written '
            f'where no kwarg expects one.'
        )
        raise AssertionError(msg)
    if isinstance(node, (NameNode, DimensionNode, LookupNode)):
        msg = (
            f'{type(node).__name__}({node.name!r}) reached the evaluator. '
            f'Expressions must go through resolution.expression_of() first '
            f'(docs/about/architecture.md hard rule 1).'
        )
        raise AssertionError(msg)

    if isinstance(node, UnaryOperatorNode):
        operand = _eval_ast(node.operand, ctx)
        if node.op == '-':
            return -operand
        return operand

    if isinstance(node, BinaryOperatorNode):
        degree.check_binary(node)
        left = _eval_ast(node.left, ctx)
        right = _eval_ast(node.right, ctx)
        return _ARITHMETIC_OPS[node.op](left, right)

    if isinstance(node, FunctionCallNode):
        if node.name not in _OPERATORS:
            raise NameError(unknown_operator_message(node.name))
        operator = _OPERATORS[node.name]
        args = [_eval_ast(a, ctx) for a in node.args]
        if node.name == 'at':
            by = node.kwargs['by']
            assert isinstance(by, LookupNode)
            return _operator_at(args[0], _lookup_array(by, ctx), into=by.into)
        if node.name == 'sum' and (by := node.kwargs.get('by')) is not None:
            assert isinstance(by, LookupNode)
            return _operator_grouped_sum(
                args[0], _lookup_array(by, ctx), into=by.into, labels=ctx.master_coords[by.into]
            )
        kwargs: dict[str, Any] = {}
        for k, v in node.kwargs.items():
            if isinstance(v, DimensionNode):
                kwargs[k] = v.name
            elif isinstance(v, EdgeNode):
                kwargs[k] = v.policy
            elif isinstance(v, LookupNode):
                kwargs[k] = _lookup_array(v, ctx)
            else:
                kwargs[k] = _eval_ast(v, ctx)
        return operator(*args, **kwargs)

    assert_never(node)


def _lookup_array(by: LookupNode, ctx: EvaluationContext) -> Any:
    """The declared lookup ``by`` as an array over the dimension it is over.

    Looked up rather than evaluated as an operand: the lookup lives on the
    dimension, not in the parameter dataset.
    """
    try:
        return ctx.dim_coords[by.dimension][by.name]
    except KeyError:
        msg = (
            f"lookup '{by.name}' over dimension '{by.dimension}' has no bound values. "
            f"Pass sources={{'{by.dimension}': <table with '{by.dimension}' and "
            f"'{by.name}' columns>}}."
        )
        raise DataError(msg) from None


# ---------------------------------------------------------------------------
# Built-in operators, eager evaluation — each operand is an xr.DataArray (a
# parameter) or a linopy Variable / LinearExpression
# ---------------------------------------------------------------------------


def _operator_sum(array: Any, *, over: str | None = None) -> Any:
    """Sum *array* over dimension *over*, or over all of them where none is named.

    A DataArray and a linopy expression both carry ``dims`` and both take the
    dim positionally, so there is one branch: if the array does not have the
    named dimension, it is returned unchanged.
    """
    if over is None:
        return array.sum()
    if over in getattr(array, 'dims', ()):
        return array.sum(over)
    return array


def _operator_grouped_sum(array: Any, mapping: Any, *, into: str, labels: pd.Index) -> Any:
    """Sum *array* through a declared lookup, producing dimension *into*.

    YAML: ``sum(p, by=gen_bus)``. *mapping* is the lookup's values as a
    one-dimensional array over the dim being grouped, from
    ``EvaluationContext.dim_coords``; that dim is summed out and *into*
    holds the group labels.

    A null lookup value says the label belongs to no group, so its terms
    contribute nowhere. linopy refuses to group by NaN at all, so those members
    are dropped before grouping rather than after.

    *labels* is ``into``'s declared index, and the result is reindexed onto it:
    a groupby yields only the labels some member actually points at, in xarray's
    sort order, so without this a label no member reaches is missing and a
    declared order that is not sorted is lost. Either one makes linopy v1 refuse
    the next combination with a coordinate mismatch, since it aligns on
    membership *and* order. Lookup values are validated against ``into``'s
    labels when they are loaded, so this only ever adds a label, never drops a
    term.
    """
    if not isinstance(mapping, xr.DataArray):
        msg = f'sum(by=) lookup must be an array (got {type(mapping).__name__}). Usage: sum(expr, by=lookup)'
        raise TypeError(msg)
    if mapping.ndim != 1:
        msg = f'sum(by=) mapping must have exactly one dimension, got {list(mapping.dims)}'
        raise LanguageError(msg)

    group = mapping.rename(into)
    present = group.notnull()
    if not bool(present.all()):
        dim = str(group.dims[0])
        group = group.isel({dim: present.to_numpy()})
        array = array.isel({dim: present.to_numpy()})
    if isinstance(array, xr.DataArray) or hasattr(array, 'groupby'):
        return _reindexed(array.groupby(group).sum(), into=into, labels=labels)
    raise _unsupported('sum(by=)', array)


def _reindexed(summed: Any, *, into: str, labels: pd.Index) -> Any:
    """*summed* over exactly *labels*, empty groups filled with an empty sum.

    A grouped parameter is a plain ``DataArray``, where the empty sum is 0. A
    grouped expression is a ``LinearExpression``, whose empty term is spelled
    per-variable — linopy's own ``_fill_value`` cannot be used, its ``const:
    nan`` propagates through the arithmetic that follows and poisons the row.
    """
    fill = {'vars': -1, 'coeffs': 0.0, 'const': 0.0} if hasattr(summed, 'const') else 0
    return summed.reindex({into: labels}, fill_value=fill)


def _operator_at(array: Any, mapping: Any, *, into: str) -> Any:
    """Read *array* through a declared lookup — the adjoint of a group.

    YAML: ``at(on, by=component)``. *mapping* is the same one-dimensional
    array ``sum`` takes; grouping sums *along* it, this indexes *through* it,
    so the operand must carry ``into`` and the result carries the mapping's
    own dim.

    xarray's vectorised selection is the pullback exactly — one ``into`` label
    read once per fine label pointing at it — so the fan-out is the indexer's
    doing rather than a broadcast arranged here.

    A null lookup value reads nothing and its row is absent, the same reading
    ``sum`` gives a null group. It cannot be selected — there is no ``into``
    label to read — so it is dropped from the indexer and the result is put
    back over the whole dim, the missing positions filled with the operand's
    own **absence** rather than a zero — which is what ``reindex`` does untold,
    on all three operand shapes. Absence is what the absence rules ask for
    here: it propagates and takes the row with it, the same mechanism the
    default ``shift`` edge relies on, where a zero would leave a row asserting
    ``x <= 0`` at a coordinate the model said nothing about. Putting the dim
    back is also what keeps the operand combinable at all — linopy v1 aligns on
    membership, so a result short of a label refuses the next arithmetic
    outright, which is how #897 surfaced.
    """
    if not isinstance(mapping, xr.DataArray):
        msg = f'at() lookup must be an array (got {type(mapping).__name__}). Usage: at(expr, by=lookup)'
        raise TypeError(msg)
    if mapping.ndim != 1:
        msg = f'at() mapping must have exactly one dimension, got {list(mapping.dims)}'
        raise LanguageError(msg)
    if not (isinstance(array, xr.DataArray) or hasattr(array, 'sel')):
        raise _unsupported('at()', array)

    present = mapping.notnull()
    if bool(present.all()):
        return array.sel({into: mapping.rename(into)})

    dim = str(mapping.dims[0])
    picked = array.sel({into: mapping.isel({dim: present.to_numpy()}).rename(into)})
    return picked.reindex({dim: mapping[dim]})


def _group_offsets(node: DimensionPositionNode, groups: np.ndarray) -> np.ndarray:
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
        if g is None or g != g:  # nan: the null a partial lookup leaves, and never equal to itself
            within[k] = np.nan
            continue
        within[k] = counts.get(g, 0)
        counts[g] = int(within[k]) + 1
    needed = node.position + 1 if node.position >= 0 else -node.position
    short = sorted(str(g) for g, n in counts.items() if n < needed)
    if short:
        msg = (
            f'where: index({node.name}, {node.position}, by={node.by}) names position '
            f'{node.position} within each group, and {len(short)} of them are shorter than '
            f'that: {short[:5]}. A boundary that names no coordinate leaves the rows it '
            f'was to seed unseeded.'
        )
        raise DataError(msg)
    sizes = np.array([counts.get(g, 0) if not (g is None or g != g) else 0 for g in groups], dtype=float)
    target = node.position if node.position >= 0 else sizes + node.position
    return within - target


def _bound_lookup(
    name: str,
    over: str,
    dim_coords: Mapping[str, Mapping[str, xr.DataArray]],
) -> xr.DataArray:
    """A lookup's bound values as an array over the dim it is over.

    The where counterpart of :func:`_lookup_array`, which reads the same store
    for a grouped sum. Kept separate because the failure differs: a predicate
    can be evaluated before any variable exists, so the message names the
    source that was wanted rather than the helper call that wanted it.
    """
    try:
        return dim_coords[over][name]
    except KeyError:
        msg = (
            f"where reads lookup '{name}' over dimension '{over}', which has no bound "
            f"values. Pass sources={{'{over}': <table with '{over}' and '{name}' columns>}}."
        )
        raise DataError(msg) from None


#: The one construct this lane accepts and cannot build, as the sentence a user
#: reads. Module-level so the test that pins the wording has something to name.
OBJECTIVE_CONSTANT_IS_A_LANE_GAP = (
    "the objective carries a constant term, and this lane cannot build one: linopy's objective "
    'takes no constant — there is no `objective_constant` slot anywhere in the package, which is '
    'why PyPSA carries its own out of band. The relational lane builds it and returns the right '
    'number, so the model is sayable and only this lane is short: run it with `lpspec.solve` / '
    '`lpspec.build`. Dropping the constant here is refused deliberately — it would answer a '
    'different model, and this lane is the differential oracle, so every test on such a file '
    "would be calibrated to the shortened objective's number (#894)."
)


def _refuse_an_objective_constant(expr: Any) -> None:
    """Refuse an objective this lane cannot build, before linopy is asked.

    linopy's own refusal names neither the file nor the other lane, and arrives
    from a setter two frames down. A check rather than a `try`, because the
    upstream message is not a contract and a nonzero constant is the whole of
    what it means.
    """
    const = getattr(expr, 'const', None)
    if const is not None and bool(np.any(np.asarray(const) != 0)):
        raise LaneError(OBJECTIVE_CONSTANT_IS_A_LANE_GAP)


def _unsupported(call: str, array: Any) -> TypeError:
    """One wording for an operand shape an operator cannot take.

    Reached only from a hand-built call: every operator's operands come from
    ``_eval_ast``, so a lane running the language proper never sees this.
    """
    return TypeError(f"{call} does not support type '{type(array).__name__}'.")


def _translation(over: str, by: float) -> Mapping[Hashable, int]:
    """The ``{dim: n}`` mapping xarray and linopy both take."""
    if int(by) != by:
        msg = f'shift() by must be an integer, got {by!r}'
        raise TypeError(msg)
    return {over: int(by)}


def _gather_by_offset(array: Any, over: str, offset: Any, *, wrap: bool, fill: float | None, card: int) -> Any:
    """Translate *array* along *over* by an offset that differs per entity.

    A scalar shift is one call; a per-entity one is a **gather**: every output
    position reads a source position of its own, so the index is an array over
    the offset's dims and *over* rather than a number.

    Selection is by *label* rather than by ordinal, because that is what linopy
    passes through to its own labels — the positions are turned back into
    coordinate values here, which also keeps a non-integer axis (a datetime
    snapshot) working for free.

    Out-of-range positions are clipped so the gather stays on the axis, then
    emptied again by ``where``, so an edge means the same thing it does for a
    scalar shift: absent by default, and :func:`_vacated` fills it where the
    model asked. Under ``wrap`` nothing is out of range and the modulo is the
    whole of it.
    """
    labels = np.asarray(array.indexes[over])
    ordinal = xr.DataArray(np.arange(card), coords={over: labels}, dims=[over])
    source = (ordinal - offset).astype(int)

    def gathered(ordinals: Any) -> Any:
        # The indexer carries no coordinate, so the result comes back with the
        # axis unlabelled; the output position *t* still means "at t", so the
        # original labels go back on before anything is combined with it.
        picked = array.sel({over: _labelled(labels, ordinals, over)})
        return picked.assign_coords({over: labels})

    if wrap:
        return gathered(source % card)
    inside = ((source >= 0) & (source < card)).assign_coords({over: labels})
    moved = gathered(source.clip(0, card - 1)).where(inside)
    return moved if fill is None else _vacated(moved, array, over, ~inside, fill)


def _gather_in_groups(array: Any, over: str, offset: Any, *, groups: Any, wrap: bool, fill: float | None) -> Any:
    """Translate *array* inside each group *groups* makes, not along the axis.

    The neighbour of a coordinate is the one *offset* back among the coordinates
    sharing its group, so the gather is by a source ordinal computed per group:
    a position past the group's start is vacated where the axis edge would
    vacate, and under *wrap* it comes round to that group's own last.

    A coordinate the lookup sends nowhere belongs to no group, so it reaches
    nothing — the null reading a partial lookup gets everywhere else. That is
    not the same as reaching *off* a group's edge, which is what a policy
    speaks for, so the two are tracked apart and only the second is filled
    (#1061).

    The relational lane computes the identical map as a rank over the dim
    table joined back on ``(group, position)``.
    """
    labels = np.asarray(array.indexes[over])
    keys = np.asarray(groups.sel({over: labels}).values, dtype=object)
    step = int(offset)

    positions: dict[object, list[int]] = {}
    grouped = np.zeros(len(labels), dtype=bool)
    for k, key in enumerate(keys):
        if key is None or key != key:  # nan: what a partial lookup leaves
            continue
        grouped[k] = True
        positions.setdefault(key, []).append(k)

    source = np.full(len(labels), -1, dtype=int)
    for members in positions.values():
        size = len(members)
        for within, k in enumerate(members):
            reached = (within - step) % size if wrap else within - step
            if 0 <= reached < size:
                source[k] = members[reached]

    inside = xr.DataArray(source >= 0, coords={over: labels}, dims=[over])
    indexer = xr.DataArray(labels[np.clip(source, 0, len(labels) - 1)], dims=[over])
    gathered = array.sel({over: indexer}).assign_coords({over: labels}).where(inside)
    if fill is None:
        return gathered
    vacated = xr.DataArray((source < 0) & grouped, coords={over: labels}, dims=[over])
    return _vacated(gathered, array, over, vacated, fill)


def _off_the_axis(array: Any, over: str, offset: float) -> Any:
    """Which positions along *over* a scalar shift of *offset* leaves vacated.

    The source a position reads, off both ends, so one expression covers a
    shift in either direction — the same verdict :func:`_gather_by_offset`
    reaches with ``inside`` negated.
    """
    labels = np.asarray(array.indexes[over])
    source = xr.DataArray(np.arange(len(labels)), coords={over: labels}, dims=[over]) - int(offset)
    return (source < 0) | (source >= len(labels))


def _labelled(labels: Any, ordinals: Any, over: str) -> Any:
    """*ordinals* as the coordinate labels they stand for, keeping their dims.

    Carries no coordinates of its own: an indexer that keeps the axis's own
    coordinate asserts the values it holds *are* that axis, and after a gather
    they are not — which xarray reports as a size conflict rather than a
    mislabelling.
    """
    return xr.DataArray(labels[ordinals.transpose(*ordinals.dims).values], dims=ordinals.dims)


def _operator_sum_back(array: Any, *, over: str, within: Any, edge: str | None = None) -> Any:
    """Sum *array* over a trailing window along one dimension.

    YAML: ``sum_back(started, over=snapshot, within=min_up)``. The result at
    *t* is the sum from *t - within + 1* through *t*, so a width of 1 is the
    operand itself and ``edge='wrap'`` lets the window reach around the axis.

    Written as a sum of scalar gathers, one per position of the widest window
    the data asks for. That bound is read from data, which is only sound
    because it decides how many *terms* are added rather than what the plan
    does — the same reading under which cardinality is data's.

    A position the window cannot reach contributes a **zero**, never an
    absence. linopy counts absence among the things that propagate, so an
    unreachable lag added to a reachable one would annihilate the whole row
    (v1 §4) — which is right for a shift, whose vacated slot really is
    unknown, and wrong here: a window at the first position is short, not
    empty. That covers a masked slot the window reaches too: absence is not a
    term, and a reduction is where absence stops (the absence reference).

    Which leaves the window that reaches **nothing** — every position it spans
    masked away, the whole of it where the width is 1. A zero there would build
    a row about constants alone, so the fill is paired with the positions any
    lag actually reached, and a window that reached none of them keeps no row
    (#1059, #1060).
    """
    card = int(array.sizes[over])
    widest = int(np.max(np.asarray(within))) if isinstance(within, xr.DataArray) else int(within)
    widest = min(widest, card)
    terms: list[Any] = []
    reached: list[Any] = []
    for lag in range(widest):
        lagged = _gather_by_offset(array, over, lag, wrap=edge == EDGE_WRAP, fill=None, card=card)
        live, term = ~lagged.isnull(), _filled(lagged, 0.0)
        if isinstance(within, xr.DataArray):
            live, term = live & (within > lag), term * (within > lag).astype(float)
        terms.append(term)
        reached.append(live)
    return reduce(operator.add, terms).where(reduce(operator.or_, reached))


def _operator_shift(array: Any, *, over: str, offset: float, edge: str | float | None = None, by: Any = None) -> Any:
    """Translate *array* along one dimension — the value at *t - offset*.

    YAML: ``shift(soc, over=snapshot, offset=1)``. ``edge`` carries all three
    policies so no two keywords can disagree: ``edge='wrap'`` is cyclic and
    vacates nothing, a number is what the vacated positions contribute, and
    omitting it leaves them **absent**, which propagates and drops the row.
    Nothing is done to the result in that default case — linopy v1 already
    gives that answer (#289).

    A DataArray shift always fills, absence not being representable in data, so
    lowering refuses a bare shift over a variable-free operand and that branch
    is only reached under a numeric ``edge=``.

    ``offset`` arrives as an array where the model named a parameter — an offset
    that differs per entity, which is a gather rather than a shift and is
    :func:`_gather_by_offset`.
    """
    if by is not None:
        return _gather_in_groups(
            array,
            over,
            offset,
            groups=by,
            wrap=edge == EDGE_WRAP,
            fill=None if isinstance(edge, str) else edge,
        )
    if isinstance(offset, xr.DataArray) and offset.ndim:
        return _gather_by_offset(
            array,
            over,
            offset,
            wrap=edge == EDGE_WRAP,
            fill=None if isinstance(edge, str) else edge,
            card=int(array.sizes[over]),
        )
    amount = _translation(over, offset)
    if edge == EDGE_WRAP:
        if isinstance(array, xr.DataArray):
            return array.roll(amount, roll_coords=False)
        if hasattr(array, 'roll'):
            return array.roll(amount)
        raise _unsupported("shift(edge='wrap')", array)
    if isinstance(edge, str):
        msg = f'shift(edge={edge!r}) reached the evaluator: only {EDGE_WRAP!r} or a number resolve.'
        raise AssertionError(msg)
    fill = edge
    if isinstance(array, xr.DataArray):
        return array.shift(amount, fill_value=fill if fill is not None else np.nan)
    if hasattr(array, 'shift'):
        shifted = array.shift(amount)
        return shifted if fill is None else _vacated(shifted, array, over, _off_the_axis(array, over, offset), fill)
    raise _unsupported('shift()', array)


#: Eager evaluation of every name in ``operators.BUILTIN_NAMES``. The two must
#: agree exactly — enforced by ``tests/test_architecture.py``, because a name
#: one lane implements and the other does not is precisely the divergence
#: that would make the differential tests a comparison of dialects.
_OPERATORS: dict[str, Callable[..., Any]] = {
    'sum': _operator_sum,
    'at': _operator_at,
    'shift': _operator_shift,
    'sum_back': _operator_sum_back,
}


# ---------------------------------------------------------------------------
# Where-mask evaluation
# ---------------------------------------------------------------------------


def evaluate_where(
    node: WhereNode | None,
    dataset: xr.Dataset,
    master_coords: dict[str, pd.Index],
    model: linopy.Model | None = None,
    dim_coords: Mapping[str, Mapping[str, xr.DataArray]] | None = None,
) -> xr.DataArray:
    """Evaluate a **resolved** where AST against a parameter dataset.

    A node, not a string: resolution has already decided what every name refers
    to, so this performs no lookups and cannot disagree with the relational lane
    about scoping. It lives here rather than in ``where_parser.py`` because it
    is xarray-only.

    ``dim_coords`` carries the bound lookup columns, which a predicate on a
    lookup reads instead of the parameter dataset — the same store the grouped
    sum reads its mapping from.

    Always a boolean DataArray. The no-mask case comes back 0-dimensional, so
    callers combine with ``&``/``|`` without case analysis.
    """
    if node is None:
        return xr.DataArray(True)

    return _eval_node(node, dataset, master_coords, model, dim_coords or {})


def _eval_node(
    node: WhereNode,
    dataset: xr.Dataset,
    master_coords: dict[str, pd.Index],
    model: linopy.Model | None = None,
    dim_coords: Mapping[str, Mapping[str, xr.DataArray]] = MappingProxyType({}),
) -> xr.DataArray:
    """One resolved where node as a boolean DataArray.

    Two absences read as exclusion rather than as an answer: a variable's
    masked-out coordinate carries label ``-1`` — linopy's own marker for an
    absent slot, which is exactly the question ``defined(v)`` asks — and a
    comparison over NaN comes back false. Comparison right-hand sides are
    literals except between two lookups, which resolution admits only over one
    dimension; every other declared name there it rejects.

    **A null lookup value is excluded explicitly rather than by ``fillna``.** A
    partial lookup arrives as an object array holding ``None``, and numpy
    answers ``None != 'north'`` with *True* rather than with null — so a ``!=``
    would keep exactly the labels that map nowhere, which is the reading law 8
    forbids and the relational lane does not give.
    """

    def evaluate(child: WhereNode) -> xr.DataArray:
        """Recurse carrying this call's bindings — what the connectives need."""
        return _eval_node(child, dataset, master_coords, model, dim_coords)

    if isinstance(node, BooleanLiteralNode):
        return xr.DataArray(node.value)

    if isinstance(node, (UnresolvedNameNode, UnresolvedComparisonNode, _UnresolvedPositionNode)):
        msg = (
            f'{type(node).__name__} reached the evaluator unresolved. '
            f'Where strings must go through resolution.resolve_where() first.'
        )
        raise AssertionError(msg)

    if isinstance(node, ParameterDefinedNode):
        arr = dataset[node.name]
        if arr.dtype == bool:
            return arr
        if arr.dtype.kind in 'OUS':
            return arr.notnull()
        return arr.notnull() & np.isfinite(arr)

    if isinstance(node, VariableDefinedNode):
        if model is None:
            msg = (
                f"where references variable '{node.name}', but no model was passed to the "
                f'evaluator — a variable mask can only be read off the model that holds it.'
            )
            raise AssertionError(msg)
        return model.variables[node.name].labels != -1

    if isinstance(node, (ParameterComparisonNode, DimensionComparisonNode)):
        if isinstance(node, ParameterComparisonNode):
            arr = dataset[node.name]
        else:
            arr = xr.DataArray(
                master_coords[node.name],
                coords={node.name: master_coords[node.name]},
                dims=[node.name],
            )

        result = _PREDICATE_OPS[node.op](arr, node.value)
        return result.fillna(False).astype(bool)

    if isinstance(node, DimensionPositionNode):
        labels = master_coords[node.name]
        if node.by is not None:
            groups = _bound_lookup(node.by, node.name, dim_coords)
            offsets = _group_offsets(node, groups.values)
            arr = xr.DataArray(offsets, coords={node.name: labels}, dims=[node.name])
            return (_PREDICATE_OPS[node.op](arr, 0) & arr.notnull()).fillna(value=False).astype(bool)
        at = node.position + len(labels) if node.position < 0 else node.position
        if not 0 <= at < len(labels):
            msg = (
                f'where: index({node.name}, {node.position}) names position {at} of '
                f"'{node.name}', which has {len(labels)} coordinate(s). A boundary that "
                f'names no coordinate leaves the rows it was to seed unseeded.'
            )
            raise DataError(msg)
        arr = xr.DataArray(np.arange(len(labels)), coords={node.name: labels}, dims=[node.name])
        return _PREDICATE_OPS[node.op](arr, at).astype(bool)

    if isinstance(node, LookupComparisonNode):
        arr = _bound_lookup(node.name, node.over, dim_coords)
        return (_PREDICATE_OPS[node.op](arr, node.value) & arr.notnull()).fillna(value=False).astype(bool)

    if isinstance(node, LookupPairComparisonNode):
        left = _bound_lookup(node.name, node.over, dim_coords)
        right = _bound_lookup(node.other, node.over, dim_coords)
        defined = left.notnull() & right.notnull()
        return (_PREDICATE_OPS[node.op](left, right) & defined).fillna(value=False).astype(bool)

    if isinstance(node, LookupDefinedNode):
        return _bound_lookup(node.name, node.over, dim_coords).notnull()

    if isinstance(node, NotNode):
        return ~evaluate(node.operand)

    if isinstance(node, AndNode):
        return evaluate(node.left) & evaluate(node.right)

    if isinstance(node, OrNode):
        return evaluate(node.left) | evaluate(node.right)

    assert_never(node)


def _coefficient(parameter: Any) -> Any:
    """A parameter in a coefficient position, its uncovered slots at zero.

    Where this lane answers linopy's v1 absence convention (linopy's
    ``doc/design/convention.rst``): the answer is *positional* — one missing
    row means zero in a coefficient, an error in ``bounds:``, false in a
    ``where`` operand — so it lives at the read, not as one fill in
    ``load_parameters`` that would be wrong for the other two. A tidy
    parameter table is a compressed dense array, not a record of absence:
    rows only for the live coordinates says the coefficient is zero elsewhere
    (the data-binding rules). ``load_parameters`` reindexes to the master coordinates, so an
    uncovered slot arrives as NaN — and v1 §5 refuses a NaN in a
    user-supplied constant. Correct under the legacy convention too, so not
    conditional on ``linopy.options['semantics']``.
    """
    return parameter.fillna(0.0)


def _vacated(shifted: Any, operand: Any, over: str, vacated: Any, fill: float) -> Any:
    """*shifted*, with the positions the shift vacated filled — and only those.

    linopy v1 counts ``.shift()`` among the operations that *create* absence
    (v1 §4), so the edge propagates and drops the row — the language's answer too
    (the operator rules, #289). This is the opt-out, reached only from ``shift(...,
    edge=0)``, and is the escape v1 itself prescribes rather than a rule of
    ours on top.

    ``fillna`` alone cannot spell it: by the time it runs, the edge the shift
    just made and a coordinate *operand* never had are both absent, and filling
    the second builds a row asserting ``x <= 0`` where the model said nothing.
    So the fill lands where the shift vacated **and** the operand carries the
    coordinate — the rule the relational lane states by crossing the edge with
    the other-dim combinations the operand has — and every other slot keeps the
    absence it arrived with.
    """
    carried = (~operand.isnull()).any(over)
    keep = carried & (~shifted.isnull() | vacated)
    return _filled(shifted, fill).where(keep)


def _filled(expression: Any, fill: float) -> Any:
    """*expression* with every absence in it standing as *fill*.

    ``to_linexpr()`` first when the operand is still a bare ``Variable``:
    ``Variable.fillna`` means a label fill on the released line and an
    expression fill on the v1 branch, and only the expression method is stable.
    """
    if hasattr(expression, 'to_linexpr'):
        expression = expression.to_linexpr()
    return expression.fillna(fill)
