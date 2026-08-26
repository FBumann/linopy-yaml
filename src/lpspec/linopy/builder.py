"""Model builder: schema + data → linopy Model.

**One section per kind of translation**, in the order a build performs them:
the four declarations (``Variables``, ``Special-ordered sets``,
``Constraints``, ``Objectives``) each ending in the ``model.add_*`` call they
exist to make, then ``AST evaluation`` for what an expression is worth.

Two questions a build asks are answered beside it rather than here, because
neither needs the schema or the model: ``operators.py`` evaluates a built-in
once its operands are values, and ``where.py`` turns a resolved ``where:`` into
the boolean array a declaration is masked by. The four positions an absent
value is spelled differently in are ``absence.py``, called qualified from here.

*Which* linopy call each construct becomes is the table in
``docs/about/linopy.md`` — one page rather than a comment per site, and
``tests/test_docs_site.py`` holds it to the operator list.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, assert_never

import numpy as np
from math_spec import (
    ArithmeticNode,
    BinaryOperatorNode,
    ComparisonNode,
    DimensionNode,
    EdgeNode,
    FunctionCallNode,
    KeywordNode,
    LookupNode,
    NameListNode,
    NameNode,
    Namespace,
    NumberNode,
    ParameterNode,
    UnaryOperatorNode,
    VariableNode,
    check_binary,
    expression_of,
    is_quadratic,
    unknown_operator_message,
    where_of,
)

from lpspec.errors import (
    DataError,
    LaneError,
    lane_cannot_build_message,
    null_bounds_message,
    unbound_lookup_message,
)
from lpspec.linopy import absence
from lpspec.linopy._notes import note
from lpspec.linopy.coverage import check_constant_side_covers, check_divisors_cover, gaps_under
from lpspec.linopy.operators import OPERATORS, operator_at, operator_grouped_sum
from lpspec.linopy.where import WhereContext, as_linopy_mask, evaluate_where

if TYPE_CHECKING:
    from collections.abc import Callable

    import linopy
    import pandas as pd
    import xarray as xr
    from math_spec import Buildable

_SIGN_MAP = {'==': '=', '<=': '<=', '>=': '>='}

#: The language's arithmetic. ``**`` is reached only over parameters — ``check_binary`` refuses it over a variable.
_ARITHMETIC_OPS: dict[str, Callable[[Any, Any], Any]] = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
    '**': operator.pow,
}


@dataclass(frozen=True)
class EvaluationContext(WhereContext):
    """Everything expression evaluation needs to resolve names.

    :class:`~lpspec.linopy.where.WhereContext` plus the schema and namespace,
    so a predicate reads it directly. Extend this rather than adding
    parameters to ``_eval_ast`` and every operator-facing seam.
    """

    #: Narrowed from the base: a build always has the model it populates.
    model: linopy.Model
    schema: Buildable = field(kw_only=True)
    ns: Namespace = field(kw_only=True)


def build_model(
    model: linopy.Model,
    schema: Buildable,
    dataset: xr.Dataset,
    master_coords: dict[str, pd.Index],
    dim_coords: dict[str, dict[str, xr.DataArray]] | None = None,
) -> None:
    """Populate a linopy Model from a parsed schema and loaded parameters.

    This mutates *model* in-place, adding variables, constraints and
    the objective as declared in *schema*.
    """
    ctx = EvaluationContext(
        dataset,
        master_coords,
        model,
        dim_coords or {},
        schema=schema,
        ns=Namespace.of(schema),
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
            mask = evaluate_where(where, ctx)

            _check_bounds_are_defined(vname, vdef, ctx.dataset, mask)

            ctx.model.add_variables(
                lower=lower,
                upper=upper,
                coords=coords,
                name=vname,
                mask=as_linopy_mask(mask),
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


def _resolve_bound(value: float | str, dataset: xr.Dataset) -> Any:
    """A bound as linopy takes it: the literal, or the named parameter's array."""
    return dataset[value] if isinstance(value, str) else value


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


def _refuse_quadratic(node: ComparisonNode) -> None:
    """A quadratic constraint is sayable and this lane cannot build one.

    Hard rule 3's amendment in the one place it bites: both lanes accept the
    same language, and ``linopy.Model.add_constraints`` refuses a
    ``QuadraticExpression`` outright. Refused *here*, in the language's own
    words and before linopy is asked, so the caller gets a sentence naming the
    lane that does build it rather than somebody else's
    ``NotImplementedError`` (:data:`lpspec.api.LANES`).
    """
    if any(is_quadratic(side) for side in (node.left, node.right)):
        raise LaneError(lane_cannot_build_message('linopy', ['quadratic_constraint']))


def _build_constraints(ctx: EvaluationContext) -> None:
    for cname, cdef in ctx.schema.constraints.items():
        with note(f"while building constraint '{cname}'"):
            c_where = where_of(cdef.where, ctx.ns, f"constraint '{cname}'")
            mask = evaluate_where(c_where, ctx)

            ast = expression_of(cdef.expression, ctx.schema, ctx.ns, f"constraint '{cname}'")
            assert isinstance(ast, ComparisonNode), 'load-time validation refuses a constraint without a comparison'

            _refuse_quadratic(ast)
            check_divisors_cover(f"constraint '{cname}'", ast, ctx, mask)
            check_constant_side_covers(f"constraint '{cname}'", ast, ctx, mask)

            lhs = _eval_ast(ast.left, ctx)
            rhs = _eval_ast(ast.right, ctx)
            if _term_free(lhs) and _term_free(rhs):
                continue
            sign = _SIGN_MAP[ast.op]

            ctx.model.add_constraints(lhs, sign, rhs, name=cname, mask=as_linopy_mask(mask))


def _term_free(side: Any) -> bool:
    """Whether *side* has nowhere for a variable term to sit.

    A bare ``Variable`` is a term; a ``LinearExpression`` over an empty axis has
    a term dimension of length zero; anything else is data.

    Both sides term-free is a constraint the *data* emptied: a dimension with no
    members reduces away to a number, so every row asserts something about
    constants alone, which the absence rules say is not a row and the relational
    lane builds none (#1108). The expression that names no variable to begin
    with cannot arrive here — validation refuses it while reading the file
    (#1171) — so this reads as data, not as a modelling mistake.
    """
    if hasattr(side, 'to_linexpr'):
        return False
    return getattr(side, 'nterm', 0) == 0


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

    The one position where a product of two variables is legal, so the degree
    ceiling is raised for this walk alone — linopy's ``*`` already answers with
    a ``QuadraticExpression``, so nothing here branches on which came back.
    """
    odef = ctx.schema.objective
    if odef is None:
        return
    with note('while building the objective'):
        ast = expression_of(odef.expression, ctx.schema, ctx.ns, 'the objective')
        assert not isinstance(ast, ComparisonNode), 'load-time validation refuses a comparison in the objective'

        check_divisors_cover('the objective', ast, ctx, None)

        expr = _eval_ast(ast, ctx, ceiling=2)
        _refuse_an_objective_constant(expr)

        sense = 'min' if odef.sense == 'minimize' else 'max'
        ctx.model.add_objective(expr, overwrite=True, sense=sense)


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


# ---------------------------------------------------------------------------
# AST evaluation
# ---------------------------------------------------------------------------


def _eval_ast(
    node: ArithmeticNode,
    ctx: EvaluationContext,
    *,
    ceiling: int = 1,
) -> Any:
    """Evaluate an expression AST node against the model namespace.

    One node kind per branch, and each is a line: a variable is its linopy
    term, a parameter its filled array, arithmetic the Python operator linopy
    overloads, and a call is :func:`_call`. The node kinds that reach here only
    through a bug say so rather than evaluating to something.

    Binary nodes go through ``check_binary`` first: ``**``, a quadratic
    product and a variable divisor are all refused by ``language/degree.py``,
    the same verdict the relational lane asks for and in the same sentence.
    """
    if isinstance(node, NumberNode):
        return node.value

    if isinstance(node, VariableNode):
        return absence.variable_term(ctx.model.variables[node.name], ctx.schema.variables[node.name].absence)

    if isinstance(node, ParameterNode):
        return absence.coefficient(ctx.dataset[node.name])

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
    if isinstance(node, (NameNode, NameListNode, DimensionNode, LookupNode)):
        shown = node.name if isinstance(node, (NameNode, DimensionNode)) else node.shown
        msg = (
            f'{type(node).__name__}({shown!r}) reached the evaluator. '
            f'Expressions must go through resolution.expression_of() first '
            f'(docs/about/architecture.md hard rule 1).'
        )
        raise AssertionError(msg)

    if isinstance(node, UnaryOperatorNode):
        operand = _eval_ast(node.operand, ctx, ceiling=ceiling)
        if node.op == '-':
            return -operand
        return operand

    if isinstance(node, BinaryOperatorNode):
        check_binary(node, ceiling=ceiling)
        left = _eval_ast(node.left, ctx, ceiling=ceiling)
        right = _eval_ast(node.right, ctx, ceiling=ceiling)
        return _ARITHMETIC_OPS[node.op](left, right)

    if isinstance(node, FunctionCallNode):
        return _call(node, ctx, ceiling=ceiling)

    assert_never(node)


def _call(node: FunctionCallNode, ctx: EvaluationContext, *, ceiling: int) -> Any:
    """One operator call, its operands and keywords evaluated.

    Two names in :data:`OPERATORS` do not reach it by that table. ``by=`` names
    a lookup rather than an operand, and it decides *which* operation runs:
    ``at`` reads through the lookup and ``sum(by=)`` groups along it, where
    plain ``sum`` reduces a dim. Both take the group's own labels, so both are
    spelled out here rather than passed a keyword.

    Every other keyword is translated to what its operator takes: a dim as its
    name, an edge as its policy, a lookup as its array, anything else evaluated
    as an expression.

    Raises:
        NameError: An operator name ``validation.py`` would have refused at
            load, which only a hand-built call can still carry.
    """
    if node.name not in OPERATORS:
        raise NameError(unknown_operator_message(node.name))
    args = [_eval_ast(a, ctx, ceiling=ceiling) for a in node.args]

    if node.name == 'at':
        by = node.kwargs['by']
        assert isinstance(by, LookupNode)
        return operator_at(args[0], _lookup_arrays(by, ctx), into=by.into)
    if node.name == 'sum' and (by := node.kwargs.get('by')) is not None:
        assert isinstance(by, LookupNode)
        return operator_grouped_sum(args[0], _lookup_arrays(by, ctx), into=by.into, labels=ctx.master_coords)

    return OPERATORS[node.name](*args, **{k: _keyword(v, ctx, ceiling=ceiling) for k, v in node.kwargs.items()})


def _keyword(value: Any, ctx: EvaluationContext, *, ceiling: int) -> Any:
    """One operator keyword, as the operator below takes it.

    A lookup arrives as its values over the dimension it is over, *named* for
    the dimension those values are labels of — the convention
    :func:`_checked_mappings` follows for the operators that take one
    positionally, and what lets a partition read a parameter declared over its
    target. A partition names one lookup, plural refused at load, so the first
    array is the whole of it.
    """
    if isinstance(value, DimensionNode):
        return value.name
    if isinstance(value, EdgeNode):
        return value.policy
    if isinstance(value, LookupNode):
        return _lookup_arrays(value, ctx)[0].rename(value.into[0])
    return _eval_ast(value, ctx, ceiling=ceiling)


def _lookup_arrays(by: LookupNode, ctx: EvaluationContext) -> tuple[Any, ...]:
    """The declared lookups ``by`` as arrays over the dimension they are over.

    Looked up rather than evaluated as operands: a lookup lives on the
    dimension, not in the parameter dataset. One array per name, in the
    order written, so the caller can pair each with the dim it targets.
    """
    arrays = []
    for name in by.names:
        try:
            arrays.append(ctx.dim_coords[by.dimension][name])
        except KeyError:
            raise DataError(unbound_lookup_message(name, by.dimension)) from None
    return tuple(arrays)
