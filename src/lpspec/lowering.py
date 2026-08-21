"""Lower a parsed YAML schema (typed AST) to the relational logical plan.

The lowering seam (docs/about/architecture.md, "The relational lane"): it consumes
the same typed AST the eager builder evaluates and emits a
:class:`~lpspec.relational.plan.Program`. It lives on the language side, so the
engine subpackage stays free of YAML knowledge and this module never imports
the eager builder.

Constructs with no lowering raise :class:`~lpspec.errors.LanguageError` naming
the construct and its rewrite, never a pointer to another backend: the two
lanes accept the same language, so a rejection here is a language gap
(docs/about/roadmap.md) rather than a routing decision.

Semantics mirror the eager builder exactly:

- a reduction over a dim the operand does not carry is an error, not a silent
  identity — ``dimensions.py`` owns that rule and this module asks it;
- a constraint is **one rule** carrying its own name, so a row is read back by
  the name the file writes, with no positional suffix to guess (#298);
- a file declares one objective, likewise one expression;
- an objective is scalar, so every reduction in it is one the file wrote and
  neither lane sums anything on its own behalf.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal, assert_never, cast

from lpspec.errors import LanguageError
from lpspec.language import (
    AndNode,
    ArithmeticNode,
    BinaryOperatorNode,
    BooleanLiteralNode,
    ComparisonNode,
    DimensionComparisonNode,
    DimensionNode,
    DimensionPositionNode,
    EdgeNode,
    FunctionCallNode,
    KeywordNode,
    LookupComparisonNode,
    LookupDefinedNode,
    LookupNode,
    LookupPairComparisonNode,
    NameListNode,
    NameNode,
    Namespace,
    NotNode,
    NumberNode,
    OrNode,
    ParameterComparisonNode,
    ParameterDefinedNode,
    ParameterNode,
    UnaryOperatorNode,
    UnresolvedComparisonNode,
    UnresolvedNameNode,
    UnresolvedPositionNode,
    VariableDefinedNode,
    VariableNode,
    WhereNode,
    call_shape_error,
    carries_variable,
    check_binary,
    dims_of,
    edge_error,
    expand_piecewise,
    expression_of,
    where_of,
)
from lpspec.relational import plan

if TYPE_CHECKING:
    from collections.abc import Callable

    from lpspec.language import Buildable, Model

_SENSES = {'==', '<=', '>='}


def lower_program(schema: Model) -> plan.Program:
    """Compile a validated :class:`Model` into a :class:`Program`.

    A ``domain: binary`` variable lowers with fixed 0/1 bounds, matching
    linopy's ``binary=True``.

    Raises:
        LanguageError: A construct outside the streaming language, named with
            its rewrite.
    """
    expanded = expand_piecewise(schema)
    ns = Namespace.of(expanded)
    parameters = tuple(
        plan.ParameterDeclaration(name, tuple(pdef.dims), pdef.dtype) for name, pdef in expanded.parameters.items()
    )

    variables = []
    for vname, vdef in expanded.variables.items():
        variable_type = cast('plan.VariableType', vdef.domain)
        if variable_type == 'binary':
            lower, upper = plan.Constant(0.0), plan.Constant(1.0)
        else:
            lower, upper = _bound_expression(vdef.bounds.lower), _bound_expression(vdef.bounds.upper)
        variables.append(
            plan.VariableDeclaration(
                vname,
                tuple(vdef.foreach),
                where=_lower_where(vdef.where, ns, f"variable '{vname}'", self_variable=vname),
                lower=lower,
                upper=upper,
                variable_type=variable_type,
                absence=cast('plan.VariableAbsence', vdef.absence),
            )
        )

    constraints = []
    for cname, cdef in expanded.constraints.items():
        where = _lower_where(cdef.where, ns, f"constraint '{cname}'")
        ast = expression_of(cdef.expression, expanded, ns, f"constraint '{cname}'")
        if not isinstance(ast, ComparisonNode):
            raise LanguageError(
                f"constraint '{cname}': expression must contain exactly one "
                f'comparison operator (<=, >=, ==). Got: {cdef.expression!r}'
            )
        if ast.op not in _SENSES:
            raise LanguageError(f"constraint '{cname}': unsupported sense '{ast.op}'")
        constraints.append(
            plan.ConstraintDeclaration(
                cname,
                tuple(cdef.foreach),
                lhs=_lower_expr(ast.left, expanded, f"constraint '{cname}'", ceiling=2),
                sense=ast.op,
                rhs=_lower_expr(ast.right, expanded, f"constraint '{cname}'", ceiling=2),
                where=where,
            )
        )

    objective = None
    if (odef := expanded.objective) is not None:
        ast = expression_of(odef.expression, expanded, ns, 'the objective')
        if isinstance(ast, ComparisonNode):
            raise LanguageError('the objective: expression must not contain a comparison operator')
        objective = plan.ObjectiveDeclaration(
            'min' if odef.sense == 'minimize' else 'max',
            _lower_expr(ast, expanded, 'the objective', ceiling=2),
        )

    dimensions = tuple(
        plan.DimensionDeclaration(
            dname,
            tuple(plan.LookupDeclaration(cname, target) for cname, target in expanded.targeted_of(dname).items()),
            tuple(expanded.labels_of(dname)),
        )
        for dname in expanded.dimensions
    )
    sos = tuple(
        plan.SosDeclaration(
            sname,
            sdef.variable,
            sdef.over,
            sos_type=cast('Literal[1, 2]', sdef.type),
            big_m=sdef.big_m,
        )
        for sname, sdef in expanded.sos.items()
    )
    return plan.Program(parameters, tuple(variables), tuple(constraints), objective, dimensions, sos)


def lower_expression(schema: Model, name: str) -> plan.Expression:
    """Compile the named expression *name* into a plan expression, on demand.

    The read-time half of ``expressions:``. :func:`lower_program` lowers none
    of them — a build pays nothing for a declared expression (the rules for named expressions) — so a
    reader asks here for the one it is reading, when it is read.

    Args:
        schema: The validated model declaring *name* under ``expressions:``.
        name: A declared expression name.

    Raises:
        KeyError: No named expression called *name*.
        LanguageError: A construct outside the streaming language.
    """
    expanded = expand_piecewise(schema)
    context = f"named expression '{name}'"
    ns = Namespace.of(expanded)
    ast = expression_of(expanded.expressions[name].expression, expanded, ns, context)
    assert not isinstance(ast, ComparisonNode), 'load-time validation refuses a comparison in a named expression'
    return _lower_expr(ast, expanded, context)


def expression_thunks(schema: Model) -> dict[str, Callable[[], plan.Expression]]:
    """One deferred :func:`lower_expression` per declared named expression.

    What a build hands the engine so a solve's result can read them: thunks,
    never plans, because building the dict is all a build may pay (the rules for named expressions).
    """
    return {name: partial(lower_expression, schema, name) for name in schema.expressions}


# ---------------------------------------------------------------------------
# expression lowering
# ---------------------------------------------------------------------------


def _lower_expr(node: ArithmeticNode, schema: Buildable, context: str, *, ceiling: int = 1) -> plan.Expression:
    """Rewrite one resolved core-AST expression as a plan expression.

    Three language rules are *asked* here and answered elsewhere: the call
    shape (``operators.call_shape_error``, re-asked so an AST that skipped
    resolution gets the language's wording rather than an ``IndexError``), the
    dim rules (``dims_of``) and degree (``check_binary``), both asked for
    their verdict rather than decided again here.

    What stays is about the plan: which node a call becomes, and the shapes a
    node cannot represent — a ``GroupSum`` groups by a declared lookup, a
    ``Translate`` distance is an integer literal. ``Sum`` and ``GroupSum`` stay
    two nodes under one surface verb, reducing a dim away and reducing it into
    another being different relational shapes.
    """
    if isinstance(node, NumberNode):
        return plan.Constant(node.value)

    if isinstance(node, VariableNode):
        return plan.Variable(node.name)

    if isinstance(node, ParameterNode):
        return plan.Parameter(node.name)

    if isinstance(node, EdgeNode):
        msg = f'EdgeNode({node.policy!r}) reached lowering: an edge policy is a shift() kwarg, not a value.'
        raise AssertionError(msg)

    if isinstance(node, KeywordNode):
        msg = (
            f'KeywordNode({node.value!r}) reached lowering. A quoted keyword is consumed '
            f'by its kwarg during resolution (docs/about/architecture.md hard rule 1).'
        )
        raise AssertionError(msg)
    if isinstance(node, (NameNode, NameListNode, DimensionNode, LookupNode)):
        shown = node.name if isinstance(node, (NameNode, DimensionNode)) else node.shown
        msg = (
            f'{type(node).__name__}({shown!r}) reached lowering. Expressions '
            f'must go through resolution.expression_of() first '
            f'(docs/about/architecture.md hard rule 1).'
        )
        raise AssertionError(msg)

    if isinstance(node, UnaryOperatorNode):
        inner = _lower_expr(node.operand, schema, context, ceiling=ceiling)
        return plan.Negate(inner) if node.op == '-' else inner

    if isinstance(node, BinaryOperatorNode):
        left = _lower_expr(node.left, schema, context, ceiling=ceiling)
        right = _lower_expr(node.right, schema, context, ceiling=ceiling)
        check_binary(node, context, ceiling=ceiling)
        match node.op:
            case '+':
                return plan.Add(left, right)
            case '-':
                return plan.Add(left, plan.Negate(right))
            case '*':
                return plan.Multiply(left, right)
            case '/':
                return plan.Divide(left, right)
            case '**':
                return plan.Power(left, right)
            case _:  # pragma: no cover — check_binary refuses every other operator
                raise AssertionError(f'{context}: operator {node.op!r} passed the degree check')

    if isinstance(node, FunctionCallNode):
        shape_error = call_shape_error(node.name, len(node.args), node.kwargs)
        if shape_error is not None:
            raise LanguageError(f'{context}: {shape_error}')

        if node.name == 'sum':
            by_node = node.kwargs.get('by')
            _check_dim_rules(node, schema, context)
            operand = _lower_expr(node.args[0], schema, context, ceiling=ceiling)
            if by_node is None and 'over' not in node.kwargs:
                return plan.Sum(operand, tuple(sorted(dims_of(node.args[0], schema, context))))
            if by_node is None:
                over_node = node.kwargs['over']
                if not isinstance(over_node, DimensionNode):
                    raise LanguageError(f'{context}: sum(over=...) must name a dimension')
                return plan.Sum(operand, (over_node.name,))
            if not isinstance(by_node, LookupNode):
                raise LanguageError(f'{context}: sum(by=...) must name a lookup')
            return plan.GroupSum(
                operand,
                over=by_node.dimension,
                coordinate=by_node.names,
                into=by_node.into,
            )

        if node.name == 'at':
            by_node = node.kwargs['by']
            if not isinstance(by_node, LookupNode):
                raise LanguageError(f'{context}: at(by=...) must name a lookup')
            _check_dim_rules(node, schema, context)
            return plan.At(
                _lower_expr(node.args[0], schema, context, ceiling=ceiling),
                over=by_node.dimension,
                coordinate=by_node.names,
                into=by_node.into,
            )

        if node.name == 'sum_back':
            over_node = node.kwargs['over']
            if not isinstance(over_node, DimensionNode):
                raise LanguageError(f'{context}: sum_back(over=...) must name a dimension')
            within_node = node.kwargs['within']
            _check_dim_rules(node, schema, context)
            operand = _lower_expr(node.args[0], schema, context, ceiling=ceiling)
            wrap = _window_edge(node.kwargs.get('edge'), context)
            width: int | str
            if isinstance(within_node, ParameterNode):
                _check_named_width(within_node.name, over_node.name, schema, context)
                width = within_node.name
            elif (
                isinstance(within_node, NumberNode)
                and within_node.value >= 1
                and int(within_node.value) == within_node.value
            ):
                width = int(within_node.value)
            else:
                raise LanguageError(f'{context}: {_window_width_message()}')
            return plan.Window(operand, over_node.name, width=width, wrap=wrap)

        if node.name == 'shift':
            over_node = node.kwargs['over']
            if not isinstance(over_node, DimensionNode):
                raise LanguageError(f'{context}: shift(over=...) must name a dimension')
            partition = _partition_of(node)
            by_node = node.kwargs['offset']
            sign = 1
            if isinstance(by_node, UnaryOperatorNode) and by_node.op == '-':
                sign, by_node = -1, by_node.operand
            if not isinstance(by_node, ParameterNode) and (
                not isinstance(by_node, NumberNode) or int(by_node.value) != by_node.value
            ):
                raise LanguageError(f'{context}: {_shift_by_message()}')
            _check_dim_rules(node, schema, context)
            operand = _lower_expr(node.args[0], schema, context, ceiling=ceiling)
            has_var = carries_variable(node.args[0])
            edge = node.kwargs.get('edge')
            wrap = isinstance(edge, EdgeNode)
            fill = None if wrap else _translate_fill(edge, context, has_var=has_var)
            if not wrap and fill is None and not has_var:
                raise LanguageError(_shift_over_data_message(context))
            by: int | str
            if isinstance(by_node, ParameterNode):
                _check_named_offset(by_node.name, node, over_node.name, schema, context, wrap=wrap, fill=fill)
                if sign < 0:
                    raise LanguageError(f'{context}: {_negated_offset_message(by_node.name)}')
                by = by_node.name
            else:
                assert isinstance(by_node, NumberNode)
                by = sign * int(by_node.value)
            return plan.Translate(operand, over_node.name, offset=by, wrap=wrap, fill=fill, partition=partition)

        raise LanguageError(f"{context}: built-in '{node.name}' declares no lowering case")

    assert_never(node)


def _check_dim_rules(node: FunctionCallNode, schema: Buildable, context: str) -> None:
    """Apply the language's dim rules to an operator call, discarding the dim set.

    Lowering wants the *raise*, not the answer. Called after the plan-shape
    checks so those speak first, and only for one call's dims — the enclosing
    frame is ``dimensions.check_schema``'s business.
    """
    dims_of(node, schema, context)


def _translate_fill(node: ArithmeticNode | None, context: str, *, has_var: bool) -> float | None:
    """The number an ``edge=`` names, or ``None`` for the absence default.

    One kwarg, three policies. ``edge='wrap'`` is cyclic and never reaches here,
    which makes a cyclic call that also asks for a fill unrepresentable rather
    than refused; a number is what the vacated slots contribute; an absent
    ``edge=`` leaves them absent.

    **The right fill is positional**, linopy v1's own reason for refusing to
    pick one (``convention.rst`` §7): 0 is the identity of a sum and 1 of a
    product, so ``x * shift(eff, over=t, offset=1, edge=1)`` wants a different
    number from ``lam <= seg + shift(seg, over=bp, offset=1, edge=0)``. Over data
    any number is accepted, both lanes filling natively.

    Over an operand carrying a **variable** the only representable fill is 0,
    the vacated slot contributing no term at all. A nonzero one would be a
    constant standing where a term was — a different fragment kind — and is
    refused rather than implemented on one lane and not the other.
    """
    if node is None:
        return None
    sign = 1.0
    if isinstance(node, UnaryOperatorNode) and node.op in ('-', '+'):
        sign, node = (-1.0 if node.op == '-' else 1.0), node.operand
    if not isinstance(node, NumberNode):
        raise LanguageError(f'{context}: {edge_error("shift", "...")}')
    fill = sign * float(node.value)
    if has_var and fill != 0:
        raise LanguageError(
            f'{context}: shift(edge={fill:g}) over an expression containing a variable — only '
            f'fill=0 is representable there, since a vacated slot contributes no term. A nonzero '
            f'fill would be a constant standing where a term was; add that constant to the '
            f'expression instead.'
        )
    return fill


def _partition_of(node: FunctionCallNode) -> str | None:
    """The lookup a translation walks inside, if the call names one.

    That it is a *single* lookup, and one *over the translated dimension*, is
    checked at load with the other dim rules (``language/dimensions.py``), where
    a model is refused before any data is read.
    """
    by_node = node.kwargs.get('by')
    if by_node is None:
        return None
    assert isinstance(by_node, LookupNode)
    return by_node.names[0]


def _shift_by_message() -> str:
    """What a ``offset=`` may be, now that it may be two things."""
    return (
        'shift(offset=...) must be a whole number, or the name of an integer '
        'parameter when the offset differs per entity — a lead time, a transit '
        'time, a minimum up time.'
    )


def _negated_offset_message(name: str) -> str:
    """Why ``offset=-lead`` is refused rather than negated.

    A literal offset is written with its sign in the call; a named one carries
    it in the values, where the reader of the data can see which way each row
    points. Allowing both spellings would let one model say ``offset=-lead`` and
    another ``offset=lead`` with negative values and mean the same thing.
    """
    return (
        f'shift(offset=-{name}) negates a named offset, which the language does not do.\n'
        f"Put the sign in '{name}' itself — a named offset carries its direction "
        f'in the data, where the row that points backwards says so.'
    )


def _window_width_message() -> str:
    return (
        'sum_back(within=...) needs a whole number of positions of at least 1, or the '
        'name of an integer parameter when the window differs per entity. A width of 1 '
        'is the operand itself.'
    )


def _window_edge(edge: ArithmeticNode | None, context: str) -> bool:
    """Whether the window wraps, refusing a fill.

    A window sums the terms it can see, so a position the axis does not reach
    contributes nothing — there is no vacated slot to fill, which is what makes
    this narrower than ``shift(edge=...)``.
    """
    if edge is None:
        return False
    if isinstance(edge, EdgeNode):
        return True
    raise LanguageError(
        f"{context}: sum_back(edge=...) takes 'wrap' or nothing. A window sums the terms "
        f'it reaches, so a position before the first contributes nothing rather than a '
        f'fill value; add the constant to the expression if you want one.'
    )


def _check_named_width(name: str, dimension: str, schema: Buildable, context: str) -> None:
    """The two rules that make a per-entity window width mean one thing.

    A width that varies along the dimension being summed over would make the
    window a different length at every position, which no longer reads as
    "the last *n*"; and a non-integral width cannot count positions.
    """
    declared = schema.parameters[name]
    if declared.dtype != 'int':
        raise LanguageError(
            f"{context}: sum_back(within={name}) needs an integer parameter, and '{name}' is "
            f"declared '{declared.dtype}'. A width counts positions rather than measuring a "
            f'distance.'
        )
    if dimension in declared.dims:
        raise LanguageError(
            f"{context}: sum_back(within={name}) sums over '{dimension}', and '{name}' is "
            f'declared over it ({sorted(declared.dims)}). A width that changes along '
            f"'{dimension}' gives a different window at every position, which is not a "
            f'trailing window. Sum over a width that is constant along it.'
        )


def _check_named_offset(
    name: str,
    node: FunctionCallNode,
    dimension: str,
    schema: Buildable,
    context: str,
    *,
    wrap: bool,
    fill: float | None,
) -> None:
    """The four rules that make a per-entity offset mean one thing.

    An offset that depends on the dimension it translates would move each
    position by a different amount *along that axis*, which is a permutation
    rather than a translation and has no reading as a lag. An offset that is
    not integral cannot land on a coordinate. An offset varying over a
    dimension neither the operand nor a partition puts within reach has no
    coordinate to be read at, and would broadcast the shifted expression onto
    a dimension the operator's dim rule says it does not have (#1161). And a
    named offset must say what the vacated positions contribute: the absent
    edge propagates through a presence frame keyed by the dimension alone,
    which a per-entity edge is not — refused here rather than answered wrongly
    (#850).
    """
    # An undeclared name never reaches here: resolution refuses it first, and
    # names the parameters that do exist, which is the better message.
    declared = schema.parameters[name]
    if declared.dtype != 'int':
        raise LanguageError(
            f"{context}: shift(offset={name}) needs an integer parameter, and '{name}' is "
            f"declared '{declared.dtype}'. An offset lands on a coordinate, so it counts "
            f'positions rather than measuring a distance.'
        )
    if dimension in declared.dims:
        raise LanguageError(
            f'{context}: shift(offset={name}) is offset by a parameter that itself spans '
            f"'{dimension}', the dimension being translated. That moves each position by "
            f'a different amount along the axis it is moving, which is a permutation and '
            f'not a lag — drop the dimension from the parameter, or state the map you mean '
            f'as a lookup.'
        )
    stray = sorted(set(declared.dims) - _within_reach(node, schema, context))
    if stray:
        raise LanguageError(f'{context}: {_stray_offset_message(name, stray, _partition_of(node))}')
    if not wrap and fill is None:
        raise LanguageError(
            f'{context}: shift(offset={name}) leaves the vacated positions absent, which a '
            f'per-entity offset cannot say yet.\n'
            f"Add edge='wrap' for a cyclic translation, or edge=<number> for what the "
            f'vacated positions contribute.'
        )


def _within_reach(node: FunctionCallNode, schema: Buildable, context: str) -> set[str]:
    """The dims a named offset may vary over.

    Two ways for a coordinate of the offset to be one the shift can read it
    at: the shifted expression carries the dim, or the partition groups the
    walked axis *into* it, and then every coordinate of a group shares the
    group's own lag.
    """
    reach = set(dims_of(node.args[0], schema, context))
    by_node = node.kwargs.get('by')
    if by_node is not None:
        assert isinstance(by_node, LookupNode)
        reach |= set(by_node.into)
    return reach


def _stray_offset_message(name: str, stray: list[str], partition: str | None) -> str:
    """Why an offset over a dim nothing puts within reach is refused.

    Without this the eager lane broadcasts the shifted expression onto that
    dim — a bigger model than the file reads as, and one the operator's own
    dim rule says it does not have — while the relational lane asks for a
    column no frame carries (#1161).
    """
    grouped = (
        f'or over what by={partition} groups into, which is one lag per group'
        if partition is not None
        else 'or, under by=<lookup>, over the dimension that lookup groups into — one lag per group'
    )
    return (
        f'shift(offset={name}) is offset by a parameter over {stray}, which the shifted '
        f'expression does not carry.\n'
        f'An offset is read at the coordinate it moves, so it varies over the dims that '
        f'expression has, {grouped}.\n'
        f"Read '{name}' onto those dims with at(), or drop {stray} from it."
    )


def _shift_over_data_message(context: str) -> str:
    """The three ways out, one of which is two things at once.

    A ``where`` is a *companion* to ``edge=``, not an alternative: the refusal
    is decided on the expression alone so a mask does not lift it, and
    ``edge=0`` alone leaves a row at the vacated coordinate whose bound is that
    zero — the silent pinning this refusal exists to prevent. Either one alone
    is wrong, so the message says so rather than listing them as alternatives.
    """
    return (
        f'{context}: shift() over a variable-free expression leaves vacated positions with no '
        f'value, and inventing one is what silently pinned a bound to zero. Say which you mean:\n'
        f"  shift(x, over=d, offset=n, edge='wrap')   the dimension really is cyclic\n"
        f'  shift(x, over=d, offset=n, edge=0)        the vacated positions contribute zero\n'
        f'  ...and a where: excluding them        the vacated rows should not exist at all\n'
        f'A where: alone does not lift this — it is decided on the expression, before any mask '
        f'is read — and edge=0 alone leaves a row whose bound is that zero.'
    )


def _bound_expression(value: float | str) -> plan.Expression:
    if isinstance(value, str):
        return plan.Parameter(value)
    return plan.Constant(value)


# ---------------------------------------------------------------------------
# where lowering
# ---------------------------------------------------------------------------


def _lower_where(
    text: str | None, ns: Namespace, context: str, self_variable: str | None = None
) -> plan.Predicate | None:
    """Lower a where string to a plan predicate, ``None`` when there is no mask.

    A predicate that resolves to the constant ``True`` is dropped too: it is
    equivalent to no mask.
    """
    node = where_of(text, ns, context, self_variable)
    if node is None:
        return None
    pred = _lower_where_node(node, context)
    if isinstance(pred, plan.BooleanConstant) and pred.value:
        return None
    return pred


def _lower_where_node(node: WhereNode, context: str) -> plan.Predicate:
    if isinstance(node, BooleanLiteralNode):
        return plan.BooleanConstant(node.value)

    if isinstance(node, ParameterDefinedNode):
        return plan.ParameterDefined(node.name)

    if isinstance(node, VariableDefinedNode):
        return plan.VariableDefined(node.name)

    if isinstance(node, ParameterComparisonNode):
        return plan.ParameterComparison(node.name, node.op, node.value)

    if isinstance(node, DimensionComparisonNode):
        return plan.DimensionComparison(node.name, node.op, node.value)

    if isinstance(node, DimensionPositionNode):
        return plan.DimensionPosition(node.name, node.op, node.position, node.by)

    if isinstance(node, LookupComparisonNode):
        return plan.LookupComparison(node.name, node.over, node.op, node.value)

    if isinstance(node, LookupPairComparisonNode):
        return plan.LookupPairComparison(node.name, node.other, node.over, node.op)

    if isinstance(node, LookupDefinedNode):
        return plan.LookupDefined(node.name, node.over)

    if isinstance(node, (UnresolvedNameNode, UnresolvedComparisonNode, UnresolvedPositionNode)):
        msg = (
            f'{type(node).__name__} reached lowering unresolved. Where strings '
            f'must go through lpspec.language.where_of() first.'
        )
        raise AssertionError(msg)

    if isinstance(node, NotNode):
        return plan.Not(_lower_where_node(node.operand, context))
    if isinstance(node, (AndNode, OrNode)):
        node_type = plan.And if isinstance(node, AndNode) else plan.Or
        return node_type(
            _lower_where_node(node.left, context),
            _lower_where_node(node.right, context),
        )

    assert_never(node)


# ---------------------------------------------------------------------------
# check-time advice
# ---------------------------------------------------------------------------


def advice(program: plan.Program) -> list[str]:
    """Modeling advice ``check`` surfaces as warnings — never errors.

    One rule today: **everything under ``dimensions:`` should be an axis** —
    indexed by something, or aggregated into. A declared dimension with neither
    is a label space wearing a dimension's clothes, or dead weight. Advice
    rather than an error because it reads intent: a dimension declared ahead of
    the declarations that will use it is a model part-written, not a wrong one,
    and ``check`` is the door that says so without refusing to build.
    """
    axes: set[str] = set()
    for declaration in (*program.parameters, *program.variables, *program.constraints):
        axes.update(declaration.dims)
    expressions = [program.objective.expression] if program.objective is not None else []
    expressions.extend(side for c in program.constraints for side in (c.lhs, c.rhs))
    for e in expressions:
        axes |= _produced_axes(e)

    targeted = {lk.target: (d.name, lk.name) for d in program.dimensions for lk in d.lookups}
    notes: list[str] = []
    for d in program.dimensions:
        if d.name in axes:
            continue
        if d.name in targeted:
            owner, cname = targeted[d.name]
            notes.append(
                f"dimension '{d.name}' is never an axis: nothing is indexed by it and nothing "
                f"aggregates into it — it only serves as the target of lookup '{cname}' over "
                f"'{owner}'. That is a label space, not a dimension of this model; declare the "
                f'lookup as one instead:\n'
                f'  lookups:\n'
                f'    {cname}: {{over: {owner}, dtype: str}}'
            )
        else:
            notes.append(
                f"dimension '{d.name}' is never used: nothing is indexed by it, nothing "
                f'aggregates into it, and no lookup targets it. Remove it — or keep it '
                f'knowingly, if the declarations that use it are still to be written.'
            )
    return notes


def _produced_axes(e: plan.Expression) -> set[str]:
    """The axes an expression *creates*, beyond what its declarations index.

    ``sum(by=)`` lands terms on its target and ``at()`` spreads onto its fine
    dimension, so both are axes even when no declaration is indexed by them —
    an objective may group into a dimension and then implicitly sum it away.
    """
    out: set[str] = set()
    if isinstance(e, plan.GroupSum):
        out |= set(e.into)
    if isinstance(e, plan.At):
        out.add(e.over)
    for child in plan.children(e):
        out |= _produced_axes(child)
    return out
