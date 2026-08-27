"""The logical plan, and the pass that builds one.

An intermediate representation in the compiler sense — the name is what this
shape *is* to the engine (duckdb, Calcite and Spark all call it a logical plan)
rather than the generic category. A :class:`Program` is a complete declarative
description of a linear program over named tidy tables; data is bound at
execution time via a source registry.

**The pass lives here because the plan is what it makes.**
:meth:`Program.from_model` is the only way to build one from a file, and both
lanes read what it returns — ``relational/`` executes the plan, ``linopy/``
walks it into a ``linopy.Model``, and neither ever sees the AST behind it.
Splitting the two apart would mean a cycle: the pass constructs every node
defined here, so the constructor would have to reach back for the pass. Hard
rule 0 says a cycle is removed rather than deferred behind a lazy import, and
this is the removal.

Constructs with no lowering raise :class:`~lpspec.errors.LanguageError` naming
the construct and its rewrite, never a pointer to another backend: the two
lanes accept the same language, so a rejection here is a language gap
(docs/about/roadmap.md) rather than a routing decision.

Semantics the eager lane used to mirror and now simply reads:

- a reduction over a dim the operand does not carry is an error, not a silent
  identity — ``dimensions.py`` owns that rule and this module asks it;
- a constraint is **one rule** carrying its own name, so a row is read back by
  the name the file writes, with no positional suffix to guess (#298);
- a file declares one objective, likewise one expression;
- an objective is scalar, so every reduction in it is one the file wrote and
  neither lane sums anything on its own behalf.

Frozen dataclasses only — no execution logic below the pass, no engine imports.
Expressions support operator sugar so plans read naturally in Python::

    balance = GroupSum(Variable('p'), over='generator', coordinate=('bus',), into=('bus',)) - Parameter('load')
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NamedTuple, TypeVar, assert_never, cast

from math_spec import (
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
    LanguageError,
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
    check_binary,
    dims_of,
    edge_error,
    expression_of,
    where_of,
)
from math_spec import (
    carries_variable as ast_carries_variable,
)

from lpspec.errors import unknown_name_message

if TYPE_CHECKING:
    import datetime
    from collections.abc import Callable

    from math_spec import Buildable

_SENSES = {'==', '<=', '>='}

ConstraintSense = Literal['==', '<=', '>=']
ObjectiveSense = Literal['min', 'max']
ComparisonOperator = Literal['==', '!=', '<=', '>=', '<', '>']
VariableType = Literal['continuous', 'binary', 'integer']

#: What a masked variable's non-existence means where it does not exist.
#: ``undefined`` is the absence rules' default — a term carrying it takes its
#: row. ``zero`` says the quantity *is* zero there, so the term contributes
#: nothing and the row stands.
VariableAbsence = Literal['undefined', 'zero']


# --------------------------------------------------------------------------
# Affine expressions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Expression:
    """Base class for expressions over variables and parameters.

    Affine everywhere but the objective, where a :class:`Multiply` of two
    variable-carrying operands is degree 2; which position allows what is
    ``language/degree.py``'s to say and no node here records.

    The four operators exist for the tests that compose plans by hand;
    constructing Programs in Python is not supported API, so there is no
    scalar coercion and no reflected form.
    """

    def __add__(self, other: Expression) -> Expression:
        return Add(self, other)

    def __sub__(self, other: Expression) -> Expression:
        return Add(self, Negate(other))

    def __mul__(self, other: Expression) -> Expression:
        return Multiply(self, other)

    def __neg__(self) -> Expression:
        return Negate(self)


@dataclass(frozen=True)
class Constant(Expression):
    """A scalar constant."""

    value: float


@dataclass(frozen=True)
class Parameter(Expression):
    """A parameter reference — contributes to the constant part."""

    name: str


@dataclass(frozen=True)
class Variable(Expression):
    """A variable reference — one term per existing variable row."""

    name: str


@dataclass(frozen=True)
class Negate(Expression):
    operand: Expression


@dataclass(frozen=True)
class Add(Expression):
    left: Expression
    right: Expression


@dataclass(frozen=True)
class Multiply(Expression):
    """Product of two operands.

    Affine where at least one factor is variable-free. **Degree 2 where neither
    is**, which the language allows in the objective alone
    (``language/degree.py``) — so a consumer that cannot represent a quadratic
    term is told which position it is compiling rather than assuming it.
    """

    left: Expression
    right: Expression


@dataclass(frozen=True)
class Power(Expression):
    """``base ** exponent``, both variable-free.

    Degree 0 in variables wherever it appears, so no consumer has to ask what
    position it stands in: the language refuses a variable anywhere under it
    (``language/degree.py``), which is what lets this fold to one number per
    coordinate like any other parameter arithmetic.
    """

    base: Expression
    exponent: Expression


@dataclass(frozen=True)
class Divide(Expression):
    """Quotient ``numerator / divisor``. The divisor must be variable-free."""

    numerator: Expression
    divisor: Expression


@dataclass(frozen=True)
class Sum(Expression):
    """Sum ``operand`` over the named dims, removing them from the result."""

    operand: Expression
    over: tuple[str, ...]


@dataclass(frozen=True)
class GroupSum(Expression):
    """Sum ``operand`` through coordinates declared on dim ``over``.

    ``coordinate`` names coordinates carried by dim ``over`` whose values are
    labels of the matching dim in ``into``; the result replaces ``over`` with
    all of them. Everything is resolved before lowering, so the engine needs
    no schema lookup to place the terms.

    Several coordinates are one grouping into a product of targets, not a
    composition of groupings — they are consumed in a single join, so the pair
    of tuples is always the same length and their order pairs them up.
    """

    operand: Expression
    over: str
    coordinate: tuple[str, ...]
    into: tuple[str, ...]


@dataclass(frozen=True)
class At(Expression):
    """Read ``operand`` through a lookup — the adjoint of :class:`GroupSum`.

    Same mapping table, walked the other way: ``GroupSum`` consumes ``over``
    and produces ``into``, this consumes ``into`` and produces ``over``. The
    fields are named for the *table* rather than the direction, so the pair
    reads as one relation; the surface says which end you stand on
    (``sum(by=)`` consumes it, ``at(by=)`` produces it, the lookup names the map).

    The join fans out, many ``over`` labels sharing one ``into`` tuple — the
    fan-out ``GroupSum`` pays in reverse, so the locality class is unchanged.
    """

    operand: Expression
    over: str
    coordinate: tuple[str, ...]
    into: tuple[str, ...]


@dataclass(frozen=True)
class Translate(Expression):
    """Re-index along one dimension: the result at *t* is ``operand`` at *t - by*.

    One node for the whole of ``shift``, whose ``edge=`` decides ``wrap``:
    ``edge='wrap'`` is periodic (``xarray.roll``), absent or numeric is not.

    ``fill`` decides what an acyclic shift leaves behind. ``None``, what bare
    ``shift`` lowers to, leaves the vacated positions **absent** so they
    propagate and drop the row — linopy v1's ``.shift()``. A number makes them
    present and contribute it, the ``.fillna(0)`` escape hatch spelled in the
    language. Always ``None`` under ``wrap``, a cyclic map vacating nothing.

    ``offset`` is how far back to reach: an integer, or the name of an integer
    parameter when it differs per entity — a construction lead time, a transit
    time, a minimum up time. A named offset may not depend on the dimension
    being translated, and carries its sign in the values.

    ``partition`` names a lookup over ``dimension``, and then the translation
    happens **inside each group** it makes: the neighbour of a coordinate is the
    one before it *in its own group*, the edge is that group's edge, and a wrap
    closes each group onto itself. A coordinate the lookup sends nowhere is in
    no group and reaches nothing.
    """

    operand: Expression
    dimension: str
    offset: int | str
    wrap: bool = True
    fill: float | None = None
    partition: str | None = None


@dataclass(frozen=True)
class Window(Expression):
    """Sum ``operand`` over a trailing window along one dimension.

    The result at *t* is the sum of the operand at every position from
    *t - width + 1* through *t*, so a width of 1 is the operand itself. The
    dimension survives: this replicates terms onto the positions that can see
    them rather than reducing anything away.

    ``width`` is a whole number, or the name of an integer parameter when the
    window differs per entity — a minimum up time, a rolling budget, a delivery
    horizon. A named width may not depend on the dimension being summed over.

    ``partition`` names a lookup over that dimension, and the window then stops
    at each group's edge: a representative day, a season, a scenario's own run
    of hours. Positions are counted inside the group rather than along the
    axis, so a coordinate the lookup places nowhere reaches nothing at all —
    not even itself.

    One node rather than a sum of ``Translate``s, because the number of terms
    would then be read from data and the plan's *shape* is fixed before any
    data is bound. What data supplies is the mask's cardinality, exactly as it
    supplies how many snapshots there are.
    """

    operand: Expression
    dimension: str
    width: int | str
    wrap: bool = False
    partition: str | None = None


def children(expression: Expression) -> tuple[Expression, ...]:
    """The sub-expressions of *expression* — the structural half of any walk.

    Every walk over a plan expression recurses through here and differs only in
    what it does at the leaves. Enumerating the children once is how a node
    added later reaches all of them rather than one.
    """
    if isinstance(expression, Negate):
        return (expression.operand,)
    if isinstance(expression, (Add, Multiply)):
        return (expression.left, expression.right)
    if isinstance(expression, Divide):
        return (expression.numerator, expression.divisor)
    if isinstance(expression, (Sum, GroupSum, At, Translate, Window)):
        return (expression.operand,)
    return ()


# --------------------------------------------------------------------------
# Predicates (where masks — row absence)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Predicate:
    """Base class for where-predicates."""


@dataclass(frozen=True)
class ParameterComparison(Predicate):
    parameter: str
    op: ComparisonOperator
    value: float | str


@dataclass(frozen=True)
class DimensionComparison(Predicate):
    """Compare a *dimension coordinate* to a literal — ``where: "snapshot > 0"``.

    Unlike :class:`ParameterComparison`, no parameter is involved: the dim
    table is already in the frame, so this is a filter on its own column.
    """

    dimension: str
    op: ComparisonOperator
    #: ``datetime`` widens this: a datetime dimension's boundary is a date,
    #: and comparing one to a number reads it as an epoch offset (#460).
    value: float | str | datetime.date


@dataclass(frozen=True)
class DimensionPosition(Predicate):
    """Compare where a row sits along a dimension against *position*.

    ``where: "position(snapshot) == 0"``. The position is resolved
    against the *bound* coordinate order rather than lowered to a label, so a
    relabelled index moves the boundary with it. Negative counts from the end.
    Both sides are integers, so every comparator reads the one way.

    ``by`` names a lookup over the same dimension, and then the position is
    counted **within each group** it makes:
    ``position(snapshot, by=period_of) == 0``
    is every period's first snapshot. A coordinate its lookup sends nowhere is
    in no group and matches nothing, as a null group does everywhere else.

    Every consuming lane already holds that order — the dim table's ``ord``
    here, the master index on the eager side — so this needs no new source.
    """

    dimension: str
    op: ComparisonOperator
    position: int
    by: str | None = None


@dataclass(frozen=True)
class LookupComparison(Predicate):
    """Compare a *lookup's values* to a literal — ``where: "period_of == 2030"``.

    Read off the ``over`` dimension's own table, one column beside the labels,
    so this is the same pointwise filter :class:`DimensionComparison` is —
    one join against a small table rather than a data join.
    """

    lookup: str
    over: str
    op: ComparisonOperator
    value: float | str | datetime.date


@dataclass(frozen=True)
class LookupPairComparison(Predicate):
    """Compare two lookups over one dimension — ``where: "from != to"``.

    Both columns sit on the same dim table, so this is one filter on two of
    its columns and never a join between them.
    """

    lookup: str
    other: str
    over: str
    op: ComparisonOperator


@dataclass(frozen=True)
class LookupDefined(Predicate):
    """True where a partial lookup has a value — a label that maps somewhere."""

    lookup: str
    over: str


@dataclass(frozen=True)
class ParameterDefined(Predicate):
    """True where the parameter has a non-null, finite value."""

    parameter: str


@dataclass(frozen=True)
class VariableDefined(Predicate):
    """True at the coordinates where the variable exists.

    A semi-join against the variable's own frame — pointwise, and the same
    shape as any mapping-table join.
    """

    variable: str


@dataclass(frozen=True)
class BooleanConstant(Predicate):
    """Constant predicate (``BooleanConstant(False)`` masks out every row)."""

    value: bool


@dataclass(frozen=True)
class And(Predicate):
    left: Predicate
    right: Predicate


@dataclass(frozen=True)
class Or(Predicate):
    left: Predicate
    right: Predicate


@dataclass(frozen=True)
class Not(Predicate):
    operand: Predicate


# --------------------------------------------------------------------------
# Declarations
# --------------------------------------------------------------------------


class LookupDeclaration(NamedTuple):
    """One declared lookup and the dimension its values are labels of."""

    name: str
    target: str


@dataclass(frozen=True)
class DimensionDeclaration:
    """A dimension and the lookups its labels carry.

    ``lookups`` names each lookup and the dimension its values are labels of,
    checked for containment once the dim tables exist — which keeps a mistyped
    label from silently dropping its terms in the join that places them.

    ``label_spaces`` are the inline kind: maps the dimension owns outright,
    with no target and so nothing to check. They are read for selection and
    rendering, and resolution refuses to group into one, so no expression node
    reaches them.
    """

    name: str
    lookups: tuple[LookupDeclaration, ...] = ()
    label_spaces: tuple[str, ...] = ()
    #: What the labels are, as the file declares them. A dimension is read from
    #: whatever table carries it, so the declared type is what that column is
    #: checked against — the same claim ``ParameterDeclaration.dtype`` makes
    #: about a value column, one axis over.
    dtype: str = 'str'

    @property
    def maps(self) -> list[str]:
        """Every map over the dimension, targeted and label-space alike.

        What binding needs a relation for: both kinds are read by a ``where``
        and both arrive the same way, and only the targeted ones have a label
        set to be checked against.
        """
        return sorted([*(lk.name for lk in self.lookups), *self.label_spaces])

    @property
    def targets(self) -> dict[str, str]:
        """Each targeted map over the dimension, to the dimension its values are labels of.

        The question every consumer of a ``by=`` asks, and asked here so it has
        one answer: an operator grouping through a lookup names the target as
        the dim it lands on, and a partition array is named for it so an amount
        declared over the group's own dim can be read through it.
        """
        return {lk.name: lk.target for lk in self.lookups}


@dataclass(frozen=True)
class ParameterDeclaration:
    """Shape declaration; data is bound at execution time by name.

    ``dtype`` is what the declaration claims the values are, and binding
    refuses a column that is not it — so the engine reads the *declaration*
    where it used to read the column, and the two cannot disagree.
    """

    name: str
    dims: tuple[str, ...]
    dtype: str = 'float'


@dataclass(frozen=True)
class VariableDeclaration:
    name: str
    dims: tuple[str, ...]
    where: Predicate | None = None
    lower: Expression = field(default_factory=lambda: Constant(float('-inf')))
    upper: Expression = field(default_factory=lambda: Constant(float('inf')))
    variable_type: VariableType = 'continuous'
    absence: VariableAbsence = 'undefined'


@dataclass(frozen=True)
class ConstraintDeclaration:
    """``lhs sense rhs`` for each coord combination of ``dims``.

    Both sides are affine; the engine normalises constants to the RHS.
    ``where`` masks out coord combinations (row absence, like variables).
    """

    name: str
    dims: tuple[str, ...]
    lhs: Expression
    sense: ConstraintSense
    rhs: Expression
    where: Predicate | None = None


@dataclass(frozen=True)
class SosDeclaration:
    """One special-ordered set per coordinate of the variable's ``foreach`` minus ``over``.

    The only declaration that adds neither a column nor a row: it names
    columns a sink already has and says what may be nonzero among them. Which
    dims those are is the variable's own ``foreach`` and is read from it: a
    copy here would be a second home for a fact
    (:meth:`Program.variable`).

    ``big_m`` caps the linking coefficient a sink without the concept
    reformulates with, and is ``None`` where the variable's own upper bound is
    the only cap.
    """

    name: str
    variable: str
    over: str
    sos_type: Literal[1, 2]
    big_m: float | None = None


@dataclass(frozen=True)
class ObjectiveDeclaration:
    """Objective; dims remaining after explicit Sums are implicitly summed."""

    sense: ObjectiveSense
    expression: Expression


_Declaration = TypeVar('_Declaration', ParameterDeclaration, VariableDeclaration, ConstraintDeclaration)


def _declared(items: tuple[_Declaration, ...], name: str, kind: str) -> _Declaration:
    """The declaration called *name*, or a ``KeyError`` naming the near miss."""
    for item in items:
        if item.name == name:
            return item
    raise KeyError(unknown_name_message(kind, name, (i.name for i in items)))


@dataclass(frozen=True)
class Program:
    """A complete linear program over named tidy tables."""

    parameters: tuple[ParameterDeclaration, ...]
    variables: tuple[VariableDeclaration, ...]
    constraints: tuple[ConstraintDeclaration, ...]
    #: ``None`` where the file declares no objective — a feasibility problem,
    #: whose answer is whether the constraints can be met at all.
    objective: ObjectiveDeclaration | None
    dimensions: tuple[DimensionDeclaration, ...] = ()
    sos: tuple[SosDeclaration, ...] = ()
    #: Declared ``expressions:``, lowered. Not part of the program a solver
    #: sees — none of them builds a row — but lowered with it, so a file whose
    #: named expression is outside the language is refused by every verb that
    #: reads the file rather than only by the one that reads the expression.
    #: Keyed rather than a tuple of declarations because a reader asks for one
    #: by the name it wrote, and nothing iterates them in order.
    expressions: dict[str, Expression] = field(default_factory=dict)

    @classmethod
    def from_model(cls, model: Buildable) -> Program:
        """Compile an expanded model into the plan both lanes build from.

        The one way in. Takes the expanded model rather than expanding one: a
        plan is built from declarations, and ``Buildable`` is the type that
        guarantees they are all there.

        A ``domain: binary`` variable lowers with fixed 0/1 bounds, matching
        linopy's ``binary=True``.

        Raises:
            LanguageError: A construct outside the streaming language, named
                with its rewrite.
        """
        return _lower_program(model)

    @property
    def advice(self) -> list[str]:
        """Modelling notes ``check`` surfaces as warnings — never errors.

        A property rather than a pass a caller runs: it reads nothing but this
        plan, so there is no order to get wrong and nothing to pass it.
        """
        return _advice(self)

    def dimension(self, name: str) -> DimensionDeclaration:
        """The dimension called *name*.

        Undeclared is not an error here: a dimension with no lookups has
        nothing to declare.
        """
        for d in self.dimensions:
            if d.name == name:
                return d
        return DimensionDeclaration(name)

    @property
    def lookups(self) -> tuple[tuple[str, LookupDeclaration], ...]:
        """Every targeted map in the program, with the dimension it is over.

        One walk for the several shapes consumers want it in — name to target,
        target to origin, the set of targets — because the nested comprehension
        that produces any of them is the same walk written again.
        """
        return tuple((d.name, lk) for d in self.dimensions for lk in d.lookups)

    def parameter(self, name: str) -> ParameterDeclaration:
        return _declared(self.parameters, name, 'parameter')

    def variable(self, name: str) -> VariableDeclaration:
        return _declared(self.variables, name, 'variable')


def is_quadratic(expression: Expression) -> bool:
    """Whether *expression* contains a product of two variable-carrying operands.

    A structural question over the plan, asked by three unrelated callers — the
    capability a program requires, which declarations the engine builds last,
    and the compiler's own ceiling — so it is answered once here beside the
    other walks rather than three times in their own terms.

    Degree is the *language's* verdict (``language/degree.py``) and this is not
    a second opinion on it: by the time a plan exists the question is no longer
    "may this be written" but "which shape is it", and only the plan is in
    hand to answer it.
    """
    if isinstance(expression, Multiply) and all(carries_variable(x) for x in (expression.left, expression.right)):
        return True
    return any(is_quadratic(child) for child in children(expression))


def declares_quadratic(c: ConstraintDeclaration) -> bool:
    """Whether constraint *c*'s expression multiplies two variable-carrying operands.

    One home, because two readers act on it — the capability a program
    requires, and which declarations the engine builds last — and a third side
    added to a constraint has to be found by both.
    """
    return is_quadratic(c.lhs) or is_quadratic(c.rhs)


def carries_variable(expression: Expression) -> bool:
    """Whether a variable appears anywhere under *expression*."""
    if isinstance(expression, Variable):
        return True
    return any(carries_variable(child) for child in children(expression))


def parameters_of(*expressions: Expression) -> frozenset[str]:
    """Every parameter named anywhere under *expressions*."""
    found: set[str] = set()

    def walk(e: Expression) -> None:
        if isinstance(e, Parameter):
            found.add(e.name)
        for child in children(e):
            walk(child)

    for e in expressions:
        walk(e)
    return frozenset(found)


def variables_of(*expressions: Expression) -> frozenset[str]:
    """Every variable named anywhere under *expressions*."""
    found: set[str] = set()

    def walk(e: Expression) -> None:
        if isinstance(e, Variable):
            found.add(e.name)
        for child in children(e):
            walk(child)

    for e in expressions:
        walk(e)
    return frozenset(found)


def quotients(*expressions: Expression) -> tuple[Divide, ...]:
    """Every division under *expressions*, each kept whole.

    The divisor and the numerator answer different questions and one consumer
    needs them paired: a divisor is judged against the rows the declaration
    builds *narrowed by the variables in its own numerator*, which the flat
    :func:`divisor_parameters` cannot say.
    """
    found: list[Divide] = []

    def walk(e: Expression) -> None:
        if isinstance(e, Divide):
            found.append(e)
        for child in children(e):
            walk(child)

    for e in expressions:
        walk(e)
    return tuple(found)


def divisor_parameters(*expressions: Expression) -> frozenset[str]:
    """Parameters appearing anywhere in a divisor position.

    Static, like :func:`parameters_of`: which names *can* reach a divisor is
    the plan's to answer, and *where* they must have values is decided by the
    rows a declaration builds.
    """
    return frozenset().union(*(parameters_of(q.divisor) for q in quotients(*expressions)))


# ---------------------------------------------------------------------------
# The pass: a resolved model in, a Program out
# ---------------------------------------------------------------------------


def _lower_program(schema: Buildable) -> Program:
    """Compile a :class:`Buildable` into a :class:`Program`.

    Takes the expanded model rather than expanding one: a plan is built from
    declarations, and `Buildable` is the type that guarantees they are all
    there. Every caller already held one — the expansion is memoised on the
    model — so this moves no work, it only stops the guarantee being a
    convention four consumers happened to observe.

    A ``domain: binary`` variable lowers with fixed 0/1 bounds, matching
    linopy's ``binary=True``.

    Raises:
        LanguageError: A construct outside the streaming language, named with
            its rewrite.
    """
    expanded = schema
    ns = Namespace.of(expanded)
    parameters = tuple(
        ParameterDeclaration(name, tuple(pdef.dims), pdef.dtype) for name, pdef in expanded.parameters.items()
    )

    variables = []
    for vname, vdef in expanded.variables.items():
        variable_type = cast('VariableType', vdef.domain)
        if variable_type == 'binary':
            lower, upper = Constant(0.0), Constant(1.0)
        else:
            lower, upper = _bound_expression(vdef.bounds.lower), _bound_expression(vdef.bounds.upper)
        variables.append(
            VariableDeclaration(
                vname,
                tuple(vdef.foreach),
                where=_lower_where(vdef.where, ns, f"variable '{vname}'", self_variable=vname),
                lower=lower,
                upper=upper,
                variable_type=variable_type,
                absence=cast('VariableAbsence', vdef.absence),
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
        lowering = _Lowering(expanded, f"constraint '{cname}'", ceiling=2)
        constraints.append(
            ConstraintDeclaration(
                cname,
                tuple(cdef.foreach),
                lhs=lowering.expr(ast.left),
                sense=ast.op,
                rhs=lowering.expr(ast.right),
                where=where,
            )
        )

    objective = None
    if (odef := expanded.objective) is not None:
        ast = expression_of(odef.expression, expanded, ns, 'the objective')
        if isinstance(ast, ComparisonNode):
            raise LanguageError('the objective: expression must not contain a comparison operator')
        objective = ObjectiveDeclaration(
            'min' if odef.sense == 'minimize' else 'max',
            _Lowering(expanded, 'the objective', ceiling=2).expr(ast),
        )

    dimensions = tuple(
        DimensionDeclaration(
            dname,
            tuple(LookupDeclaration(cname, target) for cname, target in expanded.targeted_of(dname).items()),
            tuple(expanded.labels_of(dname)),
            ddef.dtype,
        )
        for dname, ddef in expanded.dimensions.items()
    )
    sos = tuple(
        SosDeclaration(
            sname,
            sdef.variable,
            sdef.over,
            sos_type=cast('Literal[1, 2]', sdef.type),
            big_m=sdef.big_m,
        )
        for sname, sdef in expanded.sos.items()
    )
    expressions = {name: _lower_expression(expanded, name) for name in expanded.expressions}
    return Program(parameters, tuple(variables), tuple(constraints), objective, dimensions, sos, expressions)


def _lower_expression(schema: Buildable, name: str) -> Expression:
    """Compile the named expression *name* into a plan expression.

    Raises:
        KeyError: No named expression called *name*.
        LanguageError: A construct outside the streaming language.
    """
    expanded = schema
    context = f"named expression '{name}'"
    ns = Namespace.of(expanded)
    ast = expression_of(expanded.expressions[name].expression, expanded, ns, context)
    assert not isinstance(ast, ComparisonNode), 'load-time validation refuses a comparison in a named expression'
    return _Lowering(expanded, context).expr(ast)


# ---------------------------------------------------------------------------
# expression lowering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Lowering:
    """One expression walk, and the three things every step of it reads.

    ``schema``, ``context`` and ``ceiling`` are fixed for a whole walk: the
    ceiling is chosen once by the position being lowered — 2 inside a
    constraint or the objective, 1 elsewhere — and the other two never vary at
    all. So they are the walk's state rather than three arguments every
    recursion and every check repeats. Extend this rather than adding a
    parameter to :meth:`expr` and every operator seam, which is the rule
    ``linopy/builder.py``'s ``EvaluationContext`` already states for the other
    lane.

    Frozen, because a walk that could change its own ceiling half way through
    would be a degree rule no file states.
    """

    schema: Buildable
    context: str
    ceiling: int = 1

    def expr(self, node: ArithmeticNode) -> Expression:
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
            return Constant(node.value)

        if isinstance(node, VariableNode):
            return Variable(node.name)

        if isinstance(node, ParameterNode):
            return Parameter(node.name)

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
            inner = self.expr(node.operand)
            return Negate(inner) if node.op == '-' else inner

        if isinstance(node, BinaryOperatorNode):
            left = self.expr(node.left)
            right = self.expr(node.right)
            check_binary(node, self.context, ceiling=self.ceiling)
            match node.op:
                case '+':
                    return Add(left, right)
                case '-':
                    return Add(left, Negate(right))
                case '*':
                    return Multiply(left, right)
                case '/':
                    return Divide(left, right)
                case '**':
                    return Power(left, right)
                case _:  # pragma: no cover — check_binary refuses every other operator
                    raise AssertionError(f'{self.context}: operator {node.op!r} passed the degree check')

        if isinstance(node, FunctionCallNode):
            shape_error = call_shape_error(node.name, len(node.args), node.kwargs)
            if shape_error is not None:
                raise LanguageError(f'{self.context}: {shape_error}')
            try:
                lower_call = _CALLS[node.name]
            except KeyError:
                raise LanguageError(f"{self.context}: built-in '{node.name}' declares no lowering case") from None
            return lower_call(self, node)

        assert_never(node)

    def sum(self, node: FunctionCallNode) -> Expression:
        """``sum(x)``, ``sum(x, over=d)`` or ``sum(x, by=lookup)``.

        Two plan nodes under one surface verb: reducing a dim away and reducing it
        *into* another are different relational shapes, so ``by=`` decides which
        before anything else is read.
        """
        by_node = node.kwargs.get('by')
        self._dim_rules(node)
        operand = self.expr(node.args[0])
        if by_node is None and 'over' not in node.kwargs:
            return Sum(operand, tuple(sorted(dims_of(node.args[0], self.schema, self.context))))
        if by_node is None:
            over_node = node.kwargs['over']
            if not isinstance(over_node, DimensionNode):
                raise LanguageError(f'{self.context}: sum(over=...) must name a dimension')
            return Sum(operand, (over_node.name,))
        if not isinstance(by_node, LookupNode):
            raise LanguageError(f'{self.context}: sum(by=...) must name a lookup')
        return GroupSum(operand, over=by_node.dimension, coordinate=by_node.names, into=by_node.into)

    def at(self, node: FunctionCallNode) -> Expression:
        """``at(x, by=lookup)`` — the adjoint of :meth:`sum`'s ``by=`` form."""
        by_node = node.kwargs['by']
        if not isinstance(by_node, LookupNode):
            raise LanguageError(f'{self.context}: at(by=...) must name a lookup')
        self._dim_rules(node)
        return At(
            self.expr(node.args[0]),
            over=by_node.dimension,
            coordinate=by_node.names,
            into=by_node.into,
        )

    def sum_back(self, node: FunctionCallNode) -> Expression:
        """``sum_back(x, over=d, within=w)`` — a trailing window along one dimension.

        *within* is an integer literal of at least one, or a parameter naming a
        per-entity width, which the language holds to the two rules that make it
        mean one thing before this is reached.

        ``by=`` names the lookup the window stops at the edges of, and rides on
        the node the way it rides on a translation — the dim rules have already
        held it to one lookup over the walked dimension.
        """
        over_node = node.kwargs['over']
        if not isinstance(over_node, DimensionNode):
            raise LanguageError(f'{self.context}: sum_back(over=...) must name a dimension')
        within_node = node.kwargs['within']
        self._dim_rules(node)
        operand = self.expr(node.args[0])
        wrap = _window_edge(node.kwargs.get('edge'), self.context)
        width: int | str
        if isinstance(within_node, ParameterNode):
            width = within_node.name
        elif (
            isinstance(within_node, NumberNode)
            and within_node.value >= 1
            and int(within_node.value) == within_node.value
        ):
            width = int(within_node.value)
        else:
            raise LanguageError(f'{self.context}: {_window_width_message()}')
        return Window(operand, over_node.name, width=width, wrap=wrap, partition=_partition_of(node))

    def shift(self, node: FunctionCallNode) -> Expression:
        """``shift(x, over=d, offset=n)`` — the value at *t - offset* along one dim.

        The longest of the four because *offset* and *edge* are read together:
        what the vacated positions contribute decides whether a named offset is
        sayable at all, so it is settled before the offset is read.
        """
        over_node = node.kwargs['over']
        if not isinstance(over_node, DimensionNode):
            raise LanguageError(f'{self.context}: shift(over=...) must name a dimension')
        partition = _partition_of(node)
        by_node = node.kwargs['offset']
        sign = 1
        if isinstance(by_node, UnaryOperatorNode) and by_node.op == '-':
            sign, by_node = -1, by_node.operand
        if not isinstance(by_node, ParameterNode) and (
            not isinstance(by_node, NumberNode) or int(by_node.value) != by_node.value
        ):
            raise LanguageError(f'{self.context}: {_shift_by_message()}')
        self._dim_rules(node)
        operand = self.expr(node.args[0])
        has_var = ast_carries_variable(node.args[0])
        edge = node.kwargs.get('edge')
        wrap = isinstance(edge, EdgeNode)
        fill = None if wrap else _translate_fill(edge, self.context, has_var=has_var)
        if not wrap and fill is None and not has_var:
            raise LanguageError(_shift_over_data_message(self.context))
        by: int | str
        if isinstance(by_node, ParameterNode):
            if not wrap and fill is None:
                raise LanguageError(f'{self.context}: {_named_offset_edge_message(by_node.name)}')
            by = by_node.name
        else:
            assert isinstance(by_node, NumberNode)
            by = sign * int(by_node.value)
        return Translate(operand, over_node.name, offset=by, wrap=wrap, fill=fill, partition=partition)

    def _dim_rules(self, node: FunctionCallNode) -> None:
        """Apply the language's dim rules to an operator call, discarding the dim set.

        Lowering wants the *raise*, not the answer. Called after the plan-shape
        checks so those speak first, and only for one call's dims — the enclosing
        frame is ``dimensions.check_schema``'s business.
        """
        dims_of(node, self.schema, self.context)


#: One lowering per name in the language's ``BUILTIN_NAMES``. A table rather
#: than a chain of ``if``s because the set is *closed* — nothing registers into
#: it, and a name the language declares with no entry here is the failure
#: ``tests/test_architecture.py`` looks for. Each method is named for the
#: operator it lowers, so the table reads as the identity it nearly is.
_CALLS: dict[str, Callable[[_Lowering, FunctionCallNode], Expression]] = {
    'sum': _Lowering.sum,
    'at': _Lowering.at,
    'sum_back': _Lowering.sum_back,
    'shift': _Lowering.shift,
}


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
    checked with the other dim rules (``math_spec.dimensions``), where a model
    is refused before any data is read.
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


def _named_offset_edge_message(name: str) -> str:
    """Why a named offset must say what the vacated positions contribute.

    The absent edge propagates through a presence frame keyed by the translated
    dimension alone, and a per-entity offset vacates a different slot for each
    entity — which that frame cannot say. Refused rather than answered wrongly
    (#850); the two edges that write their own answer are allowed.
    """
    return (
        f'shift(offset={name}) leaves the vacated positions absent, which a '
        f'per-entity offset cannot say yet.\n'
        f"Add edge='wrap' for a cyclic translation, or edge=<number> for what the "
        f'vacated positions contribute.'
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


def _bound_expression(value: float | str) -> Expression:
    if isinstance(value, str):
        return Parameter(value)
    return Constant(value)


# ---------------------------------------------------------------------------
# where lowering
# ---------------------------------------------------------------------------


def _lower_where(text: str | None, ns: Namespace, context: str, self_variable: str | None = None) -> Predicate | None:
    """Lower a where string to a plan predicate, ``None`` when there is no mask.

    A predicate that resolves to the constant ``True`` is dropped too: it is
    equivalent to no mask.
    """
    node = where_of(text, ns, context, self_variable)
    if node is None:
        return None
    pred = _lower_where_node(node, context)
    if isinstance(pred, BooleanConstant) and pred.value:
        return None
    return pred


def _lower_where_node(node: WhereNode, context: str) -> Predicate:
    if isinstance(node, BooleanLiteralNode):
        return BooleanConstant(node.value)

    if isinstance(node, ParameterDefinedNode):
        return ParameterDefined(node.name)

    if isinstance(node, VariableDefinedNode):
        return VariableDefined(node.name)

    if isinstance(node, ParameterComparisonNode):
        return ParameterComparison(node.name, node.op, node.value)

    if isinstance(node, DimensionComparisonNode):
        return DimensionComparison(node.name, node.op, node.value)

    if isinstance(node, DimensionPositionNode):
        return DimensionPosition(node.name, node.op, node.position, node.by)

    if isinstance(node, LookupComparisonNode):
        return LookupComparison(node.name, node.over, node.op, node.value)

    if isinstance(node, LookupPairComparisonNode):
        return LookupPairComparison(node.name, node.other, node.over, node.op)

    if isinstance(node, LookupDefinedNode):
        return LookupDefined(node.name, node.over)

    if isinstance(node, (UnresolvedNameNode, UnresolvedComparisonNode, UnresolvedPositionNode)):
        msg = (
            f'{type(node).__name__} reached lowering unresolved. Where strings '
            f'must go through math_spec.where_of() first.'
        )
        raise AssertionError(msg)

    if isinstance(node, NotNode):
        return Not(_lower_where_node(node.operand, context))
    if isinstance(node, (AndNode, OrNode)):
        node_type = And if isinstance(node, AndNode) else Or
        return node_type(
            _lower_where_node(node.left, context),
            _lower_where_node(node.right, context),
        )

    assert_never(node)


# ---------------------------------------------------------------------------
# check-time advice
# ---------------------------------------------------------------------------


def _advice(program: Program) -> list[str]:
    """Modeling advice ``check`` surfaces as warnings — never errors.

    Every dimension should be an axis — indexed by something, or aggregated
    into. Each one that is neither gets a note: one that only serves as a
    lookup's target is a label space and the note says how to declare it as
    one; one nothing reaches at all is unused. A warning rather than an error
    because it reads intent: a dimension declared ahead of the declarations
    that will use it is a model part-written, not a wrong one.

    Returns:
        One note per dimension that is never an axis, in declaration order.
    """
    axes: set[str] = set()
    for declaration in (*program.parameters, *program.variables, *program.constraints):
        axes.update(declaration.dims)
    expressions = [program.objective.expression] if program.objective is not None else []
    expressions.extend(side for c in program.constraints for side in (c.lhs, c.rhs))
    for e in expressions:
        axes |= _produced_axes(e)

    targeted = {lk.target: (dimension, lk.name) for dimension, lk in program.lookups}
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


def _produced_axes(e: Expression) -> set[str]:
    """The axes an expression *creates*, beyond what its declarations index.

    ``sum(by=)`` lands terms on its target and ``at()`` spreads onto its fine
    dimension, so both are axes even when no declaration is indexed by them —
    an objective may group into a dimension and then implicitly sum it away.
    """
    out: set[str] = set()
    if isinstance(e, GroupSum):
        out |= set(e.into)
    if isinstance(e, At):
        out.add(e.over)
    for child in children(e):
        out |= _produced_axes(child)
    return out
