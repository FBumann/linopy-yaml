"""The logical plan: relational LP construction, one step above SQL.

An intermediate representation in the compiler sense — the module is named
for what it *is* to this engine (duckdb, Calcite and Spark all call this
shape a logical plan) rather than for the generic category.

The lane is described in docs/about/architecture.md, "The relational lane".

Frozen dataclasses only — no execution logic, no engine imports. A `Program`
is a complete declarative description of a linear program over named tidy
tables; actual data is bound at execution time via a source registry, and
`lowering.py` is what builds one from a resolved model.

Expressions support operator sugar so plans read naturally in Python:

balance = GroupSum(Variable("p"), over="generator", coordinate=("bus",), into=("bus",)) - Parameter("load")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple, TypeVar

from math_spec import LanguageError

from lpspec.errors import unknown_name_message

if TYPE_CHECKING:
    import datetime
    from collections.abc import Iterator


ConstraintSense = Literal['==', '<=', '>=']

#: How a shape operator's output rows relate to its input slots. ``sum`` and
#: ``sum(by=)`` are many-to-one and a window one-to-many, so an output row
#: mixes several input slots; a pullback and a translation are one-to-one —
#: one output, one input. Declared on the node because two consumers deciding
#: it separately was a bug: the compiler's absence pass kept its own list,
#: ``Window`` was missing from it, and the lanes disagreed about a constant
#: at a masked slot (#1142). The fan-in is the rule, and the node states it.
FanIn = Literal['one-to-one', 'many-to-one', 'one-to-many']
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

    def __add__(self: ExpressionNode, other: ExpressionNode) -> ExpressionNode:
        return Add(self, other)

    def __sub__(self: ExpressionNode, other: ExpressionNode) -> ExpressionNode:
        return Add(self, Negate(other))

    def __mul__(self: ExpressionNode, other: ExpressionNode) -> ExpressionNode:
        return Multiply(self, other)

    def __neg__(self: ExpressionNode) -> ExpressionNode:
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
    operand: ExpressionNode


@dataclass(frozen=True)
class Add(Expression):
    left: ExpressionNode
    right: ExpressionNode


@dataclass(frozen=True)
class Multiply(Expression):
    """Product of two operands.

    Affine where at least one factor is variable-free. **Degree 2 where neither
    is**, which the language allows in the objective alone
    (``language/degree.py``) — so a consumer that cannot represent a quadratic
    term is told which position it is compiling rather than assuming it.
    """

    left: ExpressionNode
    right: ExpressionNode


@dataclass(frozen=True)
class Power(Expression):
    """``base ** exponent``, both variable-free.

    Degree 0 in variables wherever it appears, so no consumer has to ask what
    position it stands in: the language refuses a variable anywhere under it
    (``language/degree.py``), which is what lets this fold to one number per
    coordinate like any other parameter arithmetic.
    """

    base: ExpressionNode
    exponent: ExpressionNode


@dataclass(frozen=True)
class Divide(Expression):
    """Quotient ``numerator / divisor``. The divisor must be variable-free."""

    numerator: ExpressionNode
    divisor: ExpressionNode


@dataclass(frozen=True)
class Sum(Expression):
    """Sum ``operand`` over the named dims, removing them from the result."""

    fan_in: ClassVar[FanIn] = 'many-to-one'

    operand: ExpressionNode
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

    fan_in: ClassVar[FanIn] = 'many-to-one'

    operand: ExpressionNode
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

    fan_in: ClassVar[FanIn] = 'one-to-one'

    operand: ExpressionNode
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

    fan_in: ClassVar[FanIn] = 'one-to-one'

    operand: ExpressionNode
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

    fan_in: ClassVar[FanIn] = 'one-to-many'

    operand: ExpressionNode
    dimension: str
    width: int | str
    wrap: bool = False
    partition: str | None = None


#: Every expression node, as one type. The set is *closed* — nothing registers
#: into it — so a consumer that walks it ends in ``assert_never`` and a node
#: added without a branch is a type error at the site that must grow one,
#: rather than a ``LanguageError`` raised at the first model that uses it.
#: ``Expression`` stays the base class the nodes inherit and the operators are
#: declared on; this is what a walk *takes*.
ExpressionNode = (
    Constant
    | Parameter
    | Variable
    | Negate
    | Add
    | Multiply
    | Power
    | Divide
    | Sum
    | GroupSum
    | At
    | Translate
    | Window
)


def children(expression: ExpressionNode) -> tuple[ExpressionNode, ...]:
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
    lower: ExpressionNode = field(default_factory=lambda: Constant(float('-inf')))
    upper: ExpressionNode = field(default_factory=lambda: Constant(float('inf')))
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
    lhs: ExpressionNode
    sense: ConstraintSense
    rhs: ExpressionNode
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
    expression: ExpressionNode


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
    expressions: dict[str, ExpressionNode] = field(default_factory=dict)

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

    def check(self) -> None:
        """Refuse this program unless every declaration is internally coherent.

        The boundary a hand-built program crosses where a lowered one passes
        by construction: both lanes call it where a program enters, so a
        malformed plan fails there, in plan vocabulary, rather than mid-query
        in whatever error a compiler hits first (#1134). What data alone can
        answer stays with binding: whether a lookup's source arrives, whether
        a parameter covers the rows that read it.

        Raises:
            LanguageError: A name no declaration carries, two declarations
                sharing one name, an expression whose shape no operator
                produces — a reduction over a dimension its operand does not
                span, a grouping into a dimension that is not its lookup's
                declared target, a translation whose distance varies along the
                walked dimension — a degree above the position's ceiling, or a
                bound that is not variable-free arithmetic.
        """
        _check_program(self)


def walk(*expressions: ExpressionNode) -> Iterator[ExpressionNode]:
    """Every node under *expressions*, each expression itself included, parents first.

    The traversal every *question* about a plan is a filter of — which names
    it mentions, whether a variable stands under it, which divisions it
    contains. One generator rather than that five-line recursion once per
    question: how a plan is traversed is one fact, so a node kind
    :func:`children` learns to descend into reaches every caller at once
    rather than the callers that remembered.
    """
    for expression in expressions:
        yield expression
        yield from walk(*children(expression))


def is_quadratic(expression: ExpressionNode) -> bool:
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
    return any(
        isinstance(node, Multiply) and all(carries_variable(side) for side in (node.left, node.right))
        for node in walk(expression)
    )


def declares_quadratic(c: ConstraintDeclaration) -> bool:
    """Whether constraint *c*'s expression multiplies two variable-carrying operands.

    One home, because two readers act on it — the capability a program
    requires, and which declarations the engine builds last — and a third side
    added to a constraint has to be found by both.
    """
    return is_quadratic(c.lhs) or is_quadratic(c.rhs)


def carries_variable(expression: ExpressionNode) -> bool:
    """Whether a variable appears anywhere under *expression*."""
    return any(isinstance(node, Variable) for node in walk(expression))


def parameters_of(*expressions: ExpressionNode) -> frozenset[str]:
    """Every parameter named anywhere under *expressions*."""
    return frozenset(node.name for node in walk(*expressions) if isinstance(node, Parameter))


def variables_of(*expressions: ExpressionNode) -> frozenset[str]:
    """Every variable named anywhere under *expressions*."""
    return frozenset(node.name for node in walk(*expressions) if isinstance(node, Variable))


def quotients(*expressions: ExpressionNode) -> tuple[Divide, ...]:
    """Every division under *expressions*, each kept whole.

    The divisor and the numerator answer different questions and one consumer
    needs them paired: a divisor is judged against the rows the declaration
    builds *narrowed by the variables in its own numerator*, which the flat
    :func:`divisor_parameters` cannot say.
    """
    return tuple(node for node in walk(*expressions) if isinstance(node, Divide))


def divisor_parameters(*expressions: ExpressionNode) -> frozenset[str]:
    """Parameters appearing anywhere in a divisor position.

    Static, like :func:`parameters_of`: which names *can* reach a divisor is
    the plan's to answer, and *where* they must have values is decided by the
    rows a declaration builds.
    """
    return frozenset().union(*(parameters_of(q.divisor) for q in quotients(*expressions)))


# --------------------------------------------------------------------------
# Checking — the invariants :meth:`Program.check` holds a program to
# --------------------------------------------------------------------------


def _check_program(program: Program) -> None:
    names = _flat_namespace(program)
    for v in program.variables:
        context = f"variable '{v.name}'"
        _check_bound(v.lower, v.name, names)
        _check_bound(v.upper, v.name, names)
        _check_predicate(v.where, names, context)
    for c in program.constraints:
        context = f"constraint '{c.name}'"
        _check_predicate(c.where, names, context)
        for side in (c.lhs, c.rhs):
            _checked_dims(side, program, names, context)
            _check_degree(side, context)
    if program.objective is not None:
        _checked_dims(program.objective.expression, program, names, 'the objective')
        _check_degree(program.objective.expression, 'the objective')
    for name, expression in program.expressions.items():
        _checked_dims(expression, program, names, f"named expression '{name}'")
    for s in program.sos:
        v = program.variable(s.variable)
        if s.over not in v.dims:
            raise LanguageError(f"sos '{s.name}': over {s.over!r}, which variable '{v.name}' is not indexed by")


def _flat_namespace(program: Program) -> dict[str, tuple[str, ...]]:
    """Every name to the dims it is read through — the language's one flat namespace.

    A parameter and a variable sharing a name would shadow one another in
    every consumer that merges the two, so the collision is refused here; a
    name declared twice within a kind is the same silent shadowing one
    declaration deep.
    """
    names: dict[str, tuple[str, ...]] = {}
    for declaration in (*program.parameters, *program.variables):
        if declaration.name in names:
            raise LanguageError(f"'{declaration.name}' is declared twice — one flat namespace, one home per name")
        names[declaration.name] = declaration.dims
    return names


def _named(names: dict[str, tuple[str, ...]], name: str, kind: str, context: str) -> tuple[str, ...]:
    if name not in names:
        raise LanguageError(f'{context}: {unknown_name_message(kind, name, names)}')
    return names[name]


def _checked_dims(
    expression: ExpressionNode, program: Program, names: dict[str, tuple[str, ...]], context: str
) -> tuple[str, ...]:
    """The dims *expression* spans, derived bottom-up — refusing every incoherent node.

    The order is first appearance, the union rule every fragment join uses;
    what matters to callers is the set, so only membership is promised.
    """
    if isinstance(expression, Constant):
        return ()
    if isinstance(expression, Parameter):
        return _named(names, expression.name, 'parameter', context)
    if isinstance(expression, Variable):
        return _named(names, expression.name, 'variable', context)
    if isinstance(expression, Negate):
        return _checked_dims(expression.operand, program, names, context)
    if isinstance(expression, (Add, Multiply)):
        left = _checked_dims(expression.left, program, names, context)
        right = _checked_dims(expression.right, program, names, context)
        return left + tuple(d for d in right if d not in left)
    if isinstance(expression, Divide):
        num = _checked_dims(expression.numerator, program, names, context)
        div = _checked_dims(expression.divisor, program, names, context)
        return num + tuple(d for d in div if d not in num)
    if isinstance(expression, Power):
        base = _checked_dims(expression.base, program, names, context)
        exp = _checked_dims(expression.exponent, program, names, context)
        return base + tuple(d for d in exp if d not in base)
    if isinstance(expression, Sum):
        operand = _checked_dims(expression.operand, program, names, context)
        missing = [d for d in expression.over if d not in operand]
        if missing:
            raise LanguageError(f'{context}: sum over {missing}, which the operand does not span')
        return tuple(d for d in operand if d not in expression.over)
    if isinstance(expression, GroupSum):
        operand = _checked_dims(expression.operand, program, names, context)
        if expression.over not in operand:
            raise LanguageError(f'{context}: sum(by=) over {expression.over!r}, which the operand does not span')
        _check_mapping(expression, program, context)
        return (*(d for d in operand if d != expression.over), *expression.into)
    if isinstance(expression, At):
        operand = _checked_dims(expression.operand, program, names, context)
        missing = [d for d in expression.into if d not in operand]
        if missing:
            raise LanguageError(f'{context}: at() through {missing}, which the operand does not span')
        _check_mapping(expression, program, context)
        return (*(d for d in operand if d not in expression.into), expression.over)
    if isinstance(expression, (Translate, Window)):
        return _checked_walk_dims(expression, program, names, context)
    raise LanguageError(f'{context}: unsupported expression node {type(expression).__name__}')


def _checked_walk_dims(
    expression: Translate | Window, program: Program, names: dict[str, tuple[str, ...]], context: str
) -> tuple[str, ...]:
    """A translation or a window: the walked dimension spanned, its distance independent of it."""
    verb = 'shift' if isinstance(expression, Translate) else 'sum_back'
    operand = _checked_dims(expression.operand, program, names, context)
    if expression.dimension not in operand:
        raise LanguageError(f'{context}: {verb}() along {expression.dimension!r}, which the operand does not span')
    distance = expression.offset if isinstance(expression, Translate) else expression.width
    if isinstance(distance, str) and expression.dimension in _named(names, distance, 'parameter', context):
        raise LanguageError(
            f'{context}: {verb}() distance {distance!r} varies along {expression.dimension!r}, '
            f'the dimension being walked'
        )
    return operand


def _check_mapping(node: GroupSum | At, program: Program, context: str) -> None:
    """A grouping or a pullback pairs each lookup with the dimension it targets.

    A lookup the program does not declare is left to binding, which refuses a
    source that never arrives — declaring lookups on a dimension is optional
    in a hand-built program. A lookup declared with a *different* target is a
    contradiction no data can repair, and is refused here.
    """
    if len(node.coordinate) != len(node.into):
        raise LanguageError(
            f'{context}: {len(node.coordinate)} lookup(s) paired with {len(node.into)} target dimension(s)'
        )
    targets = {lk.name: lk.target for lk in program.dimension(node.over).lookups}
    for coordinate, into in zip(node.coordinate, node.into, strict=True):
        if coordinate in targets and targets[coordinate] != into:
            raise LanguageError(f'{context}: lookup {coordinate!r} targets {targets[coordinate]!r}, not {into!r}')


def _check_degree(expression: ExpressionNode, context: str) -> int:
    """The degree of *expression* in variables, refusing what no position takes.

    Two is the ceiling everywhere a whole expression stands — the objective
    and a constraint side — so one number serves both; a divisor or an
    exponent carrying a variable is refused where it stands, since no ceiling
    admits either.
    """
    if isinstance(expression, Variable):
        return 1
    if isinstance(expression, (Constant, Parameter)):
        return 0
    if isinstance(expression, Multiply):
        degree = _check_degree(expression.left, context) + _check_degree(expression.right, context)
        if degree > 2:
            raise LanguageError(f'{context}: a product of degree {degree} — nothing takes more than a quadratic form')
        return degree
    if isinstance(expression, Divide):
        if carries_variable(expression.divisor):
            raise LanguageError(f'{context}: the divisor contains variables')
        return _check_degree(expression.numerator, context)
    if isinstance(expression, Power):
        if carries_variable(expression.base) or carries_variable(expression.exponent):
            raise LanguageError(f'{context}: a power over variables — `**` takes neither side variable')
        return 0
    if isinstance(expression, Add):
        return max(_check_degree(expression.left, context), _check_degree(expression.right, context))
    return max((_check_degree(child, context) for child in children(expression)), default=0)


#: The node kinds a bound may be built from — what each lane's bound walk
#: evaluates, stated once where the boundary refuses the rest.
_BOUND_NODES = (Constant, Parameter, Negate, Add, Multiply)


def _check_bound(expression: ExpressionNode, variable: str, names: dict[str, tuple[str, ...]]) -> None:
    """A bound is variable-free arithmetic over constants and parameters."""
    context = f"bounds of variable '{variable}'"
    if not isinstance(expression, _BOUND_NODES):
        raise LanguageError(f'{context}: unsupported node {type(expression).__name__}')
    if isinstance(expression, Parameter):
        _named(names, expression.name, 'parameter', context)
    for child in children(expression):
        _check_bound(child, variable, names)


def _check_predicate(predicate: Predicate | None, names: dict[str, tuple[str, ...]], context: str) -> None:
    """Every name a mask reads resolves against a declaration."""
    if predicate is None:
        return
    if isinstance(predicate, (ParameterComparison, ParameterDefined)):
        _named(names, predicate.parameter, 'parameter', context)
    if isinstance(predicate, VariableDefined):
        _named(names, predicate.variable, 'variable', context)
    if isinstance(predicate, Not):
        _check_predicate(predicate.operand, names, context)
    if isinstance(predicate, (And, Or)):
        _check_predicate(predicate.left, names, context)
        _check_predicate(predicate.right, names, context)
