"""The logical plan: relational LP construction, one step above SQL.

An intermediate representation in the compiler sense — the module is named
for what it *is* to this engine (duckdb, Calcite and Spark all call this
shape a logical plan) rather than for the generic category.

The lane is described in docs/about/architecture.md, "The relational lane".

Frozen dataclasses only — no execution logic, no engine imports. A `Program`
is a complete declarative description of a linear program over named tidy
tables; actual data is bound at execution time via a source registry.

Expressions support operator sugar so plans read naturally in Python:

    balance = GroupSum(Variable("p"), over="generator", coordinate=("bus",), into=("bus",)) - Parameter("load")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NamedTuple, TypeVar

from lpspec.errors import unknown_name_message

if TYPE_CHECKING:
    import datetime

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

    @property
    def maps(self) -> list[str]:
        """Every map over the dimension, targeted and label-space alike.

        What binding needs a relation for: both kinds are read by a ``where``
        and both arrive the same way, and only the targeted ones have a label
        set to be checked against.
        """
        return sorted([*(lk.name for lk in self.lookups), *self.label_spaces])


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

    def dimension(self, name: str) -> DimensionDeclaration:
        """The dimension called *name*.

        Undeclared is not an error here: a dimension with no lookups has
        nothing to declare.
        """
        for d in self.dimensions:
            if d.name == name:
                return d
        return DimensionDeclaration(name)

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


def divisor_parameters(*expressions: Expression) -> frozenset[str]:
    """Parameters appearing anywhere in a divisor position.

    Static, like :func:`parameters_of`: which names *can* reach a divisor is
    the plan's to answer, and *where* they must have values is decided by the
    rows a declaration builds.
    """
    found: set[str] = set()

    def walk(e: Expression) -> None:
        if isinstance(e, Divide):
            found.update(parameters_of(e.divisor))
        for child in children(e):
            walk(child)

    for e in expressions:
        walk(e)
    return frozenset(found)
