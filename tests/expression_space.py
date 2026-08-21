"""Every expression up to a bounded depth, and the rewrites that must not move it.

The AST is closed, which is what makes this different from sampling: at depth
three over two dimensions the space of spellings is *finite*, so the sweep in
``test_expression_sweep.py`` makes the strong claim — every one of them means
the same on both lanes — rather than "a hundred did".

**Built well-formed, not filtered.** Each node carries the dimensions and the
degree it produces, and a constructor refuses what the language would refuse:
summing over a dimension the operand does not carry, a product past degree two,
a constraint with no variable in it. Trial-loading the candidates instead cost
twenty seconds of collection, in every xdist worker, and turned a genuine load
error into one more rejected candidate. Here a load error is a failure.

**Rewrites are built, not found.** Depth bounds expression *size*, and the rules
worth checking need a *shape* — which gets rarer as the space grows, not
commoner. Enumerating to depth three and collecting the rules that happened to
fire yielded 2,460 commutativity pairs, which ``test_arithmetic_laws.py``
already states by hand, and **ten** of ``reduction-is-linear``, which is the one
the sweep exists for. So the rule-carrying shapes are constructed over an
operand pool instead, and the two commute rules are left to the curated file.

``reduction-is-linear`` is the rule with a side condition: ``sum(a + b)`` and
``sum(a) + sum(b)`` are equal only while every operand is *total*, and the
divergence when they are not went unnoticed for as long as the oracle shared it
(#311). The condition is read off the tree rather than assumed.

**Degree two goes in the objective, never a constraint row.** The eager lane
refuses a quadratic constraint outright (#942), so a degree-two expression in a
row is a lane limit rather than a disagreement, and generating one would make
the sweep report a gap it already knows about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

#: The two dimensions every expression is written over.
DIMS = {'f': {'values': ['a', 'b']}, 't': {'dtype': 'int', 'values': [0, 1]}}

#: `gate` masks `y`; `w` is a dense coefficient. The same fixture the curated
#: laws use, because a sweep that agrees with them has to be over one model.
DATA_SHAPE = {'gate': {'a': True}, 'w': {'a': 2.0, 'b': 3.0}}

#: The degree past which the language refuses a product.
MAX_DEGREE = 2


@dataclass(frozen=True)
class Node:
    """One expression, with what the language would say about it.

    Attributes:
        text: The expression as it is written in the model file.
        dims: The dimensions the expression carries.
        degree: 0 for data, 1 linear, 2 quadratic.
        total: Whether no masked variable appears anywhere inside.
    """

    text: str
    dims: frozenset[str]
    degree: int
    total: bool

    def __str__(self) -> str:
        return self.text


#: The leaves: a total variable, a masked one, a parameter, and a literal.
LEAVES = (
    Node('x', frozenset({'f', 't'}), 1, total=True),
    Node('y', frozenset({'f', 't'}), 1, total=False),
    Node('w', frozenset({'f'}), 0, total=True),
    Node('2', frozenset(), 0, total=True),
)


def negate(a: Node) -> Node | None:
    return Node(f'-({a})', a.dims, a.degree, a.total)


def add(a: Node, b: Node) -> Node | None:
    return Node(f'({a}) + ({b})', a.dims | b.dims, max(a.degree, b.degree), a.total and b.total)


def multiply(a: Node, b: Node) -> Node | None:
    """A product past degree two is not in the language, so it is not generated."""
    degree = a.degree + b.degree
    if degree > MAX_DEGREE:
        return None
    return Node(f'({a}) * ({b})', a.dims | b.dims, degree, a.total and b.total)


def summed(a: Node, over: str) -> Node | None:
    """Summing over a dimension the operand does not carry is a load error."""
    if over not in a.dims:
        return None
    return Node(f'sum({a}, over={over})', a.dims - {over}, a.degree, a.total)


def shifted(a: Node, over: str) -> Node | None:
    """Likewise for shifting along one — and the result is never total.

    With no ``edge=``, "the vacated edge is absent"
    (docs/reference/language/operators.md), so a shift introduces absence
    wherever it lands however total its operand was. Propagating ``total``
    through it made the generator emit ``reduction-is-linear`` pairs whose side
    condition does not hold, and thirty-four of them duly disagreed — the law is
    intact, the claim about the operand was not.
    """
    if over not in a.dims:
        return None
    return Node(f'shift({a}, over={over}, offset=1)', a.dims, a.degree, total=False)


#: Every way to grow an expression by one node. Ordered, so the enumeration is
#: the same list on every machine and a failing id can be found again.
UNARY: tuple[Callable[[Node], Node | None], ...] = (
    negate,
    lambda a: summed(a, 'f'),
    lambda a: summed(a, 't'),
    lambda a: shifted(a, 't'),
)
BINARY: tuple[Callable[[Node, Node], Node | None], ...] = (add, multiply)


def _grown(previous: tuple[Node, ...]) -> tuple[Node, ...]:
    grown = list(previous)
    grown += [made for grow in UNARY for a in previous if (made := grow(a)) is not None]
    grown += [made for grow in BINARY for a in previous for b in previous if (made := grow(a, b)) is not None]
    return tuple(dict.fromkeys(grown))


def expressions(depth: int) -> tuple[Node, ...]:
    """Every well-formed expression of at most *depth* nodes deep, deduplicated.

    Only those a constraint row can be written from come back: one with no
    variable in it is data, which the language refuses in a constraint, and a
    quadratic one is a row the eager lane cannot build at all (#942).
    """
    space: tuple[Node, ...] = LEAVES
    for _ in range(depth - 1):
        space = _grown(space)
    return tuple(node for node in space if node.degree == 1)


def model_of(node: Node) -> dict:
    """A model whose only variable content is *node*, in a binding row.

    ``foreach`` is the expression's own dimensions: anything else is a row
    repeated across a dimension the expression does not carry, which the
    language refuses — so it is computed rather than searched for.
    """
    return {
        'dimensions': dict(DIMS),
        'parameters': {'gate': {'dims': ['f'], 'dtype': 'bool'}, 'w': {'dims': ['f']}},
        'variables': {
            'x': {'foreach': ['f', 't'], 'bounds': {'lower': 0, 'upper': 100}},
            'y': {'foreach': ['f', 't'], 'where': 'gate', 'bounds': {'lower': 0, 'upper': 50}},
        },
        'constraints': {'c': {'foreach': sorted(node.dims), 'expression': f'{node} <= 10'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
    }


# ---------------------------------------------------------------------------
# the rewrites
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rewrite:
    """A rule and the pair it produced, so a failure names both spellings."""

    rule: str
    before: Node
    after: Node


def _pool() -> tuple[Node, ...]:
    """The operands the rule-carrying shapes are built over.

    Depth two, and total — a masked operand is what ``reduction-is-linear`` is
    conditional on, so it belongs in the curated non-laws rather than here,
    where every pair must be an equality.
    """
    space: tuple[Node, ...] = LEAVES
    space = _grown(space)
    return tuple(node for node in space if node.total)


def rewrites() -> tuple[Rewrite, ...]:
    """Every rule-carrying shape, built over the operand pool.

    Three rules, each constructed rather than waited for:

    - ``reduction-is-linear`` — ``sum(a + b, over=d)`` against
      ``sum(a, over=d) + sum(b, over=d)``, for every total pair whose sum is a
      constraint row the eager lane can build.
    - ``negate-through-sum`` — ``sum(-(a), over=d)`` against ``-(sum(a, over=d))``.
    - ``double-negation`` — ``-(-(a))`` against ``a``.

    Returns:
        The pairs, ordered, so a failing id can be found again.
    """
    pool = _pool()
    built: list[Rewrite] = []
    for over in sorted(DIMS):
        for a in pool:
            if a.degree == 1 and (negated := negate(a)) is not None and (lhs := summed(negated, over)) is not None:
                inner = summed(a, over)
                if inner is not None and (rhs := negate(inner)) is not None:
                    built.append(Rewrite('negate-through-sum', lhs, rhs))
            for b in pool:
                summand = add(a, b)
                if summand is None or summand.degree != 1:
                    continue
                lhs = summed(summand, over)
                left, right = summed(a, over), summed(b, over)
                if lhs is None or left is None or right is None:
                    continue
                built.append(Rewrite('reduction-is-linear', lhs, add(left, right)))
    built += [
        Rewrite('double-negation', doubled, a)
        for a in pool
        if a.degree == 1 and (doubled := negate(negate(a))) is not None
    ]
    return tuple(built)
