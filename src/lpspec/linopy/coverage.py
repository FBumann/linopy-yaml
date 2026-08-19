"""Is the data there where a declaration needs it? The two positions that ask.

Everything else in this lane reads an absent parameter row as a zero
coefficient (the absence rules). These are the two places that reading has no
answer for: **a divisor**, where zero is not a divisor at all, and **a constant
side**, where zero is the bound rather than the absence of one. Both are
decided against the rows the declaration actually builds, so a ``where`` that
removed the coordinate has already answered.

Walkers over the expression AST rather than over data, which is why they sit
apart from ``loader.py``: the loader coerces what a caller passed, and these
read what a declaration says before :func:`~lpspec.linopy.builder._eval_ast`
turns the gap into an infinity nothing can name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lpspec.errors import DataError, sparse_divisor_message, uncovered_constant_message
from lpspec.language.expression_parser import (
    BinaryOperatorNode,
    NameNode,
    ParameterNode,
    VariableNode,
    children,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lpspec.language.expression_parser import ComparisonNode, ExpressionNode
    from lpspec.language.model import Model


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
