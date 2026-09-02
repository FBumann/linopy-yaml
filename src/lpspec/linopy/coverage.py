"""Is the data there where a declaration needs it? The two positions that ask.

Everything else in this lane reads an absent parameter row as a zero
coefficient (the absence rules). These are the two places that reading has no
answer for: **a divisor**, where zero is not a divisor at all, and **a constant
side**, where zero is the bound rather than the absence of one. Both are
decided against the rows the declaration actually builds, so a ``where`` that
removed the coordinate has already answered. The walk itself is
``program.children`` and ``program.parameters_of``, so "which names can reach a
divisor" is answered once for both lanes.
"""

from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING, Any

from math_spec import program

from lpspec.errors import DataError, sparse_divisor_message, uncovered_constant_message
from lpspec.linopy import absence
from lpspec.linopy.where import evaluate_where

if TYPE_CHECKING:
    from collections.abc import Iterator

    from lpspec.linopy.where import EvaluationContext


def gaps_under(array: Any, mask: Any) -> int:
    """How many slots of *array* are null where *mask* still admits the row.

    The one way this lane asks "is this parameter defined where it is needed";
    ``None`` means nothing narrows the question.
    """
    missing = array.isnull()
    if mask is not None:
        missing = missing & mask
    return int(missing.sum())


def check_constant_side_covers(
    name: str, row: program.ConstraintDeclaration, ctx: EvaluationContext, mask: Any
) -> None:
    """A comparison's constant side must have values wherever the row is built.

    A missing row is read as 0, and on a side with no variable that zero *is*
    the bound — `x <= cap` becomes `x <= 0`, which binds rather than vanishing.
    Keyed to the rows the declaration builds, not to the coordinate product: a
    `where` that removed the coordinate has already answered the question.
    """
    for side in (row.lhs, row.rhs):
        if program.carries_variable(side):
            continue
        walk = _under_regions(side, ctx, mask)
        found = [(node.name, where) for node, where in walk if isinstance(node, program.Parameter)]
        for param, narrowed in sorted(found, key=itemgetter(0)):
            missing = gaps_under(ctx.dataset[param], narrowed)
            if missing:
                raise DataError(uncovered_constant_message(param, missing, name))


def _under_regions(
    node: program.ExpressionNode, ctx: EvaluationContext, mask: Any
) -> Iterator[tuple[program.ExpressionNode, Any]]:
    """Every node under *node*, each with the rows it actually has to cover.

    The mask narrows at every region of a ``cases:`` block: a region's data is
    owed only where that region applies, so asking a piece to cover the whole
    frame would refuse a model the language accepts. Sorted by the caller
    where the order decides which name an error can reach: a mask is an array,
    so the pairs are not orderable among themselves.
    """
    yield node, mask
    if isinstance(node, program.Cases):
        for region in node.regions:
            inside = evaluate_where(region.when, ctx)
            yield from _under_regions(region.value, ctx, inside if mask is None else mask & inside)
        return
    for child in program.children(node):
        yield from _under_regions(child, ctx, mask)


def check_divisors_cover(
    name: str, expressions: tuple[program.ExpressionNode, ...], ctx: EvaluationContext, mask: Any
) -> None:
    """A divisor must have a value wherever this declaration divides by it.

    Not "wherever it is indexed": sparse data is the ordinary case, and a check
    keyed to the coordinate product would refuse models that never touch the
    gap. Two things can already have removed a coordinate — the row's own
    ``where``, and the mask on a variable in the numerator — and either is
    enough, so the requirement is their conjunction, narrowed at a ``cases:``
    region like the constant side is.

    Reached before :func:`~lpspec.linopy.builder._eval`, the last moment the
    gap is visible: :func:`absence.coefficient` fills an uncovered slot with
    0.0 at the parameter leaf, and from there the division yields an infinity
    and the row is masked out silently.
    """
    for expression in expressions:
        for quotient, region in _under_regions(expression, ctx, mask):
            if not isinstance(quotient, program.Divide):
                continue
            params = program.parameters_of(quotient.divisor)
            if not params:
                continue
            needed = region
            for variable in sorted(program.variables_of(quotient.numerator)):
                present = absence.present(ctx.model, variable)
                needed = present if needed is None else (needed & present)
            for param in sorted(params):
                missing = gaps_under(ctx.dataset[param], needed)
                if missing:
                    raise DataError(f'{name}: {sparse_divisor_message(param, missing)}')
