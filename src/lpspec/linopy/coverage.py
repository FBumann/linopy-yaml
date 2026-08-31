"""Is the data there where a declaration needs it? The two positions that ask.

Everything else in this lane reads an absent parameter row as a zero
coefficient (the absence rules). These are the two places that reading has no
answer for: **a divisor**, where zero is not a divisor at all, and **a constant
side**, where zero is the bound rather than the absence of one. Both are
decided against the rows the declaration actually builds, so a ``where`` that
removed the coordinate has already answered.

Walkers over the logical plan rather than over data, which is why they sit
apart from ``loader.py``: the loader coerces what a caller passed, and these
read what a declaration says before :func:`~lpspec.linopy.builder._eval`
turns the gap into an infinity nothing can name. The walk itself is
``program.children`` and ``program.parameters_of``, so "which names can reach a
divisor" is answered once for both lanes rather than re-derived here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from math_spec import program

from lpspec.errors import DataError, sparse_divisor_message, uncovered_constant_message
from lpspec.linopy import absence
from lpspec.linopy.where import evaluate_where

if TYPE_CHECKING:
    from lpspec.linopy.builder import EvaluationContext


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


def check_constant_side_covers(
    name: str, row: program.ConstraintDeclaration, ctx: EvaluationContext, mask: Any
) -> None:
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
    for side in (row.lhs, row.rhs):
        if program.carries_variable(side):
            continue
        for param, narrowed in sorted(_constant_parameters(side, ctx, mask)):
            missing = gaps_under(ctx.dataset[param], narrowed)
            if missing:
                raise DataError(uncovered_constant_message(param, missing, name))


def _constant_parameters(node: program.ExpressionNode, ctx: EvaluationContext, mask: Any) -> list[tuple[str, Any]]:
    """Every parameter on a constant side, each with the rows it actually has to cover.

    The mask narrows at every region of a ``cases:`` block. A region's value is
    needed only where that region applies — the cap a file states for its
    flagged steps says nothing about the rest, and the ``otherwise`` carries
    them — so asking it to cover the whole frame would refuse a model the
    language accepts. The relational lane reaches the same answer from the
    other end, by exempting the pieces it built under a region
    (:attr:`~lpspec.relational.engines.polars.fragments.TermFragment.partial`).
    """
    if isinstance(node, program.Cases):
        found: list[tuple[str, Any]] = []
        for region in node.regions:
            inside = evaluate_where(region.when, ctx)
            found += _constant_parameters(region.value, ctx, inside if mask is None else mask & inside)
        return found
    if isinstance(node, program.Parameter):
        return [(node.name, mask)]
    return [found for child in program.children(node) for found in _constant_parameters(child, ctx, mask)]


def check_divisors_cover(
    name: str, expressions: tuple[program.ExpressionNode, ...], ctx: EvaluationContext, mask: Any
) -> None:
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
    for quotient in program.quotients(*expressions):
        params = program.parameters_of(quotient.divisor)
        if not params:
            continue
        needed = mask
        for variable in sorted(program.variables_of(quotient.numerator)):
            present = absence.present(ctx.model, variable)
            needed = present if needed is None else (needed & present)
        for param in sorted(params):
            missing = gaps_under(ctx.dataset[param], needed)
            if missing:
                raise DataError(f'{name}: {sparse_divisor_message(param, missing)}')
