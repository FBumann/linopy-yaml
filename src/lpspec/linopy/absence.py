"""The positions an absent value is spelled differently in.

Absence is **positional** in this lane: one missing parameter row is a zero in
a coefficient, an error in ``bounds:``, and false in a ``where`` operand — so
there is no single fill to apply once at load, and each position states its own
answer. The convention underneath is linopy v1's, which the lane selects on
import.
"""

from __future__ import annotations

from typing import Any

__all__ = ['coefficient', 'filled', 'present', 'unmapped', 'vacated', 'variable_term']


def present(model: Any, name: str) -> Any:
    """The coordinates variable *name* occupies — ``-1`` is linopy's own marker for an absent slot."""
    return model.variables[name].labels != -1


def unmapped(key: object) -> bool:
    """Whether a lookup left this member in no group: ``None``, or the NaN that never equals itself."""
    return key is None or key != key


def variable_term(variable: Any, absence: str) -> Any:
    """The variable as it enters an expression, carrying its declared ``absence:``.

    The mask stays on the variable either way — it is what keeps the absent
    coordinates out of the model. Under v1 an absent slot propagates and takes
    its row, which is ``absence: undefined``; ``fillna(0)`` is linopy's own
    per-expression escape back to the other reading: the slot contributes
    nothing and the row stands.
    """
    return variable.fillna(0) if absence == 'zero' else variable


def coefficient(parameter: Any) -> Any:
    """A parameter in a coefficient position, its uncovered slots at zero.

    ``load_parameters`` reindexes to the master coordinates, so an uncovered
    slot arrives as NaN, which linopy refuses in a user-supplied constant.
    """
    return parameter.fillna(0.0)


def vacated(shifted: Any, operand: Any, over: str, vacated: Any, fill: float) -> Any:
    """*shifted*, with the positions the shift vacated filled — and only those.

    linopy v1 counts ``.shift()`` among the operations that *create* absence,
    so the edge propagates and drops the row — the language's answer too. This
    is the opt-out, reached only under a numeric fill.

    ``fillna`` alone cannot spell it: by the time it runs, the edge the shift
    just made and a coordinate *operand* never had are both absent, and filling
    the second builds a row asserting ``x <= 0`` where the model said nothing.
    So the fill lands where the shift vacated **and** the operand carries the
    coordinate, and every other slot keeps the absence it arrived with.
    """
    carried = (~operand.isnull()).any(over)
    keep = carried & (~shifted.isnull() | vacated)
    return filled(shifted, fill).where(keep)


def filled(expression: Any, fill: float) -> Any:
    """*expression* with every absence in it standing as *fill*.

    ``to_linexpr()`` first when the operand is still a bare ``Variable``:
    ``Variable.fillna`` means a label fill on the released line and an
    expression fill on the v1 branch, and only the expression method is stable.
    """
    if hasattr(expression, 'to_linexpr'):
        expression = expression.to_linexpr()
    return expression.fillna(fill)
