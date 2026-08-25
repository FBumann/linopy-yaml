"""The four positions an absent value is spelled differently in.

Absence is **positional** in this lane: one missing parameter row is a zero in
a coefficient, an error in ``bounds:``, and false in a ``where`` operand — so
there is no single fill to apply once at load, and each position states its own
answer. They live together because the reason they differ is one reason, and a
reader asking "how does this lane spell absence" should not have to find four
places to be told.

The convention underneath is linopy v1's (``doc/design/convention.rst``), which
the lane selects on import; ``builder.py``'s module docstring says why.
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
    coordinates out of the model, and dropping it to pin them at zero instead
    would hand the solver a column per absent coordinate that the relational
    lane never emits.

    What differs is the *arithmetic*. This lane sets
    ``linopy.options['semantics'] = 'v1'`` on import (see the module docstring),
    under which an absent slot propagates and takes its row — the default, and
    ``absence: undefined``. ``fillna(0)`` is linopy's own per-expression escape
    back to the other reading: the slot contributes nothing and the row stands.
    Per use rather than per model, which is the granularity a declaration needs
    and the reason the global option alone could not express this.
    """
    return variable.fillna(0) if absence == 'zero' else variable


def coefficient(parameter: Any) -> Any:
    """A parameter in a coefficient position, its uncovered slots at zero.

    ``load_parameters`` reindexes to the master coordinates, so an uncovered
    slot arrives as NaN — and v1 §5 refuses a NaN in a user-supplied constant.
    Correct under the legacy convention too, so not conditional on
    ``linopy.options['semantics']``.
    """
    return parameter.fillna(0.0)


def vacated(shifted: Any, operand: Any, over: str, vacated: Any, fill: float) -> Any:
    """*shifted*, with the positions the shift vacated filled — and only those.

    linopy v1 counts ``.shift()`` among the operations that *create* absence
    (v1 §4), so the edge propagates and drops the row — the language's answer too
    (the operator rules, #289). This is the opt-out, reached only from ``shift(...,
    edge=0)``, and is the escape v1 itself prescribes rather than a rule of
    ours on top.

    ``fillna`` alone cannot spell it: by the time it runs, the edge the shift
    just made and a coordinate *operand* never had are both absent, and filling
    the second builds a row asserting ``x <= 0`` where the model said nothing.
    So the fill lands where the shift vacated **and** the operand carries the
    coordinate — the rule the relational lane states by crossing the edge with
    the other-dim combinations the operand has — and every other slot keeps the
    absence it arrived with.
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
