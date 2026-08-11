"""Where this lane answers linopy's v1 absence convention.

v1 (linopy's ``doc/design/convention.rst``) makes *absence* a first-class
state: §6 propagates it through every operator, §12 drops the constraint row,
and §7 refuses to fill on the caller's behalf, since the right fill is 0 for a
sum, 1 for a product, or "leave the row out".

The answers live here rather than in the loader because they are *positional* —
one missing row means three different things and only the evaluator knows
which: zero in a coefficient (:func:`coefficient`), an error in ``bounds:``,
false in a ``where`` operand. A single fill in ``load_parameters`` would pick
one and be wrong for the other two.

Both functions are correct under the legacy convention too, so neither is
conditional on ``linopy.options['semantics']``.
"""

from __future__ import annotations

from typing import Any

__all__ = ['coefficient', 'vacated']


def coefficient(parameter: Any) -> Any:
    """A parameter in a coefficient position, its uncovered slots at zero.

    A tidy parameter table is a compressed dense array, not a record of
    absence: rows only for the live coordinates says the coefficient is zero
    elsewhere (SPEC §8). ``load_parameters`` reindexes to the master
    coordinates, so an uncovered slot arrives as NaN — and v1 §5 refuses a NaN
    in a user-supplied constant, since from inside linopy a deliberate absence
    and a data error are indistinguishable.
    """
    return parameter.fillna(0.0)


def vacated(expression: Any, fill: float) -> Any:
    """A shifted expression with its vacated edge positions filled.

    linopy v1 counts ``.shift()`` among the operations that *create* absence
    (§4), so the edge propagates and drops the row — the language's answer too
    (SPEC §7, #289). This is the opt-out, reached only from ``shift(...,
    fill=0)``, and is the escape v1 itself prescribes rather than a rule of
    ours on top.

    ``to_linexpr()`` first when the operand is still a bare ``Variable``:
    ``Variable.fillna`` means a label fill on the released line and an
    expression fill on the v1 branch, and only the expression method is stable.
    """
    if hasattr(expression, 'to_linexpr'):
        expression = expression.to_linexpr()
    return expression.fillna(fill)
