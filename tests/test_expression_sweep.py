"""Every expression to a bounded depth on both lanes, and the rewrites that must not move it.

``test_arithmetic_laws.py`` states the laws a reader should know, chosen by hand.
This makes the same kind of claim without choosing: the AST is closed, so the
set of spellings at a given depth is finite and "every one of them agrees" is a
sentence that can be checked rather than sampled.

Two claims, and the second is the one the curated file could not make:

**Agreement** — each expression builds to one model on the eager lane and the
relational one, over shapes no model in the corpus writes.

**Invariance** — a rewrite that must not change the meaning does not change the
answer. ``reduction-is-linear`` is why this exists: ``sum(a + b)`` and
``sum(a) + sum(b)`` part company the moment an operand is absent, and while the
oracle shared the mistake both lanes agreed on the wrong number until somebody
wrote that pair down by hand (#311). Agreement alone would not have caught it.

**Depth two here, depth three in its own job.** Measured on this fixture, depth
three is 2,576 expressions and about four minutes of CI, against depth two's
thirty and about a second — so ``--sweep-depth 3`` is a job of its own and the
default keeps every PR fast. The rewrites do not depend on that dial: they are
built over an operand pool, so the rule that motivated the sweep gets 546 cases
either way rather than the ten depth three happened to contain.

A model this fixture makes infeasible or unbounded is skipped rather than failed
— it says nothing about either lane — and ``test_enough_of_the_sweep_reaches_an_answer``
is what stops that from quietly becoming every case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from lpspec.errors import LaneError
from tests.differential import differential
from tests.expression_space import expressions, model_of, rewrites
from tests.oracle import pd

if TYPE_CHECKING:
    from tests.expression_space import Node, Rewrite

#: The fixture the curated laws use: `x` total, `y` masked at `f=b`, `w` dense.
DATA = {'gate': pd.Series({'a': True}), 'w': pd.Series({'a': 2.0, 'b': 3.0})}

#: What the census counts, and how many answered when the floor was taken. One
#: test, so xdist cannot split the count across workers. The depth-two space is
#: small enough to take whole; the rewrites are sampled.
CENSUS_STEP = 20
#: Of the 56 sampled cases, what answered and what a lane refused when the floor
#: and the ceiling were measured. Every refusal today is the asymmetry #1137
#: settled — a `sum` acting along a dimension a constant part of the expression
#: does not carry, which the eager lane builds and the relational one refuses by
#: name. That is decided, so the ceiling is a ratchet against it spreading
#: rather than a countdown to closing it.
ANSWERS = 41
LANE_REFUSALS = 10


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrise the agreement sweep at whatever ``--sweep-depth`` asks for."""
    if 'node' in metafunc.fixturenames:
        metafunc.parametrize('node', expressions(metafunc.config.getoption('--sweep-depth')), ids=str)


def _objective(node: Node) -> tuple[float | None, str]:
    """The objective both lanes agree on, or why there is none.

    Two things are not disagreements and must not be failures. A model this
    fixture makes infeasible or unbounded says nothing about either lane. And a
    lane that refuses the expression outright is a gap already filed against it,
    not a divergence — the message names its issue.

    Returns:
        The agreed objective and an empty reason, or None and the reason.
    """
    try:
        with differential(model_of(node), DATA) as run:
            return float(run.result.objective), ''
    except LaneError as exc:
        return None, f'a lane cannot build it: {str(exc).splitlines()[0]}'
    except AssertionError as exc:
        if 'infeasible or unbounded' in str(exc):
            return None, 'this fixture admits no finite answer'
        raise


def test_every_expression_means_the_same_on_both_lanes(node: Node) -> None:
    if reason := _objective(node)[1]:
        pytest.skip(reason)


@pytest.mark.parametrize('rewrite', rewrites(), ids=lambda r: f'{r.rule}: {r.before}')
def test_a_rewrite_that_must_not_change_the_meaning_does_not_change_the_answer(rewrite: Rewrite) -> None:
    before, why_before = _objective(rewrite.before)
    after, why_after = _objective(rewrite.after)
    if before is None or after is None:
        pytest.skip(why_before or why_after)
    assert before == pytest.approx(after, rel=1e-9), (
        f'`{rewrite.before}` and `{rewrite.after}` are one model under {rewrite.rule}, and reached {before} and {after}'
    )


def test_enough_of_the_sweep_reaches_an_answer() -> None:
    """The floor under the skips, because a skip reads exactly like a pass.

    Both sweeps above skip a case this fixture cannot solve and a case a lane
    refuses, which is right — neither says anything about agreement. What is not
    right is a change that quietly makes *every* case skip, leaving thousands of
    green skips and no claim at all. So a slice is counted two ways: a floor
    under the answers, and a ceiling over the lane refusals, which is what turns
    a new gap into a red suite rather than one more skip.
    """
    outcomes = [_objective(node) for node in expressions(2)]
    outcomes += [_objective(r.before) for r in rewrites()[::CENSUS_STEP]]
    answered = sum(value is not None for value, _ in outcomes)
    refused = sum('a lane cannot build it' in reason for _, reason in outcomes)

    assert answered >= ANSWERS, (
        f'{answered} of {len(outcomes)} sampled cases reached an answer, below the {ANSWERS} measured '
        f'when this floor was set — the fixture has gone degenerate, not the language'
    )
    assert refused <= LANE_REFUSALS, (
        f'{refused} of {len(outcomes)} sampled cases were refused by a lane, above the {LANE_REFUSALS} '
        f'measured when this ceiling was set — a lane has lost ground, and a skip would have hidden it'
    )
