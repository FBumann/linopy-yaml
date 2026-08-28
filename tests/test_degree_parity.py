"""Degree 1, checked against the oracle lane itself.

The ceiling's own inventory is math-spec's and is swept there; the relational
lane relaying a refusal to a caller is `test_language_boundary.py`. This module
checks the thing that actually mattered: that the *eager* lane refuses it too,
**in the same words**.

The rule lives in ``math_spec.degree`` and this lane asks it. Where it kept a
hand-copy of the ``**`` sentence instead, no test compared the copy against the
original; and for ``x * y`` it did not refuse at all — it multiplied and let
linopy raise whatever linopy raises, so the user got a library's error instead
of the language's, with no mention of ``piecewise:``. Hard rule 3 says both
lanes accept exactly the same language; a shared *set* with two different
refusals is the weakest possible version of that.
"""

from __future__ import annotations

import pytest

import lpspec as lps
from lpspec.errors import LanguageError
from tests.conftest import dispatch_model_path
from tests.oracle import lpspec_linopy  # skips the module without the [linopy] extra


#: One entry per way the degree rule can be broken *at build time on both
#: lanes*. A product of two variables is not among them any more — the
#: objective and constraints both take one — and the position that still
#: refuses it, a named expression, is lowered only by ``check``
#: (``test_language_boundary.py``). What the eager lane refuses about a
#: quadratic *constraint* is not degree but capability, and lives in
#: ``test_quadratic_constraint.py``. The divisor that adds is affine and
#: refused all the same, because a quotient is built as one reciprocal factor.
@pytest.mark.parametrize(
    ('patch', 'match'),
    [
        pytest.param(
            {'objective.expression': 'sum(p) * sum(p)'},
            'sums of more than one term',
            id='two-reductions-multiplied-even-in-the-objective',
        ),
        pytest.param(
            {'objective.expression': 'sum(cost / p)'},
            'divisor contains variables',
            id='variable-in-a-divisor',
        ),
        pytest.param(
            {'objective.expression': 'sum(p ** 2)'},
            'over variables',
            id='a-power-over-a-variable',
        ),
        pytest.param(
            {'objective.expression': 'sum(p / (1 - cost))'},
            'must be a single Constant/Parameter factor',
            id='a-divisor-that-adds',
        ),
    ],
)
def test_both_lanes_refuse_the_same_expression(tmp_path, dispatch_model_inputs, patch, match):
    """Not just "both raise": both say the same thing.

    The relational lane prefixes the declaration it was lowering; the eager lane
    carries that as an ``add_note`` instead, so its message is the bare sentence
    and the relational one ends with it. One source, so this cannot drift into
    two dialects the way the hand-copied ``**`` message could.
    """
    data = dispatch_model_inputs
    path = dispatch_model_path(tmp_path, **patch)

    with pytest.raises(LanguageError, match=match) as eager:
        lpspec_linopy.build(path, data)

    with pytest.raises(LanguageError, match=match) as relational:
        lps.check(path)

    assert str(relational.value).endswith(str(eager.value))


def test_the_eager_lane_still_accepts_an_affine_product(tmp_path, dispatch_model_inputs):
    """The guard refuses degree 2, not multiplication — ``variable * parameter``
    is the shape the whole language is built around, and a check that broke it
    would be caught here rather than by every other test at once.
    """
    data = dispatch_model_inputs
    path = dispatch_model_path(tmp_path, **{'objective.expression': 'sum(p * cost)'})
    model = lpspec_linopy.build(path, data)
    assert model.objective is not None
