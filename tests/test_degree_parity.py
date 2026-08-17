"""Degree 1, checked against the oracle lane itself.

`test_language_boundary.py` checks that the relational lane refuses what the
ceiling refuses. This module checks the thing that actually mattered: that the
*eager* lane refuses it too, **in the same words**.

Before ``language/degree.py``, the rule lived in ``lowering.py`` and this lane
did not ask. It kept a hand-copy of the ``**`` sentence that no test compared
against the original, and for ``x * y`` it did not refuse at all — it
multiplied and let linopy raise whatever linopy raises, so the user got a
library's error instead of the language's, with no mention of ``piecewise:``.
Hard rule 3 says both lanes accept exactly the same language; a shared *set*
with two different refusals is the weakest possible version of that.
"""

from __future__ import annotations

import pytest

import lpspec as lps
from lpspec.errors import LanguageError
from tests.conftest import dispatch_model_path
from tests.oracle import lpspec_linopy  # skips the module without the [linopy] extra


#: One entry per way degree 1 can be lost.
@pytest.mark.parametrize(
    ('expression', 'match'),
    [
        pytest.param('sum(p * p, over=generator)', 'degree 2', id='variable-times-variable'),
        pytest.param('sum(cost / p, over=generator)', 'divisor contains variables', id='variable-in-a-divisor'),
        pytest.param('sum(p ** 2, over=generator)', r"operator '\*\*'", id='an-operator-outside-the-language'),
    ],
)
def test_both_lanes_refuse_the_same_expression(tmp_path, dispatch_model_inputs, expression, match):
    """Not just "both raise": both say the same thing.

    The relational lane prefixes the declaration it was lowering; the eager lane
    carries that as an ``add_note`` instead, so its message is the bare sentence
    and the relational one ends with it. One source, so this cannot drift into
    two dialects the way the hand-copied ``**`` message could.
    """
    data = dispatch_model_inputs
    path = dispatch_model_path(tmp_path, **{'objective.expression': expression})

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
    path = dispatch_model_path(tmp_path, **{'objective.expression': 'sum(p * cost, over=generator)'})
    model = lpspec_linopy.build(path, data)
    assert model.objective is not None
