"""The one resolution rule this repository still asserts.

Name resolution is a pass in ``math_spec`` and is swept there, over its own
probes: the typed AST it hands on, the flat namespace, the ill-formed ``where``,
the narrower language bounds accept — a formal shadowing a parameter included.
What is left here is the one refusal with no case at the pinned tag. A formal
may not shadow a *dimension*, because a formal in a dim position is how a
template names the axis it reduces over, and a template that shadowed one would
read as reducing over its own argument. That both lanes then refuse the same
thing in the same words is ``test_resolution_parity.py``.
"""

from __future__ import annotations

import pytest

from tests.conftest import DISPATCH_MODEL, schema_of


def test_macro_formal_may_not_shadow_a_dimension():
    with pytest.raises(ValueError, match="formal 'generator' collides with declared dimension"):
        schema_of(
            DISPATCH_MODEL,
            **{'macros.agg': {'args': ['x'], 'kwargs': ['generator'], 'template': 'sum(x, over=generator)'}},
        )
