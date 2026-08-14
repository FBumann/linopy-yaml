"""The streaming language boundary: out-of-subset constructs are load errors.

There is no runtime fallback — the streaming subset IS the language
(docs/ARCHITECTURE.md). The eager builder survives only as the opt-in
compatibility layer (`lpspec.linopy`) and the differential oracle.
Errors must carry the construct and its context, verbatim.
"""

from __future__ import annotations

import pytest

from lpspec.errors import LanguageError
from lpspec.lowering import lower_program
from tests.conftest import EXAMPLES_DIR, MODEL_PATHS, schema_of

DISPATCH = EXAMPLES_DIR / 'dispatch.yaml'


def _objective(expression: str) -> dict:
    return {'objective.expression': expression}


@pytest.mark.parametrize('path', MODEL_PATHS, ids=lambda p: p.name)
def test_every_shipped_example_is_inside_the_language(path):
    """The examples are the language's own claim about itself — one of them
    falling outside the streaming subset would be a documentation bug that
    only shows up when a reader runs it."""
    lower_program(schema_of(path))


@pytest.mark.parametrize(
    'patch',
    [
        pytest.param({'variables.p.domain': 'binary', 'variables.p.bounds': {}}, id='binary-variable'),
        pytest.param({'variables.p.where': 'snapshot > 2'}, id='where-on-a-dimension-roadmap-5b'),
        pytest.param(_objective('sum(p * cost, over=generator)'), id='affine-product'),
    ],
)
def test_inside_the_language(patch):
    """Each of these lowers, so both lanes accept it."""
    lower_program(schema_of(DISPATCH, **patch))


@pytest.mark.parametrize(
    ('patch', 'match'),
    [
        pytest.param(
            {'constraints.power_balance.expression': 'sum(p ** 2, over=generator) == load'},
            r"operator '\*\*'",
            id='power-operator',
        ),
        pytest.param(_objective('sum(p * p, over=generator)'), 'degree 2', id='degree-two'),
        pytest.param(_objective('sum(cost / p, over=generator)'), 'divisor contains variables', id='variable-divisor'),
    ],
)
def test_outside_the_language_is_a_load_error(patch, match):
    """Each of these is refused at load, with no data bound.

    ``degree-two`` is the first clause of the ceiling and the reason this runs
    here rather than in the engine: the affine guard used to need data bound,
    so ``lps.check()`` accepted the model and it only blew up at build time —
    useless as a CI verb for exactly the rule it should enforce first.
    """
    with pytest.raises(LanguageError, match=match):
        lower_program(schema_of(DISPATCH, **patch))


def test_an_unknown_helper_names_its_context_and_teaches_the_rewrite():
    """The message is the whole test: an error that pointed at another lane
    would be telling the user to leave the language rather than restate it."""
    patch = {'constraints.power_balance.expression': 'my_helper(p, over=generator) == load'}
    with pytest.raises(LanguageError, match='my_helper') as exc:
        lower_program(schema_of(DISPATCH, **patch))

    reason = str(exc.value)
    assert 'power_balance' in reason, 'the reason carries its context'
    assert 'escape' in reason, 'and the rewrite, rather than a pointer to another lane'
    assert 'eager' not in reason.lower()
