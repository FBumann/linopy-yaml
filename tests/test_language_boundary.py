"""The streaming language boundary: out-of-subset constructs are load errors.

There is no runtime fallback — the streaming subset IS the language
(docs/about/architecture.md), and both lanes are inside it: `lpspec.linopy`
builds the same file through the same `lower_program` gate.
Errors must carry the construct and its context, verbatim.
"""

from __future__ import annotations

import pytest
from math_spec import expand_piecewise

import lpspec as lps
from lpspec.errors import LanguageError
from lpspec.lowering import lower_program
from tests.conftest import EXAMPLES_DIR, MODEL_PATHS, schema_of

DISPATCH = EXAMPLES_DIR / 'dispatch.yaml'


def _objective(expression: str) -> dict:
    return {'objective.expression': expression}


@pytest.mark.parametrize('path', MODEL_PATHS, ids=lambda p: p.name)
def test_every_shipped_example_is_inside_the_language(path):
    """Every dim rule, over the corpus this repository ships, and then lowering.

    The rules live with the language and are swept there over the probes that
    travel with them; this is the same sweep over the gallery and the ports,
    which stay. Both are needed: a rule with no corpus proves nothing, and a
    corpus with no rule applied to it is a directory of files.

    Loading *is* the first half: a ``Model`` runs every dim rule on the way
    out. The second is that the result lowers, so an example falling outside
    the streaming subset is caught here rather than by a reader running it.
    """
    lower_program(expand_piecewise(schema_of(path)))


@pytest.mark.parametrize(
    'patch',
    [
        pytest.param({'variables.p.domain': 'binary', 'variables.p.bounds': {}}, id='binary-variable'),
        pytest.param({'variables.p.where': 'snapshot > 2'}, id='where-on-a-dimension-roadmap-5b'),
        pytest.param(_objective('sum(p * cost)'), id='affine-product'),
        pytest.param(_objective('sum(p * p)'), id='degree-two-in-the-objective'),
        pytest.param(
            {'constraints.power_balance.expression': 'sum(p * p, over=generator) == load'},
            id='degree-two-in-a-constraint',
        ),
    ],
)
def test_inside_the_language(patch):
    """Each of these lowers, so both lanes accept it."""
    lower_program(expand_piecewise(schema_of(DISPATCH, **patch)))


@pytest.mark.parametrize(
    ('patch', 'match'),
    [
        pytest.param(_objective('sum(cost / p)'), 'divisor contains variables', id='an-expression-the-file-writes'),
        pytest.param(
            {'expressions': {'squared': {'expression': 'sum(p * p, over=generator)'}}},
            'degree 2',
            id='an-expression-only-check-lowers',
        ),
    ],
)
def test_outside_the_language_is_a_load_error(patch, match):
    """The refusal reaches the caller through ``lps.check``, with no data bound.

    Two rows, one per position the verb has to reach — which rules it enforces
    there is the language's inventory and is swept in math-spec's own
    ``test_degree.py``. Asked of ``lps.check`` rather than of
    ``lower_program``, because the verb is the claim: the affine guard once
    needed data bound, so ``check`` accepted the model and it blew up at build
    time — useless as a CI verb for exactly the rules it should enforce first.
    Named expressions are the same argument one construct along; only ``check``
    lowers them.
    """
    with pytest.raises(LanguageError, match=match):
        lps.check(schema_of(DISPATCH, **patch))


def test_an_unknown_operator_names_its_context_and_teaches_the_rewrite():
    """The message is the whole test: an error that pointed at another lane
    would be telling the user to leave the language rather than restate it."""
    patch = {'constraints.power_balance.expression': 'my_helper(p, over=generator) == load'}
    with pytest.raises(LanguageError, match='my_helper') as exc:
        lower_program(expand_piecewise(schema_of(DISPATCH, **patch)))

    reason = str(exc.value)
    assert 'power_balance' in reason, 'the reason carries its context'
    assert 'escape' in reason, 'and the rewrite, rather than a pointer to another lane'
    assert 'eager' not in reason.lower()
