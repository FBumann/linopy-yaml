"""A model no data can make bounded is named by `check`, not by the solver (#229)."""

from __future__ import annotations

import warnings

import pytest
import yaml

import lpspec as lps
from lpspec.errors import LpspecWarning
from lpspec.language.boundedness import unbounded_notes
from lpspec.language.operators import BUILTIN_NAMES
from lpspec.language.piecewise import expand_piecewise
from tests.conftest import EXAMPLES_DIR

#: The issue's variant 1, as a mapping the cases below vary one key of:
#: ``slack`` is unbounded below, is in the objective, and no constraint names it.
FREE_SLACK = {
    'dimensions': {'t': {'dtype': 'int', 'values': [0, 1, 2]}},
    'parameters': {'cap': {'dims': ['t']}, 'cost': {'dims': ['t']}},
    'variables': {
        'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 'cap'}},
        'slack': {'foreach': ['t']},
    },
    'constraints': {'limit': {'foreach': ['t'], 'expression': 'x <= cap'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x + slack, over=t)'},
}


def _check(**overrides):
    return lps.check({**FREE_SLACK, **overrides})


@pytest.mark.parametrize(
    ('overrides', 'side'),
    [
        pytest.param({}, 'lower', id='minimize-a-plus-term-runs-down'),
        pytest.param(
            {'objective': {'sense': 'maximize', 'expression': 'sum(x + slack, over=t)'}},
            'upper',
            id='maximize-a-plus-term-runs-up',
        ),
        pytest.param(
            {'objective': {'sense': 'minimize', 'expression': 'sum(x - slack, over=t)'}},
            'upper',
            id='minimize-a-minus-term-runs-up',
        ),
        pytest.param(
            {'objective': {'sense': 'minimize', 'expression': 'sum(x + -2 * slack, over=t)'}},
            'upper',
            id='a-negative-literal-coefficient-flips-the-side',
        ),
        pytest.param(
            {'objective': {'sense': 'minimize', 'expression': 'sum(x + -slack, over=t)'}},
            'upper',
            id='a-unary-minus-flips-the-side',
        ),
        pytest.param(
            {'variables': {**FREE_SLACK['variables'], 'slack': {'foreach': ['t'], 'domain': 'integer'}}},
            'lower',
            id='an-integer-variable-keeps-its-declared-bounds',
        ),
    ],
)
def test_a_free_variable_the_objective_drives_to_infinity_is_named(overrides, side):
    with pytest.warns(LpspecWarning) as record:
        _check(**overrides)
    message = '\n'.join(str(w.message) for w in record)
    assert "Variable 'slack'" in message, 'the note names the variable, which the solver answer does not'
    assert f'bounds.{side}' in message, f'the note names the open side, which here is {side}'
    assert 'no constraint names it' in message, 'the note gives the other half of the conjunction'
    assert 'unbounded' in message, 'the note uses the word the solve would have answered with'


@pytest.mark.parametrize(
    ('overrides', 'why'),
    [
        pytest.param(
            {'constraints': {'floor': {'foreach': ['t'], 'expression': 'x + slack >= cap'}}},
            'a constraint naming it is what bounds it, and an inequality counts',
            id='constrained-somewhere',
        ),
        pytest.param(
            {'variables': {**FREE_SLACK['variables'], 'slack': {'foreach': ['t'], 'bounds': {'lower': 0}}}},
            'bounded on the improving side; the open upper side is not the one improving',
            id='bounded-on-the-improving-side',
        ),
        pytest.param(
            {'variables': {**FREE_SLACK['variables'], 'slack': {'foreach': ['t'], 'bounds': {'lower': 'cost'}}}},
            'a bound naming a parameter is finite or not by data this pass does not have',
            id='a-parameter-bound',
        ),
        pytest.param(
            {'objective': {'expression': 'sum(x + cost * slack, over=t)'}},
            'a parameter coefficient may be zero or either sign, so the improving side is data',
            id='a-parameter-coefficient',
        ),
        pytest.param(
            {'objective': {'expression': 'sum(x + slack - slack, over=t)'}},
            'both signs may cancel to a coefficient of zero',
            id='both-signs-in-the-objective',
        ),
        pytest.param(
            {'objective': {'expression': 'sum(x + 0 * slack, over=t)'}},
            'a term multiplied away drives nothing',
            id='a-zero-coefficient',
        ),
        pytest.param({'objective': None}, 'a feasibility problem improves toward nothing', id='no-objective'),
        pytest.param(
            {'objective': {'expression': 'sum(x, over=t)'}},
            'a free variable the objective never names is driven nowhere',
            id='not-in-the-objective',
        ),
        pytest.param(
            {'variables': {**FREE_SLACK['variables'], 'slack': {'foreach': ['t'], 'domain': 'binary'}}},
            'a binary lowers to 0/1 whatever its bounds block says',
            id='a-binary-variable',
        ),
        pytest.param(
            {'sos': {'pick': {'variable': 'slack', 'over': 't', 'type': 1, 'big_m': 10}}},
            'a set names the variable, so nothing-touches-it is false',
            id='named-by-an-sos-block',
        ),
    ],
)
def test_a_model_the_data_could_still_bound_is_passed_over(overrides, why):
    with warnings.catch_warnings():
        warnings.simplefilter('error', LpspecWarning)
        assert _check(**overrides) is not None, why


def test_the_note_closes_no_door():
    """Every verb but `check` is silent, and the unbounded model still builds.

    The trade the warning makes: a caller who goes straight to `solve` is told
    nothing and gets the solver's answer, which is the bare `unbounded` this
    finding exists to improve on. Pinned from both ends so neither half moves
    without the other being read.
    """
    data = {'cap': [1.0, 1.0, 1.0], 'cost': [1.0, 1.0, 1.0]}
    with warnings.catch_warnings():
        warnings.simplefilter('error', LpspecWarning)
        lps.build(FREE_SLACK, data)

    assert lps.solve(FREE_SLACK, data).termination_condition == 'unbounded', (
        'the solve is left to answer as it always did — the note is advice, not a gate'
    )


@pytest.mark.parametrize(
    'expression',
    [
        pytest.param('sum(x + slack * slack, over=t)', id='a-product-of-two-variables'),
        pytest.param('sum(x + slack**2, over=t)', id='a-power'),
    ],
)
def test_a_degree_2_operand_carries_no_sign(expression):
    """The walk claims nothing where a variable is not a term's linear factor.

    Reached through `load_model` rather than `check`, because degree 2 is
    refused by lowering (`language/degree.py`) before `check` surfaces a note —
    the language accepts what the engine will not build. The guard is what
    keeps the order of those two answers from mattering.
    """
    schema = lps.load_model({**FREE_SLACK, 'objective': {'sense': 'minimize', 'expression': expression}})
    assert unbounded_notes(schema) == [], 'a variable multiplied by a variable is driven in no direction'


def test_a_variable_a_piecewise_block_holds_gets_no_note():
    """`piecewise:` holds its variables, through the constraints it expands into.

    `load_model` returns the file as written, so the note is read off the
    expansion — which is also what `lower_program` compiles. Both halves are
    asserted, because a test that only checked the expanded schema would pass
    against a `check` that had never expanded at all.
    """
    raw = yaml.safe_load((EXAMPLES_DIR / 'piecewise.yaml').read_text())
    del raw['variables']['op_cost']['bounds']
    schema = lps.load_model(raw)

    assert unbounded_notes(schema), 'unexpanded, nothing in the file names op_cost — this is what the test bites on'
    assert unbounded_notes(expand_piecewise(schema)) == [], 'the lambda formulation is what holds it'
    with warnings.catch_warnings():
        warnings.simplefilter('error', LpspecWarning)
        lps.check(raw)


#: One objective per built-in, each driving `slack` to infinity *through* that
#: operator. Keyed by `BUILTINS` below, so an operator added to the language
#: arrives here or the census fails: `_walk` hands its sign to every call's
#: arguments unchanged, on the claim that every operator sums its argument's
#: terms with coefficient 1, and an operator that did something else would make
#: that claim wrong with nothing to notice.
THROUGH_AN_OPERATOR = {
    'sum': {'objective': {'sense': 'minimize', 'expression': 'sum(x + slack, over=t)'}},
    'shift': {
        'objective': {'sense': 'minimize', 'expression': "sum(x + shift(slack, over=t, offset=1, edge='wrap'), over=t)"}
    },
    'sum_back': {
        'objective': {
            'sense': 'minimize',
            'expression': "sum(x + sum_back(slack, over=t, within=2, edge='wrap'), over=t)",
        }
    },
    'at': {
        'dimensions': {'t': {'dtype': 'int', 'values': [0, 1, 2]}, 'z': {'dtype': 'str', 'values': ['north']}},
        'lookups': {'zone_of': {'over': 't', 'into': 'z'}},
        'variables': {
            'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 'cap'}},
            'slack': {'foreach': ['z']},
        },
        'objective': {'sense': 'minimize', 'expression': 'sum(x + at(slack, by=zone_of), over=t)'},
    },
}


@pytest.mark.parametrize('operator', sorted(BUILTIN_NAMES))
def test_every_operator_hands_its_sign_to_its_operand(operator):
    """A `+slack` term under any operator still runs down toward `lower`.

    The one arm of `_walk` that is generic: a `FunctionCallNode` passes its
    sign to every argument, so *all four* built-ins share a single line that
    nothing else reaches. A fifth that negated, took a magnitude or reversed a
    sense would inherit sign-preservation in silence, and the note it produced
    would name a model that solves — which the module's own reasoning calls the
    worse error.

    What these bite on is an operator that stops carrying the sign at all. They
    cannot bite on one that *uniformly* flips it: an operator returns the dims
    it was given and an objective is one number, so every case but `sum` nests
    inside an outer `sum` and two flips cancel. That mutation is caught by
    `a-unary-minus-flips-the-side` instead.
    """
    notes = unbounded_notes(lps.load_model({**FREE_SLACK, **THROUGH_AN_OPERATOR[operator]}))
    assert len(notes) == 1, f'{operator}() should leave exactly one note, got {notes}'
    assert "'slack'" in notes[0] and 'bounds.lower' in notes[0], (
        f'{operator}() lost the term sign on the way down: {notes[0]}'
    )
