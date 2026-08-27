"""A model no data can make bounded is named by `check`, not by the solver (#229)."""

from __future__ import annotations

import warnings

import pytest
import yaml
from math_spec import load_model, unbounded_notes

import lpspec as lps
from lpspec.errors import LpspecWarning
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


def test_the_note_reaches_the_caller_as_a_warning_off_check():
    """The surface, not the inventory: which models earn a note is
    ``unbounded_notes``' rule and is swept in math-spec's own
    ``test_boundedness.py``. What is asserted here is that ``check`` asks for
    the notes at all and hands each one to the caller whole — the wording is
    what the caller reads, so a note truncated to its first clause would pass
    a test that only counted warnings.
    """
    with pytest.warns(LpspecWarning) as record:
        _check()
    message = '\n'.join(str(w.message) for w in record)
    assert "Variable 'slack'" in message, 'the note names the variable, which the solver answer does not'
    assert 'bounds.lower' in message, 'the note names the open side'
    assert 'no constraint names it' in message, 'the note gives the other half of the conjunction'
    assert 'unbounded' in message, 'the note uses the word the solve would have answered with'


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


def test_check_expands_a_curve_before_it_reads_the_notes():
    """`piecewise:` holds its variables through the constraints it expands into.

    `load_model` returns the file as written, so unexpanded there is nothing
    naming `op_cost` — which is what this bites on. A `check` that read the
    notes off the file it was handed would warn about a model the language
    holds perfectly well.
    """
    raw = yaml.safe_load((EXAMPLES_DIR / 'piecewise.yaml').read_text())
    del raw['variables']['op_cost']['bounds']

    assert unbounded_notes(load_model(raw)), 'unexpanded, nothing in the file names op_cost'
    with warnings.catch_warnings():
        warnings.simplefilter('error', LpspecWarning)
        lps.check(raw)
