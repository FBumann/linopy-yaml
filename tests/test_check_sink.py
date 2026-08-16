"""`check(model, sink=...)`: the second axis, asked with no data bound.

Whether a model is *sayable* is solver-independent; where it can *land* is not.
What is pinned here is what makes that a separate argument rather than a
warning everyone gets: bare `check` stays silent, the answer needs no data and
no installed solver, and a refusal names the sinks that would have taken it.
"""

from __future__ import annotations

import warnings

import pytest

import lpspec as lps
from lpspec.errors import LpspecError, LpspecWarning
from lpspec.lowering import lower_program
from lpspec.relational import sinks
from lpspec.relational.sinks.capabilities import Capabilities
from lpspec.relational.sinks.solvers import SOLVERS

#: A pure LP: every sink takes it whole, so it is what "silent" is measured
#: against.
PLAIN = {
    'dimensions': {'g': {'dtype': 'str', 'values': ['a', 'b', 'c']}},
    'parameters': {'cost': {'dims': ['g']}},
    'variables': {'p': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'total': {'foreach': [], 'expression': 'sum(p, over=g) <= 5'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost, over=g)'},
}

#: The same model with a set on it — the one construct in the language today
#: that a shipped sink satisfies by rewriting rather than by taking.
WITH_A_SET = PLAIN | {'sos': {'pick': {'variable': 'p', 'over': 'g', 'type': 1}}}


def _warnings(model, **kwargs) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        lps.check(model, **kwargs)
    return [str(w.message) for w in caught if issubclass(w.category, LpspecWarning)]


@pytest.mark.parametrize('sink', ['highs', 'gurobi', '.lp'])
def test_a_plain_lp_is_silent_on_every_sink(sink):
    assert _warnings(PLAIN, sink=sink) == [], 'a model inside the common subset must say nothing about portability'


def test_bare_check_says_nothing_about_portability():
    """The default has to stay silent, or every model carries a warning about a
    sink nobody named."""
    assert _warnings(WITH_A_SET) == []


def test_a_set_on_highs_says_it_will_be_reformulated_and_what_that_costs():
    """The one verdict that exists today."""
    (note,) = _warnings(WITH_A_SET, sink='highs')
    assert "'highs'" in note and 'special-ordered sets' in note
    assert 'reformulated' in note
    assert 'without duals' in note, (
        'a model that declared no integrality of its own comes back mixed-integer here, and that is '
        'the consequence a caller is actually choosing between sinks over'
    )


@pytest.mark.parametrize('sink', ['gurobi', '.lp'])
def test_a_set_is_silent_on_the_sinks_that_carry_one(sink):
    """Gurobi branches on a set and LP text writes it, so neither rewrites
    anything — the same model, no note."""
    assert _warnings(WITH_A_SET, sink=sink) == []


def test_an_unknown_sink_names_the_ones_there_are():
    with pytest.raises(LpspecError, match='unknown sink'):
        lps.check(PLAIN, sink='cplex')
    with pytest.raises(LpspecError, match=r'\.lp, \.mps, gurobi, highs, xpress'):
        lps.check(PLAIN, sink='cplex')


def test_the_question_needs_no_solver_installed(monkeypatch):
    """`solver()` refuses a name whose package is missing, being about to hand a
    model over. This one is not — and a check that needed the solver installed
    could not validate a repository against every sink it will be solved on."""
    monkeypatch.setattr(SOLVERS['gurobi'], 'is_available', classmethod(lambda cls: False))
    assert _warnings(WITH_A_SET, sink='gurobi') == []
    with pytest.raises(ModuleNotFoundError):
        sinks.solver('gurobi')


def test_no_shipped_sink_refuses_anything_the_language_can_say():
    """The state of the world, pinned so it is visible when it changes.

    Every construct today is `native` or `reformulated` everywhere, which is
    why `check(sink=)` can only warn. The first `absent` cell the language can
    reach turns this red, which is where that should show up.
    """
    refused = {name: sinks.refusal(_program(WITH_A_SET), name) for name in (*SOLVERS, '.lp')}
    assert set(refused.values()) == {None}, f'a shipped sink now refuses a model the language can state: {refused}'


def test_a_sink_that_takes_nothing_is_refused_by_name_and_offered_the_others(monkeypatch):
    """The refusal path itself, which no shipped sink can reach today.

    A purpose-built probe, since the alternative is leaving the whole contract
    unexercised until the first `absent` cell lands.
    """

    class Stub:
        capabilities = Capabilities(supports={})

    monkeypatch.setitem(SOLVERS, 'stub', Stub)
    message = sinks.refusal(_program(WITH_A_SET), 'stub')
    assert message is not None
    assert "'stub'" in message, 'the refusal names the sink'
    assert 'special-ordered sets' in message, "the refusal names the construct, in the modeller's own words"
    assert 'gurobi' in message and 'highs' in message and '.lp' in message, 'and the sinks that do take it'


def test_a_sink_excluding_a_pair_says_so_rather_than_denying_the_half(monkeypatch):
    """The other refusal shape. A caller told "it cannot take a quadratic
    objective" of a solver whose documentation says it can would reasonably
    conclude the message is wrong."""

    class Stub:
        capabilities = Capabilities(
            supports={'integrality': 'native', 'sos': 'native'},
            excludes=(frozenset({'sos', 'integrality'}),),
        )

    monkeypatch.setitem(SOLVERS, 'stub', Stub)
    integral = WITH_A_SET | {
        'variables': {'p': {'foreach': ['g'], 'domain': 'integer', 'bounds': {'lower': 0, 'upper': 10}}}
    }
    message = sinks.refusal(_program(integral), 'stub')
    assert message is not None
    assert 'separately and refuses them together' in message
    assert 'binary or integer variables' in message and 'special-ordered sets' in message


def _program(model):
    """The lowered plan a capability question is asked of."""
    return lower_program(lps.load_model(model))
