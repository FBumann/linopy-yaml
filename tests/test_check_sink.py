"""`check(model, sink=...)`: the second axis, asked with no data bound.

Whether a model is *sayable* is solver-independent; where it can *land* is not.
What is pinned here is what makes that a separate argument rather than a
warning everyone gets: bare `check` stays silent, the answer needs no data and
no installed solver, and a refusal names the sinks that would have taken it.
"""

from __future__ import annotations

import re
import warnings

import pytest
from math_spec import expand_piecewise, load_model

import lpspec as lps
from lpspec.errors import LpspecError, LpspecWarning
from lpspec.lowering import lower_program
from lpspec.relational import sinks
from lpspec.relational.sinks.capabilities import Capabilities
from lpspec.relational.sinks.solvers import SOLVERS
from lpspec.relational.sinks.writers import WRITERS

#: A pure LP: every sink takes it whole, so it is what "silent" is measured
#: against.
PLAIN = {
    'dimensions': {'g': {'dtype': 'str'}},
    'parameters': {'cost': {'dims': ['g']}},
    'variables': {'p': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'total': {'foreach': [], 'expression': 'sum(p, over=g) <= 5'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost, over=g)'},
}

#: The same model with a set on it — the one construct in the language today
#: that a shipped sink satisfies by rewriting rather than by taking.
WITH_A_SET = PLAIN | {'sos': {'pick': {'variable': 'p', 'over': 'g', 'type': 1}}}

#: The same model at degree 2, in each of the two positions the language takes
#: it: the first constructs a shipped sink refuses outright.
WITH_A_QUADRATIC_OBJECTIVE = PLAIN | {'objective': {'sense': 'minimize', 'expression': 'sum(p * p, over=g)'}}
WITH_A_QUADRATIC_ROW = PLAIN | {
    'constraints': {**PLAIN['constraints'], 'ball': {'foreach': ['g'], 'expression': 'p * p <= 9'}}
}


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
    with pytest.raises(LpspecError, match='unknown sink') as refused:
        lps.check(PLAIN, sink='cplex')
    assert re.search(r'\.lp, \.mps, gurobi, highs, xpress', str(refused.value)), (
        'the refusal lists every sink there is, so a reader picks one instead of guessing'
    )


def test_the_question_needs_no_solver_installed(monkeypatch):
    """`solver()` refuses a name whose package is missing, being about to hand a
    model over. This one is not — and a check that needed the solver installed
    could not validate a repository against every sink it will be solved on."""
    monkeypatch.setattr(SOLVERS['gurobi'], 'is_available', classmethod(lambda cls: False))
    assert _warnings(WITH_A_SET, sink='gurobi') == []
    with pytest.raises(ModuleNotFoundError):
        sinks.solver('gurobi')


def _refusing(model) -> set[str]:
    """Every shipped sink that would turn *model* away."""
    return {name for name in (*SOLVERS, *WRITERS) if sinks.refusal(_program(model), name) is not None}


def test_which_shipped_sinks_refuse_what_the_language_can_now_say():
    """The state of the world, pinned so it is visible when it changes.

    A **set** still reaches every sink, natively or as binaries, so a model
    carrying one is warned about at worst. **Degree 2** does not: it is the
    first thing the language can say that some destinations have no spelling
    for at all, which is what turned `check(sink=)` from a warning into a
    refusal.

    Named individually rather than counted, because which sink is on the list
    is the fact: a cell moving in either direction — xpress growing a Hessian,
    a writer gaining a section — should be read here rather than inferred from
    a number.
    """
    assert _refusing(WITH_A_SET) == set(), 'a set reaches every sink, the ones that cannot branch on it rewriting it'
    assert _refusing(WITH_A_QUADRATIC_OBJECTIVE) == {'xpress', '.mps'}, (
        'the two with no path for a Hessian: xpress ships one and this package hands it none, '
        'and MPS spells it in a section this writer does not write'
    )
    assert _refusing(WITH_A_QUADRATIC_ROW) == {'highs', 'xpress', '.mps'}, (
        'and a quadratic row, which HiGHS has no entry point for at all'
    )


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


def test_a_suffix_is_a_sink_however_the_path_spelled_it():
    """``write`` resolves ``out.LP`` through the same registry, so the natural
    ``check(model, sink=path.suffix)` in a CI script cannot be the call that
    rejects it."""
    assert sinks.sink_capabilities('.LP') is sinks.sink_capabilities('.lp')


def test_a_refusal_does_not_swallow_the_solver_independent_advice(recwarn):
    """The two axes are independent, so naming a sink answers the second
    question without costing the first."""
    unused = PLAIN | {'dimensions': PLAIN['dimensions'] | {'spare': {'dtype': 'str'}}}

    class Stub:
        capabilities = Capabilities(supports={})

    bare = _warnings(unused)
    assert bare, 'the premise: this model has something to say without any sink named'

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(SOLVERS, 'stub', Stub)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(LpspecError):
                lps.check(unused | {'sos': {'pick': {'variable': 'p', 'over': 'g', 'type': 1}}}, sink='stub')
        assert [str(w.message) for w in caught] == bare, 'the advice a bare check gives is issued before the raise'


def test_a_refusal_names_only_what_the_model_declares(monkeypatch):
    """What a rewrite would cost is the sink's fact, not a requirement of the
    model: a pure LP with a set on it declares no integrality, and a refusal
    telling its author otherwise sends them looking for binaries they never
    wrote."""

    class Stub:
        capabilities = Capabilities(supports={'sos': 'reformulated'})

    monkeypatch.setitem(SOLVERS, 'stub', Stub)
    message = sinks.refusal(_program(WITH_A_SET), 'stub')
    assert message is None, 'a sink that rewrites a set takes the model; only what it lacks refuses one'


def test_a_set_beside_a_hessian_is_the_pair_highs_refuses():
    """The exclusion HiGHS's own rewrite manufactures, read through `check`.

    Its answer for a set is binaries, and it refuses those beside a Hessian —
    so the pair is declared there and named here in the model's own words,
    rather than derived into integrality the file never mentions.
    """
    from lpspec.relational.sinks.capabilities import required

    needed = required(_program(WITH_A_SET)) | {'quadratic_objective'}
    excluded = sinks.sink_capabilities('highs').excluded(needed)
    assert excluded == frozenset({'quadratic_objective', 'sos'})


def _program(model):
    """The lowered plan a capability question is asked of."""
    return lower_program(expand_piecewise(load_model(model)))
