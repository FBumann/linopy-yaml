"""A variable the objective can run away with is refused before any data binds.

`unbounded` from a solver is one of the least actionable answers a modeller
gets: it says the model is wrong and not where. A subset of it is provable from
the schema alone, and this is that subset — a variable **in the objective with
a coefficient signable from literals alone**, **unbounded on the side that
improves it**, and **named by no constraint and no set**.

The conjunction is the whole design, and the accepted cases below are what
makes it one. Each half alone describes ordinary models: a free variable held
by a constraint is how a dual is read, and a bounded variable in no constraint
is how a cost is declared. Only both together are provably unbounded, and a
check that took either half on its own would refuse models that solve.

What is deliberately *not* refused is the case whose sign needs data. A
coefficient reached through a parameter has no sign until that parameter is
bound, so `sum(x * price, over=t)` is left alone however `price` looks — this
runs before any data exists, and a check that guessed would refuse a model
whose prices are all positive.

One such term is enough. A sum is signable only where *every* term carrying the
variable is, since the term that needs data can outweigh the ones that do not
and reverse the direction the variable improves in — which is a bound on the
other side, or none. `test_a_literal_term_does_not_sign_a_sum_a_parameter_is_in`
is that model, and it solves.
"""

from __future__ import annotations

import copy
import dataclasses

import pytest

import lpspec as lps
from lpspec import lowering
from lpspec.errors import LanguageError
from lpspec.relational import plan

#: The issue's own reproducer: `slack` is free, costed, and in no constraint.
UNBOUNDED = {
    'dimensions': {'t': {'dtype': 'int', 'description': 'periods'}},
    'parameters': {'cap': {'dims': ['t'], 'description': 'the cap on x'}},
    'variables': {
        'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 'cap'}, 'description': 'the real quantity'},
        'slack': {
            'foreach': ['t'],
            'bounds': {'lower': float('-inf'), 'upper': float('inf')},
            'description': 'free, costed, and held by nothing',
        },
    },
    'constraints': {'limit': {'foreach': ['t'], 'expression': 'x <= cap', 'description': 'x stays under its cap'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x + slack, over=t)', 'description': 'total'},
}


def _with(**overrides):
    """The reproducer with one or more nested keys replaced, as `a.b.c=value`."""
    model = copy.deepcopy(UNBOUNDED)
    for path, value in overrides.items():
        target = model
        *parents, leaf = path.split('.')
        for step in parents:
            target = target.setdefault(step, {})
        target[leaf] = value
    return model


def test_the_reproducer_is_refused_by_check_naming_the_variable():
    """`check()` binds no data, so this costs nothing and runs before a solve.

    The message has to carry all three parts of the finding, because the
    modeller's next question after "which variable" is "and what do I do".
    """
    with pytest.raises(LanguageError) as caught:
        lps.check(UNBOUNDED)

    message = str(caught.value)
    assert "variable 'slack'" in message, 'names the variable'
    assert 'unbounded below' in message, 'names the side, which is the one to bound'
    assert 'appears in no constraint' in message, 'names the other half of the conjunction'
    assert 'finite lower bound' in message, 'names a fix'


def test_it_is_refused_at_build_too_and_not_only_at_check():
    """A caller who never calls `check()` gets the same answer from `build()`.

    Both doors go through `lower_program`, which is where this lives — so the
    refusal is a property of the model rather than of which verb was called.
    """
    with pytest.raises(LanguageError, match="variable 'slack'"):
        lps.build(UNBOUNDED, {'cap': [1.0], 't': [0]})


@pytest.mark.parametrize(
    ('overrides', 'why'),
    [
        pytest.param(
            {'variables.slack.bounds': {'lower': 0}},
            'a finite lower bound is all it needed',
            id='bounded-on-the-improving-side',
        ),
        pytest.param(
            {
                'constraints.holds_slack': {
                    'foreach': ['t'],
                    'expression': 'slack >= x - cap',
                    'description': 'slack absorbs the overshoot',
                }
            },
            'a constraint names it, so nothing is provable without data',
            id='held-by-a-constraint',
        ),
        pytest.param(
            {'objective.expression': 'sum(x, over=t)'},
            'not in the objective, so there is no direction to run away in',
            id='absent-from-the-objective',
        ),
        pytest.param(
            {'objective.expression': 'sum(x + slack * cap, over=t)'},
            'the coefficient is a parameter, so its sign is not knowable here',
            id='a-coefficient-with-no-sign-until-data',
        ),
        pytest.param(
            {
                'objective.expression': 'sum(x + slack + slack * cap, over=t)',
                'variables.slack.bounds': {'lower': float('-inf'), 'upper': 5},
            },
            'one term signs and one does not, so their sum does not either',
            id='a-literal-term-beside-a-parameter-one',
        ),
        pytest.param(
            {'variables.slack.domain': 'binary', 'variables.slack.bounds': {}},
            'a binary lowers with 0/1 bounds, so it is bounded before this reads it',
            id='a-binary-is-never-this',
        ),
    ],
)
def test_what_the_conjunction_deliberately_accepts(overrides, why):
    """Each of these breaks one leg of the conjunction and must build.

    They are the reason the check is three conditions rather than one: every
    line here is an ordinary model that a looser rule would refuse.
    """
    assert lps.check(_with(**overrides)) is not None, why


def test_a_literal_term_does_not_sign_a_sum_a_parameter_is_in():
    """The accepted case above, carried through to the optimum that proves it.

    A sum of a literal term and a parameter one is signable from neither: here
    `1 + cap` is negative in this data, so `slack` improves *upward* and stops
    at its finite upper bound. Signing it from the literal `+slack` alone reads
    the lower bound instead, finds `-inf`, and refuses a model whose answer is
    an ordinary `-20`.
    """
    model = _with(
        **{
            'variables.x.bounds': {'lower': 0, 'upper': 10},
            'constraints.limit.expression': 'x <= 10',
            'variables.slack.bounds': {'lower': float('-inf'), 'upper': 5},
            'objective.expression': 'sum(x + slack + slack * cap, over=t)',
        }
    )
    result = lps.solve(model, {'t': [0, 1], 'cap': [-3.0, -3.0]})
    assert result.status == 'ok', 'a model the walk must not refuse'
    assert result.objective == -20.0, 'slack sits at its upper bound of 5, twice, at a net -2 each'


def test_maximising_toward_an_open_upper_bound_is_the_mirror():
    """The same finding on the other side, so the sense is read and not assumed."""
    model = _with(**{'objective.sense': 'maximize', 'variables.slack.bounds': {'lower': 0}})
    with pytest.raises(LanguageError, match='unbounded above'):
        lps.check(model)


def test_a_negative_coefficient_reads_the_other_bound():
    """Minimising `-slack` wants `+inf`, so the *upper* bound is the one missing.

    The sign of the coefficient decides which bound is asked for, and reading
    it off the declaration rather than off the sense alone is what makes this
    right for a subtraction.
    """
    model = _with(**{'objective.expression': 'sum(x - slack, over=t)', 'variables.slack.bounds': {'lower': 0}})
    with pytest.raises(LanguageError, match='unbounded above'):
        lps.check(model)


def test_a_variable_in_a_set_is_held_by_it():
    """An `sos:` names its variable, and naming is as far as this check reasons.

    A set does not bound a member's magnitude, so this is deliberately
    conservative — refusing a model a set touches would be a false positive on
    the one construct whose whole point is restricting which members are
    nonzero.
    """
    model = _with(
        **{
            'variables.slack.bounds': {'lower': float('-inf'), 'upper': float('inf')},
            'sos.pick': {'variable': 'slack', 'over': 't', 'type': 1},
        }
    )
    assert lps.check(model) is not None


def test_a_node_the_walk_does_not_know_reads_as_unsignable_not_absent(monkeypatch):
    """The default an expression node added later lands on.

    `plan.children` is the one place a new node has to reach, and this walk
    signs a shorter list of nodes than that. So the node that arrives between
    the two must fall to *unsignable* — falling to *absent* would drop a term
    that carries the variable and sign the objective from what is left.
    """

    @dataclasses.dataclass(frozen=True)
    class Later(plan.Expression):
        operand: plan.Expression

    enumerated = plan.children
    monkeypatch.setattr(plan, 'children', lambda e: (e.operand,) if isinstance(e, Later) else enumerated(e))
    node = Later(plan.Variable('slack'))

    assert lowering._objective_sign(node, 'slack') is None, 'carries the variable, so it cannot be signed away'
    assert lowering._objective_sign(node, 'x') == 0, 'a node without the variable is absent, which is signable'
