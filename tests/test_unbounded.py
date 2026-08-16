"""A variable the objective can run away with is named by `check`, before data.

`unbounded` from a solver is one of the least actionable answers a modeller
gets: it says the model is wrong and not where, and once one integer variable
is in the model HiGHS says `infeasible_or_unbounded` instead, which does not
even say which. A subset of it is provable from the schema alone, and this is
that subset — a variable **in the objective with a coefficient signable from
literals alone**, **unbounded on the side that improves it**, and **named by no
constraint and no set**.

It is advice, not a refusal. `check` warns and every door still builds, because
a model part-written declares a variable before the constraint that will hold
it, and a draft is not a wrong model. So the tests below assert a note, and the
accepted cases assert silence rather than the absence of an exception.

The conjunction is the whole design, and the accepted cases are what makes it
one. Each half alone describes ordinary models: a free variable held by a
constraint is how a dual is read, and a bounded variable in no constraint is
how a cost is declared. Only both together are provably unbounded, and a check
that took either half on its own would speak about models that solve.

What is deliberately *not* read is the case whose sign needs data. A
coefficient reached through a parameter has no sign until that parameter is
bound, so `sum(x * price, over=t)` is left alone however `price` looks — this
runs before any data exists, and a check that guessed would name the wrong
bound on a model whose prices are all positive.

One such term is enough. A sum is signable only where *every* term carrying the
variable is, since the term that needs data can outweigh the ones that do not
and reverse the direction the variable improves in — which is a bound on the
other side, or none. `test_a_literal_term_does_not_sign_a_sum_a_parameter_is_in`
is that model, and it solves.

A `where` on the variable is left alone for the same reason, from the other
end. Every leg of the conjunction is fixed by the schema, so what is left is
unbounded under any data that gives the variable a column: the only reading
data can still change is whether it has one, and a mask is where that is
decided.
"""

from __future__ import annotations

import copy
import dataclasses
import warnings

import polars as pl
import pytest

import lpspec as lps
from lpspec import lowering
from lpspec.errors import LpspecWarning
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


def _notes(model) -> list[str]:
    """The unbounded advice `check` warns with, and nothing else it has to say."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        lps.check(model)
    return [str(w.message) for w in caught if 'unbounded' in str(w.message)]


def test_the_reproducer_is_named_by_check():
    """`check()` binds no data, so this costs nothing and runs before a solve.

    The note has to carry all three parts of the finding, because the
    modeller's next question after "which variable" is "and what do I do".
    """
    (note,) = _notes(UNBOUNDED)

    assert "variable 'slack'" in note, 'names the variable'
    assert 'unbounded below' in note, 'names the side, which is the one to bound'
    assert 'appears in no constraint' in note, 'names the other half of the conjunction'
    assert 'finite lower bound' in note, 'names a fix'


def test_the_advice_closes_no_door():
    """Advice, so the model a caller who never runs `check()` has still builds.

    The whole reason this is a warning: a variable declared before the
    constraint that will hold it is a model part-written, and refusing to build
    a draft would cost more than the bare `unbounded` this saves. Which is what
    that caller still gets, from the solver, unnamed.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error', LpspecWarning)
        lps.build(UNBOUNDED, {'cap': [1.0], 't': [0]})

    result = lps.solve(UNBOUNDED, {'cap': [1.0], 't': [0]})
    assert result.termination_condition == 'unbounded', 'the answer check offered to name in advance'


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
def test_what_the_conjunction_deliberately_says_nothing_about(overrides, why):
    """Each of these breaks one leg of the conjunction and must go unremarked.

    They are the reason the check is three conditions rather than one: every
    line here is an ordinary model that a looser rule would speak about, and a
    warning nobody can act on is what teaches a modeller to filter this one.
    """
    assert _notes(_with(**overrides)) == [], why


def test_a_literal_term_does_not_sign_a_sum_a_parameter_is_in():
    """The accepted case above, carried through to the optimum that proves it.

    A sum of a literal term and a parameter one is signable from neither: here
    `1 + cap` is negative in this data, so `slack` improves *upward* and stops
    at its finite upper bound. Signing it from the literal `+slack` alone reads
    the lower bound instead, finds `-inf`, and warns about a model whose answer
    is an ordinary `-20`.
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
    assert result.status == 'ok', 'a model the walk must not speak about'
    assert result.objective == -20.0, 'slack sits at its upper bound of 5, twice, at a net -2 each'


def test_a_masked_variable_is_left_to_the_data_that_decides_it_exists():
    """A `where` decides whether the variable has a column at all, and data owns that.

    Every other leg of the conjunction is fixed by the schema, so the reported
    shape is unbounded under *any* data that gives the variable a column. A
    mask is the one thing that can leave it with none — and then the term is
    inert and the model solves, as it does here.
    """
    model = _with(**{'variables.slack.where': 'live'})
    model['parameters']['live'] = {'dims': ['t']}
    defined_nowhere = {
        't': [0, 1],
        'cap': pl.DataFrame({'t': [0, 1], 'value': [1.0, 1.0]}),
        'live': pl.DataFrame(schema={'t': pl.Int64, 'value': pl.Float64}),
    }

    result = lps.solve(model, defined_nowhere)
    assert result.status == 'ok', 'no column of slack exists, so nothing runs away'
    assert result.objective == 0.0, 'x is the only variable left and its cost floor is 0'


def test_maximising_toward_an_open_upper_bound_is_the_mirror():
    """The same finding on the other side, so the sense is read and not assumed."""
    model = _with(**{'objective.sense': 'maximize', 'variables.slack.bounds': {'lower': 0}})
    assert 'unbounded above' in _notes(model)[0], 'maximising reads the upper bound'


def test_a_negative_coefficient_reads_the_other_bound():
    """Minimising `-slack` wants `+inf`, so the *upper* bound is the one missing.

    The sign of the coefficient decides which bound is asked for, and reading
    it off the declaration rather than off the sense alone is what makes this
    right for a subtraction.
    """
    model = _with(**{'objective.expression': 'sum(x - slack, over=t)', 'variables.slack.bounds': {'lower': 0}})
    assert 'unbounded above' in _notes(model)[0], 'the coefficient decides the side, not the sense'


def test_a_variable_in_a_set_is_held_by_it():
    """An `sos:` names its variable, and naming is as far as this check reasons.

    A set does not bound a member's magnitude, so this is deliberately
    conservative — speaking about a model a set touches would be a false
    positive on the one construct whose whole point is restricting which
    members are nonzero.
    """
    model = _with(
        **{
            'variables.slack.bounds': {'lower': float('-inf'), 'upper': float('inf')},
            'sos.pick': {'variable': 'slack', 'over': 't', 'type': 1},
        }
    )
    assert _notes(model) == [], 'a set names it, and naming is enough to stay quiet'


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
