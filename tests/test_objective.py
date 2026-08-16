"""What an objective sums, when its terms do not carry the same dims.

The rule is one line of the declaration rules — an objective is a scalar, so every dim it
names is summed — and the whole content of these tests is *which* dims belong
to *what*. A term is summed over the dims that term carries. A term is not
repeated because a sibling term carries a dim it does not.

That distinction is invisible while every term of an objective has the same
dims, which is every model the rest of the suite builds. It becomes an 8x
error the moment an objective spans a sparse ``(snapshot, node, tech)``
variable and a dense ``(snapshot, node, carrier)`` one — the shape a real cost
function has, and the shape that found #197.
"""

from __future__ import annotations

import pytest

from tests.differential import differential
from tests.oracle import pd  # through the guard: a bare import would beat it

#: Two variables pinned to 1 on disjoint dims, so the objective is arithmetic
#: with no optimisation left in it: whatever comes out is what was summed.
DISJOINT_MODEL = {
    'dimensions': {
        'i': {'dtype': 'int', 'values': [0, 1]},
        'j': {'dtype': 'int', 'values': [0, 1, 2]},
        'k': {'dtype': 'int', 'values': [0, 1]},
    },
    'parameters': {'a': {'dims': ['i']}, 'b': {'dims': ['j']}, 'c': {'dims': ['k']}},
    'variables': {
        'x': {'foreach': ['i'], 'bounds': {'lower': 1, 'upper': 1}},
        'y': {'foreach': ['j'], 'bounds': {'lower': 1, 'upper': 1}},
    },
    'constraints': {'floor': {'foreach': ['i'], 'expression': 'x >= 0'}},
    'objective': {'sense': 'minimize', 'expression': 'x * a + y * b'},
}


@pytest.fixture
def data():
    """``sum(x * a) == 2``, ``sum(y * b) == 30``, ``sum(c) == 200``.

    Distinct enough that a broadcast shows up as a different number rather than
    a coincidence: broadcasting the first two gives 66, not 32.
    """
    return {
        'a': pd.Series([1.0, 1.0], index=pd.Index([0, 1], name='i')),
        'b': pd.Series([10.0, 10.0, 10.0], index=pd.Index([0, 1, 2], name='j')),
        'c': pd.Series([100.0, 100.0], index=pd.Index([0, 1], name='k')),
    }


@pytest.mark.parametrize(
    ('expression', 'expected', 'broadcast_would_give'),
    [
        pytest.param('x * a + y * b', 32.0, 66.0, id='a-sum-of-two-terms'),
        pytest.param('x * a - y * b', -28.0, -54.0, id='a-difference'),
        pytest.param('-(x * a + y * b)', -32.0, -66.0, id='negated'),
        pytest.param('(x * a + y * b) * c', 6400.0, 13200.0, id='an-operator-applied-to-the-group'),
        pytest.param('(x * a + y * b) / 2', 16.0, 33.0, id='a-divisor-applied-to-the-group'),
        pytest.param(
            'sum(x * a, over=i) + sum(y * b, over=j)',
            32.0,
            32.0,
            id='already-scalar-per-term-the-control-that-always-agreed',
        ),
    ],
)
def test_a_term_is_summed_over_its_own_dims(data, expression, expected, broadcast_would_give):
    """Each term totals its own dims, whatever is wrapped around the group.

    ``differential`` already asserts the two lanes agree; what it cannot know is
    whether they agree on the *right* number, and before #197 they disagreed in
    exactly this shape. The operator cases are what say the rule survives
    something applied to the group rather than holding only at the top of the
    expression.
    """
    model = {**DISJOINT_MODEL, 'objective': {'sense': 'minimize', 'expression': expression}}
    with differential(model, data) as run:
        assert run.oracle == pytest.approx(expected)
        assert run.oracle != pytest.approx(broadcast_would_give) or expected == broadcast_would_give


#: No `objective:` at all — the constraints are the whole question, and the
#: answer is whether they can be met. `need` sits inside the caps, so they can.
FEASIBILITY_MODEL = {
    'dimensions': {'g': {'values': ['wind', 'gas']}},
    'parameters': {'cap': {'dims': ['g']}, 'need': {'dims': []}},
    'variables': {'x': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {'meet': {'foreach': [], 'expression': 'sum(x, over=g) >= need'}},
}


def test_a_model_with_no_objective_is_a_feasibility_problem(tmp_path):
    """Both lanes build it, and the answer is a point rather than an optimum.

    A file with no `objective:` used to lower to `LanguageError: the relational
    backend requires an objective` while the linopy lane built it happily —
    the one construct the two lanes disagreed about (#845). Nothing optimises,
    so the objective value is the zero the sink was handed.
    """
    import yaml as pyyaml

    import lpspec as lps
    from tests.oracle import lpspec_linopy

    sources = {'cap': {'wind': 40.0, 'gas': 100.0}, 'need': 90.0}

    path = tmp_path / 'feasibility.yaml'
    path.write_text(pyyaml.safe_dump(FEASIBILITY_MODEL))
    eager = lpspec_linopy.build(path, data=sources)
    assert 'meet' in eager.constraints, 'the eager lane built the same file'

    with lps.solve(FEASIBILITY_MODEL, sources) as result:
        assert result.is_ok, 'the constraints can be met, so this is not a failed solve'
        assert result.objective == 0.0, 'nothing was optimised, so the objective is the zero it was given'
        served = result.primal('x')['value'].sum()
        assert served == pytest.approx(90.0), 'the constraint is the whole model, so it binds'

    lp = lps.write(FEASIBILITY_MODEL, sources, tmp_path / 'feasibility.lp')
    assert 'obj:\n\ns.t.' in lp.read_text(), 'the objective section is written, and is empty'


def test_a_model_with_no_objective_still_says_when_it_cannot_be_met():
    """The answer a feasibility problem exists to give."""
    import lpspec as lps

    with lps.solve(FEASIBILITY_MODEL, {'cap': {'wind': 40.0, 'gas': 10.0}, 'need': 90.0}) as result:
        assert not result.is_ok
        assert result.termination_condition == 'infeasible'
