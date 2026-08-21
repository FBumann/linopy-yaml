"""What an objective sums, when its terms do not carry the same dims.

An objective is one number and says so: the expression is scalar or the file
does not load, so every reduction in it is one somebody wrote. What these
tests hold is that the two readings a bracket allows are **both sayable and
different** — ``sum(a) + sum(b)`` totals each term over its own dims, while
``sum(a + b)`` broadcasts and counts each once per coordinate of the other.

That distinction is invisible while every term of an objective has the same
dims, which is every model the rest of the suite builds. It becomes an 8x
error the moment an objective spans a sparse ``(snapshot, node, tech)``
variable and a dense ``(snapshot, node, carrier)`` one — the shape a real cost
function has, and the shape that found #197. What #197 fixed by *rule* — the
implied sum distributing over addition — is now spelled at the call site, and
#1069 is why: the rule was invisible in the file and the math block disagreed
with the LP about it (#1046).
"""

from __future__ import annotations

import pytest
from math_spec import load_model

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
    'objective': {'sense': 'minimize', 'expression': 'sum(x * a) + sum(y * b)'},
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
    ('expression', 'expected'),
    [
        pytest.param('sum(x * a) + sum(y * b)', 32.0, id='a-sum-per-term'),
        pytest.param('sum(x * a + y * b)', 66.0, id='one-sum-around-both-broadcasts'),
        pytest.param('sum(x * a) - sum(y * b)', -28.0, id='a-difference'),
        pytest.param('-(sum(x * a) + sum(y * b))', -32.0, id='negated'),
        pytest.param('sum(x * a * c) + sum(y * b * c)', 6400.0, id='a-factor-in-each-term'),
        pytest.param('sum((x * a + y * b) * c)', 13200.0, id='a-factor-over-the-broadcast-group'),
        pytest.param('(sum(x * a) + sum(y * b)) / 2', 16.0, id='a-divisor-applied-to-the-group'),
        pytest.param('sum(x * a, over=i) + sum(y * b, over=j)', 32.0, id='the-dims-named-one-at-a-time'),
    ],
)
def test_where_the_sum_is_written_decides_what_it_counts(data, expression, expected):
    """Two readings, both sayable, and the bracket is what picks.

    ``differential`` already asserts the two lanes agree; what it cannot know is
    whether they agree on the *right* number, and before #197 they disagreed in
    exactly this shape. The pairs here are what make the rule readable rather
    than remembered: 32 against 66, and 6400 against 13200, differ only in
    where the sum's bracket closes.
    """
    model = {**DISJOINT_MODEL, 'objective': {'sense': 'minimize', 'expression': expression}}
    with differential(model, data) as run:
        assert run.oracle == pytest.approx(expected)


#: #1046's model: a bracketed addition under a product, its branches on
#: different dims. The two readings differ by a factor of |j| on the first
#: term, and the file used to pick one while the math block printed the other.
BRACKETED_MODEL = {
    'dimensions': {'i': {'dtype': 'int', 'values': [0, 1]}, 'j': {'dtype': 'int', 'values': [0, 1, 2]}},
    'parameters': {'c': {'dims': ['i']}},
    'variables': {
        'x': {'foreach': ['i'], 'bounds': {'lower': 1, 'upper': 1}},
        'y': {'foreach': ['j'], 'bounds': {'lower': 1, 'upper': 1}},
    },
    'constraints': {'floor': {'foreach': ['i'], 'expression': 'x >= 0'}},
}


@pytest.mark.parametrize(
    ('expression', 'expected'),
    [
        pytest.param('sum(c * (x + y))', 660.0, id='the-bracket-broadcasts-and-the-page-says-so'),
        pytest.param('sum(c * x) + sum(c * y)', 440.0, id='distributed-by-hand-is-a-different-model'),
    ],
)
def test_a_bracketed_addition_under_a_product_means_what_it_prints(expression, expected):
    """#1046: the shape whose math block and LP file disagreed.

    ``c * (x + y)`` carried dims, so it is now refused outright and both
    readings have to be written down — 30 per unit of ``x[0]`` where the sum
    closes outside the bracket, 10 where it closes around each term. What made
    the old spelling a silent bug is that the page showed the first and the
    solver was handed the second.
    """
    data = {'c': pd.Series([10.0, 100.0], index=pd.Index([0, 1], name='i'))}
    model = {**BRACKETED_MODEL, 'objective': {'sense': 'minimize', 'expression': expression}}
    with differential(model, data, lp=True) as run:
        assert run.oracle == pytest.approx(expected)


def test_an_objective_carrying_dims_is_refused_with_the_wrapper_named():
    """The rule that used to be implied is now the load error that asks for it."""
    from lpspec.errors import DimensionError

    model = {**DISJOINT_MODEL, 'objective': {'sense': 'minimize', 'expression': 'x * a + y * b'}}
    with pytest.raises(DimensionError, match=r"carries dims \['i', 'j'\].*Wrap each additive term"):
        load_model(model)


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
    eager = lpspec_linopy.build(path, sources)
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
