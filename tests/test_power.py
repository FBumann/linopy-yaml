"""``**`` over parameters: the one exponent the language reads off the data.

The operator is degree 0 in variables wherever it appears, so it is not a
ceiling question at all — it is the arithmetic ``*`` already does, spelled the
way a discount factor is written. What it costs is one refusal per way a
variable can get underneath it, and one for an operand that adds: addition does
not distribute over ``**``, so ``(1 + rate) ** period`` is two factors wearing
one and is refused where ``growth ** period`` is not (#1175).
"""

from __future__ import annotations

import dataclasses

import polars as pl
import pytest

import lpspec as lps
from lpspec.errors import LanguageError
from lpspec.plan import Add, Constant, Parameter, Power, Variable
from lpspec.sources import tidy_sources
from tests.differential import differential

#: Three coordinates one period apart, so a discount factor orders them and a
#: hand-computed optimum is one line of arithmetic.
MODEL = {
    'dimensions': {'g': {'dtype': 'str', 'values': ['a', 'b', 'c']}},
    'parameters': {'cost': {'dims': ['g']}, 'growth': {'dims': []}, 'period': {'dims': ['g']}},
    'variables': {'p': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'meet': {'foreach': [], 'expression': 'sum(p) >= 12'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost / growth ** period)'},
}

SOURCES = {
    'cost': pl.DataFrame({'g': ['a', 'b', 'c'], 'value': [5.0, 5.0, 5.0]}),
    'growth': pl.DataFrame({'value': [1.1]}),
    'period': pl.DataFrame({'g': ['a', 'b', 'c'], 'value': [0.0, 1.0, 2.0]}),
}


def model(expression: str, **patch) -> dict:
    """MODEL with another objective — the axis every test here varies."""
    return {**MODEL, 'objective': {'sense': 'minimize', 'expression': expression}, **patch}


@pytest.mark.parametrize(
    'expression',
    [
        pytest.param('sum(p * cost / growth ** period)', id='a-discount-factor'),
        pytest.param('sum(p * cost * growth ** period)', id='a-growth-factor'),
        pytest.param('sum(p * growth ** period ** period)', id='right-associative'),
        pytest.param('sum(p * cost / growth ** 2)', id='a-literal-exponent'),
        pytest.param('sum(p * cost / 2 ** period)', id='a-literal-base'),
    ],
)
def test_both_lanes_reach_one_optimum(expression):
    """The differential oracle, which is the whole reason this is a fold and not
    a special case: a power is one number per coordinate on either lane."""
    with differential(model(expression), SOURCES):
        pass


def test_the_discount_factor_is_the_one_a_hand_computes():
    """A published number rather than a lane agreeing with itself.

    Unit costs discount to 5, 5/1.1 and 5/1.21, so the cheapest twelve units are
    all ten of `c` and two of `b` — an ordering a *linear* cost over equal
    `cost` could not produce, which is what makes the exponent load-bearing.
    """
    result = lps.solve(MODEL, SOURCES)
    assert result.objective == pytest.approx(10 * 5 / 1.21 + 2 * 5 / 1.1), (
        'the discounted optimum is not what the exponent says it is'
    )
    filled = dict(zip(*result.primal('p').sort('g').to_dict(as_series=False).values(), strict=True))
    assert filled == pytest.approx({'a': 0.0, 'b': 2.0, 'c': 10.0}), 'the cheapest period is filled first'


# ---------------------------------------------------------------------------
# the plan boundary: two guards no file can reach
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('expression', 'match'),
    [
        pytest.param(Power(Variable('p'), Constant(2.0)), 'power over variables', id='a-variable-under-it'),
        pytest.param(
            Power(Add(Constant(1.0), Parameter('growth')), Parameter('growth')),
            'refused at load',
            id='an-operand-that-adds',
        ),
    ],
)
def test_a_power_outside_the_language_is_refused_at_the_plan_boundary(expression, match):
    """Purpose-built, because `check` refuses both before a plan exists.

    The mutation table is why these are here: deleting either guard left the
    whole suite green, since every route from YAML is already closed one level
    up. A guard no test can reach is a guard nothing holds, so the plan is
    built by hand — the shape `test_relational.py` uses for the same reason.

    The second is not pedantry: addition does not distribute over `**`, so a
    two-fragment base silently folded would compile `1 ** growth` and drop the
    rate, answering a different model at full confidence. Both operands are
    the *scalar* parameter: `Program.check` refuses a variable-free part of an
    objective that carries dims, so a `period`-shaped one would be turned back
    at the boundary before the guard under test could speak.
    """
    model = {
        'dimensions': {'g': {'dtype': 'str', 'values': ['a']}},
        'parameters': {'growth': {'dims': []}, 'period': {'dims': ['g']}},
        'variables': {'p': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'meet': {'foreach': [], 'expression': 'sum(p) >= 1'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(p * growth)'},
    }
    sources = {
        'growth': pl.DataFrame({'value': [1.1]}),
        'period': pl.DataFrame({'g': ['a'], 'value': [2.0]}),
    }
    bound = lps.build(model, sources)
    program = bound._program
    patched = dataclasses.replace(program, objective=dataclasses.replace(program.objective, expression=expression))
    with pytest.raises((LanguageError, AssertionError), match=match):
        bound._engine.build(patched, tidy_sources(bound._schema, dict(bound._sources)))
