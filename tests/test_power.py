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
from math_spec import to_latex, to_typst

import lpspec as lps
from lpspec.errors import LanguageError
from lpspec.relational.plan import Add, Constant, Parameter, Power, Variable
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


@pytest.mark.parametrize(
    ('expression', 'match'),
    [
        pytest.param('sum(p ** 2)', 'over variables', id='a-variable-base'),
        pytest.param('sum(growth ** p)', 'over variables', id='a-variable-exponent'),
        pytest.param('sum(p ** p)', 'over variables', id='a-variable-on-both-sides'),
        pytest.param('sum(p * (1 + growth) ** period)', 'not a sum', id='a-base-that-adds'),
        pytest.param('sum(p * growth ** (period + period))', 'not a sum', id='an-exponent-that-adds'),
    ],
)
def test_outside_the_language_is_a_load_error(expression, match):
    """Each refused with no data bound, so `check` is the gate rather than the
    build — and the typesetter, which asks the same question, refuses to print
    math no lane would build."""
    with pytest.raises(LanguageError, match=match):
        lps.check(model(expression))
    with pytest.raises(LanguageError, match=match):
        to_latex(model(expression))


def test_a_variable_exponent_is_refused_for_its_own_reason():
    """Not the degree argument: the *degree* would be data.

    `p ** n` is affine at 1, quadratic at 2 and over the ceiling at 3, so a
    check that had to read the numbers could not answer with nothing bound —
    which is rule 2 rather than the ceiling, and the message says so.
    """
    with pytest.raises(LanguageError, match='no degree until the data arrives'):
        lps.check(model('sum(growth ** p)'))


def test_the_typesetter_prints_the_exponent_as_one():
    """A superscript, not a spelled-out product: the file said `**` and the page
    says what the file said."""
    latex = to_latex(model('sum(p * cost / growth ** period)'))
    assert r'\mathrm{growth}^{\mathrm{period}_{g}}' in latex, 'the exponent is not a superscript'
    typst = to_typst(model('sum(p * cost / growth ** period)'))
    assert 'upright("growth")^(upright("period")_(g))' in typst, 'typst spells the superscript its own way'


def test_a_power_of_a_power_keeps_its_brackets():
    """`**` is right-associative, so `(a ** b) ** c` has to print its own
    parentheses or it would read back as the other grouping."""
    nested = to_latex(model('sum(p * (growth ** period) ** period)'))
    assert r'\left( \mathrm{growth}^{\mathrm{period}_{g}} \right)^{' in nested, (
        'a parenthesised base lost its brackets, so the page says the other association'
    )


# ---------------------------------------------------------------------------
# the plan boundary: two guards no file can reach
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('expression', 'match'),
    [
        pytest.param(Power(Variable('p'), Constant(2.0)), 'power over variables', id='a-variable-under-it'),
        pytest.param(
            Power(Add(Constant(1.0), Parameter('growth')), Parameter('period')),
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
    two-fragment base silently folded would compile `1 ** period` and drop the
    rate, answering a different model at full confidence.
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
