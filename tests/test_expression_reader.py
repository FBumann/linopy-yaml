"""`Result.expression(name)`: a named expression readable after a solve (#562).

The relational lane only — the differential half, both lanes agreeing on the
same values, lives in ``test_linopy_lane.py`` with the rest of the oracle
comparisons. What is pinned here: the value is the one the primal implies, an
expression no constraint references still reads, the frame's dims are
``dims_of``'s, laziness (a build compiles no expression; a read compiles that
one), and the unknown-name refusal.
"""

from __future__ import annotations

import polars as pl
import pytest
from math_spec import Namespace, dims_of, expression_of, load_model

import lpspec as lps
from lpspec.errors import LpspecError
from lpspec.relational.engines.polars.compiler import PolarsCompiler

MODEL = {
    'dimensions': {
        'snapshot': {'dtype': 'int', 'values': [0, 1, 2]},
        'generator': {'values': ['g1', 'g2']},
    },
    'parameters': {
        'p_max': {'dims': ['generator']},
        'cost': {'dims': ['generator']},
        'load': {'dims': ['snapshot']},
    },
    'variables': {
        'p': {'foreach': ['snapshot', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}},
    },
    'expressions': {
        'total_gen': 'sum(p, over=generator)',
        'spend': 'sum(p * cost, over=generator)',
        'answer': '21 * 2',
        'total_cost': 'sum(sum(p * cost, over=generator), over=snapshot)',
    },
    'constraints': {
        'balance': {'foreach': ['snapshot'], 'expression': 'total_gen == load'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(sum(p * cost, over=generator), over=snapshot)'},
}


def sources() -> dict[str, pl.DataFrame]:
    return {
        'p_max': pl.DataFrame({'generator': ['g1', 'g2'], 'value': [100.0, 100.0]}),
        'cost': pl.DataFrame({'generator': ['g1', 'g2'], 'value': [10.0, 20.0]}),
        'load': pl.DataFrame({'snapshot': [0, 1, 2], 'value': [50.0, 120.0, 80.0]}),
    }


@pytest.fixture(scope='module')
def result():
    """One solve for the whole module — `lps.solve` closes the bound model, so
    every read below also proves the readers outlive it."""
    return lps.solve(MODEL, sources())


def test_a_referenced_expression_reads_the_value_its_constraint_pinned(result):
    frame = result.expression('total_gen')
    assert frame.columns == ['snapshot', 'value'], 'an expression frame is (dims…, value), dims in declaration order'
    got = dict(zip(frame['snapshot'], frame['value'], strict=True))
    assert got == pytest.approx({0: 50.0, 1: 120.0, 2: 80.0}), (
        'balance pins total_gen to load, so the reader must hand back exactly the load values'
    )


def test_an_expression_nothing_references_reads_the_value_the_primal_implies(result):
    external = (
        result.primal('p')
        .join(sources()['cost'].rename({'value': 'cost'}), on='generator')
        .group_by('snapshot')
        .agg((pl.col('value') * pl.col('cost')).sum().alias('value'))
        .sort('snapshot')
    )
    frame = result.expression('spend')
    assert frame.sort('snapshot').equals(external), (
        'spend is referenced by nothing, and must still equal sum(p * cost) computed from the primal by hand'
    )


def test_a_scalar_expression_is_one_row_matching_the_objective(result):
    frame = result.expression('total_cost')
    assert frame.columns == ['value'] and frame.height == 1, 'an expression with no dims is a single value row'
    assert frame.item() == pytest.approx(result.objective), (
        'total_cost restates the objective, so the two numbers must agree'
    )


def test_a_variable_free_expression_is_legal_and_reads_its_constant(result):
    assert result.expression('answer').item() == pytest.approx(42.0), (
        'the grammar admits a variable-free named expression, and its value is the constant it spells'
    )


@pytest.mark.parametrize('name', [pytest.param(n, id=n) for n in MODEL['expressions']])
def test_the_frame_carries_exactly_the_dims_dims_of_computes(result, name):
    schema = load_model(MODEL)
    ast = expression_of(schema.expressions[name].expression, schema, Namespace.of(schema), name)
    frame = result.expression(name)
    assert set(frame.columns) - {'value'} == set(dims_of(ast, schema, name)), (
        'the returned frame answers over the dim set the language computes for the expression'
    )


@pytest.mark.parametrize(
    'name',
    [
        pytest.param('nope', id='a-typo'),
        pytest.param('sum(p, over=generator)', id='an-expression-string'),
    ],
)
def test_an_unknown_name_lists_the_declared_names_and_refuses_strings(result, name):
    with pytest.raises(KeyError, match=r'answer, spend, total_cost, total_gen') as caught:
        result.expression(name)
    assert 'never an expression string' in str(caught.value), (
        'the refusal must say expression() takes declared names only, not arbitrary expression strings'
    )


def test_a_masked_coordinate_has_no_row():
    masked = {
        **MODEL,
        'variables': {
            'p': {
                'foreach': ['snapshot', 'generator'],
                'bounds': {'lower': 0, 'upper': 'p_max'},
                'where': 'p_max > 0',
            }
        },
        'expressions': {'scaled': 'p * cost'},
        'constraints': {'balance': {'foreach': ['snapshot'], 'expression': 'sum(p, over=generator) == load'}},
    }
    data = sources() | {
        'p_max': pl.DataFrame({'generator': ['g1', 'g2'], 'value': [200.0, 0.0]}),
        'load': pl.DataFrame({'snapshot': [0, 1, 2], 'value': [50.0, 120.0, 80.0]}),
    }
    frame = lps.solve(masked, data).expression('scaled')
    assert frame['generator'].unique().to_list() == ['g1'], (
        'absence propagates into a reader the way it does into a constraint (the operator rules): the masked-out '
        "generator's coordinates have no rows rather than zeros"
    )
    assert frame.height == 3, 'the surviving generator keeps one row per snapshot'


def test_a_build_compiles_no_expression_and_a_read_compiles_exactly_one(monkeypatch):
    compiled = []
    original = PolarsCompiler.expression

    def counting(self, expr, context, **kwargs):
        compiled.append(context)
        return original(self, expr, context, **kwargs)

    monkeypatch.setattr(PolarsCompiler, 'expression', counting)
    with lps.build(MODEL, sources()) as bound:
        named = [c for c in compiled if c.startswith('named expression')]
        assert named == [], 'a build lowers no named expression — fifty declared and none read must cost none'
        assert len(compiled) == 2 * len(MODEL['constraints']) + 1, (
            'a model declaring expressions compiles exactly what one without them compiles: '
            'each constraint side, and the objective'
        )
        outcome = bound.solve()
        named = [c for c in compiled if c.startswith('named expression')]
        assert named == [], 'a solve lowers none either — the readers it hands out are thunks'
        outcome.expression('spend')
        named = [c for c in compiled if c.startswith('named expression')]
        assert named == ["named expression 'spend'"], 'reading one expression compiles that one expression'


def test_a_closed_result_refuses_an_expression_read(result):
    with lps.build(MODEL, sources()) as bound:
        outcome = bound.solve()
    outcome.close()
    with pytest.raises(LpspecError, match='closed'):
        outcome.expression('spend')
