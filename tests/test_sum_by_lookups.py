"""``sum(x, by=[l, m])`` — one grouping through several maps at once.

The single-lookup form lands terms on one dimension; this lands them on a
product of dimensions, which is what a capacity limit per *location and
technology* asks for. PyPSA ships exactly that as a constraint type
(`tech_capacity_expansion_limit`, carrier and bus together), so the shape has
an outside consumer rather than only a symmetry argument.

What needs two lanes to see:

**The grouping is one join, not two.** Relationally the coordinates ride the
same dim table, so a list costs one extra column and no extra join — held by
the lowering test, which asserts a single ``GroupSum`` carrying both names
rather than a composition.

**The empty combination.** A (bus, technology) pair no generator sits on is a
group with no members, and the two lanes reach it by different routes: the
relational lane never emits a row, the eager lane's unstacked groupby invents
one and fills it with linopy's own ``const: nan``. That NaN does not stay in
the empty sum — it propagates through whatever the row adds next and takes the
whole row with it, which is why the row here carries a second term.

**Label order.** A groupby hands its groups back sorted; the dim table keeps
the declared order. Under v1 arithmetic a shared dim ordered two ways is a
refusal, so ``test_a_declared_order_the_groupby_would_not_pick`` builds the
model whose technologies are deliberately not alphabetical.
"""

from __future__ import annotations

import pytest
from math_spec import expand_piecewise

from lpspec.errors import DimensionError, LanguageError, SchemaError
from lpspec.lowering import _Lowering, lower_program
from lpspec.relational.plan import GroupSum, Variable
from tests.conftest import by_coord, override, raw_of, relation, resolved, schema_of
from tests.differential import RTOL, differential
from tests.oracle import operators, pd, xr
from tests.test_compiler import compiler

MODEL = """
description: capacity limited per bus and technology at once

dimensions:
  generator: {dtype: str, description: a generating unit}
  bus: {dtype: str, description: a node of the network}
  technology: {dtype: str, description: what a generator is built from}

lookups:
  gen_bus: {over: generator, into: bus, description: the bus a generator sits on}
  gen_tech: {over: generator, into: technology, description: the technology it is}

parameters:
  cost: {dims: [generator], description: marginal cost of a unit of output}
  limit: {dims: [bus, technology], description: how much of one technology one bus may run}
  demand: {dims: [], description: total output the system must reach}

variables:
  p:
    foreach: [generator]
    bounds: {lower: 0, upper: .inf}
    description: output of a generator

constraints:
  technology_at_bus:
    foreach: [bus, technology]
    expression: sum(p, by=[gen_bus, gen_tech]) <= limit
    description: output of one technology at one bus stays under its limit
  meet_demand:
    foreach: []
    expression: sum(p, over=generator) >= demand
    description: the system meets its demand

objective:
  sense: minimize
  expression: sum(p * cost, over=generator)
  description: total marginal cost
"""

GENERATORS = ['g1', 'g2', 'g3', 'g4']
#: `g4` shares (b, wind) with `g3`, so a limit binds across two generators;
#: nothing sits at (b, sun), which is the empty combination.
OF_BUS = ['a', 'a', 'b', 'b']
OF_TECH = ['wind', 'sun', 'wind', 'wind']


def _inputs():
    index = pd.DataFrame({'generator': GENERATORS})
    limits = pd.DataFrame(
        {
            'bus': ['a', 'a', 'b', 'b'],
            'technology': ['wind', 'sun', 'wind', 'sun'],
            'value': [10.0, 5.0, 7.0, 1.0],
        }
    )
    return {
        'cost': index.assign(value=[1.0, 2.0, 3.0, 4.0]).set_index('generator')['value'],
        'limit': limits,
        'demand': 20.0,
        'generator': index,
        'gen_bus': relation('generator', 'bus', GENERATORS, OF_BUS),
        'gen_tech': relation('generator', 'technology', GENERATORS, OF_TECH),
        'bus': pd.Index(['a', 'b'], name='bus'),
        'technology': pd.Index(['wind', 'sun'], name='technology'),
    }


# ---------------------------------------------------------------------------
# both lanes
# ---------------------------------------------------------------------------


def test_grouping_through_two_lookups_agrees_across_the_lanes():
    """The optimum, hand-derived, and the same on both lanes and the LP file.

    (a, wind) caps `g1` at 10 and (a, sun) caps `g2` at 5, which is 15 of the
    20 demanded at cost 1 and 2. The remaining 5 has to come from bus b, where
    (b, wind) allows 7 across `g3` and `g4` together — so the cheaper `g3`
    takes all 5 and `g4` stays at zero.
    """
    sources = _inputs()
    with differential(MODEL, sources, lp=True) as run:
        assert run.oracle == pytest.approx(10 * 1.0 + 5 * 2.0 + 5 * 3.0, rel=RTOL)
        built = by_coord(run.result, 'p', 'generator')

    assert built['g1'] == pytest.approx(10.0), '(a, wind) is the binding limit'
    assert built['g2'] == pytest.approx(5.0), '(a, sun) is the binding limit'
    assert built['g3'] == pytest.approx(5.0), 'the cheaper of the two generators sharing (b, wind)'
    assert built['g4'] == pytest.approx(0.0), 'priced out, and its limit is shared rather than its own'


def test_a_combination_no_member_lands_on_is_a_group_of_nothing():
    """(b, sun) has no generator, and both lanes have to read that the same way.

    An empty group is a zero-length sum, so its row asks `0 <= 1` and binds
    nothing. Tightening that limit to zero must therefore change no answer,
    which catches a lane that quietly summed the wrong members into it.

    A row left with no variables is not built at all, on either lane, so this
    cannot also say whether the row survived — that is what
    :func:`test_an_empty_combination_does_not_take_its_row_with_it` is for.
    """
    sources = _inputs()
    limit = sources['limit'].copy()
    limit.loc[(limit['bus'] == 'b') & (limit['technology'] == 'sun'), 'value'] = 0.0
    sources['limit'] = limit
    with differential(MODEL, sources) as run:
        assert run.oracle == pytest.approx(35.0, rel=RTOL), 'a limit on an empty group binds nothing'


def test_an_empty_combination_does_not_take_its_row_with_it():
    """A row whose group is empty but whose *other* terms are not is still a row.

    The eager lane reaches the combinations no member lands on by unstacking,
    which invents them carrying linopy's own ``_fill_value`` — ``const: nan``.
    Left there, that NaN does not stay in the empty sum: it propagates through
    the addition, and linopy drops the whole row, `headroom` with it, leaving
    the constraint enforced on one lane and unenforced on the other.

    `headroom` takes the slack under every limit and is paid for it, so
    (b, sun) is worth 1 if its row exists and 100 if it does not.
    """
    sources = _inputs()
    patched = override(
        raw_of(MODEL),
        **{
            'variables.headroom': {
                'foreach': ['bus', 'technology'],
                'bounds': {'lower': 0, 'upper': 100},
                'description': 'capacity left unused at one bus in one technology',
            },
            'constraints.technology_at_bus.expression': 'sum(p, by=[gen_bus, gen_tech]) + headroom <= limit',
            'objective.expression': 'sum(p * cost, over=generator) - sum(sum(headroom, over=bus), over=technology)',
        },
    )
    with differential(patched, sources, lp=True) as run:
        assert run.oracle == pytest.approx(35.0 - 3.0, rel=RTOL), 'the same dispatch, less the slack it is paid for'
        headroom = {(b, t): v for b, t, v in run.result.primal('headroom').iter_rows()}

    assert headroom[('b', 'sun')] == pytest.approx(1.0), (
        'the empty combination binds `headroom` at its limit — a dropped row would let it run to 100'
    )
    assert sum(headroom.values()) == pytest.approx(3.0), 'the four limits total 23 against 20 dispatched'


def test_a_declared_order_the_groupby_would_not_pick():
    """The technologies are declared out of alphabetical order on purpose.

    A groupby returns its groups sorted, the dim table keeps the declared
    order, and v1 arithmetic refuses to combine a shared dim ordered two ways.
    So this model builds on both lanes only because the eager lane puts its
    result back into declared order.
    """
    sources = _inputs()
    assert list(sources['technology']) != sorted(sources['technology']), 'the point of the case is the order'
    with differential(MODEL, sources) as run:
        assert run.oracle == pytest.approx(35.0, rel=RTOL)


# ---------------------------------------------------------------------------
# the eager grouper
# ---------------------------------------------------------------------------


def test_a_grouped_parameter_reads_zero_where_no_member_lands():
    """The combination the unstack invents is an empty sum, not a NaN.

    A grouped *parameter* comes back as a plain array, and no model reaches
    this through both lanes — the relational lane refuses a constant side that
    does not cover its rows — so the arm is held here rather than by a
    differential. Without it linopy refuses the model outright, naming a NaN
    the modeller never wrote.
    """
    generator = pd.Index(['g1', 'g2'], name='generator')
    cost = xr.DataArray([1.0, 2.0], coords=[generator])
    of_bus = xr.DataArray(['a', 'b'], coords=[generator])
    of_tech = xr.DataArray(['wind', 'sun'], coords=[generator])
    labels = {'bus': pd.Index(['a', 'b'], name='bus'), 'technology': pd.Index(['wind', 'sun'], name='technology')}

    grouped = operators.operator_grouped_sum(cost, (of_bus, of_tech), into=('bus', 'technology'), labels=labels)

    assert grouped.to_series().to_dict() == {
        ('a', 'wind'): 1.0,
        ('a', 'sun'): 0.0,
        ('b', 'wind'): 0.0,
        ('b', 'sun'): 2.0,
    }, 'the two combinations nobody sits at are zero-length sums, and a zero-length sum is 0'


# ---------------------------------------------------------------------------
# lowering
# ---------------------------------------------------------------------------


def test_two_lookups_lower_to_one_node_and_not_to_a_composition():
    """One grouping, so one plan node: the coordinates ride one join.

    A composition would consume `generator` twice, and the second pass would
    have nothing left to group.
    """
    (limit, _demand) = lower_program(expand_piecewise(schema_of(MODEL))).constraints
    assert limit.lhs == GroupSum(
        Variable('p'), over='generator', coordinate=('gen_bus', 'gen_tech'), into=('bus', 'technology')
    )


def test_the_one_element_list_is_the_plain_form():
    """`by=[l]` and `by=l` are the same call, so nothing branches on arity."""
    schema = schema_of(MODEL)
    assert _Lowering(schema, 't').expr(resolved('sum(p, by=[gen_bus])', schema)) == _Lowering(schema, 't').expr(
        resolved('sum(p, by=gen_bus)', schema)
    )


def test_a_hand_built_node_whose_tuples_disagree_is_refused():
    """`plan.py` is a public IR, so a node can arrive without going through
    resolution — and the two tuples pair up positionally, so a mismatch would
    otherwise drop the unpaired coordinate and group by one map too few.

    Nothing in the language can build this: resolution derives both tuples
    from one list of names. It is the shortest path to the guard.
    """
    node = GroupSum(Variable('p'), over='generator', coordinate=('gen_bus', 'gen_tech'), into=('bus',))
    with pytest.raises(ValueError, match='zip'):
        compiler().expression(node, 'a hand-built plan')


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('expression', 'match'),
    [
        pytest.param(
            'sum(p, by=[gen_bus, nope])',
            r'by=nope\) does not name a lookup',
            id='one-member-names-nothing',
        ),
        pytest.param(
            'sum(p, by=[gen_bus, bus])',
            r"by=bus\): 'bus' is a dimension, and by= takes a lookup",
            id='one-member-is-a-dimension',
        ),
        pytest.param(
            'sum(p, by=[gen_bus, gen_bus])',
            r"targets \['bus'\] more than once",
            id='two-members-target-one-dim',
        ),
    ],
)
def test_a_list_the_language_cannot_read_is_refused_at_load(expression, match):
    with pytest.raises(LanguageError, match=match):
        resolved(expression, schema_of(MODEL))


def test_a_list_is_only_legal_in_a_kwarg_value():
    """The grammar admits a bracketed list where a kwarg value goes, and nowhere
    else — a list has no meaning in arithmetic, so it never reaches resolution:
    asked for one anyway, the refusal comes from the parse."""
    with pytest.raises(SchemaError, match='Failed to parse expression'):
        resolved('p * [gen_bus, gen_tech]', schema_of(MODEL))


def test_lookups_over_different_dimensions_cannot_be_one_grouping():
    """One grouping consumes one dimension, so the members must share it.

    Refused in resolution rather than left to the dim rule: the dim rule can
    only report the *first* `over:` as the one consumed, which would name a
    dimension the author never wrote and blame the operand for lacking it.
    """
    schema = schema_of(
        MODEL.replace('sum(p, by=[gen_bus, gen_tech])', 'sum(p, by=gen_bus) * limit').replace(
            'gen_tech: {over: generator, into: technology, description: the technology it is}',
            'bus_tech: {over: bus, into: technology, description: a technology of a bus}',
        )
    )
    with pytest.raises(LanguageError, match=r'groups through lookups over different dimensions'):
        resolved('sum(p, by=[gen_bus, bus_tech])', schema)


def test_a_partition_is_one_lookup_and_says_so():
    """`shift(by=...)` takes a lookup in the other position, so a list means nothing.

    `sum` and `at` consume the dim their lookups are over and *produce* the
    targets, which is what a list is: one grouping into a product. A partition
    produces nothing — it says which rows are neighbours — so several of them
    name no shape the operator could walk, and the refusal is at load rather
    than a lane quietly walking the first.
    """
    patch = {
        'constraints.technology_at_bus.foreach': ['generator'],
        'constraints.technology_at_bus.expression': 'shift(p, over=generator, offset=1, by=[gen_bus, gen_tech]) <= 1',
    }
    with pytest.raises(DimensionError, match=r'by=\[gen_bus, gen_tech\]\) partitions by several lookups'):
        schema_of(MODEL, **patch)


def test_grouping_into_a_dim_the_operand_already_carries_is_refused():
    """The union would absorb one of the two — the single-lookup rule, plural."""
    schema = schema_of(MODEL)
    node = resolved('sum(p * limit, by=[gen_bus, gen_tech])', schema)
    with pytest.raises(LanguageError, match=r"targets \['bus', 'technology'\], which the expression already carries"):
        _Lowering(schema, 't').expr(node)
