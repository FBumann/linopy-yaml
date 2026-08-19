"""piecewise costs: the λ-formulation block, and the epigraph that needs none.

The ``piecewise:`` expansion runs before either backend, so eager and
relational receive identical affine declarations. Nonconvex correctness is
verified by checking the linked primals lie ON the curve (adjacency binaries
at work) against a numpy interpolation; the ``convex:`` flag is verified to
produce the hull instead.

The last section is the counterweight, and the piecewise rules' claim: convex piecewise
needs no formulation machinery at all. Written as epigraph constraints it is
ordinary affine YAML, relational-eligible with no ``piecewise:`` block in
sight — which is the reason the block is only for the nonconvex case.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml as pyyaml

import lpspec as lps
from lpspec.errors import DimensionError
from lpspec.language.piecewise import PiecewiseExpansionError, expand_piecewise
from lpspec.lowering import lower_program
from lpspec.sources import tidy_sources, validate_piecewise_data
from tests.conftest import by_coord, override, raw_of, schema_of
from tests.differential import differential
from tests.oracle import lpspec_linopy, pd

NONCONVEX_YAML = """
dimensions:
  snapshot: {dtype: int}
  bp: {dtype: int}

parameters:
  load: {dims: [snapshot]}
  bp_x: {dims: [bp]}
  bp_y: {dims: [bp]}

variables:
  p:
    foreach: [snapshot]
    bounds: {lower: 0, upper: 100}
  op_cost:
    foreach: [snapshot]
    bounds: {lower: 0}

piecewise:
  cost_curve:
    over: bp
    links:
      - [p, bp_x]
      - [op_cost, bp_y]

constraints:
  balance:
    foreach: [snapshot]
    expression: p == load

objective:
  sense: minimize
  expression: sum(op_cost, over=snapshot)
"""

#: The same model with the hull instead of the curve — `method: convex` and
#: nothing else changed.
CONVEX_MODEL = override(raw_of(NONCONVEX_YAML), **{'piecewise.cost_curve.method': 'convex'})

#: And the same restriction as the default's, said as a set rather than built
#: out of binaries. The two must reach the same optimum on every sink.
SOS2_MODEL = override(raw_of(NONCONVEX_YAML), **{'piecewise.cost_curve.method': 'sos2'})

#: Breakpoints whose x goes backwards — the shape the curvature guard refuses.
BACKWARDS_BP_X = pd.Series([0.0, 50.0, 40.0], index=pd.RangeIndex(3, name='bp'))


#: two dims in the frame, so the emitted ``foreach`` has an order to get wrong.
TWO_DIM_YAML = """
dimensions:
  snapshot: {dtype: int}
  generator: {dtype: str}
  bp: {dtype: int}

parameters:
  load: {dims: [snapshot]}
  bp_x: {dims: [generator, bp]}
  bp_y: {dims: [generator, bp]}

variables:
  p:
    foreach: [snapshot, generator]
    bounds: {lower: 0, upper: 100}
  op_cost:
    foreach: [snapshot, generator]
    bounds: {lower: 0}

piecewise:
  cost_curve:
    over: bp
    links:
      - [p, bp_x]
      - [op_cost, bp_y]

constraints:
  balance:
    foreach: [snapshot]
    expression: sum(p, over=generator) == load

objective:
  sense: minimize
  expression: sum(sum(op_cost, over=generator), over=snapshot)
"""


def curve(p, bp_x, bp_y) -> float:
    return float(np.interp(p, np.asarray(bp_x), np.asarray(bp_y)))


@pytest.fixture
def nonconvex_inputs():
    """A concave curve — economies of scale — and a load that reaches into it.

    The convex hull's lower envelope is the chord, which undercuts a concave
    curve, so the adjacency binaries are load-bearing on this fixture.
    """
    rng = np.random.default_rng(13)
    n_s = 12
    bp_x = pd.Series([0.0, 40.0, 100.0], index=pd.RangeIndex(3, name='bp'))
    bp_y = pd.Series([0.0, 30.0, 55.0], index=pd.RangeIndex(3, name='bp'))
    load = pd.Series(rng.uniform(5, 95, n_s).round(2), index=pd.RangeIndex(n_s, name='snapshot'))
    return {'load': load, 'bp_x': bp_x, 'bp_y': bp_y, 'snapshot': load.index, 'bp': bp_x.index}


# ---------------------------------------------------------------------------
# the λ formulation, end to end
# ---------------------------------------------------------------------------


def test_the_solution_sits_on_the_curve_not_on_its_hull(nonconvex_inputs):
    """The λ formulation reaches the curve itself, not the chord under it."""
    data = nonconvex_inputs
    expected = sum(curve(v, data['bp_x'], data['bp_y']) for v in data['load'])

    with differential(NONCONVEX_YAML, data) as run:
        assert run.oracle == pytest.approx(expected, rel=1e-6), 'ON the curve, not on the hull'

        cost = by_coord(run.result, 'op_cost', 'snapshot')
        for s, load_v in data['load'].items():
            assert cost[s] == pytest.approx(curve(load_v, data['bp_x'], data['bp_y']), abs=1e-6)


def test_the_convex_flag_gives_the_hull_and_stays_a_pure_lp(nonconvex_inputs):
    """`method: convex` drops the binaries, and says so in the answer.

    The same concave curve relaxes to its hull, whose lower envelope is the
    chord, so the objective must land below the curve.
    """
    data = nonconvex_inputs

    program = lower_program(schema_of(CONVEX_MODEL))
    assert all(v.variable_type == 'continuous' for v in program.variables), 'method: convex is a pure LP'

    on_curve = sum(curve(v, data['bp_x'], data['bp_y']) for v in data['load'])
    chord = sum(0.55 * v for v in data['load'])  # the (100, 55) chord from the origin
    with differential(CONVEX_MODEL, data) as run:
        assert run.oracle == pytest.approx(chord, rel=1e-6)
        assert run.oracle < on_curve, 'the hull undercuts a concave curve'


CHP_YAML = """
dimensions:
  snapshot: {dtype: int}
  bp: {dtype: int}

parameters:
  load: {dims: [snapshot]}
  power_bp: {dims: [bp]}
  fuel_bp: {dims: [bp]}
  heat_bp: {dims: [bp]}

variables:
  power:
    foreach: [snapshot]
    bounds: {lower: 0, upper: 100}
  fuel:
    foreach: [snapshot]
    bounds: {lower: 0}
  heat:
    foreach: [snapshot]
    bounds: {lower: 0}

piecewise:
  chp:
    over: bp
    links:
      - [power, power_bp]
      - [fuel, fuel_bp]
      - [heat, heat_bp]

constraints:
  balance:
    foreach: [snapshot]
    expression: power == load

objective:
  sense: minimize
  expression: sum(fuel, over=snapshot)
"""


def test_three_links_all_track_the_same_curve_position():
    n_s = 8
    rng = np.random.default_rng(21)
    power_bp = pd.Series([0.0, 50.0, 100.0], index=pd.RangeIndex(3, name='bp'))
    fuel_bp = pd.Series([10.0, 60.0, 140.0], index=pd.RangeIndex(3, name='bp'))
    heat_bp = pd.Series([0.0, 20.0, 60.0], index=pd.RangeIndex(3, name='bp'))
    load = pd.Series(rng.uniform(10, 90, n_s).round(2), index=pd.RangeIndex(n_s, name='snapshot'))
    data = {
        'load': load,
        'power_bp': power_bp,
        'fuel_bp': fuel_bp,
        'heat_bp': heat_bp,
        'snapshot': load.index,
        'bp': power_bp.index,
    }

    with differential(CHP_YAML, data) as run:
        fuel = by_coord(run.result, 'fuel', 'snapshot')
        heat = by_coord(run.result, 'heat', 'snapshot')
        for s, load_v in load.items():
            assert fuel[s] == pytest.approx(curve(load_v, power_bp, fuel_bp), abs=1e-6)
            assert heat[s] == pytest.approx(curve(load_v, power_bp, heat_bp), abs=1e-6)


GATED_YAML = """
dimensions:
  snapshot: {dtype: int}
  bp: {dtype: int}

parameters:
  load: {dims: [snapshot]}
  on_flag: {dims: [snapshot]}
  bp_x: {dims: [bp]}
  bp_y: {dims: [bp]}

variables:
  u:
    foreach: [snapshot]
    domain: binary
  p:
    foreach: [snapshot]
    bounds: {lower: 0, upper: 100}
  op_cost:
    foreach: [snapshot]
    bounds: {lower: 0}

piecewise:
  cost_curve:
    over: bp
    links:
      - [p, bp_x]
      - [op_cost, bp_y]
    active: u

constraints:
  commit:
    foreach: [snapshot]
    expression: u == on_flag
  balance:
    foreach: [snapshot]
    expression: p == load * on_flag

objective:
  sense: minimize
  expression: sum(op_cost, over=snapshot)
"""


def test_active_gates_the_curve_off(nonconvex_inputs):
    """`active:` decides whether the curve applies at a coordinate at all.

    Gated on, the cost sits on the curve at the pinned load; gated off, it is
    pinned to zero.
    """
    data = nonconvex_inputs
    on_flag = pd.Series([1.0, 0.0] * 6, index=pd.RangeIndex(12, name='snapshot'))
    data = {**data, 'on_flag': on_flag}

    with differential(GATED_YAML, data) as run:
        cost = by_coord(run.result, 'op_cost', 'snapshot')
        for s in on_flag.index:
            expected = curve(data['load'][s], data['bp_x'], data['bp_y']) if on_flag[s] else 0.0
            assert cost[s] == pytest.approx(expected, abs=1e-6)


def test_breakpoints_may_vary_along_another_dim():
    """examples/piecewise.yaml: convex per-generator curves (breakpoints vary
    along the generator dim — the thing flat breakpoint lists can't do).

    Each generator gets an increasing marginal cost of a different shape, and
    each one's cost has to sit on its own curve: the hull is exact here,
    because the curves are convex and the objective minimises.
    """
    example = Path('examples/piecewise.yaml')
    rng = np.random.default_rng(31)
    n_s = 24
    gens = pd.Index(['cheap', 'mid'], name='generator')
    bps = pd.RangeIndex(3, name='bp')
    p_max = pd.Series({'cheap': 100.0, 'mid': 120.0})
    per_generator = pd.MultiIndex.from_product([gens, bps], names=['generator', 'bp'])
    bp_x = pd.Series([0.0, 40.0, 100.0, 0.0, 60.0, 120.0], index=per_generator)
    bp_y = pd.Series([0.0, 200.0, 800.0, 0.0, 900.0, 2700.0], index=per_generator)
    load = pd.Series(
        (rng.uniform(0.3, 0.9, n_s) * p_max.sum()).round(1),
        index=pd.RangeIndex(n_s, name='snapshot'),
    )
    data = {
        'p_max': p_max,
        'load': load,
        'bp_x': bp_x,
        'bp_y': bp_y,
        'snapshot': load.index,
        'generator': gens,
        'bp': bps,
    }

    lower_program(schema_of(example))

    with differential(example, data) as run:
        p = by_coord(run.result, 'p', 'snapshot', 'generator')
        cost = by_coord(run.result, 'op_cost', 'snapshot', 'generator')
        for (s, g), pv in p.items():
            expected = curve(pv, bp_x.xs(g), bp_y.xs(g))
            assert cost[(s, g)] == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# what the expansion emits, and what it refuses
# ---------------------------------------------------------------------------


def test_expansion_emits_the_lambda_declarations():
    expanded = expand_piecewise(schema_of(NONCONVEX_YAML))

    assert not expanded.piecewise
    assert 'cost_curve_lam' in expanded.variables
    assert expanded.variables['cost_curve_seg'].domain == 'binary'
    assert set(expanded.constraints) >= {
        'cost_curve_convexity',
        'cost_curve_pick',
        'cost_curve_adjacency',
        'cost_curve_link0',
        'cost_curve_link1',
        'balance',
    }


def test_the_sos2_method_states_the_restriction_instead_of_building_it():
    """The same weights, the same convexity row, and no binaries at all.

    What changes is only *how λ is restricted*: the segment variable and the
    two rows that pick and neighbour it are gone, replaced by a set over the
    weights the block already emits — which is why this is a method rather
    than a second formulation.
    """
    expanded = expand_piecewise(schema_of(SOS2_MODEL))

    assert 'cost_curve_lam' in expanded.variables
    assert 'cost_curve_seg' not in expanded.variables, 'the segment binaries survived a method that has none'
    assert set(expanded.constraints) == {'cost_curve_convexity', 'cost_curve_link0', 'cost_curve_link1', 'balance'}
    emitted = expanded.sos['cost_curve']
    assert (emitted.variable, emitted.over, emitted.type, emitted.big_m) == ('cost_curve_lam', 'bp', 2, None)

    program = lower_program(schema_of(SOS2_MODEL))
    assert all(v.variable_type == 'continuous' for v in program.variables), 'sos2 emits no binary of its own'
    assert [(s.variable, s.sos_type) for s in program.sos] == [('cost_curve_lam', 2)]


def test_the_sos2_method_reaches_the_curve_the_binaries_reach(nonconvex_inputs):
    """Two spellings of one restriction, so they must agree on the answer.

    The concave fixture is what makes this a claim: the hull undercuts the
    curve there, so a set that failed to restrict anything would show up as
    the chord rather than as a near miss.
    """
    data = nonconvex_inputs
    on_curve = sum(curve(v, data['bp_x'], data['bp_y']) for v in data['load'])

    with differential(SOS2_MODEL, data) as run:
        assert run.result.objective == pytest.approx(on_curve, rel=1e-6), 'ON the curve, not on the hull'
        cost = by_coord(run.result, 'op_cost', 'snapshot')
        for s, load_v in data['load'].items():
            assert cost[s] == pytest.approx(curve(load_v, data['bp_x'], data['bp_y']), abs=1e-6)


def test_the_sos2_method_solves_natively_where_the_sink_has_the_concept(nonconvex_inputs):
    """The whole point of saying it rather than building it."""
    pytest.importorskip('gurobipy', reason='the native SOS path needs the [gurobi] extra')
    data = nonconvex_inputs
    on_curve = sum(curve(v, data['bp_x'], data['bp_y']) for v in data['load'])
    assert lps.solve(SOS2_MODEL, data, 'gurobi').objective == pytest.approx(on_curve, rel=1e-6)


def test_the_sos2_method_gates_off_like_the_binaries_do(nonconvex_inputs):
    """``active`` is a property of the weights, so every method keeps it.

    A gated-off block pins the convexity row to zero, which sets every weight
    to zero — a state the set admits, since at most two nonzero is satisfied
    by none. ``method: convex`` is the one that refuses ``active``, and does
    so because a hull with nothing pinning it is not a gate.
    """
    data = nonconvex_inputs
    gated = override(raw_of(GATED_YAML), **{'piecewise.cost_curve.method': 'sos2'})
    on_flag = pd.Series([1.0, 0.0] * 6, index=pd.RangeIndex(12, name='snapshot'))

    with differential(gated, {**data, 'on_flag': on_flag}) as run:
        cost = by_coord(run.result, 'op_cost', 'snapshot')
        for s in on_flag.index:
            expected = curve(data['load'][s], data['bp_x'], data['bp_y']) if on_flag[s] else 0.0
            assert cost[s] == pytest.approx(expected, abs=1e-6)


def test_an_emitted_set_may_not_collide_with_a_declared_one():
    """The emitted-name rule, for the one declaration kind that is new."""
    raw = override(raw_of(SOS2_MODEL), **{'sos.cost_curve': {'variable': 'p', 'over': 'bp', 'type': 1}})
    with pytest.raises(PiecewiseExpansionError, match="emitted sos 'cost_curve' collides"):
        schema_of(raw)


def test_a_method_this_project_does_not_have_is_refused():
    """`incremental` is linopy's fourth formulation and not one of ours. The
    refusal names the formulations that exist rather than picking one."""
    raw = raw_of(NONCONVEX_YAML)
    raw['piecewise']['cost_curve']['method'] = 'incremental'
    with pytest.raises(lps.SchemaError, match='unknown piecewise method'):
        schema_of(raw)


def test_a_validated_model_expands_once():
    """Validation already built the expansion, so asking again returns it.

    One object from both calls is the observable form of "once is enough": a
    second ``Model`` would be a second full validation of every emitted
    declaration.
    """
    schema = schema_of(NONCONVEX_YAML)
    assert expand_piecewise(schema) is expand_piecewise(schema)


def test_the_adjacency_row_survives_at_the_first_breakpoint(nonconvex_inputs):
    """The reason ``shift`` kept an escape hatch when it started meaning absence.

    Adjacency is ``lam <= seg + shift(seg, over=bp, offset=1, edge=0)``. At the first
    breakpoint the shifted term has no predecessor: filled it contributes zero
    and the row reads ``lam <= seg``, which is correct. Absent it would
    propagate and drop the row (#289), leaving the first lambda bounded only by
    ``[0, 1]`` — free to sit on a breakpoint the active segment does not touch,
    which is a wrong MILP that still solves.

    So this asserts the row *exists*, not just that the expansion mentions
    ``fill``: the escape hatch is only worth having if it reaches the model.
    """
    expanded = expand_piecewise(schema_of(NONCONVEX_YAML))
    assert 'edge=0' in expanded.constraints['cost_curve_adjacency'].expression

    data = nonconvex_inputs
    with differential(NONCONVEX_YAML, data) as run:
        first = run.model.constraints['cost_curve_adjacency'].labels.isel({'bp': 0}).values
        assert (first != -1).all(), 'the first breakpoint lost its adjacency row'


def test_the_emitted_foreach_follows_declaration_order():
    """The frame is a *set* of dims until something orders it, and iterating a
    set spends randomised string hashing — so the emitted ``foreach``, and every
    solver column index behind it, used to vary between processes building the
    same file. Asserted both ways round: within one process a set iterates the
    same way for the same names, so a run that reads the set rather than the
    declaration would have to fail one of the two.
    """
    raw = raw_of(TWO_DIM_YAML)
    assert list(raw['dimensions']) == ['snapshot', 'generator', 'bp']
    assert expand_piecewise(schema_of(raw)).variables['cost_curve_lam'].foreach == [
        'snapshot',
        'generator',
        'bp',
    ]

    flipped = raw_of(TWO_DIM_YAML)
    flipped['dimensions'] = {d: flipped['dimensions'][d] for d in ('generator', 'snapshot', 'bp')}
    assert expand_piecewise(schema_of(flipped)).variables['cost_curve_lam'].foreach == [
        'generator',
        'snapshot',
        'bp',
    ]


def test_an_inline_expression_is_a_legal_link():
    raw = raw_of(NONCONVEX_YAML)
    raw['piecewise']['cost_curve']['links'][0] = ['p * 2', 'bp_x']
    expanded = expand_piecewise(schema_of(raw))
    assert expanded.constraints['cost_curve_link0'].expression.startswith('(p * 2) ==')


@pytest.mark.parametrize(
    ('model', 'patch', 'match'),
    [
        pytest.param(
            NONCONVEX_YAML,
            {'piecewise.cost_curve.links': [['p', 'bp_x', '<='], ['op_cost', 'bp_y', '>=']]},
            'at most one link',
            id='at-most-one-link',
        ),
        pytest.param(
            CHP_YAML, {'piecewise.chp.method': 'convex'}, 'exactly two links', id='convex-needs-exactly-two-links'
        ),
        pytest.param(
            GATED_YAML,
            {'piecewise.cost_curve.method': 'convex'},
            'active gating is not supported',
            id='convex-cannot-be-gated',
        ),
        pytest.param(
            GATED_YAML,
            {'variables.u': {'foreach': ['snapshot'], 'bounds': {'lower': 0, 'upper': 1}}},
            'must be binary',
            id='active-must-be-binary',
        ),
        pytest.param(
            NONCONVEX_YAML,
            {'piecewise.cost_curve.links': [['p', 'bp_x'], ['op_cost', 'nope']]},
            "undeclared parameter 'nope'",
            id='undeclared-parameter',
        ),
    ],
)
def test_a_malformed_block_is_refused(model, patch, match):
    """Schema-level arity rules and the expansion's own preconditions — both
    have to fire before any data is bound."""
    with pytest.raises(ValueError, match=match):
        expand_piecewise(schema_of(model, **patch))


@pytest.mark.parametrize(
    ('link_expression', 'message'),
    [
        ('p ** 2', r"operator '\*\*'"),
        ('p * p', 'both factors of a product contain variables'),
    ],
)
def test_a_link_outside_the_language_is_named_where_the_user_wrote_it(link_expression, message):
    """The formulation checks its links itself, and that is the whole point.

    Lowering would catch these anyway — but only after expansion, so the error
    would name ``cost_curve_link0``, a declaration the user never wrote. The
    guard in ``_expr_dims`` exists to keep the message pointing at the
    ``piecewise:`` block and the link index instead.
    """
    raw = raw_of(NONCONVEX_YAML)
    block = next(iter(raw['piecewise']))
    raw['piecewise'][block]['links'][0][0] = link_expression

    with pytest.raises(PiecewiseExpansionError, match=message) as exc:
        expand_piecewise(schema_of(raw))
    assert f"piecewise '{block}' link 0" in str(exc.value)


def test_both_lanes_check_the_declarations_a_formulation_emits(tmp_path):
    """Emitted declarations are language too, so both lanes must judge them.

    A link's dims come from its values parameter, so a values parameter
    carrying a dim the links do not is a stray dim in generated math — one row
    per zone where the file reads as one per snapshot. The native lane used to
    validate the file as written, which made ``lps.check()`` pass on a model
    ``lpspec_linopy.build`` refused: the same YAML, two answers (hard rule 3).
    """
    raw = override(
        raw_of(NONCONVEX_YAML),
        **{'dimensions.zone': {'dtype': 'str'}, 'parameters.bp_y': {'dims': ['zone', 'bp']}},
    )
    stray = r"cost_curve_link1.*\['zone'\]"

    with pytest.raises(DimensionError, match=stray):
        lps.check(raw)

    path = tmp_path / 'stray_dim.yaml'
    path.write_text(pyyaml.safe_dump(raw))
    with pytest.raises(DimensionError, match=stray):
        lpspec_linopy.build(path, {})


# ---------------------------------------------------------------------------
# the data guard: `method: convex` is a promise about the breakpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('breakpoints', 'match'),
    [
        pytest.param(
            {
                'bp_x': pd.Series([0.0, 30.0, 60.0, 100.0], index=pd.RangeIndex(4, name='bp')),
                'bp_y': pd.Series([0.0, 10.0, 40.0, 50.0], index=pd.RangeIndex(4, name='bp')),
            },
            'mixed-curvature',
            id='convex-then-concave-the-hull-would-cut-corners',
        ),
        pytest.param(
            {'bp_x': BACKWARDS_BP_X},
            'strictly increasing',
            id='breakpoints-that-go-backwards',
        ),
    ],
)
def test_convex_breakpoints_that_are_not_convex_are_refused(nonconvex_inputs, breakpoints, match):
    data = nonconvex_inputs
    schema = schema_of(CONVEX_MODEL)

    with pytest.raises(PiecewiseExpansionError, match=match):
        validate_piecewise_data(schema, {**data, **breakpoints})


def test_the_curvature_guard_also_fires_through_the_relational_adapter(nonconvex_inputs):
    """`tidy_sources` is the streaming lane's only door for data, so the guard
    has to live behind it too — not only in the eager loader."""
    data = nonconvex_inputs
    schema = schema_of(CONVEX_MODEL)

    validate_piecewise_data(schema, data)  # consistent (concave) curvature passes

    bad = {**data, 'bp_x': BACKWARDS_BP_X}
    with pytest.raises(PiecewiseExpansionError, match='strictly increasing'):
        tidy_sources(schema, bad)


# ---------------------------------------------------------------------------
# the counterweight: convex piecewise with no formulation at all
# ---------------------------------------------------------------------------

EPIGRAPH_YAML = """
# The epigraph pattern: convex piecewise costs in ordinary affine YAML,
# no piecewise: block needed (the seed of issue #23's method: lp).
dimensions:
  snapshot: {dtype: int}
  generator: {dtype: str}
  segment: {dtype: str}

parameters:
  p_max: {dims: [generator]}
  load: {dims: [snapshot]}
  seg_slope: {dims: [generator, segment]}
  seg_intercept: {dims: [generator, segment]}

variables:
  p:
    foreach: [snapshot, generator]
    bounds: {lower: 0, upper: p_max}
  gen_cost:
    foreach: [snapshot, generator]
    bounds: {lower: 0}

constraints:
  balance:
    foreach: [snapshot]
    expression: sum(p, over=generator) == load
  pwl:
    foreach: [snapshot, generator, segment]
    expression: gen_cost >= p * seg_slope + seg_intercept

objective:
  sense: minimize
  expression: sum(gen_cost)
"""


@pytest.fixture
def epigraph_inputs():
    """A convex piecewise cost written as tangents, with no `piecewise:` block.

    Segment *k* of a convex curve is the tangent ``cost >= slope_k * p +
    intercept_k``, so marginal cost increases per segment. The breakpoints sit
    at 40% and 75% of each generator's ``p_max``, and the intercepts are what
    make the tangents touch there.
    """

    rng = np.random.default_rng(9)
    n_s = 24
    gens = ['cheap', 'mid']
    segments = ['s0', 's1', 's2']
    p_max = pd.Series({'cheap': 100.0, 'mid': 120.0})

    slopes = pd.DataFrame({'cheap': [5.0, 15.0, 40.0], 'mid': [20.0, 35.0, 60.0]}, index=segments)
    intercepts = {}
    for g in gens:
        b1, b2 = 0.4 * p_max[g], 0.75 * p_max[g]
        s0, s1, s2 = slopes[g]
        intercepts[g] = [0.0, (s0 - s1) * b1, (s0 - s1) * b1 + (s1 - s2) * b2]
    icepts = pd.DataFrame(intercepts, index=segments)

    load = pd.Series(
        (rng.uniform(0.3, 0.9, n_s) * p_max.sum()).round(1),
        index=pd.RangeIndex(n_s, name='snapshot'),
    )
    data = {
        'p_max': p_max,
        'load': load,
        'seg_slope': slopes.T.stack().rename_axis(['generator', 'segment']),
        'seg_intercept': icepts.T.stack().rename_axis(['generator', 'segment']),
    }
    return data | {
        'snapshot': load.index,
        'generator': pd.Index(gens, name='generator'),
        'segment': pd.Index(segments, name='segment'),
    }


def test_the_epigraph_pattern_needs_no_formulation_machinery(epigraph_inputs):
    """A convex piecewise cost is ordinary affine YAML, and stays a pure LP.

    Under minimisation the epigraph is tight, so ``gen_cost`` equals the true
    piecewise cost at the optimal dispatch.
    """
    data = epigraph_inputs

    program = lower_program(schema_of(EPIGRAPH_YAML))
    assert all(v.variable_type == 'continuous' for v in program.variables), 'the epigraph pattern is a pure LP'

    with differential(EPIGRAPH_YAML, data, lp=True) as run:
        p = by_coord(run.result, 'p', 'snapshot', 'generator')
        gc = by_coord(run.result, 'gen_cost', 'snapshot', 'generator')
        slopes = data['seg_slope'].unstack('segment')
        icepts = data['seg_intercept'].unstack('segment')
        for (s, g), pv in p.items():
            expected = max(sl * pv + ic for sl, ic in zip(slopes.loc[g], icepts.loc[g], strict=True))
            assert gc[(s, g)] == pytest.approx(expected, abs=1e-6)
