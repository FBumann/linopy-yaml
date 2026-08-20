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
import polars as pl
import pytest
import yaml as pyyaml

import lpspec as lps
from lpspec.errors import DataError, DimensionError
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


def _of(frame, generator):
    """One generator's breakpoint values, in order — the tidy-frame `xs`."""
    return frame.loc[frame['generator'] == generator, 'value'].to_numpy()


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
    per_generator = pd.MultiIndex.from_product([gens, bps], names=['generator', 'bp']).to_frame(index=False)
    bp_x = per_generator.assign(value=[0.0, 40.0, 100.0, 0.0, 60.0, 120.0])
    bp_y = per_generator.assign(value=[0.0, 200.0, 800.0, 0.0, 900.0, 2700.0])
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
            expected = curve(pv, _of(bp_x, g), _of(bp_y, g))
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
# the data guard reads the curve in the order the model builds it
# ---------------------------------------------------------------------------


#: `nonconvex_inputs`' three breakpoints, labelled the same and written in
#: another order. A tidy table carries its coordinates in its labels, so this
#: is the same curve — the engine joins it by label and reaches one model.
OUT_OF_ORDER_BP = pd.Index([2, 0, 1], name='bp')


def test_a_curve_written_out_of_order_is_the_same_curve(nonconvex_inputs):
    """A row order is not a breakpoint order, and only the second is the model's.

    Rows reach a lane in whatever order the join or the group-by that made
    them left behind; what orders the breakpoints is the `bp` index, which is
    ascending here. The guard read the rows as they arrived and refused this
    table as backwards (#1122), where the engine joins it by label and the
    eager lane builds and solves it.
    """
    shuffled = {
        **nonconvex_inputs,
        'bp_x': pd.Series([100.0, 0.0, 40.0], index=OUT_OF_ORDER_BP),
        'bp_y': pd.Series([55.0, 0.0, 30.0], index=OUT_OF_ORDER_BP),
    }

    tidy_sources(schema_of(CONVEX_MODEL), shuffled)


def test_a_breakpoint_dimension_with_no_index_keeps_its_own_message(nonconvex_inputs):
    """With no index there is no order, so the guard has no question to answer.

    It answered anyway: reading the rows as they arrived, a curve written out
    of order drew "requires strictly increasing breakpoints" — a claim about
    an order nothing had established — in front of the message that names the
    missing index. The λ methods, which have no curvature guard, always
    reached the right one.
    """
    orphaned = {k: v for k, v in nonconvex_inputs.items() if k != 'bp'}
    orphaned['bp_x'] = pd.Series([100.0, 0.0, 40.0], index=OUT_OF_ORDER_BP)
    orphaned['bp_y'] = pd.Series([55.0, 0.0, 30.0], index=OUT_OF_ORDER_BP)

    tidy_sources(schema_of(CONVEX_MODEL), orphaned)  # the guard has nothing to say

    with pytest.raises(DataError, match='has no index'):
        lps.build(CONVEX_MODEL, orphaned)


def test_the_eager_lane_reads_the_curve_in_the_index_order(nonconvex_inputs, tmp_path):
    """Which of the two lanes is right, pinned — the loader lays the values out first.

    So the order the guard walks is the dimension's, and the row order the
    table happened to arrive in is nothing: the shuffled curve binds and the
    backwards index is refused. Hard rule 3 says the streaming lane owes the
    same two answers.
    """
    path = tmp_path / 'convex.yaml'
    path.write_text(pyyaml.safe_dump(CONVEX_MODEL))
    shuffled = {
        **nonconvex_inputs,
        'bp_x': pd.Series([100.0, 0.0, 40.0], index=OUT_OF_ORDER_BP),
        'bp_y': pd.Series([55.0, 0.0, 30.0], index=OUT_OF_ORDER_BP),
    }

    lpspec_linopy.build(path, shuffled)  # a row order is not a breakpoint order

    with pytest.raises(PiecewiseExpansionError, match='strictly increasing'):
        lpspec_linopy.build(path, {**nonconvex_inputs, 'bp': pd.Index([2, 1, 0], name='bp')})


def test_a_breakpoint_index_that_runs_backwards_is_refused(nonconvex_inputs):
    """The other half of the same blindness, and this one built a wrong model.

    A dimension's index is its order — `shift` walks it and `index(bp, 0)`
    names its first label — so an index written `[2, 1, 0]` puts the fixture's
    breakpoints at x = 100, 40, 0. `adjacency` then pairs segments that are
    not neighbours, and `lp` writes its chords against a negative run. The
    guard never read the index, so it had nothing to say about the order that
    index sets (#1122).
    """
    backwards = {**nonconvex_inputs, 'bp': pd.Index([2, 1, 0], name='bp')}

    with pytest.raises(PiecewiseExpansionError, match='strictly increasing'):
        tidy_sources(schema_of(CONVEX_MODEL), backwards)


# ---------------------------------------------------------------------------
# the data guard: a curve carries a value at every breakpoint it is built over
# ---------------------------------------------------------------------------


def ragged_curve(points):
    """A per-generator curve as a tidy frame — B two breakpoints where bp has three."""
    return pl.DataFrame(
        {
            'generator': [g for g, _ in points],
            'bp': [k for _, k in points],
            'value': list(points.values()),
        }
    )


@pytest.fixture
def ragged_inputs():
    """Generator B supplies two of the three breakpoints the dimension declares.

    B's curve starts at (10, 100), so the row it never wrote is not a harmless
    repeat of its first point: read as a zero coefficient it is a vertex at
    (0, 0), and the weights mix onto it to run B below the minimum output its
    own curve states.
    """
    return {
        'snapshot': [0],
        'generator': ['A', 'B'],
        'bp': [0, 1, 2],
        'load': pd.Series([25.0], index=pd.RangeIndex(1, name='snapshot')),
        'bp_x': ragged_curve({('A', 0): 0.0, ('A', 1): 10.0, ('A', 2): 20.0, ('B', 0): 10.0, ('B', 1): 20.0}),
        'bp_y': ragged_curve({('A', 0): 0.0, ('A', 1): 50.0, ('A', 2): 140.0, ('B', 0): 100.0, ('B', 1): 130.0}),
    }


def test_a_curve_short_of_a_breakpoint_is_refused(ragged_inputs):
    """A missing breakpoint row read as a zero coefficient is a vertex at the origin.

    It built, and the answer was wrong with nothing to see: on this fixture B
    interpolated between its real (20, 130) and the (0, 0) it never declared,
    for an optimum of 147.5 where its own two points put it at 195.
    """
    schema = schema_of(raw_of(TWO_DIM_YAML))

    with pytest.raises(DataError, match=r"'bp_x' has no value at"):
        tidy_sources(schema, dict(ragged_inputs))


def test_the_curve_guard_fires_on_the_eager_lane_too(ragged_inputs, tmp_path):
    """Both lanes take the same sources, so both refuse the same table (hard rule 3)."""
    path = tmp_path / 'two_dim.yaml'
    path.write_text(TWO_DIM_YAML)

    with pytest.raises(DataError, match=r"'bp_x' has no value at"):
        lpspec_linopy.build(path, dict(ragged_inputs))


def test_a_curve_supplied_at_every_breakpoint_passes(ragged_inputs):
    """The guard is about holes, not about how the table is written."""
    whole = dict(ragged_inputs)
    whole['bp_x'] = ragged_curve(
        {('A', 0): 0.0, ('A', 1): 10.0, ('A', 2): 20.0, ('B', 0): 10.0, ('B', 1): 20.0, ('B', 2): 30.0}
    )
    whole['bp_y'] = ragged_curve(
        {('A', 0): 0.0, ('A', 1): 50.0, ('A', 2): 140.0, ('B', 0): 100.0, ('B', 1): 130.0, ('B', 2): 200.0}
    )

    tidy_sources(schema_of(raw_of(TWO_DIM_YAML)), whole)


def test_a_dict_shaped_curve_is_read_for_holes_too(ragged_inputs, tmp_path):
    """The eager lane takes the caller's mapping unspread, so the guard reads that spelling.

    A ``{label: value}`` curve is the one plain-Python shape that can be short:
    a sequence and a single number are dense against the labels they spread
    over, a dict carries only the keys it was written with.
    """
    path = tmp_path / 'one_dim.yaml'
    path.write_text(NONCONVEX_YAML)
    data = {
        'snapshot': [0],
        'bp': [0, 1, 2],
        'load': pd.Series([25.0], index=pd.RangeIndex(1, name='snapshot')),
        'bp_x': {0: 0.0, 1: 10.0},
        'bp_y': {0: 0.0, 1: 50.0},
    }

    with pytest.raises(DataError, match=r"'bp_x' has no value at"):
        lpspec_linopy.build(path, data)


def test_a_dimension_with_no_index_keeps_its_own_message(ragged_inputs):
    """The guard runs before the index is bound, and must not answer for its absence.

    Where nothing declares the breakpoints, the curve's own labels are all
    there is — it cannot be short of a breakpoint no one declared — so a
    complete curve has to reach the message that names the missing index.
    """
    whole = {k: v for k, v in ragged_inputs.items() if k != 'bp'}
    whole['bp_x'] = ragged_curve({('A', 0): 0.0, ('A', 1): 20.0, ('B', 0): 10.0, ('B', 1): 20.0})
    whole['bp_y'] = ragged_curve({('A', 0): 0.0, ('A', 1): 140.0, ('B', 0): 100.0, ('B', 1): 130.0})

    tidy_sources(schema_of(raw_of(TWO_DIM_YAML)), whole)  # the guard has nothing to say

    with pytest.raises(DataError, match='has no index'):
        lps.build(raw_of(TWO_DIM_YAML), whole)


# ---------------------------------------------------------------------------
# points: a curve shorter than its breakpoint dimension
# ---------------------------------------------------------------------------

SHORT_CURVE = """
dimensions:
  generator: {dtype: str}
  bp: {dtype: int}

parameters:
  p_max: {dims: [generator]}
  load: {dims: []}
  bp_x: {dims: [generator, bp]}
  bp_y: {dims: [generator, bp]}
  bp_present: {dims: [generator, bp], dtype: bool}

variables:
  p:
    foreach: [generator]
    bounds: {lower: 0, upper: p_max}
  op_cost:
    foreach: [generator]
    bounds: {lower: 0}

piecewise:
  cost_curve:
    over: bp
    points: bp_present
    links:
      - [p, bp_x]
      - [op_cost, bp_y]

constraints:
  balance:
    foreach: []
    expression: sum(p, over=generator) == load

objective:
  sense: minimize
  expression: sum(op_cost, over=generator)
"""

#: A three-breakpoint dimension where B is a two-point curve starting at (10, 100):
#: at a load of 25 the cheap answer runs B at 20 and A at 5, for 155.
A_AND_SHORT_B = {
    'x': {('A', 0): 0.0, ('A', 1): 10.0, ('A', 2): 20.0, ('B', 0): 10.0, ('B', 1): 20.0},
    'y': {('A', 0): 0.0, ('A', 1): 50.0, ('A', 2): 140.0, ('B', 0): 100.0, ('B', 1): 130.0},
}


def curve_frame(values):
    return pl.DataFrame(
        {
            'generator': [g for g, _ in values],
            'bp': [k for _, k in values],
            'value': list(values.values()),
        }
    )


@pytest.fixture
def short_curve_inputs():
    """B's rows stop at its second breakpoint, and the mask says so."""
    present = {(g, k): ((g, k) in A_AND_SHORT_B['x']) for g in ('A', 'B') for k in range(3)}
    return {
        'generator': ['A', 'B'],
        'bp': [0, 1, 2],
        'load': pl.DataFrame({'value': [25.0]}),
        'p_max': pl.DataFrame({'generator': ['A', 'B'], 'value': [20.0, 20.0]}),
        'bp_x': curve_frame(A_AND_SHORT_B['x']),
        'bp_y': curve_frame(A_AND_SHORT_B['y']),
        'bp_present': curve_frame(present),
    }


@pytest.mark.parametrize('method', ['adjacency', 'convex', 'lp'])
def test_both_lanes_agree_on_a_masked_curve(short_curve_inputs, method, tmp_path):
    """Whatever the mask reaches has to reach it on both lanes (hard rule 3).

    `lp` is the one whose rows the mask reaches directly, and the one whose
    domain rows sit on each curve's own first and last breakpoint rather than
    the axis'. Testing only the default method left that pair unbuilt on the
    eager lane, where the constant-side coverage guard refuses a curve its
    breakpoints stop short of.
    """
    raw = override(raw_of(SHORT_CURVE), **{'piecewise.cost_curve.method': method})
    if method == 'lp':
        raw['piecewise']['cost_curve']['links'][1] = ['op_cost', 'bp_y', '>=']
    path = tmp_path / 'masked.yaml'
    path.write_text(pyyaml.safe_dump(raw))

    built = lpspec_linopy.build(path, short_curve_inputs)
    built.solve('highs', output_flag=False)

    assert float(built.objective.value) == pytest.approx(155.0)
    assert lps.solve(raw, short_curve_inputs).objective == pytest.approx(155.0), 'and the same on the other lane'


@pytest.mark.parametrize('method', ['adjacency', 'sos2', 'convex', 'lp'])
def test_a_masked_curve_reaches_the_optimum_its_own_points_put_it_at(short_curve_inputs, method):
    """Every method reads the mask, and two of them have no other way to take a short curve.

    `convex` and `lp` require strictly increasing breakpoints, so the padding
    that serves `adjacency` and `sos2` is refused there — before this the
    shorter curve could not be written at all.
    """
    raw = override(raw_of(SHORT_CURVE), **{'piecewise.cost_curve.method': method})
    if method == 'lp':
        raw['piecewise']['cost_curve']['links'][1] = ['op_cost', 'bp_y', '>=']

    result = lps.solve(raw, short_curve_inputs)

    assert result.objective == pytest.approx(155.0), 'B runs at 20 on its own two points, A at 5'


def test_the_mask_is_smaller_than_padding_the_curve_out(short_curve_inputs):
    """What the mask buys: the padded breakpoint costs a weight and a binary."""
    padded = {k: v for k, v in short_curve_inputs.items() if k != 'bp_present'}
    padded['bp_x'] = curve_frame({**A_AND_SHORT_B['x'], ('B', 2): 20.0})
    padded['bp_y'] = curve_frame({**A_AND_SHORT_B['y'], ('B', 2): 130.0})
    unmasked = raw_of(SHORT_CURVE)
    del unmasked['piecewise']['cost_curve']['points']
    del unmasked['parameters']['bp_present']

    with lps.build(raw_of(SHORT_CURVE), short_curve_inputs) as masked_model:
        masked = masked_model.diagnostics()
        assert masked_model.solve('highs').objective == pytest.approx(155.0)
    with lps.build(unmasked, padded) as padded_model:
        grown = padded_model.diagnostics()
        assert padded_model.solve('highs').objective == pytest.approx(155.0), 'the same answer, larger'

    assert masked.columns < grown.columns, 'a masked breakpoint declares no weight and no segment binary'


def test_a_masked_breakpoint_declares_no_segment_binary(short_curve_inputs):
    """The mask is on the declarations, and the binaries are half of what it saves.

    The answer alone cannot see this: an unmasked binary at a breakpoint no
    weight reaches is slack the solver never uses, so the objective is right
    either way and the MILP is bigger for nothing.
    """
    result = lps.solve(raw_of(SHORT_CURVE), short_curve_inputs)

    built = {(row['generator'], row['bp']) for row in result.primal('cost_curve_seg').to_dicts()}

    assert ('B', 2) not in built, "B's curve stops at bp 1, so bp 2 has no segment to pick"
    assert ('A', 2) in built, 'A runs the whole axis'


@pytest.mark.parametrize(
    ('present', 'match'),
    [
        pytest.param(
            {('A', 0): True, ('A', 1): True, ('A', 2): True, ('B', 0): True, ('B', 1): False, ('B', 2): True},
            'not consecutive',
            id='a-gap-in-the-mask',
        ),
        pytest.param(
            {('A', 0): True, ('A', 1): True, ('A', 2): True, ('B', 0): False, ('B', 1): False, ('B', 2): False},
            'not consecutive',
            id='a-curve-of-no-points-at-all',
        ),
    ],
)
def test_a_mask_with_a_gap_in_it_is_refused(short_curve_inputs, present, match):
    """The emitted rows read the mask as a length, so a gap builds a different curve.

    The chord joins a breakpoint to the one before it and the upper domain row
    is written where the mask stops; across a gap both are wrong, and neither
    is wrong in a way the answer shows.
    """
    data = {**short_curve_inputs, 'bp_present': curve_frame(present)}

    with pytest.raises(DataError, match=match):
        tidy_sources(schema_of(raw_of(SHORT_CURVE)), data)


def test_values_missing_where_the_mask_says_present_are_still_refused(short_curve_inputs):
    """#1105's guard follows the mask rather than the whole product."""
    thin = {k: v for k, v in A_AND_SHORT_B['x'].items() if k != ('A', 2)}
    data = {**short_curve_inputs, 'bp_x': curve_frame(thin)}

    with pytest.raises(DataError, match=r"'bp_x' has no value at"):
        tidy_sources(schema_of(raw_of(SHORT_CURVE)), data)


def test_the_hole_message_offers_the_mask_to_a_block_that_has_none(short_curve_inputs):
    """A ragged curve meets this message first, so it is where `points:` is discovered.

    With a mask already declared the same advice would be wrong — the reader
    said how far the curve runs and the values disagree — so the way out is
    named against what the block has.
    """
    unmasked = raw_of(SHORT_CURVE)
    del unmasked['piecewise']['cost_curve']['points'], unmasked['parameters']['bp_present']
    ragged = {k: v for k, v in short_curve_inputs.items() if k != 'bp_present'}

    with pytest.raises(DataError, match='points: a mask over the curve') as offered:
        tidy_sources(schema_of(unmasked), ragged)
    assert '#1101' in str(offered.value), 'the arity escape is the other way out, and a different one'

    thin = {k: v for k, v in A_AND_SHORT_B['x'].items() if k != ('A', 2)}
    with pytest.raises(DataError, match=r"'bp_present' claims this breakpoint"):
        tidy_sources(schema_of(raw_of(SHORT_CURVE)), {**short_curve_inputs, 'bp_x': curve_frame(thin)})


def test_values_the_mask_leaves_out_are_left_alone(short_curve_inputs):
    """A table wider than the block uses is ordinary, not an error."""
    spare = {**A_AND_SHORT_B['x'], ('B', 2): 999.0}
    data = {**short_curve_inputs, 'bp_x': curve_frame(spare)}

    assert lps.solve(raw_of(SHORT_CURVE), data).objective == pytest.approx(155.0), 'the masked row is not read'


@pytest.mark.parametrize(
    ('patch', 'match'),
    [
        pytest.param({'piecewise.cost_curve.points': 'nope'}, 'undeclared parameter', id='names-nothing'),
        pytest.param(
            {'parameters.bp_present': {'dims': ['generator'], 'dtype': 'bool'}},
            "must carry dim 'bp'",
            id='does-not-run-along-the-breakpoints',
        ),
        pytest.param(
            {
                'dimensions.zone': {'dtype': 'str'},
                'parameters.bp_present': {'dims': ['zone', 'bp'], 'dtype': 'bool'},
            },
            'which the links do not',
            id='carries-a-dim-the-block-does-not',
        ),
    ],
)
def test_a_mask_the_block_cannot_use_is_a_load_error(patch, match):
    with pytest.raises(PiecewiseExpansionError, match=match):
        lps.check(override(raw_of(SHORT_CURVE), **patch))


@pytest.mark.parametrize('method', ['adjacency', 'sos2', 'convex', 'lp'])
def test_points_may_name_the_curve_that_already_says_how_long_it_is(short_curve_inputs, method):
    """`points: bp_x` — no mask table, because a length is a fact of the curve.

    What the second table was for is still done: every other link is checked
    against the one named, so a row missing from `bp_y` is refused. What is
    given up is a row missing from `bp_x` itself, which is the parameter the
    file nominated as the length.
    """
    raw = override(
        raw_of(SHORT_CURVE), **{'piecewise.cost_curve.points': 'bp_x', 'piecewise.cost_curve.method': method}
    )
    del raw['parameters']['bp_present']
    if method == 'lp':
        raw['piecewise']['cost_curve']['links'][1] = ['op_cost', 'bp_y', '>=']
    data = {k: v for k, v in short_curve_inputs.items() if k != 'bp_present'}

    assert lps.solve(raw, data).objective == pytest.approx(155.0), 'the same curve, one table fewer'

    thin = {k: v for k, v in A_AND_SHORT_B['y'].items() if k != ('A', 2)}
    with pytest.raises(DataError, match=r"'bp_y' has no value at"):
        lps.solve(raw, {**data, 'bp_y': curve_frame(thin)})


def test_a_nominated_curve_is_read_for_its_rows_not_its_values(short_curve_inputs):
    """A breakpoint at zero is a breakpoint, so the mask cannot be the values' truthiness."""
    raw = override(raw_of(SHORT_CURVE), **{'piecewise.cost_curve.points': 'bp_x'})
    del raw['parameters']['bp_present']
    data = {k: v for k, v in short_curve_inputs.items() if k != 'bp_present'}

    result = lps.solve(raw, data)

    at_zero = [row for row in result.primal('cost_curve_lam').to_dicts() if (row['generator'], row['bp']) == ('A', 0)]
    assert at_zero, "A's curve starts at x = 0, and that breakpoint carries a weight like any other"


def test_a_curve_may_start_anywhere_on_the_axis(short_curve_inputs):
    """Contiguous, not a prefix: what the rows need is a predecessor, not the axis head.

    `lp` is the method that used to need the axis' own first and last
    breakpoint; it reads each curve's own now, so a curve numbered from 1 is
    the same curve one label along.
    """
    shifted_x = {('A', 1): 0.0, ('A', 2): 10.0, ('B', 1): 10.0, ('B', 2): 20.0}
    shifted_y = {('A', 1): 0.0, ('A', 2): 50.0, ('B', 1): 100.0, ('B', 2): 130.0}
    raw = override(raw_of(SHORT_CURVE), **{'piecewise.cost_curve.points': 'bp_x', 'piecewise.cost_curve.method': 'lp'})
    del raw['parameters']['bp_present']
    raw['piecewise']['cost_curve']['links'][1] = ['op_cost', 'bp_y', '>=']
    data = {
        **{k: v for k, v in short_curve_inputs.items() if k != 'bp_present'},
        'load': pl.DataFrame({'value': [15.0]}),
        'bp_x': curve_frame(shifted_x),
        'bp_y': curve_frame(shifted_y),
    }

    assert lps.solve(raw, data).objective == pytest.approx(115.0), (
        'B runs at 15 on its own domain and A stays off — a curve one label along is the same curve'
    )


def test_a_curve_with_a_gap_is_still_refused(short_curve_inputs):
    """Contiguity is what the chord row needs; the axis head was never the point."""
    gapped = {('A', 0): 0.0, ('A', 2): 20.0, ('B', 0): 10.0, ('B', 1): 20.0}
    raw = override(raw_of(SHORT_CURVE), **{'piecewise.cost_curve.points': 'bp_x'})
    del raw['parameters']['bp_present']
    data = {k: v for k, v in short_curve_inputs.items() if k != 'bp_present'}

    with pytest.raises(DataError, match='not consecutive'):
        tidy_sources(schema_of(raw), {**data, 'bp_x': curve_frame(gapped)})


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
        'seg_slope': slopes.T.stack().rename_axis(['generator', 'segment']).rename('value').reset_index(),
        'seg_intercept': icepts.T.stack().rename_axis(['generator', 'segment']).rename('value').reset_index(),
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
        slopes = data['seg_slope'].set_index(['generator', 'segment'])['value'].unstack('segment')
        icepts = data['seg_intercept'].set_index(['generator', 'segment'])['value'].unstack('segment')
        for (s, g), pv in p.items():
            expected = max(sl * pv + ic for sl, ic in zip(slopes.loc[g], icepts.loc[g], strict=True))
            assert gc[(s, g)] == pytest.approx(expected, abs=1e-6)
