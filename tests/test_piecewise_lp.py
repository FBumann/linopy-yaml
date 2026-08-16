"""``method: lp`` — the curve as its own segment lines, and no weights at all.

Every other method interpolates: it declares a weight per breakpoint and ties
the linked expressions to a convex combination of them. This one does not
declare anything. Where the curve is convex and the linked expression is
bounded from below by it — a cost, the common case — the epigraph *is* the
intersection of the segments' half-planes, so the rows say it directly and the
K weights per frame row are simply not there.

Three things need holding:

**It is the same model.** `test_lp_and_convex_reach_one_optimum` solves the
same curve both ways; the objectives must agree, because a formulation that
merely relaxes differently is a different model wearing the same block.

**The saving is real and it is a trade.** Columns fall and rows rise, and
`test_the_saving_is_columns_paid_for_in_rows` records both — a claim about size
that only counted the half that improved would be worth nothing.

**The curvature is checked, because getting it wrong is silent.** Lines that
envelope a convex curve *cut* a concave one, and the solve comes back optimal
either way. That check needs the values, so it lives beside the one
`method: convex` already has.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import yaml as pyyaml

import lpspec as lps
from lpspec.errors import PiecewiseExpansionError
from tests.conftest import schema_of
from tests.differential import RTOL, differential
from tests.oracle import pd

MODEL = """
description: dispatch whose cost is read off a convex curve, stated as its segment lines

dimensions:
  snapshot: {dtype: int, description: dispatch periods}
  bp: {dtype: int, description: breakpoints of the cost curve}

parameters:
  load: {dims: [snapshot], description: demand to be met}
  bp_x: {dims: [bp], description: breakpoint output levels}
  bp_y: {dims: [bp], description: cost at each breakpoint}

variables:
  p:
    foreach: [snapshot]
    bounds: {lower: 0, upper: 100}
    description: dispatched power
  op_cost:
    foreach: [snapshot]
    bounds: {lower: 0}
    description: operating cost, read off the curve
  running:
    foreach: [snapshot]
    domain: binary
    description: unused here; a gate for the case lp cannot take one

piecewise:
  cost_curve:
    description: cost bounded below by the curve, which is exact where the curve is convex
    over: bp
    links:
      - [p, bp_x]
      - [op_cost, bp_y, '>=']
    method: lp

constraints:
  balance:
    foreach: [snapshot]
    expression: p == load
    description: output meets demand

objective:
  sense: minimize
  expression: sum(op_cost, over=snapshot)
  description: total operating cost
"""

#: Slopes 1, 2 then 3 — convex, and strictly increasing in x.
BREAKPOINTS_X = [0.0, 10.0, 20.0, 30.0]
BREAKPOINTS_Y = [0.0, 10.0, 30.0, 60.0]
SNAPSHOTS = [0, 1, 2]
LOAD = [5.0, 15.0, 25.0]


def _inputs():
    bp = pd.Index(range(len(BREAKPOINTS_X)), name='bp')
    snapshot = pd.Index(SNAPSHOTS, name='snapshot')
    return {
        'snapshot': snapshot,
        'bp': bp,
        'load': pd.Series(LOAD, index=snapshot),
        'bp_x': pd.Series(BREAKPOINTS_X, index=bp),
        'bp_y': pd.Series(BREAKPOINTS_Y, index=bp),
    }


def _on_the_curve(x: float) -> float:
    return float(np.interp(x, BREAKPOINTS_X, BREAKPOINTS_Y))


# ---------------------------------------------------------------------------
# it is the same model
# ---------------------------------------------------------------------------


def test_the_cost_lands_on_the_curve_and_both_lanes_agree():
    """The expansion is schema-level, so both lanes get identical affine rows.

    Under minimisation the epigraph binds, so each snapshot's cost is the
    curve read at its load — checked against a numpy interpolation, which is
    an oracle that involves no formulation at all.
    """
    with differential(MODEL, _inputs(), lp=True) as run:
        assert run.oracle == pytest.approx(sum(_on_the_curve(x) for x in LOAD), rel=RTOL)
        costs = dict(run.result.primal('op_cost').select('snapshot', 'value').iter_rows())

    for t, load in zip(SNAPSHOTS, LOAD, strict=True):
        assert costs[t] == pytest.approx(_on_the_curve(load), rel=1e-9), (
            f'the cost at load {load} is the curve read there, not a point above it'
        )


def test_a_curve_that_does_not_start_at_the_origin():
    """The case that decides whether the first breakpoint's row is excluded.

    The slope is a difference one position apart, so the first breakpoint has
    no predecessor and its row must not exist. `edge=0` makes the shift read a
    zero there, which is the *origin* — so the row that would be written is the
    line from the origin through the first breakpoint, extended.

    On a curve starting at (0, 0) that line is the first segment and the row is
    harmless, which is why this needs a curve that starts somewhere else. Here
    the ray has slope 10 against segment slopes of 1 and 2, so left in it would
    force a cost of 30 where the curve says 13 — and nothing would say so.
    """
    xs, ys, loads = [1.0, 2.0, 3.0], [10.0, 11.0, 13.0], [1.0, 2.0, 3.0]
    with lps.solve(pyyaml.safe_load(MODEL), _relational(load=loads, xs=xs, ys=ys)) as result:
        assert result.objective == pytest.approx(sum(ys)), (
            'each load sits on a breakpoint, so the total is the curve read at each'
        )


def test_lp_and_convex_reach_one_optimum():
    """The two formulations of the same convex curve are the same model.

    `convex` pins the cost to a convex combination of the breakpoints; `lp`
    bounds it below by the segment lines. Under minimisation both are exact,
    and an objective that differed would mean one of them is not.
    """
    hull = pyyaml.safe_load(MODEL)
    hull['piecewise']['cost_curve']['method'] = 'convex'
    hull['piecewise']['cost_curve']['links'] = [['p', 'bp_x'], ['op_cost', 'bp_y']]

    with lps.solve(pyyaml.safe_load(MODEL), _relational()) as lines, lps.solve(hull, _relational()) as weights:
        assert lines.objective == pytest.approx(weights.objective, rel=RTOL)


def test_the_domain_rows_hold_the_output_inside_the_curve():
    """A line does not stop where its segment does, which is why they are emitted.

    Asked for more than the last breakpoint, the model is **infeasible** rather
    than extrapolating along the last segment's slope — which is what `convex`
    does, and what `linopy`'s own `_add_lp` emits its domain rows for.
    """
    beyond = pyyaml.safe_load(MODEL)
    sources = _relational(load=[5.0, 15.0, 45.0])
    with lps.solve(beyond, sources) as result:
        assert not result.is_ok, 'a load past the last breakpoint is outside the curve, not on its last slope'


# ---------------------------------------------------------------------------
# what it costs
# ---------------------------------------------------------------------------


def test_the_saving_is_columns_paid_for_in_rows():
    """The trade, measured on one model rather than argued.

    `convex` declares a weight per breakpoint per frame row; `lp` declares
    none, and instead writes a row per segment plus two holding the domain.
    So this is columns traded for rows, and the numbers say how much.
    """
    sizes = {}
    for method, links in (
        ('convex', [['p', 'bp_x'], ['op_cost', 'bp_y']]),
        ('lp', [['p', 'bp_x'], ['op_cost', 'bp_y', '>=']]),
    ):
        model = pyyaml.safe_load(MODEL)
        model['piecewise']['cost_curve']['method'] = method
        model['piecewise']['cost_curve']['links'] = links
        built = lps.build(model, _relational())
        sizes[method] = built.diagnostics()
        built.close()

    weights = len(SNAPSHOTS) * len(BREAKPOINTS_X)
    assert sizes['convex'].columns - sizes['lp'].columns == weights, (
        f'lp declares no weights, so it is exactly the {weights} lambda columns lighter'
    )
    assert sizes['lp'].rows > sizes['convex'].rows, 'and it pays for them in rows — this is a trade, not a free win'


# ---------------------------------------------------------------------------
# the curvature that makes it exact
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('sign', 'sense', 'ys', 'wanted'),
    [
        pytest.param('>=', 'minimize', [0.0, 30.0, 50.0, 60.0], 'convex', id='a-cost-bounded-below-by-a-concave-curve'),
        pytest.param('<=', 'maximize', BREAKPOINTS_Y, 'concave', id='a-yield-bounded-above-by-a-convex-curve'),
    ],
)
def test_the_curvature_the_sign_states_is_required(sign, sense, ys, wanted):
    """The silent case, and the reason the guard is directional.

    Lines through a *concave* curve's breakpoints lie above it, so bounding a
    cost below by them cuts the feasible region — and the solve comes back
    optimal with a wrong answer. `method: convex`'s own guard would pass such a
    curve, since it is not mixed; only `lp` cares which way it bends.
    """
    model = pyyaml.safe_load(MODEL)
    model['piecewise']['cost_curve']['links'] = [['p', 'bp_x'], ['op_cost', 'bp_y', sign]]
    model['objective']['sense'] = sense
    with pytest.raises(PiecewiseExpansionError, match=f'exact only for a {wanted} curve'):
        lps.solve(model, _relational(ys=ys))
    assert schema_of(MODEL) is not None, 'and the schema alone is fine — this needs the values'


def test_breakpoints_that_do_not_increase_are_refused():
    """The run is what the row is multiplied through by, so it must be positive."""
    model = pyyaml.safe_load(MODEL)
    with pytest.raises(PiecewiseExpansionError, match='requires strictly increasing breakpoints'):
        lps.solve(model, _relational(xs=[0.0, 10.0, 10.0, 30.0]))


# ---------------------------------------------------------------------------
# the shape lp needs, refused at load
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('patch', 'match'),
    [
        pytest.param(
            {'links': [['p', 'bp_x'], ['op_cost', 'bp_y']]},
            'needs exactly one link bounded by the curve',
            id='both-links-pinned',
        ),
        pytest.param(
            {'links': [['p', 'bp_x'], ['op_cost', 'bp_y'], ['p', 'bp_x']]},
            'needs exactly one link bounded by the curve',
            id='three-links-none-bounded',
        ),
        pytest.param(
            {'active': 'running'},
            'active gating is not supported with method: lp',
            id='an-active-gate-with-nothing-to-gate',
        ),
    ],
)
def test_a_block_lp_cannot_state_is_refused_at_load(patch, match):
    """Refused rather than fallen back from: a method written down is a
    formulation chosen, and quietly building a different one is the thing a
    reviewer of the file could not see."""
    model = pyyaml.safe_load(MODEL)
    model['piecewise']['cost_curve'].update(patch)
    with pytest.raises(lps.SchemaError, match=match):
        schema_of(model)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _relational(load=None, xs=None, ys=None):
    xs = BREAKPOINTS_X if xs is None else xs
    ys = BREAKPOINTS_Y if ys is None else ys
    return {
        'snapshot': pl.DataFrame({'snapshot': SNAPSHOTS}),
        'bp': pl.DataFrame({'bp': list(range(len(xs)))}),
        'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': LOAD if load is None else load}),
        'bp_x': pl.DataFrame({'bp': list(range(len(xs))), 'value': xs}),
        'bp_y': pl.DataFrame({'bp': list(range(len(ys))), 'value': ys}),
    }
