"""``project``: the feasible region on two quantities, traced by solving along directions.

Every polygon here is one a reader can draw by hand, and the one that is not —
the CHP plant — is enumerated by brute force instead: every triple of its
constraint planes intersected, the feasible intersections kept, and their
shadow on the two axes hulled. That oracle shares no code with the tracer
beyond the hull of a point set, which the hand-drawn polygons check first.
"""

from __future__ import annotations

import itertools
import math
import sys
from typing import Any

import numpy as np
import polars as pl
import pytest

import lpspec as lps
from lpspec.errors import LpspecError, NoSolutionError
from lpspec.projection import _hull
from tests.conftest import override, port_sources, port_spec, raw_of

#: Two flows over two hours, one capped by data and one by a literal, tied by
#: a shared limit — a pentagon at the first hour, since the limit cuts the
#: corner the two caps would otherwise reach.
CORNER: dict[str, Any] = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'cap': {'dims': ['t']}, 'limit': {'dims': []}},
    'variables': {
        'a': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 'cap'}},
        'b': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 4}},
    },
    'expressions': {'total_a': 'sum(a)'},
    'constraints': {'shared': {'foreach': ['t'], 'expression': 'a + b <= limit'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(a) + sum(b)'},
}

CORNER_SOURCES: dict[str, Any] = {
    't': [0, 1],
    'cap': pl.DataFrame({'t': [0, 1], 'value': [4.0, 1.0]}),
    'limit': 6.0,
}


def hull(region: lps.Region) -> list[tuple[float, float]]:
    return [tuple(row) for row in region.hull.select(region.x, region.y).rows()]


def piece(region: lps.Region, i: int) -> list[tuple[float, float]]:
    return [tuple(row) for row in region.vertices.filter(pl.col('piece') == i).select(region.x, region.y).rows()]


def bound_by(region: lps.Region, i: int, edge: int) -> list[tuple[str, str, str]]:
    on = region.edges.filter((pl.col('piece') == i) & (pl.col('edge') == edge))
    return [tuple(row) for row in on.select('kind', 'name', 'side').rows()]


def committed(**patch: Any) -> dict[str, Any]:
    """CORNER with an on/off state per hour: ``a`` needs the unit on, and on means at least one."""
    return override(
        CORNER,
        **{
            'variables.on': {'foreach': ['t'], 'domain': 'binary'},
            'constraints.cap_on': {'foreach': ['t'], 'expression': 'a <= cap * on'},
            'constraints.min_load': {'foreach': ['t'], 'expression': 'a >= 1 * on'},
            **patch,
        },
    )


# ----------------------------------------------------------------------------
# The polygon
# ----------------------------------------------------------------------------


def test_a_region_one_axis_direction_cannot_see_is_found_by_refinement():
    """``a + b <= 6`` cuts the corner of the ``[0, 4] by [0, 4]`` box, and no
    compass direction lands on that edge's ends: only probing the outward
    normal of the edge the compass drew between ``(4, 0)`` and ``(0, 4)``
    finds ``(4, 2)`` and ``(2, 4)``."""
    region = lps.project(CORNER, CORNER_SOURCES, x='a', y='b', at={'t': 0})
    assert hull(region) == [(0, 0), (4, 0), (4, 2), (2, 4), (0, 4)], (
        'the pentagon, counter-clockwise from the origin, with the cut corner as two vertices'
    )


def test_every_frame_keeps_its_schema_when_the_binaries_are_free():
    region = lps.project(CORNER, CORNER_SOURCES, x='a', y='b', at={'t': 0})
    assert (region.x, region.y) == ('a', 'b'), 'the region names its axes as the caller did'
    assert region.vertices.columns == ['piece', 'vertex', 'a', 'b'], 'one piece, numbered, its vertices counted'
    assert region.vertices['piece'].unique().to_list() == [0], 'a free trace is piece 0'
    assert region.hull.columns == ['vertex', 'a', 'b'], 'the hull is the piece itself, without the piece column'
    assert region.pieces.columns == ['piece', 'variable', 'value'] and region.pieces.is_empty(), (
        'nothing pinned: the frame has its schema and no rows'
    )
    assert region.edges.columns == ['piece', 'edge', 'kind', 'name', 't', 'side'], (
        'what bounds each edge, the dim of the bounds it names between name and side'
    )
    assert region.optimum.columns == ['piece', 'a', 'b'], 'where the model as written lands, and in which piece'


def test_each_edge_names_what_bounds_it():
    """The pentagon's five edges, in order: the floor is ``b``'s lower bound,
    the right wall ``a``'s upper, the diagonal the shared limit, the ceiling
    ``b``'s upper, the left wall ``a``'s lower."""
    region = lps.project(CORNER, CORNER_SOURCES, x='a', y='b', at={'t': 0})
    assert [bound_by(region, 0, e) for e in range(5)] == [
        [('variable', 'b', 'lower')],
        [('variable', 'a', 'upper')],
        [('constraint', 'shared', 'upper')],
        [('variable', 'b', 'upper')],
        [('variable', 'a', 'lower')],
    ], 'one bound or row per edge, and the one the reader would name'
    assert region.edges['t'].unique().to_list() == [0], 'only the hour at names is reported; the other hour is parked'


def test_the_optimum_is_marked_where_the_model_as_written_lands():
    region = lps.project(CORNER, CORNER_SOURCES, x='a', y='b', at={'t': 0})
    assert region.optimum.rows() == [(0, 0.0, 0.0)], 'minimising both flows puts the optimum at the origin, in piece 0'


def test_a_spec_without_an_objective_has_no_optimum():
    spec = {k: v for k, v in CORNER.items() if k != 'objective'}
    region = lps.project(spec, CORNER_SOURCES, x='a', y='b', at={'t': 0})
    assert region.optimum.is_empty() and region.optimum.columns == ['piece', 'a', 'b'], (
        'nothing to solve as written, so the frame keeps its schema and has no rows'
    )
    assert hull(region) == [(0, 0), (4, 0), (4, 2), (2, 4), (0, 4)], 'and the region itself needs no objective'


def test_without_at_a_quantity_is_summed_over_every_dim():
    """Over both hours the caps add — 4 + 1 on ``a``, 4 + 4 on ``b``, 6 + 6
    shared — so the region is the same shape at a larger scale."""
    region = lps.project(CORNER, CORNER_SOURCES, x='a', y='b')
    assert hull(region) == [(0, 0), (5, 0), (5, 6), (3, 8), (0, 8)], (
        'the hour-by-hour pentagons summed: each vertex is the sum of the two hours at the same direction'
    )
    assert sorted(region.edges['t'].unique().to_list()) == [0, 1], 'both hours bound the summed region'


def test_a_scalar_expression_is_an_axis_as_it_stands():
    """``total_a`` already sums ``a``, so it is read as it is rather than summed
    again, which the language would refuse."""
    region = lps.project(CORNER, CORNER_SOURCES, x='total_a', y='b')
    assert hull(region) == [(0, 0), (5, 0), (5, 6), (3, 8), (0, 8)], (
        'the same polygon as summing the variable, since that is what the expression declares'
    )


def test_a_region_that_is_a_segment_has_two_vertices_and_one_edge():
    """Two quantities tied by an equality trace a segment, and both its ends
    are found by probing the segment's two outward normals."""
    spec = override(CORNER, **{'constraints.shared.expression': 'a == b'})
    region = lps.project(spec, CORNER_SOURCES, x='a', y='b', at={'t': 0})
    assert hull(region) == [(0, 0), (4, 4)], 'the diagonal of the box, ordered from the origin'
    assert region.edges['edge'].to_list() == [0], 'a segment is one edge, whichever way it is walked'
    assert bound_by(region, 0, 0) == [('constraint', 'shared', 'equal')], 'and the equality is what it sits on'


def test_a_region_that_is_a_point_has_one_vertex_and_no_edge():
    spec = override(
        CORNER, **{'variables.a.bounds': {'lower': 2, 'upper': 2}, 'variables.b.bounds': {'lower': 3, 'upper': 3}}
    )
    region = lps.project(spec, CORNER_SOURCES, x='a', y='b', at={'t': 0})
    assert hull(region) == [(2, 3)], 'one row for a region with no extent'
    assert region.edges.is_empty(), 'a point has nothing to bound'


def test_an_integer_model_gives_the_hull_of_its_region():
    """Integers along the diagonal are a row of dots; the trace returns the
    segment through them, which is their convex hull and not the dots."""
    spec = override(
        CORNER,
        **{
            'variables.a.domain': 'integer',
            'variables.b.domain': 'integer',
            'constraints.shared.expression': 'a == b',
        },
    )
    region = lps.project(spec, CORNER_SOURCES, x='a', y='b', at={'t': 0})
    assert hull(region) == [(0, 0), (4, 4)], 'the hull of five integer points on the diagonal is its two ends'


def test_the_probe_stays_on_the_fast_path(monkeypatch: pytest.MonkeyPatch):
    """Every direction is two costs and the optimum a third, so the solver
    holding the model is never reloaded: as many solves as probes, one load."""
    from lpspec import projection

    seen: list[Any] = []
    original = projection.Model

    class Watched(original):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            seen.append(self)

    monkeypatch.setattr(projection, 'Model', Watched)
    lps.project(CORNER, CORNER_SOURCES, x='a', y='b', at={'t': 0})
    (model,) = seen
    diagnostics = model.diagnostics()
    assert diagnostics.loads == 1, 'one hand-over; every probe after it pushes costs onto the loaded solver'
    assert diagnostics.solves >= 6, 'four compass probes, at least one along an edge, and the optimum'


# ----------------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------------


def test_at_on_a_scalar_quantity_is_refused():
    with pytest.raises(LpspecError, match="'total_a' does not carry: it is read over no dims"):
        lps.project(CORNER, CORNER_SOURCES, x='total_a', y='b', at={'t': 0})


def test_at_naming_a_dim_the_quantity_does_not_carry_is_refused():
    """A selection over a dim the quantity lacks would broadcast rather than
    select, so the check reads the quantity's dims off the built model."""
    spec = override(
        CORNER,
        **{
            'dimensions.unit': {'dtype': 'str'},
            'variables.b.foreach': ['unit'],
            'constraints.shared.foreach': ['t', 'unit'],
        },
    )
    sources = {**CORNER_SOURCES, 'unit': ['chp']}
    with pytest.raises(LpspecError, match="at names \\['t'\\], which 'b' does not carry: it is read over \\['unit'\\]"):
        lps.project(spec, sources, x='a', y='b', at={'t': 0})


def test_at_naming_no_declared_dimension_is_refused():
    with pytest.raises(LpspecError, match="at names \\['hour'\\], which the spec declares no dimension for"):
        lps.project(CORNER, CORNER_SOURCES, x='a', y='b', at={'hour': 0})


def test_an_unknown_quantity_names_the_declared_ones():
    with pytest.raises(KeyError, match="unknown variable or named expression 'c'"):
        lps.project(CORNER, CORNER_SOURCES, x='a', y='c')


def test_the_same_quantity_on_both_axes_is_refused():
    with pytest.raises(LpspecError, match="both axes are 'a'"):
        lps.project(CORNER, CORNER_SOURCES, x='a', y='a')


def test_a_name_the_probe_adds_is_refused_where_the_spec_declares_it():
    spec = override(CORNER, **{'parameters.objective_weight': {'dims': []}})
    with pytest.raises(LpspecError, match='already declares objective_weight under parameters:'):
        lps.project(spec, {**CORNER_SOURCES, 'objective_weight': 1.0}, x='a', y='b')


def test_a_lowered_program_is_refused():
    with pytest.raises(LpspecError, match='not a Program'):
        lps.project(lps.check(CORNER), CORNER_SOURCES, x='a', y='b')


def test_an_infeasible_model_has_no_region():
    spec = override(CORNER, **{'variables.a.bounds.lower': 10})
    with pytest.raises(NoSolutionError, match='infeasible'):
        lps.project(spec, CORNER_SOURCES, x='a', y='b')


def test_an_unbounded_region_names_the_direction():
    spec = override(CORNER, **{'variables.a.bounds': {'lower': 0}, 'constraints': {}})
    with pytest.raises(LpspecError, match=r'unbounded toward \(\+1·a, \+0·b\)'):
        lps.project(spec, CORNER_SOURCES, x='a', y='b')


def test_a_trace_that_never_settles_stops_rather_than_running_on(monkeypatch: pytest.MonkeyPatch):
    """The cap is the guard against solver noise past the tolerance; lowered to
    the four compass solves, the first edge probe is the one too many."""
    from lpspec import projection

    monkeypatch.setattr(projection, '_MOST_SOLVES', 4)
    with pytest.raises(LpspecError, match='for 4 solves without settling'):
        lps.project(CORNER, CORNER_SOURCES, x='a', y='b', at={'t': 0})


# ----------------------------------------------------------------------------
# Binaries: every combination, one piece each
# ----------------------------------------------------------------------------


def test_each_combination_of_the_binaries_is_its_own_piece():
    """Off leaves the ``a = 0`` segment; on cuts the strip below minimum load
    off the pentagon. Neither is the hull, which is what ``free`` gives."""
    region = lps.project(committed(), CORNER_SOURCES, x='a', y='b', at={'t': 0}, binaries='each')
    assert region.pieces.rows() == [(0, 'on', 0, 0), (1, 'on', 0, 1)], (
        'one row per pinned column per piece: the piece, the variable, its coordinate as a typed column, the value'
    )
    assert piece(region, 0) == [(0, 0), (0, 4)], 'off: a is zero, b is free'
    assert piece(region, 1) == [(1, 0), (4, 0), (4, 2), (2, 4), (1, 4)], (
        'on: the pentagon without the strip below the minimum load'
    )
    assert hull(region) == [(0, 0), (4, 0), (4, 2), (2, 4), (0, 4)], (
        'the hull of the pieces hides the strip neither piece covers'
    )


def test_a_piece_edge_names_the_pin_it_sits_on_through_the_rows_the_pin_moved():
    """With the unit off, the segment sits on ``a``'s lower bound and on the
    two rows the binary drives to zero; the pinning rows themselves are not
    reported, and the binary is not either, being always on a bound."""
    region = lps.project(committed(), CORNER_SOURCES, x='a', y='b', at={'t': 0}, binaries='each')
    assert bound_by(region, 0, 0) == [
        ('variable', 'a', 'lower'),
        ('constraint', 'cap_on', 'upper'),
        ('constraint', 'min_load', 'lower'),
    ], 'what holds a at zero with the unit off'
    assert not region.edges['name'].str.contains('pinned').any(), 'the rows the probe added to pin are left out'
    assert 'on' not in region.edges['name'].to_list(), 'and so is the binary'


def test_the_optimum_lands_in_the_piece_the_model_as_written_chooses():
    region = lps.project(committed(), CORNER_SOURCES, x='a', y='b', at={'t': 0}, binaries='each')
    assert region.optimum.rows() == [(0, 0.0, 0.0)], 'minimising the flows switches the unit off: piece 0'


def test_free_binaries_give_the_hull_the_pieces_fill():
    free = lps.project(committed(), CORNER_SOURCES, x='a', y='b', at={'t': 0})
    each = lps.project(committed(), CORNER_SOURCES, x='a', y='b', at={'t': 0}, binaries='each')
    assert free.pieces.is_empty() and free.vertices['piece'].unique().to_list() == [0], (
        'free is one piece, nothing pinned'
    )
    assert hull(free) == hull(each), 'and that piece is the hull the pieces span'


def test_labels_say_what_varies_between_the_pieces():
    """At one hour the hour is the same in every piece and is dropped; over
    the horizon it varies, and a numbered dim keeps its name."""
    at_one_hour = lps.project(committed(), CORNER_SOURCES, x='a', y='b', at={'t': 0}, binaries='each')
    assert [at_one_hour.label(i) for i in range(2)] == ['on=0', 'on=1'], 'nothing but the value varies'
    over_both = lps.project(committed(), CORNER_SOURCES, x='a', y='b', binaries='each')
    assert [over_both.label(i) for i in range(4)] == [
        'on[t=0]=0, on[t=1]=0',
        'on[t=0]=0, on[t=1]=1',
        'on[t=0]=1, on[t=1]=0',
        'on[t=0]=1, on[t=1]=1',
    ], 'every column of the binary, counted like a binary number, the hour named'
    assert piece(over_both, 3) == [(2, 0), (5, 0), (5, 6), (3, 8), (2, 8)], (
        'both on: each hour contributes its minimum load, so a starts at two'
    )


def test_a_label_along_a_string_dim_spells_the_label_alone():
    spec = override(
        CORNER,
        **{
            'dimensions.unit': {'dtype': 'str'},
            'variables.on': {'foreach': ['t', 'unit'], 'domain': 'binary'},
            'constraints.cap_on': {'foreach': ['t'], 'expression': 'a <= cap * sum(on, over=unit)'},
        },
    )
    region = lps.project(
        spec, {**CORNER_SOURCES, 'unit': ['chp', 'boiler']}, x='a', y='b', at={'t': 0}, binaries='each'
    )
    assert region.pieces.columns == ['piece', 'variable', 't', 'unit', 'value'], (
        'both dims, typed, in declaration order'
    )
    assert region.label(3) == 'on[chp]=1, on[boiler]=1', 'the unit names carry the label; the fixed hour is dropped'


def test_an_infeasible_combination_is_left_out():
    spec = committed(**{'constraints.someone_on': {'foreach': [], 'expression': 'sum(on) >= 1'}})
    region = lps.project(spec, CORNER_SOURCES, x='a', y='b', binaries='each')
    assert [region.label(i) for i in range(3)] == [
        'on[t=0]=0, on[t=1]=1',
        'on[t=0]=1, on[t=1]=0',
        'on[t=0]=1, on[t=1]=1',
    ], 'both off breaks the rule that someone is on, and is not a piece'
    assert region.vertices['piece'].unique().to_list() == [0, 1, 2], 'the pieces are numbered as they were found'


def test_an_infeasible_model_has_no_pieces_either():
    """The first solve leaves every binary free, so a model with no region at
    all says so before any combination is pinned."""
    spec = committed(**{'variables.a.bounds.lower': 10})
    with pytest.raises(NoSolutionError, match='the model is infeasible'):
        lps.project(spec, CORNER_SOURCES, x='a', y='b', binaries='each')


def test_a_masked_binary_column_is_not_pinned():
    """A column the variable's ``where`` removed does not exist to pin, so
    only the hour that has a state is a combination."""
    spec = committed(
        **{'variables.on.where': 't == 0', 'constraints.cap_on.where': 't == 0', 'constraints.min_load.where': 't == 0'}
    )
    region = lps.project(spec, CORNER_SOURCES, x='a', y='b', binaries='each')
    assert region.pieces.rows() == [(0, 'on', 0, 0), (1, 'on', 0, 1)], 'the second hour has no state'


def test_the_pieces_move_on_the_fast_path(monkeypatch: pytest.MonkeyPatch):
    """A pin is two right-hand sides, so a combination is pushed onto the
    loaded solver: one load for every piece of every combination."""
    from lpspec import projection

    seen: list[Any] = []
    original = projection.Model

    class Watched(original):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            seen.append(self)

    monkeypatch.setattr(projection, 'Model', Watched)
    lps.project(committed(), CORNER_SOURCES, x='a', y='b', binaries='each')
    (model,) = seen
    assert model.diagnostics().loads == 1, 'four combinations traced, and the solver was handed the model once'


def test_each_on_a_model_with_no_binary_is_refused():
    with pytest.raises(LpspecError, match='the spec declares none'):
        lps.project(CORNER, CORNER_SOURCES, x='a', y='b', binaries='each')


def test_more_binary_columns_than_the_cap_are_refused(monkeypatch: pytest.MonkeyPatch):
    from lpspec import projection

    monkeypatch.setattr(projection, '_MOST_PINNED', 1)
    with pytest.raises(
        LpspecError, match='every combination of 2 binary columns, which is 4 regions; the most it traces is 2'
    ):
        lps.project(committed(), CORNER_SOURCES, x='a', y='b', binaries='each')


def test_a_pin_name_the_spec_declares_is_refused():
    spec = committed(**{'parameters.on_at_least': {'dims': []}})
    with pytest.raises(LpspecError, match='already declares on_at_least under parameters:'):
        lps.project(spec, {**CORNER_SOURCES, 'on_at_least': 0.0}, x='a', y='b', binaries='each')


def test_binaries_outside_the_two_words_is_refused():
    with pytest.raises(LpspecError, match="binaries is 'free' or 'each'"):
        lps.project(CORNER, CORNER_SOURCES, x='a', y='b', binaries='all')  # pyrefly: ignore[bad-argument-type] — the refusal under test


# ----------------------------------------------------------------------------
# A plant, against brute force
# ----------------------------------------------------------------------------


def _chp_plant() -> tuple[dict[str, Any], dict[str, Any]]:
    """The multi-link CHP example with its balances relaxed to ``>=``.

    As published the balances are equalities, so what each bus receives is its
    load and the region is a point. Letting a bus be over-supplied is what
    makes heat against power a region: the gas well, the three conversions
    and the two loads bound it.
    """
    spec = override(
        raw_of(port_spec('pypsa_multilink')),
        **{
            'constraints.nodal_balance.expression': 'sum(gen, by=gen_bus) + sum(incidence * p, over=link) >= load',
            'parameters.is_heat': {'dims': ['bus']},
            'parameters.is_elec': {'dims': ['bus']},
            'expressions': {
                'delivered': 'sum(incidence * p, over=link)',
                'heat_out': 'sum(is_heat * delivered)',
                'elec_out': 'sum(is_elec * delivered)',
            },
        },
    )
    sources = {
        **port_sources('pypsa_multilink'),
        'is_heat': pl.DataFrame({'bus': ['heat'], 'value': [1.0]}),
        'is_elec': pl.DataFrame({'bus': ['elec'], 'value': [1.0]}),
    }
    return spec, sources


def _brute_force_chp_region() -> list[tuple[float, float]]:
    """The same region off the plant's own inequalities, with no solver.

    In the link draws ``(chp, boiler, ocgt)``: each capped, the gas well
    capping their sum, and the two loads as floors on what the outputs
    deliver. Every vertex of that polytope is where three of its planes
    meet, so intersecting every triple and keeping the feasible points is
    its whole vertex set, and the hull of their shadow is the region.
    """
    planes = np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 50],
            [0, 1, 0, 0],
            [0, 1, 0, 100],
            [0, 0, 1, 0],
            [0, 0, 1, 100],
            [1, 1, 1, 200],
            [0.4, 0.8, 0, 36],
            [0.4, 0, 0.5, 40],
        ]
    )
    lower = np.array([[0.4, 0.8, 0, 36], [0.4, 0, 0.5, 40], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    upper = np.array([[1, 0, 0, 50], [0, 1, 0, 100], [0, 0, 1, 100], [1, 1, 1, 200]])
    points = []
    for triple in itertools.combinations(planes, 3):
        a, b = np.array(triple)[:, :3], np.array(triple)[:, 3]
        if abs(np.linalg.det(a)) < 1e-9:
            continue
        v = np.linalg.solve(a, b)
        if np.all(lower[:, :3] @ v >= lower[:, 3] - 1e-9) and np.all(upper[:, :3] @ v <= upper[:, 3] + 1e-9):
            points.append((0.4 * v[0] + 0.8 * v[1], 0.4 * v[0] + 0.5 * v[2]))
    return _hull(points)


def test_the_chp_plant_traces_the_region_brute_force_enumerates():
    spec, sources = _chp_plant()
    region = lps.project(spec, sources, x='heat_out', y='elec_out')
    expected = _brute_force_chp_region()
    traced_vertices = hull(region)
    assert len(traced_vertices) == len(expected), (
        f'the trace found {len(traced_vertices)} vertices where enumeration finds {len(expected)}'
    )
    for traced, enumerated in zip(traced_vertices, expected, strict=True):
        assert math.isclose(traced[0], enumerated[0], abs_tol=1e-6) and math.isclose(
            traced[1], enumerated[1], abs_tol=1e-6
        ), f'vertex {traced} is not the enumerated {enumerated}, at solver precision'


def test_the_chp_plant_optimum_sits_on_the_region():
    """The published optimum delivers exactly the two loads, which is the
    region's lower-left corner: both floors bind there."""
    spec, sources = _chp_plant()
    region = lps.project(spec, sources, x='heat_out', y='elec_out')
    assert (36.0, 40.0) in hull(region), 'the two loads are a vertex of the region'
    assert region.optimum.select('heat_out', 'elec_out').rows() == [(36.0, 40.0)], (
        'and the optimum delivers exactly them'
    )
    with lps.solve(spec, sources) as result:
        assert result.objective == pytest.approx(1100.0), (
            'relaxing the balances to >= leaves the optimum where PyPSA put it'
        )


def test_the_chp_plant_edges_name_its_constraints():
    """The floor and the left wall are the two loads' balances; the well and the
    boiler's cap bound the rest."""
    spec, sources = _chp_plant()
    region = lps.project(spec, sources, x='heat_out', y='elec_out')
    named = region.edges.filter(pl.col('kind') == 'constraint').select('edge', 'name', 'bus').rows()
    assert (0, 'nodal_balance', 'elec') in named and (4, 'nodal_balance', 'heat') in named, (
        'the floor is the power balance and the left wall the heat balance, at their own buses'
    )
    assert 'nodal_balance' in set(region.edges['name']), 'the balances bound the plant'


# ----------------------------------------------------------------------------
# The picture
# ----------------------------------------------------------------------------


def traces(figure: Any, **match: Any) -> list[Any]:
    return [t for t in figure.data if all(getattr(t, k) == v for k, v in match.items())]


def test_a_polygon_is_a_filled_trace_with_its_vertices_and_edges_on_hover():
    pytest.importorskip('plotly')
    region = lps.project(CORNER, CORNER_SOURCES, x='a', y='b', at={'t': 0})
    figure = region.plot(name='hour 0')
    (polygon,) = traces(figure, fill='toself')
    assert polygon.name == 'hour 0' and list(zip(polygon.x, polygon.y, strict=True))[:-1] == hull(region), (
        'the fill is the polygon, vertex for vertex, closed, under the name the caller gave'
    )
    (vertices,) = [t for t in figure.data if t.mode == 'markers' and t.showlegend is False and t.marker.opacity is None]
    assert list(zip(vertices.x, vertices.y, strict=True)) == hull(region), 'every vertex is a hoverable marker'
    (edges,) = [t for t in figure.data if t.marker.opacity == 0]
    assert list(edges.text) == [
        'b[0] at its lower',
        'a[0] at its upper',
        'shared[0] at its upper',
        'b[0] at its upper',
        'a[0] at its lower',
    ], 'the middle of each edge reads what bounds it, the edges frame spelled out'
    assert (figure.layout.xaxis.title.text, figure.layout.yaxis.title.text) == ('a', 'b'), (
        'the axes are the two quantities'
    )


def test_the_optimum_is_a_marker_naming_where_it_landed():
    pytest.importorskip('plotly')
    region = lps.project(CORNER, CORNER_SOURCES, x='a', y='b', at={'t': 0})
    (marked,) = traces(region.plot(), name='the optimum')
    assert (list(marked.x), list(marked.y)) == ([0.0], [0.0]), 'the optimum is marked where the model as written lands'
    assert traces(region.plot(optimum=False), name='the optimum') == [], 'and left off when the caller asks'


def test_a_segment_is_a_line_and_a_point_a_marker():
    pytest.importorskip('plotly')
    segment = override(CORNER, **{'constraints.shared.expression': 'a == b'})
    (line,) = traces(lps.project(segment, CORNER_SOURCES, x='a', y='b', at={'t': 0}).plot(optimum=False), mode='lines')
    assert line.fill is None and len(line.x) == 2, 'a segment is a line, not a fill'
    point = override(
        CORNER, **{'variables.a.bounds': {'lower': 2, 'upper': 2}, 'variables.b.bounds': {'lower': 3, 'upper': 3}}
    )
    figure = lps.project(point, CORNER_SOURCES, x='a', y='b', at={'t': 0}).plot(optimum=False)
    assert [t.mode for t in figure.data] == ['markers', 'markers'], 'a point is a marker, its hover, and no edge'


def test_pieces_are_drawn_each_in_its_own_colour_under_its_label():
    pytest.importorskip('plotly')
    region = lps.project(committed(), CORNER_SOURCES, x='a', y='b', at={'t': 0}, binaries='each')
    figure = region.plot(optimum=False)
    named = [t for t in figure.data if t.showlegend is not False]
    assert [t.name for t in named] == ['on=0', 'on=1'], 'one legend entry per piece, under its label'
    assert named[0].line.color != named[1].line.color, 'each in its own colour'
    assert {t.legendgroup for t in figure.data} == {'0: on=0', '1: on=1'}, (
        'a piece and its hover markers share a legend group, so a click on the legend hides all of it'
    )


def test_a_second_region_on_the_same_figure_takes_the_next_colours():
    pytest.importorskip('plotly')
    first = lps.project(CORNER, CORNER_SOURCES, x='a', y='b', at={'t': 0})
    second = lps.project(CORNER, CORNER_SOURCES, x='a', y='b', at={'t': 1})
    figure = second.plot(first.plot(name='hour 0', optimum=False), name='hour 1', optimum=False)
    fills = traces(figure, fill='toself')
    assert [t.name for t in fills] == ['hour 0', 'hour 1'] and fills[0].line.color != fills[1].line.color, (
        'two regions on one figure, each under its own name and colour'
    )


def test_plot_without_plotly_names_the_extra(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, 'plotly.graph_objects', None)
    region = lps.project(CORNER, CORNER_SOURCES, x='a', y='b', at={'t': 0})
    with pytest.raises(ModuleNotFoundError, match=r'pip install "lpspec\[plot\]"'):
        region.plot()
