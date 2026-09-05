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


def vertices(region: lps.Region) -> list[tuple[float, float]]:
    return [tuple(round(v, 6) for v in row) for row in region.vertices.rows()]


def test_a_region_one_axis_direction_cannot_see_is_found_by_refinement():
    """``a + b <= 6`` cuts the corner of the ``[0, 4] by [0, 4]`` box, and no
    compass direction lands on that edge's ends: only probing the outward
    normal of the edge the compass drew between ``(4, 0)`` and ``(0, 4)``
    finds ``(4, 2)`` and ``(2, 4)``."""
    region = lps.project(CORNER, CORNER_SOURCES, x='a', y='b', at={'t': 0})
    assert vertices(region) == [(0, 0), (4, 0), (4, 2), (2, 4), (0, 4)], (
        'the pentagon, counter-clockwise from the origin, with the cut corner as two vertices'
    )


def test_the_columns_are_named_after_the_quantities():
    region = lps.project(CORNER, CORNER_SOURCES, x='a', y='b', at={'t': 0})
    assert region.vertices.columns == ['a', 'b'], 'x then y, under the names the caller passed'
    assert (region.x, region.y) == ('a', 'b'), 'and the region names them the same way'


def test_without_at_a_quantity_is_summed_over_every_dim():
    """Over both hours the caps add — 4 + 1 on ``a``, 4 + 4 on ``b``, 6 + 6
    shared — so the region is the same shape at a larger scale."""
    region = lps.project(CORNER, CORNER_SOURCES, x='a', y='b')
    assert vertices(region) == [(0, 0), (5, 0), (5, 6), (3, 8), (0, 8)], (
        'the hour-by-hour pentagons summed: each vertex is the sum of the two hours at the same direction'
    )


def test_a_scalar_expression_is_an_axis_as_it_stands():
    """``total_a`` already sums ``a``, so it is read as it is rather than summed
    again, which the language would refuse."""
    region = lps.project(CORNER, CORNER_SOURCES, x='total_a', y='b')
    assert vertices(region) == [(0, 0), (5, 0), (5, 6), (3, 8), (0, 8)], (
        'the same polygon as summing the variable, since that is what the expression declares'
    )


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
    spec = override(CORNER, **{'parameters.x_direction': {'dims': []}})
    with pytest.raises(LpspecError, match='already declares x_direction under parameters:'):
        lps.project(spec, {**CORNER_SOURCES, 'x_direction': 1.0}, x='a', y='b')


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


def test_a_region_that_is_a_segment_has_two_vertices():
    """Two quantities tied by an equality trace a segment, and both its ends
    are found by probing the segment's two outward normals."""
    spec = override(CORNER, **{'constraints.shared.expression': 'a == b'})
    region = lps.project(spec, CORNER_SOURCES, x='a', y='b', at={'t': 0})
    assert vertices(region) == [(0, 0), (4, 4)], 'the diagonal of the box, ordered from the origin'


def test_a_region_that_is_a_point_has_one_vertex():
    spec = override(
        CORNER, **{'variables.a.bounds': {'lower': 2, 'upper': 2}, 'variables.b.bounds': {'lower': 3, 'upper': 3}}
    )
    region = lps.project(spec, CORNER_SOURCES, x='a', y='b', at={'t': 0})
    assert vertices(region) == [(2, 3)], 'one row for a region with no extent'


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
    assert vertices(region) == [(0, 0), (4, 4)], 'the hull of five integer points on the diagonal is its two ends'


def test_the_probe_stays_on_the_fast_path(monkeypatch: pytest.MonkeyPatch):
    """Every direction is two costs, so the solver holding the model is never
    reloaded: as many solves as probes, and one load."""
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
    assert diagnostics.solves >= 5, 'four compass probes plus at least one along an edge'


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
    traced_vertices = region.vertices.rows()
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
    with lps.solve(spec, sources) as result:
        assert result.objective == pytest.approx(1100.0), (
            'relaxing the balances to >= leaves the optimum where PyPSA put it'
        )
        delivered = (result.expression('heat_out').item(), result.expression('elec_out').item())
    assert (36.0, 40.0) in vertices(region), 'the two loads are a vertex of the region'
    assert delivered == pytest.approx((36.0, 40.0)), 'and the optimum delivers exactly them'


def test_a_trace_that_never_settles_stops_rather_than_running_on(monkeypatch: pytest.MonkeyPatch):
    """The cap is the guard against solver noise past the tolerance; lowered to
    the four compass solves, the first edge probe is the one too many."""
    from lpspec import projection

    monkeypatch.setattr(projection, '_MOST_SOLVES', 4)
    with pytest.raises(LpspecError, match='for 4 solves without settling'):
        lps.project(CORNER, CORNER_SOURCES, x='a', y='b', at={'t': 0})


# ----------------------------------------------------------------------------
# The picture
# ----------------------------------------------------------------------------


@pytest.fixture
def axes():
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    yield ax
    plt.close('all')


def test_a_polygon_is_filled_and_outlined_with_the_axes_labelled(axes):
    region = lps.project(CORNER, CORNER_SOURCES, x='a', y='b', at={'t': 0})
    ax = region.plot(axes, color='tab:orange', label='hour 0')
    assert ax is axes, 'drawn on the axes handed in, which comes back for the next call'
    (patch,) = ax.patches
    assert [tuple(v) for v in patch.get_xy()[:-1]] == vertices(region), 'the fill is the polygon, vertex for vertex'
    assert patch.get_label() == 'hour 0', 'a style keyword reaches the fill, so a legend can name the region'
    assert len(ax.lines) == 1, 'one outline, closed back to the first vertex'
    assert (ax.get_xlabel(), ax.get_ylabel()) == ('a', 'b'), 'the axes are the two quantities'


def test_a_segment_is_a_line_and_a_point_a_marker(axes):
    segment = override(CORNER, **{'constraints.shared.expression': 'a == b'})
    lps.project(segment, CORNER_SOURCES, x='a', y='b', at={'t': 0}).plot(axes)
    assert (len(axes.patches), len(axes.lines), len(axes.collections)) == (0, 1, 0), 'a segment fills nothing'
    point = override(
        CORNER, **{'variables.a.bounds': {'lower': 2, 'upper': 2}, 'variables.b.bounds': {'lower': 3, 'upper': 3}}
    )
    lps.project(point, CORNER_SOURCES, x='a', y='b', at={'t': 0}).plot(axes)
    assert (len(axes.patches), len(axes.lines), len(axes.collections)) == (0, 1, 1), 'a point is one marker on top'


def test_plot_without_an_axes_makes_its_own_figure():
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ax = lps.project(CORNER, CORNER_SOURCES, x='a', y='b', at={'t': 0}).plot()
    assert ax.figure is not None
    plt.close('all')


def test_plot_without_matplotlib_names_the_extra(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', None)
    region = lps.Region('a', 'b', pl.DataFrame({'a': [0.0], 'b': [0.0]}))
    with pytest.raises(ModuleNotFoundError, match=r'pip install "lpspec\[plot\]"'):
        region.plot()


# ----------------------------------------------------------------------------
# Binaries: every combination, one piece each
# ----------------------------------------------------------------------------


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


def test_each_combination_of_the_binaries_is_its_own_piece():
    """Off leaves the ``a = 0`` segment; on cuts the strip below minimum load
    off the pentagon. Neither is the hull, which is what ``free`` gives."""
    region = lps.project(committed(), CORNER_SOURCES, x='a', y='b', at={'t': 0}, binaries='each')
    assert [piece.label for piece in region.pieces] == ['on[t=0]=0', 'on[t=0]=1'], (
        'one piece per combination, in the order the combinations are counted, spelled as a row is'
    )
    off, on = region.pieces
    assert off.fixed == {'on[t=0]': 0}, 'the pinned column and its value, as data'
    assert [tuple(r) for r in off.vertices.rows()] == [(0, 0), (0, 4)], 'off: a is zero, b is free'
    assert [tuple(r) for r in on.vertices.rows()] == [(1, 0), (4, 0), (4, 2), (2, 4), (1, 4)], (
        'on: the pentagon without the strip below the minimum load'
    )
    assert vertices(region) == [(0, 0), (4, 0), (4, 2), (2, 4), (0, 4)], (
        'the region itself is the hull of the pieces, which hides the strip neither piece covers'
    )


def test_free_binaries_give_the_hull_the_pieces_fill():
    free = lps.project(committed(), CORNER_SOURCES, x='a', y='b', at={'t': 0})
    each = lps.project(committed(), CORNER_SOURCES, x='a', y='b', at={'t': 0}, binaries='each')
    assert free.pieces == (), 'free traces one polygon and no pieces'
    assert vertices(free) == vertices(each), 'and that polygon is the hull the pieces span'


def test_at_narrows_which_binary_columns_are_pinned():
    """Without ``at`` both hours' states are pinned: four combinations, the
    quantities summed over the horizon."""
    region = lps.project(committed(), CORNER_SOURCES, x='a', y='b', binaries='each')
    assert [piece.label for piece in region.pieces] == [
        'on[t=0]=0, on[t=1]=0',
        'on[t=0]=0, on[t=1]=1',
        'on[t=0]=1, on[t=1]=0',
        'on[t=0]=1, on[t=1]=1',
    ], 'every column of the binary, counted like a binary number'
    both_on = region.pieces[-1]
    assert [tuple(r) for r in both_on.vertices.rows()] == [(2, 0), (5, 0), (5, 6), (3, 8), (2, 8)], (
        'both on: each hour contributes its minimum load, so a starts at two'
    )


def test_an_infeasible_combination_is_left_out():
    spec = committed(**{'constraints.someone_on': {'foreach': [], 'expression': 'sum(on) >= 1'}})
    region = lps.project(spec, CORNER_SOURCES, x='a', y='b', binaries='each')
    assert [piece.label for piece in region.pieces] == [
        'on[t=0]=0, on[t=1]=1',
        'on[t=0]=1, on[t=1]=0',
        'on[t=0]=1, on[t=1]=1',
    ], 'both off breaks the rule that someone is on, and is not a piece'


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
    assert [piece.label for piece in region.pieces] == ['on[t=0]=0', 'on[t=0]=1'], 'the second hour has no state'


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


def test_pieces_are_drawn_each_under_its_label(axes):
    region = lps.project(committed(), CORNER_SOURCES, x='a', y='b', at={'t': 0}, binaries='each')
    region.plot(axes)
    assert [line.get_label() for line in axes.lines][:1] == ['on[t=0]=0'], 'the segment piece is a labelled line'
    assert [patch.get_label() for patch in axes.patches] == ['on[t=0]=1'], 'the polygon piece a labelled fill'
    from matplotlib.colors import to_rgb

    segment, polygon = to_rgb(axes.lines[0].get_color()), to_rgb(axes.patches[0].get_facecolor())
    assert (segment, polygon) == (to_rgb('C0'), to_rgb('C1')), (
        'one colour per piece off one cycle: the segment takes the first, the polygon the second, though a line '
        'and a fill would each have taken the first of their own cycles'
    )


def test_a_piece_is_a_region_that_draws_itself(axes):
    region = lps.project(committed(), CORNER_SOURCES, x='a', y='b', at={'t': 0}, binaries='each')
    on = region.pieces[1]
    assert isinstance(on, lps.Region) and on.pieces == (), 'a piece is a region with nothing under it'
    on.plot(axes)
    assert [patch.get_label() for patch in axes.patches] == ['on[t=0]=1'], 'drawn alone, it carries the same label'
    assert region.plot(axes, label='all').get_legend_handles_labels()[1].count('all') == 2, (
        'a label given to the whole reaches every piece'
    )


def test_the_long_form_has_one_row_per_vertex_of_every_piece():
    region = lps.project(committed(), CORNER_SOURCES, x='a', y='b', binaries='each')
    long = region.to_frame()
    assert long.columns == ['on[t=0]', 'on[t=1]', 'vertex', 'a', 'b'], (
        'the pinned columns, the vertex index in polygon order, then the two quantities'
    )
    assert long.height == sum(len(piece.vertices) for piece in region.pieces), 'every vertex of every piece, once'
    assert long.filter((pl.col('on[t=0]') == 1) & (pl.col('on[t=1]') == 1))['vertex'].to_list() == [0, 1, 2, 3, 4], (
        'a piece is its vertices counted from zero'
    )
    assert long.schema['on[t=0]'] == pl.Int64, 'a pinned value is an integer, not the float the solver holds'


def test_the_long_form_of_a_whole_region_is_its_vertices_indexed():
    region = lps.project(CORNER, CORNER_SOURCES, x='a', y='b', at={'t': 0})
    assert region.to_frame().columns == ['vertex', 'a', 'b'], 'nothing pinned, so no pinned columns'
    assert region.to_frame().drop('vertex').equals(region.vertices), 'and the rows are the polygon as it stands'
