"""The fold: one plan per slice, and the answers stitched back together.

What is checked here is that the driver is a *driver* — every claim below is
about slicing, coupling and folding, and none of it is about the language. The
windowed model in ``WINDOW_YAML`` uses only constructs that already ship, which
is the whole argument for building this above ``api.py`` rather than inside it.
"""

from __future__ import annotations

import datetime
import multiprocessing
import sys
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from unittest import mock

import polars as pl
import pytest

import lpspec as lps
from lpspec import strategy

# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

#: Dispatch, with a scenario-free declaration — the slice column never appears
#: in the model, which is what lets `EachCoordinate` need no language support.
DISPATCH = {
    'dimensions': {'snapshot': {'dtype': 'int'}, 'generator': {'dtype': 'str'}},
    'parameters': {
        'p_max': {'dims': ['generator']},
        'cost': {'dims': ['generator']},
        'load': {'dims': ['snapshot']},
    },
    'variables': {'p': {'foreach': ['snapshot', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}}},
    'constraints': {'balance': {'foreach': ['snapshot'], 'expression': 'sum(p, over=generator) == load'}},
    'objectives': {'total': {'sense': 'minimize', 'expression': 'sum(p * cost, over=generator)'}},
}

#: Storage over a *local* index, with the seam split out by a `where` on a dim
#: literal. `soc_step` carries no `edge=`, so its vacated row drops and the
#: masked `soc_open` supplies it from a carried parameter.
WINDOW = {
    'dimensions': {'t': {'dtype': 'int'}, 'generator': {'dtype': 'str'}},
    'parameters': {
        'p_max': {'dims': ['generator']},
        'cost': {'dims': ['generator']},
        'load': {'dims': ['t']},
        'soc_initial': {'dims': []},
    },
    'variables': {
        'p': {'foreach': ['t', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}},
        'charge': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 30}},
        'discharge': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 30}},
        'soc': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 100}},
    },
    'constraints': {
        'balance': {
            'foreach': ['t'],
            'expression': 'sum(p, over=generator) + discharge - charge == load',
        },
        'soc_open': {
            'foreach': ['t'],
            'where': 't == 0',
            'expression': 'soc == soc_initial + charge * 0.9 - discharge',
        },
        'soc_step': {
            'foreach': ['t'],
            'where': 't > 0',
            'expression': 'soc == shift(soc, over=t, by=1) + charge * 0.9 - discharge',
        },
    },
    'objectives': {'total': {'sense': 'minimize', 'expression': 'sum(p * cost, over=generator)'}},
}

#: The same storage, but two of them — so `soc` is over `(t, storage)` and the
#: carried `soc_initial` over `(storage)`. The carry drops `t` and `storage`
#: rides along, which is the general shape a scalar carry is a corner of.
#:
#: `charge` and `discharge` are capped well below a window's worth so a store
#: cannot empty itself before the seam; otherwise every window ends at zero and
#: carrying the state is indistinguishable from not carrying it.
MULTI_STORE = {
    'dimensions': {'t': {'dtype': 'int'}, 'generator': {'dtype': 'str'}, 'storage': {'dtype': 'str'}},
    'parameters': {
        'p_max': {'dims': ['generator']},
        'cost': {'dims': ['generator']},
        'load': {'dims': ['t']},
        'soc_initial': {'dims': ['storage']},
        'efficiency': {'dims': ['storage']},
    },
    'variables': {
        'p': {'foreach': ['t', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}},
        'charge': {'foreach': ['t', 'storage'], 'bounds': {'lower': 0, 'upper': 5}},
        'discharge': {'foreach': ['t', 'storage'], 'bounds': {'lower': 0, 'upper': 5}},
        'soc': {'foreach': ['t', 'storage'], 'bounds': {'lower': 0, 'upper': 100}},
    },
    'constraints': {
        'balance': {
            'foreach': ['t'],
            'expression': 'sum(p, over=generator) + sum(discharge, over=storage) - sum(charge, over=storage) == load',
        },
        'soc_open': {
            'foreach': ['t', 'storage'],
            'where': 't == 0',
            'expression': 'soc == soc_initial + charge * efficiency - discharge',
        },
        'soc_step': {
            'foreach': ['t', 'storage'],
            'where': 't > 0',
            'expression': 'soc == shift(soc, over=t, by=1) + charge * efficiency - discharge',
        },
    },
    'objectives': {'total': {'sense': 'minimize', 'expression': 'sum(p * cost, over=generator)'}},
}

#: A myopic pathway: what a period builds is what the next period already has.
#: `total` and `existing` are both over `(generator)`, so the carry drops
#: nothing and the whole vector moves — no index could have said this.
MYOPIC = {
    'dimensions': {'generator': {'dtype': 'str'}},
    'parameters': {
        'existing': {'dims': ['generator']},
        'cost': {'dims': ['generator']},
        'demand': {'dims': []},
    },
    'variables': {
        'build': {'foreach': ['generator'], 'bounds': {'lower': 0, 'upper': 50}},
        'total': {'foreach': ['generator'], 'bounds': {'lower': 0, 'upper': 200}},
    },
    'constraints': {
        'accumulate': {'foreach': ['generator'], 'expression': 'total == existing + build'},
        'meet': {'foreach': [], 'expression': 'sum(total, over=generator) >= demand'},
    },
    'objectives': {'total_cost': {'sense': 'minimize', 'expression': 'sum(build * cost, over=generator)'}},
}

GENERATORS = ['wind', 'gas']
STORES = ['battery', 'pumped']
STATIC = {
    'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [10.0, 100.0]}),
    'cost': pl.DataFrame({'generator': GENERATORS, 'value': [1.0, 50.0]}),
}


def scenario_sources() -> dict[str, object]:
    """Three scenarios differing only in load — `load` carries the slice key."""
    rows = []
    for scenario, scale in (('low', 1.0), ('mid', 2.0), ('high', 3.0)):
        rows += [{'scenario': scenario, 'snapshot': t, 'value': 5.0 * scale + t} for t in range(4)]
    return {**STATIC, 'load': pl.DataFrame(rows)}


def multi_store_sources(periods: int = 12) -> dict[str, object]:
    """Two stores, each with a real starting level worth handing across a seam."""
    return {
        **horizon_sources(periods),
        'soc_initial': pl.DataFrame({'storage': STORES, 'value': [40.0, 20.0]}),
        'efficiency': pl.DataFrame({'storage': STORES, 'value': [0.9, 0.75]}),
    }


def myopic_sources() -> dict[str, object]:
    """Three periods of rising demand — `demand` carries the slice key."""
    return {
        'cost': pl.DataFrame({'generator': GENERATORS, 'value': [1.0, 50.0]}),
        'existing': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 0.0]}),
        'demand': pl.DataFrame({'period': [1, 2, 3], 'value': [10.0, 25.0, 40.0]}),
    }


def horizon_sources(periods: int = 12) -> dict[str, object]:
    """A load profile of any length — the pattern repeats past twelve."""
    load = [5.0, 9.0, 30.0, 40.0, 6.0, 8.0, 35.0, 45.0, 7.0, 10.0, 25.0, 50.0]
    return {
        **STATIC,
        'load': pl.DataFrame({'snapshot': range(periods), 'value': [load[t % len(load)] for t in range(periods)]}),
        'soc_initial': pl.DataFrame({'value': [0.0]}),
    }


#: Window geometries whose *tail* differs — the only place a windowing rule
#: goes wrong. Between them these cover a final window of one, a final window
#: of ``step``, a horizon shorter than a single window, and a tail that divides
#: exactly so there is no short window at all.
GEOMETRIES = [
    pytest.param(periods, length, step, id=f'n{periods}-l{length}-s{step}')
    for periods in (1, 2, 5, 7, 12)
    for length in (1, 2, 3, 6)
    for step in range(1, length + 1)
]


# ---------------------------------------------------------------------------
# EachCoordinate — the independent case
# ---------------------------------------------------------------------------


def test_a_scenario_sweep_solves_each_slice_and_keys_the_answers():
    """The model never mentions `scenario`; the driver filters and drops it.

    That is the whole reason this needs no language change — a slice is the
    same declaration bound to a narrower source.
    """
    runs = lps.solve_over(DISPATCH, scenario_sources(), lps.EachCoordinate('scenario'))

    assert len(runs) == 3
    assert runs.keys == ['high', 'low', 'mid'], 'keys come back sorted, not in data order'
    assert runs.objective.columns == ['scenario', 'status', 'termination_condition', 'objective']
    assert set(runs.primal('p').columns) == {'scenario', 'snapshot', 'generator', 'value'}
    assert runs.primal('p').height == 3 * 4 * 2

    by_key = dict(zip(runs.objective['scenario'], runs.objective['objective'], strict=True))
    assert by_key['low'] < by_key['mid'] < by_key['high'], 'a bigger load is a costlier dispatch'


def test_each_slice_matches_solving_that_slice_alone():
    """The fold must not change the answer — the oracle is `solve` itself."""
    runs = lps.solve_over(DISPATCH, scenario_sources(), lps.EachCoordinate('scenario'))
    folded = dict(zip(runs.objective['scenario'], runs.objective['objective'], strict=True))

    for scenario, expected in folded.items():
        one = scenario_sources()
        one['load'] = one['load'].filter(pl.col('scenario') == scenario).drop('scenario')
        with lps.solve(DISPATCH, one) as result:
            assert result.objective == pytest.approx(expected)


def test_an_axis_naming_a_column_no_source_carries_says_so():
    with pytest.raises(lps.DataError, match="no source carries a 'draw' column"):
        lps.solve_over(DISPATCH, scenario_sources(), lps.EachCoordinate('draw'))


def test_a_name_the_sweep_does_not_hold_says_what_it_does_hold():
    """Everything a slice produced is kept, so a miss is a name, not a flag."""
    runs = lps.solve_over(DISPATCH, scenario_sources(), lps.EachCoordinate('scenario'))
    with pytest.raises(lps.LpspecError, match="no variable 'q' in this sweep"):
        runs.primal('q')
    with pytest.raises(lps.LpspecError, match="no constraint 'nope' in this sweep"):
        runs.dual('nope')


def test_a_sweep_that_solved_nothing_blames_the_solve():
    """An absent frame has one cause now, and the message says which.

    The load is pushed past total capacity, so every slice is infeasible.
    """
    sources = scenario_sources()
    sources['load'] = sources['load'].with_columns(pl.col('value') + 1_000)
    runs = lps.solve_over(DISPATCH, sources, lps.EachCoordinate('scenario'))

    assert len(runs) == 3, 'an unsolvable slice is still a row of the record'
    assert runs.objective['objective'].is_nan().all()
    with pytest.raises(lps.LpspecError, match='holds no variable frames at all') as raised:
        runs.primal('p')
    assert 'infeasible' in str(raised.value), 'the message names what the slices actually did'


# ---------------------------------------------------------------------------
# EachWindow — the coupled case
# ---------------------------------------------------------------------------


def test_a_rolling_horizon_carries_state_across_the_seam():
    """Three contiguous windows, the store's level handed forward.

    `soc_initial` is rebound per window from the previous window's last `soc`,
    which is the carry doing its one job — a copy, at a named index.
    """
    runs = lps.solve_over(
        WINDOW,
        horizon_sources(),
        lps.EachWindow('snapshot', length=4, step=4, into='t'),
        carry={'soc_initial': ('soc', 3)},
    )

    assert runs.keys == [0, 4, 8]
    assert runs.primal('p').height == 3 * 4 * 2
    assert set(runs.primal('soc').columns) == {'snapshot_start', 't', 'value'}, (
        'the rows are indexed by `t`, so the key column cannot be called `snapshot`'
    )
    assert runs.objective['objective'].to_list() == pytest.approx([2270.0, 2770.0, 2655.0])


def test_overlapping_windows_advance_by_step_and_look_ahead_by_length():
    runs = lps.solve_over(
        WINDOW,
        horizon_sources(),
        lps.EachWindow('snapshot', length=6, step=3, into='t'),
        carry={'soc_initial': ('soc', 2)},  # the last *kept* row, not the last row
    )
    assert runs.keys == [0, 3, 6, 9]
    assert runs.primal('soc').filter(pl.col('snapshot_start') == 9).height == 3, (
        'the tail window is short rather than padded: 9..11 is three rows, not six'
    )


def test_stitch_drops_the_overlap_and_restores_the_global_coordinate():
    """The answer a rolling horizon is for, without the caller doing arithmetic."""
    runs = lps.solve_over(
        WINDOW,
        horizon_sources(12),
        lps.EachWindow('snapshot', length=6, step=3, into='t'),
        carry={'soc_initial': ('soc', 2)},
    )
    stitched = runs.primal('soc', original_index=True)
    assert stitched.columns == ['snapshot', 'value'], 'the slice bookkeeping is gone'
    assert stitched['snapshot'].to_list() == list(range(12)), 'and every coordinate is present once'
    # every window kept exactly the `step` coordinates it owns, out of 21 solved
    assert runs.primal('soc').height == 21


@pytest.mark.parametrize(('periods', 'length', 'step'), GEOMETRIES)
def test_a_window_geometry_covers_every_coordinate_exactly_once(periods, length, step):
    """A stitched sweep reproduces the coordinate list, whatever the tail."""
    runs = lps.solve_over(
        WINDOW,
        horizon_sources(periods),
        lps.EachWindow('snapshot', length=length, step=step, into='t'),
    )
    assert runs.primal('soc', original_index=True)['snapshot'].to_list() == list(range(periods)), (
        'the original index must reproduce the coordinate list, whatever the tail'
    )
    assert runs.primal('soc')['snapshot_start'].n_unique() == len(range(0, periods, step)), 'one slice per window start'


@pytest.mark.parametrize(('periods', 'length', 'step'), GEOMETRIES)
def test_a_carry_at_step_minus_one_is_in_range_for_every_geometry(periods, length, step):
    """A non-final window always holds at least ``step`` coordinates.

    It can still be shorter than ``length`` — 10 coordinates at ``length=6,
    step=3`` gives a window at 6 holding four — but never shorter than
    ``step``, since a later window starting ``step`` on means that many were
    left. So the carry api.md recommends, the last row each window *keeps*,
    can never fall off the end of the window it reads from.
    """
    runs = lps.solve_over(
        WINDOW,
        horizon_sources(periods),
        lps.EachWindow('snapshot', length=length, step=step, into='t'),
        carry={'soc_initial': ('soc', step - 1)},
    )
    assert runs.primal('soc', original_index=True)['snapshot'].to_list() == list(range(periods)), (
        'a carry at step - 1 is in range for every geometry, so the sweep completes'
    )


def test_stitch_keeps_the_whole_of_the_final_short_window():
    """A tail window holds at most `step`, so the owning rule keeps all of it.

    12 coordinates at length 6 step 5 leaves a final window of two. Dropping
    `t >= step` uniformly would be right for it too; the risk is a rule that
    drops the tail because it is not a full window, and it must not.
    """
    runs = lps.solve_over(
        WINDOW,
        horizon_sources(12),
        lps.EachWindow('snapshot', length=6, step=5, into='t'),
    )
    assert runs.keys == [0, 5, 10], 'three windows, the last of two coordinates'
    assert runs.primal('soc', original_index=True)['snapshot'].to_list() == list(range(12)), (
        'the short tail window is kept whole, not dropped for being short'
    )


def test_stitching_an_axis_that_re_indexed_nothing_changes_nothing():
    """A caller handed an axis should not have to ask which kind it is."""
    runs = lps.solve_over(DISPATCH, scenario_sources(), lps.EachCoordinate('scenario'))
    assert runs.primal('p', original_index=True).equals(runs.primal('p')), (
        'an axis that re-indexed nothing has nothing to restore'
    )
    assert runs.dual('balance', original_index=True).equals(runs.dual('balance')), (
        'and that holds for duals too, since it is a property of the axis'
    )


def test_duals_stitch_the_same_way_primals_do():
    """A window's price at a coordinate is the owning window's, not a blend.

    The reason `stitch` is a flag on the readers rather than a reader of its
    own: what has to be undone is a property of the *axis*, so a dual needs no
    second implementation — and a name that is both a variable and a
    constraint, which the language permits, is never dispatched on.
    """
    runs = lps.solve_over(
        WINDOW,
        horizon_sources(12),
        lps.EachWindow('snapshot', length=6, step=3, into='t'),
        carry={'soc_initial': ('soc', 2)},
    )
    keyed, stitched = runs.dual('balance'), runs.dual('balance', original_index=True)
    assert keyed.columns == ['snapshot_start', 't', 'value']
    assert stitched.columns == ['snapshot', 'value']
    assert stitched['snapshot'].to_list() == list(range(12)), 'one price per coordinate'
    assert keyed.height > stitched.height, 'the overlap is priced twice before the index collapses it'


def test_keyed_is_the_default_because_stitching_drops_rows():
    """The default may not silently discard answers the sweep computed."""
    runs = lps.solve_over(
        WINDOW,
        horizon_sources(12),
        lps.EachWindow('snapshot', length=6, step=3, into='t'),
        carry={'soc_initial': ('soc', 2)},
    )
    assert runs.primal('soc').height == 21, 'keyed keeps every row every window solved'
    assert runs.primal('soc', original_index=True).height == 12, 'only the rows each window owns'
    # keyed by the same column as `objective`, so the two still join
    assert runs.objective.join(runs.primal('soc'), on=runs.key_name).height == 21


#: Six coordinates, three windows of two, whatever the coordinates *are*.
#:
#: `length` and `step` count coordinates rather than coordinate values, and
#: every row here is a case that measuring in values got wrong. Dense integers
#: from zero were the one shape that worked, because there value equals
#: position; spacing them by ten silently produced **26** mostly-empty slices,
#: and a datetime index raised `TypeError` from `int()`.
COORDINATE_TYPES = [
    pytest.param(list(range(6)), id='dense-ints'),
    pytest.param([0, 10, 20, 30, 40, 50], id='gapped-ints'),
    pytest.param(list(range(100, 106)), id='ints-not-from-zero'),
    pytest.param([datetime.datetime(2030, 1, 1, h) for h in range(6)], id='datetimes'),
    pytest.param([f's{i}' for i in range(6)], id='strings'),
]


@pytest.mark.parametrize('coordinates', COORDINATE_TYPES)
def test_a_window_spans_coordinates_whatever_they_are_numbered(coordinates):
    """The only requirement on a windowed dimension is that it is orderable.

    Not numeric, not dense, not starting anywhere in particular — and not time,
    which is only the common case. The local index is dense `0..n-1` by
    construction, which is also what keeps the seam's `where: "t == 0"`
    matching on a dimension with gaps in it.
    """
    sources = {
        **STATIC,
        'load': pl.DataFrame({'snapshot': coordinates, 'value': [5.0] * 6}),
        'soc_initial': pl.DataFrame({'value': [0.0]}),
    }
    runs = lps.solve_over(WINDOW, sources, lps.EachWindow('snapshot', length=2, step=2, into='t'))

    assert len(runs) == 3
    assert runs.keys == coordinates[::2], 'a window is keyed by its first coordinate'
    soc = runs.primal('soc')
    assert soc.height == 6
    assert sorted(soc['t'].unique().to_list()) == [0, 1], 'the local index is dense per window'


@pytest.mark.parametrize('coordinates', COORDINATE_TYPES)
def test_stitch_recovers_coordinates_no_arithmetic_could(coordinates):
    """`snapshot_start + t` is meaningless for a datetime or a string axis.

    The window→coordinate mapping is the axis's to keep, and stitching is the
    only way back to it: nothing the caller holds could reconstruct these.
    """
    runs = lps.solve_over(
        WINDOW,
        {
            **STATIC,
            'load': pl.DataFrame({'snapshot': coordinates, 'value': [10.0] * 6}),
            'soc_initial': pl.DataFrame({'value': [0.0]}),
        },
        lps.EachWindow('snapshot', length=2, step=2, into='t'),
    )
    assert runs.primal('soc', original_index=True)['snapshot'].to_list() == coordinates


def test_a_window_key_column_never_shadows_the_dimension_it_replaced():
    """`snapshot_start` holds window starts, and there are no snapshots left.

    `EachWindow` drops the global dimension and re-indexes to `into`, so a key
    column called `snapshot` would be window starts sitting under the name of
    the coordinate they are *not* — one that joins cleanly against real
    snapshot-indexed data and silently keeps a twelfth of it.
    """
    runs = lps.solve_over(
        WINDOW,
        horizon_sources(),
        lps.EachWindow('snapshot', length=4, step=4, into='t'),
    )
    soc = runs.primal('soc')
    assert 'snapshot' not in soc.columns
    assert soc.columns[0] == 'snapshot_start'
    assert runs.objective.columns[0] == 'snapshot_start', 'both frames key the same way'
    assert sorted(soc['snapshot_start'].unique().to_list()) == [0, 4, 8]

    sweep = lps.solve_over(DISPATCH, scenario_sources(), lps.EachCoordinate('scenario'))
    assert sweep.objective.columns[0] == 'scenario', (
        'EachCoordinate keeps the plain name: there the key really is a coordinate of it'
    )


def test_the_window_geometry_is_checked_at_construction():
    """`__post_init__` is what earns these two a name on the public surface."""
    with pytest.raises(ValueError, match='exceeds length'):
        lps.EachWindow('snapshot', length=4, step=8, into='t')
    with pytest.raises(ValueError, match='must be positive'):
        lps.EachWindow('snapshot', length=0, step=1, into='t')
    with pytest.raises(ValueError, match='must differ from dim'):
        lps.EachWindow('snapshot', length=4, step=4, into='snapshot')
    with pytest.raises(ValueError, match='no default'):
        lps.EachWindow('snapshot', length=4, step=4, into='')


def test_a_carry_index_outside_the_slice_is_refused_by_name():
    with pytest.raises(lps.LpspecError, match='out of range'):
        lps.solve_over(
            WINDOW,
            horizon_sources(),
            lps.EachWindow('snapshot', length=4, step=4, into='t'),
            carry={'soc_initial': ('soc', 99)},
        )


def test_a_short_tail_window_does_not_have_to_hold_the_carry_index():
    """Nothing reads the last slice's carry, so it is never computed.

    12 coordinates at length 6 step 5 leaves a final window of two, which
    cannot answer `t == 4`. Computing a value no later slice will read would
    fail a sweep that had already solved every window.
    """
    runs = lps.solve_over(
        WINDOW,
        horizon_sources(12),
        lps.EachWindow('snapshot', length=6, step=5, into='t'),
        carry={'soc_initial': ('soc', 4)},
    )
    assert runs.keys == [0, 5, 10]
    assert runs.objective['termination_condition'].to_list() == ['optimal'] * 3
    assert runs.primal('soc').filter(pl.col('snapshot_start') == 10).height == 2


def test_a_carry_collapses_one_dimension_and_every_other_rides_along():
    """`soc` is over `(t, storage)` and `soc_initial` over `(storage)`.

    The two declarations say what is copied: `t` is what the parameter lacks,
    so `t` is what the index names, and `storage` passes through — both stores
    are handed forward, each its own level. That is the general case; a scalar
    `soc_initial` is only the one where nothing is left to ride.
    """
    window = lps.EachWindow('snapshot', length=4, step=4, into='t')
    runs = lps.solve_over(
        MULTI_STORE,
        multi_store_sources(),
        window,
        carry={'soc_initial': ('soc', 3)},
    )

    assert runs.keys == [0, 4, 8]
    assert set(runs.primal('soc').columns) == {'snapshot_start', 't', 'storage', 'value'}

    def at(name: str, start: int, t: int, store: str) -> float:
        rows = runs.primal(name).filter(
            (pl.col('snapshot_start') == start) & (pl.col('t') == t) & (pl.col('storage') == store)
        )
        return rows['value'].item()

    efficiency = dict(zip(STORES, [0.9, 0.75], strict=True))
    for previous, start in ((0, 4), (4, 8)):
        for store in STORES:
            opened = at('soc', start, 0, store)
            expected = (
                at('soc', previous, 3, store)
                + at('charge', start, 0, store) * efficiency[store]
                - at('discharge', start, 0, store)
            )
            assert opened == pytest.approx(expected, abs=1e-6), (
                'the opening row is the previous window at t == 3, for this same store'
            )

    fresh = lps.solve_over(MULTI_STORE, multi_store_sources(), window)
    assert not fresh.primal('soc').equals(runs.primal('soc')), 'the carry changed nothing'


def test_a_myopic_pathway_carries_a_whole_vector_with_no_index():
    """Capacity per generator, handed forward as a frame rather than a number.

    `total` and `existing` are both over `(generator)`, so nothing is dropped
    and there is no coordinate to name — the frame *is* the carry. This is the
    shape that a row index could never express, and the reason the index is
    read off the two declarations rather than off the frame.
    """
    runs = lps.solve_over(
        MYOPIC,
        myopic_sources(),
        lps.EachCoordinate('period', ordered=True),
        carry={'existing': ('total', None)},
    )

    assert runs.keys == [1, 2, 3]
    built = runs.primal('build').filter(pl.col('generator') == 'wind').sort('period')['value'].to_list()
    total = runs.primal('total').filter(pl.col('generator') == 'wind').sort('period')['value'].to_list()
    assert built == pytest.approx([10.0, 15.0, 15.0]), (
        'each period builds only the increment: what the last one built came back as `existing`'
    )
    assert total == pytest.approx([10.0, 25.0, 40.0]), 'demand 10 -> 25 -> 40 is met exactly'


#: The five ways a carry cannot line up. Each `id` is the case, so a failure
#: names it rather than a line number: `-k collapses-two-dimensions`.
_WINDOW_AXIS = lps.EachWindow('snapshot', length=4, step=4, into='t')
_PERIOD_AXIS = lps.EachCoordinate('period', ordered=True)
UNSOUND_CARRIES = [
    pytest.param(
        WINDOW, horizon_sources, _WINDOW_AXIS, {'soc_initial': ('p', 3)},
        r'would collapse .*at once', "['t', 'generator']",
        id='collapses-two-dimensions-where-an-index-names-one',
    ),
    pytest.param(
        WINDOW, horizon_sources, _WINDOW_AXIS, {'soc_initial': ('soc', None)},
        r"drops 't' and so needs an index", None,
        id='drops-a-dimension-without-naming-a-coordinate',
    ),
    pytest.param(
        MYOPIC, myopic_sources, _PERIOD_AXIS, {'existing': ('total', 0)},
        'has nothing to index', None,
        id='indexes-two-sides-that-already-line-up',
    ),
    pytest.param(
        WINDOW, horizon_sources, _WINDOW_AXIS, {'p_max': ('soc', 3)},
        'cannot line up', None,
        id='parameter-over-more-than-the-variable',
    ),
    pytest.param(
        WINDOW, horizon_sources, _WINDOW_AXIS, {'soc_initial': ('nope', 3)},
        'does not declare', None,
        id='a-name-neither-side-declares',
    ),
]  # fmt: skip


@pytest.mark.parametrize(('model', 'sources', 'axis', 'carry', 'expected', 'names'), UNSOUND_CARRIES)
def test_a_carry_that_cannot_line_up_says_so_before_anything_solves(model, sources, axis, carry, expected, names):
    """Every one of these is answerable from the two declarations alone."""
    with pytest.raises(lps.LpspecError, match=expected) as raised:
        lps.solve_over(model, sources(), axis, carry=carry)
    if names is not None:
        assert names in str(raised.value), 'the message names the dimensions it could not choose between'


def test_a_carry_is_refused_before_a_single_source_is_read(tmp_path):
    """ "Early" has to mean before the data, not merely before the solve.

    Every question a carry raises is answered by the two declarations, so
    answering it after the axis has scanned every parquet file to find its
    coordinates makes a typo cost a pass over the whole dataset. The unreadable
    path is the assertion: reaching it at all means the check ran too late.
    """
    missing = tmp_path / 'not-written-yet.parquet'
    sources = {**horizon_sources(), 'load': str(missing)}

    with pytest.raises(lps.LpspecError, match='does not declare'):
        lps.solve_over(
            WINDOW,
            sources,
            lps.EachWindow('snapshot', length=4, step=4, into='t'),
            carry={'soc_initial': ('nope', 3)},
        )

    with pytest.raises(Exception, match='not-written-yet') as raised:
        lps.solve_over(
            WINDOW,
            sources,
            lps.EachWindow('snapshot', length=4, step=4, into='t'),
            carry={'soc_initial': ('soc', 3)},
        )
    assert not isinstance(raised.value, lps.LpspecError), 'the file, not the carry, is what failed'


# ---------------------------------------------------------------------------
# the fold's own rules
# ---------------------------------------------------------------------------


def test_carry_and_executor_are_refused_together():
    """Sequential by definition, so the combination is a call-time error rather
    than something discovered at slice two."""
    with pytest.raises(lps.LpspecError, match='mutually exclusive'):
        lps.solve_over(
            WINDOW,
            horizon_sources(),
            lps.EachWindow('snapshot', length=4, step=4, into='t'),
            carry={'soc_initial': ('soc', 3)},
            executor=object(),
        )


def test_carry_needs_an_ordered_axis():
    """Scenarios have no "next" slice for a value to move into."""
    with pytest.raises(lps.LpspecError, match='needs an ordered axis'):
        lps.solve_over(
            DISPATCH,
            scenario_sources(),
            lps.EachCoordinate('scenario'),
            carry={'p_max': ('p', 0)},
        )


class Inline:
    """The whole protocol `solve_over` needs, in nine lines.

    Not a toy: it is the claim that ``executor=`` takes
    :class:`concurrent.futures.Executor` and not `ProcessPoolExecutor`, which
    is what lets a dask ``Client`` or any other pool plug in **without this
    package shipping a transport**. If the driver ever reaches for something
    only a stdlib pool has, this is what stops compiling.
    """

    def submit(self, fn, /, *args, **kwargs):
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # a pool reports through the future, never raises
            future.set_exception(exc)
        return future


def _process_pool(method: str):
    return ProcessPoolExecutor(2, mp_context=multiprocessing.get_context(method))


#: Every executor shape the docs name, and the reason each is here.
#:
#: **`fork` is absent and that is the statement.** polars' thread pool does not
#: survive it, and a forked worker *hangs* rather than failing — so it cannot be
#: a parametrisation without wedging CI, which is exactly why the docs refuse
#: it. Measured: `fork` never returns where all four below do.
EXECUTORS = [
    pytest.param(Inline, id='inline-protocol'),
    pytest.param(lambda: ThreadPoolExecutor(2), id='threads'),
    pytest.param(lambda: _process_pool('spawn'), id='processes-spawn'),
    pytest.param(
        lambda: _process_pool('forkserver'),
        id='processes-forkserver',
        marks=pytest.mark.skipif(
            'forkserver' not in multiprocessing.get_all_start_methods(),
            reason='forkserver is not available on this platform',
        ),
    ),
]


@pytest.mark.parametrize('make_executor', EXECUTORS)
def test_every_executor_gives_the_same_answers_in_the_same_order(make_executor):
    """One fold, four pools, one answer — and the sequential run is the oracle.

    Same numbers *and* the same order. Futures complete out of order, so a
    sweep that read them by completion would reorder itself run to run, which
    is the kind of wrong that looks fine until two runs are diffed.
    """
    sources = scenario_sources()
    sequential = lps.solve_over(DISPATCH, sources, lps.EachCoordinate('scenario'))

    pool = make_executor()
    if hasattr(pool, '__enter__'):
        with pool as live:
            parallel = lps.solve_over(DISPATCH, sources, lps.EachCoordinate('scenario'), executor=live)
    else:
        parallel = lps.solve_over(DISPATCH, sources, lps.EachCoordinate('scenario'), executor=pool)

    assert parallel.keys == sequential.keys
    assert parallel.objective.equals(sequential.objective)
    assert parallel.primal('p').equals(sequential.primal('p'))


def test_a_thread_pool_does_not_encode_for_a_boundary_it_never_crosses():
    """In-process, so a parquet round trip would be paid for nothing.

    31% of a thread-pool sweep, measured, which is what earns the one type
    check in the driver. `ThreadPoolExecutor` is public stdlib, so this is a
    documented class rather than a reach into an executor's internals — and
    every other executor is assumed to cross, because none of them can be
    asked.
    """
    seen: list[str] = []
    original = strategy._encode

    def spy(sources, memo, **kwargs):
        seen.append('encoded')
        return original(sources, memo, **kwargs)

    strategy._encode = spy
    try:
        with ThreadPoolExecutor(2) as pool:
            lps.solve_over(DISPATCH, scenario_sources(), lps.EachCoordinate('scenario'), executor=pool)
        assert seen == [], 'a thread pool encoded its sources'

        with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
            lps.solve_over(DISPATCH, scenario_sources(), lps.EachCoordinate('scenario'), executor=pool)
        assert seen, 'a process pool did not encode its sources'
    finally:
        strategy._encode = original


def test_a_failing_slice_reports_the_real_error_across_a_process_boundary():
    """An exception has to survive pickling or the cause is lost.

    A custom ``__init__`` signature is the classic way this breaks, and it
    surfaces as an unrelated ``TypeError`` raised while *unpickling* — so the
    worker's real complaint never arrives. Pinned because a future improvement
    to an error message is exactly what would break it.
    """
    broken = {**scenario_sources()}
    broken.pop('cost')
    with (
        ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool,
        pytest.raises(lps.DataError, match="no data provided for parameter 'cost'"),
    ):
        lps.solve_over(DISPATCH, broken, lps.EachCoordinate('scenario'), executor=pool)


def test_a_parquet_path_slices_without_being_read_whole(tmp_path):
    """A path source is scanned, so the per-slice filter pushes into the file."""
    sources = scenario_sources()
    path = tmp_path / 'load.parquet'
    frame = sources.pop('load')
    assert isinstance(frame, pl.DataFrame)
    frame.write_parquet(path)

    runs = lps.solve_over(DISPATCH, {**sources, 'load': str(path)}, lps.EachCoordinate('scenario'))
    assert runs.keys == ['high', 'low', 'mid']
    assert runs.primal('p').height == 3 * 4 * 2


def test_a_path_stays_a_path_for_a_local_pool_and_travels_as_bytes_for_a_remote_one(tmp_path):
    """`workers_share_fs` is inferred from the pool, and only paths are affected.

    A `ProcessPoolExecutor`'s workers are this machine's, so slurping the file
    into the message would be reading and shipping it once per slice for
    nothing. An executor this package did not ship could be anywhere, so its
    paths travel as their own bytes — which is what a caller building a remote
    transport depends on, and the reason the flag survives with no transport in
    the box. `workers_share_fs=` says it outright when the guess is wrong.
    """
    sources = scenario_sources()
    path = tmp_path / 'p_max.parquet'
    frame = sources.pop('p_max')
    assert isinstance(frame, pl.DataFrame)
    frame.write_parquet(path)
    sources['p_max'] = str(path)

    crossed: list[object] = []
    original = strategy._encode

    def spy(sliced, memo, **kwargs):
        """Record what `p_max` crossed as; the same helper also encodes answers back."""
        encoded = original(sliced, memo, **kwargs)
        if 'p_max' in encoded:
            crossed.append(encoded['p_max'])
        return encoded

    strategy._encode = spy
    try:
        with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
            local = lps.solve_over(DISPATCH, sources, lps.EachCoordinate('scenario'), executor=pool)
            assert all(v == str(path) for v in crossed), 'a local pool shipped a file it could have opened'

            crossed.clear()
            remote = lps.solve_over(
                DISPATCH, sources, lps.EachCoordinate('scenario'), executor=pool, workers_share_fs=False
            )
            assert all(v == path.read_bytes() for v in crossed), 'the file did not travel as its own bytes'
    finally:
        strategy._encode = original

    assert remote.objective.equals(local.objective), 'the path and the bytes are the same numbers'
    assert remote.primal('p').equals(local.primal('p'))

    crossed.clear()
    strategy._encode = spy
    try:
        lps.solve_over(DISPATCH, sources, lps.EachCoordinate('scenario'), executor=Inline())
    finally:
        strategy._encode = original
    assert all(v == path.read_bytes() for v in crossed), 'an executor we did not ship was assumed local'


# ---------------------------------------------------------------------------
# reading a sweep back — Result's readers, one dimension wider
# ---------------------------------------------------------------------------


def test_the_readers_mirror_result_with_the_slice_key_as_one_more_dimension():
    """A sweep is where a labelled array earns its keep.

    `(scenario, snapshot, generator)` is the shape the caller wants — `.sel` a
    scenario, take a spread across them — and assembling it out of a
    slice-keyed frame by hand is the part worth not writing twice. Every
    reader here is `Result`'s under the same name, so knowing one is knowing
    both.
    """
    pytest.importorskip('xarray')
    runs = lps.solve_over(DISPATCH, scenario_sources(), lps.EachCoordinate('scenario'))

    pandas_frame = runs.to_pandas('p')
    assert list(pandas_frame.columns) == ['scenario', 'snapshot', 'generator', 'value']
    assert len(pandas_frame) == 3 * 4 * 2

    array = runs.to_dataarray('p')
    assert array.name == 'p'
    assert array.dims == ('scenario', 'snapshot', 'generator')
    assert array.shape == (3, 4, 2)
    assert array.sel(scenario='low', generator='wind').shape == (4,), (
        'the slice key is an ordinary coordinate, which is the whole point'
    )

    dataset = runs.to_dataset()
    assert set(dataset.data_vars) == {'p'}
    assert dataset['p'].dims == ('scenario', 'snapshot', 'generator')


def test_to_parquet_writes_one_file_per_kept_variable(tmp_path):
    """The bridge out for a sweep too wide to want in one array."""
    runs = lps.solve_over(DISPATCH, scenario_sources(), lps.EachCoordinate('scenario'))

    written = runs.to_parquet(tmp_path / 'sweep')
    assert set(written) == {'p'}
    assert pl.read_parquet(written['p']).equals(runs.primal('p'))


def test_a_reader_for_a_name_the_sweep_lacks_fails_the_way_primal_does():
    """One explanation, reached through every reader."""
    runs = lps.solve_over(DISPATCH, scenario_sources(), lps.EachCoordinate('scenario'))
    for read in (runs.to_pandas, runs.to_dataarray):
        with pytest.raises(lps.LpspecError, match="no variable 'q' in this sweep"):
            read('q')


def test_a_hand_built_axis_needs_no_class_but_must_name_its_own_key():
    """`axis` also takes a plain list of `(key, sources, coords)`, so an
    irregular ladder needs no third constructor on the public surface.

    What it cannot do is say what its keys are coordinates *of*, so `key=` is
    required there — the same argument that leaves `EachWindow.into` without a
    default. A column called `slice` would be this library naming somebody
    else's draw.
    """
    base = scenario_sources()
    cuts = [
        (name, {**base, 'load': base['load'].filter(pl.col('scenario') == name).drop('scenario')}, {})
        for name in ('low', 'high')
    ]

    with pytest.raises(lps.LpspecError, match='hand-built axis needs key_name='):
        lps.solve_over(DISPATCH, base, cuts)

    runs = lps.solve_over(DISPATCH, base, cuts, key_name='draw')
    assert runs.keys == ['low', 'high']
    assert runs.meta.columns[0] == 'draw'
    assert runs.primal('p').columns[0] == 'draw', 'both frames key the same way, or they stop joining'


def test_key_overrides_what_an_axis_derived_and_refuses_a_collision():
    """The derived name is right by default and the caller's word wins.

    The refusal is the narrow one: a key that is already a dimension of a kept
    variable would collide with a column those frames carry, which polars
    reports as a duplicate with no idea why. Naming a *dropped* dimension is
    not refused — that is the caller saying it deliberately, which is a
    different thing from the library doing it silently.
    """
    runs = lps.solve_over(DISPATCH, scenario_sources(), lps.EachCoordinate('scenario'), key_name='case')
    assert runs.objective.columns[0] == 'case'
    assert set(runs.primal('p').columns) == {'case', 'snapshot', 'generator', 'value'}

    with pytest.raises(lps.LpspecError, match=r"key_name='generator' is already a dimension of \['p'\]"):
        lps.solve_over(DISPATCH, scenario_sources(), lps.EachCoordinate('scenario'), key_name='generator')


def test_duals_come_back_keyed_by_slice_and_are_never_combined():
    """A shadow price belongs to the slice that priced it.

    The refusal `Runs` used to carry was against *aggregating* duals, which is
    a different thing from not having them: a price curve concatenated across
    windows is wrong in a way nothing complains about, but so is one summed
    across scenarios, and `primal` has never been asked to guess either. Keyed
    rows say whose each price is and leave the reduction to the caller.
    """
    runs = lps.solve_over(DISPATCH, scenario_sources(), lps.EachCoordinate('scenario'))

    prices = runs.dual('balance')
    assert prices.columns[0] == runs.key_name, 'the key comes first, as it does for a primal'
    assert set(prices[runs.key_name].unique()) == set(runs.keys)
    assert prices.height == runs.primal('p').height // 2, 'one price per row, not per column'


def test_a_slice_without_duals_does_not_fail_the_sweep():
    """An integer variable leaves duals undefined, and that is one slice's news.

    `Result.dual` raises for such a model — correct there, fatal here. The
    sweep must still return, and when the caller does ask for a price it must
    say what a single solve says: which variable is not continuous, and what to
    do about it. A sweep of one model has one answer, so the first is carried
    rather than rewritten.
    """
    integral = {
        **DISPATCH,
        'variables': {**DISPATCH['variables'], 'p': {**DISPATCH['variables']['p'], 'integer': True}},
    }
    runs = lps.solve_over(integral, scenario_sources(), lps.EachCoordinate('scenario'))

    assert len(runs) == 3, 'every slice is still a row of the record'
    assert runs.primal('p').height > 0, 'primals are unaffected'
    with pytest.raises(lps.LpspecError, match='duals are undefined for a mixed-integer model') as raised:
        runs.dual('balance')
    assert "'p' is not continuous" in str(raised.value), 'the sweep names the variable, as one solve does'


def test_a_bad_name_is_reported_without_the_optional_dependency():
    """`to_pandas` answers about the model before it asks about the environment.

    The bare-install job carries no pandas, and importing it first turned "this
    sweep never held 'q'" into "no module named pandas" — a true statement about
    something the caller did not ask about. Resolving the name first is what
    makes the reader's message the same on every install — while a name the
    sweep does hold still needs the dependency.
    """
    runs = lps.solve_over(DISPATCH, scenario_sources(), lps.EachCoordinate('scenario'))
    with mock.patch.dict(sys.modules, {'pandas': None}):
        with pytest.raises(lps.LpspecError, match="no variable 'q' in this sweep"):
            runs.to_pandas('q')
        with pytest.raises(ImportError):
            runs.to_pandas('p')
