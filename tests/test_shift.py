"""shift: time-coupled recurrences through both backends.

examples/storage.yaml is dispatch plus a cyclic battery:
soc == shift(soc, over=snapshot, offset=1, edge='wrap") + charge * 0.9 - discharge. The eager backend
The eager backend implements `edge='wrap'` with linopy"s circular .roll(); the
relational backend lowers it to plan.Translate — a pointwise ord-join remap.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import lpspec as lps
from lpspec.errors import LanguageError
from lpspec.lowering import _Lowering
from lpspec.relational.plan import (
    Translate,
    Variable,
)
from tests.conftest import DISPATCH_MODEL, EXAMPLES_DIR, by_coord, override, relation, resolved, schema_of
from tests.differential import differential
from tests.oracle import pd

STORAGE_YAML = EXAMPLES_DIR / 'storage.yaml'
STORAGE_SCHEMA = schema_of(STORAGE_YAML)


@pytest.fixture
def storage_inputs():
    """Peaky load that exceeds generation capacity at the peaks, so the
    battery is *required* (not just economic) and soc is genuinely coupled."""
    n_s = 48
    p_max = pd.Series({'wind': 80.0, 'gas': 70.0})
    cost = pd.Series({'wind': 1.0, 'gas': 40.0})
    t = np.arange(n_s)
    load = pd.Series(
        (110 + 60 * np.sin(2 * np.pi * t / 24)).round(3),  # peaks above the fleet's 150
        index=pd.RangeIndex(n_s, name='snapshot'),
    )
    return {
        'p_max': p_max,
        'cost': cost,
        'load': load,
        'snapshot': pd.RangeIndex(n_s, name='snapshot'),
        'generator': pd.Index(p_max.index, name='generator'),
    }


def _soc_trace(result):
    """(soc, prev-contribution inputs) as plain arrays, sorted by snapshot."""
    return tuple(
        result.to_pandas(name).set_index('snapshot')['value'].sort_index().to_numpy()
        for name in ('soc', 'charge', 'discharge')
    )


# ---------------------------------------------------------------------------
# the recurrence, end to end
# ---------------------------------------------------------------------------


def test_a_wrapping_edge_is_cyclic_on_both_lanes(storage_inputs):
    """`edge='wrap'` closes the recurrence, so `soc[0]` reads the last slot."""
    data = storage_inputs

    with differential(STORAGE_YAML, data, lp=True) as run:
        assert float(run.model.solution['discharge'].max()) > 1e-3, (
            'the battery must actually cycle for the model to be feasible'
        )

        soc, charge, discharge = _soc_trace(run.result)
        assert np.allclose(soc, np.roll(soc, 1) + 0.9 * charge - discharge, atol=1e-6)


def test_shift_drops_the_row_it_has_no_predecessor_for_on_both_lanes(storage_inputs):
    """shift() = acyclic recurrence, and the first snapshot has *no* recurrence.

    ``soc[0]`` has no predecessor, so under #289 the vacated slot is absent, it
    propagates through the equation, and the ``t=0`` row is not built at all —
    linopy v1's own reading of ``.shift()``. It used to start from zero, which
    was a constraint the model never wrote: an initial condition invented by
    the language on the modeller's behalf.

    A model that wants one now says so, which is what the declaration rules'
    storage example
    already did with a complementary ``where``. Both lanes are asserted because
    they reach the drop differently — the eager lane from linopy's absence
    propagation, the relational one from the vacated coordinates leaving the
    presence set.
    """
    data = storage_inputs
    data = {**data, 'load': (data['load'] * 0.93).round(3)}

    original = STORAGE_YAML.read_text()
    assert "shift(soc, over=snapshot, offset=1, edge='wrap')" in original
    acyclic = original.replace(
        "shift(soc, over=snapshot, offset=1, edge='wrap')", 'shift(soc, over=snapshot, offset=1)'
    )

    with differential(acyclic, data) as run:
        soc, charge, discharge = _soc_trace(run.result)
        assert np.allclose(soc[1:], soc[:-1] + 0.9 * charge[1:] - discharge[1:], atol=1e-6), (
            'the recurrence holds from the second snapshot on'
        )
        assert run.model.constraints['soc_balance'].labels.values[0] == -1, (
            't=0 is governed by its own bounds alone, so no row is built for it'
        )


def test_a_forward_shift_drops_the_row_at_the_far_end_on_both_lanes(storage_inputs):
    """`by=-1` is the mirror of the test above: the *last* snapshot has no successor.

    Both directions are one operator and the sign is data, so nothing about the
    drop should be special to reaching backwards — but nothing built a forward
    shift until #837, where the typesetter turned out to abort on one. The
    engine had always been right; that is the half no test said.
    """
    data = storage_inputs
    data = {**data, 'load': (data['load'] * 0.93).round(3)}

    original = STORAGE_YAML.read_text()
    assert "shift(soc, over=snapshot, offset=1, edge='wrap')" in original
    forward = original.replace(
        "shift(soc, over=snapshot, offset=1, edge='wrap')", 'shift(soc, over=snapshot, offset=-1)'
    )

    with differential(forward, data) as run:
        soc, charge, discharge = _soc_trace(run.result)
        assert np.allclose(soc[:-1], soc[1:] + 0.9 * charge[:-1] - discharge[:-1], atol=1e-6), (
            'the recurrence reads forwards, and holds up to the second-to-last snapshot'
        )
        assert run.model.constraints['soc_balance'].labels.values[-1] == -1, (
            'the last snapshot has no successor, so no row is built for it'
        )


def test_a_forward_shift_with_a_zero_edge_keeps_the_far_row_on_both_lanes(storage_inputs):
    """`edge=0` fills what `by=-1` vacates, so the row at the far end survives.

    The distinction the two spellings carry is the whole of law 8 in this
    position, and it is the one #830 found the math could not state: both
    printed as `t - 1`. Here it is asserted as rows rather than as notation —
    the last snapshot keeps its equation, with the successor term contributing
    nothing.
    """
    data = storage_inputs
    data = {**data, 'load': (data['load'] * 0.93).round(3)}

    original = STORAGE_YAML.read_text()
    filled = original.replace(
        "shift(soc, over=snapshot, offset=1, edge='wrap')", 'shift(soc, over=snapshot, offset=-1, edge=0)'
    )

    with differential(filled, data) as run:
        soc, charge, discharge = _soc_trace(run.result)
        assert run.model.constraints['soc_balance'].labels.values[-1] != -1, (
            'edge=0 asks for a value at the boundary, so the last row is built rather than dropped'
        )
        assert np.allclose(soc[-1], 0.9 * charge[-1] - discharge[-1], atol=1e-6), (
            'at the last snapshot the vacated successor contributes zero, not a wraparound'
        )


#: A mask that removes one interior coordinate, so the operand's own absence
#: sits where no edge is. `edge: 0` may fill the boundary and nothing else, and
#: the two are one call to `fillna` apart on the eager lane (#987).
MASKED_INTERIOR = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'usable': {'dims': ['t']}},
    'variables': {
        'level': {'foreach': ['t'], 'where': 'usable > 0', 'bounds': {'lower': 0, 'upper': 10}},
        'take': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {'link': {'foreach': ['t'], 'expression': 'take <= shift(level, over=t, offset=1, edge=0)'}},
    'objective': {
        'sense': 'maximize',
        'expression': 'sum(take, over=t) - 1000 * sum(level, over=t)',
    },
}


def test_a_zero_edge_fills_the_boundary_and_not_an_absence_that_was_already_there():
    """`edge:` is the opt-out for the slot the shift vacated, not for the operand.

    Three snapshots and `level` masked away at the middle one. The row at `t=0`
    is the vacated edge and is filled; the row at `t=2` reads the masked slot,
    which is absent for a reason the shift had nothing to do with, so it drops
    and `take[2]` is held by its own bound alone. Filling either alone is
    indistinguishable from filling both by row count, which is why the
    objective is asserted too.
    """
    sources = {
        't': pd.Index([0, 1, 2], name='t'),
        'usable': pd.Series([1.0, 0.0, 1.0], index=pd.Index([0, 1, 2], name='t')),
    }
    with differential(MASKED_INTERIOR, sources, lp=True) as run:
        assert run.engine.diagnostics().rows == 2, 'the boundary row and the one that reads a live slot, and no other'
        assert run.oracle == pytest.approx(10.0), (
            'take[2] reaches its bound because no row caps it — 0.0 would mean the masked slot was read as a zero'
        )


#: The same question asked of the two *gathers* — an offset that differs per
#: entity, and a shift closed inside a group. Both reach the edge through their
#: own out-of-range mask rather than through `.shift()`, so each needs its own
#: case or one of the three could fill the whole operand unnoticed.
BY_PARAMETER = {
    'dimensions': {'g': {'dtype': 'str'}, 't': {'dtype': 'int'}},
    'parameters': {'lead': {'dims': ['g'], 'dtype': 'int'}, 'usable': {'dims': ['t']}},
    'variables': {
        'level': {'foreach': ['g', 't'], 'where': 'usable > 0', 'bounds': {'lower': 0, 'upper': 10}},
        'take': {'foreach': ['g', 't'], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {'link': {'foreach': ['g', 't'], 'expression': 'take <= shift(level, over=t, offset=lead, edge=0)'}},
    'objective': {
        'sense': 'maximize',
        'expression': 'sum(sum(take, over=g), over=t) - 1000 * sum(sum(level, over=g), over=t)',
    },
}

IN_GROUPS = {
    'dimensions': {'t': {'dtype': 'int'}, 'season': {'dtype': 'str'}},
    'lookups': {'season_of': {'over': 't', 'into': 'season'}},
    'parameters': {'usable': {'dims': ['t']}},
    'variables': {
        'level': {'foreach': ['t'], 'where': 'usable > 0', 'bounds': {'lower': 0, 'upper': 10}},
        'take': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {
        'link': {'foreach': ['t'], 'expression': 'take <= shift(level, over=t, offset=1, edge=0, by=season_of)'}
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(take, over=t) - 1000 * sum(level, over=t)'},
}

#: The same, with nothing masked at all: `level` carries no `where`, so the
#: operand reaches the shift with no presence frame of its own. Which of the
#: two group-less readings the lane takes used to depend on that (#1061).
IN_GROUPS_UNMASKED = {
    'dimensions': {'t': {'dtype': 'int'}, 'season': {'dtype': 'str'}},
    'lookups': {'season_of': {'over': 't', 'into': 'season'}},
    'variables': {
        'level': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 10}},
        'take': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {
        'link': {'foreach': ['t'], 'expression': 'take <= shift(level, over=t, offset=1, edge=0, by=season_of)'}
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(take, over=t) - 1000 * sum(level, over=t)'},
}

#: A fourth snapshot the lookup sends nowhere, for both models above.
GROUPLESS_SOURCES = {
    't': pd.DataFrame({'t': [0, 1, 2, 3]}),
    'season_of': relation('t', 'season', [0, 1, 2, 3], ['s1', 's1', 's2', None]),
    'season': pd.Index(['s1', 's2'], name='season'),
}


def test_a_per_entity_offset_fills_its_own_edge_and_not_the_mask_under_it():
    """The gather's edge, asked the same way as the scalar shift's.

    One technology, a lead of one month, and `level` masked away at the middle
    month: `t=0` is the vacated edge and is filled, `t=2` reads the masked slot
    and drops, so two rows are built and `take` at the last month is capped by
    nothing.

    Where a *named* offset stops being a detail: which coordinates are vacated
    is per entity, so the edge cannot be one column of labels (#1049).
    """
    sources = {
        'g': pd.Index(['a'], name='g'),
        't': pd.Index([0, 1, 2], name='t'),
        'lead': pd.Series([1], index=pd.Index(['a'], name='g')),
        'usable': pd.Series([1.0, 0.0, 1.0], index=pd.Index([0, 1, 2], name='t')),
    }
    with differential(BY_PARAMETER, sources, lp=True) as run:
        assert run.engine.diagnostics().rows == 2, 'the vacated month and the one reading a live slot'
        assert run.oracle == pytest.approx(10.0), 'take at the last month is capped by nothing'


def test_a_grouped_shift_fills_each_groups_edge_and_not_the_mask_under_it():
    """And the third path: the edge is per group, the mask is not.

    Two seasons of two snapshots, `level` masked away at the first. Each
    season's opening snapshot is vacated and filled; the snapshot that reads
    the masked one drops.
    """
    sources = {
        't': pd.DataFrame({'t': [0, 1, 2, 3]}),
        'season_of': relation('t', 'season', [0, 1, 2, 3], ['s1', 's1', 's2', 's2']),
        'season': pd.Index(['s1', 's2'], name='season'),
        'usable': pd.Series([0.0, 1.0, 1.0, 1.0], index=pd.Index([0, 1, 2, 3], name='t')),
    }
    with differential(IN_GROUPS, sources, lp=True) as run:
        assert run.engine.diagnostics().rows == 3, "both seasons' opening rows, and the one reading a live slot"
        assert run.oracle == pytest.approx(10.0), 'the snapshot whose predecessor is masked is capped by nothing'


def test_a_coordinate_the_lookup_sends_nowhere_is_absent_rather_than_vacated():
    """A snapshot in no season at all: it reaches nothing for a reason the
    shift had nothing to do with, so `edge=0` does not speak for it.

    The two readings of "reached nothing" are what separates this from the
    test above: off a group's start the shift vacated the slot and the edge
    fills it, but a coordinate the lookup sends nowhere never had a
    predecessor to lose. It is the null a partial lookup gets everywhere else
    (#969, `sum(by=)`, `at()`), and filling it asserts `take <= 0` where the
    model said nothing.

    Before #1061 the eager lane filled it and the relational one did not, so
    the lanes reported 4 rows and 3 for the same file.
    """
    with differential(IN_GROUPS_UNMASKED, GROUPLESS_SOURCES, lp=True) as run:
        assert run.engine.diagnostics().rows == 3, (
            "each season's opening row and the one reading a live slot — not the group-less snapshot"
        )
        assert run.oracle == pytest.approx(10.0), 'take at the group-less snapshot is capped by nothing'


def test_a_group_less_coordinate_stays_absent_under_a_mask_that_removes_nothing():
    """The same question with a `where` on the operand that masks no row.

    Worth its own case because it takes the other path: with a presence frame
    the edge is rebuilt from the vacated set, without one it comes from the
    grouped labels — and a mask removing nothing must not decide whether a
    row exists.
    """
    sources = {**GROUPLESS_SOURCES, 'usable': pd.Series([1.0, 1.0, 1.0, 1.0], index=pd.Index([0, 1, 2, 3], name='t'))}
    with differential(IN_GROUPS, sources, lp=True) as run:
        assert run.engine.diagnostics().rows == 3, 'a mask that removes nothing changes no row'
        assert run.oracle == pytest.approx(10.0), 'take at the group-less snapshot is capped by nothing'


#: A nonzero edge over a variable-free operand, reached with a per-entity
#: offset: the other reader of the edge frame, and the other half of #1049.
BY_PARAMETER_CONSTANT = {
    'dimensions': {'g': {'dtype': 'str'}, 't': {'dtype': 'int'}},
    'parameters': {'lead': {'dims': ['g'], 'dtype': 'int'}, 'eff': {'dims': ['g', 't']}},
    'variables': {'x': {'foreach': ['g', 't'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'link': {'foreach': ['g', 't'], 'expression': 'x * shift(eff, over=t, offset=lead, edge=1) <= 10'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(sum(x, over=g), over=t)'},
}


def test_a_per_entity_offset_writes_a_nonzero_edge_where_that_entity_vacates():
    """The fill is a value, so the vacated coordinate is written rather than
    left out — and with a lead of one month only the first month is vacated.

    Efficiencies of 1, 2 and 4 give caps of 10, 10 and 5: the first from the
    edge's own 1, the second and third from the month before.
    """
    sources = {
        'g': pd.Index(['a'], name='g'),
        't': pd.Index([0, 1, 2], name='t'),
        'lead': pd.Series([1], index=pd.Index(['a'], name='g')),
        'eff': pd.DataFrame({'g': ['a'] * 3, 't': [0, 1, 2], 'value': [1.0, 2.0, 4.0]}),
    }
    with differential(BY_PARAMETER_CONSTANT, sources, lp=True) as run:
        assert run.engine.diagnostics().rows == 3, 'every month is capped, the first by the edge it was given'
        assert run.oracle == pytest.approx(25.0), '10 + 10 + 5'


#: The two gathers *together*: a lag that is per group rather than per entity,
#: which is a parameter declared over the dimension the partition groups into.
#: A period's own construction lead time, on a flat snapshot axis (#1161).
PER_GROUP_OFFSET = {
    'dimensions': {'t': {'dtype': 'int'}, 'period': {'dtype': 'int'}},
    'lookups': {'period_of': {'over': 't', 'into': 'period'}},
    'parameters': {'lead': {'dims': ['period'], 'dtype': 'int'}, 'v': {'dims': ['t']}},
    'variables': {'p': {'foreach': ['t'], 'bounds': {'lower': -100, 'upper': 100}}},
    'constraints': {
        'reads': {'foreach': ['t'], 'expression': 'p == shift(v, over=t, offset=lead, by=period_of, edge=0)'}
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
}

#: The same lag over a *variable*, where the edge is a term that is not there
#: rather than a number written into a frame — and one snapshot the lookup
#: sends nowhere, which no lag reaches and no edge speaks for.
PER_GROUP_OFFSET_TERMS = {
    'dimensions': {'t': {'dtype': 'int'}, 'season': {'dtype': 'str'}},
    'lookups': {'season_of': {'over': 't', 'into': 'season'}},
    'parameters': {'lead': {'dims': ['season'], 'dtype': 'int'}, 'cap': {'dims': ['t']}},
    'variables': {
        'level': {'foreach': ['t'], 'bounds': {'lower': 'cap', 'upper': 'cap'}},
        'take': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 100}},
    },
    'constraints': {
        'link': {'foreach': ['t'], 'expression': 'take <= shift(level, over=t, offset=lead, by=season_of, edge=0)'}
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(take, over=t)'},
}

#: Per entity *and* per group at once: one key the frame carries and one it
#: reaches through the lookup, in the same join.
PER_ENTITY_AND_PER_GROUP = {
    'dimensions': {'g': {'dtype': 'str'}, 't': {'dtype': 'int'}, 'season': {'dtype': 'str'}},
    'lookups': {'season_of': {'over': 't', 'into': 'season'}},
    'parameters': {'lead': {'dims': ['g', 'season'], 'dtype': 'int'}, 'v': {'dims': ['g', 't']}},
    'variables': {'p': {'foreach': ['g', 't'], 'bounds': {'lower': -100, 'upper': 100}}},
    'constraints': {
        'reads': {'foreach': ['g', 't'], 'expression': 'p == shift(v, over=t, offset=lead, by=season_of, edge=0)'}
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
}


def _by_t(result, name: str) -> list[float]:
    """One variable's primals in coordinate order, which is what a lag is about."""
    return result.primal(name).sort('t')['value'].to_list()


def test_a_per_group_offset_translates_each_group_by_its_own_lag():
    """The lag is the group's, so two periods with different lead times each
    reach back their own distance and vacate their own opening rows.

    The first period leads by one and the second by two, which no single
    number reproduces: shifting the whole axis by either lands three of the
    six coordinates somewhere else.
    """
    sources = {
        't': pd.DataFrame({'t': [0, 1, 2, 3, 4, 5]}),
        'period_of': relation('t', 'period', [0, 1, 2, 3, 4, 5], [2030] * 3 + [2050] * 3),
        'period': pd.Index([2030, 2050], name='period'),
        'lead': pd.Series([1, 2], index=pd.Index([2030, 2050], name='period')),
        'v': pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], index=pd.Index(range(6), name='t')),
    }
    with differential(PER_GROUP_OFFSET, sources, lp=True) as run:
        assert list(run.model.constraints['reads'].coords) == ['t'], (
            'the row is the snapshot, and the period its lag was read at is not a coordinate of it'
        )
        assert run.model.constraints['reads'].size == run.engine.diagnostics().rows == 6, (
            'one row per snapshot, and the same number of them on both lanes'
        )
        assert _by_t(run.result, 'p') == [0.0, 10.0, 20.0, 0.0, 0.0, 40.0], (
            "each period reaches back its own lead inside its own group, and its opening rows take the edge's zero"
        )
        assert run.oracle == pytest.approx(70.0), '0 + 10 + 20 + 0 + 0 + 40'


def test_a_per_group_offset_over_a_variable_vacates_each_groups_opening_rows():
    """The same lag where the operand carries terms rather than values.

    ``level`` is pinned to ``cap`` by its bounds, so what ``take`` may reach is
    the lag read plainly: the second season leads by two, so both of its
    opening rows are capped by the edge's zero rather than by a term. The
    snapshot in no season is capped by nothing at all — it belongs to no group,
    so it reaches nothing, and ``edge=0`` does not speak for it (#1061).
    """
    sources = {
        't': pd.DataFrame({'t': [0, 1, 2, 3, 4, 5, 6]}),
        'season_of': relation('t', 'season', range(7), ['s1'] * 3 + ['s2'] * 3 + [None]),
        'season': pd.Index(['s1', 's2'], name='season'),
        'lead': pd.Series([1, 2], index=pd.Index(['s1', 's2'], name='season')),
        'cap': pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], index=pd.Index(range(7), name='t')),
    }
    with differential(PER_GROUP_OFFSET_TERMS, sources, lp=True) as run:
        assert run.engine.diagnostics().rows == 6, 'one row per snapshot in a season, and none for the one in no season'
        assert _by_t(run.result, 'take') == [0.0, 10.0, 20.0, 0.0, 0.0, 40.0, 100.0], (
            'each season reaches back its own lead, and the group-less snapshot is capped by nothing'
        )
        assert run.oracle == pytest.approx(170.0)


def test_an_offset_may_differ_per_entity_and_per_group_at_once():
    """Two keys, one join: the entity's own column and the group's, which the
    frame carries only as the lookup's value.

    The second unit leads by two in the second season and by one everywhere
    else, so it alone vacates both of that season's rows.
    """
    units, periods = ['a', 'b'], [0, 1, 2, 3]
    sources = {
        'g': pd.Index(units, name='g'),
        't': pd.DataFrame({'t': periods}),
        'season_of': relation('t', 'season', periods, ['s1', 's1', 's2', 's2']),
        'season': pd.Index(['s1', 's2'], name='season'),
        'lead': pd.DataFrame({'g': ['a', 'a', 'b', 'b'], 'season': ['s1', 's2'] * 2, 'value': [1, 1, 1, 2]}),
        'v': pd.DataFrame(
            {
                'g': [u for u in units for _ in periods],
                't': periods * 2,
                'value': [10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0],
            }
        ),
    }
    with differential(PER_ENTITY_AND_PER_GROUP, sources, lp=True) as run:
        read = run.result.primal('p').sort('g', 't')
        assert read['value'].to_list() == [0.0, 10.0, 0.0, 30.0, 0.0, 1.0, 0.0, 0.0], (
            "the lead is read at the pair, not at either key alone — 'b' vacates both rows of the season it leads by two"
        )
        assert run.oracle == pytest.approx(41.0), '(0 + 10 + 0 + 30) + (0 + 1 + 0 + 0)'


OFFSET_OUT_OF_REACH = {
    'nothing-puts-it-in-reach': ('lead', '', r"over \['season'\], which the shifted expression does not carry"),
    'not-what-the-partition-groups-into': ('far', ', by=season_of', r"over \['g'\], which the shifted expression"),
}


@pytest.mark.parametrize(
    ('offset', 'partition', 'match'), list(OFFSET_OUT_OF_REACH.values()), ids=list(OFFSET_OUT_OF_REACH)
)
def test_an_offset_over_a_dim_nothing_puts_in_reach_is_refused(offset: str, partition: str, match: str):
    """An offset is read at the coordinate it moves, so it must vary over a dim
    that coordinate has — the shifted expression's own, or the one a partition
    groups into.

    Neither refusal is pedantry: the eager lane broadcast the shifted
    expression onto the stray dim and built a bigger model than the file reads
    as, while the relational lane asked for a column no frame carries (#1161).
    """
    model = {
        'dimensions': {
            'g': {'dtype': 'str', 'values': ['a']},
            't': {'dtype': 'int', 'values': [0, 1]},
            'season': {'dtype': 'str', 'values': ['s']},
        },
        'lookups': {'season_of': {'over': 't', 'into': 'season'}},
        'parameters': {'lead': {'dims': ['season'], 'dtype': 'int'}, 'far': {'dims': ['g'], 'dtype': 'int'}},
        'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 1}}},
        'constraints': {
            'k': {'foreach': ['t'], 'expression': f"x >= shift(x, over=t, offset={offset}, edge='wrap'{partition})"}
        },
        'objective': {'sense': 'minimize', 'expression': 'sum(x)'},
    }
    with pytest.raises(LanguageError, match=match):
        lps.check(model)


def test_shift_semantics_are_positional_not_lexicographic():
    """Coords whose sorted order differs from declared order (string labels:
    lexicographic t0,t1,t10,... vs positional t0..t47). Both backends must
    couple the same neighbours."""
    n_s = 48
    labels = pd.Index([f't{i}' for i in range(n_s)], name='snapshot')
    assert list(labels.sort_values()) != list(labels), 'the fixture is only a fixture if sorted != positional'

    p_max = pd.Series({'wind': 80.0, 'gas': 70.0})
    t = np.arange(n_s)
    data = {
        'p_max': p_max,
        'cost': pd.Series({'wind': 1.0, 'gas': 40.0}),
        'load': pd.Series((110 + 60 * np.sin(2 * np.pi * t / 24)).round(3), index=labels),
    }
    data |= {'snapshot': labels, 'generator': pd.Index(p_max.index, name='generator')}

    original = STORAGE_YAML.read_text()
    assert 'dtype: int' in original
    with differential(original.replace('dtype: int', 'dtype: str'), data):
        pass  # agreement on the objective is the whole assertion


RAMP_MODEL = override(
    DISPATCH_MODEL,
    **{
        'parameters.ramp_max': {'dims': ['generator']},
        'constraints.ramp_up': {
            'foreach': ['snapshot', 'generator'],
            'where': 'snapshot > 0',
            'expression': 'p - shift(p, over=snapshot, offset=1) <= ramp_max',
        },
    },
)


def test_a_where_on_dimension_coordinates_means_the_same_on_both_lanes():
    """ROADMAP 5b: `where: "snapshot > 0"` must mean the same on both lanes.

    The README's ramp example uses exactly this — a time-coupling constraint
    that skips the first snapshot. It used to be eager-only: lowering refused
    dimension comparisons, so the same file built two different models.
    """
    n_s = 12
    rng = np.random.default_rng(11)
    data = {
        'p_max': pd.Series({'wind': 80.0, 'gas': 200.0}),
        'cost': pd.Series({'wind': 1.0, 'gas': 40.0}),
        'ramp_max': pd.Series({'wind': 100.0, 'gas': 25.0}),  # binding on gas
        'load': pd.Series(
            (rng.uniform(0.3, 0.9, n_s) * 200.0).round(3),
            index=pd.RangeIndex(n_s, name='snapshot'),
        ),
    }
    data |= {'snapshot': pd.RangeIndex(n_s, name='snapshot')}

    with differential(RAMP_MODEL, data) as run:
        active = int((run.model.constraints['ramp_up'].labels != -1).sum())
        assert active == (n_s - 1) * 2, (
            'the mask must bite: the first snapshot is dropped per generator, and a masked row on '
            'the eager lane carries label -1'
        )


# ---------------------------------------------------------------------------
# lowering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('expression', 'expected'),
    [
        pytest.param(
            "shift(soc, over=snapshot, offset=1, edge='wrap')",
            Translate(Variable('soc'), 'snapshot', 1),
            id='wrap',
        ),
        pytest.param(
            "shift(soc, over=snapshot, offset=-2, edge='wrap')",
            Translate(Variable('soc'), 'snapshot', -2),
            id='wrap-backwards',
        ),
        pytest.param(
            'shift(soc, over=snapshot, offset=1)',
            Translate(Variable('soc'), 'snapshot', 1, wrap=False),
            id='bare',
        ),
        # fill is the field both lanes branch on: None is absence, 0.0 the zero.
        pytest.param(
            'shift(soc, over=snapshot, offset=1, edge=0)',
            Translate(Variable('soc'), 'snapshot', 1, wrap=False, fill=0.0),
            id='zero-fill',
        ),
    ],
)
def test_translation_lowers_to_a_bounded_halo(expression, expected):
    assert _Lowering(STORAGE_SCHEMA, 't').expr(resolved(expression, STORAGE_SCHEMA)) == expected


@pytest.mark.parametrize(
    ('expression', 'match'),
    [
        pytest.param(
            "shift(soc, over=nope, offset=1, edge='wrap')",
            r'shift\(over=nope\) does not name a declared dimension',
            id='over-names-no-dimension',
        ),
        pytest.param(
            "shift(load, over=generator, offset=1, edge='wrap')",
            'but the expression has dims',
            id='a-dim-the-expression-lacks',
        ),
        # `edge=` is a closed keyword: one keyword carries all three policies, so
        # "cyclic, and also fill" has no spelling left to be refused.
        pytest.param(
            'shift(soc, over=snapshot, offset=1, edge=nonsense)',
            'is not an edge policy',
            id='the-edge-keyword-is-closed',
        ),
        # Over a variable only `edge=0` is sayable — a nonzero fill would put a
        # constant where a term was.
        pytest.param(
            'shift(soc, over=snapshot, offset=1, edge=1)',
            'only fill=0 is representable there',
            id='a-nonzero-fill-over-a-variable',
        ),
    ],
)
def test_a_shift_neither_lane_can_honour_is_refused_at_lowering(expression, match):
    with pytest.raises(LanguageError, match=match):
        _Lowering(STORAGE_SCHEMA, 't').expr(resolved(expression, STORAGE_SCHEMA))


FILL_IDENTITY_MODEL = """
dimensions: {t: {dtype: int, values: [0, 1, 2]}}
parameters:
  eff: {dims: [t]}
variables:
  x: {foreach: [t], bounds: {lower: 0, upper: 100}}
constraints:
  c:
    foreach: [t]
    expression: "x * shift(eff, over=t, offset=1, edge=1) <= 10"
objective: {sense: maximize, expression: "sum(x, over=t)"}
"""


def test_the_fill_a_product_wants_is_one_not_zero():
    """``fill=`` takes the identity of the *position*, which is why it takes a number.

    linopy v1 refuses to fill on the caller's behalf precisely because the right
    value is positional (``convention.rst`` §7): 0 is the identity of a sum, 1 of
    a product. ``x * shift(eff, over=t, offset=1, edge=0)`` would force ``x`` to zero at the
    first coordinate — the pin again, wearing the coefficient's hat — where
    ``fill=1`` leaves it governed by its own bound.

    Over data any number is allowed, since it is a data fill. The relational
    lane has to *write* the rows for a nonzero one: a const fragment reads a
    missing row as zero, so `fill=1` exists only if something puts it there.
    """
    with differential(FILL_IDENTITY_MODEL, {'eff': pd.Series({0: 2.0, 1: 4.0, 2: 5.0})}, lp=True) as run:
        x = by_coord(run.result, 'x', 't')
        assert x[0] == pytest.approx(10.0), 't=0: the fill is 1, so the bound is 10/1'
        assert x[1] == pytest.approx(5.0), 't=1: eff[0] = 2, so 10/2'
        assert x[2] == pytest.approx(2.5), 't=2: eff[1] = 4, so 10/4'


EDGE_MODEL = {
    'dimensions': {'t': {'dtype': 'int', 'values': [0, 1, 2]}, 'wrap': {'dtype': 'str', 'values': ['a', 'b']}},
    'parameters': {'c': {'dims': ['t']}},
    'variables': {'x': {'foreach': ['t', 'wrap'], 'bounds': {'lower': 0, 'upper': 5}}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x * c)'},
}


def _with(expr):
    return {**EDGE_MODEL, 'constraints': {'r': {'foreach': ['t', 'wrap'], 'expression': expr}}}


@pytest.mark.parametrize(
    'edge',
    ["edge='wrap'", 'edge="wrap"', 'edge=0'],
    ids=['single', 'double', 'zero fill'],
)
def test_an_edge_policy_is_quoted_or_a_number(edge):
    """The keyword is quoted; the fill is bare.

    A bare word in a kwarg value is a *name to resolve* — `over=wrap` names a
    dimension — so the one closed keyword `edge=` takes has to say it is a
    literal. Numbers need no quotes because a number is never a name.
    """
    lps.check(_with(f'x - shift(x, over=t, offset=1, {edge}) <= 1'))


def test_a_bare_wrap_names_a_dimension_and_is_refused():
    """`over=wrap, edge=wrap` was legal, and the same token meant two things.

    The model here declares a dimension actually called `wrap`, which is what
    makes the ambiguity concrete rather than theoretical: the parser resolved
    the two positions differently and a reader could not.
    """
    with pytest.raises(ValueError) as exc:
        lps.check(_with('x - shift(x, over=t, offset=1, edge=wrap) <= 1'))

    assert 'bare name where a keyword belongs' in str(exc.value)
    assert "edge='wrap'" in str(exc.value), 'the refusal has to name the rewrite'


def test_a_quoted_keyword_outside_a_kwarg_does_not_parse():
    """Quotes are for closed keywords in kwarg values, not for arithmetic.

    The *grammar* refuses this rather than resolution, which is the stronger
    place for it — a quoted word in arithmetic is not a name and not a number,
    so there is nothing for a later pass to say about it. `resolution.py` keeps
    a branch for the shape anyway, reachable only from a hand-built AST.
    """
    with pytest.raises(ValueError) as exc:
        lps.check(_with("x - 'wrap' <= 1"))

    assert 'Failed to parse expression' in str(exc.value)


def _shift_over_data(where: str | None = None, edge: str | None = None) -> dict[str, object]:
    shift = f'shift(dt, over=t, offset=1, edge={edge})' if edge else 'shift(dt, over=t, offset=1)'
    constraint: dict[str, object] = {'foreach': ['t'], 'expression': f'x <= {shift}'}
    if where is not None:
        constraint['where'] = where
    return {
        'dimensions': {'t': {'dtype': 'int', 'values': [0, 1, 2]}},
        'parameters': {'dt': {'dims': ['t']}},
        'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 5}}},
        'constraints': {'c': constraint},
        'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
    }


def test_the_bare_shift_refusal_names_the_pair_that_actually_omits_the_row():
    """`edge=` and `where:` are a companion pair here, not a choice.

    Each is wrong alone, which is why listing them as alternatives was the
    defect: a `where` does not lift the refusal, because it is decided on the
    expression before any mask is read; and `edge=0` alone leaves a row at the
    vacated coordinate whose bound is that zero — the silent pinning the
    refusal exists to prevent.

    Held as behaviour and as wording, because the wording is the only thing
    standing between a reader and the `edge=0`-alone answer, which builds and
    solves and is wrong.
    """
    with pytest.raises(LanguageError, match='vacated positions') as bare:
        lps.check(_shift_over_data())
    with pytest.raises(LanguageError, match='vacated positions') as masked:
        lps.check(_shift_over_data(where='t > 0'))
    assert str(bare.value) == str(masked.value), 'a mask lifts the refusal, so it is an alternative after all'

    message = str(bare.value)
    assert 'where' in message, 'the way to omit the row has to be reachable from the error'
    assert "edge='wrap'" in message
    assert 'edge=0 alone' in message, 'the trap has to be named, not just the remedy'


def test_edge_zero_alone_binds_the_vacated_row_and_a_where_frees_it():
    """The measurement the message is built on.

    `edge=0` alone is not a refusal and not an error — it solves, and the
    answer is wrong in the direction that looks like a tight model.
    """
    sources = {'dt': pl.DataFrame({'t': [0, 1, 2], 'value': [1.0, 1.0, 1.0]})}
    pinned = lps.solve(_shift_over_data(edge='0'), sources)
    omitted = lps.solve(_shift_over_data(edge='0', where='t > 0'), sources)

    assert pinned.primal('x')['value'].to_list()[0] == 0.0, 'edge=0 alone should pin the vacated row'
    assert omitted.primal('x')['value'].to_list()[0] == 5.0, 'the where should omit it entirely'
    assert omitted.objective > pinned.objective


NESTED_SHIFTS = {
    'same-dim': 'shift(shift(p, over=t, offset=1), over=t, offset=1)',
    'cross-dim': 'shift(shift(p, over=t, offset=1), over=g, offset=1)',
    'cross-dim-reversed': 'shift(shift(p, over=g, offset=1), over=t, offset=1)',
    'triple-mixed': 'shift(shift(shift(p, over=t, offset=1), over=g, offset=1), over=t, offset=1)',
    'inner-fill': 'shift(shift(p, over=t, offset=1, edge=0), over=t, offset=1)',
    'outer-wrap': "shift(shift(p, over=t, offset=1), over=t, offset=1, edge='wrap')",
}


@pytest.mark.parametrize('rhs', NESTED_SHIFTS.values(), ids=list(NESTED_SHIFTS))
def test_a_nested_shift_agrees_with_the_oracle(rhs: str):
    """A shift over a shift, in every arrangement of edge and dimension.

    `shift` takes any node of the right dim set (the operator rules), so nesting is inside
    what the language accepts — and the eager lane always built it. The
    relational lane raised a raw `polars.ColumnNotFoundError` instead, because
    an acyclic inner shift leaves a presence narrower than the fragment and the
    outer one projected the fragment's dims onto it.

    The coefficient and the `+ 1` are what make the row bind: without them
    every variable sits at its upper bound and the lanes agree on an answer
    neither of them computed from the shift.
    """
    model = {
        'dimensions': {'t': {'dtype': 'int', 'values': [0, 1, 2, 3, 4]}, 'g': {'dtype': 'str', 'values': ['a', 'b']}},
        'parameters': {'c': {'dims': ['g']}},
        'variables': {'p': {'foreach': ['t', 'g'], 'bounds': {'lower': 0, 'upper': 5}}},
        'constraints': {'k': {'foreach': ['t', 'g'], 'expression': f'p <= 0.5 * {rhs} + 1'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(p * c)'},
    }
    data = {'c': pd.Series([1.0, 2.0], index=pd.Index(['a', 'b'], name='g'))}
    with differential(model, data) as run:
        primal = run.result.primal('p')['value'].to_numpy()
        assert not np.allclose(primal, 5.0), 'nothing binds, so the lanes would agree on an unconstrained model'


@pytest.mark.parametrize('edge', ["'wrap'", '0'], ids=['wrap', 'fill'])
def test_an_offset_may_differ_per_entity(edge: str):
    """`by=` names a parameter: each entity is translated by its own amount.

    A lead time, a transit time, a minimum up time — every one of them is a
    column in the source data, and writing one `shift` per distinct value with
    a mask selecting the rows that carry it is what this replaces.

    The instance discriminates: an order placed at *t* arrives at *t + lead*,
    demand falls only in the last period, and the two units have different
    leads. If the offset were read once for both, one of them would order in
    the wrong period and the primal would say so.
    """
    lead = {'slow': 1, 'fast': 2}
    units, periods = list(lead), [0, 1, 2, 3]
    model = {
        'dimensions': {
            'g': {'dtype': 'str', 'values': list(lead)},
            't': {'dtype': 'int', 'values': [0, 1, 2, 3]},
        },
        'parameters': {
            'lead': {'dims': ['g'], 'dtype': 'int'},
            'c': {'dims': ['g']},
            'demand': {'dims': ['g', 't']},
        },
        'variables': {'order': {'foreach': ['g', 't'], 'bounds': {'lower': 0, 'upper': 9}}},
        'constraints': {
            'arrive': {
                'foreach': ['g', 't'],
                'expression': f'shift(order, over=t, offset=lead, edge={edge}) >= demand',
            }
        },
        'objective': {'sense': 'minimize', 'expression': 'sum(order * c)'},
    }
    data = {
        'lead': pd.Series([lead[u] for u in units], index=pd.Index(units, name='g')),
        'c': pd.Series([1.0, 1.0], index=pd.Index(units, name='g')),
        'demand': pd.DataFrame(
            {
                'g': [u for u in units for _ in periods],
                't': periods * len(units),
                'value': [0.0, 0.0, 0.0, 5.0] * len(units),
            }
        ),
    }
    with differential(model, data) as run:
        assert run.result.objective == pytest.approx(10.0)
        placed = run.result.primal('order').filter(pl.col('value') > 1e-9)
        assert dict(zip(placed['g'].to_list(), placed['t'].to_list(), strict=True)) == {'slow': 2, 'fast': 1}, (
            'each unit orders one of its own lead times before the demand, not a shared one'
        )


NAMED_OFFSET_REFUSALS = {
    'not-a-parameter': ('missing', "'missing' not found"),
    'not-an-integer': ('rate', 'needs an integer parameter'),
    'along-the-shifted-dim': ('drift', 'the dimension being translated'),
}


@pytest.mark.parametrize(('offset', 'match'), list(NAMED_OFFSET_REFUSALS.values()), ids=list(NAMED_OFFSET_REFUSALS))
def test_a_named_offset_that_cannot_mean_a_lag_is_refused(offset: str, match: str):
    """The three ways a per-entity offset stops being a translation.

    An undeclared name is a typo, and resolution refuses it before lowering
    sees it — pinned here so the better message stays the one that prints. A
    non-integral offset cannot land on a coordinate: it counts positions rather
    than measuring a distance.
    And one that spans the dimension it translates moves each position by a
    different amount *along the axis it is moving*, which is a permutation with
    no reading as a lag.
    """
    model = {
        'dimensions': {'g': {'dtype': 'str', 'values': ['a']}, 't': {'dtype': 'int', 'values': [0, 1]}},
        'parameters': {
            'lead': {'dims': ['g'], 'dtype': 'int'},
            'rate': {'dims': ['g']},
            'drift': {'dims': ['g', 't'], 'dtype': 'int'},
        },
        'variables': {'x': {'foreach': ['g', 't'], 'bounds': {'lower': 0, 'upper': 1}}},
        'constraints': {
            'k': {'foreach': ['g', 't'], 'expression': f"x >= shift(x, over=t, offset={offset}, edge='wrap')"}
        },
        'objective': {'sense': 'minimize', 'expression': 'sum(x * 1.0)'},
    }
    with pytest.raises(LanguageError, match=match):
        lps.check(model)


def test_a_named_offset_must_say_what_the_vacated_positions_contribute():
    """The absent edge is refused for a named offset, deliberately and for now.

    A bare `shift` leaves the vacated positions absent, and absence is carried
    by a presence frame keyed by the translated dimension alone. A per-entity
    offset vacates a *different* slot for each entity, which that frame cannot
    say — so the case is refused rather than answered wrongly, and the two
    edges that write their own answer are allowed.
    """
    model = {
        'dimensions': {'g': {'dtype': 'str', 'values': ['a']}, 't': {'dtype': 'int', 'values': [0, 1]}},
        'parameters': {'lead': {'dims': ['g'], 'dtype': 'int'}},
        'variables': {'x': {'foreach': ['g', 't'], 'bounds': {'lower': 0, 'upper': 1}}},
        'constraints': {'k': {'foreach': ['g', 't'], 'expression': 'x >= shift(x, over=t, offset=lead)'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(x * 1.0)'},
    }
    with pytest.raises(LanguageError, match='vacated positions absent'):
        lps.check(model)


def test_a_named_offset_carries_its_sign_in_the_data():
    """`by=-lead` is refused rather than negated, so one spelling means one thing."""
    model = {
        'dimensions': {'g': {'dtype': 'str', 'values': ['a']}, 't': {'dtype': 'int', 'values': [0, 1]}},
        'parameters': {'lead': {'dims': ['g'], 'dtype': 'int'}},
        'variables': {'x': {'foreach': ['g', 't'], 'bounds': {'lower': 0, 'upper': 1}}},
        'constraints': {'k': {'foreach': ['g', 't'], 'expression': "x >= shift(x, over=t, offset=-lead, edge='wrap')"}},
        'objective': {'sense': 'minimize', 'expression': 'sum(x * 1.0)'},
    }
    with pytest.raises(LanguageError, match='negates a named offset'):
        lps.check(model)


def _reindexed_parameter_model(op: str) -> dict:
    return {
        'dimensions': {'t': {'dtype': 'int', 'values': [0, 1, 2]}},
        'parameters': {'dt': {'dims': ['t']}},
        'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 100}}},
        'constraints': {'r': {'foreach': ['t'], 'expression': f'x <= {op}'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x, over=t)'},
    }


@pytest.mark.parametrize(
    ('op', 'expected'),
    [
        pytest.param(
            "shift(dt, over=t, offset=1, edge='wrap')",
            {0: 7.0, 1: 5.0, 2: 6.0},
            id='cyclic-vacates-nothing-so-t0-reads-the-last-value',
        ),
        pytest.param(
            'shift(dt, over=t, offset=1, edge=0)',
            {0: 0.0, 1: 5.0, 2: 6.0},
            id='the-vacated-position-contributes-zero-which-pins',
        ),
    ],
)
def test_roll_and_filled_shift_re_index_a_parameter_not_only_a_variable(op, expected):
    """``array`` is any node, so these operators read a parameter.

    Worth its own test because every documented example took a variable, and a
    downstream consumer built and shipped a hand-shifted copy of a parameter
    table before probing revealed this works.

    ``fill=0`` is what a *bare* ``shift`` used to mean here, and the pin it
    produces at ``t=0`` is why it stopped being the default — see the refusal
    below. Spelled out, it is a legitimate thing to ask for, so it still works.
    """
    data = {'dt': pd.Series({0: 5.0, 1: 6.0, 2: 7.0})}
    with differential(_reindexed_parameter_model(op), data, lp=True) as run:
        x = by_coord(run.result, 'x', 't')
        for t, want in expected.items():
            assert x[t] == pytest.approx(want, abs=1e-9), f'{op} at t={t}'


def test_a_bare_shift_over_data_is_refused_rather_than_filled():
    """The pin, removed at its source (#289).

    ``x <= shift(dt, over=t, offset=1)`` used to build ``x <= 0`` at the first coordinate:
    a bound invented from a slot that has no value. Absence would be the
    consistent answer, but a parameter has no absence to propagate — a missing
    row is a zero coefficient (the absence rules) — so this follows linopy v1
    and refuses,
    at load time, naming the three things the author might have meant.

    Decidable without data, so ``lps.check()`` catches it: the operand is
    variable-free by declaration, not by what arrives in ``sources``.
    """
    model = _reindexed_parameter_model('shift(dt, over=t, offset=1)')
    with pytest.raises(LanguageError) as exc:
        lps.check(model)
    assert 'edge=0' in str(exc.value), 'the refusal must name the escape hatch'
    assert "edge='wrap'" in str(exc.value), 'and the policy for a genuinely cyclic horizon'
