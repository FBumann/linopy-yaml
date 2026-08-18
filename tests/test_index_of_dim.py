"""``index(dim, i)`` — a boundary named by position, not by the label there.

A recurrence needs its first position seeded, and every seeding clause in the
corpus used to name the label that happened to sit there (``snapshot == 0``).
That is correct only while the instance starts at zero: relabel the horizon and
the clause matches nothing, so the row it was to seed is never written and the
recurrence is left unanchored.

The failure is loud but names nothing about its cause — an infeasible solve —
which is why the relabel is the test that matters here, and why an
out-of-range position is an error rather than a mask that is false everywhere.

Both lanes read the position off the coordinate order they already hold: the
dim table's ``ord`` relationally, the master index on the eager side. So the
one thing a single-lane test could not see is whether the two orders agree,
which every case below checks differentially.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import yaml as pyyaml

import lpspec as lps
from lpspec.errors import DataError, LanguageError, LpspecError
from tests.conftest import schema_of
from tests.differential import RTOL, differential
from tests.oracle import pd

MODEL = """
description: a storage level carried across a horizon, seeded at its first position

dimensions:
  snapshot: {dtype: int, description: dispatch periods in order}

parameters:
  soc_initial: {dims: [], description: the level carried in from before the horizon}
  inflow: {dims: [snapshot], description: energy arriving in each period}
  price: {dims: [snapshot], description: what a unit of output earns}

variables:
  soc:
    foreach: [snapshot]
    bounds: {lower: 0, upper: 100}
    description: energy stored at the end of a period
  out:
    foreach: [snapshot]
    bounds: {lower: 0, upper: 100}
    description: energy released in a period

constraints:
  soc_start:
    foreach: [snapshot]
    where: "snapshot == index(snapshot, 0)"
    expression: soc == soc_initial + inflow - out
    description: the first period has no predecessor, so it carries the initial level
  soc_carry:
    foreach: [snapshot]
    where: "snapshot != index(snapshot, 0)"
    expression: soc == shift(soc, over=snapshot, offset=1) + inflow - out
    description: every later period carries the previous one's level

objective:
  sense: maximize
  expression: sum(out * price, over=snapshot)
  description: revenue from what is released
"""

#: Deliberately not starting at zero — the whole point of the construct. A
#: clause written `snapshot == 0` matches nothing on this horizon.
SNAPSHOTS = [4, 5, 6]


def _inputs(snapshots=SNAPSHOTS):
    index = pd.Index(snapshots, name='snapshot')
    return {
        'snapshot': index,
        'soc_initial': 20.0,
        'inflow': pd.Series([5.0, 5.0, 5.0], index=index),
        'price': pd.Series([1.0, 2.0, 3.0], index=index),
    }


# ---------------------------------------------------------------------------
# both lanes
# ---------------------------------------------------------------------------


def test_a_boundary_survives_the_horizon_being_relabelled():
    """The optimum, hand-derived, on a horizon that starts at 4.

    35 units of energy exist across the horizon and no more — 20 carried in,
    5 arriving in each of three periods — and the price rises, so everything
    is held back and released in the last period. Revenue is 35·3.

    The seeded row is what caps it: without it nothing ties the first period's
    level to `soc_initial`, and the total stops being 35 at all.
    """
    sources = _inputs()
    with differential(MODEL, sources, lp=True) as run:
        assert run.oracle == pytest.approx(35 * 3.0, rel=RTOL)
        levels = dict(run.result.primal('soc').select('snapshot', 'value').iter_rows())

    assert levels[4] == pytest.approx(25.0), 'the seeded row: 20 carried in, 5 arriving, nothing released'
    assert levels[6] == pytest.approx(0.0), 'and the recurrence carries it to the end, where it all goes'


def test_the_label_that_happens_to_be_first_is_not_the_rule():
    """The same model with the horizon at 0..2 gives the same answer.

    Which is the claim: the clause names the position, so relabelling the index
    moves the boundary with it rather than silently seeding nothing.
    """
    sources = _inputs(snapshots=[0, 1, 2])
    with differential(MODEL, sources) as run:
        assert run.oracle == pytest.approx(105.0, rel=RTOL)


def test_the_hardcoded_label_is_what_this_replaces():
    """Written the old way, the relabelled horizon seeds no row — and solves.

    This is the failure #707 was filed for, and it is the silent kind. The
    seeding clause matches nothing, the recurrence's own first row is dropped
    because the level it shifts from does not exist, and the first period's
    release is left in no constraint at all. The solve comes back `optimal`
    with an objective four times the true one.

    Held here so the replacement has something to be better than, and so the
    number says how wrong it was rather than that it was wrong.
    """
    sources = _inputs()
    hardcoded = pyyaml.safe_load(MODEL.replace('index(snapshot, 0)', '0'))
    with lps.solve(hardcoded, _relational(sources)) as result:
        assert result.is_ok, 'the point is that nothing complains'
        assert result.objective == pytest.approx(420.0), (
            'an unanchored recurrence releases energy the model never had — 420 against the true 105'
        )


def test_a_negative_position_counts_from_the_end():
    """`-1` is the last coordinate — the cyclic boundary's other half."""
    sources = _inputs()
    cyclic = MODEL.replace(
        'expression: soc == soc_initial + inflow - out',
        'expression: soc == soc_initial + inflow - out',
    ).replace(
        """objective:""",
        """  soc_final:
    foreach: [snapshot]
    where: "snapshot == index(snapshot, -1)"
    expression: soc >= 10
    description: the last period ends with at least ten stored

objective:""",
    )
    with differential(cyclic, sources) as run:
        assert run.oracle == pytest.approx(25 * 3.0, rel=RTOL), (
            'ten held back at the end is ten not sold at the top price'
        )


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def _relational(sources):
    snapshots = list(sources['snapshot'])
    return {
        'snapshot': pl.DataFrame({'snapshot': snapshots}),
        'soc_initial': pl.DataFrame({'value': [sources['soc_initial']]}),
        'inflow': pl.DataFrame({'snapshot': snapshots, 'value': list(sources['inflow'])}),
        'price': pl.DataFrame({'snapshot': snapshots, 'value': list(sources['price'])}),
    }


@pytest.mark.parametrize('position', [3, -4], ids=['past-the-end', 'past-the-start'])
def test_a_position_no_coordinate_occupies_is_an_error_at_bind(tmp_path, position):
    """Not an empty mask: seeding no row is exactly what the construct prevents.

    Three snapshots, so positions 0..2 and -1..-3 exist and nothing else. Both
    lanes have to say so — a lane that quietly matched nothing would put the
    model back where the hardcoded label left it.
    """
    sources = _inputs()
    model = MODEL.replace('index(snapshot, 0)', f'index(snapshot, {position})')
    path = tmp_path / 'model.yaml'
    path.write_text(model)

    with pytest.raises(DataError, match=r'which has 3 coordinate\(s\)'):
        lps.solve(pyyaml.safe_load(model), _relational(sources))

    from tests.oracle import lpspec_linopy

    with pytest.raises(DataError, match=r'which has 3 coordinate\(s\)'):
        lpspec_linopy.build(path, sources)


@pytest.mark.parametrize(
    ('where', 'match'),
    [
        pytest.param(
            'inflow == index(inflow, 0)',
            r"index\(\) counts along a dimension's coordinates, and 'inflow' is a parameter",
            id='index-of-a-parameter',
        ),
        pytest.param(
            'snapshot == index(nowhere, 0)',
            r"index\(\) counts along a dimension's coordinates, and 'nowhere' is not declared",
            id='index-of-nothing',
        ),
    ],
)
def test_a_name_index_cannot_count_along_is_refused(where, match):
    with pytest.raises(LanguageError, match=match):
        resolved_where(where)


def test_two_different_dimensions_cannot_be_compared_by_position():
    """One dimension's coordinate against another's masks everything out.

    The same refusal a bare `generator == snapshot` gets, and for the same
    reason: no label of one is a label of the other.
    """
    model = MODEL.replace(
        'dimensions:\n  snapshot: {dtype: int, description: dispatch periods in order}',
        'dimensions:\n  snapshot: {dtype: int, description: dispatch periods in order}\n'
        '  other: {dtype: int, description: a second axis}',
    ).replace('"snapshot == index(snapshot, 0)"', '"snapshot == index(other, 0)"')
    with pytest.raises(LanguageError, match=r'compares a .snapshot. coordinate against an? .other. one'):
        schema_of(model)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def resolved_where(where: str):
    """The predicate a backend would receive — resolution is where these fail."""
    from lpspec.language.resolution import Namespace, where_of

    schema = schema_of(MODEL)
    return where_of(where, Namespace.of(schema), 't')


# ---------------------------------------------------------------------------
# by=lookup — the boundary of each group
# ---------------------------------------------------------------------------

#: Irregular on purpose: two snapshots in the first period, three in the second,
#: and one belonging to no period at all. A rectangular grid would let a wrong
#: implementation pass by counting along the whole axis.
GROUPED_SNAPSHOTS = [10, 11, 20, 21, 22, 99]
GROUPED_PERIODS = [0, 0, 1, 1, 1, None]

MASK = """
dimensions:
  snapshot: {dtype: int}
  period: {dtype: int}

lookups:
  period_of: {over: snapshot, into: period}

parameters:
  price: {dims: [snapshot]}

variables:
  soc: {foreach: [snapshot], bounds: {lower: 0, upper: 100}}

constraints:
  pin:
    foreach: [snapshot]
    where: "WHERE"
    expression: soc == 5

objective:
  sense: minimize
  expression: sum(soc * price, over=snapshot)
"""


def _grouped_sources():
    """One mapping both lanes take, the lookup arriving as a column of the index.

    Arrow tables rather than pandas: a partial lookup read out of a pandas frame
    arrives as ``float64`` beside an ``i64`` target, which is a binding question
    of its own and not the one under test here.
    """
    return {
        'snapshot': pl.DataFrame({'snapshot': GROUPED_SNAPSHOTS, 'period_of': GROUPED_PERIODS}),
        'period': pl.DataFrame({'period': [0, 1]}),
        'price': pl.DataFrame({'snapshot': GROUPED_SNAPSHOTS, 'value': [1.0] * len(GROUPED_SNAPSHOTS)}),
    }


def _masked(where: str) -> list[int]:
    """The snapshots *where* selects, agreed by both lanes.

    Read off the primal rather than the plan: minimising a positive price holds
    `soc` at zero everywhere the row was not built, so what comes back non-zero
    is exactly the mask — which is the thing under test, and the one an engine
    could get wrong on its own.
    """
    with differential(MASK.replace('WHERE', where), _grouped_sources()) as run:
        rows = run.result.primal('soc').filter(pl.col('value') > 1e-9)
        return sorted(int(s) for s in rows.select('snapshot').to_series())


def test_each_group_is_seeded_at_its_own_first_position():
    """The whole point: one boundary per group, not one for the axis."""
    assert _masked('snapshot == index(snapshot, 0, by=period_of)') == [10, 20], (
        "each period's first snapshot, not just the horizon's"
    )
    assert _masked('snapshot == index(snapshot, 0)') == [10], 'and the ungrouped spelling still names one'


def test_a_negative_position_is_each_group_s_last():
    """`-1` per group — the tail an ungrouped `index` cannot reach.

    With periods of different lengths there is no single position that is the
    last of both, which is why this is the case that decided the design.
    """
    assert _masked('snapshot == index(snapshot, -1, by=period_of)') == [11, 22]


def test_a_comparator_reads_the_same_grouped_as_ungrouped():
    """`>=` is 'at or after that position in my group', for every group.

    A guard against arithmetic that only works at zero: the offsets a row is
    compared through are signed, and an unsigned rank subtracted past zero
    wraps to a huge positive number instead — which `==` cannot see and every
    other comparator reads backwards.
    """
    assert _masked('snapshot >= index(snapshot, 1, by=period_of)') == [11, 21, 22], (
        "everything from each period's second snapshot on"
    )
    assert _masked('snapshot < index(snapshot, 1, by=period_of)') == [10, 20], 'and its complement'


def test_a_coordinate_in_no_group_has_no_boundary():
    """Snapshot 99 maps nowhere, so no group's boundary is its own.

    The same reading a null lookup value gets everywhere else — it belongs to
    no group, so `sum(by=)` places its terms nowhere and this places no row.
    """
    for where in (
        'snapshot == index(snapshot, 0, by=period_of)',
        'snapshot == index(snapshot, -1, by=period_of)',
        'snapshot >= index(snapshot, 0, by=period_of)',
    ):
        assert 99 not in _masked(where), f'{where} claimed a coordinate that is in no group'


def test_a_group_shorter_than_the_position_is_an_error_at_bind(tmp_path):
    """Not a mask that is false there: one short period would go unseeded.

    Position 2 exists in the three-snapshot period and not in the two-snapshot
    one, which is precisely the failure grouping makes easy to write and
    impossible to see in the answer.
    """
    model = MASK.replace('WHERE', 'snapshot == index(snapshot, 2, by=period_of)')
    path = tmp_path / 'model.yaml'
    path.write_text(model)
    sources = _grouped_sources()

    with pytest.raises(DataError, match=r'1 of them are shorter than that'):
        lps.solve(pyyaml.safe_load(model), sources)

    from tests.oracle import lpspec_linopy

    with pytest.raises(DataError, match=r'1 of them are shorter than that'):
        lpspec_linopy.build(path, sources)


@pytest.mark.parametrize(
    ('by', 'match'),
    [
        pytest.param('price', r"groups by 'price', which is a parameter", id='by-a-parameter'),
        pytest.param('period', r"groups by 'period', which is a dimension", id='by-a-dimension'),
        pytest.param('nowhere', r"groups by 'nowhere', which is not declared", id='by-nothing'),
    ],
)
def test_by_takes_a_lookup(by, match):
    """`by=` is the same word it is in `sum(by=)` and `at(by=)`, or it is nothing."""
    model = MASK.replace('WHERE', f'snapshot == index(snapshot, 0, by={by})')
    with pytest.raises(LanguageError, match=match):
        schema_of(model)


def test_a_lookup_over_another_dimension_carries_no_position():
    """Grouping needs a lookup over the dimension being counted.

    A lookup over something else names groups no row of this dimension is in,
    so there is no position within a group for the clause to be about.
    """
    model = (
        MASK.replace('WHERE', 'snapshot == index(snapshot, 0, by=plant_period)')
        .replace('  period: {dtype: int}', '  period: {dtype: int}\n  plant: {dtype: str}')
        .replace(
            '  period_of: {over: snapshot, into: period}',
            '  period_of: {over: snapshot, into: period}\n  plant_period: {over: plant, into: period}',
        )
    )
    with pytest.raises(LanguageError, match=r"counts positions along 'snapshot' but groups by a lookup over 'plant'"):
        schema_of(model)


# ---------------------------------------------------------------------------
# the teaching model
# ---------------------------------------------------------------------------

SEASONS = Path('examples/seasons.yaml')
SEASONS_PAGE = Path('docs/examples/seasons.md')


def _seasons_sources():
    """Snapshots numbered from **1**, and seasons of four and three.

    Both are the model's argument in the data: no clause names a label, and no
    single position along the axis is the last of a four-snapshot season *and*
    of a three-snapshot one.
    """
    snapshots = [1, 2, 3, 4, 5, 6, 7]
    return {
        'snapshot': pl.DataFrame({'snapshot': snapshots, 'season_of': ['winter'] * 4 + ['summer'] * 3}),
        'season': pl.DataFrame({'season': ['winter', 'summer']}),
        'inflow': pl.DataFrame({'snapshot': snapshots, 'value': [0.0, 10.0, 0.0, 0.0, 0.0, 6.0, 0.0]}),
        'price': pl.DataFrame({'snapshot': snapshots, 'value': [1.0, 2.0, 5.0, 3.0, 4.0, 1.0, 2.0]}),
    }


def test_the_seasons_page_number():
    """The optimum `docs/examples/seasons.md` quotes, and the trajectory behind it.

    Summer is what the numbers are for: it opens holding 6 and sells that at the
    price-4 snapshot *before* its own inflow arrives, which only a cycle closed
    per season allows — the 6 is its own, returned by snapshot 7.
    """
    with differential(SEASONS, _seasons_sources()) as run:
        assert run.oracle == pytest.approx(74.0, rel=RTOL), 'winter 50 and summer 24, agreed by both lanes'
        released = {int(r['snapshot']): r['value'] for r in run.result.primal('release').to_dicts()}
        held = {int(r['snapshot']): r['value'] for r in run.result.primal('soc').to_dicts()}

    assert released[3] == pytest.approx(10.0), "winter's inflow leaves at winter's own best price"
    assert held[4] == pytest.approx(0.0), 'and winter closes where it opened, handing summer nothing'
    assert released[5] == pytest.approx(6.0), 'summer sells a snapshot before its inflow arrives'
    assert held[7] == pytest.approx(6.0), 'and closes holding what it opened with, three snapshots later'

    assert '74.0' in SEASONS_PAGE.read_text(), 'the page quotes an optimum this test does not hold'


#: One balance row whose edge the case under test swaps in. `release` carries an
#: upper bound because a dropped row leaves that snapshot's release in no
#: constraint at all, and an unbounded lane has no optimum to agree on.
PARTITIONED = """
dimensions:
  snapshot: {dtype: int}
  season: {dtype: str}

lookups:
  season_of: {over: snapshot, into: season}

parameters:
  inflow: {dims: [snapshot]}
  price: {dims: [snapshot]}

variables:
  soc: {foreach: [snapshot], bounds: {lower: 0, upper: 60}}
  release: {foreach: [snapshot], bounds: {lower: 0, upper: 100}}

constraints:
  season_balance:
    foreach: [snapshot]
    expression: soc == shift(soc, over=snapshot, offset=1, EDGE) + inflow - release

objective:
  sense: maximize
  expression: sum(release * price, over=snapshot)
"""


def _partitioned(edge: str) -> str:
    return PARTITIONED.replace('EDGE', edge)


def test_a_bare_partitioned_shift_vacates_each_group_s_first():
    """Two rows drop, one per season — not one for the horizon.

    Bare, the vacated position is absent and takes its row with it, and with
    `by=` the position vacated is each *season's* first rather than the axis's.
    """
    model = _partitioned('by=season_of')
    with differential(model, _seasons_sources()) as run:
        assert run.result.is_ok, 'both lanes reach the same answer with two rows missing from it'
    with lps.build(pyyaml.safe_load(model), _seasons_sources()) as built:
        omitted = {r['constraint']: r['rows_not_built'] for r in built.diagnostics().omissions.to_dicts()}
    assert omitted['season_balance'] == 2, 'one row per season, not one for the horizon'


def test_a_filled_partitioned_edge_builds_every_row():
    """`edge=0` per group: the row survives and its first snapshot starts empty."""
    model = _partitioned('edge=0, by=season_of')
    with differential(model, _seasons_sources()) as run:
        held = {int(r['snapshot']): r['value'] for r in run.result.primal('soc').to_dicts()}
    with lps.build(pyyaml.safe_load(model), _seasons_sources()) as built:
        assert built.diagnostics().omissions.is_empty(), 'a filled edge builds every row'
    assert held[5] == pytest.approx(0.0), "summer's first snapshot starts from the 0 its own edge was filled with"


def test_the_axis_wrap_is_a_different_model():
    """The same balance wrapped over the horizon leaks across the boundary.

    This is what `by=` is for, and it is not a convenience: without it winter
    opens on summer's closing level, which is feasible, higher, and wrong.
    """
    with differential(_partitioned("edge='wrap'"), _seasons_sources()) as run:
        assert run.oracle == pytest.approx(80.0, rel=RTOL), 'the horizon as one cycle, worth 6 more to winter'
        held = {int(r['snapshot']): r['value'] for r in run.result.primal('soc').to_dicts()}
    assert held[1] == pytest.approx(6.0), "winter's first snapshot opens on what summer left"

    with differential(_partitioned("edge='wrap', by=season_of"), _seasons_sources()) as run:
        assert run.oracle == pytest.approx(74.0, rel=RTOL), 'each season closed on itself, and 6 poorer for it'


def test_coordinates_in_no_group_translate_from_nothing():
    """Snapshots the lookup sends nowhere are in no group, so they reach nothing.

    **Two** of them, which is the case that separates "in no group" from "in a
    group of its own": a lane that let the nulls fall together would give the
    second a predecessor — the first — and write a balance row about a season
    that does not exist. Under a wrap it would close them onto each other.
    """
    sources = _seasons_sources()
    snapshots = [1, 2, 3, 4, 5, 6, 7, 98, 99]
    sources['snapshot'] = pl.DataFrame(
        {'snapshot': snapshots, 'season_of': ['winter'] * 4 + ['summer'] * 3 + [None, None]}
    )
    for name in ('inflow', 'price'):
        sources[name] = pl.concat([sources[name], pl.DataFrame({'snapshot': [98, 99], 'value': [1.0, 1.0]})])

    for edge in ("edge='wrap', by=season_of", 'by=season_of'):
        model = _partitioned(edge)
        with differential(model, sources) as run:
            held = {int(r['snapshot']): r['value'] for r in run.result.primal('soc').to_dicts()}
        with lps.build(pyyaml.safe_load(model), sources) as built:
            omitted = {r['constraint']: r['rows_not_built'] for r in built.diagnostics().omissions.to_dicts()}

        expected = 2 if edge.startswith('edge=') else 4  # bare drops each season's first as well
        assert omitted['season_balance'] == expected, f'{edge}: the two group-less snapshots build no row'
        assert held[98] == pytest.approx(0.0), f'{edge}: and their levels sit in no balance at all'
        assert held[99] == pytest.approx(0.0), f'{edge}: neither reads the other'


def test_a_lookup_over_another_dimension_cannot_partition_a_translation():
    """`by=` groups the axis being walked, or no coordinate has a neighbour in one."""
    model = (
        _partitioned("edge='wrap', by=plant_season")
        .replace(
            'lookups:\n  season_of:',
            'lookups:\n  plant_season: {over: plant, into: season}\n  season_of:',
        )
        .replace('  season: {dtype: str}', '  season: {dtype: str}\n  plant: {dtype: str}')
    )
    with pytest.raises(LpspecError, match=r"walks 'snapshot' but groups by a lookup over 'plant'"):
        schema_of(model)
