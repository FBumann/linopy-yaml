"""``cases:`` on a named expression — one quantity whose value varies by region.

Three regimes that multiply into eight constraint rows add into three cases,
and the inequality using them is written once. The declaration is checked to
**partition** its frame before any data binds (``math_spec.partition``), and
that is what lets both lanes treat the arms as a value rather than a choice:
exactly one applies at every coordinate.

Which is also the hazard. The relational lane sums the arms' fragments, so a
partition it did not have would be a silent *total* rather than a selection —
`test_two_arms_never_reach_one_coordinate` is the guard, and it is written
against the arithmetic rather than against the checker that makes it safe.

The eager lane zeroes each arm outside its own region and adds; the relational
lane cuts each arm's rows to its region and lets the terminal ``sum(coeff)``
add them. Two different mechanisms for one meaning, which is why every case
below is differential.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import yaml as pyyaml

import lpspec as lps
from lpspec.errors import LaneError, LpspecError
from tests.conftest import schema_of
from tests.differential import RTOL, differential
from tests.oracle import lpspec_linopy

MODEL = """
description: output capped by the commitment state a unit carries into a snapshot

dimensions:
  snapshot: {dtype: int, description: dispatch periods in order}
  generator: {dtype: str, description: units}

parameters:
  committable: {dims: [generator], dtype: bool, description: whether the unit commits}
  status_initial: {dims: [generator], description: the status carried in from before the horizon}
  cap: {dims: [generator], description: capacity}
  price: {dims: [snapshot], description: what a unit of output earns}

variables:
  out:
    foreach: [snapshot, generator]
    bounds: {lower: 0, upper: 100}
    description: energy released in a period

expressions:
  previous_status:
    description: the commitment state a unit carries into a snapshot
    foreach: [snapshot, generator]
    cases:
      always_on:
        when: "not committable"
        expression: 1
        description: a unit that never commits is always on
      boundary:
        when: "committable and position(snapshot) == 0"
        expression: status_initial
        description: the first snapshot reads what was carried in
      interior:
        when: "committable and position(snapshot) > 0"
        expression: 0.5
        description: every later snapshot is half-committed

constraints:
  cap_by_status:
    foreach: [snapshot, generator]
    expression: out <= cap * previous_status
    description: output is capped by the carried-in status

objective:
  sense: maximize
  expression: sum(out * price)
  description: revenue from what is released
"""

#: Deliberately not starting at zero, so `position(snapshot) == 0` is the only
#: thing that names the boundary — see test_position_of_dim.py.
SNAPSHOTS = [4, 5, 6]
GENERATORS = ['g1', 'g2']


def _sources(**overrides: pl.DataFrame) -> dict[str, pl.DataFrame]:
    sources = {
        'snapshot': pl.DataFrame({'snapshot': SNAPSHOTS}),
        'generator': pl.DataFrame({'generator': GENERATORS}),
        'committable': pl.DataFrame({'generator': GENERATORS, 'value': [False, True]}),
        'status_initial': pl.DataFrame({'generator': GENERATORS, 'value': [1.0, 1.0]}),
        'cap': pl.DataFrame({'generator': GENERATORS, 'value': [10.0, 20.0]}),
        'price': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [1.0, 2.0, 3.0]}),
    }
    return sources | overrides


#: `out` is capped at `cap * previous_status` and the price is positive, so each
#: bound is met exactly: g1 is never committable and reads 1 everywhere; g2 is,
#: so its first snapshot reads `status_initial` and its later ones 0.5.
EXPECTED_CAP = {
    (4, 'g1'): 10.0,
    (5, 'g1'): 10.0,
    (6, 'g1'): 10.0,
    (4, 'g2'): 20.0,
    (5, 'g2'): 10.0,
    (6, 'g2'): 10.0,
}
EXPECTED_OBJECTIVE = sum(
    EXPECTED_CAP[(t, g)] * p for t, p in zip(SNAPSHOTS, [1.0, 2.0, 3.0], strict=True) for g in GENERATORS
)


def test_a_cased_expression_reaches_a_constraint_and_both_lanes_agree():
    """Three regimes, one inequality — and the same answer either way built."""
    independent = EXPECTED_OBJECTIVE
    assert independent == pytest.approx(130.0), 'the bound is met exactly in every row, so the total is arithmetic'

    with differential(MODEL, _sources(), lp=True) as run:
        assert run.oracle == pytest.approx(independent, rel=RTOL), (
            'both lanes and the LP file reach the hand-computed total, so the cases select rather than sum'
        )


def test_each_arm_supplies_the_coordinates_its_when_claims():
    """The value read back is one arm's per coordinate, not a blend of three."""
    with lps.solve(pyyaml.safe_load(MODEL), _sources()) as result:
        got = result.expression('previous_status').sort('snapshot', 'generator')
        coordinates = zip(got['snapshot'], got['generator'], strict=True)
        assert dict(zip(coordinates, got['value'], strict=True)) == {
            (4, 'g1'): 1.0,
            (5, 'g1'): 1.0,
            (6, 'g1'): 1.0,
            (4, 'g2'): 1.0,
            (5, 'g2'): 0.5,
            (6, 'g2'): 0.5,
        }, 'always_on for g1, boundary at the first snapshot for g2, interior after it'


def test_both_lanes_read_the_same_cased_value(tmp_path):
    """A cased expression is read through its *reference*, so neither lane assembles the arms itself."""
    sources = _sources()
    with lps.solve(pyyaml.safe_load(MODEL), sources) as result:
        streaming = result.expression('previous_status').sort('generator', 'snapshot')['value'].to_numpy()

    path = tmp_path / 'model.yaml'
    path.write_text(MODEL)
    built = lpspec_linopy.build(path, sources)
    built.solve(solver_name='highs', output_flag=False)
    eager = lpspec_linopy.expression(built, path, 'previous_status', sources)
    assert eager.transpose('generator', 'snapshot').to_numpy().ravel() == pytest.approx(streaming, rel=RTOL), (
        'the eager lane zeroes each arm outside its region and adds; the streaming one cuts rows — same six values'
    )


def test_two_arms_never_reach_one_coordinate():
    """The arms are summed, so overlap would be a total — this is what forbids it.

    `always_on` and `interior` both claim g2's later snapshots if the mask is
    dropped from either, and `previous_status` reads 1.5 rather than 0.5. The
    partition check refuses that at load; this asserts the arithmetic downstream
    of it, which is what the checker is protecting.
    """
    with lps.solve(pyyaml.safe_load(MODEL), _sources()) as result:
        got = result.expression('previous_status').sort('snapshot', 'generator')
        assert got['value'].to_list() == [1.0, 1.0, 1.0, 0.5, 1.0, 0.5], (
            'one arm per coordinate, in (snapshot, generator) order — a sum of two would exceed every entry'
        )


def test_an_arm_narrower_than_the_frame_broadcasts():
    """`always_on` is a scalar and its `when` reads only `generator`.

    It still has to reach every snapshot of a non-committable unit, which is
    the broadcast a parameter with fewer dims already gets. An arm widened only
    to its own dims would seed one snapshot and leave the rest uncapped.
    """
    with lps.solve(pyyaml.safe_load(MODEL), _sources()) as result:
        g1 = result.expression('previous_status').filter(pl.col('generator') == 'g1')
        assert g1['value'].to_list() == [1.0, 1.0, 1.0], 'a scalar arm reaches all three snapshots, not one'


#: The same model with a *second* variable inside one arm, so the case restricts
#: **terms** rather than constants. A distinct variable is the point: two `out`
#: terms on one row would collapse in the terminal `sum(coeff)` and the count
#: below could not tell a restricted arm from an unrestricted one.
VARIABLE_ARM = MODEL.replace(
    """  out:
    foreach: [snapshot, generator]
    bounds: {lower: 0, upper: 100}
    description: energy released in a period
""",
    """  out:
    foreach: [snapshot, generator]
    bounds: {lower: 0, upper: 100}
    description: energy released in a period
  ready:
    foreach: [snapshot, generator]
    bounds: {lower: 0, upper: 1}
    description: how ready a committed unit is in a later snapshot
""",
).replace('expression: 0.5', 'expression: ready')


def test_an_arm_carrying_a_variable_contributes_no_row_outside_itself():
    """A term under a case exists only where its arm applies.

    The constant path fills the rows outside an arm with zero and the term path
    must not: an explicit zero coefficient per non-applying arm is an entry
    handed to the solver for every row it says nothing about. Counted here
    rather than argued — the nonzeros are what a sink receives.
    """
    with differential(VARIABLE_ARM, _sources()) as run:
        assert run.engine.diagnostics().nonzeros == 8, (
            'six rows each carrying their own `out`, plus `ready` on the two rows the '
            'interior arm claims — and nothing for the four rows it does not'
        )


def test_a_value_missing_inside_its_own_arm_is_still_reported():
    """A case does not excuse a gap — it only says which rows have to be covered.

    Outside its arm a constant fragment carries a zero so the coverage check
    can tell "this arm does not apply" from "nobody supplied this row". Inside
    it, a missing row is the ordinary uncovered constant and says so.
    """
    short = pl.DataFrame({'generator': ['g1'], 'value': [1.0]})
    sources = _sources(status_initial=short)
    with pytest.raises(LpspecError, match=r'fewer coordinates than the rows built here'):
        lps.build(pyyaml.safe_load(MODEL), sources)


def test_the_declaration_states_its_frame():
    """`foreach:` is the frame the cases partition, and the dims a reference carries."""
    schema = schema_of(MODEL)
    assert schema.expressions['previous_status'].foreach == ['snapshot', 'generator'], (
        'the declared frame, in declaration order'
    )
    assert sorted(schema.expressions['previous_status'].cases) == ['always_on', 'boundary', 'interior'], (
        'all three arms, keyed by the name each prints under'
    )


def test_a_gap_in_the_cases_is_refused_at_load():
    """An expression is total over its frame or it is refused.

    Dropping `always_on` leaves every non-committable unit with no value, and
    rule 7 would spread that to the constraint referencing it — deleting rows
    the constraint never masked. Decided here with no data bound.
    """
    holed = MODEL.replace(
        """      always_on:
        when: "not committable"
        expression: 1
        description: a unit that never commits is always on
""",
        '',
    )
    with pytest.raises(LpspecError, match=r'do not partition'):
        schema_of(holed)


def test_the_two_forms_do_not_mix():
    """One `expression:` or a set of `cases:`, never both."""
    both = MODEL.replace(
        '    foreach: [snapshot, generator]\n    cases:',
        '    expression: cap\n    foreach: [snapshot, generator]\n    cases:',
    )
    with pytest.raises(LpspecError, match=r'one `expression:` or a set of `cases:`'):
        schema_of(both)


def test_the_objective_takes_a_cased_expression():
    """Nothing about a case is constraint-shaped — the objective reads one too."""
    model = MODEL.replace('expression: sum(out * price)', 'expression: sum(out * price * previous_status)')
    with differential(model, _sources()) as run:
        assert np.isfinite(run.oracle), 'a cased coefficient in the objective is a coefficient like any other'


#: A recurrence stated as cases: the seed reads a parameter, every later
#: snapshot reads the snapshot before. The `carried` arm is a bare `shift`, so
#: it **vacates** the first position — and the arm that owns that position is
#: the other one. This is `pypsa_mixed_cycling` in miniature.
RECURRENCE = """
description: a level carried across a horizon, opened by cases

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

expressions:
  opening_level:
    description: the level the store opens a snapshot with
    foreach: [snapshot]
    cases:
      seeded:
        when: "position(snapshot) == 0"
        expression: soc_initial
      carried:
        when: "position(snapshot) > 0"
        expression: shift(soc, over=snapshot, offset=1)

constraints:
  balance:
    foreach: [snapshot]
    expression: soc == opening_level + inflow - out
    description: what is held is what was opened with, plus inflow, less what was released

objective:
  sense: maximize
  expression: sum(out * price)
  description: revenue from what is released
"""


def _recurrence_sources():
    return {
        'snapshot': pl.DataFrame({'snapshot': SNAPSHOTS}),
        'soc_initial': pl.DataFrame({'value': [10.0]}),
        'inflow': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [1.0, 2.0, 3.0]}),
        'price': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [1.0, 2.0, 3.0]}),
    }


def test_an_arm_that_vacates_a_position_does_not_delete_the_arm_that_owns_it():
    """A shift's vacated edge is absence *within its own arm*, and nowhere else.

    Absence propagates into a comparison and deletes the row, and the engine
    collects that across every term of a row at once. So the `carried` arm's
    vacated first position — which its own `when` already excludes — would
    delete the seeded row that defines that very coordinate, and the recurrence
    would go unanchored with nothing to say so.

    Three snapshots, so all three rows must be built: the seed plus two
    carries. `pypsa_mixed_cycling` is the corpus case, where it showed as an
    objective of 3200 against PyPSA's 4800.
    """
    with differential(RECURRENCE, _recurrence_sources()) as run:
        assert run.engine.diagnostics().rows == 3, (
            'one row per snapshot — the seeded row at position 0 included, which the '
            "carried arm's vacated edge must not delete"
        )
        assert run.oracle == pytest.approx(48.0, rel=RTOL), (
            'the store holds everything back for the dearest snapshot, so the seed of 10 and the '
            'three inflows leave together at price 3 — and the seed only counts while its row is built'
        )


# ---------------------------------------------------------------------------
# what the review found: an arm is narrower than its frame, and the lanes have
# to agree about that in every position a cased value can stand in
# ---------------------------------------------------------------------------


def test_a_cased_value_reduced_by_sum_agrees_across_lanes():
    """`sum(over=)` needs the dim *present*, not broadcast at the row join.

    An arm compiled only as wide as its own mask leaves a fragment with no
    slots for the operator to act on, and this lane refused a file the eager
    one built. A cased expression's dims are its declared `foreach`, so the
    fragments carry that whatever any one arm names.
    """
    model = MODEL.replace(
        """  cap_by_status:
    foreach: [snapshot, generator]
    expression: out <= cap * previous_status""",
        """  cap_by_status:
    foreach: [generator]
    expression: sum(out, over=snapshot) <= sum(cap * previous_status, over=snapshot)""",
    )
    with differential(model, _sources()) as run:
        assert run.oracle == pytest.approx(210.0, rel=RTOL), (
            'g1 is capped at 10 in all three snapshots and g2 at 20 + 10 + 10, and the whole '
            'horizon is sold at the dearest price the cap allows'
        )


def test_the_read_back_carries_the_declared_frame():
    """Every arm here names only `generator`, and the value is still over both dims.

    The declaration says `foreach: [snapshot, generator]`, so that is the
    quantity's shape — a read that came back keyed by `generator` alone would
    be answering about a different quantity, and both lanes would agree on it.
    """
    narrow = MODEL.replace(
        """      boundary:
        when: "committable and position(snapshot) == 0"
        expression: status_initial
        description: the first snapshot reads what was carried in
      interior:
        when: "committable and position(snapshot) > 0"
        expression: 0.5
        description: every later snapshot is half-committed""",
        """      commits:
        when: "committable"
        expression: status_initial
        description: a committable unit reads what it was handed""",
    )
    with lps.solve(pyyaml.safe_load(narrow), _sources()) as result:
        got = result.expression('previous_status')
        assert got.columns == ['snapshot', 'generator', 'value'], 'the declared foreach, in declaration order'
        assert got.height == 6, 'three snapshots times two generators, not two rows keyed by generator'


def test_a_cased_divisor_is_refused_by_this_lane_with_its_rewrite():
    """Each arm is its own frame, and a quotient inverts one.

    Not a language error: the file is inside the language and the eager lane
    evaluates it, so the message says which lane is short and how to say the
    same model here.
    """
    model = MODEL.replace('expression: out <= cap * previous_status', 'expression: out <= cap / previous_status')
    with pytest.raises(LaneError, match=r'a divisor is a cased expression, and this lane cannot build that'):
        lps.build(pyyaml.safe_load(model), _sources())


def test_a_parameter_one_arm_reads_is_asked_only_where_that_arm_applies():
    """`status_initial` is the boundary arm's, so g1 never needs a row.

    Both lanes have to say so. The eager lane asked the whole constraint frame
    and refused a file the streaming lane built — a dialect split, and the
    differential tests are an oracle only while there is none.
    """
    only_g2 = pl.DataFrame({'generator': ['g2'], 'value': [1.0]})
    with differential(MODEL, _sources(status_initial=only_g2)) as run:
        assert run.oracle == pytest.approx(EXPECTED_OBJECTIVE, rel=RTOL), (
            'the unread row changes nothing: g1 is not committable, so no arm of its reads status_initial'
        )


def test_a_gap_inside_an_arm_is_still_refused_on_both_lanes(tmp_path):
    """Narrowing the question must not stop it being asked.

    g2 *is* committable, so the boundary arm reads its `status_initial` — a
    missing row there is the ordinary uncovered constant, on either lane.
    """
    only_g1 = pl.DataFrame({'generator': ['g1'], 'value': [1.0]})
    sources = _sources(status_initial=only_g1)
    with pytest.raises(LpspecError, match=r'fewer coordinates than the rows built here'):
        lps.build(pyyaml.safe_load(MODEL), sources)

    path = tmp_path / 'model.yaml'
    path.write_text(MODEL)
    with pytest.raises(LpspecError, match=r'fewer coordinates than the rows built here'):
        lpspec_linopy.build(path, sources)
