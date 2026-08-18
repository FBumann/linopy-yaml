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

import polars as pl
import pytest
import yaml as pyyaml

import lpspec as lps
from lpspec.errors import DataError, LanguageError
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
    expression: soc == shift(soc, over=snapshot, by=1) + inflow - out
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
