"""sum_back: a trailing window whose width is data, through both backends.

A minimum up time, a rolling emissions budget, a delivery horizon — the width
is a column in the source data, and the alternative to saying so is either a
run of hand-written shifts (which fixes the width in the model text) or an
incidence table over the dimension twice (which is what the GenX port built
before this existed).
"""

from __future__ import annotations

from copy import deepcopy

import polars as pl
import pytest

import lpspec as lps
from lpspec.errors import DimensionError, LanguageError
from lpspec.lowering import _lower_expr
from lpspec.relational.plan import Window
from tests.conftest import resolved, schema_of
from tests.differential import differential
from tests.oracle import pd

MIN_UP = {'slow': 3, 'fast': 1}
UNITS, PERIODS = list(MIN_UP), [0, 1, 2, 3, 4]


def up_time_model(edge: str | None) -> dict:
    """A unit that starts must stay on for its own minimum up time.

    The start is forced at the *last* period, which is what makes the edge
    visible: acyclically the window reaches nothing beyond the axis, so only
    that period is held on, while under ``wrap`` it reaches around into the
    first two.
    """
    window = 'sum_back(started, over=t, within=min_up' + (f", edge='{edge}'" if edge else '') + ')'
    return {
        'dimensions': {'g': {'dtype': 'str', 'values': UNITS}, 't': {'dtype': 'int', 'values': PERIODS}},
        'parameters': {
            'min_up': {'dims': ['g'], 'dtype': 'int'},
            'must_start': {'dims': ['g', 't']},
            'cost': {'dims': ['g']},
        },
        'variables': {
            'started': {'foreach': ['g', 't'], 'domain': 'binary'},
            'on': {'foreach': ['g', 't'], 'domain': 'binary'},
        },
        'constraints': {
            'starts_when_told': {'foreach': ['g', 't'], 'expression': 'started >= must_start'},
            'stays_up_its_own_time': {'foreach': ['g', 't'], 'expression': f'{window} <= on'},
        },
        'objective': {'sense': 'minimize', 'expression': 'sum(on * cost)'},
    }


UP_TIME_DATA = {
    'min_up': pd.Series([MIN_UP[u] for u in UNITS], index=pd.Index(UNITS, name='g')),
    'cost': pd.Series([1.0, 1.0], index=pd.Index(UNITS, name='g')),
    'must_start': pd.DataFrame(
        {
            'g': [u for u in UNITS for _ in PERIODS],
            't': PERIODS * len(UNITS),
            'value': [0.0, 0.0, 0.0, 0.0, 1.0] * len(UNITS),
        }
    ),
}


@pytest.mark.parametrize(
    ('edge', 'expected'),
    [
        pytest.param(None, {('slow', 4), ('fast', 4)}, id='acyclic'),
        pytest.param('wrap', {('slow', 4), ('slow', 0), ('slow', 1), ('fast', 4)}, id='wrap'),
    ],
)
def test_a_window_width_may_differ_per_entity(edge: str | None, expected: set):
    """`within=` names a parameter: each entity gets a window of its own length.

    The instance discriminates twice over. The two units carry different
    widths, so a width read once for both would hold the wrong one on; and the
    start sits at the last period, so an edge read wrongly would either lose
    the wrapped periods or invent them.
    """
    with differential(up_time_model(edge), UP_TIME_DATA) as run:
        assert run.result.objective == pytest.approx(float(len(expected)))
        held = run.result.primal('on').filter(pl.col('value') > 0.5)
        assert set(zip(held['g'].to_list(), held['t'].to_list(), strict=True)) == expected, (
            'each unit is held on for its own window, reaching the edge the way the model asked'
        )


def test_a_literal_width_is_the_last_n_positions():
    """A number where the width is the same everywhere, and 1 is the operand."""
    model = up_time_model(None)
    model['constraints']['stays_up_its_own_time']['expression'] = 'sum_back(started, over=t, within=2) <= on'
    with differential(model, UP_TIME_DATA) as run:
        held = run.result.primal('on').filter(pl.col('value') > 0.5)
        assert set(zip(held['g'].to_list(), held['t'].to_list(), strict=True)) == {('slow', 4), ('fast', 4)}, (
            'the window reaches back two positions, and only the last one is on the axis'
        )


#: A window over an operand that is masked away at one interior position. The
#: width decides what the mask means: a wider window reaches it *and* live
#: positions, a width of 1 reaches nothing else at all (#1059, #1060).
MASKED_WINDOW = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'usable': {'dims': ['t']}},
    'variables': {
        'level': {'foreach': ['t'], 'where': 'usable > 0', 'bounds': {'lower': 0, 'upper': 10}},
        'take': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {'held': {'foreach': ['t'], 'expression': 'take <= sum_back(level, over=t, within=1)'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(take, over=t) - 1000 * sum(level, over=t)'},
}

MASKED_WINDOW_SOURCES = {
    't': pd.Index([0, 1, 2, 3], name='t'),
    'usable': pd.Series([1.0, 0.0, 1.0, 1.0], index=pd.Index([0, 1, 2, 3], name='t')),
}


def masked_window_model(window: str) -> dict:
    """`MASKED_WINDOW` with the window respelled, the rest of it fixed."""
    model = deepcopy(MASKED_WINDOW)
    model['constraints']['held']['expression'] = f'take <= sum_back(level, over=t, {window})'
    return model


def test_a_window_that_reaches_nothing_builds_no_row():
    """A window is short, not empty — until every position it spans is masked.

    Width 1 at the masked snapshot: the window is exactly the slot that is not
    there, so nothing live is left to sum and the row is a statement about
    constants alone. It is not built, and `take[1]` — capped by no row —
    reaches its own bound, which is what tells the two apart from the objective
    rather than only from the row count (#1059).
    """
    with differential(MASKED_WINDOW, MASKED_WINDOW_SOURCES, lp=True) as run:
        assert run.engine.diagnostics().rows == 3, 'the snapshot whose whole window is masked away holds no row'
        assert run.oracle == pytest.approx(10.0), (
            'take[1] is capped by nothing — 0.0 would mean a row asserting take <= 0 was built there'
        )


@pytest.mark.parametrize('window', ['within=2', "within=2, edge='wrap'"], ids=['acyclic', 'wrap'])
def test_a_masked_slot_the_window_reaches_is_a_zero_not_an_absence(window: str):
    """The masked slot contributes nothing and takes nothing with it.

    Two positions wide, so every snapshot's window reaches at least one live
    slot even where it also reaches the masked one. Absence stops at a
    reduction, so all four rows are asserted and every `take` is capped —
    losing a row would show up as an uncapped `take` worth 10 (#1059, #1060).
    """
    with differential(masked_window_model(window), MASKED_WINDOW_SOURCES, lp=True) as run:
        assert run.engine.diagnostics().rows == 4, 'a window reaching one live slot is a row, masked neighbour or not'
        assert run.oracle == pytest.approx(0.0), 'every take is capped, so raising one costs more than it earns'


#: The same question asked of a width that differs per entity: one unit's
#: window is the masked slot alone, the other's reaches past it.
PER_ENTITY_WINDOW = {
    'dimensions': {'g': {'dtype': 'str'}, 't': {'dtype': 'int'}},
    'parameters': {'width': {'dims': ['g'], 'dtype': 'int'}, 'usable': {'dims': ['t']}},
    'variables': {
        'level': {'foreach': ['g', 't'], 'where': 'usable > 0', 'bounds': {'lower': 0, 'upper': 10}},
        'take': {'foreach': ['g', 't'], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {'held': {'foreach': ['g', 't'], 'expression': 'take <= sum_back(level, over=t, within=width)'}},
    'objective': {
        'sense': 'maximize',
        'expression': 'sum(sum(take, over=g), over=t) - 1000 * sum(sum(level, over=g), over=t)',
    },
}


def test_a_per_entity_window_reaching_nothing_is_that_entitys_row_alone():
    """Whose window reached something is asked per entity, not per position.

    `narrow` spans one snapshot and `wide` two, and the masked snapshot is the
    same one for both. The row `narrow` holds there reaches nothing and is not
    built; the one `wide` holds reaches the snapshot before it and is. A width
    read only as a coefficient — zeroing the terms outside it but still
    counting them as reached — would keep `narrow`'s row asserting `take <= 0`
    (#1059).
    """
    sources = {
        'g': pd.Index(['narrow', 'wide'], name='g'),
        't': pd.Index([0, 1, 2, 3], name='t'),
        'width': pd.Series([1, 2], index=pd.Index(['narrow', 'wide'], name='g')),
        'usable': pd.Series([1.0, 0.0, 1.0, 1.0], index=pd.Index([0, 1, 2, 3], name='t')),
    }
    with differential(PER_ENTITY_WINDOW, sources, lp=True) as run:
        assert run.engine.diagnostics().rows == 7, "every row but narrow's at the masked snapshot"
        assert run.oracle == pytest.approx(10.0), 'take[narrow, 1] is capped by nothing, and no other take is free'


WIDTH_REFUSALS = {
    'not-an-integer': ('rate', 'needs an integer parameter'),
    'along-the-summed-dim': ('drift', 'a different window at every position'),
}


@pytest.mark.parametrize(('name', 'expected'), list(WIDTH_REFUSALS.values()), ids=list(WIDTH_REFUSALS))
def test_a_named_width_that_cannot_mean_a_window_is_refused(name: str, expected: str):
    """A width counts positions along one axis and is constant along the other."""
    model = up_time_model(None)
    model['parameters']['rate'] = {'dims': ['g']}
    model['parameters']['drift'] = {'dims': ['g', 't'], 'dtype': 'int'}
    model['constraints']['stays_up_its_own_time']['expression'] = f'sum_back(started, over=t, within={name}) <= on'
    with pytest.raises(LanguageError, match=expected):
        lps.check(model)


@pytest.mark.parametrize('width', ['0', '1.5', '-2'], ids=['zero', 'fractional', 'negative'])
def test_a_literal_width_is_a_whole_number_of_positions(width: str):
    """Below one position there is no window, and no reading that says so."""
    model = up_time_model(None)
    model['constraints']['stays_up_its_own_time']['expression'] = f'sum_back(started, over=t, within={width}) <= on'
    with pytest.raises(LanguageError, match='whole number of positions of at least 1'):
        lps.check(model)


def test_a_window_refuses_a_numeric_edge():
    """A window has no vacated slot to fill — a short window is simply short."""
    model = up_time_model(None)
    model['constraints']['stays_up_its_own_time']['expression'] = (
        'sum_back(started, over=t, within=min_up, edge=0) <= on'
    )
    with pytest.raises(LanguageError, match="takes 'wrap' or nothing"):
        lps.check(model)


def test_a_window_needs_the_dimension_it_sums_over():
    """Summing back along an axis the operand does not carry says nothing."""
    model = up_time_model(None)
    model['constraints']['stays_up_its_own_time']['expression'] = (
        'sum_back(sum(started, over=t), over=t, within=min_up) <= on'
    )
    with pytest.raises(DimensionError, match='sum_back\\(over=t\\)'):
        lps.check(model)


def test_the_window_lowers_to_one_node():
    """One node, not a sum of translations.

    The number of terms a window adds is read from data, so lowering it into
    that many ``Translate``s would make the plan's *shape* depend on data —
    the line the ceiling draws. What data supplies here is how many rows the
    one node's mask has.
    """
    schema = schema_of(up_time_model(None))
    plan = _lower_expr(resolved('sum_back(started, over=t, within=min_up)', schema), schema, 'k')
    assert isinstance(plan, Window), 'a window is its own plan node'
    assert plan.width == 'min_up', 'the width travels as the parameter name, resolved at bind'


@pytest.mark.parametrize(
    ('edge', 'difference'),
    [pytest.param(None, 't - t', id='acyclic'), pytest.param('wrap', r't \ominus t', id='wrap')],
)
def test_the_window_typesets_as_the_condition_on_its_index(edge: str | None, difference: str):
    """A window is a sum over a *restricted* domain, and the restriction prints.

    Falling through to the plain summation would render it as a sum over the
    whole dimension — a different equation, and one the reader cannot tell
    from this one.
    """
    from math_spec.typeset import to_latex

    line = next(line for line in to_latex(schema_of(up_time_model(edge))).splitlines() if 'stays' in line)
    assert rf"\sum_{{t' \in \mathcal{{T}} \,:\, 0 \le {difference}' < \mathit{{min\_up}}}}" in line, (
        'the summation names the window rather than the whole dimension'
    )


def test_a_window_at_the_first_position_is_short_not_empty():
    """The row at the start of the axis survives, holding what it can see.

    The lags that reach past the start contribute a zero rather than an
    absence. Absence propagates in the eager lane, so an unreachable lag added
    to a reachable one would annihilate the whole row and leave the first
    positions unconstrained — silently, and only near the edge.
    """
    model = up_time_model(None)
    data = dict(UP_TIME_DATA)
    data['must_start'] = pd.DataFrame(
        {
            'g': [u for u in UNITS for _ in PERIODS],
            't': PERIODS * len(UNITS),
            'value': [1.0, 0.0, 0.0, 0.0, 0.0] * len(UNITS),
        }
    )
    with differential(model, data) as run:
        held = run.result.primal('on').filter(pl.col('value') > 0.5)
        assert set(zip(held['g'].to_list(), held['t'].to_list(), strict=True)) == {
            ('slow', 0),
            ('slow', 1),
            ('slow', 2),
            ('fast', 0),
        }, 'a start at the first position is held for its window, and the row that says so exists'


def test_a_window_wider_than_the_axis_counts_each_position_once():
    """A cyclic window is capped at the dimension, not wrapped around twice.

    Nothing else pins the cap: acyclically the lags past the end contribute
    zeros and the answer survives them, so only the cyclic case can tell a
    capped window from one that reads some positions twice.
    """
    model = up_time_model('wrap')
    data = dict(UP_TIME_DATA)
    data['min_up'] = pd.Series([len(PERIODS) + 2] * len(UNITS), index=pd.Index(UNITS, name='g'))
    with differential(model, data) as run:
        assert run.result.objective == pytest.approx(float(len(UNITS) * len(PERIODS))), (
            'a window at least as wide as the axis holds every position on, once'
        )
