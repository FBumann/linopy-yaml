"""shift: time-coupled recurrences through both backends.

examples/storage.yaml is dispatch plus a cyclic battery:
soc == shift(soc, over=snapshot, by=1, edge='wrap") + charge * 0.9 - discharge. The eager backend
The eager backend implements `edge='wrap'` with linopy"s circular .roll(); the
relational backend lowers it to plan.Translate — a pointwise ord-join remap.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import lpspec as lps
from lpspec.errors import LanguageError
from lpspec.lowering import _lower_expr
from lpspec.relational.plan import (
    Translate,
    Variable,
)
from tests.conftest import DISPATCH_MODEL, EXAMPLES_DIR, by_coord, override, resolved, schema_of
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
    assert "shift(soc, over=snapshot, by=1, edge='wrap')" in original
    acyclic = original.replace("shift(soc, over=snapshot, by=1, edge='wrap')", 'shift(soc, over=snapshot, by=1)')

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
    assert "shift(soc, over=snapshot, by=1, edge='wrap')" in original
    forward = original.replace("shift(soc, over=snapshot, by=1, edge='wrap')", 'shift(soc, over=snapshot, by=-1)')

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
        "shift(soc, over=snapshot, by=1, edge='wrap')", 'shift(soc, over=snapshot, by=-1, edge=0)'
    )

    with differential(filled, data) as run:
        soc, charge, discharge = _soc_trace(run.result)
        assert run.model.constraints['soc_balance'].labels.values[-1] != -1, (
            'edge=0 asks for a value at the boundary, so the last row is built rather than dropped'
        )
        assert np.allclose(soc[-1], 0.9 * charge[-1] - discharge[-1], atol=1e-6), (
            'at the last snapshot the vacated successor contributes zero, not a wraparound'
        )


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
            'expression': 'p - shift(p, over=snapshot, by=1) <= ramp_max',
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
            "shift(soc, over=snapshot, by=1, edge='wrap')",
            Translate(Variable('soc'), 'snapshot', 1),
            id='wrap',
        ),
        pytest.param(
            "shift(soc, over=snapshot, by=-2, edge='wrap')",
            Translate(Variable('soc'), 'snapshot', -2),
            id='wrap-backwards',
        ),
        pytest.param(
            'shift(soc, over=snapshot, by=1)',
            Translate(Variable('soc'), 'snapshot', 1, wrap=False),
            id='bare',
        ),
        # fill is the field both lanes branch on: None is absence, 0.0 the zero.
        pytest.param(
            'shift(soc, over=snapshot, by=1, edge=0)',
            Translate(Variable('soc'), 'snapshot', 1, wrap=False, fill=0.0),
            id='zero-fill',
        ),
    ],
)
def test_translation_lowers_to_a_bounded_halo(expression, expected):
    assert _lower_expr(resolved(expression, STORAGE_SCHEMA), STORAGE_SCHEMA, 't') == expected


@pytest.mark.parametrize(
    ('expression', 'match'),
    [
        pytest.param(
            "shift(soc, over=nope, by=1, edge='wrap')",
            r'shift\(over=nope\) does not name a declared dimension',
            id='over-names-no-dimension',
        ),
        pytest.param(
            "shift(load, over=generator, by=1, edge='wrap')",
            'but the expression has dims',
            id='a-dim-the-expression-lacks',
        ),
        # `edge=` is a closed keyword: one keyword carries all three policies, so
        # "cyclic, and also fill" has no spelling left to be refused.
        pytest.param(
            'shift(soc, over=snapshot, by=1, edge=nonsense)',
            'is not an edge policy',
            id='the-edge-keyword-is-closed',
        ),
        # Over a variable only `edge=0` is sayable — a nonzero fill would put a
        # constant where a term was.
        pytest.param(
            'shift(soc, over=snapshot, by=1, edge=1)',
            'only fill=0 is representable there',
            id='a-nonzero-fill-over-a-variable',
        ),
    ],
)
def test_a_shift_neither_lane_can_honour_is_refused_at_lowering(expression, match):
    with pytest.raises(LanguageError, match=match):
        _lower_expr(resolved(expression, STORAGE_SCHEMA), STORAGE_SCHEMA, 't')


FILL_IDENTITY_MODEL = """
dimensions: {t: {dtype: int, values: [0, 1, 2]}}
parameters:
  eff: {dims: [t]}
variables:
  x: {foreach: [t], bounds: {lower: 0, upper: 100}}
constraints:
  c:
    foreach: [t]
    expression: "x * shift(eff, over=t, by=1, edge=1) <= 10"
objective: {sense: maximize, expression: "sum(x, over=t)"}
"""


def test_the_fill_a_product_wants_is_one_not_zero():
    """``fill=`` takes the identity of the *position*, which is why it takes a number.

    linopy v1 refuses to fill on the caller's behalf precisely because the right
    value is positional (``convention.rst`` §7): 0 is the identity of a sum, 1 of
    a product. ``x * shift(eff, over=t, by=1, edge=0)`` would force ``x`` to zero at the
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
    'objective': {'sense': 'maximize', 'expression': 'x * c'},
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
    lps.check(_with(f'x - shift(x, over=t, by=1, {edge}) <= 1'))


def test_a_bare_wrap_names_a_dimension_and_is_refused():
    """`over=wrap, edge=wrap` was legal, and the same token meant two things.

    The model here declares a dimension actually called `wrap`, which is what
    makes the ambiguity concrete rather than theoretical: the parser resolved
    the two positions differently and a reader could not.
    """
    with pytest.raises(ValueError) as exc:
        lps.check(_with('x - shift(x, over=t, by=1, edge=wrap) <= 1'))

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
    shift = f'shift(dt, over=t, by=1, edge={edge})' if edge else 'shift(dt, over=t, by=1)'
    constraint: dict[str, object] = {'foreach': ['t'], 'expression': f'x <= {shift}'}
    if where is not None:
        constraint['where'] = where
    return {
        'dimensions': {'t': {'dtype': 'int', 'values': [0, 1, 2]}},
        'parameters': {'dt': {'dims': ['t']}},
        'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 5}}},
        'constraints': {'c': constraint},
        'objective': {'sense': 'maximize', 'expression': 'x'},
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
    'same-dim': 'shift(shift(p, over=t, by=1), over=t, by=1)',
    'cross-dim': 'shift(shift(p, over=t, by=1), over=g, by=1)',
    'cross-dim-reversed': 'shift(shift(p, over=g, by=1), over=t, by=1)',
    'triple-mixed': 'shift(shift(shift(p, over=t, by=1), over=g, by=1), over=t, by=1)',
    'inner-fill': 'shift(shift(p, over=t, by=1, edge=0), over=t, by=1)',
    'outer-wrap': "shift(shift(p, over=t, by=1), over=t, by=1, edge='wrap')",
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
        'objective': {'sense': 'maximize', 'expression': 'p * c'},
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
                'expression': f'shift(order, over=t, by=lead, edge={edge}) >= demand',
            }
        },
        'objective': {'sense': 'minimize', 'expression': 'order * c'},
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
        'constraints': {'k': {'foreach': ['g', 't'], 'expression': f"x >= shift(x, over=t, by={offset}, edge='wrap')"}},
        'objective': {'sense': 'minimize', 'expression': 'x * 1.0'},
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
        'constraints': {'k': {'foreach': ['g', 't'], 'expression': 'x >= shift(x, over=t, by=lead)'}},
        'objective': {'sense': 'minimize', 'expression': 'x * 1.0'},
    }
    with pytest.raises(LanguageError, match='vacated positions absent'):
        lps.check(model)


def test_a_named_offset_carries_its_sign_in_the_data():
    """`by=-lead` is refused rather than negated, so one spelling means one thing."""
    model = {
        'dimensions': {'g': {'dtype': 'str', 'values': ['a']}, 't': {'dtype': 'int', 'values': [0, 1]}},
        'parameters': {'lead': {'dims': ['g'], 'dtype': 'int'}},
        'variables': {'x': {'foreach': ['g', 't'], 'bounds': {'lower': 0, 'upper': 1}}},
        'constraints': {'k': {'foreach': ['g', 't'], 'expression': "x >= shift(x, over=t, by=-lead, edge='wrap')"}},
        'objective': {'sense': 'minimize', 'expression': 'x * 1.0'},
    }
    with pytest.raises(LanguageError, match='negates a named offset'):
        lps.check(model)
