"""`cases:` — one quantity, a value per region, on both lanes.

The regions of a cased expression are disjoint and total before any data binds,
so neither lane ranks them: each builds a region against that region's own mask
and adds the results. What the tests below hold is the three things that
follow, and each of them was wrong in a first cut of this feature:

* a region's value reaches the coordinates it claims, and only those;
* a region's **absence** stays inside it — an ``otherwise`` that shifts with no
  ``edge=`` has nothing at the first position, and must not unmake a row the
  other regions do cover;
* a region's data is required **where that region applies**, so a parameter
  standing for one region is not asked to cover the whole frame, and a hole
  inside the region it does stand for is still refused.

The last two are the pair that can silently disagree between the lanes, which
is why every case here runs through ``differential``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

import lpspec as lps
from lpspec.errors import DataError
from tests.differential import RTOL, differential
from tests.oracle import lpspec_linopy

CAPPED_BY_REGION = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {
        'flag': {'dims': ['t'], 'dtype': 'bool'},
        'hi': {'dims': ['t']},
        'cost': {'dims': ['t']},
    },
    'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 1000}}},
    'expressions': {
        'cap': {
            'foreach': ['t'],
            'cases': {'flagged': {'when': 'flag', 'expression': 'hi'}},
            'otherwise': 5,
        }
    },
    'constraints': {'under_cap': {'foreach': ['t'], 'expression': 'x <= cap'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x * cost)'},
}

#: `flag` holds at 0 and 2, so `hi` carries those and `otherwise`'s 5 the rest.
CAPPED_SOURCES = {
    't': [0, 1, 2, 3],
    'flag': {'t': [0, 1, 2, 3], 'value': [True, False, True, False]},
    'hi': {'t': [0, 2], 'value': [40.0, 60.0]},
    'cost': {'t': [0, 1, 2, 3], 'value': [1.0, 1.0, 1.0, 1.0]},
}


def _frames(sources):
    """A dimension's labels as a Series and every parameter as a frame, the shape both lanes take."""
    import polars as pl

    built = {}
    for name, value in sources.items():
        if not isinstance(value, list):
            built[name] = pl.DataFrame(value)
            continue
        dtype = pl.Int64 if isinstance(value[0], int) else pl.String
        built[name] = pl.Series(name, value, dtype=dtype)
    return built


def test_each_region_carries_the_coordinates_it_claims():
    with differential(CAPPED_BY_REGION, _frames(CAPPED_SOURCES), lp=True) as run:
        assert run.oracle == pytest.approx(110.0, rel=RTOL), 'the flagged steps cap at 40 and 60, the rest at 5'
        caps = run.result.activity('under_cap').sort('t')
        assert list(caps.get_column('value')) == [40.0, 5.0, 60.0, 5.0], (
            'a region reaches its own coordinates and the otherwise carries the rest, in t order'
        )


@pytest.mark.parametrize(
    ('hi', 'objective'),
    [
        pytest.param({'t': [0, 2], 'value': [40.0, 60.0]}, 110.0, id='every flagged step has a cap'),
        pytest.param(
            {'t': [0, 1, 2, 3], 'value': [40.0, 9.0, 60.0, 9.0]}, 110.0, id='a cap outside the region is spare'
        ),
    ],
)
def test_a_constant_side_is_asked_for_data_only_where_its_region_applies(hi, objective):
    """`hi` is the flagged region's cap, so it answers for that region and is never asked about the rest.

    A parameter under a region used to be held to the whole frame, which
    refused the first case outright — the ``otherwise`` carries steps 1 and 3
    and ``hi`` says nothing about them.
    """
    with differential(CAPPED_BY_REGION, _frames(CAPPED_SOURCES | {'hi': hi})) as run:
        assert run.oracle == pytest.approx(objective, rel=RTOL), 'rows outside the region change nothing'


@pytest.mark.parametrize('lane', ['relational', 'eager'])
def test_a_hole_inside_the_region_is_still_refused_on_each_lane(lane):
    """Narrowing the question to the region must not stop it being asked there.

    Asserted lane by lane rather than through ``differential``: either lane
    refusing satisfies a ``pytest.raises`` around both of them, so a harness
    that runs the two together cannot tell which one spoke — and the first cut
    exempted the relational side altogether while the eager side narrowed,
    which is exactly the divergence that hides behind the shared assertion.
    """
    sources = _frames(CAPPED_SOURCES | {'hi': {'t': [0], 'value': [40.0]}})
    with tempfile.TemporaryDirectory() as work:
        path = Path(work) / 'capped.yaml'
        path.write_text(yaml.safe_dump(CAPPED_BY_REGION))
        build = (
            (lambda: lps.build(path, sources))
            if lane == 'relational'
            else (lambda: lpspec_linopy.build(path, dict(sources)))
        )
        with pytest.raises(DataError, match=r"parameter 'hi' covers 1 fewer coordinate"):
            build()


@pytest.mark.parametrize(
    ('when', 'objective', 'reads'),
    [
        pytest.param(
            'true', 130.0, 'the region takes the whole frame, so hi caps every step', id='a region that is everywhere'
        ),
        pytest.param(
            'false', 20.0, 'the region takes nothing, so the otherwise 5 caps every step', id='a region that is nowhere'
        ),
    ],
)
def test_a_region_whose_mask_reads_no_dimension(when, objective, reads):
    """A `when` of `true` names no coordinate set to cut the region down by, so it cuts by its own constant.

    The language takes a boolean literal there, and its ``otherwise`` is the
    negation — between them one region is the whole frame and the other is
    empty. Building a coordinate frame to cross against instead returned a row
    from an empty frame, so both regions landed everywhere and were summed:
    150 rather than 130, on the relational lane only.
    """
    model = CAPPED_BY_REGION | {
        'expressions': {
            'cap': {'foreach': ['t'], 'cases': {'flagged': {'when': when, 'expression': 'hi'}}, 'otherwise': 5}
        }
    }
    sources = _frames(CAPPED_SOURCES | {'hi': {'t': [0, 1, 2, 3], 'value': [40.0, 10.0, 60.0, 20.0]}})
    with differential(model, sources) as run:
        assert run.oracle == pytest.approx(objective, rel=RTOL), reads


CARRIED_IN = {
    'dimensions': {'t': {'dtype': 'int'}, 'g': {}},
    'parameters': {
        'switchable': {'dims': ['g'], 'dtype': 'bool'},
        'before': {'dims': ['g']},
        'cap': {'dims': ['g']},
        'step': {'dims': ['g']},
        'first_step': {'dims': ['g']},
        'load': {'dims': ['t']},
        'cost': {'dims': ['g']},
    },
    'variables': {
        'p': {'foreach': ['t', 'g'], 'bounds': {'lower': 0, 'upper': 'cap'}},
        'on': {'foreach': ['t', 'g'], 'domain': 'binary'},
    },
    'expressions': {
        'carried': {
            'foreach': ['t', 'g'],
            'cases': {
                'never_off': {'when': 'not switchable', 'expression': 1},
                'boundary': {'when': 'switchable and position(t) == 0', 'expression': 'before'},
            },
            # no `edge=`, so this region has nothing at t == 0 - which no region claims it at
            'otherwise': 'shift(on, over=t, offset=1)',
        }
    },
    'constraints': {
        'meet_load': {'foreach': ['t'], 'expression': 'sum(p, over=g) == load'},
        'runs_only_when_on': {'foreach': ['t', 'g'], 'expression': 'p <= on * cap'},
        'ramp': {
            'foreach': ['t', 'g'],
            'expression': 'p - shift(p, over=t, offset=1, edge=0) <= step * carried + first_step * (1 - carried)',
        },
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
}


def _carried_sources(switchable, before):
    return _frames(
        {
            't': [0, 1, 2, 3],
            'g': ['base', 'peak'],
            'switchable': {'g': ['base', 'peak'], 'value': switchable},
            'before': {'g': ['base', 'peak'], 'value': before},
            'cap': {'g': ['base', 'peak'], 'value': [80.0, 60.0]},
            'step': {'g': ['base', 'peak'], 'value': [40.0, 35.0]},
            'first_step': {'g': ['base', 'peak'], 'value': [70.0, 55.0]},
            'cost': {'g': ['base', 'peak'], 'value': [12.0, 45.0]},
            'load': {'t': [0, 1, 2, 3], 'value': [90.0, 60.0, 60.0, 60.0]},
        }
    )


def test_a_region_that_claims_no_coordinate_does_not_unmake_the_row():
    """The `otherwise` shifts with no `edge=`, so it is empty at t == 0 — where no region claims it.

    Its emptiness there is not the quantity's: ``never_off`` and ``boundary``
    between them carry every unit at the first position. A region's absence
    reaching out of the region it applies to took all four t == 0 rows out of
    the build, on both lanes and for different reasons — the relational one
    through the shift's presence, the eager one through a NaN that survived
    being multiplied by a false mask.
    """
    with differential(CARRIED_IN, _carried_sources([False, True], [1.0, 0.0])) as run:
        rows = run.result.activity('ramp')
        assert rows.height == 8, 'every (t, g) coordinate has a ramp row, the first position included'
        assert sorted(set(rows.get_column('t'))) == [0, 1, 2, 3], 't == 0 is built like any other position'
        assert int((run.model.constraints['ramp'].labels == -1).sum()) == 0, (
            'the eager lane masks out no ramp row either'
        )


@pytest.mark.parametrize(
    ('switchable', 'before', 'objective', 'reads'),
    [
        pytest.param(
            [False, True],
            [1.0, 0.0],
            4890.0,
            'boundary: peak was off before the horizon, so it may start at first_step 55',
            id='peak starts cold',
        ),
        pytest.param(
            [True, True],
            [0.0, 0.0],
            3900.0,
            'never_off no longer claims base, so boundary gives it first_step 70 instead of step 40',
            id='base becomes switchable',
        ),
    ],
)
def test_a_region_is_read_and_its_own_data_decides_the_answer(switchable, before, objective, reads):
    """Vary the data each region reads and the answer moves — which is the only proof the region is built.

    A cased expression that quietly collapsed to one region would still solve,
    and would still agree lane to lane. What it could not do is respond to
    ``before`` at the first position while ``switchable`` decides which region
    reads it at all.
    """
    with differential(CARRIED_IN, _carried_sources(switchable, before)) as run:
        assert run.oracle == pytest.approx(objective, rel=RTOL), reads


def test_a_region_binding_tighter_makes_the_model_infeasible_on_both_lanes():
    """`boundary` reading a unit that was already on holds it to `step`, and the load can no longer be met.

    The companion to the case above, where the same edit only moved the
    objective: here it decides feasibility, which is the sharpest evidence the
    region is read. It is asserted lane by lane rather than through
    ``differential``, whose oracle is a finite objective by construction.
    """
    sources = _carried_sources([False, True], [1.0, 1.0])
    with tempfile.TemporaryDirectory() as work:
        path = Path(work) / 'carried.yaml'
        path.write_text(yaml.safe_dump(CARRIED_IN))

        relational = lps.solve(path, sources, solver_name='highs').objective
        eager = lpspec_linopy.build(path, dict(sources))
        eager.solve(solver_name='highs', output_flag=False)

    assert relational != relational, 'the relational lane reports no objective — peak is held to step 35'
    assert eager.objective.value != eager.objective.value, 'and the eager lane reaches the same infeasibility'


@pytest.mark.parametrize(
    'when',
    [
        pytest.param('true', id='a boolean literal'),
        pytest.param('everywhere', id='a scalar parameter'),
    ],
)
def test_a_region_that_claims_nothing_does_not_unmake_the_row(when):
    """The complement of a mask reading no dimension claims nothing, and so may restrict nothing.

    The companion to the case above, one branch over. A dimensionless mask has
    no coordinate set to cut a region down by, so the piece is filtered by the
    mask's own constant — and where that constant is *true* the ``otherwise``
    is left claiming nothing at all, while its ``shift`` with no ``edge=`` is
    still absent at the first position. Letting that presence through
    unrelaxed took every first-position row out of the relational build and
    left the eager one whole: 3570 against an infeasible model.
    """
    model = CARRIED_IN | {
        'parameters': CARRIED_IN['parameters'] | {'everywhere': {'dims': [], 'dtype': 'bool'}},
        'expressions': {
            'carried': {
                'foreach': ['t', 'g'],
                'cases': {'always': {'when': when, 'expression': 1}},
                'otherwise': 'shift(on, over=t, offset=1)',
            }
        },
    }
    sources = _carried_sources([False, True], [1.0, 0.0]) | _frames({'everywhere': {'value': [True]}})
    sources['load'] = _frames({'load': {'t': [0, 1, 2, 3], 'value': [70.0, 60.0, 60.0, 60.0]}})['load']
    with differential(model, sources) as run:
        rows = run.result.activity('ramp')
        assert rows.height == 8, 'every (t, g) coordinate has a ramp row, the first position included'
        assert int((run.model.constraints['ramp'].labels == -1).sum()) == 0, (
            'the eager lane masks out no ramp row either'
        )
        assert run.oracle == pytest.approx(3990.0, rel=RTOL), (
            'carried is 1 everywhere, so the first position is held to step rather than first_step'
        )


def test_one_parameter_answering_for_two_regions():
    """`hi` caps the flagged steps and, doubled, the rest — which is one name owed two answers.

    The pairs a coverage walk collects are ``(name, mask)``, and one parameter
    under two regions makes the names equal and the masks differ. Ordering
    them by the pair rather than by the name asks whether one mask is less
    than another, which an array answers with an array: the eager lane raised
    numpy's ambiguous truth value where the relational lane built.
    """
    model = CAPPED_BY_REGION | {
        'expressions': {
            'cap': {
                'foreach': ['t'],
                'cases': {
                    'flagged': {'when': 'flag', 'expression': 'hi'},
                    'unflagged': {'when': 'not flag', 'expression': 'hi * 2'},
                },
                'otherwise': 0,
            }
        },
    }
    sources = _frames(CAPPED_SOURCES | {'hi': {'t': [0, 1, 2, 3], 'value': [40.0, 10.0, 60.0, 20.0]}})
    with differential(model, sources) as run:
        assert run.oracle == pytest.approx(160.0, rel=RTOL), 'the flagged steps read hi and the rest read twice it'


def test_a_divisor_is_asked_for_data_only_where_its_region_applies():
    """A rate stated for the steps its region claims is not asked about the rest.

    The constant side's rule, one position over: the divisor check walks the
    same tree and had kept its own idea of which rows a piece owes data at, so
    the eager lane refused a model the relational lane built.
    """
    model = CAPPED_BY_REGION | {
        'expressions': {
            'cap': {
                'foreach': ['t'],
                'cases': {'flagged': {'when': 'flag', 'expression': 'hi / rate'}},
                'otherwise': 5,
            },
        },
        'parameters': CAPPED_BY_REGION['parameters'] | {'rate': {'dims': ['t']}},
    }
    sources = _frames(CAPPED_SOURCES | {'rate': {'t': [0, 2], 'value': [2.0, 2.0]}})
    with differential(model, sources) as run:
        assert run.oracle == pytest.approx(60.0, rel=RTOL), 'the flagged steps cap at 40/2 and 60/2, the rest at 5'


def test_a_hole_in_a_divisor_inside_its_region_is_still_refused():
    """Narrowing the divisor's question to the region must not stop it being asked there.

    The eager lane alone, because only it holds a divisor to the rows that
    divide by it: the relational lane reads a missing divisor row as a dropped
    coefficient wherever it stands, region or not, which is #NNNN and not
    something ``cases:`` introduced.
    """
    model = CAPPED_BY_REGION | {
        'expressions': {
            'cap': {
                'foreach': ['t'],
                'cases': {'flagged': {'when': 'flag', 'expression': 'hi / rate'}},
                'otherwise': 5,
            },
        },
        'parameters': CAPPED_BY_REGION['parameters'] | {'rate': {'dims': ['t']}},
    }
    sources = _frames(CAPPED_SOURCES | {'rate': {'t': [0], 'value': [2.0]}})
    with tempfile.TemporaryDirectory() as work:
        path = Path(work) / 'capped.yaml'
        path.write_text(yaml.safe_dump(model))
        with pytest.raises(DataError, match=r"parameter 'rate' is used as a divisor but covers 1 fewer coordinate"):
            lpspec_linopy.build(path, dict(sources))
