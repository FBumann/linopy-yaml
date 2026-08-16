"""``at()`` — the pullback, and the models that had no formulation without it.

`sum` walks a mapping table from the fine dim into the coarse one; this
walks it back out. They take the same one argument on purpose: ``by=`` names
one lookup, and the helper says which direction.

Two things are checked here that a single-lane test could not reach.

**The answer, against the oracle.** A pullback duplicates a variable's label
across the fine dim, so the relational lane's key claim has to weaken and the
terminal aggregate has to run. Whether it does is only observable as a *wrong
objective*, which is what the differential case is for.

**The model the ledger cares about.** `examples/multi_period.yaml` is ragged —
four snapshots in 2030 against two in 2050 — which a ``period x snapshot``
rectangle cannot express at all. Its capacity bound reads a per-period
*variable* at every snapshot, and a variable cannot be pre-joined in data prep
the way a parameter can. The number `docs/examples/multi_period.md` quotes is
held by :func:`test_the_multi_period_page_number`.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import lpspec as lps
from tests.conftest import EXAMPLES_DIR

MULTI_PERIOD = EXAMPLES_DIR / 'multi_period.yaml'
PAGE = Path('docs/examples/multi_period.md')

#: 2030 is modelled at four snapshots and 2050 at two — the whole reason the
#: index is flat. `weight` is what keeps them comparable: a 2050 snapshot
#: stands for four hours and is charged for four.
SNAPSHOTS = [0, 1, 2, 3, 4, 5]
PERIOD_OF = [2030, 2030, 2030, 2030, 2050, 2050]
GENERATORS = ['wind', 'gas']


def _sources(capex_2050_wind: float = 8.0):
    return {
        'snapshot': pl.DataFrame({'snapshot': SNAPSHOTS, 'period_of': PERIOD_OF}),
        'period': pl.DataFrame({'period': [2030, 2050]}),
        'generator': pl.DataFrame({'generator': GENERATORS}),
        'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [10.0, 20.0, 30.0, 20.0, 40.0, 60.0]}),
        'weight': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [1.0, 1.0, 1.0, 1.0, 4.0, 4.0]}),
        'opex': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 5.0]}),
        'capex': pl.DataFrame(
            {
                'generator': GENERATORS * 2,
                'period': [2030, 2030, 2050, 2050],
                'value': [10.0, 2.0, capex_2050_wind, 2.0],
            }
        ),
    }


def test_the_multi_period_page_number():
    """The optimum `docs/examples/multi_period.md` quotes, and the build behind it.

    Capacity is per period and binds at every snapshot in that period, so the
    two periods must be able to differ — which is the claim the table on the
    page makes and this holds.
    """
    with lps.solve(MULTI_PERIOD, _sources()) as result:
        nominal = result.primal('p_nom').sort('period', 'generator')
        assert result.objective == pytest.approx(750.0)

    built = {(row['period'], row['generator']): row['value'] for row in nominal.to_dicts()}
    assert built[(2030, 'wind')] == pytest.approx(20.0), '2030 peaks at 30 and splits the build'
    assert built[(2030, 'gas')] == pytest.approx(10.0), '2030 peaks at 30 and splits the build'
    assert built[(2050, 'wind')] == pytest.approx(60.0), (
        '2050 peaks at 60 with every snapshot weighted four times, so the operating term takes it all to wind'
    )
    assert built[(2050, 'gas')] == pytest.approx(0.0), 'gas is priced out of 2050 entirely'

    assert '750.0' in PAGE.read_text(), 'the page quotes an optimum this test does not hold'


def test_a_period_bound_actually_binds():
    """Not vacuous: halving what 2050 may build has to move the answer.

    A pullback that silently dropped its rows would leave `p` unbounded above
    and the objective unchanged, which is the failure this rules out. The cap
    is applied by making 2050 capacity ruinously expensive.
    """
    with lps.solve(MULTI_PERIOD, _sources()) as unbounded:
        base = unbounded.objective

    with lps.solve(MULTI_PERIOD, _sources(capex_2050_wind=80.0)) as dearer:
        assert dearer.objective > base, 'the per-period capacity bound is not reaching the snapshots'


COMPONENT_GATE = {
    'dimensions': {
        'flow': {'dtype': 'str'},
        'component': {'dtype': 'str'},
        't': {'dtype': 'int', 'values': [0, 1]},
    },
    'lookups': {'component_of': {'over': 'flow', 'into': 'component'}},
    'parameters': {'cost': {'dims': ['flow']}, 'oncost': {'dims': ['component']}},
    'variables': {
        'rate': {'foreach': ['flow', 't'], 'bounds': {'lower': 0, 'upper': 10}},
        'on': {'foreach': ['component', 't'], 'domain': 'binary'},
    },
    'constraints': {
        'gate': {'foreach': ['flow', 't'], 'expression': 'rate <= at(on, by=component_of) * 10'},
        'need': {'foreach': ['t'], 'expression': 'sum(rate, over=flow) >= 12'},
    },
    'objective': {'sense': 'minimize', 'expression': 'rate * cost + on * oncost'},
}


def test_one_binary_gates_every_flow_of_its_component():
    """The shape #185 was filed for: a per-component decision read on each of
    that component's flows.

    Reads a **variable** through the map, not a parameter — which is the half
    that matters, since a parameter could be pre-joined in data prep and a
    variable cannot.
    """
    flows, components = ['f1', 'f2', 'f3'], ['c1', 'c2']
    sources = {
        'flow': pl.DataFrame({'flow': flows, 'component_of': ['c1', 'c1', 'c2']}),
        'component': pl.DataFrame({'component': components}),
        'cost': pl.DataFrame({'flow': flows, 'value': [1.0, 2.0, 1.5]}),
        'oncost': pl.DataFrame({'component': components, 'value': [5.0, 7.0]}),
    }
    with lps.solve(COMPONENT_GATE, sources) as result:
        assert result.objective == pytest.approx(38.0)
        running = {(r['component'], r['t']): r['value'] for r in result.primal('on').to_dicts()}
        rates = {(r['flow'], r['t']): r['value'] for r in result.primal('rate').to_dicts()}

    for t in (0, 1):
        for flow, component in zip(flows, ['c1', 'c1', 'c2'], strict=True):
            if rates[(flow, t)] > 1e-9:
                assert running[(component, t)] == pytest.approx(1.0), (
                    f'{flow} runs at t={t} while its component {component} is off — the gate did not reach this flow'
                )


def test_at_agrees_with_the_oracle_through_a_reduction():
    """The differential half, and the case the key claim is about.

    A pullback duplicates a variable's label across the fine dim, so a later
    reduction can bring two copies into one constraint row. If the relational
    lane still claimed its ``(row, col)`` key were unique, the terminal
    aggregate would be skipped and the frame would hold a cell twice — which a
    solver reads as whichever copy it saw last, not as their sum.

    Summing the pulled-back term back over `flow` is what forces that, so this
    is deliberately not the pointwise case the tests above cover.

    The oracle is imported in the body rather than at module scope: every other
    test here is linopy-free and has to keep running on the bare install, so
    this one test skips there instead of failing on a missing pandas.
    """
    from tests.differential import differential
    from tests.oracle import pd

    model = {
        'dimensions': {
            'flow': {'dtype': 'str'},
            'component': {'dtype': 'str'},
        },
        'lookups': {'component_of': {'over': 'flow', 'into': 'component'}},
        'parameters': {'cost': {'dims': ['flow']}, 'share': {'dims': ['flow']}},
        'variables': {
            'level': {'foreach': ['component'], 'bounds': {'lower': 0, 'upper': 10}},
            'take': {'foreach': ['flow'], 'bounds': {'lower': 0, 'upper': 10}},
        },
        'constraints': {
            # summed, so one `level` label lands in this row once per flow of its component
            'draw': {'foreach': [], 'expression': 'sum(at(level, by=component_of) * share, over=flow) >= 9'},
            'link': {'foreach': ['flow'], 'expression': 'take <= at(level, by=component_of)'},
        },
        'objective': {'sense': 'minimize', 'expression': 'level * 1.0 + take * cost'},
    }
    flows, components = ['f1', 'f2', 'f3'], ['c1', 'c2']
    data = {
        'cost': pd.Series([1.0, 2.0, 1.5], index=flows),
        'share': pd.Series([1.0, 2.0, 3.0], index=flows),
    }
    coords = {
        'flow': pd.DataFrame({'flow': flows, 'component_of': ['c1', 'c1', 'c2']}),
        'component': pd.Index(components, name='component'),
    }
    with differential(model, data, coords) as run:
        assert run.result.objective > 0


def test_the_retired_onto_kwarg_names_its_rewrite():
    """`at(onto=, by=)` restated what the lookup already knows.

    The lookup carries the dim it is over, so `onto=` could only agree with it
    or contradict it — and a kwarg whose every legal value is derivable is one
    the file should not have to write. There is no alias, so the load error is
    the whole migration story.
    """
    model = {
        'dimensions': {'flow': {'dtype': 'str'}, 'component': {'dtype': 'str'}},
        'lookups': {'component_of': {'over': 'flow', 'into': 'component'}},
        'parameters': {'cap': {'dims': ['flow']}},
        'variables': {'level': {'foreach': ['component'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'link': {'foreach': ['flow'], 'expression': 'at(level, onto=flow, by=component_of) <= cap'}},
        'objective': {'sense': 'minimize', 'expression': 'level'},
    }
    with pytest.raises(lps.LanguageError) as exc:
        lps.load_model(model)
    assert 'at(onto=...) was removed' in str(exc.value)
    assert 'at(<expr>, by=<lookup>)' in str(exc.value), 'the refusal has to name the rewrite'


def test_a_window_whose_length_is_read_from_data_is_an_incidence_table():
    """The window family's data-dependent member, and what to write instead of a shift chain.

    A minimum up time — a unit that starts must stay committed for its own *T*
    snapshots — is `sum(start over the last T) <= status`. Where *T* is fixed in
    the file that is a chain of shifts, a macro. Where each unit carries its
    own, the *number of terms* differs per unit and no chain can be written
    down, which the ledger read as the constraint being unsayable.

    It is not. The window is a relation between snapshots — one row per pair
    inside it — so it is an incidence table contracted along a mirror of the
    snapshot axis, the shape `pypsa_kvl` already uses for a cycle basis. The
    plan's *shape* is fixed before any data is read; only its cardinality comes
    from data, which is as true of `foreach: [snapshot]`.

    The mirror needs no second commitment variable and no identity table: `tf`
    maps back to `t` single-valuedly, which is a lookup, and `at()` reads the
    commitment onto the mirror axis where the start recurrence needs it.

    Two units, one with T=3 and one with T=1, and a no-load cost so idling is
    not free. The slow unit is cheaper to run, so it is chosen and must then
    stay up its full three hours: **13.0**. Relaxing the window row gives 11.0
    with one hour up, which is what makes it bind rather than decorate.

    Relational lane only: the differential harness cannot carry a 3-D
    parameter, because the two lanes take a wide frame and a tidy one
    respectively (#60). The claim here is about what the language can state,
    and the engine is what states it.
    """
    import polars as pl

    import lpspec as lps

    up_time = {'slow': 3, 'fast': 1}
    hours = list(range(6))
    model = {
        'dimensions': {'unit': {'dtype': 'str'}, 't': {'dtype': 'int'}, 'tf': {'dtype': 'int'}},
        # every `tf` is the same moment as one `t` — single-valued, so a lookup
        'lookups': {'same_moment': {'over': 'tf', 'into': 't'}},
        'parameters': {
            'window': {'dims': ['unit', 't', 'tf']},
            'load': {'dims': ['t']},
            'cap': {'dims': ['unit']},
            'run_cost': {'dims': ['unit']},
            'idle_cost': {'dims': ['unit']},
        },
        'variables': {
            'p': {'foreach': ['unit', 't'], 'bounds': {'lower': 0}},
            'on': {'foreach': ['unit', 't'], 'domain': 'binary'},
            'started': {'foreach': ['unit', 'tf'], 'domain': 'binary'},
        },
        'constraints': {
            # the commitment read onto the mirror axis, where the recurrence lives
            'a_start_turns_it_on': {
                'foreach': ['unit', 'tf'],
                'expression': (
                    'started >= at(on, by=same_moment) - shift(at(on, by=same_moment), over=tf, by=1, edge=0)'
                ),
            },
            'stays_up_its_own_time': {
                'foreach': ['unit', 't'],
                'expression': 'sum(started * window, over=tf) <= on',
            },
            'within_capacity': {'foreach': ['unit', 't'], 'expression': 'p <= on * cap'},
            'meet_load': {'foreach': ['t'], 'expression': 'sum(p, over=unit) >= load'},
        },
        'objective': {'sense': 'minimize', 'expression': 'p * run_cost + on * idle_cost'},
    }
    rows = [(u, t, tf) for u, k in up_time.items() for t in hours for tf in hours if 0 <= t - tf < k]
    sources = {
        'window': pl.DataFrame(
            {
                'unit': [r[0] for r in rows],
                't': [r[1] for r in rows],
                'tf': [r[2] for r in rows],
                'value': [1.0] * len(rows),
            }
        ),
        'load': pl.DataFrame({'t': hours, 'value': [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]}),
        'cap': pl.DataFrame({'unit': list(up_time), 'value': [10.0, 10.0]}),
        'run_cost': pl.DataFrame({'unit': list(up_time), 'value': [1.0, 5.0]}),
        'idle_cost': pl.DataFrame({'unit': list(up_time), 'value': [1.0, 1.0]}),
    }
    coords = {
        'unit': pl.DataFrame({'unit': list(up_time)}),
        't': pl.DataFrame({'t': hours}),
        'tf': pl.DataFrame({'tf': hours, 'same_moment': hours}),
    }
    with lps.solve(model, sources, coords=coords) as solution:
        assert solution.objective == pytest.approx(13.0), (
            'the slow unit runs and is held up its own three hours; 11.0 would mean the window read nothing'
        )
        on = solution.primal('on').filter(pl.col('value') > 0.5)
        assert on.height == 3, 'exactly the three snapshots its own minimum up time forces'
        assert set(on['unit']) == {'slow'}, 'and it is the slow unit that is held, not the fast one'
