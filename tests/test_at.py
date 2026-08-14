"""``at()`` — the pullback, and the models that had no formulation without it.

`sum` walks a mapping table from the fine dim into the coarse one; this
walks it back out. They take the same two arguments on purpose: ``(onto, by)``
names one table, and the helper says which direction.

Two things are checked here that a single-lane test could not reach.

**The answer, against the oracle.** A pullback duplicates a variable's label
across the fine dim, so the relational lane's key claim has to weaken and the
terminal aggregate has to run. Whether it does is only observable as a *wrong
objective*, which is what the differential case is for.

**The model the ledger cares about.** `examples/multi_period.yaml` is ragged —
four snapshots in 2030 against two in 2050 — which a ``period x snapshot``
rectangle cannot express at all. Its capacity bound reads a per-period
*variable* at every snapshot, and a variable cannot be pre-joined in data prep
the way a parameter can. The number `docs/models/multi_period.md` quotes is
held by :func:`test_the_multi_period_page_number`.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import lpspec as lps
from tests.conftest import EXAMPLES_DIR

MULTI_PERIOD = EXAMPLES_DIR / 'multi_period.yaml'
PAGE = Path('docs/models/multi_period.md')

#: 2030 is modelled at four snapshots and 2050 at two — the whole reason the
#: index is flat. `weight` is what keeps them comparable: a 2050 snapshot
#: stands for four hours and is charged for four.
SNAPSHOTS = [0, 1, 2, 3, 4, 5]
PERIOD_OF = [2030, 2030, 2030, 2030, 2050, 2050]
GENERATORS = ['wind', 'gas']


def _sources(capex_2050_wind: float = 8.0):
    return {
        'snapshot': pl.DataFrame({'snapshot': SNAPSHOTS, 'period': PERIOD_OF}),
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
    """The optimum `docs/models/multi_period.md` quotes, and the build behind it.

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
        'flow': {'dtype': 'str', 'coords': ['component']},
        'component': {'dtype': 'str'},
        't': {'dtype': 'int', 'values': [0, 1]},
    },
    'parameters': {'cost': {'dims': ['flow']}, 'oncost': {'dims': ['component']}},
    'variables': {
        'rate': {'foreach': ['flow', 't'], 'bounds': {'lower': 0, 'upper': 10}},
        'on': {'foreach': ['component', 't'], 'binary': True},
    },
    'constraints': {
        'gate': {'foreach': ['flow', 't'], 'expression': 'rate <= at(on, onto=flow, by=component) * 10'},
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
        'flow': pl.DataFrame({'flow': flows, 'component': ['c1', 'c1', 'c2']}),
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
            'flow': {'dtype': 'str', 'coords': ['component']},
            'component': {'dtype': 'str'},
        },
        'parameters': {'cost': {'dims': ['flow']}, 'share': {'dims': ['flow']}},
        'variables': {
            'level': {'foreach': ['component'], 'bounds': {'lower': 0, 'upper': 10}},
            'take': {'foreach': ['flow'], 'bounds': {'lower': 0, 'upper': 10}},
        },
        'constraints': {
            # summed, so one `level` label lands in this row once per flow of its component
            'draw': {'foreach': [], 'expression': 'sum(at(level, onto=flow, by=component) * share, over=flow) >= 9'},
            'link': {'foreach': ['flow'], 'expression': 'take <= at(level, onto=flow, by=component)'},
        },
        'objective': {'sense': 'minimize', 'expression': 'level * 1.0 + take * cost'},
    }
    flows, components = ['f1', 'f2', 'f3'], ['c1', 'c2']
    data = {
        'cost': pd.Series([1.0, 2.0, 1.5], index=flows),
        'share': pd.Series([1.0, 2.0, 3.0], index=flows),
    }
    coords = {
        'flow': pd.DataFrame({'flow': flows, 'component': ['c1', 'c1', 'c2']}),
        'component': pd.Index(components, name='component'),
    }
    with differential(model, data, coords) as run:
        assert run.result.objective > 0
