"""A myopic investment pathway on lpspec, one period at a time.

    pixi run python examples/myopic/run.py

**This is evidence, not a feature.** It is the shape capacity-expansion models
are actually run in: a handful of investment periods, each solved on a few
typical days, each inheriting the fleet the last one left.

One file, `pathway.yaml`, written for *a* period. The driver supplies which:

    lps.solve_over(model, sources, lps.EachCoordinate('year', ordered=True),
                   carry={'existing': ('total', None)})

`ordered=True` is what makes a carry legal — scenarios have no "next", years
do. The carry has **no index** because nothing is dropped: `total` is over
`(generator)` and so is `existing`, so the whole fleet vector moves forward.
That is the myopic case in one line, and the rolling-horizon case
(`examples/rolling/`) is the same keyword with an index because there a
dimension *is* dropped.

**The accumulation is in the YAML, not here.** `carry` copies; it never adds.
`total == existing + build` is a constraint in `pathway.yaml`, where it is
reviewable and the linopy oracle can check it — which is the whole reason the
driver has no `combine: add`.

What the run asserts rather than prints:

- period *i+1* starts from exactly the fleet period *i* ended with, which is
  the promise `carry` makes
- capacity never falls, since nothing here retires
"""

from __future__ import annotations

import itertools
from pathlib import Path

import polars as pl

import lpspec as lps

HERE = Path(__file__).parent
MODEL = HERE / 'pathway.yaml'

YEARS = [2030, 2035, 2040]
DAYS = ['winter', 'summer']
HOURS = [0, 6, 12, 18]
GENERATORS = ['solar', 'gas']

#: Two typical days standing for a year between them.
WEIGHT = {'winter': 120.0, 'summer': 245.0}

#: Solar works in the middle of the day and better in summer; gas always runs.
AVAIL = {
    ('winter', 'solar'): [0.0, 0.15, 0.35, 0.0],
    ('summer', 'solar'): [0.0, 0.55, 0.95, 0.10],
    ('winter', 'gas'): [1.0, 1.0, 1.0, 1.0],
    ('summer', 'gas'): [1.0, 1.0, 1.0, 1.0],
}

#: Demand grows every period, which is what forces a build.
BASE_LOAD = {'winter': [70.0, 90.0, 95.0, 85.0], 'summer': [45.0, 70.0, 110.0, 60.0]}
GROWTH = {2030: 1.0, 2035: 1.25, 2040: 1.5}

#: Solar gets cheaper; gas fuel gets dearer. So each period's answer differs.
INVEST = {2030: {'solar': 42000.0, 'gas': 55000.0}, 2035: {'solar': 30000.0, 'gas': 55000.0}}
INVEST[2040] = {'solar': 24000.0, 'gas': 55000.0}
FUEL = {2030: 55.0, 2035: 70.0, 2040: 90.0}


def sources() -> dict[str, object]:
    """Every table carrying `year`, which is the column the axis slices on.

    ``existing`` is the fleet the pathway starts from: a little gas, and
    nothing else.
    """
    return {
        'day': pl.DataFrame({'day': DAYS}),
        'hour': pl.DataFrame({'hour': HOURS}),
        'generator': pl.DataFrame({'generator': GENERATORS}),
        'weight': pl.DataFrame({'day': DAYS, 'value': [WEIGHT[d] for d in DAYS]}),
        'load': pl.DataFrame(
            [
                {'year': y, 'day': d, 'hour': h, 'value': BASE_LOAD[d][i] * GROWTH[y]}
                for y in YEARS
                for d in DAYS
                for i, h in enumerate(HOURS)
            ]
        ),
        'avail': pl.DataFrame(
            [
                {'day': d, 'hour': h, 'generator': g, 'value': AVAIL[d, g][i]}
                for d in DAYS
                for g in GENERATORS
                for i, h in enumerate(HOURS)
            ]
        ),
        'invest': pl.DataFrame([{'year': y, 'generator': g, 'value': INVEST[y][g]} for y in YEARS for g in GENERATORS]),
        'cost': pl.DataFrame(
            [{'year': y, 'generator': g, 'value': 0.0 if g == 'solar' else FUEL[y]} for y in YEARS for g in GENERATORS]
        ),
        'existing': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 60.0]}),
    }


def main() -> None:
    runs = lps.solve_over(
        MODEL,
        sources(),
        lps.EachCoordinate('year', ordered=True),
        carry={'existing': ('total', None)},
    )

    print('myopic pathway — each period sees only itself, and inherits the last')
    print()
    print(runs.objective.select('year', 'termination_condition', pl.col('objective').round(0)))
    print()

    fleet = runs.primal('total', original_index=True).pivot('generator', index='year', values='value')
    built = runs.primal('build', original_index=True).pivot('generator', index='year', values='value')
    print('fleet after each period (MW)')
    print(fleet.select('year', pl.col(GENERATORS).round(1)))
    print()
    print('built in each period (MW)')
    print(built.select('year', pl.col(GENERATORS).round(1)))
    print()

    _check_the_carry_moved_the_fleet(runs)
    assert fleet['solar'].to_list() == sorted(fleet['solar'].to_list()), 'nothing retires'
    print(f'each period starts from the fleet the last one left, across {len(YEARS)} periods')


def _check_the_carry_moved_the_fleet(runs: lps.Runs) -> None:
    """Period *i+1* inherited exactly the fleet period *i* ended with.

    `existing` is never read back — it is a parameter, not a variable — so the
    check is on what the model did with it: `total - build` is what the period
    started from, and it must equal the previous period's `total`.
    """
    totals, builds = runs.primal('total'), runs.primal('build')

    def per_year(frame: pl.DataFrame, year: int) -> list[float]:
        return frame.filter(pl.col('year') == year).sort('generator')['value'].to_list()

    for earlier, later in itertools.pairwise(YEARS):
        ended = per_year(totals, earlier)
        added = per_year(builds, later)
        inherited = [total - built for total, built in zip(per_year(totals, later), added, strict=True)]
        assert all(abs(a - b) < 1e-6 for a, b in zip(ended, inherited, strict=True)), (
            f'{later} started from {inherited}, but {earlier} ended with {ended}'
        )


if __name__ == '__main__':
    main()
