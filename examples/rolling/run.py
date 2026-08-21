"""A rolling horizon on lpspec, checked against the horizon it approximates.

    pixi run python examples/rolling/run.py

**This is evidence, not a feature.** `solve_over` ships; what this shows is
that the pieces compose into the thing people actually want — a storage
schedule over a whole year, solved a day at a time — and that the answer lines
up with the one full foresight gives.

One file, `horizon.yaml`, written over a *local* index `t`. The same YAML is
solved as one window over the whole horizon — the reference — and then as
rolling windows of increasing lookahead, each handing its state of charge to
the next through `carry`.

**The store cycles in every schedule.** This is deliberately not a model where
myopia switches storage off: wind blows nightly and load triples by day, so
charging and discharging is worth doing inside any window. What lookahead buys
is the *end* of one. Charge a window cannot spend before its horizon runs out
is worth nothing to it, so it arrives empty and hands zero to the next; a
window that can see far enough has somewhere to spend it.

Windows advance eight hours against a twelve-hour cycle, so a boundary lands at
a different point of the day each time. A horizon that happened to align with
the cycle would end each window at the natural trough and show no gap at all —
which is a property of that arithmetic, not of rolling horizons.

Three properties are asserted rather than printed:

- the stitched schedule covers every snapshot exactly once — no overlap
  double-counted, no tail dropped
- rolling never beats full foresight, which is what myopia means
- the store is used in every schedule, so the gap is a quality difference and
  not storage quietly disappearing
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import lpspec as lps

HERE = Path(__file__).parent
MODEL = HERE / 'horizon.yaml'

DAY = 12  #: hours in a day — six of night, then six of daylight
DAYS = 4
PERIODS = DAY * DAYS
GENERATORS = ['wind', 'gas']

#: Wind blows at night when load is low; by day there is none and load triples.
#: The store therefore cycles every day, in every schedule below — what changes
#: is how *well*.
NIGHT = [t % DAY < DAY // 2 for t in range(PERIODS)]
LOAD = [25.0 if night else 85.0 for night in NIGHT]
WIND = [70.0 if night else 0.0 for night in NIGHT]

#: Windows advance eight hours against a twelve-hour cycle, so a window
#: boundary falls at a different point of the day each time — which is the
#: ordinary case, and the one a horizon that happened to align would hide.
STEP = 8

SOURCES = {
    'generator': pl.DataFrame({'generator': GENERATORS}),
    'p_max': pl.DataFrame(
        {
            'snapshot': [t for t in range(PERIODS) for _ in GENERATORS],
            'generator': GENERATORS * PERIODS,
            'value': [v for t in range(PERIODS) for v in (WIND[t], 200.0)],
        }
    ),
    'cost': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 40.0]}),
    'load': pl.DataFrame({'snapshot': range(PERIODS), 'value': LOAD}),
    'soc_initial': pl.DataFrame({'value': [0.0]}),
}


def full_foresight() -> lps.Runs:
    """One window over the whole horizon — the answer rolling is measured against."""
    return lps.solve_over(
        MODEL,
        SOURCES,
        lps.EachWindow('snapshot', length=PERIODS, step=PERIODS, into='t'),
    )


def rolling(length: int, step: int) -> lps.Runs:
    """Windows of *length*, advancing *step*, each carrying its final kept level.

    The carry reads local index ``step - 1``: the last row this window *keeps*,
    not the last it solved. With lookahead those differ, and taking the last
    solved row would carry a level the next window is about to recompute.
    """
    return lps.solve_over(
        MODEL,
        SOURCES,
        lps.EachWindow('snapshot', length=length, step=step, into='t'),
        carry={'soc_initial': ('soc', step - 1)},
    )


def cost_of(runs: lps.Runs) -> float:
    """What the schedule cost, summed over the snapshots each window owns.

    A window objective covers its lookahead too, so summing them double-counts.
    `spend` is the model's own per-snapshot definition, and the stitched read
    keeps only the rows a window owns — the same quantity the objective
    minimises, never restated in a second language.
    """
    return float(runs.expression('spend', original_index=True)['value'].sum())


def main() -> None:
    reference = full_foresight()
    best = cost_of(reference)
    peak = reference.primal('soc', original_index=True)['value'].max()
    print(f'{DAYS} days of {DAY} hours: six of cheap wind, then six of none.')
    print(f'full foresight   one window                      cost {best:>9.2f}   peak soc {peak:>6.1f}')
    print()

    for length in (STEP, STEP + 4, STEP + 8):
        runs = rolling(length, STEP)
        stitched = runs.primal('soc', original_index=True)
        assert stitched['snapshot'].to_list() == list(range(PERIODS)), 'the stitch must cover the horizon'

        cost = cost_of(runs)
        assert cost >= best - 1e-6, 'rolling cannot beat full foresight'
        assert stitched['value'].max() > 0, 'the store must be used in every schedule'
        print(
            f'rolling  length={length:<3} step={STEP:<3} lookahead={length - STEP:<3} '
            f'windows {len(runs):>2}   cost {cost:>9.2f}   peak soc {stitched["value"].max():>6.1f}'
            f'   +{100 * (cost - best) / best:>5.1f}%'
        )

    print()
    print('The store cycles in every schedule — this is not a model where storage')
    print('stops being worth having. What myopia costs is the end of each window:')
    print('charge a window cannot spend before its horizon runs out is worth')
    print('nothing to it, so it arrives empty and the next window starts from zero.')
    print('Lookahead gives it somewhere to spend that charge, and the gap closes.')
    print()
    print(f'every schedule above covers all {PERIODS} snapshots exactly once')


if __name__ == '__main__':
    main()
