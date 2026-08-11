# ruff: noqa: T201  a spike reports by printing; it is not shipped code
"""The same Benders, expressed through `driver.iterate`.

`benders.py` is the algorithm in a hand-written loop. This is the identical
algorithm with the loop taken away, and it exists to answer one question:
**does the sibling driver actually carry it, or does it just relocate the
loop?**

The comparison is the point. Both must reach the monolith, and the second must
be smaller in the part that is about *iterating* while being no larger in the
part that is about *Benders*.
"""

from __future__ import annotations

import polars as pl

import lpspec as lps
from spike.benders import GENERATORS, SOURCES, monolith
from spike.driver import Iteration, Step, iterate

SUB_SOURCES = {name: frame for name, frame in SOURCES.items() if name != 'invest'}


def _spend(cap_hat: pl.DataFrame) -> float:
    """What the master's capacity choice costs before it is operated."""
    priced = cap_hat.join(SOURCES['invest'], on='generator', suffix='_rate')
    return priced.select((pl.col('value') * pl.col('value_rate')).sum()).item()


def _cut(cap_hat: pl.DataFrame, sub: lps.Result) -> tuple[float, pl.DataFrame]:
    """One optimality cut as `(constant, slope per generator)`, tight at *cap_hat*."""
    slope = (
        sub.dual('capacity')
        .join(SOURCES['avail'], on=['snapshot', 'generator'], suffix='_avail')
        .with_columns((pl.col('value') * pl.col('value_avail')).alias('term'))
        .group_by('generator')
        .agg(pl.col('term').sum().alias('slope'))
    )
    at_hat = slope.join(cap_hat, on='generator').select((pl.col('slope') * pl.col('value')).sum()).item()
    return sub.objective - at_hat, slope


def benders(tolerance: float = 1e-6) -> Iteration:
    """Benders as one `step`, with the loop, the bound and the growth outside it."""
    upper = float('inf')

    def step(index: int, state: dict[str, pl.DataFrame]) -> tuple[Step, dict[str, pl.DataFrame]]:
        nonlocal upper
        cap_hat = state['cap_hat']

        with lps.solve('spike/sub.yaml', {**SUB_SOURCES, 'cap_hat': cap_hat}) as sub:
            constant, slope = _cut(cap_hat, sub)
            upper = min(upper, _spend(cap_hat) + sub.objective)

        grown = {
            'cut_const': pl.concat([state['cut_const'], pl.DataFrame({'cut': [index], 'value': [constant]})]),
            'cut_slope': pl.concat(
                [
                    state['cut_slope'],
                    slope.select(
                        pl.lit(index, dtype=pl.Int64).alias('cut'), 'generator', pl.col('slope').alias('value')
                    ),
                ]
            ),
        }

        master_sources = {'invest': SOURCES['invest'], **grown}
        with lps.solve(
            'spike/master.yaml', master_sources, coords={'cut': grown['cut_const']['cut'].to_list()}
        ) as master:
            lower = master.objective
            grown['cap_hat'] = master.primal('cap').select('generator', 'value')

        gap = upper - lower
        return Step(index, upper, lower, gap - tolerance * max(1.0, abs(upper))), grown

    return iterate(
        step,
        state={
            'cap_hat': pl.DataFrame({'generator': GENERATORS, 'value': [0.0] * len(GENERATORS)}),
            'cut_const': pl.DataFrame(schema={'cut': pl.Int64, 'value': pl.Float64}),
            'cut_slope': pl.DataFrame(schema={'cut': pl.Int64, 'generator': pl.String, 'value': pl.Float64}),
        },
        until=lambda taken: taken.gap <= 0.0,
    )


if __name__ == '__main__':
    truth = monolith()
    run = benders()
    print(run.history())
    print(f'\nmonolith          = {truth:.6f}')
    print(f'benders on driver = {run.steps[-1].objective:.6f} after {len(run.steps)} steps')
    assert run.converged, 'the driver stopped without closing the bound'
    assert abs(run.steps[-1].objective - truth) < 1e-6 * max(1.0, abs(truth))
    print(f'cuts accumulated  = {run.state["cut_const"].height}')
    print('\nORACLE HOLDS through the driver as well as through the hand-written loop.')
