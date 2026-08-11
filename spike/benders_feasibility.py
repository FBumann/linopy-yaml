# ruff: noqa: T201  a spike reports by printing; it is not shipped code
"""Benders with the crutch removed: a subproblem that can be infeasible.

`benders.py` kept its subproblem always-feasible with `unserved` at VOLL, which
makes the value function finite everywhere and needs only optimality cuts. That
is a legitimate formulation and it hid the question this file asks: **what does
the algorithm surface actually cost, on an engine that hands back no Farkas
ray?**

Three findings, each of them structural rather than incidental:

1. **lpspec cannot give a feasibility cut directly.** An infeasible solve has no
   readable status, so `dual` raises — correctly, since the values would be a
   full-length vector of zeros. There is no dual-ray accessor. So the cut has to
   come from somewhere else.
2. **That somewhere else is a fourth YAML file.** A model declares one
   objective, so "minimise violation" cannot be a second objective on the
   subproblem — it is `strict_feas.yaml`, and it duplicates the subproblem's
   constraints to change what is minimised over them.
3. **The master grows a second cut family**, because the two cuts say different
   things: one bounds `theta`, the other bounds `cap` alone.

Each is a consequence of decisions that are right on their own. Together they
are what "shipping the algorithm surface" would mean.
"""

from __future__ import annotations

import polars as pl

import lpspec as lps
from spike.benders import GENERATORS, SOURCES
from spike.driver import Iteration, Step, iterate

STRICT = {name: frame for name, frame in SOURCES.items() if name not in {'invest', 'voll'}}


def monolith() -> float:
    """The oracle, with no unserved-energy escape."""
    with lps.solve('spike/strict_monolith.yaml', {k: v for k, v in SOURCES.items() if k != 'voll'}) as sol:
        return sol.objective


def _slope(sol: lps.Result, cap_hat: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    """`dQ/dcap` from the capacity duals, and its value at *cap_hat*."""
    slope = (
        sol.dual('capacity')
        .join(SOURCES['avail'], on=['snapshot', 'generator'], suffix='_avail')
        .with_columns((pl.col('value') * pl.col('value_avail')).alias('term'))
        .group_by('generator')
        .agg(pl.col('term').sum().alias('slope'))
    )
    at_hat = slope.join(cap_hat, on='generator').select((pl.col('slope') * pl.col('value')).sum()).item()
    return slope, at_hat


def _append(state: dict[str, pl.DataFrame], family: str, index: int, constant: float, slope: pl.DataFrame) -> None:
    state[f'{family}_const'] = pl.concat(
        [state[f'{family}_const'], pl.DataFrame({family: [index], 'value': [constant]})]
    )
    state[f'{family}_slope'] = pl.concat(
        [
            state[f'{family}_slope'],
            slope.select(pl.lit(index, dtype=pl.Int64).alias(family), 'generator', pl.col('slope').alias('value')),
        ]
    )


def benders(tolerance: float = 1e-6) -> Iteration:
    upper = float('inf')

    def step(index: int, state: dict[str, pl.DataFrame]) -> tuple[Step, dict[str, pl.DataFrame]]:
        nonlocal upper
        cap_hat = state['cap_hat']
        grown = dict(state)

        with lps.solve('spike/strict_sub.yaml', {**STRICT, 'cap_hat': cap_hat}) as sub:
            feasible = sub.has_primal
            if feasible:
                slope, at_hat = _slope(sub, cap_hat)
                spend = cap_hat.join(SOURCES['invest'], on='generator', suffix='_rate')
                spend = spend.select((pl.col('value') * pl.col('value_rate')).sum()).item()
                upper = min(upper, spend + sub.objective)
                _append(grown, 'cut', state['cut_const'].height, sub.objective - at_hat, slope)

        if not feasible:
            # No ray to read, so the violation is minimised instead and its own
            # duals give the cut: v(cap_hat) + slope . (cap - cap_hat) <= 0.
            with lps.solve('spike/strict_feas.yaml', {**STRICT, 'cap_hat': cap_hat}) as feas:
                slope, at_hat = _slope(feas, cap_hat)
                _append(grown, 'fcut', state['fcut_const'].height, at_hat - feas.objective, slope)

        sources = {'invest': SOURCES['invest'], **{k: v for k, v in grown.items() if k != 'cap_hat'}}
        coords = {'cut': grown['cut_const']['cut'].to_list(), 'fcut': grown['fcut_const']['fcut'].to_list()}
        with lps.solve('spike/strict_master.yaml', sources, coords=coords) as master:
            lower = master.objective
            grown['cap_hat'] = master.primal('cap').select('generator', 'value')

        gap = upper - lower
        print(
            f'  step {index}: {"optimality" if feasible else "FEASIBILITY"} cut  '
            f'lower {lower:11.4f}  upper {upper:12.4f}  gap {gap:12.4f}'
        )
        return Step(index, upper, lower, gap - tolerance * max(1.0, abs(upper))), grown

    return iterate(
        step,
        state={
            'cap_hat': pl.DataFrame({'generator': GENERATORS, 'value': [0.0] * len(GENERATORS)}),
            'cut_const': pl.DataFrame(schema={'cut': pl.Int64, 'value': pl.Float64}),
            'cut_slope': pl.DataFrame(schema={'cut': pl.Int64, 'generator': pl.String, 'value': pl.Float64}),
            'fcut_const': pl.DataFrame(schema={'fcut': pl.Int64, 'value': pl.Float64}),
            'fcut_slope': pl.DataFrame(schema={'fcut': pl.Int64, 'generator': pl.String, 'value': pl.Float64}),
        },
        until=lambda taken: taken.gap <= 0.0,
    )


if __name__ == '__main__':
    truth = monolith()
    print(f'strict monolith = {truth:.6f}\n')
    run = benders()
    got = run.steps[-1].objective
    print(f'\nbenders         = {got:.6f} after {len(run.steps)} steps')
    print(f'optimality cuts = {run.state["cut_const"].height},  feasibility cuts = {run.state["fcut_const"].height}')
    assert abs(got - truth) < 1e-6 * max(1.0, abs(truth)), 'the decomposition missed the monolith'
    print('\nORACLE HOLDS with a subproblem that can be infeasible.')
