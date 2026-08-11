# ruff: noqa: T201  a spike reports by printing; it is not shipped code
"""Benders on today's lpspec, to learn what a decomposition driver would need.

Not a proposal and not shipped code. It answers three questions by running:

1. Is a cut sayable as data? — the master YAML declares `cut` with members from
   data and never changes again; an iteration appends two rows.
2. Does the free oracle work? — the decomposition must reach the monolith's
   objective, built from a third file over the same sources.
3. Where does `solve_over`'s seam stop? — every wall met is recorded in
   FINDINGS at the bottom, which is the actual output of this spike.
"""

from __future__ import annotations

import polars as pl

import lpspec as lps

SNAPSHOTS = [0, 1, 2, 3]
GENERATORS = ['wind', 'gas']

SOURCES = {
    'invest': pl.DataFrame({'generator': GENERATORS, 'value': [90.0, 30.0]}),
    'cost': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 25.0]}),
    'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [40.0, 80.0, 55.0, 95.0]}),
    'avail': pl.DataFrame(
        {
            'snapshot': [s for s in SNAPSHOTS for _ in GENERATORS],
            'generator': GENERATORS * len(SNAPSHOTS),
            'value': [0.9, 1.0, 0.2, 1.0, 0.6, 1.0, 0.1, 1.0],
        }
    ),
    'voll': pl.DataFrame({'value': [3000.0]}),
}


def monolith() -> float:
    """The ground truth: one plan, one solve, over the same sources."""
    with lps.solve('spike/monolith.yaml', SOURCES) as sol:
        return sol.objective


def _cut_from(sub_sol, cap_hat: pl.DataFrame) -> tuple[float, pl.DataFrame]:
    """One optimality cut, as `(constant, slope per generator)`.

    The subgradient form rather than the dual-of-the-dual: `Q` is convex in
    capacity and `dQ/dcap[g]` is the capacity constraint's shadow price summed
    over snapshots and weighted by availability, so

        Q(cap) >= Q(cap_hat) + slope . (cap - cap_hat)

    is tight at `cap_hat` by construction. Written this way the cut needs only
    one dual family and no sign reasoning about the balance row — and any sign
    error shows up immediately as a decomposition that fails to converge to
    :func:`monolith`, which is the point of having the oracle.
    """
    mu = sub_sol.dual('capacity')
    slope = (
        mu.join(SOURCES['avail'], on=['snapshot', 'generator'], suffix='_avail')
        .with_columns((pl.col('value') * pl.col('value_avail')).alias('term'))
        .group_by('generator')
        .agg(pl.col('term').sum().alias('slope'))
    )
    at_hat = slope.join(cap_hat, on='generator').select((pl.col('slope') * pl.col('value')).sum()).item()
    return sub_sol.objective - at_hat, slope


def benders(max_iterations: int = 25, tolerance: float = 1e-6) -> tuple[float, int]:
    """Loop until the bound closes. Returns `(objective, iterations)`.

    **Subproblem first.** The master is solved only once a cut exists, which is
    what every implementation does anyway — but here it is also forced: an
    empty ``cut`` dimension is refused by the engine (FINDING 1), so there is
    no "master with no cuts" state to be in.
    """
    cut_const = pl.DataFrame(schema={'cut': pl.Int64, 'value': pl.Float64})
    cut_slope = pl.DataFrame(schema={'cut': pl.Int64, 'generator': pl.String, 'value': pl.Float64})
    cap_hat = pl.DataFrame({'generator': GENERATORS, 'value': [0.0] * len(GENERATORS)})
    upper, lower = float('inf'), -float('inf')
    sub_sources = {k: v for k, v in SOURCES.items() if k != 'invest'}

    for iteration in range(max_iterations):
        with lps.solve('spike/sub.yaml', {**sub_sources, 'cap_hat': cap_hat}) as sub:
            const, slope = _cut_from(sub, cap_hat)
            spend = cap_hat.join(SOURCES['invest'], on='generator', suffix='_i')
            spend = spend.select((pl.col('value') * pl.col('value_i')).sum()).item()
            upper = min(upper, spend + sub.objective)

        index = cut_const.height
        cut_const = pl.concat([cut_const, pl.DataFrame({'cut': [index], 'value': [const]})])
        cut_slope = pl.concat(
            [
                cut_slope,
                slope.select(pl.lit(index, dtype=pl.Int64).alias('cut'), 'generator', pl.col('slope').alias('value')),
            ]
        )

        master_sources = {'invest': SOURCES['invest'], 'cut_const': cut_const, 'cut_slope': cut_slope}
        with lps.solve('spike/master.yaml', master_sources, coords={'cut': cut_const['cut'].to_list()}) as master:
            lower = master.objective
            cap_hat = master.primal('cap').select('generator', 'value')

        print(f'  iter {iteration}: lower {lower:12.4f}  upper {upper:12.4f}  gap {upper - lower:12.6f}')
        if upper - lower <= tolerance * max(1.0, abs(upper)):
            return upper, iteration + 1

    raise AssertionError(f'no convergence in {max_iterations} iterations')


if __name__ == '__main__':
    truth = monolith()
    print(f'monolith objective = {truth:.6f}\n')
    got, iterations = benders()
    print(f'\nbenders objective  = {got:.6f} after {iterations} iterations')
    print(f'gap to monolith    = {abs(got - truth):.3e}')
    assert abs(got - truth) < 1e-6 * max(1.0, abs(truth)), 'decomposition did not reach the monolith'
    print('\nORACLE HOLDS: the decomposition reached the monolith on the same sources.')
