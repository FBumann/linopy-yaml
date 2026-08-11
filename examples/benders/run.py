"""Benders decomposition on lpspec, checked against the model it decomposes.

    uv run python examples/benders/run.py

**This is evidence, not a feature.** It shows what the language can express and
that the answer is right; lpspec ships no decomposition driver, and whether it
should is https://github.com/fluxopt/lpspec/issues/596.

Four files, and the split is the whole idea:

- ``monolith.yaml``  the problem in one plan — the answer everything else must reach
- ``master.yaml``    capacity, plus a placeholder for what operating it will cost
- ``sub.yaml``       dispatch at a capacity someone else chose; infeasible if it is too small
- ``feasibility.yaml``  how far from dispatchable a capacity is, when it is too small

The master's cuts are **data**. It declares ``cut`` and ``fcut`` with members
from data (SPEC §8) and never changes; an iteration appends rows to their
parameter tables. No YAML is written at runtime, so the model a reviewer reads
is the model that runs — which is the point of writing models in YAML at all.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import lpspec as lps

HERE = Path(__file__).parent
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
}
DISPATCH = {name: frame for name, frame in SOURCES.items() if name != 'invest'}

EMPTY = {
    'cut_const': pl.DataFrame(schema={'cut': pl.Int64, 'value': pl.Float64}),
    'cut_slope': pl.DataFrame(schema={'cut': pl.Int64, 'generator': pl.String, 'value': pl.Float64}),
    'fcut_const': pl.DataFrame(schema={'fcut': pl.Int64, 'value': pl.Float64}),
    'fcut_slope': pl.DataFrame(schema={'fcut': pl.Int64, 'generator': pl.String, 'value': pl.Float64}),
}


def slope_at(solution: lps.Result, capacity: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    """How the subproblem's value moves with capacity, and its value there.

    The capacity constraint's shadow price is that derivative, weighted by
    availability and summed over snapshots. Reading it needs nothing but
    ``dual`` and a join against the model's own ``avail`` table.
    """
    slope = (
        solution.dual('capacity')
        .join(SOURCES['avail'], on=['snapshot', 'generator'], suffix='_avail')
        .with_columns((pl.col('value') * pl.col('value_avail')).alias('term'))
        .group_by('generator')
        .agg(pl.col('term').sum().alias('slope'))
    )
    here = slope.join(capacity, on='generator').select((pl.col('slope') * pl.col('value')).sum()).item()
    return slope, here


def appended(tables: dict[str, pl.DataFrame], family: str, constant: float, slope: pl.DataFrame) -> None:
    """One more cut in *family*, in place. This is the whole of "a cut is data"."""
    index = tables[f'{family}_const'].height
    tables[f'{family}_const'] = pl.concat(
        [tables[f'{family}_const'], pl.DataFrame({family: [index], 'value': [constant]})]
    )
    tables[f'{family}_slope'] = pl.concat(
        [
            tables[f'{family}_slope'],
            slope.select(pl.lit(index, dtype=pl.Int64).alias(family), 'generator', pl.col('slope').alias('value')),
        ]
    )


def main() -> None:
    with lps.solve(HERE / 'monolith.yaml', SOURCES) as whole:
        truth = whole.objective
    print(f'the whole problem, in one plan: {truth:.2f}\n')

    tables = dict(EMPTY)
    capacity = pl.DataFrame({'generator': GENERATORS, 'value': [0.0] * len(GENERATORS)})
    upper = float('inf')

    for step in range(25):
        with lps.solve(HERE / 'sub.yaml', {**DISPATCH, 'cap_hat': capacity}) as sub:
            dispatchable = sub.has_primal
            if dispatchable:
                slope, here = slope_at(sub, capacity)
                spent = capacity.join(SOURCES['invest'], on='generator', suffix='_rate')
                upper = min(upper, spent.select((pl.col('value') * pl.col('value_rate')).sum()).item() + sub.objective)
                appended(tables, 'cut', sub.objective - here, slope)

        if not dispatchable:
            # An infeasible solve has no duals to read — correctly, since its
            # values would be a vector of zeros indistinguishable from an
            # answer. So the violation is minimised instead, and *its* duals
            # say which way capacity has to move.
            with lps.solve(HERE / 'feasibility.yaml', {**DISPATCH, 'cap_hat': capacity}) as short:
                slope, here = slope_at(short, capacity)
                appended(tables, 'fcut', here - short.objective, slope)

        coordinates = {'cut': tables['cut_const']['cut'].to_list(), 'fcut': tables['fcut_const']['fcut'].to_list()}
        with lps.solve(HERE / 'master.yaml', {'invest': SOURCES['invest'], **tables}, coords=coordinates) as master:
            lower = master.objective
            capacity = master.primal('cap').select('generator', 'value')

        kind = 'optimality' if dispatchable else 'feasibility'
        bound = f'{upper:.2f}' if upper < float('inf') else 'none yet'
        print(f'  step {step}  {kind:11}  lower {lower:8.2f}   upper {bound}')
        # There is no gap to close until some capacity has proved dispatchable:
        # every feasibility cut leaves the upper bound at infinity.
        if upper < float('inf') and upper - lower <= 1e-6 * abs(upper):
            break

    print(f'\ndecomposed: {upper:.2f} in {step + 1} steps')
    print(f'monolithic: {truth:.2f}')
    print(f'difference: {abs(upper - truth):.1e}')
    print(f'cuts: {tables["cut_const"].height} optimality, {tables["fcut_const"].height} feasibility')


if __name__ == '__main__':
    main()
