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
from data (the data-binding rules) and never changes; an iteration appends rows to their
parameter tables. No YAML is written at runtime, so the model a reviewer reads
is the model that runs — which is the point of writing models in YAML at all.

Because no file changes, each model is parsed **once** above the loop and
*built* once: ``lps.build`` binds the data and ``rebind`` puts the next
iteration's numbers on the model that is already there, where a path would
re-parse a model that cannot have moved and a rebuild would re-derive a model
that did not change. The subproblem's ``cap_hat`` reaches its rows as a
right-hand side, so HiGHS keeps the model it holds and re-solves from the last
basis; the master grows a row a step and is loaded again, which
``diagnostics().loads`` is what says. Any driver over a fixed model does this, whether it decomposes,
rolls a horizon or sweeps data.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import lpspec as lps

HERE = Path(__file__).parent
SNAPSHOTS = [0, 1, 2, 3]
GENERATORS = ['wind', 'gas']

SUB = lps.load_model(HERE / 'sub.yaml')
FEASIBILITY = lps.load_model(HERE / 'feasibility.yaml')
MASTER = lps.load_model(HERE / 'master.yaml')

SOURCES = {
    'snapshot': pl.DataFrame({'snapshot': SNAPSHOTS}),
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


def slice_for(model, **extra):
    """The part of ``SOURCES`` *model* declares, plus what this call adds.

    One bag of data and four models, each taking its own slice — `sub` reads a
    `cost` that `feasibility` does not, and the two are otherwise the same
    call. Binding refuses a name a model does not declare, so a driver over
    several models says which slice it means; that refusal is what turns a
    misspelled key into an error instead of a table nobody read.
    """
    known = {**model.parameters, **model.dimensions}
    return {name: frame for name, frame in {**SOURCES, **extra}.items() if name in known}


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
    """The decomposition loop, against the same problem solved in one plan.

    An infeasible subproblem has no duals to read — correctly, since its values
    would be a vector of zeros indistinguishable from an answer — so the
    violation is minimised instead and *its* duals say which way capacity has
    to move.

    The gap is only checked once some capacity has proved dispatchable: every
    feasibility cut leaves the upper bound at infinity.
    """
    with lps.solve(HERE / 'monolith.yaml', SOURCES) as whole:
        truth = whole.objective
    print(f'the whole problem, in one plan: {truth:.2f}\n')

    tables = dict(EMPTY)
    capacity = pl.DataFrame({'generator': GENERATORS, 'value': [0.0] * len(GENERATORS)})
    upper = float('inf')
    empty = {'cut': [], 'fcut': []}

    with (
        lps.build(SUB, slice_for(SUB, cap_hat=capacity)) as sub_model,
        lps.build(FEASIBILITY, slice_for(FEASIBILITY, cap_hat=capacity)) as short_model,
        lps.build(MASTER, {'invest': SOURCES['invest'], **tables, **empty}) as master,
    ):
        for step in range(25):
            sub = sub_model.rebind({'cap_hat': capacity}).solve()
            dispatchable = sub.has_primal
            if dispatchable:
                slope, here = slope_at(sub, capacity)
                spent = capacity.join(SOURCES['invest'], on='generator', suffix='_rate')
                upper = min(upper, spent.select((pl.col('value') * pl.col('value_rate')).sum()).item() + sub.objective)
                appended(tables, 'cut', sub.objective - here, slope)
            else:
                short = short_model.rebind({'cap_hat': capacity}).solve()
                slope, here = slope_at(short, capacity)
                appended(tables, 'fcut', here - short.objective, slope)

            coordinates = {'cut': tables['cut_const']['cut'].to_list(), 'fcut': tables['fcut_const']['fcut'].to_list()}
            answer = master.rebind({**tables, **coordinates}).solve()
            lower = answer.objective
            capacity = answer.primal('cap').select('generator', 'value')

            kind = 'optimality' if dispatchable else 'feasibility'
            bound = f'{upper:.2f}' if upper < float('inf') else 'none yet'
            print(f'  step {step}  {kind:11}  lower {lower:8.2f}   upper {bound}')
            if upper < float('inf') and upper - lower <= 1e-6 * abs(upper):
                break

        print(f'\ndecomposed: {upper:.2f} in {step + 1} steps')
        print(f'monolithic: {truth:.2f}')
        print(f'difference: {abs(upper - truth):.1e}')
        print(f'cuts: {tables["cut_const"].height} optimality, {tables["fcut_const"].height} feasibility')
        for name, model in (('the subproblem', sub_model), ('the master', master)):
            seen = model.diagnostics()
            print(f'{name} loaded the solver {seen.loads} time(s) in {seen.solves} solves')


if __name__ == '__main__':
    main()
