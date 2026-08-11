# ruff: noqa: T201  a spike reports by printing; it is not shipped code
"""The same loop, attempted through `solve_over`. Each wall is *run*, not argued.

`benders.py` shows the algorithm works on today's engine with a hand-written
loop. This file asks the question the spike exists for: how much of that loop
can `solve_over` carry, and what exactly stops it. Every section below executes
and prints the real failure.

**One of the five has come down since.** #586 made duals cross the seam, so
wall 2 now reports what it found rather than what it could not. The other four
stand, and each remains a correct decision for a flat fold.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing

import polars as pl

import lpspec as lps
from lpspec.strategy import Runs, solve_over
from spike.benders import GENERATORS, SOURCES

SUB_SOURCES = {k: v for k, v in SOURCES.items() if k != 'invest'}
CAP_ZERO = pl.DataFrame({'generator': GENERATORS, 'value': [0.0] * len(GENERATORS)})


def wall_1_the_axis_is_materialised_before_any_slice_runs() -> str:
    """A cut list is `list(axis)`-ed up front, so slice i+1 cannot depend on slice i.

    Demonstrated with a generator whose second cut needs the first cut's answer.
    Benders' whole shape is that dependency: cut i+1 exists *because* of solve i.
    """
    answers: list[float] = []

    def cuts():
        yield ('first', {**SUB_SOURCES, 'cap_hat': CAP_ZERO}, {})
        if not answers:
            raise RuntimeError('asked for cut 2 before cut 1 had been solved')
        yield ('second', {**SUB_SOURCES, 'cap_hat': CAP_ZERO}, {})

    try:
        solve_over('spike/sub.yaml', SUB_SOURCES, cuts(), key_name='k')
    except Exception as exc:
        return f'{type(exc).__name__}: {exc}'
    return 'no error — the axis was consumed lazily after all'


def wall_2_duals_now_cross_the_seam() -> str:
    """**No longer a wall.** #586 gave `Runs.dual` `Runs.primal`'s shape.

    When this spike was written a slice returned `result.primal(name)` and
    nothing else, so the one value a cut is built from could not leave the
    fold. It can now, keyed by slice and never combined.

    What is still missing is not the values. It is the *carry* — getting them
    back in as the next plan's parameter, which is wall 3.
    """
    full = {**SUB_SOURCES, 'cap_hat': CAP_ZERO}
    runs = solve_over('spike/sub.yaml', full, [('one', full, {})], key_name='k')
    prices = runs.dual('capacity')
    return (
        f'Runs.dual exists: {hasattr(Runs, "dual")}; '
        f'{prices.height} capacity prices, keyed by {runs.key_name!r}, columns {prices.columns}'
    )


def wall_3_carry_replaces_a_parameter_and_cannot_grow_one() -> str:
    """`carry` maps parameter -> (variable, index): a same-shape copy forward.

    A cut table does not replace, it *appends* — `cut` gains a member every
    iteration, so the parameter's own dimension grows. That is a different
    operation from the one `carry` performs, on a dimension `carry` has no way
    to extend.
    """
    try:
        solve_over(
            'spike/master.yaml',
            {'invest': SOURCES['invest']},
            [('one', {}, {})],
            carry={'cut_const': ('cap', None)},
            keep=('cap',),
            key_name='k',
        )
    except Exception as exc:
        return f'{type(exc).__name__}: {str(exc).splitlines()[0]}'
    return 'accepted — carry can grow a dimension after all'


def wall_4_sequential_outer_and_parallel_inner_is_refused() -> str:
    """The shape Benders actually has, and the one `solve_over` forbids.

    Iterations are sequential — cut i+1 needs solve i. Subproblems *within* an
    iteration (one per scenario, multi-cut L-shaped) are independent and belong
    on the pool. `carry` and `executor` together are exactly that, and they are
    refused at call time.
    """
    ctx = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(2, mp_context=ctx) as pool:
        try:
            solve_over(
                'spike/sub.yaml',
                SUB_SOURCES,
                [('a', {'cap_hat': CAP_ZERO}, {}), ('b', {'cap_hat': CAP_ZERO}, {})],
                carry={'cap_hat': ('p', None)},
                keep=('p',),
                executor=pool,
                key_name='k',
            )
        except Exception as exc:
            return f'{type(exc).__name__}: {str(exc).splitlines()[0]}'
    return 'accepted'


def wall_5_there_is_no_stopping_rule() -> str:
    """A fold visits every slice. Benders stops when the bound closes.

    Nothing in the signature expresses "keep going until", so the number of
    iterations would have to be guessed before the first solve.
    """
    import inspect

    parameters = list(inspect.signature(solve_over).parameters)
    convergence = [p for p in parameters if p in {'until', 'while_', 'stop', 'tolerance', 'max_iterations'}]
    return f'solve_over parameters: {parameters}\n    convergence-related: {convergence or "none"}'


def cost_of_rebuilding() -> str:
    """Need 5 from the issue, measured rather than asserted."""
    import time

    n = len(GENERATORS)
    cut_const = pl.DataFrame({'cut': [0], 'value': [0.0]})
    cut_slope = pl.DataFrame({'cut': [0] * n, 'generator': GENERATORS, 'value': [0.0] * n})
    sources = {'invest': SOURCES['invest'], 'cut_const': cut_const, 'cut_slope': cut_slope}

    best = float('inf')
    for _ in range(5):
        start = time.perf_counter()
        with lps.solve('spike/master.yaml', sources, coords={'cut': [0]}):
            pass
        best = min(best, time.perf_counter() - start)
    return f'one master solve at this toy size: {best * 1000:.1f} ms, all of it rebuild — nothing is reused between iterations'


if __name__ == '__main__':
    checks = [
        ('1. the axis is materialised before any slice runs', wall_1_the_axis_is_materialised_before_any_slice_runs),
        ('2. duals now cross the seam (no longer a wall)', wall_2_duals_now_cross_the_seam),
        ('3. carry replaces, it cannot grow a dimension', wall_3_carry_replaces_a_parameter_and_cannot_grow_one),
        ('4. sequential-outer / parallel-inner is refused', wall_4_sequential_outer_and_parallel_inner_is_refused),
        ('5. there is no stopping rule', wall_5_there_is_no_stopping_rule),
        ('cost of rebuilding per iteration', cost_of_rebuilding),
    ]
    for title, check in checks:
        print(f'\n=== {title}')
        try:
            print(f'    {check()}')
        except Exception as exc:  # a wall that fails differently is still a wall
            print(f'    unexpected {type(exc).__name__}: {str(exc).splitlines()[0]}')
