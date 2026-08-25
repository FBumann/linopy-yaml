"""The ladder: one model, two lanes, one seam — measured by whichever plugin is loaded.

    pixi run refresh    # every rung, then both writers, in order — or by hand:
    pixi run -e bench pytest bench --benchmark-memory --benchmark-json=bench/results/latest.json \\
        --sizes xs s m l
    pixi run -e bench python -m bench.report bench/results/latest.json    # -> markdown
    pixi run -e bench python -m bench.plot                                # -> the chart page

Selection is `--cases / --sizes / --arms / --sinks` (see `conftest.py`), so the
published ladder and a one-rung smoke test are the same command with different
flags — and `-k` narrows further without any of them.

**Peak RSS is the published metric, and it needs `isolate=True`.** It is a
property of a *process*: a second arm in the same interpreter inherits the
first's high-water mark and its warm allocator. `isolate=True` is what gives a
fresh process per pass, and with it the whole-process `rss` beside the memray
peak — the two measure different things and both are recorded, because
`docs/about/benchmarks.md` publishes a cross-library claim and only `rss` is honest
across libraries. memray counts polars' reserved arenas as allocated and does
not count the interpreter at all, so the same pair of runs is 0.51x by RSS and
0.07x by memray. Within one lane that bias cancels; across two it does not.

**What is not measured, deliberately:** solve time (that is HiGHS, identical
either way, and it would swamp the build) and anything about expressiveness.
"""

from __future__ import annotations

from typing import Any

import pytest

from bench.arms import ARMS
from bench.conftest import shape_of


def _record(benchmark: Any, counts: dict[str, Any], case_name: str, size: str) -> None:
    """Attach the dims the published tables read, and check the model is the right one.

    A benchmark that silently built the wrong model is worse than none. With
    one arm there is nothing to compare an objective against, so this
    arithmetic check on every measurement is the whole of it.

    Written only when the fixture carries `extra_info` — CodSpeed's reports to a
    service rather than to a JSON file and has none, and an assertion that held
    under one instrument and raised under another is the failure this whole file
    is arranged to prevent.

    ``live_fraction`` is measured rather than declared: `dispatch` masks on a
    ``p_max`` that is always positive, so its ``where`` removes nothing and the
    engine pays for it anyway. ``variables`` is the real x of a scaling curve —
    ``size`` is a rung *label* and sorts alphabetically, where benchmem plots
    the numeric dimension.
    """
    shape = shape_of(case_name, size)
    assert 0 < counts['columns'] <= shape.nominal_variables
    info = getattr(benchmark, 'extra_info', None)
    if info is None:
        return
    info['columns'] = counts['columns']
    info['rows'] = counts['rows']
    info['nonzeros'] = counts['nonzeros']
    info['live_fraction'] = counts['columns'] / shape.nominal_variables
    info['variables'] = shape.nominal_variables


@pytest.mark.benchmem(isolate=True)
def test_emit(benchmark: Any, paths: Any, case_name: str, size: str, arm: str, sink: str) -> None:
    """Build the model and hand it over — an LP file on disk, or a populated solver.

    Both arms start from the same parquet and stop at the same seam, so each
    pays for its own data ingestion. That is the honest unit, and it is the only
    reason the two are comparable at all.

    ``checked_sources`` runs before the clock: it is harness bookkeeping, and the
    linopy arm has no counterpart to be charged for it.
    """
    if sink == 'gurobi':
        pytest.importorskip('gurobipy')

    module = ARMS[arm]
    prepared = module.prepare(case_name, size, paths(case_name, size), {})
    counts = benchmark(module.build_and_emit, sink, prepared)
    _record(benchmark, counts, case_name, size)


def test_rebuild(benchmark: Any, paths: Any, builds: int, case_name: str, size: str, arm: str) -> None:
    """First build against every later one, in one process.

    Two questions, two numbers. **First** is what a caller pays who builds one
    model and solves it — a fresh interpreter, and whatever lazy work each lane
    does on its first call lands here. **Steady** is what a rolling horizon pays
    for every model after the first. They differ by more than an order of
    magnitude on the eager lane, so a single figure would misreport one of the
    two use cases whichever it was.

    Deliberately **not** `isolate=True`, and deliberately sink-free: repeated
    builds in one process are the whole question, so a fresh process per pass
    would answer a different one — and a peak read here would be the high-water
    mark of five builds rather than of one.

    Not run under CodSpeed at all — its instruments ignore `rounds`, so there is
    no second build to compare the first against. `conftest.py` deselects it.
    """
    if builds < 1:
        pytest.skip('--builds 0')
    module = ARMS[arm]
    counts = benchmark.pedantic(
        module.build_only,
        args=(module.prepare(case_name, size, paths(case_name, size), {}),),
        rounds=builds,
        iterations=1,
        warmup_rounds=0,
    )
    _record(benchmark, counts, case_name, size)
