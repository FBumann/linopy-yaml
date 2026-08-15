"""What is measured — one definition, for every instrument that measures it.

The harness is `pytest` now (`bench/test_ladder.py`), and the instruments are
plugins: `pytest-benchmark` times, `pytest-benchmem` adds a memray peak and —
under ``benchmem(isolate=True)`` — the whole-process ``rss`` the published
comparison is built on, and CodSpeed measures the same tests in CI. None of
them can share a *measurement*. All of them must share the **workload**, or two
numbers reported under one name describe different work.

Every verb here is **top-level and picklable**, because ``isolate=True`` sends
it to a fresh process: peak RSS is a property of a process, and two
measurements in one interpreter report the larger of them twice.

`lpspec` is imported *inside* the verbs, never at module scope. The import is
part of what an arm costs — linopy's alone exceeds lpspec's entire build at the
`xs` rung — so a harness that had already paid for it before measuring would be
charging one arm for the other's work.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bench.cases import CASES

if TYPE_CHECKING:
    from bench.cases import Case

#: What every verb returns: enough to prove the model is the right one, and the
#: counts the published tables carry. Read after the action, never during.
Counts = dict[str, Any]


def split_sources(case: Case, size: str, paths: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Parameters from dimension index tables, by what the model declares.

    Harness bookkeeping, and it runs *before* the clock on the lpspec arm: it
    re-parses the YAML only because the runner, not lpspec, decides which
    parquet file is which. The linopy arm has no counterpart — its own
    ``read_parquet`` and reshape are inside its build, where they belong.

    A path the model declares nothing for is an error rather than a silent
    drop, which would leave the case measuring a build that never saw it. The
    way it happens is a stale parquet in the case's cache directory: the
    generator's output is globbed on a cache hit, so a file an older generator
    wrote outlives the declaration it was written for.
    """
    import yaml as pyyaml

    model = case.model_path(case.shape(size))
    schema = pyyaml.safe_load(model.read_text())
    params = set(schema.get('parameters', {}))
    dims = set(schema.get('dimensions', {}))
    undeclared = sorted(set(paths) - params - dims)
    if undeclared:
        raise ValueError(
            f'{case.name}: {undeclared} declared as neither parameter nor dimension in '
            f'{model} — the build would not see it. Stale files under bench/.cache/?'
        )
    return (
        {k: v for k, v in paths.items() if k in params},
        {k: v for k, v in paths.items() if k in dims},
    )


def _tables(handle: Any) -> Any:
    """The built model's frames, wherever the checkout under test keeps them.

    ``build`` returns a handle *over* the engine; a checkout from before it
    returned the engine itself. Written the tolerant way for the same reason
    the nonzero count below is optional — the ladder is run across checkouts,
    and a comparison that cannot reach the older one measures nothing.
    """
    return getattr(handle, '_engine', handle)._tables()


def lpspec_build_and_emit(
    case_name: str, size: str, sink: str, sources: dict[str, str], coords: dict[str, str]
) -> Counts:
    """Build relationally and hand the model over — an LP file, or a solver.

    ``run()`` / ``optimize()`` is never called. The simplex is the solver's work
    whoever filled the model, so timing it would swamp the phase this harness
    exists to measure and publish a number about HiGHS under our name.
    ``Model.to_highspy()`` is the same seam on linopy's side, which is the only
    reason the two arms are comparable.

    The counts are read after the action, so they are the harness's work and
    not the engine's. ``matrix`` is this engine's frame and an older checkout
    exposes its own shape, so the nonzero count stays optional.
    """
    import lpspec as lps

    case = CASES[case_name]
    with (
        tempfile.TemporaryDirectory(prefix='lpspec-bench-') as tmp,
        lps.build(case.model_path(case.shape(size)), sources, coords=coords) as bound,
    ):
        if sink == 'lp':
            bound.write(Path(tmp) / 'model.lp')
        elif sink == 'gurobi':
            from lpspec.relational.sinks.solvers.gurobi import build_gurobi

            _handle = build_gurobi(_tables(bound))
        else:
            from lpspec.relational.sinks.solvers.highs import build_highs

            _handle = build_highs(_tables(bound))

        tables = _tables(bound)
        matrix = getattr(tables, 'matrix', None)
        return {
            'columns': tables.column_count,
            'rows': tables.row_count,
            'nonzeros': getattr(matrix, 'height', None),
        }


def linopy_build_and_emit(
    case_name: str, size: str, sink: str, paths: dict[str, str], io_api: str = 'lp-polars'
) -> Counts:
    """The same YAML, the same parquet, the same seam — on the eager lane.

    ``set_names=False`` is load-bearing. linopy names every variable and
    constraint by default and neither of our solver sinks names anything, so
    the default call would time a feature only one arm's model carries — and it
    is not a rounding error: naming is 82% of linopy's HiGHS hand-off and 35%
    of its Gurobi one.

    ``progress=False`` for the same reason in the other direction: linopy's
    default is ``m._xCounter > 10_000``, so every rung above `xs` would render
    tqdm bars the lpspec arm has no equivalent of — ~7% of the write at 10M
    variables.
    """
    from lpspec import linopy as lpspec_linopy

    case = CASES[case_name]
    with tempfile.TemporaryDirectory(prefix='lpspec-bench-') as tmp:
        data, coords = case.eager_inputs(paths)
        m = lpspec_linopy.build(case.model_path(case.shape(size)), data=data, coords=coords)
        if sink == 'lp':
            m.to_file(Path(tmp) / 'model.lp', io_api=io_api, progress=False)
        elif sink == 'gurobi':
            _handle = m.to_gurobipy(set_names=False)
        else:
            _handle = m.to_highspy(set_names=False)
        return {'columns': int(m.nvars), 'rows': int(m.ncons), 'nonzeros': None}


def build_only(arm: str, case_name: str, size: str, paths: dict[str, str]) -> Counts:
    """Just the build — no sink, nothing to release.

    The verb behind the *first vs steady* question: what a caller pays who
    builds one model, against what a rolling horizon pays for every model after
    it. Sink-free because a repeated write would conflate warm-up in the writer
    with warm-up in the engine.
    """
    case = CASES[case_name]
    model = case.model_path(case.shape(size))
    if arm == 'linopy':
        from lpspec import linopy as lpspec_linopy

        data, coords = case.eager_inputs(paths)
        m = lpspec_linopy.build(model, data=data, coords=coords)
        return {'columns': int(m.nvars), 'rows': int(m.ncons), 'nonzeros': None}

    import lpspec as lps

    sources, coords_ = split_sources(case, size, paths)
    with lps.build(model, sources, coords=coords_) as bound:
        tables = _tables(bound)
        return {'columns': tables.column_count, 'rows': tables.row_count, 'nonzeros': None}


def objective(arm: str, case_name: str, size: str, paths: dict[str, str]) -> float:
    """Solve, and return the objective the parity gate compares.

    Not a measurement — the one thing the harness does that is allowed to be
    slow, because a performance number describing two different models is worse
    than none.

    The two lanes carry two axes: ``status`` is the coarse rollup (``'ok'``)
    and the solver's verdict is ``termination_condition`` (``'optimal'``).
    Checking the wrong one aborts every run with a parity failure that is
    really a vocabulary mismatch.
    """
    case = CASES[case_name]
    model = case.model_path(case.shape(size))
    if arm == 'linopy':
        from lpspec import linopy as lpspec_linopy

        data, coords = case.eager_inputs(paths)
        m = lpspec_linopy.build(model, data=data, coords=coords)
        m.solve(solver_name='highs', output_flag=False)
        if m.status != 'ok':
            raise RuntimeError(f'linopy solve finished {m.status!r}, not ok')
        return float(m.objective.value)

    import lpspec as lps

    sources, coords_ = split_sources(case, size, paths)
    with lps.solve(model, sources, coords=coords_) as sol:
        if sol.termination_condition != 'optimal':
            raise RuntimeError(f'lpspec solve terminated {sol.termination_condition!r}, not optimal')
        return float(sol.objective)
