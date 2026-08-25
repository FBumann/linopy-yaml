"""The relational lane: the YAML, the parquet, and a sink it never runs."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bench.cases import CASES

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bench.arms import Counts
    from bench.cases import Case


def checked_sources(case: Case, size: str, paths: dict[str, str]) -> dict[str, str]:
    """Every generated parquet, checked against what the model declares.

    Harness bookkeeping, and it runs *before* the clock: it re-parses the YAML
    only because the runner, not lpspec, decides which parquet file is which.

    A path the model declares nothing for is an error rather than a silent
    drop, which would leave the case measuring a build that never saw it. The
    way it happens is a stale parquet in the case's cache directory: the
    generator's output is globbed on a cache hit, so a file an older generator
    wrote outlives the declaration it was written for.
    """
    import yaml as pyyaml

    model = case.model_path(case.shape(size))
    schema = pyyaml.safe_load(model.read_text())
    declared = set().union(*(schema.get(block, {}) for block in ('parameters', 'dimensions', 'lookups')))
    undeclared = sorted(set(paths) - declared)
    if undeclared:
        raise ValueError(
            f'{case.name}: {undeclared} declared as neither parameter, dimension nor lookup in '
            f'{model} — the build would not see it. Stale files under bench/.cache/?'
        )
    return dict(paths)


def prepare(
    case_name: str, size: str, paths: dict[str, str], options: Mapping[str, Any]
) -> tuple[Path, dict[str, str]]:
    """The model to build and the sources to build it from, both already checked."""
    del options
    case = CASES[case_name]
    return case.model_path(case.shape(size)), checked_sources(case, size, paths)


def _tables(handle: Any) -> Any:
    """The built model's frames, wherever the checkout under test keeps them.

    ``build`` returns a handle *over* the engine; a checkout from before it
    returned the engine itself, and one from before ``BuiltModel`` kept the
    frames on the engine rather than on a value. Written the tolerant way for
    the same reason the nonzero count below is optional — the ladder is run
    across checkouts, and a comparison that cannot reach the older one measures
    nothing.
    """
    engine = getattr(handle, '_engine', handle)
    built = getattr(engine, '_model', None)
    return built.tables() if built is not None else engine._tables()


def _counts(tables: Any, *, nonzeros: bool) -> Counts:
    """The dims the published tables read.

    ``matrix`` is this engine's frame and an older checkout exposes its own
    shape, so the nonzero count stays optional — and a build with no sink has
    no assembled matrix to count at all.
    """
    matrix = getattr(tables, 'matrix', None) if nonzeros else None
    return {
        'columns': tables.column_count,
        'rows': tables.row_count,
        'nonzeros': getattr(matrix, 'height', None),
    }


def build_and_emit(sink: str, prepared: tuple[Path, dict[str, str]]) -> Counts:
    """Build relationally and hand the model over — an LP file, or a solver.

    ``run()`` / ``optimize()`` is never called. The simplex is the solver's work
    whoever filled the model, so timing it would swamp the phase this harness
    exists to measure and publish a number about HiGHS under our name.
    ``Model.to_highspy()`` is the same seam on linopy's side, which is the only
    reason the two arms are comparable.

    The counts are read after the action, so they are the harness's work and
    not the engine's.
    """
    import lpspec as lps

    model, sources = prepared
    with tempfile.TemporaryDirectory(prefix='lpspec-bench-') as tmp, lps.build(model, sources) as bound:
        if sink == 'lp':
            bound.write(Path(tmp) / 'model.lp')
        elif sink == 'gurobi':
            from lpspec.relational.sinks.solvers.gurobi import build_gurobi

            _handle = build_gurobi(_tables(bound))
        else:
            from lpspec.relational.sinks.solvers.highs import build_highs

            _handle = build_highs(_tables(bound))

        return _counts(_tables(bound), nonzeros=True)


def build_only(prepared: tuple[Path, dict[str, str]]) -> Counts:
    """Just the build — no sink, nothing to release."""
    import lpspec as lps

    model, sources = prepared
    with lps.build(model, sources) as bound:
        return _counts(_tables(bound), nonzeros=False)


def objective(prepared: tuple[Path, dict[str, str]]) -> float:
    """Solve, and return the objective the parity gate compares.

    The two lanes carry two axes: ``status`` is the coarse rollup (``'ok'``) and
    the solver's verdict is ``termination_condition`` (``'optimal'``). Checking
    the wrong one aborts every run with a parity failure that is really a
    vocabulary mismatch.
    """
    import lpspec as lps

    model, sources = prepared
    with lps.solve(model, sources) as sol:
        if sol.termination_condition != 'optimal':
            raise RuntimeError(f'lpspec solve terminated {sol.termination_condition!r}, not optimal')
        return float(sol.objective)
