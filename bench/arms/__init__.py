"""Every arm, and the four verbs the harness asks of one.

An *arm* is one library's answer to the same question: same parquet, same
model, same seam. The harness knows nothing about any of them beyond the names
below, so adding one is a module here and an entry in `ARMS` — not a branch in
the runner.

Each arm module defines:

    prepare(case_name, size, paths, options) -> Prepared
    build_and_emit(sink, prepared) -> Counts
    build_only(prepared) -> Counts
    objective(prepared) -> float

``Prepared`` is opaque to the harness: it hands the token from `prepare` to the
verb without looking inside, so an arm's own bookkeeping — validating paths,
resolving a writer backend — is described once and lands where it belongs.

**`prepare` is the pre-clock hook, and it is the reason it exists.** Whatever
an arm needs before it can build, that the *harness* rather than the library
imposed, is charged to nobody: the lpspec arm re-parses the case's YAML only
because the runner decides which parquet file is which, and the linopy arm has
no counterpart to be charged for it.

**Every verb is top-level and picklable**, because ``benchmem(isolate=True)``
sends it to a fresh process: peak RSS is a property of a process, and two
measurements in one interpreter report the larger of them twice.

**The library is imported inside the verb, never at module scope.** The import
is part of what an arm costs — a modelling library's alone can exceed lpspec's
entire build at the `xs` rung — so a harness that had already paid for it
before measuring would be charging one arm for another's work. That is also
why `ARMS` maps to modules rather than to imported callables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bench.arms import gurobipy_loop, gurobipy_matrix, linopy, lpspec

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType

#: What every verb returns: enough to prove the model is the right one, and the
#: counts the published tables carry. Read after the action, never during.
Counts = dict[str, Any]

#: Name to the module that speaks for it. Written out rather than discovered by
#: scanning: a misnamed module would go missing as an *absent arm*, which reads
#: as "not measured" rather than as the error it is.
ARMS: dict[str, ModuleType] = {
    'lpspec': lpspec,
    'linopy': linopy,
    'gurobipy-loop': gurobipy_loop,
    'gurobipy-matrix': gurobipy_matrix,
}


def unmeasurable(arm: str, case_name: str, sink: str) -> str | None:
    """Why this cell is not measured, or None when it is.

    A cell can be missing for three honest reasons — the arm's library is not
    installed here, the arm cannot reach that sink, or nobody has written the
    case in that arm's dialect — and all three are results. Returned as a sentence rather than a bool so the skip says which
    it was, and so a table can print the reason where a reader is looking.
    """
    import importlib.util

    module = ARMS[arm]
    absent = [r for r in getattr(module, 'REQUIRES', ()) if importlib.util.find_spec(r) is None]
    if absent:
        return f'{arm} needs {", ".join(absent)}, which this environment does not have'
    if sink not in module.SINKS:
        return f'{arm} does not reach the {sink} sink — it has {", ".join(module.SINKS)}'
    dialect = getattr(module, 'DIALECT', None)
    if dialect is not None:
        from bench.models import formulation

        if formulation(case_name, dialect) is None:
            return f'{case_name} has no {dialect} formulation'
    return None


def solved(arm: str, case_name: str, size: str, paths: dict[str, str], options: Mapping[str, Any]) -> float:
    """Prepare and solve on *arm* — what `bench.floor` checks its own answer against.

    Not a measurement — the one thing the harness does that is allowed to be
    slow, because a performance number describing two different models is worse
    than none.
    """
    module = ARMS[arm]
    return float(module.objective(module.prepare(case_name, size, paths, options)))
