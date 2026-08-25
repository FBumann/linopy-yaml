"""The eager lane: the same YAML, through `lpspec.linopy`, into linopy's own model.

This arm is the *oracle* rather than a rival dialect: it accepts exactly the
same model file (docs/about/architecture.md hard rule 3), so it cannot silently
be measured on a different model. Every other arm has to earn that by parity.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from bench.cases import CASES

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bench.arms import Counts


class Prepared(NamedTuple):
    """What the timed verbs need. The parquet is *not* read here.

    Reading it is the eager arm's own cost and belongs inside its build — that
    is how linopy is actually used, and handing the verb data already in memory
    would flatter it.
    """

    case_name: str
    model: Path
    paths: dict[str, str]
    io_api: str


def prepare(case_name: str, size: str, paths: dict[str, str], options: Mapping[str, Any]) -> Prepared:
    """The model, the parquet paths, and which LP writer backend to use."""
    case = CASES[case_name]
    return Prepared(case_name, case.model_path(case.shape(size)), dict(paths), str(options.get('io_api', 'lp-polars')))


def _build(prepared: Prepared) -> Any:
    """The model, data read included — every timed verb starts here."""
    from lpspec import linopy as lpspec_linopy

    data = CASES[prepared.case_name].eager_inputs(prepared.paths)
    return lpspec_linopy.build(prepared.model, data)


def _counts(m: Any) -> Counts:
    return {'columns': int(m.nvars), 'rows': int(m.ncons), 'nonzeros': None}


def build_and_emit(sink: str, prepared: Prepared) -> Counts:
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
    with tempfile.TemporaryDirectory(prefix='lpspec-bench-') as tmp:
        m = _build(prepared)
        if sink == 'lp':
            m.to_file(Path(tmp) / 'model.lp', io_api=prepared.io_api, progress=False)
        elif sink == 'gurobi':
            _handle = m.to_gurobipy(set_names=False)
        else:
            _handle = m.to_highspy(set_names=False)
        return _counts(m)


def build_only(prepared: Prepared) -> Counts:
    """Just the build — no sink, nothing to release."""
    return _counts(_build(prepared))


def objective(prepared: Prepared) -> float:
    """Solve, and return the objective the parity gate compares.

    The two lanes carry two axes: ``status`` is the coarse rollup (``'ok'``) and
    the solver's verdict is ``termination_condition`` (``'optimal'``). Checking
    the wrong one aborts every run with a parity failure that is really a
    vocabulary mismatch.
    """
    m = _build(prepared)
    m.solve(solver_name='highs', output_flag=False)
    if m.status != 'ok':
        raise RuntimeError(f'linopy solve finished {m.status!r}, not ok')
    return float(m.objective.value)
