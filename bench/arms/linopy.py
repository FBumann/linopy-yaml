"""linopy, hand-written — the arm a reader means when they see the name.

Not `lpspec.linopy`: that lane read our YAML and measured our own lowering on
top of linopy's work, which is why it was retired (#1268). Here the model is
typed out per case in `bench/models/<case>/linopy.py`, the way linopy's own
docs and this repo's `examples/ports/references/linopy/` write it — the same
scripts the gallery publishes, which is what makes this arm's formulations
reviewable rather than a strawman.

**The parquet is read with pandas**, not polars: linopy's inputs are pandas and
xarray, so a polars read plus a conversion would charge it for a hop its users
do not take. Each arm reads the way its own library expects.

**Three defaults are switched off, and they are load-bearing:**

- `set_names=False` on both solver hand-offs. linopy names every variable and
  constraint while our sinks name nothing, so the default would time a feature
  only one arm's model carries — naming is **82% of linopy's HiGHS hand-off**
  (0.11s against 0.02s at 200k variables) and 35% of its Gurobi one.
- `progress=False` on the LP writer. Its default is `m._xCounter > 10_000`, so
  every rung above `xs` renders tqdm bars no other arm draws — ~7% of the write
  at 10M variables, and stderr noise in a harness that parses stdout.
- `io_api='lp-polars'`, its fastest writer rather than its default one. The
  correction runs against us, which is the direction an honest harness errs in.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bench.arms import Counts

#: Every sink linopy can hand a model to — the same three as ours.
SINKS = ('lp', 'highs', 'gurobi')

#: What has to be importable for this arm to run. The `codspeed` environment
#: deliberately leaves `dev` out — it resolves linopy from a git branch and the
#: job measured no linopy model until this arm existed — so an absent library
#: has to skip the cell with its reason rather than error the run.
REQUIRES = ('linopy',)

#: Which formulation module in `bench/models/<case>/` this arm builds from.
DIALECT = 'linopy'


class Prepared(NamedTuple):
    case_name: str
    paths: dict[str, str]


def prepare(case_name: str, size: str, paths: dict[str, str], options: Mapping[str, Any]) -> Prepared:
    del size, options
    return Prepared(case_name, dict(paths))


def _built(prepared: Prepared) -> Any:
    """The model, parquet read included — every timed verb starts here."""
    import pandas as pd

    from bench.models import formulation

    tables = {name: pd.read_parquet(path) for name, path in prepared.paths.items()}
    return formulation(prepared.case_name, DIALECT).build(tables)


def _counts(m: Any) -> Counts:
    return {'columns': int(m.nvars), 'rows': int(m.ncons), 'nonzeros': None}


def build_and_emit(sink: str, prepared: Prepared) -> Counts:
    """Build the model and hand it over — an LP file, or a populated solver."""
    with tempfile.TemporaryDirectory(prefix='lpspec-bench-') as tmp:
        m = _built(prepared)
        if sink == 'lp':
            m.to_file(Path(tmp) / 'model.lp', io_api='lp-polars', progress=False)
        elif sink == 'gurobi':
            _handle = m.to_gurobipy(set_names=False)
        else:
            _handle = m.to_highspy(set_names=False)
        return _counts(m)


def build_only(prepared: Prepared) -> Counts:
    """Just the build — no sink, nothing to release."""
    return _counts(_built(prepared))


def objective(prepared: Prepared) -> float:
    """Solve, and return what the arms are checked against.

    ``status`` is linopy's coarse rollup; the solver's own verdict rides on
    ``termination_condition``, and checking the wrong one turns a vocabulary
    mismatch into a parity failure.
    """
    m = _built(prepared)
    m.solve(solver_name='highs', output_flag=False)
    if m.status != 'ok':
        raise RuntimeError(f'linopy solve finished {m.status!r}, not ok')
    return float(m.objective.value)
