"""pyomo, hand-written — the incumbent, and the baseline most readers already have.

Slow is the expected answer and not the point. A comparison that leaves pyomo
out looks chosen, and a reader who runs pyomo today wants to know what the
difference is in the units this harness measures rather than in an anecdote.

**The model is a `ConcreteModel` with `Set`/`Var`/`Constraint` rules**, which is
what pyomo's own documentation and every textbook using it writes. Each case's
is in `bench/models/<case>/pyomo.py`.

**All three sinks, through appsi's persistent interfaces.** `set_instance`
populates the solver's own model and stops there, which is the same seam
`build_highs` and `build_gurobi` reach — no `solve()` anywhere near a
measurement. The LP writer is pyomo's own.

**`symbolic_solver_labels` is left off**, which is pyomo's default: the labels
it would otherwise generate are the same feature `set_names=False` switches off
on the linopy arm, and the default is already the cheap side. Nothing is
switched off here that pyomo does not switch off itself.

**The parquet is read with pandas.** pyomo's own examples build their
`initialize=` mappings out of dicts and pandas frames; a polars read would
charge it for a hop its users do not take.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bench.arms import Counts

#: Every sink pyomo can hand a model to.
SINKS = ('lp', 'highs', 'gurobi')

#: What has to be importable for this arm to run. pyomo is in `dev` so the
#: default and bench environments carry it; the `codspeed` one deliberately does
#: not, and an absent library skips the cell with its reason rather than
#: erroring the run.
REQUIRES = ('pyomo',)

#: Which formulation module in `bench/models/<case>/` this arm builds from.
DIALECT = 'pyomo'


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
    """What pyomo has, counted the way pyomo counts.

    ``active=True`` is the honest reading: a `ConcreteModel` can carry
    deactivated blocks, and this harness's models do not, so the count is the
    model's own size rather than an assertion about how it was built.
    """
    from pyomo.environ import Constraint, Var

    columns = sum(len(v) for v in m.component_objects(Var, active=True))
    rows = sum(len(c) for c in m.component_objects(Constraint, active=True))
    return {'columns': columns, 'rows': rows, 'nonzeros': None}


def _persistent(sink: str) -> Any:
    from pyomo.contrib.appsi.solvers import Gurobi, Highs

    return Gurobi() if sink == 'gurobi' else Highs()


def build_and_emit(sink: str, prepared: Prepared) -> Counts:
    """Build the model and hand it over — an LP file, or a populated solver.

    ``set_instance`` is where appsi writes pyomo's expressions into the solver's
    own model, and it is the whole hand-off: nothing here calls ``solve``.
    """
    with tempfile.TemporaryDirectory(prefix='lpspec-bench-') as tmp:
        m = _built(prepared)
        if sink == 'lp':
            m.write(str(Path(tmp) / 'model.lp'))
        else:
            _handle = _persistent(sink)
            _handle.set_instance(m)
        return _counts(m)


def build_only(prepared: Prepared) -> Counts:
    """Just the build — no sink, nothing to release."""
    return _counts(_built(prepared))


def objective(prepared: Prepared) -> float:
    """Solve, and return what the arms are checked against."""
    from pyomo.contrib.appsi.base import TerminationCondition
    from pyomo.environ import Objective, value

    m = _built(prepared)
    result = _persistent('highs').solve(m)
    if result.termination_condition != TerminationCondition.optimal:
        raise RuntimeError(f'pyomo/appsi finished {result.termination_condition!r}, not optimal')
    return float(value(next(iter(m.component_objects(Objective, active=True)))))
