"""The Gurobi column of the sink-capability table, measured rather than read.

`test_sink_capability_probes.py`'s twin, and it exists because this column was
the *unverified* one — the table said its entries came "from the API and
linopy's `SolverFeature` table, and want a spike before they are relied on".

Every model is two columns wide, so it runs under the size-limited licence
gurobipy ships in its own wheel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

gurobipy = pytest.importorskip('gurobipy', reason='the gurobi sink needs the [gurobi] extra')

TABLE = 'docs/about/benchmarks.md, "Sink capabilities"'


@pytest.fixture
def model() -> Iterator[Callable[..., Any]]:
    """A silent, disposable model — one environment per test, since each holds a licence."""
    with gurobipy.Env(params={'OutputFlag': 0}) as env:
        made: list[Any] = []

        def build(**params: Any) -> Any:
            m = gurobipy.Model(env=env)
            for name, value in params.items():
                m.setParam(name, value)
            made.append(m)
            return m

        yield build
        for m in made:
            m.dispose()


def _solved(m: Any) -> float:
    m.optimize()
    assert m.Status == gurobipy.GRB.OPTIMAL, f'expected an optimal solve, got status {m.Status}'
    return float(m.ObjVal)


def test_gurobi_takes_a_convex_quadratic_objective(model):
    m = model()
    x = m.addVars(2, lb=0, ub=10)
    m.addConstr(x[0] + x[1] >= 2)
    m.setObjective(x[0] * x[0] + x[1] * x[1], gurobipy.GRB.MINIMIZE)
    assert _solved(m) == pytest.approx(2.0), f'{TABLE} says a quadratic objective is native on Gurobi'


def test_gurobi_takes_a_nonconvex_quadratic_objective_at_default_parameters(model):
    """`NonConvex=2` is the folklore answer and was not needed: the automatic
    default reaches spatial branch-and-bound by itself. Pinned because the
    refusal contract will name Gurobi, and "…if you set a parameter" is a
    different sentence."""
    m = model()
    x = m.addVars(2, lb=0, ub=10)
    m.addConstr(x[0] + x[1] >= 2)
    m.setObjective(-x[0] * x[0] - x[1] * x[1], gurobipy.GRB.MINIMIZE)
    assert _solved(m) == pytest.approx(-200.0), (
        f'{TABLE} says Gurobi solves a nonconvex quadratic objective, at default parameters. A '
        f'refusal here means the nonconvex row needs a parameter beside it.'
    )


def test_gurobi_takes_a_quadratic_objective_beside_integrality(model):
    m = model()
    x = m.addVars(2, lb=0, ub=10, vtype=gurobipy.GRB.INTEGER)
    m.addConstr(x[0] + x[1] >= 3)
    m.setObjective(x[0] * x[0] + x[1] * x[1], gurobipy.GRB.MINIMIZE)
    assert _solved(m) == pytest.approx(5.0), f'{TABLE} says MIQP is native on Gurobi'


def test_gurobi_takes_a_quadratic_constraint(model):
    m = model()
    x = m.addVars(2, lb=0, ub=10)
    m.addConstr(x[0] * x[1] >= 4)
    m.setObjective(x[0] + x[1], gurobipy.GRB.MINIMIZE)
    assert _solved(m) == pytest.approx(4.0), f'{TABLE} says a quadratic constraint is native on Gurobi'


def test_both_quadratic_parts_have_a_bulk_entry_point():
    """`addSOS` has no bulk form; the quadratic parts do, which is why the
    hand-off cost is about memory rather than call overhead."""
    missing = [name for name in ('setMObjective', 'addQConstr', 'addMQConstr') if not hasattr(gurobipy.Model, name)]
    assert missing == [], f'{TABLE} names these as the bulk quadratic entry points, and gurobipy lacks {missing}'
