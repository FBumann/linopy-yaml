"""The Gurobi column of the sink-capability table, measured rather than read.

`test_sink_capability_probes.py`'s twin, and it exists because this column was
the *unverified* one — the table said its entries came "from the API and
linopy's `SolverFeature` table, and want a spike before they are relied on".

Every model is two columns wide, so it runs under the size-limited licence
gurobipy ships in its own wheel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

gurobipy = pytest.importorskip('gurobipy', reason='the gurobi sink needs the [gurobi] extra')

from lpspec.relational.sinks import SOLVERS  # noqa: E402 — after the guard, or a bare install fails at import

TABLE = 'docs/about/benchmarks.md, "Sink capabilities"'

#: What these probes measure is acceptance, not precision: a spatial or mixed-integer
#: solve stops on ``MIPGap`` (1e-4 by default), so a vertex proved to that gap is a yes.
ACCEPTED = 1e-3


@pytest.fixture
def model() -> Iterator[Callable[[], Any]]:
    """A silent, disposable model — one environment per test, since each holds a licence."""
    with gurobipy.Env(params={'OutputFlag': 0}) as env:
        made: list[Any] = []

        def build() -> Any:
            m = gurobipy.Model(env=env)
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
    assert _solved(m) == pytest.approx(-200.0, rel=ACCEPTED), (
        f'{TABLE} says Gurobi solves a nonconvex quadratic objective, at default parameters. A '
        f'refusal here means the nonconvex row needs a parameter beside it.'
    )


def test_gurobi_takes_a_quadratic_objective_beside_integrality(model):
    m = model()
    x = m.addVars(2, lb=0, ub=10, vtype=gurobipy.GRB.INTEGER)
    m.addConstr(x[0] + x[1] >= 3)
    m.setObjective(x[0] * x[0] + x[1] * x[1], gurobipy.GRB.MINIMIZE)
    assert _solved(m) == pytest.approx(5.0, rel=ACCEPTED), f'{TABLE} says MIQP is native on Gurobi'


def test_gurobi_takes_a_quadratic_constraint(model):
    m = model()
    x = m.addVars(2, lb=0, ub=10)
    m.addConstr(x[0] * x[1] >= 4)
    m.setObjective(x[0] + x[1], gurobipy.GRB.MINIMIZE)
    assert _solved(m) == pytest.approx(4.0, rel=ACCEPTED), f'{TABLE} says a quadratic constraint is native on Gurobi'


def test_both_quadratic_parts_take_a_matrix_through_their_bulk_entry_point(model):
    """`addSOS` has no bulk form; the quadratic parts do, which is why the
    hand-off cost is about memory rather than call overhead. Measured through
    the matrix APIs the table names, since those are what this package would
    call and a shape they stop accepting is a cell going wrong."""
    m = model()
    x = m.addVars(2, lb=0, ub=10)
    columns = [x[0], x[1]]
    m.addMQConstr(np.array([[0.0, 1.0], [0.0, 0.0]]), None, gurobipy.GRB.GREATER_EQUAL, 4.0, columns, columns)
    m.setMObjective(np.eye(2), None, 0.0, columns, columns, sense=gurobipy.GRB.MINIMIZE)
    assert _solved(m) == pytest.approx(8.0, rel=ACCEPTED), (
        f'{TABLE} names setMObjective and addMQConstr as the bulk quadratic entry points. Q = I '
        f'over x0·x1 >= 4 minimises x0² + x1² at x = (2, 2), so 8.'
    )


def test_the_gurobi_descriptor_says_what_this_sink_does_with_what_it_measured():
    """The claim beside its evidence — and the gap between them, stated.

    `test_sink_capability_probes.py`'s twin, with one difference this column
    has and HiGHS's does not: **a descriptor describes the sink as shipped**,
    not the library it wraps. Everything above solved, and nothing here passes
    gurobipy a Hessian yet, so the quadratic entries are `absent` — claiming
    otherwise would drop the quadratic part of an objective and answer a
    different model's optimum.

    So this is two assertions, not one: what the sink does today, and that the
    gap is the hand-off rather than the solver.
    """
    capabilities = SOLVERS['gurobi'].capabilities
    assert capabilities.support('sos') == 'native', 'a set is loaded natively, and was probed above'
    assert capabilities.support('integrality') == 'native'
    assert capabilities.excludes == (), 'every combination probed above solved; nothing here is excluded'
    for capability in ('quadratic_objective', 'nonconvex_quadratic_objective', 'quadratic_constraint'):
        assert capabilities.support(capability) == 'absent', (
            f'gurobipy takes {capability} — the probes above measure it — but this sink does not '
            f'hand it one. When the hand-off lands, this line moves to native and the refusal '
            f'contract starts naming gurobi.'
        )
