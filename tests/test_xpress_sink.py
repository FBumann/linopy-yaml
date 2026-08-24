"""The ``xpress`` sink, against the sink that was already here.

Three sinks loading one :class:`ModelTables` must produce the same model, so
HiGHS is the oracle for Xpress the way it is for Gurobi: the interesting
assertions are agreements, not values. Where a value *is* asserted it comes
from ``examples/ports/references.json`` — somebody else's published optimum,
which no sink can talk another into.

Every test skips without ``xpress``. Its wheel carries a Community licence
that is active on import — no file and no signup, which is what makes this
runnable in CI at all — and refuses a model whose rows *plus* columns exceed
5000. The models here stay well under; the one port that does not is named in
``OVER_THE_XPRESS_LIMIT`` and skipped rather than shrunk, the corpus being
checked against a published optimum for the whole model.

What this sink does not share with the other two is worth naming, since each
is a place a member could be wrong on its own: the objective's constant is a
*column* here, discarding a solve is a control rather than a call, and an
unsolved problem hands back a trivial basis where Gurobi refuses one.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

import lpspec as lps
from lpspec.errors import LpspecError, NoSolutionError
from lpspec.relational.sinks.solvers.xpress import Xpress, build_xpress
from tests.conftest import port_sources
from tests.test_gurobi_sink import CASES

xpress = pytest.importorskip('xpress', reason='the xpress sink needs the [xpress] extra')


@pytest.mark.parametrize(
    ('name', 'variable', 'constraint', 'has_duals'),
    [('LP', 'p', 'meet', True), ('MAX', 'p', 'lim', True), ('MIP', 'x', 'budget', False)],
)
def test_xpress_and_highs_agree(name: str, variable: str, constraint: str, has_duals: bool) -> None:
    """The claim a third solver has to earn, on all four quantities.

    ``test_gurobi_sink``'s cases, deliberately reused rather than restated:
    the models were chosen for what a sink states *outside* the frames — a
    maximisation, an objective constant, an integrality — which is exactly
    what a third member spells in its own third way.

    Coordinates as well as values, so a sink that loaded the columns in a
    different order fails here rather than reaching the same objective
    quietly. Activity is two derivations as well as two solvers: HiGHS reads
    its own ``row_value`` and this sink subtracts the slack from the
    right-hand side.
    """
    with lps.solve(*CASES[name]) as highs, lps.solve(*CASES[name], solver_name='xpress') as xp:
        assert xp.termination_condition == highs.termination_condition
        assert xp.objective == pytest.approx(highs.objective)

        expected, got = highs.primal(variable), xp.primal(variable)
        assert got.columns == expected.columns
        assert got.drop('value').equals(expected.drop('value'))
        assert got['value'].to_list() == pytest.approx(expected['value'].to_list())

        assert xp.activity(constraint)['value'].to_list() == pytest.approx(
            highs.activity(constraint)['value'].to_list()
        )
        if has_duals:
            assert xp.dual(constraint)['value'].to_list() == pytest.approx(highs.dual(constraint)['value'].to_list())


#: `osemosys_utopia` builds 10,857 rows plus columns against the Community
#: licence's 5,000. Nothing about the model is xpress-specific — it is skipped
#: for the licence, not for the answer.
OVER_THE_XPRESS_LIMIT = {'osemosys_utopia'}


def test_every_port_reaches_its_reference_optimum_on_xpress(port: dict[str, Any]) -> None:
    """``test_ports.py``'s corpus, solved by the third solver.

    The one assertion here no part of this package produced. A sink that
    mis-loads the matrix — a block boundary off by a row, a sense inverted —
    still reaches *a* number; this is what that number is checked against.
    """
    if port['name'] in OVER_THE_XPRESS_LIMIT:
        pytest.skip(f'{port["name"]} exceeds the bundled xpress licence — see OVER_THE_XPRESS_LIMIT')
    with lps.solve(port['model'], port_sources(port['name']), solver_name='xpress') as solution:
        assert solution.is_ok, f'{port["name"]} did not solve: {solution.status}'
        assert solution.objective == pytest.approx(port['objective'], rel=port['rtol'])


@pytest.mark.parametrize('batch_rows', [None, 1, 2, 10_000], ids=lambda n: f'batch-{n}')
def test_block_boundaries_do_not_move_the_answer(batch_rows: int | None) -> None:
    """The matrix goes in a block at a time, and a block is a slice of rows.

    A boundary that dropped or repeated a row's entries would still solve, so
    the budget is varied against a fixed answer rather than asserted about.
    """
    model, data = CASES['LP']
    with lps.build(model, data) as bound:
        tables = bound._engine._model.tables()
        reference = bound.solve().objective
    problem = build_xpress(tables, batch_rows=batch_rows)
    problem.optimize()
    assert float(problem.attributes.objval) == pytest.approx(reference), f'batch_rows={batch_rows} moved the answer'


def test_an_infeasible_solve_reports_both_axes_in_xpress_wording() -> None:
    """The status pair, and the solver's own word for it where a user reads it."""
    with lps.solve(*CASES['INFEASIBLE'], solver_name='xpress') as solution:
        assert solution.status == 'warning'
        assert solution.termination_condition == 'infeasible'
        assert not solution.has_primal
        assert solution.objective != solution.objective, 'nan, not 0.0'
        with pytest.raises(NoSolutionError, match='INFEASIBLE'):
            solution.primal('p')


def test_a_mixed_integer_model_has_no_duals() -> None:
    """Xpress refuses the read rather than handing back zeros, and the refusal
    is the answer — a zero vector would be indistinguishable from real prices."""
    with (
        lps.solve(*CASES['MIP'], solver_name='xpress') as solution,
        pytest.raises(LpspecError, match='mixed-integer'),
    ):
        solution.dual('budget')


def test_the_objective_constant_rides_on_the_model_not_the_answer() -> None:
    """The one quantity this sink spells as a *column*.

    Xpress carries it as the objective coefficient of column ``-1``, negated,
    where the other two sinks set an attribute — so a sign error here is a
    model that solves and answers wrong by a constant.
    """
    with lps.solve(*CASES['MAX'], solver_name='xpress') as solution:
        assert solution.objective == pytest.approx(12.0), 'cap 3 + cap 4, plus the declared 5'


def test_forgetting_makes_the_next_solve_start_cold() -> None:
    """``keepbasis``, and the reason it is not ``problem.reset()``.

    Reset on this solver clears the problem itself, so what the middle rung of
    ``keep=`` needs is the control. The observable is the iteration counter:
    a re-solve that kept the basis does no simplex work, and one that forgot
    it does the same work as the first.
    """
    from tests.test_warm_start import DISPATCH, SNAPSHOTS, dispatch_sources

    with lps.build(DISPATCH, dispatch_sources() | {'snapshot': SNAPSHOTS}) as bound:
        tables = bound._engine._model.tables()
    session = Xpress(tables)
    try:
        session.run(tables)
        cold = int(session._p.attributes.simplexiter)
        assert cold > 0, 'the model must make the simplex work, or this is unobservable'

        session.run(tables)
        assert int(session._p.attributes.simplexiter) == 0, 'a re-solve that kept the basis has nothing to do'

        session.forget()
        session.run(tables)
        assert int(session._p.attributes.simplexiter) == cold, 'forget() must send the next solve back to scratch'
    finally:
        session.close()


def test_solver_options_reach_xpress() -> None:
    """Forwarded verbatim, in the solver's own vocabulary — a control name here."""
    model, data = CASES['LP']
    with lps.build(model, data) as bound:
        tables = bound._engine._model.tables()
    problem = build_xpress(tables, solver_options={'timelimit': 42})
    assert int(problem.controls.timelimit) == 42, 'the option did not reach the problem'


def test_build_xpress_loads_the_model_and_stops() -> None:
    """The seam `bench/` measures: a loaded problem, unsolved."""
    model, data = CASES['LP']
    with lps.build(model, data) as bound:
        tables = bound._engine._model.tables()
    problem = build_xpress(tables)
    assert (problem.attributes.rows, problem.attributes.cols) == (tables.row_count, tables.column_count)
    assert int(problem.attributes.solvestatus) == 0, 'build_xpress loads the model and does not solve it'


def test_a_set_reaches_the_solver_natively() -> None:
    """``sos = 'native'``, so the family hands the sets over rather than the
    reformulation — asserted on the optimum a reformulation would also reach,
    plus the count the solver itself reports."""
    from tests.test_sos import DATA, best, model

    with lps.build(model(2), DATA) as bound:
        tables = bound._engine._model.tables()
    problem = build_xpress(tables)
    assert int(problem.attributes.sets) == 2, 'both declared sets reached the solver as sets'
    problem.optimize()
    assert float(problem.attributes.objval) == pytest.approx(best(2))


#: A model whose optimum leaves a row **slack**. Every case in ``CASES`` binds
#: every constraint, so ``rhs - slack`` and ``rhs`` agree there and the
#: subtraction is invisible — this is the model that separates them.
SLACK = {
    'dimensions': {'t': {'dtype': 'int', 'values': [0, 1]}},
    'parameters': {'cap': {'dims': ['t']}, 'price': {'dims': ['t']}},
    'variables': {'p': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 100}}},
    'constraints': {'lim': {'foreach': ['t'], 'expression': 'p <= cap'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * price, over=t) + 0'},
}

SLACK_DATA = {
    'cap': pl.DataFrame({'t': [0, 1], 'value': [10.0, 20.0]}),
    'price': pl.DataFrame({'t': [0, 1], 'value': [1.0, 2.0]}),
}


def test_activity_is_the_row_value_and_not_its_right_hand_side() -> None:
    """A non-binding row, which no case in ``CASES`` has.

    Xpress reports a *slack* and this sink subtracts it from the right-hand
    side to recover the row's own value. Every model above binds every row, so
    slack is zero and the subtraction cannot be seen — here the minimum parks
    both variables at 0 against caps of 10 and 20, so activity and rhs differ
    by the whole of each bound.
    """
    with lps.solve(SLACK, SLACK_DATA, solver_name='xpress') as solution:
        assert solution.activity('lim')['value'].to_list() == pytest.approx([0.0, 0.0]), (
            'activity is the row value at the solution, not the bound it was compared against'
        )


def test_a_solve_that_errored_is_not_reported_as_unknown() -> None:
    """The second axis, which linopy's map does not read.

    A unit probe rather than a solve: making the Optimizer *fail* is not
    something the suite can arrange reliably, and the branch is one line —
    what it has to answer is that a failure and a model nobody solved do not
    arrive as the same word.
    """
    from types import SimpleNamespace

    from lpspec.relational.sinks.solvers.xpress import _status_of

    failed = _status_of(SimpleNamespace(attributes=SimpleNamespace(solvestatus=2, solstatus=0)))
    assert failed.termination_condition == 'internal_solver_error'
    assert failed.status == 'error'
    assert not failed.has_primal

    unsolved = _status_of(SimpleNamespace(attributes=SimpleNamespace(solvestatus=0, solstatus=0)))
    assert unsolved.termination_condition == 'unknown', 'nothing solved yet is not a solver failure'


def test_the_missing_extra_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller without the package is told which extra to install, once —
    the same sentence whether the refusal lands early or at the import."""
    monkeypatch.setattr(Xpress, 'requires', ('xpress_not_installed',))
    assert not Xpress.is_available()
    with pytest.raises(ModuleNotFoundError, match=r'\[xpress\] extra'):
        lps.solve(*CASES['LP'], solver_name='xpress')
