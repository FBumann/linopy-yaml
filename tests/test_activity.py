"""Activity: a constraint row's left-hand side, read back after a solve.

The value is the solver's own — HiGHS hands ``row_value`` back in the same
``getSolution()`` call the primals come from — so a CSR recomputation agreeing
with it is an independent check of the whole chain: matrix, hand-off, solve,
read-back. Beside ``dual`` it is the constraint-side reader without the MILP
hole: an activity exists at any incumbent, where a mixed-integer model has no
dual solution at all.
"""

from __future__ import annotations

import numpy as np
import pytest

import charter as lps
from charter.errors import CharterError, NoSolutionError
from tests.differential import differential
from tests.oracle import pd  # through the guard: a bare import would beat it
from tests.test_milp import COMMITMENT_YAML

ACTIVITY_RTOL = 1e-9


def _csr_row_values(run) -> np.ndarray:
    """Every row's Ax at the solution, recomputed from the CSR block.

    ``np.add.reduceat(coeff * x[cols], row_starts[:-1])`` — nothing the solver
    produced except the primal vector itself, which is what makes agreement
    with `activity` an independent check rather than a tautology.
    """
    tables = run.engine._tables()
    x = np.zeros(tables.column_count)
    for name, block in run.engine._variable_blocks.items():
        x[block.start : block.start + block.height] = run.result.primal(name)['value'].to_numpy()
    products = tables.matrix['coeff'].to_numpy() * x[tables.matrix['col'].to_numpy()]
    assert (np.diff(tables.row_starts) > 0).all(), 'reduceat repeats on an empty row — pick a model without one'
    return np.add.reduceat(products, tables.row_starts[:-1])


def _agrees_with_csr(run) -> None:
    """Each constraint's activity against its slice of the recomputation."""
    recomputed = _csr_row_values(run)
    for name, block in run.engine._constraint_blocks.items():
        got = run.result.activity(name)['value'].to_numpy()
        assert got == pytest.approx(recomputed[block.start : block.start + block.height], rel=ACTIVITY_RTOL), (
            f'the solver judged feasibility against a different {name!r} LHS than the CSR block holds. '
            f'activity is Ax only while every constraint is linear: if quadratic constraints ever land, '
            f'it must grow the x^T Q x term (#563) — this disagreement is that invariant failing.'
        )


def test_activity_matches_the_csr_recomputation(dispatch_yaml, dispatch_inputs):
    """The solver's row values against Ax recomputed from the model's own CSR."""
    data, coords = dispatch_inputs
    with differential(dispatch_yaml, data, coords) as run:
        got = run.result.activity('power_balance')
        assert got.columns == ['snapshot', 'value']
        assert got.height == len(coords['snapshot'])
        _agrees_with_csr(run)


def test_activity_matches_the_eager_lane(dispatch_yaml, dispatch_inputs):
    """The linopy lane has no accessor, so its half is lhs evaluated at the solution."""
    data, coords = dispatch_inputs
    with differential(dispatch_yaml, data, coords) as run:
        oracle = run.model.constraints['power_balance'].lhs.solution
        expected = pd.Series(np.asarray(oracle), index=np.asarray(oracle.indexes['snapshot'])).sort_index()
        actual = run.result.activity('power_balance').sort('snapshot')['value'].to_numpy()
        assert actual == pytest.approx(expected.to_numpy(), rel=ACTIVITY_RTOL)


def test_a_milp_returns_activity_where_dual_refuses(commitment_inputs):
    """Activity is gated on `has_primal` alone: an integer incumbent has one.

    The constraint-side reader without the MILP hole — the same model on which
    `dual` must refuse (`tests/test_duals.py`) reads its activities back, and
    they agree with the CSR recomputation at the incumbent.
    """
    data, coords = commitment_inputs
    with differential(COMMITMENT_YAML, data, coords) as run:
        assert run.result.has_primal
        _agrees_with_csr(run)

        balance = run.result.activity('balance').sort('snapshot')['value'].to_numpy()
        assert balance == pytest.approx(data['load'].sort_index().to_numpy(), rel=ACTIVITY_RTOL), (
            'the == balance row is met exactly at any feasible incumbent, so its activity is the load'
        )
        with pytest.raises(CharterError, match='mixed-integer'):
            run.result.dual('balance')


def test_equality_row_activity_equals_rhs(dispatch_yaml, dispatch_frame_inputs):
    """On an `==` row activity equals the rhs up to solver tolerance, by construction."""
    data, coords = dispatch_frame_inputs
    with lps.solve(dispatch_yaml, data, coords=coords) as sol:
        got = sol.activity('power_balance').sort('snapshot')['value'].to_numpy()
        assert got == pytest.approx(data['load'].sort('snapshot')['value'].to_numpy(), rel=ACTIVITY_RTOL), (
            'an == row holds at the solution, so its activity is its rhs — a residual check, not a bug'
        )

        with pytest.raises(KeyError, match='unknown constraint'):
            sol.activity('zzz')


def test_infeasible_solve_refuses_activity(dispatch_yaml, dispatch_inputs):
    """No values at all is the refusal `primal` shares — same class, same gate."""
    data, coords = dispatch_inputs
    data = dict(data, load=pd.Series(1e6, index=coords['snapshot']))  # more than every generator together

    with lps.solve(dispatch_yaml, data, coords=coords) as result:
        assert not result.has_primal
        with pytest.raises(NoSolutionError, match='cannot read the activity'):
            result.activity('power_balance')


def test_a_closed_result_refuses_activity(dispatch_yaml, dispatch_frame_inputs):
    """close() releases the activity frames with the primal and dual ones."""
    data, coords = dispatch_frame_inputs
    with lps.solve(dispatch_yaml, data, coords=coords) as sol:
        pass
    with pytest.raises(CharterError, match='closed'):
        sol.activity('power_balance')
