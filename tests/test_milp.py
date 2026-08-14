"""vtype: a unit-commitment MILP through both backends.

Binary commitment variables u with p <= p_max * u and a fixed commitment
cost. Verifies the relational backend's vtype path end to end: cols vtype
column, HiGHS changeColsIntegrality in the `highs` solver, and the LP binary
section.
"""

from __future__ import annotations

from typing import get_args

import numpy as np
import polars as pl
import pytest

from lpspec.relational import plan
from tests.differential import differential

COMMITMENT_YAML = """
dimensions:
  snapshot: {dtype: int}
  generator: {dtype: str}

parameters:
  p_max: {dims: [generator]}
  cost: {dims: [generator]}
  fix_cost: {dims: [generator]}
  load: {dims: [snapshot]}

variables:
  u:
    foreach: [snapshot, generator]
    binary: true
  p:
    foreach: [snapshot, generator]
    bounds: {lower: 0, upper: p_max}

constraints:
  commitment:
    foreach: [snapshot, generator]
    expression: p <= p_max * u
  balance:
    foreach: [snapshot]
    expression: sum(p, over=generator) == load

objective:
  sense: minimize
  expression: sum(p * cost, over=generator) + sum(u * fix_cost, over=generator)
"""


@pytest.fixture
def commitment_run(commitment_inputs):
    """The commitment model solved through both lanes, engine still open."""
    data, coords = commitment_inputs
    with differential(COMMITMENT_YAML, data, coords) as run:
        yield run


def test_commitment_milp_agrees_and_stays_integral(commitment_inputs):
    """Both lanes agree, the binaries are integral, and the LP file says so."""
    data, coords = commitment_inputs

    with differential(COMMITMENT_YAML, data, coords, lp=True) as run:
        assert float(run.model.solution['u'].sum()) < run.model.solution['u'].size, (
            'u is not all-1 at the optimum, so commitment actually binds'
        )

        u = run.result.to_pandas('u')['value'].to_numpy()
        assert np.allclose(u, np.round(u), atol=1e-6), 'a binary variable takes an integral value'
        assert set(np.round(u)) <= {0.0, 1.0}, 'and that value is 0 or 1'

        assert 'binary' in run.lp.read_text(), 'the LP file carries integrality, not just bounds'


@pytest.mark.parametrize('batch_rows', [7, 13, 100_000], ids=['tiny-chunks', 'odd-chunks', 'one-chunk'])
def test_the_highs_solver_ingests_columns_in_order_whatever_the_chunking(commitment_run, batch_rows):
    """Columns reach HiGHS in label order however the range loop splits them.

    ``addCols`` appends, so column *k* must be the *k*-th row handed over. The
    sink used to get that from one ``ORDER BY c.col`` over the whole table — a
    global sort, which is the operator that does not stay inside
    ``memory_limit``. It now walks bounded ``col_chunks`` instead, which is
    only equivalent if every chunk is ordered *and* the chunks themselves are
    consecutive and gapless.

    A binary model is the sharp case: integrality is applied by column index,
    so a chunking bug relabels which variables are integral and the objective
    moves. Prime batch sizes make the last chunk short and stop a bug that
    only shows on ragged splits from hiding behind a round number.
    """
    chunked = commitment_run.engine.solve(batch_rows=batch_rows)
    assert chunked.is_ok
    assert chunked.objective == pytest.approx(commitment_run.oracle, rel=1e-9)

    u = chunked.primal('u')['value'].to_numpy()
    assert set(np.round(u)) <= {0.0, 1.0}, 'integrality landed on the wrong columns'


def test_cols_vtype_is_an_enum_over_every_declared_variable_type(commitment_run):
    """``cols.vtype`` is an Enum, and its members are ``plan.VariableType``.

    The storage choice is a performance one — one word per column, the same
    handful of words for the whole model, and the widest thing on the row as a
    string. The *members* are a contract: an Enum rejects a value outside it,
    so a fourth variable type added to the plan and not reaching the column
    fails where the column is built rather than in whichever sink first
    compares against a name it does not know.
    """
    vtype = commitment_run.engine._tables().cols.schema['vtype']
    held = set(commitment_run.engine._tables().cols['vtype'].unique().to_list())

    assert isinstance(vtype, pl.Enum), f'vtype is {vtype}, so it stores a word per column'
    assert set(vtype.categories.to_list()) == set(get_args(plan.VariableType))
    assert held == {'continuous', 'binary'}, 'this model declares both, so both must survive the stack'
