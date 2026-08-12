"""The ``highs`` solver: COO batches straight into HiGHS.

The default, and the only one whose dependency ships with the package. Columns
and rows arrive as numpy slices, in batches, with no float→text→parse round
trip — which is why this exists beside
:mod:`~lpspec.relational.sinks.writers.lp_file`.

**Nothing textual crosses into numpy**: a row's ``'<='`` becomes a
:data:`~lpspec.relational.sinks.tables.SENSE_CODES` byte before it is read
here, the rule
:meth:`~lpspec.relational.sinks.tables.ModelTables.dense_columns` measured.

``highspy`` is imported inside the function, being optional: importing this
module stays free for callers that only write LP files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lpspec.errors import LpspecError
from lpspec.relational.sinks.tables import SENSE_CODES, solver_vector
from lpspec.relational.status import SolveStatus

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl

    from lpspec.relational.sinks.tables import ModelTables


#: Elements per hand-off chunk. No *build-side* pass batches any more, labels
#: having become positional, so this is the sink's own budget rather than a
#: copy of a build knob — as the LP writer's ``EMIT_BUDGET`` is its. Spent
#: through
#: :mod:`~lpspec.relational.chunking`, which asks a caller to state what one
#: unit costs: a column is one element, a constraint row is as many as it has
#: nonzeros.
#:
#: Deliberately small. Both columns and rows are numpy slices, so more chunks
#: cost almost nothing and only residency scales with the budget — where an
#: engine whose every chunk re-ran an ordered query would want the opposite. A
#: wider budget buys a fraction of a second on a hand-off that precedes a
#: minute of simplex, and pays for it in a large fraction of the invariant this
#: budget exists to hold (#189).
HANDOFF_BUDGET = 100_000

#: HiGHS model status -> termination condition. Copied from linopy's own
#: ``Highs.CONDITION_MAP``; ``tests/test_solve_status.py`` asserts it still
#: matches, so a HiGHS release that adds a status shows up as a failure here
#: rather than as a silent ``unknown``.
_CONDITION_OF_HIGHS_STATUS = {
    'kNotset': 'unknown',
    'kLoadError': 'internal_solver_error',
    'kModelError': 'internal_solver_error',
    'kPresolveError': 'internal_solver_error',
    'kSolveError': 'internal_solver_error',
    'kPostsolveError': 'internal_solver_error',
    'kModelEmpty': 'unknown',
    'kMemoryLimit': 'resource_interrupt',
    'kOptimal': 'optimal',
    'kInfeasible': 'infeasible',
    'kUnboundedOrInfeasible': 'infeasible_or_unbounded',
    'kUnbounded': 'unbounded',
    'kObjectiveBound': 'terminated_by_limit',
    'kObjectiveTarget': 'terminated_by_limit',
    'kTimeLimit': 'time_limit',
    'kIterationLimit': 'iteration_limit',
    'kSolutionLimit': 'terminated_by_limit',
    'kInterrupt': 'user_interrupt',
    'kUnknown': 'unknown',
}


def build_highs(
    model: ModelTables,
    batch_rows: int | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> Any:
    """Load the model into a :class:`highspy.Highs` and stop there.

    The hand-off without the simplex, which is the same work whoever filled the
    model — so a measurement including it says nothing about the lane that
    filled it. `bench/` ends here, as linopy's ``Model.to_highspy()`` does on
    that side.

    Args:
        model: The built model, as every sink reads it.
        batch_rows: The budget in *elements*, spent through
            :mod:`~lpspec.relational.chunking`. The parameter stays so tests
            can force ragged chunks.
        solver_options: Set on the solver before anything is loaded.
    """
    import highspy
    import numpy as np

    batch = HANDOFF_BUDGET if batch_rows is None else batch_rows
    inf = highspy.kHighsInf
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    for option, value in (solver_options or {}).items():
        h.setOptionValue(option, value)

    empty_i = np.empty(0, dtype=np.int32)
    empty_f = np.empty(0, dtype=np.float64)
    lb, ub, cost, integral = model.dense_columns(inf)
    for lo, hi in model.col_chunks(batch):
        _loaded(h, h.addCols(hi - lo, cost[lo:hi], lb[lo:hi], ub[lo:hi], 0, empty_i, empty_i, empty_f), 'columns')
        noncontinuous = np.flatnonzero(integral[lo:hi]).astype(np.int32) + np.int32(lo)
        if len(noncontinuous):
            integrality = np.full(len(noncontinuous), int(highspy.HighsVarType.kInteger), dtype=np.uint8)
            h.changeColsIntegrality(len(noncontinuous), noncontinuous, integrality)

    sense, rhs = model.dense_rows(inf)
    rlb = np.where(sense == SENSE_CODES['<='], -inf, rhs)
    rub = np.where(sense == SENSE_CODES['>='], inf, rhs)
    for lo, hi, a, starts in model.row_blocks(batch):
        _loaded(
            h,
            h.addRows(
                hi - lo,
                rlb[lo:hi],
                rub[lo:hi],
                a.height,
                starts.astype(np.int32),
                a['col'].to_numpy().astype(np.int32, copy=False),
                a['coeff'].to_numpy(),
            ),
            'rows',
        )

    if model.objective_sense == 'max':
        h.changeObjectiveSense(highspy.ObjSense.kMaximize)
    return h


def solve_highs(
    model: ModelTables,
    batch_rows: int | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> tuple[SolveStatus, float, pl.Series | None, pl.Series | None]:
    """Feed the model to HiGHS and solve it.

    Returns:
        ``(status, objective, primal, dual)``, the last two the solver's own
        vectors — positional in its index, which *is* our label, so there is
        nothing to key them by and nothing to join.

        Either vector can be ``None``, for different reasons: no primal means
        the solve left nothing worth reading, while no dual is narrower, a
        mixed-integer model having none at all and neither does a run stopped
        short of a simplex basis. HiGHS hands back full-length vectors of
        zeros either way, and returning them would only make them reachable.
    """
    import highspy

    h = build_highs(model, batch_rows, solver_options)
    h.run()

    status = _status_of(h, highspy)
    if not status.is_readable:
        return status, float('nan'), None, None

    objective = h.getInfo().objective_function_value + model.objective_constant
    solution = h.getSolution()
    primal = solver_vector(solution.col_value)
    dual = solver_vector(solution.row_dual) if solution.dual_valid else None
    return status, objective, primal, dual


def _loaded(h: Any, status: Any, what: str) -> None:
    """Check that the solver accepted the batch.

    HiGHS reports a rejected batch by return value and carries on with an empty
    model, so an unchecked call turns a malformed hand-off into a confident
    answer to a different problem — an unconstrained one, if it was the rows.

    Raises:
        LpspecError: If the batch was refused.
    """
    import highspy

    if status == highspy.HighsStatus.kError:
        raise LpspecError(
            f'the solver refused a batch of {what}: {h.modelStatusToString(h.getModelStatus())!r}. '
            f'Nothing was loaded, so any answer would describe a different model. '
            f'This is an engine bug rather than a problem with the model — please report it.'
        )


def _status_of(h: Any, highspy: Any) -> SolveStatus:
    """What the solve concluded, on both axes.

    ``has_primal`` is the solver's own answer to "is there anything here",
    which the termination condition does not give: a run stopped at a time
    limit may or may not have found an incumbent.
    """
    model_status = h.getModelStatus()
    return SolveStatus(
        termination_condition=_CONDITION_OF_HIGHS_STATUS.get(str(model_status).rsplit('.', 1)[-1], 'unknown'),
        solver_wording=h.modelStatusToString(model_status),
        has_primal=h.getInfo().primal_solution_status == int(highspy.SolutionStatus.kSolutionStatusFeasible),
    )
