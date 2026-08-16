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

:class:`Highs` is the same hand-off held open — what a driver that re-solves
one model with new numbers uses, and where the warm basis lives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lpspec.errors import LpspecError, nonconvex_objective_message
from lpspec.relational.sinks.capabilities import Capabilities
from lpspec.relational.sinks.solvers.base import SolveAnswer, Solver, WarmStart
from lpspec.relational.sinks.tables import SENSE_CODES, solver_vector
from lpspec.relational.status import SolveStatus

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lpspec.relational.sinks.tables import ModelTables, RowVectors


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

    ``batch_rows`` is the budget in *elements*, spent through
    :mod:`~lpspec.relational.chunking`; the parameter stays so tests can force
    ragged chunks.
    """
    import highspy
    import numpy as np

    if model.qmatrix.height:
        raise LpspecError(
            'HiGHS has no quadratic-constraint concept at all — no entry point takes one — and '
            f'this model has {model.row_count - model.linear_row_count} such rows. Solving through '
            'lps.solve() '
            'refuses this earlier and names the sinks that do take it; reaching build_highs '
            'directly skips that, and loading the rows without their quadratic part would be a '
            'different model that solves.'
        )

    batch = HANDOFF_BUDGET if batch_rows is None else batch_rows
    inf = highspy.kHighsInf
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    for option, value in (solver_options or {}).items():
        h.setOptionValue(option, value)

    empty_i = np.empty(0, dtype=np.int32)
    empty_f = np.empty(0, dtype=np.float64)
    cols = model.dense_columns(inf)
    for lo, hi in model.col_chunks(batch):
        _loaded(
            h,
            h.addCols(hi - lo, cols.cost[lo:hi], cols.lb[lo:hi], cols.ub[lo:hi], 0, empty_i, empty_i, empty_f),
            'a batch of columns',
        )
        noncontinuous = np.flatnonzero(cols.integral[lo:hi]).astype(np.int32) + np.int32(lo)
        if len(noncontinuous):
            integrality = np.full(len(noncontinuous), int(highspy.HighsVarType.kInteger), dtype=np.uint8)
            h.changeColsIntegrality(len(noncontinuous), noncontinuous, integrality)

    rlb, rub = _row_bounds(model.dense_rows(inf), inf)
    for block in model.row_blocks(batch):
        _loaded(
            h,
            h.addRows(
                block.height,
                rlb[block.lo : block.hi],
                rub[block.lo : block.hi],
                block.entries.height,
                block.starts.astype(np.int32),
                block.entries['col'].to_numpy().astype(np.int32, copy=False),
                block.entries['coeff'].to_numpy(),
            ),
            'a batch of rows',
        )

    if model.objective_sense == 'max':
        h.changeObjectiveSense(highspy.ObjSense.kMaximize)
    _pass_hessian(h, model)
    return h


def _pass_hessian(h: Any, model: ModelTables) -> None:
    r"""The objective's quadratic part, as the Hessian HiGHS reads.

    ``passHessian`` takes :math:`Q` in :math:`\frac12 x^\top Q x`, lower
    triangle only, in column-major (CSC) order — so the conversion from the
    unordered-pair form the engine hands over is two rules, and they differ:

    * a **diagonal** pair states :math:`q\,x_i^2`, and :math:`\frac12 Q_{ii}
      x_i^2 = q\,x_i^2` needs :math:`Q_{ii} = 2q`;
    * an **off-diagonal** pair states :math:`q\,x_i x_j` once, where the
      symmetric matrix holds it twice — :math:`\frac12 (Q_{ij} + Q_{ji}) = q`
      — so the stored value is :math:`q` itself.

    The whole part goes over at once — there is no incremental Hessian API —
    but onto the model already loaded, which is what lets :meth:`Highs.push`
    replace it without a reload.
    """
    import highspy
    import numpy as np

    if not model.quad.height:
        return
    lower = model.quad['col_r'].to_numpy().astype(np.int32, copy=False)
    upper = model.quad['col_l'].to_numpy().astype(np.int32, copy=False)
    diagonal = lower == upper
    values = np.where(diagonal, model.quad['coeff'].to_numpy() * 2.0, model.quad['coeff'].to_numpy())

    order = np.lexsort((lower, upper))
    starts = np.zeros(model.column_count + 1, dtype=np.int32)
    np.add.at(starts, upper + 1, 1)
    _loaded(
        h,
        h.passHessian(
            model.column_count,
            len(order),
            int(highspy.HessianFormat.kTriangular),
            np.cumsum(starts, out=starts),
            lower[order],
            values[order],
        ),
        'the quadratic objective',
    )


class Highs(Solver):
    """HiGHS, holding one model — :class:`Solver`'s member for the default sink.

    What makes an iterative driver cheap. The second solve of a rebound model
    changes bounds, costs and right-hand sides on the model HiGHS already
    holds and starts from the basis the last solve ended on, where loading
    again would hand over the matrix a second time and start cold — unless
    the caller carries the basis across with :meth:`warm_start` and
    :meth:`~lpspec.relational.sinks.solvers.base.Solver.warm`.

    **Values are re-pushed, never diffed** — the previous model is *gone* by
    the time the new one exists, so there is nothing held to diff against;
    the trade is argued once, in ``../README.md``. Pushing the whole vectors
    costs a pass over the columns and the rows, against the matrix pass that
    loading would cost.
    """

    #: The loaded model. Declared rather than inferred, ``close`` dropping it.
    _handle: Any

    requires = ('highspy',)
    unavailable_message = 'highspy ships with lpspec, so a build without it is broken rather than missing an extra'

    #: No SOS concept at all, so a set arrives already written as binaries and
    #: linking rows. A *convex* Hessian goes in through ``passHessian``; the
    #: exclusions beside it are why this is a descriptor rather than a set of
    #: features, and the pair is probed in ``test_sink_capability_probes.py``.
    #: A set is that same refusal one step removed: the rewrite that gets one
    #: in here *is* binaries, so it cannot stand beside a Hessian either.
    capabilities = Capabilities(
        supports={
            'integrality': 'native',
            'sos': 'reformulated',
            'quadratic_objective': 'native',
        },
        excludes=(
            frozenset({'quadratic_objective', 'integrality'}),
            frozenset({'quadratic_objective', 'sos'}),
        ),
    )

    def _load(self, model: ModelTables, batch_rows: int | None) -> None:
        self._handle = build_highs(model, batch_rows, self._options)

    def push(self, model: ModelTables) -> None:
        """*model*'s bounds, costs and right-hand sides onto the loaded model.

        Everything a rebind may change without moving a label. The index
        vectors are built here rather than held, an ``arange`` being cheaper
        to make than to keep.
        """
        import highspy
        import numpy as np

        inf = highspy.kHighsInf
        cols = model.dense_columns(inf)
        columns = np.arange(model.column_count, dtype=np.int32)
        _loaded(self._handle, self._handle.changeColsCost(model.column_count, columns, cols.cost), 'new costs')
        _loaded(
            self._handle, self._handle.changeColsBounds(model.column_count, columns, cols.lb, cols.ub), 'new bounds'
        )

        rows = np.arange(model.row_count, dtype=np.int32)
        rlb, rub = _row_bounds(model.dense_rows(inf), inf)
        _loaded(self._handle, self._handle.changeRowsBounds(model.row_count, rows, rlb, rub), 'new right-hand sides')
        _pass_hessian(self._handle, model)

    def warm_start(self) -> WarmStart | None:
        """The basis the last solve left, or its incumbent where none is valid.

        A solved MIP is the model that holds an answer but no valid basis —
        ``getBasis().valid`` is false — so what crosses is ``col_value`` as an
        incumbent. A model not yet solved holds neither.
        """
        import highspy
        import numpy as np

        basis = self._handle.getBasis()
        if basis.valid:
            return WarmStart(
                solver='highs',
                column_statuses=np.fromiter((int(status) for status in basis.col_status), dtype=np.int8),
                row_statuses=np.fromiter((int(status) for status in basis.row_status), dtype=np.int8),
                column_values=None,
            )
        if self._handle.getInfo().primal_solution_status == int(highspy.SolutionStatus.kSolutionStatusFeasible):
            values = np.asarray(self._handle.getSolution().col_value, dtype=np.float64)
            return WarmStart(solver='highs', column_statuses=None, row_statuses=None, column_values=values)
        return None

    def _warm(self, ws: WarmStart) -> None:
        """``setBasis`` for a basis, ``setSolution`` for an incumbent.

        Both report a refusal by return value, like every hand-off here, so
        both go through :func:`_took` — an unchecked call would start cold and
        call it warm.
        """
        import highspy

        if ws.column_statuses is not None and ws.row_statuses is not None:
            basis = highspy.HighsBasis()
            basis.col_status = [highspy.HighsBasisStatus(int(status)) for status in ws.column_statuses]
            basis.row_status = [highspy.HighsBasisStatus(int(status)) for status in ws.row_statuses]
            basis.valid = True
            _took(self._handle.setBasis(basis), 'the carried basis')
        else:
            assert ws.column_values is not None, (
                'a warm start with no basis carries an incumbent — it holds nothing else'
            )
            solution = highspy.HighsSolution()
            solution.col_value = [float(value) for value in ws.column_values]
            _took(self._handle.setSolution(solution), 'the carried incumbent')

    def _run(self, model: ModelTables) -> SolveAnswer:
        """Solve, and read the one error HiGHS reports as a refusal to start.

        A ``kError`` from ``run()`` leaves the model status unset — there is no
        solve to read back — so a quadratic model that gets one is refused with
        the sentence the curvature earns rather than as an unreadable status.
        The pair a Hessian is otherwise refused for, integrality beside it, is
        declared on the descriptor and never reaches a load.
        """
        import highspy

        if self._handle.run() == highspy.HighsStatus.kError and model.quad.height:
            raise LpspecError(nonconvex_objective_message())
        status = _status_of(self._handle, highspy)
        if not status.is_readable:
            return SolveAnswer.unreadable(status)

        objective = self._handle.getInfo().objective_function_value + model.objective_constant
        solution = self._handle.getSolution()
        primal = solver_vector(solution.col_value)
        dual = solver_vector(solution.row_dual) if solution.dual_valid else None
        activity = solver_vector(solution.row_value)
        return SolveAnswer(status, objective, primal, dual, activity)

    def forget(self) -> None:
        """``clearSolver``: the basis and the solution go, the model stays.

        What this buys back is presolve. HiGHS skips it for a run that starts
        from a basis, so a model presolve can crack is one where keeping the
        answer is the slower path — and that is decided per model, which is
        why it is the caller's word and not a rule here.
        """
        self._handle.clearSolver()

    def close(self) -> None:
        """Release the loaded model. Idempotent."""
        if self._handle is not None:
            self._handle.clear()
        self._handle = None


def _row_bounds(rows: RowVectors, inf: float) -> tuple[Any, Any]:
    """HiGHS's ``(lower, upper)`` spelling of a sense code and right-hand side.

    The one rule for it, asked by the load and the push alike, so the two
    cannot drift: an inequality is open on the side its sense does not bound.
    """
    import numpy as np

    return (
        np.where(rows.sense == SENSE_CODES['<='], -inf, rows.rhs),
        np.where(rows.sense == SENSE_CODES['>='], inf, rows.rhs),
    )


def _loaded(h: Any, status: Any, what: str) -> None:
    """Raise unless the solver accepted the hand-off.

    HiGHS reports a rejected call by return value and carries on with whatever
    it had, so an unchecked call turns a malformed hand-off into a confident
    answer to a different problem — an unconstrained one, if it was the rows.

    Raises:
        LpspecError: If the batch was refused.
    """
    import highspy

    if status == highspy.HighsStatus.kError:
        raise LpspecError(
            f'the solver refused {what}: {h.modelStatusToString(h.getModelStatus())!r}. '
            f'The model it holds is not the one handed over, so any answer would describe a '
            f'different one. This is an engine bug rather than a problem with the model — '
            f'please report it.'
        )


def _took(status: Any, what: str) -> None:
    """Raise unless the solver accepted a warm-start hint.

    HiGHS reports a refusal by return value and carries on, and a dropped
    hint would not corrupt the model — the solve would just silently start
    cold, a wrong answer in the time dimension that the value dimension can
    never show.

    Raises:
        LpspecError: If the hint was refused.
    """
    import highspy

    if status == highspy.HighsStatus.kError:
        raise LpspecError(
            f'HiGHS refused {what} even though it spans the loaded model, so the solve would '
            f'silently start cold instead of warm. This is an engine bug rather than a problem '
            f'with the model — please report it.'
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
