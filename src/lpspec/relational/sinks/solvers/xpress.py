"""The ``xpress`` solver: the model in two calls, straight into the Optimizer.

The same hand-off as :mod:`~lpspec.relational.sinks.solvers.highs`, reading the
same ``dense_columns``, ``dense_rows`` and ``row_blocks``, so no two sinks can
disagree about the model they load. What differs:

- **The matrix is row-major and stays that way.** ``addRows`` takes the CSR
  triple a block already is, so this sink is the HiGHS one's shape rather than
  the Gurobi one's — no ``scipy`` wrapper, and the extra carries nothing but
  ``xpress`` itself.
- **The objective's constant is a column.** Xpress spells it as the objective
  coefficient of column ``-1``, *negated*, where the other two have an
  attribute for it.
- **Forgetting is a control, not a call.** ``problem.reset()`` clears the whole
  problem here — it is not Gurobi's — so what discards the last solve's work is
  ``keepbasis``; see :meth:`Xpress.forget`.

``xpress`` is imported inside the functions, so importing this module stays
free for a caller who never solves with it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lpspec.relational.sinks.capabilities import Capabilities
from lpspec.relational.sinks.solvers.base import SolveAnswer, Solver, WarmStart
from lpspec.relational.sinks.tables import solver_vector, spelled_senses
from lpspec.relational.status import SolveStatus

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl

    from lpspec.relational.sinks.tables import ModelTables


#: Xpress solution status -> termination condition. Copied from linopy's own
#: ``Xpress.CONDITION_MAP``, which is the whole of theirs;
#: ``tests/test_solve_status.py`` asserts the copy, so linopy moving fails here
#: rather than silently. Keyed by the enum's *value*, the enum being an
#: optional import and this module level.
_CONDITION_OF_SOL_STATUS = {
    0: 'unknown',
    1: 'optimal',
    2: 'terminated_by_limit',
    3: 'infeasible',
    4: 'unbounded',
}

#: Which solution statuses carry values worth reading. ``FEASIBLE`` is an
#: incumbent found before the run stopped, so it does; ``NOTFOUND`` is the
#: case :attr:`~lpspec.relational.status.SolveStatus.is_readable` exists for.
_HAS_PRIMAL = frozenset({1, 2})

#: What this sink answers that linopy's map cannot, and why — an *addition*
#: rather than a divergence, the table above being copied entry for entry.
#: linopy reads ``solstatus`` alone, which cannot separate a solve that
#: *failed* from one that merely found nothing, so a solver error arrives as
#: ``unknown`` and a caller cannot tell it from a model nobody has solved. The
#: second axis is read here instead; the word stays linopy's vocabulary.
_BEYOND_LINOPY = {
    'solvestatus FAILED': 'a solve that errored reports internal_solver_error, where solstatus alone says unknown',
}

#: ``SolveStatus.FAILED`` and ``SolveStatus.UNSTARTED``, by value. The second
#: axis, read for two things ``solstatus`` cannot answer: whether the run
#: errored (:data:`_BEYOND_LINOPY`), and whether there has been a run at all —
#: which on this solver is the difference between a basis and a trivial one.
_SOLVE_UNSTARTED = 0
_SOLVE_FAILED = 2


def build_xpress(
    model: ModelTables,
    batch_rows: int | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> Any:
    """Load the model into an :class:`xpress.problem` and stop there.

    :func:`~lpspec.relational.sinks.solvers.highs.build_highs`'s seam, drawn
    for its reason: the search is the same work whoever filled the model.

    The problem owns its licence and releases it when it is collected, so
    unlike :func:`~lpspec.relational.sinks.solvers.gurobi.build_gurobi` this
    needs no finalizer — there is one object, and dropping it is the release.
    """
    return _built(model, batch_rows, solver_options)


class Xpress(Solver):
    """FICO Xpress, holding one model — :class:`Solver`'s member for the second opt-in sink.

    :class:`~lpspec.relational.sinks.solvers.highs.Highs`'s twin in how the
    model is handed over and :class:`~lpspec.relational.sinks.solvers.gurobi.Gurobi`'s
    in what it costs to hold. Three things are the Optimizer's shape rather
    than a choice:

    - **A push writes by index**, whole vectors through ``chgBounds`` /
      ``chgObj`` / ``chgRHS``. There is no handle to keep: the problem *is*
      the read-back.
    - **Nothing pushes a row's comparison.** A sense comes from the YAML and no
      data can move it, so a model whose senses differ is one
      :attr:`~lpspec.relational.sinks.tables.ModelTables.structure` has already
      sent back to be loaded again.
    - **Duals are refused rather than zero-filled** on a model that has none,
      as on Gurobi, so the refusal is the answer.
    """

    #: The loaded problem. One object, unlike Gurobi's four: ``close`` drops it
    #: and the licence goes with it.
    _p: Any

    #: One package, and it carries its own solver library.
    requires = ('xpress',)
    unavailable_message = 'The xpress sink requires the [xpress] extra: pip install "lpspec[xpress]"'

    #: Xpress branches on a set itself, which is the whole reason to declare
    #: one: no binaries, no big-M, and no bound a member has to have. The
    #: Optimizer takes a Hessian; **this sink does not hand it one**, and a
    #: descriptor says what the sink ingests rather than what the library
    #: could, so the quadratic entries are absent until something here writes
    #: them.
    capabilities = Capabilities(supports={'integrality': 'native', 'sos': 'native'})

    def _load(self, model: ModelTables, batch_rows: int | None) -> None:
        self._p = _built(model, batch_rows, self._options)

    def push(self, model: ModelTables) -> None:
        """Whole vectors by index, in three calls.

        Both bounds go in one ``chgBounds``: it takes a column per entry and a
        letter saying which bound, so the pair is one call over a doubled index
        rather than two over the model.
        """
        import numpy as np

        xpress = _xpress()
        cols = model.dense_columns(xpress.infinity)
        every = np.arange(model.column_count, dtype=np.int64)
        self._p.chgBounds(
            np.concatenate([every, every]),
            ['L'] * model.column_count + ['U'] * model.column_count,
            np.concatenate([cols.lb, cols.ub]),
        )
        self._p.chgObj(np.append(every, -1), np.append(cols.cost, -model.objective_constant))
        self._p.chgRHS(np.arange(model.row_count, dtype=np.int64), model.dense_rows(xpress.infinity).rhs)

    def warm_start(self) -> WarmStart | None:
        """The basis the last solve left, or its incumbent where that is not valid.

        **Asked of the problem, not caught from it.** Gurobi refuses ``VBasis``
        where no basis exists and the refusal routes the answer; Xpress hands
        back the trivial all-slack basis instead — before any solve, and after
        a mixed-integer one — so a member reading it that way would carry a
        start that means nothing and call it warm. The two questions are asked
        directly: has anything been solved, and is what is loaded a MIP.

        Xpress hands the basis back as ``(rows, columns)``, the opposite order
        to :class:`WarmStart`'s fields.
        """
        import numpy as np

        if int(self._p.attributes.solvestatus) == _SOLVE_UNSTARTED:
            return None
        if int(self._p.attributes.mipents):
            if int(self._p.attributes.solstatus) not in _HAS_PRIMAL:
                return None
            values = np.asarray(self._p.getSolution(), dtype=np.float64)
            return WarmStart(solver='xpress', column_statuses=None, row_statuses=None, column_values=values)
        rows, columns = self._p.getBasis()
        return WarmStart(
            solver='xpress',
            column_statuses=np.asarray(columns, dtype=np.int32),
            row_statuses=np.asarray(rows, dtype=np.int32),
            column_values=None,
        )

    def _warm(self, ws: WarmStart) -> None:
        """``loadBasis`` for a basis, ``addMipSol`` for an incumbent.

        ``keepbasis`` goes back on with the basis: :meth:`forget` is what turns
        it off, and a caller asking for a warm start after one has asked for
        the opposite of what that control says.
        """
        if (basis := ws.basis()) is not None:
            column_statuses, row_statuses = basis
            self._p.controls.keepbasis = 1
            self._p.loadBasis(row_statuses, column_statuses)
        else:
            assert ws.column_values is not None, (
                'a warm start with no basis carries an incumbent — it holds nothing else'
            )
            self._p.addMipSol(ws.column_values)

    def _run(self, model: ModelTables) -> SolveAnswer:
        """Solve what is loaded and read it back.

        The objective constant is already the loaded model's, so *model* is
        asked for nothing.
        """
        self._p.optimize()
        status = _status_of(self._p)
        if not status.is_readable:
            return SolveAnswer.unreadable(status)
        return SolveAnswer(
            status,
            float(self._p.attributes.objval),
            solver_vector(self._p.getSolution()),
            _duals(self._p),
            _activity(self._p),
        )

    def forget(self) -> None:
        """``keepbasis = 0``: the next solve ignores the basis this one left.

        Not ``problem.reset()``, which on Xpress clears the whole problem and
        is documented as returning it to the state it had at construction —
        the model would go with the solution. The control is what separates
        the two here, and it is durable rather than per-run on purpose: a
        caller taking the default ``keep='solver'`` calls this before every
        re-solve, and one asking for ``keep='progress'`` never calls it, so
        the flag already tracks the question. :meth:`_warm` turns it back on,
        that being a caller asking for the opposite.
        """
        self._p.controls.keepbasis = 0

    def close(self) -> None:
        """Release the problem, and the licence it holds.

        ``reset`` rather than dropping the reference: it is documented to clear
        everything the problem holds, where a dropped reference leaves the
        release to the collector — and a solver kept between solves holds a
        licence no frame in this process accounts for.
        """
        if self._p is not None:
            self._p.reset()
            self._p = None


def _built(
    model: ModelTables,
    batch_rows: int | None,
    solver_options: Mapping[str, Any] | None,
) -> Any:
    """The loaded problem, columns first and then the matrix a block at a time.

    Columns arrive with no entries — ``start`` is all zeros — because the
    matrix goes in row-wise afterwards, which is the form
    :meth:`~lpspec.relational.sinks.tables.ModelTables.row_blocks` already
    hands over. Loading it column-wise instead would mean sorting the model.

    ``chgColType`` is called only when some column is integral, for the reason
    the Gurobi sink skips ``vtype`` on an LP: an array of one repeated letter
    over every column is a cost an LP should not pay.

    ``outputlog`` leads the controls so a caller can put the log back.
    """
    import numpy as np

    xpress = _xpress()
    p = xpress.problem()
    p.setControl({'outputlog': 0, **dict(solver_options or {})})

    cols = model.dense_columns(xpress.infinity)
    p.addCols(
        objcoef=cols.cost,
        start=np.zeros(model.column_count + 1, dtype=np.int64),
        rowind=np.empty(0, dtype=np.int64),
        rowcoef=np.empty(0, dtype=np.float64),
        lb=cols.lb,
        ub=cols.ub,
    )
    if cols.integral.any():
        integral = np.flatnonzero(cols.integral)
        p.chgColType(integral, ['I'] * integral.size)

    rows = model.dense_rows(xpress.infinity)
    spelling = spelled_senses(_XPRESS_SENSE)
    for chunk in model.row_blocks(batch_rows):
        entries = chunk.entries
        p.addRows(
            rowtype=spelling[rows.sense[chunk.lo : chunk.hi]].tolist(),
            rhs=rows.rhs[chunk.lo : chunk.hi],
            start=np.append(chunk.starts, entries.height),
            colind=entries['col'].to_numpy(),
            rowcoef=entries['coeff'].to_numpy(),
        )

    _add_sets(p, model, xpress)
    if model.objective_sense == 'max':
        p.chgObjSense(xpress.maximize)
    if model.objective_constant:
        p.chgObj([-1], [-model.objective_constant])
    return p


def _add_sets(p: Any, model: ModelTables, xpress: Any) -> None:
    """Every special-ordered set, one ``addSOS`` call each.

    The one stream with no bulk form, as on Gurobi — a set is a call, its
    members a list of column indices and their weights.
    """
    if not model.sos.height:
        return
    for set_type, cols, weights in model.sets():
        p.addSOS(cols.to_list(), weights.cast(float).to_list(), type=set_type)


#: Our spelling of a comparison against the Optimizer's row types.
_XPRESS_SENSE = {'<=': 'L', '>=': 'G', '==': 'E'}


def _xpress() -> Any:
    """The optional dependency, or :attr:`Xpress.unavailable_message`."""
    return Xpress.imported()


def _status_of(p: Any) -> SolveStatus:
    """What the solve concluded, on both axes.

    ``solstatus`` carries the condition and whether anything is readable;
    ``solvestatus`` is read only to separate a solve that *errored* from one
    that found nothing, which is where this goes beyond linopy
    (:data:`_BEYOND_LINOPY`).
    """
    solution = int(p.attributes.solstatus)
    if int(p.attributes.solvestatus) == _SOLVE_FAILED:
        return SolveStatus('internal_solver_error', _wording(solution), has_primal=False)
    return SolveStatus(
        termination_condition=_CONDITION_OF_SOL_STATUS.get(solution, 'unknown'),
        solver_wording=_wording(solution),
        has_primal=solution in _HAS_PRIMAL,
    )


def _wording(solution: int) -> str:
    """Xpress's own name for a solution status.

    Read off the enum rather than tabulated — so one this package has never
    heard of still arrives searchable.
    """
    xpress = _xpress()
    names = {int(member): member.name for member in xpress.SolStatus}
    return names.get(solution, str(solution))


def _activity(p: Any) -> pl.Series:
    """Each row's left-hand side at the solution, in row order.

    Xpress exposes no row value of its own — only the slack, which is
    ``rhs - activity`` uniformly across senses, as on Gurobi — so the one
    subtraction recovers the solver's number.
    """
    import numpy as np

    slack = np.asarray(p.getSlacks(), dtype=np.float64)
    rhs = np.asarray(p.getRHS(), dtype=np.float64)
    return solver_vector(rhs - slack)


def _duals(p: Any) -> pl.Series | None:
    """Shadow prices in row order, or ``None`` where the model has none.

    Xpress refuses ``getDuals`` on a mixed-integer model, and that refusal *is*
    the answer — there is no zero vector to tell apart from real prices.
    """
    xpress = _xpress()
    try:
        return solver_vector(p.getDuals())
    except (xpress.SolverError, xpress.ModelError):
        return None
