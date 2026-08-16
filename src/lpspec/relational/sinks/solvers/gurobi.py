"""The ``gurobi`` solver: the model in two calls, straight into gurobipy.

The same hand-off as :mod:`~lpspec.relational.sinks.solvers.highs`, reading the
same ``dense_columns``, ``dense_rows`` and ``row_blocks``, so the two cannot
disagree about the model they load. Two things differ:

- **The matrix's currency.** HiGHS takes the three CSR arrays; gurobipy's
  matrix API takes a matrix *object*, so they are wrapped in a
  ``scipy.sparse.csr_matrix`` — a view, not a copy. That wrapper is why the
  ``[gurobi]`` extra carries scipy: the alternative is a Python call per row.
- **Nothing is batched.** The columns cannot be, since ``addMConstr`` writes
  into one ``MVar`` spanning the model — and the matrix *should* not be, which
  is where this sink parts company with the HiGHS one. See
  :meth:`~lpspec.relational.sinks.tables.ModelTables.row_blocks`.

``gurobipy`` and ``scipy`` are imported inside the functions, so importing
this module stays free for a caller who never solves with it.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

from lpspec.errors import LpspecError, nonconvex_row_message
from lpspec.relational.sinks.capabilities import Capabilities
from lpspec.relational.sinks.solvers.base import SolveAnswer, Solver, WarmStart
from lpspec.relational.sinks.tables import SENSE_CODES, solver_vector
from lpspec.relational.status import SolveStatus

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl

    from lpspec.relational.sinks.tables import ModelTables


#: Gurobi status -> termination condition. Copied from linopy's own
#: ``Gurobi.CONDITION_MAP`` bar three entries (:data:`_LINOPY_DIVERGENCES`);
#: ``tests/test_solve_status.py`` asserts both halves, so linopy moving — or
#: fixing — fails here rather than silently.
_CONDITION_OF_GUROBI_STATUS = {
    1: 'unknown',
    2: 'optimal',
    3: 'infeasible',
    4: 'infeasible_or_unbounded',
    5: 'unbounded',
    6: 'other',
    7: 'iteration_limit',
    8: 'terminated_by_limit',
    9: 'time_limit',
    10: 'terminated_by_limit',
    11: 'user_interrupt',
    12: 'other',
    13: 'suboptimal',
    14: 'unknown',
    15: 'terminated_by_limit',
    16: 'terminated_by_limit',
    17: 'resource_interrupt',
}

#: Where the table above does not copy linopy's, and why: each contradicts a
#: status Gurobi documents, so copying it would import a wrong answer rather
#: than a shared vocabulary — the trade
#: :attr:`~lpspec.relational.status.SolveStatus.is_readable` already refused
#: once. The words stay linopy's; only the verdicts differ.
_LINOPY_DIVERGENCES = {
    10: 'SOLUTION_LIMIT stopped early after n incumbents; linopy calls it optimal',
    16: 'WORK_LIMIT is a limit, not a solver failure; linopy calls it internal_solver_error',
    17: 'MEM_LIMIT is the resource_interrupt linopy itself maps kMemoryLimit to on HiGHS',
}


def build_gurobi(
    model: ModelTables,
    batch_rows: int | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> Any:
    """Load the model into a :class:`gurobipy.Model` and stop there.

    :func:`~lpspec.relational.sinks.solvers.highs.build_highs`'s seam, drawn
    for its reason: the search is the same work whoever filled the model.
    ``batch_rows`` is a *nonzero* budget that splits the matrix across calls;
    it defaults to one call — see
    :meth:`~lpspec.relational.sinks.tables.ModelTables.row_blocks` for why.

    **The caller owns the model, so the environment follows it.** gurobipy has
    no ``Model.getEnv()``, so a caller handed only the model could never
    release the licence it holds; a finalizer disposes the environment when the
    model is collected, which under refcounting is when the caller drops it.
    One thing to own rather than two, which is why this returns a model rather
    than a pair.
    """
    m, _x, _blocks, _quadratic, environment = _built(model, batch_rows, solver_options)
    weakref.finalize(m, environment.dispose)
    return m


class Gurobi(Solver):
    """Gurobi, holding one model — :class:`Solver`'s member for the opt-in sink.

    :class:`~lpspec.relational.sinks.solvers.highs.Highs`'s twin, and the same
    lifecycle. Three things are gurobipy's shape rather than a choice:

    - **A push writes through the read-back handles.** The ``MVar`` and the
      constraint blocks are what carry the attributes, so this keeps what
      :func:`_built` returns rather than the model alone — and it is why
      :func:`build_gurobi` is not reused: its finalizer disposes an
      environment a held solver ends explicitly.
    - **Nothing pushes ``Sense``.** A row's comparison comes from the YAML and
      no data can move it, so a model whose senses differ is one
      :attr:`~lpspec.relational.sinks.tables.ModelTables.structure` has already
      sent back to be loaded again. gurobipy would refuse the array anyway.
    - **``update`` before ``optimize``**, gurobipy's changes being queued.
    """

    #: The loaded model, the two handles that read it back, and the
    #: environment to release. Declared rather than inferred, ``close``
    #: dropping all four.
    _m: Any
    _x: Any
    _blocks: list[Any]
    #: The quadratic constraints, in row order and **after** every linear one:
    #: they are the tail of the label space, so the read-back concatenates
    #: rather than scatters.
    _qrows: list[Any]
    _env: Any

    #: Both halves of the extra, for the reason :attr:`unavailable_message`
    #: names both: the missing one is as often scipy.
    requires = ('gurobipy', 'scipy')
    unavailable_message = 'The gurobi sink requires the [gurobi] extra (gurobipy, scipy): pip install "lpspec[gurobi]"'

    #: Gurobi branches on a set itself, which is the whole reason to declare
    #: one: no binaries, no big-M, and no bound a member has to have.
    #:
    #: The only sink with no quadratic exclusion: a Hessian stands beside
    #: integrality, and a nonconvex one reaches spatial branch-and-bound at
    #: default parameters, both measured in
    #: ``tests/test_gurobi_capability_probes.py``.
    #:
    #: ``quadratic_constraint`` joined them when the stream that carries one
    #: did. This is the only consumer in the package that builds one at all —
    #: the linopy lane cannot (:data:`lpspec.api.LANES`).
    capabilities = Capabilities(
        supports={
            'integrality': 'native',
            'sos': 'native',
            'quadratic_objective': 'native',
            'nonconvex_quadratic_objective': 'native',
            'quadratic_constraint': 'native',
        }
    )

    def _load(self, model: ModelTables, batch_rows: int | None) -> None:
        self._m, self._x, self._blocks, self._qrows, self._env = _built(model, batch_rows, self._options)

    def push(self, model: ModelTables) -> None:
        """Whole vectors, in as many calls as there are blocks.

        The matrix API writes an attribute across an ``MVar`` or an
        ``MConstr`` at a time, so there is nothing here to batch that was not
        batched at the load.
        """
        gurobipy = _gurobipy()
        cols = model.dense_columns(gurobipy.GRB.INFINITY)
        self._x.LB, self._x.UB, self._x.Obj = cols.lb, cols.ub, cols.cost

        rhs = model.dense_rows(gurobipy.GRB.INFINITY).rhs
        at = 0
        for block in self._blocks:
            block.RHS = rhs[at : at + block.shape[0]]
            at += block.shape[0]
        self._m.ObjCon = model.objective_constant
        _set_quadratic(self._m, self._x, model, cols.cost)
        self._m.update()

    def warm_start(self) -> WarmStart | None:
        """The basis the last solve left, or its incumbent where Gurobi holds none.

        Gurobi refuses ``VBasis`` outright where no basis exists — after a
        mixed-integer solve, and before any — so the refusal itself routes to
        the incumbent, and to ``None`` where ``SolCount`` says there is not
        one of those either. Row statuses concatenate across the constraint
        blocks the way :func:`_duals` reads prices.
        """
        import numpy as np

        gurobipy = _gurobipy()
        try:
            columns = np.asarray(self._x.VBasis, dtype=np.int32)
            slices = [np.asarray(block.CBasis, dtype=np.int32) for block in self._blocks]
        except (AttributeError, gurobipy.GurobiError):
            if self._m.SolCount > 0:
                values = np.asarray(self._x.X, dtype=np.float64)
                return WarmStart(solver='gurobi', column_statuses=None, row_statuses=None, column_values=values)
            return None
        rows = np.concatenate(slices) if slices else np.empty(0, dtype=np.int32)
        return WarmStart(solver='gurobi', column_statuses=columns, row_statuses=rows, column_values=None)

    def _warm(self, ws: WarmStart) -> None:
        """``VBasis``/``CBasis`` for a basis, ``Start`` for an incumbent.

        Written through the same handles a push writes, the row statuses
        sliced per block the way a push slices the right-hand sides —
        and ``update`` after, gurobipy's changes being queued.
        """
        if ws.column_statuses is not None and ws.row_statuses is not None:
            self._x.VBasis = ws.column_statuses
            at = 0
            for block in self._blocks:
                block.CBasis = ws.row_statuses[at : at + block.shape[0]]
                at += block.shape[0]
        else:
            assert ws.column_values is not None, (
                'a warm start with no basis carries an incumbent — it holds nothing else'
            )
            self._x.Start = ws.column_values
        self._m.update()

    def _run(self, model: ModelTables) -> SolveAnswer:
        """Solve what is loaded and read it back.

        Gurobi refuses the attribute where there is no primal or no dual
        rather than handing back zeros, which is the one place it makes this
        easier than HiGHS.

        The one error translated here is the convexity refusal a *caller's own
        option* can provoke: ``QCPDual`` puts the solve on the convex path, so
        a nonconvex quadratic constraint that solves without it fails with it.
        Left alone that reaches the caller as a ``GurobiError`` naming a
        parameter they set for an unrelated reason.
        """
        gurobipy = _gurobipy()
        try:
            self._m.optimize()
        except gurobipy.GurobiError as exc:
            if 'not PSD' not in str(exc):
                raise
            raise LpspecError(nonconvex_row_message(str(exc))) from None
        status = _status_of(self._m)
        if not status.is_readable:
            return SolveAnswer.unreadable(status)
        return SolveAnswer(
            status,
            self._m.ObjVal,
            solver_vector(self._x.X),
            _duals(self._blocks, self._qrows),
            _activity(self._blocks, self._qrows),
        )

    def forget(self) -> None:
        """``Model.reset``: the solution and the basis go, the model stays.

        The default depth, which discards the solution without touching the
        parameters the caller set through ``solver_options`` — those are the
        model's configuration and outlive any one run.
        """
        self._m.reset()

    def close(self) -> None:
        """Release the model and the licence its environment holds.

        Both, and in that order: gurobipy has no ``Model.getEnv()``, so an
        environment left behind holds the licence until the collector reaches
        it — the hazard :func:`build_gurobi`'s finalizer exists for, and which
        a solver held between solves answers by ending explicitly.
        """
        if self._m is not None:
            self._m.dispose()
            self._env.dispose()
            self._m = self._x = self._env = None
            self._blocks = []
            self._qrows = []


def _built(
    model: ModelTables,
    batch_rows: int | None,
    solver_options: Mapping[str, Any] | None,
) -> tuple[Any, Any, list[Any], list[Any], Any]:
    """The model, the handles to read it back, and the environment to release.

    ``x.X`` and ``block.Pi`` are numpy arrays; ``getVars()``/``getConstrs()``
    would build one Python object per column and row for the same numbers.

    **Options go on the environment, not the model.** A licence parameter —
    ``WLSAccessID``, ``ComputeServer``, ``TokenServer`` — can only be set
    before an environment starts, and ``setParam`` on the model refuses it,
    so a Compute-Server or WLS user could not reach this sink at all. Nothing
    else is affected: an environment's parameters are the defaults of every
    model built on it. ``OutputFlag`` leads so a caller can put the log back.

    ``vtype`` is passed only when some column is integral — an LP otherwise
    pays a double-digit percentage of the column hand-off for an array of one
    repeated letter (#434), and linopy skips it the same way. ``batch_rows``
    goes straight through un-defaulted: one call unless a caller asks
    otherwise (#434).
    """
    gurobipy = _gurobipy()
    import numpy as np
    import scipy.sparse

    environment = gurobipy.Env(params={'OutputFlag': 0, **dict(solver_options or {})})
    m = gurobipy.Model(env=environment)

    cols = model.dense_columns(gurobipy.GRB.INFINITY)
    discrete: dict[str, Any] = {'vtype': np.where(cols.integral, 'I', 'C')} if cols.integral.any() else {}
    x = m.addMVar(model.column_count, lb=cols.lb, ub=cols.ub, obj=cols.cost, **discrete)

    rows = model.dense_rows(gurobipy.GRB.INFINITY)
    spelling = _spelled(gurobipy)
    blocks = []
    for chunk in model.row_blocks(batch_rows):
        entries = chunk.entries
        block = scipy.sparse.csr_matrix(
            (entries['coeff'].to_numpy(), entries['col'].to_numpy(), np.append(chunk.starts, entries.height)),
            shape=(chunk.height, model.column_count),
        )
        blocks.append(m.addMConstr(block, x, spelling[rows.sense[chunk.lo : chunk.hi]], rows.rhs[chunk.lo : chunk.hi]))

    _add_sets(m, x, model, gurobipy)
    quadratic = _add_quadratic_rows(m, x, model, gurobipy)
    if model.objective_sense == 'max':
        m.ModelSense = gurobipy.GRB.MAXIMIZE
    m.ObjCon = model.objective_constant
    _set_quadratic(m, x, model, cols.cost)
    m.update()
    return m, x, blocks, quadratic, environment


def _add_quadratic_rows(m: Any, x: Any, model: ModelTables, gurobipy: Any) -> list[Any]:
    r"""Every quadratic constraint, one ``addMQConstr`` call each.

    The second stream with no bulk form — ``addSOS`` is the first — and for the
    same reason it does not matter: the API takes one constraint per call, and
    a model with enough quadratic rows for that to cost anything is one no
    spatial branch-and-bound would finish.

    Each row is assembled from **both** matrices: its quadratic entries as
    :math:`Q` in :math:`x^	op Q x` (no halving, the convention
    :func:`_set_quadratic` already takes) and its linear entries from the
    ordinary matrix, where they sit at the same row label. A quadratic row
    keeps its place in the linear matrix precisely so that the two halves are
    read from one label rather than kept in step by hand.

    They are the **tail** of the label space, so the handles returned here
    concatenate onto the linear blocks and the read-back stays two runs rather
    than a scatter (:func:`_duals`).
    """
    import numpy as np
    import scipy.sparse

    spelling = _spelled(gurobipy)
    rows = model.dense_rows(gurobipy.GRB.INFINITY)
    added = []
    for row, pairs in model.quadratic_blocks():
        quadratic = scipy.sparse.csr_matrix(
            (pairs['coeff'].to_numpy(), (pairs['col_l'].to_numpy(), pairs['col_r'].to_numpy())),
            shape=(model.column_count, model.column_count),
        )
        entries = model.matrix_block(row, row + 1)
        linear = np.zeros(model.column_count, dtype=np.float64)
        linear[entries['col'].to_numpy()] = entries['coeff'].to_numpy()
        added.append(m.addMQConstr(quadratic, linear, spelling[rows.sense[row]], float(rows.rhs[row]), x, x, x))
    return added


def _set_quadratic(m: Any, x: Any, model: ModelTables, cost: Any) -> None:
    r"""The objective's quadratic part, as the matrix Gurobi reads.

    ``setMObjective`` takes :math:`Q` in :math:`x^\top Q x` — **no halving** —
    so the unordered-pair form the engine hands over goes in as it stands, one
    entry per pair in the upper triangle. Third of three conventions for one
    form: HiGHS halves and doubles its diagonal, the LP section doubles
    everything, and this one does neither.

    It sets the *whole* objective, so the cost vector already on the columns is
    passed again rather than overwritten with zeros — and that replacement is
    what makes a push safe, where accumulating would answer twice the curvature
    on the second solve. Nothing is called at all for an affine model.
    """
    import scipy.sparse

    if not model.quad.height:
        return
    pairs = scipy.sparse.csr_matrix(
        (
            model.quad['coeff'].to_numpy(),
            (model.quad['col_l'].to_numpy(), model.quad['col_r'].to_numpy()),
        ),
        shape=(model.column_count, model.column_count),
    )
    m.setMObjective(pairs, cost, model.objective_constant, x, x, x)


def _add_sets(m: Any, x: Any, model: ModelTables, gurobipy: Any) -> None:
    """Every special-ordered set, one ``addSOS`` call each.

    The one stream with no bulk form: ``addSOS`` takes a list of ``Var`` and
    their weights, so a set is a call and its members are Python objects. The
    ``MVar`` is sliced rather than ``getVars()`` walked, which keeps that cost
    proportional to the *members* — a model whose sets cover a corner of it
    pays for the corner.

    Nothing here is pushed on a rebind: a set is structure, so a model whose
    members moved is one
    :attr:`~lpspec.relational.sinks.tables.ModelTables.structure` has already
    sent back to be loaded again.
    """
    if not model.sos.height:
        return
    order = {1: gurobipy.GRB.SOS_TYPE1, 2: gurobipy.GRB.SOS_TYPE2}
    columns = x.tolist()
    for members in model.sos.partition_by('set', maintain_order=True):
        m.addSOS(
            order[members.item(0, 'type')],
            [columns[at] for at in members.get_column('col')],
            members.get_column('weight').to_list(),
        )


#: Our spelling of a comparison against Gurobi's, by ``GRB`` attribute name —
#: a name rather than a value because ``gurobipy`` is an optional import and
#: this is module level.
_GUROBI_SENSE = {'<=': 'LESS_EQUAL', '>=': 'GREATER_EQUAL', '==': 'EQUAL'}


def _spelled(gurobipy: Any) -> Any:
    """:data:`SENSE_CODES` as the characters ``addMConstr`` wants, by code.

    Built from the mapping rather than written out in its order: a wrong order
    is a model whose comparisons are silently permuted, which every solver
    answers confidently. A sense added to :data:`SENSE_CODES` and not to
    :data:`_GUROBI_SENSE` raises instead.
    """
    import numpy as np

    spelling = np.empty(len(SENSE_CODES), dtype='<U1')
    for sense, code in SENSE_CODES.items():
        spelling[code] = getattr(gurobipy.GRB, _GUROBI_SENSE[sense])
    return spelling


def _gurobipy() -> Any:
    """The optional dependency, or :attr:`Gurobi.unavailable_message`.

    The same sentence the early refusal prints, since it is the same fact
    arriving later.
    """
    try:
        import gurobipy
        import scipy.sparse  # noqa: F401 — guarded here so the message covers it
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(Gurobi.unavailable_message) from exc
    return gurobipy


def _status_of(m: Any) -> SolveStatus:
    """What the solve concluded, on both axes.

    ``SolCount`` answers "is there anything here", which the termination
    condition does not: a run stopped at a limit may or may not hold an
    incumbent.
    """
    code = int(m.Status)
    return SolveStatus(
        termination_condition=_CONDITION_OF_GUROBI_STATUS.get(code, 'unknown'),
        solver_wording=_wording(code),
        has_primal=m.SolCount > 0,
    )


def _wording(code: int) -> str:
    """Gurobi's own name for a status code.

    Read off ``GRB.Status`` rather than tabulated — so one this package has
    never heard of still arrives searchable.
    """
    gurobipy = _gurobipy()
    names = {getattr(gurobipy.GRB.Status, name): name for name in dir(gurobipy.GRB.Status) if not name.startswith('_')}
    return names.get(code, str(code))


def _activity(blocks: list[Any], qrows: list[Any]) -> pl.Series:
    r"""Each row's left-hand side at the solution, in row order.

    Gurobi exposes no row value of its own — only ``Slack``, which is
    ``rhs - activity`` uniformly across senses — so the one subtraction
    recovers the solver's number. ``Slack`` exists whenever a solution does,
    mixed-integer included, and a readable status guarantees one by the time
    this is asked. Blocks were added in ascending row ranges, the same fact
    :func:`_duals` leans on.

    **A quadratic row's activity is not** :math:`Ax`: ``QCSlack`` is measured
    against the whole left-hand side, :math:`x^\top Q x + a^\top x`, so the
    same subtraction returns what the row actually asserts. Nothing here
    recomputes it — the solver's own number is the one feasibility was judged
    against.
    """
    import numpy as np

    slices = [block.RHS - block.Slack for block in blocks]
    slices += [np.asarray([row.QCRHS - row.QCSlack], dtype=np.float64) for row in qrows]
    values = np.concatenate(slices) if slices else np.empty(0, dtype=np.float64)
    return solver_vector(values)


def _duals(blocks: list[Any], qrows: list[Any]) -> pl.Series | None:
    """Shadow prices in row order, or ``None`` where the model has none.

    Blocks were added in ascending row ranges and the quadratic rows after
    them, so concatenating their slices reproduces the row index without a
    sort — and :meth:`Solver.run` checks the vector spans the model. Gurobi
    refuses ``Pi`` on a mixed-integer model, and that refusal *is* the answer
    — no zero vector to test.

    A quadratic row prices through ``QCPi``, which exists **only under
    ``QCPDual``** — off by default and deliberately left off: asking for it
    puts the solve on the convex path, and a nonconvex row that solves without
    it then fails outright (measured: ``Constraint Q not PSD``). A caller who
    wants prices asks with ``solver_options={'QCPDual': 1}``; without it the
    attribute is refused, and that refusal *is* the answer.
    """
    import numpy as np

    gurobipy = _gurobipy()
    try:
        slices = [block.Pi for block in blocks]
        slices += [np.asarray([row.QCPi], dtype=np.float64) for row in qrows]
    except (AttributeError, gurobipy.GurobiError):
        return None
    values = np.concatenate(slices) if slices else np.empty(0, dtype=np.float64)
    return solver_vector(values)
