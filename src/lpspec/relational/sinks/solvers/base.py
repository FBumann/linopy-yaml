"""What every solver is: a loaded model, and the rule for keeping it.

A solver sink holds the model it was given and outlives the solve it was loaded
for, so that a rebound model (:meth:`~lpspec.api.BoundModel.rebind`) has its new
numbers *pushed* onto what the solver already has and re-solves from the basis
the last one ended on. linopy's shape, and its word: their ``Solver`` is the
persistent object too, and a one-shot solve is one you throw away
(``solve_<name>`` here). Copied rather than imported, and tested here.

**This module imports no solver.** It is the one thing ``solvers/`` members may
read besides ``tables.py``, and that is the whole reason it can exist: the fence
`tests/test_architecture.py` draws keeps ``gurobipy`` off the import path of a
caller who solves with HiGHS, and a base that reaches for neither cannot carry
one across. Sharing through it is what stops the alternative — one leaf importing
the other — from ever being the tempting option.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from lpspec.errors import LpspecError

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl

    from lpspec.relational.sinks.tables import ModelTables
    from lpspec.relational.status import SolveStatus

    #: What a solve concluded, and the two vectors it left: the status pair,
    #: the objective, the primal and the dual. Either vector may be ``None``,
    #: for different reasons — see :meth:`Solver.run`.
    Answer = tuple[SolveStatus, float, pl.Series | None, pl.Series | None]


class Solver(ABC):
    """One solver, holding one model. Subclassed once per member of ``SOLVERS``.

    The two halves are split by who can answer them. **This class owns the
    rule** — whether the loaded model may be kept for another solve — because
    it is a property of the *tables*, identical for every solver, and a second
    copy of it would be a second answer that could drift. **A subclass owns the
    hand-off**: loading, pushing values, running, releasing, all of which are
    its own library's shape and nothing else's.

    The lifecycle a driver walks, and the only order that is defined::

        solver = Solver(tables, batch_rows, options)  # loaded
        solver.run(tables)  # …repeatedly
        solver.remember(tables)  # before the tables go
        if solver.takes(tables, options):  # after they are rebuilt
            solver.push(tables)
        solver.close()

    :meth:`takes` is asked of every reuse and is the correctness floor: a model
    whose :meth:`~lpspec.relational.sinks.tables.ModelTables.structure` moved is
    a different model wearing the same labels, and pushing values onto it would
    answer a question nobody asked.
    """

    def __init__(
        self,
        model: ModelTables,
        batch_rows: int | None = None,
        solver_options: Mapping[str, Any] | None = None,
    ) -> None:
        self._options = dict(solver_options or {})
        #: The structure of the model this holds, recorded when something
        #: rebuilt the tables under it. ``None`` while nothing has, which is
        #: every solve of a model nobody rebound — and the reason a one-shot
        #: solve pays for no digest.
        self._structure: bytes | None = None
        self._load(model, batch_rows)

    def remember(self, model: ModelTables) -> None:
        """Record what is loaded, before the frames it was loaded from go.

        The last moment anything can say what the solver holds: a rebuild
        releases the previous model *before* it starts, which is what keeps a
        rebound build at one model's peak, so nothing afterwards could compare
        against it.
        """
        self._structure = model.structure()

    def takes(self, model: ModelTables, solver_options: Mapping[str, Any] | None) -> bool:
        """Whether *model* is the loaded one differing in nothing but numbers.

        Options count: they are set when the model is loaded, so a solve asking
        for others has to be given something that was told them.
        """
        if dict(solver_options or {}) != self._options:
            return False
        return self._structure is None or self._structure == model.structure()

    @abstractmethod
    def _load(self, model: ModelTables, batch_rows: int | None) -> None:
        """Hand *model* to the solver and hold whatever reads it back.

        Called by ``__init__`` rather than by a caller, so that a subclass
        cannot exist in a state where the other four have nothing to work on.
        """

    @abstractmethod
    def push(self, model: ModelTables) -> None:
        """*model*'s bounds, costs and right-hand sides onto the loaded model.

        Everything a rebind may change without moving a label, and only ever
        after :meth:`takes` said so. Whole vectors rather than a diff: the
        model that would say which cells moved is the one this replaces.
        """

    def run(self, model: ModelTables) -> Answer:
        """Solve what is loaded, read it back, and refuse a vector that lies.

        Reading a solution back is positional, so a vector that does not span
        the model is an answer about a *different* one. Refused here, where the
        solver hands it over, rather than where it is read: the objective comes
        back directly, so a result built on a broken vector would report a
        plausible number and only fail if someone asked for a coordinate.

        A member writes :meth:`_run`; this is the contract around it, so no
        sink can be added that forgets to be checked.
        """
        status, objective, primal, dual = self._run(model)
        self._spans('primal', primal, model.column_count)
        self._spans('dual', dual, model.row_count)
        return status, objective, primal, dual

    def _spans(self, quantity: str, values: pl.Series | None, expected: int) -> None:
        """``None`` is not a wrong length — a mixed-integer model has no duals,
        and neither does a run stopped short of a simplex basis."""
        if values is not None and len(values) != expected:
            raise LpspecError(
                f'{type(self).__name__} returned {len(values)} {quantity} values for a model with '
                f'{expected}. Reading a solution back is positional, so a vector that does not span '
                f'the model describes a different one. This is an engine bug rather than a problem '
                f'with the model — please report it.'
            )

    @abstractmethod
    def _run(self, model: ModelTables) -> Answer:
        """Solve what is loaded and read it back.

        *model* is asked only for what has no column and so was never loaded —
        the objective's constant. Either vector may be ``None``, for different
        reasons: no primal means the solve left nothing worth reading, while no
        dual is narrower, a mixed-integer model having none at all and neither
        does a run stopped short of a simplex basis.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the loaded model, and anything outside this process with it.

        Idempotent, and the counterpart to holding one: a solver kept between
        solves is memory — and, for one of them, a licence — that no frame in
        this process accounts for.
        """
