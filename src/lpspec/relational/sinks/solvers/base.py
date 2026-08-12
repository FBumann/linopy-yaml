"""What every solver is: a loaded model, and the rule for keeping it.

A solver sink holds the model it was given and outlives the solve it was loaded
for, so that a rebound model (:meth:`~lpspec.api.BoundModel.rebind`) has its new
numbers *pushed* onto what the solver already has and re-solves from the basis
the last one ended on. linopy's shape, and its word: their ``Solver`` is the
persistent object too. Copied rather than imported, and tested here.

**This module imports no solver.** It is the one thing ``solvers/`` members may
read besides ``tables.py``, and that is the whole reason it can exist: the fence
`tests/test_architecture.py` draws keeps ``gurobipy`` off the import path of a
caller who solves with HiGHS, and a base that reaches for neither cannot carry
one across. Sharing through it is what stops the alternative — one leaf importing
the other — from ever being the tempting option.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

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

    A driver never constructs one directly: :func:`~lpspec.relational.sinks.solvers.loaded`
    is the whole of "reuse or load again", and what it hands back is run and,
    eventually, closed::

        solver = solvers.loaded(held, name, tables, batch_rows, options)
        solver.run(tables)  # …repeatedly
        solver.close()

    The two halves are split by who can answer them. **This class records the
    rule's evidence** — the digest of what was loaded and the options it was
    loaded with, identical bookkeeping for every solver. **A subclass owns the
    hand-off**: loading, pushing values, running, releasing, all of which are
    its own library's shape and nothing else's.
    """

    def __init__(
        self,
        model: ModelTables,
        batch_rows: int | None = None,
        solver_options: Mapping[str, Any] | None = None,
    ) -> None:
        #: The options the loaded model was told. Set at the load, so a solve
        #: asking for others has to be given something that was.
        self._options = dict(solver_options or {})
        self._load(model, batch_rows)
        #: What the loaded model *is* —
        #: :attr:`~lpspec.relational.sinks.tables.ModelTables.structure`, the
        #: digest of everything a re-solve may not change. Sixteen bytes, where
        #: holding the frames themselves would keep two models alive across a
        #: rebuild.
        self._structure = model.structure

    #: The packages this member imports lazily, and so the ones an environment
    #: has to have for it to run at all. Data rather than a probe per member:
    #: how availability is *decided* is one rule and lives in
    #: :meth:`is_available`, where a copy of it in each leaf could drift.
    requires: ClassVar[tuple[str, ...]]

    #: What to tell a caller when :meth:`is_available` says no — which package
    #: is missing, and whether it ships or needs an extra. The member's own
    #: fact, so the sentence a caller reads is the one written beside the
    #: import that needs it, and there is only one of it. Named for when it
    #: prints rather than for what it advises: it is a message, not a verb.
    unavailable_message: ClassVar[str]

    @classmethod
    def is_available(cls) -> bool:
        """Whether this build can actually run this solver.

        A name in ``SOLVERS`` says the package *knows* the solver, not that the
        environment has it: ``gurobi`` is a name here on an install that never
        took the extra. Asked where the sink is resolved, which is before the
        build, so naming one this environment cannot run costs no model
        (:func:`~lpspec.relational.sinks.solvers.solver`).

        A probe of the import system rather than an import: answering must not
        cost the load it is asked in order to avoid, and must not raise.
        Uncached, being asked once per solve — against a solve.
        """
        return all(importlib.util.find_spec(package) is not None for package in cls.requires)

    @abstractmethod
    def _load(self, model: ModelTables, batch_rows: int | None) -> None:
        """Hand *model* to the solver and hold whatever reads it back.

        Called by ``__init__`` rather than by a caller, so that a subclass
        cannot exist in a state where the other three have nothing to work on.
        """

    @abstractmethod
    def push(self, model: ModelTables) -> None:
        """*model*'s bounds, costs and right-hand sides onto the loaded model.

        Everything a rebind may change without moving a label, and only ever
        after *model*'s digest matched the loaded one. Whole vectors rather
        than a diff: the model that would say which cells moved is the one
        this replaces.
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

        Returns:
            The status pair, the objective, and the primal and dual vectors —
            either of which may be ``None``, as :meth:`_run` describes.
        """
        status, objective, primal, dual = self._run(model)
        self._spans('primal', primal, model.column_count)
        self._spans('dual', dual, model.row_count)
        return status, objective, primal, dual

    def _spans(self, quantity: str, values: pl.Series | None, expected: int) -> None:
        """Check that a solver vector spans the model.

        ``None`` is not a wrong length — a mixed-integer model has no duals,
        and neither does a run stopped short of a simplex basis.

        Raises:
            LpspecError: A vector of any other length, which describes a
                different model.
        """
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
