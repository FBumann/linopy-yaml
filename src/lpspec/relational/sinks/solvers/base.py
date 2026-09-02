"""What every solver is: a loaded model, and the rule for keeping it.

A solver sink holds the model it was given and outlives the solve it was loaded
for, so that an updated model (:meth:`~lpspec.api.Model.update`) has its new
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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from lpspec.errors import LpspecError

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl

    from lpspec.relational.sinks.capabilities import Capabilities
    from lpspec.relational.sinks.tables import Tables
    from lpspec.relational.status import SolveStatus


@dataclass(frozen=True)
class WarmStart:
    """What one solve leaves for a later session: a basis, or an incumbent.

    Read with :meth:`Solver.warm_start`, applied with :meth:`Solver.warm`.
    Which fields are filled is the reading solver's decision: an LP leaves its
    simplex basis (both status vectors, paired), a mixed-integer solve leaves
    no valid basis anywhere and carries its incumbent instead.

    Opaque, and the statuses are the reading solver's own encoding, so only a
    session of the same solver takes them back. Machinery, not a surface:
    nothing above ``solvers/`` carries one.
    """

    #: Which member of ``SOLVERS`` read it; only that member takes it back.
    solver: str
    #: Basis status per column in label order, or ``None`` after a solve that
    #: left no valid basis.
    column_statuses: Any | None
    #: Basis status per row in label order; filled exactly when
    #: :attr:`column_statuses` is.
    row_statuses: Any | None
    #: Primal value per column in label order — a mixed-integer incumbent —
    #: or ``None`` where the basis carries the start instead.
    column_values: Any | None

    def basis(self) -> tuple[Any, Any] | None:
        """Both status vectors — filled exactly together — or ``None`` where the incumbent carries the start."""
        if self.column_statuses is not None and self.row_statuses is not None:
            return self.column_statuses, self.row_statuses
        return None


@dataclass(frozen=True)
class SolveAnswer:
    """What a solve concluded, and the vectors it left.

    Any vector may be ``None``, for different reasons: no ``primal`` means the
    solve left nothing worth reading — and ``activity``, each row's left-hand
    side at that point, travels with it — while no ``dual`` is narrower: a
    mixed-integer model has none at all, and neither does a run stopped short
    of a simplex basis.
    """

    status: SolveStatus
    objective: float
    primal: pl.Series | None
    dual: pl.Series | None
    activity: pl.Series | None

    @classmethod
    def unreadable(cls, status: SolveStatus) -> SolveAnswer:
        """The answer for a solve that left nothing worth reading.

        One home for the fact that an unreadable status carries a NaN
        objective and no vector at all, so two sinks cannot spell it apart.
        """
        return cls(status, float('nan'), None, None, None)


class Solver(ABC):
    """One solver, holding one model. Subclassed once per member of ``SOLVERS``.

    A driver never constructs one directly: :func:`~lpspec.relational.sinks.solvers.loaded`
    is the whole of "reuse or load again", and what it hands back is run and,
    eventually, closed::

        solver = solvers.loaded(held, name, tables, options)
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
        tables: Tables,
        batch_rows: int | None = None,
        solver_options: Mapping[str, Any] | None = None,
    ) -> None:
        #: The options the loaded model was told. Set at the load, so a solve
        #: asking for others has to be given something that was.
        self._options = dict(solver_options or {})
        self._load(tables, batch_rows)
        #: What the loaded model *is* —
        #: :attr:`~lpspec.relational.sinks.tables.Tables.structure`, the
        #: digest of everything a re-solve may not change. Sixteen bytes, where
        #: holding the frames themselves would keep two models alive across a
        #: rebuild.
        self._structure = tables.structure
        #: The loaded model's spans, read by :meth:`_takes` alone — of the
        #: *ingested* tables, which on a reformulating sink are wider than what
        #: was built, so a warm start is checked against the model the solver
        #: actually holds.
        self._columns = tables.column_count
        self._rows = tables.row_count

    #: The packages this member imports lazily, and so the ones an environment
    #: has to have for it to run at all. Data rather than a probe per member:
    #: how availability is *decided* is one rule and lives in
    #: :meth:`is_available`, where a copy of it in each leaf could drift.
    requires: ClassVar[tuple[str, ...]]

    #: What this member can ingest, and what it refuses in combination. A
    #: member states it; the family acts on it
    #: (:func:`~lpspec.relational.sinks.ingestible`), so no ``_load``
    #: has to remember to ask.
    capabilities: ClassVar[Capabilities]

    #: What to tell a caller when :meth:`is_available` says no — which package
    #: is missing, and whether it ships or needs an extra. The member's own
    #: fact, so the sentence a caller reads is the one written beside the
    #: import that needs it, and there is only one of it. Named for when it
    #: prints rather than for what it advises: it is a message, not a verb.
    unavailable_message: ClassVar[str]

    def keeps(self, tables: Tables, solver_options: Mapping[str, Any] | None) -> bool:
        """Whether this held solver may keep its load and take *tables* by value.

        Both halves of the recorded evidence live here — the digest of what was
        loaded and the options it was loaded with — so the reuse test reads
        them where they were written.
        """
        return self._options == dict(solver_options or {}) and self._structure == tables.structure

    @classmethod
    def imported(cls) -> Any:
        """Every package in :attr:`requires`, imported — or :attr:`unavailable_message`.

        Returns the first, the member's own library; the rest are imported only
        to fail here, where the message covers them, rather than mid-load.
        Through ``__import__``, the hook the members' own ``import`` statements
        use, so an absence fails here rather than at the first statement past
        the guard.
        """
        try:
            modules = [__import__(package) for package in cls.requires]
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(cls.unavailable_message) from exc
        return modules[0]

    @classmethod
    def is_available(cls) -> bool:
        """Whether this build can actually run this solver.

        A probe of the import system rather than an import: answering must not
        cost the load it is asked to avoid, and must not raise. Probed at the
        top-level name — ``find_spec`` on a dotted one imports the parent.
        """
        return all(importlib.util.find_spec(package.partition('.')[0]) is not None for package in cls.requires)

    @abstractmethod
    def _load(self, tables: Tables, batch_rows: int | None) -> None:
        """Hand *tables* to the solver and hold whatever reads it back.

        Called by ``__init__`` rather than by a caller, so that a subclass
        cannot exist in a state where the other three have nothing to work on.
        """

    @abstractmethod
    def push(self, tables: Tables) -> None:
        """*tables*'s bounds, costs and right-hand sides onto the loaded model.

        Everything an update may change without moving a label, and only ever
        after *tables*'s digest matched the loaded one. Whole vectors rather
        than a diff: the model that would say which cells moved is the one
        this replaces.
        """

    @abstractmethod
    def warm_start(self) -> WarmStart | None:
        """What the loaded model holds to warm a later session, if anything.

        Returns:
            The basis after an LP solve, the incumbent after a mixed-integer
            one — a solved MIP leaves no valid basis on any solver — and
            ``None`` where the model holds neither, which is every model not
            yet solved.
        """

    def warm(self, ws: WarmStart) -> None:
        """Start the next :meth:`run` from *ws* instead of from scratch.

        The caller vouches that *ws* was read from a model with this one's
        label set; what is checked here is what can be — the sink it came
        from, and that its vectors span the loaded model. A member writes
        :meth:`_warm`; this is the contract around it, as :meth:`run` is
        around :meth:`_run`.

        Raises:
            LpspecError: A warm start read from another solver, or whose
                vectors do not span the loaded model.
        """
        self._takes(ws)
        self._warm(ws)

    def _takes(self, ws: WarmStart) -> None:
        """Refuse a warm start that describes a different model.

        Raises:
            LpspecError: A start from another solver — statuses are each
                solver's own encoding — or one whose vectors have the wrong
                span, which a basis being positional makes a start about a
                different model.
        """
        mine = type(self).__name__.lower()
        if ws.solver != mine:
            raise LpspecError(
                f'this warm start was read from {ws.solver!r} and cannot warm a {mine!r} session: '
                f"basis statuses and incumbents are the reading solver's own encoding, so applied "
                f'elsewhere they would start the solve from a state that means something else. '
                f'Read a warm start from the solver that will take it back.'
            )
        spans = (
            ('column statuses', ws.column_statuses, self._columns, 'columns'),
            ('row statuses', ws.row_statuses, self._rows, 'rows'),
            ('column values', ws.column_values, self._columns, 'columns'),
        )
        for quantity, values, expected, axis in spans:
            if values is not None and len(values) != expected:
                raise LpspecError(
                    f'this warm start carries {len(values)} {quantity} for a model with {expected} '
                    f'{axis}. A basis and an incumbent are positional, so one read from a '
                    f'differently shaped model would start the solve from a state about a '
                    f'different one — carry a warm start only across builds whose label set '
                    f'is unchanged.'
                )

    @abstractmethod
    def _warm(self, ws: WarmStart) -> None:
        """Apply *ws* onto the loaded model, its spans already checked.

        Reached only through :meth:`warm`, so a member may assume the vectors
        span the model it holds and that filled fields pair the way
        :class:`WarmStart` says they do.
        """

    def run(self, tables: Tables) -> SolveAnswer:
        """Solve what is loaded, read it back, and refuse a vector that lies.

        Reading a solution back is positional, so a vector that does not span
        the model is an answer about a *different* one. Refused here, where the
        solver hands it over, rather than where it is read: the objective comes
        back directly, so a result built on a broken vector would report a
        plausible number and only fail if someone asked for a coordinate.

        A member writes :meth:`_run`; this is the contract around it, so no
        sink can be added that forgets to be checked.
        """
        answer = self._run(tables)
        self._check_span('primal', answer.primal, tables.column_count)
        self._check_span('dual', answer.dual, tables.row_count)
        self._check_span('activity', answer.activity, tables.row_count)
        return answer

    def _check_span(self, quantity: str, values: pl.Series | None, expected: int) -> None:
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
    def _run(self, tables: Tables) -> SolveAnswer:
        """Solve what is loaded and read it back.

        *tables* is asked only for what has no column and so was never loaded —
        the objective's constant. When either vector may be ``None`` is
        :class:`SolveAnswer`'s docstring.
        """

    @abstractmethod
    def forget(self) -> None:
        """Discard the work the last solve did, keeping the model loaded.

        The middle rung of :data:`~lpspec.relational.result.KEEPS`: the matrix
        stays handed over, and the next run begins as if it had never been
        solved. A member with nothing to discard implements this as a no-op.
        """

    @property
    @abstractmethod
    def handle(self) -> Any:
        """The native object the load handed back, or ``None`` once closed.

        The library's own model — what ``build_<solver>`` gives a caller who
        stops at the hand-off, and what a test reads the load back through.
        Owned by this holder: the caller does not release it, :meth:`close`
        does.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the loaded model, and anything outside this process with it.

        Idempotent, and the counterpart to holding one: a solver kept between
        solves is memory — and, for one of them, a licence — that no frame in
        this process accounts for. Afterwards :attr:`handle` is ``None``.

        **The same release happens to a holder dropped without closing.** A
        member whose library releases its object on collection has that for
        free; one that does not — or that holds two objects, a model on an
        environment, where the order is innermost first — registers a
        finalizer over the objects rather than over itself, so a half-torn
        holder is never what runs it. ``tests/test_solver_release.py`` asks
        every member.
        """

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False
