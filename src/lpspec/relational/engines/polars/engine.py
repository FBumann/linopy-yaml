"""Polars engine: build the model frames, hand them to a sink, read the answer back.

The engine owns the lifecycle — a build, its solver, the counters and clocks
:meth:`PolarsEngine.diagnostics` reports — and none of the three questions it
asks on the way: what the data is (:mod:`~lpspec.relational.engines.polars.attaching`),
what each declaration contributes (:mod:`~lpspec.relational.engines.polars.assembly`),
how a row or a solve reads back (:mod:`~lpspec.relational.engines.polars.readback`).
The lane is described in docs/about/architecture.md.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from lpspec.errors import LpspecError
from lpspec.relational import sinks
from lpspec.relational.engines.polars import readback
from lpspec.relational.engines.polars.assembly import (
    Assembly,
    BuiltModel,
    Measured,
    declares_quadratic,
    short_parameters,
)
from lpspec.relational.engines.polars.attaching import attach
from lpspec.relational.engines.polars.compiler import PolarsCompiler
from lpspec.relational.result import KEEPS, ConstraintRow, Diagnostics, Keep, Result, unknown_keep_message

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from math_spec import program


def _no_built_model(doing: str) -> str:
    """Why there is no model *doing*, in the two ways that happens."""
    return (
        f'there is no built model {doing}: it was closed, or an update raised and released '
        f'it rather than leaving half of one behind. Build it again — update() with data it can '
        f'attach, or build() from the start.'
    )


class PolarsEngine:
    """Build a :class:`Program` into polars frames, then sink it."""

    def __init__(self) -> None:
        #: The build, or ``None`` where there is not one — closed, released by
        #: an update that raised, or never run.
        self._built: BuiltModel | None = None
        #: What the last build measured about itself. Outlives ``_built``,
        #: since :meth:`diagnostics` answers after :meth:`close`.
        self._measured = Measured()
        #: The solver holding this model, kept between solves — the only thing
        #: a rebuild does *not* throw away. ``None`` until one has been solved.
        self._solver: sinks.Solver | None = None
        #: How many solves this model has been through, and how many of them
        #: had to load the solver from scratch instead of pushing values onto
        #: one that already held it.
        self._solves = 0
        self._loads = 0
        #: What the last solve's sink had to add to take the model — nothing,
        #: unless it had no concept of a set the model declares. A fact about a
        #: *solve*, so a rebuild does not clear it.
        self._sink_columns = 0
        self._sink_rows = 0
        #: Wall seconds each phase has spent, cumulatively. Time spent is a
        #: fact about what ran, so a rebuild adds to it rather than clearing it.
        self._timings: dict[str, float] = {}

    @property
    def _model(self) -> BuiltModel:
        """The built model, or why there is not one."""
        if self._built is None:
            raise LpspecError(_no_built_model('to hand over'))
        return self._built

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def build(self, program: program.Program, sources: Mapping[str, pl.LazyFrame]) -> None:
        """Attach *sources*, then build every declaration into the model frames.

        **A second call rebuilds over the same object**, which is what
        ``update`` is. The previous build is released *before* this one starts,
        so a driver that re-solves in a loop stays at one model's peak; what
        the loaded solver holds survives as the digest it recorded at its load.
        A build that raises leaves no model at all rather than half of one,
        and ``diagnostics()`` answers from what was measured by then.
        """
        self._built = None
        self._measured = Measured()
        with _clocked(self._timings, 'attach'):
            attached = attach(program, sources)
        self._measured.sparse = short_parameters(program, attached)
        assembly = Assembly(program, attached, self._measured)
        with _clocked(self._timings, 'build'):
            self._built = assembly.run()

    # ------------------------------------------------------------------
    # sinks — see relational/sinks/; the engine only supplies the frames
    # ------------------------------------------------------------------

    def row(self, name: str, coordinate: Mapping[str, Any]) -> ConstraintRow:
        """One built constraint row, spelled back out. See :meth:`~lpspec.api.Model.row`."""
        if self._built is None:
            raise LpspecError(_no_built_model(f"to read '{name}' out of"))
        return readback.row(self._built, name, coordinate)

    def write(self, path: str | Path) -> None:
        """Stream the built model to *path*, in the format its suffix names.

        A construct the format has no section for is refused here, the way the
        solve path refuses one a solver cannot ingest
        (:func:`~lpspec.relational.sinks.ingestible`) and with the sentence
        ``check(spec, sink=...)`` would have given: written anyway, the file
        would parse, solve, and be a different model.

        Raises:
            ValueError: A suffix nothing writes.
            LpspecError: A construct this format cannot spell.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        chosen = sinks.writer(suffix)
        tables = self._model.tables()
        if (refused := sinks.refusal(self._model.program, suffix)) is not None:
            raise LpspecError(refused)
        with _clocked(self._timings, 'write'):
            chosen.write(tables, path)

    def solve(
        self,
        solver_name: str = 'highs',
        *,
        solver_options: Mapping[str, Any] | None = None,
        keep: Keep = 'solver',
    ) -> Result:
        """Hand the built model to a solver and solve it.

        The solver stays loaded where it can, which is
        :func:`~lpspec.relational.sinks.solvers.loaded`'s decision: an updated
        model has its new numbers pushed onto what the solver already holds,
        and one whose structure moved is loaded again. What the solver is
        handed may be wider than what was built
        (:func:`~lpspec.relational.sinks.ingestible`): a sink with no SOS
        concept takes the sets as binaries and rows appended past the model's
        own. The read-back is unaffected — a declaration's share is a slice,
        and nothing was appended before one.

        Args:
            solver_name: One of :data:`~lpspec.relational.sinks.SOLVERS`.
            solver_options: Forwarded to the solver verbatim, in its own
                vocabulary (``{'time_limit': 60, 'mip_rel_gap': 0.01}``).
            keep: How much of the session this solve may keep — one of
                :data:`~lpspec.relational.result.KEEPS`. A preference, not a
                guarantee: a model whose structure moved is loaded again
                whatever was asked, and
                :attr:`~lpspec.relational.result.Result.kept` reports what
                happened. ``nothing`` is held to structurally, the held solver
                being closed before the load decision.

        Returns:
            The solution, holding this engine and the build it answered.

        Raises:
            LpspecError: A *keep* outside
                :data:`~lpspec.relational.result.KEEPS`.
        """
        if keep not in KEEPS:
            raise LpspecError(unknown_keep_message(keep))
        built = self._model.tables()
        with _clocked(self._timings, 'handoff'):
            tables = sinks.ingestible(solver_name, built, self._model.program)
            self._sink_columns = tables.column_count - built.column_count
            self._sink_rows = tables.row_count - built.row_count
            if keep == 'nothing' and self._solver is not None:
                self._solver.close()
                self._solver = None
            held = self._solver
            self._solver = sinks.loaded(held, solver_name, tables, solver_options)
            kept: Keep = keep if self._solver is held else 'nothing'
            if kept == 'solver':
                self._solver.forget()
        self._solves += 1
        if self._solver is not held:
            self._loads += 1
        with _clocked(self._timings, 'solve'):
            answer = self._solver.run(tables)
        assert answer.primal is not None or not answer.status.is_readable, (
            'a readable status must come with a primal vector'
        )
        assert (answer.activity is None) == (answer.primal is None), (
            'activity travels with the primal: every sink reads it whenever a solution exists, mixed-integer included'
        )
        primals, duals, activities = self._read_back(answer.primal, answer.dual, answer.activity)
        return Result(
            _status=answer.status,
            _objective=answer.objective,
            _primals=primals,
            _duals=duals,
            _activities=activities,
            _kept=kept,
            _expressions=self._expression_readers(answer.primal),
            _no_duals=None
            if answer.dual is not None
            else _no_duals_message(
                self._discrete(),
                answer.status.termination_condition,
                sets=self._reformulated_sets(tables is not built),
                quadratic_rows=self._quadratic_constraints(),
            ),
        )

    def diagnostics(self) -> Diagnostics:
        """What this build and its solves did that the answer does not show.

        Answerable after :meth:`close`: every field is a count, a clock or a
        small frame this keeps, not a read of the model it releases.
        """
        return Diagnostics(
            columns=self._measured.columns,
            rows=self._measured.rows,
            nonzeros=self._measured.nonzeros,
            sink_columns=self._sink_columns,
            sink_rows=self._sink_rows,
            omissions=pl.DataFrame(
                {'constraint': list(self._measured.omitted), 'rows_not_built': list(self._measured.omitted.values())},
                schema={'constraint': pl.String, 'rows_not_built': pl.UInt32},
            ),
            coefficient_range=pl.DataFrame(
                {
                    'constraint': list(self._measured.coefficients),
                    'smallest': [low for low, _ in self._measured.coefficients.values()],
                    'largest': [high for _, high in self._measured.coefficients.values()],
                },
                schema={'constraint': pl.String, 'smallest': pl.Float64, 'largest': pl.Float64},
            ),
            bound_range=pl.DataFrame(
                {
                    'variable': list(self._measured.bounds),
                    'smallest': [low for low, _ in self._measured.bounds.values()],
                    'largest': [high for _, high in self._measured.bounds.values()],
                },
                schema={'variable': pl.String, 'smallest': pl.Float64, 'largest': pl.Float64},
            ),
            rhs_range=pl.DataFrame(
                {
                    'constraint': list(self._measured.rhs),
                    'smallest': [low for low, _ in self._measured.rhs.values()],
                    'largest': [high for _, high in self._measured.rhs.values()],
                },
                schema={'constraint': pl.String, 'smallest': pl.Float64, 'largest': pl.Float64},
            ),
            sparse_parameters=pl.DataFrame(
                {
                    'parameter': list(self._measured.sparse),
                    'coordinates': [reach for reach, _ in self._measured.sparse.values()],
                    'rows': [rows for _, rows in self._measured.sparse.values()],
                    'missing': [reach - rows for reach, rows in self._measured.sparse.values()],
                },
                schema={
                    'parameter': pl.String,
                    'coordinates': pl.UInt64,
                    'rows': pl.UInt64,
                    'missing': pl.UInt64,
                },
            ),
            objective_range=self._measured.objective_range,
            solves=self._solves,
            loads=self._loads,
            timings=dict(self._timings),
        )

    def _read_back(
        self, primal: pl.Series | None, dual: pl.Series | None, activity: pl.Series | None
    ) -> tuple[dict[str, pl.LazyFrame], dict[str, pl.LazyFrame], dict[str, pl.LazyFrame]]:
        """One solve's answer as one frame per declaration — a :class:`Result`'s own.

        References rather than copies: the frames point at this build's label
        frames, and :meth:`build` replacing the registries takes nothing from
        what an earlier result still holds. Lazy, so composing every
        declaration's plan here costs nothing for the ones nobody reads. A
        vector that is ``None`` yields no frames at all rather than empty
        ones, which is the state :class:`Result` reports through the status.
        """
        model = self._model
        program = model.program

        def rows(values: pl.Series | None) -> dict[str, pl.LazyFrame]:
            if values is None:
                return {}
            return {
                name: readback.laid_out(model.attached, model.constraints[name], c.dims, values)
                for name, c in program.constraints.items()
            }

        return (
            {
                name: readback.laid_out(model.attached, model.variables[name], v.dims, primal)
                for name, v in program.variables.items()
            }
            if primal is not None
            else {},
            rows(dual),
            rows(activity),
        )

    def _expression_readers(self, primal: pl.Series | None) -> dict[str, Callable[[], pl.DataFrame]]:
        """One deferred reader per declared named expression — nothing compiled yet.

        A closure compiles its expression when it is first called, so a solve
        over fifty declared expressions that reads none pays for a dict of
        closures. Each captures a snapshot the result *owns* — the program,
        the attached data, a copy of this build's variable-frame registry and
        the solver's primal vector — so it keeps answering after an update or
        ``close()``, at the cost of keeping those frames alive.
        """
        if primal is None:
            return {}
        model = self._model
        compiler = PolarsCompiler(model.program, model.attached, dict(model.variables))
        values = pl.DataFrame(
            {'var_label': pl.int_range(primal.len(), dtype=pl.Int64, eager=True), readback.SOLUTION: primal}
        ).lazy()

        def reader(name: str, expression: program.ExpressionNode) -> Callable[[], pl.DataFrame]:
            return lambda: readback.expression_frame(name, expression, compiler, values)

        return {name: reader(name, e) for name, e in model.program.named_expressions.items()}

    def _discrete(self) -> list[str]:
        """The variables this model declared as anything but continuous."""
        return sorted(n for n, v in self._model.program.variables.items() if v.variable_type != 'continuous')

    def _quadratic_constraints(self) -> list[str]:
        """The constraints this model declared as quadratic — a fact about the model, not the solve."""
        return sorted(n for n, c in self._model.program.constraints.items() if declares_quadratic(c))

    def _reformulated_sets(self, reformulated: bool) -> list[str]:
        """The sets that reached the solver as binaries, if any did.

        The one reason for a missing dual no declaration shows: the model
        declares no integrality, and the sink added some.
        """
        return sorted(self._model.program.sos) if reformulated else []

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Drop the built model. A :class:`Result` keeps its own frames.

        One assignment, because the build is one value. A loaded solver goes
        first, being the one thing here that is not this process's memory.
        :meth:`diagnostics` still answers afterwards.
        """
        if self._solver is not None:
            self._solver.close()
            self._solver = None
        self._built = None

    def __enter__(self) -> PolarsEngine:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False


def _no_duals_message(
    discrete: Sequence[str],
    termination_condition: str,
    sets: Sequence[str],
    quadratic_rows: Sequence[str],
) -> str:
    """Why a solve that *did* leave values still has no duals.

    Integrality is decidable from the model, and naming the variable is
    actionable where "the solver reported none" is not.

    *sets* are the special-ordered sets a sink without the concept turned into
    binaries. They come first because a model that declared none of its own
    integrality would otherwise be told it is mixed-integer with nothing named
    — and because the fix is a different one: another sink, not a different
    model.

    *quadratic_rows* are the quadratic constraints, whose prices are off by
    default: asking for them puts the solve on the convex path, and a nonconvex
    row that solves without them fails with them. The one case here where
    nothing is wrong with the model.
    """
    if quadratic_rows and not discrete:
        names = ', '.join(f"'{n}'" for n in quadratic_rows)
        return (
            f"a quadratic constraint prices only under gurobi's QCPDual, which is off by default: "
            f'{names} {"is" if len(quadratic_rows) == 1 else "are"} quadratic. Asking for those '
            f'prices makes the solver take the convex path, so a nonconvex row that solves without '
            f'them fails with them — which is why this is yours to ask for rather than ours to '
            f"assume. Re-solve with solver_options={{'QCPDual': 1}} if the model is convex."
        )
    if sets:
        names = ', '.join(f"'{n}'" for n in sets)
        return (
            f'duals are undefined for a mixed-integer model, and this sink has no SOS concept, so '
            f'{names} reached it as binaries. Solve with a sink that takes a set natively (gurobi) '
            f'to keep the LP, or drop the set to price the relaxation.'
        )
    if discrete:
        names = ', '.join(f"'{n}'" for n in discrete)
        return (
            f'duals are undefined for a mixed-integer model: {names} '
            f'{"is" if len(discrete) == 1 else "are"} not continuous. '
            f'Drop the integrality to price the LP relaxation instead.'
        )
    return (
        f'the solver returned no dual solution, though the solve terminated '
        f'{termination_condition!r}. Duals come from a simplex basis, which a '
        f'run stopped short of one does not have.'
    )


@contextmanager
def _clocked(timings: dict[str, float], phase: str) -> Iterator[None]:
    """Add the block's wall time onto ``timings[phase]`` — the diagnostics clocks.

    Cumulative, so a phase that runs again adds to its total the way the
    counters count. Recorded on failure too: a build that died mid-phase spent
    its time there.
    """
    started = perf_counter()
    try:
        yield
    finally:
        timings[phase] = timings.get(phase, 0.0) + perf_counter() - started
