"""What a caller reads back — a solve's :class:`Result`, a build's :class:`Diagnostics`.

The objects ``lps.solve`` and ``bound.diagnostics()`` hand back, so they are
the pieces of this subpackage a reader meets without going looking. They live
beside the engine rather than in it because they answer different questions:
the engine *builds* a model, and these *read* one — a :class:`Result` holds one
finished frame per declaration, its values already laid out over the build's
coordinates, so no reader ever goes back to the engine.

Named for linopy's envelope (``Result`` = status + solution + report) rather
than for our own decomposition, because our audience arrives from linopy and a
second vocabulary for one fact is a tax on every one of them
(docs/about/architecture.md, "Where a concept is already linopy's").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from lpspec.errors import LpspecError, NoSolutionError, unknown_name_message

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import pandas as pd
    import polars as pl
    import xarray as xr

    from lpspec.relational.status import SolveStatus


#: How much of the session a solve keeps, as a request to
#: :meth:`lpspec.api.BoundModel.solve` and as the report in
#: :attr:`Result.kept`. One word rather than a pair of flags because the two
#: things a session holds — the solver with the model on it, and the work that
#: solver did — can only be dropped in that order: there is no carrying on from
#: a solver that was closed, so the fourth combination does not exist.
Keep = Literal['nothing', 'solver', 'progress']

#: What each word keeps, in the order of how much that is. Deliberately about
#: provenance and not mechanism: whether *progress* is a basis, an incumbent
#: or a sink's own notion is the sink's business, so a solver with no simplex
#: fits these words unchanged.
KEEPS: Mapping[Keep, str] = {
    'nothing': 'the model is handed to a fresh solver, which has nothing to begin from',
    'solver': 'the solver already holding the model is reused, and the work the last solve did is discarded',
    'progress': 'the solver is reused and carries on from where the last solve got to',
}


def unknown_keep_message(keep: object) -> str:
    """Why *keep* is not one, and what the three are."""
    options = '\n'.join(f'  {name}: {what}' for name, what in KEEPS.items())
    return f'unknown keep {keep!r}. A solve may keep:\n{options}'


def tidy_to_pandas(frame: pl.DataFrame) -> pd.DataFrame:
    """A tidy polars frame as pandas, column by column.

    The three bridges below are shared by :class:`Result` and
    :class:`~lpspec.strategy.Runs`, which differ only in where the tidy frame
    comes from — a sweep's is the same frame one slice wider. Built column by
    column because polars' own ``to_pandas`` reaches for pyarrow; pandas itself
    ships with the ``[linopy]`` extra.
    """
    import pandas as pd

    return pd.DataFrame({column: frame[column].to_numpy() for column in frame.columns})


def tidy_to_dataarray(frame: pd.DataFrame, name: str) -> xr.DataArray:
    """The same, labelled by its non-``value`` columns.

    A scalar declaration has none and comes back 0-dimensional.
    """
    dims = [column for column in frame.columns if column != 'value']
    if not dims:
        return frame['value'].to_xarray().rename(name)
    return frame.set_index(dims).to_xarray()['value'].rename(name)


def tidy_to_dataset(names: Sequence[str], one: Callable[[str], xr.DataArray]) -> xr.Dataset:
    """*names* as one dataset, each array built by *one*."""
    first, *rest = names
    dataset = one(first).to_dataset(name=first)
    for name in rest:
        dataset[name] = one(name)
    return dataset


@dataclass(frozen=True)
class Diagnostics:
    """What a build and its solves did that the answer does not show.

    Advisory, all of it: no answer depends on any field, and a caller who
    branches on one has made this engine's bookkeeping part of their model.
    Read them when a loop is slower or smaller than it should be.
    """

    #: The shape the build produced: columns, rows, and matrix entries. What
    #: ``check`` cannot answer, needing no data where this needs all of it,
    #: and the thing to report when a model is bigger than its author
    #: expected — a broadcast that multiplied rows shows up here first.
    columns: int
    rows: int
    nonzeros: int

    #: What the **last solve's sink** had to add on top of those to take the
    #: model, and zero for every sink that took it as built. A sink with no
    #: SOS concept is handed the sets as binaries and linking rows
    #: (:mod:`lpspec.relational.sinks.sos`), which is the one thing that grows
    #: a model after the build and the one growth no declaration accounts
    #: for — so a solve that is larger than the model reads it here rather
    #: than nowhere. Zero until something has been solved: a *writer* is
    #: handed the model as built, and reports nothing.
    sink_columns: int
    sink_rows: int

    #: ``(constraint, rows_not_built)`` — every declared row that did not reach
    #: the solver (the absence rules), by either route: one emptied of all its
    #: terms, and one a **propagated absence** deleted while its other terms were
    #: still live. Without this record a declared constraint could go unenforced
    #: with no way to notice, which is what the second route did until #944.
    #: Empty for a model whose every declared row was built — a recurrence's
    #: first coordinate counting as a row it declared and did not get, so a
    #: ``shift`` against the horizon's edge reports here and is the boundary
    #: rather than a fault. Counts rather than coordinates: the label of an
    #: unbuilt row does not exist, so naming which went would mean holding the
    #: pre-drop frame — memory proportional to the omission, on the path this
    #: package measures hardest.
    omissions: pl.DataFrame

    #: How many times this model has been solved, and how many of those solves
    #: loaded the solver from scratch instead of pushing values onto one that
    #: already held it. Read together: ``loads == 1`` is a driver on the fast
    #: path — the first solve had nothing to keep — and ``loads == solves`` on
    #: an iterating driver is the difference between "lpspec is slow" and
    #: "this model masks on a parameter that varies", unless the driver asked
    #: for ``keep='nothing'``, which loads by construction. ``loads`` ticks on
    #: exactly the solves that report :attr:`Result.kept` of ``nothing`` —
    #: the same event, counted here and named there.
    solves: int
    loads: int

    #: Cumulative wall-clock seconds per phase, keyed by the phase's name:
    #: ``bind`` (the caller's sources onto the plan), ``build`` (declarations
    #: into the model frames), ``handoff`` (the built model into a solver),
    #: ``solve`` (the solver's own run), ``write`` (the built model to a
    #: file). A phase that never ran has no key; one that ran again holds the
    #: sum — a rebind's bind and build land on top of the first's, the way
    #: ``solves`` keeps counting. Clocks rather than a profile: enough to say
    #: which phase a slow loop spends its time in, not why.
    timings: Mapping[str, float]


def _named(frames: Mapping[str, pl.LazyFrame], name: str, kind: str) -> pl.LazyFrame:
    try:
        return frames[name]
    except KeyError:
        raise KeyError(unknown_name_message(kind, name, frames)) from None


@dataclass
class Result:
    """What a solve returned — the outcome, and access to any values.

    Returned whatever the solve concluded: test :attr:`has_primal` before
    reading values, or catch :class:`~lpspec.errors.NoSolutionError`. The
    values are this result's own, so a later solve on the same model does not
    rewrite them, and there is no lifetime to manage — :meth:`close` releases
    what this result holds early, and nothing breaks without it.

    A rebind is no exception. A result owns everything it reads — one finished
    frame per declaration, its own values already laid out over the label frames
    of the build it answered — so it outlives anything done to the model
    afterwards: a rebind, another solve, ``bound.close()``. What retaining one
    costs is those label frames staying alive, which matters once a caller keeps
    several, as a sweep, a rolling horizon and Benders all do.
    """

    _status: SolveStatus
    _objective: float
    #: One ``(dims…, value)`` frame per declaration, lazy and in label order —
    #: a read is a collect. ``None`` is what :meth:`close` leaves behind, and
    #: the primal's absence is what "closed" means: both go together, and an
    #: empty mapping is a solve that left nothing, which the status reports.
    _primals: Mapping[str, pl.LazyFrame] | None
    _duals: Mapping[str, pl.LazyFrame] | None
    #: The constraints' left-hand sides at the solution, laid out exactly as
    #: :attr:`_duals` — same frames, same row order — and present whenever the
    #: primals are: unlike a dual, an activity exists at any incumbent.
    _activities: Mapping[str, pl.LazyFrame] | None
    #: How much of the session this solve kept, read off what actually ran —
    #: never off what was asked for.
    _kept: Keep
    #: One deferred reader per declared named expression. A callable rather
    #: than a frame because deferral is the contract (the rules for named expressions): nothing about
    #: an expression is lowered or compiled until its reader is called, so a
    #: model that reads none pays for none. Released with the primals by
    #: :meth:`close`, since each holds this build's frames and values.
    _expressions: Mapping[str, Callable[[], pl.DataFrame]] | None = None
    #: Why there are no duals, when a solve that left values still has none.
    #: ``None`` whenever :attr:`_duals` holds them.
    _no_duals: str | None = None

    @property
    def status(self) -> str:
        """Coarse outcome: ``ok`` / ``warning`` / ``error`` / ``aborted`` / ``unknown``."""
        return self._status.status

    @property
    def termination_condition(self) -> str:
        """What the solver said — ``optimal``, ``infeasible``, ``time_limit`` and so on."""
        return self._status.termination_condition

    @property
    def is_ok(self) -> bool:
        """The linopy rollup: not an error, an abort or a refusal."""
        return self._status.is_ok

    @property
    def has_primal(self) -> bool:
        """Whether there are values to read — what the accessors gate on.

        Narrower than :attr:`is_ok`: a run stopped at a time limit before any
        incumbent is ``ok`` with nothing to read.
        """
        return self._status.is_readable

    @property
    def objective(self) -> float:
        """The objective value, or ``nan`` when there is no solution."""
        return self._objective

    @property
    def kept(self) -> Keep:
        """How much of the session this solve kept — one of :data:`KEEPS`.

        What *happened*, not what was asked: ``keep=`` is a preference, and a
        first solve or a structure that moved keeps ``nothing`` whatever it
        requested, the solver having been loaded again. So a driver that asked
        to keep ``progress`` and reads ``nothing`` back is being told its
        labels moved. Advisory, like :class:`Diagnostics`: no answer depends
        on it.
        """
        return self._kept

    def _readable(self, frames: Mapping[str, pl.LazyFrame] | None, what: str) -> Mapping[str, pl.LazyFrame]:
        """*frames*, or why they cannot be read — closed first, then the status.

        Closedness is read off the primals whichever mapping was asked for,
        because :meth:`close` releases both together and a solve may
        legitimately leave the duals empty.
        """
        if self._primals is None:
            raise LpspecError(
                f'cannot read {what}: this result was closed, and closing releases its values and its '
                f'hold on the coordinates they lay out over. Frames already read stay valid — they are '
                f'their own data — so read what you need before close(), or drop the `with` and close '
                f'when you are done.'
            )
        if not self._status.is_readable:
            raise NoSolutionError(
                f'cannot read {what}: the solve terminated {self.termination_condition!r} '
                f'({self._status.solver_wording}), so there are no values to read. Test '
                f'`has_primal` first. This raises rather than returning, because the solver '
                f'hands back a full-length vector of zeros either way and it is '
                f'indistinguishable from an answer.'
            )
        assert frames is not None, 'close() releases the primal, dual and activity frames together'
        return frames

    def primal(self, name: str) -> pl.DataFrame:
        """The tidy solution of variable *name* — ``(dims…, value)``.

        Rows come back in label order, row-major over the variable's coordinate
        product, so two reads and two runs agree.

        Raises:
            NoSolutionError: The solve left no values to read.
            LpspecError: The model was rebound after this solve.
            KeyError: No variable is called *name*.
        """
        frames = self._readable(self._primals, f"the primal of '{name}'")
        return _named(frames, name, 'variable').collect(engine='streaming')

    def dual(self, name: str) -> pl.DataFrame:
        """Shadow prices of constraint *name* — ``(dims…, value)``.

        :meth:`primal`'s shape and order, over constraint rows.

        Raises:
            NoSolutionError: The solve left no values at all.
            LpspecError: It left primals but no duals — an integer variable
                makes them undefined — or the model was rebound since.
            KeyError: No constraint is called *name*.
        """
        frames = self._readable(self._duals, f"the dual of '{name}'")
        if self._no_duals is not None:
            raise LpspecError(self._no_duals)
        return _named(frames, name, 'constraint').collect(engine='streaming')

    def activity(self, name: str) -> pl.DataFrame:
        """The left-hand side of constraint *name* at the solution — ``(dims…, value)``.

        :meth:`dual`'s shape and order, and the other half of a row's story:
        how far each row's ``Σ aᵢxᵢ`` sits from its bound. The solver's own
        number, not a recomputation. Readable whenever there is a solution —
        unlike :meth:`dual` it is well-defined on a mixed-integer model. On an
        ``==`` row it equals the right-hand side up to solver tolerance by
        construction.

        Raises:
            NoSolutionError: The solve left no values to read.
            LpspecError: This result was closed.
            KeyError: No constraint is called *name*.
        """
        frames = self._readable(self._activities, f"the activity of '{name}'")
        return _named(frames, name, 'constraint').collect(engine='streaming')

    def expression(self, name: str) -> pl.DataFrame:
        """The value of named expression *name* at this solution — ``(dims…, value)``.

        The quantity the model declares under ``expressions:``, evaluated at
        the solve's primal values and aggregated to the expression's own dims —
        :meth:`primal`'s shape and order, over those dims in declaration order.
        Lowered and compiled on this call, not at build, so a model that reads
        no expression pays for none.

        Takes a **declared name only**, never an expression string: what is
        readable is exactly what the file names, so the quantity a constraint
        bounds and the quantity a report reads are one definition.

        Raises:
            NoSolutionError: The solve left no values to read.
            LpspecError: This result was closed.
            DataError: A divisor with no value where the expression divides.
            KeyError: No named expression is called *name*.
        """
        self._readable(self._primals, f"expression '{name}'")
        readers = self._expressions or {}
        try:
            reader = readers[name]
        except KeyError:
            raise KeyError(
                unknown_name_message('named expression', name, readers)
                + ' expression() takes a name declared under expressions:, never an expression string.'
            ) from None
        return reader()

    def to_pandas(self, name: str) -> pd.DataFrame:
        """:meth:`primal` as a tidy :class:`pandas.DataFrame`."""
        return tidy_to_pandas(self.primal(name))

    def to_dataarray(self, name: str) -> xr.DataArray:
        """:meth:`primal` as a labelled :class:`xarray.DataArray`.

        Dense over the variable's dims: a masked coordinate comes back NaN.
        """
        return tidy_to_dataarray(self.to_pandas(name), name)

    def to_dataset(self, *names: str) -> xr.Dataset:
        """The named variables as one :class:`xarray.Dataset`; all by default.

        Each arrives dense over its own dims, all at once — on a large model
        name the few you need, or use :meth:`to_parquet`.
        """
        wanted = names or tuple(self._readable(self._primals, 'the solution'))
        return tidy_to_dataset(wanted, self.to_dataarray)

    def to_parquet(self, directory: str | Path) -> dict[str, Path]:
        """Write one parquet file per variable into *directory*.

        Streamed to disk in :meth:`primal`'s order, so the same model and data
        write the same bytes.

        Returns:
            Each variable's name, mapped to the file it was written to.
        """
        frames = self._readable(self._primals, 'the solution')
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        written = {name: out / f'{name}.parquet' for name in frames}
        for name, frame in frames.items():
            frame.sink_parquet(written[name])
        return written

    def close(self) -> None:
        """Release what this result holds early. Optional.

        Its frames, which carry both its own values and its hold on the label
        frames of the build it answered. Frames already read stay valid. Never
        the model or the solver, which are the
        :class:`~lpspec.api.BoundModel`'s to close: a result closed on the way
        out of a ``with`` block must not take down the model a loop is still
        solving, and a sibling result keeps its own.
        """
        self._primals = self._duals = self._activities = self._expressions = None

    def __enter__(self) -> Result:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False
