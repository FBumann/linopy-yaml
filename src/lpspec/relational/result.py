"""What a caller reads back — a solve's :class:`Result`, a build's :class:`Diagnostics`.

The objects ``lps.solve`` and ``bound.diagnostics()`` hand back, so they are
the pieces of this subpackage a reader meets without going looking. They live
beside the engine rather than in it because they answer different questions:
the engine *builds* a model, and these *read* one — a :class:`Result` owns its
solver vectors and a :class:`ReadBack`, the layout that lays them out over the
build's coordinates, so no reader ever goes back to the engine.

Named for linopy's envelope (``Result`` = status + solution + report) rather
than for our own decomposition, because our audience arrives from linopy and a
second vocabulary for one fact is a tax on every one of them
(docs/ARCHITECTURE.md, "Where a concept is already linopy's").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl

from lpspec.errors import LpspecError, NoSolutionError, unknown_name_message

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import pandas as pd
    import xarray as xr

    from lpspec.relational.status import SolveStatus


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

    #: The shape the solver was handed: columns, rows, and matrix entries.
    #: What ``check`` cannot answer, needing no data where this needs all of
    #: it, and the thing to report when a model is bigger than its author
    #: expected — a broadcast that multiplied rows shows up here first.
    columns: int
    rows: int
    nonzeros: int

    #: ``(constraint, rows_not_built)`` — every row that lost all its terms and
    #: so was not built (SPEC §6): without this record a declared constraint
    #: could go unenforced with no way to notice. Empty for a model whose every
    #: declared row reached the solver. Counts rather than coordinates: the
    #: label of an unbuilt row does not exist, so naming which went would mean
    #: holding the pre-drop frame — memory proportional to the omission, on the
    #: path this package measures hardest.
    omissions: pl.DataFrame

    #: How many times this model has been solved, and how many of those solves
    #: loaded the solver from scratch instead of pushing values onto one that
    #: already held it. Read together: ``loads == 1`` is a driver on the fast
    #: path — the first solve had nothing to keep — and ``loads == solves`` on
    #: an iterating driver is the difference between "lpspec is slow" and
    #: "this model masks on a parameter that varies".
    solves: int
    loads: int


@dataclass(frozen=True)
class Reader:
    """One declaration's read-back: its coordinates, and its slice of a vector.

    The coordinate frame is the build's and shared; the slice bounds are what
    make a declaration's share of a solver vector a slice rather than a join —
    the labeller hands every declaration a contiguous, dense run of labels.
    """

    coordinates: pl.LazyFrame
    dims: tuple[str, ...]
    start: int
    height: int
    #: Those of :attr:`dims` the binder encoded as ``Enum`` — its string ones,
    #: recorded here because the binder is gone by the time anyone reads.
    string_dims: tuple[str, ...]

    def frame(self, values: pl.Series) -> pl.LazyFrame:
        """This declaration's coordinates in label order, beside its values.

        **The order is not re-established here, because it was never lost**:
        the labeller numbers a sorted frame and hands back a label-ascending
        one, and the solver's vector is positional in the same index, so
        coordinates and values line up by construction. The slice is attached
        as a column rather than concatenated as a frame, so a mismatched
        length raises instead of padding with nulls.

        **Dim columns leave in ``String``**, where the build holds them as
        ``pl.Enum`` (#541). That encoding is internal and every gram of its
        win is upstream of here, but a returned frame is something a caller
        *joins against their own data* — and polars refuses ``Enum`` against
        ``String`` with a message about dtypes that names nothing about the
        cause. Two frames of one sweep will not even concatenate when their
        slices bound different members.

        The cast sits inside this projection rather than after it, so the
        string column is produced once instead of widened from an Enum that
        also exists, which is cheaper in both wall and peak (#593).
        Declaration order is the *row* order and survives, never having been
        the dtype's to carry.
        """
        labelled = self.coordinates.select(*self.dims).with_columns(values.slice(self.start, self.height))
        return labelled.with_columns(pl.col(d).cast(pl.String) for d in self.string_dims)


@dataclass(frozen=True)
class ReadBack:
    """How solver vectors lay out over one build's coordinates — a result's own.

    Captured at solve time and owned by the :class:`Result`, so a reader never
    goes back to the engine. References, not copies: every result of one build
    points at the same label frames, and a rebind builds new frames without
    touching these. What a retained result costs is exactly these frames
    staying alive until it is dropped or closed — the four model frames and
    the solver are never held here.
    """

    #: One :class:`Reader` per declaration, in declaration order.
    variables: Mapping[str, Reader]
    constraints: Mapping[str, Reader]
    #: The variables that are not continuous — what decides whether "no duals"
    #: means "undefined for this model" or "the solve left none".
    discrete: tuple[str, ...]

    def variable(self, name: str) -> Reader:
        return _named(self.variables, name, 'variable')

    def constraint(self, name: str) -> Reader:
        return _named(self.constraints, name, 'constraint')

    def no_duals_reason(self, termination_condition: str) -> str:
        """Why a solve that *did* leave values still has no duals.

        Integrality is decidable from the model, and naming the variable is
        actionable where "the solver reported none" is not.
        """
        if self.discrete:
            names = ', '.join(f"'{n}'" for n in self.discrete)
            return (
                f'duals are undefined for a mixed-integer model: {names} '
                f'{"is" if len(self.discrete) == 1 else "are"} not continuous. '
                f'Drop the integrality to price the LP relaxation instead.'
            )
        return (
            f'the solver returned no dual solution, though the solve terminated '
            f'{termination_condition!r}. Duals come from a simplex basis, which a '
            f'run stopped short of one does not have.'
        )


def _named(readers: Mapping[str, Reader], name: str, kind: str) -> Reader:
    try:
        return readers[name]
    except KeyError:
        raise KeyError(unknown_name_message(kind, name, readers)) from None


@dataclass
class Result:
    """What a solve returned — the outcome, and access to any values.

    Returned whatever the solve concluded: test :attr:`has_primal` before
    reading values, or catch :class:`~lpspec.errors.NoSolutionError`. The
    values are this result's own, so a later solve on the same model does not
    rewrite them, and there is no lifetime to manage — :meth:`close` releases
    what this result holds early, and nothing breaks without it.

    A rebind is no exception. A result owns everything it reads — its solver
    vectors, and a :class:`ReadBack` referencing the label frames of the build
    it answered — so it outlives anything done to the model afterwards: a
    rebind, another solve, ``bound.close()``. What retaining one costs is those
    label frames staying alive, which matters once a caller keeps several, as a
    sweep, a rolling horizon and Benders all do.
    """

    _status: SolveStatus
    _objective: float
    #: The layout the readers lay values out over. ``None`` is what
    #: :meth:`close` leaves behind — the one unreadable state.
    _read: ReadBack | None
    _primal_values: pl.Series | None = None
    _dual_values: pl.Series | None = None

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

    def _readable(self, what: str) -> ReadBack:
        if self._read is None:
            raise LpspecError(
                f'cannot read {what}: this result was closed, and closing releases its values and its '
                f'hold on the coordinates they lay out over. Frames already read stay valid — they are '
                f'their own data — so read what you need before close(), or drop the `with` and close '
                f'when you are done.'
            )
        if self._status.is_readable:
            return self._read
        raise NoSolutionError(
            f'cannot read {what}: the solve terminated {self.termination_condition!r} '
            f'({self._status.solver_wording}), so there are no values to read. Test '
            f'`has_primal` first. This raises rather than returning, because the solver '
            f'hands back a full-length vector of zeros either way and it is '
            f'indistinguishable from an answer.'
        )

    def _primal_frame(self, name: str) -> pl.LazyFrame:
        read = self._readable(f"the primal of '{name}'")
        assert self._primal_values is not None, 'no solve has stored a primal'
        return read.variable(name).frame(self._primal_values)

    def primal(self, name: str) -> pl.DataFrame:
        """The tidy solution of variable *name* — ``(dims…, value)``.

        Rows come back in label order, row-major over the variable's coordinate
        product, so two reads and two runs agree.

        Raises:
            NoSolutionError: The solve left no values to read.
            LpspecError: The model was rebound after this solve.
            KeyError: No variable is called *name*.
        """
        return self._primal_frame(name).collect(engine='streaming')

    def dual(self, name: str) -> pl.DataFrame:
        """Shadow prices of constraint *name* — ``(dims…, value)``.

        :meth:`primal`'s shape and order, over constraint rows.

        Raises:
            NoSolutionError: The solve left no values at all.
            LpspecError: It left primals but no duals — an integer variable
                makes them undefined — or the model was rebound since.
            KeyError: No constraint is called *name*.
        """
        read = self._readable(f"the dual of '{name}'")
        if self._dual_values is None:
            raise LpspecError(read.no_duals_reason(self.termination_condition))
        return read.constraint(name).frame(self._dual_values).collect(engine='streaming')

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
        wanted = names or tuple(self._readable('the solution').variables)
        return tidy_to_dataset(wanted, self.to_dataarray)

    def to_parquet(self, directory: str | Path) -> dict[str, Path]:
        """Write one parquet file per variable into *directory*.

        Streamed to disk in :meth:`primal`'s order, so the same model and data
        write the same bytes.

        Returns:
            Each variable's name, mapped to the file it was written to.
        """
        read = self._readable('the solution')
        assert self._primal_values is not None, 'no solve has stored a primal'
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for name, reader in read.variables.items():
            written[name] = out / f'{name}.parquet'
            reader.frame(self._primal_values).sink_parquet(written[name])
        return written

    def close(self) -> None:
        """Release what this result holds early. Optional.

        Frames already read stay valid; this result's own stop working. Never
        the model or the solver, which are the
        :class:`~lpspec.api.BoundModel`'s to close: a result closed on the way
        out of a ``with`` block must not take down the model a loop is still
        solving, and a sibling result keeps its own read-back.
        """
        self._primal_values = self._dual_values = None
        self._read = None

    def __enter__(self) -> Result:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False
