"""What a solve returned — the caller's end of the relational lane.

The object ``lps.solve`` hands back, so it is the one piece of this subpackage a
reader meets without going looking. It lives beside the executor rather than in
it because the two answer different questions: the executor *builds* a model,
and this *reads* one — the accessors are joins against the label frames plus
whatever vector the solver left.

Named for linopy's envelope (``Result`` = status + solution + report) rather
than for our own decomposition, because our audience arrives from linopy and a
second vocabulary for one fact is a tax on every one of them
(docs/ARCHITECTURE.md, "Where a concept is already linopy's").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from lpspec.errors import LpspecError, NoSolutionError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd
    import polars as pl
    import xarray as xr

    from lpspec.relational.engines.polars.executor import PolarsExecutor
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


@dataclass
class Result:
    """What a solve returned — the outcome, and access to any values.

    Named for linopy's envelope rather than its ``Solution``: it is returned
    when the solve produced *nothing*, so "solution" would be a lie in exactly
    the case a caller most needs to notice.

    **No lifetime to manage.** The model is frames this process owns, so the
    readers stay valid as long as the object does; :meth:`close` releases a
    large one early but nothing breaks without it.

    **Its values are its own** — held here rather than read back off the
    executor, so a later solve cannot rewrite what an earlier result reports.
    Only the label frames are shared, and a re-solve does not touch them. That
    matters once a caller retains several results, which a sweep, a rolling
    horizon and Benders all do.

    **A rebind is the exception, and it is the one lifetime here.** Rebinding
    replaces the label frames the readers join through, so a result from before
    it stops reading rather than laying its values out over coordinates that
    are no longer the ones they were computed on. Read out what you need first
    — ``to_pandas``, ``to_parquet``, or ``primal`` itself, all of which return
    frames that are their own data.
    """

    _status: SolveStatus
    _objective: float
    _executor: PolarsExecutor
    #: The build this answers. The executor's own counter moves past it when
    #: something rebinds, which is how a stale result knows it is one.
    _generation: int = 0
    _primal_values: pl.Series | None = None
    _dual_values: pl.Series | None = None
    _closed: bool = False

    @property
    def _current(self) -> bool:
        """Whether the model behind this result is still the one it answered."""
        return self._generation == self._executor._generation

    @property
    def status(self) -> str:
        """Coarse outcome: ``ok`` / ``warning`` / ``error`` / ``aborted`` / ``unknown``."""
        return self._status.status

    @property
    def termination_condition(self) -> str:
        """What the solver said — ``optimal``, ``infeasible``, ``time_limit``…"""
        return self._status.termination_condition

    @property
    def is_ok(self) -> bool:
        """linopy's rollup: not an error, an abort or a refusal."""
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

    def _require_solution(self, what: str) -> None:
        if self._closed:
            raise LpspecError(
                f'cannot read {what}: this result was closed, and closing releases the model the '
                f'readers join against. Frames already read stay valid — they are their own data — so '
                f'read what you need before close(), or drop the `with` and close when you are done.'
            )
        if not self._current:
            raise LpspecError(
                f'cannot read {what}: the model was rebound after this solve, and a rebind replaces '
                f'the label frames the readers join through — laying these values out over them would '
                f'report an answer over coordinates it was not computed on. Read what you need before '
                f'rebinding; a frame already read is its own data and stays valid.'
            )
        if self._status.is_readable:
            return
        raise NoSolutionError(
            f'cannot read {what}: the solve terminated {self.termination_condition!r} '
            f'({self._status.solver_wording}), so there are no values to read. Test '
            f'`has_primal` first. This raises rather than returning, because the solver '
            f'hands back a full-length vector of zeros either way and it is '
            f'indistinguishable from an answer.'
        )

    def primal(self, name: str) -> pl.DataFrame:
        """The tidy solution of *name* — ``(dims…, value)``.

        Rows come back in **label order** — row-major over the variable's
        coordinate product, what ``var_label`` already encodes and what the LP
        sink writes — so two reads agree, two runs agree, and a solution file
        can be diffed.
        """
        self._require_solution(f"the primal of '{name}'")
        return self._executor._primal(name, self._primal_values)

    def dual(self, name: str) -> pl.DataFrame:
        """Shadow prices of constraint *name*: ``(dims…, value)``.

        :meth:`primal` against the row frame rather than a column one, in that
        method's order. The two empty cases are different failures and both
        raise rather than return zeros: no values at all is
        :class:`~lpspec.errors.NoSolutionError`, primals without duals — any
        integer variable makes them undefined — a
        :class:`~lpspec.errors.LpspecError`.
        """
        self._require_solution(f"the dual of '{name}'")
        if self._dual_values is None:
            raise LpspecError(self._executor._no_duals_reason(self.termination_condition))
        return self._executor._dual(name, self._dual_values)

    def to_pandas(self, name: str) -> pd.DataFrame:
        """:meth:`primal` as a tidy :class:`pandas.DataFrame`."""
        self._require_solution(f"the primal of '{name}'")
        return tidy_to_pandas(self._executor._primal(name, self._primal_values))

    def to_dataarray(self, name: str) -> xr.DataArray:
        """``primal(name)`` as a labelled :class:`xarray.DataArray`.

        The bridge to array post-processing — ``.sel``, resampling, duration
        curves. A masked coordinate has no row and comes back NaN.
        """
        return tidy_to_dataarray(self.to_pandas(name), name)

    def to_dataset(self, *names: str) -> xr.Dataset:
        """Variables as one :class:`xarray.Dataset`; all of them by default.

        Costs what it says: each variable arrives dense over its own dims,
        whatever the mask removed, and all of them at once. On a large model,
        name the few you need or use :meth:`to_parquet`.
        """
        assert self._executor._program is not None
        wanted = names or tuple(v.name for v in self._executor._program.variables)
        return tidy_to_dataset(wanted, self.to_dataarray)

    def to_parquet(self, directory: str | Path) -> dict[str, Path]:
        """One parquet file per variable, ``(dims…, value)``. Returns name → path.

        Sunk straight to disk, never copied into a second representation, in
        :meth:`primal`'s order — so the same model and data write the same
        bytes.
        """
        self._require_solution('the solution')
        return self._executor._solution_to_parquet(Path(directory), self._primal_values)

    def close(self) -> None:
        """Release the built model early. Optional — see the class docstring.

        Drops this result's own values as well as the shared model, so closing
        still frees everything one solve allocated; a sibling result keeps its.

        A result the model has outgrown releases only its own values: the
        built model it names is no longer the one it answered, and closing a
        rebound model on the way out of an old ``with`` block would take the
        live one down.
        """
        self._primal_values = self._dual_values = None
        self._closed = True
        if self._current:
            self._executor.close()

    def __enter__(self) -> Result:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False
