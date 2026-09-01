"""Solving strategies: one plan per slice, folded.

A plan cannot contain a loop; a *process* may loop over plans
(docs/about/ceiling.md). So a strategy is a driver above :mod:`lpspec.api`,
built from the public verbs — never a language or engine feature.

Every strategy is the same fold: **partition → bind → solve → carry → stitch**.
Only how slices are cut and whether they couple differs. A serial fold builds
once and rebinds each slice (:func:`_serially`); under a process pool it builds
per slice (:func:`_pooled`), a built model being the one thing that cannot
cross. Both yield an :class:`_Answer`, and the fold that absorbs them is
written once.

    scenario / sweep    ``EachCoordinate('scenario')``              independent
    myopic pathway      ``EachCoordinate('period', ordered=True)``  + ``carry``
    rolling horizon     ``EachWindow('snapshot', 48, 24, 't')``     + ``carry``

**A partition is a filter on the sources, not a narrower index** — the
containment check refuses parameter rows outside a narrowed index, so an axis
rewrites the rows and the index together.

The caller-facing rules are [docs/reference/sweeps.md](../../docs/reference/sweeps.md).
"""

from __future__ import annotations

import io
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import polars as pl
from math_spec import to_program

from lpspec.api import build, check
from lpspec.api import solve as _solve
from lpspec.errors import DataError, LpspecError
from lpspec.frames import as_frame
from lpspec.relational.result import tidy_to_dataarray, tidy_to_dataset, tidy_to_pandas

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence

    import pandas as pd
    import xarray as xr
    from math_spec.program import Program

    from lpspec.api import Buildable
    from lpspec.relational.result import Keep

#: Parquet rather than pickle, and not a knob: zstd measured smaller *and*
#: faster than pickling the frame, on compressible and incompressible data
#: alike (#459).
_COMPRESSION = 'zstd'


class _Cut(NamedTuple):
    """One slice of a sweep: the key, and the sources that build it.

    A tuple on purpose: a hand-built axis is a plain list of ``(key, sources)``,
    and those unpack the same way.
    """

    key: Any
    sources: Mapping[str, Any]


class _SliceMeta(NamedTuple):
    """One row of :attr:`Runs.objective`: how a slice terminated, and its objective."""

    status: str
    termination_condition: str
    objective: float


@dataclass(frozen=True)
class _CarryRule:
    """One resolved carry: which variable moves into a parameter, and how.

    ``dropped`` is the one dimension the carry collapses — ``None`` when the
    whole frame moves forward — and ``index`` names a coordinate of it.
    """

    variable: str
    dropped: str | None
    index: int | None

    @classmethod
    def resolved(cls, program: Program, parameter: str, variable: str, index: int | None) -> _CarryRule:
        """One carry checked against the plan — construction and validation, together.

        The variable's dims minus the parameter's is the one dimension the
        carry collapses; everything else passes through, so a myopic pathway
        hands a whole capacity vector forward rather than one number at a time.
        Nothing here reads data, which is why the plan resolves before the axis
        cuts any.
        """
        if parameter not in program.parameters:
            raise LpspecError(f'carry writes parameter {parameter!r}, which the spec does not declare')
        if variable not in program.variables:
            raise LpspecError(f'carry reads variable {variable!r}, which the spec does not declare')
        over = list(program.parameters[parameter].dims)
        source = list(program.variables[variable].dims)
        if missing := [d for d in over if d not in source]:
            raise LpspecError(
                f'carry {parameter!r} <- {variable!r} cannot line up: {parameter!r} is over {over}, and '
                f'{variable!r} is over {source}, which has no {missing}. A carry copies a variable into a '
                f'parameter, so the parameter cannot be over more than the variable is.'
            )
        dropped = [d for d in source if d not in over]
        if len(dropped) > 1:
            raise LpspecError(
                f'carry {parameter!r} <- {variable!r} would collapse {dropped} at once: {variable!r} is over '
                f'{source} and {parameter!r} over {over}. An index names a coordinate of one dimension, so '
                f'reduce the others in the YAML — a derived variable is where the oracle can see the math.'
            )
        if dropped and index is None:
            raise LpspecError(
                f'carry {parameter!r} <- {variable!r} drops {dropped[0]!r} and so needs an index: '
                f'({variable!r}, <{dropped[0]}>). With overlap the coordinate to carry is the last one you '
                f'*keep* — EachWindow(length=48, step=24) carries at 23, not 47 — which is why there is no default.'
            )
        if not dropped and index is not None:
            raise LpspecError(
                f'carry {parameter!r} <- ({variable!r}, {index}) has nothing to index: both are over {over}, '
                f'so the whole frame is what moves forward. Pass ({variable!r}, None).'
            )
        return cls(variable, dropped[0] if dropped else None, index)

    def value_from(self, frames: Mapping[str, pl.DataFrame], parameter: str, key: Any) -> pl.DataFrame:
        """What this rule hands the next slice, read out of one slice's primals.

        ``index`` is a **coordinate** of the dropped dimension, never a row
        number — the only one of the two that still means something once a
        second dimension is there.

        Raises:
            LpspecError: ``index`` names a coordinate the slice does not have
                — a short tail window has fewer than a full one.
        """
        frame = frames[self.variable]
        if self.dropped is None:
            return frame
        picked = frame.filter(pl.col(self.dropped) == self.index).drop(self.dropped)
        if picked.is_empty():
            coordinates = frame[self.dropped].unique().sort().to_list()
            raise LpspecError(
                f'carry {parameter!r} <- ({self.variable!r}, {self.index}) is out of range: slice {key!r} has '
                f'no {self.dropped} == {self.index}. Its coordinates run '
                f'{coordinates[0]!r}..{coordinates[-1]!r}, and a short tail window has fewer than a full one.'
            )
        return picked


@dataclass(frozen=True)
class _Answer:
    """One slice, solved and read out — what the fold absorbs.

    What :func:`_serially` and :func:`_pooled` both produce. Plain data
    throughout, because a worker returns it and so it has to pickle: frames,
    strings and numbers, never a result or a model.
    """

    meta: _SliceMeta
    primals: dict[str, pl.DataFrame]
    duals: dict[str, pl.DataFrame]
    #: Every declared named expression, evaluated at this slice's solution.
    expressions: dict[str, pl.DataFrame]
    #: Why this slice has none, when it has none. Carried rather than raised:
    #: one mixed-integer slice must not fail a whole sweep.
    no_duals: str | None
    #: Per expression, why this slice could not evaluate it — the same
    #: carried-not-raised rule, per name because each fails on its own data.
    no_expressions: dict[str, str]


@dataclass(frozen=True)
class _OriginalIndex:
    """The way back from a windowed sweep's slices to the dimension it sliced.

    ``owned`` is ``(key, local, dim)`` for the coordinates each window is
    *responsible* for — its first ``step``, the rest being lookahead the next
    window recomputes. One-way: the lookahead rows are not in it, so a sliced
    frame cannot be rebuilt from it — slicing stays
    :meth:`EachWindow._slices`' business.
    """

    local: str
    dim: str
    owned: pl.DataFrame

    def restore(self, frame: pl.DataFrame, key_name: str) -> pl.DataFrame:
        """*frame* over the dimension the axis sliced, rather than over its slices.

        The inner join against ``owned`` is the whole operation: it restores
        the original coordinate, and because a coordinate may appear only once
        under its own index, the lookahead rows have nowhere to go. Sorted on
        the restored dimension, so a sweep reads back in the caller's order.

        Raises:
            LpspecError: *frame* has no ``local`` column — the quantity was
                reduced over the sliced dimension, so there is no way back.
        """
        if self.local not in frame.columns:
            raise LpspecError(
                f'cannot read this over {self.dim!r}: the frame has no {self.local!r} column, because the '
                f'quantity was reduced over the sliced dimension — each row already covers a whole slice, '
                f'lookahead included under an overlapping window. Read it without original_index for the '
                f'per-slice values, or read a quantity that keeps {self.local!r} and aggregate the '
                f'stitched frame.'
            )
        keys = [key_name, self.local]
        restored = frame.join(self.owned, on=keys, how='inner').drop(keys)
        rest = [column for column in restored.columns if column not in (self.dim, 'value')]
        return restored.select(self.dim, *rest, 'value').sort(self.dim, *rest)


# ---------------------------------------------------------------------------
# axes — how slices are cut
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EachCoordinate:
    """One slice per coordinate of *dim* — a column the sources carry.

    Scenarios, draws, investment periods. Sources carrying *dim* are filtered
    to one coordinate and the column dropped, so the model never mentions it;
    every other source passes through untouched. ``ordered=True`` says the
    coordinates are a sequence, which a ``carry`` needs.
    """

    dim: str
    ordered: bool = False

    def _slices(self, sources: Mapping[str, Any], key_name: str) -> tuple[list[_Cut], _OriginalIndex | None]:
        """One cut per coordinate, keyed by it. Sources without *dim* pass through.

        No :class:`_OriginalIndex`: nothing was re-indexed, so a slice's frames
        already carry the coordinates they were solved over.
        """
        del key_name
        carrying, coordinates = _coordinates(sources, self.dim, 'slice')
        out: list[_Cut] = []
        for key in coordinates:
            cut = {name: _lazy(sources[name]).filter(pl.col(self.dim) == key).drop(self.dim) for name in carrying}
            out.append(_Cut(key, {**sources, **cut}))
        return out, None


@dataclass(frozen=True)
class EachWindow:
    """One slice per window of consecutive coordinates of *dim*.

    ``length`` is what the solver sees, ``step`` is what the window keeps, and
    ``length > step`` is overlap. Both count coordinates rather than coordinate
    values, so *dim* need only be orderable — datetimes, strings and gapped
    integers all work. The dimension is re-indexed rather than dropped, into a
    dense ``0..n-1`` column the model addresses by the name ``into`` gives it.
    """

    dim: str
    length: int
    step: int
    into: str

    ordered: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.length < 1 or self.step < 1:
            raise ValueError(f'length and step must be positive (got length={self.length}, step={self.step})')
        if self.step > self.length:
            raise ValueError(
                f'step={self.step} exceeds length={self.length}, which would skip coordinates between '
                f'windows. step == length is contiguous; step < length overlaps.'
            )
        if not self.into:
            raise ValueError('into must name the local index the spec declares — it has no default')
        if self.into == self.dim:
            raise ValueError(f'into={self.into!r} must differ from dim — the local index replaces the global one')

    def _slices(self, sources: Mapping[str, Any], key_name: str) -> tuple[list[_Cut], _OriginalIndex]:
        """One cut per window, keyed by its **first coordinate**.

        Keyed by the coordinate rather than the window's position, which is
        what names a window in the caller's own terms. Sources without *dim*
        pass through untouched.

        The filter leads because it is what a scan can push down; the
        re-indexing that follows is over a frame already cut to one window.

        **A window owns its first ``step`` coordinates**, and the
        :class:`_OriginalIndex` records which — the rest is lookahead the next window
        recomputes. The final window can hold no more than ``step``, its start
        being the last multiple of ``step`` below the end, so the same rule
        keeps all of it and nothing is dropped off the tail.
        """
        carrying, coordinates = _coordinates(sources, self.dim, 'window')
        out: list[_Cut] = []
        owned: list[dict[str, Any]] = []
        for start in range(0, len(coordinates), self.step):
            window = coordinates[start : start + self.length]
            local = {coordinate: position for position, coordinate in enumerate(window)}
            cut = {
                name: (
                    _lazy(sources[name])
                    .filter(pl.col(self.dim).is_in(window))
                    .with_columns(pl.col(self.dim).replace_strict(local, return_dtype=pl.Int64).alias(self.into))
                    .drop(self.dim)
                )
                for name in carrying
            }
            out.append(_Cut(window[0], {**sources, **cut, self.into: range(len(window))}))
            owned.extend(
                {key_name: window[0], self.into: position, self.dim: coordinate}
                for position, coordinate in enumerate(window[: self.step])
            )
        return out, _OriginalIndex(self.into, self.dim, pl.DataFrame(owned))


#: What ``axis=`` accepts. A plain list of ``(key, sources)`` is also
#: taken, so an irregular ladder or a hand-built draw needs no third class.
Axis = EachCoordinate | EachWindow


# ---------------------------------------------------------------------------
# the result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Runs:
    """What a fold returned: frames keyed by slice, never a scalar.

    :class:`~lpspec.relational.result.Result`'s readers one dimension wider —
    same names, same shapes, the slice key prepended. Nothing is combined
    across slices: each row says which slice computed it. A windowed sweep
    reads over that key unless a reader asks ``original_index=True``, which
    gives the dimension the axis sliced and drops the lookahead rows every
    overlapping window recomputed.
    """

    key_name: str
    meta: pl.DataFrame
    #: Per slice, not concatenated. Joining them is the reader's work so a
    #: sweep pays it for the names actually read, and so the concatenated copy
    #: never exists beside the pieces it was built from.
    _primals: dict[str, list[pl.DataFrame]] = field(repr=False, default_factory=dict)
    _duals: dict[str, list[pl.DataFrame]] = field(repr=False, default_factory=dict)
    _expressions: dict[str, list[pl.DataFrame]] = field(repr=False, default_factory=dict)
    _no_duals: str | None = field(repr=False, default=None)
    _no_expressions: dict[str, str] = field(repr=False, default_factory=dict)
    _original: _OriginalIndex | None = field(repr=False, default=None)

    @property
    def objective(self) -> pl.DataFrame:
        """``(key, status, termination_condition, objective)``, in slice order."""
        return self.meta

    @property
    def keys(self) -> list[Any]:
        return self.meta[self.key_name].to_list()

    def _read(
        self, held: Mapping[str, list[pl.DataFrame]], kind: str, name: str, absent: str | None = None
    ) -> pl.DataFrame:
        """*name*'s frames from *held*, concatenated — or why there are none.

        *absent* is a reason the fold already knows, which beats one derived
        from what the sweep happens to hold.
        """
        if name not in held:
            raise LpspecError(absent or _nothing_to_read(kind, name, held, self.meta))
        return pl.concat(held[name])

    def primal(self, name: str, *, original_index: bool = False) -> pl.DataFrame:
        """One variable's values across every slice, the slice key prepended.

        A slice that reached no solution contributes no rows, so this can be
        shorter than the sweep; :attr:`objective` is one row per slice always.

        Args:
            name: A variable the sweep's spec declares.
            original_index: Read over the dimension the axis sliced instead of
                over the slice key.

        Raises:
            LpspecError: No slice of the sweep produced *name*.
        """
        return self._reindexed(self._read(self._primals, 'variable', name), original_index=original_index)

    def dual(self, name: str, *, original_index: bool = False) -> pl.DataFrame:
        """One constraint's shadow prices across every slice, the key prepended.

        :meth:`primal`'s shape and arguments. A slice whose model had an
        integer variable contributes no duals; over the original index each
        coordinate carries the price of the window that owns it, never a blend
        of several.

        Raises:
            LpspecError: No slice produced duals for *name* — the message says
                which of the two it was.
        """
        return self._reindexed(
            self._read(self._duals, 'constraint', name, self._no_duals), original_index=original_index
        )

    def expression(self, name: str, *, original_index: bool = False) -> pl.DataFrame:
        """One named expression's values across every slice, the slice key prepended.

        :meth:`primal`'s shape and arguments, for the quantities the spec
        declares under ``expressions:`` — each slice's value was evaluated at
        that slice's solution when the fold read it.

        Over the original index each coordinate carries the value of the window
        that owns it — the recomputed lookahead rows are dropped, which is what
        makes summing the stitched frame safe where summing per-window values
        double-counts.

        Raises:
            LpspecError: No slice produced *name* — an evaluation that failed
                on every slice carries its own reason — or ``original_index``
                on a quantity reduced over the sliced dimension.
        """
        return self._reindexed(
            self._read(self._expressions, 'named expression', name, self._no_expressions.get(name)),
            original_index=original_index,
        )

    def _reindexed(self, frame: pl.DataFrame, *, original_index: bool) -> pl.DataFrame:
        """*frame* over the dimension the axis sliced, rather than over its slices.

        Every axis answers it: :class:`EachCoordinate` and a hand-built axis
        re-indexed nothing — their key column already *is* a coordinate of the
        answer — so there the frame comes back unchanged, a satisfied request
        rather than an ignored one.
        """
        if not original_index or self._original is None:
            return frame
        return self._original.restore(frame, self.key_name)

    def to_pandas(self, name: str, *, original_index: bool = False) -> pd.DataFrame:
        """:meth:`primal` as a tidy :class:`pandas.DataFrame`.

        The name is resolved before pandas is imported, so a sweep that never
        held *name* says so on any install.
        """
        return tidy_to_pandas(self.primal(name, original_index=original_index))

    def to_dataarray(self, name: str, *, original_index: bool = False) -> xr.DataArray:
        """:meth:`primal` as a :class:`xarray.DataArray`, the slice key a dimension.

        The extra dimension is named by the axis — a scenario sweep gives
        ``(scenario, …)`` and a window ``(<dim>_start, …)``. A slice that
        reached no solution has no rows and comes back NaN, the same answer a
        masked coordinate gets from ``Result``. ``original_index=True`` gives
        the array over the dimension the axis sliced instead, so a rolling
        horizon comes back indexed by time.
        """
        return tidy_to_dataarray(self.to_pandas(name, original_index=original_index), name)

    def to_dataset(self, *names: str) -> xr.Dataset:
        """Kept variables as one :class:`xarray.Dataset`; all of them by default.

        Costs more than ``Result``'s does — each variable arrives dense over
        its own dims *and* over every slice. Name the few you need, or use
        :meth:`to_parquet`.

        No ``original_index``: this and :meth:`to_parquet` export what the
        sweep *holds*, and the original index is lossy — a bulk export is the
        wrong place to drop the lookahead rows.

        Raises:
            LpspecError: The sweep holds no variable values at all.
        """
        wanted = names or tuple(sorted(self._primals))
        if not wanted:
            raise LpspecError(_nothing_to_read('variable', 'anything', self._primals, self.meta))
        return tidy_to_dataset(wanted, self.to_dataarray)

    def to_parquet(self, directory: str | Path) -> dict[str, Path]:
        """One parquet file per variable the sweep holds, ``(key, dims…, value)``.

        Written in :meth:`primal`'s order, so the same sweep writes the same
        bytes. A *copy out* of frames already held — what a sweep holds is #610.

        Returns:
            Each variable's name, mapped to the file it was written to.

        Raises:
            LpspecError: The sweep holds no variable values at all.
        """
        if not self._primals:
            raise LpspecError(_nothing_to_read('variable', 'anything', self._primals, self.meta))
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for name in sorted(self._primals):
            path = directory / f'{name}.parquet'
            self.primal(name).write_parquet(path)
            written[name] = path
        return written

    def __len__(self) -> int:
        return self.meta.height


def _nothing_to_read(kind: str, name: str, held: Mapping[str, object], meta: pl.DataFrame) -> str:
    """Why *name* has no frame.

    A sweep keeps everything every slice produced, so a declared name arrives
    here only when no slice produced it; an undeclared name arrives here too,
    and the two are told apart by what the sweep did hold.
    """
    conditions = ', '.join(sorted(set(meta['termination_condition'].to_list())))
    if held:
        listed = ', '.join(repr(k) for k in sorted(held))
        return (
            f'no {kind} {name!r} in this sweep — it holds {listed}. '
            f'If the spec declares it, no slice produced one: all {meta.height} terminated {conditions}.'
        )
    return (
        f'this sweep holds no {kind} frames at all — every one of its {meta.height} slices '
        f'terminated {conditions}. The fold ran; the models did not solve. '
        f'runs.objective carries the status of each slice.'
    )


def solve_over(
    spec: Buildable,
    sources: Mapping[str, Any],
    axis: Axis | Sequence[tuple[Any, Mapping[str, Any]]],
    *,
    carry: Mapping[str, tuple[str, int | None]] | None = None,
    key_name: str | None = None,
    executor: Any = None,
    workers_share_fs: bool | None = None,
    solver_options: Mapping[str, Any] | None = None,
    solver_name: str = 'highs',
    keep: Keep = 'solver',
) -> Runs:
    """Solve *spec* once per slice of *axis* and fold the answers together.

    The caller-facing rules — what a carry copies, how a key column is named,
    which executor to choose — are the table in
    [docs/reference/sweeps.md](../../docs/reference/sweeps.md). What this
    docstring adds is the order the work happens in.

    **Everything answerable from the declarations is answered before a source
    is read** — a mistyped carry, a key column that collides, an axis a carry
    cannot run on each cost a parse rather than a scan of every parquet file.
    The plan then rides down to the slices already parsed, so no slice — and
    no worker — reads the same YAML again.

    **It is a fold.** The previous slice's model is released as the loop goes,
    so build peak stays at one slice however many there are; what accumulates
    is the answer.

    **``keep`` reaches every slice unchanged**, defaulting as
    :meth:`~lpspec.api.Model.solve` does. Whether ``keep='progress'``
    pays is a question about one *model* — on some, carrying is the slower
    path by a wide margin — so the fold offers the option and picks neither.
    The pooled branch builds per slice and can keep nothing at all; asking
    there is not an error.

    **The last slice carries nothing**, there being no next slice to read it —
    a short tail window can hold fewer coordinates than the carry index names.

    **A process pool must not use the ``fork`` start method.** polars' thread
    pool does not survive a fork, and a forked worker hangs rather than
    failing. Pass a ``spawn`` context, and give the entry point the ``__main__``
    guard it requires:

    .. code-block:: python

        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(4, mp_context=ctx) as pool:
            runs = lps.solve_over(spec, sources, axis, executor=pool)

    Returns:
        Every slice's answers, keyed by slice.

    Raises:
        LpspecError: A carry together with an executor — a carried value makes
            each slice depend on the one before, so they cannot run
            concurrently — or a carry on an unordered axis.
    """
    if carry and executor is not None:
        raise LpspecError(
            'carry and executor are mutually exclusive: a carried value makes slice i+1 depend on '
            "slice i's answer, so the slices cannot run concurrently. Drop the executor, or drop the carry."
        )
    if isinstance(axis, (EachCoordinate, EachWindow)) and carry and not axis.ordered:
        raise LpspecError(
            f'carry needs an ordered axis: {axis!r} has no defined "next" slice for a value to move into. '
            f'EachCoordinate(..., ordered=True) says the coordinates are a sequence.'
        )
    program = check(spec)
    plan = {p: _CarryRule.resolved(program, p, v, i) for p, (v, i) in (carry or {}).items()}
    key_name = _key_column(axis, key_name, program)

    if isinstance(axis, (EachCoordinate, EachWindow)):
        cut, original = axis._slices(sources, key_name)
    else:
        cut, original = list(axis), None
    cuts = [_Cut(*entry) for entry in cut]
    if not cuts:
        raise DataError('the axis produced no slices')
    solving = {'solver_name': solver_name, 'solver_options': dict(solver_options or {}) or None}
    rows: list[dict[str, Any]] = []
    primals: defaultdict[str, list[pl.DataFrame]] = defaultdict(list)
    shadow: defaultdict[str, list[pl.DataFrame]] = defaultdict(list)
    valued: defaultdict[str, list[pl.DataFrame]] = defaultdict(list)
    no_duals: str | None = None
    no_expressions: dict[str, str] = {}

    answered = (
        _serially(program, cuts, solving, plan, keep)
        if executor is None
        else _pooled(executor, workers_share_fs, spec, cuts, solving)
    )
    with closing(answered) as stream:
        for key, answer in stream:
            no_duals = no_duals or answer.no_duals
            for name, reason in answer.no_expressions.items():
                no_expressions.setdefault(name, reason)
            rows.append({key_name: key, **answer.meta._asdict()})
            for into, produced in ((primals, answer.primals), (shadow, answer.duals), (valued, answer.expressions)):
                for name, frame in produced.items():
                    into[name].append(frame.select(pl.lit(key).alias(key_name), pl.all()))

    return Runs(
        key_name=key_name,
        meta=pl.DataFrame(rows),
        _primals=dict(primals),
        _duals=dict(shadow),
        _expressions=dict(valued),
        _no_duals=no_duals,
        _no_expressions=no_expressions,
        _original=original,
    )


def _serially(
    program: Program,
    cuts: Sequence[_Cut],
    solving: Mapping[str, Any],
    plan: Mapping[str, _CarryRule],
    keep: Keep,
) -> Generator[tuple[Any, _Answer], None, None]:
    """Each slice's answer, off one model rebound in place.

    Every slice of a sweep is the same math over different numbers, which is
    what :meth:`~lpspec.api.Model.rebind` is for; a rebuild releases the
    previous model before it starts, so the fold holds one slice's model
    however many there are.

    **A slice that names something else is rebuilt, not rebound.** A cut is
    *total* where ``rebind`` is partial by construction: the two agree only
    while every slice names the same sources, which the class axes guarantee
    and a hand-built list does not. Compared by *name* — values are what a
    rebind exists to replace.

    **A generator because of the carry**: slice ``i+1``'s sources are not
    known until slice ``i``'s frames have been read, and resuming after the
    yield is where that happens. The caller closes this — that is what
    releases the model when a fold is abandoned part way.
    """
    model: Any = None
    named: frozenset[str] | None = None
    state: dict[str, Any] = {}
    try:
        for position, cut in enumerate(cuts):
            sources = {**cut.sources, **state}
            names = frozenset(sources)
            if names == named:
                model.rebind(sources)
            else:
                if model is not None:
                    model.close()
                model, named = build(program, sources), names
            answer = _answers(model.solve(**solving, keep=keep), program)
            yield cut.key, answer
            if plan and position < len(cuts) - 1:
                state = {p: rule.value_from(answer.primals, p, cut.key) for p, rule in plan.items()}
    finally:
        if model is not None:
            model.close()


def _pooled(
    executor: Any,
    workers_share_fs: bool | None,
    spec: Buildable,
    cuts: Sequence[_Cut],
    solving: Mapping[str, Any],
) -> Generator[tuple[Any, _Answer], None, None]:
    """The same, from slices built independently and possibly elsewhere.

    Yielded in **slice order, never completion order**: the futures are walked
    in the order they were submitted, so a sweep cannot reorder itself under a
    pool. A built model cannot cross a process, so this branch builds per
    slice — the same fact that makes ``carry`` and ``executor`` mutually
    exclusive.

    **The spec crosses as the caller wrote it**, not as the plan: a
    ``Program`` holds its declarations in ``MappingProxyType``, which pickle
    refuses, and a worker that has to build anyway lowers the file it is given
    at no extra cost. Which is why a *pooled* sweep over a spec passed as an
    already-lowered ``Program`` is the one shape this cannot carry.
    """
    crosses = _crosses_a_process(executor)
    shared = _shares_filesystem(executor, workers_share_fs)
    call = dict(solving)
    memo: dict[str, tuple[Any, Any]] = {}
    futures = [
        executor.submit(
            _run_slice,
            spec,
            _encode(cut.sources, memo, workers_share_fs=shared) if crosses else dict(cut.sources),
            crosses,
            call,
        )
        for cut in cuts
    ]
    for cut, future in zip(cuts, futures, strict=True):
        answer = future.result()
        yield (
            cut.key,
            replace(
                answer,
                primals=_decode(answer.primals),
                duals=_decode(answer.duals),
                expressions=_decode(answer.expressions),
            ),
        )


def _answers(result: Any, program: Program) -> _Answer:
    """One slice's answer, read out of *result*: its meta row, and its frames.

    Read here rather than held, so that what a sweep accumulates is frames and
    never results — holding a result per slice would hold that slice's label
    frames with it (#634). Every declared expression is evaluated here,
    eagerly, for the same reason: the deferred reader holds the build's frames.

    **A slice that answered nothing is not a failure**, and neither is one
    whose duals are undefined: an integer variable makes them so, and one such
    slice must not fail a whole sweep. ``Result.dual`` already writes the
    sentence saying why, so it is caught and carried rather than rewritten.
    """
    meta = _SliceMeta(
        status=result.status,
        termination_condition=result.termination_condition,
        objective=result.objective if result.has_primal else float('nan'),
    )
    if not result.has_primal:
        return _Answer(meta, {}, {}, {}, None, {})
    primals = {name: result.primal(name) for name in program.variables}
    expressions: dict[str, pl.DataFrame] = {}
    no_expressions: dict[str, str] = {}
    for name in program.named_expressions:
        try:
            expressions[name] = result.expression(name)
        except LpspecError as exc:
            no_expressions[name] = str(exc)
    try:
        duals = {name: result.dual(name) for name in program.constraints}
        return _Answer(meta, primals, duals, expressions, None, no_expressions)
    except LpspecError as exc:
        return _Answer(meta, primals, {}, expressions, str(exc), no_expressions)


def _run_slice(
    spec: Buildable,
    encoded: dict[str, Any],
    encode_out: bool,
    call: dict[str, Any],
) -> _Answer:
    """One slice, start to finish, over plain data — the *pooled* branch.

    Module-level and closure-free on purpose: a remote executor has to pickle
    what it is handed, and a bound method or a lambda over the axis object
    cannot cross.
    """
    program = to_program(spec)
    with _solve(program, _decode(encoded), **call) as result:
        answer = _answers(result, program)
        if not encode_out:
            return answer
        return replace(
            answer,
            primals=_encode(answer.primals, {}),
            duals=_encode(answer.duals, {}),
            expressions=_encode(answer.expressions, {}),
        )


def _key_column(
    axis: Axis | Sequence[tuple[Any, Mapping[str, Any]]],
    key_name: str | None,
    program: Program,
) -> str:
    """What to call the column holding the slice key.

    The class axes know: a coordinate sweep keys on the dimension it cut, a
    window on where it *started* — never on ``dim`` itself, since a column
    called ``snapshot`` holding window starts would join against real
    snapshot-indexed data and keep a fraction of it, silently. A hand-built
    list knows neither and has to be told.
    """
    if key_name is None:
        if isinstance(axis, EachWindow):
            key_name = f'{axis.dim}_start'
        elif isinstance(axis, EachCoordinate):
            key_name = axis.dim
        else:
            raise LpspecError(
                'a hand-built axis needs key_name=: a list of cuts does not say what its keys are '
                "coordinates of, and 'slice' would be this library naming your axis for you. Pass "
                "key_name='draw', key_name='period', or whatever the keys actually are."
            )
    clashing = sorted(name for name, block in program.variables.items() if key_name in block.dims)
    if clashing:
        raise LpspecError(
            f'key_name={key_name!r} is already a dimension of {clashing}, so the slice key would collide '
            f'with a column those frames already carry. Name it something the spec does not use.'
        )
    return key_name


# ---------------------------------------------------------------------------
# the wire
# ---------------------------------------------------------------------------


def _shares_filesystem(executor: Any, declared: bool | None) -> bool:
    """Whether *executor*'s workers can read this process's paths.

    The two stdlib pools are the ones whose deployment is knowable: both run
    here, so both read the paths here. An executor this package did not ship is
    a transport it cannot ask, so it is assumed remote until *declared* says
    otherwise.
    """
    if declared is not None:
        return declared
    return isinstance(executor, ProcessPoolExecutor)


def _crosses_a_process(executor: Any) -> bool:
    """Whether a slice's sources have to be encoded to reach *executor*.

    A thread pool runs in this process, so encoding would be a parquet round
    trip for a boundary that is not there, and it measured as a large share of
    such a sweep (#459). Every other executor is assumed to cross, none of them
    being answerable.
    """
    return not isinstance(executor, ThreadPoolExecutor)


def _encode(
    sources: Mapping[str, Any], memo: dict[str, tuple[Any, Any]], *, workers_share_fs: bool = False
) -> dict[str, Any]:
    """Sources in the shape a worker can be handed.

    A path the workers can reach stays a path. A path they cannot travels as
    **its own bytes, untouched** — decoding and re-encoding a parquet file
    produces byte-identical output for 79x the CPU (#459).
    Anything held in memory is written to parquet, which beats pickling the
    frame on size and time.

    *memo* keeps a source no slice rewrote — the static tables, which is most
    of them — from being encoded once per slice. ``bytes`` is what
    :func:`_decode` reads back, and cannot be confused with a path.
    """
    out: dict[str, Any] = {}
    for name, obj in sources.items():
        is_path = isinstance(obj, (str, Path))
        if is_path and workers_share_fs:
            out[name] = obj
            continue
        cached = memo.get(name)
        if cached is not None and cached[0] is obj:
            out[name] = cached[1]
            continue
        if is_path:
            out[name] = Path(obj).read_bytes()
        else:
            buffer = io.BytesIO()
            _lazy(obj).collect().write_parquet(buffer, compression=_COMPRESSION)
            out[name] = buffer.getvalue()
        memo[name] = (obj, out[name])
    return out


def _decode(encoded: Mapping[str, Any]) -> dict[str, Any]:
    """The inverse of :func:`_encode`, and a pass-through for what never crossed.

    Called on every returned frame rather than only the encoded ones: a frame
    that stayed in this process is not ``bytes`` and comes back untouched, so
    the caller needs no branch and the two paths cannot answer differently.
    """
    return {name: pl.read_parquet(io.BytesIO(v)) if isinstance(v, bytes) else v for name, v in encoded.items()}


# ---------------------------------------------------------------------------
# reading a source without binding it
# ---------------------------------------------------------------------------


def _lazy(obj: Any) -> pl.LazyFrame:
    """One source as a lazy frame — a scan for a path, so a filter pushes down."""
    if isinstance(obj, (str, Path)):
        return pl.scan_parquet(obj)
    frame = as_frame(obj)
    if frame is None:
        raise DataError(
            f'cannot slice a source of type {type(obj).__name__} — pass a parquet path or a table '
            f'polars can read (polars, pyarrow, pandas)'
        )
    return frame


def _coordinates(sources: Mapping[str, Any], dim: str, verb: str) -> tuple[list[str], list[Any]]:
    """The sources a slice has to filter, and the ordered coordinates to cut.

    *carrying* is derived rather than declared: a source that carries the slice
    key and is *not* filtered produces a duplicate-coordinate error at bind
    time, so the derivation cannot silently miss one.

    The coordinates are sorted as **values of the column**, so a window is a
    span of those and never of the numbers in them.
    """
    carrying = [name for name, obj in sources.items() if dim in _lazy(obj).collect_schema().names()]
    if not carrying:
        raise DataError(
            f"no source carries a '{dim}' column, so there is nothing to {verb} over. "
            f'EachCoordinate names a column the data has; a span of consecutive coordinates is EachWindow.'
        )
    return carrying, sorted(
        {c for name in carrying for c in _lazy(sources[name]).select(pl.col(dim).unique()).collect()[dim]}
    )
