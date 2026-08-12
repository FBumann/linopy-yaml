"""Solving strategies: one plan per slice, folded.

A plan cannot contain a loop; a *process* may loop over plans, each with its
shape fixed before its own data (docs/design/ceiling.md). So a strategy is
never a language feature and never an engine feature — it is a driver above
:mod:`lpspec.api`, built from the public verbs, and no lane learns a new word.

Every strategy is the same fold: **partition → bind → solve → carry →
stitch**. Only how slices are cut and whether they couple differs. (The
*stage* stitches; a reader asks for its result by the index it wants, which is
:meth:`Runs.primal`'s ``original_index``.)

*Bind* rather than *build*, because a serial fold builds once and rebinds each
slice onto that model (:func:`_one_model`) — every slice being the same math
over different numbers, which is what ``rebind`` is for. A fold under a process
pool builds per slice, a built model being the one thing that cannot cross.

    scenario / sweep    ``EachCoordinate('scenario')``              independent
    myopic pathway      ``EachCoordinate('period', ordered=True)``  + ``carry``
    rolling horizon     ``EachWindow('snapshot', 48, 24, 't')``     + ``carry``

**A partition is a filter on the sources, not a narrower ``coords``.** Passing
``coords`` alone leaves the parameter rows outside the window in place, and the
containment check refuses them by design, so an axis rewrites the sources and
supplies the matching ``coords`` together.

The caller-facing rules are [docs/api.md](../../docs/api.md#solving-one-model-many-times).
"""

from __future__ import annotations

import io
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from lpspec.api import build, check
from lpspec.api import solve as _solve
from lpspec.errors import DataError, LpspecError
from lpspec.relational.frames import as_frame
from lpspec.relational.result import tidy_to_dataarray, tidy_to_dataset, tidy_to_pandas

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    import pandas as pd
    import xarray as xr

    from lpspec.language.model import Model

#: Parquet rather than pickle, and not a knob: zstd measured smaller *and*
#: faster than pickling the frame, on compressible and incompressible data
#: alike (#459).
_COMPRESSION = 'zstd'

#: One cut: the slice key, its sources, and the coords the axis re-indexed.
Cut = tuple[Any, dict[str, Any], dict[str, Any]]


@dataclass(frozen=True)
class _OriginalIndex:
    """The way back from a windowed sweep's slices to the dimension it sliced.

    ``owned`` is ``(key, local, dim)`` for the coordinates each window is
    *responsible* for — its first ``step``, the rest being lookahead the next
    window recomputes. An inner join against it is the whole operation: it
    restores the original coordinate, and because a coordinate may appear only
    once under its own index, the lookahead rows have nowhere to go.

    **One-way, and that is why it holds only the owned coordinates.** It is the
    return path, not a record of the slicing: the lookahead rows are not in it,
    so a sliced frame cannot be rebuilt from it. Slicing is
    :meth:`EachWindow._slices`' business and stays there.
    """

    local: str
    dim: str
    owned: pl.DataFrame


# ---------------------------------------------------------------------------
# axes — how slices are cut
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EachCoordinate:
    """One slice per coordinate of *dim* — a column the sources carry.

    Scenarios, draws, investment periods. Sources carrying *dim* are filtered
    to one coordinate and the column dropped, so **the model never mentions
    it**; every other source passes through untouched.

    ``ordered`` says the coordinates have a meaningful sequence, which is what
    a ``carry`` needs — scenarios have no "next", investment periods do.
    """

    dim: str
    ordered: bool = False

    def _slices(self, sources: Mapping[str, Any], key_name: str) -> tuple[list[Cut], _OriginalIndex | None]:
        """One cut per coordinate, keyed by it. Sources without *dim* pass through.

        No :class:`_OriginalIndex`: nothing was re-indexed, so a slice's frames
        already carry the coordinates they were solved over.
        """
        del key_name
        carrying, coordinates = _coordinates(sources, self.dim, 'slice')
        out: list[Cut] = []
        for key in coordinates:
            cut = {name: _lazy(sources[name]).filter(pl.col(self.dim) == key).drop(self.dim) for name in carrying}
            out.append((key, {**sources, **cut}, {}))
        return out, None


@dataclass(frozen=True)
class EachWindow:
    """One slice per window of consecutive coordinates of *dim*.

    The dimension is re-indexed rather than dropped, a window holding many
    coordinates whose order the model has to be able to name. ``length`` is
    what the solver sees, ``step`` is what you keep, and ``length > step`` is
    overlap — the one thing this class uniquely offers. Non-positional grouping
    (every calendar month) is a precomputed column plus
    :class:`EachCoordinate`.

    Both count **coordinates, not coordinate values**, so *dim* need only be
    orderable — datetimes, strings and gapped integers all work.

    **``into`` is structural, and has no default.** The seam row of a windowed
    model is ``where: "t == 0"``, which needs a literal, and "the first
    coordinate of *this* window" is not one in global numbering. The local
    index is dense ``0..n-1`` by construction, and its name belongs to whoever
    wrote the model.
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
            raise ValueError('into must name the local index the model declares — it has no default')
        if self.into == self.dim:
            raise ValueError(f'into={self.into!r} must differ from dim — the local index replaces the global one')

    def _slices(self, sources: Mapping[str, Any], key_name: str) -> tuple[list[Cut], _OriginalIndex]:
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
        out: list[Cut] = []
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
            out.append((window[0], {**sources, **cut}, {self.into: range(len(window))}))
            owned.extend(
                {key_name: window[0], self.into: position, self.dim: coordinate}
                for position, coordinate in enumerate(window[: self.step])
            )
        return out, _OriginalIndex(self.into, self.dim, pl.DataFrame(owned))


#: What ``axis=`` accepts. A plain list of ``(key, sources, coords)`` is also
#: taken, so an irregular ladder or a hand-built draw needs no third class.
Axis = EachCoordinate | EachWindow


# ---------------------------------------------------------------------------
# the result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Runs:
    """What a fold returned: frames keyed by slice, never a scalar.

    :class:`~lpspec.relational.result.Result`'s readers one dimension wider —
    same names, same shapes, the slice key prepended.

    **Keyed is the default; the original index is asked for.** Reading a
    windowed sweep over the dimension it re-indexed is the nicer answer, but a
    *lossy* one — a coordinate may appear only once under its own index, so the
    lookahead rows every overlapping window solved have nowhere to go. A reader
    that discarded them by default would be throwing away computed answers
    silently, and it would key differently from :attr:`objective`, which is one
    row per slice always, so the two would stop joining.
    ``original_index=True`` is one word at the call site and says which was
    meant.

    **Nothing is aggregated, and that is the decision.** Scenarios are a
    distribution rather than a sum, summing window objectives double-counts
    whatever the overlap discards, and a window's shadow price is that
    window's — concatenating them into a price curve is wrong in a way nothing
    complains about. Keyed rows say whose each number is; a reduction would
    have to know what the caller meant.
    """

    key_name: str
    meta: pl.DataFrame
    #: Per slice, not concatenated. Joining them is the reader's work so a
    #: sweep pays it for the names actually read, and so the concatenated copy
    #: never exists beside the pieces it was built from.
    _primals: dict[str, list[pl.DataFrame]] = field(repr=False, default_factory=dict)
    _duals: dict[str, list[pl.DataFrame]] = field(repr=False, default_factory=dict)
    _no_duals: str | None = field(repr=False, default=None)
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

        A slice that reached no solution contributes no rows, so this frame can
        be shorter than the sweep — :attr:`objective` is the record of which
        slices those were, and it is one row per slice always.

        ``original_index=True`` asks for the same values over the dimension the
        axis sliced instead — see :meth:`_reindexed`.
        """
        return self._reindexed(self._read(self._primals, 'variable', name), original_index=original_index)

    def dual(self, name: str, *, original_index: bool = False) -> pl.DataFrame:
        """One constraint's shadow prices across every slice, the key prepended.

        :meth:`primal`'s shape, caveats and ``original_index``, plus one of its
        own: a slice whose model had an integer variable has no duals to
        contribute, so a mixed sweep can be shorter here than there.

        **Nothing is combined** either way. Averaging window prices and taking
        the last are both defensible and neither is done; over the original
        index each coordinate carries the price of *the window that owns it*,
        which is one window's answer rather than a blend of several.
        """
        return self._reindexed(
            self._read(self._duals, 'constraint', name, self._no_duals), original_index=original_index
        )

    def _reindexed(self, frame: pl.DataFrame, *, original_index: bool) -> pl.DataFrame:
        """*frame* over the dimension the axis sliced, rather than over its slices.

        **One operation, not two.** Restoring the original coordinate is the
        whole of it; dropping the overlap falls out, because a coordinate may
        appear only once under its own index and the lookahead rows have
        nowhere to go. That is why a single window with no overlap at all still
        needs this — the re-index fires regardless of how many pieces there
        are, which is what the word *stitch* got wrong.

        Each window contributes the ``step`` coordinates it owns, the final one
        included, which can hold no more and so keeps all of it.

        **Every axis answers it**, so code handed one need not ask which it
        got. :class:`EachCoordinate` and a hand-built axis re-indexed nothing —
        their key column already *is* a coordinate of the answer — so the frame
        comes back unchanged, which is a satisfied request rather than an
        ignored one.

        A flag on the readers rather than a reader of its own, because what has
        to be undone depends on the *axis* and not on which quantity was read:
        duals get it for free, and a name that is both a variable and a
        constraint (which the language permits) never has to be dispatched.

        It takes a *frame* rather than a name, so making it public would be a
        rename plus a precondition — the frame has to carry the key column and
        the local index — for a caller with a quantity the sweep does not hold.
        Nothing here forecloses that; it is private because the readers are
        where a sweep's frames come from.
        """
        if not original_index or self._original is None:
            return frame
        keys = [self.key_name, self._original.local]
        restored = frame.join(self._original.owned, on=keys, how='inner').drop(keys)
        rest = [column for column in restored.columns if column not in (self._original.dim, 'value')]
        return restored.select(self._original.dim, *rest, 'value').sort(self._original.dim, *rest)

    def to_pandas(self, name: str, *, original_index: bool = False) -> pd.DataFrame:
        """:meth:`primal` as a tidy :class:`pandas.DataFrame`.

        **The name is resolved before pandas is imported.** A sweep that never
        held *name* should say so on any install, where importing first answers
        a question about the environment when the caller asked one about their
        model.
        """
        return tidy_to_pandas(self.primal(name, original_index=original_index))

    def to_dataarray(self, name: str, *, original_index: bool = False) -> xr.DataArray:
        """:meth:`primal` as a :class:`xarray.DataArray`, the slice key a dimension.

        The extra dimension is named by the axis, not by this class — a
        scenario sweep gives ``(scenario, …)`` and a window ``(<dim>_start, …)``
        — which is what a sweep is *for*: ``.sel`` one slice, take a quantile
        across them, plot the spread. A slice that reached no solution has no
        rows and comes back NaN, the same answer a masked coordinate gets from
        ``Result``.

        ``original_index=True`` gives the array over the dimension the axis
        sliced instead, so a rolling horizon comes back indexed by time.
        """
        return tidy_to_dataarray(self.to_pandas(name, original_index=original_index), name)

    def to_dataset(self, *names: str) -> xr.Dataset:
        """Kept variables as one :class:`xarray.Dataset`; all of them by default.

        Costs more than ``Result``'s does — each variable arrives dense over
        its own dims *and* over every slice. Name the few you need, or use
        :meth:`to_parquet`.

        No ``original_index``: this and :meth:`to_parquet` export what the
        sweep *holds*, and the original index is lossy — the lookahead rows
        every overlapping window computed cannot survive it. A bulk export is
        the wrong place to lose them.
        """
        wanted = names or tuple(sorted(self._primals))
        if not wanted:
            raise LpspecError(_nothing_to_read('variable', 'anything', self._primals, self.meta))
        return tidy_to_dataset(wanted, self.to_dataarray)

    def to_parquet(self, directory: str | Path = '.') -> dict[str, Path]:
        """One parquet file per variable the sweep holds, ``(key, dims…, value)``.

        Returns name → path, in :meth:`primal`'s order, so the same sweep
        writes the same bytes. This is a *copy out* of frames already held and
        bounds nothing — what a sweep holds is #610.
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
            f'If the model declares it, no slice produced one: all {meta.height} terminated {conditions}.'
        )
    return (
        f'this sweep holds no {kind} frames at all — every one of its {meta.height} slices '
        f'terminated {conditions}. The fold ran; the models did not solve. '
        f'runs.objective carries the status of each slice.'
    )


def solve_over(
    model: Any,
    sources: Mapping[str, Any],
    axis: Axis | Sequence[Cut],
    *,
    carry: Mapping[str, tuple[str, int | None]] | None = None,
    key_name: str | None = None,
    executor: Any = None,
    workers_share_fs: bool | None = None,
    solver_options: Mapping[str, Any] | None = None,
    solver_name: str = 'highs',
    **build_kwargs: Any,
) -> Runs:
    """Solve *model* once per slice of *axis* and fold the answers together.

    A serial fold builds one model and rebinds each slice onto it; a pooled one
    builds per slice. Either way one slice's model is alive at a time, so build
    peak does not grow with the sweep, and what accumulates is the answers. A
    carry hands each slice's result to the next, and the last slice carries
    nothing. What a carry copies and how a key column is named are the table in
    [docs/api.md](../../docs/api.md#solving-one-model-many-times).

    Every declaration is checked before a source is read: a mistyped carry, a
    key column that collides, an axis a carry cannot run on all cost a parse
    rather than a scan of every parquet file.

    **A process pool must not use the ``fork`` start method.** polars' thread
    pool does not survive a fork, and a forked worker hangs rather than
    failing. Pass a ``spawn`` context, and give the entry point the ``__main__``
    guard it requires:

    .. code-block:: python

        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(4, mp_context=ctx) as pool:
            runs = lps.solve_over(model, sources, axis, executor=pool)

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
    schema = check(model)
    plan: dict[str, tuple[str, str | None, int | None]] = _carry_plan(schema, carry) if carry else {}
    key_name = _key_column(axis, key_name, schema)

    if isinstance(axis, (EachCoordinate, EachWindow)):
        cuts, original = axis._slices(sources, key_name)
    else:
        cuts, original = list(axis), None
    if not cuts:
        raise DataError('the axis produced no slices')

    caller_coords = dict(build_kwargs.pop('coords', None) or {})
    call = {'solver_name': solver_name, 'solver_options': dict(solver_options or {}) or None, **build_kwargs}
    shared = _shares_filesystem(executor, workers_share_fs)
    memo: dict[str, tuple[Any, Any]] = {}
    rows: list[dict[str, Any]] = []
    primals: dict[str, list[pl.DataFrame]] = {name: [] for name in schema.variables}
    shadow: dict[str, list[pl.DataFrame]] = {name: [] for name in schema.constraints}
    no_duals: str | None = None

    def arguments(sliced: dict[str, Any], coords: dict[str, Any], crosses: bool) -> tuple[Any, ...]:
        """One slice's ``_run_slice`` arguments, as plain data.

        The caller's own ``coords`` sit *under* the axis's, which owns the dim
        it re-indexed. They are popped once rather than per slice, so the loop
        stays a pure function of the cut.
        """
        return (
            schema,
            _encode(sliced, memo, workers_share_fs=shared) if crosses else dict(sliced),
            {**caller_coords, **coords} or None,
            crosses,
            call,
        )

    def absorb(
        key: Any, meta: dict[str, Any], frames: dict[str, pl.DataFrame], priced: dict[str, pl.DataFrame]
    ) -> None:
        """Fold one slice's answer in, its key prepended to every frame.

        Called in **slice order, never completion order** — the concurrent
        branch below walks the futures in the order it submitted them, so a
        sweep cannot reorder itself under a pool.
        """
        rows.append({key_name: key, **meta})
        for into, produced in ((primals, frames), (shadow, priced)):
            for name, frame in produced.items():
                into[name].append(frame.select(pl.lit(key).alias(key_name), pl.all()))

    if executor is None:
        state: dict[str, Any] = {}
        last = len(cuts) - 1
        with _one_model(schema, call) as bound:
            for position, (key, sliced, coords) in enumerate(cuts):
                result = bound.on({**sliced, **state}, {**caller_coords, **coords})
                meta, frames, priced, reason = _answers(result, schema)
                no_duals = no_duals or reason
                absorb(key, meta, frames, priced)
                if plan and position < last:
                    state = _carried(plan, frames, key)
    else:
        crosses = _crosses_a_process(executor)
        futures = [executor.submit(_run_slice, *arguments(sliced, coords, crosses)) for _key, sliced, coords in cuts]
        for (key, _sliced, _coords), future in zip(cuts, futures, strict=True):
            meta, returned, priced, reason = future.result()
            no_duals = no_duals or reason
            absorb(key, meta, _decode(returned), _decode(priced))

    return Runs(
        key_name=key_name,
        meta=pl.DataFrame(rows),
        _primals={name: frames for name, frames in primals.items() if frames},
        _duals={name: frames for name, frames in shadow.items() if frames},
        _no_duals=no_duals,
        _original=original,
    )


@contextmanager
def _one_model(schema: Model, call: dict[str, Any]) -> Iterator[_Rebound]:
    """One built model for the whole fold, rebound per slice.

    Every slice of a sweep is the same math over different numbers, which is
    what :meth:`~lpspec.api.BoundModel.rebind` is: the YAML is parsed once, the
    plan lowered once, and a slice whose structure matches the last one keeps
    the loaded solver and re-solves from its basis. The peak is unchanged —
    a rebuild releases the previous model before it starts — so this is still
    a fold that holds one slice's model however many there are.

    Serial only. The pooled branch hands plain data to :func:`_run_slice` in
    another process, and a built model is the one thing that cannot cross.

    Yields:
        The holder to solve each slice on; its model is closed on the way out.
    """
    holder = _Rebound(schema, call)
    try:
        yield holder
    finally:
        holder.close()


class _Rebound:
    """The fold's model, and which slice it currently holds."""

    def __init__(self, schema: Model, call: dict[str, Any]) -> None:
        self._schema = schema
        self._solve_with = {'solver_name': call['solver_name'], 'solver_options': call['solver_options']}
        self._build_with = {k: v for k, v in call.items() if k not in self._solve_with}
        self._bound: Any = None

    def on(self, sources: Mapping[str, Any], coords: Mapping[str, Any]) -> Any:
        """Solve this slice — the first builds, the rest rebind."""
        if self._bound is None:
            self._bound = build(self._schema, sources, coords=dict(coords) or None, **self._build_with)
        else:
            self._bound.rebind(sources, coords=coords)
        return self._bound.solve(**self._solve_with)

    def close(self) -> None:
        if self._bound is not None:
            self._bound.close()


def _answers(result: Any, model: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None]:
    """One slice's answer, read out of *result* before anything rebinds it.

    Read here rather than held, which is the discipline a rebound model asks
    of every driver and the fold already kept: what a sweep accumulates is
    frames, never results.
    """
    meta = {
        'status': result.status,
        'termination_condition': result.termination_condition,
        'objective': result.objective if result.has_primal else float('nan'),
    }
    frames = {name: result.primal(name) for name in model.variables} if result.has_primal else {}
    priced, no_duals = _prices(result, model)
    return meta, frames, priced, no_duals


def _run_slice(
    model: Any,
    encoded: dict[str, Any],
    coords: dict[str, Any] | None,
    encode_out: bool,
    call: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None]:
    """One slice, start to finish, over plain data — the *pooled* branch.

    Module-level and closure-free on purpose: a remote executor has to pickle
    what it is handed, and a bound method or a lambda over the axis object
    cannot cross. Everything in the signature is a path, a frame, a string or a
    number — which is also why this one builds per slice where the serial
    branch rebinds.
    """
    with _solve(model, _decode(encoded), coords=coords, **call) as result:
        meta, frames, priced, no_duals = _answers(result, model)
        if not encode_out:
            return meta, frames, priced, no_duals
        return meta, _encode(frames, {}), _encode(priced, {}), no_duals


def _prices(result: Any, model: Any) -> tuple[dict[str, Any], str | None]:
    """Every constraint's duals, or the one reason there are none.

    An integer variable leaves duals undefined, and one such slice must not
    fail a whole sweep, so the frames are simply absent as they are for a slice
    that did not solve. ``Result.dual`` already writes the sentence saying why
    and names the variable, so it is caught and carried rather than rewritten —
    a sweep of one model has one answer, so the first is the answer.
    """
    if not result.has_primal:
        return {}, None
    priced: dict[str, Any] = {}
    for name in model.constraints:
        try:
            priced[name] = result.dual(name)
        except LpspecError as exc:
            return {}, str(exc)
    return priced, None


def _key_column(
    axis: Axis | Sequence[Cut],
    key_name: str | None,
    schema: Model,
) -> str:
    """What to call the column holding the slice key.

    The class axes know: a coordinate sweep keys on the dimension it cut, a
    window on where it *started* — never on ``dim`` itself, which the slice
    dropped and re-indexed. A column called ``snapshot`` holding window starts
    would join against real snapshot-indexed data and keep a fraction of it,
    silently.

    A hand-built list knows neither and has to be told, for the reason
    :attr:`EachWindow.into` has no default: ``'slice'`` would be this library's
    word for the caller's draw, ladder or pathway.
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
    clashing = sorted(
        name for name in schema.variables if key_name in getattr(schema.variables.get(name), 'foreach', ())
    )
    if clashing:
        raise LpspecError(
            f'key_name={key_name!r} is already a dimension of {clashing}, so the slice key would collide '
            f'with a column those frames already carry. Name it something the model does not use.'
        )
    return key_name


def _carry_plan(
    schema: Model,
    carry: Mapping[str, tuple[str, int | None]],
) -> dict[str, tuple[str, str | None, int | None]]:
    """Resolve each carry against the schema: what is dropped, and what rides.

    The variable's dims minus the parameter's is the one dimension the carry
    collapses, and ``index`` names a coordinate of it; everything else passes
    through, which is what lets a myopic pathway hand a whole capacity vector
    forward rather than one number at a time.

    **Nothing here reads data**, which is why it runs before the axis cuts any.
    """
    plan: dict[str, tuple[str, str | None, int | None]] = {}
    for parameter, (variable, index) in carry.items():
        if parameter not in schema.parameters:
            raise LpspecError(f'carry writes parameter {parameter!r}, which the model does not declare')
        if variable not in schema.variables:
            raise LpspecError(f'carry reads variable {variable!r}, which the model does not declare')
        over = list(schema.parameters[parameter].dims)
        source = list(schema.variables[variable].foreach)
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
        plan[parameter] = (variable, dropped[0] if dropped else None, index)
    return plan


def _carried(
    plan: Mapping[str, tuple[str, str | None, int | None]],
    frames: Mapping[str, pl.DataFrame],
    key: Any,
) -> dict[str, Any]:
    """The next slice's carried parameters, read out of this slice's primals.

    ``index`` is a **coordinate** of the dropped dimension, never a row number.
    That is the same integer for the case this started with — ``into`` is dense
    ``0..n-1``, so window position and coordinate coincide — and it is the only
    one of the two that still means something once a second dimension is there.
    """
    state: dict[str, Any] = {}
    for parameter, (variable, dropped, index) in plan.items():
        frame = frames[variable]
        if dropped is None:
            state[parameter] = frame
            continue
        picked = frame.filter(pl.col(dropped) == index).drop(dropped)
        if picked.is_empty():
            coordinates = frame[dropped].unique().sort().to_list()
            raise LpspecError(
                f'carry {parameter!r} <- ({variable!r}, {index}) is out of range: slice {key!r} has no '
                f'{dropped} == {index}. Its coordinates run {coordinates[0]!r}..{coordinates[-1]!r}, and a '
                f'short tail window has fewer than a full one.'
            )
        state[parameter] = picked
    return state


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
    produces byte-identical output for a large multiple of the CPU (#459).
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
