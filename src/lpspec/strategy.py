"""Solving strategies: one plan per slice, folded.

A plan cannot contain a loop; a *process* may loop over plans, each with its
shape fixed before its own data (docs/design/ceiling.md). So a strategy is
never a language feature and never an engine feature — it is a driver above
:mod:`lpspec.api`, built from the public verbs, and no lane learns a new word.

Every strategy is the same fold: **partition → build → solve → carry →
stitch**. What differs is only how slices are cut and whether they couple.

    scenario / sweep    ``EachCoordinate('scenario')``              independent
    myopic pathway      ``EachCoordinate('period', ordered=True)``  + ``carry``
    rolling horizon     ``EachWindow('snapshot', 48, 24, 't')``     + ``carry``

**A partition is a filter on the sources, not a narrower ``coords``.** Passing
``coords`` alone leaves the parameter rows outside the window in place, and the
containment check refuses them by design. So an axis rewrites the sources and
supplies the matching ``coords`` together.

The caller-facing rules are [docs/api.md](../../docs/api.md#solving-one-model-many-times).
"""

from __future__ import annotations

import io
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from lpspec.api import check
from lpspec.api import solve as _solve
from lpspec.errors import DataError, LpspecError
from lpspec.relational.frames import as_frame

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd
    import xarray as xr

    from lpspec.language.schema import MathSchema

#: Parquet rather than pickle, and not a knob: measured over 1M rows, zstd is
#: 8.3x smaller *and* 3x faster than pickling the frame, and still smaller and
#: faster on incompressible float64.
_COMPRESSION = 'zstd'

#: One cut: the slice key, its sources, and the coords the axis re-indexed.
Cut = tuple[Any, dict[str, Any], dict[str, Any]]


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

    def _slices(self, sources: Mapping[str, Any]) -> list[Cut]:
        carrying, coordinates = _coordinates(sources, self.dim, 'slice')
        out: list[Cut] = []
        for key in coordinates:
            sliced = dict(sources)
            for name in carrying:
                sliced[name] = _lazy(sources[name]).filter(pl.col(self.dim) == key).drop(self.dim)
            out.append((key, sliced, {}))
        return out


@dataclass(frozen=True)
class EachWindow:
    """One slice per window of consecutive coordinates of *dim*.

    Unlike :class:`EachCoordinate` the dimension is re-indexed rather than
    dropped, because a window holds many coordinates and the model has to be
    able to name their order. ``length`` is what the solver sees, ``step`` is
    what you keep, and ``length > step`` is overlap.

    ``length`` and ``step`` count **coordinates, not coordinate values**, so
    the only thing *dim* has to be is orderable — datetimes, strings and gapped
    integers all work.

    **``into`` is structural, and has no default.** The seam row of a windowed
    model is ``where: "t == 0"``, which needs a literal, and "the first
    coordinate of *this* window" is not one in global numbering. Re-indexing is
    the mechanism, the local index is dense ``0..n-1`` by construction, and the
    name belongs to whoever wrote the model.

    For grouping that is not positional — every calendar month, say — precompute
    the group column and use :class:`EachCoordinate`. What this class uniquely
    offers is **overlap**.
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

    def _slices(self, sources: Mapping[str, Any]) -> list[Cut]:
        carrying, coordinates = _coordinates(sources, self.dim, 'window')
        out: list[Cut] = []
        for start in range(0, len(coordinates), self.step):
            window = coordinates[start : start + self.length]
            local = {coordinate: position for position, coordinate in enumerate(window)}
            sliced = dict(sources)
            for name in carrying:
                sliced[name] = (
                    _lazy(sources[name])
                    # the filter is what a scan can push down; the mapping that
                    # follows is over a frame already cut to one window
                    .filter(pl.col(self.dim).is_in(window))
                    .with_columns(pl.col(self.dim).replace_strict(local, return_dtype=pl.Int64).alias(self.into))
                    .drop(self.dim)
                )
            # keyed by the window's first coordinate, not its position: that is
            # what names the window in the caller's own terms
            out.append((window[0], sliced, {self.into: range(len(window))}))
        return out


#: What ``axis=`` accepts. A plain list of ``(key, sources, coords)`` is also
#: taken, so an irregular ladder or a hand-built draw needs no third class.
Axis = EachCoordinate | EachWindow


# ---------------------------------------------------------------------------
# the result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Runs:
    """What a fold returned: frames keyed by slice, never a scalar.

    **There is no aggregate objective**, and that is deliberate. Scenarios are
    a distribution rather than a sum, and summing window objectives
    double-counts whatever the overlap discards. :attr:`objective` is a frame;
    the caller reduces it having said what they mean.

    **Duals are not aggregated**, which is different from not being available:
    :meth:`dual` hands a constraint's shadow prices back slice by slice, keyed
    like everything else. What is refused is doing the reduction *for* the
    caller — a window's shadow price is that window's, and concatenating them
    into a price curve is wrong in a way nothing complains about. Keyed rows
    say whose each price is; a sum would not.

    Everything else here is :class:`~lpspec.relational.result.Result`'s reader
    one dimension wider — same names, same shapes, the slice key prepended. A
    sweep is where a labelled array earns its keep, and building one out of a
    slice-keyed frame by hand is the part worth not writing twice.
    """

    key_name: str
    meta: pl.DataFrame
    #: Per slice, not concatenated. Joining them is the reader's work so a
    #: sweep pays it for the names actually read, and so the concatenated copy
    #: never exists beside the pieces it was built from.
    _primals: dict[str, list[pl.DataFrame]] = field(repr=False, default_factory=dict)
    _duals: dict[str, list[pl.DataFrame]] = field(repr=False, default_factory=dict)
    _no_duals: str | None = field(repr=False, default=None)

    @property
    def objective(self) -> pl.DataFrame:
        """``(key, status, termination_condition, objective)``, in slice order."""
        return self.meta

    @property
    def keys(self) -> list[Any]:
        return self.meta[self.key_name].to_list()

    def primal(self, name: str) -> pl.DataFrame:
        """One variable's values across every slice, the slice key prepended.

        A slice that reached no solution contributes no rows, so this frame can
        be shorter than the sweep — :attr:`objective` is the record of which
        slices those were, and it is one row per slice always.
        """
        if name not in self._primals:
            raise LpspecError(_nothing_to_read('variable', name, self._primals, self.meta))
        return pl.concat(self._primals[name])

    def dual(self, name: str) -> pl.DataFrame:
        """One constraint's shadow prices across every slice, the key prepended.

        :meth:`primal`'s shape and its caveats, plus one of its own: a slice
        whose model had an integer variable has no duals to contribute, so a
        mixed sweep can be shorter here than it is there. :attr:`objective`
        carries what each slice did.

        **Nothing is combined.** Averaging window prices, taking the last, and
        reading one slice alone are all defensible, and this cannot know which
        was meant.
        """
        if name not in self._duals:
            raise LpspecError(self._no_duals or _nothing_to_read('constraint', name, self._duals, self.meta))
        return pl.concat(self._duals[name])

    def to_pandas(self, name: str) -> pd.DataFrame:
        """:meth:`primal` as a tidy :class:`pandas.DataFrame`.

        Needs pandas, which ships with the ``[linopy]`` extra. Column by column
        for the same reason ``Result.to_pandas`` does it: polars' own
        ``to_pandas`` reaches for pyarrow.
        """
        import pandas as pd

        frame = self.primal(name)
        return pd.DataFrame({column: frame[column].to_numpy() for column in frame.columns})

    def to_dataarray(self, name: str) -> xr.DataArray:
        """:meth:`primal` as a :class:`xarray.DataArray`, the slice key a dimension.

        ``(scenario, snapshot, generator)`` is what a sweep is *for* — ``.sel``
        a scenario, take a quantile across them, plot the spread. A slice that
        reached no solution has no rows and so comes back NaN, which is the
        same answer a masked coordinate gets from ``Result``.
        """
        frame = self.to_pandas(name)
        dims = [column for column in frame.columns if column != 'value']
        return frame.set_index(dims).to_xarray()['value'].rename(name)

    def to_dataset(self, *names: str) -> xr.Dataset:
        """Kept variables as one :class:`xarray.Dataset`; all of them by default.

        Costs what it says, and more than ``Result``'s does — each variable
        arrives dense over its own dims *and* over every slice. Name the few you
        need, or use :meth:`to_parquet`.
        """
        wanted = names or tuple(sorted(self._primals))
        if not wanted:
            raise LpspecError(_nothing_to_read('variable', 'anything', self._primals, self.meta))
        first, *rest = wanted
        dataset = self.to_dataarray(first).to_dataset(name=first)
        for name in rest:
            dataset[name] = self.to_dataarray(name)
        return dataset

    def to_parquet(self, directory: str | Path = '.') -> dict[str, Path]:
        """One parquet file per variable the sweep holds, ``(key, dims…, value)``.

        Returns name → path, in :meth:`primal`'s order, so the same sweep
        writes the same bytes. This is a *copy out* of frames already held and
        bounds nothing; spilling per slice, which would, is #477.
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


def _nothing_to_read(kind: str, name: str, held: dict[str, pl.DataFrame], meta: pl.DataFrame) -> str:
    """Why *name* has no frame.

    A sweep keeps everything every slice produced, so there is only one way to
    arrive here with a name the model declares: no slice produced it. A name
    the model does not declare arrives here too, and the two are told apart by
    what the sweep did hold.
    """
    if held:
        listed = ', '.join(repr(k) for k in sorted(held))
        conditions = ', '.join(sorted(set(meta['termination_condition'].to_list())))
        return (
            f'no {kind} {name!r} in this sweep — it holds {listed}. '
            f'If the model declares it, no slice produced one: all {meta.height} terminated {conditions}.'
        )
    conditions = ', '.join(sorted(set(meta['termination_condition'].to_list())))
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

    ``carry`` maps a **parameter** to ``(variable, index)`` — that variable's
    values in slice *i* become the parameter's in slice *i+1*. **The two
    declarations say what is copied**: whichever dimension the variable has and
    the parameter does not is the one the carry collapses, and ``index`` names a
    coordinate of it. Every other dimension rides along, so a myopic pathway
    hands a whole capacity vector forward (``('capacity', None)``, nothing
    dropped) while a rolling horizon hands one row of a time series
    (``('soc', 23)``, dropping ``t``). The index is explicit because with
    ``EachWindow(48, 24)`` the state to carry sits at 23, not 47.

    It is a **copy and never arithmetic**: accumulation (``existing += built``)
    is a derived variable in the YAML, where the oracle can see it.

    **Everything a slice produced is kept** — every variable's primals, every
    constraint's duals — and read back through :meth:`Runs.primal` and
    :meth:`Runs.dual`. It is still a fold: each slice's *model* is released as
    the loop goes, so build peak stays at one slice however many there are.
    What accumulates is the answer, which is what the caller asked for.

    Narrowing that is a later addition and an easy one — a keyword naming what
    to hold on to. It is absent because it would have to be *two* keywords, a
    constraint being allowed to carry a variable's name, and neither has been
    wanted yet.

    ``key_name`` names the column holding the slice key — the same word in
    :attr:`Runs.objective` and in every frame :meth:`Runs.primal` returns, so
    the two still join. The class axes derive it (``EachCoordinate('scenario')``
    keys on ``scenario``; a window keys on ``<dim>_start``, since ``dim`` itself
    was dropped), but **a hand-built list of cuts has to be told**, for the
    reason :attr:`EachWindow.into` has no default.

    ``executor`` is any :class:`concurrent.futures.Executor`; ``None`` is
    sequential. A ``carry`` makes slices sequential by definition, so the two
    are refused together rather than one silently winning.

    ``workers_share_fs`` says whether the executor's workers can read the caller's
    paths, and it only affects path sources — a frame crosses as parquet either
    way. Left unset it is **inferred from the pool**: a stdlib
    :class:`~concurrent.futures.ProcessPoolExecutor` runs on this machine and
    reads this machine's files, so its paths stay paths; anything else is a
    transport this package did not ship and is assumed remote, so its paths
    travel as bytes. Say it outright when the inference is wrong — a cluster
    that mounts the same filesystem is ``workers_share_fs=True``.

    **A process pool must not use the ``fork`` start method.** polars' thread
    pool does not survive a fork, and a forked worker hangs rather than failing.
    This cannot be enforced here — a remote executor has no start method to
    inspect — so pass the context, and give the entry point the ``__main__``
    guard that ``spawn`` requires.

    .. code-block:: python

        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(4, mp_context=ctx) as pool:
            runs = lps.solve_over(model, sources, axis, executor=pool)
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
    # Everything above and below this line is answerable from the declarations
    # and the keywords, so it is answered before a single source is read — a
    # mistyped carry should not cost a scan of every parquet file first. The
    # schema then rides down to the slices already parsed, rather than each of
    # them (and each worker) reading the same YAML again.
    schema = check(model)
    plan: dict[str, tuple[str, str | None, int | None]] = _carry_plan(schema, carry) if carry else {}
    key_name = _key_column(axis, key_name, schema)

    cuts = axis._slices(sources) if isinstance(axis, (EachCoordinate, EachWindow)) else list(axis)
    if not cuts:
        raise DataError('the axis produced no slices')

    # the caller's own coords are merged under the axis's, which owns the dim it
    # re-indexed; popped once rather than per slice, so the loop stays pure
    caller_coords = dict(build_kwargs.pop('coords', None) or {})
    call = {'solver_name': solver_name, 'solver_options': dict(solver_options or {}) or None, **build_kwargs}
    # the two stdlib pools are the ones whose deployment is knowable: both run
    # here, so both read the paths here. An executor this package did not ship
    # is a transport it cannot ask, so it is assumed remote until told otherwise
    shared = workers_share_fs if workers_share_fs is not None else isinstance(executor, ProcessPoolExecutor)
    memo: dict[str, tuple[Any, Any]] = {}
    rows: list[dict[str, Any]] = []
    primals: dict[str, list[pl.DataFrame]] = {name: [] for name in schema.variables}
    shadow: dict[str, list[pl.DataFrame]] = {name: [] for name in schema.constraints}
    no_duals: str | None = None

    def arguments(sliced: dict[str, Any], coords: dict[str, Any], crosses: bool) -> tuple[Any, ...]:
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
        rows.append({key_name: key, **meta})
        for into, produced in ((primals, frames), (shadow, priced)):
            for name, frame in produced.items():
                into[name].append(frame.select(pl.lit(key).alias(key_name), pl.all()))

    if executor is None:
        state: dict[str, Any] = {}
        for key, sliced, coords in cuts:
            meta, frames, priced, reason = _run_slice(*arguments({**sliced, **state}, coords, False))
            no_duals = no_duals or reason
            absorb(key, meta, frames, priced)
            if plan:
                state = _carried(plan, frames, key)
    else:
        # A thread pool runs in this process, so encoding would be a parquet
        # round trip for a boundary that is not there — 31% of a thread-pool
        # sweep, measured. Every other executor is assumed to cross, because
        # none of them can be asked.
        crosses = not isinstance(executor, ThreadPoolExecutor)
        futures = [executor.submit(_run_slice, *arguments(sliced, coords, crosses)) for _key, sliced, coords in cuts]
        # in slice order, never completion order — a sweep must not reorder itself
        for (key, _sliced, _coords), future in zip(cuts, futures, strict=True):
            meta, returned, priced, reason = future.result()
            no_duals = no_duals or reason
            absorb(key, meta, _decode(returned) if crosses else returned, _decode(priced) if crosses else priced)

    return Runs(
        key_name=key_name,
        meta=pl.DataFrame(rows),
        _primals={name: frames for name, frames in primals.items() if frames},
        _duals={name: frames for name, frames in shadow.items() if frames},
        _no_duals=no_duals,
    )


def _run_slice(
    model: Any,
    encoded: dict[str, Any],
    coords: dict[str, Any] | None,
    encode_out: bool,
    call: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None]:
    """One slice, start to finish, over plain data.

    Module-level and closure-free on purpose: a remote executor has to pickle
    what it is handed, and a bound method or a lambda over the axis object
    cannot cross. Everything in the signature is a path, a frame, a string or a
    number.
    """
    with _solve(model, _decode(encoded), coords=coords, **call) as result:
        meta = {
            'status': result.status,
            'termination_condition': result.termination_condition,
            'objective': result.objective if result.has_primal else float('nan'),
        }
        frames = {name: result.primal(name) for name in model.variables} if result.has_primal else {}
        priced, no_duals = _prices(result, model)
        if not encode_out:
            return meta, frames, priced, no_duals
        return meta, _encode(frames, {}), _encode(priced, {}), no_duals


def _prices(result: Any, model: Any) -> tuple[dict[str, Any], str | None]:
    """Every constraint's duals, or the one reason there are none.

    An integer variable leaves duals undefined, and one such slice must not
    fail a whole sweep — so the frames are simply absent, as they are for a
    slice that did not solve. But *absent* is where a sweep used to lose what a
    single solve says plainly, and `Result.dual` already writes that sentence
    (it names the integer variable). It is caught and carried rather than
    rewritten: a sweep of one model has one answer, so the first is the answer.
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
    schema: MathSchema,
) -> str:
    """What to call the column holding the slice key.

    The two class axes know: a coordinate sweep keys on the dimension it cut,
    and a window keys on where it *started* — never on ``dim`` itself, which the
    slice dropped and re-indexed to ``into``. A column called ``snapshot``
    holding window starts joins against real snapshot-indexed data and keeps a
    fraction of it, silently.

    A hand-built list knows neither, so it has to be told, for the same reason
    :attr:`EachWindow.into` has no default: naming somebody else's axis is
    guessing at their decision, and ``'slice'`` would be the library's word for
    the caller's draw, ladder or pathway.
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
    schema: MathSchema,
    carry: Mapping[str, tuple[str, int | None]],
) -> dict[str, tuple[str, str | None, int | None]]:
    """Resolve each carry against the schema: what is dropped, and what rides.

    The declaration says which dimensions the two sides have, so it says what a
    carry *is*: the variable's dims minus the parameter's is the one dimension
    the carry collapses, and ``index`` names a coordinate of it. Everything else
    passes through, which is what lets a myopic pathway hand a whole capacity
    vector forward rather than one number at a time.

    **Nothing here reads data**, which is why it runs before the axis cuts one.
    Every question is answered by the YAML and the keywords, so a carry that
    cannot work costs a parse rather than a scan of every source.
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


def _encode(
    sources: Mapping[str, Any], memo: dict[str, tuple[Any, Any]], *, workers_share_fs: bool = False
) -> dict[str, Any]:
    """Sources in the shape a worker can be handed.

    A path the workers can reach stays a path and costs nothing. A path they
    cannot reach travels as **its own bytes, untouched** — decoding and
    re-encoding a parquet file produces byte-identical output for 79x the CPU,
    so the caller must never be pushed into doing it by hand. Anything held in
    memory is written to parquet, which beats pickling the frame on both size
    and time.

    *memo* keeps a source that no slice rewrote — the static tables, which is
    most of them — from being encoded once per slice. ``bytes`` is what
    :func:`_decode` reads back, and it cannot be confused with a path.

    Not called on the sequential or thread path at all: nothing crosses a
    boundary there, so encoding would be a round trip paid for nothing.
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
    """
    carrying = [name for name, obj in sources.items() if dim in _lazy(obj).collect_schema().names()]
    if not carrying:
        raise DataError(
            f"no source carries a '{dim}' column, so there is nothing to {verb} over. "
            f'EachCoordinate names a column the data has; a span of consecutive coordinates is EachWindow.'
        )
    # a window is a span of *these*, never of the numbers in them
    return carrying, sorted(
        {c for name in carrying for c in _lazy(sources[name]).select(pl.col(dim).unique()).collect()[dim]}
    )
