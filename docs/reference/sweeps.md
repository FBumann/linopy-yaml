# Sweeps and rolling horizons

`solve_over` runs the same model once per slice and folds the answers together.
Scenarios, rolling horizons and myopic pathways are all the same fold: a plan
cannot contain a loop, but a process may loop over plans.

```python
import lpspec as lps

runs = lps.solve_over('model.yaml', sources, lps.EachCoordinate('scenario'))
runs.objective  # (scenario, status, termination_condition, objective)
runs.primal('p')  # (scenario, snapshot, generator, value)
```

## The axes

| | |
|---|---|
| `lps.EachCoordinate(dim, ordered=False)` | one slice per coordinate of `dim` — scenarios, draws, investment periods. Sources carrying `dim` are filtered to one coordinate and the column dropped, so the model never mentions it; every other source passes through untouched. `ordered=True` says the coordinates are a sequence, which a `carry` needs |
| `lps.EachWindow(dim, length, step, into)` | one slice per window of consecutive coordinates of `dim`. `length` is what the solver sees, `step` is what the window keeps, and `length > step` is overlap. The dimension is re-indexed rather than dropped, into a dense `0..n-1` column the model addresses by the name `into` gives it |
| a sequence of `(key, sources)` pairs | a hand-built axis: each cut says what the *whole* model binds for that slice, and the call must pass `key_name=` |

```python
runs = lps.solve_over(
    'window.yaml',
    sources,
    lps.EachWindow('snapshot', length=48, step=24, into='t'),
    carry={'soc_initial': ('soc', 23)},
)
runs.primal('soc')  # (snapshot_start, t, value) — the window, and the index inside it
```

**A window spans coordinates, not values.** `length=48` is forty-eight
snapshots however they are numbered, so the dimension only has to be
*orderable* — datetimes, strings and gapped integers all work. `into` is the
dense local index, which is what keeps a seam's `where: "t == 0"` matching, and
it has no default because the name belongs to the model.

*"Each calendar month"* has unequal groups, so it is a precomputed column plus
`EachCoordinate`. What `EachWindow` uniquely offers is **overlap**.

## Reading a sweep

**`Runs` reads like `Result`, one dimension wider** — `primal`, `dual`,
`expression`, `to_pandas`, `to_dataarray`, `to_dataset`, `to_parquet`, under
the same names and with the slice key prepended.

That extra dimension is **named by you, not by the library**:
`EachCoordinate('scenario')` keys on `scenario`, a window on `<dim>_start`, and
`key_name=` overrides either. So `runs.to_dataarray('p')` on a scenario sweep
is `(scenario, snapshot, generator)`, which is what a sweep is *for*: `.sel` one
scenario, take a spread across them, plot the band.

**`original_index=` asks for the answer over real coordinates**, and it is a
keyword on the readers rather than a reader of its own:

```python
runs.primal('soc')  # (snapshot_start, t, value) — keyed by slice
runs.primal('soc', original_index=True)  # (snapshot, value) — the answer
runs.dual('balance', original_index=True)  # the same, for a price
runs.expression('spend', original_index=True)  # the model's own quantity, over real coordinates
```

For `EachWindow` that is the answer a rolling horizon is *for*: the overlap
dropped and the global coordinate restored, each window contributing the `step`
coordinates it owns — the final one included, which can hold no more and so
keeps all of it. For `EachCoordinate` and a hand-built axis nothing was
re-indexed, so the frame comes back unchanged.

**Keyed is the default**, because stitching is lossy: it keeps only what each
window owns and drops the lookahead rows the sweep solved. For the same reason
`to_dataset` and `to_parquet` have no `original_index` — a bulk export of what
the sweep holds is the wrong place to lose rows.

Per slice is a partition of a frame you already have, so there is no reader for
it: `runs.primal('p').partition_by(runs.key_name, as_dict=True)`.

| Rule | |
|---|---|
| **everything a slice produced is kept** | every variable's primals and every constraint's duals, read back through `runs.primal(name)` and `runs.dual(name)`. Each slice's *model* is released as the loop goes, so build peak stays at one slice however many there are; what accumulates is the answer |
| **duals are keyed, never combined** | `runs.dual(name)` is `runs.primal(name)`'s shape. Averaging window prices, taking the last, and reading one slice alone are all defensible, so the reduction is yours. A slice whose model had an integer variable contributes none, and `runs.objective` says which |
| **expressions are evaluated per slice** | every declared `expressions:` name, evaluated at each slice's solution, back through `runs.expression(name)`. Over `original_index=True` only the rows each window owns survive, so summing the stitched frame cannot double-count the lookahead — and a quantity *reduced over* the sliced dimension has no way back and is refused there, naming the per-slice read as the alternative |
| **no aggregate objective** | `objective` is a frame keyed by slice. Scenarios are a distribution, not a sum; summing window objectives double-counts whatever the overlap discards |
| **the lookahead is `t >= step`** | overlapping windows return every row they solved, including the tail the next window recomputes. Keeping only what each window owns is one clause and no special case: `runs.primal('soc').filter(pl.col('t') < step)` |
| a slice that did not solve | contributes no `primal` rows, so that frame can be shorter than the sweep. `objective` is one row per slice always, and is the record of which slices those were |
| **a window keys as `<dim>_start`** | `EachWindow('snapshot', …)` drops `snapshot` and re-indexes to `into`, so the key column is `snapshot_start` and holds where each window began |
| **a hand-built axis names its own key** | a plain list of cuts cannot say what its keys are coordinates *of*, so it must pass `key_name='draw'`. `key_name` overrides the derived name anywhere, and is refused only when it collides with a dimension a kept variable already carries |
| **a sweep's memory grows with its answer** | the models are released as the fold goes; the extracted frames accumulate, and nothing bounds them. `to_parquet` copies out frames already in memory: a bridge, not a bound. Whether to bound it, and how, is [#610](https://github.com/fluxopt/lpspec/issues/610) |

## Carrying state between slices

`carry` copies one slice's answer into the next slice's data:
`{parameter: (variable, index)}`.

```python
runs = lps.solve_over(
    'window.yaml',
    sources,
    lps.EachWindow('snapshot', length=48, step=24, into='t'),
    carry={'soc_initial': ('soc', 23)},
)
```

| Rule | |
|---|---|
| **a copy, never arithmetic** | accumulation — `existing += built` — is a derived variable in the YAML, where the math is reviewable |
| **the two declarations say what is copied** | whichever dimension the *variable* has and the *parameter* does not is the one the carry collapses, and `index` names a coordinate of it. Everything else rides along. So `soc` over `(t, storage)` into `soc_initial` over `(storage)` drops `t` and hands both stores forward, and `total` over `(generator)` into `existing` over `(generator)` drops nothing and needs no index — pass `None` |
| **the index is explicit** | with `EachWindow(…, 48, 24, …)` the state to carry sits at coordinate 23 of `into`, not 47. An implicit "last" is correct until overlap is introduced and silently wrong after |
| **checked before anything is read** | the dims come from the YAML, so a carry that cannot line up — collapsing two dimensions at once, a parameter over more than the variable is, an index where the sides already match — raises before the axis has scanned a single source. `check` cannot answer this for you: `carry` is an argument to the call, not part of the model |
| **the last slice carries nothing** | there is no next slice to read it |
| **`carry` excludes `executor`** | a carried value makes slice *i+1* depend on slice *i*, so the slices cannot run concurrently. Refused rather than one silently winning |

## Running slices in parallel

`executor` is any
[`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#executor-objects) —
a `submit` returning a `Future`, and nothing else. This package ships no remote
transport and no vendor integration, so the executor is the extension point,
and it has to be one anybody can implement.

| | Use it when | Notes |
|---|---|---|
| `None` *(default)* | always, until measurement says otherwise | sequential. Nothing is serialised, because nothing crosses a boundary |
| `ThreadPoolExecutor` | rarely | works, and sources are **not** encoded — but polars is already multithreaded, so slices contend with its pool, and threads share an address space so peak is additive rather than per-worker |
| `ProcessPoolExecutor` | genuine local parallelism | **must not use `fork`** — below. Sources cross as parquet |
| anything remote | a cluster you already run | dask's `Client`, ray's wrappers, loky. Assumed not to share your filesystem, so paths travel as bytes; pass `workers_share_fs=True` if the workers really do mount it |

**A forked worker hangs.** polars' thread pool does not survive `fork`, and the
failure is a hang rather than an error — indistinguishable from a slow solve,
which makes it the worst shape a failure can take. It cannot be enforced from
inside `solve_over`, because a remote executor has no start method to inspect,
so pass the context yourself:

```python
import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def main():
    ctx = multiprocessing.get_context('spawn')  # or 'forkserver'
    with ProcessPoolExecutor(4, mp_context=ctx) as pool:
        runs = lps.solve_over('model.yaml', sources, lps.EachCoordinate('scenario'), executor=pool)


if __name__ == '__main__':  # spawn re-imports your module; without this it recurses
    main()
```

**Parallel is N × peak.** Each worker holds its own slice's model, so a
four-way pool wants four times the memory of one slice. That is a machine
decision, and the reason `None` is the default rather than a pool sized for
you.

Sources cross a process boundary as parquet, never as pickled frames. A path
the workers can reach stays a path; one they cannot travels as its own bytes
untouched. Either way a source no slice rewrote is encoded once for the whole
sweep rather than once per slice. **Pass paths or frames, whichever you already
have** — there is nothing to tune. (`df.lazy()` is not an optimisation: an
eager frame is embedded in the plan, so it pickles *larger* than the frame.
Only `scan_parquet` is a reference.)

## What a sweep does under the hood

| | |
|---|---|
| **a partition is a filter on the sources** | not a narrower index: the containment check refuses parameter rows outside the declared coordinates, by design. The axis rewrites the rows and the index it is over in one mapping |
| **one model, rebound per slice** | every slice is the same math over different numbers, so a serial sweep builds once and [rebinds](api.md#re-solving-with-new-numbers): the YAML is parsed once, the plan lowered once, and a slice whose structure matches the last keeps the loaded solver. A sweep under `executor=` cannot — a built model is the one thing that does not cross a process — so it builds per slice |
| **`keep=` reaches every slice, and the fold chooses none of them** | it defaults exactly as [`solve`](api.md#how-much-of-the-session-a-solve-keeps) does, to `'solver'`. A fold is where `keep='progress'` has something to carry, consecutive slices differing by one step — but whether carrying pays is a fact about the *model*, and the driver knows no more about that than you do, so it does not decide for you. Under `executor=` it cannot apply at all: a pooled sweep builds per slice, so every slice is a first solve and keeps `'nothing'` |
| **the model is parsed once** | `solve_over` validates it up front and hands every slice the schema, so a model outside the language fails before the data is touched and no worker re-reads the YAML |
| **a cut is total** | a cut says what the *whole* model binds, not what changed since the one before it. The class axes always do; a hand-built list has to keep the rule |
