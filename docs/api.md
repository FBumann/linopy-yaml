# Python API

How you *run* a model. The model itself is the YAML file —
[SPEC](SPEC.md) is what it may contain and what it means; this page is the
nineteen names that load, check, build, solve and read one back. The surface is
pinned by a test and the reasoning behind its size is
[ARCHITECTURE](ARCHITECTURE.md#the-python-surface).

Six verbs — `check`, `load_model`, `build`, `solve`, `solve_over`, `write` — and the
exception tree rooted at `LpspecError`: `LanguageError` (with `SchemaError`,
`DimensionError`, `PiecewiseExpansionError`) for the model, `DataError` for what
was bound to it.

```python
import lpspec as lps

lps.check('model.yaml')  # parse → validate → lower, no data bound
model = lps.load_model('model.yaml')  # Model — the declared math
model.to_dict()  # ...and back out, as data
model.to_yaml()  # ...or as the file a reviewer reads

result = lps.solve('model.yaml', sources, solver_options={'time_limit': 60})
# ...or solver_name='gurobi', the other solver sink — same model either way
result.status, result.termination_condition, result.objective
result.is_ok  # rolled-up verdict: not an error, abort or refusal
result.has_primal  # narrower: are there values to read
result.primal('p')  # tidy frame (dims…, value) in label order — the native shape
result.dual('power_balance')  # shadow prices, the same shape and the same join
result.activity('power_balance')  # each row's left-hand side at the solution — defined for a MILP, unlike dual
result.expression('co2')  # a named expression at the solution — same shape, its own dims
result.to_pandas('p')  # the same, as a DataFrame
result.to_dataarray('p')  # the same, labelled: .sel / resample / plot
result.to_dataset()  # every variable by default; names for a subset
result.to_parquet(directory)  # streamed to disk, never through this process

lps.write('model.yaml', sources, 'model.lp')  # sink chosen by the suffix
```

**Nothing has to be released.** The built model is frames this process owns, so
`primal` and the `to_*` readers stay valid for as long as the `Result` does.
`close()` and the context-manager protocol exist to hand a large model back
early, not because forgetting them breaks anything. `lps.build` returns a
`BoundModel` — the math with your data on it — when one build should feed more
than one sink, or be solved more than once:

```python
bound = lps.build('model.yaml', sources)
bound.write('model.lp')
bound.diagnostics()  # what the build and its solves did that the answer does not show
result = bound.solve()
```

### Re-solving with new numbers

`rebind` puts new data on a model that is already built, so a loop that solves
the same math over and over pays for the YAML, the plan and the build once:

```python
bound = lps.build('sub.yaml', sources)
for capacity in search:
    result = bound.rebind({'cap_hat': capacity}).solve()
    price = result.dual('capacity')  # each result reads its own build, rebinds notwithstanding
    bound.diagnostics()  # did that push values, or load the model again?
```

| | |
|---|---|
| **it names what changed** | everything else keeps what `build` bound. A parameter, or a dimension index — a coordinate set grows by handing over a longer table and the `coords=` to match, which is how a Benders cut family is *data* |
| **the answer is the reference build's** | `bound.rebind(x)` solves what `build(model, sources \| x)` solves, always. That is an equality a test asserts, not a promise — it is also the oracle to reach for when a loop looks wrong |
| **it never refuses** | there is no capability to query and no shape of data it rejects. What new values can cost is the *fast path*, never the answer |
| **the solver stays loaded where it can** | new bounds, costs and right-hand sides go onto the model HiGHS already holds, so the matrix is not handed over twice; whether the next solve also carries on from the work the last one did is `keep=`, below. A rebind that moves a **mask** — a parameter a `where` compares against — renumbers labels, and that model is loaded again and solved cold |
| **which one ran is `bound.diagnostics()`** | `loads` counts the solves that had to load the model from scratch, against `solves` as its denominator. A driver on the fast path leaves `loads` at one however many times it goes round; `loads == solves` is the difference between "lpspec is slow" and "this model masks on a parameter that varies". Advisory — nothing about the answer depends on it |
| **earlier results keep reading** | a `Result` owns its values and a reference to the label frames of the build it answered; a rebind builds new frames without touching those, so an old answer stays an answer over its own coordinates. What retaining one costs is those frames staying alive until it is dropped or `close()`d |
| **a rebind that raises releases the model** | the same rule as `build`: half a model would answer the next `solve` with a mixture of two |

`solve_over` is the other spelling and the one to reach for first — a sweep,
a rolling horizon or a myopic pathway is a *fold*, and it is written for you.
`rebind` is the primitive underneath: reach for it when the next set of numbers
depends on the last answer, which is what a fold cannot express. Where the next
set of numbers depends on *you* — a notebook — it is
[Change a model](interactive.ipynb), which runs this loop beside the two
costlier ones a session also has: growing a coordinate set, and patching the
declarations.

### How much of the session a solve keeps

A session holds two things: the solver with the model on it, and the work that
solver did. A rebind keeps the first, so a second solve never hands the matrix
over again. Whether it keeps the second is `keep=`, and the two can only be
dropped in that order — there is no carrying on from a solver that was closed:

```python
result = bound.rebind({'load': load}).solve()
result.kept  # 'solver' — reused, and the work it did discarded

again = bound.rebind({'load': more}).solve(keep='progress')
again.kept  # 'progress' — it carried on from where the last solve got to

baseline = bound.solve(keep='nothing')  # whatever the session held, gone
baseline.kept  # 'nothing'
```

| | |
|---|---|
| `keep='solver'` (default) | the solver already holding the model is reused and the hand-off skipped; the work it did is discarded, so the run begins as if the model had never been solved |
| `keep='progress'` | that work is kept too — **opt-in, because it swings both ways**. A solver told where to begin skips the presolve it would otherwise run, and which of the two is worth more is a fact about your model. Six rebinds, HiGHS, measured both ways (#815): on a dispatch presolve cracks outright, carrying cost **76.6 s against 4.3 s** — an 18× *loss*; on a storage model with a cyclic recurrence it cannot crack, carrying cost **111.2 s against 213.9 s** — a 1.9× *win*. Same procedure, opposite answers, so nothing here guesses on your behalf — measure it, below |
| `keep='nothing'` | held to **structurally**: the held solver is discarded before the load, so the fresh one *has* nothing to start from — no basis, no incumbent, no solver-internal state, and nothing a member has to remember to scrub. `diagnostics().loads` ticks, the whole model having been transferred again. **The baseline, never the fast path**: a fresh solver does the same work as a reused one — same iterations, solve for solve — and the hand-off is paid on top of that (#815) |

`result.kept` is read off what happened, never off what was asked, so a rebind
that had to rebuild reports the `'nothing'` it got rather than the `'progress'`
it hoped for. It is what a benchmark needs — a cold baseline you can prove
is cold — and what an iterating driver reads when a loop is slower than it
should be: `'nothing'` every iteration means the session is being rebuilt away,
and `loads` ticks on exactly those solves.
**Provenance, deliberately, not mechanism**: whether progress is a basis, an
incumbent or a solver's own notion stays the sink's business, so a solver with
no simplex fits the same words and a word can be added the day something else
can be kept.

**Which keep your model wants is measured, not reasoned about.** Run the loop
each way and read the clock the package already keeps; `kept` confirms the
request was honoured rather than quietly downgraded:

```python
for keep in ('solver', 'progress'):
    bound = lps.build('model.yaml', sources)
    for numbers in walk:
        assert bound.rebind(numbers).solve(keep=keep).kept in {keep, 'nothing'}
    print(keep, bound.diagnostics().timings['solve'])
```

Take the faster one. Nothing about the answer changes either way — across both
models above the objectives agreed to 2e-15 relative — so this is a timing
question and only a timing question.

**Carrying progress across a rebuild is not here yet.** The sinks can read a
start out of a session and set it on another, but nothing above them does: the case
that wants it most — a cutting-plane master re-solved after gaining a cut — is
a model that gained a *row*, and a basis spans the model it was read from.
[#382](https://github.com/fluxopt/lpspec/issues/382) is where that is being
worked out.

`diagnostics()` is what a build and its solves did that the answer does not
show: the shape the build produced (`columns`, `rows`, `nonzeros` — what
`check` cannot answer, needing no data where this needs all of it, and where a
broadcast that multiplied rows shows up first), what the last solve's sink had
to *add* to that shape (`sink_columns`, `sink_rows` — zero unless it had no
concept of a set the model declares, in which case this is the binaries and
linking rows it was handed instead), `omissions` (rows a constraint declared but
did not build, and why that matters), `solves` with `loads` (above; `solves`
is the denominator to read `loads` against), and `timings` (cumulative wall
seconds per phase — `bind`, `build`, `handoff`, `solve`, `write` — so a run
that is slower than it should be can say which phase the time went to). It
answers after `close()` too, every field being a count, a clock or a small
frame it keeps rather than a read of the model it releases.

Advisory, all of it: nothing about an answer depends on any of them, and a
caller who branches on one has made this engine's bookkeeping part of their
model.

**Inspecting a model is `build`'s job, not `solve`'s.** `solve` hands back an
answer and `write` a path; the questions *about the model* — how big is it, what
did it not build, how did its re-solves go — belong to the handle that **is** the
model, so a caller who wants them builds and keeps it. That is also what keeps
the record honest: read off a `Result`, `solves` and `loads` would have to
report what happened *after* that answer was produced, which is not a fact about
it. A `Result` reports its own solve, and nothing else does.

What `sources` accepts is [SPEC §8](SPEC.md#8-data-binding). Nothing on this
path imports linopy, and `primal` returns a `polars.DataFrame` — Arrow-backed, so it exports the same protocol the
loader recognises. `to_pandas` and `to_dataarray` are the bridges out and need
pandas / xarray, which ship with the `[linopy]` extra. The only build knob is
`coords`; **`solver_options` is not a build knob** and is forwarded verbatim to
the solver.

**Every verb takes the model four ways**: a path, a `str`, a `dict`, or a
`Model`. `check`, `build`, `solve` and `write` share one first argument, so a
framework that emits declarations never writes a temporary file to run them:

```python
model = {'dimensions': ..., 'variables': ..., 'constraints': ..., 'objective': ...}

lps.solve(model, sources)  # a dict runs like a file
checked = lps.load_model(model)  # ...or validate once and keep it
checked.to_yaml()  # the review copy — a dict-built model still gets a file
lps.solve(checked, sources)  # a Model is passed through, not revalidated
```

**This is the supported path for a framework**, and it is the one closing #29
and #30 chose: a library composing optional features emits *data*, not YAML
text, and never merges files. What keeps it honest is the last two lines — a
generated model that cannot show you a file is the failure mode hard rule 5
exists to prevent, so `to_yaml()` is not a convenience here, it is the
condition. Hand-written math still starts as a file; nothing about this path
asks it not to.

**A `Model` goes back out two ways, and they agree.** `to_dict()` is the model
as data; `to_yaml()` is that dict as the file hard rule 5 says you review and
diff — which a model built as a *dict*, the way a framework emits one, would
otherwise never have. `tests/test_roundtrip.py` holds `load → out → load` for
both forms over every example and every port, holds that the two forms match,
and holds that dumping twice gives the same bytes, since a review copy that
changes per run is a diff nobody can read.

**Every value is written; only what is absent is dropped** — a null, an
infinite bound, or a mapping that declares nothing. One mechanical rule, on purpose: omitting
*defaults* reads better but needs a list of which ones are consequential, and
that list is a second copy of the schema. An empty **list** stays, because a
list carries cardinality here and zero is one of its values — `foreach: []` is
a scalar declaration.

An infinite bound is in that list because it is not a bound — it is the
unbounded side, which is exactly what omitting the bound already means. That
also makes JSON lossless: JSON has no infinity, so anything reaching
`model_dump_json` as `inf` came back as `null` and read as absent regardless.

The rule lives on the model's **serializer**, so `model_dump`, `model_dump_json`,
`to_dict` and `to_yaml` all give the same content — a helper beside them would
have left pydantic's own methods describing the model differently from the
file.

**Which solver is a caller's choice, not the file's.** `solver_name` is
`highs` (ships with the package) or `gurobi` (needs the `[gurobi]` extra), and
nothing in the YAML names one — the same file means the same model whichever
takes it. Options travel in the chosen solver's own vocabulary,
`{'time_limit': 60}` for HiGHS against `{'TimeLimit': 60}` for Gurobi, because
forwarding verbatim is the contract and translating names would mean holding an
opinion about every option either one has. A name outside the two is an error
listing them, never a quiet fallback to the default.

**Gurobi's remote and licensing options travel the same way**, so Compute
Server, Instant Cloud and WLS need nothing from this package:

```python
options = {'ComputeServer': 'srv:61000', 'ServerPassword': '…'}
lps.solve('model.yaml', sources, solver_name='gurobi', solver_options=options)
```

They are applied when Gurobi's environment is created, which is what
`ComputeServer`, `TokenServer` and `WLSAccessID` require.

Reading a result:

| Rule | |
|---|---|
| **`is_ok` is not `has_primal`** | `is_ok` rolls up the termination condition; `has_primal` adds the solver's verdict on whether an incumbent exists, and is what every reader gates on. A MIP that hits `time_limit` before finding a feasible point is `ok` with nothing to read |
| reading anyway | `NoSolutionError`; `objective` is `nan` |
| **`expression` reads what the model named** | the value of a declared named expression ([SPEC §3](SPEC.md#3-expressions-and-macros)) at the solution, aggregated to the expression's own dims (declaration order — an expression has no `foreach`). Takes a declared name only, never an expression string; an unknown one is a `KeyError` listing what is declared. Lowered and compiled **at the read**, through the same compiler the constraints use, so a build with fifty declared expressions that reads none pays for none. Semantics are a constraint's: an uncovered parameter coordinate contributes zero (SPEC §8), a coordinate where a term's variable is absent has no row (SPEC §7), an undefined divisor is a `DataError`. On the linopy lane the same read is `lpspec.linopy.expression(m, path, name, data=…)` |
| `dual` **raises rather than zero-filling** | no values at all is `NoSolutionError`; values but no duals — any integer or binary variable makes them undefined — is `LpspecError`, because only this quantity is missing |
| **the sink can make a model mixed-integer** | a `sos:` set ([SPEC §4.1](SPEC.md#41-sos)) reaches a solver with no SOS concept as binaries, so an otherwise continuous model solved on `highs` has no duals and says so. Solving it on `gurobi`, which branches on the set itself, keeps them |
| duals exist only where a solver ran | either solver sink hands them back through the same join; a model written to LP and solved elsewhere never passes back through here. Reduced costs and slacks ride that join too and are not exposed yet |
| `to_dataset` costs what it says | each variable arrives dense over its own dims — name a subset, or use `to_parquet` |
| `write` | the **suffix** picks the writer — `.lp` today, anything else a `ValueError` listing what can be written. Checked before the build, so a format nothing can write costs no model |

## Solving one model many times

`solve_over` runs the same model once per slice and folds the answers. It is a
driver over `solve`, not a second engine: **a plan cannot contain a loop; a
process may loop over plans** ([the ceiling](design/ceiling.md)). Scenarios,
rolling horizons and myopic pathways are all the same fold.

```python
runs = lps.solve_over('model.yaml', sources, lps.EachCoordinate('scenario'), executor=ProcessPoolExecutor(4))
runs.objective  # (scenario, status, termination_condition, objective)
runs.primal('p')  # (scenario, snapshot, generator, value)

runs = lps.solve_over(
    'window.yaml',
    sources,
    lps.EachWindow('snapshot', length=48, step=24, into='t'),
    carry={'soc_initial': ('soc', 23)},
)
runs.primal('soc')  # (snapshot_start, t, value) — the window, and the index inside it
```

**`Runs` reads like `Result`, one dimension wider** — `primal`, `dual`,
`expression`, `to_pandas`, `to_dataarray`, `to_dataset`, `to_parquet`, under the same names
and with the slice key prepended. That extra dimension is **named by you, not
by the library**: `EachCoordinate('scenario')` keys on `scenario` and
`EachCoordinate('draw')` on `draw`, a window on `<dim>_start`, and `key_name=`
overrides either. So `runs.to_dataarray('p')` on a scenario sweep is
`(scenario, snapshot, generator)`, which is what a sweep is *for*: `.sel` one
scenario, take a spread across them, plot the band.

**`original_index=` is how you ask for the answer over real coordinates**, and
it is a keyword on the readers rather than a reader of its own:

```python
runs.primal('soc')  # (snapshot_start, t, value) — keyed by slice
runs.primal('soc', original_index=True)  # (snapshot, value)          — the answer
runs.dual('balance', original_index=True)  # the same, for a price
runs.expression('spend', original_index=True)  # the model's own quantity, over real coordinates
```

A flag rather than a method because what has to be undone is a property of the
*axis*, not of the quantity — so duals get it for free, and a name that is both
a variable and a constraint (which the language permits) is never dispatched
on. `to_pandas` and `to_dataarray` take it too.

**Every axis answers it.** For `EachWindow` it is the answer a rolling horizon
is *for*: the overlap dropped and the global coordinate restored, each window
contributing the `step` coordinates it owns — the final one included, which can
hold no more and so keeps all of it. `snapshot_start + t` would not do, because
a window spans coordinates rather than values and there is nothing to add for a
datetime or a string axis. For `EachCoordinate` and a hand-built axis nothing
was re-indexed, the key column already *is* a coordinate, and the frame comes
back unchanged.

**Keyed is the default**, because stitching is lossy: it keeps only what each
window owns and drops the lookahead rows the sweep solved. A default that
discarded computed answers would also key differently from `objective`, which
is one row per slice always, and the two would stop joining. For the same
reason `to_dataset` and `to_parquet` have no `original_index` — a bulk export
of what the sweep holds is the wrong place to lose rows.

Per slice is a partition of a frame you already have, so there is no reader for
it: `runs.primal('p').partition_by(runs.key_name, as_dict=True)`.

| Rule | |
|---|---|
| **a partition is a filter on the sources** | not a narrower `coords` — the containment check refuses parameter rows outside the declared coordinates, by design. The axis rewrites the sources and supplies the matching `coords` together |
| **one model, rebound per slice** | every slice is the same math over different numbers, so a serial sweep builds once and [rebinds](#re-solving-with-new-numbers): the YAML is parsed once, the plan lowered once, and a slice whose structure matches the last keeps the loaded solver. Peak is unchanged, a rebuild releasing the previous model before it starts. A sweep under `executor=` cannot — a built model is the one thing that does not cross a process — so it builds per slice, which is also why `carry` and `executor` are mutually exclusive |
| **`keep` reaches every slice, and the fold chooses none of them** | it defaults exactly as [`solve`](#how-much-of-the-session-a-solve-keeps) does, `'solver'`. A fold is where `keep='progress'` has something to carry, consecutive slices differing by one step — but whether carrying pays is a fact about the *model*, and the driver knows no more about that than you do, so it does not decide for you. Under `executor=` it cannot apply at all: a pooled sweep builds per slice, so every slice is a first solve and keeps `'nothing'` |
| **everything a slice produced is kept** | every variable's primals and every constraint's duals, read back through `runs.primal(name)` and `runs.dual(name)`. It is still a fold — each slice's *model* is released as the loop goes, so build peak stays at one slice however many there are, and what accumulates is the answer. Narrowing that is a later addition and an easy one; it is absent because it would need *two* keywords, a constraint being allowed to carry a variable's name |
| **duals are keyed, never combined** | `runs.dual(name)` is `runs.primal(name)`'s shape. Averaging window prices, taking the last, and reading one slice alone are all defensible, so the reduction is the caller's. A slice whose model had an integer variable contributes none, and `runs.objective` says which |
| **expressions are evaluated per slice** | every declared `expressions:` name, evaluated at each slice's solution as the fold reads it, back through `runs.expression(name)` in `runs.primal(name)`'s shape. Eager where `Result.expression` defers, because a deferred reader holds its build's frames and the fold releases each slice's model as it goes; per slice it costs what one more variable read does. Over `original_index=True` only the rows each window owns survive, so summing the stitched frame cannot double-count the lookahead — and a quantity *reduced over* the sliced dimension has no way back and is refused there, with the per-slice read named as the alternative |
| **no aggregate objective** | `objective` is a frame keyed by slice. Scenarios are a distribution, not a sum; summing window objectives double-counts whatever the overlap discards |
| `carry` is a copy, never arithmetic | `{parameter: (variable, index)}`. Accumulation — `existing += built` — is a derived variable in the YAML, where the math is reviewable |
| **the two declarations say what is copied** | whichever dimension the *variable* has and the *parameter* does not is the one the carry collapses, and `index` names a coordinate of it. Everything else rides along. So `soc` over `(t, storage)` into `soc_initial` over `(storage)` drops `t` and hands both stores forward, and `total` over `(generator)` into `existing` over `(generator)` drops nothing and needs no index — pass `None` |
| the carry index is explicit | with `EachWindow(…, 48, 24, …)` the state to carry is at coordinate 23 of `into`, not 47. An implicit "last" is correct until overlap is introduced and silently wrong after |
| **a carry is checked before anything is read** | the dims come from the YAML, so a carry that cannot line up — collapsing two dimensions at once, a parameter over more than the variable is, an index where the sides already match — raises before the axis has scanned a single source, never mind solved a slice. `check` cannot answer this for you: `carry` is an argument to the call, not part of the model |
| **a hand-built axis names its own key** | the class axes derive the key column — `EachCoordinate('scenario')` keys on `scenario`, a window on `<dim>_start` — but a plain list of cuts cannot say what its keys are coordinates *of*, so it must pass `key_name='draw'`. `'slice'` would be this library naming somebody else's axis, which is the same reason `into` has no default. `key_name` overrides the derived name anywhere, and is refused only when it collides with a dimension a kept variable already carries |
| **a cut is total** | a cut says what the *whole* model binds, not what changed since the one before it. The class axes always do — each rewrites a copy of the whole source mapping and supplies its own coords per slice — so this is a rule a hand-built list has to keep. A serial fold rebuilds rather than rebinds where a slice names other sources or other coords, since `rebind` is partial by construction and a slice inheriting the last one's data would answer differently from the same sweep under `executor=` |
| the model is parsed once | `solve_over` validates it up front and hands every slice the schema, so a model outside the streaming language fails before the data is touched and no worker re-reads the YAML |
| **a window keys as `<dim>_start`** | `EachWindow('snapshot', …)` drops `snapshot` and re-indexes to `into`, so the key column is `snapshot_start` and holds where each window began. Naming it `snapshot` would put window starts under the name of the coordinate they are not, and join cleanly against real data |
| a slice that did not solve | contributes no `primal` rows, so that frame can be shorter than the sweep. `objective` is one row per slice always, and is the record of which slices those were |
| **the lookahead is `t >= step`** | overlapping windows return every row they solved, including the tail the next window recomputes. Keeping only what each window owns is one clause and no special case — `runs.primal('soc').filter(pl.col('t') < step)` — because the final window can never hold more than `step` rows |
| **a sweep's memory grows with its answer** | each slice's *model* is released as the fold goes, so build peak stays at one slice — but the extracted frames accumulate, and nothing bounds them. `to_parquet` copies out frames already in memory: a bridge, not a bound. Whether to bound it, and how, is [#610](https://github.com/fluxopt/lpspec/issues/610) |
| **a window spans coordinates, not values** | `length=48` is forty-eight snapshots however they are numbered, so the dimension only has to be **orderable** — datetimes, strings and gapped integers all work. `into` is a dense `0..n-1` local index, which is what keeps the seam's `where: "t == 0"` matching, and it has no default because the name belongs to the model |
| non-positional grouping | *"each calendar month"* has unequal groups, so it is a precomputed column plus `EachCoordinate`. What `EachWindow` uniquely offers is **overlap** |
| `carry` excludes `executor` | a carried value makes slice *i+1* depend on slice *i*, so the slices cannot run concurrently. Refused rather than one silently winning |
| `workers_share_fs` | whether the executor's workers can read your paths. Inferred from the pool — a stdlib `ProcessPoolExecutor` runs here and reads what is here, anything else is assumed remote — and only path sources are affected |

### Choosing an executor

`executor` is any [`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#executor-objects) —
a `submit` returning a `Future`, and nothing else. That is deliberate: **this
package ships no remote transport and no vendor integration**, so the executor
is the only extension point there is, and it has to be one anybody can
implement. `tests/test_strategy.py` runs the whole sweep through a nine-line
class to keep that honest.

| | use it when | notes |
|---|---|---|
| `None` *(default)* | always, until measurement says otherwise | sequential. Nothing is serialised, because nothing crosses a boundary |
| `ThreadPoolExecutor` | rarely | works, and sources are **not** encoded — but polars is already multithreaded, so slices contend with its pool, and threads share an address space so peak is additive rather than per-worker |
| `ProcessPoolExecutor` | genuine local parallelism | **must not use `fork`** — see below. Sources cross as parquet |
| anything remote | a cluster you already run | dask's `Client`, ray's wrappers, loky. Assumed not to share your filesystem, so paths travel as bytes; pass `workers_share_fs=True` if the workers really do mount it |

**A forked worker hangs.** polars' thread pool does not survive `fork`, and the
failure is a **hang** rather than an error — indistinguishable from a slow
solve, which makes it the worst shape a failure can take. Measured: `fork` never
returns where `spawn` and `forkserver` both do. It cannot be enforced from
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

Sources cross a process boundary as parquet, never as pickled frames — smaller
and faster on every shape it was measured against. A path the workers can reach
stays a path; one they cannot travels as its own bytes untouched, because
decoding and re-encoding a parquet file produces identical output at a large
multiple of the CPU. Either way a source no slice rewrote is encoded once for the whole sweep
rather than once per slice. **Pass paths or frames, whichever you already
have** — there is nothing to tune. (`df.lazy()` is not an optimisation: an eager
frame is embedded in the plan, so it pickles *larger* than the frame. Only
`scan_parquet` is a reference.)

**The linopy shim** (`lpspec.linopy.build` / `.extend` / `.expression`,
`[linopy]` extra) puts the same YAML math on a `linopy.Model` that already
exists in memory, and reads a named expression back off a solved one. It is
documented with everything else about that relationship in
[docs/design/linopy.md](design/linopy.md#3-the-shim).
