# How the benchmarks were taken

**Cost is a property of the engine, not of the language.** The rules in the
architecture notes constrain what a file may say and would survive an engine
swap untouched; what a build *costs* is settled by measurement.

The results are on the [benchmark page](benchmarks-scaling.html) — five
libraries over four models, with the numbers under each chart. This page is the
method: how to reproduce a figure, what the harness refuses to measure, and
what each sink can carry.

## How to reproduce it

```bash
uv run --locked bench/reproduce.py
```

`bench/reproduce.py` carries the published selection and
`bench/reproduce.py.lock` beside it freezes every version it runs on — **git
commits included**, which matters here more than it usually would: two of the
five libraries install from git, and one of those is a branch. `--locked`
refuses to start if the resolution has drifted.

Everything the tables are drawn from is in
[`latest.json`](https://github.com/fluxopt/lpspec/blob/main/bench/results/latest.json):
the machine, the library versions and the commit that produced them, and every
round of every measurement rather than the minimum the tables print. The CSV
form is `pixi run table`, which prints and commits nothing — the JSON is the
archive because it keeps the rounds, and a second copy of a reduction would be
free to drift from it.

Re-taking the numbers rather than reproducing them is `pixi run refresh`, which
runs the ladders and then writes the tables into this page between its fences
and the chart's data literal into its own. Nothing here is pasted by hand.

## First model against every model after it

What a caller pays who builds one model and solves it, against what a
rolling horizon pays for every window after the first.

<!-- bench:marginal -->

### Marginal cost per model

Build only, repeated in one process. **first** is the first recorded round and **steady** the best of the rounds after it, so the pair is what a rolling horizon pays for its second window against its first. The harness warms up before it records, so neither column carries the one-time import cost: the median gap between them is +2.7 ms on lpspec and +3.0 ms on linopy and -19.5 ms on pyomo and +2.6 ms on gurobipy-loop and +1.4 ms on gurobipy-matrix.

**Read down a column, not across the row.** The build is not the same work in every library — one that defers materialising its coefficients to its writer spends almost nothing here and pays it at the seam — so these columns carry no ratios. The tables above measure to a common artifact and are where a comparison belongs.

| case | vars | lpspec: first | lpspec: steady | linopy: first | linopy: steady | pyomo: first | pyomo: steady | gurobipy-loop: first | gurobipy-loop: steady | gurobipy-matrix: first | gurobipy-matrix: steady |
|---|---|---|---|---|---|---|---|---|---|---|---|
| transport | 9.8k | 20.1 ms | **17.3 ms** | 32.5 ms | 27.9 ms | 26.2 ms | 25.8 ms | 25.0 ms | 23.8 ms | 10.2 ms | 8.9 ms |
| dispatch | 10k | 13.6 ms | **8.3 ms** | 15.0 ms | 11.7 ms | 20.5 ms | 19.8 ms | 13.0 ms | 11.9 ms | 7.1 ms | 6.6 ms |
| storage | 10k | 20.7 ms | **18.8 ms** | 44.3 ms | 42.8 ms | 22.8 ms | 21.7 ms | 27.4 ms | 26.3 ms | 11.5 ms | 9.1 ms |
| fleet | 12k | 42.4 ms | **39.1 ms** | 89.4 ms | 89.3 ms | 24.7 ms | 23.6 ms | 39.7 ms | 38.7 ms | 13.3 ms | 12.1 ms |
| transport | 98k | 24.6 ms | **23.4 ms** | 42.2 ms | 37.7 ms | 374.0 ms | 413.4 ms | 232.0 ms | 252.6 ms | 64.8 ms | 63.3 ms |
| dispatch | 100k | 12.6 ms | **11.2 ms** | 15.2 ms | 12.5 ms | 259.9 ms | 353.8 ms | 121.1 ms | 116.4 ms | 55.2 ms | 53.9 ms |
| storage | 100k | 25.8 ms | **25.1 ms** | 47.8 ms | 45.4 ms | 377.6 ms | 320.1 ms | 263.9 ms | 322.3 ms | 71.8 ms | 72.4 ms |
| fleet | 120k | 47.7 ms | **45.0 ms** | 91.2 ms | 91.9 ms | 375.3 ms | 432.9 ms | 455.0 ms | 452.5 ms | 113.1 ms | 112.8 ms |
| transport | 980k | 78.0 ms | **76.0 ms** | 145.7 ms | 140.6 ms | 3683.9 ms | 3814.8 ms | 2601.4 ms | 2645.8 ms | 584.8 ms | 576.6 ms |
| dispatch | 1M | 34.9 ms | **33.6 ms** | 23.5 ms | 19.7 ms | 3049.5 ms | 3171.8 ms | 1557.9 ms | 1522.6 ms | 516.1 ms | 510.3 ms |
| storage | 1M | 80.1 ms | **76.5 ms** | 71.0 ms | 58.6 ms | 3595.0 ms | 3447.9 ms | 3179.5 ms | 2978.6 ms | 638.2 ms | 637.1 ms |
| fleet | 1.2M | 107.5 ms | **103.9 ms** | 111.7 ms | 105.3 ms | 3660.3 ms | 4128.4 ms | 4456.0 ms | 4375.8 ms | 1064.1 ms | 1040.4 ms |
| transport | 9.8M | 660.9 ms | **653.3 ms** | 1298.9 ms | 1300.2 ms | — | — | 37471.4 ms | 26315.6 ms | 5762.0 ms | 5785.1 ms |
| dispatch | 10M | 265.6 ms | **258.7 ms** | 104.7 ms | 107.7 ms | — | — | 16739.4 ms | 15394.1 ms | 5053.6 ms | 5035.5 ms |
| storage | 10M | 760.6 ms | **712.7 ms** | 262.3 ms | 265.9 ms | — | — | 31534.2 ms | 29882.8 ms | 6415.8 ms | 6387.8 ms |
| fleet | 12M | 776.0 ms | **850.1 ms** | 288.5 ms | 262.4 ms | — | — | — | — | 10147.9 ms | 10030.2 ms |

<!-- bench:/marginal -->

## The same size, reached by widening

Entity counts x N with the snapshots held fixed. Each rung matches one of
the size rungs above variable for variable, so the pair is one model at one
size in two shapes.

<!-- bench:sweeps -->

### The width ladder

Entity counts x N with the snapshot count held fixed, through the `highs` sink. Each rung matches one of the size ladder rungs above variable for variable — `w10` is `s`, `w1000` is `l` — so the pair reads as one model at one size in two shapes.

| case | entities x | variables | wall: lpspec | wall: linopy | wall ÷ linopy | peak: lpspec | peak: linopy | peak ÷ linopy |
|---|---|---|---|---|---|---|---|---|
| storage | 1 | 10k | 0.02 s | 0.05 s | 0.45x | 0.20 GB | 0.22 GB | 0.91x |
| storage | 10 | 100k | 0.03 s | 0.06 s | 0.54x | 0.24 GB | 0.24 GB | 0.98x |
| storage | 100 | 1M | 0.13 s | 0.17 s | 0.76x | 0.59 GB | 0.46 GB | 1.27x |
| storage | 1000 | 10M | 1.34 s | 1.48 s | 0.90x | 2.70 GB | 2.58 GB | 1.04x |
| transport | 1 | 9.8k | 0.02 s | 0.03 s | 0.63x | 0.20 GB | 0.23 GB | 0.89x |
| transport | 10 | 98k | 0.03 s | 0.23 s | 0.12x | 0.25 GB | 1.08 GB | 0.23x |
| transport | 100 | 980k | 0.11 s | 53.53 s | 0.00x | 0.59 GB | 14.26 GB | 0.04x |
| transport | 1000 | 9.8M | 1.07 s | >30 s | — | 2.74 GB | — | — |

<!-- bench:/sweeps -->

## Sink capabilities

What each sink can ingest, measured against the shipped solvers rather than
assumed. The architectural reading is in
[the ceiling](https://math-spec.readthedocs.io/en/latest/about/ceiling/#capability-is-not-the-ceiling); the plan is
[Track 3](https://github.com/fluxopt/lpspec/issues/472).

Three rows have since been acted on. Both quadratic rows are in the language —
the math takes degree 2 — and each sink answers for itself: HiGHS by its two
exclusions and by having no quadratic-constraint concept at all, Gurobi by
having none, `lp_file` by writing a section either way. A quadratic constraint
is also the first construct one *lane* cannot build, which is what hard rule 3's
`accepts ≠ builds` amendment is for. The third is `sos:`
([sos](https://math-spec.readthedocs.io/en/latest/reference/language/piecewise/#sos)) ships to every sink, natively where the row says
so and as binaries plus linking rows where it says *no concept*.

| | `lp_file` | `mps_file` | HiGHS direct | Gurobi direct | Xpress direct |
|---|---|---|---|---|---|
| affine rows, COO, integrality | text | text, `MARKER` | native | native | native |
| semi-continuous | text | **not written** — no `SC` bound | `kSemiContinuous` | native | native |
| SOS1 / SOS2 | text section | `SOS` section | **no concept** — `HighsLp` has no SOS field and no `addSos` | `addSOS` | `addSOS` |
| indicator | text section | **not written** | **no concept** | `addGenConstrIndicator` | native |
| convex quadratic objective | text section | **not written** — the section is an extension | `passHessian` | `setMObjective` | **no path here** |
| nonconvex quadratic objective | text section | **not written** | **refused** — *"Cannot solve non-convex QP problems with HiGHS"* | native, at default parameters | **no path here** |
| quadratic objective **and** integrality | text section | **not written** | **refused** — `run()` returns `kError` | native (MIQP) | **no path here** |
| quadratic constraint | text section, unreadable | **not written** | **no concept** — no entry point at all | `addQConstr` / `addMQConstr` | **no path here** |

**A rewrite is not free, and the cost is what comes back.** An LP carrying a
set returns from HiGHS without duals — the reformulation makes it a MIP — and
from Gurobi with them. That asymmetry is the argument for declaring capability
rather than papering over it ([the ceiling](https://math-spec.readthedocs.io/en/latest/about/ceiling/#capability-is-not-the-ceiling)).

**"No path here" is about this tree, not about Xpress.** The Optimizer takes a
Hessian and quadratic rows; the sink in `solvers/xpress.py` never hands it one,
and a descriptor says what the sink ingests rather than what the library could
— so the entries are `absent` and a model needing one is refused by name.

The four quadratic rows are probed rather than remembered, as are the two
sections HiGHS writes and will not read back —
`tests/test_sink_capability_probes.py` and
`tests/test_gurobi_capability_probes.py`, each assertion naming this table.
Capabilities move on somebody else's release, and nothing here calls
`passHessian` yet, so without the probes a row would go wrong with the suite
green. The four rows above them are still read off the APIs rather than
measured, and so is the whole **Xpress** column — this repository probes the
two sinks a quadratic model can actually reach. Three readings:

- **HiGHS excludes quadratic twice**, by *convexity* and by *conjunction* with
  integrality — and neither is a set membership. linopy declares HiGHS with
  `INTEGER_VARIABLES` and `QUADRATIC_OBJECTIVE` in one flat `frozenset`, so its
  own model reports MIQP as available.
- **The `lp_file` column says what can be *written*, not what will be read
  back.** The same HiGHS parser takes the quadratic-objective section and
  refuses both the `sos` and the quadratic-constraint one, so only the round
  trip says which — and a differential oracle that re-solves the written file
  has the reader's answer, not the writer's.
- **Gurobi's column was the unverified one** and is now measured, retiring one
  piece of folklore: a nonconvex quadratic objective needs no `NonConvex=2`.

### The quadratic handoff

Neither direct API has a per-coefficient counterpart to `changeCoeff`:
`passHessian` and `setMObjective` take the quadratic part whole. Under the
aligned-only scope (`variable × variable` at the same coordinates) `Q` is
**diagonal**, so it costs 16 bytes per quadratic column:

| quadratic cols | diagonal Hessian |
|---|---|
| 10⁷ | 0.16 GB |
| 3.56×10⁷ | 0.57 GB |
| 10⁸ | 1.60 GB |

Against a `solver_direct` peak already dominated by HiGHS's own model, that is
a small fraction. On `lp_file` a quadratic
objective is a text section and sinks like any other. So this is a cost, not an
invariant violation. Two caveats:

- HiGHS accepts `dim_ < num_col` (verified), so ordering quadratic variables
  first bounds the Hessian to that block rather than the whole model.
- **The diagonal argument dies as soon as the product is not aligned**, and
  the shipped language does not restrict it to aligned: `x[i] * y[i, j]`
  broadcasts and `x[i] * y[j] * a[i, j]` joins through a table. The replacement
  bound is **one entry per pair the expression states** — `nnz` of whatever
  couples the factors — which is a declared-shape quantity and still tracks the
  model. What does *not* is the cross join of two reductions, and that is the
  shape the language refuses (`language/degree.py`).

**Whole is not the same as reloading.** A second `passHessian` lands on the
model already loaded, replacing `Q` and leaving the LP standing — so a moved
quadratic *coefficient* is pushed like a cost, and only the sparsity *pattern*
is structure.

## Not measured yet

This section exists so that a claim with no table under it is visible as one.
Two of its entries are load-bearing elsewhere — `README.md` and the roadmap
lead on cost, and until these land they lead on the hand-off numbers above and
nothing else.

In rough order of what would change a decision:

- **The LP-file route as a cold floor.** The hand-off tables compare against
  linopy's *best* path deliberately. What they do not price is the route the
  claim "there is no file" is really about: write the LP, then have a solver
  read it back. The one figure in that direction is anecdotal and single-case —
  `dispatch/l` through linopy's `io_api='lp'` peaks at 6.92 GB against 3.38 GB
  direct — and it prices only the *writing* half, in the eager lane.
- **Marginal cost per model in a loop.** The architectural claim is that
  nothing accumulates between builds, so the hundredth rolling-horizon window
  costs what the first did. It follows from there being no process-wide state
  and no lifetime to leak, and every rung here is a single build in a fresh
  process — which is exactly why none of them tests it.
- **`storage` — the cyclic `shift` recurrence.** The one plan shape in the
  language whose cost is not obviously linear in the model. The case now exists
  — `bench/models/storage.yaml`, held at `dispatch`'s width on `dispatch`'s
  ladder so the two read against each other — but every number on this page
  predates it, so it is unmeasured here rather than unwritten.
- **A MILP**, where solve time dwarfs build and the hand-off is the whole
  comparison. The case now exists — `commitment` in `bench/cases.py`, a binary
  commitment gating every generator, the only case whose `vtype` stream is not
  all-continuous — but every number on this page predates it, so it is
  unmeasured here rather than unwritten.
- **The speed-of-light floor.** Without it, every ratio here has linopy as its
  only denominator. The mechanism now exists — `bench/floor.py` hand-writes
  `transport` from the case's cached parquet into numpy arrays and a CSR
  matrix, ending at the same populated-`Highs` seam with `run()` never called —
  but no number from it is published yet. When one is, the sentence becomes
  *"we are at Nx the floor and linopy is at Mx"*.

Two entries that used to be here are now measured and have moved into the file:
`solver_direct` end to end (the `highs` sink, which now runs by default) and the
mask-density sweep.

## Method

Recorded in [`bench/README.md`](https://github.com/fluxopt/lpspec/blob/main/bench/README.md) — one process per
measurement, `ru_maxrss` rather than a tracker, import excluded from
`wall_seconds` and teardown included, and a parity gate that aborts the run
before anything is timed if the two lanes disagree. Failures are results and are
rendered as cells.

Measurement pitfall worth keeping: memray's tracker slows an allocation-heavy
engine several-fold and overcounts reserved arenas, so it can attribute memory
but must never time anything. Peak RSS is the gate metric; memray is for
attribution only.

### The allocator is in the number, and only on one arm

**Every peak on this page includes a jemalloc decay component that the
relational arm pays and the eager arm does not.** polars ships its own jemalloc
settings, so a peak measured through it holds pages that have been freed and
not yet returned; the eager arm's build is xarray and numpy on the system
allocator and never enters jemalloc at all. The asymmetry is not an estimate —
linopy's build arm measures the same to three digits with the decay clock on
and off, where ours moves by 12–27% (#896). It runs one way, against the
relational lane, on every row.

The numbers are published at polars' default anyway, because that is what a
caller who sets nothing actually pays. A caller near a memory ceiling can turn
the decay off:

```bash
_RJEM_MALLOC_CONF=dirty_decay_ms:0,muzzy_decay_ms:0 python build_my_model.py
```

Two things make it look inert when it is not. It has to be set **before
`import polars`** — jemalloc reads the variable at its first allocation, so
assigning to `os.environ` afterwards is silently a no-op. And on macOS only `0`
does anything, jemalloc's background purge thread being unavailable there. It
costs wall time, which is why it is not a default, and what it costs has not
been measured on this ladder.
