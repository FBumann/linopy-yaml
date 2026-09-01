# How the benchmarks were taken

The results are on the [benchmark page](benchmarks-scaling.html): five
libraries over four models, with the numbers under each chart. This page is
how they were taken.

**Every published number is the median of a measurement's rounds, and every
band is the first to the third quartile of the same rounds.** Nine rounds is
the floor. Not the fastest round — it is a best-of-n and n is not equal, since
the harness calibrates by duration. Not the mean — one round in forty of a
20 ms measurement took 1.5 s here, which drags a mean to 2.9x its median.

That choice cost us: nine cells flipped against lpspec, all on the `gurobi`
sink, where our build alternates between a fast and a slow state round after
round and no other library's does
([#1288](https://github.com/fluxopt/lpspec/issues/1288)).

## How to reproduce it

```bash
uv run --locked bench/reproduce.py
```

`bench/reproduce.py.lock` freezes every version, git commits included — two
of the five libraries install from git and one of those is a branch, so
without it "the versions that produced this number" is unrepeatable.
`--locked` refuses to start if the resolution has drifted.

Everything the tables are drawn from is in
[`bench/results`](https://github.com/fluxopt/lpspec/blob/main/bench/results) —
one file per sink and case, since each is measured in a process of its own, and
each carries the machine, the versions, the commit and every round of every
measurement. A case the box could not finish leaves no file behind, so the
directory holds what ran and nothing else.
`pixi run table` prints it as one long CSV and commits nothing — the JSON is
the archive because it keeps the rounds. Re-taking rather than reproducing is
`pixi run refresh`, which writes the tables into their fences and the chart's
data literal into its own.

## First model against every model after it

One model built, then built again in the same process.

<!-- bench:marginal -->

### Marginal cost per model

Build only, repeated in one process. **first** is the first recorded round and **steady** the best of the rounds after it, so the pair is what a rolling horizon pays for its second window against its first. The harness warms up before it records, so neither column carries the one-time import cost: the median gap between them is +21.5 ms on lpspec and +2.2 ms on linopy and +2.1 ms on pyomo and +18.3 ms on gurobipy-loop and +13.4 ms on gurobipy-matrix.

**Read down a column, not across the row.** The build is not the same work in every library — one that defers materialising its coefficients to its writer spends almost nothing here and pays it at the seam — so these columns carry no ratios. The tables above measure to a common artifact and are where a comparison belongs.

| case | vars | lpspec: first | lpspec: steady | linopy: first | linopy: steady | pyomo: first | pyomo: steady | gurobipy-loop: first | gurobipy-loop: steady | gurobipy-matrix: first | gurobipy-matrix: steady |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dispatch | 10k | 39.6 ms | **29.1 ms** | 27.1 ms | 25.5 ms | 36.5 ms | 34.4 ms | 29.0 ms | 28.9 ms | 17.4 ms | 17.4 ms |
| fleet | 12k | 159.7 ms | **133.5 ms** | 164.5 ms | 165.2 ms | 40.4 ms | 38.6 ms | 90.0 ms | 74.2 ms | 27.8 ms | 27.1 ms |
| dispatch | 100k | 45.3 ms | **33.2 ms** | 28.3 ms | 27.1 ms | 284.6 ms | 282.4 ms | 234.5 ms | 235.0 ms | 87.4 ms | 86.9 ms |
| fleet | 120k | 182.3 ms | **150.3 ms** | 167.5 ms | 164.6 ms | 749.4 ms | 747.6 ms | 877.3 ms | 859.0 ms | 163.5 ms | 156.3 ms |
| dispatch | 1M | 106.9 ms | **90.1 ms** | 36.7 ms | 35.3 ms | 4760.3 ms | 4725.0 ms | 2806.1 ms | 2776.1 ms | 920.4 ms | 900.8 ms |
| fleet | 1.2M | 234.1 ms | **251.1 ms** | 205.4 ms | 190.7 ms | 5927.8 ms | 5887.4 ms | 8406.0 ms | 8346.0 ms | 1694.4 ms | 1673.9 ms |
| dispatch | 10M | 595.5 ms | **545.9 ms** | 162.4 ms | 159.0 ms | — | — | 29212.6 ms | 28915.9 ms | 9626.4 ms | 9485.9 ms |
| fleet | 12M | 1597.2 ms | **1495.8 ms** | 476.8 ms | 468.5 ms | — | — | — | — | 17573.4 ms | 17277.9 ms |

<!-- bench:/marginal -->

## The same size, reached by widening

Entity counts x N, snapshots fixed — the same sizes as the ladder, in a
different shape.

<!-- bench:sweeps -->

### The width ladder

Entity counts x N with the snapshot count held fixed, through the `highs` sink. Each rung matches one of the size ladder rungs above variable for variable — `w10` is `s`, `w1000` is `l` — so the pair reads as one model at one size in two shapes.

| case | entities x | variables | wall: lpspec | wall: linopy | wall: pyomo | wall ÷ linopy | wall ÷ pyomo | peak: lpspec | peak: linopy | peak: pyomo | peak ÷ linopy | peak ÷ pyomo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| storage | 1 | 10k | 0.04 s | 0.08 s | 0.31 s | 0.55x | 0.14x | 0.22 GB | 0.24 GB | 0.19 GB | 0.90x | 1.16x |
| storage | 10 | 100k | 0.06 s | 0.10 s | 2.91 s | 0.62x | 0.02x | 0.25 GB | 0.25 GB | 0.32 GB | 0.96x | 0.76x |
| transport | 1 | 9.8k | 0.04 s | 0.06 s | 0.26 s | 0.71x | 0.16x | 0.22 GB | 0.25 GB | 0.19 GB | 0.87x | 1.13x |
| transport | 10 | 98k | 0.06 s | 0.31 s | 2.45 s | 0.19x | 0.02x | 0.25 GB | 0.84 GB | 0.35 GB | 0.30x | 0.72x |

<!-- bench:/sweeps -->

## Not measured yet

Listed so that a claim with no table under it is visible as one.

- **Solve time.** Every number stops at the hand-off; the simplex is the
  solver's work whoever filled the model.
- **The LP-file round trip.** The tables price writing a file, never reading
  one back.
- **Sizes past `l`.** `xl` and `2xl` exist in the harness and no run
  publishes them.
- **The width ladder past `w10`.** `w100` and `w1000` are left out of the
  published run rather than measured and dropped. `transport/w100` on linopy
  peaks at 14.26 GB, and a measurement holds the model twice, so the cell wants
  more machine than the box has; the budget cannot stop it either, projecting
  the next rung linearly off a `w10` cell that took under a gigabyte. What is
  lost is the runner rather than the rung
  ([#1416](https://github.com/fluxopt/lpspec/issues/1416)). The last numbers
  taken there — lpspec 0.11 s and 0.59 GB against linopy 53.53 s and 14.26 GB
  at `transport/w100` — are in
  [#1285](https://github.com/fluxopt/lpspec/pull/1285), on the machine that
  could hold them.
- **Anything about expressiveness.** Four models say nothing about a fifth.

## Method

One process per measurement, `ru_maxrss` for peak rather than a tracker,
import excluded from the timing and teardown included. A run refuses to
start on a machine that is already working. The rest — every flag, every
default switched off and what it costs — is in
[`bench/README.md`](https://github.com/fluxopt/lpspec/blob/main/bench/README.md).

**Peak carries an allocator cost that only the polars arms pay.** polars
ships its own jemalloc settings, so a peak measured through it holds pages
freed and not yet returned; an arm on the system allocator never enters
jemalloc at all. Ours moves 12–27% with the decay clock on and off where
linopy's does not move at three digits ([#896](https://github.com/fluxopt/lpspec/issues/896)).
It runs against us and is left in.

**memray never times anything.** Its tracker slows an allocation-heavy
engine several-fold and overcounts reserved arenas. Peak RSS is the metric;
memray is for attribution.
