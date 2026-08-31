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

Everything the tables are drawn from is in `bench/results` — one file per sink
and case, since each is measured in a process of its own, and each carries the
machine, the versions, the commit and every round of every measurement
([`latest-highs-transport.json`](https://github.com/fluxopt/lpspec/blob/main/bench/results/latest-highs-transport.json)
is one of them).
`pixi run table` prints it as one long CSV and commits nothing — the JSON is
the archive because it keeps the rounds. Re-taking rather than reproducing is
`pixi run refresh`, which writes the tables into their fences and the chart's
data literal into its own.

## First model against every model after it

One model built, then built again in the same process.

<!-- bench:marginal -->

### Marginal cost per model

Build only, repeated in one process. **first** is the first recorded round and **steady** the best of the rounds after it, so the pair is what a rolling horizon pays for its second window against its first. The harness warms up before it records, so neither column carries the one-time import cost: the median gap between them is +2.2 ms on lpspec and +1.9 ms on linopy and +2.6 ms on pyomo and +2.8 ms on gurobipy-loop and +1.1 ms on gurobipy-matrix.

**Read down a column, not across the row.** The build is not the same work in every library — one that defers materialising its coefficients to its writer spends almost nothing here and pays it at the seam — so these columns carry no ratios. The tables above measure to a common artifact and are where a comparison belongs.

| case | vars | lpspec: first | lpspec: steady | linopy: first | linopy: steady | pyomo: first | pyomo: steady | gurobipy-loop: first | gurobipy-loop: steady | gurobipy-matrix: first | gurobipy-matrix: steady |
|---|---|---|---|---|---|---|---|---|---|---|---|
| transport | 9.8k | 41.1 ms | **39.2 ms** | 54.3 ms | 53.0 ms | 43.1 ms | 41.8 ms | 48.5 ms | 42.7 ms | 20.1 ms | 19.2 ms |
| dispatch | 10k | 21.2 ms | **19.8 ms** | 26.7 ms | 24.7 ms | 31.8 ms | 28.2 ms | 24.9 ms | 24.3 ms | 13.9 ms | 13.9 ms |
| storage | 10k | 39.8 ms | **38.7 ms** | 81.9 ms | 80.6 ms | 35.1 ms | 33.5 ms | 46.3 ms | 45.3 ms | 22.0 ms | 19.5 ms |
| fleet | 12k | 79.4 ms | **77.2 ms** | 160.6 ms | 156.1 ms | 34.8 ms | 34.1 ms | 69.8 ms | 66.0 ms | 22.7 ms | 21.5 ms |
| transport | 98k | 50.2 ms | **48.1 ms** | 67.4 ms | 62.1 ms | 735.0 ms | 754.6 ms | 523.3 ms | 514.5 ms | 96.9 ms | 100.1 ms |
| dispatch | 100k | 29.8 ms | **22.8 ms** | 27.0 ms | 25.3 ms | 233.2 ms | 226.1 ms | 200.1 ms | 213.2 ms | 76.0 ms | 74.3 ms |
| storage | 100k | 51.9 ms | **50.7 ms** | 84.2 ms | 83.5 ms | 694.5 ms | 676.4 ms | 526.7 ms | 523.9 ms | 103.7 ms | 105.0 ms |
| fleet | 120k | 85.2 ms | **84.1 ms** | 162.1 ms | 158.1 ms | 834.1 ms | 829.4 ms | 830.6 ms | 839.0 ms | 133.5 ms | 135.0 ms |
| transport | 980k | 153.3 ms | **145.3 ms** | 201.0 ms | 198.3 ms | 6091.8 ms | 6167.3 ms | 4865.6 ms | 4870.9 ms | 893.9 ms | 850.5 ms |
| dispatch | 1M | 75.1 ms | **64.7 ms** | 34.6 ms | 33.2 ms | 4596.7 ms | 4540.0 ms | 2577.7 ms | 2570.8 ms | 751.3 ms | 743.0 ms |
| storage | 1M | 152.5 ms | **151.4 ms** | 104.4 ms | 102.6 ms | 5673.0 ms | 5619.4 ms | 5343.1 ms | 5403.8 ms | 963.0 ms | 974.1 ms |
| fleet | 1.2M | 182.3 ms | **183.1 ms** | 180.9 ms | 180.1 ms | 5975.8 ms | 6029.2 ms | 7946.7 ms | 7795.6 ms | 1433.8 ms | 1409.2 ms |
| transport | 9.8M | 1236.1 ms | **1169.4 ms** | 1445.4 ms | 1456.3 ms | — | — | — | — | 8994.1 ms | 9014.2 ms |
| dispatch | 10M | 528.0 ms | **482.4 ms** | 168.3 ms | 164.8 ms | — | — | 27697.7 ms | 27120.1 ms | 8038.7 ms | 8012.3 ms |
| storage | 10M | 1264.6 ms | **1155.6 ms** | 351.9 ms | 305.4 ms | — | — | — | — | 9706.0 ms | 9743.1 ms |
| fleet | 12M | 1342.0 ms | **1299.5 ms** | 459.3 ms | 447.3 ms | — | — | — | — | 14922.4 ms | 14714.5 ms |

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
