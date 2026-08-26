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
[`latest.json`](https://github.com/fluxopt/lpspec/blob/main/bench/results/latest.json):
the machine, the versions, the commit, and every round of every measurement.
`pixi run table` prints it as one long CSV and commits nothing — the JSON is
the archive because it keeps the rounds. Re-taking rather than reproducing is
`pixi run refresh`, which writes the tables into their fences and the chart's
data literal into its own.

## First model against every model after it

One model built, then built again in the same process.

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

Entity counts x N, snapshots fixed — the same sizes as the ladder, in a
different shape.

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

## Not measured yet

Listed so that a claim with no table under it is visible as one.

- **Solve time.** Every number stops at the hand-off; the simplex is the
  solver's work whoever filled the model.
- **The LP-file round trip.** The tables price writing a file, never reading
  one back.
- **Sizes past `l`.** `xl` and `2xl` exist in the harness and no run
  publishes them.
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
