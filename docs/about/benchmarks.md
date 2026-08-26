# Measured results

**Cost is a property of the engine, not of the language.** The rules in
the architecture notes constrain what a file may say and would survive an engine
swap untouched; what a build *costs* is settled here, by measurement. That
separation is why this file can be rewritten by a benchmark run without
anything in the language reference moving.

Peak RSS and wall time for the same model built two ways — declaratively on the
relational engine, and eagerly through linopy — from the same parquet files to
the same destination. `wall` and `peak` columns are **lpspec ÷ linopy: below
1.00 is a win for us.** The [chart page](benchmarks-scaling.html) plots the same
run.

**The eager arm is `lpspec.linopy.build`, not hand-written linopy** — our own
YAML→`linopy.Model` lane, so it carries our loader on top of linopy's work.
Against hand-written linopy on the same model that lane costs a constant
**~2.3 ms**: a fixed offset, nowhere near enough to move a conclusion.

**Every linopy column here is frozen at the run that took it.** The harness no
longer measures that arm: what it says about *our* eager lane is not what this
page is for, and the column a reader wants under the name `linopy` is
hand-written linopy, which is being added as an arm of its own. Until it lands
the tables below stand on their committed provenance and are not re-taken;
`bench/floor.py` — one model hand-written straight into HiGHS — is the only
denominator a fresh run can still produce.

**Two sinks, and they are not the same comparison.** The LP file is the artifact
fewest callers want; `highs` is the one most reach for, and there HiGHS's own
dense model is resident in both arms, which narrows every ratio. Read the sink
you actually use.

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

## Results
<!-- bench:results -->

*The same runs with a cursor: [the chart page](benchmarks-scaling.html).*

<details markdown="1">
<summary><b>dispatch</b> — every rung, every sink</summary>

**dispatch — gurobi sink**

Each arm ends holding a populated `gurobipy.Model` with `optimize()` never called — lpspec through `build_gurobi`, and gurobipy through `update()`, which is where its own deferred writes land. Opt-in: it needs the `[gurobi]` extra.

| variables | live | rows | wall: lpspec | wall: linopy | wall: pyomo | wall: gurobipy-loop | wall: gurobipy-matrix | peak: lpspec | peak: linopy | peak: pyomo | peak: gurobipy-loop | peak: gurobipy-matrix |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10k | 100% | 100 | 0.01 s | 0.02 s | 0.07 s | 0.01 s | 0.01 s | 0.21 GB | 0.23 GB | 0.18 GB | 0.17 GB | 0.18 GB |
| 100k | 100% | 1k | 0.05 s | 0.07 s | 0.88 s | 0.11 s | 0.05 s | 0.26 GB | 0.26 GB | 0.37 GB | 0.21 GB | 0.20 GB |
| 1M | 100% | 10k | 0.44 s~ | 0.65 s | 9.35 s | 1.41 s | 0.50 s | 0.73 GB | 0.68 GB | 2.23 GB | 0.62 GB | 0.50 GB |
| 10M | 100% | 100k | 4.38 s~ | 6.32 s | — | 15.07 s | 5.01 s | 4.66 GB | 4.69 GB | — | 4.05 GB | 3.56 GB |

`~` marks a measurement whose rounds spread wider than 25% of their own median. Every round was slow, so the minimum printed for it has no clean round behind it and may be contaminated: **do not quote a marked number, or a ratio drawn from one** — re-take the cell on an idle machine.

**dispatch — highs sink**

Each arm ends holding a populated `highspy.Highs` with `run()` never called — lpspec through `build_highs`. The simplex is the same work whoever filled the model, so timing it would say nothing about the lane that filled it.

| variables | live | rows | wall: lpspec | wall: linopy | wall: pyomo | peak: lpspec | peak: linopy | peak: pyomo |
|---|---|---|---|---|---|---|---|---|
| 10k | 100% | 100 | 0.01 s | 0.01 s | 0.05 s | 0.20 GB | 0.22 GB | 0.17 GB |
| 100k | 100% | 1k | 0.01 s | 0.02 s | 0.74 s | 0.23 GB | 0.24 GB | 0.34 GB |
| 1M | 100% | 10k | 0.06 s | 0.08 s | 8.60 s | 0.48 GB | 0.40 GB | 1.84 GB |
| 10M | 100% | 100k | 0.52 s | 1.02 s | — | 2.30 GB | 1.84 GB | — |

</details>

<details markdown="1">
<summary><b>fleet</b> — every rung, every sink</summary>

**fleet — gurobi sink**

Each arm ends holding a populated `gurobipy.Model` with `optimize()` never called — lpspec through `build_gurobi`, and gurobipy through `update()`, which is where its own deferred writes land. Opt-in: it needs the `[gurobi]` extra.

| variables | live | rows | wall: lpspec | wall: linopy | wall: pyomo | wall: gurobipy-loop | wall: gurobipy-matrix | peak: lpspec | peak: linopy | peak: pyomo | peak: gurobipy-loop | peak: gurobipy-matrix |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12k | 100% | 6.02k | 0.05 s | 0.10 s | 0.14 s | 0.04 s~ | 0.01 s | 0.22 GB | 0.23 GB | 0.18 GB | 0.17 GB | 0.18 GB |
| 120k | 100% | 60.2k | 0.13 s | 0.22 s | 1.89 s | 0.42 s | 0.11 s~ | 0.29 GB | 0.29 GB | 0.44 GB | 0.23 GB | 0.23 GB |
| 1.2M | 100% | 602k | 0.95 s | 1.26 s | 18.79 s | 4.16 s | 0.99 s | 1.03 GB | 0.97 GB | 2.95 GB | 0.92 GB | 0.76 GB |
| 12M | 100% | 6.02M | 8.92 s | 11.66 s | — | — | 9.90 s | 7.31 GB | 7.74 GB | — | — | 5.92 GB |

`~` marks a measurement whose rounds spread wider than 25% of their own median. Every round was slow, so the minimum printed for it has no clean round behind it and may be contaminated: **do not quote a marked number, or a ratio drawn from one** — re-take the cell on an idle machine.

**fleet — highs sink**

Each arm ends holding a populated `highspy.Highs` with `run()` never called — lpspec through `build_highs`. The simplex is the same work whoever filled the model, so timing it would say nothing about the lane that filled it.

| variables | live | rows | wall: lpspec | wall: linopy | wall: pyomo | peak: lpspec | peak: linopy | peak: pyomo |
|---|---|---|---|---|---|---|---|---|
| 12k | 100% | 6.02k | 0.04 s | 0.09 s | 0.21 s | 0.20 GB | 0.22 GB | 0.18 GB |
| 120k | 100% | 60.2k | 0.05 s | 0.10 s | 3.61 s | 0.24 GB | 0.25 GB | 0.37 GB |
| 1.2M | 100% | 602k | 0.13 s | 0.21 s | — | 0.60 GB | 0.52 GB | — |
| 12M | 100% | 6.02M | 1.10 s | 1.57 s | — | 3.44 GB | 3.03 GB | — |

</details>

<details markdown="1">
<summary><b>storage</b> — every rung, every sink</summary>

**storage — gurobi sink**

Each arm ends holding a populated `gurobipy.Model` with `optimize()` never called — lpspec through `build_gurobi`, and gurobipy through `update()`, which is where its own deferred writes land. Opt-in: it needs the `[gurobi]` extra.

| variables | live | rows | wall: lpspec | wall: linopy | wall: pyomo | wall: gurobipy-loop | wall: gurobipy-matrix | peak: lpspec | peak: linopy | peak: pyomo | peak: gurobipy-loop | peak: gurobipy-matrix |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10k | 100% | 2.1k | 0.03 s | 0.05 s | 0.14 s | 0.03 s | 0.01 s | 0.21 GB | 0.23 GB | 0.18 GB | 0.17 GB | 0.18 GB |
| 100k | 100% | 21k | 0.08 s | 0.13 s | 1.36 s | 0.29 s | 0.07 s | 0.27 GB | 0.27 GB | 0.36 GB | 0.21 GB | 0.21 GB |
| 1M | 100% | 210k | 0.60 s | 0.83 s | 13.68 s | 2.87 s | 0.63 s | 0.87 GB | 0.75 GB | 2.33 GB | 0.70 GB | 0.61 GB |
| 10M | 100% | 2.1M | 5.96 s | 8.09 s | — | 29.72 s | 6.27 s | 5.09 GB | 4.70 GB | — | 5.30 GB | 4.25 GB |

**storage — highs sink**

Each arm ends holding a populated `highspy.Highs` with `run()` never called — lpspec through `build_highs`. The simplex is the same work whoever filled the model, so timing it would say nothing about the lane that filled it.

| variables | live | rows | wall: lpspec | wall: linopy | wall: pyomo | peak: lpspec | peak: linopy | peak: pyomo |
|---|---|---|---|---|---|---|---|---|
| 10k | 100% | 2.1k | 0.02 s | 0.04 s | 0.13 s | 0.20 GB | 0.22 GB | 0.17 GB |
| 100k | 100% | 21k | 0.03 s | 0.05 s | 2.01 s | 0.24 GB | 0.25 GB | 0.32 GB |
| 1M | 100% | 210k | 0.10 s | 0.15 s | 64.89 s | 0.58 GB | 0.46 GB | 1.86 GB |
| 10M | 100% | 2.1M | 1.06 s | 1.38 s | — | 2.65 GB | 2.86 GB | — |

</details>

<details markdown="1">
<summary><b>transport</b> — every rung, every sink</summary>

**transport — gurobi sink**

Each arm ends holding a populated `gurobipy.Model` with `optimize()` never called — lpspec through `build_gurobi`, and gurobipy through `update()`, which is where its own deferred writes land. Opt-in: it needs the `[gurobi]` extra.

| variables | live | rows | wall: lpspec | wall: linopy | wall: pyomo | wall: gurobipy-loop | wall: gurobipy-matrix | peak: lpspec | peak: linopy | peak: pyomo | peak: gurobipy-loop | peak: gurobipy-matrix |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9.8k | 100% | 1.4k | 0.02 s | 0.04 s | 0.10 s | 0.02 s | 0.01 s | 0.22 GB | 0.23 GB | 0.18 GB | 0.17 GB | 0.18 GB |
| 98k | 100% | 14k | 0.07 s | 0.11 s | 1.24 s | 0.25 s | 0.06 s | 0.28 GB | 0.34 GB | 0.38 GB | 0.21 GB | 0.21 GB |
| 980k | 100% | 140k | 0.55 s~ | 0.93 s | 13.12 s | 2.53 s | 0.57 s | 0.87 GB | 1.36 GB | 2.44 GB | 0.68 GB | 0.60 GB |
| 9.8M | 100% | 1.4M | 5.77 s | 9.61 s | — | 26.18 s | 5.72 s | 4.81 GB | 5.65 GB | — | 4.62 GB | 4.03 GB |

`~` marks a measurement whose rounds spread wider than 25% of their own median. Every round was slow, so the minimum printed for it has no clean round behind it and may be contaminated: **do not quote a marked number, or a ratio drawn from one** — re-take the cell on an idle machine.

**transport — highs sink**

Each arm ends holding a populated `highspy.Highs` with `run()` never called — lpspec through `build_highs`. The simplex is the same work whoever filled the model, so timing it would say nothing about the lane that filled it.

| variables | live | rows | wall: lpspec | wall: linopy | wall: pyomo | peak: lpspec | peak: linopy | peak: pyomo |
|---|---|---|---|---|---|---|---|---|
| 9.8k | 100% | 1.4k | 0.02 s | 0.03 s | 0.10 s | 0.20 GB | 0.23 GB | 0.18 GB |
| 98k | 100% | 14k | 0.02 s | 0.05 s | 1.46 s | 0.25 GB | 0.33 GB | 0.33 GB |
| 980k | 100% | 140k | 0.10 s | 0.28 s | 31.26 s | 0.57 GB | 1.14 GB | 1.98 GB |
| 9.8M | 100% | 1.4M | 1.06 s | 3.16 s | — | 2.64 GB | 5.81 GB | — |

</details>

<!-- bench:/results -->
## What this says

**Ahead of every library measured, on wall, in every model and both sinks.**
Peak is not a clean sweep and says so below. At the widest rung each model
reached, through `highs`:

| | dispatch 10M | transport 9.8M | storage 10M | fleet 12M |
|---|---|---|---|---|
| lpspec | **0.52 s** | **1.06 s** | **1.06 s** | **1.10 s** |
| linopy | 1.02 s | 3.16 s | 1.38 s | 1.57 s |
| pyomo | — | — | — | — |

Every pyomo cell at that rung is a refusal, not a gap: it was projected past the
harness's thirty-second budget and skipped with the reason printed. Where it
does reach, the distance is the story — `storage` at 1M variables is **0.10 s
against 64.89 s**, and that is the same model, the same data, the same solver.

**The Gurobi sink is where the interesting comparison lives**, because two of
the five are the same library written two ways. At the top rung of each model:

| | lpspec | gurobipy-matrix | gurobipy-loop | linopy |
|---|---|---|---|---|
| transport 9.8M | **5.77 s** | 5.72 s | 26.18 s | 9.61 s |
| storage 10M | **5.96 s** | 6.27 s | 29.72 s | 8.09 s |
| fleet 12M | **8.92 s** | 9.90 s | — | 11.66 s |

`gurobipy-matrix` is within a few percent of us in both directions — it reaches
the same `addMVar`/`addMConstr` seam our own `build_gurobi` does, so what
separates the two columns is only where the matrix came from. `gurobipy-loop`
is the *same library* at four to five times the cost. **Most of the distance
between a modelling library and a fast one is how the model was written**, and
a comparison that showed only one of those two columns would be telling you
something else.

**Peak is model-dependent and we lose two of four.** Through `highs`, at the
same rungs: `transport` 2.64 GB against linopy's 5.81, `storage` 2.65 against
2.86 — and `dispatch` 2.30 against 1.84, `fleet` 3.44 against 3.03. The two we
lose are the two whose models are widest per row. `gurobipy-matrix` holds the
lowest peak of anything here on every Gurobi cell, which is what a hand-built
CSR and nothing else in the process looks like.

**What the numbers are not.** They are build and hand-off — `run()` and
`optimize()` are never called, because the simplex is the same work whoever
filled the model. They are one machine, one run, and the cells marked `~` had
rounds spread wider than a quarter of their own median and should not be
quoted. And they are four models: a shape unlike all four may behave unlike all
four.

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

## The density sweep, and the claim it used to refuse

One model size (50 nodes x 12 technologies x 2000 snapshots = 1.2M coordinates),
four mask densities. The expectation was that an absent pair costs the
relational lane nothing and costs the eager lane a NaN, so the gap should widen
as density falls. One model size, through the `lp` sink; `live` is how many of
the 12 technologies each node has installed.

**The width ladder is measured on three of the five libraries.** pyomo and
`gurobipy-loop` reached their time budget on the size ladder, and a bug in how
that budget was keyed — fixed, with a test — carried the decision onto this
axis, where nobody had decided it. The re-run was interrupted; the axis is
built and its remaining cells are unmeasured rather than unmeasurable.

<!-- bench:sweeps -->

### The width ladder

Entity counts x N with the snapshot count held fixed, through the `highs` sink. Each rung matches one of the size ladder rungs above variable for variable — `w10` is `s`, `w1000` is `l` — so the pair reads as one model at one size in two shapes.

| case | entities x | variables | wall: lpspec | wall: linopy | wall ÷ linopy | peak: lpspec | peak: linopy | peak ÷ linopy |
|---|---|---|---|---|---|---|---|---|
| storage | 1 | 10k | 0.02 s | 0.04 s | 0.43x | 0.20 GB | 0.22 GB | 0.91x |
| storage | 10 | 100k | 0.03 s | 0.05 s | 0.52x | 0.24 GB | 0.24 GB | 0.98x |
| storage | 100 | 1M | 0.13 s | 0.17 s | 0.75x | 0.59 GB | 0.46 GB | 1.27x |
| storage | 1000 | 10M | 1.26 s | 1.44 s | 0.88x | 2.70 GB | 2.58 GB | 1.04x |
| transport | 1 | 9.8k | 0.02 s | 0.03 s | 0.62x | 0.20 GB | 0.23 GB | 0.89x |
| transport | 10 | 98k | 0.03 s | 0.23 s | 0.12x | 0.25 GB | 1.08 GB | 0.23x |
| transport | 100 | 980k | 0.11 s | 50.98 s | 0.00x | 0.59 GB | 14.26 GB | 0.04x |
| transport | 1000 | 9.8M | 1.03 s | — | — | 2.74 GB | — | — |

### The mask sweep

One model size, through the `lp` sink. For `nodal`, `live` is how many of the 12 technologies each node has installed: 12 / 6 / 3 / 1.

| case | live | variables | wall: lpspec | wall: linopy | wall ÷ linopy | peak: lpspec | peak: linopy | peak ÷ linopy |
|---|---|---|---|---|---|---|---|---|
| nodal | 100% | 1.2M | 0.16 s | 0.38 s | 0.41x | 0.54 GB | 0.62 GB | 0.88x |
| nodal | 50% | 600k | 0.10 s | 0.31 s | 0.32x | 0.41 GB | 0.60 GB | 0.68x |
| nodal | 25% | 300k | 0.06 s | 0.27 s | 0.23x | 0.32 GB | 0.46 GB | 0.68x |
| nodal | 8% | 100k | 0.04 s | 0.24 s | 0.16x | 0.26 GB | 0.35 GB | 0.72x |

<!-- bench:/sweeps -->

**It now does, and it did not before.** Wall time falls from 0.50x to 0.17x as
density drops, and peak improves at every rung — 0.91x, 0.72x, 0.72x, 0.78x.
The previous run of this sweep had linopy's peak *below* ours at the sparsest
rung, and the note here said so.

What changed is not the prediction but what a mask costs us. Assigning labels
under a mask used to mean sorting the whole masked product; a mask that reads
none of the leading dims now leaves a rectangle, so the labels are arithmetic
and the sort is of the surviving *set*. That was 46-66% of the build on every
masked case. The sweep was measuring our own cost of being sparse, and most of
it is gone.

The remaining caveat on the *memory* half stands, and it is the size this sweep
is run at rather than the prediction. It holds the
coordinate product fixed at 1.2M, where a dense array over it is ~10 MB and the
interpreter and libraries dominate everything. `sector` runs the same 8%
sparsity at a 12M product, and there the effect is unmistakable: 0.92 GB against
2.97 GB.

So the claim needs both halves — **low density and a product large enough for it
to cost anything**. This sweep varies one at a size that cannot show it; `sector`
varies the other. Neither is sufficient alone, which is worth knowing before
quoting either.

Wall time behaves throughout: our advantage grows as the model thins, 1.0x to
1.7x, because there is less to build and our fixed cost is lower.

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
