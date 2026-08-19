# seasons

A store that cycles inside each season rather than across the horizon, with
seasons of different lengths — one balance row says it, because the wrap is the
season's own and no level is carried from one season into the next.

## The problem

A store that must come back to where it started needs its first position to read
its last. `edge='wrap'` says that about **the axis**:

```yaml
soc == shift(soc, over=snapshot, offset=1, edge='wrap') + inflow - release
```

which on this instance links snapshot 7 to snapshot 1 and makes the whole
horizon one cycle: winter opens holding what summer left, and sells it at
winter's best price. That is a different model, and a plausible-looking one —
it solves, and its objective is *higher*.

The cycle a multi-period model means is per period, and `by=` says which:

```yaml
soc == shift(soc, over=snapshot, offset=1, edge='wrap', by=season_of) + inflow - release
```

$$\mathit{soc}_{t} = \mathit{soc}_{t \ominus_{\mathrm{season\_of}(t)} 1} + \mathit{inflow}_{t} - \mathit{release}_{t}$$

The translation walks inside the group the lookup makes, so a season's first
snapshot reads that season's last, whatever length each season happens to be —
four snapshots and three here, and nothing in the file says so.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

A store that cycles inside each season rather than across the horizon, with seasons of different lengths — one balance row says it, because the wrap is the season's own and no level is carried from one season into the next.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ --- `snapshot` with $\mathrm{season\_of}: \mathcal{T} \to \mathcal{S}$ --- dispatch periods in order |
| $\mathcal{S}$ | index $s$ --- `season` --- the blocks the store cycles over |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\mathit{inflow}$ | `inflow` over $\mathcal{T}$ --- energy arriving in a snapshot, whether or not it is wanted then |
| $\mathit{price}$ | `price` over $\mathcal{T}$ --- what one unit of release earns in a snapshot |

#### Variables

| Symbol | Meaning |
|---|---|
| $\mathit{soc}$ | `soc` over $\mathcal{T}$ --- energy held at the end of a snapshot |
| $\mathit{release}$ | `release` over $\mathcal{T}$ --- energy released in a snapshot |

$t \ominus k$ denotes cyclic translation: index $t-k$ taken modulo the size of the dimension (`roll`). Plain $t-k$ (`shift`) has no wraparound --- terms translated past the edge are simply absent.

#### Objective

$$\max \sum_{t \in \mathcal{T}} \mathit{release}_{t} \cdot \mathit{price}_{t}$$

#### Subject to

**`season_balance`**

$$\mathit{soc}_{t} = \mathit{soc}_{t \ominus_{\mathrm{season\_of}(t)} 1} + \mathit{inflow}_{t} - \mathit{release}_{t} \qquad \forall\thinspace t \in \mathcal{T}$$

#### Variable domains

**`soc`**

$$0 \le \mathit{soc}_{t} \le 60 \qquad \forall\thinspace t \in \mathcal{T}$$

**`release`**

$$\mathit{release}_{t} \ge 0 \qquad \forall\thinspace t \in \mathcal{T}$$

</details>
<!-- math:end -->

The tabs start from [the instance's tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    description: >-
      A store that cycles inside each season rather than across the horizon, with
      seasons of different lengths — one balance row says it, because the wrap is
      the season's own and no level is carried from one season into the next.

    dimensions:
      snapshot:
        description: dispatch periods in order
        dtype: int
      season:
        description: the blocks the store cycles over
        dtype: str

    lookups:
      season_of:
        description: the season a snapshot falls in
        over: snapshot
        into: season

    parameters:
      inflow:
        description: energy arriving in a snapshot, whether or not it is wanted then
        dims: [snapshot]
      price:
        description: what one unit of release earns in a snapshot
        dims: [snapshot]

    variables:
      soc:
        description: energy held at the end of a snapshot
        foreach: [snapshot]
        bounds:
          lower: 0
          upper: 60
      release:
        description: energy released in a snapshot
        foreach: [snapshot]
        bounds:
          lower: 0

    constraints:
      season_balance:
        description: >-
          the level carried into a snapshot is the previous snapshot's, and a
          season's first snapshot carries from that season's own last — so each
          season ends where it began and hands the next one nothing
        foreach: [snapshot]
        expression: soc == shift(soc, over=snapshot, offset=1, edge='wrap', by=season_of) + inflow - release

    objective:
      sense: maximize
      description: revenue from what the store releases
      expression: sum(release * price)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/seasons.yaml', sources) as solution:
        solution.objective  # 74.0
        solution.primal('soc')
    ```

## What the answer looks like

```text
snapshot  season  price  inflow  release  soc
1         winter  1      0       0        0
2         winter  2      10      0        10
3         winter  5      0       10       0    ← winter's inflow, at winter's best price
4         winter  3      0       0        0    ← closes where it opened
5         summer  4      0       6        0    ← sells *before* its inflow arrives
6         summer  1      6       0        6
7         summer  2      0       0        6    ← closes where it opened, three snapshots later
```

Objective **74.0**. Summer is the half worth reading: it opens holding 6, sells
that at the price-4 snapshot before its own inflow has arrived, and the inflow at
snapshot 6 puts the 6 back so the season closes where it opened. Nothing
constrains the level a season starts at except that it must return to it — which
is what a cycle is, and what no clause here has to name.

Written against the axis instead, the same instance gives **80.0**: the extra 6
comes out of winter, which had it to give only because summer's closing level
leaked across the boundary.

## Why it is one row

Per-season cycling is expressible without this operator, as a level each season
begins and ends at: a variable per season, an opening row that reads it, and a
closing row that pins it. Substituting the closing row into the opening one
gives exactly the equation above — three constraints and an auxiliary variable
saying what one `by=` says, and the substitution is the proof they are the same
model.

What the operator adds is that the boundary stops being something the file has to
mention. There is no first snapshot named anywhere, so there is nothing to
re-check when the horizon is renumbered, extended, or cut into different seasons.

## The grouping is data

`season_of` is a column of the snapshot index, so the same file expresses
maintenance campaigns, representative days, market quarters or contract windows.
Move two snapshots between seasons by editing that column and no clause changes:

```text
shape: (7, 2)
┌──────────┬───────────┐
│ snapshot ┆ season_of │
│ ---      ┆ ---       │
│ i64      ┆ str       │
╞══════════╪═══════════╡
│ 1        ┆ winter    │
│ 2        ┆ winter    │
│ 3        ┆ winter    │
│ 4        ┆ winter    │
│ 5        ┆ summer    │
│ 6        ┆ summer    │
│ 7        ┆ summer    │
└──────────┴───────────┘
```

A snapshot the column leaves null belongs to no season, so it reaches nothing and
its row is not built — the same reading a null gets in
[`sum(by=)`](../reference/language/operators.md).

Compare [monthly budget](monthly_budget.md), where such a column groups a *sum*,
and [multi-period](multi_period.md), where it carries a capacity decision down
onto the snapshots it covers. One lookup, three jobs.
