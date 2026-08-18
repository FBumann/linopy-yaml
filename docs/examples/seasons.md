# seasons

A store handed a level at the start of every season and required to leave one at
the end, with seasons of different lengths — every boundary row is named by its
position inside its own season, so no clause depends on which snapshot label
happens to sit at an edge.

## The problem

A store run in blocks needs two rows said about each block: the block's first
snapshot starts from a level it is handed, and the block's last leaves the
reserve it owes. Written against labels, both are claims about *which snapshot*
sits at an edge:

```yaml
where: "snapshot == 1"   # winter opens
where: "snapshot == 7"   # summer's reserve
```

Both are true only of this instance. Renumber the horizon from zero, extend it,
or move a snapshot between seasons, and each clause either lands on the wrong
row or matches nothing — and matching nothing is the quiet failure: the level is
never handed over, the reserve is never required, and the model still solves.

The edges a model means are **positions inside a season**:

$$\mathit{soc}_{t} = \mathit{opening}_{\mathrm{season\_of}(t)} + \mathit{inflow}_{t} - \mathit{release}_{t}
\qquad \forall\thinspace t = \mathrm{index}(\mathcal{T}, 0, \mathrm{season\_of}(t))$$

`index(snapshot, 0, by=season_of)` is the first snapshot **of each season** and
`-1` is each season's last, whatever length each season happens to be. The two
seasons here are four snapshots and three, and nothing in the file says so.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

A store handed a level at the start of every season and required to leave one at the end, with seasons of different lengths — every boundary row is named by its position inside its own season, so no clause depends on which snapshot label happens to sit at an edge.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ --- `snapshot` with $\mathrm{season\_of}: \mathcal{T} \to \mathcal{S}$ --- dispatch periods in order |
| $\mathcal{S}$ | index $s$ --- `season` --- the blocks the store is handed over and handed back |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\mathit{inflow}$ | `inflow` over $\mathcal{T}$ --- energy arriving in a snapshot, whether or not it is wanted then |
| $\mathit{price}$ | `price` over $\mathcal{T}$ --- what one unit of release earns in a snapshot |
| $\mathit{opening}$ | `opening` over $\mathcal{S}$ --- the level the store is handed at the start of a season |
| $\mathit{reserve}$ | `reserve` over $\mathcal{S}$ --- the level a season must leave behind when it ends |

#### Variables

| Symbol | Meaning |
|---|---|
| $\mathit{soc}$ | `soc` over $\mathcal{T}$ --- energy held at the end of a snapshot |
| $\mathit{release}$ | `release` over $\mathcal{T}$ --- energy released in a snapshot |

#### Objective

$$\max \sum_{t \in \mathcal{T}} \mathit{release}_{t} \cdot \mathit{price}_{t}$$

#### Subject to

**`season_opens`**

$$\mathit{soc}_{t} = \mathit{opening}_{\mathrm{season\_of}(t)} + \mathit{inflow}_{t} - \mathit{release}_{t} \qquad \forall\thinspace t \in \mathcal{T} \thinspace:\thinspace t = \mathrm{index}(\mathcal{T}, 0, \mathrm{season\_of}(t))$$

**`season_carries`**

$$\mathit{soc}_{t} = \mathit{soc}_{t - 1} + \mathit{inflow}_{t} - \mathit{release}_{t} \qquad \forall\thinspace t \in \mathcal{T} \thinspace:\thinspace t \neq \mathrm{index}(\mathcal{T}, 0, \mathrm{season\_of}(t))$$

**`season_ends_stocked`**

$$\mathit{soc}_{t} \ge \mathit{reserve}_{\mathrm{season\_of}(t)} \qquad \forall\thinspace t \in \mathcal{T} \thinspace:\thinspace t = \mathrm{index}(\mathcal{T}, -1, \mathrm{season\_of}(t))$$

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
      A store handed a level at the start of every season and required to leave one
      at the end, with seasons of different lengths — every boundary row is named by
      its position inside its own season, so no clause depends on which snapshot
      label happens to sit at an edge.

    dimensions:
      snapshot:
        description: dispatch periods in order
        dtype: int
      season:
        description: the blocks the store is handed over and handed back
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
      opening:
        description: the level the store is handed at the start of a season
        dims: [season]
      reserve:
        description: the level a season must leave behind when it ends
        dims: [season]

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
      season_opens:
        description: >-
          the first snapshot of a season starts from that season's opening level,
          never from what the previous season happened to leave behind
        foreach: [snapshot]
        where: "snapshot == index(snapshot, 0, by=season_of)"
        expression: soc == at(opening, by=season_of) + inflow - release
      season_carries:
        description: every later snapshot of a season carries the one before it
        foreach: [snapshot]
        where: "snapshot != index(snapshot, 0, by=season_of)"
        expression: soc == shift(soc, over=snapshot, offset=1) + inflow - release
      season_ends_stocked:
        description: >-
          and the last snapshot of a season is left holding at least the reserve
          that season owes, whichever snapshot that turns out to be
        foreach: [snapshot]
        where: "snapshot == index(snapshot, -1, by=season_of)"
        expression: soc >= at(reserve, by=season_of)

    objective:
      sense: maximize
      description: revenue from what the store releases
      expression: release * price
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/seasons.yaml', sources) as solution:
        solution.objective  # 160.0
        solution.primal('soc')
    ```

## Why the boundary is a position

The instance deliberately numbers its snapshots from **1**, and the seasons are
of different lengths:

```text
snapshot  1  2  3  4  |  5  6  7
season    winter      |  summer
```

Every clause survives that, and would survive the same horizon numbered from
zero, dated, or cut in half:

| Clause | What it names |
|---|---|
| `snapshot == index(snapshot, 0, by=season_of)` | 1 and 5 — each season's first |
| `snapshot != index(snapshot, 0, by=season_of)` | 2, 3, 4, 6, 7 — everything that carries |
| `snapshot == index(snapshot, -1, by=season_of)` | 4 and 7 — each season's last |

The third row is the one no ungrouped spelling reaches. `index(snapshot, -1)` is
the last snapshot of the *horizon*, so it would put a reserve on summer and none
on winter; there is no single position along the axis that is the last of a
four-snapshot season *and* of a three-snapshot one.

## What the answer looks like

```text
snapshot  season  price  inflow  release  soc
1         winter  1      0       0        20   ← handed 20
2         winter  2      10      0        30
3         winter  5      0       26       4    ← sells all but the reserve, at winter's best price
4         winter  3      0       0        4    ← leaves the 4 it owes
5         summer  4      0       5        0    ← handed 5, sells it at the best price it will see
6         summer  1      6       0        6
7         summer  2      0       5        1    ← leaves the 1 it owes
```

Objective **160.0**. Winter's opening level and summer's are independent — no row
links snapshot 4 to snapshot 5 — so what winter leaves is *its* reserve, not
summer's opening stock. The seasons share an axis and nothing else.

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

A snapshot the column leaves null belongs to no season, so it is no season's
first or last — the same reading a null gets in
[`sum(by=)`](../reference/language/operators.md) — and a season shorter than a
position the model asks for is an error when the data binds, not a boundary that
quietly seeds nothing.

Compare [monthly budget](monthly_budget.md), where such a column groups a *sum*,
and [multi-period](multi_period.md), where it carries a capacity decision down
onto the snapshots it covers. One construct, three jobs.
