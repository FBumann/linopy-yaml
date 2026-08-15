# OSeMOSYS — UTOPIA

What to build and how hard to run it, 1990–2010, to meet three end-use demands at least discounted cost.

> **✔ Verified against OSeMOSYS** — objective **29446.86269**, matched to `rtol=1e-09`. Asserted upstream in `tests/test_gnu_mathprog.py`, and re-run here directly on GLPK.

**The only optimum in this corpus that comes from outside Python.** UTOPIA is
the reference system bundled with MARKAL and the case OSeMOSYS validates itself
against. Its model is GNU MathProg, its solver is GLPK, and neither shares a
line of code, a data model or a language family with anything here — which is
the strongest independence a port can have.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

OSeMOSYS's UTOPIA: what to build and how hard to run it, 1990-2010, to meet three end-use demands at least discounted cost. The reference system bundled with MARKAL. Discounting, the annuity, salvage value and the operational-life window are arithmetic over years, so they are folded into coefficients before the model is built; what is left is the decision. Optimum 29446.86269, from OSeMOSYS itself under GLPK.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ --- `technology` --- the plants and processes that may be built and run |
| $\mathcal{F}$ | index $f$ --- `fuel` --- energy carriers, produced by one technology and consumed by another |
| $\mathcal{I}$ | index $i$ --- `timeslice` --- the slices a year is dispatched over |
| $\mathcal{M}$ | index $m$ --- `mode` --- the way a technology is being operated |
| $\mathcal{Y}$ | index $y$ --- `year` --- the years the pathway covers |
| $\mathcal{V}$ | index $v$ --- `vintage` --- a second axis over the same years — capacity standing in a year was built in some vintage |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\mathit{still\_live}$ | `still_live` over $\mathcal{T} \times \mathcal{Y} \times \mathcal{V}$ --- 1 where a vintage is still inside its technology's operational life in that year |
| $\mathit{residual\_capacity}$ | `residual_capacity` over $\mathcal{T} \times \mathcal{Y}$ --- capacity that already stood in 1990 and has not yet retired |
| $\mathit{build}^{\mathrm{cost}}$ | `build_cost` over $\mathcal{T} \times \mathcal{V}$ --- discounted cost of building a unit of capacity in a vintage |
| $\mathit{holding\_cost}$ | `holding_cost` over $\mathcal{T} \times \mathcal{Y}$ --- discounted fixed cost of holding a unit of capacity through a year |
| $\mathit{running\_cost}$ | `running_cost` over $\mathcal{I} \times \mathcal{T} \times \mathcal{M} \times \mathcal{Y}$ --- discounted variable cost of a unit of activity |
| $\mathit{year}^{\mathrm{split}}$ | `year_split` over $\mathcal{I} \times \mathcal{Y}$ --- share of the year a timeslice stands for |
| $\mathit{capacity\_available}$ | `capacity_available` over $\mathcal{T} \times \mathcal{I} \times \mathcal{Y}$ --- share of its capacity a technology can offer in a timeslice |
| $\mathit{input\_ratio}$ | `input_ratio` over $\mathcal{T} \times \mathcal{F} \times \mathcal{M} \times \mathcal{Y}$ --- fuel a technology consumes per unit of activity |
| $\mathit{output\_ratio}$ | `output_ratio` over $\mathcal{T} \times \mathcal{F} \times \mathcal{M} \times \mathcal{Y}$ --- fuel a technology produces per unit of activity |
| $\mathit{sliced\_demand}$ | `sliced_demand` over $\mathcal{F} \times \mathcal{I} \times \mathcal{Y}$ --- demand for a fuel placed on one timeslice |
| $\mathit{annual\_demand}$ | `annual_demand` over $\mathcal{F} \times \mathcal{Y}$ --- demand for a fuel placed on the year as a whole |
| $\mathit{max\_capacity}$ | `max_capacity` over $\mathcal{T} \times \mathcal{Y}$ --- most capacity a technology may stand at |
| $\mathit{min\_capacity}$ | `min_capacity` over $\mathcal{T} \times \mathcal{Y}$ --- least capacity a technology must stand at |
| $\mathit{reserve\_margin}$ | `reserve_margin` over $\mathcal{Y}$ --- how far firm capacity must exceed the demand of the moment |
| $\mathit{reserve\_tagged}$ | `reserve_tagged` over $\mathcal{T} \times \mathcal{Y}$ --- 1 where a technology's capacity counts towards the reserve |
| $\mathit{reserve\_demand}$ | `reserve_demand` over $\mathcal{T} \times \mathcal{F} \times \mathcal{M} \times \mathcal{Y}$ --- the activity the reserve margin is measured against |
| $\mathit{residual\_holding}$ | `residual_holding` (scalar) --- fixed operating cost owed on the capacity that already stood in 1990 |

#### Variables

| Symbol | Meaning |
|---|---|
| $\mathit{activity}$ | `activity` over $\mathcal{I} \times \mathcal{T} \times \mathcal{M} \times \mathcal{Y}$ --- how hard a technology runs, per timeslice and mode |
| $\mathit{build}$ | `build` over $\mathcal{T} \times \mathcal{V}$ --- how much capacity is built, and when |

#### Objective

$$\min \sum_{t \in \mathcal{T},\enspace i \in \mathcal{I},\enspace m \in \mathcal{M},\enspace y \in \mathcal{Y},\enspace v \in \mathcal{V}} \left( \mathit{build}_{t,v} \cdot \mathit{build}^{\mathrm{cost}}_{t,v} + \left( \sum_{v \in \mathcal{V}} \mathit{build}_{t,v} \cdot \mathit{still\_live}_{t,y,v} \right) \cdot \mathit{holding\_cost}_{t,y} + \mathit{activity}_{i,t,m,y} \cdot \mathit{running\_cost}_{i,t,m,y} + \mathit{residual\_holding} \right)$$

#### Subject to

**`within_capacity`**

$$\sum_{m \in \mathcal{M}} \mathit{activity}_{i,t,m,y} \le \left( \sum_{v \in \mathcal{V}} \mathit{build}_{t,v} \cdot \mathit{still\_live}_{t,y,v} + \mathit{residual\_capacity}_{t,y} \right) \cdot \mathit{capacity\_available}_{t,i,y} \qquad \forall\thinspace i \in \mathcal{I},\enspace t \in \mathcal{T},\enspace y \in \mathcal{Y}$$

**`fuel_balance`**

$$\left( \sum_{t \in \mathcal{T}} \sum_{m \in \mathcal{M}} \mathit{activity}_{i,t,m,y} \cdot \mathit{output\_ratio}_{t,f,m,y} \right) \cdot \mathit{year}^{\mathrm{split}}_{i,y} \ge \mathit{sliced\_demand}_{f,i,y} + \left( \sum_{t \in \mathcal{T}} \sum_{m \in \mathcal{M}} \mathit{activity}_{i,t,m,y} \cdot \mathit{input\_ratio}_{t,f,m,y} \right) \cdot \mathit{year}^{\mathrm{split}}_{i,y} \qquad \forall\thinspace i \in \mathcal{I},\enspace f \in \mathcal{F},\enspace y \in \mathcal{Y}$$

**`annual_balance`**

$$\sum_{i \in \mathcal{I}} \sum_{t \in \mathcal{T}} \sum_{m \in \mathcal{M}} \mathit{activity}_{i,t,m,y} \cdot \mathit{output\_ratio}_{t,f,m,y} \cdot \mathit{year}^{\mathrm{split}}_{i,y} \ge \mathit{annual\_demand}_{f,y} + \sum_{i \in \mathcal{I}} \sum_{t \in \mathcal{T}} \sum_{m \in \mathcal{M}} \mathit{activity}_{i,t,m,y} \cdot \mathit{input\_ratio}_{t,f,m,y} \cdot \mathit{year}^{\mathrm{split}}_{i,y} \qquad \forall\thinspace f \in \mathcal{F},\enspace y \in \mathcal{Y}$$

**`capacity_ceiling`**

$$\sum_{v \in \mathcal{V}} \mathit{build}_{t,v} \cdot \mathit{still\_live}_{t,y,v} + \mathit{residual\_capacity}_{t,y} \le \mathit{max\_capacity}_{t,y} \qquad \forall\thinspace t \in \mathcal{T},\enspace y \in \mathcal{Y}$$

**`capacity_floor`**

$$\sum_{v \in \mathcal{V}} \mathit{build}_{t,v} \cdot \mathit{still\_live}_{t,y,v} + \mathit{residual\_capacity}_{t,y} \ge \mathit{min\_capacity}_{t,y} \qquad \forall\thinspace t \in \mathcal{T},\enspace y \in \mathcal{Y}$$

**`reserve`**

$$\left( \sum_{t \in \mathcal{T}} \sum_{f \in \mathcal{F}} \sum_{m \in \mathcal{M}} \mathit{activity}_{i,t,m,y} \cdot \mathit{reserve\_demand}_{t,f,m,y} \right) \cdot \mathit{reserve\_margin}_{y} \le \sum_{t \in \mathcal{T}} \left( \sum_{v \in \mathcal{V}} \mathit{build}_{t,v} \cdot \mathit{still\_live}_{t,y,v} + \mathit{residual\_capacity}_{t,y} \right) \cdot \mathit{reserve\_tagged}_{t,y} \qquad \forall\thinspace i \in \mathcal{I},\enspace y \in \mathcal{Y}$$

#### Variable domains

**`activity`**

$$\mathit{activity}_{i,t,m,y} \ge 0 \qquad \forall\thinspace i \in \mathcal{I},\enspace t \in \mathcal{T},\enspace m \in \mathcal{M},\enspace y \in \mathcal{Y}$$

**`build`**

$$\mathit{build}_{t,v} \ge 0 \qquad \forall\thinspace t \in \mathcal{T},\enspace v \in \mathcal{V}$$

</details>
<!-- math:end -->

The tabs start from [the instance’s tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    description: >-
      OSeMOSYS's UTOPIA: what to build and how hard to run it, 1990-2010, to meet
      three end-use demands at least discounted cost. The reference system bundled
      with MARKAL. Discounting, the annuity, salvage value and the operational-life
      window are arithmetic over years, so they are folded into coefficients before
      the model is built; what is left is the decision. Optimum 29446.86269, from
      OSeMOSYS itself under GLPK.

    dimensions:
      technology:
        description: the plants and processes that may be built and run
        dtype: str
      fuel:
        description: energy carriers, produced by one technology and consumed by another
        dtype: str
      timeslice:
        description: the slices a year is dispatched over
        dtype: str
      mode:
        description: the way a technology is being operated
        dtype: int
      year:
        description: the years the pathway covers
        dtype: int
      vintage:
        description: >-
          a second axis over the same years — capacity standing in a year was built
          in some vintage
        dtype: int

    parameters:
      still_live:
        description: 1 where a vintage is still inside its technology's operational life in that year
        dims: [technology, year, vintage]
      residual_capacity:
        description: capacity that already stood in 1990 and has not yet retired
        dims: [technology, year]

      build_cost:
        description: discounted cost of building a unit of capacity in a vintage
        dims: [technology, vintage]
      holding_cost:
        description: discounted fixed cost of holding a unit of capacity through a year
        dims: [technology, year]
      running_cost:
        description: discounted variable cost of a unit of activity
        dims: [timeslice, technology, mode, year]

      year_split:
        description: share of the year a timeslice stands for
        dims: [timeslice, year]
      capacity_available:
        description: share of its capacity a technology can offer in a timeslice
        dims: [technology, timeslice, year]
      input_ratio:
        description: fuel a technology consumes per unit of activity
        dims: [technology, fuel, mode, year]
      output_ratio:
        description: fuel a technology produces per unit of activity
        dims: [technology, fuel, mode, year]

      sliced_demand:
        description: demand for a fuel placed on one timeslice
        dims: [fuel, timeslice, year]
      annual_demand:
        description: demand for a fuel placed on the year as a whole
        dims: [fuel, year]

      max_capacity:
        description: most capacity a technology may stand at
        dims: [technology, year]
      min_capacity:
        description: least capacity a technology must stand at
        dims: [technology, year]

      reserve_margin:
        description: how far firm capacity must exceed the demand of the moment
        dims: [year]
      reserve_tagged:
        description: 1 where a technology's capacity counts towards the reserve
        dims: [technology, year]
      reserve_demand:
        description: the activity the reserve margin is measured against
        dims: [technology, fuel, mode, year]

      residual_holding:
        description: fixed operating cost owed on the capacity that already stood in 1990
        dims: []

    variables:
      activity:
        description: how hard a technology runs, per timeslice and mode
        foreach: [timeslice, technology, mode, year]
        bounds:
          lower: 0
      build:
        description: how much capacity is built, and when
        foreach: [technology, vintage]
        bounds:
          lower: 0

    expressions:
      built_capacity:
        expression: sum(build * still_live, over=vintage)
        description: >-
          capacity standing in a year from every vintage still inside its life. A
          plant's life is read from data and differs by technology, so the window
          cannot be a fixed shift — it is an incidence table, the shape the KVL port
          uses for a cycle basis.
      capacity:
        expression: built_capacity + residual_capacity
        description: all the capacity standing in a year, including what was already there in 1990

    constraints:
      within_capacity:
        description: a technology cannot run beyond the capacity standing that year
        foreach: [timeslice, technology, year]
        expression: sum(activity, over=mode) <= capacity * capacity_available

      fuel_balance:
        description: >-
          every fuel balances in every timeslice — what is produced covers the
          demand placed on it plus what other technologies consume
        foreach: [timeslice, fuel, year]
        expression: >-
          sum(sum(activity * output_ratio, over=mode), over=technology) * year_split
          >= sliced_demand
          + sum(sum(activity * input_ratio, over=mode), over=technology) * year_split

      annual_balance:
        description: and balances again over the year, for demands that are not sliced
        foreach: [fuel, year]
        expression: >-
          sum(sum(sum(activity * output_ratio * year_split, over=mode), over=technology), over=timeslice)
          >= annual_demand
          + sum(sum(sum(activity * input_ratio * year_split, over=mode), over=technology), over=timeslice)

      capacity_ceiling:
        description: a technology stands at no more capacity than it is allowed
        foreach: [technology, year]
        expression: capacity <= max_capacity

      capacity_floor:
        description: a technology stands at no less capacity than it is required to
        foreach: [technology, year]
        expression: capacity >= min_capacity

      reserve:
        description: firm capacity exceeds the electricity demand of the moment by the reserve margin
        foreach: [timeslice, year]
        expression: >-
          sum(sum(sum(activity * reserve_demand, over=mode), over=fuel), over=technology) * reserve_margin
          <= sum(capacity * reserve_tagged, over=technology)

    objective:
      sense: minimize
      description: discounted cost of building, holding and running the system over the pathway
      expression: >-
        build * build_cost
        + built_capacity * holding_cost
        + activity * running_cost
        + residual_holding
    ```

## What the port had to decide

**An operational life is a window read from data, and that is an incidence
table.** Capacity standing in a year is every vintage still inside its
technology's life — and the lives differ, so this is not a fixed `shift`. The
port carries `still_live[technology, year, vintage]`, one row per pair that is
still live, and the standing capacity is a contraction against it. That is the
same shape [`pypsa_kvl`](pypsa_kvl.md) uses for a cycle basis, and building it
is arithmetic over years, which is where
[the ceiling](../about/ceiling.md) puts it.

**Discounting, the annuity and salvage value never reach the model.** OSeMOSYS
spends four parameters and four constraint families on them —
`CapitalRecoveryFactor`, `PvAnnuity`, `DiscountFactor`, and `SV1`–`SV4` with
three depreciation cases. Every one is a function of the year and the
technology, so all of it folds into a single coefficient per `(technology,
vintage)`. What survives into the model is the decision: how much to build, and
when.

The one piece that cannot fold is the fixed cost owed on capacity that already
stood in 1990 — it is owed whatever the model chooses, so it enters the
objective as a constant.

## Same answer, a twenty-third of the model

| | rows | columns |
|---|---|---|
| OSeMOSYS, as generated by GLPK | 119,273 | 147,171 |
| this port | 5,124 | 5,733 |

Not a fair fight, and worth being precise about why: OSeMOSYS's long
formulation defines an intermediate *variable* for every accounting quantity —
`RateOfProductionByTechnologyByMode` alone is one per region × timeslice ×
technology × mode × fuel × year — and ties each to its definition with an
equality. Those are not decisions; they are names for expressions. Substituting
them is what any modeller does by hand when the formulation is not being
generated, and it is what writing the model as expressions does automatically.

The point is not that 5,124 beats 119,273. It is that **both reach
29446.86269**, so the substitution is exact, and the smaller model is the one a
reader can hold.

## What this port does not carry

**Storage.** UTOPIA declares a reservoir and OSeMOSYS carries fifteen
constraints for it, but the instance builds none: `NewStorageCapacity` is empty
in the reference solution, as is `Trade`. Their constraints are satisfied at
zero, so the port omits them — and the optimum agreeing to ten digits is what
says the omission was safe.

That has a consequence worth recording, because it was the reason this model
was first proposed for the corpus. `Conversionls`, `Conversionld` and
`Conversionlh` — the three maps from a timeslice to its season, day type and
daily time bracket — appear **only** in those storage constraints. With storage
inert they read nothing, so this port cannot be evidence about grouping one
axis several ways. A model that needs it is still wanted.

## What it exercises

A window read from data as an incidence table; a second axis over the same
years joined to the first; and a cost chain that is entirely data preparation.
No construct here is new — which, for a model of this size from a stack this
far away, is the result.
