# multi_period

Capacity decided once per investment period, binding at every snapshot inside
it — and the periods need not be the same size.

## The problem

$$p_{t,g} \quad\le\quad \hat p_{\thinspace\mathrm{period}(t),\thinspace g}$$

Two dimensions cannot state this at the resolution a real study wants.
`period × snapshot` is a **rectangle**, so every period gets the same number of
snapshots — and a study that models 2030 hourly and 2050 in four-hour blocks is
asking for exactly the opposite.

So `snapshot` is one flat dimension carrying $\mathrm{period}$ as a
[coordinate](../SPEC.md#2-declarations), the same way `generator` carries
$\mathrm{bus}$ in [transport](transport.md). Ragged periods then cost nothing:
a coordinate is a per-row column, and four snapshots in 2030 beside two in 2050
is just a column with four of one value and two of another.

## Why this needed a new primitive

The flat index gives the grouping direction for free —
`group_sum(p, over=snapshot, by=period)` is a per-period CO₂ budget, and
[monthly_budget](monthly_budget.md) is the same construct on a different
coordinate.

`within_cap` needs the **other** direction. Capacity lives on `period` and has
to be read at each `snapshot`, so a coarse quantity is being pulled onto a fine
one:

```yaml
within_cap:
  foreach: [snapshot, generator]
  expression: p <= at(p_nom, over=snapshot, by=period)
```

A per-period *parameter* never needed this — data prep can join it onto the
snapshot index before the model sees it. **A variable cannot be pre-joined**,
and `p_nom` is a decision. Before `at`, this model had no formulation at all:
the rectangle could not express ragged time, and the flat index could not
express the capacity bound.

`at` and `group_sum` are one mapping table read in the two directions, which is
why they take the same two arguments and differ only in the verb.

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ --- `snapshot` with $\mathrm{period}: \mathcal{T} \to \mathcal{E}$ |
| $\mathcal{E}$ | index $e$ --- `period` |
| $\mathcal{G}$ | index $g$ --- `generator` |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\mathit{load}$ | `load` over $\mathcal{T}$ |
| $\mathit{weight}$ | `weight` over $\mathcal{T}$ |
| $\mathit{opex}$ | `opex` over $\mathcal{G}$ |
| $\mathit{capex}$ | `capex` over $\mathcal{G} \times \mathcal{E}$ |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{T} \times \mathcal{G}$ |
| $p^{\mathrm{nom}}$ | `p_nom` over $\mathcal{E} \times \mathcal{G}$ |

#### Objective

**`total_cost`**

$$\min \sum_{t \in \mathcal{T},\enspace e \in \mathcal{E},\enspace g \in \mathcal{G}} \left( p_{t,g} \cdot \mathit{opex}_{g} \cdot \mathit{weight}_{t} + p^{\mathrm{nom}}_{e,g} \cdot \mathit{capex}_{g,e} \right)$$

#### Subject to

**`within_cap`**

$$p_{t,g} \le p^{\mathrm{nom}}_{\mathrm{period}(t),g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`balance`**

$$\sum_{g \in \mathcal{G}} p_{t,g} = \mathit{load}_{t} \qquad \forall\thinspace t \in \mathcal{T}$$

#### Variable domains

**`p`**

$$p_{t,g} \ge 0 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`p_nom`**

$$0 \le p^{\mathrm{nom}}_{e,g} \le 100 \qquad \forall\thinspace e \in \mathcal{E},\enspace g \in \mathcal{G}$$

</details>
<!-- math:end -->

```yaml
# Multi-period investment on a *flat* snapshot index.
#
# `snapshot` is one dimension carrying `period` as a coordinate, rather than a
# `period x snapshot` rectangle. That is what lets the periods be **ragged** —
# four snapshots in 2030 and two in 2050 here — because a coordinate is a
# per-row column and nothing assumes every period has the same shape. A
# rectangle cannot say it: it forces one resolution on every period.
#
# The line that needs `at()` is `within_cap`. Capacity is decided per period and
# binds at every snapshot in that period, so a coarse-dim *variable* has to be
# read at a fine-dim row. A per-period *parameter* could be pre-joined in data
# prep; a variable cannot, which is why this model had no formulation before.
dimensions:
  snapshot:
    dtype: int
    coords: [period]
  period:
    dtype: int
  generator:
    dtype: str

parameters:
  load:
    dims: [snapshot]
  # what one snapshot stands for: a 2050 snapshot represents four hours, so the
  # operating cost of a coarse period is not understated against a fine one
  weight:
    dims: [snapshot]
  opex:
    dims: [generator]
  capex:
    dims: [generator, period]

variables:
  p:
    foreach: [snapshot, generator]
    bounds:
      lower: 0
  p_nom:
    foreach: [period, generator]
    bounds:
      lower: 0
      upper: 100

constraints:
  within_cap:
    foreach: [snapshot, generator]
    expression: p <= at(p_nom, over=snapshot, by=period)
  balance:
    foreach: [snapshot]
    expression: sum(p, over=generator) == load

objectives:
  total_cost:
    sense: minimize
    expression: p * opex * weight + p_nom * capex
```

## Reading the answer

Costs are chosen so each period picks a different technology, which is what
makes the per-period capacity visible rather than incidental:

| period | wind | gas |
|---|---|---|
| 2030 | 20 | 10 |
| 2050 | 60 | 0 |

2030 peaks at 30 and splits the build — wind is dearer to install but free to
run. 2050 peaks at 60 with every snapshot weighted four times, so the operating
term dominates and the whole build goes to wind. Objective **750.0**, agreed
integer for integer by both lanes.

The weights are the reason the two periods are comparable at all: a coarse
snapshot standing for four hours contributes four hours of operating cost, so a
period is not made cheap by being modelled coarsely.
