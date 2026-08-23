# piecewise_lp

A piecewise-linear cost curve stated as **the lines its segments lie on** —
[piecewise](piecewise.md) with one line changed, and the only method that
declares no auxiliary variable at all.

## The problem

The weight forms all say the same thing: put the operating point somewhere on
the curve, and read the cost off beside it. Saying it costs one weight per
breakpoint per row, plus a convexity row to make the weights a combination.

A convex curve does not need any of that, because it **is** the upper envelope
of its own segment lines:

$$f(x) = \max_k \left( f(x_k) + \frac{f(x_{k+1}) - f(x_k)}{x_{k+1} - x_k} \cdot (x - x_k) \right)$$

So `cost` $\ge$ every one of those lines says `cost` $\ge f(p)$, and a
minimising objective pushes it down onto the curve. One row per segment, no
weights, no convexity row, and nothing to declare — which is why the variable
list below is the model's own two and stops there.

The saving is a **trade**, not a free win. Against `method: convex` on 20
generators × 48 snapshots × 6 breakpoints:

| | columns | rows | nonzeros |
|---|---:|---:|---:|
| `method: convex` | 7680 | 2928 | 18240 |
| `method: lp` | **1920** | 6768 | **12480** |

Three quarters of the columns and a third of the nonzeros, paid for in rows.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

The same least-cost dispatch as `piecewise.yaml`, with each generator's cost curve stated as the lines its segments lie on rather than interpolated between its breakpoints. The curve is convex and the objective pushes the cost down, so a cost above every segment line settles on the curve — which needs no interpolation weights, and so declares no auxiliary variable at all.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ — `snapshot` — dispatch periods |
| $\mathcal{G}$ | index $g$ — `generator` — dispatchable units |
| $\mathcal{B}$ | index $b$ — `bp` — breakpoints of the cost curve |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\mathrm{p}^{\mathrm{max}}$ | `p_max` over $\mathcal{G}$ — maximum dispatch |
| $\mathrm{load}$ | `load` over $\mathcal{T}$ — demand to be met |
| $\mathrm{bp\_x}$ | `bp_x` over $\mathcal{G} \times \mathcal{B}$ — breakpoint dispatch levels, one curve per generator |
| $\mathrm{bp\_y}$ | `bp_y` over $\mathcal{G} \times \mathcal{B}$ — cost at each breakpoint, one curve per generator |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{T} \times \mathcal{G}$ — dispatched power |
| $\mathit{op\_cost}$ | `op_cost` over $\mathcal{T} \times \mathcal{G}$ — operating cost, held above every segment of the generator's curve |

Upright is what the model is given — a parameter such as $\mathrm{p}^{\mathrm{max}}$, a coordinate map, a label — and italic is what the solver chooses, such as $p$. An index is italic too, being what a quantifier chooses, and a set is script.

$t \boxminus_{v} k$ denotes translation with $v$ standing where index $t-k$ leaves the dimension (`shift(edge=v)`), so the row at that boundary is built and carries $v$ rather than being dropped.

$\mathrm{pos}(t)$ denotes where index $t$ sits along its dimension's own order — the order `shift` walks, not the order labels sort in — counted from $0$. The index itself stays the coordinate, so $t$ compares against labels and $\mathrm{pos}(t)$ against positions.

$\lvert \mathcal{T} \rvert$ denotes the size of the set being counted along, and a position counted from the end prints against it — $\lvert \mathcal{T} \rvert - 1$ is the last position, one less than the size because the first is $0$.

#### Objective

$$\min \sum_{t \in \mathcal{T},\enspace g \in \mathcal{G}} \mathit{op\_cost}_{t,g}$$

#### Subject to

**`balance`**

$$\sum_{g \in \mathcal{G}} p_{t,g} = \mathrm{load}_{t} \qquad \forall\thinspace t \in \mathcal{T}$$

**`cost_curve_chord`**

$$\mathit{op\_cost}_{t,g} \cdot \left( \mathrm{bp\_x}_{g,b} - \mathrm{bp\_x}_{g,b \boxminus_{0} 1} \right) \ge \left( \mathrm{bp\_y}_{g,b} - \mathrm{bp\_y}_{g,b \boxminus_{0} 1} \right) \cdot \left( p_{t,g} - \mathrm{bp\_x}_{g,b} \right) + \mathrm{bp\_y}_{g,b} \cdot \left( \mathrm{bp\_x}_{g,b} - \mathrm{bp\_x}_{g,b \boxminus_{0} 1} \right) \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G},\enspace b \in \mathcal{B} \thinspace:\thinspace \mathrm{pos}(b) \neq 0$$

**`cost_curve_domain_lo`**

$$p_{t,g} \ge \mathrm{bp\_x}_{g,b} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G},\enspace b \in \mathcal{B} \thinspace:\thinspace \mathrm{pos}(b) = 0$$

**`cost_curve_domain_hi`**

$$p_{t,g} \le \mathrm{bp\_x}_{g,b} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G},\enspace b \in \mathcal{B} \thinspace:\thinspace \mathrm{pos}(b) = \lvert \mathcal{B} \rvert - 1$$

#### Variable domains

**`p`**

$$0 \le p_{t,g} \le \mathrm{p}^{\mathrm{max}}_{g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`op_cost`**

$$\mathit{op\_cost}_{t,g} \ge 0 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

</details>
<!-- math:end -->

```yaml
description: >-
  The same least-cost dispatch as `piecewise.yaml`, with each generator's cost
  curve stated as the lines its segments lie on rather than interpolated
  between its breakpoints. The curve is convex and the objective pushes the
  cost down, so a cost above every segment line settles on the curve — which
  needs no interpolation weights, and so declares no auxiliary variable at all.

dimensions:
  snapshot:
    description: dispatch periods
    dtype: int
  generator:
    description: dispatchable units
    dtype: str
  bp:
    description: breakpoints of the cost curve
    dtype: int

parameters:
  p_max:
    description: maximum dispatch
    dims: [generator]
  load:
    description: demand to be met
    dims: [snapshot]
  bp_x:
    description: breakpoint dispatch levels, one curve per generator
    dims: [generator, bp]
  bp_y:
    description: cost at each breakpoint, one curve per generator
    dims: [generator, bp]

variables:
  p:
    description: dispatched power
    foreach: [snapshot, generator]
    bounds:
      lower: 0
      upper: p_max
  op_cost:
    description: operating cost, held above every segment of the generator's curve
    foreach: [snapshot, generator]
    bounds:
      lower: 0

piecewise:
  cost_curve:
    description: >-
      cost bounded below by the curve — the `>=` is what says which side of the
      lines the cost sits on, and the curvature has to match it: lines that
      envelope a convex curve would cut a concave one, and the solve comes back
      optimal either way
    over: bp
    links:
      - [p, bp_x]
      - [op_cost, bp_y, ">="]
    method: lp

constraints:
  balance:
    foreach: [snapshot]
    expression: sum(p, over=generator) == load

objective:
  sense: minimize
  description: total operating cost, taken off the curves rather than from a marginal rate
  expression: sum(op_cost)
```

## What it exercises

The `>=` on the second link is the whole declaration. It says which side of the
lines the cost sits on, and **the curvature has to match it**: lines that
envelope a convex curve *cut* a concave one, and the solve comes back
`optimal` either way with an answer below the curve it was told to price. So
`>=` requires a convex curve and `<=` a concave one, checked against the
breakpoint values once they are bound — strictly stronger than the mixed-curvature
guard `method: convex` needs, which a wholly concave curve passes.

A segment *line* does not stop where its segment does, so two more rows pin the
operating point inside the curve's own range — `p >= min(bp_x)` and
`p <= max(bp_x)`, at the first and last breakpoint. Without them the
formulation extrapolates along the end segments, where the weight forms cannot
go at all.

Everything above is emitted as ordinary constraints before the plan exists:
there is no plan node for a curve and no engine case, so both lanes receive
identical affine rows and the LP file agrees with them.

| | what it declares | what it emits |
|---|---|---|
| `method: adjacency` | λ per breakpoint, binary per segment | convexity, links, adjacency, pick |
| `method: sos2` | λ per breakpoint | convexity, links, and a set for the sink |
| `method: convex` | λ per breakpoint | convexity and links — a pure LP |
| `method: lp` | **nothing** | a row per segment line, and two holding the domain |

This model and [piecewise](piecewise.md) are the same instance to the number:
both reach 3850.0, and both reach the same shadow prices on `balance`, which is
the check worth having — two formulations of one curve agreeing on the dual as
well as the primal.

Compare [sos](sos.md), the other one-line variant, which moves the restriction
*outward* to the solver where this one removes the need for it.

---

[`examples/piecewise_lp.yaml`](https://github.com/fluxopt/lpspec/blob/main/examples/piecewise_lp.yaml) · back to [all models](index.md)
