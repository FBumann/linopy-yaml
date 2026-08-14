# sos

A piecewise-linear cost curve stated as a **special-ordered set** — the same
restriction [piecewise](piecewise.md) writes with binaries, handed to the
solver as a set it branches on itself.

## The problem

A convex-combination curve needs one thing said about its weights: at most two
may be nonzero, and they must be **neighbours**. Otherwise the weights mix
distant breakpoints and the model prices the chord under the curve instead of
the curve:

$$p = \sum_k \lambda_k x_k, \quad
\mathrm{cost} = \sum_k \lambda_k y_k, \quad
\sum_k \lambda_k = 1, \quad
\lambda \in \mathrm{SOS2}$$

There are two ways to say the last line and the language now has both.
`piecewise:` (§4) *builds* it: a binary per segment, an adjacency row per
breakpoint, and one more row picking a segment. `sos:` (§4.1) *declares* it and
leaves the formulation to the sink — which is the point, because a solver that
knows what SOS2 means branches on the set directly rather than searching the
binaries someone wrote for it.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ --- `snapshot` |
| $\mathcal{G}$ | index $g$ --- `generator` |
| $\mathcal{B}$ | index $b$ --- `bp` |

#### Parameters

| Symbol | Meaning |
|---|---|
| $p^{\mathrm{max}}$ | `p_max` over $\mathcal{G}$ |
| $\mathit{load}$ | `load` over $\mathcal{T}$ |
| $\mathit{bp}^{\mathrm{x}}$ | `bp_x` over $\mathcal{G} \times \mathcal{B}$ |
| $\mathit{bp}^{\mathrm{y}}$ | `bp_y` over $\mathcal{G} \times \mathcal{B}$ |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{T} \times \mathcal{G}$ |
| $\mathit{op\_cost}$ | `op_cost` over $\mathcal{T} \times \mathcal{G}$ |
| $\mathit{lam}$ | `lam` over $\mathcal{T} \times \mathcal{G} \times \mathcal{B}$ |

#### Objective

$$\min \sum_{t \in \mathcal{T},\enspace g \in \mathcal{G}} \mathit{op\_cost}_{t,g}$$

#### Subject to

**`convexity`**

$$\sum_{b \in \mathcal{B}} \mathit{lam}_{t,g,b} = 1 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`dispatch`**

$$p_{t,g} = \sum_{b \in \mathcal{B}} \mathit{lam}_{t,g,b} \cdot \mathit{bp}^{\mathrm{x}}_{g,b} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`cost`**

$$\mathit{op\_cost}_{t,g} = \sum_{b \in \mathcal{B}} \mathit{lam}_{t,g,b} \cdot \mathit{bp}^{\mathrm{y}}_{g,b} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`balance`**

$$\sum_{g \in \mathcal{G}} p_{t,g} = \mathit{load}_{t} \qquad \forall\thinspace t \in \mathcal{T}$$

#### Variable domains

**`p`**

$$0 \le p_{t,g} \le p^{\mathrm{max}}_{g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`op_cost`**

$$\mathit{op\_cost}_{t,g} \ge 0 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`lam`**

$$0 \le \mathit{lam}_{t,g,b} \le 1 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G},\enspace b \in \mathcal{B}$$

**`lam sos`**

$$\left( \mathit{lam}_{t,g,b} \right)_{b \in \mathcal{B}} \in \mathrm{SOS}2 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

</details>
<!-- math:end -->

```yaml
dimensions:
  snapshot:
    dtype: int
  generator:
    dtype: str
  bp:
    dtype: int

parameters:
  p_max:
    dims: [generator]
  load:
    dims: [snapshot]
  bp_x:
    dims: [generator, bp]  # per-generator breakpoint positions
  bp_y:
    dims: [generator, bp]  # per-generator cost at each breakpoint

variables:
  p:
    foreach: [snapshot, generator]
    bounds:
      lower: 0
      upper: p_max
  op_cost:
    foreach: [snapshot, generator]
    bounds:
      lower: 0
  lam:  # convex-combination weight on each breakpoint
    foreach: [snapshot, generator, bp]
    bounds:
      lower: 0
      upper: 1

sos:
  on_one_segment:  # at most two neighbouring weights — the curve, not its hull
    variable: lam
    over: bp
    type: 2

constraints:
  convexity:
    foreach: [snapshot, generator]
    expression: sum(lam, over=bp) == 1
  dispatch:
    foreach: [snapshot, generator]
    expression: p == sum(lam * bp_x, over=bp)
  cost:
    foreach: [snapshot, generator]
    expression: op_cost == sum(lam * bp_y, over=bp)
  balance:
    foreach: [snapshot]
    expression: sum(p, over=generator) == load

objective:
  sense: minimize
  expression: op_cost
```

## What it exercises

`sos:` is the one declaration that adds neither a column nor a row. It names
columns the variable already made and says which of them may be nonzero
together, so it leaves the engine as a **fifth stream** beside `cols`, `obj`,
`rows` and the matrix.

That stream is also the one a sink may not be able to take, which is what makes
this model worth reading beside `piecewise`:

| | what it does with this model |
|---|---|
| `gurobi` | `addSOS` — branches on the set, no binaries in the model at all |
| `lp_file` | an `sos` section, read by any solver whose parser has one |
| `highs` | **no SOS concept** — the set arrives reformulated, as a binary per segment and a linking row per member |

So the same file runs everywhere, and what differs is the *search*, not the
answer. On HiGHS the reformulation is very nearly what `piecewise:` would have
emitted, which is the honest summary of what a capability gap costs here: a
worse relaxation, never a refusal. Two conditions come with it — every member
needs a finite upper bound (`lam` has one), and the result is mixed-integer, so
an otherwise-continuous model gives up its duals.

Compare [piecewise](piecewise.md), which reaches the same feasible set through
a declaration that expands *before* the plan exists: by the time the model is
built there is nothing left called piecewise, where a set is still a set right
up to the sink that takes it.

---

[`examples/sos.yaml`](https://github.com/fluxopt/lpspec/blob/main/examples/sos.yaml) · back to [all models](index.md)
