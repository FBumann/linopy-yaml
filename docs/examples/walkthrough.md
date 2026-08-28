# walkthrough

The dispatch model plus a macro and a named expression — the one used to print every pipeline stage.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

The dispatch model of README.md, plus one macro and one named expression — small enough to print in full, complete enough that every pipeline stage in examples/walkthrough.py has something to show.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{S}$ | index $s$ — `snapshot` — dispatch periods |
| $\mathcal{G}$ | index $g$ — `generator` — generating units, including oil, which is retired and gets no columns at all |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\bar p$ | `p_max` over $\mathcal{G}$ — installed capacity, zero for a retired unit |
| $\ell$ | `load` over $\mathcal{S}$ — demand to be met |
| $c$ | `cost` over $\mathcal{G}$ — marginal cost |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{S} \times \mathcal{G}$ — output of a generator in a snapshot — the `where` drops the retired unit entirely, so the built model is smaller than the coordinate product |

#### Objective

$$\min \sum_{s \in \mathcal{S}} \sum_{g \in \mathcal{G}} p_{s,g} \cdot c_{g}$$

#### Subject to

**`power_balance`**

$$\sum_{g \in \mathcal{G}} p_{s,g} = \ell_{s} \qquad \forall\thinspace s \in \mathcal{S}$$

#### Variable domains

**`p`**

$$0 \le p_{s,g} \le \bar p_{g} \qquad \forall\thinspace s \in \mathcal{S},\enspace g \in \mathcal{G} \thinspace:\thinspace \bar p_{g} > 0$$

</details>
<!-- math:end -->

```yaml
description: >-
  The dispatch model of README.md, plus one macro and one named expression —
  small enough to print in full, complete enough that every pipeline stage in
  examples/walkthrough.py has something to show.

dimensions:
  snapshot:
    description: dispatch periods
    dtype: int
  generator:
    description: >-
      generating units, including oil, which is retired and gets no columns at
      all

parameters:
  p_max: {dims: [generator], description: "installed capacity, zero for a retired unit"}
  load: {dims: [snapshot], description: "demand to be met"}
  cost: {dims: [generator], description: "marginal cost"}

expressions:
  total_supply:
    expression: sum(p, over=generator)
    description: what the whole fleet produces in a snapshot

macros:
  weighted_sum:
    description: an array priced by a second one and summed over a dimension
    args: [array, weights]
    kwargs: [over]
    template: sum(array * weights, over=over)

variables:
  p:
    description: >-
      output of a generator in a snapshot — the `where` drops the retired unit
      entirely, so the built model is smaller than the coordinate product
    foreach: [snapshot, generator]
    where: "p_max > 0"
    bounds:
      lower: 0
      upper: p_max

constraints:
  power_balance:
    description: the fleet meets the load exactly in every snapshot
    foreach: [snapshot]
    expression: total_supply == load

objective:
  sense: minimize
  description: total cost of generation over the horizon
  expression: sum(weighted_sum(p, cost, over=generator))
```

## What it exercises

This is the model behind `python examples/walkthrough.py`, which runs it
through every stage — YAML → schema → core AST → logical plan → model frames →
LP text → solution — printing what each stage produces, then two models the
language refuses and why. The committed output is
[examples/walkthrough.out](https://github.com/fluxopt/lpspec/blob/main/examples/walkthrough.out)
if you would rather read than run.

It is the only model here that uses **tier 2**: a macro and a named
expression. The macro does not survive the language's expansion — nothing
downstream of `math_spec` knows it existed, which is what makes it free. The named
expression is substituted the same way wherever a constraint uses it, but its
name survives on the model: stage 6 reads `total_supply` back at the solution
with `expression()`, lowered on that read rather than at build, so declaring
it still costs the build nothing.

---

[`examples/walkthrough.yaml`](https://github.com/fluxopt/lpspec/blob/main/examples/walkthrough.yaml) · back to [all models](index.md)
