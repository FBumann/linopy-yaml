# Choosing the mode of transport

Moving 180 tonnes of chemicals out of four depots, where a depot may reach a centre by rail *or* by road at different cost.

> **✔ Verified against the published optimum** — **1715**, from Guéret, Prins, Sevaux & Heipcke, *Applications of Optimization with Xpress-MP* §10.2.3.

**The connection has a name, so two of them may join the same depot and centre**
and keep their own cost, their own minimum and their own capacity. That is the
whole model, and the source states the problem it solves better than we can. On
p. 142 the book observes that its data *"cannot be coded as a (two-dimensional)
matrix: for instance the element COST\(_{ij}\) of a cost matrix can only define a
single cost"* — and works around it by inventing **a fictitious node per mode
per connection**, six of them, turning each parallel pair into two paths through
distinct intermediates.

Here the connection is the axis, so the six nodes are not needed: `d2_c2_rail`
and `d2_c2_road` are two rows that disagree on cost (12 against 14) and on band
(10–50 against unbounded). The book's graph carries 15 nodes; this model carries
four.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{D}$ | index $d$ --- `depot` |
| $\mathcal{C}$ | index $c$ --- `connection` with $\mathrm{origin}: \mathcal{C} \to \mathcal{D}$ |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\mathit{stock}$ | `stock` over $\mathcal{D}$ |
| $\mathit{cost}$ | `cost` over $\mathcal{C}$ |
| $\mathit{min\_load}$ | `min_load` over $\mathcal{C}$ |
| $\mathit{max\_load}$ | `max_load` over $\mathcal{C}$ |
| $\mathit{total\_to\_move}$ | `total_to_move` (scalar) |

#### Variables

| Symbol | Meaning |
|---|---|
| $\mathit{moved}$ | `moved` over $\mathcal{C}$ |

#### Objective

$$\min \sum_{c \in \mathcal{C}} \mathit{moved}_{c} \cdot \mathit{cost}_{c}$$

#### Subject to

**`within_stock`**

$$\sum_{c \in \mathcal{C} \thinspace:\thinspace \mathrm{origin}(c) = d} \mathit{moved}_{c} \le \mathit{stock}_{d} \qquad \forall\thinspace d \in \mathcal{D}$$

**`move_the_lot`**

$$\sum_{c \in \mathcal{C}} \mathit{moved}_{c} = \mathit{total\_to\_move}$$

#### Variable domains

**`moved`**

$$\mathit{min\_load}_{c} \le \mathit{moved}_{c} \le \mathit{max\_load}_{c} \qquad \forall\thinspace c \in \mathcal{C}$$

</details>
<!-- math:end -->

The tabs start from [the instance’s tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    # Moving 180 tonnes of chemicals out of four depots to three recycling
    # centres, where a depot may reach a centre by rail *or* by road at different
    # cost. Guéret, Prins, Sevaux & Heipcke, Applications of Optimization with
    # Xpress-MP, §10.2. Optimum 1715, published in §10.2.3.
    #
    # The connection is a thing with a name, so two of them may join the same
    # depot and centre and keep their own cost and their own band. A cost matrix
    # indexed by (depot, centre) has one cell for the pair and cannot.

    dimensions:
      depot:
        dtype: str
      connection:
        dtype: str

    lookups:
      origin:
        description: the depot a connection leaves
        over: connection
        into: depot

    parameters:
      stock:
        dims: [depot]
      cost:
        dims: [connection]
      # Rail carries between 10 and 50 tonnes per delivery; road is unconstrained.
      # The band is per connection, which is the point: `d2_c2_rail` and
      # `d2_c2_road` share a depot and a centre and disagree on all three numbers.
      min_load:
        dims: [connection]
      max_load:
        dims: [connection]
      total_to_move:
        dims: []

    variables:
      moved:
        foreach: [connection]
        bounds:
          lower: min_load
          upper: max_load

    constraints:
      within_stock:
        foreach: [depot]
        expression: sum(moved, over=connection, group_by=origin) <= stock

      move_the_lot:
        foreach: []
        expression: sum(moved, over=connection) == total_to_move

    objective:
      sense: minimize
      expression: moved * cost
    ```

**The centre is in the connection's name and in nothing else, deliberately.**
This problem sets no per-centre demand — only a total of 180 t — so no
constraint reads where a connection ends, and a `destination` coordinate would
be a declaration nothing consumes. The language warns about exactly that, so it
is not written. The pairing is still visible where it matters: two connections
out of `d2` differing only in mode, which is what a `(depot, centre)` table
cannot hold.

The consequence is worth stating plainly, because it cuts against the obvious
reading: the second leg was load-bearing in the *book's* formulation — its
fictitious nodes need conservation, which reads both ends of every arc — and
reifying the connection is what removed its job. Where both legs earn their
keep is a problem with per-destination demand — §12.3 of the same book is one,
and is a port of its own.

**The rail band is a hard bound here, as in the source.** The prose says rail
carries *"at least 10 tonnes and at most 50 tonnes for any single delivery"*,
which reads as semi-continuous — either nothing or at least ten. The book's own
Mosel model writes `flow(a) >= MINCAP(a)` unconditionally, so every rail
connection carries at least 10 t whether it is wanted or not, and the published
1715 is that reading. The port matches the model that produced the number, and
the semi-continuous variant is a different question ([#383](https://github.com/fluxopt/lpspec/issues/383)) rather than a
workaround hidden here.

## What it exercises

`sum(group_by=)` through a coordinate whose target carries a constraint of its
own, and per-label bounds read from data — with the point being what the *label*
is. Reifying the connection is not a trick: it is what lets two rows describe
the same pair, and the six nodes the source spends are the price of not having
it.
