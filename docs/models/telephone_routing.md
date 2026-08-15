# Routing telephone calls

How many of 425 requested circuits a five-city network can carry at once — and by which routes.

> **✔ Verified against the published optimum** — **380 circuits**, from Guéret, Prins, Sevaux & Heipcke, *Applications of Optimization with Xpress-MP* §12.3.3.

**Both kinds of relation, in one model, each said the way it is.** A path serves
exactly one city pair, so `call_of` is a coordinate and the demand limit groups
through it. A path traverses several arcs, so `uses` is a `(path, arc)` table
and the capacity limit contracts against it. Neither could stand in for the
other: a coordinate is single-valued, and an incidence table cannot be grouped
by.

Five cities, six undirected links, five city pairs asking for circuits, and 17
elementary paths between them. A circuit reserves both directions of every link
it crosses, which is why the network is undirected and the flow carries no sign.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{A}$ | index $a$ --- `arc` |
| $\mathcal{C}$ | index $c$ --- `call` |
| $\mathcal{P}$ | index $p$ --- `path` with $\mathrm{call\_of}: \mathcal{P} \to \mathcal{C}$ |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\mathit{capacity}$ | `capacity` over $\mathcal{A}$ |
| $\mathit{demand}$ | `demand` over $\mathcal{C}$ |
| $\mathit{uses}$ | `uses` over $\mathcal{P} \times \mathcal{A}$ |

#### Variables

| Symbol | Meaning |
|---|---|
| $\mathit{flow}$ | `flow` over $\mathcal{P}$ |

#### Objective

$$\max \sum_{p \in \mathcal{P}} \mathit{flow}_{p}$$

#### Subject to

**`within_demand`**

$$\sum_{p \in \mathcal{P} \thinspace:\thinspace \mathrm{call\_of}(p) = c} \mathit{flow}_{p} \le \mathit{demand}_{c} \qquad \forall\thinspace c \in \mathcal{C}$$

**`within_capacity`**

$$\sum_{p \in \mathcal{P}} \mathit{flow}_{p} \cdot \mathit{uses}_{p,a} \le \mathit{capacity}_{a} \qquad \forall\thinspace a \in \mathcal{A}$$

#### Variable domains

**`flow`**

$$\mathit{flow}_{p} \ge 0, \mathit{flow}_{p} \in \mathbb{Z} \qquad \forall\thinspace p \in \mathcal{P}$$

</details>
<!-- math:end -->

The tabs start from [the instance’s tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    # Routing telephone calls over a five-city network: how many of the 425
    # requested circuits can be carried at once. Guéret, Prins, Sevaux & Heipcke,
    # Applications of Optimization with Xpress-MP, §12.3.
    # Optimum 380 circuits, published in §12.3.3.
    #
    # A path serves exactly one city pair — a coordinate — and traverses several
    # arcs — an incidence parameter. Both relations, each said the way it is.

    dimensions:
      arc:
        dtype: str  # an undirected link, capacity in circuits
      call:
        dtype: str  # a city pair with circuits to place
      path:
        dtype: str
        coords: {call_of: call}  # every path serves one pair, end to end

    parameters:
      capacity:
        dims: [arc]
      demand:
        dims: [call]
      # Which arcs a path traverses: many-to-many, so rows and not a coordinate.
      # A path uses an arc or it does not, so the value is 1 and absence is 0.
      uses:
        dims: [path, arc]

    variables:
      # Circuits carried on a path. A circuit reserves both directions of every arc
      # it crosses, which is why the network is undirected and the flow is not
      # signed. Integral because a multi-commodity flow is not integral by nature,
      # even though this instance's relaxation happens to be.
      flow:
        foreach: [path]
        domain: integer
        bounds:
          lower: 0

    constraints:
      # A pair cannot be carried more than it asked for, however many paths serve it.
      within_demand:
        foreach: [call]
        expression: sum(flow, over=path, group_by=call_of) <= demand

      # An arc carries every path that traverses it, and no more than its capacity.
      within_capacity:
        foreach: [arc]
        expression: sum(flow * uses, over=path) <= capacity

    objective:
      sense: maximize
      expression: sum(flow, over=path)
    ```

**Why the answer is 380 and not 425.** Troyes is reachable only over
`troyes_nice` (80) and `troyes_valenciennes` (70), so at most 150 circuits can
terminate there — and Troyes must absorb the 80 Nantes–Troyes and the 70
Paris–Troyes, which is 150 exactly. Valenciennes is reached only over
`paris_valenciennes` (200) and `troyes_valenciennes` (70), capping everything
ending there at 270 against a demand of 175 plus whatever transits. The binding
cut leaves 45 of the Nantes–Troyes circuits unplaced, and the published routing
says the same: 35 of 80 carried.

That is worth stating because the number can be derived before any solver runs,
which is the strongest form a corpus entry can take — the optimum is not merely
somebody else's output, it is arithmetic anybody can check.

## What it exercises

A coordinate and an incidence parameter side by side, each carrying the relation
it fits, in a model neither we nor an energy system wrote. `reserves` already
proves both idioms, but as our own model built to prove them; this is the
outside witness.

Integrality is the source's, not ours: a multi-commodity flow is not integral by
nature, and the book says so before observing that this instance's relaxation
happens to be. A MILP has no dual solution, so the entry records none.
