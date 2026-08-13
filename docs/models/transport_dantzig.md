# Dantzig transport

Dantzig's transportation problem — GAMS model library #1, and the oldest LP in the corpus.

> **✔ Verified against published with GAMS model library #1 (trnsport)** — objective **153.675**, matched to `rtol=1e-09`.

## The problem

Ship from plants to markets at least cost, respecting capacity and meeting demand:

$$\min \sum_{i,j} c_{ij} x_{ij}
\quad\text{s.t.}\quad \sum_j x_{ij} \le a_i ,\quad \sum_i x_{ij} \ge b_j ,\quad x \ge 0$$

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{I}$ | index $i$ --- `plant` --- canning plants, with limited capacity |
| $\mathcal{J}$ | index $j$ --- `market` --- markets, with demand to be met |

#### Parameters

| Symbol | Meaning |
|---|---|
| $a$ | `capacity` over $\mathcal{I}$ --- capacity of each plant |
| $b$ | `demand` over $\mathcal{J}$ --- demand at each market |
| $d$ | `distance` over $\mathcal{I} \times \mathcal{J}$ --- distance from plant to market |
| $f$ | `freight` (scalar) --- freight rate per case per unit distance |

#### Variables

| Symbol | Meaning |
|---|---|
| $x$ | `shipment` over $\mathcal{I} \times \mathcal{J}$ --- cases shipped from plant $i$ to market $j$ |

#### Objective

**`total_cost`**

$$\min \sum_{i \in \mathcal{I},\enspace j \in \mathcal{J}} \frac{x_{i,j} \cdot d_{i,j} \cdot f}{1000}$$

#### Subject to

**`within_capacity`**

$$\sum_{j \in \mathcal{J}} x_{i,j} \le a_{i} \qquad \forall\thinspace i \in \mathcal{I}$$

**`meet_demand`**

$$\sum_{i \in \mathcal{I}} x_{i,j} \ge b_{j} \qquad \forall\thinspace j \in \mathcal{J}$$

#### Variable domains

**`shipment`**

$$x_{i,j} \ge 0 \qquad \forall\thinspace i \in \mathcal{I},\enspace j \in \mathcal{J}$$

</details>
<!-- math:end -->

The tabs start from [the instance’s tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    # Dantzig's transportation problem (GAMS model library #1). Optimum 153.675,
    # published with the model. See docs/models/index.md.

    dimensions:
      plant:
        values: [seattle, san-diego]
      market:
        values: [new-york, chicago, topeka]

    parameters:
      capacity:
        dims: [plant]
      demand:
        dims: [market]
      distance:
        dims: [plant, market]
      freight:
        dims: []

    variables:
      shipment:
        foreach: [plant, market]
        bounds:
          lower: 0

    constraints:
      within_capacity:
        foreach: [plant]
        expression: sum(shipment, over=market) <= capacity
      meet_demand:
        foreach: [market]
        expression: sum(shipment, over=plant) >= demand

    objectives:
      total_cost:
        sense: minimize
      # c(i,j) = f * d(i,j) / 1000 in the source, kept as arithmetic here
      # rather than precomputed, so the file states the model and not a
      # derived table.
        expression: shipment * distance * freight / 1000
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/ports/transport_dantzig.yaml', sources) as solution:
        solution.objective  # 153.675
        solution.dual('within_capacity')
    ```

=== "linopy"

    The same problem written by hand in linopy — a fair comparison, because linopy
    is what a user of this project would otherwise reach for. Both formulations
    solve to **153.675**; this script is run out of band and its number is recorded
    in `references.json`.

    The model-building half of `examples/ports/references/linopy/transport_dantzig.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
        """The port's tables as a linopy model, term for term.

        ``tables`` is the same mapping the lpspec call binds as ``sources``.
        """
        capacity: pd.Series = tables['capacity'].set_index('plant')['value']
        demand: pd.Series = tables['demand'].set_index('market')['value']
        distance: pd.DataFrame = (
            tables['distance']
            .pivot(index='plant', columns='market', values='value')
            .reindex(index=capacity.index)[demand.index]
        )
        cost: pd.DataFrame = distance * tables['freight'] / 1000

        m = linopy.Model()
        shipment = m.add_variables(lower=0, coords=[capacity.index, demand.index], name='shipment')
        m.add_constraints(shipment.sum('market') <= capacity, name='within_capacity')
        m.add_constraints(shipment.sum('plant') >= demand, name='meet_demand')
        m.add_objective((shipment * cost).sum())
        return m
    ```

The YAML is 40 lines and names the maths; the linopy version is ~20 lines
of Python and names the *data structures* the maths is carried in — a pivot, a
reindex, two `.sum()` calls over named axes. Neither is obviously better and
that is the honest read: what the declarative form buys here is not brevity but
that the file is the model, with no host language between the reader and it.

## What it exercises

The freight rate is kept as arithmetic — `distance * freight / 1000` — rather
than precomputed into a cost table, so the file states the model and not a
derived table. `freight` is declared with `dims: []`: a scalar is a parameter
with no dimensions, not a special case.

The objective is checked, never the primal. This model reaches 153.675 at a
**different vertex** than the source prints, so a corpus pinned to a solution
would fail on a solver upgrade that broke nothing.

---

[`examples/ports/transport_dantzig.yaml`](https://github.com/fluxopt/lpspec/blob/main/examples/ports/transport_dantzig.yaml) · back to [all models](index.md)
