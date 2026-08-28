# Dantzig transport with economies of scale

GAMS model library `trnspwl`: the same shipping problem, but a big consignment is cheaper per unit — cost grows as `sqrt(x)`, not linearly.

> **✔ Verified against linopy 0.9.0's own `add_piecewise_formulation`** — objective **8.786852757777865**, matched to `rtol=1e-09`.

**The corpus's `piecewise` entry**, and the last hole in the construct matrix.

It is also the port with the sharpest kind of independence. Every other
reference is independent of lpspec because it is a different program; this one
is independent of *the construct under test*. `piecewise:` and linopy's
`add_piecewise_formulation` are two separate implementations of the same
λ convex-combination idea, and this compares them on a model neither was
written for.

## The curve

GAMS discretises `sqrt(x)` into eight breakpoints: a straight line up to 50, six
sample points to 400, and a line out to 600 (the largest supply). It is chosen
to pass through the origin — so an unused route picks up no fixed cost — and to
underestimate `sqrt` everywhere in between.

| x | 0 | 50 | 120 | 190 | 260 | 330 | 400 | 600 |
|---|---|---|---|---|---|---|---|---|
| f(x) | 0 | 7.071 | 10.954 | 13.784 | 16.125 | 18.166 | 20 | 24.495 |

The [instance](https://github.com/fluxopt/lpspec/blob/main/examples/ports/data/transport_pwl.json) is otherwise
[Dantzig's](transport_dantzig.md), unchanged.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

Dantzig's transportation problem with economies of scale — GAMS model library trnspwl. Shipping cost grows as the square root of the consignment rather than linearly, so a big consignment is cheaper per unit. Optimum 8.786852757777865, from linopy's own piecewise formulation.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{P}$ | index $p$ — `plant` — canning plants, with limited capacity |
| $\mathcal{M}$ | index $m$ — `market` — markets, with demand to be met |
| $\mathcal{B}$ | index $b$ — `bp` — breakpoints of the discretised square-root curve |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\mathrm{capacity}$ | `capacity` over $\mathcal{P}$ — capacity of each plant |
| $\mathrm{demand}$ | `demand` over $\mathcal{M}$ — demand at each market |
| $\mathrm{distance}$ | `distance` over $\mathcal{P} \times \mathcal{M}$ — distance from plant to market |
| $\mathrm{freight}$ | `freight` (scalar) — freight rate per case per unit distance |
| $\mathrm{bp\_x}$ | `bp_x` over $\mathcal{B}$ — breakpoint shipment levels — one curve, the same on every route, so it carries the breakpoint dimension alone and broadcasts across the pairs |
| $\mathrm{bp\_y}$ | `bp_y` over $\mathcal{B}$ — the curve's value at each breakpoint |

#### Variables

| Symbol | Meaning |
|---|---|
| $\mathit{shipment}$ | `shipment` over $\mathcal{P} \times \mathcal{M}$ — cases shipped from a plant to a market |
| $\mathit{scaled}$ | `scaled` over $\mathcal{P} \times \mathcal{M}$ — what the objective is charged on — the square root of the shipment, read off the curve rather than computed |
| $\mathit{economies\_of\_scale\_lam}$ | `economies_of_scale_lam` over $\mathcal{P} \times \mathcal{M} \times \mathcal{B}$ — convex-combination weight on a breakpoint |
| $\mathit{economies\_of\_scale\_seg}$ | `economies_of_scale_seg` over $\mathcal{P} \times \mathcal{M} \times \mathcal{B}$ |

Upright is what the model is given — a parameter such as $\mathrm{capacity}$, a coordinate map, a label — and italic is what the solver chooses, such as $\mathit{shipment}$. An index is italic too, being what a quantifier chooses, and a set is script.

$t \boxminus_{v} k$ denotes translation with $v$ standing where index $t-k$ leaves the dimension (`shift(edge=v)`), so the row at that boundary is built and carries $v$ rather than being dropped.

#### Objective

$$\min \sum_{p \in \mathcal{P},\enspace m \in \mathcal{M}} \frac{\mathit{scaled}_{p,m} \cdot \mathrm{distance}_{p,m} \cdot \mathrm{freight}}{1000}$$

#### Subject to

**`within_capacity`**

$$\sum_{m \in \mathcal{M}} \mathit{shipment}_{p,m} \le \mathrm{capacity}_{p} \qquad \forall\thinspace p \in \mathcal{P}$$

**`meet_demand`**

$$\sum_{p \in \mathcal{P}} \mathit{shipment}_{p,m} \ge \mathrm{demand}_{m} \qquad \forall\thinspace m \in \mathcal{M}$$

**`economies_of_scale_convexity`**

$$\sum_{b \in \mathcal{B}} \mathit{economies\_of\_scale\_lam}_{p,m,b} = 1 \qquad \forall\thinspace p \in \mathcal{P},\enspace m \in \mathcal{M}$$

**`economies_of_scale_link0`**

$$\mathit{shipment}_{p,m} = \sum_{b \in \mathcal{B}} \mathit{economies\_of\_scale\_lam}_{p,m,b} \cdot \mathrm{bp\_x}_{b} \qquad \forall\thinspace p \in \mathcal{P},\enspace m \in \mathcal{M}$$

**`economies_of_scale_link1`**

$$\mathit{scaled}_{p,m} = \sum_{b \in \mathcal{B}} \mathit{economies\_of\_scale\_lam}_{p,m,b} \cdot \mathrm{bp\_y}_{b} \qquad \forall\thinspace p \in \mathcal{P},\enspace m \in \mathcal{M}$$

**`economies_of_scale_pick`**

$$\sum_{b \in \mathcal{B}} \mathit{economies\_of\_scale\_seg}_{p,m,b} = 1 \qquad \forall\thinspace p \in \mathcal{P},\enspace m \in \mathcal{M}$$

**`economies_of_scale_adjacency`**

$$\mathit{economies\_of\_scale\_lam}_{p,m,b} \le \mathit{economies\_of\_scale\_seg}_{p,m,b} + \mathit{economies\_of\_scale\_seg}_{p,m,b \boxminus_{0} 1} \qquad \forall\thinspace p \in \mathcal{P},\enspace m \in \mathcal{M},\enspace b \in \mathcal{B}$$

#### Variable domains

**`shipment`**

$$\mathit{shipment}_{p,m} \ge 0 \qquad \forall\thinspace p \in \mathcal{P},\enspace m \in \mathcal{M}$$

**`scaled`**

$$\mathit{scaled}_{p,m} \ge 0 \qquad \forall\thinspace p \in \mathcal{P},\enspace m \in \mathcal{M}$$

**`economies_of_scale_lam`**

$$0 \le \mathit{economies\_of\_scale\_lam}_{p,m,b} \le 1 \qquad \forall\thinspace p \in \mathcal{P},\enspace m \in \mathcal{M},\enspace b \in \mathcal{B}$$

**`economies_of_scale_seg`**

$$\mathit{economies\_of\_scale\_seg}_{p,m,b} \in \{0, 1\} \qquad \forall\thinspace p \in \mathcal{P},\enspace m \in \mathcal{M},\enspace b \in \mathcal{B}$$

</details>
<!-- math:end -->

The tabs start from [the instance's tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    description: >-
      Dantzig's transportation problem with economies of scale — GAMS model library
      trnspwl. Shipping cost grows as the square root of the consignment rather
      than linearly, so a big consignment is cheaper per unit. Optimum
      8.786852757777865, from linopy's own piecewise formulation.

    dimensions:
      plant:
        description: canning plants, with limited capacity
      market:
        description: markets, with demand to be met
      bp:
        description: breakpoints of the discretised square-root curve
        dtype: int

    parameters:
      capacity:
        description: capacity of each plant
        dims: [plant]
      demand:
        description: demand at each market
        dims: [market]
      distance:
        description: distance from plant to market
        dims: [plant, market]
      freight:
        description: freight rate per case per unit distance
        dims: []
      bp_x:
        description: >-
          breakpoint shipment levels — one curve, the same on every route, so it
          carries the breakpoint dimension alone and broadcasts across the pairs
        dims: [bp]
      bp_y:
        description: the curve's value at each breakpoint
        dims: [bp]

    variables:
      shipment:
        description: cases shipped from a plant to a market
        foreach: [plant, market]
        bounds:
          lower: 0
      scaled:
        description: >-
          what the objective is charged on — the square root of the shipment, read
          off the curve rather than computed
        foreach: [plant, market]
        bounds:
          lower: 0

    piecewise:
      economies_of_scale:
        description: >-
          the shipment priced through the discretisation GAMS publishes, on segment
          binaries and deliberately not the convex method: the curve is concave and
          this is a
          minimisation, so the convex-hull relaxation would let the solver ride the
          chord underneath the true curve and buy transport cheaper than the model
          allows. The binaries are what make the answer right — and what make this
          port a MILP.
        over: bp
        links:
          - [shipment, bp_x]
          - [scaled, bp_y]

    constraints:
      within_capacity:
        foreach: [plant]
        expression: sum(shipment, over=market) <= capacity
      meet_demand:
        foreach: [market]
        expression: sum(shipment, over=plant) >= demand

    objective:
      sense: minimize
      description: total freight, charged on the scaled consignment rather than on the shipment
      expression: sum(scaled * distance * freight / 1000)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/ports/transport_pwl.yaml', sources) as solution:
        solution.objective  # 8.786852757777865
    ```

=== "linopy"

    The model-building half of `examples/ports/references/linopy/transport_pwl.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
        """The port's tables as a linopy model, column for column.

        ``tables`` is the same mapping the lpspec call binds as ``sources``.
        ``scaled`` is what the objective is actually charged on — ``sqrt(shipment)``
        read off the discretised curve rather than computed.
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
        scaled = m.add_variables(lower=0, coords=[capacity.index, demand.index], name='scaled')

        m.add_piecewise_formulation(
            (shipment, list(tables['bp_x']['value'])),
            (scaled, list(tables['bp_y']['value'])),
        )

        m.add_constraints(shipment.sum('market') <= capacity, name='within_capacity')
        m.add_constraints(shipment.sum('plant') >= demand, name='meet_demand')
        m.add_objective((scaled * cost).sum())
        return m
    ```

**`method: convex` would be wrong here, and quietly so.** `sqrt` is concave and
this is a minimisation, so the convex-hull relaxation lets the solver ride the
chord *underneath* the true curve and buy transport cheaper than the model
allows. Leaving it off emits segment binaries and an adjacency row, which is
what makes the answer right — and what makes this port a MILP rather than an LP.

That is the one judgement a reader has to make when writing a `piecewise:`
block, and it is not one the language can make for you: the curvature guard
catches *mixed* curvature, but a consistently concave curve under minimisation
is a modelling error, not a data error.

## What it exercises

`piecewise:` on its non-convex path — segment binaries, the adjacency row, and
the integrality that follows — plus `sum` and parameter arithmetic in the
objective.

**It is also the first port whose numbers are not bit-identical.** lpspec
returns `8.786852757777858` against linopy's `8.786852757777865`: a relative
difference of about 8 × 10⁻¹⁶, which is branch-and-bound arriving at the same
vertex by a different order of floating-point operations. The shipment plan is
identical. This is what per-port `rtol` is for, and why the corpus compares
objectives rather than bit patterns.
