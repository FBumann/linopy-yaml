# PyPSA LOPF — rung 2, ramp limits

[Rung 1](pypsa_transport.md) plus a limit on how fast each generator may change output between snapshots.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.9.0)** — objective **18200**, matched to `rtol=1e-09`.

PyPSA states a ramp limit as a fraction of `p_nom` bounding the change between
consecutive snapshots, written from the *second* snapshot on — there is no
dispatch before the first for it to ramp from.

**The rung binds, and that took a redesign.** Rung 1's links run saturated,
which fixes every generator's output exactly; a ramp limit on that instance can
only make it infeasible, never change the answer. So this rung widens the
ratings to 200 and lets merit order pick the dispatch. Gas then moves
70 → 100 → 80 → 50, hitting its ±30 limit twice and calling oil on at the two
middle snapshots. Without the limits the same instance costs **17000**; with
them, 18200.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ --- `snapshot` |
| $\mathcal{B}$ | index $b$ --- `bus` |
| $\mathcal{G}$ | index $g$ --- `generator` with $\mathrm{bus}: \mathcal{G} \to \mathcal{B}$ |
| $\mathcal{L}$ | index $l$ --- `link` with $\mathrm{from}: \mathcal{L} \to \mathcal{B},\enspace \mathrm{to}: \mathcal{L} \to \mathcal{B}$ |

#### Parameters

| Symbol | Meaning |
|---|---|
| $p^{\mathrm{nom}}$ | `p_nom` over $\mathcal{G}$ |
| $\mathit{marginal\_cost}$ | `marginal_cost` over $\mathcal{G}$ |
| $\mathit{ramp\_limit\_up}$ | `ramp_limit_up` over $\mathcal{G}$ |
| $\mathit{ramp\_limit\_down}$ | `ramp_limit_down` over $\mathcal{G}$ |
| $\mathit{rating}$ | `rating` over $\mathcal{L}$ |
| $\mathit{neg\_rating}$ | `neg_rating` over $\mathcal{L}$ |
| $\mathit{load}$ | `load` over $\mathcal{T} \times \mathcal{B}$ |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{T} \times \mathcal{G}$ |
| $f$ | `f` over $\mathcal{T} \times \mathcal{L}$ |

#### Objective

**`total_cost`**

$$\min \sum_{t \in \mathcal{T},\enspace g \in \mathcal{G}} p_{t,g} \cdot \mathit{marginal\_cost}_{g}$$

#### Subject to

**`nodal_balance`**

$$\sum_{g \in \mathcal{G} \thinspace:\thinspace \mathrm{bus}(g) = b} p_{t,g} + \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{to}(l) = b} f_{t,l} - \left( \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{from}(l) = b} f_{t,l} \right) = \mathit{load}_{t,b} \qquad \forall\thinspace t \in \mathcal{T},\enspace b \in \mathcal{B}$$

**`ramp_up`**

$$p_{t,g} - p_{t - 1,g} \le \mathit{ramp\_limit\_up}_{g} \cdot p^{\mathrm{nom}}_{g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`ramp_down`**

$$p_{t - 1,g} - p_{t,g} \le \mathit{ramp\_limit\_down}_{g} \cdot p^{\mathrm{nom}}_{g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

#### Variable domains

**`p`**

$$0 \le p_{t,g} \le p^{\mathrm{nom}}_{g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`f`**

$$\mathit{neg\_rating}_{l} \le f_{t,l} \le \mathit{rating}_{l} \qquad \forall\thinspace t \in \mathcal{T},\enspace l \in \mathcal{L}$$

</details>
<!-- math:end -->

The tabs start from [the instance’s tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    # PyPSA linear optimal power flow, rung 2: rung 1 plus generator ramp limits.
    # Optimum 18200.0, from PyPSA itself.

    dimensions:
      snapshot:
        dtype: int
      bus:
        dtype: str
      generator:
        dtype: str
        coords: [bus]  # every generator sits on a bus
      link:
        dtype: str
        coords: {from: bus, to: bus}  # both endpoints are buses

    parameters:
      p_nom:
        dims: [generator]
      marginal_cost:
        dims: [generator]
      ramp_limit_up:
        dims: [generator]
      ramp_limit_down:
        dims: [generator]
      rating:
        dims: [link]
      neg_rating:
        dims: [link]
      load:
        dims: [snapshot, bus]

    variables:
      p:
        foreach: [snapshot, generator]
        bounds:
          lower: 0
          upper: p_nom
      f:
        foreach: [snapshot, link]
        bounds:
          lower: neg_rating
          upper: rating

    constraints:
      nodal_balance:
        foreach: [snapshot, bus]
        expression: >-
          sum(p, over=generator, group_by=bus)
          + sum(f, over=link, group_by=to)
          - sum(f, over=link, group_by=from)
          == load

      ramp_up:
        foreach: [snapshot, generator]
        expression: p - shift(p, over=snapshot, by=1) <= ramp_limit_up * p_nom

      ramp_down:
        foreach: [snapshot, generator]
        expression: shift(p, over=snapshot, by=1) - p <= ramp_limit_down * p_nom

    objectives:
      total_cost:
        sense: minimize
        expression: p * marginal_cost
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/ports/pypsa_ramp.yaml', sources) as solution:
        solution.objective  # 18200.0
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The reference builds the same network with PyPSA's own objects. The delta from
    rung 1 is two keyword arguments:

    The model-building half of `examples/ports/references/pypsa/pypsa_ramp.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the lpspec call binds as ``sources``.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.add('Bus', tables['bus']['bus'])

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        links: pd.DataFrame = tables['link'].set_index('link')

        n.add(
            'Generator',
            generators.index,
            bus=generators['bus'],
            p_nom=tables['p_nom'].set_index('generator')['value'],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
            ramp_limit_up=tables['ramp_limit_up'].set_index('generator')['value'],
            ramp_limit_down=tables['ramp_limit_down'].set_index('generator')['value'],
        )
        n.add(
            'Link',
            links.index,
            bus0=links['from'],
            bus1=links['to'],
            p_nom=tables['rating'].set_index('link')['value'],
            p_min_pu=-1.0,
            efficiency=1.0,
        )

        load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
        for bus in tables['bus']['bus']:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
        return n
    ```

`shift` vacates the first snapshot, and a vacated position is *absent*, so the
row there drops on its own — which is the boundary PyPSA wants, since nothing
precedes it to ramp from. No `where` states it. Asking for the wrap,
`edge='wrap'`, would put the last snapshot onto the first and quietly build a different
model; it needs a gate, and a gate written as `snapshot > 0` hardcodes the
index origin, so it stops being the boundary on a horizon that starts anywhere
else. [Rung 4](pypsa_cyclic_storage.md) wants the wrap and asks for it by name.

## What it exercises

`shift` — the first externally verified model in the corpus to translate along a
dimension, and the acyclic boundary it carries. Also
parameter arithmetic on a constraint's right-hand side (`ramp_limit_up *
p_nom`), kept as arithmetic rather than a precomputed column so the file states
what PyPSA states.
