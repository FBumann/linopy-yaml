# PyPSA LOPF — ramp limits

[The transport model](pypsa_transport.md) plus a limit on how fast each generator may change output between snapshots.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.9.0)** — objective **18200**, matched to `rtol=1e-09`.

PyPSA states a ramp limit as a fraction of `p_nom` bounding the change between
consecutive snapshots, written from the *second* snapshot on — there is no
dispatch before the first for it to ramp from.

**The limit binds, and that took a redesign.** [The transport
model](pypsa_transport.md)'s links run saturated, which fixes every generator's
output exactly; a ramp limit on that instance can only make it infeasible,
never change the answer. So this model widens the
ratings to 200 and lets merit order pick the dispatch. Gas then moves
70 → 100 → 80 → 50, hitting its ±30 limit twice and calling oil on at the two
middle snapshots. Without the limits the same instance costs **17000**; with
them, 18200.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA linear optimal power flow with a limit on how fast a generator may change output between snapshots. Optimum 18200.0, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ — `snapshot` — dispatch periods |
| $\mathcal{B}$ | index $b$ — `bus` — network nodes |
| $\mathcal{G}$ | index $g$ — `generator` with $\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}$ — generating units, each sitting on one bus |
| $\mathcal{L}$ | index $l$ — `link` with $\mathrm{link\_from}: \mathcal{L} \to \mathcal{B},\enspace \mathrm{link\_to}: \mathcal{L} \to \mathcal{B}$ — controllable connections, each joining two buses |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\mathrm{p}^{\mathrm{nom}}$ | `p_nom` over $\mathcal{G}$ — installed capacity of a generator |
| $\mathrm{marginal\_cost}$ | `marginal_cost` over $\mathcal{G}$ — cost of one unit of output |
| $\mathrm{ramp\_limit\_up}$ | `ramp_limit_up` over $\mathcal{G}$ — share of capacity output may rise by from one snapshot to the next |
| $\mathrm{ramp\_limit\_down}$ | `ramp_limit_down` over $\mathcal{G}$ — share of capacity output may fall by from one snapshot to the next |
| $\mathrm{rating}$ | `rating` over $\mathcal{L}$ — most a link may carry towards its `link_to` bus |
| $\mathrm{neg\_rating}$ | `neg_rating` over $\mathcal{L}$ — most a link may carry the other way, negative by convention |
| $\mathrm{load}$ | `load` over $\mathcal{T} \times \mathcal{B}$ — demand at each bus in each snapshot |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{T} \times \mathcal{G}$ — output of a generator in a snapshot |
| $f$ | `f` over $\mathcal{T} \times \mathcal{L}$ — flow on a link, signed towards its `link_to` bus |

Upright is what the model is given — a parameter such as $\mathrm{p}^{\mathrm{nom}}$, a coordinate map, a label — and italic is what the solver chooses, such as $p$. An index is italic too, being what a quantifier chooses, and a set is script.

#### Objective

$$\min \sum_{t \in \mathcal{T},\enspace g \in \mathcal{G}} p_{t,g} \cdot \mathrm{marginal\_cost}_{g}$$

#### Subject to

**`nodal_balance`**

$$\sum_{g \in \mathcal{G} \thinspace:\thinspace \mathrm{gen\_bus}(g) = b} p_{t,g} + \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{link\_to}(l) = b} f_{t,l} - \left( \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{link\_from}(l) = b} f_{t,l} \right) = \mathrm{load}_{t,b} \qquad \forall\thinspace t \in \mathcal{T},\enspace b \in \mathcal{B}$$

**`ramp_up`**

$$p_{t,g} - p_{t - 1,g} \le \mathrm{ramp\_limit\_up}_{g} \cdot \mathrm{p}^{\mathrm{nom}}_{g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`ramp_down`**

$$p_{t - 1,g} - p_{t,g} \le \mathrm{ramp\_limit\_down}_{g} \cdot \mathrm{p}^{\mathrm{nom}}_{g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

#### Variable domains

**`p`**

$$0 \le p_{t,g} \le \mathrm{p}^{\mathrm{nom}}_{g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`f`**

$$\mathrm{neg\_rating}_{l} \le f_{t,l} \le \mathrm{rating}_{l} \qquad \forall\thinspace t \in \mathcal{T},\enspace l \in \mathcal{L}$$

</details>
<!-- math:end -->

The tabs start from [the instance's tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    description: >-
      PyPSA linear optimal power flow with a limit on how fast a generator may
      change output between snapshots. Optimum 18200.0, from PyPSA
      itself.

    dimensions:
      snapshot:
        description: dispatch periods
        dtype: int
      bus:
        description: network nodes
        dtype: str
      generator:
        description: generating units, each sitting on one bus
        dtype: str
      link:
        description: controllable connections, each joining two buses
        dtype: str

    lookups:
      gen_bus:
        description: the bus a generator sits on
        over: generator
        into: bus
      link_from:
        description: the bus a link leaves
        over: link
        into: bus
      link_to:
        description: the bus a link arrives at
        over: link
        into: bus

    parameters:
      p_nom:
        description: installed capacity of a generator
        dims: [generator]
      marginal_cost:
        description: cost of one unit of output
        dims: [generator]
      ramp_limit_up:
        description: share of capacity output may rise by from one snapshot to the next
        dims: [generator]
      ramp_limit_down:
        description: share of capacity output may fall by from one snapshot to the next
        dims: [generator]
      rating:
        description: most a link may carry towards its `link_to` bus
        dims: [link]
      neg_rating:
        description: most a link may carry the other way, negative by convention
        dims: [link]
      load:
        description: demand at each bus in each snapshot
        dims: [snapshot, bus]

    variables:
      p:
        description: output of a generator in a snapshot
        foreach: [snapshot, generator]
        bounds:
          lower: 0
          upper: p_nom
      f:
        description: flow on a link, signed towards its `link_to` bus
        foreach: [snapshot, link]
        bounds:
          lower: neg_rating
          upper: rating

    constraints:
      nodal_balance:
        description: what is generated at a bus plus what arrives over the links meets the load there
        foreach: [snapshot, bus]
        expression: >-
          sum(p, by=gen_bus)
          + sum(f, by=link_to)
          - sum(f, by=link_from)
          == load

      ramp_up:
        foreach: [snapshot, generator]
        expression: p - shift(p, over=snapshot, offset=1) <= ramp_limit_up * p_nom

      ramp_down:
        foreach: [snapshot, generator]
        expression: shift(p, over=snapshot, offset=1) - p <= ramp_limit_down * p_nom

    objective:
      sense: minimize
      description: total cost of generation; moving power over a link is free here
      expression: sum(p * marginal_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/ports/pypsa_ramp.yaml', sources) as solution:
        solution.objective  # 18200.0
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The reference builds the same network with PyPSA's own objects. The delta from
    the transport model is two keyword arguments:

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
            bus=generators['gen_bus'],
            p_nom=tables['p_nom'].set_index('generator')['value'],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
            ramp_limit_up=tables['ramp_limit_up'].set_index('generator')['value'],
            ramp_limit_down=tables['ramp_limit_down'].set_index('generator')['value'],
        )
        n.add(
            'Link',
            links.index,
            bus0=links['link_from'],
            bus1=links['link_to'],
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
else. [Cyclic storage](pypsa_cyclic_storage.md) wants the wrap and asks for it by name.

## What it exercises

`shift` — the first externally verified model in the corpus to translate along a
dimension, and the acyclic boundary it carries. Also
parameter arithmetic on a constraint's right-hand side (`ramp_limit_up *
p_nom`), kept as arithmetic rather than a precomputed column so the file states
what PyPSA states.
