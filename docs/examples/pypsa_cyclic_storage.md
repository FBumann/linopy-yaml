# PyPSA LOPF — rung 4, cyclic storage

[Rung 3](pypsa_storage.md) with the horizon closed on itself: the first snapshot's state of charge carries over from the *last*.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.9.0)** — objective **17228.77962151063**, matched to `rtol=1e-09`.

**The rung that makes the model smaller.** Rung 3 needs two equations for the
energy balance — one seeding the first snapshot from `soc_initial`, one carrying
over every other. Closing the cycle *removes the first*, and what is left
changes by one token: `shift` vacates the first snapshot and drops that row,
`edge='wrap'` puts it onto the last.

```diff
-  energy_balance_initial:
-    where: "position(snapshot) == 0"
-    expression: soc == soc_initial + p_store * ... - p_dispatch / ...
   energy_balance:
-    expression: soc == shift(soc, over=snapshot, offset=1) * (1 - standing_loss) + ...
+    expression: soc == shift(soc, over=snapshot, offset=1, edge='wrap') * (1 - standing_loss) + ...
```

`soc_initial` leaves the instance with it — a cyclic horizon has no seed to
give. In PyPSA the same change is `cyclic_state_of_charge=True`, which is
shorter still; the difference is that theirs is a flag on a component and ours
is the absence of a special case.

Closing the loop costs money: **17228.78** against rung 3's **15253.18**. The
battery can no longer end the horizon empty, so it has to buy back what it
spends.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA linear optimal power flow, rung 4: rung 3's storage, closed into a cycle — the first snapshot's state of charge carries over from the last. Optimum 17228.77962151063, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ — `snapshot` — dispatch periods, cyclic at the horizon |
| $\mathcal{B}$ | index $b$ — `bus` — network nodes |
| $\mathcal{G}$ | index $g$ — `generator` with $\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}$ — generating units, each sitting on one bus |
| $\mathcal{L}$ | index $l$ — `link` with $\mathrm{link\_from}: \mathcal{L} \to \mathcal{B},\enspace \mathrm{link\_to}: \mathcal{L} \to \mathcal{B}$ — controllable connections, each joining two buses |
| $\mathcal{S}$ | index $s$ — `storage` with $\mathrm{storage\_bus}: \mathcal{S} \to \mathcal{B}$ — storage units, each sitting on one bus |

#### Parameters

| Symbol | Meaning |
|---|---|
| $p^{\mathrm{nom}}$ | `p_nom` over $\mathcal{G}$ — installed capacity of a generator |
| $\mathit{marginal\_cost}$ | `marginal_cost` over $\mathcal{G}$ — cost of one unit of output |
| $\mathit{ramp\_limit\_up}$ | `ramp_limit_up` over $\mathcal{G}$ — share of capacity output may rise by from one snapshot to the next |
| $\mathit{ramp\_limit\_down}$ | `ramp_limit_down` over $\mathcal{G}$ — share of capacity output may fall by from one snapshot to the next |
| $\mathit{rating}$ | `rating` over $\mathcal{L}$ — most a link may carry towards its `link_to` bus |
| $\mathit{neg\_rating}$ | `neg_rating` over $\mathcal{L}$ — most a link may carry the other way, negative by convention |
| $\mathit{storage}^{\mathrm{p,nom}}$ | `storage_p_nom` over $\mathcal{S}$ — most a storage unit may charge or discharge in one snapshot |
| $\mathit{soc}^{\mathrm{max}}$ | `soc_max` over $\mathcal{S}$ — how much energy a storage unit holds when full |
| $\mathit{efficiency\_store}$ | `efficiency_store` over $\mathcal{S}$ — share of charging energy that reaches the store |
| $\mathit{efficiency\_dispatch}$ | `efficiency_dispatch` over $\mathcal{S}$ — share of stored energy that reaches the bus on the way out |
| $\mathit{standing\_loss}$ | `standing_loss` over $\mathcal{S}$ — share of the carried-over level lost between snapshots |
| $\mathit{load}$ | `load` over $\mathcal{T} \times \mathcal{B}$ — demand at each bus in each snapshot |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{T} \times \mathcal{G}$ — output of a generator in a snapshot |
| $f$ | `f` over $\mathcal{T} \times \mathcal{L}$ — flow on a link, signed towards its `link_to` bus |
| $p^{\mathrm{dispatch}}$ | `p_dispatch` over $\mathcal{T} \times \mathcal{S}$ — power a storage unit puts onto its bus |
| $p^{\mathrm{store}}$ | `p_store` over $\mathcal{T} \times \mathcal{S}$ — power a storage unit takes off its bus |
| $\mathit{soc}$ | `soc` over $\mathcal{T} \times \mathcal{S}$ — energy in the store at the end of a snapshot |

$t \ominus k$ denotes cyclic translation: index $t-k$ taken modulo the size of the dimension (`roll`). Plain $t-k$ (`shift`) has no wraparound — terms translated past the edge are simply absent.

#### Objective

$$\min \sum_{t \in \mathcal{T},\enspace g \in \mathcal{G}} p_{t,g} \cdot \mathit{marginal\_cost}_{g}$$

#### Subject to

**`nodal_balance`**

$$\sum_{g \in \mathcal{G} \thinspace:\thinspace \mathrm{gen\_bus}(g) = b} p_{t,g} + \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{link\_to}(l) = b} f_{t,l} - \left( \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{link\_from}(l) = b} f_{t,l} \right) + \sum_{s \in \mathcal{S} \thinspace:\thinspace \mathrm{storage\_bus}(s) = b} p^{\mathrm{dispatch}}_{t,s} - \left( \sum_{s \in \mathcal{S} \thinspace:\thinspace \mathrm{storage\_bus}(s) = b} p^{\mathrm{store}}_{t,s} \right) = \mathit{load}_{t,b} \qquad \forall\thinspace t \in \mathcal{T},\enspace b \in \mathcal{B}$$

**`ramp_up`**

$$p_{t,g} - p_{t - 1,g} \le \mathit{ramp\_limit\_up}_{g} \cdot p^{\mathrm{nom}}_{g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`ramp_down`**

$$p_{t - 1,g} - p_{t,g} \le \mathit{ramp\_limit\_down}_{g} \cdot p^{\mathrm{nom}}_{g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`energy_balance`**

$$\mathit{soc}_{t,s} = \mathit{soc}_{t \ominus 1,s} \cdot \left( 1 - \mathit{standing\_loss}_{s} \right) + p^{\mathrm{store}}_{t,s} \cdot \mathit{efficiency\_store}_{s} - \frac{p^{\mathrm{dispatch}}_{t,s}}{\mathit{efficiency\_dispatch}_{s}} \qquad \forall\thinspace t \in \mathcal{T},\enspace s \in \mathcal{S}$$

#### Variable domains

**`p`**

$$0 \le p_{t,g} \le p^{\mathrm{nom}}_{g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`f`**

$$\mathit{neg\_rating}_{l} \le f_{t,l} \le \mathit{rating}_{l} \qquad \forall\thinspace t \in \mathcal{T},\enspace l \in \mathcal{L}$$

**`p_dispatch`**

$$0 \le p^{\mathrm{dispatch}}_{t,s} \le \mathit{storage}^{\mathrm{p,nom}}_{s} \qquad \forall\thinspace t \in \mathcal{T},\enspace s \in \mathcal{S}$$

**`p_store`**

$$0 \le p^{\mathrm{store}}_{t,s} \le \mathit{storage}^{\mathrm{p,nom}}_{s} \qquad \forall\thinspace t \in \mathcal{T},\enspace s \in \mathcal{S}$$

**`soc`**

$$0 \le \mathit{soc}_{t,s} \le \mathit{soc}^{\mathrm{max}}_{s} \qquad \forall\thinspace t \in \mathcal{T},\enspace s \in \mathcal{S}$$

</details>
<!-- math:end -->

The tabs start from [the instance's tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    description: >-
      PyPSA linear optimal power flow, rung 4: rung 3's storage, closed into a
      cycle — the first snapshot's state of charge carries over from the last.
      Optimum 17228.77962151063, from PyPSA itself.

    dimensions:
      snapshot:
        description: dispatch periods, cyclic at the horizon
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
      storage:
        description: storage units, each sitting on one bus
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
      storage_bus:
        description: the bus a storage unit sits on
        over: storage
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
      storage_p_nom:
        description: most a storage unit may charge or discharge in one snapshot
        dims: [storage]
      soc_max:
        description: how much energy a storage unit holds when full
        dims: [storage]
      efficiency_store:
        description: share of charging energy that reaches the store
        dims: [storage]
      efficiency_dispatch:
        description: share of stored energy that reaches the bus on the way out
        dims: [storage]
      standing_loss:
        description: share of the carried-over level lost between snapshots
        dims: [storage]
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
      p_dispatch:
        description: power a storage unit puts onto its bus
        foreach: [snapshot, storage]
        bounds:
          lower: 0
          upper: storage_p_nom
      p_store:
        description: power a storage unit takes off its bus
        foreach: [snapshot, storage]
        bounds:
          lower: 0
          upper: storage_p_nom
      soc:
        description: energy in the store at the end of a snapshot
        foreach: [snapshot, storage]
        bounds:
          lower: 0
          upper: soc_max

    constraints:
      nodal_balance:
        description: >-
          what is generated at a bus, plus what arrives over the links and out of
          the stores, meets the load there
        foreach: [snapshot, bus]
        expression: >-
          sum(p, by=gen_bus)
          + sum(f, by=link_to)
          - sum(f, by=link_from)
          + sum(p_dispatch, by=storage_bus)
          - sum(p_store, by=storage_bus)
          == load

      ramp_up:
        foreach: [snapshot, generator]
        expression: p - shift(p, over=snapshot, offset=1) <= ramp_limit_up * p_nom

      ramp_down:
        foreach: [snapshot, generator]
        expression: shift(p, over=snapshot, offset=1) - p <= ramp_limit_down * p_nom

      energy_balance:
        description: >-
          the level carried into a snapshot, decayed, plus what was stored and less
          what was taken — and it wraps at the horizon, so the first snapshot
          inherits from the last
        foreach: [snapshot, storage]
        expression: >-
          soc == shift(soc, over=snapshot, offset=1, edge='wrap') * (1 - standing_loss)
          + p_store * efficiency_store
          - p_dispatch / efficiency_dispatch

    objective:
      sense: minimize
      description: total cost of generation; storage and transmission are free here
      expression: sum(p * marginal_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/ports/pypsa_cyclic_storage.yaml', sources) as solution:
        solution.objective  # 17228.77962151063
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_cyclic_storage.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the lpspec call binds as ``sources``.

        ``max_hours`` is the ratio PyPSA stores; the port carries the product it
        implies (``soc_max``), because a bound there takes a name, not arithmetic.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.add('Bus', tables['bus']['bus'])

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        links: pd.DataFrame = tables['link'].set_index('link')
        storages: pd.DataFrame = tables['storage'].set_index('storage')

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
        p_nom: pd.Series = tables['storage_p_nom'].set_index('storage')['value']
        n.add(
            'StorageUnit',
            storages.index,
            bus=storages['storage_bus'],
            p_nom=p_nom,
            max_hours=tables['soc_max'].set_index('storage')['value'] / p_nom,
            efficiency_store=tables['efficiency_store'].set_index('storage')['value'],
            efficiency_dispatch=tables['efficiency_dispatch'].set_index('storage')['value'],
            standing_loss=tables['standing_loss'].set_index('storage')['value'],
            cyclic_state_of_charge=True,
        )

        load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
        for bus in tables['bus']['bus']:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
        return n
    ```

## What it exercises

`edge='wrap'`, against rung 3's bare `shift` — plus division by a parameter and the same
five-term `sum(by=)` balance, with one fewer equation and one fewer parameter.
Worth reading the two side by side: neither boundary needs a clause to state it.
The operator names which one is meant, and picking the wrong one is a different
model rather than a missing guard.
