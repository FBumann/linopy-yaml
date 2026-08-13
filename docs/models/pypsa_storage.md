# PyPSA LOPF — rung 3, storage

[Rung 2](pypsa_ramp.md) plus a `StorageUnit` carrying energy between snapshots.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.9.0)** — objective **15253.178322993519**, matched to `rtol=1e-09`.

Non-cyclic: the horizon starts at `soc_initial` and its end is free. Closing
that loop is [rung 4](pypsa_cyclic_storage.md), kept separate so it can fail on
its own.

The battery sits at `south`, where the expensive oil generator is. It displaces
oil **entirely** — every snapshot runs oil at zero — and drains to empty by the
third, which is what a free end-of-horizon buys you.

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
| $\mathcal{S}$ | index $s$ --- `storage` with $\mathrm{bus}: \mathcal{S} \to \mathcal{B}$ |

#### Parameters

| Symbol | Meaning |
|---|---|
| $p^{\mathrm{nom}}$ | `p_nom` over $\mathcal{G}$ |
| $\mathit{marginal\_cost}$ | `marginal_cost` over $\mathcal{G}$ |
| $\mathit{ramp\_limit\_up}$ | `ramp_limit_up` over $\mathcal{G}$ |
| $\mathit{ramp\_limit\_down}$ | `ramp_limit_down` over $\mathcal{G}$ |
| $\mathit{rating}$ | `rating` over $\mathcal{L}$ |
| $\mathit{neg\_rating}$ | `neg_rating` over $\mathcal{L}$ |
| $\mathit{storage}^{\mathrm{p,nom}}$ | `storage_p_nom` over $\mathcal{S}$ |
| $\mathit{soc}^{\mathrm{max}}$ | `soc_max` over $\mathcal{S}$ |
| $\mathit{soc}^{\mathrm{initial}}$ | `soc_initial` over $\mathcal{S}$ |
| $\mathit{efficiency\_store}$ | `efficiency_store` over $\mathcal{S}$ |
| $\mathit{efficiency\_dispatch}$ | `efficiency_dispatch` over $\mathcal{S}$ |
| $\mathit{standing\_loss}$ | `standing_loss` over $\mathcal{S}$ |
| $\mathit{load}$ | `load` over $\mathcal{T} \times \mathcal{B}$ |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{T} \times \mathcal{G}$ |
| $f$ | `f` over $\mathcal{T} \times \mathcal{L}$ |
| $p^{\mathrm{dispatch}}$ | `p_dispatch` over $\mathcal{T} \times \mathcal{S}$ |
| $p^{\mathrm{store}}$ | `p_store` over $\mathcal{T} \times \mathcal{S}$ |
| $\mathit{soc}$ | `soc` over $\mathcal{T} \times \mathcal{S}$ |

#### Objective

**`total_cost`**

$$\min \sum_{t \in \mathcal{T},\enspace g \in \mathcal{G}} p_{t,g} \cdot \mathit{marginal\_cost}_{g}$$

#### Subject to

**`nodal_balance`**

$$\sum_{g \in \mathcal{G} \thinspace:\thinspace \mathrm{bus}(g) = b} p_{t,g} + \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{to}(l) = b} f_{t,l} - \left( \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{from}(l) = b} f_{t,l} \right) + \sum_{s \in \mathcal{S} \thinspace:\thinspace \mathrm{bus}(s) = b} p^{\mathrm{dispatch}}_{t,s} - \left( \sum_{s \in \mathcal{S} \thinspace:\thinspace \mathrm{bus}(s) = b} p^{\mathrm{store}}_{t,s} \right) = \mathit{load}_{t,b} \qquad \forall\thinspace t \in \mathcal{T},\enspace b \in \mathcal{B}$$

**`ramp_up`**

$$p_{t,g} - p_{t - 1,g} \le \mathit{ramp\_limit\_up}_{g} \cdot p^{\mathrm{nom}}_{g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`ramp_down`**

$$p_{t - 1,g} - p_{t,g} \le \mathit{ramp\_limit\_down}_{g} \cdot p^{\mathrm{nom}}_{g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`energy_balance_initial`**

$$\mathit{soc}_{t,s} = \mathit{soc}^{\mathrm{initial}}_{s} + p^{\mathrm{store}}_{t,s} \cdot \mathit{efficiency\_store}_{s} - \frac{p^{\mathrm{dispatch}}_{t,s}}{\mathit{efficiency\_dispatch}_{s}} \qquad \forall\thinspace t \in \mathcal{T},\enspace s \in \mathcal{S} \thinspace:\thinspace t = 0$$

**`energy_balance`**

$$\mathit{soc}_{t,s} = \mathit{soc}_{t - 1,s} \cdot \left( 1 - \mathit{standing\_loss}_{s} \right) + p^{\mathrm{store}}_{t,s} \cdot \mathit{efficiency\_store}_{s} - \frac{p^{\mathrm{dispatch}}_{t,s}}{\mathit{efficiency\_dispatch}_{s}} \qquad \forall\thinspace t \in \mathcal{T},\enspace s \in \mathcal{S}$$

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

=== "lpspec"

    ```yaml
    # PyPSA linear optimal power flow, rung 3: rung 2 plus a storage unit carrying
    # energy between snapshots. Non-cyclic — the horizon starts at soc_initial and
    # ends free. Optimum 15253.178322993519, from PyPSA itself. See docs/models/index.md.

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
      storage:
        dtype: str
        coords: [bus]  # a storage unit sits on a bus too

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
      storage_p_nom:
        dims: [storage]
      # p_nom * max_hours in PyPSA. Carried as a column because a bound takes a
      # name or a number, never arithmetic (#31) — the third model to want that.
      soc_max:
        dims: [storage]
      soc_initial:
        dims: [storage]
      efficiency_store:
        dims: [storage]
      efficiency_dispatch:
        dims: [storage]
      standing_loss:
        dims: [storage]
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
      # PyPSA splits a storage unit's power into two non-negative variables rather
      # than one signed one, so the two efficiencies can differ.
      p_dispatch:
        foreach: [snapshot, storage]
        bounds:
          lower: 0
          upper: storage_p_nom
      p_store:
        foreach: [snapshot, storage]
        bounds:
          lower: 0
          upper: storage_p_nom
      soc:
        foreach: [snapshot, storage]
        bounds:
          lower: 0
          upper: soc_max

    constraints:
      nodal_balance:
        foreach: [snapshot, bus]
        expression: >-
          sum(p, over=generator, group_by=bus)
          + sum(f, over=link, group_by=to)
          - sum(f, over=link, group_by=from)
          + sum(p_dispatch, over=storage, group_by=bus)
          - sum(p_store, over=storage, group_by=bus)
          == load

      ramp_up:
        foreach: [snapshot, generator]
        expression: p - shift(p, over=snapshot, by=1) <= ramp_limit_up * p_nom

      ramp_down:
        foreach: [snapshot, generator]
        expression: shift(p, over=snapshot, by=1) - p <= ramp_limit_down * p_nom

      # Charging is derated on the way in and discharging on the way out, so the
      # two efficiencies enter on opposite sides of the division. `standing_loss`
      # decays only what was *carried over* — PyPSA does not apply it to
      # soc_initial, which is why the first snapshot is its own equation rather
      # than a carry-over with a seeded value.
      energy_balance_initial:
        foreach: [snapshot, storage]
        where: "snapshot == 0"
        expression: >-
          soc == soc_initial
          + p_store * efficiency_store
          - p_dispatch / efficiency_dispatch

      energy_balance:
        foreach: [snapshot, storage]
        expression: >-
          soc == shift(soc, over=snapshot, by=1) * (1 - standing_loss)
          + p_store * efficiency_store
          - p_dispatch / efficiency_dispatch

    objectives:
      total_cost:
        sense: minimize
        expression: p * marginal_cost
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/ports/pypsa_storage.yaml', sources) as solution:
        solution.objective  # 15253.178322993519
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_storage.py`:

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

        generators = tables['generator'].set_index('generator')
        links = tables['link'].set_index('link')
        storages = tables['storage'].set_index('storage')

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
        p_nom = tables['storage_p_nom'].set_index('storage')['value']
        n.add(
            'StorageUnit',
            storages.index,
            bus=storages['bus'],
            p_nom=p_nom,
            max_hours=tables['soc_max'].set_index('storage')['value'] / p_nom,
            state_of_charge_initial=tables['soc_initial'].set_index('storage')['value'],
            efficiency_store=tables['efficiency_store'].set_index('storage')['value'],
            efficiency_dispatch=tables['efficiency_dispatch'].set_index('storage')['value'],
            standing_loss=tables['standing_loss'].set_index('storage')['value'],
            cyclic_state_of_charge=False,
        )

        load = tables['load'].pivot(index='snapshot', columns='bus', values='value')
        for bus in tables['bus']['bus']:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
        return n
    ```

**Two efficiencies, on opposite sides of the division.** PyPSA splits a storage
unit's power into two non-negative variables rather than one signed one,
precisely so charging and discharging can be derated differently:
`p_store * efficiency_store` on the way in, `p_dispatch / efficiency_dispatch`
on the way out.

**`standing_loss` decays only what was carried over.** PyPSA does not apply it
to `soc_initial`, so the first snapshot is its own equation rather than a
carry-over with a seeded value. Applying the loss to the seed as well — a one-token
change, and the reading most people would call obvious — moves the objective to
**15272.957445031367**, about 20 out of 15253. Wrong by 0.13%: far too small to
notice by eye on a plot, far too large to be rounding. That gap is the entire
argument for checking against somebody else's number instead of against a
result that merely looks sensible.

## What it exercises

`shift` across a boundary condition, division of a variable by a parameter, and
a five-term `sum(group_by=)` nodal balance — generators, both ends of every link,
and both directions of storage, all projected onto `bus`.

It also asks for [#31](https://github.com/fluxopt/lpspec/issues/31) a third
time: `soc_max` is `p_nom × max_hours` in PyPSA, and a bound here takes a name
or a number, so the product ships as a column.
