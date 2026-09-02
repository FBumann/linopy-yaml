# PyPSA global limits — four bounds, one shape

Limits that hold over a whole set at once: an energy total, a capacity at one bus, and the built network measured twice.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.9.0)** — objective **127211.66666666666**, matched to `rtol=1e-09`.

Every other bound in this corpus belongs to a component. A *global* limit
belongs to a set PyPSA selects by an attribute: all generators burning gas, all
wind capacity at one bus, every extendable link. [AC-DC](pypsa_ac_dc.md) ports one of them — the CO₂ cap — but that one groups nothing, being a single bound
over everything.

Four limits here, and each is the same sentence in the language: **a sum over a
group, against a bound only some rows carry**.

| PyPSA | what it selects | the port's row |
|---|---|---|
| `operational_limit` | generators burning a carrier | `sum(sum(p, by=gen_carrier), over=snapshot) <= energy_cap` |
| per-`(bus, carrier)` capacity cap | that carrier's capacity at one bus | `sum(p_nom, by=[gen_bus, gen_carrier]) <= bus_capacity_cap` |
| `transmission_volume_expansion_limit` | extendable links, weighted by length | `sum(link_p_nom * link_length, over=link) <= volume_cap` |
| `transmission_expansion_cost_limit` | the same links, weighted by money | `sum(link_p_nom * link_capital_cost, over=link) <= expansion_cost_cap` |

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA's global constraints: four limits over four different selected sets — the energy a carrier may deliver, the capacity a carrier may hold at one bus, and the built transmission measured twice, once as length and once as money. Each is one grouped sum against a bound only some rows carry. Optimum 127211.66666666666, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ — `snapshot` — dispatch periods |
| $\mathcal{B}$ | index $b$ — `bus` — network nodes |
| $\mathcal{C}$ | index $c$ — `carrier` — what a generator burns, and what a global limit selects on |
| $\mathcal{E}$ | index $e$ — `generator` with $\mathrm{gen\_bus}: \mathcal{E} \to \mathcal{B},\enspace \mathrm{gen\_carrier}: \mathcal{E} \to \mathcal{C}$ — generating units, each sitting on a bus and burning a carrier |
| $\mathcal{L}$ | index $l$ — `link` with $\mathrm{link\_from}: \mathcal{L} \to \mathcal{B},\enspace \mathrm{link\_to}: \mathcal{L} \to \mathcal{B}$ — controllable connections, each joining two buses |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\mathrm{load}$ | `load` over $\mathcal{T} \times \mathcal{B}$ — demand at each bus in each snapshot |
| $\mathrm{p}^{\mathrm{max,pu}}$ | `p_max_pu` over $\mathcal{T} \times \mathcal{E}$ — share of built capacity a generator can produce in a snapshot |
| $\mathrm{marginal\_cost}$ | `marginal_cost` over $\mathcal{E}$ — cost of one unit of output |
| $\mathrm{gen\_capital\_cost}$ | `gen_capital_cost` over $\mathcal{E}$ — annualised cost of a unit of generator capacity |
| $\mathrm{link\_capital\_cost}$ | `link_capital_cost` over $\mathcal{L}$ — annualised cost of a unit of link capacity |
| $\mathrm{link\_length}$ | `link_length` over $\mathcal{L}$ — how far a link reaches — what turns built capacity into a volume |
| $\mathrm{energy\_cap}$ | `energy_cap` over $\mathcal{C}$ — energy a carrier may deliver over the whole horizon, for the carriers that have such a limit |
| $\mathrm{bus\_capacity\_cap}$ | `bus_capacity_cap` over $\mathcal{B} \times \mathcal{C}$ — capacity of one carrier a bus may hold, for the pairs that cap it — PyPSA writes the carrier into a column name (`nom_max_wind`), so the pair is the limit's own key |
| $\mathrm{volume\_cap}$ | `volume_cap` (scalar) — capacity times length the whole network may build |
| $\mathrm{expansion\_cost\_cap}$ | `expansion_cost_cap` (scalar) — money the whole network may spend building links |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{T} \times \mathcal{E}$ — output of a generator in a snapshot |
| $p^{\mathrm{nom}}$ | `p_nom` over $\mathcal{E}$ — capacity built at a generator |
| $g$ | `g` over $\mathcal{T} \times \mathcal{L}$ — flow on a link, towards the bus it delivers at |
| $\mathit{link\_p\_nom}$ | `link_p_nom` over $\mathcal{L}$ — capacity built on a link |

Upright is what the model is given — a parameter such as $\mathrm{load}$, a coordinate map, a label — and italic is what the solver chooses, such as $p$. An index is italic too, being what a quantifier chooses, and a set is script.

#### Objective

$$\min \sum_{t \in \mathcal{T},\enspace e \in \mathcal{E}} p_{t,e} \cdot \mathrm{marginal\_cost}_{e} + \sum_{e \in \mathcal{E}} p^{\mathrm{nom}}_{e} \cdot \mathrm{gen\_capital\_cost}_{e} + \sum_{l \in \mathcal{L}} \mathit{link\_p\_nom}_{l} \cdot \mathrm{link\_capital\_cost}_{l}$$

#### Subject to

**`within_capacity`**

$$p_{t,e} \le p^{\mathrm{nom}}_{e} \cdot \mathrm{p}^{\mathrm{max,pu}}_{t,e} \qquad \forall\thinspace t \in \mathcal{T},\enspace e \in \mathcal{E}$$

**`within_link_capacity`**

$$g_{t,l} \le \mathit{link\_p\_nom}_{l} \qquad \forall\thinspace t \in \mathcal{T},\enspace l \in \mathcal{L}$$

**`nodal_balance`**

$$\sum_{e \in \mathcal{E} \thinspace:\thinspace \mathrm{gen\_bus}(e) = b} p_{t,e} + \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{link\_to}(l) = b} g_{t,l} - \left( \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{link\_from}(l) = b} g_{t,l} \right) = \mathrm{load}_{t,b} \qquad \forall\thinspace t \in \mathcal{T},\enspace b \in \mathcal{B}$$

**`carrier_energy`**

$$\sum_{t \in \mathcal{T}} \sum_{e \in \mathcal{E} \thinspace:\thinspace \mathrm{gen\_carrier}(e) = c} p_{t,e} \le \mathrm{energy\_cap}_{c} \qquad \forall\thinspace c \in \mathcal{C} \thinspace:\thinspace \mathrm{energy\_cap}_{c} \text{ is defined}$$

**`carrier_capacity_at_bus`**

$$\sum_{e \in \mathcal{E} \thinspace:\thinspace \mathrm{gen\_bus}(e) = b \wedge \mathrm{gen\_carrier}(e) = c} p^{\mathrm{nom}}_{e} \le \mathrm{bus\_capacity\_cap}_{b,c} \qquad \forall\thinspace b \in \mathcal{B},\enspace c \in \mathcal{C} \thinspace:\thinspace \mathrm{bus\_capacity\_cap}_{b,c} \text{ is defined}$$

**`transmission_volume`**

$$\sum_{l \in \mathcal{L}} \mathit{link\_p\_nom}_{l} \cdot \mathrm{link\_length}_{l} \le \mathrm{volume\_cap}$$

**`transmission_cost`**

$$\sum_{l \in \mathcal{L}} \mathit{link\_p\_nom}_{l} \cdot \mathrm{link\_capital\_cost}_{l} \le \mathrm{expansion\_cost\_cap}$$

#### Variable domains

**`p`**

$$p_{t,e} \ge 0 \qquad \forall\thinspace t \in \mathcal{T},\enspace e \in \mathcal{E}$$

**`p_nom`**

$$p^{\mathrm{nom}}_{e} \ge 0 \qquad \forall\thinspace e \in \mathcal{E}$$

**`g`**

$$g_{t,l} \ge 0 \qquad \forall\thinspace t \in \mathcal{T},\enspace l \in \mathcal{L}$$

**`link_p_nom`**

$$\mathit{link\_p\_nom}_{l} \ge 0 \qquad \forall\thinspace l \in \mathcal{L}$$

</details>
<!-- math:end -->

The tabs start from [the instance's tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    description: >-
      PyPSA's global constraints: four limits over four different selected sets —
      the energy a carrier may deliver, the capacity a carrier may hold at one bus,
      and the built transmission measured twice, once as length and once as money.
      Each is one grouped sum against a bound only some rows carry.
      Optimum 127211.66666666666, from PyPSA itself.

    dimensions:
      snapshot:
        description: dispatch periods
        dtype: int
      bus:
        description: network nodes
        dtype: str
      carrier:
        description: what a generator burns, and what a global limit selects on
        dtype: str
      generator:
        description: generating units, each sitting on a bus and burning a carrier
        dtype: str
      link:
        description: controllable connections, each joining two buses
        dtype: str

    lookups:
      gen_bus:
        description: the bus a generator sits on
        over: generator
        into: bus
      gen_carrier:
        description: the carrier a generator burns
        over: generator
        into: carrier
      link_from:
        description: the bus a link leaves
        over: link
        into: bus
      link_to:
        description: the bus a link arrives at
        over: link
        into: bus

    parameters:
      load:
        description: demand at each bus in each snapshot
        dims: [snapshot, bus]
      p_max_pu:
        description: share of built capacity a generator can produce in a snapshot
        dims: [snapshot, generator]
      marginal_cost:
        description: cost of one unit of output
        dims: [generator]
      gen_capital_cost:
        description: annualised cost of a unit of generator capacity
        dims: [generator]
      link_capital_cost:
        description: annualised cost of a unit of link capacity
        dims: [link]
      link_length:
        description: how far a link reaches — what turns built capacity into a volume
        dims: [link]
      energy_cap:
        coverage: masked
        description: >-
          energy a carrier may deliver over the whole horizon, for the carriers that
          have such a limit
        dims: [carrier]
      bus_capacity_cap:
        coverage: masked
        description: >-
          capacity of one carrier a bus may hold, for the pairs that cap it — PyPSA
          writes the carrier into a column name (`nom_max_wind`), so the pair is the
          limit's own key
        dims: [bus, carrier]
      volume_cap:
        description: capacity times length the whole network may build
        dims: []
      expansion_cost_cap:
        description: money the whole network may spend building links
        dims: []

    variables:
      p:
        description: output of a generator in a snapshot
        foreach: [snapshot, generator]
        bounds:
          lower: 0
      p_nom:
        description: capacity built at a generator
        foreach: [generator]
        bounds:
          lower: 0
      g:
        description: flow on a link, towards the bus it delivers at
        foreach: [snapshot, link]
        bounds:
          lower: 0
      link_p_nom:
        description: capacity built on a link
        foreach: [link]
        bounds:
          lower: 0

    constraints:
      within_capacity:
        description: a generator produces no more than the built capacity available to it
        foreach: [snapshot, generator]
        expression: p <= p_nom * p_max_pu

      within_link_capacity:
        foreach: [snapshot, link]
        expression: g <= link_p_nom

      nodal_balance:
        description: what is generated at a bus plus what arrives over the links meets the load there
        foreach: [snapshot, bus]
        expression: >-
          sum(p, by=gen_bus)
          + sum(g, by=link_to) - sum(g, by=link_from)
          == load

      carrier_energy:
        description: >-
          a carrier with an energy limit delivers no more than it over the horizon —
          the generators are grouped onto the carrier they burn, which is the
          selection PyPSA makes by querying its own table
        foreach: [carrier]
        where: energy_cap
        expression: sum(sum(p, by=gen_carrier), over=snapshot) <= energy_cap

      carrier_capacity_at_bus:
        description: >-
          a bus that caps a carrier holds no more of it than that — one grouping
          lands the built capacity on the (bus, carrier) pair the limit is keyed by,
          so no selector column and no mask stand between the two
        foreach: [bus, carrier]
        where: bus_capacity_cap
        expression: sum(p_nom, by=[gen_bus, gen_carrier]) <= bus_capacity_cap

      transmission_volume:
        description: >-
          capacity times length, summed over the links — a limit on how much network
          is built, in the unit a planner is granted
        foreach: []
        expression: sum(link_p_nom * link_length, over=link) <= volume_cap

      transmission_cost:
        description: >-
          the same set weighted by money instead of distance, which is why the two
          limits are not each other: the long link is the cheap one
        foreach: []
        expression: sum(link_p_nom * link_capital_cost, over=link) <= expansion_cost_cap

    objective:
      sense: minimize
      description: what the fleet costs to run, plus what the generation and link capacity cost to build
      expression: >-
        sum(p * marginal_cost)
        + sum(p_nom * gen_capital_cost)
        + sum(link_p_nom * link_capital_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/ports/pypsa_global_limits.yaml', sources) as solution:
        solution.objective  # 127211.66666666666
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_global_limits.py`:

    ```python
    def build(
        tables: dict[str, pd.DataFrame],
        limits: dict[str, dict[str, object]] | None = None,
        bus_capacity_cap: bool = True,
    ) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the lpspec call attaches as ``sources``.

        ``limits`` defaults to all three global-constraint rows and
        ``bus_capacity_cap`` to on; dropping one is how ``main`` measures what it is
        worth. Every generator and every link is extendable, because a limit on
        capacity has nothing to bind on a component whose capacity is data.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.add('Bus', tables['bus']['bus'])
        n.add('Carrier', tables['carrier']['carrier'])
        n.add('Carrier', 'DC')

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        p_max_pu: pd.DataFrame = tables['p_max_pu'].pivot(index='snapshot', columns='generator', values='value')
        n.add(
            'Generator',
            generators.index,
            bus=generators['gen_bus'],
            carrier=generators['gen_carrier'],
            p_nom_extendable=True,
            p_max_pu=p_max_pu[generators.index],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
            capital_cost=tables['gen_capital_cost'].set_index('generator')['value'],
        )

        links: pd.DataFrame = tables['link'].set_index('link')
        n.add(
            'Link',
            links.index,
            bus0=links['link_from'],
            bus1=links['link_to'],
            carrier='DC',
            p_nom_extendable=True,
            length=tables['link_length'].set_index('link')['value'],
            capital_cost=tables['link_capital_cost'].set_index('link')['value'],
        )

        load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
        for bus in tables['bus']['bus']:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])

        for name, attributes in (LIMITS if limits is None else limits).items():
            n.add('GlobalConstraint', name, **attributes)

        if bus_capacity_cap:
            bus, column, cap = BUS_CAPACITY_CAP
            n.buses.loc[bus, column] = cap
        return n
    ```

**All four bind, and each is worth something.** Dropping one at a time, on the
same instance: the gas energy cap is worth **7411.67**, the volume limit
**541.67**, the cost limit **176.67**, and the cap on wind at `east` **283.89**.
A global limit that does not bind proves nothing about the language, so the
reference drops each in turn and prints the four numbers.

**The two link limits are not each other.** `north_south` is 100 km at 200/MW,
`east_south` 50 km at 400/MW, so length and money rank the two links in
*opposite* orders. Both limits are tight at the optimum, which fixes the build
exactly: 27.83 MW and 12.33 MW solve `100a + 50b = 3400` and
`200a + 400b = 10500` together.

**The selection is data, not a construct.** PyPSA selects by querying its own
tables — `carrier == "gas"` — and writes the per-bus one into a *column name*,
`nom_max_wind`. That column name is a `(bus, carrier)` pair, and the port says
so: one grouping through both maps lands the built capacity on exactly that
pair, and `bus_capacity_cap` is a table keyed by it.

The alternative is what this port shipped before
[#704](https://github.com/fluxopt/lpspec/issues/704): a 0/1 `capped_carrier`
column pulled down with `at()` and multiplied in, which caps **one** carrier
and re-spells the `gen_carrier` lookup as data a second time. Losing it is the
point — a lookup's values are checked against the dimension they target when
they are bound, where a parameter's are not.

## The fifth limit, which PyPSA does not build

`tech_capacity_expansion_limit` — a carrier's capacity across the whole network
— is missing from the table above, and not because the language cannot say it.
In pypsa 1.2.4 a single-period network cannot get one built at all:

```
no investment_period   emits no constraint at all
investment_period=0    raises ValueError: Investment period not in `n.investment_periods`
```

`global_constraints.py:48` groups the rows by
`["carrier_attribute", "sense", "investment_period"]`; where no period is given
that key is `NaN`, pandas drops NaN keys, and the row leaves no constraint
behind. The next line reads `period = None if isnan(period) else int(period)`,
so NaN is plainly expected to arrive — which makes this theirs to fix rather
than ours to work around, and
[#966](https://github.com/fluxopt/lpspec/issues/966) tracks reporting it. The
limit itself will be proved by [multi-period
investment](pypsa_multi_period.md), where a period exists to name.

## What it exercises

A reduction over two dimensions at once (`sum(sum(p, by=gen_carrier), over=snapshot)`),
one grouping landing on a pair of dimensions (`sum(p_nom, by=[gen_bus, gen_carrier])`),
and two scalar-bounded sums over one set with different weights. No construct here is new — which is the
result, for five constraints PyPSA implements in five functions.
