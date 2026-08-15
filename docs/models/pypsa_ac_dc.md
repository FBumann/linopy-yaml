# PyPSA LOPF — rung 6, two coordinates on one dimension

A meshed AC–DC network under a CO₂ budget. **PyPSA's own `ac-dc-meshed` example.**

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.9.0)** — objective **18441021.477729216**, matched to `rtol=1e-09`, nodal prices included.

Every earlier rung put a generator on a bus and stopped there. Here a generator
also burns a **carrier**, and both maps do work: the nodal balance groups
generation through `bus`, while the CO₂ budget reads an emission rate back down
through `carrier`. Two coordinates on one dimension, landing on two different
axes.

Nine buses, six generators, seven passive lines and four controllable links
across three sub-networks — the first ported network large enough that the two
kinds of branch matter. Capacity is a decision everywhere, so the model prices
what it builds as well as what it runs.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ --- `snapshot` |
| $\mathcal{B}$ | index $b$ --- `bus` |
| $\mathcal{C}$ | index $c$ --- `carrier` |
| $\mathcal{E}$ | index $e$ --- `generator` with $\mathrm{gen\_bus}: \mathcal{E} \to \mathcal{B},\enspace \mathrm{gen\_carrier}: \mathcal{E} \to \mathcal{C}$ |
| $\mathcal{L}$ | index $l$ --- `line` with $\mathrm{from}: \mathcal{L} \to \mathcal{B},\enspace \mathrm{to}: \mathcal{L} \to \mathcal{B}$ |
| $\mathcal{I}$ | index $i$ --- `link` with $\mathrm{link\_from}: \mathcal{I} \to \mathcal{B},\enspace \mathrm{link\_to}: \mathcal{I} \to \mathcal{B}$ |
| $\mathcal{Y}$ | index $y$ --- `cycle` |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\mathit{load}$ | `load` over $\mathcal{T} \times \mathcal{B}$ |
| $p^{\mathrm{max,pu}}$ | `p_max_pu` over $\mathcal{T} \times \mathcal{E}$ |
| $p^{\mathrm{nom,min}}$ | `p_nom_min` over $\mathcal{E}$ |
| $\mathit{marginal\_cost}$ | `marginal_cost` over $\mathcal{E}$ |
| $\mathit{gen\_capital\_cost}$ | `gen_capital_cost` over $\mathcal{E}$ |
| $\mathit{efficiency}$ | `efficiency` over $\mathcal{E}$ |
| $\mathit{co2\_per\_mwh}$ | `co2_per_mwh` over $\mathcal{C}$ |
| $\mathit{line}^{\mathrm{capital,cost}}$ | `line_capital_cost` over $\mathcal{L}$ |
| $\mathit{link}^{\mathrm{capital,cost}}$ | `link_capital_cost` over $\mathcal{I}$ |
| $\mathit{link}^{\mathrm{p,max,pu}}$ | `link_p_max_pu` over $\mathcal{I}$ |
| $\mathit{link}^{\mathrm{p,min,pu}}$ | `link_p_min_pu` over $\mathcal{I}$ |
| $\mathit{cycle}^{\mathrm{incidence}}$ | `cycle_incidence` over $\mathcal{Y} \times \mathcal{L}$ |
| $\mathit{co2\_limit}$ | `co2_limit` (scalar) |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{T} \times \mathcal{E}$ |
| $p^{\mathrm{nom}}$ | `p_nom` over $\mathcal{E}$ |
| $f$ | `f` over $\mathcal{T} \times \mathcal{L}$ |
| $s^{\mathrm{nom}}$ | `s_nom` over $\mathcal{L}$ |
| $g$ | `g` over $\mathcal{T} \times \mathcal{I}$ |
| $\mathit{link}^{\mathrm{p,nom}}$ | `link_p_nom` over $\mathcal{I}$ |

#### Objective

$$\min \sum_{t \in \mathcal{T},\enspace e \in \mathcal{E},\enspace l \in \mathcal{L},\enspace i \in \mathcal{I}} \left( p_{t,e} \cdot \mathit{marginal\_cost}_{e} + p^{\mathrm{nom}}_{e} \cdot \mathit{gen\_capital\_cost}_{e} + s^{\mathrm{nom}}_{l} \cdot \mathit{line}^{\mathrm{capital,cost}}_{l} + \mathit{link}^{\mathrm{p,nom}}_{i} \cdot \mathit{link}^{\mathrm{capital,cost}}_{i} \right)$$

#### Subject to

**`within_capacity`**

$$p_{t,e} \le p^{\mathrm{nom}}_{e} \cdot p^{\mathrm{max,pu}}_{t,e} \qquad \forall\thinspace t \in \mathcal{T},\enspace e \in \mathcal{E}$$

**`line_upper`**

$$f_{t,l} \le s^{\mathrm{nom}}_{l} \qquad \forall\thinspace t \in \mathcal{T},\enspace l \in \mathcal{L}$$

**`line_lower`**

$$f_{t,l} \ge -s^{\mathrm{nom}}_{l} \qquad \forall\thinspace t \in \mathcal{T},\enspace l \in \mathcal{L}$$

**`link_upper`**

$$g_{t,i} \le \mathit{link}^{\mathrm{p,nom}}_{i} \cdot \mathit{link}^{\mathrm{p,max,pu}}_{i} \qquad \forall\thinspace t \in \mathcal{T},\enspace i \in \mathcal{I}$$

**`link_lower`**

$$g_{t,i} \ge \mathit{link}^{\mathrm{p,nom}}_{i} \cdot \mathit{link}^{\mathrm{p,min,pu}}_{i} \qquad \forall\thinspace t \in \mathcal{T},\enspace i \in \mathcal{I}$$

**`nodal_balance`**

$$\sum_{e \in \mathcal{E} \thinspace:\thinspace \mathrm{gen\_bus}(e) = b} p_{t,e} + \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{to}(l) = b} f_{t,l} - \left( \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{from}(l) = b} f_{t,l} \right) + \sum_{i \in \mathcal{I} \thinspace:\thinspace \mathrm{link\_to}(i) = b} g_{t,i} - \left( \sum_{i \in \mathcal{I} \thinspace:\thinspace \mathrm{link\_from}(i) = b} g_{t,i} \right) = \mathit{load}_{t,b} \qquad \forall\thinspace t \in \mathcal{T},\enspace b \in \mathcal{B}$$

**`kirchhoff_voltage_law`**

$$\sum_{l \in \mathcal{L}} f_{t,l} \cdot \mathit{cycle}^{\mathrm{incidence}}_{y,l} = 0 \qquad \forall\thinspace t \in \mathcal{T},\enspace y \in \mathcal{Y}$$

**`co2_budget`**

$$\sum_{t \in \mathcal{T}} \sum_{e \in \mathcal{E}} \frac{p_{t,e} \cdot \mathit{co2\_per\_mwh}_{\mathrm{gen\_carrier}(e)}}{\mathit{efficiency}_{e}} \le \mathit{co2\_limit}$$

#### Variable domains

**`p`**

$$p_{t,e} \ge 0 \qquad \forall\thinspace t \in \mathcal{T},\enspace e \in \mathcal{E}$$

**`p_nom`**

$$p^{\mathrm{nom}}_{e} \ge p^{\mathrm{nom,min}}_{e} \qquad \forall\thinspace e \in \mathcal{E}$$

**`f`**

$$f_{t,l} \in \mathbb{R} \qquad \forall\thinspace t \in \mathcal{T},\enspace l \in \mathcal{L}$$

**`s_nom`**

$$s^{\mathrm{nom}}_{l} \ge 0 \qquad \forall\thinspace l \in \mathcal{L}$$

**`g`**

$$g_{t,i} \in \mathbb{R} \qquad \forall\thinspace t \in \mathcal{T},\enspace i \in \mathcal{I}$$

**`link_p_nom`**

$$\mathit{link}^{\mathrm{p,nom}}_{i} \ge 0 \qquad \forall\thinspace i \in \mathcal{I}$$

</details>
<!-- math:end -->

The tabs start from [the instance’s tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    # PyPSA linear optimal power flow, rung 6: a meshed AC-DC network whose
    # generators sit on a bus *and* burn a carrier, with a CO2 budget priced
    # through the second map. PyPSA's own `ac-dc-meshed` example.
    # Optimum 18441021.477729216, from PyPSA itself.

    dimensions:
      snapshot:
        dtype: int
      bus:
        dtype: str
      carrier:
        dtype: str
      generator:
        dtype: str
      line:
        dtype: str
      link:
        dtype: str
      cycle:
        dtype: str  # one independent loop per meshed sub-network

    # A generator sits on a bus and burns a carrier: two maps out of one dimension,
    # landing on two different axes. The balance groups through the first; the CO2
    # budget reads a price back down through the second. Passive lines and
    # controllable links are separate axes with separate physics, and both have two
    # ends on the same bus dimension.
    lookups:
      gen_bus:
        description: the bus a generator sits on
        over: generator
        into: bus
      gen_carrier:
        description: the carrier a generator burns
        over: generator
        into: carrier
      from:
        description: the bus a line leaves
        over: line
        into: bus
      to:
        description: the bus a line arrives at
        over: line
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
      load:
        dims: [snapshot, bus]
      p_max_pu:
        dims: [snapshot, generator]
      p_nom_min:
        dims: [generator]
      marginal_cost:
        dims: [generator]
      gen_capital_cost:
        dims: [generator]
      efficiency:
        dims: [generator]
      # Emissions are a property of the carrier, not of the generator burning it.
      co2_per_mwh:
        dims: [carrier]
      line_capital_cost:
        dims: [line]
      link_capital_cost:
        dims: [link]
      link_p_max_pu:
        dims: [link]
      link_p_min_pu:
        dims: [link]
      # The cycle basis, as a sparse (cycle, line) table of impedance x direction.
      # A line may belong to several cycles, so this cannot be a coordinate.
      cycle_incidence:
        dims: [cycle, line]
      co2_limit:
        dims: []

    variables:
      p:
        foreach: [snapshot, generator]
        bounds:
          lower: 0
      p_nom:
        foreach: [generator]
        bounds:
          lower: p_nom_min
      # A line's flow is not chosen: it is whatever the voltage law leaves.
      f:
        foreach: [snapshot, line]
      s_nom:
        foreach: [line]
        bounds:
          lower: 0
      # A link's flow is chosen — that is what makes it a link and not a line.
      g:
        foreach: [snapshot, link]
      link_p_nom:
        foreach: [link]
        bounds:
          lower: 0

    constraints:
      within_capacity:
        foreach: [snapshot, generator]
        expression: p <= p_nom * p_max_pu

      line_upper:
        foreach: [snapshot, line]
        expression: f <= s_nom
      line_lower:
        foreach: [snapshot, line]
        expression: f >= -s_nom
      link_upper:
        foreach: [snapshot, link]
        expression: g <= link_p_nom * link_p_max_pu
      link_lower:
        foreach: [snapshot, link]
        expression: g >= link_p_nom * link_p_min_pu

      nodal_balance:
        foreach: [snapshot, bus]
        expression: >-
          sum(p, over=generator, group_by=gen_bus)
          + sum(f, over=line, group_by=to) - sum(f, over=line, group_by=from)
          + sum(g, over=link, group_by=link_to) - sum(g, over=link, group_by=link_from)
          == load

      kirchhoff_voltage_law:
        foreach: [snapshot, cycle]
        expression: sum(f * cycle_incidence, over=line) == 0

      # PyPSA's primary_energy constraint: a generator's emissions are its output
      # divided by its efficiency, priced at its carrier's rate.
      co2_budget:
        foreach: []
        expression: >-
          sum(sum(p * at(co2_per_mwh, onto=generator, by=gen_carrier) / efficiency, over=generator), over=snapshot)
          <= co2_limit

    objective:
      sense: minimize
      expression: >-
        p * marginal_cost
        + p_nom * gen_capital_cost
        + s_nom * line_capital_cost
        + link_p_nom * link_capital_cost
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/ports/pypsa_ac_dc.yaml', sources) as solution:
        solution.objective  # 18441021.477729216
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_ac_dc.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the lpspec call binds as ``sources``.

        The port carries its cycle basis as ``cycle_incidence`` because computing
        one is a graph algorithm and so data preparation; PyPSA derives its own
        from ``line_x`` / ``line_r``, which is why both are in the instance. The
        two must describe the same cycle space, and the objectives agreeing is
        what says they do.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'].tolist())

        carrier = _series(tables['bus_carrier'], 'bus')
        for bus, ct in carrier.items():
            n.add('Bus', bus, carrier=ct)

        co2 = _series(tables['co2_per_mwh'], 'carrier')
        for name, rate in co2.items():
            n.add('Carrier', name, co2_emissions=rate)

        generators = tables['generator'].set_index('generator')
        p_max_pu = _wide(tables['p_max_pu'], 'generator')
        for g, row in generators.iterrows():
            n.add(
                'Generator',
                g,
                bus=row['bus'],
                carrier=row['carrier'],
                p_nom_extendable=True,
                p_nom=_series(tables['gen_p_nom_existing'], 'generator')[g],
                p_nom_min=_series(tables['p_nom_min'], 'generator')[g],
                marginal_cost=_series(tables['marginal_cost'], 'generator')[g],
                capital_cost=_series(tables['gen_capital_cost'], 'generator')[g],
                efficiency=_series(tables['efficiency'], 'generator')[g],
                p_max_pu=p_max_pu[g],
            )

        for line, ends in tables['line'].set_index('line').iterrows():
            bus0, bus1 = ends['from'], ends['to']
            n.add(
                'Line',
                line,
                bus0=bus0,
                bus1=bus1,
                x=_series(tables['line_x'], 'line')[line],
                r=_series(tables['line_r'], 'line')[line],
                s_nom=_series(tables['line_s_nom_existing'], 'line')[line],
                s_nom_extendable=True,
                capital_cost=_series(tables['line_capital_cost'], 'line')[line],
            )

        for link, ends in tables['link'].set_index('link').iterrows():
            bus0, bus1 = ends['link_from'], ends['link_to']
            n.add(
                'Link',
                link,
                bus0=bus0,
                bus1=bus1,
                p_nom_extendable=True,
                p_nom=_series(tables['link_p_nom_existing'], 'link')[link],
                p_min_pu=_series(tables['link_p_min_pu'], 'link')[link],
                p_max_pu=_series(tables['link_p_max_pu'], 'link')[link],
                capital_cost=_series(tables['link_capital_cost'], 'link')[link],
            )

        load = _wide(tables['load'], 'bus')
        for bus in load.columns:
            if load[bus].any():
                n.add('Load', bus, bus=bus, p_set=load[bus])

        n.add(
            'GlobalConstraint',
            'co2_limit',
            type='primary_energy',
            carrier_attribute='co2_emissions',
            sense='<=',
            constant=float(tables['co2_limit']),
        )
        return n
    ```

**An emission rate is a property of the carrier, and `at()` is how a generator
reads it.** `co2_per_mwh` is dimensioned over `carrier` alone — six generators,
two rates — and `at(co2_per_mwh, onto=generator, by=carrier)` walks the map
backwards to put the right rate beside each generator's output. PyPSA does the
same join through `n.carriers`; the difference is that here the map is declared
once and checked at load.

The alternative is a `(generator, carrier)` incidence table contracted away,
which reaches the same number and no longer says that a generator burns exactly
one fuel.

**The recorded optimum is the system cost, not `n.objective`.** Every component
here is extendable, so PyPSA credits the capital already standing in `p_nom` and
reports the change against that starting point — a *negative* number on this
network. The port has no starting point to credit and states the cost outright,
so the figure recorded is `n.objective + n.objective_constant`. Worth knowing
before comparing any PyPSA capacity-expansion result against anything.

## What it exercises

Two coordinates on one dimension into different targets, and `at()` reading a
parameter that lives only on the coarse end. Beside them, the shapes rungs 1–5
already established: `sum(group_by=)` on both ends of two different branch
dimensions, and a cycle basis as a sparse `(cycle, line)` parameter.

The cycle basis carries **impedance** rather than reactance alone: PyPSA applies
the voltage law with `x` inside an AC sub-network and `r` inside a DC one, and
this network has one meshed loop of each. Which value belongs in the row is
decided in data preparation, where [the ceiling](../design/ceiling.md) puts
graph work — the language sees one incidence table either way.

No new construct was needed.
