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

PyPSA linear optimal power flow, rung 6: a meshed AC-DC network whose generators sit on a bus and burn a carrier, with capacity to build and a CO2 budget priced through the second map: the nodal balance groups through the bus coordinate, and the budget reads an emissions rate back down through the carrier. PyPSA's own ac-dc-meshed example. Optimum 18441021.477729216, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ --- `snapshot` --- dispatch periods |
| $\mathcal{B}$ | index $b$ --- `bus` --- network nodes |
| $\mathcal{C}$ | index $c$ --- `carrier` --- what a generator burns, and what its emissions are a property of |
| $\mathcal{E}$ | index $e$ --- `generator` with $\mathrm{gen\_bus}: \mathcal{E} \to \mathcal{B},\enspace \mathrm{gen\_carrier}: \mathcal{E} \to \mathcal{C}$ --- generating units, each sitting on a bus and burning a carrier — two coordinates on one dimension, landing on two different axes |
| $\mathcal{L}$ | index $l$ --- `line` with $\mathrm{line\_from}: \mathcal{L} \to \mathcal{B},\enspace \mathrm{line\_to}: \mathcal{L} \to \mathcal{B}$ --- passive AC lines, each joining two buses |
| $\mathcal{I}$ | index $i$ --- `link` with $\mathrm{link\_from}: \mathcal{I} \to \mathcal{B},\enspace \mathrm{link\_to}: \mathcal{I} \to \mathcal{B}$ --- controllable connections, each joining two buses |
| $\mathcal{Y}$ | index $y$ --- `cycle` --- one independent loop per meshed sub-network |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\mathit{load}$ | `load` over $\mathcal{T} \times \mathcal{B}$ --- demand at each bus in each snapshot |
| $p^{\mathrm{max,pu}}$ | `p_max_pu` over $\mathcal{T} \times \mathcal{E}$ --- share of built capacity a generator can produce in a snapshot |
| $p^{\mathrm{nom,min}}$ | `p_nom_min` over $\mathcal{E}$ --- capacity a generator already has, and cannot fall below |
| $\mathit{marginal\_cost}$ | `marginal_cost` over $\mathcal{E}$ --- cost of one unit of output |
| $\mathit{gen\_capital\_cost}$ | `gen_capital_cost` over $\mathcal{E}$ --- annualised cost of a unit of generator capacity |
| $\mathit{efficiency}$ | `efficiency` over $\mathcal{E}$ --- share of the carrier's energy a generator turns into output |
| $\mathit{co2\_per\_mwh}$ | `co2_per_mwh` over $\mathcal{C}$ --- emissions per unit of carrier burned, a property of the carrier |
| $\mathit{line}^{\mathrm{capital,cost}}$ | `line_capital_cost` over $\mathcal{L}$ --- annualised cost of a unit of line capacity |
| $\mathit{link}^{\mathrm{capital,cost}}$ | `link_capital_cost` over $\mathcal{I}$ --- annualised cost of a unit of link capacity |
| $\mathit{link}^{\mathrm{p,max,pu}}$ | `link_p_max_pu` over $\mathcal{I}$ --- share of its capacity a link may carry forwards |
| $\mathit{link}^{\mathrm{p,min,pu}}$ | `link_p_min_pu` over $\mathcal{I}$ --- share of its capacity a link may carry backwards, negative by convention |
| $\mathit{cycle}^{\mathrm{incidence}}$ | `cycle_incidence` over $\mathcal{Y} \times \mathcal{L}$ --- the cycle basis, as a sparse table of impedance times direction. A line may belong to several cycles, so this cannot be a coordinate. |
| $\mathit{co2\_limit}$ | `co2_limit` (scalar) --- emissions the whole horizon is allowed |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{T} \times \mathcal{E}$ --- output of a generator in a snapshot |
| $p^{\mathrm{nom}}$ | `p_nom` over $\mathcal{E}$ --- generator capacity to hold, built on top of what already stands |
| $f$ | `f` over $\mathcal{T} \times \mathcal{L}$ --- flow on a line, signed towards its `line_to` bus — not chosen, but whatever the voltage law leaves |
| $s^{\mathrm{nom}}$ | `s_nom` over $\mathcal{L}$ --- line capacity to build |
| $g$ | `g` over $\mathcal{T} \times \mathcal{I}$ --- flow on a link, signed towards the bus it delivers at — chosen, which is what makes it a link and not a line |
| $\mathit{link}^{\mathrm{p,nom}}$ | `link_p_nom` over $\mathcal{I}$ --- link capacity to build |

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

$$\sum_{e \in \mathcal{E} \thinspace:\thinspace \mathrm{gen\_bus}(e) = b} p_{t,e} + \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{line\_to}(l) = b} f_{t,l} - \left( \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{line\_from}(l) = b} f_{t,l} \right) + \sum_{i \in \mathcal{I} \thinspace:\thinspace \mathrm{link\_to}(i) = b} g_{t,i} - \left( \sum_{i \in \mathcal{I} \thinspace:\thinspace \mathrm{link\_from}(i) = b} g_{t,i} \right) = \mathit{load}_{t,b} \qquad \forall\thinspace t \in \mathcal{T},\enspace b \in \mathcal{B}$$

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
    description: >-
      PyPSA linear optimal power flow, rung 6: a meshed AC-DC network whose
      generators sit on a bus and burn a carrier, with capacity to build and a CO2
      budget priced through the second map: the nodal balance groups through the
      bus coordinate, and the budget reads an emissions rate back down through the
      carrier. PyPSA's own ac-dc-meshed example.
      Optimum 18441021.477729216, from PyPSA itself.

    dimensions:
      snapshot:
        description: dispatch periods
        dtype: int
      bus:
        description: network nodes
        dtype: str
      carrier:
        description: what a generator burns, and what its emissions are a property of
        dtype: str
      generator:
        description: >-
          generating units, each sitting on a bus and burning a carrier — two
          coordinates on one dimension, landing on two different axes
        dtype: str
      line:
        description: passive AC lines, each joining two buses
        dtype: str
      link:
        description: controllable connections, each joining two buses
        dtype: str
      cycle:
        description: one independent loop per meshed sub-network
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
      line_from:
        description: the bus a line leaves
        over: line
        into: bus
      line_to:
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
        description: demand at each bus in each snapshot
        dims: [snapshot, bus]
      p_max_pu:
        description: share of built capacity a generator can produce in a snapshot
        dims: [snapshot, generator]
      p_nom_min:
        description: capacity a generator already has, and cannot fall below
        dims: [generator]
      marginal_cost:
        description: cost of one unit of output
        dims: [generator]
      gen_capital_cost:
        description: annualised cost of a unit of generator capacity
        dims: [generator]
      efficiency:
        description: share of the carrier's energy a generator turns into output
        dims: [generator]
      co2_per_mwh:
        description: emissions per unit of carrier burned, a property of the carrier
        dims: [carrier]
      line_capital_cost:
        description: annualised cost of a unit of line capacity
        dims: [line]
      link_capital_cost:
        description: annualised cost of a unit of link capacity
        dims: [link]
      link_p_max_pu:
        description: share of its capacity a link may carry forwards
        dims: [link]
      link_p_min_pu:
        description: share of its capacity a link may carry backwards, negative by convention
        dims: [link]
      cycle_incidence:
        description: >-
          the cycle basis, as a sparse table of impedance times direction. A line
          may belong to several cycles, so this cannot be a coordinate.
        dims: [cycle, line]
      co2_limit:
        description: emissions the whole horizon is allowed
        dims: []

    variables:
      p:
        description: output of a generator in a snapshot
        foreach: [snapshot, generator]
        bounds:
          lower: 0
      p_nom:
        description: generator capacity to hold, built on top of what already stands
        foreach: [generator]
        bounds:
          lower: p_nom_min
      f:
        description: >-
          flow on a line, signed towards its `line_to` bus — not chosen, but whatever
          the voltage law leaves
        foreach: [snapshot, line]
      s_nom:
        description: line capacity to build
        foreach: [line]
        bounds:
          lower: 0
      g:
        description: >-
          flow on a link, signed towards the bus it delivers at — chosen, which is
          what makes it a link and not a line
        foreach: [snapshot, link]
      link_p_nom:
        description: link capacity to build
        foreach: [link]
        bounds:
          lower: 0

    constraints:
      within_capacity:
        description: a generator produces no more than the built capacity available to it
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
        description: >-
          what is generated at a bus plus what arrives over the lines and links
          meets the load there
        foreach: [snapshot, bus]
        expression: >-
          sum(p, by=gen_bus)
          + sum(f, by=line_to) - sum(f, by=line_from)
          + sum(g, by=link_to) - sum(g, by=link_from)
          == load

      kirchhoff_voltage_law:
        description: around each independent cycle the impedance-weighted flows sum to zero
        foreach: [snapshot, cycle]
        expression: sum(f * cycle_incidence, over=line) == 0

      co2_budget:
        description: >-
          PyPSA's primary-energy constraint — a generator's emissions are its
          output divided by its efficiency, priced at its carrier's rate, and the
          horizon's total stays inside the budget
        foreach: []
        expression: >-
          sum(sum(p * at(co2_per_mwh, by=gen_carrier) / efficiency, over=generator), over=snapshot)
          <= co2_limit

    objective:
      sense: minimize
      description: what the fleet costs to run, plus what the generation and network capacity cost to build
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
            bus0, bus1 = ends['line_from'], ends['line_to']
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
two rates — and `at(co2_per_mwh, by=gen_carrier)` walks the map
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

Two lookups over one dimension into different targets, and `at()` reading a
parameter that lives only on the coarse end. Beside them, the shapes rungs 1–5
already established: `sum(by=)` on both ends of two different branch
dimensions, and a cycle basis as a sparse `(cycle, line)` parameter.

The cycle basis carries **impedance** rather than reactance alone: PyPSA applies
the voltage law with `x` inside an AC sub-network and `r` inside a DC one, and
this network has one meshed loop of each. Which value belongs in the row is
decided in data preparation, where [the ceiling](../about/ceiling.md) puts
graph work — the language sees one incidence table either way.

No new construct was needed.
