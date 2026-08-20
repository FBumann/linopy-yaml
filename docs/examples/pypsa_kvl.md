# PyPSA LOPF — rung 5, Kirchhoff's voltage law

Passive AC lines: flow is decided by physics, not chosen. **The last rung of the ladder.**

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.9.0)** — objective **17000**, matched to `rtol=1e-09`, nodal prices and line flows included.

Every earlier rung moved power over `Link` objects, whose flow is a decision
variable — a transport model. A `Line` is passive: around every independent
cycle of the network, the reactance-weighted flows must sum to zero.

It builds on [rung 1](pypsa_transport.md) rather than on
[rung 4](pypsa_cyclic_storage.md), and that is deliberate. Rungs 2–4 are
**time**-coupling — ramps, state of charge, a closed horizon. This one is
**space**-coupling. The two axes are independent, so stacking them would only
make a mismatch ambiguous about which caused it.

Three buses in a triangle, so the cycle space has exactly one dimension. The
flows come out fractional — 46.67 / 26.67 / −33.33 at the first snapshot —
because the split is forced by reactance rather than chosen by cost, which is
the whole difference between a line and a link.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA linear optimal power flow, rung 5: passive AC lines under Kirchhoff's voltage law, rather than links whose flow is chosen. Optimum 17000.0, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ --- `snapshot` --- dispatch periods |
| $\mathcal{B}$ | index $b$ --- `bus` --- network nodes |
| $\mathcal{G}$ | index $g$ --- `generator` with $\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}$ --- generating units, each sitting on one bus |
| $\mathcal{L}$ | index $l$ --- `line` with $\mathrm{line\_from}: \mathcal{L} \to \mathcal{B},\enspace \mathrm{line\_to}: \mathcal{L} \to \mathcal{B}$ --- passive AC lines, each joining two buses |
| $\mathcal{C}$ | index $c$ --- `cycle` --- one independent loop of the network |

#### Parameters

| Symbol | Meaning |
|---|---|
| $p^{\mathrm{nom}}$ | `p_nom` over $\mathcal{G}$ --- installed capacity of a generator |
| $\mathit{marginal\_cost}$ | `marginal_cost` over $\mathcal{G}$ --- cost of one unit of output |
| $s^{\mathrm{nom}}$ | `s_nom` over $\mathcal{L}$ --- most a line may carry towards its `line_to` bus |
| $\mathit{neg\_s\_nom}$ | `neg_s_nom` over $\mathcal{L}$ --- most a line may carry the other way, negative by convention |
| $\mathit{cycle}^{\mathrm{incidence}}$ | `cycle_incidence` over $\mathcal{C} \times \mathcal{L}$ --- the cycle basis, as a sparse table of reactance times direction — a line may belong to several cycles, so this is a parameter over both dimensions rather than a coordinate, and rows are absent where a line is in no cycle |
| $\mathit{load}$ | `load` over $\mathcal{T} \times \mathcal{B}$ --- demand at each bus in each snapshot |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{T} \times \mathcal{G}$ --- output of a generator in a snapshot |
| $f$ | `f` over $\mathcal{T} \times \mathcal{L}$ --- flow on a line, signed towards its `line_to` bus — not chosen, but whatever the voltage law leaves |

#### Objective

$$\min \sum_{t \in \mathcal{T},\enspace g \in \mathcal{G}} p_{t,g} \cdot \mathit{marginal\_cost}_{g}$$

#### Subject to

**`nodal_balance`**

$$\sum_{g \in \mathcal{G} \thinspace:\thinspace \mathrm{gen\_bus}(g) = b} p_{t,g} + \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{line\_to}(l) = b} f_{t,l} - \left( \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{line\_from}(l) = b} f_{t,l} \right) = \mathit{load}_{t,b} \qquad \forall\thinspace t \in \mathcal{T},\enspace b \in \mathcal{B}$$

**`kirchhoff_voltage_law`**

$$\sum_{l \in \mathcal{L}} f_{t,l} \cdot \mathit{cycle}^{\mathrm{incidence}}_{c,l} = 0 \qquad \forall\thinspace t \in \mathcal{T},\enspace c \in \mathcal{C}$$

#### Variable domains

**`p`**

$$0 \le p_{t,g} \le p^{\mathrm{nom}}_{g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`f`**

$$\mathit{neg\_s\_nom}_{l} \le f_{t,l} \le s^{\mathrm{nom}}_{l} \qquad \forall\thinspace t \in \mathcal{T},\enspace l \in \mathcal{L}$$

</details>
<!-- math:end -->

The tabs start from [the instance's tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    description: >-
      PyPSA linear optimal power flow, rung 5: passive AC lines under Kirchhoff's
      voltage law, rather than links whose flow is chosen. Optimum 17000.0, from
      PyPSA itself.

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
      line:
        description: passive AC lines, each joining two buses
        dtype: str
      cycle:
        description: one independent loop of the network
        dtype: str

    lookups:
      gen_bus:
        description: the bus a generator sits on
        over: generator
        into: bus
      line_from:
        description: the bus a line leaves
        over: line
        into: bus
      line_to:
        description: the bus a line arrives at
        over: line
        into: bus

    parameters:
      p_nom:
        description: installed capacity of a generator
        dims: [generator]
      marginal_cost:
        description: cost of one unit of output
        dims: [generator]
      s_nom:
        description: most a line may carry towards its `line_to` bus
        dims: [line]
      neg_s_nom:
        description: most a line may carry the other way, negative by convention
        dims: [line]
      cycle_incidence:
        description: >-
          the cycle basis, as a sparse table of reactance times direction — a line
          may belong to several cycles, so this is a parameter over both dimensions
          rather than a coordinate, and rows are absent where a line is in no cycle
        dims: [cycle, line]
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
        description: >-
          flow on a line, signed towards its `line_to` bus — not chosen, but whatever
          the voltage law leaves
        foreach: [snapshot, line]
        bounds:
          lower: neg_s_nom
          upper: s_nom

    constraints:
      nodal_balance:
        description: what is generated at a bus plus what arrives over the lines meets the load there
        foreach: [snapshot, bus]
        expression: >-
          sum(p, by=gen_bus)
          + sum(f, by=line_to)
          - sum(f, by=line_from)
          == load

      kirchhoff_voltage_law:
        description: >-
          around each independent cycle the reactance-weighted flows sum to zero.
          The incidence table carries both which lines are in the cycle and which
          way round they run, so this is one equation rather than a case analysis
          over the topology — and a coordinate could not hold it, being
          single-valued per label.
        foreach: [snapshot, cycle]
        expression: sum(f * cycle_incidence, over=line) == 0

    objective:
      sense: minimize
      description: total cost of generation; the lines carry power for nothing
      expression: sum(p * marginal_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/ports/pypsa_kvl.yaml', sources) as solution:
        solution.objective  # 17000.0
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_kvl.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the lpspec call binds as ``sources``.

        ``r=0`` keeps a line purely reactive: the linearised power flow is a
        function of ``x`` alone, and a resistance would only add losses the DC
        approximation does not model anyway.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.add('Bus', tables['bus']['bus'])

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        lines: pd.DataFrame = tables['line'].set_index('line')

        n.add(
            'Generator',
            generators.index,
            bus=generators['gen_bus'],
            p_nom=tables['p_nom'].set_index('generator')['value'],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
        )
        n.add(
            'Line',
            lines.index,
            bus0=lines['line_from'],
            bus1=lines['line_to'],
            x=tables['reactance'].set_index('line')['value'],
            r=0.0,
            s_nom=tables['s_nom'].set_index('line')['value'],
        )

        load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
        for bus in tables['bus']['bus']:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
        return n
    ```

**The cycle basis is a parameter, not a coordinate.** This is the one shape
decision worth reading twice. A line may belong to *several* cycles, and a
declared coordinate is single-valued per label — so `cycle_incidence` is a
parameter over `(cycle, line)` carrying reactance × direction, with rows simply
absent where a line is not in a cycle. Row absence is how this language spells
sparsity everywhere else, and a cycle-line incidence matrix is exactly the
sparse thing it is good at.

That makes the constraint one equation, `sum(f * cycle_incidence, over=line) ==
0`, rather than a case analysis over the topology — and it keeps
[topology as data](pypsa_transport.md): a fourth bus is more rows, not a
different file.

**Computing the basis is data preparation, and stays outside the language.**
Finding a cycle basis is a graph algorithm — iteration over a structure
discovered from data, which the
[ceiling](../about/ceiling.md#two-tiers-and-the-ceiling) refuses by design. The
reference prints the rows PyPSA derived so the two can be checked against each
other. They need only agree on the *cycle space*: PyPSA scales its coefficients
for conditioning, and since the row is `= 0`, any nonzero multiple of a cycle
says the same thing.

## What it exercises

A parameter over two dimensions multiplying a variable over one, reduced along
the shared dimension — the shape that makes an incidence matrix sayable at all.
Plus `sum(by=)` on both line endpoints for the nodal balance, as in rung 1.

No new construct was needed for the last rung of the ladder, which is the
result worth reporting.
