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

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ --- `snapshot` |
| $\mathcal{B}$ | index $b$ --- `bus` |
| $\mathcal{G}$ | index $g$ --- `generator` with $\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}$ |
| $\mathcal{L}$ | index $l$ --- `line` with $\mathrm{from}: \mathcal{L} \to \mathcal{B},\enspace \mathrm{to}: \mathcal{L} \to \mathcal{B}$ |
| $\mathcal{C}$ | index $c$ --- `cycle` |

#### Parameters

| Symbol | Meaning |
|---|---|
| $p^{\mathrm{nom}}$ | `p_nom` over $\mathcal{G}$ |
| $\mathit{marginal\_cost}$ | `marginal_cost` over $\mathcal{G}$ |
| $s^{\mathrm{nom}}$ | `s_nom` over $\mathcal{L}$ |
| $\mathit{neg\_s\_nom}$ | `neg_s_nom` over $\mathcal{L}$ |
| $\mathit{cycle}^{\mathrm{incidence}}$ | `cycle_incidence` over $\mathcal{C} \times \mathcal{L}$ |
| $\mathit{load}$ | `load` over $\mathcal{T} \times \mathcal{B}$ |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{T} \times \mathcal{G}$ |
| $f$ | `f` over $\mathcal{T} \times \mathcal{L}$ |

#### Objective

$$\min \sum_{t \in \mathcal{T},\enspace g \in \mathcal{G}} p_{t,g} \cdot \mathit{marginal\_cost}_{g}$$

#### Subject to

**`nodal_balance`**

$$\sum_{g \in \mathcal{G} \thinspace:\thinspace \mathrm{gen\_bus}(g) = b} p_{t,g} + \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{to}(l) = b} f_{t,l} - \left( \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{from}(l) = b} f_{t,l} \right) = \mathit{load}_{t,b} \qquad \forall\thinspace t \in \mathcal{T},\enspace b \in \mathcal{B}$$

**`kirchhoff_voltage_law`**

$$\sum_{l \in \mathcal{L}} f_{t,l} \cdot \mathit{cycle}^{\mathrm{incidence}}_{c,l} = 0 \qquad \forall\thinspace t \in \mathcal{T},\enspace c \in \mathcal{C}$$

#### Variable domains

**`p`**

$$0 \le p_{t,g} \le p^{\mathrm{nom}}_{g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`f`**

$$\mathit{neg\_s\_nom}_{l} \le f_{t,l} \le s^{\mathrm{nom}}_{l} \qquad \forall\thinspace t \in \mathcal{T},\enspace l \in \mathcal{L}$$

</details>
<!-- math:end -->

The tabs start from [the instance’s tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    # PyPSA linear optimal power flow, rung 5: passive AC lines under Kirchhoff's
    # voltage law, rather than links whose flow is chosen.
    # Optimum 17000.0, from PyPSA itself.

    dimensions:
      snapshot:
        dtype: int
      bus:
        dtype: str
      generator:
        dtype: str
      line:
        dtype: str
      cycle:
        dtype: str  # one independent loop of the network

    lookups:
      gen_bus: {over: generator, into: bus}  # every generator sits on a bus
      from: {over: line, into: bus}  # both endpoints are buses
      to: {over: line, into: bus}

    parameters:
      p_nom:
        dims: [generator]
      marginal_cost:
        dims: [generator]
      s_nom:
        dims: [line]
      neg_s_nom:
        dims: [line]
      # The cycle basis, as a sparse (cycle, line) table of reactance x direction.
      # A line may belong to several cycles, so this cannot be a lookup over
      # `line` — a lookup is single-valued per label. It is a parameter over
      # both dims, and rows are absent where a line is not in a cycle.
      cycle_incidence:
        dims: [cycle, line]
      load:
        dims: [snapshot, bus]

    variables:
      p:
        foreach: [snapshot, generator]
        bounds:
          lower: 0
          upper: p_nom
      # A line's flow is not chosen: it is whatever the voltage law leaves.
      f:
        foreach: [snapshot, line]
        bounds:
          lower: neg_s_nom
          upper: s_nom

    constraints:
      nodal_balance:
        foreach: [snapshot, bus]
        expression: >-
          sum(p, by=gen_bus)
          + sum(f, by=to)
          - sum(f, by=from)
          == load

      # Kirchhoff's voltage law: around each independent cycle, the
      # reactance-weighted flows sum to zero. The incidence table carries both
      # which lines are in the cycle and which way round they run, so this is one
      # equation rather than a case analysis over the topology.
      kirchhoff_voltage_law:
        foreach: [snapshot, cycle]
        expression: sum(f * cycle_incidence, over=line) == 0

    objective:
      sense: minimize
      expression: p * marginal_cost
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
            bus0=lines['from'],
            bus1=lines['to'],
            x=tables['reactance'].set_index('line')['value'],
            r=0.0,
            s_nom=tables['s_nom'].set_index('line')['value'],
        )

        load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
        for bus in tables['bus']['bus']:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
        return n
    ```

**The cycle basis is a parameter, not a lookup.** This is the one shape
decision worth reading twice. A line may belong to *several* cycles, and a
declared lookup is single-valued per label — so `cycle_incidence` is a
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
[ceiling](../design/ceiling.md#two-tiers-and-the-ceiling) refuses by design. The
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
