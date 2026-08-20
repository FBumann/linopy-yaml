# PyPSA link delay — power that arrives later than it left

A shipment is the input shifted along time: withdrawn at one snapshot, delivered at another, derated on the way.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.9.0)** — objective **4311.111111111111**, matched to `rtol=1e-09`.

Every `shift` in the corpus so far relates a variable to *itself* — a ramp
limit, a state of charge. Here it relates two buses' balances: what `port_a`
gives up in snapshot 0 is what `port_b` receives in snapshot 2, times the link's
efficiency.

Two links serve the same demand, and the delay is a column, not a constant:
`ship` takes two snapshots and loses 10%, `wire` arrives at once and loses
nothing. So the first two snapshots at `port_b` have nothing shipped to them yet
and are served by the expensive unit standing beside the load — which is what
makes the delay cost something.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA's delayed link: power withdrawn at one snapshot arrives at a later one, so a shipment is the input shifted along time and derated by the link's efficiency. Two links serve the same demand — one takes two snapshots to arrive, one arrives at once — and the first snapshots, which nothing has reached yet, are served by the expensive unit standing beside the load. Optimum 4311.111111111111, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ --- `snapshot` --- dispatch periods |
| $\mathcal{B}$ | index $b$ --- `bus` --- network nodes |
| $\mathcal{E}$ | index $e$ --- `generator` with $\mathrm{gen\_bus}: \mathcal{E} \to \mathcal{B}$ --- generating units, each sitting on one bus |
| $\mathcal{L}$ | index $l$ --- `link` with $\mathrm{link\_from}: \mathcal{L} \to \mathcal{B},\enspace \mathrm{link\_to}: \mathcal{L} \to \mathcal{B}$ --- controllable connections, each joining two buses |

#### Parameters

| Symbol | Meaning |
|---|---|
| $p^{\mathrm{nom}}$ | `p_nom` over $\mathcal{E}$ --- installed capacity of a generator |
| $\mathit{marginal\_cost}$ | `marginal_cost` over $\mathcal{E}$ --- cost of one unit of output |
| $\mathit{link}^{\mathrm{p,nom}}$ | `link_p_nom` over $\mathcal{L}$ --- most a link may take in during one snapshot |
| $\mathit{efficiency}$ | `efficiency` over $\mathcal{L}$ --- share of what entered a link that arrives at the other end |
| $\mathit{delay}$ | `delay` over $\mathcal{L}$ --- how many snapshots a link takes to deliver what it took in |
| $\mathit{load}$ | `load` over $\mathcal{T} \times \mathcal{B}$ --- demand at each bus in each snapshot |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{T} \times \mathcal{E}$ --- output of a generator in a snapshot |
| $g$ | `g` over $\mathcal{T} \times \mathcal{L}$ --- what a link takes in during a snapshot, at the bus it leaves |

$t \boxminus_{v} k$ denotes translation with $v$ standing where index $t-k$ leaves the dimension (`shift(edge=v)`), so the row at that boundary is built and carries $v$ rather than being dropped.

#### Objective

$$\min \sum_{t \in \mathcal{T},\enspace e \in \mathcal{E}} p_{t,e} \cdot \mathit{marginal\_cost}_{e}$$

#### Subject to

**`nodal_balance`**

$$\sum_{e \in \mathcal{E} \thinspace:\thinspace \mathrm{gen\_bus}(e) = b} p_{t,e} + \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{link\_to}(l) = b} g_{t \boxminus_{0} \mathit{delay},l} \cdot \mathit{efficiency}_{l} - \left( \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{link\_from}(l) = b} g_{t,l} \right) = \mathit{load}_{t,b} \qquad \forall\thinspace t \in \mathcal{T},\enspace b \in \mathcal{B}$$

#### Variable domains

**`p`**

$$0 \le p_{t,e} \le p^{\mathrm{nom}}_{e} \qquad \forall\thinspace t \in \mathcal{T},\enspace e \in \mathcal{E}$$

**`g`**

$$0 \le g_{t,l} \le \mathit{link}^{\mathrm{p,nom}}_{l} \qquad \forall\thinspace t \in \mathcal{T},\enspace l \in \mathcal{L}$$

</details>
<!-- math:end -->

The tabs start from [the instance's tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    description: >-
      PyPSA's delayed link: power withdrawn at one snapshot arrives at a later one,
      so a shipment is the input shifted along time and derated by the link's
      efficiency. Two links serve the same demand — one takes two snapshots to
      arrive, one arrives at once — and the first snapshots, which nothing has
      reached yet, are served by the expensive unit standing beside the load.
      Optimum 4311.111111111111, from PyPSA itself.

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
      link_p_nom:
        description: most a link may take in during one snapshot
        dims: [link]
      efficiency:
        description: share of what entered a link that arrives at the other end
        dims: [link]
      delay:
        description: how many snapshots a link takes to deliver what it took in
        dims: [link]
        dtype: int
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
      g:
        description: what a link takes in during a snapshot, at the bus it leaves
        foreach: [snapshot, link]
        bounds:
          lower: 0
          upper: link_p_nom

    constraints:
      nodal_balance:
        description: >-
          what is generated at a bus, plus what arrives over the links — the input
          of `delay` snapshots ago, derated — less what the links took in there,
          meets the load. `edge=0` is what a non-cyclic delay means: a snapshot
          earlier than a link's delay receives nothing over it, there being no such
          snapshot to have taken anything in.
        foreach: [snapshot, bus]
        expression: >-
          sum(p, by=gen_bus)
          + sum(shift(g, over=snapshot, offset=delay, edge=0) * efficiency, by=link_to)
          - sum(g, by=link_from)
          == load

    objective:
      sense: minimize
      description: what the fleet costs to run; moving power costs nothing but time and losses
      expression: sum(p * marginal_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/ports/pypsa_link_delay.yaml', sources) as solution:
        solution.objective  # 4311.111111111111
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_link_delay.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the lpspec call binds as ``sources``.

        Both links are given a ``delay`` — 2 for ``ship`` and 0 for ``wire`` — so the
        column is read rather than a constant applied to everything, and neither
        link is extendable: a delay is about *when* energy arrives, and a capacity
        decision would give a mismatch a second thing to be about.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.add('Bus', tables['bus']['bus'])

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        n.add(
            'Generator',
            generators.index,
            bus=generators['gen_bus'],
            p_nom=tables['p_nom'].set_index('generator')['value'],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
        )

        links: pd.DataFrame = tables['link'].set_index('link')
        n.add(
            'Link',
            links.index,
            bus0=links['link_from'],
            bus1=links['link_to'],
            p_nom=tables['link_p_nom'].set_index('link')['value'],
            efficiency=tables['efficiency'].set_index('link')['value'],
            delay=tables['delay'].set_index('link')['value'],
            cyclic_delay=False,
        )

        load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
        for bus in tables['bus']['bus']:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
        return n
    ```

**`cyclic_delay=False` is `edge=0`, one for one.** PyPSA's own attribute table
says of the non-cyclic case that *energy is lost at the tail and first snapshots
receive nothing from delayed links*. That is exactly what `edge=0` states: the
vacated positions contribute zero. The cyclic case is `edge='wrap'`, which
[rung 4](pypsa_cyclic_storage.md) already ports on a different component, so
this rung takes the non-cyclic one — the case with a boundary to say something
about.

The language **refuses** a per-entity shift with no `edge=` at all:

```
LanguageError: constraint 'nodal_balance': shift(offset=delay) leaves the vacated
positions absent, which a per-entity offset cannot say yet.
Add edge='wrap' for a cyclic translation, or edge=<number> for what the vacated
positions contribute.
```

Which is the right refusal here: PyPSA does not leave those positions absent, it
zeroes them, and the two readings build different models.

**Both ends of the horizon show.** Shipments of 44.44 leave in snapshots 0 and 1
and arrive as 40 in snapshots 2 and 3; nothing is shipped in snapshots 4 and 5,
because it would arrive after the horizon ends and be lost. The prices say the
same thing: `port_b` pays 100 while it waits, then 11.11 — the cheap unit's 10
divided by the ship's 0.9.

## What it exercises

`shift(x, over=dim, offset=p, edge=0)` with `p` an integer column, inside a grouped
sum that lands on a *different* entity's row — the first model in the corpus
where a shift moves a quantity between two places rather than along one.
