# PyPSA LOPF — rung 1

PyPSA linear optimal power flow, first rung: transport model, linear marginal cost, no KVL.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.9.0)** — objective **22000**, matched to `rtol=1e-09`.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA linear optimal power flow, rung 1: a transport model — linear marginal cost, controllable links, no voltage law. Optimum 22000.0, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ --- `snapshot` --- dispatch periods |
| $\mathcal{B}$ | index $b$ --- `bus` --- network nodes |
| $\mathcal{G}$ | index $g$ --- `generator` with $\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}$ --- generating units, each sitting on one bus |
| $\mathcal{L}$ | index $l$ --- `link` with $\mathrm{from}: \mathcal{L} \to \mathcal{B},\enspace \mathrm{to}: \mathcal{L} \to \mathcal{B}$ --- controllable connections, each joining two buses |

#### Parameters

| Symbol | Meaning |
|---|---|
| $p^{\mathrm{nom}}$ | `p_nom` over $\mathcal{G}$ --- installed capacity of a generator |
| $\mathit{marginal\_cost}$ | `marginal_cost` over $\mathcal{G}$ --- cost of one unit of output |
| $\mathit{rating}$ | `rating` over $\mathcal{L}$ --- most a link may carry towards its `to` bus |
| $\mathit{neg\_rating}$ | `neg_rating` over $\mathcal{L}$ --- most a link may carry the other way, negative by convention |
| $\mathit{load}$ | `load` over $\mathcal{T} \times \mathcal{B}$ --- demand at each bus in each snapshot |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{T} \times \mathcal{G}$ --- output of a generator in a snapshot |
| $f$ | `f` over $\mathcal{T} \times \mathcal{L}$ --- PyPSA's p0 — flow measured at the link's `from` end, so a positive value withdraws there and injects at `to` |

#### Objective

$$\min \sum_{t \in \mathcal{T},\enspace g \in \mathcal{G}} p_{t,g} \cdot \mathit{marginal\_cost}_{g}$$

#### Subject to

**`nodal_balance`**

$$\sum_{g \in \mathcal{G} \thinspace:\thinspace \mathrm{gen\_bus}(g) = b} p_{t,g} + \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{to}(l) = b} f_{t,l} - \left( \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{from}(l) = b} f_{t,l} \right) = \mathit{load}_{t,b} \qquad \forall\thinspace t \in \mathcal{T},\enspace b \in \mathcal{B}$$

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
    description: >-
      PyPSA linear optimal power flow, rung 1: a transport model — linear marginal
      cost, controllable links, no voltage law. Optimum 22000.0, from PyPSA itself.

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
      from:
        description: the bus a link leaves
        over: link
        into: bus
      to:
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
      rating:
        description: most a link may carry towards its `to` bus
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
        description: >-
          PyPSA's p0 — flow measured at the link's `from` end, so a positive value
          withdraws there and injects at `to`
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
          + sum(f, by=to)
          - sum(f, by=from)
          == load

    objective:
      sense: minimize
      description: total cost of generation; moving power over a link is free here
      expression: p * marginal_cost
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/ports/pypsa_transport.yaml', sources) as solution:
        solution.objective  # 22000.0
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_transport.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the lpspec call binds as ``sources``.

        ``p_min_pu = -1`` makes a link bidirectional. The port cannot say that in
        a bound — bounds take a name or a number, never arithmetic (the declaration rules) — so
        it ships ``neg_rating`` as data instead. That is the ledger row.
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

**Read this comparison carefully — it flatters neither side fairly.** PyPSA is
a *domain package*: `n.add('Generator', ...)` and `n.add('Link', ...)` carry a
power-systems model inside them, so the reference is short because someone
already wrote the power flow. Against that, the YAML looks more explicit rather
than shorter, and it should — it is stating the constraint PyPSA implies.

The comparison against a general-purpose alternative is on
[the Dantzig page](transport_dantzig.md), where both sides write the maths out.

## What it exercises

Rung 1 of a ladder. Reproducing a full PyPSA objective means reproducing
marginal *and* capital cost, ramp limits, storage cycling and KVL at once, and
a mismatch then implicates five features instead of one. So each feature is
switched off in PyPSA and reproduced here separately: **1 transport model**
(this one) · 2 ramp limits · 3 storage with state of charge · 4 cyclic
boundary condition · 5 KVL.

**This rung hit the ceiling once**, and that is recorded rather than worked
around quietly: PyPSA's `p_min_pu = -1` is a bound of `-rating`, an expression
this language cannot yet put in `bounds:`. It ships as a `neg_rating` column
instead, and the gap is [issue #31](https://github.com/fluxopt/lpspec/issues/31)
with the verdict *primitive*. See
[the ledger](index.md#ledger--what-a-port-could-not-say).

---

[`examples/ports/pypsa_transport.yaml`](https://github.com/fluxopt/lpspec/blob/main/examples/ports/pypsa_transport.yaml) · back to [all models](index.md)
