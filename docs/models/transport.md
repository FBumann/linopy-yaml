# transport

A network: generators sit on buses, lines connect buses, and power balances at every bus.

> **✔ Agrees with hand-written linopy 0.9.0** — objective **4400**, matched to `rtol=1e-09`.

## The problem

$$\sum_{g \thinspace:\thinspace \mathrm{bus}(g) = b} p_{s,g} \quad+\quad \sum_{\ell \thinspace:\thinspace \mathrm{to}(\ell) = b} f_{s,\ell} \quad-\quad \sum_{\ell \thinspace:\thinspace \mathrm{from}(\ell) = b} f_{s,\ell} \quad=\quad d_{s,b}$$

Each sum is over the lines or generators a *coordinate map* sends to bus $b$ —
$\mathrm{bus}$, $\mathrm{to}$ and $\mathrm{from}$ are the coordinates the
dimensions declare, not sets in their own right. Load is $d$ here, because
$\ell$ is already the line index.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

Least-cost dispatch over a network, where a generator sits on a bus, a line joins two of them, and what is generated has to reach the load over the lines.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{S}$ | index $s$ --- `snapshot` --- dispatch periods |
| $\mathcal{G}$ | index $g$ --- `generator` with $\mathrm{bus}: \mathcal{G} \to \mathcal{B}$ --- generating units, each sitting on one bus |
| $\mathcal{B}$ | index $b$ --- `bus` --- network nodes |
| $\mathcal{L}$ | index $\ell$ --- `line` with $\mathrm{from}: \mathcal{L} \to \mathcal{B},\enspace \mathrm{to}: \mathcal{L} \to \mathcal{B}$ --- transmission lines, each joining two buses |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\bar p$ | `p_max` over $\mathcal{G}$ --- installed capacity |
| $c$ | `cost` over $\mathcal{G}$ --- marginal cost |
| $\bar f$ | `cap` over $\mathcal{L}$ --- forward transmission limit |
| $\underline{f}$ | `neg_cap` over $\mathcal{L}$ --- reverse transmission limit |
| $d$ | `load` over $\mathcal{S} \times \mathcal{B}$ --- demand at each bus |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{S} \times \mathcal{G}$ --- output of a generator in a snapshot |
| $f$ | `f` over $\mathcal{S} \times \mathcal{L}$ --- flow on a line, signed towards its `to` bus |

#### Objective

$$\min \sum_{s \in \mathcal{S},\enspace g \in \mathcal{G}} p_{s,g} \cdot c_{g}$$

#### Subject to

**`balance`**

$$\sum_{g \in \mathcal{G} \thinspace:\thinspace \mathrm{bus}(g) = b} p_{s,g} + \sum_{\ell \in \mathcal{L} \thinspace:\thinspace \mathrm{to}(\ell) = b} f_{s,\ell} - \left( \sum_{\ell \in \mathcal{L} \thinspace:\thinspace \mathrm{from}(\ell) = b} f_{s,\ell} \right) = d_{s,b} \qquad \forall\thinspace s \in \mathcal{S},\enspace b \in \mathcal{B}$$

#### Variable domains

**`p`**

$$0 \le p_{s,g} \le \bar p_{g} \qquad \forall\thinspace s \in \mathcal{S},\enspace g \in \mathcal{G}$$

**`f`**

$$\underline{f}_{\ell} \le f_{s,\ell} \le \bar f_{\ell} \qquad \forall\thinspace s \in \mathcal{S},\enspace \ell \in \mathcal{L}$$

</details>
<!-- math:end -->

The tabs start from [the instance’s tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    description: >-
      Least-cost dispatch over a network, where a generator sits on a bus, a line
      joins two of them, and what is generated has to reach the load over the
      lines.

    dimensions:
      snapshot:
        description: dispatch periods
        dtype: int
      generator:
        description: generating units, each sitting on one bus
        dtype: str
        coords: [bus]
      bus:
        description: network nodes
        dtype: str
      line:
        description: transmission lines, each joining two buses
        dtype: str
        coords: {from: bus, to: bus}

    parameters:
      p_max:
        description: installed capacity
        dims: [generator]
      cost:
        description: marginal cost
        dims: [generator]
      cap:
        description: forward transmission limit
        dims: [line]
      neg_cap:
        description: reverse transmission limit
        dims: [line]
      load:
        description: demand at each bus
        dims: [snapshot, bus]

    variables:
      p:
        description: output of a generator in a snapshot
        foreach: [snapshot, generator]
        bounds:
          lower: 0
          upper: p_max
      f:
        description: flow on a line, signed towards its `to` bus
        foreach: [snapshot, line]
        bounds:
          lower: neg_cap
          upper: cap

    expressions:
      gen_at_bus:
        expression: sum(p, over=generator, group_by=bus)
        description: what the generators sitting on a bus produce there
      net_inflow:
        expression: sum(f, over=line, group_by=to) - sum(f, over=line, group_by=from)
        description: flow arriving at a bus minus flow leaving it, so a negative value is a net export

    constraints:
      balance:
        description: what is generated at a bus plus what arrives over the lines meets the load there
        foreach: [snapshot, bus]
        expression: gen_at_bus + net_inflow == load

    objective:
      sense: minimize
      description: total cost of generation; moving power over a line is free here
      expression: p * cost
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/transport.yaml', sources) as solution:
        solution.objective  # 4400.0
        solution.dual('balance')
    ```

=== "linopy"

    The model-building half of `examples/ports/references/linopy/transport.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
        """The instance's tables as a linopy model, row for row.

        ``tables`` is the same mapping the lpspec call binds as ``sources``.
        """
        p_max: pd.Series = tables['p_max'].set_index('generator')['value']
        cost: pd.Series = tables['cost'].set_index('generator')['value']
        cap: pd.Series = tables['cap'].set_index('line')['value']
        neg_cap: pd.Series = tables['neg_cap'].set_index('line')['value']
        load = xr.DataArray(tables['load'].pivot(index='snapshot', columns='bus', values='value'))
        snapshots, buses = load.indexes['snapshot'], load.indexes['bus']

        gen_at = pd.DataFrame(0.0, index=buses, columns=p_max.index)
        for gen, bus in zip(tables['generator']['generator'], tables['generator']['bus'], strict=True):
            gen_at.loc[bus, gen] = 1.0
        flow_in = pd.DataFrame(0.0, index=buses, columns=cap.index)
        for line, src, dst in zip(tables['line']['line'], tables['line']['from'], tables['line']['to'], strict=True):
            flow_in.loc[dst, line] += 1.0
            flow_in.loc[src, line] -= 1.0

        m = linopy.Model()
        p = m.add_variables(lower=0, upper=p_max, coords=[snapshots, p_max.index], name='p')
        f = m.add_variables(lower=neg_cap, upper=cap, coords=[snapshots, cap.index], name='f')
        m.add_constraints(
            (p * xr.DataArray(gen_at)).sum('generator') + (f * xr.DataArray(flow_in)).sum('line') == load,
            name='balance',
        )
        m.add_objective((p * cost).sum())
        return m
    ```

## What it exercises

Three `sum(group_by=)` calls, and they are what a network *is* in this language.
A dimension can carry **coordinates** — `generator` carries `bus`, `line`
carries `from` and `to` — and `sum(f, over=line, group_by=to)` sums along a
line's `to` coordinate, landing the result on `bus`. The same `f` is summed
twice through two different coordinates, once as an inflow and once as an
outflow.

No adjacency matrix, and no join written by the modeller: the topology is
data on the dimension.

The two halves of the balance are **named expressions** — substituted into the
constraint before either backend sees the model, so naming them costs nothing
at build or solve; what it buys is a constraint that reads as the sentence it
is, and a quantity the solution can hand back: `expression('net_inflow')` is
the bus-by-bus net flow the balance constrained, one definition for the
constraint and the report.

---

[`examples/transport.yaml`](https://github.com/fluxopt/lpspec/blob/main/examples/transport.yaml) · back to [all models](index.md)
