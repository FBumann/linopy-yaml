# transport

A network: generators sit on buses, lines connect buses, and power balances at every bus.

> **✔ Verified against linopy 0.9.0** — objective **4400**, matched to `rtol=1e-09`. A teaching model, so the check is agreement with an independent hand-written formulation, not a published figure.

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

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{S}$ | index $s$ --- `snapshot` --- dispatch periods |
| $\mathcal{G}$ | index $g$ --- `generator` with $\mathrm{bus}: \mathcal{G} \to \mathcal{B}$ --- generating units |
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
| $p$ | `p` over $\mathcal{S} \times \mathcal{G}$ --- output of generator $g$ in snapshot $s$ |
| $f$ | `f` over $\mathcal{S} \times \mathcal{L}$ --- flow on line $\ell$, signed towards its `to` bus |

#### Objective

**`total_cost`**

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

=== "lpspec"

    ```yaml
    dimensions:
      snapshot:
        dtype: int
      generator:
        dtype: str
        coords: [bus]  # every generator sits on a bus
      bus:
        dtype: str
      line:
        dtype: str
        coords: {from: bus, to: bus}  # both endpoints are buses

    parameters:
      p_max:
        dims: [generator]
      cost:
        dims: [generator]
      cap:
        dims: [line]
      neg_cap:
        dims: [line]
      load:
        dims: [snapshot, bus]

    variables:
      p:
        foreach: [snapshot, generator]
        bounds:
          lower: 0
          upper: p_max
      f:
        foreach: [snapshot, line]
        bounds:
          lower: neg_cap
          upper: cap

    # Naming the two halves of the nodal balance. Pure substitution before either
    # backend sees the model, so this costs nothing at build and nothing at solve —
    # what it buys is a constraint that reads as the sentence it is.
    expressions:
      gen_at_bus: sum(p, over=generator, group_by=bus)
      net_inflow: sum(f, over=line, group_by=to) - sum(f, over=line, group_by=from)

    constraints:
      balance:
        foreach: [snapshot, bus]
        expression: gen_at_bus + net_inflow == load

    objectives:
      total_cost:
        sense: minimize
        expression: p * cost
    ```

    Run against the committed instance:

    ```python
    import json
    from pathlib import Path

    import lpspec as lps
    import polars as pl

    tables = json.loads(Path('examples/ports/data/transport.json').read_text())
    sources = {k: pl.DataFrame(v) if isinstance(v, dict) else v for k, v in tables.items()}

    with lps.solve('examples/transport.yaml', sources) as solution:
        print(solution.objective)  # 4400.0
        print(solution.dual('balance'))
    ```

=== "linopy"

    `examples/ports/references/linopy/transport.py`:

    ```python
    from __future__ import annotations

    import json
    from pathlib import Path

    import linopy
    import pandas as pd
    import xarray as xr

    DATA = Path(__file__).resolve().parents[2] / 'data' / 'transport.json'


    def build(data: dict) -> linopy.Model:
        """The instance's tables as a linopy model, row for row."""
        generators = pd.Index(data['generator']['generator'], name='generator')
        lines = pd.Index(data['line']['line'], name='line')
        buses = pd.Index(sorted(set(data['load']['bus'])), name='bus')
        snapshots = pd.Index(sorted(set(data['load']['snapshot'])), name='snapshot')

        p_max = pd.Series(data['p_max']['value'], index=generators)
        cost = pd.Series(data['cost']['value'], index=generators)
        cap = pd.Series(data['cap']['value'], index=lines)
        neg_cap = pd.Series(data['neg_cap']['value'], index=lines)
        load = xr.DataArray(
            pd.DataFrame(data['load']).pivot(index='snapshot', columns='bus', values='value').reindex(columns=buses)
        )

        gen_at = pd.DataFrame(0.0, index=buses, columns=generators)
        for gen, bus in zip(generators, data['generator']['bus'], strict=True):
            gen_at.loc[bus, gen] = 1.0
        flow_in = pd.DataFrame(0.0, index=buses, columns=lines)
        for line, src, dst in zip(lines, data['line']['from'], data['line']['to'], strict=True):
            flow_in.loc[dst, line] += 1.0
            flow_in.loc[src, line] -= 1.0

        m = linopy.Model()
        p = m.add_variables(lower=0, upper=p_max, coords=[snapshots, generators], name='p')
        f = m.add_variables(lower=neg_cap, upper=cap, coords=[snapshots, lines], name='f')
        m.add_constraints(
            (p * xr.DataArray(gen_at)).sum('generator') + (f * xr.DataArray(flow_in)).sum('line') == load,
            name='balance',
        )
        m.add_objective((p * cost).sum())
        return m


    def nodal_prices(m: linopy.Model) -> dict[str, list]:
        """The balance dual, tidy: one price per (snapshot, bus)."""
        dual = m.constraints['balance'].dual.transpose('snapshot', 'bus')
        return {
            'snapshot': [int(s) for s in dual.indexes['snapshot'] for _ in dual.indexes['bus']],
            'bus': [str(b) for _ in dual.indexes['snapshot'] for b in dual.indexes['bus']],
            'value': [float(v) for v in dual.values.ravel()],
        }


    def main() -> float:
        m = build(json.loads(DATA.read_text()))
        status, condition = m.solve(solver_name='highs')
        assert status == 'ok', f'{status}: {condition}'
        print(f'linopy {linopy.__version__}')
        print(f'objective {float(m.objective.value)!r}')
        print(f'duals {json.dumps({"balance": nodal_prices(m)})}')
        return float(m.objective.value)


    if __name__ == '__main__':
        main()
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

---

[`examples/transport.yaml`](https://github.com/fluxopt/lpspec/blob/main/examples/transport.yaml) · back to [all models](index.md)
