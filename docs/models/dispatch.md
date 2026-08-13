# dispatch

Least-cost generation against a load profile — the smallest model that is still a model.

> **✔ Verified against linopy 0.9.0** — objective **10500**, matched to `rtol=1e-09`. A teaching model, so the check is agreement with an independent hand-written formulation, not a published figure.

## The problem

Pick an output $p_{s,g}$ for every generator in every snapshot, so that the
fleet meets the load exactly and costs as little as possible:

$$\min \sum_{s,g} c_g \thinspace p_{s,g}
\quad\text{s.t.}\quad \sum_g p_{s,g} = \ell_s ,\quad 0 \le p_{s,g} \le \bar p_g
\quad\text{where}\quad \bar p_g > 0$$

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{S}$ | index $s$ --- `snapshot` --- dispatch periods |
| $\mathcal{G}$ | index $g$ --- `generator` --- generating units |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\bar p$ | `p_max` over $\mathcal{G}$ --- installed capacity |
| $\ell$ | `load` over $\mathcal{S}$ --- demand to be met |
| $c$ | `cost` over $\mathcal{G}$ --- marginal cost |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{S} \times \mathcal{G}$ --- output of generator $g$ in snapshot $s$ |

#### Objective

**`total_cost`**

$$\min \sum_{s \in \mathcal{S},\enspace g \in \mathcal{G}} p_{s,g} \cdot c_{g}$$

#### Subject to

**`power_balance`**

$$\sum_{g \in \mathcal{G}} p_{s,g} = \ell_{s} \qquad \forall\thinspace s \in \mathcal{S}$$

#### Variable domains

**`p`**

$$0 \le p_{s,g} \le \bar p_{g} \qquad \forall\thinspace s \in \mathcal{S},\enspace g \in \mathcal{G} \thinspace:\thinspace \bar p_{g} > 0$$

</details>
<!-- math:end -->

=== "lpspec"

    ```yaml
    dimensions:
      snapshot:
        dtype: int
      generator:
        values: [wind, solar, gas]

    parameters:
      p_max:
        dims: [generator]
      load:
        dims: [snapshot]
      cost:
        dims: [generator]

    variables:
      p:
        foreach: [snapshot, generator]
        where: "p_max > 0"
        bounds:
          lower: 0
          upper: p_max

    constraints:
      power_balance:
        foreach: [snapshot]
        expression: sum(p, over=generator) == load

    objectives:
      total_cost:
        sense: minimize
        expression: p * cost
    ```

=== "linopy"

    `examples/ports/references/linopy/dispatch.py`:

    ```python
    from __future__ import annotations

    import json
    from pathlib import Path

    import linopy
    import pandas as pd

    DATA = Path(__file__).resolve().parents[2] / 'data' / 'dispatch.json'


    def build(data: dict) -> linopy.Model:
        """The instance's tables as a linopy model, row for row."""
        generators = pd.Index(data['p_max']['generator'], name='generator')
        snapshots = pd.Index(data['load']['snapshot'], name='snapshot')

        p_max = pd.Series(data['p_max']['value'], index=generators)
        cost = pd.Series(data['cost']['value'], index=generators)
        load = pd.Series(data['load']['value'], index=snapshots)

        m = linopy.Model()
        p = m.add_variables(lower=0, upper=p_max, coords=[snapshots, generators], name='p')
        m.add_constraints(p.sum('generator') == load, name='power_balance')
        m.add_objective((p * cost).sum())
        return m


    def marginal_prices(m: linopy.Model) -> dict[str, list]:
        """The power-balance dual: the classic price signal.

        One price per snapshot — the cost of the marginal generator, which is what
        makes dispatch worth checking on duals: a snapshot where wind covers the
        load prices at wind, the moment gas has to run the price jumps to gas.
        """
        dual = m.constraints['power_balance'].dual
        return {'snapshot': [int(v) for v in dual.indexes['snapshot']], 'value': [float(v) for v in dual.values]}


    def main() -> float:
        m = build(json.loads(DATA.read_text()))
        status, condition = m.solve(solver_name='highs')
        assert status == 'ok', f'{status}: {condition}'
        print(f'linopy {linopy.__version__}')
        print(f'objective {float(m.objective.value)!r}')
        print(f'duals {json.dumps({"power_balance": marginal_prices(m)})}')
        return float(m.objective.value)


    if __name__ == '__main__':
        main()
    ```

## What it exercises

`where: "p_max > 0"` is the one line worth pausing on. A generator with no
capacity gets **no columns at all** — not a column pinned to zero — so a
retired unit costs nothing to carry in the data. That is row absence, and it
is how sparsity is spelled throughout: see [`where`](../SPEC.md) in the
language reference.

---

[`examples/dispatch.yaml`](https://github.com/fluxopt/lpspec/blob/main/examples/dispatch.yaml) · back to [all models](index.md)
