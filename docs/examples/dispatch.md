# dispatch

Least-cost generation against a load profile — the smallest model that is still a model.

> **✔ Agrees with hand-written linopy 0.9.0** — objective **10500**, matched to `rtol=1e-09`.

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

Least-cost dispatch of a generator fleet against an hourly load.

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
| $p$ | `p` over $\mathcal{S} \times \mathcal{G}$ --- output of a generator in a snapshot |

#### Objective

$$\min \sum_{s \in \mathcal{S},\enspace g \in \mathcal{G}} p_{s,g} \cdot c_{g}$$

#### Subject to

**`power_balance`**

$$\sum_{g \in \mathcal{G}} p_{s,g} = \ell_{s} \qquad \forall\thinspace s \in \mathcal{S}$$

#### Variable domains

**`p`**

$$0 \le p_{s,g} \le \bar p_{g} \qquad \forall\thinspace s \in \mathcal{S},\enspace g \in \mathcal{G} \thinspace:\thinspace \bar p_{g} > 0$$

</details>
<!-- math:end -->

The tabs start from [the instance's tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    description: Least-cost dispatch of a generator fleet against an hourly load.

    dimensions:
      snapshot:
        description: dispatch periods
        dtype: int
      generator:
        description: generating units
        values: [wind, solar, gas]

    parameters:
      p_max:
        description: installed capacity
        dims: [generator]
      load:
        description: demand to be met
        dims: [snapshot]
      cost:
        description: marginal cost
        dims: [generator]

    variables:
      p:
        description: output of a generator in a snapshot
        foreach: [snapshot, generator]
        where: "p_max > 0"
        bounds:
          lower: 0
          upper: p_max

    constraints:
      power_balance:
        foreach: [snapshot]
        expression: sum(p, over=generator) == load

    objective:
      sense: minimize
      description: total cost of generation over the horizon
      expression: sum(p * cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/dispatch.yaml', sources) as solution:
        solution.objective  # 10500.0
        solution.dual('power_balance')
    ```

=== "linopy"

    The model-building half of `examples/ports/references/linopy/dispatch.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
        """The instance's tables as a linopy model, row for row.

        ``tables`` is the same mapping the lpspec call binds as ``sources``.
        """
        p_max: pd.Series = tables['p_max'].set_index('generator')['value']
        cost: pd.Series = tables['cost'].set_index('generator')['value']
        load: pd.Series = tables['load'].set_index('snapshot')['value']

        m = linopy.Model()
        p = m.add_variables(lower=0, upper=p_max, coords=[load.index, p_max.index], name='p')
        m.add_constraints(p.sum('generator') == load, name='power_balance')
        m.add_objective((p * cost).sum())
        return m
    ```

## What it exercises

`where: "p_max > 0"` is the one line worth pausing on. A generator with no
capacity gets **no columns at all** — not a column pinned to zero — so a
retired unit costs nothing to carry in the data. That is row absence, and it
is how sparsity is spelled throughout: see [`where`](../reference/language/absence.md) in the
language reference.

---

[`examples/dispatch.yaml`](https://github.com/fluxopt/lpspec/blob/main/examples/dispatch.yaml) · back to [all models](index.md)
