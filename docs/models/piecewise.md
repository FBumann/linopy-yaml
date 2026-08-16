# piecewise

Per-generator convex cost curves, expanded into a λ-formulation.

> **✔ Agrees with hand-written linopy 0.9.0** — objective **3850**, matched to `rtol=1e-09`.

## The problem

Each generator gets its own breakpoint list, so the curve varies per unit —
something a flat breakpoint list cannot express:

$$p_g = \sum_k \lambda_{g,k}\, x_{g,k}, \quad
\mathrm{cost}_g = \sum_k \lambda_{g,k}\, y_{g,k}, \quad
\sum_k \lambda_{g,k} = 1, \quad \lambda \ge 0$$

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

Least-cost dispatch where each generator's cost curve is piecewise-linear in its output, expanded into a lambda formulation.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ --- `snapshot` --- dispatch periods |
| $\mathcal{G}$ | index $g$ --- `generator` --- dispatchable units |
| $\mathcal{K}$ | index $k$ --- `bp` --- breakpoints of the cost curve |

#### Parameters

| Symbol | Meaning |
|---|---|
| $p^{\mathrm{max}}$ | `p_max` over $\mathcal{G}$ --- maximum dispatch |
| $\mathit{load}$ | `load` over $\mathcal{T}$ --- demand to be met |
| $x$ | `bp_x` over $\mathcal{G} \times \mathcal{K}$ --- breakpoint dispatch levels, one curve per generator |
| $y$ | `bp_y` over $\mathcal{G} \times \mathcal{K}$ --- cost at each breakpoint, one curve per generator |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{T} \times \mathcal{G}$ --- dispatched power |
| $\mathrm{cost}$ | `op_cost` over $\mathcal{T} \times \mathcal{G}$ --- operating cost, piecewise-linear in dispatch |
| $\lambda$ | `cost_curve_lam` over $\mathcal{T} \times \mathcal{G} \times \mathcal{K}$ --- convex-combination weight on a breakpoint |

#### Objective

$$\min \sum_{t \in \mathcal{T},\enspace g \in \mathcal{G}} \mathrm{cost}_{t,g}$$

#### Subject to

**`balance`**

$$\sum_{g \in \mathcal{G}} p_{t,g} = \mathit{load}_{t} \qquad \forall\thinspace t \in \mathcal{T}$$

**`cost_curve_convexity`**

$$\sum_{k \in \mathcal{K}} \lambda_{t,g,k} = 1 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`cost_curve_link0`**

$$p_{t,g} = \sum_{k \in \mathcal{K}} \lambda_{t,g,k} \cdot x_{g,k} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`cost_curve_link1`**

$$\mathrm{cost}_{t,g} = \sum_{k \in \mathcal{K}} \lambda_{t,g,k} \cdot y_{g,k} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

#### Variable domains

**`p`**

$$0 \le p_{t,g} \le p^{\mathrm{max}}_{g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`op_cost`**

$$\mathrm{cost}_{t,g} \ge 0 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`cost_curve_lam`**

$$0 \le \lambda_{t,g,k} \le 1 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G},\enspace k \in \mathcal{K}$$

</details>
<!-- math:end -->

The tabs start from [the instance’s tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    description: >-
      Least-cost dispatch where each generator's cost curve is piecewise-linear in
      its output, expanded into a lambda formulation.

    dimensions:
      snapshot:
        description: dispatch periods
        dtype: int
      generator:
        description: dispatchable units
        dtype: str
      bp:
        description: breakpoints of the cost curve
        dtype: int

    parameters:
      p_max:
        description: maximum dispatch
        dims: [generator]
      load:
        description: demand to be met
        dims: [snapshot]
      bp_x:
        description: breakpoint dispatch levels, one curve per generator
        dims: [generator, bp]
      bp_y:
        description: cost at each breakpoint, one curve per generator
        dims: [generator, bp]

    variables:
      p:
        description: dispatched power
        foreach: [snapshot, generator]
        bounds:
          lower: 0
          upper: p_max
      op_cost:
        description: operating cost, piecewise-linear in dispatch
        foreach: [snapshot, generator]
        bounds:
          lower: 0

    piecewise:
      cost_curve:
        description: >-
          cost read off the generator's curve — convex, so the weights need no
          binaries to keep them on one segment
        over: bp
        links:
          - [p, bp_x]
          - [op_cost, bp_y]
        method: convex

    constraints:
      balance:
        foreach: [snapshot]
        expression: sum(p, over=generator) == load

    objective:
      sense: minimize
      description: total operating cost, taken off the curves rather than from a marginal rate
      expression: op_cost
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/piecewise.yaml', sources) as solution:
        solution.objective  # 3850.0
        solution.dual('balance')
    ```

=== "linopy"

    The model-building half of `examples/ports/references/linopy/piecewise.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
        """The instance's tables as a linopy model, row for row.

        ``tables`` is the same mapping the lpspec call binds as ``sources``.
        """
        p_max: pd.Series = tables['p_max'].set_index('generator')['value']
        load: pd.Series = tables['load'].set_index('snapshot')['value']
        curve_x: pd.DataFrame = tables['bp_x'].pivot(index='generator', columns='bp', values='value').reindex(p_max.index)
        curve_y: pd.DataFrame = tables['bp_y'].pivot(index='generator', columns='bp', values='value').reindex(p_max.index)
        bp_x = linopy.breakpoints(curve_x, dim='generator')
        bp_y = linopy.breakpoints(curve_y, dim='generator')

        m = linopy.Model()
        p = m.add_variables(lower=0, upper=p_max, coords=[load.index, p_max.index], name='p')
        op_cost = m.add_variables(lower=0, coords=[load.index, p_max.index], name='op_cost')
        m.add_piecewise_formulation((p, bp_x), (op_cost, bp_y, '>='))
        m.add_constraints(p.sum('generator') == load, name='balance')
        m.add_objective(op_cost.sum())
        return m
    ```

## What it exercises

`piecewise:` is a **declaration, not an operator** — it expands before lowering
into the λ-formulation above. With `method: convex` the expansion emits no
binaries at all: the convex hull is exact for a convex curve under
minimisation, so the model stays a pure LP. `method: adjacency`, the default,
adds segment binaries and adjacency constraints instead, and the model becomes
a MILP that is still entirely inside the relational subset — while
`method: sos2` states that same restriction as a [set](sos.md) and leaves the
binaries to whichever sink needs them.

By the time the logical plan exists there is nothing left called *piecewise* —
which is why the construct matrix reads it from the surface declaration.

---

[`examples/piecewise.yaml`](https://github.com/fluxopt/lpspec/blob/main/examples/piecewise.yaml) · back to [all models](index.md)
