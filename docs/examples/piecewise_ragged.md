# piecewise_ragged

Per-generator cost curves of **different lengths**, each as long as its own data.

> **✔ Agrees with hand-written linopy 0.9.0** — objective **426**, matched to `rtol=1e-09`.

## The problem

A breakpoint dimension is one axis for the whole system, but a curve is per
unit: the hydro unit here has two breakpoints, the coal plant three, the gas
turbine four. Nothing in the model should have to know which is longest.

$$p_{t,g} = \sum_{k \in \mathcal{K}_g} \lambda_{t,g,k}\, x_{g,k}, \quad
\mathrm{cost}_{t,g} \ge \sum_{k \in \mathcal{K}_g} \lambda_{t,g,k}\, y_{g,k}, \quad
\sum_{k \in \mathcal{K}_g} \lambda_{t,g,k} = 1$$

The set the weights run over is $\mathcal{K}_g$, the curve's own — which is
what `points: bp_x` says. Without it the shorter curves have to be padded out
to the longest, and padding is not free: it buys a weight per unused
breakpoint, and `method: convex` refuses it outright, since a repeated point is
not a strictly increasing breakpoint.

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

Least-cost dispatch where each generator's cost curve has as many breakpoints as its data gives it — two for the hydro unit, four for the gas turbine — rather than as many as the longest curve in the system.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ — `snapshot` — dispatch periods |
| $\mathcal{G}$ | index $g$ — `generator` — dispatchable units |
| $\mathcal{B}$ | index $b$ — `bp` — breakpoints, as many as the longest curve needs |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\mathrm{p}^{\mathrm{max}}$ | `p_max` over $\mathcal{G}$ — maximum dispatch |
| $\mathrm{load}$ | `load` over $\mathcal{T}$ — demand to be met |
| $\mathrm{bp\_x}$ | `bp_x` over $\mathcal{G} \times \mathcal{B}$ — breakpoint dispatch levels, one curve per generator and no two the same length |
| $\mathrm{bp\_y}$ | `bp_y` over $\mathcal{G} \times \mathcal{B}$ — cost at each breakpoint |
| $\mathrm{cost\_curve\_points}$ | `cost_curve_points` over $\mathcal{G} \times \mathcal{B}$ — where 'bp_x' has a row, and so where the curve runs |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{T} \times \mathcal{G}$ — dispatched power |
| $\mathit{op\_cost}$ | `op_cost` over $\mathcal{T} \times \mathcal{G}$ — operating cost, piecewise-linear in dispatch |
| $\mathit{cost\_curve\_lam}$ | `cost_curve_lam` over $\mathcal{T} \times \mathcal{G} \times \mathcal{B}$ — convex-combination weight on a breakpoint |

Upright is what the model is given — a parameter such as $\mathrm{p}^{\mathrm{max}}$, a coordinate map, a label — and italic is what the solver chooses, such as $p$. An index is italic too, being what a quantifier chooses, and a set is script.

#### Objective

$$\min \sum_{t \in \mathcal{T}} \sum_{g \in \mathcal{G}} \mathit{op\_cost}_{t,g}$$

#### Subject to

**`balance`**

$$\sum_{g \in \mathcal{G}} p_{t,g} = \mathrm{load}_{t} \qquad \forall\thinspace t \in \mathcal{T}$$

**`cost_curve_convexity`**

$$\sum_{b \in \mathcal{B}} \mathit{cost\_curve\_lam}_{t,g,b} = 1 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`cost_curve_link0`**

$$p_{t,g} = \sum_{b \in \mathcal{B}} \mathit{cost\_curve\_lam}_{t,g,b} \cdot \mathrm{bp\_x}_{g,b} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`cost_curve_link1`**

$$\mathit{op\_cost}_{t,g} \ge \sum_{b \in \mathcal{B}} \mathit{cost\_curve\_lam}_{t,g,b} \cdot \mathrm{bp\_y}_{g,b} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

#### Variable domains

**`p`**

$$0 \le p_{t,g} \le \mathrm{p}^{\mathrm{max}}_{g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`op_cost`**

$$\mathit{op\_cost}_{t,g} \ge 0 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`cost_curve_lam`**

$$0 \le \mathit{cost\_curve\_lam}_{t,g,b} \le 1 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G},\enspace b \in \mathcal{B} \thinspace:\thinspace \mathrm{cost\_curve\_points}_{g,b}$$

</details>
<!-- math:end -->

The tabs start from [the instance's tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    description: >-
      Least-cost dispatch where each generator's cost curve has as many breakpoints
      as its data gives it — two for the hydro unit, four for the gas turbine —
      rather than as many as the longest curve in the system.

    dimensions:
      snapshot:
        description: dispatch periods
        dtype: int
      generator:
        description: dispatchable units
        dtype: str
      bp:
        description: breakpoints, as many as the longest curve needs
        dtype: int

    parameters:
      p_max:
        description: maximum dispatch
        dims: [generator]
      load:
        description: demand to be met
        dims: [snapshot]
      bp_x:
        description: breakpoint dispatch levels, one curve per generator and no two the same length
        dims: [generator, bp]
      bp_y:
        description: cost at each breakpoint
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
          each unit's own curve — `points: bp_x` says a curve runs as far as its own
          breakpoints do, so the hydro unit pays for two weights and the gas turbine
          for four
        over: bp
        points: bp_x
        links:
          - [p, bp_x]
          - [op_cost, bp_y, ">="]
        method: convex

    constraints:
      balance:
        description: every period's demand is met
        foreach: [snapshot]
        expression: sum(p, over=generator) == load

    objective:
      sense: minimize
      description: total operating cost, taken off each unit's own curve
      expression: sum(sum(op_cost, over=generator), over=snapshot)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/piecewise_ragged.yaml', sources) as solution:
        solution.objective  # 426.0
        solution.dual('balance')
    ```

=== "linopy"

    The model-building half of `examples/ports/references/linopy/piecewise_ragged.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
        """The instance's tables as a linopy model, row for row.

        ``tables`` is the same mapping the lpspec call binds as ``sources``.
        """
        p_max: pd.Series = tables['p_max'].set_index('generator')['value']
        load: pd.Series = tables['load'].set_index('snapshot')['value']
        reach = tables['bp_x'].groupby('generator')['value'].max().reindex(p_max.index)

        m = linopy.Model()
        p = m.add_variables(lower=0, upper=reach, coords=[load.index, p_max.index], name='p')
        op_cost = m.add_variables(lower=0, coords=[load.index, p_max.index], name='op_cost')
        for g, lines in segments(tables).items():
            for k, (slope, intercept) in enumerate(lines):
                m.add_constraints(
                    op_cost.sel(generator=g) - slope * p.sel(generator=g) >= intercept,
                    name=f'chord_{g}_{k}',
                )
        m.add_constraints(p.sum('generator') == load, name='balance')
        m.add_objective(op_cost.sum())
        return m
    ```

## What it exercises

**`points:` is what makes the curve its own.** The weights, and the segment
binaries where a method declares them, exist only where the mask does, so the
hydro unit carries two rather than four. The values are not asked for at the
breakpoints it leaves out, and a gap in the middle of a curve is refused when
data binds — the marked breakpoints have to follow one another, though they
need not start at the head of the axis.

The reference next door reaches the same optimum from the **other**
formulation: segment lines rather than weights, which is exact for a convex
curve under minimisation and needs no auxiliary variable at all. Two
formulations agreeing is worth more than two spellings of one, and it is also
why this model keeps its duals — both sides stay a pure LP.

---

[`examples/piecewise_ragged.yaml`](https://github.com/fluxopt/lpspec/blob/main/examples/piecewise_ragged.yaml) · back to [all models](index.md)
