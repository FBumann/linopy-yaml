# PyPSA CVaR — a risk preference is three variables and two rows

The same three futures as [the stochastic rung](pypsa_stochastic.md), planned against the tail as well as the expectation.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.9.0)** — objective **35410.0**, matched to `rtol=1e-09`.

`n.set_risk_preference(alpha, omega)` turns on `define_cvar_variables`
(`variables.py:291`) — `CVaR-a` over the scenarios, and the two scalars
`CVaR-theta` and `CVaR` — and `define_objective` adds the rows that link them
(`optimize.py:377-419`):

```
a_s   >= OPEX_s - theta                       one per scenario
theta + 1/(1-alpha) * sum_s p_s a_s  <= CVaR  one, over all of them
min      CAPEX + (1-omega) * E[OPEX] + omega * CVaR
```

That is Rockafellar–Uryasev: the average of the worst `1-alpha` of the
distribution, which is a quantile average and sounds nonlinear, is an epigraph
over a level `theta` the model is free to place. The objective's weight on it is
what pulls it down onto the true value at risk.

Two things worth reading off the source rather than the phrase "risk aversion".
**The tail is the operating cost only** — capital cost sits outside the blend, so
`omega` prices what a future costs to *run*. And **`alpha` and `omega` are
independent**: `alpha` says where the tail starts, `omega` how much of the
objective it is.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA's CVaR risk preference on a stochastic network: the plan is chosen against the expectation *and* the tail, which Rockafellar and Uryasev make linear with three auxiliary quantities — an excess per future, the level the tail starts at, and the tail average itself. The risk-averse fleet is not the risk-neutral one. Optimum 35410.0, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{S}$ | index $s$ — `scenario` — the futures the fleet is built against, one of which will happen |
| $\mathcal{T}$ | index $t$ — `snapshot` — dispatch periods, the same in every future |
| $\mathcal{G}$ | index $g$ — `generator` — generating units, each built once and run in every future |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\mathrm{probability}$ | `probability` over $\mathcal{S}$ — how likely a future is — the weights the expectation is taken with |
| $\mathrm{load}$ | `load` over $\mathcal{S} \times \mathcal{T}$ — demand to be met, and the one thing that differs between futures |
| $\mathrm{capex}$ | `capex` over $\mathcal{G}$ — cost of holding one unit of capacity over the horizon |
| $\mathrm{opex}$ | `opex` over $\mathcal{G}$ — cost of one unit of output |
| $\mathrm{alpha}$ | `alpha` (scalar) — where the tail begins: the confidence level whose worst 1 - alpha of the probability mass the risk term averages over |
| $\mathrm{omega}$ | `omega` (scalar) — how much of the objective is the tail rather than the expectation — 0 is the risk-neutral plan, 1 prices nothing but the worst futures |

#### Variables

| Symbol | Meaning |
|---|---|
| $p^{\mathrm{nom}}$ | `p_nom` over $\mathcal{G}$ — capacity built at a generator — the first-stage decision, taken before anyone knows which future arrived |
| $p$ | `p` over $\mathcal{S} \times \mathcal{T} \times \mathcal{G}$ — output of a generator in a snapshot of a future |
| $\mathit{excess}$ | `excess` over $\mathcal{S}$ — how far a future's operating cost runs past the level the tail begins at, and zero for the futures that do not reach it |
| $\mathit{tail\_start}$ | `tail_start` (scalar) — the level the tail begins at — the value at risk, which the epigraph rows pin to the alpha quantile of the operating cost rather than the model declaring it |
| $\mathit{tail\_average}$ | `tail_average` (scalar) — the average operating cost of the futures beyond that level |

Upright is what the model is given — a parameter such as $\mathrm{probability}$, a coordinate map, a label — and italic is what the solver chooses, such as $p^{\mathrm{nom}}$. An index is italic too, being what a quantifier chooses, and a set is script.

#### Objective

$$\min \sum_{g \in \mathcal{G}} p^{\mathrm{nom}}_{g} \cdot \mathrm{capex}_{g} + \sum_{s \in \mathcal{S}} \left( 1 - \mathrm{omega} \right) \cdot \mathrm{probability}_{s} \cdot \left( \sum_{t \in \mathcal{T}} \sum_{g \in \mathcal{G}} p_{s,t,g} \cdot \mathrm{opex}_{g} \right) + \mathrm{omega} \cdot \mathit{tail\_average}$$

#### Subject to

**`within_capacity`**

$$p_{s,t,g} \le p^{\mathrm{nom}}_{g} \qquad \forall\thinspace s \in \mathcal{S},\enspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`power_balance`**

$$\sum_{g \in \mathcal{G}} p_{s,t,g} = \mathrm{load}_{s,t} \qquad \forall\thinspace s \in \mathcal{S},\enspace t \in \mathcal{T}$$

**`tail_excess`**

$$\mathit{excess}_{s} \ge \sum_{t \in \mathcal{T}} \sum_{g \in \mathcal{G}} p_{s,t,g} \cdot \mathrm{opex}_{g} - \mathit{tail\_start} \qquad \forall\thinspace s \in \mathcal{S}$$

**`tail_definition`**

$$\left( 1 - \mathrm{alpha} \right) \cdot \left( \mathit{tail\_average} - \mathit{tail\_start} \right) \ge \sum_{s \in \mathcal{S}} \mathrm{probability}_{s} \cdot \mathit{excess}_{s}$$

#### Variable domains

**`p_nom`**

$$p^{\mathrm{nom}}_{g} \ge 0 \qquad \forall\thinspace g \in \mathcal{G}$$

**`p`**

$$p_{s,t,g} \ge 0 \qquad \forall\thinspace s \in \mathcal{S},\enspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`excess`**

$$\mathit{excess}_{s} \ge 0 \qquad \forall\thinspace s \in \mathcal{S}$$

**`tail_start`**

$$\mathit{tail\_start} \in \mathbb{R}$$

**`tail_average`**

$$\mathit{tail\_average} \in \mathbb{R}$$

</details>
<!-- math:end -->

The tabs start from [the instance's tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    description: >-
      PyPSA's CVaR risk preference on a stochastic network: the plan is chosen
      against the expectation *and* the tail, which Rockafellar and Uryasev make
      linear with three auxiliary quantities — an excess per future, the level the
      tail starts at, and the tail average itself. The risk-averse fleet is not the
      risk-neutral one.
      Optimum 35410.0, from PyPSA itself.

    dimensions:
      scenario:
        description: the futures the fleet is built against, one of which will happen
        dtype: str
      snapshot:
        description: dispatch periods, the same in every future
        dtype: int
      generator:
        description: generating units, each built once and run in every future
        dtype: str

    parameters:
      probability:
        description: how likely a future is — the weights the expectation is taken with
        dims: [scenario]
      load:
        description: demand to be met, and the one thing that differs between futures
        dims: [scenario, snapshot]
      capex:
        description: cost of holding one unit of capacity over the horizon
        dims: [generator]
      opex:
        description: cost of one unit of output
        dims: [generator]
      alpha:
        description: >-
          where the tail begins: the confidence level whose worst 1 - alpha of the
          probability mass the risk term averages over
        dims: []
      omega:
        description: >-
          how much of the objective is the tail rather than the expectation — 0 is
          the risk-neutral plan, 1 prices nothing but the worst futures
        dims: []

    variables:
      p_nom:
        description: >-
          capacity built at a generator — the first-stage decision, taken before
          anyone knows which future arrived
        foreach: [generator]
        bounds:
          lower: 0
      p:
        description: output of a generator in a snapshot of a future
        foreach: [scenario, snapshot, generator]
        bounds:
          lower: 0
      excess:
        description: >-
          how far a future's operating cost runs past the level the tail begins at,
          and zero for the futures that do not reach it
        foreach: [scenario]
        bounds:
          lower: 0
      tail_start:
        description: >-
          the level the tail begins at — the value at risk, which the epigraph rows
          pin to the alpha quantile of the operating cost rather than the model
          declaring it
        foreach: []
      tail_average:
        description: the average operating cost of the futures beyond that level
        foreach: []

    expressions:
      operating_cost:
        description: what one future costs to run, over the whole horizon
        expression: sum(sum(p * opex, over=generator), over=snapshot)

    constraints:
      within_capacity:
        description: >-
          a generator produces no more than the capacity built for it, in every
          snapshot of every future
        foreach: [scenario, snapshot, generator]
        expression: p <= p_nom

      power_balance:
        description: what runs in this snapshot of this future meets the load there
        foreach: [scenario, snapshot]
        expression: sum(p, over=generator) == load

      tail_excess:
        description: >-
          a future's excess reaches at least past the level the tail begins at,
          which with the lower bound of zero makes it the positive part
        foreach: [scenario]
        expression: excess >= operating_cost - tail_start

      tail_definition:
        description: >-
          the tail average is at least the level it starts at plus the expected
          excess divided by the tail's own probability, both sides multiplied by that
          probability — the epigraph that makes a quantile average linear, and the
          objective's weight on it is what pulls it tight
        foreach: []
        expression: >-
          (1 - alpha) * (tail_average - tail_start)
          >= sum(probability * excess, over=scenario)

    objective:
      sense: minimize
      description: >-
        what the fleet costs to build, plus a blend of what it is expected to cost to
        run and what it costs to run in the tail — capital cost sits outside the
        blend, so the risk preference prices operation rather than investment
      expression: >-
        sum(p_nom * capex)
        + sum((1 - omega) * probability * operating_cost)
        + omega * tail_average
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/ports/pypsa_cvar.yaml', sources) as solution:
        solution.objective  # 35410.0
        solution.dual('power_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_cvar.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame | float]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the lpspec call binds as ``sources``.

        The scenario half is ``pypsa_stochastic``'s: ``probability`` is
        ``scenario_weightings`` and ``load`` over ``(scenario, snapshot)`` is a
        ``p_set`` frame with a scenario level. On top of it, ``alpha`` and ``omega``
        — the two scalars the port declares — are exactly ``set_risk_preference``'s
        arguments, which is where the three auxiliary variables and their two rows
        come from.
        """
        n = pypsa.Network()
        n.set_snapshots(list(tables['snapshot']['snapshot']))
        n.set_scenarios(tables['probability'].set_index('scenario')['value'])
        n.set_risk_preference(alpha=float(tables['alpha']), omega=float(tables['omega']))

        n.add('Bus', 'hub')
        generators: pd.DataFrame = tables['generator'].set_index('generator')
        n.add(
            'Generator',
            generators.index,
            bus='hub',
            p_nom_extendable=True,
            capital_cost=tables['capex'].set_index('generator')['value'],
            marginal_cost=tables['opex'].set_index('generator')['value'],
        )

        n.add('Load', 'l', bus='hub')
        load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='scenario', values='value')
        n.loads_t.p_set = pd.DataFrame(
            {(s, 'l'): load[s].to_numpy() for s in tables['probability']['scenario']}, index=n.snapshots
        )
        return n
    ```

**The risk preference binds, and the fleet is what changes.** `omega = 0` makes
the objective the risk-neutral expectation exactly, so the comparison is one
number changed rather than two models:

| | `base` | `peak` | objective |
|---|---|---|---|
| `omega = 0.5` | 170 | 40 | **35410.0** |
| `omega = 0` | 160 | 50 | 33940.0 |

Both fleets total 210 MW, which the severe future needs whatever the planner's
appetite for risk. What risk aversion buys is the *mix*: 10 MW moves from the
cheap-to-build peaker to the cheap-to-run base plant, because the severe future's
operating cost is what the tail term prices. The risk-neutral row is the optimum
of [`pypsa_stochastic`](pypsa_stochastic.md) — the same instance, the same
number, reached twice.

**The tail is found, not declared.** With `alpha = 0.85` the tail holds the worst
15% of the probability mass, which is all of `severe` (10%) and a third of `cold`
(30%). The model places `tail_start` at 4000.0 — `cold`'s operating cost exactly,
the 85th percentile — and `excess` comes out `[0, 0, 3600]`, so `tail_average` is
`4000 + (1/0.15) × 0.1 × 3600 = 6400`. Nothing in the file names a quantile; two
inequalities and a minimisation find it.

**Prices stop being round.** The nodal price under a risk preference is no longer
the scenario weight times a marginal cost. A future outside the tail is priced by
`(1-omega) * p_s` of its costs — `mild` at 0.3 — and one inside it by that plus
`omega * p_s/(1-alpha)`, which for `severe` is `0.05 + 0.333`.

| | snapshot 0 | 1 | 2 |
|---|---|---|---|
| `mild` | 3 | 3 | 3 |
| `cold` | 3.1666… | 3.1666… | 3.1666… |
| `severe` | 10.8333… | **146.8333…** | 3.8333… |

`cold` straddles the boundary: a third of its probability is inside the tail, so
it prices at 0.3166… of a marginal cost rather than 0.15. `severe` snapshot 1 is
the large one because it prices the peaker's 70 at 0.3833 *and* carries the 120
scarcity rent of the capacity it is running out of. All nine are asserted against
PyPSA to `rtol=1e-09`.

**One rewrite, and it is the divisor rule.** PyPSA writes the epigraph with
`1/(1-alpha)` on the sum; a divisor here must be a single variable-free factor
rather than a sum, so the port multiplies through by `1 - alpha` instead:

```yaml
(1 - alpha) * (tail_average - tail_start) >= sum(probability * excess, over=scenario)
```

The same halfspace with the same solutions — `1 - alpha` is positive by
construction — scaled by a constant. It is worth knowing that the row's own dual
carries that scale; the nodal prices, which is what a PyPSA user reads, do not.

## What it exercises

Scalar variables and a scalar row (`foreach: []`) beside dimensioned ones, a
named expression reused in two places — the epigraph rows and the objective — and
an auxiliary variable bounded below by an expression over *other* variables,
which is the shape every linearised risk, regret or minimax measure takes. The
model is degree 1 throughout: a quantile average is not a nonlinear thing here,
it is two more rows.
