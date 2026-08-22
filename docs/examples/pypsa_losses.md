# PyPSA transmission losses — a quadratic curve, held up by its own tangents

The loss on a line is `r · s²`. PyPSA approximates it from below with a fan of tangents.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.9.0)** — objective **24114.237385131008**, matched to `rtol=1e-09`.

For each segment *k* PyPSA takes the point `p_k = k/segments · s_nom`, draws the
tangent to the loss curve there, and adds it once for each sign of the flow.
Every one is a half-plane on `(loss, s)`, so the whole approximation is linear
rows and **no auxiliary variable at all** — the objective pushes the loss down,
the tangents hold it up, and it settles on the curve.

Six snapshots, three busy and three quiet: the flows have to reach the **early**
segments of the fan as well as the top of it. With the busy snapshots alone only
two of five half-planes ever bound, and the other three coefficients could have
been anything without the port noticing.

The network is a path, `b0—b1—b2—b3`, radial on purpose: with no independent
cycle there is no voltage law to satisfy, so a mismatch here implicates the loss
approximation rather than [rung 5](pypsa_kvl.md)'s technique. The last line has
**no resistance** — PyPSA gives every passive branch a loss variable and lets
`r = 0` pin it to nothing, while the port declares one only where there is a
curve to approximate.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA transmission losses, tangent form: the quadratic loss on a line, underestimated from below by a fan of tangent half-planes and subtracted half at each end. No auxiliary variable and no piecewise construct — a tangent is one linear row per segment and per sign of the flow. Optimum 3692.705905599654, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ — `snapshot` — dispatch periods |
| $\mathcal{B}$ | index $b$ — `bus` — network nodes |
| $\mathcal{G}$ | index $g$ — `generator` with $\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}$ — generating units, each sitting on one bus |
| $\mathcal{L}$ | index $l$ — `line` with $\mathrm{from}: \mathcal{L} \to \mathcal{B},\enspace \mathrm{to}: \mathcal{L} \to \mathcal{B}$ — passive branches, each joining two buses |
| $\mathcal{S}$ | index $s$ — `segment` — the tangent points the loss curve is approximated at |

#### Parameters

| Symbol | Meaning |
|---|---|
| $p^{\mathrm{nom}}$ | `p_nom` over $\mathcal{G}$ — installed capacity of a generator |
| $\mathit{marginal\_cost}$ | `marginal_cost` over $\mathcal{G}$ — cost of one unit of output |
| $s^{\mathrm{nom}}$ | `s_nom` over $\mathcal{L}$ — most a line may carry towards its `to` bus |
| $\mathit{neg\_s\_nom}$ | `neg_s_nom` over $\mathcal{L}$ — most a line may carry the other way, negative by convention |
| $\mathit{loss}^{\mathrm{max}}$ | `loss_max` over $\mathcal{L}$ — the loss at a line's rating — the top of the curve being approximated, carried as a column because a bound takes a name or a number, and given only for the lines that dissipate anything |
| $\mathit{loss}^{\mathrm{slope}}$ | `loss_slope` over $\mathcal{L} \times \mathcal{S}$ — the slope of this segment's half-plane — how much loss the flow buys along it. Where the segments come from is the instance's business, not the model's: a tangent to the loss curve and a secant across it both arrive here as a slope and an offset. |
| $\mathit{loss}^{\mathrm{offset}}$ | `loss_offset` over $\mathcal{L} \times \mathcal{S}$ — where this segment's half-plane meets the loss axis, negative for a curve through the origin |
| $\mathit{load}$ | `load` over $\mathcal{T} \times \mathcal{B}$ — demand at each bus in each snapshot |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{T} \times \mathcal{G}$ — output of a generator in a snapshot |
| $f$ | `f` over $\mathcal{T} \times \mathcal{L}$ — flow on a line, signed towards its `to` bus — unbounded here, because the rating covers the flow *and* its loss and so is a row rather than a bound |
| $\mathit{loss}$ | `loss` over $\mathcal{T} \times \mathcal{L}$ — the energy a line dissipates carrying its flow — pushed down by the objective and held up by the tangents, so it settles on the approximated curve rather than needing an equality of its own. A line with no resistance dissipates nothing, which is a loss of zero rather than a quantity with no value, so the balances and ratings that name it keep their rows. |

#### Objective

$$\min \sum_{t \in \mathcal{T},\enspace g \in \mathcal{G}} p_{t,g} \cdot \mathit{marginal\_cost}_{g}$$

#### Subject to

**`nodal_balance`**

$$\sum_{g \in \mathcal{G} \thinspace:\thinspace \mathrm{gen\_bus}(g) = b} p_{t,g} + \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{to}(l) = b} f_{t,l} - \left( \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{from}(l) = b} f_{t,l} \right) - 0.5 \cdot \left( \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{from}(l) = b} \mathit{loss}_{t,l} \right) - 0.5 \cdot \left( \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{to}(l) = b} \mathit{loss}_{t,l} \right) = \mathit{load}_{t,b} \qquad \forall\thinspace t \in \mathcal{T},\enspace b \in \mathcal{B}$$

**`within_rating_forward`**

$$f_{t,l} + \mathit{loss}_{t,l} \le s^{\mathrm{nom}}_{l} \qquad \forall\thinspace t \in \mathcal{T},\enspace l \in \mathcal{L}$$

**`within_rating_reverse`**

$$f_{t,l} - \mathit{loss}_{t,l} \ge \mathit{neg\_s\_nom}_{l} \qquad \forall\thinspace t \in \mathcal{T},\enspace l \in \mathcal{L}$$

**`loss_above_segment_forward`**

$$\mathit{loss}_{t,l} + \mathit{loss}^{\mathrm{slope}}_{l,s} \cdot f_{t,l} \ge \mathit{loss}^{\mathrm{offset}}_{l,s} \qquad \forall\thinspace t \in \mathcal{T},\enspace l \in \mathcal{L},\enspace s \in \mathcal{S} \thinspace:\thinspace \mathit{loss}^{\mathrm{max}}_{l} \text{ is defined}$$

**`loss_above_segment_reverse`**

$$\mathit{loss}_{t,l} - \mathit{loss}^{\mathrm{slope}}_{l,s} \cdot f_{t,l} \ge \mathit{loss}^{\mathrm{offset}}_{l,s} \qquad \forall\thinspace t \in \mathcal{T},\enspace l \in \mathcal{L},\enspace s \in \mathcal{S} \thinspace:\thinspace \mathit{loss}^{\mathrm{max}}_{l} \text{ is defined}$$

#### Variable domains

**`p`**

$$0 \le p_{t,g} \le p^{\mathrm{nom}}_{g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`f`**

$$f_{t,l} \in \mathbb{R} \qquad \forall\thinspace t \in \mathcal{T},\enspace l \in \mathcal{L}$$

**`loss`**

$$0 \le \mathit{loss}_{t,l} \le \mathit{loss}^{\mathrm{max}}_{l} \qquad \forall\thinspace t \in \mathcal{T},\enspace l \in \mathcal{L} \thinspace:\thinspace \mathit{loss}^{\mathrm{max}}_{l} \text{ is defined}$$

</details>
<!-- math:end -->

The tabs start from [the instance's tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    description: >-
      PyPSA transmission losses, tangent form: the quadratic loss on a line,
      underestimated from below by a fan of tangent half-planes and subtracted half
      at each end. No auxiliary variable and no piecewise construct — a tangent is
      one linear row per segment and per sign of the flow.
      Optimum 3692.705905599654, from PyPSA itself.

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
      line:
        description: passive branches, each joining two buses
        dtype: str
      segment:
        description: the tangent points the loss curve is approximated at
        dtype: int

    lookups:
      gen_bus:
        description: the bus a generator sits on
        over: generator
        into: bus
      from:
        description: the bus a line leaves
        over: line
        into: bus
      to:
        description: the bus a line arrives at
        over: line
        into: bus

    parameters:
      p_nom:
        description: installed capacity of a generator
        dims: [generator]
      marginal_cost:
        description: cost of one unit of output
        dims: [generator]
      s_nom:
        description: most a line may carry towards its `to` bus
        dims: [line]
      neg_s_nom:
        description: most a line may carry the other way, negative by convention
        dims: [line]
      loss_max:
        description: >-
          the loss at a line's rating — the top of the curve being approximated,
          carried as a column because a bound takes a name or a number, and given
          only for the lines that dissipate anything
        dims: [line]
      loss_slope:
        description: >-
          the slope of this segment's half-plane — how much loss the flow buys
          along it. Where the segments come from is the instance's business, not
          the model's: a tangent to the loss curve and a secant across it both
          arrive here as a slope and an offset.
        dims: [line, segment]
      loss_offset:
        description: >-
          where this segment's half-plane meets the loss axis, negative for a curve
          through the origin
        dims: [line, segment]
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
          flow on a line, signed towards its `to` bus — unbounded here, because the
          rating covers the flow *and* its loss and so is a row rather than a bound
        foreach: [snapshot, line]
      loss:
        description: >-
          the energy a line dissipates carrying its flow — pushed down by the
          objective and held up by the tangents, so it settles on the approximated
          curve rather than needing an equality of its own. A line with no
          resistance dissipates nothing, which is a loss of zero rather than a
          quantity with no value, so the balances and ratings that name it keep
          their rows.
        foreach: [snapshot, line]
        where: loss_max
        absence: zero
        bounds:
          lower: 0
          upper: loss_max

    constraints:
      nodal_balance:
        description: >-
          what is generated at a bus plus what arrives over the lines meets the load
          there, less half of each incident line's loss — PyPSA's convention is that
          a branch dissipates half at either end
        foreach: [snapshot, bus]
        expression: >-
          sum(p, by=gen_bus)
          + sum(f, by=to) - sum(f, by=from)
          - 0.5 * sum(loss, by=from) - 0.5 * sum(loss, by=to)
          == load

      within_rating_forward:
        description: >-
          a line's rating limits what it carries plus what it dissipates, so the
          loss eats into the capacity rather than riding on top of it
        foreach: [snapshot, line]
        expression: f + loss <= s_nom

      within_rating_reverse:
        description: the same limit for flow the other way
        foreach: [snapshot, line]
        expression: f - loss >= neg_s_nom

      loss_above_segment_forward:
        description: >-
          the loss sits above every one of its half-planes, which for a convex
          curve is the whole approximation. Only the lines that have a curve get a
          fan.
        foreach: [snapshot, line, segment]
        where: loss_max
        expression: loss + loss_slope * f >= loss_offset

      loss_above_segment_reverse:
        description: the same fan mirrored, because the loss depends on the flow's magnitude
        foreach: [snapshot, line, segment]
        where: loss_max
        expression: loss - loss_slope * f >= loss_offset

    objective:
      sense: minimize
      description: what the fleet costs to run; the losses are paid for as extra generation
      expression: sum(p * marginal_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/ports/pypsa_losses.yaml', sources) as solution:
        solution.objective  # 24114.237385131008
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_losses.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the lpspec call binds as ``sources``.

        PyPSA is given ``r``, ``x`` and ``s_nom`` and derives the tangents itself.
        The port is given the tangents, because a slope of ``2 * r * p_k`` is
        arithmetic and the language's coefficients take a name or a number — the
        same reason ``pypsa_storage`` ships ``soc_max`` rather than a ratio. Both
        sides therefore describe one model from the same instance, and
        ``SEGMENTS`` is the one number that has to agree between them.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.add('Bus', tables['bus']['bus'])

        lines: pd.DataFrame = tables['line'].set_index('line')
        n.add(
            'Line',
            lines.index,
            bus0=lines['from'],
            bus1=lines['to'],
            r=tables['r'].set_index('line')['value'],
            x=tables['x'].set_index('line')['value'],
            s_nom=tables['s_nom'].set_index('line')['value'],
        )

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        n.add(
            'Generator',
            generators.index,
            bus=generators['gen_bus'],
            p_nom=tables['p_nom'].set_index('generator')['value'],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
        )

        load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
        for bus in load.columns:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
        return n
    ```

**The rating covers the flow *and* its loss.** This is the one row that is not
obvious from the formulation, and getting it wrong is invisible in two snapshots
out of three: with losses enabled PyPSA replaces the flow bound `|s| ≤ s_nom`
with `s + loss ≤ s_nom` and `s − loss ≥ −s_nom`. So the loss eats into the
capacity rather than riding on top of it, and the flow limit stops being a bound
and becomes a row. A first draft of this port kept the bound, matched PyPSA
exactly at snapshots 0 and 2, and pushed the flow to a full 120 at snapshot 1
where PyPSA stops at 115.97 — 115.97 + 4.03 being exactly 120.

**Losses put a gradient in the prices.** Between `b0` and `b2` the recorded duals
climb `10.00 → 85.78 → 90.00`: power is worth more the further it has to travel,
which is the entire point of modelling losses and something no objective figure
shows. Across the **lossless** line the price does not move at all — `b3` differs
from `b2` only because the local generator sets it.

**The mask needs `absence: zero`, and the instance proves it.** `loss` is
declared `where: loss_max`, so the resistanceless line has no loss variable. But
`loss` also appears as a bare term in the rating rows — `f + loss <= s_nom` —
and a constraint naming a masked variable loses its **row**. Without
`absence: zero` those two rows vanish for that line, it becomes uncapacitated,
and the model reports a cheaper answer with nothing flagged:

| spelling | rows | `omissions` | objective |
|---|---|---|---|
| `where:` + `absence: zero` | 132 | 0 | **24114.24** ✔ |
| `where:` alone | 120 | **2** | 17514.24 ✘ |

Twenty-seven per cent low, and an optimal status — but no longer silent: since
[#944](https://github.com/fluxopt/lpspec/issues/944) a row a propagated absence
deleted is counted, so `diagnostics().omissions` names the two rating rows that
went. The wrong model announces itself where it used to shrug.

**`r` is 0.0003, not a per-unit textbook figure.** PyPSA's loss term is
`r_pu_eff · s²` with `s` in MW, so a resistance chosen for a per-unit base makes
the loss exceed the flow — at `r = 0.05` this instance is *infeasible*, the
generators unable to cover a loss larger than the demand. At this value losses
run about 3% of throughput.

**PyPSA's *other* loss mode is this same model.** Its default is secants rather
than tangents — secants lie above a convex curve where tangents lie below, so
they overestimate the losses these underestimate — and it emits the identical
rows: one half-plane per segment per sign of the flow. Only the coefficients
differ, and how many of them there are.

So it gets no model file of its own. `test_the_two_loss_approximations_are_one_model`
binds *this* model to those coefficients and reaches PyPSA's own secant optimum,
which is the stronger claim: the two approximations are one thing the language
says once.

That is also why the model's parameters are called `loss_slope` and
`loss_offset` rather than anything with *tangent* in it, and why the
coefficients are **dumped** from PyPSA rather than recomputed here. Where the
breakpoints fall is the instance's business — a secant's come out of an error
tolerance, and even their number does — and the model asks only for a slope and
an offset per segment.

## What it exercises

A third dimension that exists only to index an approximation — `segment` is not
a thing in the network, it is a row multiplier — and a variable pinned between
an objective pushing down and a fan of constraints pushing up, with no equality
defining it anywhere.

The tangent slopes and offsets ship as data. `2 · r · p_k` is arithmetic, and a
coefficient here takes a name or a number, the same reason
[rung 3](pypsa_storage.md) ships `soc_max` rather than a ratio. That is the
ergonomics case for a method deriving the fan from the curve: two columns and a
segment count instead of six precomputed rows. What it is **not** is a
capability gap — this port needs no construct the language lacks.

[`method: lp`](https://math-spec.readthedocs.io/en/latest/reference/language/piecewise/#lp-the-one-that-declares-nothing)
emits rows of exactly this shape — one linear row per piece, no auxiliary
variable — but it is **not** a drop-in here, and the difference is the whole
approximation. It states the lines through consecutive breakpoints, which for a
convex curve lie *above* it; PyPSA's tangents lie *below*. The two bracket
`r · s²` from opposite sides, so swapping one for the other moves the optimum
rather than restating it, and this port keeps the fan PyPSA publishes.
