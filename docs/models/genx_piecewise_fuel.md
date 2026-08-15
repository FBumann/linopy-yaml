# GenX — piecewise fuel

A day of dispatch for two carbon-capture plants and a wind farm under a net-zero carbon cap, where the gas plant's fuel use bends with its output.

> **✔ Verified against GenX** — objective **2341.8230753008093**, matched to `rtol=1e-09`. Asserted upstream in `test/test_piecewisefuel.jl`, and re-run here on GenX itself.

**A plant burns one fuel, and that fuel's price moves hour by hour.** So the
price a plant pays is a `(fuel, hour)` table read through the plant's own fuel —
a lookup whose target keeps a dimension after the read. Every other reach-through
in this corpus lands on a single value; this one lands on a row.

The instance folds that read into `fuel_price[plant, hour]` because the
mapping is one-to-one here, and the page says so rather than pretending
otherwise: what the model needs is the *joined* price, and where a plant's fuel
is a declared map the join is the language's to do.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{P}$ | index $p$ --- `plant` |
| $\mathcal{H}$ | index $h$ --- `hour` |
| $\mathcal{S}$ | index $s$ --- `segment` |
| $\mathcal{T}$ | index $t$ --- `step` |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\mathit{unit\_size}$ | `unit_size` over $\mathcal{P}$ |
| $\mathit{units}^{\mathrm{available}}$ | `units_available` over $\mathcal{P}$ |
| $\mathit{availability}$ | `availability` over $\mathcal{P} \times \mathcal{H}$ |
| $\mathit{min\_output}$ | `min_output` over $\mathcal{P}$ |
| $\mathit{ramp}$ | `ramp` over $\mathcal{P}$ |
| $\mathit{start\_headroom}$ | `start_headroom` over $\mathcal{P} \times \mathcal{H}$ |
| $\mathit{is\_thermal}$ | `is_thermal` over $\mathcal{P}$ |
| $\mathit{uses\_curve}$ | `uses_curve` over $\mathcal{P}$ |
| $\mathit{fuel\_slope}$ | `fuel_slope` over $\mathcal{P} \times \mathcal{S}$ |
| $\mathit{fuel\_intercept}$ | `fuel_intercept` over $\mathcal{P} \times \mathcal{S}$ |
| $\mathit{heat\_rate}$ | `heat_rate` over $\mathcal{P}$ |
| $\mathit{start\_fuel}$ | `start_fuel` over $\mathcal{P}$ |
| $\mathit{fuel\_price}$ | `fuel_price` over $\mathcal{P} \times \mathcal{H}$ |
| $\mathit{run\_cost}$ | `run_cost` over $\mathcal{P}$ |
| $\mathit{start\_cost}$ | `start_cost` over $\mathcal{P}$ |
| $\mathit{weight}$ | `weight` over $\mathcal{H}$ |
| $\mathit{emitted}$ | `emitted` over $\mathcal{P}$ |
| $\mathit{emitted}^{\mathrm{start}}$ | `emitted_start` over $\mathcal{P}$ |
| $\mathit{captured}$ | `captured` over $\mathcal{P}$ |
| $\mathit{captured}^{\mathrm{start}}$ | `captured_start` over $\mathcal{P}$ |
| $\mathit{carbon\_cap}$ | `carbon_cap` (scalar) |
| $\mathit{demand}$ | `demand` over $\mathcal{H}$ |
| $\mathit{shed}^{\mathrm{cost}}$ | `shed_cost` over $\mathcal{T}$ |
| $\mathit{shed}^{\mathrm{limit}}$ | `shed_limit` over $\mathcal{T}$ |

#### Variables

| Symbol | Meaning |
|---|---|
| $\mathit{output}$ | `output` over $\mathcal{P} \times \mathcal{H}$ |
| $\mathit{burned}$ | `burned` over $\mathcal{P} \times \mathcal{H}$ |
| $\mathit{burned}^{\mathrm{starting}}$ | `burned_starting` over $\mathcal{P} \times \mathcal{H}$ |
| $\mathit{committed}$ | `committed` over $\mathcal{P} \times \mathcal{H}$ |
| $\mathit{starting}$ | `starting` over $\mathcal{P} \times \mathcal{H}$ |
| $\mathit{shutting}$ | `shutting` over $\mathcal{P} \times \mathcal{H}$ |
| $\mathit{shed}$ | `shed` over $\mathcal{T} \times \mathcal{H}$ |
| $\mathit{units}$ | `units` over $\mathcal{P}$ |

$t \ominus k$ denotes cyclic translation: index $t-k$ taken modulo the size of the dimension (`roll`). Plain $t-k$ (`shift`) has no wraparound --- terms translated past the edge are simply absent.

#### Objective

$$\min \sum_{p \in \mathcal{P},\enspace h \in \mathcal{H},\enspace t \in \mathcal{T}} \left( \mathit{output}_{p,h} \cdot \mathit{run\_cost}_{p} \cdot \mathit{weight}_{h} + \mathit{burned}_{p,h} \cdot \mathit{fuel\_price}_{p,h} \cdot \mathit{weight}_{h} + \mathit{burned}^{\mathrm{starting}}_{p,h} \cdot \mathit{fuel\_price}_{p,h} \cdot \mathit{weight}_{h} + \mathit{starting}_{p,h} \cdot \mathit{start\_cost}_{p} \cdot \mathit{weight}_{h} + \mathit{shed}_{t,h} \cdot \mathit{shed}^{\mathrm{cost}}_{t} \cdot \mathit{weight}_{h} + \mathit{burned}_{p,h} \cdot \mathit{captured}_{p} \cdot \mathit{weight}_{h} + \mathit{burned}^{\mathrm{starting}}_{p,h} \cdot \mathit{captured}^{\mathrm{start}}_{p} \cdot \mathit{weight}_{h} \right)$$

#### Subject to

**`meet_demand`**

$$\sum_{p \in \mathcal{P}} \mathit{output}_{p,h} + \sum_{t \in \mathcal{T}} \mathit{shed}_{t,h} = \mathit{demand}_{h} \qquad \forall\thinspace h \in \mathcal{H}$$

**`shed_within_step`**

$$\mathit{shed}_{t,h} \le \mathit{shed}^{\mathrm{limit}}_{t} \cdot \mathit{demand}_{h} \qquad \forall\thinspace t \in \mathcal{T},\enspace h \in \mathcal{H}$$

**`committed_units_exist`**

$$\mathit{committed}_{p,h} \le \mathit{units}_{p} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathit{is\_thermal}_{p} > 0$$

**`thermal_ceiling`**

$$\mathit{output}_{p,h} \le \mathit{committed}_{p,h} \cdot \mathit{unit\_size}_{p} \cdot \mathit{availability}_{p,h} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathit{is\_thermal}_{p} > 0$$

**`thermal_floor`**

$$\mathit{output}_{p,h} \ge \mathit{committed}_{p,h} \cdot \mathit{unit\_size}_{p} \cdot \mathit{min\_output}_{p} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathit{is\_thermal}_{p} > 0$$

**`variable_ceiling`**

$$\mathit{output}_{p,h} \le \mathit{units}_{p} \cdot \mathit{unit\_size}_{p} \cdot \mathit{availability}_{p,h} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathit{is\_thermal}_{p} = 0$$

**`commitment_tracks_starts`**

$$\mathit{committed}_{p,h} - \mathit{committed}_{p,h \ominus 1} = \mathit{starting}_{p,h} - \mathit{shutting}_{p,h} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathit{is\_thermal}_{p} > 0$$

**`stay_up_once_started`**

$$\mathit{committed}_{p,h} \ge \mathit{starting}_{p,h} + \mathit{starting}_{p,h \ominus 1} + \mathit{starting}_{p,h \ominus 2} + \mathit{starting}_{p,h \ominus 3} + \mathit{starting}_{p,h \ominus 4} + \mathit{starting}_{p,h \ominus 5} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathit{is\_thermal}_{p} > 0$$

**`stay_down_once_shut`**

$$\mathit{units}_{p} - \mathit{committed}_{p,h} \ge \mathit{shutting}_{p,h} + \mathit{shutting}_{p,h \ominus 1} + \mathit{shutting}_{p,h \ominus 2} + \mathit{shutting}_{p,h \ominus 3} + \mathit{shutting}_{p,h \ominus 4} + \mathit{shutting}_{p,h \ominus 5} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathit{is\_thermal}_{p} > 0$$

**`ramp_up`**

$$\mathit{output}_{p,h} - \mathit{output}_{p,h \ominus 1} \le \mathit{ramp}_{p} \cdot \mathit{unit\_size}_{p} \cdot \left( \mathit{committed}_{p,h} - \mathit{starting}_{p,h} \right) + \mathit{start\_headroom}_{p,h} \cdot \mathit{unit\_size}_{p} \cdot \mathit{starting}_{p,h} - \mathit{min\_output}_{p} \cdot \mathit{unit\_size}_{p} \cdot \mathit{shutting}_{p,h} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathit{is\_thermal}_{p} > 0$$

**`ramp_down`**

$$\mathit{output}_{p,h \ominus 1} - \mathit{output}_{p,h} \le \mathit{ramp}_{p} \cdot \mathit{unit\_size}_{p} \cdot \left( \mathit{committed}_{p,h} - \mathit{starting}_{p,h} \right) - \mathit{min\_output}_{p} \cdot \mathit{unit\_size}_{p} \cdot \mathit{starting}_{p,h} + \mathit{start\_headroom}_{p,h} \cdot \mathit{unit\_size}_{p} \cdot \mathit{shutting}_{p,h} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathit{is\_thermal}_{p} > 0$$

**`fuel_above_each_piece`**

$$\mathit{burned}_{p,h} \ge \mathit{fuel\_slope}_{p,s} \cdot \mathit{output}_{p,h} + \mathit{fuel\_intercept}_{p,s} \cdot \mathit{committed}_{p,h} \qquad \forall\thinspace p \in \mathcal{P},\enspace s \in \mathcal{S},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathit{uses\_curve}_{p} > 0$$

**`fuel_at_the_heat_rate`**

$$\mathit{burned}_{p,h} = \mathit{heat\_rate}_{p} \cdot \mathit{output}_{p,h} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathit{uses\_curve}_{p} = 0$$

**`fuel_to_start`**

$$\mathit{burned}^{\mathrm{starting}}_{p,h} = \mathit{unit\_size}_{p} \cdot \mathit{starting}_{p,h} \cdot \mathit{start\_fuel}_{p} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H}$$

**`carbon_budget`**

$$\sum_{p \in \mathcal{P}} \sum_{h \in \mathcal{H}} \left( \mathit{burned}_{p,h} \cdot \mathit{emitted}_{p} \cdot \mathit{weight}_{h} + \mathit{burned}^{\mathrm{starting}}_{p,h} \cdot \mathit{emitted}^{\mathrm{start}}_{p} \cdot \mathit{weight}_{h} \right) \le \mathit{carbon\_cap}$$

#### Variable domains

**`output`**

$$\mathit{output}_{p,h} \ge 0 \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H}$$

**`burned`**

$$\mathit{burned}_{p,h} \ge 0 \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H}$$

**`burned_starting`**

$$\mathit{burned}^{\mathrm{starting}}_{p,h} \ge 0 \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H}$$

**`committed`**

$$\mathit{committed}_{p,h} \ge 0 \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H}$$

**`starting`**

$$\mathit{starting}_{p,h} \ge 0 \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H}$$

**`shutting`**

$$\mathit{shutting}_{p,h} \ge 0 \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H}$$

**`shed`**

$$\mathit{shed}_{t,h} \ge 0 \qquad \forall\thinspace t \in \mathcal{T},\enspace h \in \mathcal{H}$$

**`units`**

$$0 \le \mathit{units}_{p} \le \mathit{units}^{\mathrm{available}}_{p} \qquad \forall\thinspace p \in \mathcal{P}$$

</details>
<!-- math:end -->

The tabs start from [the instance’s tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    # GenX's piecewise-fuel case: a day of dispatch for two carbon-capture plants
    # and a wind farm under a net-zero carbon cap, where the gas plant's fuel use
    # is a piecewise-linear function of its output.
    # Optimum 2341.82308, from GenX itself.
    #
    # A plant burns one fuel and the fuel's price moves hour by hour, so the price
    # a plant pays is its fuel's price — a table with a dimension left over after
    # the plant has chosen its fuel.

    dimensions:
      plant:
        dtype: str
      hour:
        dtype: int
      segment:
        dtype: int  # a piece of the fuel curve
      step:
        dtype: int  # a block of demand that may be shed, each dearer than the last

    parameters:
      unit_size:
        dims: [plant]
      units_available:
        dims: [plant]
      availability:
        dims: [plant, hour]
      min_output:
        dims: [plant]
      ramp:
        dims: [plant]
      start_headroom:
        dims: [plant, hour]
      is_thermal:
        dims: [plant]
      uses_curve:
        dims: [plant]

      fuel_slope:
        dims: [plant, segment]
      fuel_intercept:
        dims: [plant, segment]
      heat_rate:
        dims: [plant]
      start_fuel:
        dims: [plant]
      fuel_price:
        dims: [plant, hour]

      run_cost:
        dims: [plant]
      start_cost:
        dims: [plant]
      weight:
        dims: [hour]

      emitted:
        dims: [plant]
      emitted_start:
        dims: [plant]
      captured:
        dims: [plant]
      captured_start:
        dims: [plant]
      carbon_cap:
        dims: []

      demand:
        dims: [hour]
      shed_cost:
        dims: [step]
      shed_limit:
        dims: [step]

    variables:
      output:
        foreach: [plant, hour]
        bounds:
          lower: 0
      burned:
        foreach: [plant, hour]
        bounds:
          lower: 0
      burned_starting:
        foreach: [plant, hour]
        bounds:
          lower: 0
      # Commitment is counted in units and relaxed to a continuous variable, which
      # is what GenX's UCommit=2 does.
      committed:
        foreach: [plant, hour]
        bounds:
          lower: 0
      starting:
        foreach: [plant, hour]
        bounds:
          lower: 0
      shutting:
        foreach: [plant, hour]
        bounds:
          lower: 0
      shed:
        foreach: [step, hour]
        bounds:
          lower: 0
      units:
        foreach: [plant]
        bounds:
          lower: 0
          upper: units_available

    expressions:
      # The day is a representative period that repeats, so hour 1 follows hour 24.
      started_recently: >-
        starting + shift(starting, over=hour, by=1, edge='wrap')
        + shift(starting, over=hour, by=2, edge='wrap') + shift(starting, over=hour, by=3, edge='wrap')
        + shift(starting, over=hour, by=4, edge='wrap') + shift(starting, over=hour, by=5, edge='wrap')
      shut_recently: >-
        shutting + shift(shutting, over=hour, by=1, edge='wrap')
        + shift(shutting, over=hour, by=2, edge='wrap') + shift(shutting, over=hour, by=3, edge='wrap')
        + shift(shutting, over=hour, by=4, edge='wrap') + shift(shutting, over=hour, by=5, edge='wrap')

    constraints:
      meet_demand:
        foreach: [hour]
        expression: sum(output, over=plant) + sum(shed, over=step) == demand

      shed_within_step:
        foreach: [step, hour]
        expression: shed <= shed_limit * demand

      committed_units_exist:
        foreach: [plant, hour]
        where: "is_thermal > 0"
        expression: committed <= units

      thermal_ceiling:
        foreach: [plant, hour]
        where: "is_thermal > 0"
        expression: output <= committed * unit_size * availability

      thermal_floor:
        foreach: [plant, hour]
        where: "is_thermal > 0"
        expression: output >= committed * unit_size * min_output

      variable_ceiling:
        foreach: [plant, hour]
        where: "is_thermal == 0"
        expression: output <= units * unit_size * availability

      commitment_tracks_starts:
        foreach: [plant, hour]
        where: "is_thermal > 0"
        expression: committed - shift(committed, over=hour, by=1, edge='wrap') == starting - shutting

      stay_up_once_started:
        foreach: [plant, hour]
        where: "is_thermal > 0"
        expression: committed >= started_recently

      stay_down_once_shut:
        foreach: [plant, hour]
        where: "is_thermal > 0"
        expression: units - committed >= shut_recently

      ramp_up:
        foreach: [plant, hour]
        where: "is_thermal > 0"
        expression: >-
          output - shift(output, over=hour, by=1, edge='wrap')
          <= ramp * unit_size * (committed - starting)
          + start_headroom * unit_size * starting
          - min_output * unit_size * shutting

      ramp_down:
        foreach: [plant, hour]
        where: "is_thermal > 0"
        expression: >-
          shift(output, over=hour, by=1, edge='wrap') - output
          <= ramp * unit_size * (committed - starting)
          - min_output * unit_size * starting
          + start_headroom * unit_size * shutting

      # Fuel use is above every piece of the curve, so at the optimum it sits on
      # the binding one. The no-load intercept is charged per committed unit.
      fuel_above_each_piece:
        foreach: [plant, segment, hour]
        where: "uses_curve > 0"
        expression: burned >= fuel_slope * output + fuel_intercept * committed

      fuel_at_the_heat_rate:
        foreach: [plant, hour]
        where: "uses_curve == 0"
        expression: burned == heat_rate * output

      fuel_to_start:
        foreach: [plant, hour]
        expression: burned_starting == unit_size * starting * start_fuel

      # A net-zero cap: what escapes capture, less what the biomass took up.
      carbon_budget:
        foreach: []
        expression: >-
          sum(sum(burned * emitted * weight + burned_starting * emitted_start * weight, over=hour), over=plant)
          <= carbon_cap

    objective:
      sense: minimize
      expression: >-
        output * run_cost * weight
        + burned * fuel_price * weight
        + burned_starting * fuel_price * weight
        + starting * start_cost * weight
        + shed * shed_cost * weight
        + burned * captured * weight
        + burned_starting * captured_start * weight
    ```

## What the port had to decide

**A piecewise fuel curve is a floor per piece.** GenX gives the gas plant two
segments — 6.0 MMBtu/MWh above a 0.4 no-load intercept, then 7.2 above 0.208 —
and requires fuel use to be at least each of them. At the optimum it rests on
whichever binds, so the curve needs no binaries and no `piecewise:` block: it is
one constraint over a `segment` axis. The intercept is charged per *committed
unit*, which is why commitment has to be a variable even though nothing here is
integral.

**Commitment is continuous, and the day wraps.** `UCommit=2` relaxes the
commitment variables, and the 24 hours are a representative period that repeats,
so `shift(edge='wrap')` is exactly right: hour 1 follows hour 24. The six-hour
minimum up and down times are the same for both plants, so they expand as six
shifted terms — the macro form the ledger describes. Where the times differ by
plant, the window becomes an incidence table instead
([#791](https://github.com/fluxopt/lpspec/issues/791)).

**Negative emissions are a coefficient, not a special case.** The biomass plant
captures 90% of its carbon and the fuel counts its own uptake, so a burned
MMBtu emits `0.05306 × ((1 − 0.9) − 1)` — **−0.0855**. Against a cap of zero
that is what lets the gas plant run at all. Startup burns at a lower capture
fraction (0.6), so it carries its own coefficient.

## Checked component by component

GenX exposes its cost expressions, so the port is checked against six numbers
rather than one:

| | GenX | port |
|---|---|---|
| variable O&M | 208.8176259987514 | 208.8176259988 |
| fuel | 1397.790027451872 | 1397.7900274519 |
| start fuel | 32.36623248939999 | 32.3662324894 |
| start | 368.1658945669249 | 368.1658945669 |
| non-served energy | 0.0 | 0.0 |
| carbon disposal | *(residual)* | 334.6832947939 |

The residual is what the objective leaves after the five GenX prints, and it
lands on the port's own carbon term to ten digits. A formulation error that
happened to preserve the total would still move one of these.

## What it exercises

`shift(edge='wrap')` on a representative day, a piecewise curve as a floor per
piece rather than a formulation, and a carbon budget whose coefficients carry
capture and uptake. No new construct — the interest is that a framework's
dispatch model, settings and all, is a page of declarations.
