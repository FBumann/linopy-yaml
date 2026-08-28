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

GenX's piecewise-fuel case: a day of dispatch for two carbon-capture plants and a wind farm under a net-zero carbon cap, where the gas plant's fuel use is a piecewise-linear function of its output. A plant burns one fuel and the fuel's price moves hour by hour, so the price a plant pays is its fuel's price — a table with a dimension left over after the plant has chosen its fuel. Optimum 2341.82308, from GenX itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{P}$ | index $p$ — `plant` carrying labels $\mathrm{commitment},\enspace \mathrm{fuel\_use}$ — the units dispatched over the day |
| $\mathcal{H}$ | index $h$ — `hour` — hours of a representative day that repeats |
| $\mathcal{S}$ | index $s$ — `segment` — a piece of the fuel curve |
| $\mathcal{T}$ | index $t$ — `step` — a block of demand that may be shed, each dearer than the last |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\mathrm{unit\_size}$ | `unit_size` over $\mathcal{P}$ — capacity of one unit of a plant |
| $\mathrm{units}^{\mathrm{available}}$ | `units_available` over $\mathcal{P}$ — how many units of a plant may be committed |
| $\mathrm{availability}$ | `availability` over $\mathcal{P} \times \mathcal{H}$ — share of its capacity a plant can offer in an hour |
| $\mathrm{min\_output}$ | `min_output` over $\mathcal{P}$ — share of unit size a committed unit must produce |
| $\mathrm{ramp}$ | `ramp` over $\mathcal{P}$ — share of unit size output may change by from one hour to the next |
| $\mathrm{start\_headroom}$ | `start_headroom` over $\mathcal{P} \times \mathcal{H}$ — share of unit size a unit may reach in the hour it starts |
| $\mathrm{fuel\_slope}$ | `fuel_slope` over $\mathcal{P} \times \mathcal{S}$ — fuel per unit of output on one piece of the curve |
| $\mathrm{fuel\_intercept}$ | `fuel_intercept` over $\mathcal{P} \times \mathcal{S}$ — no-load fuel of one piece, charged per committed unit |
| $\mathrm{heat\_rate}$ | `heat_rate` over $\mathcal{P}$ — fuel per unit of output for a plant with no curve |
| $\mathrm{start\_fuel}$ | `start_fuel` over $\mathcal{P}$ — fuel burned per unit of capacity started |
| $\mathrm{fuel\_price}$ | `fuel_price` over $\mathcal{P} \times \mathcal{H}$ — what a unit of the plant's fuel costs in that hour |
| $\mathrm{run\_cost}$ | `run_cost` over $\mathcal{P}$ — variable cost of one unit of output, fuel aside |
| $\mathrm{start\_cost}$ | `start_cost` over $\mathcal{P}$ — what starting one unit of capacity costs |
| $\mathrm{weight}$ | `weight` over $\mathcal{H}$ — how many real hours an hour of the representative day stands for |
| $\mathrm{emitted}$ | `emitted` over $\mathcal{P}$ — net CO2 per unit of fuel burned after capture, negative where the fuel took it up |
| $\mathrm{emitted}^{\mathrm{start}}$ | `emitted_start` over $\mathcal{P}$ — net CO2 per unit of start-up fuel burned |
| $\mathrm{captured}$ | `captured` over $\mathcal{P}$ — what capturing the CO2 from a unit of fuel costs |
| $\mathrm{captured}^{\mathrm{start}}$ | `captured_start` over $\mathcal{P}$ — what capturing the CO2 from a unit of start-up fuel costs |
| $\mathrm{carbon\_cap}$ | `carbon_cap` (scalar) — emissions the day is allowed, net of uptake |
| $\mathrm{demand}$ | `demand` over $\mathcal{H}$ — demand to be met in an hour |
| $\mathrm{shed}^{\mathrm{cost}}$ | `shed_cost` over $\mathcal{T}$ — what shedding a unit of demand in this block costs |
| $\mathrm{shed}^{\mathrm{limit}}$ | `shed_limit` over $\mathcal{T}$ — share of the hour's demand this block may shed |

#### Variables

| Symbol | Meaning |
|---|---|
| $\mathit{output}$ | `output` over $\mathcal{P} \times \mathcal{H}$ — what a plant produces in an hour |
| $\mathit{burned}$ | `burned` over $\mathcal{P} \times \mathcal{H}$ — fuel a plant burns running in an hour |
| $\mathit{burned}^{\mathrm{starting}}$ | `burned_starting` over $\mathcal{P} \times \mathcal{H}$ — fuel a plant burns starting units in an hour |
| $\mathit{committed}$ | `committed` over $\mathcal{P} \times \mathcal{H}$ — how many units of a plant are committed in an hour — counted in units and relaxed to a continuous variable, which is what GenX's UCommit=2 does |
| $\mathit{starting}$ | `starting` over $\mathcal{P} \times \mathcal{H}$ — units of a plant brought up entering an hour |
| $\mathit{shutting}$ | `shutting` over $\mathcal{P} \times \mathcal{H}$ — units of a plant taken down entering an hour |
| $\mathit{shed}$ | `shed` over $\mathcal{T} \times \mathcal{H}$ — demand shed out of a block in an hour |
| $\mathit{units}$ | `units` over $\mathcal{P}$ — how many units of a plant stand available all day |

Upright is what the model is given — a parameter such as $\mathrm{unit\_size}$, a coordinate map, a label — and italic is what the solver chooses, such as $\mathit{output}$. An index is italic too, being what a quantifier chooses, and a set is script.

$t \ominus k$ denotes cyclic translation: index $t-k$ taken modulo the size of the dimension (`roll`). Plain $t-k$ (`shift`) has no wraparound — terms translated past the edge are simply absent.

#### Objective

$$\min \sum_{p \in \mathcal{P},\enspace h \in \mathcal{H}} \mathit{output}_{p,h} \cdot \mathrm{run\_cost}_{p} \cdot \mathrm{weight}_{h} + \sum_{p \in \mathcal{P},\enspace h \in \mathcal{H}} \mathit{burned}_{p,h} \cdot \mathrm{fuel\_price}_{p,h} \cdot \mathrm{weight}_{h} + \sum_{p \in \mathcal{P},\enspace h \in \mathcal{H}} \mathit{burned}^{\mathrm{starting}}_{p,h} \cdot \mathrm{fuel\_price}_{p,h} \cdot \mathrm{weight}_{h} + \sum_{p \in \mathcal{P},\enspace h \in \mathcal{H}} \mathit{starting}_{p,h} \cdot \mathrm{start\_cost}_{p} \cdot \mathrm{weight}_{h} + \sum_{h \in \mathcal{H},\enspace t \in \mathcal{T}} \mathit{shed}_{t,h} \cdot \mathrm{shed}^{\mathrm{cost}}_{t} \cdot \mathrm{weight}_{h} + \sum_{p \in \mathcal{P},\enspace h \in \mathcal{H}} \mathit{burned}_{p,h} \cdot \mathrm{captured}_{p} \cdot \mathrm{weight}_{h} + \sum_{p \in \mathcal{P},\enspace h \in \mathcal{H}} \mathit{burned}^{\mathrm{starting}}_{p,h} \cdot \mathrm{captured}^{\mathrm{start}}_{p} \cdot \mathrm{weight}_{h}$$

#### Subject to

**`meet_demand`**

$$\sum_{p \in \mathcal{P}} \mathit{output}_{p,h} + \sum_{t \in \mathcal{T}} \mathit{shed}_{t,h} = \mathrm{demand}_{h} \qquad \forall\thinspace h \in \mathcal{H}$$

**`shed_within_step`**

$$\mathit{shed}_{t,h} \le \mathrm{shed}^{\mathrm{limit}}_{t} \cdot \mathrm{demand}_{h} \qquad \forall\thinspace t \in \mathcal{T},\enspace h \in \mathcal{H}$$

**`committed_units_exist`**

$$\mathit{committed}_{p,h} \le \mathit{units}_{p} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathrm{commitment}(p) = \text{'}\mathrm{unit}\text{'}$$

**`thermal_ceiling`**

$$\mathit{output}_{p,h} \le \mathit{committed}_{p,h} \cdot \mathrm{unit\_size}_{p} \cdot \mathrm{availability}_{p,h} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathrm{commitment}(p) = \text{'}\mathrm{unit}\text{'}$$

**`thermal_floor`**

$$\mathit{output}_{p,h} \ge \mathit{committed}_{p,h} \cdot \mathrm{unit\_size}_{p} \cdot \mathrm{min\_output}_{p} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathrm{commitment}(p) = \text{'}\mathrm{unit}\text{'}$$

**`variable_ceiling`**

$$\mathit{output}_{p,h} \le \mathit{units}_{p} \cdot \mathrm{unit\_size}_{p} \cdot \mathrm{availability}_{p,h} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathrm{commitment}(p) = \text{'}\mathrm{free}\text{'}$$

**`commitment_tracks_starts`**

$$\mathit{committed}_{p,h} - \mathit{committed}_{p,h \ominus 1} = \mathit{starting}_{p,h} - \mathit{shutting}_{p,h} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathrm{commitment}(p) = \text{'}\mathrm{unit}\text{'}$$

**`stay_up_once_started`**

$$\mathit{committed}_{p,h} \ge \mathit{starting}_{p,h} + \mathit{starting}_{p,h \ominus 1} + \mathit{starting}_{p,h \ominus 2} + \mathit{starting}_{p,h \ominus 3} + \mathit{starting}_{p,h \ominus 4} + \mathit{starting}_{p,h \ominus 5} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathrm{commitment}(p) = \text{'}\mathrm{unit}\text{'}$$

**`stay_down_once_shut`**

$$\mathit{units}_{p} - \mathit{committed}_{p,h} \ge \mathit{shutting}_{p,h} + \mathit{shutting}_{p,h \ominus 1} + \mathit{shutting}_{p,h \ominus 2} + \mathit{shutting}_{p,h \ominus 3} + \mathit{shutting}_{p,h \ominus 4} + \mathit{shutting}_{p,h \ominus 5} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathrm{commitment}(p) = \text{'}\mathrm{unit}\text{'}$$

**`ramp_up`**

$$\mathit{output}_{p,h} - \mathit{output}_{p,h \ominus 1} \le \mathrm{ramp}_{p} \cdot \mathrm{unit\_size}_{p} \cdot \left( \mathit{committed}_{p,h} - \mathit{starting}_{p,h} \right) + \mathrm{start\_headroom}_{p,h} \cdot \mathrm{unit\_size}_{p} \cdot \mathit{starting}_{p,h} - \mathrm{min\_output}_{p} \cdot \mathrm{unit\_size}_{p} \cdot \mathit{shutting}_{p,h} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathrm{commitment}(p) = \text{'}\mathrm{unit}\text{'}$$

**`ramp_down`**

$$\mathit{output}_{p,h \ominus 1} - \mathit{output}_{p,h} \le \mathrm{ramp}_{p} \cdot \mathrm{unit\_size}_{p} \cdot \left( \mathit{committed}_{p,h} - \mathit{starting}_{p,h} \right) - \mathrm{min\_output}_{p} \cdot \mathrm{unit\_size}_{p} \cdot \mathit{starting}_{p,h} + \mathrm{start\_headroom}_{p,h} \cdot \mathrm{unit\_size}_{p} \cdot \mathit{shutting}_{p,h} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathrm{commitment}(p) = \text{'}\mathrm{unit}\text{'}$$

**`fuel_above_each_piece`**

$$\mathit{burned}_{p,h} \ge \mathrm{fuel\_slope}_{p,s} \cdot \mathit{output}_{p,h} + \mathrm{fuel\_intercept}_{p,s} \cdot \mathit{committed}_{p,h} \qquad \forall\thinspace p \in \mathcal{P},\enspace s \in \mathcal{S},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathrm{fuel\_use}(p) = \text{'}\mathrm{curve}\text{'}$$

**`fuel_at_the_heat_rate`**

$$\mathit{burned}_{p,h} = \mathrm{heat\_rate}_{p} \cdot \mathit{output}_{p,h} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H} \thinspace:\thinspace \mathrm{fuel\_use}(p) = \text{'}\mathrm{flat}\text{'}$$

**`fuel_to_start`**

$$\mathit{burned}^{\mathrm{starting}}_{p,h} = \mathrm{unit\_size}_{p} \cdot \mathit{starting}_{p,h} \cdot \mathrm{start\_fuel}_{p} \qquad \forall\thinspace p \in \mathcal{P},\enspace h \in \mathcal{H}$$

**`carbon_budget`**

$$\sum_{p \in \mathcal{P}} \sum_{h \in \mathcal{H}} \left( \mathit{burned}_{p,h} \cdot \mathrm{emitted}_{p} \cdot \mathrm{weight}_{h} + \mathit{burned}^{\mathrm{starting}}_{p,h} \cdot \mathrm{emitted}^{\mathrm{start}}_{p} \cdot \mathrm{weight}_{h} \right) \le \mathrm{carbon\_cap}$$

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

$$0 \le \mathit{units}_{p} \le \mathrm{units}^{\mathrm{available}}_{p} \qquad \forall\thinspace p \in \mathcal{P}$$

</details>
<!-- math:end -->

The tabs start from [the instance's tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    description: >-
      GenX's piecewise-fuel case: a day of dispatch for two carbon-capture plants
      and a wind farm under a net-zero carbon cap, where the gas plant's fuel use
      is a piecewise-linear function of its output. A plant burns one fuel and the
      fuel's price moves hour by hour, so the price a plant pays is its fuel's
      price — a table with a dimension left over after the plant has chosen its
      fuel. Optimum 2341.82308, from GenX itself.

    dimensions:
      plant:
        description: the units dispatched over the day
        dtype: str
      hour:
        description: hours of a representative day that repeats
        dtype: int
      segment:
        description: a piece of the fuel curve
        dtype: int
      step:
        description: a block of demand that may be shed, each dearer than the last
        dtype: int

    lookups:
      commitment:
        description: whether a plant is committed unit by unit or dispatched freely
        over: plant
        dtype: str
      fuel_use:
        description: whether a plant's fuel use is read off the piecewise curve or a flat heat rate
        over: plant
        dtype: str

    parameters:
      unit_size:
        description: capacity of one unit of a plant
        dims: [plant]
      units_available:
        description: how many units of a plant may be committed
        dims: [plant]
      availability:
        description: share of its capacity a plant can offer in an hour
        dims: [plant, hour]
      min_output:
        description: share of unit size a committed unit must produce
        dims: [plant]
      ramp:
        description: share of unit size output may change by from one hour to the next
        dims: [plant]
      start_headroom:
        description: share of unit size a unit may reach in the hour it starts
        dims: [plant, hour]
      fuel_slope:
        description: fuel per unit of output on one piece of the curve
        dims: [plant, segment]
      fuel_intercept:
        description: no-load fuel of one piece, charged per committed unit
        dims: [plant, segment]
      heat_rate:
        description: fuel per unit of output for a plant with no curve
        dims: [plant]
      start_fuel:
        description: fuel burned per unit of capacity started
        dims: [plant]
      fuel_price:
        description: what a unit of the plant's fuel costs in that hour
        dims: [plant, hour]

      run_cost:
        description: variable cost of one unit of output, fuel aside
        dims: [plant]
      start_cost:
        description: what starting one unit of capacity costs
        dims: [plant]
      weight:
        description: how many real hours an hour of the representative day stands for
        dims: [hour]

      emitted:
        description: net CO2 per unit of fuel burned after capture, negative where the fuel took it up
        dims: [plant]
      emitted_start:
        description: net CO2 per unit of start-up fuel burned
        dims: [plant]
      captured:
        description: what capturing the CO2 from a unit of fuel costs
        dims: [plant]
      captured_start:
        description: what capturing the CO2 from a unit of start-up fuel costs
        dims: [plant]
      carbon_cap:
        description: emissions the day is allowed, net of uptake
        dims: []

      demand:
        description: demand to be met in an hour
        dims: [hour]
      shed_cost:
        description: what shedding a unit of demand in this block costs
        dims: [step]
      shed_limit:
        description: share of the hour's demand this block may shed
        dims: [step]

    variables:
      output:
        description: what a plant produces in an hour
        foreach: [plant, hour]
        bounds:
          lower: 0
      burned:
        description: fuel a plant burns running in an hour
        foreach: [plant, hour]
        bounds:
          lower: 0
      burned_starting:
        description: fuel a plant burns starting units in an hour
        foreach: [plant, hour]
        bounds:
          lower: 0
      committed:
        description: >-
          how many units of a plant are committed in an hour — counted in units and
          relaxed to a continuous variable, which is what GenX's UCommit=2 does
        foreach: [plant, hour]
        bounds:
          lower: 0
      starting:
        description: units of a plant brought up entering an hour
        foreach: [plant, hour]
        bounds:
          lower: 0
      shutting:
        description: units of a plant taken down entering an hour
        foreach: [plant, hour]
        bounds:
          lower: 0
      shed:
        description: demand shed out of a block in an hour
        foreach: [step, hour]
        bounds:
          lower: 0
      units:
        description: how many units of a plant stand available all day
        foreach: [plant]
        bounds:
          lower: 0
          upper: units_available

    expressions:
      started_recently:
        expression: >-
          starting + shift(starting, over=hour, offset=1, edge='wrap')
          + shift(starting, over=hour, offset=2, edge='wrap') + shift(starting, over=hour, offset=3, edge='wrap')
          + shift(starting, over=hour, offset=4, edge='wrap') + shift(starting, over=hour, offset=5, edge='wrap')
        description: >-
          units started in this hour or the five before it — the day is a
          representative period that repeats, so the first hour follows the last
      shut_recently:
        expression: >-
          shutting + shift(shutting, over=hour, offset=1, edge='wrap')
          + shift(shutting, over=hour, offset=2, edge='wrap') + shift(shutting, over=hour, offset=3, edge='wrap')
          + shift(shutting, over=hour, offset=4, edge='wrap') + shift(shutting, over=hour, offset=5, edge='wrap')
        description: units shut in this hour or the five before it

    constraints:
      meet_demand:
        description: what is produced plus what is shed meets the demand of the hour
        foreach: [hour]
        expression: sum(output, over=plant) + sum(shed, over=step) == demand

      shed_within_step:
        description: a block sheds no more than its share of the hour's demand
        foreach: [step, hour]
        expression: shed <= shed_limit * demand

      committed_units_exist:
        foreach: [plant, hour]
        where: "commitment == unit"
        expression: committed <= units

      thermal_ceiling:
        description: a committed unit produces no more than its available capacity
        foreach: [plant, hour]
        where: "commitment == unit"
        expression: output <= committed * unit_size * availability

      thermal_floor:
        description: a committed unit produces no less than its minimum
        foreach: [plant, hour]
        where: "commitment == unit"
        expression: output >= committed * unit_size * min_output

      variable_ceiling:
        description: a plant with no commitment produces no more than its available capacity
        foreach: [plant, hour]
        where: "commitment == free"
        expression: output <= units * unit_size * availability

      commitment_tracks_starts:
        description: what is committed changes only by what starts and what shuts
        foreach: [plant, hour]
        where: "commitment == unit"
        expression: committed - shift(committed, over=hour, offset=1, edge='wrap') == starting - shutting

      stay_up_once_started:
        description: a unit that started within the last six hours is still committed
        foreach: [plant, hour]
        where: "commitment == unit"
        expression: committed >= started_recently

      stay_down_once_shut:
        description: a unit that shut within the last six hours is still down
        foreach: [plant, hour]
        where: "commitment == unit"
        expression: units - committed >= shut_recently

      ramp_up:
        description: output rises no faster than the ramp allows, with extra room in the hour a unit starts
        foreach: [plant, hour]
        where: "commitment == unit"
        expression: >-
          output - shift(output, over=hour, offset=1, edge='wrap')
          <= ramp * unit_size * (committed - starting)
          + start_headroom * unit_size * starting
          - min_output * unit_size * shutting

      ramp_down:
        foreach: [plant, hour]
        where: "commitment == unit"
        expression: >-
          shift(output, over=hour, offset=1, edge='wrap') - output
          <= ramp * unit_size * (committed - starting)
          - min_output * unit_size * starting
          + start_headroom * unit_size * shutting

      fuel_above_each_piece:
        description: >-
          fuel use is above every piece of the curve, so at the optimum it sits on
          the binding one, and the no-load intercept is charged per committed unit
        foreach: [plant, segment, hour]
        where: "fuel_use == curve"
        expression: burned >= fuel_slope * output + fuel_intercept * committed

      fuel_at_the_heat_rate:
        description: a plant with no curve burns fuel at a flat heat rate
        foreach: [plant, hour]
        where: "fuel_use == flat"
        expression: burned == heat_rate * output

      fuel_to_start:
        description: starting a unit burns its own fuel, on top of what running it burns
        foreach: [plant, hour]
        expression: burned_starting == unit_size * starting * start_fuel

      carbon_budget:
        description: a net-zero cap — what escapes capture, less what the biomass took up
        foreach: []
        expression: >-
          sum(sum(burned * emitted * weight + burned_starting * emitted_start * weight, over=hour), over=plant)
          <= carbon_cap

    objective:
      sense: minimize
      description: >-
        the day's cost, scaled to the year — running, fuel, starts, shed demand and
        the capture the carbon-capture plants pay for
      expression: >-
        sum(output * run_cost * weight)
        + sum(burned * fuel_price * weight)
        + sum(burned_starting * fuel_price * weight)
        + sum(starting * start_cost * weight)
        + sum(shed * shed_cost * weight)
        + sum(burned * captured * weight)
        + sum(burned_starting * captured_start * weight)
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
shifted terms. Where they differ by plant, `sum_back(within=)` reads the width
off the column — [minimum up and down times](pypsa_min_up_down.md).

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
