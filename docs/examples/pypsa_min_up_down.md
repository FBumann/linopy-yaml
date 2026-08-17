# PyPSA minimum up and down times — the rows that make commitment bite

A unit that has started must stay on; one that has stopped must stay off.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.9.0)** — objective **32750.0**, matched to `rtol=1e-09`.

[Unit commitment](pypsa_unit_commitment.md) takes the status and the two
transition variables and stops there. On its own that lets a unit start and stop
in consecutive snapshots, paying the charges and doing as it likes. The windows
are what make commitment a scheduling problem: over any `min_up_time`
consecutive snapshots a unit may have started at most as often as it is now
running, and the mirror holds for stopping.

Each window's length is a property of the **generator**, not of the model — 3, 2
and 1 here — because a single shared length would be satisfied by an operator
that ignored the parameter and used a constant.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA minimum up and down times: a unit that has started must stay on, and one that has stopped must stay off. Each window is a backward-looking sum whose length is a property of the generator, and the three units here carry different lengths. Optimum 32750.0, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ --- `snapshot` --- dispatch periods |
| $\mathcal{G}$ | index $g$ --- `generator` --- generating units, each either committed or off |

#### Parameters

| Symbol | Meaning |
|---|---|
| $p^{\mathrm{nom}}$ | `p_nom` over $\mathcal{G}$ --- installed capacity of a generator |
| $p^{\mathrm{min,pu}}$ | `p_min_pu` over $\mathcal{G}$ --- share of capacity a committed unit must produce at least |
| $\mathit{marginal\_cost}$ | `marginal_cost` over $\mathcal{G}$ --- cost of one unit of output |
| $\mathit{start\_up\_cost}$ | `start_up_cost` over $\mathcal{G}$ --- what bringing a unit up costs, once per start |
| $\mathit{shut\_down\_cost}$ | `shut_down_cost` over $\mathcal{G}$ --- what taking a unit down costs, once per stop |
| $\mathit{min\_up\_time}$ | `min_up_time` over $\mathcal{G}$ --- how many snapshots a unit must stay on once it has started |
| $\mathit{min\_down\_time}$ | `min_down_time` over $\mathcal{G}$ --- how many snapshots a unit must stay off once it has stopped |
| $\mathit{load}$ | `load` over $\mathcal{T}$ --- demand to be met |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{T} \times \mathcal{G}$ --- output of a generator in a snapshot |
| $\mathit{status}$ | `status` over $\mathcal{T} \times \mathcal{G}$ --- is this unit committed in this snapshot? |
| $\mathit{start\_up}$ | `start_up` over $\mathcal{T} \times \mathcal{G}$ --- does this unit come up entering this snapshot? |
| $\mathit{shut\_down}$ | `shut_down` over $\mathcal{T} \times \mathcal{G}$ --- does this unit go down entering this snapshot? |

#### Objective

$$\min \sum_{t \in \mathcal{T},\enspace g \in \mathcal{G}} \left( p_{t,g} \cdot \mathit{marginal\_cost}_{g} + \mathit{start\_up}_{t,g} \cdot \mathit{start\_up\_cost}_{g} + \mathit{shut\_down}_{t,g} \cdot \mathit{shut\_down\_cost}_{g} \right)$$

#### Subject to

**`power_balance`**

$$\sum_{g \in \mathcal{G}} p_{t,g} = \mathit{load}_{t} \qquad \forall\thinspace t \in \mathcal{T}$$

**`commitment_max`**

$$p_{t,g} - p^{\mathrm{nom}}_{g} \cdot \mathit{status}_{t,g} \le 0 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`commitment_min`**

$$p_{t,g} - p^{\mathrm{min,pu}}_{g} \cdot p^{\mathrm{nom}}_{g} \cdot \mathit{status}_{t,g} \ge 0 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`start_up_initial`**

$$\mathit{start\_up}_{t,g} - \mathit{status}_{t,g} \ge 0 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G} \thinspace:\thinspace t = 0$$

**`start_up`**

$$\mathit{start\_up}_{t,g} - \mathit{status}_{t,g} + \mathit{status}_{t - 1,g} \ge 0 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`shut_down`**

$$\mathit{shut\_down}_{t,g} + \mathit{status}_{t,g} - \mathit{status}_{t - 1,g} \ge 0 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`min_up_time`**

$$\sum_{t' \in \mathcal{T} \thinspace:\thinspace 0 \le t - t' < \mathit{min\_up\_time}} \mathit{start\_up}_{t',g} \le \mathit{status}_{t,g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G} \thinspace:\thinspace t > 0$$

**`min_down_time`**

$$\mathit{status}_{t,g} + \sum_{t' \in \mathcal{T} \thinspace:\thinspace 0 \le t - t' < \mathit{min\_down\_time}} \mathit{shut\_down}_{t',g} \le 1 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G} \thinspace:\thinspace t > 0$$

#### Variable domains

**`p`**

$$p_{t,g} \ge 0 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`status`**

$$\mathit{status}_{t,g} \in \{0, 1\} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`start_up`**

$$\mathit{start\_up}_{t,g} \in \{0, 1\} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`shut_down`**

$$\mathit{shut\_down}_{t,g} \in \{0, 1\} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

</details>
<!-- math:end -->

The tabs start from [the instance’s tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    description: >-
      PyPSA minimum up and down times: a unit that has started must stay on, and one
      that has stopped must stay off. Each window is a backward-looking sum whose
      length is a property of the generator, and the three units here carry
      different lengths. Optimum 32750.0, from PyPSA itself.

    dimensions:
      snapshot:
        description: dispatch periods
        dtype: int
      generator:
        description: generating units, each either committed or off
        dtype: str

    parameters:
      p_nom:
        description: installed capacity of a generator
        dims: [generator]
      p_min_pu:
        description: share of capacity a committed unit must produce at least
        dims: [generator]
      marginal_cost:
        description: cost of one unit of output
        dims: [generator]
      start_up_cost:
        description: what bringing a unit up costs, once per start
        dims: [generator]
      shut_down_cost:
        description: what taking a unit down costs, once per stop
        dims: [generator]
      min_up_time:
        description: how many snapshots a unit must stay on once it has started
        dims: [generator]
        dtype: int
      min_down_time:
        description: how many snapshots a unit must stay off once it has stopped
        dims: [generator]
        dtype: int
      load:
        description: demand to be met
        dims: [snapshot]

    variables:
      p:
        description: output of a generator in a snapshot
        foreach: [snapshot, generator]
        bounds:
          lower: 0
      status:
        description: is this unit committed in this snapshot?
        foreach: [snapshot, generator]
        domain: binary
      start_up:
        description: does this unit come up entering this snapshot?
        foreach: [snapshot, generator]
        domain: binary
      shut_down:
        description: does this unit go down entering this snapshot?
        foreach: [snapshot, generator]
        domain: binary

    constraints:
      power_balance:
        foreach: [snapshot]
        expression: sum(p, over=generator) == load

      commitment_max:
        description: a committed unit runs at no more than its capacity, an uncommitted one at zero
        foreach: [snapshot, generator]
        expression: p - p_nom * status <= 0

      commitment_min:
        description: a committed unit runs at no less than its minimum, an uncommitted one at zero
        foreach: [snapshot, generator]
        expression: p - p_min_pu * p_nom * status >= 0

      start_up_initial:
        description: >-
          every unit begins the horizon off, so one that is committed in the first
          snapshot pays for the start
        foreach: [snapshot, generator]
        where: "snapshot == 0"
        expression: start_up - status >= 0

      start_up:
        description: a unit whose status rises entering this snapshot pays for a start
        foreach: [snapshot, generator]
        expression: start_up - status + shift(status, over=snapshot, by=1) >= 0

      shut_down:
        description: >-
          a unit whose status falls entering this snapshot pays for a stop. There is
          no first-snapshot counterpart: a unit that begins off has nothing to stop,
          and the row PyPSA emits there holds for every value of both variables.
        foreach: [snapshot, generator]
        expression: shut_down + status - shift(status, over=snapshot, by=1) >= 0

      min_up_time:
        description: >-
          over the last `min_up_time` snapshots a unit may have started at most as
          often as it is running now — which is what stops it starting and stopping
          again inside its own window
        foreach: [snapshot, generator]
        where: "snapshot > 0"
        expression: sum_back(start_up, over=snapshot, within=min_up_time) <= status

      min_down_time:
        description: >-
          the mirror: having stopped inside the window and running now cannot both
          be true
        foreach: [snapshot, generator]
        where: "snapshot > 0"
        expression: status + sum_back(shut_down, over=snapshot, within=min_down_time) <= 1

    objective:
      sense: minimize
      description: what the fleet costs to run, plus what its starts and stops cost
      expression: p * marginal_cost + start_up * start_up_cost + shut_down * shut_down_cost
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/ports/pypsa_min_up_down.yaml', sources) as solution:
        solution.objective  # 32750.0
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_min_up_down.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the lpspec call binds as ``sources``.

        One bus and no network: a rung that fails to match should implicate one
        feature, and here it is the window length. ``committable`` is what turns the
        status into a variable at all, and the two ``*_time_before`` values are set
        rather than defaulted for the reason in the module docstring.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.add('Bus', 'hub')

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        n.add(
            'Generator',
            generators.index,
            bus='hub',
            committable=True,
            up_time_before=0,
            down_time_before=0,
            p_nom=tables['p_nom'].set_index('generator')['value'],
            p_min_pu=tables['p_min_pu'].set_index('generator')['value'],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
            start_up_cost=tables['start_up_cost'].set_index('generator')['value'],
            shut_down_cost=tables['shut_down_cost'].set_index('generator')['value'],
            min_up_time=tables['min_up_time'].set_index('generator')['value'],
            min_down_time=tables['min_down_time'].set_index('generator')['value'],
        )

        n.add('Load', 'l', bus='hub', p_set=tables['load'].set_index('snapshot')['value'])
        return n
    ```

**The windows bind, and the schedule shows where.** `mid` runs, drops out over
snapshots 4 and 5, and comes back at 6 — and then has to stay on through 7 even
though the load there is met more cheaply without it. Set every window to 1 and
the same instance solves at **31800.0**, with `mid` free to cycle in and out of
every trough. The 950 between the two numbers is the whole point of the rung.

**`sum_back` truncates at the start of the axis, and so does PyPSA.** At snapshot
1 with a window of 3, the sum has two terms rather than three — there is no
snapshot −1 to reach for. Both implementations shorten the window rather than
wrapping or dropping the row, which is why no `edge=` argument appears here.

**The initial conditions are switched off rather than defaulted.**
`up_time_before` defaults to 1 in PyPSA — the unit was already running — which
emits a further block pinning the status on for the remainder of its minimum up
time. That is real behaviour and a *second* feature; this rung is about the
windows, so both `*_time_before` values are set to 0. Every unit therefore
begins the horizon **off**, and the first snapshot's transition rows are the
mirror of the ones [unit commitment](pypsa_unit_commitment.md) ports: a unit
committed in the first snapshot pays for a start, and nothing is charged for a
stop, where a unit that began the horizon running pays for no start and is
charged if it goes down.

## What it exercises

`sum_back(x, over=dim, within=p)` with `p` an integer parameter — the
per-entity window length, against a variable, inside a MILP. The `dtype: int`
declaration is required and load-time validation says so by name: *a width
counts positions rather than measuring a distance.*
