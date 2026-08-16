# PyPSA unit commitment

Which generators are *on*, not just how much they produce — a binary per generator per snapshot, with start-up and shut-down charges.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.9.0)** — objective **24900**, matched to `rtol=1e-09`.

**The corpus's MILP entry.** Every other verified model is a pure continuous
LP; this one carries integrality, which is what the gallery's construct matrix
had no verified example of. One bus and no network, deliberately: a model that
fails to match should implicate one feature, and here that feature is
commitment.

`min_up_time` and `min_down_time` are left at 0. They need a rolling window sum
over a horizon, which is a different question from whether the language can say
commitment at all — see [the ledger](index.md#ledger--what-a-port-could-not-say).

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA unit commitment: which generators are on, not just how much they produce — a binary status per generator per snapshot, with start-up and shut-down charges. Optimum 24900.0, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ --- `snapshot` --- dispatch periods |
| $\mathcal{G}$ | index $g$ --- `generator` --- generating units, each either committed or off |

#### Parameters

| Symbol | Meaning |
|---|---|
| $p^{\mathrm{nom}}$ | `p_nom` over $\mathcal{G}$ --- installed capacity of a generator |
| $\mathit{marginal\_cost}$ | `marginal_cost` over $\mathcal{G}$ --- cost of one unit of output |
| $p^{\mathrm{min,pu}}$ | `p_min_pu` over $\mathcal{G}$ --- share of capacity a committed unit must produce at least |
| $\mathit{start\_up\_cost}$ | `start_up_cost` over $\mathcal{G}$ --- what bringing a unit up costs, once per start |
| $\mathit{shut\_down\_cost}$ | `shut_down_cost` over $\mathcal{G}$ --- what taking a unit down costs, once per stop |
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

$$\mathit{start\_up}_{t,g} - \mathit{status}_{t,g} \ge -1 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G} \thinspace:\thinspace t = \mathrm{index}(\mathcal{T}, 0)$$

**`start_up`**

$$\mathit{start\_up}_{t,g} - \mathit{status}_{t,g} + \mathit{status}_{t - 1,g} \ge 0 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`shut_down_initial`**

$$\mathit{shut\_down}_{t,g} + \mathit{status}_{t,g} \ge 1 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G} \thinspace:\thinspace t = \mathrm{index}(\mathcal{T}, 0)$$

**`shut_down`**

$$\mathit{shut\_down}_{t,g} + \mathit{status}_{t,g} - \mathit{status}_{t - 1,g} \ge 0 \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

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
      PyPSA unit commitment: which generators are on, not just how much they
      produce — a binary status per generator per snapshot, with start-up and
      shut-down charges. Optimum 24900.0, from PyPSA itself.

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
      marginal_cost:
        description: cost of one unit of output
        dims: [generator]
      p_min_pu:
        description: share of capacity a committed unit must produce at least
        dims: [generator]
      start_up_cost:
        description: what bringing a unit up costs, once per start
        dims: [generator]
      shut_down_cost:
        description: what taking a unit down costs, once per stop
        dims: [generator]
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
        description: >-
          a committed unit runs at no more than its capacity and an uncommitted one
          is pinned to zero — capacity times status is a parameter against a
          variable, so the product stays degree 1
        foreach: [snapshot, generator]
        expression: p - p_nom * status <= 0

      commitment_min:
        description: a committed unit runs at no less than its minimum, an uncommitted one at zero
        foreach: [snapshot, generator]
        expression: p - p_min_pu * p_nom * status >= 0

      start_up_initial:
        description: >-
          the first snapshot has no predecessor, and PyPSA's default is that the
          unit was already up before the horizon — so the start-up row is slackened
          here and never binds
        foreach: [snapshot, generator]
        where: "snapshot == index(snapshot, 0)"
        expression: start_up - status >= -1

      start_up:
        description: >-
          a unit whose status rises entering this snapshot pays for a start. The
          start-up and shut-down variables are implied by these transitions, but
          PyPSA declares them binary rather than leaving it to the status, and the
          port matches that.
        foreach: [snapshot, generator]
        expression: start_up - status + shift(status, over=snapshot, by=1) >= 0

      shut_down_initial:
        description: >-
          the mirror of the start-up row, and not slackened: a unit that begins the
          horizon off is charged for the shut-down, which is PyPSA's asymmetry and
          worth 50 on this instance
        foreach: [snapshot, generator]
        where: "snapshot == index(snapshot, 0)"
        expression: shut_down + status >= 1

      shut_down:
        description: a unit whose status falls entering this snapshot pays for a stop
        foreach: [snapshot, generator]
        expression: shut_down + status - shift(status, over=snapshot, by=1) >= 0

    objective:
      sense: minimize
      description: what the fleet costs to run, plus what its starts and stops cost
      expression: p * marginal_cost + start_up * start_up_cost + shut_down * shut_down_cost
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/ports/pypsa_unit_commitment.yaml', sources) as solution:
        solution.objective  # 24900.0
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_unit_commitment.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the lpspec call binds as ``sources``.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.add('Bus', 'bus')

        generators: pd.DataFrame = tables['generator'].set_index('generator')

        n.add(
            'Generator',
            generators.index,
            bus='bus',
            committable=True,
            p_nom=tables['p_nom'].set_index('generator')['value'],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
            p_min_pu=tables['p_min_pu'].set_index('generator')['value'],
            start_up_cost=tables['start_up_cost'].set_index('generator')['value'],
            shut_down_cost=tables['shut_down_cost'].set_index('generator')['value'],
        )

        load: pd.Series = tables['load'].set_index('snapshot')['value']
        n.add('Load', 'load', bus='bus', p_set=load)
        return n
    ```

**The first snapshot is not like the others.** PyPSA's default is that a unit
was already up before the horizon began, so the start-up row is slackened to
`>= -1` there and never binds, while the shut-down row still charges a unit
that begins the horizon *off*. `peak` does, so the instance pays a shut-down it
never visibly performs. That asymmetry is PyPSA's, it is worth 50 here, and
reproducing it is most of what makes this a fidelity test rather than a
plausible-looking rewrite.

Two `where` clauses on one constraint block is how the language says "this row
differs at the boundary" — the same shape [storage](storage.md) uses for its
initial state of charge.

## What it costs

| | |
|---|---|
| energy | `30 × 520 + 90 × 100` = 24600 |
| start-ups | `peak` at snapshot 1 = 200 |
| shut-downs | `peak` at snapshot 0 (begins off) and at 3 = 100 |
| **total** | **24900** |

`base` runs throughout; `peak` covers the two peak snapshots. lpspec and PyPSA
agree on the schedule as well as the cost.

## What it exercises

`binary` variables and the integrality path through to HiGHS, `shift` across a
boundary condition, and a three-term objective mixing an energy cost with two
transition charges.
