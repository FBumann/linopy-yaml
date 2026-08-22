# pypsa_mixed_cycling

PyPSA's `cyclic_state_of_charge` is a per-unit flag, so one network runs both
regimes at once: a unit that must end each horizon where it began, beside one
handed a level it may simply spend.

## The problem

Rung 3 seeds every storage from `state_of_charge_initial`; [rung 4](pypsa_cyclic_storage.md)
closes every storage on itself. A real PyPSA model has both in the same
`StorageUnit` frame, because the flag is a column:

```python
n.add('StorageUnit', ..., cyclic_state_of_charge=[True, False])
```

The two regimes are **one rule and a different predecessor**. A cyclic unit's
first snapshot carries from its last; a seeded unit's carries from a level in the
data. So the thing that varies is not the rule, it is the *opening level* — one
quantity with three answers, which is what a named expression's `cases:` says:

| Case | When | What the unit opens with |
|---|---|---|
| `wrapped` | `cyclic` | the unit's own last snapshot (`edge='wrap'`) |
| `seeded` | `not cyclic and position(snapshot) == 0` | `soc_initial` |
| `carried` | `not cyclic and position(snapshot) > 0` | the snapshot before |

The balance is then written **once**, against `opening_level`. The cases have to
partition `[snapshot, storage]` and are checked to at load, with no data read:
`not` is a real complement over a boolean column, and the two position clauses
split the rest, so every coordinate falls in exactly one — including a unit whose
flag row is missing, which reads as not cyclic.

Written as three constraints instead, the seeded unit's first row is the one
`shift` leaves absent, so a fourth block has to write it back and
`diagnostics().omissions` reports the dropped row. Under `cases:` there is no
dropped row to report: the arm that vacates the first position is the arm that
does not claim it.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA's `cyclic_state_of_charge` is a column of the StorageUnit frame, so one network runs both regimes at once: a unit that must end each horizon where it began, beside one handed a level it may simply spend. What differs between them is the level a unit opens a snapshot with, so that is the cased quantity and the balance is written once. Optimum 4800.0, from PyPSA itself.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ — `snapshot` — dispatch periods |
| $\mathcal{B}$ | index $b$ — `bus` — network nodes |
| $\mathcal{G}$ | index $g$ — `generator` with $\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}$ — generating units, each sitting on one bus |
| $\mathcal{S}$ | index $s$ — `storage` with $\mathrm{storage\_bus}: \mathcal{S} \to \mathcal{B}$ — storage units, each sitting on one bus |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\mathrm{cyclic}$ | `cyclic` over $\mathcal{S}$ — whether a unit closes its own horizon. PyPSA's flag, and the one column that decides which of the two balance rules a unit obeys |
| $\mathrm{p}^{\mathrm{nom}}$ | `p_nom` over $\mathcal{G}$ — installed capacity of a generator |
| $\mathrm{marginal\_cost}$ | `marginal_cost` over $\mathcal{G}$ — cost of one unit of generation |
| $\mathrm{storage\_p\_nom}$ | `storage_p_nom` over $\mathcal{S}$ — how fast a storage unit may charge or discharge |
| $\mathrm{soc}^{\mathrm{max}}$ | `soc_max` over $\mathcal{S}$ — most energy a storage unit may hold |
| $\mathrm{soc}^{\mathrm{initial}}$ | `soc_initial` over $\mathcal{S}$ — the level a seeded unit begins with. A cyclic unit has one in the data and never reads it — its own last snapshot is its opening level |
| $\mathrm{load}$ | `load` over $\mathcal{T} \times \mathcal{B}$ — demand to be met at a bus |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{T} \times \mathcal{G}$ — output of a generator in a snapshot |
| $p^{\mathrm{dispatch}}$ | `p_dispatch` over $\mathcal{T} \times \mathcal{S}$ — power a storage unit puts onto its bus |
| $p^{\mathrm{store}}$ | `p_store` over $\mathcal{T} \times \mathcal{S}$ — power a storage unit takes off its bus |
| $\mathit{soc}$ | `soc` over $\mathcal{T} \times \mathcal{S}$ — energy in the store at the end of a snapshot |

Upright is what the model is given — a parameter such as $\mathrm{cyclic}$, a coordinate map, a label — and italic is what the solver chooses, such as $p$. An index is italic too, being what a quantifier chooses, and a set is script.

$t \ominus k$ denotes cyclic translation: index $t-k$ taken modulo the size of the dimension (`roll`). Plain $t-k$ (`shift`) has no wraparound — terms translated past the edge are simply absent.

$\mathrm{pos}(t)$ denotes where index $t$ sits along its dimension's own order — the order `shift` walks, not the order labels sort in — counted from $0$. The index itself stays the coordinate, so $t$ compares against labels and $\mathrm{pos}(t)$ against positions.

#### Objective

$$\min \sum_{t \in \mathcal{T},\enspace g \in \mathcal{G}} p_{t,g} \cdot \mathrm{marginal\_cost}_{g}$$

#### Subject to

**`nodal_balance`**

$$\sum_{g \in \mathcal{G} \thinspace:\thinspace \mathrm{gen\_bus}(g) = b} p_{t,g} + \sum_{s \in \mathcal{S} \thinspace:\thinspace \mathrm{storage\_bus}(s) = b} p^{\mathrm{dispatch}}_{t,s} - \left( \sum_{s \in \mathcal{S} \thinspace:\thinspace \mathrm{storage\_bus}(s) = b} p^{\mathrm{store}}_{t,s} \right) = \mathrm{load}_{t,b} \qquad \forall\thinspace t \in \mathcal{T},\enspace b \in \mathcal{B}$$

**`energy_balance`**

$$\mathit{soc}_{t,s} = \mathit{opening\_level}_{t,s} + p^{\mathrm{store}}_{t,s} - p^{\mathrm{dispatch}}_{t,s} \qquad \forall\thinspace t \in \mathcal{T},\enspace s \in \mathcal{S}$$

#### Definitions

**`opening_level`**

$$\mathit{opening\_level}_{t,s} = \begin{cases} \mathit{soc}_{t \ominus 1,s} & \text{if } \mathrm{cyclic}_{s} \cr \mathrm{soc}^{\mathrm{initial}}_{s} & \text{if } \neg \mathrm{cyclic}_{s} \wedge \mathrm{pos}(t) = 0 \cr \mathit{soc}_{t - 1,s} & \text{if } \neg \mathrm{cyclic}_{s} \wedge \mathrm{pos}(t) > 0 \end{cases} \qquad \forall\thinspace t \in \mathcal{T},\enspace s \in \mathcal{S}$$

#### Variable domains

**`p`**

$$0 \le p_{t,g} \le \mathrm{p}^{\mathrm{nom}}_{g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`p_dispatch`**

$$0 \le p^{\mathrm{dispatch}}_{t,s} \le \mathrm{storage\_p\_nom}_{s} \qquad \forall\thinspace t \in \mathcal{T},\enspace s \in \mathcal{S}$$

**`p_store`**

$$0 \le p^{\mathrm{store}}_{t,s} \le \mathrm{storage\_p\_nom}_{s} \qquad \forall\thinspace t \in \mathcal{T},\enspace s \in \mathcal{S}$$

**`soc`**

$$0 \le \mathit{soc}_{t,s} \le \mathrm{soc}^{\mathrm{max}}_{s} \qquad \forall\thinspace t \in \mathcal{T},\enspace s \in \mathcal{S}$$

</details>
<!-- math:end -->

The tabs start from [the instance's tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    description: >-
      PyPSA's `cyclic_state_of_charge` is a column of the StorageUnit frame, so one
      network runs both regimes at once: a unit that must end each horizon where it
      began, beside one handed a level it may simply spend. What differs between them
      is the level a unit opens a snapshot with, so that is the cased quantity and
      the balance is written once. Optimum 4800.0, from PyPSA itself.

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
      storage:
        description: storage units, each sitting on one bus
        dtype: str

    lookups:
      gen_bus:
        description: the bus a generator sits on
        over: generator
        into: bus
      storage_bus:
        description: the bus a storage unit sits on
        over: storage
        into: bus

    parameters:
      cyclic:
        description: >-
          whether a unit closes its own horizon. PyPSA's flag, and the one column
          that decides which of the two balance rules a unit obeys
        dims: [storage]
        dtype: bool
      p_nom:
        description: installed capacity of a generator
        dims: [generator]
      marginal_cost:
        description: cost of one unit of generation
        dims: [generator]
      storage_p_nom:
        description: how fast a storage unit may charge or discharge
        dims: [storage]
      soc_max:
        description: most energy a storage unit may hold
        dims: [storage]
      soc_initial:
        description: >-
          the level a seeded unit begins with. A cyclic unit has one in the data and
          never reads it — its own last snapshot is its opening level
        dims: [storage]
      load:
        description: demand to be met at a bus
        dims: [snapshot, bus]

    variables:
      p:
        description: output of a generator in a snapshot
        foreach: [snapshot, generator]
        bounds:
          lower: 0
          upper: p_nom
      p_dispatch:
        description: power a storage unit puts onto its bus
        foreach: [snapshot, storage]
        bounds:
          lower: 0
          upper: storage_p_nom
      p_store:
        description: power a storage unit takes off its bus
        foreach: [snapshot, storage]
        bounds:
          lower: 0
          upper: storage_p_nom
      soc:
        description: energy in the store at the end of a snapshot
        foreach: [snapshot, storage]
        bounds:
          lower: 0
          upper: soc_max

    expressions:
      opening_level:
        description: >-
          the level a unit carries into a snapshot. The two regimes and the seeded
          unit's boundary are three answers to that one question, so they are three
          cases of one quantity rather than three copies of one inequality
        foreach: [snapshot, storage]
        cases:
          wrapped:
            when: "cyclic"
            expression: shift(soc, over=snapshot, offset=1, edge='wrap')
            description: a cyclic unit's first snapshot inherits from its last, so it ends where it began
          seeded:
            when: "not cyclic and position(snapshot) == 0"
            expression: soc_initial
            description: a seeded unit's first snapshot reads the level it was handed
          carried:
            when: "not cyclic and position(snapshot) > 0"
            expression: shift(soc, over=snapshot, offset=1)
            description: and every later one carries from the snapshot before

    constraints:
      nodal_balance:
        description: what is generated at a bus, plus what the stores give back, meets the load there
        foreach: [snapshot, bus]
        expression: >-
          sum(p, by=gen_bus)
          + sum(p_dispatch, by=storage_bus)
          - sum(p_store, by=storage_bus)
          == load

      energy_balance:
        description: >-
          one rule for both regimes: what a unit holds is what it opened with, plus
          what it took off its bus, less what it gave back
        foreach: [snapshot, storage]
        expression: soc == opening_level + p_store - p_dispatch

    objective:
      sense: minimize
      description: total cost of generation; storage is free to operate here
      expression: sum(p * marginal_cost)
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/ports/pypsa_mixed_cycling.yaml', sources) as solution:
        solution.objective  # 4800.0
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_mixed_cycling.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network, column for column.

        ``tables`` is the same mapping the lpspec call binds as ``sources``.

        ``max_hours`` is the ratio PyPSA stores; the port carries the product it
        implies (``soc_max``), because a bound there takes a name, not arithmetic.
        """
        n = pypsa.Network()
        n.set_snapshots(tables['snapshot']['snapshot'])
        n.add('Bus', tables['bus']['bus'])

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        storages: pd.DataFrame = tables['storage'].set_index('storage')

        n.add(
            'Generator',
            generators.index,
            bus=generators['gen_bus'],
            p_nom=tables['p_nom'].set_index('generator')['value'],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
        )
        p_nom: pd.Series = tables['storage_p_nom'].set_index('storage')['value']
        n.add(
            'StorageUnit',
            storages.index,
            bus=storages['storage_bus'],
            p_nom=p_nom,
            max_hours=tables['soc_max'].set_index('storage')['value'] / p_nom,
            cyclic_state_of_charge=tables['cyclic'].set_index('storage')['value'],
            state_of_charge_initial=tables['soc_initial'].set_index('storage')['value'],
        )

        load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
        for bus in tables['bus']['bus']:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
        return n
    ```

## Both flags bind

Neither regime is decoration on this instance — flipping either changes the
answer, in opposite directions:

| Instance | Objective |
|---|---|
| as shipped — `ring` cyclic, `seasonal` seeded | **4800.0** |
| `ring` flipped to seeded | 4400.0 |
| `seasonal` flipped to cyclic | 7200.0 |

`ring` carries a `soc_initial` of 20 that it never reads, which is the point of
the first row: a cyclic unit ignores the level in the data, and flipping its flag
hands it 400 of free energy. `seasonal`'s 30 is worth 2400, because a cyclic unit
must give back everything it spends.

## What the answer looks like

```text
snapshot  price   ring   seasonal
0         20      15     30      ← seasonal opens at its given level; ring opens where it will close
1         80      0      15      ← both discharge into the peak
2         20      10     15
3         20      0      0       ← ring is back where it started; seasonal need not be
```

The price is the dual of `nodal_balance`: 20 where the base plant is marginal
and 80 at the snapshot where the peaker runs, which is what makes moving energy
worth anything at all.

## What this rung is for

It is the shape neither storage rung has. Rung 3 and rung 4 differ by one deleted
`where`, and each is uniform — so nothing in the corpus exercised *a data column
choosing between two boundary regimes* until this one.

It was first written as three blocks under complementary masks, which worked and
needed no language feature. What `cases:` changes is where the variation sits:
the regimes are three answers to *what does this unit open with*, and the model
now says that rather than repeating one inequality three times under three
masks. The inequality is written once, and the partition is proved at load
instead of being a property of three `where` strings a reader has to check
against each other.
