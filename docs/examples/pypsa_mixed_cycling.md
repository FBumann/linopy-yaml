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
data. Here that is three blocks under complementary masks, and the masks are the
flag itself:

| Block | Where | What carries into the first snapshot |
|---|---|---|
| `energy_balance_cyclic` | `cyclic` | the unit's own last snapshot (`edge='wrap'`) |
| `energy_balance_carry` | `NOT cyclic` | nothing — the vacated position is absent, so no row is built there |
| `energy_balance_seed` | `NOT cyclic AND position(snapshot) == 0` | `soc_initial` |

`NOT` is a real complement over a boolean column, so every unit falls in exactly
one regime — including one whose flag row is missing, which reads as not cyclic.
The row `energy_balance_carry` does not build at each seeded unit's first
snapshot is reported: `diagnostics().omissions` gives 1, and
`energy_balance_seed` writes it instead.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

PyPSA's `cyclic_state_of_charge` is a column of the StorageUnit frame, so one network runs both regimes at once: a unit that must end each horizon where it began, beside one handed a level it may simply spend. The two are one rule under complementary masks, and the seeded unit's boundary row is what a cyclic one does not have. Optimum 4800.0, from PyPSA itself.

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
| $\mathit{cyclic}$ | `cyclic` over $\mathcal{S}$ — whether a unit closes its own horizon. PyPSA's flag, and the one column that decides which of the two balance rules a unit obeys |
| $p^{\mathrm{nom}}$ | `p_nom` over $\mathcal{G}$ — installed capacity of a generator |
| $\mathit{marginal\_cost}$ | `marginal_cost` over $\mathcal{G}$ — cost of one unit of generation |
| $\mathit{storage}^{\mathrm{p,nom}}$ | `storage_p_nom` over $\mathcal{S}$ — how fast a storage unit may charge or discharge |
| $\mathit{soc}^{\mathrm{max}}$ | `soc_max` over $\mathcal{S}$ — most energy a storage unit may hold |
| $\mathit{soc}^{\mathrm{initial}}$ | `soc_initial` over $\mathcal{S}$ — the level a seeded unit begins with. A cyclic unit has one in the data and never reads it — its own last snapshot is its opening level |
| $\mathit{load}$ | `load` over $\mathcal{T} \times \mathcal{B}$ — demand to be met at a bus |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{T} \times \mathcal{G}$ — output of a generator in a snapshot |
| $p^{\mathrm{dispatch}}$ | `p_dispatch` over $\mathcal{T} \times \mathcal{S}$ — power a storage unit puts onto its bus |
| $p^{\mathrm{store}}$ | `p_store` over $\mathcal{T} \times \mathcal{S}$ — power a storage unit takes off its bus |
| $\mathit{soc}$ | `soc` over $\mathcal{T} \times \mathcal{S}$ — energy in the store at the end of a snapshot |

$t \ominus k$ denotes cyclic translation: index $t-k$ taken modulo the size of the dimension (`roll`). Plain $t-k$ (`shift`) has no wraparound — terms translated past the edge are simply absent.

#### Objective

$$\min \sum_{t \in \mathcal{T},\enspace g \in \mathcal{G}} p_{t,g} \cdot \mathit{marginal\_cost}_{g}$$

#### Subject to

**`nodal_balance`**

$$\sum_{g \in \mathcal{G} \thinspace:\thinspace \mathrm{gen\_bus}(g) = b} p_{t,g} + \sum_{s \in \mathcal{S} \thinspace:\thinspace \mathrm{storage\_bus}(s) = b} p^{\mathrm{dispatch}}_{t,s} - \left( \sum_{s \in \mathcal{S} \thinspace:\thinspace \mathrm{storage\_bus}(s) = b} p^{\mathrm{store}}_{t,s} \right) = \mathit{load}_{t,b} \qquad \forall\thinspace t \in \mathcal{T},\enspace b \in \mathcal{B}$$

**`energy_balance_cyclic`**

$$\mathit{soc}_{t,s} = \mathit{soc}_{t \ominus 1,s} + p^{\mathrm{store}}_{t,s} - p^{\mathrm{dispatch}}_{t,s} \qquad \forall\thinspace t \in \mathcal{T},\enspace s \in \mathcal{S} \thinspace:\thinspace \mathit{cyclic}_{s}$$

**`energy_balance_carry`**

$$\mathit{soc}_{t,s} = \mathit{soc}_{t - 1,s} + p^{\mathrm{store}}_{t,s} - p^{\mathrm{dispatch}}_{t,s} \qquad \forall\thinspace t \in \mathcal{T},\enspace s \in \mathcal{S} \thinspace:\thinspace \neg \mathit{cyclic}_{s}$$

**`energy_balance_seed`**

$$\mathit{soc}_{t,s} = \mathit{soc}^{\mathrm{initial}}_{s} + p^{\mathrm{store}}_{t,s} - p^{\mathrm{dispatch}}_{t,s} \qquad \forall\thinspace t \in \mathcal{T},\enspace s \in \mathcal{S} \thinspace:\thinspace \neg \mathit{cyclic}_{s} \wedge \mathrm{pos}(t) = 0$$

#### Variable domains

**`p`**

$$0 \le p_{t,g} \le p^{\mathrm{nom}}_{g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

**`p_dispatch`**

$$0 \le p^{\mathrm{dispatch}}_{t,s} \le \mathit{storage}^{\mathrm{p,nom}}_{s} \qquad \forall\thinspace t \in \mathcal{T},\enspace s \in \mathcal{S}$$

**`p_store`**

$$0 \le p^{\mathrm{store}}_{t,s} \le \mathit{storage}^{\mathrm{p,nom}}_{s} \qquad \forall\thinspace t \in \mathcal{T},\enspace s \in \mathcal{S}$$

**`soc`**

$$0 \le \mathit{soc}_{t,s} \le \mathit{soc}^{\mathrm{max}}_{s} \qquad \forall\thinspace t \in \mathcal{T},\enspace s \in \mathcal{S}$$

</details>
<!-- math:end -->

The tabs start from [the instance's tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    description: >-
      PyPSA's `cyclic_state_of_charge` is a column of the StorageUnit frame, so one
      network runs both regimes at once: a unit that must end each horizon where it
      began, beside one handed a level it may simply spend. The two are one rule
      under complementary masks, and the seeded unit's boundary row is what a cyclic
      one does not have. Optimum 4800.0, from PyPSA itself.

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

    constraints:
      nodal_balance:
        description: what is generated at a bus, plus what the stores give back, meets the load there
        foreach: [snapshot, bus]
        expression: >-
          sum(p, by=gen_bus)
          + sum(p_dispatch, by=storage_bus)
          - sum(p_store, by=storage_bus)
          == load

      energy_balance_cyclic:
        description: >-
          a cyclic unit's level wraps at the horizon, so its first snapshot inherits
          from its last and it ends every horizon where it began
        foreach: [snapshot, storage]
        where: "cyclic"
        expression: soc == shift(soc, over=snapshot, offset=1, edge='wrap') + p_store - p_dispatch

      energy_balance_carry:
        description: >-
          a seeded unit carries from the snapshot before, and has no predecessor at
          the first — the vacated position is absent, so that row is not built here
        foreach: [snapshot, storage]
        where: "NOT cyclic"
        expression: soc == shift(soc, over=snapshot, offset=1) + p_store - p_dispatch

      energy_balance_seed:
        description: >-
          and the row the vacated position left is written here instead, from the
          level the unit was handed
        foreach: [snapshot, storage]
        where: "NOT cyclic AND position(snapshot) == 0"
        expression: soc == soc_initial + p_store - p_dispatch

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
choosing between two boundary regimes* until this one. The finding is that it
needs no language feature: three blocks, complementary masks, and the dropped row
visible in `omissions`.
