# storage

Dispatch plus a battery, and the only construct in the language whose cost is not obviously linear.

> **✔ Agrees with hand-written linopy 0.9.0** — objective **5650**, matched to `rtol=1e-09`.

## The problem

State of charge links each snapshot to the one before it, cyclically:

$$\mathrm{soc}_s = \mathrm{soc}_{s-1} + 0.9\,\mathrm{charge}_s - \mathrm{discharge}_s$$

with $s-1$ wrapping at the horizon, so the battery ends where it started. The
charging efficiency is written into the model as a literal `0.9` rather than
declared as a parameter — which is why it appears as a number here and not as
an $\eta$.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

Dispatch plus a battery whose state of charge is closed into a cycle: the horizon ends where it began.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{S}$ | index $s$ --- `snapshot` --- dispatch periods, cyclic at the horizon |
| $\mathcal{G}$ | index $g$ --- `generator` --- generating units |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\bar p$ | `p_max` over $\mathcal{G}$ --- installed capacity |
| $c$ | `cost` over $\mathcal{G}$ --- marginal cost |
| $\ell$ | `load` over $\mathcal{S}$ --- demand to be met |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{S} \times \mathcal{G}$ --- output of a generator in a snapshot |
| $\mathrm{charge}$ | `charge` over $\mathcal{S}$ --- energy into the store |
| $\mathrm{discharge}$ | `discharge` over $\mathcal{S}$ --- energy out of the store |
| $\mathrm{soc}$ | `soc` over $\mathcal{S}$ --- state of charge carried into the next snapshot |

$t \ominus k$ denotes cyclic translation: index $t-k$ taken modulo the size of the dimension (`roll`). Plain $t-k$ (`shift`) has no wraparound --- terms translated past the edge are simply absent.

#### Objective

$$\min \sum_{s \in \mathcal{S},\enspace g \in \mathcal{G}} p_{s,g} \cdot c_{g}$$

#### Subject to

**`power_balance`**

$$\sum_{g \in \mathcal{G}} p_{s,g} + \mathrm{discharge}_{s} - \mathrm{charge}_{s} = \ell_{s} \qquad \forall\thinspace s \in \mathcal{S}$$

**`soc_balance`**

$$\mathrm{soc}_{s} = \mathrm{soc}_{s \ominus 1} + \mathrm{charge}_{s} \cdot 0.9 - \mathrm{discharge}_{s} \qquad \forall\thinspace s \in \mathcal{S}$$

#### Variable domains

**`p`**

$$0 \le p_{s,g} \le \bar p_{g} \qquad \forall\thinspace s \in \mathcal{S},\enspace g \in \mathcal{G}$$

**`charge`**

$$0 \le \mathrm{charge}_{s} \le 30 \qquad \forall\thinspace s \in \mathcal{S}$$

**`discharge`**

$$0 \le \mathrm{discharge}_{s} \le 30 \qquad \forall\thinspace s \in \mathcal{S}$$

**`soc`**

$$0 \le \mathrm{soc}_{s} \le 100 \qquad \forall\thinspace s \in \mathcal{S}$$

</details>
<!-- math:end -->

The tabs start from [the instance’s tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    description: >-
      Dispatch plus a battery whose state of charge is closed into a cycle: the
      horizon ends where it began.

    dimensions:
      snapshot:
        description: dispatch periods, cyclic at the horizon
        dtype: int
      generator:
        description: generating units
        dtype: str

    parameters:
      p_max:
        description: installed capacity
        dims: [generator]
      cost:
        description: marginal cost
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
          upper: p_max
      charge:
        description: energy into the store
        foreach: [snapshot]
        bounds:
          lower: 0
          upper: 30
      discharge:
        description: energy out of the store
        foreach: [snapshot]
        bounds:
          lower: 0
          upper: 30
      soc:
        description: state of charge carried into the next snapshot
        foreach: [snapshot]
        bounds:
          lower: 0
          upper: 100

    constraints:
      power_balance:
        description: generation plus what the store gives back covers the load, net of charging
        foreach: [snapshot]
        expression: sum(p, over=generator) + discharge - charge == load
      soc_balance:
        description: >-
          the level carried out of a snapshot is the one carried in plus what was
          stored, minus what was taken — and it wraps at the horizon, so the first
          snapshot inherits from the last
        foreach: [snapshot]
        expression: soc == shift(soc, over=snapshot, by=1, edge='wrap') + charge * 0.9 - discharge

    objective:
      sense: minimize
      description: total cost of generation; storing and releasing energy is free here
      expression: p * cost
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/storage.yaml', sources) as solution:
        solution.objective  # 5650.0
        solution.dual('power_balance')
    ```

=== "linopy"

    The model-building half of `examples/ports/references/linopy/storage.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
        """The instance's tables as a linopy model, row for row.

        ``tables`` is the same mapping the lpspec call binds as ``sources``.
        """
        p_max: pd.Series = tables['p_max'].set_index('generator')['value']
        cost: pd.Series = tables['cost'].set_index('generator')['value']
        load: pd.Series = tables['load'].set_index('snapshot')['value']
        snapshots = load.index

        m = linopy.Model()
        p = m.add_variables(lower=0, upper=p_max, coords=[snapshots, p_max.index], name='p')
        charge = m.add_variables(lower=0, upper=30, coords=[snapshots], name='charge')
        discharge = m.add_variables(lower=0, upper=30, coords=[snapshots], name='discharge')
        soc = m.add_variables(lower=0, upper=100, coords=[snapshots], name='soc')

        m.add_constraints(p.sum('generator') + discharge - charge == load, name='power_balance')
        m.add_constraints(soc == soc.roll(snapshot=1) + 0.9 * charge - discharge, name='soc_balance')
        m.add_objective((p * cost).sum())
        return m
    ```

## What it exercises

`shift(soc, over=snapshot, by=1, edge='wrap')` is the whole of it. One term reaches one position
back along `snapshot`, and `edge='wrap'` wraps — the first snapshot reads the
last, which is what makes the storage cyclic without a boundary condition
written out by hand. Omitting `edge=` is the same node without the wrap, where
positions translated past the edge simply contribute nothing.

It is also the one plan shape whose cost is not obviously linear in the model
size, which is why it is named in *Not measured yet* in
[the benchmarks](../about/benchmarks.md).

---

[`examples/storage.yaml`](https://github.com/fluxopt/lpspec/blob/main/examples/storage.yaml) · back to [all models](index.md)
