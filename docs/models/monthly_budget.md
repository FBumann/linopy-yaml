# monthly_budget

A cap on what each technology may generate per calendar month — an aggregate
over a *coarser grouping of time*, written with the same primitive that places
a generator on a bus.

## The problem

$$\sum_{t \thinspace:\thinspace \mathrm{month}(t) = m} p_{t,g} \quad\le\quad \bar E_{m,g}$$

$\mathrm{month}$ is a **coordinate the snapshot dimension declares**, not a
calendar the language understands. Its values arrive as a column in the
snapshot index, so the same model expresses weeks, seasons, fiscal quarters,
peak/off-peak blocks or representative days by changing that one column and
nothing else.

Compare [transport](transport.md): there, $\mathrm{bus}$ is a coordinate on
`generator` and the sum is over generators at a bus. Here $\mathrm{month}$ is a
coordinate on `snapshot` and the sum is over snapshots in a month. **It is the
same construct** — `group_sum` — and time is not a special axis.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | index $t$ --- `snapshot` with $\mathrm{month}: \mathcal{T} \to \mathcal{M}$ --- dispatch periods, each falling in one month |
| $\mathcal{M}$ | index $m$ --- `month` --- the grouping the budget is stated over |
| $\mathcal{G}$ | index $g$ --- `generator` --- generating units |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\bar p$ | `p_max` over $\mathcal{G}$ |
| $c$ | `cost` over $\mathcal{G}$ |
| $\ell$ | `load` over $\mathcal{T}$ |
| $\bar E$ | `monthly_cap` over $\mathcal{M} \times \mathcal{G}$ |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{T} \times \mathcal{G}$ |

#### Objective

**`total_cost`**

$$\min \sum_{t \in \mathcal{T}} \sum_{g \in \mathcal{G}} p_{t,g} \cdot c_{g}$$

#### Subject to

**`balance`**

$$\sum_{g \in \mathcal{G}} p_{t,g} = \ell_{t} \qquad \forall\thinspace t \in \mathcal{T}$$

**`monthly_budget`**

$$\sum_{t \in \mathcal{T} \thinspace:\thinspace \mathrm{month}(t) = m} p_{t,g} \le \bar E_{m,g} \qquad \forall\thinspace m \in \mathcal{M},\enspace g \in \mathcal{G}$$

#### Variable domains

**`p`**

$$0 \le p_{t,g} \le \bar p_{g} \qquad \forall\thinspace t \in \mathcal{T},\enspace g \in \mathcal{G}$$

</details>
<!-- math:end -->

```yaml
dimensions:
  snapshot:
    dtype: datetime
    coords: [month]  # every snapshot falls in a month, exactly as a generator sits on a bus
  month:
    dtype: str
  generator:
    dtype: str

parameters:
  p_max:
    dims: [generator]
  cost:
    dims: [generator]
  load:
    dims: [snapshot]
  # the budget the group sum is checked against, one per month and technology
  monthly_cap:
    dims: [month, generator]

variables:
  p:
    foreach: [snapshot, generator]
    bounds:
      lower: 0
      upper: p_max

constraints:
  balance:
    foreach: [snapshot]
    expression: sum(p, over=generator) == load
  # The per-month total, and the whole point of this model: `month` is a
  # coordinate the snapshot dimension declares, so the grouping is a column in
  # the snapshot index rather than anything the language knows about calendars.
  monthly_budget:
    foreach: [month, generator]
    expression: group_sum(p, over=snapshot, by=month) <= monthly_cap

objectives:
  total_cost:
    sense: minimize
    expression: sum(p * cost, over=generator)
```

## The grouping is data

The `month` column is produced before the model, by whatever rule you want:

```python
index = pl.DataFrame({'snapshot': hours}).with_columns(pl.col('snapshot').dt.strftime('%Y-%m').alias('month'))
```

What that produces is the snapshot index the model binds against — a second
column beside the timestamps, and nothing else:

```text
snapshot              month
2030-01-01 00:00:00   2030-01
2030-01-16 00:00:00   2030-01
2030-01-31 00:00:00   2030-01
2030-02-15 00:00:00   2030-02
2030-03-02 00:00:00   2030-03
2030-03-17 00:00:00   2030-03
```

Three snapshots in January, one in February, two in March: `group_sum` needs a
partition, not equal groups.

That one expression is the only place a calendar appears anywhere. Swap it for
`dt.quarter()`, a fiscal-year lookup, or a hand-built table of representative
periods and the model is unchanged — which is why there is no `resample:` or
`reduce_to_monthly()` in the language and never will be. A domain helper would
cover one of those cases; a mapping column covers all of them.

## Reading it back

The budget's dual is a shadow price per month and technology, so a binding cap
says what relaxing it would be worth:

```text
month     generator   dual
2030-01   wind       -49.0   ← binding: displacing gas (50) with wind (1)
2030-02   wind        -0.0
2030-03   wind        -0.0
```

Per-month *results* need no language support at all — a primal is a tidy frame,
so it is a join and a `group_by`:

```python
sol.primal('p').join(index, on='snapshot').group_by('month').agg(pl.col('value').sum())
```

## Why `month` is a dimension

A `coords:` entry is a **function between two dimensions**, so it needs a
codomain. `month` being one is not ceremony — three things rest on it:

1. **`group_sum` replaces `over` with the dimension the coordinate targets.**
   The expression's dims are therefore `[month, generator]`, and a `foreach:`
   can only name declared dimensions.
2. **`monthly_cap` is indexed *by* month.** A parameter carries values *at*
   coordinates; it cannot be the thing a `foreach` ranges over. So month could
   not be a parameter even if the grouping did not need it.
3. **It is what makes a typo an error.** A value in the snapshot index that is
   not a coordinate of `month` is rejected at bind time:

```text
DataError: dimension 'snapshot' coordinate 'month' has value(s) that are
           not 'month' coordinates: '2030-3'
```

That third one is the load-bearing reason. With no declared target there is
nothing to check against, and `2030-3` sitting beside `2030-03` would quietly
become a fourth group with a budget of its own — a smaller problem, solved
without a word. It is the same check that catches a generator assigned to a bus
that does not exist.

**Null is still legal.** A snapshot belonging to no month contributes its terms
nowhere, exactly as a generator on no bus does (SPEC §2). Absent is a claim;
misspelled is a mistake.

## What this cannot do

`group_sum` takes a **partition**: `coords` is a function from snapshot to
month, so each snapshot belongs to exactly one group. Unequal groups are fine,
and a group with no members contributes nothing.

What it cannot express is an **overlapping** aggregate — *"trailing twelve
months, at every month"* — because each snapshot would belong to twelve groups
and no single column can say so. That is a sliding window over a variable, and
it is [ROADMAP](../ROADMAP.md#track-1--primitives) Track 1 item 4.

The same split shows up one level up, where a *process* loops over plans
rather than an expression looping over rows
([#457](https://github.com/FBumann/lpspec/issues/457)): slicing a model per
coordinate is a partition, slicing it per window overlaps. Here `group_sum`
partitions, and the overlapping counterpart is the piece that has not landed.
