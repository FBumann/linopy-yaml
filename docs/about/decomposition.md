# Decomposition, as evidence

**This page is not a feature announcement.** lpspec ships no decomposition
driver, and whether it should is
[#596](https://github.com/fluxopt/lpspec/issues/596). What it shows is narrower
and checkable: the language can *express* a decomposition, and the answer it
reaches is right.

The whole example is in
[`examples/benders/`](https://github.com/fluxopt/lpspec/blob/main/examples/benders/run.py)
and every block below is validated against it.

## Why anyone wants it

A modeller with a model too large to solve has one move: make it smaller.
Representative days instead of a year, forty nodes instead of three hundred, one
weather year instead of many. Each answers a *different question*, with no bound
on how wrong it is for the one that was asked.

Decomposition answers the question that was asked, with a **gap**. Stop at 1% and
you know you are within 1%. A bound rather than a caveat.

## The problem, whole

Choose generator capacity, then dispatch it. Investment and operation decided
together, which is what makes it worth decomposing — the capacity choice is
small and the dispatch is large.

```yaml
dimensions:
  snapshot: {dtype: int}
  generator: {values: [wind, gas]}
parameters:
  invest: {dims: [generator]}
  cost: {dims: [generator]}
  load: {dims: [snapshot]}
  avail: {dims: [snapshot, generator]}
variables:
  cap:
    foreach: [generator]
    bounds: {lower: 0, upper: 100}
  p:
    foreach: [snapshot, generator]
    bounds: {lower: 0}
constraints:
  capacity:
    foreach: [snapshot, generator]
    expression: p <= cap * avail
  balance:
    foreach: [snapshot]
    expression: sum(p, over=generator) >= load
objective:
  sense: minimize
  expression: sum(cap * invest) + sum(p * cost)
```

## The split is one substitution

The subproblem is the same dispatch, at a capacity someone else chose. **`cap`
stops being a variable and becomes a parameter** — that single change is the
whole of the decomposition:

```yaml
dimensions:
  snapshot: {dtype: int}
  generator: {values: [wind, gas]}
parameters:
  cost: {dims: [generator]}
  load: {dims: [snapshot]}
  avail: {dims: [snapshot, generator]}
  cap_hat: {dims: [generator]}          # was `cap`, a variable
variables:
  p:
    foreach: [snapshot, generator]
    bounds: {lower: 0}
constraints:
  capacity:
    foreach: [snapshot, generator]
    expression: p <= cap_hat * avail
  balance:
    foreach: [snapshot]
    expression: sum(p, over=generator) >= load
objective:
  sense: minimize
  expression: sum(p * cost)
```

Note what is *not* here: no `invest`, no `cap` bounds, no investment term. The
subproblem does not know it is part of anything.

## The master, where a cut is data

The master keeps the capacity decision and replaces the dispatch it can no
longer see with `theta`, a single variable standing for what operating that
capacity will cost. Cuts teach it what `theta` really is:

```yaml
dimensions:
  generator: {values: [wind, gas]}
  cut: {dtype: int}
  fcut: {dtype: int}
parameters:
  invest: {dims: [generator]}
  cut_const: {dims: [cut]}
  cut_slope: {dims: [cut, generator]}
  fcut_const: {dims: [fcut]}
  fcut_slope: {dims: [fcut, generator]}
variables:
  cap:
    foreach: [generator]
    bounds: {lower: 0, upper: 100}
  theta:
    foreach: []
    bounds: {lower: 0}
constraints:
  optimality_cut:
    foreach: [cut]
    expression: theta >= cut_const + sum(cut_slope * cap, over=generator)
  feasibility_cut:
    foreach: [fcut]
    expression: sum(fcut_slope * cap, over=generator) <= fcut_const
objective:
  sense: minimize
  expression: sum(cap * invest) + theta
```

**`cut` and `fcut` take their members from data**
([the data contract](../reference/data.md)), so an iteration appends rows
to their parameter tables and this file never changes.
No YAML is generated at runtime, which is what keeps the model a reviewer reads
the model that runs.

`theta` is a scalar variable — `foreach: []` — and starts at `lower: 0`, which
is the only thing keeping the first master bounded before any cut exists.

## Reading a cut out of an answer

A cut is the subproblem's value and its slope, at the capacity that was tried.
The slope is the capacity constraint's shadow price, weighted by availability
and summed over snapshots:

```python
import lpspec as lps
import polars as pl

with lps.solve('examples/benders/sub.yaml', sources) as sub:
    slope = (
        sub.dual('capacity')
        .join(avail, on=['snapshot', 'generator'], suffix='_avail')
        .with_columns((pl.col('value') * pl.col('value_avail')).alias('term'))
        .group_by('generator')
        .agg(pl.col('term').sum().alias('slope'))
    )
```

That is the whole interface with the engine: `dual`, and a join against the
model's own `avail` table. Appending the cut is two `pl.concat` calls onto the
parameter tables the master already declares.

## When the subproblem is infeasible

Below some capacity there is no dispatch at all, and the subproblem says so by
being infeasible. lpspec hands back **no Farkas ray** — an infeasible solve has
no readable status, so `dual()` raises rather than returning a vector of zeros
indistinguishable from an answer.

So the cut comes from a fourth model, which asks *how far from dispatchable*
this capacity is. It is the subproblem with a slack and a different objective:

```yaml
dimensions:
  snapshot: {dtype: int}
  generator: {values: [wind, gas]}
parameters:
  load: {dims: [snapshot]}
  avail: {dims: [snapshot, generator]}
  cap_hat: {dims: [generator]}
variables:
  p:
    foreach: [snapshot, generator]
    bounds: {lower: 0}
  short:
    foreach: [snapshot]
    bounds: {lower: 0}
constraints:
  capacity:
    foreach: [snapshot, generator]
    expression: p <= cap_hat * avail
  balance:
    foreach: [snapshot]
    expression: sum(p, over=generator) + short >= load
objective:
  sense: minimize
  expression: sum(short)
```

Its optimum is zero exactly when the subproblem is feasible, and its capacity
duals are the slope the feasibility cut needs. It is a separate file because a
model declares one objective — "minimise the violation" cannot be a second
objective on the subproblem.

## The loop

```python
sub_model, feasibility_model, master_model = (load_model(path) for path in paths)

for step in range(25):
    with lps.solve(sub_model, {**dispatch, 'cap_hat': capacity}) as sub:
        dispatchable = sub.has_primal
        if dispatchable:
            slope, here_value = slope_at(sub, capacity)
            upper = min(upper, spent(capacity) + sub.objective)
            appended(tables, 'cut', sub.objective - here_value, slope)

    if not dispatchable:
        with lps.solve(feasibility_model, {**dispatch, 'cap_hat': capacity}) as short:
            slope, here_value = slope_at(short, capacity)
            appended(tables, 'fcut', here_value - short.objective, slope)

    with lps.solve(master_model, {**master_sources, **coordinates}) as master:
        lower = master.objective
        capacity = master.primal('cap').select('generator', 'value')

    if upper < float('inf') and upper - lower <= 1e-6 * abs(upper):
        break
```

Twenty lines, three `lps.solve` calls, and a growing pair of tables. **A reader
could write this**, which is the observation that matters most for
[#596](https://github.com/fluxopt/lpspec/issues/596).

The models are loaded above the loop because none of them changes — a cut is a
row in a parameter table, not an edit to a file. `lps.solve` accepts a `Model`
anywhere it accepts a path, so parse and validation are paid once for the run
instead of three times an iteration. That is not decomposition-specific: it is
what any driver over a fixed model does, and `solve_over` already does it.

## Running it

```bash
pixi run python examples/benders/run.py
```

```text
the whole problem, in one plan: 9600.00

  step 0  feasibility  lower  2025.00   upper none yet
  step 1  feasibility  lower  2625.00   upper none yet
  step 2  feasibility  lower  2850.00   upper none yet
  step 3  optimality   lower  9600.00   upper 9600.00

decomposed: 9600.00 in 4 steps
monolithic: 9600.00
difference: 0.0e+00
cuts: 1 optimality, 3 feasibility
```

Three capacities are excluded as undispatchable before one proves feasible; the
first optimality cut then closes the gap exactly.

## The check is the algorithm's own

A decomposed answer is only interesting if it is the *same* answer, and lpspec
can always build the monolith from the same sources. So the example solves both
and prints the difference — `0.0e+00` above, asserted in
`tests/test_benders_example.py`.

This is the two-lane differential test aimed at an algorithm instead of an
engine, and it is a property of writing models declaratively: the undecomposed
form is always available, because it is another file over the same data.

## What is deliberately absent

The loop above is not the hard part. What is missing is everything that makes a
decomposition survive a real model: cut management as the master grows,
stabilisation, multi-cut, tolerances that hold when duals are degenerate, and an
answer for when convergence does not happen.

That is the surface [#596](https://github.com/fluxopt/lpspec/issues/596) asks
whether to own. Nothing here settles it. What this page settles is that the
*language* is not the obstacle.
