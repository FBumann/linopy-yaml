# Decomposition, as evidence

**This page is not a feature announcement.** lpspec ships no decomposition
driver, and whether it should is
[#596](https://github.com/fluxopt/lpspec/issues/596). What this page shows is
narrower and checkable: the language can *express* a decomposition, and the
answer it reaches is right.

That distinction matters because the two halves have very different costs. The
modelling is free — it needs nothing the language does not already have. Owning
the algorithm is not, and that is the open question.

## Why anyone wants it

A modeller with a model too large to solve has one move today: make it smaller.
Representative days instead of a year, forty nodes instead of three hundred, one
weather year instead of many, myopic pathways instead of foresight. Each is a
budget choice wearing modelling clothes, and each answers a *different question*
with no bound on how wrong it is for the one that was asked.

Decomposition answers the question that was asked, with a **gap**. Stop at 1%
and you know you are within 1%. That is the difference between a caveat and a
bound, and it is the whole of the motivation.

## The worked example

[`examples/benders/run.py`](https://github.com/fluxopt/lpspec/blob/main/examples/benders/run.py)
is a capacity-expansion problem — choose generator capacity, then dispatch it —
solved twice: once whole, once split. Four files:

| file | what it is |
|---|---|
| `monolith.yaml` | the problem in one plan; the answer everything else must reach |
| `master.yaml` | capacity, plus a placeholder for what operating it will cost |
| `sub.yaml` | dispatch at a capacity someone else chose — *infeasible* if it is too small |
| `feasibility.yaml` | how far from dispatchable a capacity is, when it is too small |

Run it:

```bash
uv run python examples/benders/run.py
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

## A cut is data

The master declares its cut families once and never changes:

```yaml
dimensions:
  cut: {dtype: int}
  fcut: {dtype: int}
parameters:
  cut_const: {dims: [cut]}
  cut_slope: {dims: [cut, generator]}
constraints:
  optimality_cut:
    foreach: [cut]
    expression: theta >= cut_const + sum(cut_slope * cap, over=generator)
```

`cut` takes its members from data ([§8](../SPEC.md)), so an iteration appends
rows to two parameter tables and the file stays exactly as written. **No YAML is
generated at runtime**, which is what keeps the model a reviewer reads the model
that runs — the same reason `lpspec` exists at all, applied to an algorithm
instead of a network.

## The check is the algorithm's own

A decomposed answer is only interesting if it is the *same* answer. lpspec can
always build the monolith from the same sources, so the example solves both and
prints the difference — `0.0e+00` above, asserted in
`tests/test_benders_example.py`.

This is the two-lane differential test aimed at an algorithm rather than at an
engine, and it is a property of building models declaratively: the undecomposed
form is always available, because it is just another file over the same data.

## What it costs, honestly

Three things this example needed, each a consequence of a decision that is right
on its own:

**No Farkas ray.** An infeasible solve has no readable status, so `dual()`
raises rather than handing back a vector of zeros indistinguishable from an
answer. The feasibility cut therefore cannot be read off the infeasible
subproblem — it comes from minimising the violation instead.

**So there is a fourth file.** A model declares one objective, so *"minimise the
violation"* cannot be a second objective on the subproblem. `feasibility.yaml`
restates the subproblem's constraints in order to minimise something else over
them.

**And a second cut family.** An optimality cut bounds `theta`; a feasibility cut
bounds capacity alone and mentions no `theta`. They cannot share a dimension.

## What is deliberately absent

The loop in `run.py` is about twenty lines and a reader could write it. What is
not there is everything that makes a decomposition survive a real model: cut
management as the master grows, stabilisation, multi-cut, tolerances that hold
when duals are degenerate, and an answer for when convergence simply does not
happen.

That is the surface [#596](https://github.com/fluxopt/lpspec/issues/596) asks
whether to own, and nothing on this page settles it. What the page settles is
that the *language* is not the obstacle.
