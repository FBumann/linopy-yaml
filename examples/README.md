# Examples

**These are fixtures, not samples.** Eleven test modules load them by path, and
`docs/models/` embeds them, so the file you read here is the file CI runs.
Renaming one breaks tests; changing one changes what the docs claim, and a test
will say so.

**Read them explained** in [docs/models/](../docs/models/index.md) — the maths,
what each construct exercises, and for a port a side-by-side against the
reference implementation. This directory is the source; that is the guided tour.

| | |
|---|---|
| `dispatch.yaml` | least-cost generation against a load profile — the smallest complete model |
| `storage.yaml` | dispatch plus a cyclic battery (`shift(edge='wrap')`) |
| `transport.yaml` | a network: coordinates on a dimension *are* the topology (`sum(group_by=)`) |
| `piecewise.yaml` | per-generator convex cost curves (`piecewise:`) |
| `sos.yaml` | the same curve stated as a set the solver branches on, rather than built from binaries (`method: sos2`) |
| `monthly_budget.yaml` | a cap per calendar month: time grouped through a coordinate, exactly as a generator sits on a bus (`sum(group_by=)`) |
| `multi_period.yaml` | capacity decided once per investment period and binding at every snapshot in it (`at()`) |
| `walkthrough.yaml` | the model `walkthrough.py` prints every pipeline stage for |
| `rolling/` | a storage schedule solved a window at a time, and what the lookahead buys (`solve_over`, `EachWindow`) |
| `myopic/` | an investment pathway over periods of typical days, each inheriting the last one's fleet (`solve_over`, `carry`) |
| `benders/` | the problem split in two and reassembled, checked against the monolith it decomposes |
| `ports/` | eleven models somebody else already solved, checked against an optimum that did not come from us |

`walkthrough.py` runs one model through YAML → schema → AST → plan → frames →
LP text → solution, printing what each stage produces, then two models the
language refuses and why. Its output is committed as `walkthrough.out` and
asserted, so it cannot drift from what the code does:

```bash
python examples/walkthrough.py
```

`ports/` carries three or four files per model — the YAML, the instance, the
recorded objective with its provenance, and a reference implementation
importing no lpspec. That last one is absent where the optimum is *published*
and needs nothing of ours to reproduce it: `facility_location` (OR-Library) and
`tsp_mtz` (TSPLIB) cite the literature instead, which is the strongest tier the
corpus has.

Reference scripts are **never run by CI** and carry their dependencies inline;
adding a port is described in
[CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-ported-model).
