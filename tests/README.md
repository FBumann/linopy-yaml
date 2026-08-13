# How this suite is built

Four tiers. A new test lands in the cheapest tier that can catch its failure,
using the harness that serves it — not a hand-rolled copy.
[`differential.py`](differential.py)'s docstring records what copies cost last
time: twelve hand-rolled versions in seven files, each carrying a different
fraction of the claim.

| tier | the claim | the harness |
|---|---|---|
| **corpus sweeps** | every model in the repo loads, round-trips, stays inside the language, and its gallery page is current | `conftest.MODEL_PATHS` (= `tools.constructs.models()`) — one list, so a model added anywhere is covered the day it lands |
| **differential** | the same YAML means the same thing on the eager lane and the relational one — and in the written LP file, with `lp=True` | `tests.differential.differential()`; importing it *is* the `[linopy]` guard |
| **probes** | one mechanism each, pinned on the smallest model whose data can reach it | `conftest.DISPATCH_MODEL` + `override`, or a purpose-built module constant |
| **goldens** | an example prints what the docs show | `conftest.run_example` + `assert_golden`; regenerate with `--update-golden` |

`examples/ports/` cuts across the tiers: the one corpus checked against optima
that did not come from us (`conftest.port`, `references.json`) — data and
reference committed *because* the provenance is external.

## Rules

- **A probe model is purpose-built.** A shared model cannot be assumed to
  express a failure: no data change moves a coefficient of
  `examples/dispatch.yaml`, because they are all 1 (#658). Build the smallest
  model whose data reaches the guard, with data small enough to read in a
  failure.
- **A probe moves to `conftest.py` on its second importer.** Not before.
- **Data is generated, seeded and feasible by construction**
  (`conftest.transport_data` is the model). Committed data needs one of two
  reasons: external provenance (the ports) or a golden whose diff is the
  review artifact.
- **linopy, xarray and pandas arrive through `tests.oracle`.** Its docstring
  says why any other spelling is a guard bypass; a fixture that hands out
  pandas imports it in its own body (`conftest.py`'s docstring says why).
- **A correctness guard lands with its mutation table** (AGENTS.md, Part 2):
  delete the guard, run the suite, show the result in the PR — a deletion the
  suite survives gets a probe first.
