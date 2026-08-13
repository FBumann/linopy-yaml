# Sinks

How a built model leaves the engine — the boxes downstream of the engine in
[docs/ARCHITECTURE.md](../../../../docs/ARCHITECTURE.md)'s pipeline, which
carries the argument for the split. This page is the membership list.

**Two families.** A **solver** takes the tables and runs them; a **writer**
takes the tables and renders them to a file. Everything else follows.

| | solvers/ | writers/ |
|---|---|---|
| answers | a `Solver` subclass holding one model | `(tables, path) -> None` |
| chosen by | **name**, at the call — `solver_name='gurobi'` | **suffix**, from the output — `model.lp` |
| registry | `SOLVERS`, closed, holding the classes | `WRITERS` + `PLANNED_WRITERS`, closed |
| members | `highs.py` (`highspy`, ships), `gurobi.py` (`[gurobi]`: `gurobipy`, `scipy`), over `base.py` | `lp_file.py` (nothing beyond polars) |

## Staying loaded

`base.py` is what a solver **is**: a loaded model with a lifecycle, which is
linopy's shape and its word — their `Solver` is the persistent object too. It
holds no solver of its own, which is the whole reason it is allowed to exist
beside the leaves: it cannot carry an optional dependency across the fence
below, and sharing through it is what stops one leaf importing the other.

The split is by who can answer. `solvers.loaded(held, name, …)` is the whole of
**reuse or load again**: it keeps a held solver exactly when it is the named
class whose recorded digest and options match the new tables — and then pushes
the new numbers onto it — closing and replacing it otherwise. The base records
that evidence at the load; a subclass owns **the hand-off**:

| | |
|---|---|
| `solvers.loaded(held, name, …)` | reuse or load again — the whole of that decision |
| `Solver.run(tables)` | `_run`, plus the refusal of a vector that does not span the model |
| `_load(tables, batch_rows)` | hand the model over and hold what reads it back |
| `push(tables)` | only after `loaded` matched the digest — new bounds, costs and right-hand sides |
| `_run(tables)` | solve what is loaded, and read it back |
| `close()` | drop the handle, and any licence with it |

The first two are the family's and identical for everyone; the last four are a
member's, and are its own library's shape. Nothing above the family decides
which solver to keep or checks what one returned — an engine hands over tables
and is given an answer.

So a model rebuilt with new numbers (`bound.rebind`) has them pushed onto what
the solver already holds and solves from the basis the last one ended on. Both
sinks do this; a solver that could not would be slower to re-solve and nothing
else.

The guard is `ModelTables.structure` — a digest of everything a re-solve may
not change, recorded by the solver at its load and cached on the tables. **Values are re-pushed, not diffed**: linopy's persistent layer
(`persistent/diff.py`) computes a delta against a snapshot of the previous
model, where here the previous model is released before the new one exists, that
release being what keeps a rebound build at one model's peak. What a diff would
need is exactly what is not kept.

`tables.py` is what both read. Neither family imports the other and no member
imports a sibling — `tests/test_architecture.py` reads all of that off the
path, which is what keeps `gurobipy` off the import path of a caller who
solves with HiGHS.

## The contract

A sink takes a `ModelTables` and nothing else: the frames `cols`
(col, lb, ub, vtype), `obj` (col, coeff), `rows` (row, sense, rhs) and `matrix`
(row, col, coeff), plus the counts it chunks by and the objective's sense and
constant — those last two live outside the tables because a constant has no
column to attach to.

A sink never learns how the tables were filled, and the engine never learns
how they are drained. That is the point: adding `mps` is a new module in
`writers/`, not another method on `PolarsEngine`.

The one thing sinks may share is a *projection* of those frames, never a step
of the work — `ModelTables.dense_columns`, which both solvers read.

## Adding one

**A solver:** `solvers/<name>.py` named for the solver, defining a `Solver`
subclass named for it — `_load`, `push`, `_run`, `close` — plus the
`build_<name>` seam `bench/` measures, and one line in `SOLVERS` holding the
class.
Import the solver **inside the function** and declare an extra for it — the
module boundary is the fence, the lazy import is what keeps this package free
to import for callers who will never use it. Copy linopy's status map for it
and pin the copy in `tests/test_solve_status.py`, including anywhere you
deliberately diverge.

**A writer:** `writers/<format>.py`, one line in `WRITERS` keyed by suffix,
moved out of `PLANNED_WRITERS` if it was there.

Either way: stream — nothing here may materialise the model a second time —
and nothing above changes. No method on the engine, no branch in `api.py`,
no name on the Python surface.

## When Track 3 lands

[Track 3](https://github.com/fluxopt/lpspec/issues/472)
gives each sink a declared capability table so `check(model, sink=...)` can
answer "will this sink take it". The table belongs in the sink's own module,
collected by the family `__init__` rather than owned by it — and it stops
being uniform at exactly the seam this directory draws: SOS is native in
`gurobi`, a text section in `lp_file`, absent in `highs`.

## Stable output

Two runs of one model produce the same bytes
([#109](https://github.com/fluxopt/lpspec/issues/109)). It is not free and it
is easy to lose: a parallel join hands back a group in whatever order it
finished it, so a sink that gathers a row's terms and *then* orders the rows
has already lost the order within one. `lp_file` emits one frame of lines
carrying its own sort key instead, and sorts once. The solvers are ordered for
a different reason — `searchsorted` requires it, and the CSR `indptr` it
produces is only a row's extent if the rows it indexes are sorted.
