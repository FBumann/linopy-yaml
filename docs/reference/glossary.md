# Glossary

The names this project uses, and the one distinction the rest hang off:

> A **spec** is the math you write. A **model** is that spec with your data on
> it. A **result** is one answer read back.

```
check ──▶ Program ──▶ build ──▶ Model ──▶ solve ──▶ Result
 (spec)   (lowered)   (+data)             (answer)
```

## The chain

**Spec**
: The math, before any data — a YAML file, a mapping, or an object the language
  has already read (a `Spec` from `math_spec.to_spec`). It *declares* the
  dimensions, parameters, variables, constraints and objective; it is what is
  checked for being *sayable*. It carries no numbers. This is the input every
  verb takes, spelled `spec` in the signatures.

**Program**
: A spec after the language has parsed, expanded, validated and *lowered* it to
  the internal plan. What [`check`](api.md) returns and what a build reads its
  rows off. Still no data. It is `math_spec`'s own type — typeset it or read its
  declarations through that package.

**Model**
: A spec with your data attached to it — what [`build`](api.md) returns (the class
  `lpspec.Model`). One model feeds any number of sinks: `model.solve()`,
  `model.write(path)`, `model.row(...)`, `model.diagnostics()`, and
  `model.update(...)` puts new numbers on it in place. Instances are named
  `model` in the code.

**Result**
: One answer, read back from a solve: `result.objective`, `result.primal(name)`,
  `result.dual(name)`, `result.expression(name)`. It owns the frames it reads,
  so it outlives the model it came from.

## The verbs

**check** · **build** · **solve** · **write**
: The four things you can do, all on a spec plus (for the last three) sources.
  `check(spec)` validates and lowers; `build(spec, sources)` returns a
  [Model](#the-chain); `solve` and `write` are the one-shot spellings that build
  and then solve or stream in a single call. There is no Python API for
  *constructing* a spec — the math is written in YAML.

**update**
: `model.update(sources)` — put new numbers on the same model, in place, without
  re-reading the YAML or re-lowering the plan. Only what changed is named. When
  the change moves a mask (renumbering labels) the model is rebuilt and solved
  cold rather than pushed onto a loaded solver.

**Buildable**
: The type alias for anything the verbs accept as the spec — `str | Path |
  dict | Spec | Program`.

## How it runs

**Lane**
: One of the two ways a spec is executed. The **relational lane** (the default,
  `lpspec.build`/`solve`) validates at load time, lowers to the plan, and
  streams relationally on polars. The **linopy lane** (`lpspec.linopy`, needs
  the `[linopy]` extra) constructs the same spec as a `linopy.Model`. Both
  accept *exactly* the same language — the reason the differential tests are an
  oracle rather than a comparison of dialects.

**Engine**
: The relational lane's builder. It fills the model's frames from the attached
  data and hands them to a sink.

**Sink**
: Where the built [tables](#the-built-form) land — a solver (`highs`, `gurobi`,
  `xpress`) or a file writer (`.lp`, `.mps`). `linopy` is a lane, not a sink.

**Sources**
: The data you attach — a mapping of parameter, dimension and lookup names to
  tables (parquet paths or in-memory frames), and dimension names to their
  labels.

**attach**
: Fitting your sources onto a spec to make a [Model](#the-chain) — what `build`
  does, and what `update` does again with new numbers. There is no separate
  public verb for it. The data operation is called *attach*, never "bind", so
  that `bound` is free to mean one thing only (below).

## The built form

**Tables** (a `tables` value)
: The built model as a sink sees it: the numeric problem in frames —
  `cols` (bounds, type), `obj`, `rows`, `matrix` (CSR), plus `quad` and `sos`.
  The class name is `Tables`; the variables that hold one are named
  `tables`. It is the *built* form of a model, not the spec.

**keep**
: How much of a solve session `model.solve` may carry to the next solve —
  `solver` (reuse the loaded solver, default), `progress` (keep its work too),
  or `nothing` (a cold baseline). See [the verbs](api.md).

**solve_over** (a sweep)
: Solve one spec once per slice of an axis — scenarios, windows, periods — and
  fold the answers together into a `Runs`. A fold: the previous slice's model is
  released as the loop goes.

## `bound` means one thing

**bound**
: A lower or upper limit on a variable or a constraint row — the `bounds:` of a
  declaration, the `BOUNDS` section of an `.mps` file, an *absent bound* the
  solver reads as infinity. Nothing else. Attaching data to a spec is
  [**attach**](#how-it-runs), never "bind", precisely so that `bound` carries no
  second meaning.
