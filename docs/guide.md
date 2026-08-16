# Write a model

Five ideas carry the whole language. Each one is shown below in a model that
lives in the repo and is run by the test suite, so nothing here is a snippet
that only works on this page.

If you would rather see the machinery than read about it,
`python examples/walkthrough.py` prints every stage — YAML → schema → AST →
plan → frames → LP text → solution — for one small model. And once a model
exists, [Change a model](interactive.ipynb) is the notebook loop — new numbers,
more rows, new math, and how to tell which of the three you are paying for.
Every output on that page is the site build's own run.

## 1. A dimension is declared; its coordinates usually are not

```yaml
dimensions:
  snapshot: {dtype: int}  # coordinates come from the data
  generator: {values: [wind, solar, gas]}  # coordinates are given here
```

A dimension is an axis. You either list its coordinates in the file, or leave
them to be read off whatever data binds — `{"snapshot": range(6)}` in `sources` at
call time, or the union of what the parameters carry.

**One master coordinate set per dimension, resolved before any data binds.**
Every parameter is reindexed onto it, so two tables that disagree about which
snapshots exist is an error you get at load time rather than a silently
truncated model.

## 2. Absence is how you say "sparse"

```yaml
variables:
  p:
    foreach: [snapshot, generator]
    where: "p_max > 0"
```

`where` does not zero a variable out — it means the variable **has no column
there at all**. A retired generator with `p_max = 0` costs nothing to carry in
the data, and the built model is smaller than the coordinate product.

The same idea runs through data binding, with one distinction worth learning
early: a **variable** the mask removed is *absent*, and a term carrying it takes
its whole row with it — while a **parameter** row that is simply missing is a
zero coefficient, and the row survives without it. Absence is a property of
variables. → [dispatch](examples/dispatch.md), [absence](reference/language/absence.md)

## 3. A lookup maps one dimension onto another, and that is your topology

```yaml
lookups:
  gen_bus: {over: generator, into: bus}  # each generator sits on a bus
  line_from: {over: line, into: bus}  # both endpoints are buses
  line_to: {over: line, into: bus}
```

```yaml
- expression: >-
    sum(p, by=gen_bus)
    + sum(f, by=line_to)
    - sum(f, by=line_from)
    == load
```

`sum(by=)` sums along a lookup, landing the result on the dimension the
lookup points at. The same `f` is summed twice through two different lookups —
once as inflow, once as outflow.

No adjacency matrix and no join written by hand: the network is data on the
dimension. → [transport](examples/transport.md)

## 4. `shift` reaches along an axis

```yaml
- expression: soc == shift(soc, over=snapshot, offset=1, edge='wrap') + charge * 0.9 - discharge
```

One operator, and `edge=` says what happens at the boundary. `edge='wrap'` is
cyclic — the first snapshot reads the last, which is what makes a battery
cyclic without writing the boundary condition out. Omit `edge` and positions
translated past the edge are **absent**, so the row they would have fed is not
built. `edge=0` keeps the row and contributes zero there instead.

This is the only construct whose cost is not obviously linear in model size.
→ [storage](examples/storage.md)

## 5. The dims of an equation must equal its `foreach`

```yaml
constraints:
  power_balance:
    foreach: [snapshot]
    expression: sum(p, over=generator) == load
```

`p` has dims `(snapshot, generator)`; summing over `generator` leaves
`(snapshot)`; `load` has `(snapshot)`. The union is `(snapshot)`, which is what
`foreach` says — so it compiles.

Get it wrong and you are told at load time, not at solve time. A stray dim
would multiply rows and an unused `foreach` dim would repeat one row across
them; either way you would build a different model than the file reads as.
→ [the dim algebra](reference/language/expressions.md#dim-algebra)

## Then: check, build, solve

```python
import lpspec as lps

lps.check('model.yaml')  # compiles? no data needed
sol = lps.solve('model.yaml', sources)  # to an answer
sol.objective
sol.primal('p')  # a polars.DataFrame
sol.dual('power_balance')
sol.activity('power_balance')  # the row's left-hand side at the solution
```

`lps.check` is the CI verb — it parses, expands, resolves and lowers without
binding anything, so a model repository can be validated on every commit
without shipping the data.

Sources accept polars, pandas, pyarrow, or parquet paths — anything exposing
the Arrow PyCapsule protocol, and the recogniser imports none of them.
Results come back as frames; `to_pandas`, `to_dataarray` and `to_parquet` are
the bridges out. → [data binding](reference/language/data.md), [Python API](reference/api.md)

## Editor completion and offline checking

The YAML surface ships as a JSON Schema —
[`schema/lpspec.schema.json`](https://github.com/fluxopt/lpspec/blob/main/schema/lpspec.schema.json),
generated from the same models `lps.check` runs, held current by a test. With
the [Red Hat YAML extension](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml)
it gives key completion, hover docs, the closed vocabulary behind `dtype:`,
`domain:`, `sense:` and the rest, and a squiggle on a misspelled key — before Python runs. Map it per workspace:

```jsonc
// .vscode/settings.json
"yaml.schemas": { "./schema/lpspec.schema.json": ["*.model.yaml"] }
```

or per file, with a modeline on the first line:

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/fluxopt/lpspec/main/schema/lpspec.schema.json
```

The same file checks a model without Python, which is what a pre-commit hook
or a non-Python CI job wants:

```bash
uvx check-jsonschema --schemafile schema/lpspec.schema.json model.yaml
```

It validates structure only. `expression:` and `where:` are strings to the
schema, so everything inside them — the actual math — is checked by
`lps.check`, not by the schema.

## What it will not do

Worth knowing before you start, rather than after:

- **Bounds take a name or a number, never arithmetic.** `upper: p_max` is
  fine; `upper: -rating` is not. This one has bitten a real port —
  [#31](https://github.com/fluxopt/lpspec/issues/31), and the workaround is to
  ship the negated column as data.
- **Every expression is affine in the variables — except the objective**,
  which takes `variable * variable`. Everywhere else a product needs a
  variable-free factor. That ceiling is what the whole design is built around,
  and where it sits is a choice with reasons. →
  [The ceiling](about/ceiling.md#two-tiers-and-the-ceiling)
- **Several plausible features are refused on purpose**, with reasons.
  → [the roadmap](about/roadmap.md)

## Where next

| | |
|---|---|
| [Examples](examples/index.md) | every model in the repo, and which constructs each exercises |
| [Language reference](reference/language/index.md) | what a file may contain, exactly |
| [Python API](reference/api.md) | building, solving, and reading an answer back |
| [Typeset the math](reference/typeset.md) | the same file as LaTeX, Typst or Markdown |
| [About](about/index.md) | why it is shaped this way, what it costs, where it is going |
