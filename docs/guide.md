# Run a model

A model is two things: a YAML file, and a mapping of the names it declares to
your tables. This page is the path from those two to an answer you can read
back. What may be *in* the file is the language, and the language is
[documented with itself](https://math-spec.readthedocs.io/en/latest/reference/language/) —
it is a package this one depends on, not a chapter of this site.

If you would rather see the machinery than read about it,
`python examples/walkthrough.py` prints every stage — YAML → schema → AST →
plan → frames → LP text → solution — for one small model. And once a model
exists, [Change a model](interactive.ipynb) is the notebook loop — new numbers,
more rows, new math, and how to tell which of the three you are paying for.
Every output on that page is the site build's own run.

## The language, in five links

Enough to read a model file, each shown in a model that lives in the repo and
is run by the test suite:

| | |
|---|---|
| **A dimension is an axis, and its coordinates usually come from the data.** One master set per dimension, resolved before anything attaches, so two tables disagreeing about which snapshots exist is a load-time error rather than a truncated model. | [dimensions](https://math-spec.readthedocs.io/en/latest/reference/language/dimensions/) · [dispatch](examples/dispatch.md) |
| **Absence is how you say "sparse".** A `where:` does not zero a variable out — the variable has no column there at all, and the built model is smaller than the coordinate product. | [absence](https://math-spec.readthedocs.io/en/latest/reference/language/absence/) · [dispatch](examples/dispatch.md) |
| **A lookup maps one dimension onto another, and that is your topology.** `sum(p, by=gen_bus)` lands on the dimension the lookup points at; no adjacency matrix and no join written by hand. | [lookups](https://math-spec.readthedocs.io/en/latest/reference/language/dimensions/#lookups) · [transport](examples/transport.md) |
| **`shift` reaches along an axis**, and `edge=` says what happens at the boundary — `'wrap'` is what makes a battery cyclic without writing the boundary condition out. | [shift](https://math-spec.readthedocs.io/en/latest/reference/language/operators/#shift) · [storage](examples/storage.md) |
| **The dims of an equation must equal its `foreach`.** Get it wrong and you are told at load time: a stray dim would multiply rows, an unused one would repeat a row across them. | [dim algebra](https://math-spec.readthedocs.io/en/latest/reference/language/expressions/#dim-algebra) · [monthly budget](examples/monthly_budget.md) |

## Check, build, solve

```python
import lpspec as lps

lps.check('spec.yaml')  # compiles? no data needed
sol = lps.solve('spec.yaml', sources)  # to an answer
sol.objective
sol.primal('p')  # a polars.DataFrame
sol.dual('power_balance')
sol.activity('power_balance')  # the row's left-hand side at the solution
```

`lps.check` is the CI verb — it parses, expands, resolves and lowers without
attaching anything, so a model repository can be validated on every commit
without shipping the data. With `sink=` it also answers the second question:
will *that* solver take this model.

Between the two sits `lps.build(spec, sources)`, which attaches and builds
without solving — what you want when the same built model is solved many times
with new numbers, and what [`update`](reference/api.md#re-solving-with-new-numbers)
re-uses. → [Python API](reference/api.md)

## Your numbers go in as tables

`sources` maps each declared name to a **parquet path, or any table exposing
the Arrow PyCapsule protocol** — polars, pandas, pyarrow, duckdb — and the
recogniser imports none of them on your behalf. Results come back as frames;
`to_pandas`, `to_dataarray` and `to_parquet` are the bridges out.

Two pages cover the whole of it: [preparing the data](examples/data.md) is the
journey from the files an instance actually arrives in to that mapping, and
[the data contract](reference/data.md) is what attaches, what is refused, and
which sentence you get when it is.

## Editor completion and offline checking

The YAML surface ships as a JSON Schema —
[`schema/math-spec.schema.json`](https://github.com/energy-models/math-spec/blob/main/schema/math-spec.schema.json),
generated from the same declarations `lps.check` validates against and held
current by a test. It travels with the language, so the examples below read it
from math-spec over the network; a vendored copy takes a path in the same slot.
With
the [Red Hat YAML extension](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml)
it gives key completion, hover docs, the closed vocabulary behind `dtype:`,
`domain:`, `sense:` and the rest, and a squiggle on a misspelled key — before Python runs. Map it per workspace:

```jsonc
// .vscode/settings.json
"yaml.schemas": { "https://raw.githubusercontent.com/energy-models/math-spec/main/schema/math-spec.schema.json": ["*.model.yaml"] }
```

or per file, with a modeline on the first line:

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/energy-models/math-spec/main/schema/math-spec.schema.json
```

The same file checks a model without Python, which is what a pre-commit hook
or a non-Python CI job wants:

```bash
uvx check-jsonschema --schemafile https://raw.githubusercontent.com/energy-models/math-spec/main/schema/math-spec.schema.json model.yaml
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
- **The math takes degree 2; what stands beside it does not.** The objective
  and constraints take `variable * variable`; a bound, a named expression and a
  `piecewise:` link need a variable-free factor. Where a quadratic model can be
  *solved* is a second question — `check(spec, sink=…)` answers it. →
  [The ceiling](https://math-spec.readthedocs.io/en/latest/about/ceiling/#two-tiers-and-the-ceiling)
- **Several plausible features are refused on purpose**, with reasons.
  → [the roadmap](about/roadmap.md)

## Where next

| | |
|---|---|
| [Preparing the data](examples/data.md) · [the contract](reference/data.md) | files to frames, and what attaching refuses |
| [Python API](reference/api.md) · [Sweeps](reference/sweeps.md) | building, solving, reading back; and solving once per slice |
| [Examples](examples/index.md) | every model in the repo, and which constructs each exercises |
| [Language reference](https://math-spec.readthedocs.io/en/latest/reference/language/) | what a file may contain, exactly |
| [Typeset the math](https://math-spec.readthedocs.io/en/latest/reference/typeset/) | the same file as LaTeX, Typst or Markdown |
| [About](about/index.md) | why it is shaped this way, what it costs, where it is going |
