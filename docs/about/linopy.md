# Relationship to linopy

Everything about [linopy](https://github.com/PyPSA/linopy) in one place, because
it is otherwise the kind of thing that gets mentioned everywhere and explained
nowhere. Three separate relationships, and conflating them is what made the rest
of the docs noisy:

| | What | Where it matters |
|---|---|---|
| **Not a dependency** | the product path never imports it | packaging |
| **The oracle** | how we know the answers are right | testing |
| **The shim** | an opt-in way to put YAML math on a `linopy.Model` | a narrow, real use case |

## 1. It is not a runtime dependency

`lps.solve`, `lps.build`, `lps.write` and `lps.check` go YAML → polars → HiGHS
or LP file, and import nothing from linopy, xarray or pandas. CI proves it: the
bare-install job runs the whole suite with none of them present.

`pip install "lpspec[linopy]"` adds linopy, xarray and pandas, which buys two
things and nothing else — the shim below, and the `to_pandas` / `to_dataarray`
bridges out of a result.

**Nothing on the product path names linopy, including in a traceback.** The
public exception tree is rooted at `LpspecError`, with no alias
([#389](https://github.com/fluxopt/lpspec/issues/389)) — a name from this
extra has no business reaching a caller who never installed it.

## 2. It is the oracle

Correctness here is not "the tests pass"; it is **the same YAML, built both
ways, produces the same model**. The differential suite builds a model through
the relational engine and through linopy, and compares.

That is only meaningful because both paths consume the *same resolved AST* and
neither may hold its own opinion about what a name means — the narrow waist in
[the architecture notes](architecture.md#one-contract-many-consumers). If they resolved
names independently, the suite would be comparing two dialects rather than
checking one language.

It also has a known blind spot, which is why the model gallery exists: a
**shared misreading** passes the differential suite green. Only an outside
published optimum catches that, and
[docs/models/index.md](../models/index.md) is where those live.

**Where a concept is already linopy's, we copy its name** — solve statuses,
`status` / `termination_condition` as two axes with `is_ok` as the rollup, the
shape of a result. Our audience arrives from linopy and PyPSA, and a second
vocabulary for one fact is a tax on all of them. But **copy it, do not import
it**: the engine may not import linopy, so the tables live here and a test
imports linopy to assert the copy still matches. A copy nobody checks is a copy
that rots.

## 3. The shim

The same file, built as a `linopy.Model` instead of bound relationally — the
caller picks the lane by an import, and the call is the one `lps.build` takes:
same first argument (a path, a mapping or a loaded `Model`), same `sources`,
same `coords=`.

```python
from lpspec import linopy as lpspec_linopy

m = lpspec_linopy.build('model.yaml', {...}, coords={...})  # -> linopy.Model
m.solve(...)
lpspec_linopy.expression(m, 'model.yaml', 'co2', {...})  # a named quantity, read back
```

Both are *pure*: YAML in, a model or a value out, nothing retained. `build`
returns a plain `linopy.Model` — no accessor, no attached schema, no patched
attributes — so nothing is lost across `pickle`, `deepcopy` or `to_netcdf`. To
inspect the math, re-read the file with `lps.load_model`.
`expression` is the reader the same purity forces to take `sources` again: it
evaluates a declared named expression ([named expressions](../reference/language/expressions.md#named-expressions))
on the solved model and hands back linopy's native `.solution` — the eager
half of `result.expression(name)`, so the differential suite can hold the two
lanes to one answer.

**This lane constructs; it does not attach.** Math for a `linopy.Model`
something else built — a PyPSA network, say — had a verb here and no longer
does ([#845](https://github.com/fluxopt/lpspec/issues/845)): it was the one
file allowed to reference names it did not declare, and paying for that
exception across the whole language layer bought one use case. Build a second
model and merge it.

### The same language, different data inputs

The shim accepts **exactly the same language** — that equality is what makes the
oracle an oracle, and a construct outside the language is a load error naming
the rewrite, never a redirection to the other path.

A parameter takes the same shapes on both — a parquet path, any table exporting
the Arrow PyCapsule protocol, a `pd.Series` carrying its dims in an index, a
`dict` or a sequence over one dimension, or one number spread over the
coordinates it covers. Neither reads an `xr.DataArray`: this package reads
tables and hands arrays back.

What still differs is narrower, and is the remainder of
[#60](https://github.com/fluxopt/lpspec/issues/60):

| | product path | linopy lane |
|---|---|---|
| dimension labels | `sources`, then `coords=`, then `values:`, then **derived from the parameter tables** | `coords=`, then `values:`, then error — no derivation |
| unnamed index levels | — | bind positionally to the declared dims; named levels bind by name |

The derivation row is the one that bites: on the product path a dimension some
parameter already spans needs no second declaration, but it costs the *declared
order*, which `shift` reads positionally — so pass an explicit index whenever
order matters. A dimension carrying lookups cannot be derived at all, since
derivation reads index columns only.

## What we deliberately do not take

Array operations (`merge`, `reindex`, `stack`), the Python modeling API, and the
solver layer. The first is data prep
([the limits](../reference/language/errors.md#what-the-language-will-not-say)), the second is
[hard rule 5](architecture.md#hard-rules) — the model is the file you review
and diff — and the third is
[#106](https://github.com/fluxopt/lpspec/issues/106), where we adopt linopy's
*design* for declared solver capabilities without adopting its code.

The modeling API is the one a reader arriving from linopy misses first, and
what replaces it is two notebook pages: [Change a model](../interactive.ipynb)
for the loops — `rebind` for new numbers, a longer table for more rows, a
patched `dict` for new math — and [Fix, relax, remove](../lifecycle.ipynb) for
the verbs, which are the same loops aimed at `fix`, `relax` and
`remove_constraints`. What neither replaces is the *debugging*: an IIS, and
printing a built row. Both pages say so.

Where linopy is genuinely ahead, and why none of it is a ceiling question, is the
honest snapshot in [the roadmap](roadmap.md#honest-snapshot).

What is *owed* to linopy rather than merely true of it — and the same for
Calliope, whose math language this surface is derived from — is
[prior art and credit](prior-art.md).
