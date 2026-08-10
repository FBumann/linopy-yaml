# Python API

How you *run* a model. The model itself is the YAML file —
[SPEC](SPEC.md) is what it may contain and what it means; this page is the
sixteen names that load, check, build, solve and read one back. The surface is
pinned by a test and the reasoning behind its size is
[ARCHITECTURE](ARCHITECTURE.md#the-python-surface).

Five verbs — `check`, `load_model`, `build`, `solve`, `write` — and the
exception tree rooted at `LpspecError`: `LanguageError` (with `SchemaError`,
`DimensionError`, `PiecewiseExpansionError`) for the model, `DataError` for what
was bound to it.

```python
import lpspec as lps

lps.check('model.yaml')  # parse → validate → lower, no data bound
model = lps.load_model('model.yaml')  # Model — the declared math
model.to_dict()  # ...and back out, as data
model.to_yaml()  # ...or as the file a reviewer reads

result = lps.solve('model.yaml', sources, solver_options={'time_limit': 60})
# ...or solver_name='gurobi', the other solver sink — same model either way
result.status, result.termination_condition, result.objective
result.is_ok  # rolled-up verdict: not an error, abort or refusal
result.has_primal  # narrower: are there values to read
result.primal('p')  # tidy frame (dims…, value) — the native shape
result.dual('power_balance')  # shadow prices, the same shape and the same join
result.to_pandas('p')  # the same, as a DataFrame
result.to_dataarray('p')  # the same, labelled: .sel / resample / plot
result.to_dataset()  # every variable by default; names for a subset
result.to_parquet(directory)  # streamed to disk, never through this process

lps.write('model.yaml', sources, 'model.lp')  # sink chosen by the suffix
```

**Nothing has to be released.** The built model is frames this process owns, so
`primal` and the `to_*` readers stay valid for as long as the `Result` does.
`close()` and the context-manager protocol exist to hand a large model back
early, not because forgetting them breaks anything. `lps.build` returns the
executor when one build should feed more than one sink:

```python
ex = lps.build('model.yaml', sources)
ex.write('model.lp')
result = ex.solve()
```

What `sources` accepts is [SPEC §8](SPEC.md#8-data-binding). Nothing on this
path imports linopy, and `primal` returns a `polars.DataFrame` — Arrow-backed, so it exports the same protocol the
loader recognises. `to_pandas` and `to_dataarray` are the bridges out and need
pandas / xarray, which ship with the `[linopy]` extra. The only build knob is
`coords`; **`solver_options` is not a build knob** and is forwarded verbatim to
the solver.

**Every verb takes the model four ways**: a path, a `str`, a `dict`, or a
`Model`. `check`, `build`, `solve` and `write` share one first argument, so a
framework that emits declarations never writes a temporary file to run them:

```python
model = {'dimensions': ..., 'variables': ..., 'constraints': ..., 'objectives': ...}

lps.solve(model, sources)  # a dict runs like a file
checked = lps.load_model(model)  # ...or validate once and keep it
checked.to_yaml()  # the review copy — a dict-built model still gets a file
lps.solve(checked, sources)  # a Model is passed through, not revalidated
```

**This is the supported path for a framework**, and it is the one closing #29
and #30 chose: a library composing optional features emits *data*, not YAML
text, and never merges files. What keeps it honest is the last two lines — a
generated model that cannot show you a file is the failure mode hard rule 5
exists to prevent, so `to_yaml()` is not a convenience here, it is the
condition. Hand-written math still starts as a file; nothing about this path
asks it not to.

**A `Model` goes back out two ways, and they agree.** `to_dict()` is the model
as data; `to_yaml()` is that dict as the file hard rule 5 says you review and
diff — which a model built as a *dict*, the way a framework emits one, would
otherwise never have. `tests/test_roundtrip.py` holds `load → out → load` for
both forms over every example and every port, holds that the two forms match,
and holds that dumping twice gives the same bytes, since a review copy that
changes per run is a diff nobody can read.

**Every value is written; only what is absent is dropped** — a null, an
infinite bound, or a mapping that declares nothing. One mechanical rule, on purpose: omitting
*defaults* reads better but needs a list of which ones are consequential, and
that list is a second copy of the schema. An empty **list** stays, because a
list carries cardinality here and zero is one of its values — `foreach: []` is
a scalar declaration.

An infinite bound is in that list because it is not a bound — it is the
unbounded side, which is exactly what omitting the bound already means. That
also makes JSON lossless: JSON has no infinity, so anything reaching
`model_dump_json` as `inf` came back as `null` and read as absent regardless.

The rule lives on the model's **serializer**, so `model_dump`, `model_dump_json`,
`to_dict` and `to_yaml` all give the same content — a helper beside them would
have left pydantic's own methods describing the model differently from the
file.

**Which solver is a caller's choice, not the file's.** `solver_name` is
`highs` (ships with the package) or `gurobi` (needs the `[gurobi]` extra), and
nothing in the YAML names one — the same file means the same model whichever
takes it. Options travel in the chosen solver's own vocabulary,
`{'time_limit': 60}` for HiGHS against `{'TimeLimit': 60}` for Gurobi, because
forwarding verbatim is the contract and translating names would mean holding an
opinion about every option either one has. A name outside the two is an error
listing them, never a quiet fallback to the default.

**Gurobi's remote and licensing options travel the same way**, so Compute
Server, Instant Cloud and WLS need nothing from this package:

```python
options = {'ComputeServer': 'srv:61000', 'ServerPassword': '…'}
lps.solve('model.yaml', sources, solver_name='gurobi', solver_options=options)
```

They are applied when Gurobi's environment is created, which is what
`ComputeServer`, `TokenServer` and `WLSAccessID` require.

Reading a result:

| Rule | |
|---|---|
| **`is_ok` is not `has_primal`** | `is_ok` rolls up the termination condition; `has_primal` adds the solver's verdict on whether an incumbent exists, and is what every reader gates on. A MIP that hits `time_limit` before finding a feasible point is `ok` with nothing to read |
| reading anyway | `NoSolutionError`; `objective` is `nan` |
| `dual` **raises rather than zero-filling** | no values at all is `NoSolutionError`; values but no duals — any integer or binary variable makes them undefined — is `LpspecError`, because only this quantity is missing |
| duals exist only where a solver ran | either solver sink hands them back through the same join; a model written to LP and solved elsewhere never passes back through here. Reduced costs and slacks ride that join too and are not exposed yet |
| `to_dataset` costs what it says | each variable arrives dense over its own dims — name a subset, or use `to_parquet` |
| `write` | the **suffix** picks the writer — `.lp` today, `.mps` a `NotImplementedError` naming it as planned, anything else a `ValueError` listing both sets. Checked before the build, so a format nothing can write costs no model |

**The linopy shim** (`lpspec.linopy.build` / `.extend`, `[linopy]` extra) puts
the same YAML math on a `linopy.Model` that already exists in memory. It is
documented with everything else about that relationship in
[docs/design/linopy.md](design/linopy.md#3-the-shim).
