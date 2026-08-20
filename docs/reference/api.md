# Python API

How you *run* a model. The model itself is the YAML file — what it may contain
is [the language](language/index.md); this page is what loads, checks, builds,
solves and reads one back.

```python
import lpspec as lps

lps.check('model.yaml')  # compiles? no data needed

result = lps.solve('model.yaml', sources)
result.objective
result.primal('p')  # a polars.DataFrame
result.dual('power_balance')
```

## The verbs

| | |
|---|---|
| `lps.check(model, sink=None)` | parse, expand, validate and lower; bind no data. With a `sink`, also whether that sink will take it. Returns the validated `Model` |
| `lps.load_model(model)` | the same parse, without the lowering pass and its warnings |
| `lps.build(model, sources)` | bind data and build it — returns a `BoundModel` |
| `lps.solve(model, sources, solver_name='highs', solver_options=None)` | build and solve in one call — returns a `Result` |
| `lps.solve_over(model, sources, axis, ...)` | solve once per slice and fold the answers — [sweeps](sweeps.md) |
| `lps.write(model, sources, out)` | build and stream to a file; the suffix picks the format |
| `bound.row(name, **coordinate)` | what one built constraint row says — terms, comparison, right-hand side |
| `lps.to_latex` / `to_typst` / `to_markdown` | the math as a document — [typeset](typeset.md) |

Errors are one tree: `LpspecError` at the root, `LanguageError` (with
`SchemaError`, `DimensionError`, `PiecewiseExpansionError`) for the model, and
`DataError` for what was bound to it
([errors](language/errors.md#which-error-you-get)).

**`check` is the CI verb.** It parses, expands, resolves and lowers without
binding anything, so a model repository can be validated on every commit
without shipping the data.

### `sink=`, the second question

Whether a model is *sayable* is solver-independent. Where it can *land* is a
separate axis — [what a sink can
ingest](../about/ceiling.md#capability-is-not-the-ceiling) — and `sink=` is how
you ask about it:

```python
lps.check('model.yaml')  # sayable?
lps.check('model.yaml', sink='highs')  # ...and will HiGHS take it?
lps.check('model.yaml', sink='.lp')  # ...will the LP writer?
```

A solver name (`highs`, `gurobi`) or an output suffix (`.lp`). It is **optional
and silent by default** — most models never leave the common subset, so warning
about a sink nobody named would be noise on every one of them. You get back:

- **A refusal** (`LpspecError`) if the sink has no such concept, or refuses the
  combination — naming the construct, the sink, *and* the sinks that do take
  it. Degree 2 is what reaches one: no sink but Gurobi and the LP writer takes
  a quadratic row, and HiGHS refuses a quadratic objective *beside* integrality
  while taking either alone.
- **A warning** if the sink takes it only by rewriting. `sos:` on HiGHS is the
  one that exists: it arrives as binaries, so a model that declared no
  integrality comes back mixed-integer and without duals — better read before
  the solve than inferred from an empty `dual()`.

Answered off a declared table with **no data and no installed solver**, so
`check(m, sink='gurobi')` answers on a machine that has never had gurobipy.

Asking is optional; being refused is not. `solve` and `write` read the same
table when they get there, so `lps.write(m, sources, 'model.mps')` on a model
carrying a quadratic term is refused by name rather than handed back as a file
whose quadratic rows are missing — which would parse, solve, and answer for a
different model. What `sink=` buys is the same sentence before the build.

## Sources

`sources` maps declared names to data: parquet paths, or any table exposing the
Arrow PyCapsule protocol — polars, pandas, pyarrow. A dimension's own key supplies
dimension labels that neither the sources nor the YAML carries. The exact rules
are [data binding](language/data.md).

```python
result = lps.solve(
    'dispatch.yaml',
    {'load': 'load.parquet', 'cost': cost_frame, 'p_max': p_max_frame},
)
```

`sources` is the whole of the build's input — parameters and dimension indexes
in one mapping. **`solver_options` is not a build knob** — it is forwarded to
the solver verbatim.

## Reading a result

```python
result.status, result.termination_condition, result.objective
result.is_ok  # rolled-up verdict: not an error, abort or refusal
result.has_primal  # narrower: are there values to read
result.kept  # how much of the session this solve kept: 'nothing', 'solver' or 'progress'

result.primal('p')  # tidy frame (dims…, value) in label order — the native shape
result.dual('power_balance')  # shadow prices, same shape, same join
result.activity('power_balance')  # each row's left-hand side at the solution
result.expression('co2')  # a named expression at the solution, over its own dims

result.to_pandas('p')  # the same, as a DataFrame
result.to_dataarray('p')  # the same, labelled: .sel / resample / plot
result.to_dataset()  # every variable by default; names for a subset
result.to_parquet(directory)  # streamed to disk, never through this process
```

`primal` returns a `polars.DataFrame` — Arrow-backed, so it exports the same
protocol the loader recognises. `to_pandas` and `to_dataarray` are the bridges
out and need pandas / xarray, which ship with the `[linopy]` extra.

| Rule | |
|---|---|
| **`is_ok` is not `has_primal`** | `is_ok` rolls up the termination condition; `has_primal` adds the solver's verdict on whether an incumbent exists, and is what every reader gates on. A MIP that hits `time_limit` before finding a feasible point is `ok` with nothing to read |
| reading anyway | `NoSolutionError`; `objective` is `nan` |
| **`expression` takes a declared name** | the value of a [named expression](language/expressions.md#named-expressions) at the solution, aggregated to its own dims. Never an expression string; an unknown name is a `KeyError` listing what is declared. It is compiled at the read, so a build with fifty declared expressions that reads none pays for none |
| `dual` **raises rather than zero-filling** | no values at all is `NoSolutionError`; values but no duals — any integer or binary variable makes them undefined — is `LpspecError`, because only this quantity is missing |
| **a solver can make a model mixed-integer** | an [`sos:`](language/piecewise.md#sos) set reaches a solver with no SOS concept as binaries, so an otherwise continuous model solved on `highs` has no duals and says so. On `gurobi` and `xpress`, which branch on the set itself, it keeps them |
| duals exist only where a solver ran | a model written to LP and solved elsewhere never passes back through here. Reduced costs and slacks are not exposed yet |
| `to_dataset` costs what it says | each variable arrives dense over its own dims — name a subset, or use `to_parquet` |

**Nothing has to be released.** The built model is frames this process owns, so
`primal` and the `to_*` readers stay valid for as long as the `Result` does.
`close()` and the context-manager protocol exist to hand a large model back
early, not because forgetting them breaks anything.

## Building once, solving many times

`lps.build` returns a `BoundModel` — the math with your data on it — for when
one build should feed more than one sink, or be solved more than once:

```python
bound = lps.build('model.yaml', sources)
bound.write('model.lp')
result = bound.solve()
bound.diagnostics()  # what the build and its solves did that the answer does not show
bound.row('balance', snapshot=17)  # what one row actually says
```

**Inspecting a model is `build`'s job, not `solve`'s.** `solve` hands back an
answer and `write` a path; the questions *about the model* — how big is it,
what did it not build, what does this row say, how did its re-solves go —
belong to the handle that **is** the model.

### Reading one row

`to_latex` and its siblings render the model as math **before any data**, and
`result.dual('balance')` gives a row's number **without its terms**. `row` is
the third question, and the one a wrong model is debugged by: what does this
constraint, at this coordinate, actually say?

```python
print(bound.row('balance', snapshot=1))
# balance[snapshot=1]: +1 p[1, wind] +50 p[1, gas] +30 p[1, coal] >= 60
```

That line is **linopy's**, on purpose — their `Constraint.print()` renders a row
the same way, and a reader arriving from there should not have to learn a
second way to read a constraint. What is added is the row's own identity on the
same line, where linopy prints it as a header.

The same content is a frame, for the row too wide to read and for anything that
filters or joins:

```python
row = bound.row('balance', snapshot=17)
row.terms  # (variable, coordinate, coefficient), one row per term
row.sense  # '=='
row.rhs  # 80.0
```

A row too wide to spell out **summarises rather than truncating** — twelve
terms of three hundred are twelve arbitrary ones:

```python
print(bound.row('balance', t=0))
# balance[t=0]: 301 terms — p: 300 (|coef| 0.001…0.3), slack: 1 (|coef| 1000) >= 5
```

That is the two questions a wide row is actually asked, on one line: how much
of it each declaration contributes, and whether its coefficients span an order
of magnitude the solve will pay for. The thousand-fold spread above is the
fault `diagnostics().coefficient_range` reports per *declaration* and nothing
reported per row. `display_terms` is where a line stops spelling terms out.

It reads the **built** row, which is the whole of its value:

- a coefficient is the number the *data* produced, where the file shows a
  parameter name — every digit of it, since a rendering that rounded would
  agree with the file in exactly the case worth reading;
- a term whose variable was masked out by a `where` is **not there**, so
  a row shorter than the file suggests says so;
- a term whose coefficient the data made **exactly zero** is not there either.
  What a zero states, absence already states, so the build prunes it and the
  row reads the matrix the solver was handed rather than a reconstruction of
  it;
- a row a `where` removed raises rather than answering, and the message names
  the three things that cause it.

It needs no solve — a model too wrong to solve is exactly the one whose rows
need reading — and the coordinate must name **every** dim of the declaration,
since a partial one names a set of rows rather than one. The constraint is
**positional**, so a dimension may be called `name` and still be named in the
coordinate; a label the dimension cannot hold — a string against an integer
dim, a stranger against a declared label set — is refused naming the dim, not
the dtypes.

There is no verb for a *column*: a variable's bounds are `to_yaml()`'s and its
coefficients are the transpose of this, which nothing has asked for yet.

### Re-solving with new numbers

`rebind` puts new data on a model that is already built, so a loop that solves
the same math over and over pays for the YAML, the plan and the build once:

```python
bound = lps.build('sub.yaml', sources)
for capacity in search:
    result = bound.rebind({'cap_hat': capacity}).solve()
    price = result.dual('capacity')
```

| | |
|---|---|
| **it names what changed** | everything else keeps what `build` bound. A parameter, or a dimension index under its own key — a coordinate set grows by handing over a longer table |
| **the answer is the reference build's** | `bound.rebind(x)` solves what `build(model, sources \| x)` solves, always |
| **it never refuses** | there is no capability to query and no shape of data it rejects. What new values can cost is the *fast path*, never the answer |
| **the solver stays loaded where it can** | new bounds, costs and right-hand sides go onto the model the solver already holds, so the matrix is never handed over twice. Whether the next solve also carries on from the *work* the last one did is [`keep=`](#how-much-of-the-session-a-solve-keeps). A rebind that moves a **mask** — a parameter a `where` compares against — renumbers labels, so that model is loaded again and keeps nothing |
| **earlier results keep reading** | a `Result` owns its values and the label frames of the build it answered, so an old answer stays an answer over its own coordinates. Retaining one keeps those frames alive until it is dropped or closed |
| **a rebind that raises releases the model** | the same rule as `build`: half a model would answer the next `solve` with a mixture of two |
| **a name the model does not declare raises** | `DataError` — a rebind that named nothing would silently re-solve the numbers already bound |

For a sweep, a rolling horizon or a myopic pathway, reach for
[`solve_over`](sweeps.md) first: it is this loop written for you. `rebind` is
the primitive underneath, and what you want when the next set of numbers
depends on the last answer. Where the next set depends on *you*,
[Change a model](../interactive.ipynb) is the notebook loop.

### How much of the session a solve keeps

A session holds two things: the solver with the model on it, and the work that
solver did. A rebind keeps the first, so a second solve never hands the matrix
over again. Whether it keeps the second is `keep=`, and the two can only be
dropped in that order — there is no carrying on from a solver that was closed.

```python
result = bound.rebind({'load': load}).solve()
result.kept  # 'solver' — reused, and the work it did discarded

again = bound.rebind({'load': more}).solve(keep='progress')
again.kept  # 'progress' — it carried on from where the last solve got to

baseline = bound.solve(keep='nothing')  # whatever the session held, gone
baseline.kept  # 'nothing'
```

| | What it asks for | Ask for it when |
|---|---|---|
| `keep='nothing'` | the model handed over again, into a solver that has never seen it — `diagnostics().loads` ticks with it | you are **measuring**. The held solver is discarded *before* the load, so cold is structural rather than scrubbed: no basis, no incumbent, no solver-internal state. That is what a benchmark needs, and what comparing two sets of `solver_options` needs so the first run cannot flatter the second |
| `keep='solver'` *(default)* | the hand-off skipped, and a solver asked to run as though the model were new | **until you have measured otherwise.** It gives the solver back the run it would have had on a fresh load, without paying for the load. Every ordinary rebind loop wants this and nothing else |
| `keep='progress'` | that, and the solver left holding what its last run reached | the model is **hard for its solver's preprocessing** *and* consecutive solves differ by a small step — a rolling horizon, a myopic pathway, a search that inches |

**`keep='progress'` swings both ways, and the two ways are far apart.** Over
six rebinds on HiGHS, measured both ways
([#815](https://github.com/fluxopt/lpspec/pull/815)): on a dispatch model,
whose presolve cracks the problem outright, carrying the solver's work cost
**76.6 s against 4.3 s** — an 18× *loss*; on a storage model whose cyclic
recurrence presolve cannot crack, carrying cost **111.2 s against 213.9 s** — a
1.9× *win*. Same procedure, opposite answers, and the downside was an order of
magnitude where the upside was a factor of two. That asymmetry is why it is
opt-in.

**Which one your model wants is measured, not reasoned about.** Run the loop
each way and read the clock the package already keeps; `kept` confirms the
request was honoured rather than quietly downgraded:

```python
for keep in ('solver', 'progress'):
    bound = lps.build('model.yaml', sources)
    for numbers in walk:
        assert bound.rebind(numbers).solve(keep=keep).kept in {keep, 'nothing'}
    print(keep, bound.diagnostics().timings['solve'])
```

Take the faster one. **Nothing about the answer changes either way** — across
both models above the objectives agreed to 2e-15 relative — so this is a timing
question and only a timing question.

`result.kept` is read off what happened, never off what was asked, so a rebind
that had to rebuild reports the `'nothing'` it got rather than the `'progress'`
it hoped for. `'nothing'` every iteration means the session is being rebuilt
away, and `loads` ticks on exactly those solves.

Whether progress is a basis, an incumbent or a solver's own notion stays the
solver's business: this surface says how much was kept, not what it was made
of. It is also not reachable by setting a solver option — on both solvers that
ship, an option asking for the same thing did not produce it
([#815](https://github.com/fluxopt/lpspec/pull/815)).

**Carrying progress across a rebuild is not here yet** — the case that wants it
most, a cutting-plane master re-solved after gaining a cut, is a model that
gained a *row*, and a basis spans the model it was read from.
[#382](https://github.com/fluxopt/lpspec/issues/382) is where that is being
worked out.

### `diagnostics`

What a build and its solves did that the answer does not show. Advisory, all of
it: nothing about an answer depends on any field, and a caller who branches on
one has made this engine's bookkeeping part of their model.

| Field | |
|---|---|
| `columns`, `rows`, `nonzeros` | the shape the build produced — what `check` cannot answer, needing no data where this needs all of it, and where a broadcast that multiplied rows shows up first |
| `sink_columns`, `sink_rows` | what the last solve's solver had to *add* to that shape. Zero unless it had no concept of a set the model declares, in which case this is the binaries and linking rows it was handed instead |
| `omissions` | rows a constraint declared but did not build ([absence](language/absence.md#a-row-with-no-variable-terms-is-not-built)) |
| `sparse_parameters` | `(parameter, coordinates, rows, missing)` — one row per parameter whose source is short of the coordinates its dims reach, empty where every one is complete. Sparsity is how a model masks, so this reports rather than judges: a table that lost a row and a `where:` that removed one build the same model, and nothing else would say which parameters could be either |
| `coefficient_range` | `(constraint, smallest, largest)` — the coefficient **magnitudes** each block put in the matrix. A solver prints one range for the whole model, which says a repair is needed and not where; this says which declaration holds the outlier, and `largest / smallest` over the frame is the conditioning to compare against the solver's own |
| `objective_range` | the same pair for the costs, or `None` where the model declares no objective. Beside the frame rather than in it: badly scaled costs and a badly scaled matrix are different faults with different repairs |
| `solves`, `loads` | how many solves ran, and how many of them had to load the model from scratch. A driver on the fast path leaves `loads` at one however many times it goes round; `loads == solves` is the difference between "lpspec is slow" and "this model masks on a parameter that varies" |
| `timings` | cumulative wall seconds per phase — `bind`, `build`, `handoff`, `solve`, `write` |

It answers after `close()` too: every field is a count, a clock or a small
frame the model keeps rather than a read of what it releases.

## Choosing a solver

**Which solver is a caller's choice, not the file's.** `solver_name` is
`highs` (ships with the package), `gurobi` (the `[gurobi]` extra) or `xpress`
(the `[xpress]` extra), and nothing in the YAML names one — the same file means
the same model whichever takes it. A name outside the three is an error listing
them, never a quiet fallback.

Options travel in the chosen solver's own vocabulary, because forwarding
verbatim is the contract — a time limit is three different words:

```python
lps.solve('model.yaml', sources, solver_options={'time_limit': 60})
lps.solve('model.yaml', sources, solver_name='gurobi', solver_options={'TimeLimit': 60})
lps.solve('model.yaml', sources, solver_name='xpress', solver_options={'timelimit': 60})
```

**Gurobi's remote and licensing options travel the same way**, so Compute
Server, Instant Cloud and WLS need nothing from this package:

```python
options = {'ComputeServer': 'srv:61000', 'ServerPassword': '…'}
lps.solve('model.yaml', sources, solver_name='gurobi', solver_options=options)
```

They are applied when Gurobi's environment is created, which is what
`ComputeServer`, `TokenServer` and `WLSAccessID` require.

## Writing a file instead of solving

```python
lps.write('model.yaml', sources, 'model.lp')
```

The **suffix** picks the writer — `.lp` and `.mps`, anything else a
`ValueError` listing what can be written. It is checked before the build, so a
format nothing can write costs no model.

The two describe one model and name their columns and rows the same way, so a
reader holding both files is reading one thing twice. Which to write is the
reader's, not the model's: LP is the one a person diffs, MPS the one a
decade-old toolchain accepts.

## A model four ways

**Every verb takes the model as a path, a `str`, a `dict`, or a `Model`.**
`check`, `build`, `solve` and `write` share one first argument, so a framework
that emits declarations never writes a temporary file to run them:

```python
model = {'dimensions': ..., 'variables': ..., 'constraints': ..., 'objective': ...}

lps.solve(model, sources)  # a dict runs like a file
checked = lps.load_model(model)  # ...or validate once and keep it
checked.to_yaml()  # the review copy — a dict-built model still gets a file
lps.solve(checked, sources)  # a Model is passed through, not revalidated
```

**This is the supported path for a framework**: a library composing optional
features emits *data*, not YAML text, and never merges files. The last two
lines are the condition rather than a convenience — a generated model that
cannot show you a file is exactly the failure the file exists to prevent.
Hand-written math still starts as a file; nothing here asks it not to.

**A `Model` goes back out two ways, and they agree.** `to_dict()` is the model
as data; `to_yaml()` is that dict as the file you review and diff. Loading,
dumping and loading again is stable for both forms, and dumping twice gives the
same bytes — a review copy that changed per run would be a diff nobody can
read.

**Every value is written; only what is absent is dropped** — a null, an
infinite bound, or a mapping that declares nothing. An infinite bound is absent
because it is not a bound: it is the unbounded side, which is what omitting the
bound already means. An empty **list** stays, because a list carries
cardinality here and zero is one of its values — `foreach: []` is a scalar
declaration.

## The linopy lane

`lpspec.linopy.build` / `.expression` (the `[linopy]` extra) build the same YAML
as a `linopy.Model` instead of binding it relationally, and read a named
expression back off a solved one. It is documented with everything else
about that relationship in [Relationship to linopy](../about/linopy.md#3-it-is-a-lane).
