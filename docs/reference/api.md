# Python API

How you *run* a spec. The spec itself is the YAML file — what it may contain
is [the language](https://math-spec.readthedocs.io/en/latest/reference/language/); this page is what loads, checks, builds,
solves and reads one back.

```python
import lpspec as lps

lps.check('spec.yaml')  # compiles? no data needed

result = lps.solve('spec.yaml', sources)
result.objective
result.primal('p')  # a polars.DataFrame
result.dual('power_balance')
```

## The verbs

| | |
|---|---|
| `lps.check(spec, sink=None)` | parse, expand, validate and lower; attach no data. With a `sink`, also whether that sink will take it. Returns the lowered `Program`, which every verb here takes back |
| `math_spec.to_spec(spec)` | the file as written, for editing and typesetting it — the language's own verb, from the package that owns it |
| `lps.build(spec, sources)` | attach data and build it — returns a `Model` |
| `lps.solve(spec, sources, solver_name='highs', solver_options=None)` | build and solve in one call — returns a `Result` |
| `lps.solve_over(spec, sources, axis, ...)` | solve once per slice and fold the answers — [sweeps](sweeps.md) |
| `lps.write(spec, sources, out)` | build and stream to a file; the suffix picks the format |
| `model.row(name, **coordinate)` | what one built constraint row says — terms, comparison, right-hand side |
| `math_spec.to_latex` / `to_typst` / `to_markdown` | the math as a document — [typeset](https://math-spec.readthedocs.io/en/latest/reference/typeset/) |
| `lps.Model` / `lps.Result` / `lps.Runs` | the types the verbs hand back, importable — a wrapper annotates its own signature with them rather than reaching a submodule for the name. The model going *in* is the language's: `math_spec.Spec` or `math_spec.program.Program`, from the package a caller already called to get one |

Errors are one tree: `LpspecError` at the root, `LanguageError` (with
`SchemaError`, `DimensionError`, `PiecewiseExpansionError`) for the spec,
`DataError` for what was bound to it, `LaneError` for a spec one lane cannot
build, and `NoSolutionError` for a solve that left nothing to read
([errors](https://math-spec.readthedocs.io/en/latest/reference/language/errors/#which-error-you-get)).
`LpspecWarning` is the one warning category — `check`'s advice — so
`warnings.simplefilter('error', lps.LpspecWarning)` is how a spec repository
fails CI on it.

**`check` is the CI verb.** It parses, expands, resolves and lowers without
attaching anything, so a spec repository can be validated on every commit
without shipping the data.

### `sink=`, the second question

Whether a spec is *sayable* is solver-independent. Where it can *land* is a
separate axis — [what a sink can
ingest](https://math-spec.readthedocs.io/en/latest/about/ceiling/#capability-is-not-the-ceiling) — and `sink=` is how
you ask about it:

```python
lps.check('spec.yaml')  # sayable?
lps.check('spec.yaml', sink='highs')  # ...and will HiGHS take it?
lps.check('spec.yaml', sink='.lp')  # ...will the LP writer?
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
  one that exists: it arrives as binaries, so a spec that declared no
  integrality comes back mixed-integer and without duals — better read before
  the solve than inferred from an empty `dual()`.

Answered off a declared table with **no data and no installed solver**, so
`check(m, sink='gurobi')` answers on a machine that has never had gurobipy.

#### What each sink takes

Measured against the shipped solvers rather than assumed. The four quadratic
rows and the two sections HiGHS writes but will not read back are *probed* —
`tests/test_sink_capability_probes.py` and
`tests/test_gurobi_capability_probes.py`, each assertion naming this table —
because a capability moves on somebody else's release and a stale row would go
wrong with the suite green. The rest are read off the APIs.

| | `lp_file` | `mps_file` | HiGHS direct | Gurobi direct | Xpress direct |
|---|---|---|---|---|---|
| affine rows, COO, integrality | text | text, `MARKER` | native | native | native |
| semi-continuous | text | **not written** — no `SC` bound | `kSemiContinuous` | native | native |
| SOS1 / SOS2 | text section | `SOS` section | **no concept** — rewritten to binaries | `addSOS` | native |
| indicator | text section | **not written** | **no concept** | `addGenConstrIndicator` | native |
| convex quadratic objective | text section | **not written** | `passHessian` | `setMObjective` | **no path here** |
| nonconvex quadratic objective | text section | **not written** | **refused** | native, at default parameters | **no path here** |
| quadratic objective **and** integrality | text section | **not written** | **refused** | native (MIQP) | **no path here** |
| quadratic constraint | text section, unreadable | **not written** | **no concept** | `addQConstr` | **no path here** |

Three readings worth having:

- **HiGHS excludes quadratic twice** — by convexity, and by conjunction with
  integrality. Neither is a set membership, which is why a flat capability set
  gets it wrong: linopy declares HiGHS with `INTEGER_VARIABLES` and
  `QUADRATIC_OBJECTIVE` together, so its own model reports MIQP as available.
- **The `lp_file` column says what can be written, not what reads back.** The
  same HiGHS parser takes the quadratic-objective section and refuses both the
  `sos` and the quadratic-constraint one.
- **"No path here" is about this tree, not about Xpress.** The Optimizer takes
  a Hessian; the sink in `solvers/xpress.py` never hands it one, and a
  descriptor says what the sink ingests rather than what the library could.

**A rewrite is not free, and the cost is what comes back.** A model carrying a
set returns from HiGHS without duals — the reformulation makes it a MIP — and
from Gurobi with them. That asymmetry is the argument for declaring capability
rather than papering over it, and it is why `sink=` warns rather than staying
silent.

Asking is optional; being refused is not. `solve` and `write` read the same
table when they get there, so `lps.write(m, sources, 'model.mps')` on a model
carrying a quadratic term is refused by name rather than handed back as a file
whose quadratic rows are missing — which would parse, solve, and answer for a
different model. What `sink=` buys is the same sentence before the build.

## Sources

`sources` maps declared names to data: parquet paths, or any table exposing the
Arrow PyCapsule protocol — polars, pandas, pyarrow. A dimension's own key supplies
dimension labels that neither the sources nor the YAML carries. The exact rules
are [the data contract](data.md).

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
| **`expression` takes a declared name** | the value of a [named expression](https://math-spec.readthedocs.io/en/latest/reference/language/expressions/#named-expressions) at the solution, aggregated to its own dims. Never an expression string; an unknown name is a `KeyError` listing what is declared. It is compiled at the read, so a build with fifty declared expressions that reads none pays for none |
| `dual` **raises rather than zero-filling** | no values at all is `NoSolutionError`; values but no duals — any integer or binary variable makes them undefined — is `LpspecError`, because only this quantity is missing |
| **a solver can make a model mixed-integer** | an [`sos:`](https://math-spec.readthedocs.io/en/latest/reference/language/piecewise/#sos) set reaches a solver with no SOS concept as binaries, so an otherwise continuous model solved on `highs` has no duals and says so. On `gurobi` and `xpress`, which branch on the set itself, it keeps them |
| duals exist only where a solver ran | a model written to LP and solved elsewhere never passes back through here. Reduced costs and slacks are not exposed yet |
| `to_dataset` costs what it says | each variable arrives dense over its own dims — name a subset, or use `to_parquet` |

**Nothing has to be released.** The built model is frames this process owns, so
`primal` and the `to_*` readers stay valid for as long as the `Result` does.
`close()` and the context-manager protocol exist to hand a large model back
early, not because forgetting them breaks anything.

## Building once, solving many times

`lps.build` returns a `Model` — the math with your data on it — for when
one build should feed more than one sink, or be solved more than once:

```python
model = lps.build('spec.yaml', sources)
model.write('model.lp')
result = model.solve()
model.diagnostics()  # what the build and its solves did that the answer does not show
model.row('balance', snapshot=17)  # what one row actually says
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
print(model.row('balance', snapshot=1))
# balance[snapshot=1]: +1 p[1, wind] +50 p[1, gas] +30 p[1, coal] >= 60
```

That line is **linopy's**, on purpose — their `Constraint.print()` renders a row
the same way, and a reader arriving from there should not have to learn a
second way to read a constraint. What is added is the row's own identity on the
same line, where linopy prints it as a header.

The same content is a frame, for the row too wide to read and for anything that
filters or joins:

```python
row = model.row('balance', snapshot=17)
row.terms  # (variable, coordinate, coefficient), one row per term
row.sense  # '=='
row.rhs  # 80.0
```

A row too wide to spell out **summarises rather than truncating** — twelve
terms of three hundred are twelve arbitrary ones:

```python
print(model.row('balance', t=0))
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

`update` puts new data on a model that is already built, so a loop that solves
the same math over and over pays for the YAML, the plan and the build once:

```python
model = lps.build('sub.yaml', sources)
for capacity in search:
    result = model.update({'cap_hat': capacity}).solve()
    price = result.dual('capacity')
```

| | |
|---|---|
| **it names what changed** | everything else keeps what `build` bound. A parameter, or a dimension index under its own key — a coordinate set grows by handing over a longer table |
| **the answer is the reference build's** | `model.update(x)` solves what `build(spec, sources \| x)` solves, always |
| **it never refuses** | there is no capability to query and no shape of data it rejects. What new values can cost is the *fast path*, never the answer |
| **the solver stays loaded where it can** | new bounds, costs and right-hand sides go onto the model the solver already holds, so the matrix is never handed over twice. Whether the next solve also carries on from the *work* the last one did is [`keep=`](#how-much-of-the-session-a-solve-keeps). An update that moves a **mask** — a parameter a `where` compares against — renumbers labels, so that model is loaded again and keeps nothing |
| **earlier results keep reading** | a `Result` owns its values and the label frames of the build it answered, so an old answer stays an answer over its own coordinates. Retaining one keeps those frames alive until it is dropped or closed |
| **an update that raises releases the model** | the same rule as `build`: half a model would answer the next `solve` with a mixture of two |
| **a name the spec does not declare raises** | `DataError` — an update that named nothing would silently re-solve the numbers already bound |

For a sweep, a rolling horizon or a myopic pathway, reach for
[`solve_over`](sweeps.md) first: it is this loop written for you. `update` is
the primitive underneath, and what you want when the next set of numbers
depends on the last answer. Where the next set depends on *you*,
[Change a model](../interactive.ipynb) is the notebook loop.

### How much of the session a solve keeps

A session holds two things: the solver with the model on it, and the work that
solver did. An update keeps the first, so a second solve never hands the matrix
over again. Whether it keeps the second is `keep=`, and the two can only be
dropped in that order — there is no carrying on from a solver that was closed.

```python
result = model.update({'load': load}).solve()
result.kept  # 'solver' — reused, and the work it did discarded

again = model.update({'load': more}).solve(keep='progress')
again.kept  # 'progress' — it carried on from where the last solve got to

baseline = model.solve(keep='nothing')  # whatever the session held, gone
baseline.kept  # 'nothing'
```

| | What it asks for | Ask for it when |
|---|---|---|
| `keep='nothing'` | the model handed over again, into a solver that has never seen it — `diagnostics().loads` ticks with it | you are **measuring**. The held solver is discarded *before* the load, so cold is structural rather than scrubbed: no basis, no incumbent, no solver-internal state. That is what a benchmark needs, and what comparing two sets of `solver_options` needs so the first run cannot flatter the second |
| `keep='solver'` *(default)* | the hand-off skipped, and a solver asked to run as though the model were new | **until you have measured otherwise.** It gives the solver back the run it would have had on a fresh load, without paying for the load. Every ordinary update loop wants this and nothing else |
| `keep='progress'` | that, and the solver left holding what its last run reached | the model is **hard for its solver's preprocessing** *and* consecutive solves differ by a small step — a rolling horizon, a myopic pathway, a search that inches |

**`keep='progress'` swings both ways, and the two ways are far apart.** Over
six updates on HiGHS, measured both ways
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
    model = lps.build('spec.yaml', sources)
    for numbers in walk:
        assert model.update(numbers).solve(keep=keep).kept in {keep, 'nothing'}
    print(keep, model.diagnostics().timings['solve'])
```

Take the faster one. **Nothing about the answer changes either way** — across
both models above the objectives agreed to 2e-15 relative — so this is a timing
question and only a timing question.

`result.kept` is read off what happened, never off what was asked, so an update
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
| `sink_columns`, `sink_rows` | what the last solve's solver had to *add* to that shape. Zero unless it had no concept of a set the spec declares, in which case this is the binaries and linking rows it was handed instead |
| `omissions` | rows a constraint declared but did not build ([absence](https://math-spec.readthedocs.io/en/latest/reference/language/absence/#a-row-with-no-variable-terms-is-not-built)) |
| `sparse_parameters` | `(parameter, coordinates, rows, missing)` — one row per parameter whose source is short of the coordinates its dims reach, empty where every one is complete. Sparsity is how a model masks, so this reports rather than judges: a table that lost a row and a `where:` that removed one build the same model, and nothing else would say which parameters could be either |
| `coefficient_range` | `(constraint, smallest, largest)` — the coefficient **magnitudes** each block put in the matrix. A solver prints one range for the whole model, which says a repair is needed and not where; this says which declaration holds the outlier, and `largest / smallest` over the frame is the conditioning to compare against the solver's own |
| `bound_range` | `(variable, smallest, largest)` — the **bound** magnitudes each variable block put on its columns. The axis a solver reports and does not repair: HiGHS equilibrates the matrix by itself and answers the bounds with `Consider scaling the bounds by …`, so a model can be clean on `coefficient_range` and still be the one it is complaining about. Zero and infinity are excluded — a `lower: 0` and an unbounded side are nothing the solver represents — which is also what makes the pair comparable with the line it prints. A large `largest` is usually a big number standing in for "uncapped", and wants no upper bound rather than a rounder one |
| `rhs_range` | `(constraint, smallest, largest)` — the same for each block's right-hand sides, over the rows that survived. The fourth of the four ranges a solver prints, and the last of them answerable per declaration |
| `objective_range` | the same pair for the costs, or `None` where the spec declares no objective. Beside the frame rather than in it: badly scaled costs and a badly scaled matrix are different faults with different repairs |
| `solves`, `loads` | how many solves ran, and how many of them had to load the model from scratch. A driver on the fast path leaves `loads` at one however many times it goes round; `loads == solves` is the difference between "lpspec is slow" and "this model masks on a parameter that varies" |
| `timings` | cumulative wall seconds per phase — `attach`, `build`, `handoff`, `solve`, `write` |

It answers after `close()` too: every field is a count, a clock or a small
frame the model keeps rather than a read of what it releases.

## Tracing the feasible region

`lps.project` draws what a model *can* do on two quantities you name — the
question a modeller asks of a plant before asking what it should do.
[See the feasible region](../region.ipynb) is the notebook walk, hour by hour
and state by state:

```python
region = lps.project('plant.yaml', sources, x='heat', y='power', at={'t': 5})
region.vertices  # (heat, power), one row per vertex, counter-clockwise
region.plot()  # filled on a matplotlib axes — the [plot] extra
```

A plant with on/off states is a union of regions rather than one, and the
states are what a modeller wants to see apart:

```python
region = lps.project('plant.yaml', sources, x='heat', y='power', at={'t': 5}, binaries='each')
region.pieces  # one Region per feasible combination of the units' states at t=5
ax = region.plot()  # each piece in its own colour, labelled `on[t=5, unit=chp]=1, …`
ax.legend()
region.pieces[0].plot()  # one state on its own, under the same label
region.to_frame()  # every vertex of every piece: (on[t=5, unit=chp], …, vertex, heat, power)
```

`to_frame` is the long form: one row per vertex, the pinned columns as
columns, a `vertex` index in polygon order, then the two quantities — what a
sweep over hours concatenates, and what a plotting library other than
matplotlib takes.

The picture is one call, and what it should say beyond the region is a call on
the axes it returns — the optimum on it, say, which `project` does not solve for
because the region does not depend on it:

```python
ax = region.plot(label='what the plant can do')
with lps.solve('plant.yaml', sources) as result:
    ax.scatter(result.expression('heat_at_5').item(), result.expression('power_at_5').item(), label='the optimum')
ax.legend()
```

`x` and `y` are a declared variable or named expression each. `at` fixes
coordinates, and every dim it leaves free is summed: `heat` over `(t, unit)`
with `at={'t': 5}` is the plant's heat in hour five, over all its units; with
no `at` it is the whole horizon's. A dim in `at` that a quantity does not carry
is refused, because multiplying a selection into a quantity that lacks its dim
broadcasts rather than selects.

**The objective plays no part.** The file's is set aside and each solve is
driven by a direction instead: maximise `x` and `y` weighted by that direction,
and the optimum is the vertex it points at. Four compass directions enclose the
region; from there every edge of the polygon so far is probed along its
outward normal, a solve that reaches beyond the edge is a new vertex, and one
that does not settles it. The trace ends when every edge is settled, which is
what makes the polygon **exact** for a continuous model rather than a sample of
it.

Between solves only two costs change, so the model stays on the solver and
each solve carries on from the last vertex: `diagnostics().loads` stays at one
however many vertices the region has.

| | |
|---|---|
| **binaries make it a union, and `free` gives the hull** | each solve still returns an extreme point, so with the binaries free the polygon is the convex hull of the union of what each combination allows; what it encloses may have holes it cannot show |
| `binaries='each'` **traces every combination** | every binary column `at` reaches is pinned to each of its values in turn, and the region each combination leaves comes back as a `Region` of its own under `region.pieces` — `fixed`, the pinned columns and their values, its own `vertices` and its own `plot`. `region.vertices` is then the hull of the pieces. An infeasible combination is left out; a model with no binary, or with more columns at `at` than a trace of every combination can afford, is refused, and `at` is how to ask about fewer. A pin is two rows whose right-hand sides are data, so a combination is a push onto the loaded solver rather than a rebuild. An `integer` variable is never pinned |
| `NoSolutionError` | the model is infeasible, so there is no region |
| **unbounded is a finding, not a picture** | the error names the direction nothing caps — `(+1·heat, +0·power)` — which is the variable missing its bound |
| **`plot` needs matplotlib** | the `[plot]` extra, like `to_pandas` needs `[linopy]`; the vertices need nothing added |
| **it takes the file, not a `Program`** | the probe is ordinary declarations added to the spec — two weights, two expressions, a selection parameter per axis — so it needs the spec as written; `check`'s output has already been lowered |
| a name the probe adds | `x_axis`, `y_axis`, `x_direction`, `y_direction`, `x_selection`, `y_selection` — a spec already declaring one is refused rather than quietly overridden |

## Choosing a solver

**Which solver is a caller's choice, not the file's.** `solver_name` is
`highs` (ships with the package), `gurobi` (the `[gurobi]` extra) or `xpress`
(the `[xpress]` extra), and nothing in the YAML names one — the same file means
the same model whichever takes it. A name outside the three is an error listing
them, never a quiet fallback.

Options travel in the chosen solver's own vocabulary, because forwarding
verbatim is the contract — a time limit is three different words:

```python
lps.solve('spec.yaml', sources, solver_options={'time_limit': 60})
lps.solve('spec.yaml', sources, solver_name='gurobi', solver_options={'TimeLimit': 60})
lps.solve('spec.yaml', sources, solver_name='xpress', solver_options={'timelimit': 60})
```

**Gurobi's remote and licensing options travel the same way**, so Compute
Server, Instant Cloud and WLS need nothing from this package:

```python
options = {'ComputeServer': 'srv:61000', 'ServerPassword': '…'}
lps.solve('spec.yaml', sources, solver_name='gurobi', solver_options=options)
```

They are applied when Gurobi's environment is created, which is what
`ComputeServer`, `TokenServer` and `WLSAccessID` require.

## Writing a file instead of solving

```python
lps.write('spec.yaml', sources, 'model.lp')
```

The **suffix** picks the writer — `.lp` and `.mps`, anything else a
`ValueError` listing what can be written. It is checked before the build, so a
format nothing can write costs no model.

The two describe one model and name their columns and rows the same way, so a
reader holding both files is reading one thing twice. Which to write is the
reader's, not the model's: LP is the one a person diffs, MPS the one a
decade-old toolchain accepts.

## A spec four ways

**Every verb takes the spec as a path, a `str`, a `dict`, a `Spec` or a
`Program` — exactly what `math_spec.to_program` takes, because that is who
opens it.** `check`, `build`, `solve`, `write`, `solve_over`, `Model` and
both linopy-lane verbs share one first argument, so a framework that emits
declarations never writes a temporary file to run them:

```python
spec = {'dimensions': ..., 'variables': ..., 'constraints': ..., 'objective': ...}

lps.solve(spec, sources)  # a dict runs like a file
checked = lps.check(spec)  # ...or lower once and keep the plan
lps.solve(checked, sources)  # a Program is passed through, not re-lowered

to_spec(spec).to_yaml()  # the review copy — a dict-built spec still gets a file
```

**This is the supported path for a framework**: a library composing optional
features emits *data*, not YAML text, and never merges files. The last line is
the condition rather than a convenience — a generated spec that cannot show
you a file is exactly the failure the file exists to prevent. Hand-written math
still starts as a file; nothing here asks it not to.

**A `Spec` goes back out two ways, and they agree.** `to_dict()` is the spec
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
as a `linopy.Model` instead of attaching it relationally, and read a named
expression back off a solved one. It is documented with everything else
about that relationship in [Relationship to linopy](../about/linopy.md#3-it-is-a-lane).
