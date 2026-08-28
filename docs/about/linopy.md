# Relationship to linopy

Everything about [linopy](https://github.com/PyPSA/linopy) in one place, because
it is otherwise the kind of thing that gets mentioned everywhere and explained
nowhere. Three separate relationships, and conflating them is what made the rest
of the docs noisy:

| | What | Where it matters |
|---|---|---|
| **Not a dependency** | solving a model never imports it | packaging |
| **The oracle** | how we know the answers are right | testing |
| **The lane** | the second thing a file can be built as | what a caller chooses |

## 1. It is not a runtime dependency

`lps.solve`, `lps.build`, `lps.write` and `lps.check` go YAML → polars → HiGHS
or file, and import nothing from linopy, xarray or pandas. CI proves it: the
bare-install job runs the whole suite with none of them present.

`pip install "lpspec[linopy]"` adds linopy, xarray and pandas, which buys two
things and nothing else — the lane below, and the `to_pandas` /
`to_dataarray` bridges out of a result. The lane is a peer, not a fallback:
nothing routes to it, and a bare install is a complete one.

**Nothing a bare install can reach names linopy, including in a traceback.** The
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
[docs/examples/index.md](../examples/index.md) is where those live.

**Where a concept is already linopy's, we copy its name** — solve statuses,
`status` / `termination_condition` as two axes with `is_ok` as the rollup, the
shape of a result. Our audience arrives from linopy and PyPSA, and a second
vocabulary for one fact is a tax on all of them. But **copy it, do not import
it**: the engine may not import linopy, so the tables live here and a test
imports linopy to assert the copy still matches. A copy nobody checks is a copy
that rots.

## 3. It is a lane

The same file, built as a `linopy.Model` instead of bound relationally — the
caller picks the lane by an import, and the call is the one `lps.build` takes:
same first argument (a path, a mapping or a loaded `Spec`), same `sources`,
same index sources.

```python
from lpspec import linopy as lpspec_linopy

m = lpspec_linopy.build('model.yaml', {...})  # -> linopy.Model
m.solve(...)
lpspec_linopy.expression(m, 'model.yaml', 'co2', {...})  # a named quantity, read back
```

Both are *pure*: YAML in, a model or a value out, nothing retained. `build`
returns a plain `linopy.Model` — no accessor, no attached schema, no patched
attributes — so nothing is lost across `pickle`, `deepcopy` or `to_netcdf`. To
inspect the math, re-read the file with `to_spec`.
`expression` is the reader the same purity forces to take `sources` again: it
evaluates a declared named expression ([named expressions](https://math-spec.readthedocs.io/en/latest/reference/language/expressions/#named-expressions))
on the solved model and hands back linopy's native `.solution` — the eager
half of `result.expression(name)`, so the differential suite can hold the two
lanes to one answer.

**This lane constructs; it does not attach.** Math for a `linopy.Model`
something else built — a PyPSA network, say — had a verb here and no longer
does ([#845](https://github.com/fluxopt/lpspec/issues/845)): it was the one
file allowed to reference names it did not declare, and paying for that
exception across the whole language layer bought one use case. Build a second
model and merge it.

### What a construct becomes

The whole translation, in one place — what `lpspec.linopy.build` calls for each
thing a file can say. `linopy/builder.py` is where each row lives, one section
per group below.

| Declaration | linopy |
|---|---|
| `variables:` | `Model.add_variables(lower, upper, coords, name, mask, binary, integer)` |
| `sos:` | `Model.add_sos_constraints(variable, sos_type, sos_dim, big_m)` — the block handed over, not a formulation rebuilt |
| `constraints:` | `Model.add_constraints(lhs, sign, rhs, name, mask)`, one rule per declaration |
| `objective:` | `Model.add_objective(expr, sense)`, each additive term summed over the dims it carries |
| `expressions:` | evaluated on the solved model, and linopy's own `.solution` handed back |

| In an expression | linopy or xarray |
|---|---|
| `x` — a variable | `Model.variables['x']`, `.fillna(0)` under `absence: zero` |
| `p` — a parameter | its `xr.DataArray`, `.fillna(0.0)` where it stands as a coefficient |
| `+` `-` `*` `/` | the Python operators linopy overloads |
| `sum(x, over=t)` | `.sum('t')` |
| `sum(x, by=lk)` | the lookup attached as a coordinate, then `.groupby()`, reindexed onto the target dimension's declared labels — one key per lookup, so `by=[lk1, lk2]` groups by both at once |
| `at(p, by=lk)` | `.sel({into: lookup})` — xarray's vectorised selection *is* the pullback, and one entry per lookup reads a tuple of labels at once |
| `shift(x, over=t, offset=n)` | `.shift({t: n})`; `.roll({t: n})` under `edge: wrap`; a `.sel()` gather where the offset differs per entity or `by=` groups it |
| `sum_back(x, over=t, within=w)` | a sum of `w` scalar gathers, each unreachable position contributing zero; under `by=` each gather reads inside the group, so the window stops at its edge |

| A `where:` | linopy |
|---|---|
| on a declaration | the `mask=` argument — a mask that excludes nothing is passed as `None` |
| `defined(x)` | `Model.variables['x'].labels != -1`, linopy's own marker for an absent slot |
| a comparison | the Python comparison operators element-wise, absence reading as false |

Absence is the one thing with no single row: it is positional, so a missing
parameter row is zero in a coefficient, an error in `bounds:`, and false in a
`where` operand. `linopy/absence.py` holds all four spellings together, and the
builder calls them qualified — `absence.coefficient(...)` — so a reader meets
the name at the call rather than only at the definition.

### The same language, and the same data

The lane accepts **exactly the same language** — that equality is what makes the
oracle an oracle, and it is now structural: both run the same `to_program`
gate, so a construct one refuses the other refuses in the same sentence, never
with a redirection to the other lane.

**Accepting is not building, and two constructs part them — one in each
direction.** Neither is a language limit: both files pass `check`, and each is
built by the lane the other cannot.

**The first is this lane's: an objective carrying a constant.** `linopy.Objective`'s expression setter rejects any
expression whose `const` is nonzero — *"Constant values in objective function
not supported."* — and there is no slot to put one in, which is why PyPSA
carries `n.objective_constant` out of band. So a model like
`examples/ports/osemosys_utopia.yaml`, whose objective owes a fixed cost on
capacity that already stood in 1990, builds relationally and raises linopy's
`ValueError` on this lane. **Dropping the constant is the one repair that must
not happen**: the lane is the oracle, so a quietly shortened objective would
recalibrate every differential test on such a model to the wrong number.
Adding it back as a variable pinned to `[1, 1]` reaches the right answer and
was refused too — it puts a column on the caller's model that the other lane
does not have.
So the lane says it in its own words: `builder.py` checks for a constant before
linopy is asked and raises `LaneError`, naming the wall and the route that does
build the model. `tests/test_corpus_parity.py` carries the strict xfail, typed
to that error rather than to any `ValueError`, so the day linopy grows a slot it
XPASSes and the check comes out with it
([#894](https://github.com/fluxopt/lpspec/issues/894)).

**The second is the relational lane's, and it is the mirror: an operator acting
along a dimension a constant part does not carry**, beside a term that does —
say `sum(x * k + d, over=t)` where `d` is a scalar. That lane compiles a
constant part as its own frame, so a fragment with no rows for `t` has no slots
for the operator to act on, and under a mask which slots those are is known
only to the rows. This lane has no such split — the operand is one masked
expression, so the constant is dropped wherever the term is — and so it builds
the file as written.

It is **one wall, reached by all four operators that act along a dimension**
(`sum(over=)`, `sum(by=)`, `shift`, `sum_back`), which is why they share a
refusal rather than each wording its own: a fix for one that left the others
would be a fix for a symptom. The relational lane names the rewrite that
reaches the same number — declare the parameter over the dimension and supply
it there ([#1137](https://github.com/fluxopt/lpspec/issues/1137)).

Finding that wall is what turned up a real disagreement behind it: `sum_back`
read a constant at a slot the variable was absent from, where every other
operator drops it, so the two lanes answered 2.5 and 3.0 on a file **neither**
refused. A reduction consumes its operand before any row exists, so absence has
to be pushed into the operand first — `sum` and `sum(by=)` did that and the
window did not. Fixed by giving the window the same pass, with a differential
test over every operator that moves along a dimension
([#1142](https://github.com/fluxopt/lpspec/issues/1142)).

Both are worth reading twice, because the shape is easy to mistake for a
language limit and is not one: a `LaneError` names the wall *and* the route
around it, which is the difference between the two classes.

It takes the same *data* too, which it did not always
([#60](https://github.com/fluxopt/lpspec/issues/60)). A parameter is a parquet
path, any table exporting the Arrow PyCapsule protocol, a `pd.Series` carrying
its dims in an index, a `dict` or a sequence over one dimension, or one number
spread over the coordinates it covers. Neither reads an `xr.DataArray`: this
package reads tables and hands arrays back. A dimension index is any of those
tables too, under the dimension's own key in `sources`, and labels come from
`sources` or from what the file declares — exactly one of the two, since a dimension the file declares and the caller also
supplies is refused by both lanes in the same sentence. A dimension with none of
the three has no index, and is refused in the same sentence again rather than
derived from the parameters that span it: a parameter carries a label, never the
set of labels that exist, nor what a label maps to. The index is also what fixes
the *order*, so pass one wherever order matters.

So one `sources` mapping goes to either, and which lane builds a file is
decided by an import and nothing else.

## What we deliberately do not take

Array operations (`merge`, `reindex`, `stack`), the Python modeling API, and the
solver layer. The first is data prep
([the limits](https://math-spec.readthedocs.io/en/latest/reference/language/errors/#what-the-language-will-not-say)), the second is
[hard rule 5](architecture.md#hard-rules) — the model is the file you review
and diff — and the third is
[#106](https://github.com/fluxopt/lpspec/issues/106), where we adopt linopy's
*design* for declared solver capabilities without adopting its code.

The modeling API is the one a reader arriving from linopy misses first, and
what replaces it is two notebook pages: [Change a model](../interactive.ipynb)
for the loops — `rebind` for new numbers, a longer table for more rows, a
patched `dict` for new math — and [Fix, relax, remove](../lifecycle.ipynb) for
the verbs, which are the same loops aimed at `fix`, `relax` and
`remove_constraints`. What neither replaces is the *debugging*: an IIS. Both
pages say so — a built row is read with
[`row`](../reference/api.md#reading-one-row), in linopy's own form.

Where linopy is genuinely ahead, and why none of it is a ceiling question, is the
honest snapshot in [the roadmap](roadmap.md#honest-snapshot).

What is *owed* to linopy rather than merely true of it — and the same for
Calliope, whose math language this surface is derived from — is
[prior art and credit](prior-art.md).
