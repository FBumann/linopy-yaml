# Errors and limits

## Everything decidable without data is decided without data

Anything detectable before building is detected before building. The worst
error this language could hand you is an opaque solver or array exception with
no pointer back to a YAML declaration, so a model is parsed, expanded, resolved
and dim-checked — including *uncalled* macro templates and every `where` string
— before a single source is read.

`lps.check('model.yaml')` runs exactly that and binds nothing, which is why it
is the CI verb: a model repository can be validated on every commit without
shipping the data.

Every message names what went wrong, what to do about it, and where it helps,
the valid options:

```text
Constraint 'balance', equation 0: 'p_charge' not found.
  Variables: ['p', 'soc']
  Parameters: ['p_max', 'load', 'efficiency']
Check for typos, or ensure 'p_charge' is declared.
```

A construct outside the language names the construct and its rewrite, never a
silent fallback.

## Which error you get

| | |
|---|---|
| `LpspecError` | the root of the tree; everything below is an instance of it |
| `LanguageError` | the model: a construct outside the language, a dim set that does not compose, a name nothing declares |
| `SchemaError` | the file: an unknown key, a malformed declaration, a bad symbol table |
| `DimensionError` | dims that disagree — a constraint whose expression does not equal its `foreach` |
| `PiecewiseExpansionError` | a `piecewise:` block that cannot be expanded |
| `DataError` | what was bound: a missing source, an unreadable one, a coordinate outside the master index |

The split is the useful one for a caller: `LanguageError` and its subclasses
are the *file* being wrong, and are reproducible from the YAML alone;
`DataError` is the numbers being wrong for a file that is fine.

`check` also issues an `LpspecWarning` for advice short of an error — a
declared dimension nothing uses as an axis, say. It is the only place warnings
come from.

## One wrong *answer* is decidable without data too

Everything above is the file being wrong. One thing is not: a model that
parses, composes and builds, and has **no optimum to find**. `check` names it
rather than refusing it, because the same shape is what a half-written model
looks like — a variable declared before the constraint that will hold it.

```text
variable 'slack' is unbounded below and appears in no constraint, so the
objective can improve without limit — this model has no optimum to find.
Give it a finite lower bound, constrain it, or drop it from the objective.
Solved as it stands, the answer is a bare 'unbounded' that names nothing.
```

The advice needs all three of: the variable is **in the objective with a
coefficient signable from literals alone**, it is **unbounded on the side that
improves it**, and **no constraint and no set names it**. Each alone is an
ordinary model — a free variable held by a constraint is how a dual is read, a
bounded one in no constraint is how a cost is declared — so only the
conjunction is remarked on.

Signable means signable *everywhere the variable appears*. A coefficient a
**parameter** reaches has no sign until that parameter is bound, and one such
term leaves the whole sum unsigned: `sum(slack + slack * price, over=t)` says
nothing however `price` looks, because the data decides which way `slack`
improves. This runs before any data exists, and a guess here would name the
wrong bound on a model that solves.

A variable carrying a `where` is left alone for the same reason from the other
end: every leg above is fixed by the schema, so what is reported is unbounded
under *any* data that gives the variable a column, and a mask is the one thing
that can leave it with none. The per-coordinate case, where a `where` leaves
one slice of a variable undefined rather than all of them, is not answered here
either — it needs the built rows.

Skip `check` and nothing changes about the model: it builds, and the solver
answers as it always did — `unbounded` for an LP, and `infeasible_or_unbounded`
once one integer variable is in it, which is the answer this advice exists to
get ahead of.

## What the language will not say

Refusals, and what to reach for instead. None of them is an unimplemented
feature list: each is a boundary the design keeps on purpose, and
[the ceiling](../../about/ceiling.md) is the argument for where it sits.

| Not here | Instead |
|---|---|
| variable × variable, or `**` | nothing — degree 1 is the ceiling ([expressions](expressions.md#degree-1-always)) |
| arithmetic in `bounds:` | a name or a number; ship the derived column as data ([#31](https://github.com/fluxopt/lpspec/issues/31)) |
| time-series processing (resample, cluster, interpolate, align), file IO, units | data prep; pass a parameter |
| solver breadth | two solvers — HiGHS, which ships, and Gurobi via the `[gurobi]` extra — chosen at the call and never in the file; LP files for everything else ([#106](https://github.com/fluxopt/lpspec/issues/106)) |
| indicator constraints | planned as a *solver capability* rather than a language question, the same axis `sos:` landed on ([#220](https://github.com/fluxopt/lpspec/issues/220)) |
| multi-objective | one `objective:` block — a second is unsayable; weight them into one expression |
| arbitrary array ops (`merge`, `reindex`, `apply_ufunc`) | data prep — the closed operator set is what makes streaming possible |
| filling a missing value (`.fillna`) | data prep, or a `where` if you meant the coordinate not to exist. In the language only where the data cannot reach: `shift(..., edge=)` ([absence](absence.md)) |
| schema migrations | — |

A model built partly in Python has no readable `.yaml` representation and will
not get one: the *math* side is feasible, but expression and `where` strings
come back as anonymous arrays, so the round trip would be functional and not
reviewable — which is the whole point of the file. A framework that wants to
*emit* declarations passes a dict, and gets `to_yaml()` back
([Python API](../api.md#a-model-four-ways)).

Where the language genuinely cannot say the math, the escape hatch is a
declared `escape:` island — named in the file, bounded by the preceding `where`
mask, terminal, and billed against a label budget before any Python runs. It is
[#38](https://github.com/fluxopt/lpspec/issues/38) and not shipped.
