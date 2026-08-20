# Piecewise curves and SOS

Two blocks for the shapes a purely affine language cannot state directly: a
curve through breakpoints, and a family of variables of which only one — or two
neighbours — may be nonzero.

## `piecewise`

N expressions jointly pinned to a breakpoint-indexed piecewise-linear curve.

<!-- doctest: wrap=piecewise -->
```yaml
chp:
  over: bp  # breakpoint dimension
  foreach: [snapshot, generator]  # where the weights live
  links:  # each under a name the emitted row takes
    power: {expression: power, values: power_bp}
    fuel: {expression: fuel, values: fuel_bp}
    heat: {expression: heat, values: heat_bp}
  method: adjacency  # how the weights are restricted — below
  active: null  # optional gating expression: formulation pinned to 0

# a two-link block may bound one side instead of pinning it
fuel_cap:
  over: bp
  foreach: [snapshot, generator]
  links:
    power: {expression: power, values: power_bp}
    fuel: {expression: fuel, values: fuel_bp, sign: "<="}
```

| Part of a link | |
|---|---|
| *expression* | any affine expression — a bare variable name being the simplest |
| *values* | a parameter carrying the `over` dim, and any dims the link *expressions* carry — so curves may vary per generator where the links do, and a dim they do not carry is refused |
| *sign* | `<=` or `>=`, at most one per block and only with exactly two links: bounds the link instead of pinning it |

`points:` says how far each curve runs where they are not all the same length —
below. `active:` is a different question again: whether a curve *applies*, gated
by a variable, rather than how long it is.

A block **expands before building** into plain variables and constraints, for
three of the four methods via a λ convex combination — weights in `[0,1]` with
a convexity row, and one link row per tuple. That expansion is what the rest of
the model, and the [typeset output](../typeset.md), sees.

**A curve is supplied everywhere it is built.** The expansion emits one weight
per breakpoint over the whole product of its dims and masks none of them, so a
values parameter short of a row does not build a shorter curve: the
[absence rules](absence.md#what-creates-absence) read the missing row as a zero
coefficient, which is a breakpoint at the origin the file never declared. Such a
table is refused when data binds.

**The breakpoint order is `over`'s index order**, the one every dimension has:
the order its labels are first written in, which `shift` walks and
`index(bp, 0)` names. So the `bp` index is the curve's x-axis, and a values
parameter is a lookup against it — a table is a function of its coordinates and
the order its rows arrive in means nothing, on either lane. "Strictly
increasing breakpoints" below is increasing *in that order*: write the index
backwards and the curve really does run backwards, which is refused.

**A curve with fewer breakpoints than the dimension holds says how far it
runs**, with `points:`. Name one of the block's own values parameters and the
curve is as long as its rows:

<!-- doctest: wrap=piecewise -->
```yaml
cost_curve:
  over: bp
  foreach: [snapshot, generator]
  points: bp_x  # this curve runs as far as its own breakpoints do
  links:
    dispatch: {expression: p, values: bp_x}
    cost: {expression: op_cost, values: bp_y}
```

A length is a fact of the curve, so this keeps it there — and the other links
are still read against the one named, so a row missing from `bp_y` is refused.
Name a **boolean parameter** instead where the length is its own data, which is
a different question: not *how long the curve is* but *how much of it to use*.

The breakpoint left out declares no weight and no segment binary, and its values
are not asked for. **The marked breakpoints must be consecutive**, though they
need not start at the head of the axis: a curve numbered from 1 is the same
curve one label along. A gap, or a curve with no points at all, is refused when
data binds — the chord row joins a breakpoint to the one before it, and the two
domain rows sit on the curve's own first and last.

Where the *arity* is data, and one component ties three expressions where
another ties two, the λ formulation is written out directly rather than through
this block ([#1101](https://github.com/fluxopt/lpspec/issues/1101)).

### A link below the weights, for a curve whose arity is data

A link's rows sit on the frame the weights live on — one per generator, one per
converter. A link whose expression carries a **finer** dimension sits below it,
and `by:` is the lookup that carries each row up to the weights it reads:

<!-- doctest: wrap=piecewise -->
```yaml
conversion:
  over: bp
  foreach: [converter, time]
  points: bp_present
  method: adjacency
  links:
    fuel_in: {expression: fuel, values: bp_fuel}   # per converter — on the weights' frame
    on_the_curve:                                  # per flow — one row each
      expression: rate
      values: bp_rate
      by: converter_of                             # and this is how they reach the weights
```

The shape it is about: the block makes one set of weights per coordinate of
`foreach:`, and every link is rows that read a set. Which set is the only
question.

```
one set of weights ─────►  λ[boiler, t]            λ[chp, t]

fuel_in:                   fuel[boiler, t]         fuel[chp, t]
  one row per set

on_the_curve:              rate[boiler_fuel, t]    rate[chp_fuel, t]
  more rows than sets,     rate[boiler_heat, t]    rate[chp_heat, t]
  grouped by converter_of                          rate[chp_power, t]
```

**The per-row answer is the lookup's own contents** — a table with one entry per
flow, `boiler_heat → boiler` — so `by:` carries no per-row information. It names
which table.

**`by:` sits beside `sign:`, on the link**, because both say how that tie meets
the weights: one which set it reads, the other how it relates to them. That is
how a curve ties **as many expressions as the data gives it** — a converter with
a fourth flow is a row in the lookup rather than an edit to the model — and the
block still writes the tie, so the weights stay its own. Nothing in the file
names λ.

The contract is one sentence: **every link lands on the frame, or on a
refinement of it named by `by:`.** Three things follow:

- **The frame is what the links imply**, each with its own map applied: a link
  over `[flow, time]` mapping `converter_of` puts the weights on
  `[converter, time]`, and nothing declares that twice. A link carrying *fewer*
  dims than the others is refused rather than broadcast — tied at every
  coordinate of what it lacks, one capacity would pin every snapshot to a single
  operating point, and the model would solve.
- **A values parameter follows its link**: `bp_rate` is over `[flow, bp]` because
  that is where its rows sit, and `bp_fuel` over `[converter, bp]`.
- **A map that takes its rows nowhere is refused** — a lookup out of a dim the
  link's expression does not carry.
- **A member the lookup leaves null is on no curve**, so the link builds it no
  row and asks it for no breakpoints. A flow that belongs to no converter is
  tied by nothing, which is what a partial lookup already means everywhere else.
- **One link is enough when it carries a map**, and it may be bounded. The
  two-link minimum is about quantities, and a mapped link's quantities are the
  members its lookup carries; the same goes for a `<=` or `>=`, which needs two
  links to say which side is bounded only when the links are the quantities.

  A sign under a map reaches **every member alike**, so it fits a family whose
  members are all pushed the same way. `>=` is exact for a quantity something
  drives *onto* the curve — a cost or an input under minimisation, where the
  bound binds at the optimum — and slack for one the model *wants*, where
  nothing stops it rising: bound a converter's heat output from below and the
  model buys heat without fuel. The block cannot tell those apart, because
  which way a quantity is pushed is a property of the rest of the model.

A link takes no mask of its own: its rows exist where its expression does, so a
member missing the flow a curve ties builds no row for it and the build reports
the omission.

[The model](../../examples/piecewise_conversion.md) is a boiler tying two flows
and a CHP tying three, on one axis, with neither number in the file.

**`method` is the one thing that varies**, and for those three it varies in
exactly one place: how the weights are restricted, once they exist.

| `method` | What it adds | |
|---|---|---|
| `adjacency` *(default)* | a binary per segment, and `lam <= seg + shift(seg, over=bp, offset=1, edge=0)` | the curve, built |
| `sos2` | an [`sos:`](#sos) block over the same weights | the curve, *said* — for a solver that branches on the set itself |
| `convex` | nothing | the hull, which is a pure LP |
| `lp` | no weights at all — a row per segment line, and two holding the domain | the curve as its own lines |

`adjacency` and `sos2` state the same restriction and reach the same optimum;
they differ in what the solver is handed, so which is faster is a property of
the solver and not of the model.

`convex` is a **different model** — exact only for a curve of matching
curvature under optimisation pressure, which is checked against the breakpoint
*values* when data binds. It takes exactly two links and no `active`.

### `lp`, the one that declares nothing

`lp` states the curve as its **segment lines** instead of interpolating between
its breakpoints, so it declares no auxiliary variable at all — where the others
carry one weight per breakpoint per frame row. It needs exactly two links, one
of them bounded (`<=` or `>=`), and no `active:` — there are no weights for a
gate to pin down.

<!-- doctest: wrap=piecewise -->
```yaml
cost_curve:
  over: bp
  foreach: [snapshot, generator]
  method: lp
  links:
    dispatch: {expression: p, values: bp_x}
    cost: {expression: op_cost, values: bp_y, sign: ">="}  # bounded below by the curve
```

The trade is **columns for rows**: one row per segment plus the two domain
rows, against K weight columns. On a 20-generator, 48-snapshot, 6-breakpoint
dispatch it is 7680 → 1920 columns and 2928 → 6768 rows, at the same optimum
([#926](https://github.com/fluxopt/lpspec/pull/926)).

Two things follow from stating lines rather than weights:

- **The curvature has to match the sign**, and getting it wrong is silent —
  lines that envelope a convex curve *cut* a concave one, and the solve comes
  back optimal with a wrong answer. `>=` requires a convex curve and `<=` a
  concave one, checked against the values when data binds. This is stricter
  than `convex`'s check, which only refuses a *mixed* curve.
- **A line does not stop where its segment does**, so the block emits the two
  domain rows that hold the pinned link inside the breakpoint range. Without
  them the formulation would extrapolate along the end segments, where the
  weight forms cannot go. They are the rows `linopy`'s own `lp` method emits.

## `sos`

A **special-ordered set**: one dimension of one variable, and how many of that
family may be nonzero at once.

<!-- doctest: wrap=sos -->
```yaml
pick_one_size:
  variable: build  # the variable the set is over
  over: size  # the dim it runs along — one set per coordinate of the rest
  type: 1  # 1: at most one nonzero; 2: at most two, and consecutive
  big_m: 500  # optional, and only read by a solver that has to reformulate
```

`type: 1` is a **choice** — at most one member of the family is nonzero.
`type: 2` is an **interpolation** — at most two, and those two *consecutive*,
which is what makes it the native spelling of a piecewise-linear curve.

**A set is over one variable, and a variable holds one set.** A second block
naming the same variable is a load error.

**Membership is the variable's own.** Its `where` decides which coordinates
exist, so a masked-out member is not in the set — and for `type: 2`,
consecutive means consecutive *among the members present*, leaving no hole
where a coordinate was masked away.

**Order is the `over` dimension's declared order** — the same order `shift`
walks ([data binding](data.md)) — so reordering the set means reordering that
index. There is no per-set weight to supply.

### What a solver without SOS does with it

Where the chosen solver has no SOS concept, the set is handed over as binaries
and big-M rows instead. Two consequences reach the model, so neither is silent:

- that rewrite is **mixed-integer**, so a set on an otherwise continuous model
  gives up its [duals](../api.md#reading-a-result) there;
- **M has to be finite**, so every member needs `bounds.upper` or a `big_m:`,
  and a negative `bounds.lower` is refused. `big_m` caps a loose bound — the
  *tighter* of the two is used, tighter being a better relaxation.

Both are conditions of the *rewrite*, so a model that fails them still solves
on a solver that takes the set, and the message says so. HiGHS, which ships
with the package, reformulates; Gurobi branches on the set itself.
