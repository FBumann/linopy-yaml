# Examples

Every model in the repo, what it says, and what it exercises. Three questions,
in the order you probably have them: **[can it say my model?](#can-it-say-my-model)**
· **[is it readable?](#every-model)** · **[does it get the right
answer?](#does-it-get-the-right-answer)** The first and the third are the two
tables on this page; the second is each model page itself, where the file sits
beside the same model written on another stack.

Every page starts from data in the shape the call wants, and
[Preparing the data](data.md) is where that shape comes from.

## Every model

<!-- catalogue:begin -->
### Teaching examples

| | |
|---|---|
| [dispatch](dispatch.md) | Least-cost generation against a load profile — the smallest model that is still a model. |
| [storage](storage.md) | Dispatch plus a battery, and the only construct in the language whose cost is not obviously linear. |
| [transport](transport.md) | A network: generators sit on buses, lines connect buses, and power balances at every bus. |
| [piecewise](piecewise.md) | Per-generator convex cost curves, expanded into a λ-formulation. |
| [special-ordered sets](sos.md) | A piecewise-linear cost curve stated as a **special-ordered set** — [piecewise](piecewise.md) with one line changed, handed to the solver as a set it branches on itself. |
| [monthly budget](monthly_budget.md) | A cap on what each technology may generate per calendar month — an aggregate over a *coarser grouping of time*, written with the same operator that places a generator on a bus. |
| [multi-period](multi_period.md) | Capacity decided once per investment period, binding at every snapshot inside it — and the periods need not be the same size. |
| [reserves](reserves.md) | Energy and reserve co-optimization on a two-bus grid: offers are (generator, market, tranche) triples, reserve zones overlap, and one line dangles. The model exists to prove a claim — every many-to-many shape the language covers, in one instance, each one load-bearing. |
| [walkthrough](walkthrough.md) | The dispatch model plus a macro and a named expression — the one used to print every pipeline stage. |

### The PyPSA ladder

| | |
|---|---|
| [rung 1 — transport](pypsa_transport.md) | PyPSA linear optimal power flow, first rung: transport model, linear marginal cost, no KVL. |
| [rung 2 — ramp limits](pypsa_ramp.md) | [Rung 1](pypsa_transport.md) plus a limit on how fast each generator may change output between snapshots. |
| [rung 3 — storage](pypsa_storage.md) | [Rung 2](pypsa_ramp.md) plus a `StorageUnit` carrying energy between snapshots. |
| [rung 4 — cyclic storage](pypsa_cyclic_storage.md) | [Rung 3](pypsa_storage.md) with the horizon closed on itself: the first snapshot's state of charge carries over from the *last*. |
| [rung 5 — KVL](pypsa_kvl.md) | Passive AC lines: flow is decided by physics, not chosen. **The last rung of the ladder.** |
| [rung 6 — AC-DC, two coordinates](pypsa_ac_dc.md) | A meshed AC–DC network under a CO₂ budget. **PyPSA's own `ac-dc-meshed` example.** |

### PyPSA components and modes

| | |
|---|---|
| [unit commitment](pypsa_unit_commitment.md) | Which generators are *on*, not just how much they produce — a binary per generator per snapshot, with start-up and shut-down charges. |
| [multi-link](pypsa_multilink.md) | One `Link`, one input bus, several output buses, each output derated by its own efficiency — PyPSA's spelling for a CHP plant, an electrolyser with waste heat, any conversion with more than one product. |
| [modular capacity](pypsa_modular.md) | Capacity that comes in whole modules: an integer count decides it, not a continuous bound. |

### Published optima

| | |
|---|---|
| [Dantzig transport](transport_dantzig.md) | Dantzig's transportation problem — GAMS model library #1, and the oldest LP in the corpus. |
| [Dantzig, economies of scale](transport_pwl.md) | GAMS model library `trnspwl`: the same shipping problem, but a big consignment is cheaper per unit — cost grows as `sqrt(x)`, not linearly. |
| [Stigler's diet](stigler_diet.md) | The cheapest way to eat for a year and stay alive. 77 foods, 9 nutrients, 1939 prices. |
| [Facility location](facility_location.md) | Where do you put the warehouses? Open a set of them, assign every customer to one, and trade the fixed cost of opening against the cost of serving from further away. |
| [GenX piecewise fuel](genx_piecewise_fuel.md) | A day of dispatch for two carbon-capture plants and a wind farm under a net-zero carbon cap, where the gas plant's fuel use bends with its output. |
| [Routing telephone calls](telephone_routing.md) | How many of 425 requested circuits a five-city network can carry at once — and by which routes. |
| [Choosing the mode of transport](transport_modes.md) | Moving 180 tonnes of chemicals out of four depots, where a depot may reach a centre by rail *or* by road at different cost. |
| [OSeMOSYS UTOPIA](osemosys_utopia.md) | What to build and how hard to run it, 1990–2010, to meet three end-use demands at least discounted cost. |
| [Travelling salesman](tsp_mtz.md) | Visit every city once and come home, as cheaply as possible. The most famous problem in combinatorial optimisation, and the one most often assumed to be out of reach here. |
<!-- catalogue:end -->

Everything on this page is **generated** — the catalogue off the site nav and
each page's own opening line, the constructs matrix off each model's resolved
plan, the reference table off `examples/ports/references.json`, which is the
same file the tests assert against. Regenerate with
`uv run python -m tools.constructs`; a test fails if any of the three is stale.

Every page also carries the model **as math**, typeset from the same file the
engine builds (`uv run python -m tools.gallery_math`, likewise gated). Where a
model has a symbol table in `examples/symbols/`, the block uses the notation
that model's prose already does.

## Can it say my model?

Read off the resolved plan of each model rather than its text, so it cannot
drift from what the engine builds.

<!-- constructs:begin -->
| model | verified | `sum` | `sum(by=)` | `at()` | `shift` | `shift(edge='wrap')` | `where` | `bounds` | `piecewise` | `sos` | MILP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [dispatch](dispatch.md) | **✔** 10500 | **✓** | · | · | · | · | **✓** | **✓** | · | · | · |
| [monthly_budget](monthly_budget.md) | **✔** 9500 | **✓** | **✓** | · | · | · | · | **✓** | · | · | · |
| [multi_period](multi_period.md) | **✔** 10020 | **✓** | · | **✓** | · | · | · | **✓** | · | · | · |
| [piecewise](piecewise.md) | **✔** 3850 | **✓** | · | · | · | · | · | **✓** | **✓** | · | · |
| [reserves](reserves.md) | **✔** 915 | **✓** | **✓** | **✓** | · | · | · | **✓** | · | · | · |
| [sos](sos.md) | · | **✓** | · | · | · | · | · | **✓** | **✓** | **✓** | · |
| [storage](storage.md) | **✔** 5650 | **✓** | · | · | · | **✓** | · | **✓** | · | · | · |
| [transport](transport.md) | **✔** 4400 | · | **✓** | · | · | · | · | **✓** | · | · | · |
| [walkthrough](walkthrough.md) | · | **✓** | · | · | · | · | **✓** | **✓** | · | · | · |
| [facility_location](facility_location.md) | **✔** 932616 | **✓** | · | · | · | · | · | **✓** | · | · | **✓** |
| [genx_piecewise_fuel](genx_piecewise_fuel.md) | **✔** 2341.82 | **✓** | · | · | · | **✓** | **✓** | **✓** | · | · | · |
| [osemosys_utopia](osemosys_utopia.md) | **✔** 29446.9 | **✓** | · | · | · | · | · | **✓** | · | · | · |
| [pypsa_ac_dc](pypsa_ac_dc.md) | **✔** 1.8441e+07 | **✓** | **✓** | **✓** | · | · | · | **✓** | · | · | · |
| [pypsa_cyclic_storage](pypsa_cyclic_storage.md) | **✔** 17228.8 | · | **✓** | · | **✓** | **✓** | · | **✓** | · | · | · |
| [pypsa_kvl](pypsa_kvl.md) | **✔** 17000 | **✓** | **✓** | · | · | · | · | **✓** | · | · | · |
| [pypsa_modular](pypsa_modular.md) | **✔** 56700 | · | **✓** | · | · | · | · | **✓** | · | · | **✓** |
| [pypsa_multilink](pypsa_multilink.md) | **✔** 1100 | **✓** | **✓** | · | · | · | · | **✓** | · | · | · |
| [pypsa_ramp](pypsa_ramp.md) | **✔** 18200 | · | **✓** | · | **✓** | · | · | **✓** | · | · | · |
| [pypsa_storage](pypsa_storage.md) | **✔** 15253.2 | · | **✓** | · | **✓** | · | **✓** | **✓** | · | · | · |
| [pypsa_transport](pypsa_transport.md) | **✔** 22000 | · | **✓** | · | · | · | · | **✓** | · | · | · |
| [pypsa_unit_commitment](pypsa_unit_commitment.md) | **✔** 24900 | **✓** | · | · | **✓** | · | **✓** | **✓** | · | · | **✓** |
| [stigler_diet](stigler_diet.md) | **✔** 0.108662 | **✓** | · | · | · | · | · | **✓** | · | · | · |
| [telephone_routing](telephone_routing.md) | **✔** 380 | **✓** | **✓** | · | · | · | · | **✓** | · | · | **✓** |
| [transport_dantzig](transport_dantzig.md) | **✔** 153.675 | **✓** | · | · | · | · | · | **✓** | · | · | · |
| [transport_modes](transport_modes.md) | **✔** 1715 | **✓** | **✓** | · | · | · | · | **✓** | · | · | · |
| [transport_pwl](transport_pwl.md) | **✔** 8.78685 | **✓** | · | · | **✓** | · | · | **✓** | **✓** | · | **✓** |
| [tsp_mtz](tsp_mtz.md) | **✔** 2085 | **✓** | **✓** | · | · | · | **✓** | **✓** | · | · | **✓** |
<!-- constructs:end -->

**No holes left.** Every construct in the table has at least one model behind
it whose optimum came from somebody else. The column was worth keeping
precisely because it named each gap out loud before it was filled: `roll /
shift` went first ([ramp limits](pypsa_ramp.md)), then integrality
([unit commitment](pypsa_unit_commitment.md)), and `piecewise` last
([economies of scale](transport_pwl.md)).

That is a floor, not a ceiling. A tick means *one* verified model exercises the
construct — not that every shape of it is covered, and not that the constructs
are exercised in combination. The table's job from here is to stay honest as
the language grows, which is why it is generated rather than maintained by
hand.

## Does it get the right answer?

**✔ means the optimum did not come from lpspec** — a figure published with the
model, or a reference implementation hand-written on another stack, each row's
provenance saying which. Every model on this page is run by the test suite, so
"there is a test" distinguishes nothing. What the badge marks is narrower, and
it is the only check that can catch a *shared misreading* — both lanes of the
implementation agreeing on a meaning the modeller did not intend, which passes
every lpspec-against-lpspec test green.

Even the differential harness compares two lanes consuming the *same resolved
AST* ([hard rule 1](../about/architecture.md#hard-rules)), which is what makes them
an oracle for each other and also what they cannot see. This is the net for
that class, and the evidence behind
[the ceiling](../about/ceiling.md#two-tiers-and-the-ceiling).

<!-- references:begin -->
| port | optimum | `rtol` | duals | reference |
|---|---|---|---|---|
| [dispatch](dispatch.md) | 10500.0 | 1e-09 | **✔** | linopy 0.9.0, via examples/ports/references/linopy/dispatch.py — agreement, not a published figure |
| [facility_location](facility_location.md) | 932615.75 | 1e-09 | · | published by OR-Library (Beasley) for instance cap71 of the uncapacitated warehouse location set, in the file uncapopt: http://people.brunel.ac.uk/~mastjjb/jeb/orlib/uncapinfo.html |
| [genx_piecewise_fuel](genx_piecewise_fuel.md) | 2341.8230753008093 | 1e-09 | · | published by GenX: asserted in test/test_piecewisefuel.jl as obj_true = 2341.82308 under genx_setup UCommit=2, CO2Cap=1, ParameterScale=1, and reproduced here by running GenX itself (julia 1.12.6, HiGHS) which reports 2341.8230753008093 |
| [monthly_budget](monthly_budget.md) | 9500.0 | 1e-09 | **✔** | linopy 0.9.0, via examples/ports/references/linopy/monthly_budget.py — agreement, not a published figure |
| [multi_period](multi_period.md) | 10020.0 | 1e-09 | **✔** | linopy 0.9.0, via examples/ports/references/linopy/multi_period.py — agreement, not a published figure |
| [osemosys_utopia](osemosys_utopia.md) | 29446.86269 | 1e-09 | · | published by OSeMOSYS: asserted in OSeMOSYS_GNU_MathProg tests/test_gnu_mathprog.py as obj = 2.944686269e+04 for tests/utopia.txt, and reproduced here by running GLPK directly (glpsol 5.0, src/osemosys.txt) — an oracle outside Python entirely |
| [piecewise](piecewise.md) | 3850.0 | 1e-09 | **✔** | linopy 0.9.0, via examples/ports/references/linopy/piecewise.py — agreement, not a published figure |
| [pypsa_ac_dc](pypsa_ac_dc.md) | 18441021.477729216 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_ac_dc.py — n.objective + n.objective_constant, the system cost |
| [pypsa_cyclic_storage](pypsa_cyclic_storage.md) | 17228.77962151063 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_cyclic_storage.py |
| [pypsa_kvl](pypsa_kvl.md) | 17000.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_kvl.py |
| [pypsa_modular](pypsa_modular.md) | 56700.0 | 1e-09 | · | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_modular.py |
| [pypsa_multilink](pypsa_multilink.md) | 1100.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_multilink.py |
| [pypsa_ramp](pypsa_ramp.md) | 18200.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_ramp.py |
| [pypsa_storage](pypsa_storage.md) | 15253.178322993519 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_storage.py |
| [pypsa_transport](pypsa_transport.md) | 22000.0 | 1e-09 | **✔** | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_transport.py |
| [pypsa_unit_commitment](pypsa_unit_commitment.md) | 24900.0 | 1e-09 | · | pypsa 1.2.4 (its own linopy 0.9.0), via examples/ports/references/pypsa/pypsa_unit_commitment.py |
| [reserves](reserves.md) | 915.0 | 1e-09 | **✔** | linopy 0.9.0, via examples/ports/references/linopy/reserves.py — agreement, not a published figure |
| [stigler_diet](stigler_diet.md) | 0.10866227820675685 | 1e-09 | **✔** | linopy 0.9.0, via examples/ports/references/linopy/stigler_diet.py — dollars per day; x365 = $39.6617/year[^stigler_diet] |
| [storage](storage.md) | 5650.0 | 1e-09 | **✔** | linopy 0.9.0, via examples/ports/references/linopy/storage.py — agreement, not a published figure |
| [telephone_routing](telephone_routing.md) | 380.0 | 1e-09 | · | published by Gueret, Prins, Sevaux & Heipcke, Applications of Optimization with Xpress-MP (Dash Optimization, 2002) SS12.3.3 p. 182 — "380 out of the required 425 calls are routed"; problem and data in SS12.3, pp. 180-182 |
| [transport](transport.md) | 4400.0 | 1e-09 | **✔** | linopy 0.9.0, via examples/ports/references/linopy/transport.py — agreement, not a published figure |
| [transport_dantzig](transport_dantzig.md) | 153.675 | 1e-09 | **✔** | published with GAMS model library #1 (trnsport), after Dantzig, Linear Programming and Extensions (1963) ch. 3.3[^transport_dantzig] |
| [transport_modes](transport_modes.md) | 1715.0 | 1e-09 | · | published by Gueret, Prins, Sevaux & Heipcke, Applications of Optimization with Xpress-MP (Dash Optimization, 2002) SS10.2.3 p. 143 — "The minimum cost is EUR 1,715k"; problem and data in SS10.2, p. 142 |
| [transport_pwl](transport_pwl.md) | 8.786852757777865 | 1e-09 | · | linopy 0.9.0's own add_piecewise_formulation, via examples/ports/references/linopy/transport_pwl.py; the model is GAMS model library trnspwl (Dantzig transport with economies of scale), which publishes the formulation and its discretisation but no optimal objective |
| [tsp_mtz](tsp_mtz.md) | 2085.0 | 1e-09 | · | published by TSPLIB for instance gr17 (Groetschel, 17 cities, EXPLICIT lower-diagonal distance matrix); optimum 2085 as listed in the TSPLIB solutions file |

[^stigler_diet]: Laderman (1947) at the National Bureau of Standards published $39.69/year for this data, the first serious test of the simplex method. This LP's exact optimum is 0.08% under it — his rounding, not a different model — and both select the same five foods: wheat flour, liver, cabbage, spinach, navy beans.

[^transport_dantzig]: examples/ports/references/linopy/transport_dantzig.py — the same LP hand-written in linopy 0.9.0, which reaches 153.675 independently. Secondary: the published figure is what verifies the port.
<!-- references:end -->

**The objective is not the only thing checked.** A port with a `duals` tick
also records the reference's **shadow prices** and is asserted against them —
for the PyPSA models that is `buses_t.marginal_price`, the nodal price, which
is the output this audience reads most often after the cost.

That matters because an objective is one number and hides a great deal. A dual
vector is where two implementations most reliably disagree quietly: which side
of a constraint the price belongs to, and what sign an inequality carries.
[Dantzig transport](transport_dantzig.md) is in that set specifically because
both of its constraints are inequalities pointing opposite ways. A MILP has no
dual solution, and lpspec refuses to invent one — which is what the `·` rows
are.

Adding a port is four files and five rules:
[CONTRIBUTING.md](https://github.com/fluxopt/lpspec/blob/main/CONTRIBUTING.md#adding-a-ported-model).

## The ladder

Reproducing a full PyPSA objective means reproducing marginal *and* capital
cost, ramp limits, storage cycling and KVL at once, and a mismatch then
implicates five features instead of one. So each network is a ladder, one
feature per rung, each switched off in PyPSA and reproduced here:
**1 transport model** ✔ · **2 ramp limits** ✔ · **3 storage with state of
charge** ✔ · **4 cyclic boundary condition** ✔ · **5 KVL** ✔ · **6 a meshed
AC-DC network under a CO₂ budget** ✔. Rungs 1–5 are one feature at a time on a
three-bus network; rung 6 is the first that puts several of them on a network
somebody else designed, which is a different question — not *can it say this
feature* but *does the whole thing still read*.
**The ladder is the six rungs.** A feature that needs no network gets a one-bus
model of its own instead — *PyPSA components and modes* above — where a
mismatch implicates the one thing switched on.

A rung that matches is a row in the table above; one that **cannot be said** is
a row in the ledger. Both are evidence, so no rung is wasted.

Rung 2 needed the instance widened before it meant anything. Rung 1's links run
saturated, which fixes every generator's output exactly, so a ramp limit on that
network can only make it infeasible — never change the answer. A rung that
cannot bind is not evidence that it works.

**Rung 6 is where a second coordinate first earns its keep.** A generator
sits on a bus *and* burns a carrier, and both maps are load-bearing — the
balance groups through one, the CO₂ budget reads an emission rate back down
through the other. It also carries passive lines and controllable links at
once, so both branch kinds group onto the same bus dimension in one equation.

**The ladder reached rung 5 without a new primitive.** Rung 5 is Kirchhoff's voltage
law, and it needed nothing added to the language: a cycle basis is a sparse
`(cycle, line)` incidence *parameter*, and the constraint is one
`sum(f * cycle_incidence, over=line) == 0`. A line can belong to several
cycles, so the incidence cannot be a declared lookup — that is the shape
finding, and it is the same "topology is data" claim the corpus started with.
Computing the basis is a graph algorithm and stays in data preparation, where
the ceiling puts it.

**Rung 4 made the model smaller**, which is the ladder paying off in the
direction nobody plans for. Closing the horizon deletes rung 3's boundary
equation outright: `edge='wrap'` is cyclic already, so the wrap onto the last snapshot
is what it does unguarded, and the *acyclic* case is the one needing an extra
clause. Two rungs written a day apart, differing by one deleted `where`, is a
sharper statement about the language than either alone.

## Ledger — what a port could not say

Feeds [the roadmap](../about/roadmap.md), with the verdict
[AGENTS.md](https://github.com/fluxopt/lpspec/blob/main/AGENTS.md) asks for:
macro, primitive, or escape.

| Port | What could not be said | Worked around by | Verdict |
|---|---|---|---|
| PyPSA rung 1 | a bound of `-rating` — PyPSA's `p_min_pu = -1` | shipping `neg_rating` as data | **primitive**: bounds as expressions, [#31](https://github.com/fluxopt/lpspec/issues/31). A second model asking for it |
| PyPSA unit commitment | `min_up_time` — a unit that starts must stay up for *T* snapshots | left at 0, so the constraint is not written | **sayable**, and the row used to say otherwise — below |
| Travelling salesman | subtour cuts **generated lazily** inside branch-and-cut, which is how every serious TSP code works | [MTZ](tsp_mtz.md), O(n²) and static | **refused, and correctly**: a solve loop is an algorithm, not a model |

`min_up_time` is the row worth reading twice, because it was **wrong** until
recently and the correction is instructive. The constraint is
`sum(start_up over the last T snapshots) <= status`. For a *single T fixed in
the file* that is `start_up + shift(start_up, over=snapshot, by=1, edge=0) + …`
— a **macro**, free, and `edge=0` because a window reaching before the horizon
is short a term rather than undefined ([law 8](../reference/language/index.md#ten-rules-the-language-reduces-to)).

For PyPSA's actual signature, where `T` is a column and each generator may have
its own, this row used to claim the constraint was refused by design, because
the number of *terms* is read from data. That confused a spelling with the
constraint. No chain of shifts can be written down — but the window is a
relation between snapshots, one row per pair inside it, and a relation is an
incidence table:

```yaml
lookups:
  same_moment: {over: snapshot_from, into: snapshot}
constraints:
  a_start_turns_it_on:
    foreach: [unit, snapshot_from]
    expression: >-
      started >= at(on, by=same_moment)
      - shift(at(on, by=same_moment), over=snapshot_from, by=1, edge=0)
  stays_up_its_own_time:
    foreach: [unit, snapshot]
    expression: sum(started * window, over=snapshot_from) <= on
```

with `window[unit, snapshot, snapshot_from]` built in data preparation. That is
the shape [`pypsa_kvl`](pypsa_kvl.md) already uses for a cycle basis and
[UTOPIA](osemosys_utopia.md) for an operational life. The plan's *shape* is
fixed before any data is read; only its cardinality comes from data, which is
as true of `foreach: [snapshot]`.

**What it costs is one mirror of the snapshot axis** — the window sum is taken
on `snapshot_from` while capacity and cost are rows on `snapshot`, and
[`tsp_mtz`](tsp_mtz.md) carries a mirror for the same reason. It costs nothing
else: `snapshot_from` maps back to `snapshot` single-valuedly, so the mirror is
a **lookup**, and `at()` reads the commitment across it. No second commitment
variable, no identity table. A cost, then, and a small one — not a refusal.

Three rows from eighteen ports — a rate worth watching once the corpus has hit
the ceiling a few more times.

### Shapes still without a witness

Not things the language cannot say — things no *outside* model in the corpus
has yet been found to need. Each was searched for and not found, so the row is
a standing request rather than a gap in the language:

| Shape | Where it was looked for |
|---|---|
| one axis grouped several ways, all of them load-bearing | OSeMOSYS UTOPIA declares three maps out of its timeslice, but they feed only storage constraints and the instance builds none |
| a chain whose coarse end carries a constraint | PyPSA's `ac-dc-meshed` has a country per bus and nothing constrains a country; GAMSLIB `alum` composes three such chains and its shipped scenario switches two of them off |
| a group that carries its own constraint and appears nowhere else | GAMSLIB `mexls` is exactly this, and its optimum is published only in a book with no reachable text |
| a partial map whose null membership moves an optimum | `reserves` exercises it, but that model is ours and was built to |
| opposite-sign legs onto one dimension, plus a second hop a constraint reads | an airline fleet-assignment text has it; the data and both optima are not obtainable |

A model that needs one of these is worth more to this corpus than another that
exercises a shape already covered.

**The TSP row is the one to read**, and it is narrower than it first looked.
Writing DFJ's subtour rows out in full *is* sayable — the subsets go in as data
exactly the way [KVL's cycle basis](pypsa_kvl.md) does, and an 8-city instance
with all 246 subsets solves to a correct tour. There are 2ⁿ of them, so it
stops being practical around twenty cities, but that is a data-size wall rather
than a ceiling.

What is genuinely outside is *lazy* generation: solve, find the violated
subsets, add rows, re-solve. That is an algorithm, and this language describes
models. Since lazy generation is what every serious TSP code actually does,
"lpspec can express TSP" and "lpspec is a good way to solve a large TSP" are
different sentences and only the first is true.

A data-dependent row count is not what rules DFJ out: the cycle basis has one
and is ordinary. What the corpus is for is catching exactly that kind of
mis-attribution, where a data-size wall reads as a ceiling.
