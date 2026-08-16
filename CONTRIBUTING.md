# Contributing

Procedure lives here. **Why** the project is shaped the way it is lives in
[docs/about/architecture.md](docs/about/architecture.md), and that split is deliberate: this file
should be readable in one sitting and go stale only when a command changes.

## Setup

```bash
uv sync  # dev group is the default: tools + the linopy oracle
uv run pre-commit install  # once per clone
```

`uv sync` installs the `[linopy]` extra too, because the differential test
suite needs a second lane to compare against, and `[gurobi]` and `[xpress]`,
because a solver sink is checked against another solver. Both of those wheels
carry a size-limited licence of their own — gurobipy's needs nothing, and
xpress's Community licence is active on import — so those tests run on a plain
checkout with no licence of your own. They skip where the package is absent.
The engine itself never imports linopy, xarray, pandas, gurobipy or xpress —
see *the bare install* below.

## The loop

```bash
uv run pytest -q -n auto  # ~20 s
uv run ruff check --fix . && uv run ruff format .
uv run pyrefly check
```

Narrower runs while working:

```bash
uv run pytest tests/test_relational.py -q
uv run pytest -k piecewise -q
uv run pytest --lf  # last failures only
```

## What each CI gate means

`main` requires two checks: **`ci`** and **`Conventional commit subject`**.
Everything below is the first one, in the order it runs.

| gate | what a failure means |
|---|---|
| `ruff check .` | a lint rule fired. `--fix` handles most; if the finding is wrong, silence the one line with a `# noqa: RULE` and say why. |
| `ruff format --check .` | formatting drifted. Run `ruff format .`. |
| `pyrefly check` | a type is wrong. **Fix the type, don't widen it** — if the finding is genuinely wrong, `# pyrefly: ignore[rule-name]` on the one line with a reason, never the rule off globally. |
| `pytest -q` | the suite. Includes the differential lanes and the ported models. |
| `mkdocs build --strict` | the site. A dead cross-link, an anchor that no longer resolves, or a page under `docs/` with no `nav:` entry — see *the docs* below. |
| **bare install, at the floors** | the engine reached for something it does not declare. |

**The bare-install job is the one worth understanding.** It reinstalls with
`--resolution lowest-direct` and *no* dev group, asserts `linopy` is absent,
and runs the suite. It proves two things at once: that the relational lane
builds, solves and reads results back with no pandas, pyarrow, linopy or
xarray; and that the declared lower bounds are real rather than decorative.
Tests that need a second lane route through `tests/oracle.py`, which skips
them when it is not installed — a bare `import pandas` in a test file breaks
this job, and only this job.

Raise a floor when the code relies on that version's behaviour. Do not raise
one to chase a newer interpreter.

## The docs

`docs/` is both the site and what you read on GitHub. Write for the repo —
relative links, no site-only syntax — and the build handles the difference.

```bash
uv sync --group docs
uv run mkdocs serve  # http://127.0.0.1:8000, live-reloading
uv run mkdocs build --strict  # what CI runs
```

**A page states a rule or argues for one, and that decides where it goes.**
`docs/guide.md`, `docs/examples/` and `docs/reference/` are the path a reader
writing or running a model takes; design notes, measured cost, project
direction and the changelog live under `docs/about/`, reachable and out of the
way.

Three rules on top of that, each enforced, so none has to be remembered:

- **Every page under `docs/` needs a `nav:` entry** in `mkdocs.yml`. Adding a
  model page without one fails the build rather than shipping an unreachable
  page. `docs/README.md` is the deliberate exception — it is the folder view
  GitHub renders, and `exclude_docs` keeps it out of the site, where
  `docs/index.md` is the home page.
- **Inside `docs/`, link relatively.** `../reference/api.md`,
  `examples/index.md`. mkdocs resolves and validates these; a dead one fails the
  build.
- **Outside `docs/`, write the full GitHub URL** —
  `https://github.com/fluxopt/lpspec/blob/main/bench/README.md`, not
  `../bench/README.md`. The site has no file above `docs/` to resolve to, and
  mkdocs does *not* flag the relative form: it ships as a silent 404. This is
  the same convention the model pages already use to link at their `.yaml`.

`tests/test_docs_site.py` enforces the last two in both directions — no
relative link may escape `docs/`, and every blob URL must name a file that
exists. Neither is checkable by mkdocs, which is why they are tests.

Headings are slugged the way GitHub slugs them, so `#track-4--sink-capabilities`
means the same thing in both places.

Read the Docs builds and publishes from `main` (`.readthedocs.yaml`); nothing
needs deploying by hand.

## Branches, commits, PRs

**Never commit on `main`.** It takes squash merges through a PR only, and the
ruleset enforces it.

The PR title is parsed by release-please and becomes the changelog entry, so it
has to be a conventional-commit subject:

```
feat: streaming engine for indexed constraints
fix(parser): where clauses with a trailing comma
refactor(api): closed helper set, no monkey-patch
```

Types: `feat` `fix` `perf` `refactor` `docs` — these appear in the changelog —
plus `chore` `test` `ci` `build` `style` `revert`, which are hidden. A subject
that will not parse fails the required check rather than silently dropping the
entry. Fixing it is an edit to the PR, not a branch rewrite.

**No `!`, and no `BREAKING CHANGE:` footer.** The same check refuses both while
the version is pinned to the alpha stream, because a breaking marker moves the
*base* version rather than the alpha counter — the accident is written up in
[RELEASING.md](RELEASING.md). Describe the break in the PR body instead; the
next section is why there is nothing for the version to announce.

`main` is protected: no force-push, no deletion, squash-only through a PR, and
the two required checks above. Approvals are not required, but the PR is.

Versioning, the release PR, and how to force a specific version:
[RELEASING.md](RELEASING.md).

## Filing issues

**Cite behaviour and a file, not a private symbol and a line range.** An issue
outlives several refactors; one written against internals dies with the next and
takes its argument with it — four have had to be closed and re-filed for exactly
that. So write `bounds accept a parameter name or a number, not an expression
(language/model.py)`, not a line number inside the loop that enforces it.

**`now` is the only order label**, capped at five; everything else is backlog.
Grouping is sub-issue parentage — a track is a parent issue — because parentage
is structural where a label mirroring a list is a copy that drifts.
`blocked:upstream` and `blocked:decision` say what an issue waits on. A
`decision` closes by *resolution*, not by work: on yes it becomes `roadmap`, on
no it becomes a row in the deliberate non-primitives table.

## Breaking changes are free

**The project is `0.0.1aN` until the first official release, and holds no
compatibility promise.** So a construct that is named wrong, a default that is
wrong, or a permissive input that hides a silent wrong answer gets **fixed in
place**: rename, move and delete outright — no alias for the old spelling, no
`DeprecationWarning` cycle, no `legacy_` path beside the new one.

Spend nothing on the retirement either. The closed schema already fails at load
naming the valid keys and the near miss, and the operator table already names
what it accepts — that is the whole migration story an alpha owes anyone. A
hand-written message per retired spelling is a second place the old surface
lives, it needs a test of its own, and it outlives every file it was written
for: `shift(by=)` had one for a day before `by=` became a legal keyword again
and the message started refusing the new spelling.

This binds **agents working in this repo** too, and it is the habit most often
imported from elsewhere: asked to change something, change it — do not add
backwards compatibility nobody requested.

It is the *surface* that is unfrozen, not the behaviour. What exists is tested
and differentially verified against linopy; a break is a deliberate rewrite, not
licence for churn.

## Changing the language

**Triage first: macro, primitive, or escape?** Most requests are compositions
and cost nothing. A genuinely new shape earns a primitive only if it clears the
expressive ceiling — relational ∩ local, degree 2 in the math and 1 beside it. Unsayable math goes to a
declared `escape:` island rather than into the language.

Read, in order:

1. [the deliberate non-primitives](docs/about/ceiling.md#deliberate-non-primitives) — parity with
   another tool is not by itself a reason to add anything, and several
   plausible-sounding features are refused there on purpose;
2. [the ceiling in docs/about/ceiling.md](docs/about/ceiling.md#two-tiers-and-the-ceiling) —
   the admissibility test;
3. [the extension checklists](docs/about/architecture.md#extension-checklists), which sit directly under that
   test. They stay there rather than moving here: *may I?* and *how?* are one
   question, and splitting them invites answering the second without the first.

A PR that adds, renames or retires a construct updates the [language
reference](docs/reference/language/index.md).
Rationale belongs in the PR description or a code comment; "this used to work
differently" belongs in git.

## Adding a ported model

A port is a model somebody else already solved, said again in this language and
checked against **an optimum that did not come from us**. It is the only test
class that can catch a *shared misreading* — both lanes agreeing on a meaning
the modeller did not intend — because every other test compares lpspec against
lpspec. The corpus, its ladder and the ledger of what a port could *not* say
are in [docs/examples/index.md](docs/examples/index.md), where the reference table
is generated from `examples/ports/references.json` — the same file the tests
assert against. Each port's page there shows the model and a side-by-side
against its reference.

**Check the claim against the shipped instance before writing a file.** A model
is picked because its structure exercises something; twelve candidates chosen
that way produced six whose structure was not in the data they ship. Three
checks, in the order they catch things:

1. **Count the rows.** A map into a one-member dimension cannot change an
   answer, and a table with no rows is not structure. An empty `zone` table and
   a one-member `DAYTYPE` both read as topology until counted.
2. **Quote the source's own constraint that reads the map** — not its schema,
   not its prose. A source that models the same problem over three separate
   sets needs no map at all, whatever its data dictionary suggests.
3. **Run the reference and read the *solution*.** The first two are about the
   model; this one is about the instance. A constraint that holds at zero
   reads nothing, and a scenario parameter can switch one off — OSeMOSYS's
   UTOPIA passes 1 and 2 and still fails, because its season and day-type maps
   feed only storage constraints and the instance builds no storage.

Running the reference costs minutes and settles what a port otherwise
discovers after days of transcription. It is also what tells you the published
figure is still the model's answer: one candidate's asserted optimum had drifted
from what its own code computes, hidden by a loose tolerance.

Four files per port:

```
examples/ports/<name>.yaml                  the model
examples/ports/data/<name>.json             the instance
examples/ports/references/<arm>/<name>.py   a reference implementation, importing no lpspec
examples/ports/references.json              the recorded objective and where it came from
docs/examples/<name>.md                     the gallery page — maths, model, side-by-side
```

`<arm>` is the library the reference is written in — `linopy` or `pypsa` today,
`pyomo` when one arrives. The gallery page shows each arm as a tab beside the
YAML, and `tests/test_models_gallery.py` holds the two in lockstep: a script
with no tab fails, and so does a tab with no script behind it.

A *teaching* model may carry the same three reference files — the instance, the
script, the `references.json` entry — while its model file stays in
`examples/`, where the guide points. Its check is weaker and the provenance
says so: agreement with an independent hand-written formulation, not a
published figure.

- **A published optimum needs no script.** `transport_dantzig`'s number comes
  from the literature, and the citation *is* its provenance. It also ships a
  reference implementation, but as `corroborated_by` rather than as what
  verifies it — a second opinion, where the published figure is the first.
- **Reference scripts are never run by CI.** Pinning PyPSA into this project
  would hand their release cadence a veto over the suite. They carry their
  dependencies inline ([PEP 723](https://peps.python.org/pep-0723/)), pinned to
  whatever produced the recorded number, and are run out of band:
  ```bash
  uv run --script examples/ports/references/pypsa/pypsa_transport.py
  ```
- **Both sides read the same instance.** A reference optimum against a
  different instance means nothing. What must stay independent is the
  formulation, not the data.
- **`rtol` is per port.** A published optimum is rounded; a solved one is not.
- **Record the duals too, if the model has any.** An objective is one number;
  a dual vector is where two implementations disagree quietly — which side of a
  constraint the price belongs to, and what sign an inequality carries. The
  reference script prints a `duals {...}` line keyed by constraint name; paste
  it into the port's entry as a `duals` block. A MILP has no dual solution, so
  it records none and the test skips rather than passing vacuously.
- **Regenerate the gallery's tables**, don't write them: both the construct
  matrix and the reference table in `docs/examples/index.md` come from
  `uv run python -m tools.constructs`, and a test fails if either is stale. The
  reference table is rendered straight from `references.json`, so the published
  optimum and the asserted one cannot disagree.
- **A rung that cannot be said is also a result.** It goes in the ledger with a
  verdict — macro, primitive or escape — and feeds docs/about/roadmap.md. Do not work
  around a gap silently.

## Refreshing the benchmarks

Full method, and why each measurement is taken the way it is, in
[bench/README.md](bench/README.md). The short version:

```bash
uv sync --group bench
uv run pytest bench --benchmark-memory --sizes xs s m l \
    --benchmark-json=bench/results/latest.json
uv run pytest bench --benchmark-memory --sizes d100 d50 d25 d08 --skip-gate \
    --benchmark-json=bench/results/density.json
uv run pytest bench --benchmark-memory --sizes n002 n008 n032 n128 --skip-gate \
    --benchmark-json=bench/results/declarations.json
uv run python -m bench.report bench/results/latest.json bench/results/density.json \
    bench/results/declarations.json
uv run python -m bench.plot
```

Three things that have each cost us a wrong published number:

- **Measure on an idle machine.** A ladder taken while the laptop was busy
  inflated one case by 55% — enough to turn "level" into "the one case we lose".
- **A run replaces its output file.** Anything narrower than the published
  ladder goes to `--benchmark-json=/tmp/something.json`, or the tables keep
  their old numbers with a fingerprint that no longer describes them.
- **Never retype a number.** `bench.report` prints the markdown and
  `bench.plot` rewrites the chart page's data, both from the results file. A
  figure typed by hand outlives the run that produced it.
