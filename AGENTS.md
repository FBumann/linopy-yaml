# AGENTS.md

How to work in this repo. **What** the project is lives in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/SPEC.md](docs/SPEC.md);
**which command to run** lives in [CONTRIBUTING.md](CONTRIBUTING.md). This file
is the third thing — the habits, each one written down because its absence cost
a wrong number, a lost diff, or a round trip.

Everything here binds humans too. It is addressed to agents because agents are
who keep re-learning it.

## Where the rules live

| question | file |
|---|---|
| what may a YAML file say | [docs/SPEC.md](docs/SPEC.md) — §0 is the ten laws, every section elaborates one |
| how does it fit together | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — hard rules, module map |
| may this enter the language | [docs/design/ceiling.md](docs/design/ceiling.md) — the two tiers and the admissibility test |
| where is it going | [docs/ROADMAP.md](docs/ROADMAP.md) — motivation and end state, no work items |
| how do I run/ship it | [CONTRIBUTING.md](CONTRIBUTING.md) — setup, gates, ports, benchmarks |

**Reference pages carry rules, design notes carry arguments.** A change that
makes one longer is usually a change that belongs in the other.

## Start from the base you claim to start from

`main` moves several times a day, often while you are working. The git status in
your prompt is a **snapshot taken before you started** and goes stale silently —
once it read `docs/spec-drift-cleanup` while `HEAD` was a different, unmerged
branch, and a doc PR landed on top of somebody else's in-flight work.

```bash
git fetch origin && git branch --show-current   # costs nothing, run it first
git worktree add ../wt/<topic> -b <type>/<topic> origin/main
```

- **Branch from `origin/main`, not from the tree you happen to be sitting in.**
- **Work in a worktree.** One topic, one worktree, one PR.
- **Verifying a claim about shipped behaviour? Verify it against `origin/main`.**
  The working tree may carry unmerged code that makes the claim true only here.
- **When told to rebase or resolve conflicts, re-measure anything the PR claims.**
  A benchmark taken before the merge describes a base that no longer exists.

## Assume another agent is working in this repo right now

Several sessions run against this checkout at once. One of them rewrote
`helpers.py` mid-run and broke `import lpspec`; a branch switch in the primary
tree wiped another session's uncommitted changes.

- **Commit to your branch early** rather than holding a large working-tree diff.
- **If files change under you, stop and say so.** Do not untangle whose edit is
  whose, and do not `git checkout` your way out of it.
- **Never push a branch you did not create in this session**, and never push
  while the user has said another agent is mid-run.
- **Check whether the work is already done** before redoing it: a PR may have
  been merged, closed, or superseded since it was last mentioned. `gh pr view`
  before `git rebase`.

## Two gates that pass locally and fail in CI

Both bite specifically when running from a worktree:

- **`pyrefly check` in project mode silently skips gitignored directories** — it
  prints `WARN Skipping include pattern …` and **exits 0**, which reads as
  success. `.claude/` is gitignored, so a worktree under it has type checking
  effectively off. Use `uv run pyrefly check $(git ls-files 'src/**/*.py')`.
- **`ruff format --check` runs on `.`, never on the changed `.py` files.** It
  formats python blocks inside markdown, so a docs edit fails a gate that every
  `.py` file passes.

## Finish the loop, and say where it landed

The most-repeated correction in this repo is one word: *pushed?* Work that is
not committed, pushed and open as a PR is not done, and "I'll open the PR" in a
summary is not an open PR.

- Report the PR **URL**, not the intent.
- **The user merges.** Never merge or force-push, and never delete a branch you
  did not create.
- **Green CI is part of done.** Watch the run out; a failure you did not wait
  for becomes the user's next message.
- **One issue, one PR.** Related-but-separable work is a stacked PR, not extra
  commits on this one. Split rather than bundle.
- PR titles are conventional-commit subjects — release-please parses them into
  the changelog. `CONTRIBUTING.md` has the types and the two forbidden markers.

## A number without provenance is not a result

Performance is a product claim here, so measurement discipline is not optional:

- **Measure on an idle machine.** A ladder taken while the laptop was busy
  inflated one case by 55% — enough to invent a regression that was not there.
- **Say what base, at which commit, on what date.** PR descriptions carrying
  benchmarks name the hash they were taken at; after a rebase they are re-run or
  removed, not left standing.
- **A/B or it did not happen.** Compare candidate against base built the same
  way, same sizes, same process, and re-run when the two are close enough that
  the ranking could flip. Noise is the default explanation for a small delta.
- **State what is counted and what is not**, next to the number — build vs
  solve, LP file vs direct sink, in-memory vs parquet input.
- **Never retype a number.** `bench.report` and `bench.plot` regenerate the
  tables and charts from the results file; a hand-typed figure outlives the run.
- **Do not accept a number from the user either.** "My benchmarks are
  preliminary — check for yourself" is standing instruction.

## Change it, do not wrap it

The project is `0.0.1aN` and holds no compatibility promise. This is the habit
most often imported from other repos, and it is wrong here in both directions:

- Asked to change something, **change it** — rename, move, delete. No alias, no
  deprecation cycle, no `legacy_` path beside the new one.
- **A test asserting the old behaviour may be rewritten or deleted.** Do not let
  one block a simplification; say in the PR what coverage moved where.
- **Timid diffs get sent back.** Asked to simplify a subsystem, the expected
  output is fewer lines and fewer concepts, not a defensive rename. If the
  cleanup turns out smaller than it looked, say why in one line rather than
  padding it.

## The repo holds facts, not narration

- **No explanatory inline comments.** Complex logic becomes a helper whose
  docstring carries the constraint. Measured numbers and issue refs move into
  that docstring — deleting a comment must never delete a fact. What stays
  inline: pragmas (`# pyrefly: ignore[…]` with its reason, `# fmt: skip`), `#:`
  attribute docs, section dividers.
- **Docstrings are informative and short.** A docstring that restates the
  signature, or narrates the change that introduced it, is noise.
- **Nothing transitive.** "Previously this used to…", "as of the polars
  rewrite…", "renamed from…" belong in the PR description or git, never in the
  tree.
- **Rationale for a change goes in the PR description.** That is what it is for,
  and it is where the user reads it.

## Docs move with the change, or they contradict it

A stale sentence outranks a correct implementation in every reader's head. The
spec once still described a router with four fallback triggers that no longer
existed, and every reader concluded the no-fallback claim was aspirational.

- A PR that adds, renames or retires a construct **updates `docs/SPEC.md`** —
  §0 if it changes a law, the section if it changes a detail.
- A PR that changes structure **updates `docs/ARCHITECTURE.md`**, diagrams
  included.
- **When a decision is made in conversation, sweep for what now contradicts it.**
  Half-updated docs are the failure mode: the rule changes in one file and the
  roadmap still argues the old position.
- **Stale rationale is corrected even when the item it justified is done.** A
  number from a closed issue should not stay quotable.
- Links inside `docs/` are relative; a link to anything **outside** `docs/` is a
  full GitHub URL — the relative form resolves in the repo and 404s on the site,
  silently. `tests/test_docs_site.py` enforces it.

## Verify before you act on it

Findings age faster than code. Review comments, issue bodies, and your own
earlier analysis are **claims to check**, not work orders:

- Re-read the current code before fixing a reported finding. Fix what is still
  valid, skip the rest with a one-line reason, and say which you skipped.
- An issue whose body was invalidated by a rewrite gets **closed and re-filed**
  against current code, not edited or annotated — except coverage checklists and
  scope decisions, which stay open with a status comment.
- File issues against **behaviour and a file**, not a private symbol and a line
  range, and **label them** — four issues have had to be re-filed for exactly
  this.

## Answer the question that was asked

- **"Discuss", "should we", "is it worth" are questions.** Answer them, then
  stop. Building the thing before the answer lands wastes the answer.
- **Do not widen the scope on your own.** Touching an unrelated PR or issue
  because it seemed adjacent is the correction, every time. Finish what was
  asked, then name the adjacent thing in one sentence.
- **Recommend, do not survey.** A ranked answer with the reason attached beats
  an even-handed list of options.
- **Before proposing a language feature, triage it: macro, primitive, or
  escape?** Most requests are compositions (macro, free). A genuinely new shape
  earns a primitive only if it clears the ceiling — degree 1 ∩ relational ∩
  local. Unsayable math goes to a declared `escape:` island. Check the
  deliberate non-primitives in [docs/design/ceiling.md](docs/design/ceiling.md)
  first: parity with another tool is not by itself a reason to add anything.

## House rules for the code itself

- **The engine imports no linopy.** The `lpspec.linopy` shim is a pure consumer
  of linopy's *public* API; where the two model the same concept, copy linopy's
  names and shapes and add a test that the copy still matches — never import.
- **Validate at load time**, with an error that names the construct and its
  rewrite.
- **Fix the type, do not widen it.** pyrefly runs `strict` with zero errors; a
  genuinely wrong finding is silenced on the one line with a reason.
- **Keep the dependency footprint minimal**: polars, numpy, pyparsing, pydantic,
  pyyaml, highspy — and no dataframe library beyond polars. pandas and xarray
  are bridges *out*, shipped with the `[linopy]` extra, and the bare-install job
  proves the engine never needs them.
- **The git tag is the version.** Never hardcode one in `pyproject.toml`.
