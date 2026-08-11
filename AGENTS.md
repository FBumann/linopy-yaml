# AGENTS.md

House preferences: how code, docstrings, commits, PRs and issues are written
here. **What** the project is lives in [docs/](docs/README.md); **which command
to run** lives in [CONTRIBUTING.md](CONTRIBUTING.md). Every rule below is here
because its absence cost a wrong number, a lost diff, or a round trip.

## Code

**Cut lines. Remove duplication. Apply YAGNI.** Delete the abstraction with one
caller, the option nobody sets, the branch nothing reaches. A cleanup pass whose
output is a defensive rename rather than fewer lines and fewer concepts gets
sent back — if the cut turned out smaller than it looked, say why in one line.

**Breaking changes are free.** The project is `0.0.1aN` and holds no
compatibility promise. Asked to change something, change it: rename, move,
delete. No alias, no deprecation cycle, no `legacy_` path beside the new one, no
back-compat nobody requested. Spend that effort on the error instead — a retired
spelling should fail at load naming its rewrite.

**A test asserting the old behaviour is not a blocker.** Rewrite or delete it if
that makes the codebase simpler, and say in the PR what coverage moved where.

**No explanatory inline comments.** Complex logic becomes a helper whose
docstring carries the constraint; a rule the code cannot show moves to the
nearest docstring, and everything else moves to the PR.

```python
# no
solver(solver_name)  # before the build, for the reason `write` checks the suffix first

# yes — the sentence moved into solve()'s docstring, the call reads as itself
solver(solver_name)
```

Kept inline: pragmas (`# pyrefly: ignore[…]` with its reason, `# fmt: skip`),
`#:` attribute docs, section banners. Nothing else.

Also standing: the engine imports **no linopy** (the shim is a pure consumer of
its public API — copy names and shapes, test the copy, never import); validate
at **load time** with an error naming the construct and its rewrite; **fix the
type, do not widen it** (pyrefly `strict`, zero errors); keep the dependency set
at polars, numpy, pyparsing, pydantic, pyyaml, highspy and no second dataframe
library; the **git tag is the version**, never hardcoded.

## Docstrings

Informative, concise, and about the reader's problem — not the author's.
[#592](https://github.com/fluxopt/lpspec/pull/592) cut 4,012 docstring lines to
3,474 (42% → 39% of non-blank source) and named the three kinds it removed:

| cut | example of the shape |
|---|---|
| **restatement** | a message builder explaining at length what the message it returns already says |
| **argument for a settled decision** | two paragraphs defending a choice the first sentence stated |
| **narration of how the answer was found** | the trials and measurement runs that led here, rather than the number they produced |

What earns its lines: the rule, the invariant, the non-obvious reason something
is safe, and a `#nnn` when the argument is longer than the docstring should be.
`relational/chunking.py`'s module docstring is the model — one rule, then the
trap that makes the rule necessary, then the scope it does *not* claim.

**Measured numbers do not live in the tree.** They belong in the PR description
that produced them, where the method, the base commit and the ladder are next to
them; a number copied into a docstring loses all three and ages silently. Say
the conclusion and point at the PR — *"ordered joins were measured slower on
masked models (#581)"* — and the numbers stay one click away, for as long as
they are worth having.

## Commit messages and PR titles

`main` takes squash merges, so **the PR title becomes the commit subject** and
release-please parses it into the changelog. One conventional-commit line:
`type(scope): subject`, types `feat fix perf refactor docs` (shown) plus
`chore test ci build style revert` (hidden). **No `!`, no `BREAKING CHANGE:`
footer** — the check refuses both while the version is pinned to the alpha
stream.

The subject names **the problem solved**, in the indicative present, and stops.
It is read in `git log --oneline` and in the changelog by people who will never
open the diff, so: scannable, one line, no mechanism, no "why", no trailing
clause.

```
yes  fix(api): a closed result says it was closed
yes  feat(language): a coordinate may declare its own label space
yes  fix(data): an empty index keeps the dimension's declared dtype
yes  perf(engine): bounds stop dominating build on wide models

no   perf(engine): the bound attach reads the ordinal off the Enum, not a dictionary
     ^ how it was done — belongs in the description
no   perf(engine): the objective keeps the hash count — bought order was a shape-dependent bet
     ^ mechanism plus the argument for it; two descriptions in a title
no   fix(data): fix dtype bug
     ^ names an activity, not an outcome
```

For a `perf` change, the outcome is what got cheaper and for whom — not which
call changed. **Mechanism, rationale, numbers and alternatives go in the body.**
Nothing transitive lives in the code: "previously this used to…", "renamed
from…", "as of the polars rewrite…" belong in the commit, the PR, or docs.

## PR descriptions

Two readers: the one deciding whether to merge today, and the one who reaches it
through `git blame` in six months. Neither wants a log of your session.

- **Lead with the claim**, then the evidence. [#592](https://github.com/fluxopt/lpspec/pull/592)
  and [#581](https://github.com/fluxopt/lpspec/pull/581) are the pattern.
- **This is where numbers live**, and the only place. Each carries the base
  commit it was taken against, what is counted (build vs solve, LP file vs
  direct sink), and how it was taken. A rebase invalidates them — re-run or
  remove, never leave standing.
- **Say what was verified**: gates run, test counts, and what you could *not*
  check.
- **Name what you deliberately did not do**, and why.
- **One issue, one PR.** Separable work is a stacked PR, not extra commits.
  Split rather than bundle.

## Branch, worktree, and the other agents

`main` moves several times a day, often while you work, and the git status in
your prompt is a **snapshot from before you started** — once it read one branch
while `HEAD` was a different, unmerged one, and a doc PR landed on top of
somebody else's work.

```bash
git fetch origin && git branch --show-current
git worktree add ../wt/<topic> -b <type>/<topic> origin/main
```

- Branch from `origin/main`, not from the tree you are sitting in. One topic,
  one worktree.
- Verifying a claim about shipped behaviour? Verify it against `origin/main` —
  the local tree may carry unmerged code that makes it true only here.
- **Several sessions share this checkout.** Commit early rather than holding a
  large working-tree diff; if files change under you, stop and say so instead of
  untangling whose edit is whose; never push a branch you did not create, and
  never while another agent is mid-run.
- Before rebasing or reviving anything, `gh pr view` it — it may already be
  merged, closed or superseded.

**Two gates pass locally and fail in CI**, both from a worktree: `pyrefly check`
silently skips gitignored directories and **exits 0** (use
`uv run pyrefly check $(git ls-files 'src/**/*.py')`), and `ruff format --check`
must run on `.`, never on the changed `.py` files — it formats python inside
markdown, so a docs edit fails a gate every `.py` file passes.

## Finishing

The most-repeated correction in this repo is one word: *pushed?* Committed,
pushed, PR open, CI green — then report, with the **URL**. "I'll open the PR" in
a summary is not an open PR, and a CI failure you did not wait for becomes the
user's next message. **The user merges**: never merge, force-push, or delete a
branch you did not create.

## Measurements

Performance is a product claim here, so:

- **Measure on an idle machine.** A ladder taken while the laptop was busy
  inflated one case by 55%, inventing a regression.
- **A/B against the same base**, same sizes, same process, and re-run when the
  two are close enough that the ranking could flip. Noise is the default
  explanation for a small delta.
- **Never retype a number** — `bench.report` and `bench.plot` regenerate the
  tables and charts from the results file.
- **Do not take the user's numbers on faith either**: *"my benchmarks are
  preliminary — check for yourself"* is standing instruction.

## Docs

A stale sentence outranks a correct implementation in every reader's head — the
spec once still described a router with fallback triggers that no longer
existed, and readers concluded the no-fallback claim was aspirational.

- Adding, renaming or retiring a construct updates [docs/SPEC.md](docs/SPEC.md)
  — §0 if it changes a law, the section if it changes a detail. Changing
  structure updates [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), diagrams
  included.
- **Reference pages carry rules, design notes carry arguments.** A change that
  makes one longer usually belongs in the other.
- **When a decision is made in conversation, sweep for what now contradicts it.**
  Half-updated docs are the failure mode.
- Stale *rationale* is corrected even when the thing it justified is done: a
  number from a closed issue should not stay quotable.
- Links inside `docs/` are relative; a link to anything outside `docs/` is a
  full GitHub URL — the relative form resolves in the repo and 404s on the site,
  silently.

## Issues

File against **behaviour and a file**, never a private symbol and a line range —
four issues have had to be re-filed for exactly that — and **label them**. An
issue whose body a rewrite invalidated is closed and re-filed against current
code, not edited or annotated; coverage checklists and scope decisions stay open
with a status comment.

Review findings, issue bodies and your own earlier analysis are **claims to
check**, not work orders: re-read the current code, fix what is still valid,
skip the rest with a one-line reason, and say which you skipped.

## Working with the user

- **"Discuss", "should we", "is it worth" are questions.** Answer them and stop;
  building before the answer lands wastes the answer.
- **Do not widen scope.** Touching an adjacent PR or issue unasked is the
  correction, every time. Finish what was asked, then name the adjacent thing in
  one sentence.
- **Recommend, do not survey.** A ranked answer with its reason beats an
  even-handed list.
- **Before proposing a language feature, triage it: macro, primitive, or
  escape?** Most requests are compositions (macro, free). A new shape earns a
  primitive only if it clears the ceiling — degree 1 ∩ relational ∩ local.
  Unsayable math goes to a declared `escape:` island. Read the deliberate
  non-primitives in [docs/design/ceiling.md](docs/design/ceiling.md) first:
  parity with another tool is not by itself a reason to add anything.
