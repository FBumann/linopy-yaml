# Docs

This folder is both the published site and what you read on GitHub.
[index.md](index.md) is the site's front door and this page is the folder view;
start at [running a model](guide.md), then
[the rules](https://math-spec.readthedocs.io/en/latest/reference/language/#ten-rules-the-language-reduces-to) for
what may be in the file.

**The layout is the reader's path**, and the site's three sections are this
package's three subjects: the **data** it attaches (`examples/data.md`,
`reference/data.md`), the **API** that attaches it (`reference/api.md`,
`reference/sweeps.md`, the two notebooks) and the **examples** that do both.
The language is a dependency, documented with itself, so the nav links out to
math-spec rather than keeping a second copy. Everything else — design notes,
measured cost, project direction, changelog — is under `about/`, reachable and
out of the way. A page that argues for a decision belongs there; a page that
states a rule belongs in `reference/`.

Two link rules make one set of files serve both places, and
`tests/test_docs_site.py` enforces them: **inside `docs/`, link relatively**;
**outside it, write the full GitHub URL** — the relative form resolves in the
repo and 404s on the site, silently. The rest is in *the docs* in
[CONTRIBUTING.md](../CONTRIBUTING.md#the-docs).

**Generated, so do not hand-edit:** the catalogue, the construct matrix and the
reference table in [examples/index.md](examples/index.md) (`tools/constructs.py`),
the *"the same model, as math"* block on each model page
(`tools/gallery_math.py`), and the tables in
[benchmarks.md](about/benchmarks.md) (`bench.report`, `bench.plot`). The
catalogue is read off `mkdocs.yml`'s nav,
so a model is added to the gallery list by adding it to the sidebar — one list,
not two.
The YAML and Python shown on the model pages, and the model in `README.md`
that [index.md](index.md) includes, are asserted against the files that run, so
a page cannot quietly drift from what it describes. The guide shows no model
YAML at all now: what a file may contain is the language's to state, and it
states it upstream.

**What stays hand-written, and what checks it.** A model page opens with a
summary — a sentence and, on six pages, the math stated the way that problem is
usually written. The sentence is the model's one description anywhere: the
gallery catalogue quotes it rather than keeping a second one. It is allowed to
be loose in a way the generated block beneath it is not: it is read at a
glance, and three summaries had drifted far enough to be wrong before the block
existed to check them against. So the
looseness is bounded rather than assumed. `tests/test_typeset_gallery.py` requires each
of those six to **either** use only symbols the generator can reach — the
hand-written notation is then an oracle *for* the typesetter, since the point
of the format is that a gallery page could be generated — **or** name why it
deviates. `tsp_mtz` states DFJ, the formulation the language refuses; `storage`
writes `soc_{s-1}` where the model rolls and the generator writes the cyclic
`⊖`. A page in neither list fails, so a new summary cannot quietly opt out.
