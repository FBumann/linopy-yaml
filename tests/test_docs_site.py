"""The docs are read in two places; these are the checks that keep them honest in both.

``docs/`` is browsed on GitHub and served as a site, from one set of files. A
link *inside* ``docs/`` is relative and mkdocs validates it — ``build --strict``
in CI fails on a dead one. A link *outside* ``docs/`` cannot be relative,
because the site has no `../CONTRIBUTING.md` to resolve to, so it is written as
a full GitHub URL.

That convention is the whole mechanism, and it is unenforceable by mkdocs in
both directions: a relative link escaping ``docs/`` builds a silent 404, and a
blob URL is opaque to every checker there is — the file it names can be deleted
and nothing anywhere fails. Hence this module.

``docs/README.md`` is the one exception and is exempted throughout: it is
excluded from the site (``exclude_docs``), exists only as the folder view
GitHub renders, and its relative links out of the tree are correct there.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / 'docs'
REPO_URL = 'https://github.com/fluxopt/lpspec'
BLOB = f'{REPO_URL}/blob/main'

#: `](target)` and `[label]: target`, the two ways markdown names a destination.
_TARGETS = re.compile(r'\]\(\s*([^)\s]+)|^\[[^\]]+\]:\s+(\S+)', re.MULTILINE)

#: Already absolute, a bare fragment, or a protocol that names no path.
_ABSOLUTE = re.compile(r'^([a-z][a-z0-9+.-]*:|//|#|/)', re.IGNORECASE)


@functools.cache
def _pages() -> tuple[Path, ...]:
    """Every page mkdocs builds — so, not `docs/README.md`.

    A notebook is one of them: mkdocs-jupyter renders it into the site and
    GitHub renders it in the tree, so a link in a markdown cell is read in both
    places and lives under exactly the convention above.
    """
    pages = (*DOCS.rglob('*.md'), *DOCS.rglob('*.ipynb'))
    return tuple(p for p in sorted(pages) if p.relative_to(DOCS).as_posix() != 'README.md')


def _all_pages() -> tuple[Path, ...]:
    """The site's pages plus `docs/README.md`, the folder view GitHub renders."""
    return (*_pages(), DOCS / 'README.md')


def _prose(page: Path) -> str:
    """What a reader sees: the whole file, or a notebook's markdown cells."""
    if page.suffix != '.ipynb':
        return page.read_text()
    cells = json.loads(page.read_text())['cells']
    return '\n'.join(''.join(cell['source']) for cell in cells if cell['cell_type'] == 'markdown')


def _targets(page: Path) -> list[str]:
    return [inline or reference for inline, reference in _TARGETS.findall(_prose(page))]


def test_no_relative_link_escapes_the_docs_tree():
    """The failure mkdocs cannot see.

    `[x](../CONTRIBUTING.md)` is correct in the repo and a 404 on the site.
    mkdocs resolves it against `docs/`, finds nothing above the root, and —
    because the target is outside the tree it knows about — does not treat it
    as a broken internal link. It just ships. Write the full GitHub URL.
    """
    escaping = []
    for page in _pages():
        for target in _targets(page):
            if _ABSOLUTE.match(target):
                continue
            path = target.partition('#')[0]
            if not path:
                continue
            resolved = (page.parent / path).resolve()
            if resolved != DOCS and DOCS not in resolved.parents:
                escaping.append(f'{page.relative_to(REPO)} -> {target}')
    assert not escaping, (
        f'relative links pointing outside docs/, which 404 on the site: {escaping}\nwrite them as {BLOB}/<path> instead'
    )


def test_every_blob_url_names_a_file_that_exists():
    """The other half: a blob URL is checked by nothing at all.

    mkdocs treats it as external and never follows it; the repo has no reason
    to notice it. So a page can go on pointing at `bench/results/latest.jsonl`
    long after the file moves, and the first report is a reader hitting
    GitHub's 404.
    """
    broken = []
    for page in _all_pages():
        for target in _targets(page):
            if not target.startswith(BLOB):
                continue
            relative = target.removeprefix(f'{BLOB}/').partition('#')[0]
            if not (REPO / relative).exists():
                broken.append(f'{page.relative_to(REPO)} -> {relative}')
    assert not broken, f'links to repo files that no longer exist: {broken}'


def test_links_to_our_own_files_are_all_spelled_the_same_way():
    """One spelling, so the check above cannot be dodged.

    A link at a file in this repo written any other way — `tree/`, `raw/`, a
    permalinked sha, a branch that will vanish — reaches the right page today
    and is skipped by the existence check, which only recognises `blob/main`.
    Issue and PR links are not file links and are left alone.
    """
    file_shaped = re.compile(rf'^{re.escape(REPO_URL)}/(blob|tree|raw|blame)/')
    stray = [
        f'{page.relative_to(REPO)} -> {target}'
        for page in _all_pages()
        for target in _targets(page)
        if file_shaped.match(target) and not target.startswith(f'{BLOB}/')
    ]
    assert not stray, f'links at repo files not written as {BLOB}/<path>: {stray}'


def test_the_convention_is_actually_in_use():
    """A guard on the guards.

    Every assertion above passes vacuously on a docs tree with no outbound
    links at all — including one where a bad refactor stripped them. Pin that
    the arrangement they describe exists.
    """
    urls = [t for page in _pages() for t in _targets(page) if t.startswith(BLOB)]
    assert len(urls) >= 15, f'expected the docs to link out to the repo; found {len(urls)}'


def test_the_home_page_still_carries_its_math_block():
    """`tools.gallery_math --check` also fills the tabs on `docs/index.md`, and
    it fills what it finds — a page whose markers were dropped in an edit stops
    being checked without anything failing. Pin that they are there.

    The content itself is not asserted here; that is the generator's job, and
    `test_the_gallery_math_is_current` runs it.
    """
    from tools import gallery_math

    page = (DOCS / 'index.md').read_text()
    assert gallery_math.HOME_BEGIN in page and gallery_math.HOME_END in page, (
        f'docs/index.md lost its {gallery_math.HOME_BEGIN}/{gallery_math.HOME_END} markers — '
        f'the LaTeX tabs are generated, and an unmarked page silently opts out'
    )


# --------------------------------------------------------------------------
# the ten rules, and the pages that elaborate them
# --------------------------------------------------------------------------

LANGUAGE = DOCS / 'reference' / 'language'
RULES = LANGUAGE / 'index.md'

#: A rule row: `| 7 | text | [Absence](absence.md#how-absence-travels) |`
_RULE_ROW = re.compile(r'^\|\s*(\d+)\s*\|(.+?)\|([^|]*)\|\s*$', re.MULTILINE)


def _rules() -> list[tuple[str, str, str]]:
    text = RULES.read_text()
    start = text.index('## Ten rules the language reduces to')
    return _RULE_ROW.findall(text[start : text.index('\n## The pages', start)])


def _headings(page: Path) -> set[str]:
    """Every heading in *page* as GitHub would slug it."""
    slugs = set()
    for line in page.read_text().splitlines():
        if line.startswith('#'):
            title = line.lstrip('#').strip()
            slugs.add(re.sub(r'[^a-z0-9 -]', '', title.lower()).replace(' ', '-'))
    return slugs


def test_every_rule_cites_the_page_that_elaborates_it():
    """A rule is the canonical statement and a page below is the detail.

    An unlinked rule is the failure that would otherwise pass silently: mkdocs
    fails the build on a *dead* anchor, but a row that cites nothing at all
    resolves fine and quietly becomes a second, drifting home for the rule.
    Splitting the reference across pages adds the other half — a citation whose
    *page* moved — so both are checked here.
    """
    rules = _rules()
    assert len(rules) >= 10, f'expected the rule block to be found and populated; got {len(rules)} rows'

    broken = []
    for number, _, citation in rules:
        targets = re.findall(r'\]\(([a-z0-9_.-]+\.md)(?:#([a-z0-9-]+))?\)', citation)
        if not targets:
            broken.append(f'rule {number} cites no page')
        for page, anchor in targets:
            target = LANGUAGE / page
            if not target.is_file():
                broken.append(f'rule {number} -> {page} (no such page)')
            elif anchor and anchor not in _headings(target):
                broken.append(f'rule {number} -> {page}#{anchor}')
    assert not broken, f'rules whose citation does not resolve under {LANGUAGE.relative_to(REPO)}: {broken}'


# --------------------------------------------------------------------------
# the operators as math


def test_the_operator_math_is_current():
    """The generated block equals what the probe models render.

    The same bargain the gallery makes: a page showing a model is a copy, and a
    copy rots unless something asserts it. Here the copy is one equation per
    operator, and what it would rot into is a reference page describing an
    operator the language stopped having.
    """
    from tools import spec_math

    assert spec_math.main(['--check']) == 0, 'stale operator math'


def test_every_operator_in_the_table_has_a_probe():
    """The operator table and the math block are the same list, in the same order.

    "As math" says it shows *each row above*, and nothing else makes that
    true: an operator added to the language and to the table, but given no
    probe, would leave a section quietly claiming to be all of them. The order
    is asserted too — the prose table is the order a reader meets them, and two
    tables that disagree about it are read as two different sets.
    """
    from tools import spec_math

    assert spec_math.table_operators() == list(spec_math.OPERATORS), (
        f'{spec_math.PAGE.name} and tools/spec_math.OPERATORS name different operators, '
        'or name them in a different order — every row of the operator table needs '
        'a probe in examples/operators/, and every probe needs its row'
    )


def test_no_probe_without_a_row():
    """The reverse: a model in `examples/operators/` that the page never shows."""
    from tools import spec_math

    orphans = sorted(p.stem for p in spec_math.PROBES.glob('*.yaml') if p.stem not in set(spec_math.OPERATORS.values()))
    assert not orphans, f'operator probes nothing renders: {orphans}'
