"""The gallery's catalogue and its two evidence tables: which models there are,
what each exercises, and what somebody else says its answer is.

    pixi run python -m tools.constructs           # rewrite all three blocks
    pixi run python -m tools.constructs --check   # fail if any has drifted

The point of generating them is that a hand-kept table is the exact shape of
claim that rots: it is written once when it is true, and nothing fails when a
model changes underneath it. ``tests/test_models_gallery.py`` asserts the
committed blocks equal what this produces.

**The catalogue** is read off ``mkdocs.yml``'s nav and each page's own opening
line, so the list on the page and the list in the sidebar are one list. The nav
is already the complete one — ``strict: true`` with ``nav.omitted_files: warn``
fails the build on a page missing from it — which is what lets the section head
say *every* model. The hand-kept version it replaced had drifted to sixteen of
twenty-three.

**Constructs** are read off the **logical plan**, not the YAML text. Grepping
for ``shift(`` would count a construct inside a macro that never expands, miss
one a macro introduces, and disagree with itself about whether a bound written
as ``0`` is a bound. ``to_program`` needs no data, so the plan is available
for any model in the repo — and it is what the engine actually builds.

**References** are read off ``examples/ports/references.json``, which is the
same file ``tests/test_ports.py`` asserts against. That is the whole point: a
published optimum and an asserted optimum that can disagree is a correctness
claim with nothing behind it. Adding a port is a JSON entry and a regenerate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterator

from math_spec import program, to_program, to_spec, where_parser

ROOT = Path(__file__).resolve().parent.parent
GALLERY = ROOT / 'docs' / 'examples'
PAGE = GALLERY / 'index.md'
MKDOCS = ROOT / 'mkdocs.yml'
REFERENCES = json.loads((ROOT / 'examples' / 'ports' / 'references.json').read_text())
CAT_BEGIN, CAT_END = '<!-- catalogue:begin -->', '<!-- catalogue:end -->'
BEGIN, END = '<!-- constructs:begin -->', '<!-- constructs:end -->'
REF_BEGIN, REF_END = '<!-- references:begin -->', '<!-- references:end -->'

#: Column order is the order a reader meets these in docs/reference/language/, not
#: alphabetical and not by how many models happen to use them.
COLUMNS = ('sum', 'sum(by=)', 'at()', 'shift', "shift(edge='wrap')", 'where', 'bounds', 'piecewise', 'sos', 'MILP')


def walk(node: Any) -> Iterator[Any]:
    """Every dataclass node reachable from *node*, itself included.

    Structural rather than a visitor with a case per type: a new expression
    node then shows up in this table by existing, instead of by someone
    remembering to add it here.
    """
    if is_dataclass(node) and not isinstance(node, type):
        yield node
        for f in fields(node):
            yield from walk(getattr(node, f.name))
    elif isinstance(node, Mapping):
        for item in node.values():
            yield from walk(item)
    elif isinstance(node, tuple | list):
        for item in node:
            yield from walk(item)


def constructs(model: Path) -> set[str]:
    """The set of columns *model* exercises.

    ``shift`` and ``shift(edge='wrap')`` are two columns rather than one: the
    two spellings are the acyclic and the cyclic boundary, which is the
    distinction #330 was about, and ``wrap`` is what the node keeps them apart
    by.

    A bound counts as *declared* only where it is not the open default.
    Reading the plan rather than the YAML is what makes ``lower: 0`` and an
    omitted lower distinguishable.

    ``piecewise:`` is the one construct read off the surface schema: it lowers
    away into a lambda formulation, so by the time the plan exists there is
    nothing left to recognise. ``sos:`` does not — a set survives lowering as a
    declaration of its own, so it is read off the plan like the rest.
    """
    schema = to_spec(model)
    lowered = to_program(schema)
    nodes = list(walk(lowered))
    used: set[str] = set()

    for node in nodes:
        if isinstance(node, program.Sum):
            used.add('sum')
        elif isinstance(node, program.GroupSum):
            used.add('sum(by=)')
        elif isinstance(node, program.At):
            used.add('at()')
        elif isinstance(node, program.Translate):
            used.add("shift(edge='wrap')" if node.wrap else 'shift')

    if any(isinstance(n, where_parser.WhereNode) for n in nodes):
        used.add('where')
    if lowered.footprint.variable_types - {'continuous'}:
        used.add('MILP')
    if any(_bounded(v) for v in lowered.variables.values()):
        used.add('bounds')
    if schema.piecewise:
        used.add('piecewise')
    if lowered.sos:
        used.add('sos')
    return used


def _bounded(v: program.VariableDeclaration) -> bool:
    open_at = {float('-inf'): 'lower', float('inf'): 'upper'}
    for side in ('lower', 'upper'):
        bound = getattr(v, side)
        if not (isinstance(bound, program.Constant) and open_at.get(bound.value) == side):
            return True
    return False


class _NavLoader(yaml.SafeLoader):
    """``mkdocs.yml`` is not safe-loadable: its markdown extensions are
    configured with ``!!python/name:`` and ``!!python/object/apply:`` tags.

    Resolving them would mean importing pymdownx to read a nav, so every tag
    becomes ``None`` instead — the nav itself is plain lists and strings.
    """


_NavLoader.add_multi_constructor('tag:yaml.org,2002:python/', lambda *_: None)


def nav_groups() -> list[tuple[str, list[tuple[str, str]]]]:
    """The Examples section of the site nav: group title, then its pages.

    Each page is ``(label, name)`` — the label the sidebar shows and the
    model's name, which is also its page and its YAML file. An entry in the
    section that is not a group — the index itself — is skipped: it is not a
    model, and :func:`catalogue` is checked against :func:`models` for the ones
    that are.
    """
    nav = yaml.load(MKDOCS.read_text(), Loader=_NavLoader)['nav']
    section = next(entry['Examples'] for entry in nav if isinstance(entry, dict) and 'Examples' in entry)
    groups = []
    for entry in section:
        if not isinstance(entry, dict):
            continue
        ((title, target),) = entry.items()
        if isinstance(target, list):
            groups.append((title, [(label, Path(path).stem) for page in target for label, path in page.items()]))
    return groups


def _summary(name: str) -> str:
    """The opening line of a model's page, as one line.

    The page is where a model describes itself, so the catalogue quotes it
    rather than keeping a second description that can disagree with the first.
    Links inside it resolve unchanged — every gallery page is a sibling of the
    index.
    """
    _, _, body = (GALLERY / f'{name}.md').read_text().partition('\n')
    summary = next(block for block in body.split('\n\n') if block.strip())
    return ' '.join(summary.split())


def catalogue() -> str:
    """Every model, grouped and described exactly as the sidebar groups it."""
    blocks = []
    for title, pages in nav_groups():
        rows = '\n'.join(f'| [{label}]({name}.md) | {_summary(name)} |' for label, name in pages)
        blocks.append(f'### {title}\n\n| | |\n|---|---|\n{rows}')
    return '\n\n'.join(blocks)


def table(models: list[tuple[str, Path]]) -> str:
    """Markdown, one row per model, `·` where a construct is absent.

    A dot rather than an empty cell: an empty one reads as "not checked", and
    the holes in this table are the informative part.

    The ``verified`` badge means *external* verification, not "there is a
    test": every model here is exercised by the suite, and only the ported
    ones are checked against a number that did not come from us.
    """
    lines = [
        '| model | verified | ' + ' | '.join(f'`{c}`' if c != 'MILP' else c for c in COLUMNS) + ' |',
        '|---' * (len(COLUMNS) + 2) + '|',
    ]
    for name, path in models:
        used = constructs(path)
        cells = ['**✓**' if c in used else '·' for c in COLUMNS]
        badge = f'**✔** {REFERENCES[name]["objective"]:g}' if name in REFERENCES else '·'
        lines.append(f'| [{name}]({name}.md) | {badge} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def references_table() -> str:
    """One row per verified port, straight from ``references.json``.

    The optimum is written as ``repr`` rather than rounded: this is the number
    the assertion uses, and a table that rounds it is a different claim from
    the one the test makes.

    ``rtol`` is a column rather than a footnote even though every port shares
    one today — a footnote saying "all matched to 1e-09" becomes quietly false
    the first time one does not, and nothing would catch it.

    Corroboration runs to a paragraph, so it lands under the table as a
    footnote rather than in a cell. ``footnotes`` is on in ``mkdocs.yml``, and
    the repo view renders the same text as plain prose that still reads.
    """
    lines = [
        '| port | optimum | `rtol` | duals | reference |',
        '|---|---|---|---|---|',
    ]
    notes = []
    for name, entry in sorted(REFERENCES.items()):
        duals = '**✔**' if entry.get('duals') else '·'
        mark = f'[^{name}]' if entry.get('corroborated_by') else ''
        lines.append(
            f'| [{name}]({name}.md) | {entry["objective"]!r} | {entry["rtol"]:g} | '
            f'{duals} | {entry["provenance"]}{mark} |'
        )
        if corroborated := entry.get('corroborated_by'):
            notes.append(f'[^{name}]: {corroborated}')
    return '\n'.join(lines) + ('\n\n' + '\n\n'.join(notes) if notes else '')


def ports() -> list[Path]:
    """The ported models — somebody else's model, against somebody else's optimum."""
    return sorted((ROOT / 'examples' / 'ports').glob('*.yaml'))


def models() -> list[tuple[str, Path]]:
    """Every model the gallery shows, examples before ports."""
    examples = sorted((ROOT / 'examples').glob('*.yaml'))
    return [(p.stem, p) for p in examples] + [(p.stem, p) for p in ports()]


def replace_between(page: str, begin: str, end: str, body: str) -> str:
    """*page* with the generated block between two markers replaced by *body*."""
    i, j = page.index(begin) + len(begin), page.index(end)
    return page[:i] + '\n' + body + '\n' + page[j:]


def rendered(page: str) -> str:
    """*page* with all three generated blocks replaced."""
    page = replace_between(page, CAT_BEGIN, CAT_END, catalogue())
    page = replace_between(page, BEGIN, END, table(models()))
    return replace_between(page, REF_BEGIN, REF_END, references_table())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='fail if any committed block has drifted')
    opts = ap.parse_args(argv)

    page = PAGE.read_text()
    updated = rendered(page)
    if opts.check:
        if updated != page:
            print(f'{PAGE} is stale — run `pixi run python -m tools.constructs`', file=sys.stderr)
            return 1
        print(f'{PAGE} matches the models')
        return 0
    PAGE.write_text(updated)
    print(f'{PAGE} refreshed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
