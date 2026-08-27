"""Every rung page shows the files that ran, byte for byte, and is current.

The projection, the script and the tables under `differential/pypsa/` are what
the parity runner wrote from the pinned math-spec; the `PyPSA parity` workflow
holds them to the corpus. Here, with no pypsa on the install, what is held is
the page: each fence is one of those files verbatim, and every page is what
`tools.ladder` prints from them and the certificate.
"""

from __future__ import annotations

import json

import pytest
import yaml

from tests.test_models_gallery import _fences
from tools import ladder

STEMS = ladder.stems()


def test_the_ladder_pages_are_current():
    assert ladder.main(['--check']) == 0, 'stale ladder pages — pixi run python -m tools.ladder'


def test_every_stamped_rung_has_a_page_and_a_projection():
    stamped = json.loads((ladder.LADDER / 'references.json').read_text())
    bound = {s for s, r in stamped.items() if 'unbound' not in r['parity']}
    assert bound == set(STEMS), 'a certified rung without a projection, or a projection no run certifies'
    missing = [s for s in STEMS if not (ladder.PAGES / f'{s}.md').exists()]
    assert not missing, f'rungs without a page: {missing}'
    index = ladder.INDEX.read_text()
    unlisted = [s for s in stamped if s not in bound and 'prep cannot bind' not in index]
    assert not unlisted, f'unbound rungs the index does not list: {unlisted}'


@pytest.mark.parametrize('stem', STEMS, ids=STEMS)
def test_the_lpspec_tab_shows_the_projection_that_solves(stem: str):
    fences = _fences((ladder.PAGES / f'{stem}.md').read_text(), 'yaml')
    assert (ladder.RUNGS / f'{stem}.yaml').read_text().rstrip() + '\n' in fences, 'the projected model has drifted'


@pytest.mark.parametrize('stem', STEMS, ids=STEMS)
def test_the_pypsa_tab_shows_the_script_that_builds_the_network(stem: str):
    fences = _fences((ladder.PAGES / f'{stem}.md').read_text(), 'python')
    assert (ladder.RUNGS / f'{stem}.py').read_text().rstrip() + '\n' in fences, 'the rung script has drifted'


@pytest.mark.parametrize('stem', STEMS, ids=STEMS)
def test_the_data_section_shows_the_tables_the_binding_produced(stem: str):
    fences = _fences((ladder.PAGES / f'{stem}.md').read_text(), 'csv')
    for path in sorted((ladder.LADDER / 'tables' / stem).glob('*.csv')):
        assert path.read_text().rstrip() + '\n' in fences, f'{path.name} is not embedded byte for byte'


def test_every_structure_difference_has_a_reason_and_is_on_the_index():
    stamped = json.loads((ladder.LADDER / 'references.json').read_text())
    reasons = yaml.safe_load((ladder.LADDER / 'deviations.yaml').read_text()) or {}
    index = ladder.INDEX.read_text()
    for stem in STEMS:
        for kind in ('structure', 'duals'):
            for name, d in stamped[stem]['parity'][kind]['differences'].items():
                assert d['reason'] and reasons.get(name, {}).get(kind) == d['reason'], (
                    f'{stem}: {kind} of {name} differ without a reason'
                )
                assert f'`{name}' in index and d['reason'] in index, f'{name} and its reason are not on the index'
        for name, reason in stamped[stem]['structural'].get('recorded', {}).items():
            assert reason and reasons.get(name, {}).get('blocks') == reason, (
                f'{stem}: the linopy-lane comparison records {name} without a reason'
            )
            assert f'`{name}' in index and reason in index, f'{name} and its reason are not on the index'


def test_every_rung_page_is_in_the_nav():
    """`mkdocs build --strict` refuses a page the nav does not list, and the nav is written by hand."""
    nav = (ladder.ROOT / 'mkdocs.yml').read_text()
    unlisted = [s for s in STEMS if f'examples/pypsa_ladder/{s}.md' not in nav]
    assert not unlisted, f'rung pages missing from the nav in mkdocs.yml: {unlisted}'


def test_the_reasons_file_names_each_pypsa_name_once():
    """A YAML loader keeps the last of two mappings for one key, so a second entry silently drops the first's reasons."""
    import re

    keys = re.findall(r'^([^\s#][^:\n]*):$', (ladder.LADDER / 'deviations.yaml').read_text(), flags=re.MULTILINE)
    doubled = sorted({k for k in keys if keys.count(k) > 1})
    assert not doubled, f'deviations.yaml names these more than once: {doubled}'


def test_the_index_lists_every_rung():
    text = ladder.INDEX.read_text()
    missing = [s for s in STEMS if f'pypsa_ladder/{s}.md' not in text]
    assert not missing, f'rungs the index does not list: {missing}'
