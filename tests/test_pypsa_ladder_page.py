"""The ladder page shows the tables lpspec was actually handed, and is current.

The tables are what `differential/pypsa/parity.py` wrote through
`tidy_sources` from math-spec's networks and binding; the `PyPSA parity`
workflow holds them to the corpus. Here, with no pypsa on the install, what
is held is the page: every CSV fence is a committed table byte for byte, and
the page is what the generator prints from those files and the certificate.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tests.test_models_gallery import _fences
from tools import ladder

if TYPE_CHECKING:
    from pathlib import Path

TABLES = sorted(ladder.LADDER.glob('tables/*/*.csv'))


def test_the_ladder_page_is_current():
    assert ladder.main(['--check']) == 0, 'stale ladder page'


@pytest.mark.parametrize('path', TABLES, ids=[f'{p.parent.name}/{p.name}' for p in TABLES])
def test_every_committed_table_is_on_the_page(path: Path):
    fences = _fences(ladder.PAGE.read_text(), 'csv')
    assert path.read_text().rstrip() + '\n' in fences, f'{path.parent.name}/{path.name} is not embedded byte for byte'


def test_the_binding_on_the_page_is_the_one_that_runs():
    fences = _fences(ladder.PAGE.read_text(), 'python')
    assert (ladder.LADDER / 'prep.py').read_text().rstrip() + '\n' in fences, 'prep.py on the page has drifted'


def test_every_rung_in_the_certificate_has_its_tables():
    rungs = set(json.loads((ladder.LADDER / 'references.json').read_text()))
    folders = {p.name for p in (ladder.LADDER / 'tables').iterdir() if p.is_dir()}
    assert rungs == folders, 'a stamped rung without its tables, or tables no rung stamps'
