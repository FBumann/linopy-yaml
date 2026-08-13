"""The data-preparation page shows files that exist and code that runs.

The page establishes the shared starting point every gallery tab assumes —
files, then one frame per parameter — so it is the one place a reader checks
whether the comparison is rigged. Three ways it could quietly become a lie:
the CSV shown drifts from the committed file, the committed file drifts from
the JSON instance the verification machinery reads, or the preparation code
stops producing tables lpspec accepts. One test each.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl
import pytest

import lpspec as lps

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / 'docs' / 'models' / 'data.md'
FOLDER = ROOT / 'examples' / 'ports' / 'data' / 'dispatch'


def _fences(lang: str) -> list[str]:
    return re.findall(rf'^```{lang}\n(.*?)^```', PAGE.read_text(), re.MULTILINE | re.DOTALL)


@pytest.mark.parametrize('name', ['generators', 'load'])
def test_the_page_shows_the_file_that_is_committed(name: str) -> None:
    text = (FOLDER / f'{name}.csv').read_text()
    assert text in _fences('csv'), f'docs/models/data.md has drifted from {name}.csv'


def test_the_preparation_code_runs_and_solves(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one fresh code block on the page, executed verbatim.

    Everything else the page says about frameworks points at tab content the
    gallery tests already byte-assert; this block is new, so it runs here —
    all the way to the optimum, because tables lpspec merely *accepts* could
    still be the wrong tables.
    """
    monkeypatch.chdir(ROOT)
    (block,) = _fences('python')
    scope: dict = {}
    exec(block, scope)  # the page's code, run as the reader would
    with lps.solve(str(ROOT / 'examples' / 'dispatch.yaml'), scope['sources']) as solution:
        assert solution.objective == pytest.approx(10500.0, rel=1e-9), (
            'the prepared sources do not reach the optimum references.json records for dispatch'
        )


def test_the_folder_and_the_instance_agree() -> None:
    """The CSV folder and dispatch.json carry the same instance.

    Two committed copies of one instance is the drift the rest of the corpus
    avoids; this page needs the entity shape and the machinery reads the JSON,
    so the copy is allowed — held equal here.
    """
    instance = json.loads((FOLDER.parent / 'dispatch.json').read_text())
    generators = pl.read_csv(FOLDER / 'generators.csv')
    for parameter in ('p_max', 'cost'):
        from_csv = generators.select('generator', pl.col(parameter).alias('value'))
        assert from_csv.equals(pl.DataFrame(instance[parameter])), f'{parameter} differs between CSV and JSON'
    assert pl.read_csv(FOLDER / 'load.csv').equals(pl.DataFrame(instance['load'])), 'load differs between CSV and JSON'
