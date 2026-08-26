"""What the committed PyPSA certificate says about this engine, read from the file alone.

`differential/pypsa/corpus/references.json` carries two records per rung: what
PyPSA saw when `reference.py` solved the rung's network, and what the parity
runner (`differential/pypsa/parity.py`) stamped when it bound the same network
through both lanes — one objective and one set of prices across the fence,
block-level coverage, and the model-for-model verdict where `lpspec.linopy`
builds. The `PyPSA parity` workflow keeps the stamps current by failing when a
run of this tree would rewrite them; these tests hold what the stamps say,
with no solver and no pypsa, on every install.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from math_spec import load_model

CORPUS = Path(__file__).resolve().parent.parent / 'differential' / 'pypsa' / 'corpus'
RECORDED: dict[str, dict] = json.loads((CORPUS / 'references.json').read_text())
RUNGS = sorted(path.name for path in (CORPUS / 'data').iterdir() if path.is_dir() and path.name != 'base')
STEMS = sorted(RECORDED)

#: Model file (as the stamps name it) -> the loaded model and the rungs that bind it.
BINDINGS: dict[str, tuple] = {}
for _stem, _record in RECORDED.items():
    BINDINGS.setdefault(_record['parity']['model'], (load_model(CORPUS / _record['parity']['model']), []))[1].append(
        _stem
    )


def test_every_rung_folder_has_a_stamped_solve():
    assert set(RUNGS) == set(RECORDED), 'a rung folder without a record, or a record without a folder'


@pytest.mark.parametrize('stem', STEMS, ids=STEMS)
def test_both_lanes_solve_the_rung_to_one_objective(stem: str):
    parity = RECORDED[stem]['parity']
    assert parity['matches'], f'lpspec {parity["lpspec_objective"]} against pypsa {RECORDED[stem]["objective"]}'
    assert math.isclose(parity['lpspec_objective'], RECORDED[stem]['objective'], rel_tol=1e-9, abs_tol=1e-6), (
        'a re-recorded network with unrefreshed stamps would certify another network'
    )


@pytest.mark.parametrize('stem', STEMS, ids=STEMS)
def test_the_nodal_prices_are_pypsas_where_the_lane_prices(stem: str):
    prices = RECORDED[stem]['parity']['prices']
    if prices['compared']:
        assert prices['matches'], f'lpspec prices differ from marginal_price by up to {prices["max_abs_diff"]}'
    else:
        assert 'mixed-integer' in prices['skipped'], (
            'a rung without prices names the integer variables that undefine them'
        )


@pytest.mark.parametrize('stem', STEMS, ids=STEMS)
def test_the_two_linopy_models_are_one_or_the_blocker_is_named(stem: str):
    structural = RECORDED[stem]['structural']
    assert not structural.get('mismatch'), f'the two lanes build different models: {structural.get("mismatch")}'
    assert structural.get('error') or 'equal' in structural, 'a structural stamp carries a verdict or its blocker'


def _blocks():
    for rel, (model, stems) in BINDINGS.items():
        for kind, blocks in (('built_rows', model.constraints), ('built_columns', model.variables)):
            for name, block in blocks.items():
                yield rel, kind, name, block, stems


@pytest.mark.parametrize(
    ('rel', 'kind', 'name', 'block', 'stems'),
    list(_blocks()),
    ids=[f'{rel}:{name}' for rel, _, name, _, _ in _blocks()],
)
def test_every_block_is_built_by_some_rung(rel, kind, name, block, stems):
    """A declared block no rung builds is a silent regime: its rows have never been compared to anything."""
    assert sum(RECORDED[stem]['parity'][kind][name] for stem in stems), (
        f'no rung builds {name} of {rel} — extend a rung until its rows exist somewhere'
    )


@pytest.mark.parametrize(
    ('rel', 'kind', 'name', 'block', 'stems'),
    [entry for entry in _blocks() if entry[3].where],
    ids=[f'{rel}:{name}' for rel, _, name, block, _ in _blocks() if block.where],
)
def test_every_masked_block_is_partially_masked_somewhere(rel, kind, name, block, stems):
    """A `where:` no rung leaves half-true is untested as a mask — full or empty proves only all-or-nothing."""
    partial = any(
        0 < RECORDED[stem]['parity'][kind][name] < math.prod(RECORDED[stem]['parity']['dims'][d] for d in block.foreach)
        for stem in stems
    )
    assert partial, f'{name} of {rel} is always all-or-nothing — give some rung a label its mask excludes'


@pytest.mark.parametrize('rel', sorted(BINDINGS), ids=sorted(BINDINGS))
def test_every_parameter_is_bound_nonempty_by_some_rung(rel):
    """A parameter every rung leaves empty is data no comparison has ever weighed."""
    model, stems = BINDINGS[rel]
    fed = set().union(*(RECORDED[stem]['parity']['bound_nonempty'] for stem in stems))
    unfed = {*model.parameters, *model.lookups} - fed
    assert not unfed, f'no rung feeds these: {sorted(unfed)}'
