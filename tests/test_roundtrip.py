"""``Model.to_yaml`` gives back the same model — held over the whole corpus.

The method is three lines; **this file is the deliverable**. A `to_yaml` that
drifts from what the engine builds is worse than none: a reviewer would be
reading a model that never ran, and nothing about the output would look wrong.

The ways it could drift are all quiet. A field gaining a default, a validator
normalising a value, an alias, a dict that stops preserving order — each turns
the dumped file into a *different* model while every existing test still
passes, because every existing test builds from the original.

So the property is checked against the corpus rather than a fixture: every
example and every ported model, which between them exercise every construct the
language has (`docs/models/index.md` generates the coverage table from exactly
this set).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import yaml as pyyaml

import lpspec as lps
from tests.conftest import MODEL_PATHS

if TYPE_CHECKING:
    from pathlib import Path


def test_the_corpus_is_not_empty():
    """A guard on the guard: the parametrised tests below pass vacuously if
    ``constructs.models()`` stops finding anything, which a directory rename
    would do silently."""
    assert len(MODEL_PATHS) >= 10, f'the model corpus looks wrong: {MODEL_PATHS}'


@pytest.mark.parametrize('path', MODEL_PATHS, ids=lambda p: p.stem)
def test_a_model_survives_a_round_trip(path: Path):
    """`load -> to_yaml -> load` is the same model, field for field."""
    original = lps.load_model(path)
    dumped = original.to_yaml()
    reloaded = lps.load_model(pyyaml.safe_load(dumped))

    assert reloaded.model_dump() == original.model_dump(), f'{path} does not survive a round trip'


@pytest.mark.parametrize('path', MODEL_PATHS, ids=lambda p: p.stem)
def test_the_two_out_forms_agree(path: Path):
    """`to_dict` is what `to_yaml` writes, so a caller cannot get two answers.

    The rule lives on the model's *serializer*, so pydantic's own methods carry
    it too. A helper beside them would have left `model_dump` — public, and not
    ours to remove — describing the same model with different content, and
    which one a consumer got would depend on which name they reached for.
    """
    model = lps.load_model(path)
    assert pyyaml.safe_load(model.to_yaml()) == model.to_dict()
    assert model.model_dump() == model.to_dict(), "pydantic's own dump has to agree too"
    assert lps.load_model(model.to_dict()).model_dump() == model.model_dump()


@pytest.mark.parametrize('path', MODEL_PATHS, ids=lambda p: p.stem)
def test_the_dump_is_stable(path: Path):
    """Dumping twice gives the same bytes.

    Not pedantry: an unstable dump means a framework that emits a model for
    review produces a different file on every run, so the diff a reviewer is
    supposed to read is noise.
    """
    once = lps.load_model(path).to_yaml()
    twice = lps.load_model(pyyaml.safe_load(once)).to_yaml()
    assert once == twice, f'{path} dumps differently the second time'


def test_a_dict_built_model_gets_a_file():
    """The case this exists for: a model that never had a file.

    #30 and #29 were closed in favour of frameworks building a dict and handing
    it over. This is what keeps that path honest against hard rule 5 — the dict
    gets a reviewable file, and it is the same model.
    """
    built = {
        'dimensions': {'t': {'dtype': 'int', 'values': [0, 1, 2]}},
        'parameters': {'cost': {'dims': ['t']}},
        'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'cap': {'foreach': ['t'], 'expression': 'x <= 4'}},
        'objectives': {'total': {'sense': 'maximize', 'expression': 'x * cost'}},
    }
    text = lps.load_model(built).to_yaml()

    assert lps.load_model(pyyaml.safe_load(text)).model_dump() == lps.load_model(built).model_dump()
    assert text.startswith('version: 0\n'), 'a generated file should say which surface it targets'
    assert 'dimensions:' in text
    assert 'piecewise' not in text, 'an absent section is absence, not a value'


def test_a_declared_version_survives():
    """`version:` is not a default when the file states it — a dumped model has
    to keep saying which surface it targets (#67)."""
    text = lps.load_model({'version': 0, 'dimensions': {'t': {'dtype': 'int', 'values': [0]}}}).to_yaml()
    assert 'version: 0' in text


@pytest.mark.parametrize('path', MODEL_PATHS, ids=lambda p: p.stem)
def test_the_review_copy_states_the_objective_sense(path: Path):
    """`sense` is emitted even at its default — the one word a reviewer must
    not have to infer.

    It round-trips either way, since absent means minimize. What it does not do
    either way is *read*: an objective with no direction makes the reviewer
    know a default to know whether the model minimises or maximises, and every
    model in the corpus writes it, so dropping it made the review copy differ
    from the file in the place that matters most.
    """
    model = lps.load_model(path)
    if not model.objectives:
        pytest.skip('no objective to state')
    text = model.to_yaml()
    for name, objective in model.objectives.items():
        assert f'sense: {objective.sense}' in text, f"{path}: '{name}' lost its direction"


def test_absence_is_dropped_and_values_are_kept():
    """The whole rule, both halves.

    Judging *defaults* would need a list of which ones matter, and that list is
    a second copy of the schema — it drifted on its first day, keeping `version`
    and `sense` while dropping `dtype`. Absence needs no list.
    """
    text = lps.load_model(
        {
            'dimensions': {'t': {'dtype': 'int', 'values': [0]}},
            'variables': {'x': {'foreach': ['t']}},
            'objectives': {'o': {'sense': 'minimize', 'expression': 'x'}},
        }
    ).to_yaml()

    for absent in ('where: null', 'coords: {}', 'macros:', 'piecewise:'):
        assert absent not in text, f'{absent!r} is absence and should not be written'
    for stated in ('dtype: int', 'sense: minimize', 'binary: false', 'version: 0'):
        assert stated in text, f'{stated!r} is a value and should be written'


def test_json_carries_a_model_too():
    """`model_dump_json` round-trips, because nothing infinite survives to it.

    JSON has no infinity, so an unbounded `-inf` bound used to come back as
    `null` and read as *absent*. It is absent — that is what an infinite bound
    means — so the serializer drops it and the two agree instead of one being
    quietly wrong. Held here because the fix is easy to undo by "restoring" a
    bound that was never information.
    """
    model = lps.load_model(
        {
            'dimensions': {'t': {'dtype': 'int', 'values': [0]}},
            'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0}}, 'y': {'foreach': ['t']}},
            'objectives': {'o': {'sense': 'minimize', 'expression': 'x + y'}},
        }
    )
    assert json.loads(model.model_dump_json()) == model.to_dict()
    assert lps.load_model(json.loads(model.model_dump_json())).to_dict() == model.to_dict()

    out = model.to_dict()['variables']
    assert out['x']['bounds'] == {'lower': 0.0}, 'a real bound stays, its infinite partner does not'
    assert 'bounds' not in out['y'], 'unbounded on both sides is no bounds block at all'
    assert lps.load_model(model.to_dict()).variables['y'].bounds.lower == float('-inf')
