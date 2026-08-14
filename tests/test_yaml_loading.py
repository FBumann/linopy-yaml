"""What the loader must refuse to do to a file before anyone else sees it."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

import lpspec as lps
from lpspec.language._yaml import read_yaml
from lpspec.language.validation import load_model
from tests.conftest import raw_of

MODEL = """dimensions:
  snapshot: {dtype: int, values: [0, 1]}
  generator: {values: [wind, gas]}
parameters:
  cost: {dims: [generator]}
variables:
  p:
    foreach: [snapshot, generator]
    where: "cost > 0"
    bounds: {lower: 0, upper: 100}
constraints:
  balance:
    foreach: [snapshot]
    expression: sum(p, over=generator) == 5
objective:
  expression: sum(p * cost, over=generator)
"""


def _write(tmp_path, text, name='m.yaml'):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_only_true_and_false_are_booleans(tmp_path):
    """YAML 1.1 resolved these to bools, so the rows they keyed silently vanished.

    ``no`` is Norway, ``on`` is Ontario, ``y``/``n`` are perfectly ordinary
    labels. Two 1.1 coercions deliberately survive — the implicit timestamp
    and sexagesimal ints — because both interact with the unimplemented
    ``dtype: datetime``. They belong to the dtype guard in #65.
    """
    path = _write(tmp_path, 'dimensions:\n  c: {dtype: str, values: [no, se, on, off, yes, n, y]}\n')

    assert read_yaml(path)['dimensions']['c']['values'] == ['no', 'se', 'on', 'off', 'yes', 'n', 'y']


def test_the_harness_reads_a_model_the_way_the_product_does(tmp_path):
    """``raw_of`` is the door every test walks through, and it was a different one.

    It read YAML 1.1, so a ``no`` label reached the schema as ``False`` while
    ``load_model`` saw the string — and a differential test on such a model
    compared two lanes that had loaded different files. The corpus has no
    ``no``/``yes``/``on``/``off`` label today, which is why nothing said so.
    """
    text = 'dimensions:\n  country: {dtype: str, values: [uk, de, no]}\n'
    path = _write(tmp_path, text)

    assert raw_of(path) == read_yaml(path), 'a path through the harness reads what the product reads'
    assert raw_of(text)['dimensions']['country']['values'] == ['uk', 'de', 'no'], (
        'and so does YAML text, which is the form most fixtures take'
    )


def test_real_booleans_still_parse(tmp_path):
    """The narrowed resolver keeps 1.2's `true`/`false` as booleans, not labels."""
    path = _write(tmp_path, 'flags:\n  a: true\n  b: false\n')

    assert read_yaml(path)['flags'] == {'a': True, 'b': False}


def test_the_loader_yields_plain_types(tmp_path):
    """No loader wrapper may reach the schema, the AST, the plan, or the engine."""
    raw = read_yaml(_write(tmp_path, MODEL))
    assert type(raw) is dict

    schema = load_model(raw)
    assert type(schema.dimensions['generator'].values) is list
    assert all(type(v) is str for v in schema.dimensions['generator'].values)
    assert type(schema.variables['p'].foreach) is list


def test_duplicate_key_is_an_error_naming_both_lines(tmp_path):
    """PyYAML keeps the last one, discarding a declaration the file contains."""
    path = _write(
        tmp_path, MODEL.replace('constraints:\n', 'constraints:\n  balance:\n    foreach: []\n    equations: []\n')
    )

    with pytest.raises(ValueError, match=r"duplicate key 'balance' .* first declared on line 12"):
        lps.check(path)


def test_duplicate_top_level_section_is_an_error(tmp_path):
    path = _write(tmp_path, MODEL + 'parameters:\n  other: {dims: [snapshot]}\n')

    with pytest.raises(ValueError, match="duplicate key 'parameters'"):
        lps.check(path)


def test_a_merge_key_override_is_not_a_duplicate(tmp_path):
    """`<<:` then a key of the same name is an override — the point of merging."""
    path = _write(
        tmp_path,
        'defaults: &d\n  foreach: [generator]\n'
        'dimensions:\n  generator: {values: [wind]}\n'
        'variables:\n  p:\n    <<: *d\n    foreach: [generator]\n',
    )

    assert read_yaml(path)['variables']['p']['foreach'] == ['generator']


def test_a_non_mapping_document_is_a_load_error(tmp_path):
    """Otherwise `Model(**raw)` raises a bare TypeError about `**`."""
    for text in ('- a\n- b\n', 'just a string\n'):
        path = _write(tmp_path, text)
        with pytest.raises(ValueError, match='must be a mapping of sections'):
            lps.check(path)


def test_an_empty_file_is_an_empty_model(tmp_path):
    assert read_yaml(_write(tmp_path, '')) == {}
    assert read_yaml(_write(tmp_path, '# only a comment\n')) == {}


@pytest.mark.parametrize(
    'name',
    [
        pytest.param('m.json', id='json'),
        pytest.param('M.JSON', id='suffix-case-insensitive'),
    ],
)
def test_a_json_model_is_the_same_model(tmp_path, name):
    """`.json` dispatches to the stdlib reader — a feature, not a YAML accident (#136)."""
    raw = read_yaml(_write(tmp_path, MODEL))
    path = _write(tmp_path, json.dumps(raw), name=name)

    assert load_model(path).model_dump() == load_model(raw).model_dump()


@pytest.mark.parametrize(
    'name',
    [
        pytest.param('m.toml', id='another-format'),
        pytest.param('m.yaml.txt', id='only-the-last-suffix-counts'),
        pytest.param('model', id='no-suffix-at-all'),
    ],
)
def test_an_unknown_suffix_is_refused_naming_the_supported_set(tmp_path, name):
    path = _write(tmp_path, MODEL, name=name)

    with pytest.raises(ValueError, match=r"suffix declares its format.*not one of '\.yaml', '\.yml' or '\.json'"):
        load_model(path)


@pytest.mark.parametrize(
    ('text', 'match'),
    [
        pytest.param('[1, 2]', 'must be a mapping of sections', id='a-list-is-not-a-model'),
        pytest.param('{"dimensions": ', 'not valid JSON', id='truncated-json'),
    ],
)
def test_a_json_document_that_is_not_a_model_is_a_load_error(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        load_model(_write(tmp_path, text, name='m.json'))


def test_a_missing_yaml_parser_names_the_extra(tmp_path):
    """Without pyyaml a `.yaml` path errors with the install line; `.json` still loads.

    A subprocess with the import blocked, because this suite runs with pyyaml
    installed — the bare-install CI job proves the same claim on a genuinely
    bare environment, at the dependency floors.
    """
    json_path = _write(tmp_path, json.dumps(read_yaml(_write(tmp_path, MODEL))), name='m.json')
    script = textwrap.dedent(f"""
        import sys

        class NoYaml:
            def find_spec(self, name, *args):
                if name == 'yaml' or name.startswith('yaml.'):
                    raise ModuleNotFoundError("No module named 'yaml'")

        sys.meta_path.insert(0, NoYaml())

        from lpspec.language.validation import load_model

        load_model({str(json_path)!r})
        try:
            load_model({str(json_path.with_suffix('.yaml'))!r})
        except ModuleNotFoundError as exc:
            assert 'pip install "lpspec[yaml]"' in str(exc), str(exc)
        else:
            raise AssertionError('a .yaml path loaded with the parser blocked')
        print('YAML_FREE_OK')
    """)
    out = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert 'YAML_FREE_OK' in out.stdout
