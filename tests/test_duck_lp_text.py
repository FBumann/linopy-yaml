"""The experiment's evidence: LP text rendered in SQL is the same LP text.

#399 asks whether the LP file can be written from inside duckdb, so the model
never crosses into polars on the way out. Whether that is *worth* doing is a
measurement and lives in the PR. Whether it is even the same file is this, and
it has to hold before any measurement means anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import polars as pl
import pytest

import lpspec as lps
from lpspec.relational.engines.duck.executor import DuckExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import duck_lp

PORTS = Path(__file__).resolve().parent.parent / 'examples' / 'ports'
REFERENCES: dict[str, dict[str, Any]] = json.loads((PORTS / 'references.json').read_text())


def _sources(name: str) -> dict[str, Any]:
    data = json.loads((PORTS / 'data' / f'{name}.json').read_text())
    return {k: pl.DataFrame(v) if isinstance(v, dict) else v for k, v in data.items()}


def _render_in_sql(ex: DuckExecutor, path: Path) -> None:
    tables = ex._tables()
    duck_lp.write(
        ex._con,
        path,
        col_tables=list(ex._col_tables),
        obj_tables=list(ex._obj_tables),
        row_tables=list(ex._row_shares),
        matrix_tables=[name for name, _, _ in ex._matrix_spans],
        vtype_runs=[(vtype, height) for _, vtype, height in ex._col_runs],
        sense=tables.objective_sense,
        constant=tables.objective_constant,
    )


@pytest.mark.parametrize('name', sorted(REFERENCES))
def test_the_sql_writer_renders_the_same_bytes(name: str, tmp_path: Path, duck_internals: None) -> None:
    """Every port, both writers, compared as bytes.

    The corpus is the point: it carries the sections a small model does not —
    ``binary`` and ``general`` for the four MILPs, termless rows, infinite
    bounds, an objective constant, and coefficients across enough magnitudes to
    reach both places duckdb's float rendering departs from polars'.
    """
    program = str(PORTS / f'{name}.yaml')
    with lps.build(program, _sources(name)) as ex:
        assert isinstance(ex, DuckExecutor), 'this run is not on the duckdb engine'
        ex.write(tmp_path / 'ordinary.lp')
        _render_in_sql(ex, tmp_path / 'sql.lp')
    ordinary = (tmp_path / 'ordinary.lp').read_bytes()
    assert (tmp_path / 'sql.lp').read_bytes() == ordinary, f'{name}: the SQL writer rendered different bytes'


@pytest.mark.parametrize(
    'value',
    [
        pytest.param(1e-7, id='padded-negative-exponent'),
        pytest.param(2.5e-8, id='padded-negative-exponent-with-mantissa'),
        pytest.param(1.2345678901234568e-5, id='the-decade-polars-keeps-positional'),
        pytest.param(-9.99e-5, id='the-same-decade-negative'),
        pytest.param(1e16, id='the-upper-switch-both-agree-on'),
        pytest.param(-0.0, id='negative-zero'),
    ],
)
def test_the_number_repairs_are_the_ones_that_matter(value: float, tmp_path: Path, duck_internals: None) -> None:
    """The two departures :mod:`duck_lp` repairs, each as its own case.

    A coefficient is where they reach the file, so the check is the rendered
    LP rather than the SQL: a repair that stopped working would show up here as
    a term a solver reads as a different number.
    """
    model = {
        'dimensions': {'i': {'dtype': 'int', 'values': [0]}},
        'parameters': {'c': {'dims': ['i']}},
        'variables': {'x': {'foreach': ['i'], 'bounds': {'lower': 0, 'upper': 1}}},
        'constraints': {'k': {'foreach': ['i'], 'expression': 'x * c >= 0'}},
        'objectives': {'o': {'sense': 'minimize', 'expression': 'sum(x, over=i)'}},
    }
    sources = {'c': pl.DataFrame({'i': [0], 'value': [value]})}
    with lps.build(model, sources) as ex:
        ex.write(tmp_path / 'ordinary.lp')
        _render_in_sql(ex, tmp_path / 'sql.lp')
    assert (tmp_path / 'sql.lp').read_bytes() == (tmp_path / 'ordinary.lp').read_bytes()
