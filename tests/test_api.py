"""Native API: YAML → streaming engine → solver, with linopy never imported.

The linopy-free guarantee is asserted in a subprocess so conftest's optional
lpspec_linopy import cannot pollute the check.

This module is deliberately **pandas-free**: it is the bare install's proof
that the native path — frames in, build, solve, frames out — needs no
dataframe library beyond the engine's own. The tests that exercise the bridges
*out* (``to_pandas``, ``to_dataarray``) say so with an ``importorskip``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from unittest import mock

import numpy as np
import polars as pl
import pytest

import lpspec as lps
from lpspec.language.model import Model
from tests.conftest import (
    DISPATCH_COST,
    DISPATCH_GENERATORS,
    DISPATCH_P_MAX,
    _dispatch_load,
    schema_of,
    solve_written_file,
)


@pytest.fixture
def dispatch_solution(dispatch_yaml, dispatch_frame_inputs):
    """The dispatch model solved on the native lane, closed after the test."""
    sources = dispatch_frame_inputs
    with lps.solve(dispatch_yaml, sources) as result:
        yield result


def test_solve(dispatch_solution, dispatch_frame_inputs):
    sources = dispatch_frame_inputs
    assert dispatch_solution.is_ok
    assert np.isfinite(dispatch_solution.objective)
    balance = dispatch_solution.primal('p').group_by('snapshot').agg(pl.col('value').sum()).sort('snapshot')
    assert np.allclose(balance['value'], sources['load']['value'])


def test_build_context_manager_and_write(dispatch_yaml, dispatch_frame_inputs, tmp_path):
    sources = dispatch_frame_inputs
    with lps.build(dispatch_yaml, sources) as bound:
        result = bound.solve()
        assert result.is_ok
        objective_direct = result.objective

    lp = lps.write(dispatch_yaml, sources, tmp_path / 'm.lp')
    assert solve_written_file(lp) == pytest.approx(objective_direct, rel=1e-9)


def test_parquet_path_sources(dispatch_yaml, dispatch_frame_inputs, tmp_path):
    sources = dispatch_frame_inputs
    paths = {}
    for name, frame in sources.items():
        p = tmp_path / f'{name}.parquet'
        frame.write_parquet(p)
        paths[name] = str(p)

    with lps.solve(dispatch_yaml, paths) as result:
        assert result.is_ok
        objective = result.objective

    with lps.solve(dispatch_yaml, sources) as ref:
        assert objective == pytest.approx(ref.objective, rel=1e-9)


#: ``examples/dispatch.yaml``'s numbers written out in Python rather than
#: handed over as tables — the shapes a hand-written model reaches for.
_PLAIN = {
    'dict': {
        'p_max': dict(zip(DISPATCH_GENERATORS, DISPATCH_P_MAX, strict=True)),
        'cost': dict(zip(DISPATCH_GENERATORS, DISPATCH_COST, strict=True)),
        'load': dict(enumerate(_dispatch_load())),
    },
    'sequence': {
        'p_max': list(DISPATCH_P_MAX),
        'cost': list(DISPATCH_COST),
        'load': _dispatch_load(),
    },
}


@pytest.mark.parametrize('shape', sorted(_PLAIN), ids=sorted(_PLAIN))
def test_plain_python_sources_reach_the_same_answer_as_tables(dispatch_yaml, dispatch_frame_inputs, shape):
    """A dict and a sequence are sources, and mean what the tables mean.

    A dict carries its own labels; a sequence is positional against the index,
    which is why the dimensions are resolved before any parameter is read.
    """
    frames = dispatch_frame_inputs
    with lps.solve(dispatch_yaml, frames) as tables:
        expected = tables.objective

    with lps.solve(dispatch_yaml, _PLAIN[shape] | {'snapshot': frames['snapshot']}) as plain:
        assert plain.objective == pytest.approx(expected, rel=1e-9)


def test_one_number_stands_for_every_coordinate(dispatch_yaml, dispatch_frame_inputs):
    """A scalar covers the dims the parameter declares, not just a 0-D one.

    Dense by construction, and materialised here — which is the cost of saying
    it this way rather than declaring the parameter ``dims: []``.
    """
    frames = dispatch_frame_inputs
    flat = {**frames, 'cost': 7.0}
    spelled = {**frames, 'cost': pl.DataFrame({'generator': list(DISPATCH_GENERATORS), 'value': [7.0] * 3})}

    with (
        lps.solve(dispatch_yaml, flat) as broadcast,
        lps.solve(dispatch_yaml, spelled) as written,
    ):
        assert broadcast.objective == pytest.approx(written.objective, rel=1e-9)


@pytest.mark.parametrize(
    ('sources', 'match'),
    [
        pytest.param({'p_max': [100.0, 60.0]}, 'one entry per label', id='a-sequence-of-the-wrong-length'),
        pytest.param({'p_max': object()}, 'cannot adapt', id='nothing-table-shaped-at-all'),
    ],
)
def test_a_plain_python_source_that_does_not_fit_is_refused(dispatch_yaml, dispatch_frame_inputs, sources, match):
    frames = dispatch_frame_inputs
    with pytest.raises(lps.DataError, match=match):
        lps.build(dispatch_yaml, {**frames, **sources}).close()


#: One parameter over two dims — what a dict and a sequence cannot cover.
_TWO_DIMS = {
    'dimensions': {'g': {'values': ['wind', 'gas']}, 't': {'dtype': 'int', 'values': [0, 1]}},
    'parameters': {'cap': {'dims': ['g', 't']}},
    'variables': {'x': {'foreach': ['g', 't'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
}


@pytest.mark.parametrize(
    ('source', 'match'),
    [
        pytest.param({('wind', 0): 1.0}, 'a dict maps one label to one value', id='a-dict'),
        pytest.param([1.0, 2.0, 3.0, 4.0], 'a sequence runs along one dimension', id='a-sequence'),
    ],
)
def test_a_flat_shape_cannot_cover_two_dimensions(source, match):
    """Both carry one axis, and the rewrite is the table that carries both."""
    with pytest.raises(lps.DataError, match=match):
        lps.build(_TWO_DIMS, {'cap': source}).close()


def test_a_positional_source_needs_the_labels_it_is_written_against():
    """A sequence says what the values are and not what they are labelled, and no
    lane reads labels off the parameters."""
    model = {
        'dimensions': {'g': {}},
        'parameters': {'cap': {'dims': ['g']}},
        'variables': {'x': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x, over=g)'},
    }
    with pytest.raises(lps.DataError, match='nothing else supplies an index'):
        lps.build(model, {'cap': [1.0, 2.0]}).close()


def test_runtime_is_linopy_free(dispatch_yaml):
    """Import the package, build and solve on Arrow sources — linopy never loads.

    pandas and pyarrow are on the list too, and that is newer than it looks:
    on the duckdb engine they could not be, because duckdb imported pandas
    opportunistically when registering any Python object, so "not in
    ``sys.modules``" was not a claim this package could keep. polars imports
    neither until asked, so the stronger claim is now available and is pinned
    here — a bridge out (``to_pandas``, ``to_dataarray``) must stay a bridge
    and never become something the build path walks over on its own.

    Distinct from, and weaker than, the claim that they need not be
    *installed*: the bare-install CI job is what proves that, running this
    suite with no dataframe library beyond polars present at all.
    """
    absent = ('linopy', 'xarray', 'pandas', 'pyarrow')
    script = textwrap.dedent(f"""
        import sys
        assert "linopy" not in sys.modules

        import polars as pl
        import lpspec as lps
        for lib in {absent!r}:
            assert lib not in sys.modules, f"package import pulled in {{lib}}"

        result = lps.solve(
            {str(dispatch_yaml)!r},
            {{
                "p_max": pl.DataFrame({{"generator": ["wind", "solar", "gas"],
                                       "value": [100.0, 60.0, 200.0]}}),
                "cost": pl.DataFrame({{"generator": ["wind", "solar", "gas"],
                                      "value": [1.0, 2.0, 50.0]}}),
                "load": pl.DataFrame({{"snapshot": [0, 1, 2],
                                      "value": [80.0, 120.0, 150.0]}}),
                "snapshot": range(3),
            }},
        )
        assert result.is_ok
        assert isinstance(result.primal("p"), pl.DataFrame), "no dataframe on either side"
        assert result.primal("p").height == 9
        result.close()
        for lib in {absent!r}:
            assert lib not in sys.modules, f"solve pulled in {{lib}}"
        print("LINOPY_FREE_OK")
    """)
    out = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert 'LINOPY_FREE_OK' in out.stdout


def test_check_and_load_model_need_no_data(dispatch_yaml):
    """The model stands for itself: the schema is read from the file when
    wanted, never carried on a built model."""
    for schema in (lps.check(dispatch_yaml), lps.load_model(dispatch_yaml)):
        assert schema.variables['p'].foreach == ['snapshot', 'generator']
        assert schema.parameters['load'].dims == ['snapshot']


@pytest.mark.parametrize(
    ('expression', 'match'),
    [
        pytest.param('sum(p ** 2)', r"operator '\*\*'", id='an-operator-outside-the-language'),
        pytest.param('sum(p * p)', 'degree 2', id='degree-2-caught-with-no-data-bound'),
    ],
)
def test_check_reports_language_errors_before_any_data_is_bound(
    dispatch_yaml, dispatch_frame_inputs, expression, match
):
    """The CI verb enforces the ceiling with no data bound (docs/about/ceiling.md).

    ``build`` is asserted to say the same thing rather than defer it to the
    solver.
    """
    raw = schema_of(dispatch_yaml, **{'objective.expression': expression}).model_dump()

    with pytest.raises(lps.LanguageError, match=match):
        lps.check(raw)
    sources = dispatch_frame_inputs
    with pytest.raises(lps.LanguageError, match=match):
        lps.build(raw, sources)


def test_error_hierarchy_is_one_catchable_tree():
    """One ``except`` covers the package, and the model/run split is real."""
    for cls in (lps.LanguageError, lps.DataError):
        assert issubclass(cls, lps.LpspecError)
    for cls in (lps.SchemaError, lps.DimensionError, lps.PiecewiseExpansionError):
        assert issubclass(cls, lps.LanguageError)
    assert not issubclass(lps.DataError, lps.LanguageError)
    assert issubclass(lps.LpspecError, ValueError)


def test_an_unknown_solver_is_refused_with_the_alternatives(dispatch_yaml, dispatch_frame_inputs):
    """The set of solvers is closed, and a name outside it never falls back to
    the default — solving with a solver other than the one asked for is the one
    answer that cannot be right. Here rather than in ``test_gurobi_sink.py``,
    which skips without the extra: the closed set is a property of the package,
    not of gurobi. Refused before the build, as an unwritable suffix is."""
    from lpspec.relational.sinks import SOLVERS

    sources = dispatch_frame_inputs
    with pytest.raises(lps.LpspecError, match='unknown solver'):
        lps.solve(dispatch_yaml, sources, solver_name='cplex')
    assert set(SOLVERS) == {'highs', 'gurobi', 'xpress'}


def test_a_solver_this_environment_cannot_run_is_refused_before_the_build(
    dispatch_yaml, dispatch_frame_inputs, monkeypatch
):
    """A name in the closed set is not a promise the package is installed.

    `gurobi` is a name lpspec knows on an install that never took the extra, so
    the two mistakes are different and get different sentences. Both refuse
    where the sink is resolved, which is before the build: resolving it there is
    what makes naming a sink nothing can serve cost no model, and that was only
    half true while a known name always resolved.

    Faked by naming a package nothing has rather than by uninstalling gurobipy,
    so the check runs wherever the suite does and still goes through the real
    probe.
    """
    from lpspec import api
    from lpspec.relational.sinks import SOLVERS

    sources = dispatch_frame_inputs
    monkeypatch.setattr(SOLVERS['gurobi'], 'requires', ('a_package_no_environment_has',))
    monkeypatch.setattr(
        api.PolarsEngine, 'build', lambda *_a, **_k: pytest.fail('the model was built before the refusal')
    )

    with pytest.raises(ModuleNotFoundError, match=r'not installed here.*\[gurobi\] extra'):
        lps.solve(dispatch_yaml, sources, solver_name='gurobi')


def test_a_list_of_models_is_refused(dispatch_yaml):
    """Composition is merging declarations, not passing several models.

    The message points at the dict, because a caller holding two files has
    somewhere to go — #30 declined the native merge rather than deferring it.
    """
    with pytest.raises(TypeError, match='merge the declarations'):
        lps.check([dispatch_yaml, dispatch_yaml])


def test_write_suffix_dispatch(dispatch_yaml, dispatch_frame_inputs, tmp_path):
    sources = dispatch_frame_inputs
    out = lps.write(dispatch_yaml, sources, tmp_path / 'm.lp')
    assert out.stat().st_size > 0
    with pytest.raises(ValueError, match='unsupported output format'):
        lps.write(dispatch_yaml, sources, tmp_path / 'm.nc')


def test_solution_to_parquet(dispatch_solution, tmp_path):
    """One file per variable, tidy, streamed straight to disk."""
    assert dispatch_solution.is_ok
    written = dispatch_solution.to_parquet(tmp_path / 'solution')
    assert set(written) == {'p'}
    frame = pl.read_parquet(written['p'])
    assert set(frame.columns) == {'snapshot', 'generator', 'value'}
    assert frame.height == dispatch_solution.primal('p').height


def test_read_back_is_in_label_order_and_stays_there(dispatch_yaml, dispatch_frame_inputs, tmp_path):
    """A read is a join, and a join settles no order — so the read states one.

    Was: every call came back in whatever order the hash join finished in, so
    two reads of one unchanged result disagreed and five writes of one solution
    produced five different files. Nothing was wrong with the numbers, which is
    what made it worth stating rather than leaving to the planner.

    Label order is row-major over the coordinate product, so it is checkable
    against the coordinates themselves: `snapshot` varies slowest, and within
    it `generator` follows the order the file declares.
    """
    sources = dispatch_frame_inputs
    generators = list(sources['p_max']['generator'])
    with lps.solve(dispatch_yaml, sources) as result:
        first = result.primal('p')
        assert first.equals(result.primal('p')), 'a second read agrees, to the row'

        by_declaration = first.with_columns(
            pl.col('generator').replace_strict(generators, range(len(generators))).alias('ord')
        )
        assert by_declaration.equals(by_declaration.sort('snapshot', 'ord'))

        written = [result.to_parquet(tmp_path / f'solution{i}')['p'].read_bytes() for i in range(3)]
        assert len(set(written)) == 1, 'the same solution writes the same bytes'


def test_a_result_stays_readable_until_it_is_closed(dispatch_yaml, dispatch_frame_inputs):
    """No lifetime to manage: reading is valid until you say otherwise.

    A result owns its read-back, so nothing expires it from outside and a
    caller who never closes loses nothing but memory. `close()` is there to
    release the label frames it pins early, and it means what it says — after
    it, there is nothing left to read.
    """
    sources = dispatch_frame_inputs
    result = lps.solve(dispatch_yaml, sources)
    height = result.primal('p').height
    assert height > 0
    assert result.primal('p').height == height, 'still readable, with no close in sight'

    result.close()
    with pytest.raises(lps.LpspecError, match='this result was closed'):
        result.primal('p')


def test_a_second_solve_does_not_rewrite_the_first_result(dispatch_yaml, dispatch_frame_inputs):
    """A result reports its own solve, not the engine's latest.

    Was: the values lived on the engine and every reader went back to them,
    so `objective` was a snapshot while `primal` was live — one result
    disagreeing with itself after a second solve, silently and with plausible
    numbers. Nothing supported re-binds data yet, so the bound has to be moved
    the way the planned in-place update will (#382: `changeColsBounds`
    against labels that are already solver indices).
    """
    key = ['snapshot', 'generator']  # a read is a join, so compare on coordinates
    sources = dispatch_frame_inputs
    with lps.build(dispatch_yaml, sources) as bound:
        first = bound.solve()
        before = first.primal('p').sort(key)
        assert first.is_ok

        assert bound._engine._obj is not None
        bound._engine._obj = bound._engine._obj.with_columns(-pl.col('coeff'))
        second = bound.solve()

        assert not second.primal('p').sort(key).equals(before), 'the second solve really moved'
        assert first.primal('p').sort(key).equals(before), 'and the first still reports its own'
        assert first.objective != pytest.approx(second.objective)


def test_primal_is_a_frame_and_to_pandas_is_the_bridge(dispatch_solution):
    """A frame is the shape results come in; pandas is an exit, not a shape.

    The two must describe the same table — the bridge is a conversion, not a
    second query with its own opinion about column order or dtypes.
    """
    frame = dispatch_solution.primal('p')
    assert isinstance(frame, pl.DataFrame)
    assert frame.columns == ['snapshot', 'generator', 'value']

    pandas = pytest.importorskip('pandas')
    converted = dispatch_solution.to_pandas('p')
    assert isinstance(converted, pandas.DataFrame)
    assert list(converted.columns) == frame.columns
    assert len(converted) == frame.height
    assert frame['value'].sum() == pytest.approx(converted['value'].sum())


@pytest.mark.parametrize(
    ('absent', 'bridge'),
    [
        pytest.param('pandas', 'to_pandas', id='to_pandas-without-pandas'),
        pytest.param('xarray', 'to_dataarray', id='to_dataarray-without-xarray'),
        pytest.param('xarray', 'to_dataset', id='to_dataset-without-xarray'),
    ],
)
def test_a_bridge_out_names_the_extra_that_carries_it(dispatch_solution, absent, bridge):
    """A bridge out of a bare install says which extra to add.

    pandas and xarray ship with ``[linopy]`` rather than with the engine, so
    the bare `No module named 'pandas'` names a package no install instruction
    mentions and leaves the reader to guess. The gurobi sink already answers
    the same question with the extra; these three did not.

    The assertion is the extra, not the missing package: on an install that
    has neither, `to_dataarray` fails at the pandas half and reports that one.
    """
    with (
        mock.patch.dict(sys.modules, {absent: None}),
        pytest.raises(ModuleNotFoundError, match=r'pip install "lpspec\[linopy\]"'),
    ):
        getattr(dispatch_solution, bridge)('p')


def test_no_operator_registry_anywhere():
    """The operator set is closed — there is no way to register more, on any
    surface (#38's ``escape:`` island replaces the idea).

    This is what makes the two lanes accept the same language, and hence what
    makes the differential tests an oracle rather than a comparison of
    dialects (docs/about/architecture.md, "The expressive ceiling").
    """
    import lpspec.language.operators as operators

    assert not hasattr(lps, 'register')
    assert not hasattr(operators, 'register')
    assert not hasattr(operators, '_REGISTRY')


def test_solution_to_dataarray(dispatch_solution):
    """Long tables are right for joining, wrong for the array math that
    post-processing is mostly made of. `to_dataarray` is the bridge."""
    pytest.importorskip('xarray')
    arr = dispatch_solution.to_dataarray('p')
    tidy = dispatch_solution.to_pandas('p')

    assert arr.name == 'p', "named for the variable, not 'value' — the tidy column it came from"
    assert sorted(arr.dims) == ['generator', 'snapshot']
    assert arr.sizes['generator'] == 3
    wind_0 = tidy.query("generator == 'wind' and snapshot == 0")['value'].iloc[0]
    assert float(arr.sel(generator='wind', snapshot=0)) == pytest.approx(wind_0), (
        'the labelled form is the tidy form, indexed'
    )


def test_solution_to_dataset(dispatch_solution):
    """Several variables at once, each keeping its own dims."""
    pytest.importorskip('xarray')
    ds = dispatch_solution.to_dataset('p')
    tidy = dispatch_solution.to_pandas('p')

    assert list(ds.data_vars) == ['p']
    assert sorted(ds['p'].dims) == ['generator', 'snapshot']
    first = tidy.iloc[0]
    assert float(ds['p'].sel(snapshot=first['snapshot'], generator=first['generator'])) == pytest.approx(first['value'])


TWO_VARIABLE_MODEL = {
    'dimensions': {'snapshot': {'dtype': 'int'}, 'generator': {'values': ['wind', 'gas']}},
    'parameters': {'p_max': {'dims': ['generator']}, 'load': {'dims': ['snapshot']}},
    'variables': {
        'p': {'foreach': ['snapshot', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}},
        'shed': {'foreach': ['snapshot'], 'bounds': {'lower': 0}},
    },
    'constraints': {
        'balance': {
            'foreach': ['snapshot'],
            'expression': 'sum(p, over=generator) + shed == load',
        }
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(shed)'},
}


def test_to_dataset_defaults_to_every_variable():
    """A small model wants all of them at once, as linopy's model.solution
    gives you — naming them would be busywork."""
    pytest.importorskip('xarray')
    n = 4
    sources = {
        'p_max': pl.DataFrame({'generator': ['wind', 'gas'], 'value': [100.0, 200.0]}),
        'load': pl.DataFrame({'snapshot': list(range(n)), 'value': np.full(n, 90.0)}),
    }

    with lps.solve(TWO_VARIABLE_MODEL, sources | {'snapshot': range(n)}) as result:
        ds = result.to_dataset()
        subset = result.to_dataset('shed')

    assert set(ds.data_vars) == {'p', 'shed'}
    assert sorted(ds['p'].dims) == ['generator', 'snapshot']
    assert list(ds['shed'].dims) == ['snapshot'], 'each variable keeps its own dims'
    assert set(subset.data_vars) == {'shed'}


@pytest.mark.parametrize(
    'raw',
    [
        pytest.param({'dimensionz': {}}, id='unknown-key'),
        pytest.param({'dimensions': {'g': {'dtype': 'complex'}}}, id='bad-dtype'),
        pytest.param({'version': 99}, id='unknown-version'),
        pytest.param(
            {
                'dimensions': {'g': {'dtype': 'str', 'values': ['a']}},
                'constraints': {'c': {'foreach': ['g'], 'expression': 'nope <= 1'}},
            },
            id='undeclared-name',
        ),
    ],
)
def test_a_wrong_model_raises_one_tree(raw: dict[str, object], tmp_path):
    """Every documented door answers with `LpspecError` (#527).

    Model checking happens in two places — pydantic's validators and the
    language checkers — and they failed differently, so `except LpspecError`,
    the thing `docs/reference/api.md` tells a caller to write, missed the majority of
    model mistakes and a caller had no way to know which.

    `Model.__init__` is *not* in this list, and cannot be: defining one makes
    pydantic route validation through it, which runs every after-validator
    twice and the first time with no context, breaking `extend()`.
    """
    doors = {
        'lps.load_model': lambda: lps.load_model(raw),
        'lps.check': lambda: lps.check(raw),
        'lps.solve': lambda: lps.solve(raw, {}),
        'lps.write': lambda: lps.write(raw, {}, str(tmp_path / 'm.lp')),
        'Model.model_validate': lambda: Model.model_validate(raw),
        'Model.model_validate_json': lambda: Model.model_validate_json(json.dumps(raw)),
    }
    for door, call in doors.items():
        with pytest.raises(lps.LpspecError) as ei:
            call()
        assert 'errors.pydantic.dev' not in str(ei.value), f"{door} leaks pydantic's envelope"


def test_a_closed_result_says_it_was_closed(dispatch_yaml, dispatch_frame_inputs):
    """`close` releases the read-back the readers lay values over, and they say so.

    The status gate cannot notice: closing releases the coordinates, not the
    solve, so `is_readable` stays true and the reader used to fall through to
    a bare `AssertionError`. Frames read before the close are their own data
    and stay valid, which is the half worth stating in the message.
    """
    sources = dispatch_frame_inputs
    sol = lps.solve(dispatch_yaml, sources)
    frame = sol.primal('p')
    objective = sol.objective
    sol.close()

    assert frame.height > 0, 'a frame read before the close is its own data'
    assert sol.objective == objective, 'and the outcome needs no model to report'
    for read in (lambda: sol.primal('p'), lambda: sol.dual('power_balance')):
        with pytest.raises(lps.LpspecError, match='this result was closed'):
            read()
