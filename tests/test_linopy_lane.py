"""The opt-in linopy lane: its verbs, its loader, its where evaluator, its notes.

Everything here needs the ``[linopy]`` extra and nothing here is reachable
from the native lane, so it is one module rather than four: the guard and the
"write a YAML file, feed it to the lane" idiom were being restated in each of
them.

The lane *constructs*, so it holds no state to begin with: one call, one
model, nothing retained. What is left to pin is that it puts nothing on
``linopy.Model`` either — the first test below.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import polars as pl
import pytest

from lpspec.errors import DataError, LanguageError
from tests.conftest import schema_of
from tests.oracle import builder, linopy, loader, lpspec_linopy, pd, xr

if TYPE_CHECKING:
    from lpspec.language.model import Model


@pytest.fixture
def yaml_file(tmp_path):
    """Write YAML text to a file — the only shape the eager lane accepts."""

    def write(text: str, name: str = 'm.yaml'):
        path = tmp_path / name
        path.write_text(textwrap.dedent(text).lstrip())
        return path

    return write


# ---------------------------------------------------------------------------
# the lane is a pure producer
# ---------------------------------------------------------------------------


def test_nothing_is_patched_onto_linopy_model():
    """Importing lpspec_linopy must not touch linopy.Model."""
    assert not hasattr(linopy.Model, 'from_yaml')
    assert not hasattr(linopy.Model, 'yaml')


# ---------------------------------------------------------------------------
# loader: master coords and parameter coercion
# ---------------------------------------------------------------------------


def _schema(dims=None, params=None) -> Model:
    raw = {}
    if dims:
        raw['dimensions'] = dims
    if params:
        raw['parameters'] = params
    return schema_of(raw)


class TestBuildMasterCoords:
    @pytest.mark.parametrize(
        ('dim', 'coords', 'expected'),
        [
            pytest.param({'values': [1, 2, 3], 'dtype': 'int'}, None, [1, 2, 3], id='from-yaml-values'),
            pytest.param({}, {'x': [10, 20]}, [10, 20], id='from-coords-kwarg'),
        ],
    )
    def test_labels_come_from_values_or_the_coords_kwarg(self, dim, coords, expected):
        mc = loader.build_master_coords(_schema(dims={'x': dim}), coords)
        assert list(mc['x']) == expected

    def test_a_dimension_cannot_take_its_labels_from_both(self):
        """One home, and no precedence to remember — the two ways above are exclusive."""
        with pytest.raises(DataError, match=r'dimensions\.x\.values'):
            loader.build_master_coords(_schema(dims={'x': {'values': [1, 2], 'dtype': 'int'}}), {'x': [99]})

    def test_a_dimension_with_no_index_is_refused(self):
        """Third in the precedence there is not one: the index is the authority.

        Labels read out of the parameters would *be* the definition, so a
        mistyped one could not be told from a new one — and both lanes say so
        in the same sentence.
        """
        schema = _schema(dims={'x': {}}, params={'a': {'dims': ['x']}})

        with pytest.raises(ValueError, match="dimension 'x' has no index"):
            loader.build_master_coords(schema, None, {'a': {'wind': 1.0}})


class TestLoadParameters:
    """Every shape a user may hand a parameter, coerced onto the master coords."""

    @pytest.mark.parametrize(
        ('values', 'data', 'select', 'expected'),
        [
            pytest.param([1, 2], 5.0, {'x': 1}, 5.0, id='scalar-broadcasts'),
            pytest.param(['a', 'b'], {'a': 100, 'b': 60}, {'x': 'a'}, 100.0, id='dict'),
            pytest.param(
                ['a', 'b'],
                pd.Series([1.0, 2.0], index=pd.Index(['a', 'b'], name='x')),
                {'x': 'b'},
                2.0,
                id='series',
            ),
            pytest.param(
                [0, 1],
                pd.DataFrame({'x': [0, 1], 'value': [10.0, 20.0]}),
                {'x': 1},
                20.0,
                id='tidy-frame',
            ),
            pytest.param([0, 1], [10.0, 20.0], {'x': 1}, 20.0, id='sequence'),
            pytest.param(
                [0, 1],
                pl.DataFrame({'x': [0, 1], 'value': [10.0, 20.0]}),
                {'x': 1},
                20.0,
                id='a-polars-frame',
            ),
            pytest.param(
                [0, 1],
                pl.LazyFrame({'x': [0, 1], 'value': [10.0, 20.0]}),
                {'x': 1},
                20.0,
                id='a-polars-scan',
            ),
        ],
    )
    def test_accepted_shapes(self, values, data, select, expected):
        dtype = 'int' if isinstance(values[0], int) else 'str'
        s = _schema(dims={'x': {'values': values, 'dtype': dtype}}, params={'a': {'dims': ['x']}})
        ds = loader.load_parameters(s, {'a': data}, loader.build_master_coords(s, None))
        assert float(ds['a'].sel(**select)) == expected

    @pytest.mark.parametrize(
        ('dims', 'params', 'data', 'match'),
        [
            pytest.param(
                {'x': {'values': [1], 'dtype': 'int'}},
                {'a': {'dims': ['x']}},
                {},
                'required',
                id='missing-required',
            ),
            pytest.param(
                {'x': {'values': [1], 'dtype': 'int'}, 'y': {'values': [2], 'dtype': 'int'}},
                {'a': {'dims': ['x']}},
                {'a': pd.DataFrame({'x': [1], 'y': [2], 'value': [1.0]}).set_index(['x', 'y'])['value']},
                'index has 2 level',
                id='an-index-deeper-than-the-declared-dims',
            ),
            pytest.param(
                {'x': {'values': [1], 'dtype': 'int'}},
                {'a': {'dims': ['x']}},
                {'a': xr.DataArray([1], dims=['x'], coords={'x': [1]})},
                'not a source',
                id='a-dense-array',
            ),
            pytest.param(
                {'g': {'values': ['a', 'b']}},
                {'p': {'dims': ['g']}},
                {'p': pd.Series([1.0], index=pd.Index(['z'], name='g'))},
                'not in the master coordinate',
                id='unknown-coord',
            ),
        ],
    )
    def test_refused_shapes(self, dims, params, data, match):
        s = _schema(dims=dims, params=params)
        with pytest.raises(ValueError, match=match):
            loader.load_parameters(s, data, loader.build_master_coords(s, None))


# ---------------------------------------------------------------------------
# builder.evaluate_where: the eager reading of a resolved where AST
# ---------------------------------------------------------------------------


@pytest.fixture
def gens():
    """A dataset and its master coords, with one generator masked out by p_max."""
    ds = xr.Dataset({'p_max': xr.DataArray([100, 0, 50], dims=['g'], coords={'g': ['wind', 'solar', 'gas']})})
    return ds, {'g': pd.Index(['wind', 'solar', 'gas'], name='g')}


def _resolved(text, parameters=('p_max',), dimensions=('g',)):
    """Resolve then evaluate — the evaluator no longer takes strings."""
    from lpspec.language.resolution import Namespace, where_of

    return where_of(text, Namespace((), parameters, dimensions), 'test')


def test_no_where_is_a_scalar_true(gens):
    mask = builder.evaluate_where(None, *gens)
    assert mask.ndim == 0
    assert bool(mask) is True


def test_a_bare_parameter_name_is_an_existence_check(gens):
    assert builder.evaluate_where(_resolved('p_max'), *gens).all()


def test_a_comparison_masks_per_coordinate(gens):
    mask = builder.evaluate_where(_resolved('p_max > 0'), *gens)
    assert [bool(mask.sel(g=g)) for g in ('wind', 'solar', 'gas')] == [True, False, True]


def test_a_dimension_comparison_masks_on_the_coordinate_itself():
    node = _resolved('t > 0', parameters=(), dimensions=('t',))
    mask = builder.evaluate_where(node, xr.Dataset(), {'t': pd.Index([0, 1, 2], name='t')})
    assert [bool(mask.sel(t=t)) for t in (0, 1, 2)] == [False, True, True]


def test_a_missing_parameter_is_a_load_error():
    """Was: a scalar-False mask, i.e. a silently empty model. Resolution
    makes an undeclared name a load error in both lanes."""
    with pytest.raises(LanguageError, match="'nonexistent' not found"):
        _resolved('nonexistent')


# ---------------------------------------------------------------------------
# error notes: the context add_note() carries out of build
# ---------------------------------------------------------------------------

_MINIMAL = """
    dimensions:
      g: {values: [a]}
    variables:
      p:
        foreach: [g]
"""


def _has_note(exc: BaseException, substring: str) -> bool:
    return any(substring in n for n in getattr(exc, '__notes__', []))


@pytest.mark.parametrize(
    ('tail', 'error', 'match', 'context'),
    [
        pytest.param(
            "    where: '<<<'\n",
            ValueError,
            'Failed to parse where string',
            "Variable 'p'",
            id='malformed-where',
        ),
        pytest.param(
            "constraints:\n  c:\n    foreach: [g]\n    expression: 'p + 1'\n",
            ValueError,
            'exactly one comparison',
            "Constraint 'c'",
            id='constraint-without-comparison',
        ),
        pytest.param(
            "objective:\n  expression: 'p == 1'\n",
            ValueError,
            'must not contain a comparison',
            'The objective',
            id='objective-with-comparison',
        ),
        pytest.param(
            "constraints:\n  c:\n    foreach: []\n    expression: '1 <= 2'\n",
            TypeError,
            None,
            "while building constraint 'c'",
            id='valid-syntax-and-dims-but-no-variable-so-only-the-build-sees-it',
        ),
    ],
)
def test_a_failure_names_the_declaration_and_the_file(yaml_file, tail, error, match, context):
    bad = yaml_file(textwrap.dedent(_MINIMAL).lstrip() + tail, 'bad.yaml')

    with pytest.raises(error, match=match) as ei:
        lpspec_linopy.build(bad, {})

    assert context in str(ei.value) or _has_note(ei.value, context)
    assert _has_note(ei.value, f"while loading YAML '{bad}'")


# --------------------------------------------------------------------------
# the convention this lane speaks
# --------------------------------------------------------------------------


def test_importing_the_lane_selects_the_v1_convention():
    """In a *fresh* process, because the harness would otherwise answer for it.

    ``tests/oracle.py`` sets ``semantics = 'v1'`` at import, so an in-process
    assertion here would pass whether or not the package sets it — which is
    precisely how the package shipped without setting it: the suite proved the
    two lanes agree under a configuration no user ran. linopy's default is
    ``legacy``, which fills an absent slot with 0 rather than dropping the row,
    so under it this lane answered 25.0 where the native engine answered 125.0.

    A subprocess is the only place the claim is falsifiable, so it is the only
    place worth making it.
    """
    probe = 'import linopy, lpspec.linopy; print(linopy.options["semantics"])'
    out = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == 'v1', f'the lane must select v1 on import, got {out.stdout.strip()!r}'


def test_the_two_lanes_agree_about_a_masked_variable_without_the_harness(tmp_path):
    """The divergence the missing opt-in caused, pinned end to end.

    Not a ``differential()`` case on purpose: that helper runs inside the suite,
    where the convention is already set. This one drives both lanes from a
    subprocess with nothing but the package imported, which is the user's
    situation.
    """
    model = tmp_path / 'masked.yaml'
    model.write_text(
        textwrap.dedent("""
            dimensions: {f: {values: [a, b]}}
            parameters:
              gate: {dims: [f], dtype: bool}
              relmax: {dims: [f]}
            variables:
              x: {foreach: [f], bounds: {lower: 0, upper: 100}}
              size: {foreach: [f], where: gate, bounds: {lower: 0, upper: 50}}
            constraints:
              env:
                foreach: [f]
                expression: "x - relmax * size <= 0"
            objective:
              sense: maximize
              expression: "sum(x, over=f)"
        """).lstrip()
    )
    probe = textwrap.dedent(f"""
        import warnings; warnings.simplefilter('ignore')
        import pandas as pd, polars as pl
        import lpspec as lps
        from lpspec import linopy as fkl
        data = {{'gate': pd.Series({{'a': True}}), 'relmax': pd.Series({{'a': 0.5, 'b': 0.5}})}}
        m = fkl.build({str(model)!r}, data)
        m.solve(solver_name='highs', output_flag=False)
        native = lps.solve({str(model)!r}, {{
            'gate': pl.DataFrame({{'f': ['a'], 'value': [True]}}),
            'relmax': pl.DataFrame({{'f': ['a', 'b'], 'value': [0.5, 0.5]}}),
        }})
        print(float(m.objective.value), native.objective)
    """)
    out = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, check=True)
    eager, native = (float(v) for v in out.stdout.split())
    assert eager == pytest.approx(native), f'lanes disagree outside the harness: {eager} vs {native}'
    assert native == pytest.approx(125.0), 'the masked row should be dropped, leaving x[b] at its bound'


def test_a_missing_bound_is_refused_at_build_with_the_native_lane_s_message(yaml_file):
    """It used to surface two phases later, from inside linopy.

    ``build()`` returned a model whose bounds carried NaN, and the failure came
    at solve or write as ``ValueError: Continuous Variable x contains nan's in
    field(s) ['upper']`` — linopy's own message, naming an internal rather than
    the YAML, the declaration or the fix. The native lane had raised a
    ``DataError`` at build the whole time, so the two lanes agreed on the
    verdict and disagreed on everything a reader needs (#313).

    The mask is the other half. A coordinate the variable does not occupy needs
    no bound, and supplying data only where the variable exists is the ordinary
    idiom — so this must refuse the gap and accept the masked one, which is what
    the second half asserts.
    """
    model = yaml_file("""
        dimensions:
          f: {values: [a, b]}
        parameters:
          ub: {dims: [f]}
          live: {dims: [f], dtype: bool}
        variables:
          x: {foreach: [f], bounds: {lower: 0, upper: ub}}
        constraints:
          c:
            foreach: [f]
            expression: x <= 100
        objective:
          sense: maximize
          expression: sum(x, over=f)
        """)
    data = {
        'ub': pd.Series([10.0], index=pd.Index(['a'], name='f')),
        'live': pd.Series([True], index=pd.Index(['a'], name='f')),
    }

    with pytest.raises(DataError, match='NULL bounds'):
        lpspec_linopy.build(model, data)

    masked = yaml_file(
        model.read_text().replace('{foreach: [f], bounds:', '{foreach: [f], where: live, bounds:'),
        'masked.yaml',
    )
    built = lpspec_linopy.build(masked, data)
    assert 'x' in built.variables


# ---------------------------------------------------------------------------
# named expressions: one reader per lane, one answer (#562)
# ---------------------------------------------------------------------------

EXPRESSION_YAML = """
dimensions:
  snapshot: {dtype: int, values: [0, 1, 2]}
  generator: {dtype: str, values: [g1, g2]}
parameters:
  p_max: {dims: [generator]}
  cost: {dims: [generator]}
  load: {dims: [snapshot]}
variables:
  p:
    foreach: [snapshot, generator]
    bounds: {lower: 0, upper: p_max}
expressions:
  total_gen: sum(p, over=generator)
  spend: sum(p * cost, over=generator)
constraints:
  balance:
    foreach: [snapshot]
    expression: total_gen == load
objective:
  sense: minimize
  expression: sum(sum(p * cost, over=generator), over=snapshot)
"""

#: Distinct costs and a load exceeding the cheap generator's capacity make the
#: dispatch unique, so the two lanes' expression values are comparable exactly
#: rather than up to an alternative optimum.
EXPRESSION_DATA = {
    'p_max': pd.Series({'g1': 100.0, 'g2': 100.0}),
    'cost': pd.Series({'g1': 10.0, 'g2': 20.0}),
    'load': pd.Series({0: 50.0, 1: 120.0, 2: 80.0}),
}


@pytest.mark.parametrize(
    'name',
    [
        pytest.param('total_gen', id='referenced-by-a-constraint'),
        pytest.param('spend', id='declared-but-never-referenced'),
    ],
)
def test_the_two_lanes_agree_on_a_named_expression(yaml_file, name):
    """`result.expression(name)` and the lane's `expression` read one value.

    Including the standalone case: the rules for named expressions guarantees a never-referenced
    expression is parsed and name-checked, and #562 makes it readable — on
    the eager lane by building the declared expression on the solved model and
    taking linopy's native `.solution`.
    """
    from tests.differential import differential

    path = yaml_file(EXPRESSION_YAML, 'expressions.yaml')
    with differential(path, EXPRESSION_DATA) as run:
        tidy = run.result.expression(name)
        eager = lpspec_linopy.expression(run.model, path, name, dict(EXPRESSION_DATA))
        got = {int(k): v for k, v in zip(tidy['snapshot'], tidy['value'], strict=True)}
        want = {int(k): float(v) for k, v in eager.to_series().items()}
        assert got == pytest.approx(want), f"the two lanes disagree about named expression '{name}'"


def test_the_lane_refuses_an_unknown_expression_name(yaml_file):
    path = yaml_file(EXPRESSION_YAML, 'expressions.yaml')
    m = lpspec_linopy.build(path, dict(EXPRESSION_DATA))
    with pytest.raises(KeyError, match='never an expression string'):
        lpspec_linopy.expression(m, path, 'sum(p, over=generator)', dict(EXPRESSION_DATA))


def test_one_set_of_tables_reaches_both_lanes(dispatch_yaml, dispatch_frame_inputs, tmp_path):
    """The claim the shape work is for: one `sources` mapping, either lane.

    polars frames and a parquet path, handed to both unchanged — no per-lane
    conversion at the call site, which is what made the two accepted-input sets
    a divergence a user hit directly (#60).
    """
    from tests.differential import differential

    frames, coords = dispatch_frame_inputs
    path = tmp_path / 'load.parquet'
    frames['load'].write_parquet(path)
    sources = {**frames, 'load': path}

    with differential(dispatch_yaml, sources, coords) as run:
        assert run.result.primal('p').height, 'the relational lane built no rows'
        assert float(run.model.variables['p'].labels.count()), 'the eager lane built no variables'


@pytest.mark.parametrize(
    'as_model',
    [
        pytest.param(lambda raw, path: path, id='a-path'),
        pytest.param(lambda raw, path: raw, id='a-mapping'),
        pytest.param(lambda raw, path: schema_of(raw), id='a-loaded-model'),
    ],
)
def test_the_lane_takes_a_model_the_same_three_ways_the_runner_does(tmp_path, as_model):
    """`lps.build` and this take the same first argument, so neither decides the lane.

    A path was the only spelling here while the runner took all three, which
    made "convert this to a linopy.Model instead" a rewrite of the call rather
    than a change of import (#845).
    """
    import yaml as pyyaml

    raw = {
        'dimensions': {'g': {'values': ['wind', 'gas']}},
        'parameters': {'cap': {'dims': ['g']}},
        'variables': {'x': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'objective': {'sense': 'maximize', 'expression': 'x'},
    }
    path = tmp_path / 'm.yaml'
    path.write_text(pyyaml.safe_dump(raw))

    built = lpspec_linopy.build(as_model(raw, path), {'cap': {'wind': 40.0, 'gas': 100.0}})
    assert 'x' in built.variables, 'the same file, whichever way it was handed over'


#: A shift over a variable-free expression: the vacated positions have no
#: value, and inventing one silently pins a bound to zero.
_BARE_SHIFT = {
    'dimensions': {'t': {'dtype': 'int', 'values': [0, 1, 2]}},
    'parameters': {'eff': {'dims': ['t']}},
    'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 5}}},
    'constraints': {'c': {'foreach': ['t'], 'expression': 'x <= shift(eff, over=t, by=1)'}},
    'objective': {'sense': 'maximize', 'expression': 'x'},
}


def test_a_construct_the_streaming_lane_refuses_is_refused_here_too():
    """One gate, both lanes — hard rule 3 held mechanically rather than by care.

    This lane used to load and expand and stop there, so the fourteen refusals
    in `lowering.py` never fired on it: a bare `shift()` over data built a
    model whose vacated positions were `NaN`, and died two phases later inside
    linopy's IO with a sentence naming neither the YAML nor the fix.
    """
    import lpspec as lps

    with pytest.raises(LanguageError, match='vacated positions') as native:
        lps.check(_BARE_SHIFT)
    with pytest.raises(LanguageError, match='vacated positions') as eager:
        lpspec_linopy.build(_BARE_SHIFT, {'eff': {0: 1.0, 1: 2.0, 2: 3.0}})

    assert str(native.value) == str(eager.value), 'one refusal, one wording, whichever lane was asked'


def test_a_file_that_declares_no_labels_at_all_is_refused_on_both_lanes():
    """The index is what says which labels exist, on either lane.

    Neither `values:` nor a table under `g` says what it holds, so a mistyped
    label in `cost` would define a generator rather than fail. Both lanes
    refuse, in the same sentence.
    """
    import lpspec as lps
    from lpspec.errors import DataError

    model = {
        'dimensions': {'g': {}},
        'parameters': {'cap': {'dims': ['g']}, 'cost': {'dims': ['g']}},
        'variables': {'x': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'objective': {'sense': 'maximize', 'expression': 'x * cost'},
    }
    sources = {
        'cap': pd.Series({'wind': 40.0, 'gas': 100.0}).rename_axis('g'),
        'cost': pd.Series({'wind': 3.0, 'gas': 1.0}).rename_axis('g'),
    }

    with pytest.raises(DataError, match="dimension 'g' has no index") as native:
        lps.build(model, sources).close()
    with pytest.raises(DataError, match="dimension 'g' has no index") as eager:
        lpspec_linopy.build(model, sources)
    assert str(native.value) == str(eager.value), 'one refusal, one wording'

    indexed = {**sources, 'g': pd.DataFrame({'g': ['wind', 'gas']})}
    assert 'x' in lpspec_linopy.build(model, indexed).variables
