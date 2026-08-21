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

import re
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import polars as pl
import pytest

from lpspec.errors import DataError, LaneError, LanguageError
from lpspec.sources import tidy_sources
from tests.conftest import schema_of
from tests.oracle import builder, linopy, loader, lpspec_linopy, pd, xr

if TYPE_CHECKING:
    from math_spec.model import Model


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


def _master_coords(schema: Model, sources=None) -> dict:
    """The labels, through the front door both lanes enter by."""
    return loader.dimension_coords(schema, tidy_sources(schema, sources or {}))[0]


class TestMasterCoords:
    @pytest.mark.parametrize(
        ('dim', 'coords', 'expected'),
        [
            pytest.param({'values': [1, 2, 3], 'dtype': 'int'}, None, [1, 2, 3], id='from-yaml-values'),
            pytest.param({}, {'x': [10, 20]}, [10, 20], id='from-coords-kwarg'),
        ],
    )
    def test_labels_come_from_values_or_the_coords_kwarg(self, dim, coords, expected):
        assert list(_master_coords(_schema(dims={'x': dim}), coords)['x']) == expected

    def test_a_dimension_cannot_take_its_labels_from_both(self):
        """One home, and no precedence to remember — the two ways above are exclusive."""
        with pytest.raises(DataError, match=r'dimensions\.x\.values'):
            _master_coords(_schema(dims={'x': {'values': [1, 2], 'dtype': 'int'}}), {'x': [99]})

    def test_a_dimension_with_no_index_is_refused(self):
        """Third in the precedence there is not one: the index is the authority.

        Labels read out of the parameters would *be* the definition, so a
        mistyped one could not be told from a new one — and both lanes say so
        in the same sentence.
        """
        schema = _schema(dims={'x': {}}, params={'a': {'dims': ['x']}})

        with pytest.raises(ValueError, match="dimension 'x' has no index"):
            _master_coords(schema, {'a': {'wind': 1.0}})

    def test_the_labels_keep_the_order_and_the_first_of_each_duplicate(self):
        """A caller's index is read in its own order on both lanes, deduplicated.

        The ordinals a translation moves by are positions in this list, so a
        sort here would move `shift` somewhere else than it moves relationally.
        """
        schema = _schema(dims={'x': {}}, params={'a': {'dims': ['x']}})
        labels = _master_coords(schema, {'x': ['z', 'a', 'z', 'm'], 'a': {'z': 1.0}})['x']

        assert list(labels) == ['z', 'a', 'm'], 'source order, each label once'

    @pytest.mark.parametrize('index', ['pandas', 'polars', 'a bare list'])
    def test_a_temporal_axis_is_the_same_instant_whichever_library_brought_it(self, index):
        """`datetime.date` out of pandas and `datetime64` out of polars are one label.

        They compare unequal, so which library a caller reached for used to
        decide whether their parameter aligned with their index at all — and
        whether a `where` boundary could be compared against the axis. The
        declaration is what knows, and it always answers, so the axis is
        canonical past the read and the source library stops being visible.
        """
        import datetime

        days = [datetime.date(2030, 1, d) for d in (1, 2, 3)]
        sources = {
            'pandas': pd.Index(days, name='t'),
            'polars': pl.DataFrame({'t': days}),
            'a bare list': days,
        }
        schema = _schema(dims={'t': {'dtype': 'datetime'}}, params={'a': {'dims': ['t']}})

        labels = _master_coords(schema, {'t': sources[index], 'a': {days[0]: 1.0}})['t']

        assert labels.dtype.kind == 'M', f'a {index} index reads as the instants it holds'
        assert list(labels) == list(pd.DatetimeIndex(days)), 'and as the same three of them'


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
        tidy = tidy_sources(s, {'a': data})
        ds = loader.load_parameters(s, tidy, loader.dimension_coords(s, tidy)[0])
        assert float(ds['a'].sel(**select)) == expected

    @pytest.mark.parametrize(
        ('dims', 'params', 'data', 'match'),
        [
            pytest.param(
                {'x': {'values': [1], 'dtype': 'int'}},
                {'a': {'dims': ['x']}},
                {},
                'no data provided',
                id='missing-required',
            ),
            pytest.param(
                {'x': {'values': [1], 'dtype': 'int'}, 'y': {'values': [2], 'dtype': 'int'}},
                {'a': {'dims': ['x']}},
                {'a': pd.DataFrame({'x': [1], 'y': [2], 'value': [1.0]}).set_index(['x', 'y'])['value']},
                'a pandas Series with a MultiIndex is not a source',
                id='a-multi-indexed-series',
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
            tidy = tidy_sources(s, data)
            loader.load_parameters(s, tidy, loader.dimension_coords(s, tidy)[0])


# ---------------------------------------------------------------------------
# builder: the operand shapes an operator refuses
# ---------------------------------------------------------------------------


class TestOperandShapesAnOperatorRefuses:
    """Reachable only by a hand-built call, and therefore only from here.

    `validation.py` refuses every one of these at load, so a model cannot carry
    them and no end-to-end test can reach the guards. Deleting any of the four
    left the whole suite green, which is the case the rules say gets a
    purpose-built probe rather than a shrug: they are the difference between a
    caller of `lpspec.linopy.builder` seeing the sentence and seeing an
    `AttributeError` from inside xarray.
    """

    @pytest.mark.parametrize(
        ('call', 'kwargs'),
        [
            pytest.param(
                builder._operator_grouped_sum, {'into': ('b',), 'labels': {'b': pd.Index(['n'], name='b')}}, id='sum-by'
            ),
            pytest.param(builder._operator_at, {'into': ('b',)}, id='at'),
        ],
    )
    def test_a_lookup_that_is_not_an_array_names_what_arrived(self, call, kwargs):
        array = xr.DataArray([1.0, 2.0], dims=['g'], coords={'g': ['w', 's']})

        with pytest.raises(TypeError, match='lookup must be an array'):
            call(array, ({'w': 'n'},), **kwargs)

    @pytest.mark.parametrize(
        ('call', 'kwargs'),
        [
            pytest.param(
                builder._operator_grouped_sum, {'into': ('b',), 'labels': {'b': pd.Index(['n'], name='b')}}, id='sum-by'
            ),
            pytest.param(builder._operator_at, {'into': ('b',)}, id='at'),
        ],
    )
    def test_a_lookup_over_two_dims_is_refused_as_language(self, call, kwargs):
        """A lookup is one column of one index, so two dims is not a shape it has."""
        array = xr.DataArray([1.0, 2.0], dims=['g'], coords={'g': ['w', 's']})
        wide = xr.DataArray([['n', 'e']], dims=['t', 'g'], coords={'t': [0], 'g': ['w', 's']})

        with pytest.raises(LanguageError, match='exactly one dimension'):
            call(array, (wide,), **kwargs)

    @pytest.mark.parametrize(
        ('call', 'kwargs', 'named'),
        [
            pytest.param(
                builder._operator_grouped_sum,
                {'into': ('b',), 'labels': {'b': pd.Index(['n'], name='b')}},
                'sum(by=)',
                id='sum-by',
            ),
            pytest.param(builder._operator_at, {'into': ('b',)}, 'at()', id='at'),
        ],
    )
    def test_an_operand_the_operator_cannot_read_names_the_call(self, call, kwargs, named):
        """The operand reaches the guard, not xarray — so the message says which operator."""
        mapping = xr.DataArray(['n', 'n'], dims=['g'], coords={'g': ['w', 's']})

        with pytest.raises(TypeError, match=re.escape(named)):
            call(object(), (mapping,), **kwargs)


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
    from math_spec.resolution import Namespace, where_of

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


#: A constant side the data does not cover, which only the *build* can see: the
#: rows exist, the parameter has no row at one of them, and the fill would be
#: the bound. The declaration it names is reached through `note()` rather than
#: written into the message, which is the half of this the load errors cannot
#: exercise.
_UNCOVERED_BOUND = "parameters:\n  cap: {dims: [g]}\nconstraints:\n  c:\n    foreach: [g]\n    expression: 'p <= cap'\n"
_NO_ROWS = {'cap': pd.Series([], index=pd.Index([], name='g', dtype='object'), dtype='float64')}


@pytest.mark.parametrize(
    ('tail', 'data', 'error', 'match', 'context'),
    [
        pytest.param(
            "    where: '<<<'\n",
            {},
            ValueError,
            'Failed to parse where string',
            "Variable 'p'",
            id='malformed-where',
        ),
        pytest.param(
            "constraints:\n  c:\n    foreach: [g]\n    expression: 'p + 1'\n",
            {},
            ValueError,
            'exactly one comparison',
            "Constraint 'c'",
            id='constraint-without-comparison',
        ),
        pytest.param(
            "objective:\n  expression: 'p == 1'\n",
            {},
            ValueError,
            'must not contain a comparison',
            'The objective',
            id='objective-with-comparison',
        ),
        pytest.param(
            "constraints:\n  c:\n    foreach: []\n    expression: '1 <= 2'\n",
            {},
            ValueError,
            'decides nothing',
            "Constraint 'c'",
            id='a-comparison-with-no-variable-in-it',
        ),
        pytest.param(
            _UNCOVERED_BOUND,
            _NO_ROWS,
            DataError,
            'fewer coordinates than the rows built here',
            "while building constraint 'c'",
            id='a-constant-side-the-data-misses-so-only-the-build-sees-it',
        ),
    ],
)
def test_a_failure_names_the_declaration_and_the_file(yaml_file, tail, data, error, match, context):
    bad = yaml_file(textwrap.dedent(_MINIMAL).lstrip() + tail, 'bad.yaml')

    with pytest.raises(error, match=match) as ei:
        lpspec_linopy.build(bad, data)

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
              expression: "sum(x)"
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
          expression: sum(x)
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


#: A curve masked by ``points:``, so the expansion declares a parameter the file
#: does not — ``cost_curve_points``, derived from ``bp_x``'s own rows. Ragged on
#: purpose: hydro states two breakpoints where the axis has four, which is the
#: whole reason a mask exists.
MASKED_CURVE_YAML = """
dimensions:
  snapshot: {dtype: int, values: [0]}
  generator: {dtype: str, values: [hydro, gas]}
  bp: {dtype: int, values: [0, 1, 2, 3]}
parameters:
  p_max: {dims: [generator]}
  load: {dims: [snapshot]}
  bp_x: {dims: [generator, bp]}
  bp_y: {dims: [generator, bp]}
variables:
  p:
    foreach: [snapshot, generator]
    bounds: {lower: 0, upper: p_max}
  op_cost:
    foreach: [snapshot, generator]
    bounds: {lower: 0}
piecewise:
  cost_curve:
    over: bp
    points: bp_x
    links:
      - [p, bp_x]
      - [op_cost, bp_y, ">="]
    method: convex
expressions:
  spend: sum(op_cost, over=generator)
constraints:
  balance:
    foreach: [snapshot]
    expression: sum(p, over=generator) == load
objective:
  sense: minimize
  expression: sum(sum(op_cost, over=generator), over=snapshot)
"""


def _curve(points):
    """A per-generator curve as the tidy frame it is supplied as."""
    return pl.DataFrame(
        {
            'generator': [g for g, _ in points],
            'bp': [k for _, k in points],
            'value': list(points.values()),
        }
    )


MASKED_CURVE_DATA = {
    'p_max': pd.Series({'hydro': 40.0, 'gas': 80.0}),
    'load': pd.Series([50.0], index=pd.RangeIndex(1, name='snapshot')),
    'bp_x': _curve(
        {('hydro', 0): 0.0, ('hydro', 1): 40.0, ('gas', 0): 0.0, ('gas', 1): 20.0, ('gas', 2): 50.0, ('gas', 3): 80.0}
    ),
    'bp_y': _curve(
        {
            ('hydro', 0): 0.0,
            ('hydro', 1): 200.0,
            ('gas', 0): 0.0,
            ('gas', 1): 150.0,
            ('gas', 2): 450.0,
            ('gas', 3): 900.0,
        }
    ),
}


def test_a_named_expression_reads_off_a_masked_curve(yaml_file):
    """A curve's derived mask is the file's to supply, so the file is what asks for it.

    `points:` names a values parameter, and the mask that says where the curve
    runs is then derived from that parameter's rows — by `derive_curve_masks`,
    which reads `piecewise:`. The expansion has cleared `piecewise:`, so handing
    it the expanded model asks for `cost_curve_points` and derives nothing:
    `DataError: no data provided for parameter 'cost_curve_points'`, naming a
    parameter no caller wrote and none can supply. `build` passed the file and
    this reader passed the expansion, which is the only reason one worked.
    """
    from tests.differential import differential

    path = yaml_file(MASKED_CURVE_YAML, 'masked_curve.yaml')
    with differential(path, MASKED_CURVE_DATA) as run:
        tidy = run.result.expression('spend')
        eager = lpspec_linopy.expression(run.model, path, 'spend', dict(MASKED_CURVE_DATA))
        got = {int(k): v for k, v in zip(tidy['snapshot'], tidy['value'], strict=True)}
        want = {int(k): float(v) for k, v in eager.to_series().items()}
        assert got == pytest.approx(want), 'the two lanes disagree about a named expression over a masked curve'


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

    frames = dispatch_frame_inputs
    path = tmp_path / 'load.parquet'
    frames['load'].write_parquet(path)
    sources = {**frames, 'load': path}

    with differential(dispatch_yaml, sources) as run:
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
        'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
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
    'constraints': {'c': {'foreach': ['t'], 'expression': 'x <= shift(eff, over=t, offset=1)'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
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


#: The one construct this lane accepts and cannot build: a bare parameter term
#: in the objective, which linopy has no slot for. `osemosys_utopia` owes one
#: as the fixed cost of capacity that already stood in 1990.
OBJECTIVE_CONSTANT = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'standing': {'dims': []}},
    'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 1}}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x) + standing'},
}


def test_a_construct_this_lane_cannot_build_is_refused_in_its_own_words():
    """The mirror of the test above: the streaming lane builds this one.

    So it is not a refusal of the *language* — the model is sayable, lowers,
    and solves relationally. What the reader has to be told is that the wall
    is this lane's, which linopy's `Constant values in objective function not
    supported.` cannot say: it names no file, no declaration and no other
    route. Before #894 that sentence was what escaped, from a linopy setter
    two frames down.
    """
    import lpspec as lps

    assert lps.solve(OBJECTIVE_CONSTANT, {'t': [0, 1], 'standing': 5.0}).objective == pytest.approx(5.0), (
        'the streaming lane builds it, so the model is not the problem'
    )
    with pytest.raises(LaneError) as refusal:
        lpspec_linopy.build(OBJECTIVE_CONSTANT, {'t': [0, 1], 'standing': 5.0})

    assert str(refusal.value) == builder.OBJECTIVE_CONSTANT_IS_A_LANE_GAP, (
        "the sentence is the lane's own, which is the whole of the fix"
    )


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
        'objective': {'sense': 'maximize', 'expression': 'sum(x * cost)'},
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


class TestLoadTimeIntegration:
    """The eager lane's entry points, so only this class needs the [linopy] extra.

    ``tests.oracle`` is imported per test rather than at module scope: the rest
    of the module is pure load-time validation and runs on a bare install.
    """

    def test_from_yaml_fails_before_data_validation(self, tmp_path):
        """A typo in an expression errors even when data= is absent."""
        from tests.oracle import lpspec_linopy

        f = tmp_path / 'm.yaml'
        f.write_text(
            'dimensions:\n'
            '  g:\n'
            '    values: [wind, solar]\n'
            'variables:\n'
            '  p:\n'
            '    foreach: [g]\n'
            'constraints:\n'
            '  cap:\n'
            '    foreach: [g]\n'
            '    expression: pp <= 100\n'
        )
        with pytest.raises(ValueError, match="'pp' not found"):
            lpspec_linopy.build(f, {})
