"""The linopy lane: its verbs, its loader, its where evaluator, its notes.

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

import numpy as np
import polars as pl
import pytest

from lpspec.errors import DataError, LaneError, LanguageError
from lpspec.sources import tidy_sources
from tests.conftest import EXAMPLES_DIR, schema_of
from tests.differential import differential
from tests.oracle import builder, linopy, loader, lpspec_linopy, pd, where, xr
from tests.piecewise_models import curve_frame

if TYPE_CHECKING:
    from math_spec import Spec


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


def _schema(dims=None, params=None) -> Spec:
    raw = {}
    if dims:
        raw['dimensions'] = dims
    if params:
        raw['parameters'] = params
    return schema_of(raw)


def _program(schema: Spec):
    """The plan the loader reads its declarations off, as a build makes one."""
    from math_spec import to_program

    return to_program(schema)


def _master_coords(schema: Spec, sources=None) -> dict:
    """The labels, through the front door both lanes enter by."""
    return loader.dimension_coords(_program(schema), tidy_sources(_program(schema), sources or {}))[0]


class TestMasterCoords:
    def test_labels_come_from_the_source_under_the_dimensions_own_key(self):
        assert list(_master_coords(_schema(dims={'x': {'dtype': 'int'}}), {'x': [10, 20]})['x']) == [10, 20]

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
        s = _schema(dims={'x': {'dtype': dtype}}, params={'a': {'dims': ['x']}})
        tidy = tidy_sources(_program(s), {'x': values, 'a': data})
        ds = loader.load_parameters(_program(s), tidy, loader.dimension_coords(_program(s), tidy)[0])
        assert float(ds['a'].sel(**select)) == expected

    @pytest.mark.parametrize(
        ('dtype', 'value', 'kind'),
        [
            pytest.param('bool', False, 'b', id='bool'),
            pytest.param('str', 'a', 'U', id='str'),
            pytest.param('int', 3, 'i', id='int'),
            pytest.param('float', 2.5, 'f', id='float'),
        ],
    )
    def test_a_dims_less_parameter_keeps_the_dtype_it_declares(self, dtype, value, kind):
        s = _schema(params={'a': {'dims': [], 'dtype': dtype}})
        tidy = tidy_sources(_program(s), {'a': pd.DataFrame({'value': [value]})})
        ds = loader.load_parameters(_program(s), tidy, loader.dimension_coords(_program(s), tidy)[0])
        assert ds['a'].dtype.kind == kind, f'declared {dtype}, loaded as {ds["a"].dtype}'
        assert ds['a'].item() == value

    @pytest.mark.parametrize(
        ('dims', 'params', 'data', 'match'),
        [
            pytest.param(
                {'x': {'dtype': 'int'}},
                {'a': {'dims': ['x']}},
                {'x': [1]},
                'no data provided',
                id='missing-required',
            ),
            pytest.param(
                {'x': {'dtype': 'int'}, 'y': {'dtype': 'int'}},
                {'a': {'dims': ['x']}},
                {
                    'x': [1],
                    'y': [2],
                    'a': pd.DataFrame({'x': [1], 'y': [2], 'value': [1.0]}).set_index(['x', 'y'])['value'],
                },
                'a pandas Series with a MultiIndex is not a source',
                id='a-multi-indexed-series',
            ),
            pytest.param(
                {'x': {'dtype': 'int'}},
                {'a': {'dims': ['x']}},
                {'x': [1], 'a': xr.DataArray([1], dims=['x'], coords={'x': [1]})},
                'not a source',
                id='a-dense-array',
            ),
            pytest.param(
                {'g': {'dtype': 'str'}},
                {'p': {'dims': ['g']}},
                {'g': ['a', 'b'], 'p': pd.Series([1.0], index=pd.Index(['z'], name='g'))},
                'that are not coordinates',
                id='unknown-coord',
            ),
        ],
    )
    def test_refused_shapes(self, dims, params, data, match):
        s = _schema(dims=dims, params=params)
        with pytest.raises(ValueError, match=match):
            tidy = tidy_sources(_program(s), data)
            loader.load_parameters(_program(s), tidy, loader.dimension_coords(_program(s), tidy)[0])


# ---------------------------------------------------------------------------
# where.evaluate_where: the eager reading of a lowered predicate
# ---------------------------------------------------------------------------


@pytest.fixture
def gens():
    """A dataset and its master coords, with one generator masked out by p_max."""
    ds = xr.Dataset({'p_max': xr.DataArray([100, 0, 50], dims=['g'], coords={'g': ['wind', 'solar', 'gas']})})
    return ds, {'g': pd.Index(['wind', 'solar', 'gas'], name='g')}


def _lowered(text, parameters=('p_max',), dimensions=('g',)):
    """The program a model with predicate *text* lowers to, and that predicate.

    Through a whole model rather than the resolver directly: the resolver is
    the language's, and a model with the right names in it is the only handle
    this side of the seam has on one predicate.
    """
    from math_spec import to_program

    spec = {
        'dimensions': {d: {'dtype': 'int' if d == 't' else 'str'} for d in dimensions},
        'parameters': {name: {'dims': list(dimensions)} for name in parameters},
        'variables': {'x': {'foreach': list(dimensions), 'where': text, 'bounds': {'lower': 0, 'upper': 1}}},
        'objective': {'sense': 'minimize', 'expression': 'sum(x)'},
    }
    program = to_program(spec)
    return program, program.variables['x'].where


def _context(program, dataset, master_coords):
    return where.EvaluationContext(dataset, master_coords, linopy.Model(), {}, program)


def test_no_where_is_a_scalar_true(gens):
    program, _ = _lowered('p_max')
    mask = where.evaluate_where(None, _context(program, *gens))
    assert mask.ndim == 0
    assert bool(mask) is True


def test_a_bare_parameter_name_is_an_existence_check(gens):
    program, node = _lowered('p_max')
    assert where.evaluate_where(node, _context(program, *gens)).all()


def test_a_comparison_masks_per_coordinate(gens):
    program, node = _lowered('p_max > 0')
    mask = where.evaluate_where(node, _context(program, *gens))
    assert [bool(mask.sel(g=g)) for g in ('wind', 'solar', 'gas')] == [True, False, True]


def test_a_dimension_comparison_masks_on_the_coordinate_itself():
    program, node = _lowered('t > 0', parameters=(), dimensions=('t',))
    mask = where.evaluate_where(node, _context(program, xr.Dataset(), {'t': pd.Index([0, 1, 2], name='t')}))
    assert [bool(mask.sel(t=t)) for t in (0, 1, 2)] == [False, True, True]


def test_a_missing_parameter_is_a_load_error():
    """Was: a scalar-False mask, i.e. a silently empty model. Resolution
    makes an undeclared name a load error in both lanes."""
    with pytest.raises(LanguageError, match="'nonexistent' not found"):
        _lowered('nonexistent')


# ---------------------------------------------------------------------------
# error notes: the context add_note() carries out of build
# ---------------------------------------------------------------------------

_MINIMAL = """
    dimensions:
      g: {dtype: str}
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
_NO_ROWS = {'g': ['a'], 'cap': pd.Series([], index=pd.Index([], name='g', dtype='object'), dtype='float64')}


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
        lpspec_linopy.build(bad, {'g': ['a'], **data})

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
    spec = tmp_path / 'masked.yaml'
    spec.write_text(
        textwrap.dedent("""
            dimensions: {f: {dtype: str}}
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
        data = {{'f': ['a', 'b'], 'gate': pd.Series({{'a': True}}), 'relmax': pd.Series({{'a': 0.5, 'b': 0.5}})}}
        m = fkl.build({str(spec)!r}, data)
        m.solve(solver_name='highs', output_flag=False)
        native = lps.solve({str(spec)!r}, {{
            'f': ['a', 'b'],
            'gate': pl.DataFrame({{'f': ['a'], 'value': [True]}}),
            'relmax': pl.DataFrame({{'f': ['a', 'b'], 'value': [0.5, 0.5]}}),
        }})
        print(float(m.objective.value), native.objective)
    """)
    out = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, check=True)
    eager, native = (float(v) for v in out.stdout.split())
    assert eager == pytest.approx(native), f'lanes disagree outside the harness: {eager} vs {native}'
    assert native == pytest.approx(125.0), 'the masked row should be dropped, leaving x[b] at its bound'


#: A scalar switch gates one variable; the other keeps the model non-empty
#: whichever way the switch is thrown.
SCALAR_SWITCH = {
    'dimensions': {'i': {'dtype': 'int'}},
    'parameters': {'on': {'dims': [], 'dtype': 'bool'}},
    'variables': {
        'x': {'foreach': ['i'], 'bounds': {'lower': 1, 'upper': 5}, 'where': 'on'},
        'y': {'foreach': ['i'], 'bounds': {'lower': 2, 'upper': 5}},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(x) + sum(y)'},
}


@pytest.mark.parametrize(('on', 'expected'), [pytest.param(True, 6.0, id='on'), pytest.param(False, 4.0, id='off')])
def test_a_where_on_a_scalar_bool_agrees_on_both_lanes(on, expected):
    """Was: the lane loaded a dims-less bool through ``float()``, so ``False``
    arrived as ``0.0`` and a bare ``where: on`` read it as *defined* — x was
    built at every coordinate while the relational lane built none.
    """
    with differential(SCALAR_SWITCH, {'i': [1, 2], 'on': on}) as agreed:
        assert agreed.oracle == pytest.approx(expected), 'x is built only where the switch is on'


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
    spec = yaml_file("""
        dimensions:
          f: {dtype: str}
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
        'f': ['a', 'b'],
        'ub': pd.Series([10.0], index=pd.Index(['a'], name='f')),
        'live': pd.Series([True], index=pd.Index(['a'], name='f')),
    }

    with pytest.raises(DataError, match='NULL bounds'):
        lpspec_linopy.build(spec, data)

    masked = yaml_file(
        spec.read_text().replace('{foreach: [f], bounds:', '{foreach: [f], where: live, bounds:'),
        'masked.yaml',
    )
    built = lpspec_linopy.build(masked, data)
    assert 'x' in built.variables


# ---------------------------------------------------------------------------
# named expressions: one reader per lane, one answer (#562)
# ---------------------------------------------------------------------------

EXPRESSION_YAML = """
dimensions:
  snapshot: {dtype: int}
  generator: {dtype: str}
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
    'snapshot': [0, 1, 2],
    'generator': ['g1', 'g2'],
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
  snapshot: {dtype: int}
  generator: {dtype: str}
  bp: {dtype: int}
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


MASKED_CURVE_DATA = {
    'snapshot': [0],
    'generator': ['hydro', 'gas'],
    'bp': [0, 1, 2, 3],
    'p_max': pd.Series({'hydro': 40.0, 'gas': 80.0}),
    'load': pd.Series([50.0], index=pd.RangeIndex(1, name='snapshot')),
    'bp_x': curve_frame(
        {('hydro', 0): 0.0, ('hydro', 1): 40.0, ('gas', 0): 0.0, ('gas', 1): 20.0, ('gas', 2): 50.0, ('gas', 3): 80.0}
    ),
    'bp_y': curve_frame(
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
    runs is then derived from that parameter's rows — by `derive_curve_sources`,
    which reads `piecewise:`. The expansion has cleared `piecewise:`, so handing
    it the expanded model asks for `cost_curve_points` and derives nothing:
    `DataError: no data provided for parameter 'cost_curve_points'`, naming a
    parameter no caller wrote and none can supply. `build` passed the file and
    this reader passed the expansion, which is the only reason one worked.
    """
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


# ---------------------------------------------------------------------------
# reported-grade named expressions: duals and nonlinear bodies (#287)
# ---------------------------------------------------------------------------

#: A reported name calls `dual()` or is nonlinear past the degree-2 ceiling, so
#: no linopy term can hold it and the reader folds it over the solved primal and
#: the duals. `where: p_max > 0` lets the absence case mask a generator out by
#: giving it a zero cap. `sqsum` sits at the ceiling and stays math grade — the
#: nod that the boundary move did not sweep a quadratic into the reported path.
REPORTED_YAML = """
dimensions:
  snapshot: {dtype: int}
  generator: {dtype: str}
parameters:
  p_max: {dims: [generator]}
  cost: {dims: [generator]}
  load: {dims: [snapshot]}
  rate: {dims: []}
  horizon: {dims: []}
variables:
  p:
    foreach: [snapshot, generator]
    bounds: {lower: 0, upper: p_max}
    where: p_max > 0
expressions:
  price: dual(balance)
  lcoe: sum(p * cost, over=generator) / sum(p, over=generator)
  sqsum: sum(p * p, over=generator)
  share: p / sum(p, over=generator)
  growth: (1 + rate) ** horizon
constraints:
  balance:
    foreach: [snapshot]
    expression: sum(p, over=generator) == load
objective:
  sense: minimize
  expression: sum(sum(p * cost, over=generator), over=snapshot)
"""

REPORTED_DATA = {
    'snapshot': [0, 1, 2],
    'generator': ['g1', 'g2'],
    'p_max': pd.Series({'g1': 100.0, 'g2': 100.0}),
    'cost': pd.Series({'g1': 10.0, 'g2': 20.0}),
    'load': pd.Series({0: 50.0, 1: 120.0, 2: 80.0}),
    'rate': 0.05,
    'horizon': 3,
}


@pytest.fixture
def reported_run(yaml_file):
    """A built and solved model plus its file, for reading reported names off."""
    path = yaml_file(REPORTED_YAML, 'reported.yaml')
    model = lpspec_linopy.build(path, dict(REPORTED_DATA))
    model.solve(solver_name='highs', output_flag=False)
    return model, path


def test_a_dual_reads_the_constraints_shadow_price(reported_run):
    model, path = reported_run
    got = lpspec_linopy.expression(model, path, 'price', dict(REPORTED_DATA))
    assert got.equals(model.constraints['balance'].dual), (
        'the linopy lane has no dual accessor of its own — reading dual() through expression() is it, '
        "so it must hand back exactly the constraint's shadow price"
    )


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        pytest.param('lcoe', lambda p, cost: (p * cost).sum('generator') / p.sum('generator'), id='a-variable-divisor'),
        pytest.param('sqsum', lambda p, cost: (p * p).sum('generator'), id='a-degree-two-product-stays-math-grade'),
    ],
)
def test_a_nonlinear_reported_expression_folds_over_the_primal(reported_run, name, expected):
    model, path = reported_run
    p = model.variables['p'].solution
    cost = REPORTED_DATA['cost'].to_xarray().rename(index='generator')
    got = lpspec_linopy.expression(model, path, name, dict(REPORTED_DATA))
    assert got.round(9).equals(expected(p, cost).round(9)), (
        f"'{name}' must equal the ratio hand-computed from the solved primal"
    )


def test_a_data_only_reported_expression_evaluates_without_a_solve(yaml_file):
    path = yaml_file(REPORTED_YAML, 'reported.yaml')
    built = lpspec_linopy.build(path, dict(REPORTED_DATA))
    got = lpspec_linopy.expression(built, path, 'growth', dict(REPORTED_DATA))
    assert float(got) == pytest.approx(1.05**3), (
        '(1 + rate) ** horizon carries no variable and no dual, so it reads off the bound parameters '
        'with no solution present'
    )


def _reported_spec(absence: str) -> dict:
    """The reported model with ``share`` and a variable carrying a chosen ``absence:``."""
    return {
        'dimensions': {'snapshot': {'dtype': 'int'}, 'generator': {'dtype': 'str'}},
        'parameters': {
            'p_max': {'dims': ['generator']},
            'cost': {'dims': ['generator']},
            'load': {'dims': ['snapshot']},
        },
        'variables': {
            'p': {
                'foreach': ['snapshot', 'generator'],
                'bounds': {'lower': 0, 'upper': 'p_max'},
                'where': 'p_max > 0',
                'absence': absence,
            }
        },
        'expressions': {'share': 'p / sum(p, over=generator)'},
        'constraints': {'balance': {'foreach': ['snapshot'], 'expression': 'sum(p, over=generator) == load'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(sum(p * cost, over=generator), over=snapshot)'},
    }


@pytest.mark.parametrize(
    ('absence', 'masked_reads_nan'),
    [
        pytest.param('undefined', True, id='undefined-leaves-a-masked-slot-nan'),
        pytest.param('zero', False, id='zero-fills-a-masked-slot'),
    ],
)
def test_absence_in_a_reported_read_mirrors_the_variable_term(absence, masked_reads_nan):
    """A masked coordinate reads the way its variable's ``absence:`` says — NaN under
    ``undefined``, a hard zero under ``zero`` — the same choice the term path makes."""
    spec = _reported_spec(absence)
    data = {k: REPORTED_DATA[k] for k in ('snapshot', 'generator', 'cost', 'load')}
    data['p_max'] = pd.Series({'g1': 200.0, 'g2': 0.0})
    model = lpspec_linopy.build(spec, data)
    model.solve(solver_name='highs', output_flag=False)
    share = lpspec_linopy.expression(model, spec, 'share', data)

    masked = share.sel(generator='g2')
    assert bool(masked.isnull().all()) is masked_reads_nan, (
        f"under absence '{absence}' the masked generator's share must read "
        f'{"NaN" if masked_reads_nan else "a filled zero"}'
    )
    if not masked_reads_nan:
        assert float(masked.max()) == pytest.approx(0.0), 'a zero-absence masked slot contributes a hard zero'
    assert not bool(share.sel(generator='g1').isnull().any()), 'the surviving generator keeps a value at every snapshot'


@pytest.mark.parametrize(
    ('name', 'reported'),
    [
        pytest.param('price', True, id='a-dual'),
        pytest.param('lcoe', True, id='a-variable-divisor'),
        pytest.param('share', True, id='a-variable-divisor-keeping-a-dim'),
        pytest.param('sqsum', False, id='a-degree-two-product-is-math-grade'),
        pytest.param('growth', False, id='a-variable-free-power-needs-no-solve'),
    ],
)
def test_the_reader_grades_a_body_by_whether_a_term_can_hold_it(yaml_file, name, reported):
    """The degree-2 ceiling is the boundary: a quadratic stays math grade, a
    variable divisor or a dual does not."""
    from math_spec import to_program

    from lpspec.linopy.builder import reads_off_the_solution

    program = to_program(yaml_file(REPORTED_YAML, 'reported.yaml'))
    assert reads_off_the_solution(program.named_expressions[name]) is reported, (
        f"'{name}' is graded on the wrong side of the degree-2 ceiling"
    )


def test_one_set_of_tables_reaches_both_lanes(dispatch_yaml, dispatch_frame_inputs, tmp_path):
    """The claim the shape work is for: one `sources` mapping, either lane.

    polars frames and a parquet path, handed to both unchanged — no per-lane
    conversion at the call site, which is what made the two accepted-input sets
    a divergence a user hit directly (#60).
    """
    frames = dispatch_frame_inputs
    path = tmp_path / 'load.parquet'
    frames['load'].write_parquet(path)
    sources = {**frames, 'load': path}

    with differential(dispatch_yaml, sources) as run:
        assert run.result.primal('p').height, 'the relational lane built no rows'
        assert float(run.model.variables['p'].labels.count()), 'the eager lane built no variables'


@pytest.mark.parametrize(
    'as_spec',
    [
        pytest.param(lambda raw, path: path, id='a-path'),
        pytest.param(lambda raw, path: raw, id='a-mapping'),
        pytest.param(lambda raw, path: schema_of(raw), id='a-loaded-model'),
    ],
)
def test_the_lane_takes_a_model_the_same_three_ways_the_runner_does(tmp_path, as_spec):
    """`lps.build` and this take the same first argument, so neither decides the lane.

    A path was the only spelling here while the runner took all three, which
    made "convert this to a linopy.Model instead" a rewrite of the call rather
    than a change of import (#845).
    """
    import yaml as pyyaml

    raw = {
        'dimensions': {'g': {'dtype': 'str'}},
        'parameters': {'cap': {'dims': ['g']}},
        'variables': {'x': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
    }
    path = tmp_path / 'm.yaml'
    path.write_text(pyyaml.safe_dump(raw))

    built = lpspec_linopy.build(as_spec(raw, path), {'g': ['wind', 'gas'], 'cap': {'wind': 40.0, 'gas': 100.0}})
    assert 'x' in built.variables, 'the same file, whichever way it was handed over'


#: A shift over a variable-free expression: the vacated positions have no
#: value, and inventing one silently pins a bound to zero.
_BARE_SHIFT = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'eff': {'dims': ['t']}},
    'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 5}}},
    'constraints': {'c': {'foreach': ['t'], 'expression': 'x <= shift(eff, over=t, offset=1)'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
}


def test_a_construct_the_streaming_lane_refuses_is_refused_here_too():
    """One gate, both lanes — hard rule 3 held mechanically rather than by care.

    This lane used to load and expand and stop there, so the lowering pass's
    refusals never fired on it: a bare `shift()` over data built a
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

    spec = {
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
        lps.build(spec, sources).close()
    with pytest.raises(DataError, match="dimension 'g' has no index") as eager:
        lpspec_linopy.build(spec, sources)
    assert str(native.value) == str(eager.value), 'one refusal, one wording'

    indexed = {**sources, 'g': pd.DataFrame({'g': ['wind', 'gas']})}
    assert 'x' in lpspec_linopy.build(spec, indexed).variables


def test_from_yaml_fails_before_data_validation(tmp_path):
    """A typo in an expression errors even when data= is absent."""
    f = tmp_path / 'm.yaml'
    f.write_text(
        'dimensions:\n'
        '  g: {dtype: str}\n'
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


def test_dispatch_yaml_agrees_variable_by_variable(dispatch_inputs):
    """The two lanes agree variable by variable, not only in total.

    An objective can agree while the dispatch behind it differs, which is what
    this rules out.
    """
    data = dispatch_inputs

    with differential(EXAMPLES_DIR / 'dispatch.yaml', data, lp=True) as run:
        eager_p = run.model.solution['p'].to_dataframe(name='value').reset_index()
        rel_p = run.result.to_pandas('p')
        merged = eager_p.merge(rel_p, on=['snapshot', 'generator'], suffixes=('_eager', '_rel'))
        assert len(merged) == len(rel_p), 'nothing is masked here, so the rows align 1:1'
        assert np.allclose(merged['value_eager'], merged['value_rel'], atol=1e-6)
