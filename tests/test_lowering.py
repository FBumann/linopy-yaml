"""Phase-3 gate: YAML lowers to the logical plan and matches the eager backend.

The dispatch example runs through both backends with the same data, and the
lowered ``Program`` is read back node by node — the plan is the contract
between the language and the engine, so its shape is asserted directly rather
than only through the answer it produces.
"""

from __future__ import annotations

import numpy as np
import pytest

import lpspec as lps
from lpspec.errors import DataError, DimensionError, LanguageError
from lpspec.language.model import Model
from lpspec.language.resolution import Namespace
from lpspec.lowering import _lower_expr, _lower_where, lower_program
from lpspec.relational.plan import (
    At,
    DimensionComparison,
    Divide,
    Parameter,
    ParameterComparison,
    ParameterDefined,
    Sum,
    Variable,
    divisor_parameters,
)
from lpspec.sources import tidy_sources
from tests.conftest import EXAMPLES_DIR, resolved, schema_of
from tests.differential import differential

DISPATCH_YAML = EXAMPLES_DIR / 'dispatch.yaml'


@pytest.fixture
def dispatch_schema() -> Model:
    return schema_of(DISPATCH_YAML)


def test_dispatch_yaml_agrees_variable_by_variable(dispatch_inputs):
    """The two lanes agree variable by variable, not only in total.

    An objective can agree while the dispatch behind it differs, which is what
    this rules out.
    """
    data, coords = dispatch_inputs

    with differential(DISPATCH_YAML, data, coords, lp=True) as run:
        eager_p = run.model.solution['p'].to_dataframe(name='value').reset_index()
        rel_p = run.result.to_pandas('p')
        merged = eager_p.merge(rel_p, on=['snapshot', 'generator'], suffixes=('_eager', '_rel'))
        assert len(merged) == len(rel_p), 'nothing is masked here, so the rows align 1:1'
        assert np.allclose(merged['value_eager'], merged['value_rel'], atol=1e-6)


# ---------------------------------------------------------------------------
# the plan the language lowers to
# ---------------------------------------------------------------------------


def test_lower_program_structure(dispatch_schema):
    program = lower_program(dispatch_schema)

    assert [p.name for p in program.parameters] == ['p_max', 'load', 'cost']
    (v,) = program.variables
    assert v.name == 'p'
    assert v.dims == ('snapshot', 'generator')
    assert v.where == ParameterComparison('p_max', '>', 0.0)
    assert v.upper == Parameter('p_max')

    (c,) = program.constraints
    assert c.name == 'power_balance'
    assert c.dims == ('snapshot',)
    assert c.lhs == Sum(Variable('p'), ('generator',))
    assert c.sense == '=='
    assert c.rhs == Parameter('load')

    assert program.objective.sense == 'min'
    assert program.objective.expression == Variable('p') * Parameter('cost'), (
        'no sum: an objective totals every dim it carries, so writing one would restate it'
    )


@pytest.mark.parametrize(
    ('where', 'expected'),
    [
        pytest.param(None, None, id='no-where-at-all'),
        pytest.param('True', None, id='True-is-no-mask'),
        pytest.param('p_max', ParameterDefined('p_max'), id='a-bare-parameter-name'),
        pytest.param(
            'snapshot > 5',
            DimensionComparison('snapshot', '>', 5),
            id='a-dimension-coordinate-compares-like-a-parameter',
        ),
    ],
)
def test_where_lowering(dispatch_schema, where, expected):
    assert _lower_where(where, Namespace.of(dispatch_schema), 't') == expected


def test_a_compound_where_lowers_to_something(dispatch_schema):
    assert _lower_where('p_max > 0 AND NOT load == 0', Namespace.of(dispatch_schema), 't') is not None


def test_an_unknown_where_name_is_an_error_at_lowering_too(dispatch_schema):
    """It used to be a scalar-False mask in the eager lane: a model that
    builds, solves, and is silently empty. Resolution makes it a load error."""
    with pytest.raises(LanguageError, match="'no_such_param' not found"):
        _lower_where('no_such_param', Namespace.of(dispatch_schema), 't')


def test_sum_over_absent_dim_raises_at_lowering_too(dispatch_schema):
    """A no-op sum is an error at *every* layer, not only at the front door.

    SPEC §"dims" and alpha.4 settled the language question: summing over a dim
    the operand does not carry builds a model that solves and is wrong, so it
    is an error rather than the silent identity it once was. ``check_schema``
    raises it for anything entering through ``lps.check``; this pins that
    ``_lower_expr`` does not quietly disagree one layer down, which is what it
    used to do — it returned the operand unchanged, and the comment claiming
    eager parity outlived the parity.
    """
    with pytest.raises(DimensionError, match='no-op that builds and solves wrong'):
        _lower_expr(resolved('sum(load, over=generator)', dispatch_schema), dispatch_schema, 't')


def test_the_power_operator_stays_outside_the_relational_subset(dispatch_schema):
    """roll/shift lower to plan.Translate and binary/integer to variable_type;
    '**' has no affine reading at all, so it has nowhere to go."""
    with pytest.raises(LanguageError, match=r"operator '\*\*'"):
        _lower_expr(resolved('p ** 2', dispatch_schema), dispatch_schema, 't')


def test_a_binary_variable_lowers_to_a_vtype():
    program = lower_program(schema_of(DISPATCH_YAML, **{'variables.p.domain': 'binary', 'variables.p.bounds': {}}))
    assert program.variable('p').variable_type == 'binary'


# ---------------------------------------------------------------------------
# binding an index to a dim: by name where there is one, by position otherwise
# ---------------------------------------------------------------------------

NETWORK = {
    'dimensions': {'from_bus': {'values': ['n1', 'n2']}, 'to_bus': {'values': ['n1', 'n2']}},
    'parameters': {'cap': {'dims': ['from_bus', 'to_bus']}},
    'variables': {'f': {'foreach': ['from_bus', 'to_bus'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'objective': {'sense': 'maximize', 'expression': 'f'},
}

#: Asymmetric, so a transposition changes the answer rather than hiding in it.
CAPS = {('n1', 'n1'): 1.0, ('n2', 'n1'): 5.0, ('n1', 'n2'): 500.0, ('n2', 'n2'): 1.0}


def _tidy_cap(names):
    """`cap` keyed by (from_bus, to_bus), read back off the normalised frame.

    ``tidy_sources`` normalises to a frame, so the columns come back by name —
    which is the point: a transposition shows up as swapped values.
    """
    import pandas as pd

    index = pd.MultiIndex.from_tuples(list(CAPS), names=names)
    frame = tidy_sources(Model(**NETWORK), {'cap': pd.Series(list(CAPS.values()), index=index)})['cap'].collect()
    table = frame.to_dict(as_series=False)
    return dict(zip(zip(table['from_bus'], table['to_bus'], strict=True), table['value'], strict=True))


def test_a_named_index_binds_by_name_not_position():
    """Two dims over the same label space make a transposed index type-check
    and cover every coordinate, so nothing downstream can catch it. Was: the
    declared dims overwrote the user's level names and the matrix came out
    transposed, with no error.
    """
    assert _tidy_cap(['from_bus', 'to_bus']) == CAPS
    assert _tidy_cap(['to_bus', 'from_bus']) == {(f, t): v for (t, f), v in CAPS.items()}


def test_an_unnamed_index_still_binds_positionally():
    assert _tidy_cap([None, None]) == CAPS


def test_an_index_name_outside_the_declared_dims_is_an_error():
    """Refused by binding, which asks it of a parquet path as well as a frame.

    ``tidy_sources`` only ever sees the in-memory half, so asking there too
    would be a second wording of one defect covering fewer sources.
    """
    import pandas as pd

    index = pd.MultiIndex.from_tuples(list(CAPS), names=['banana', 'to_bus'])
    with pytest.raises(DataError, match='is missing columns'):
        lps.build(Model(**NETWORK), {'cap': pd.Series(list(CAPS.values()), index=index)})


def test_a_divisor_under_a_pullback_is_still_named():
    """`children` has to descend through every node, or a refusal loses its name.

    `divisor_parameters` is what turns "a coefficient came out null" into a
    message naming the parameter the caller has to fix, and it finds those names
    by walking `children`. `At` was missing from that walk, so a quotient inside
    `at(...)` reported an uncovered divisor with an empty list where the name
    belongs — the refusal still fired, and stopped saying what to do about it.

    Asked of the walk directly rather than through a build: the walk is static,
    and a test that needed data to reach it would be testing the assembly.
    """
    quotient = Divide(Variable('x'), Parameter('rate'))
    pulled = At(quotient, over='flow', coordinate='component', into='component')

    assert divisor_parameters(pulled) == frozenset({'rate'})
    assert divisor_parameters(Sum(pulled, ('flow',))) == frozenset({'rate'})
