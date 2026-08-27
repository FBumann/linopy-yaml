"""The lowering pass: a resolved model in, a logical plan out.

The plan is read back node by node rather than through the answer it produces:
it is the contract both lanes are written against, so its *shape* is the thing
under test here and what either lane then builds from it is not.

Nothing in this module binds data, builds a model or names a lane. That is the
point of it — the pass has one input and one output, both of them values, and a
test that needed a solver to reach it would be testing the assembly instead.
Lowering's verdict reaching a caller is ``test_language_boundary.py``; the two
lanes agreeing about it is ``test_degree_parity.py`` and its siblings.
"""

from __future__ import annotations

import pytest
from math_spec import DimensionError, LanguageError, Model, Namespace, expand_piecewise

from lpspec.lowering import _lower_where, _Lowering, lower_program
from lpspec.plan import (
    At,
    DimensionComparison,
    DimensionDeclaration,
    Divide,
    LookupDeclaration,
    Parameter,
    ParameterComparison,
    ParameterDefined,
    Power,
    Program,
    Sum,
    Variable,
    divisor_parameters,
    quotients,
    variables_of,
)
from tests.conftest import EXAMPLES_DIR, resolved, schema_of

DISPATCH_YAML = EXAMPLES_DIR / 'dispatch.yaml'


@pytest.fixture
def dispatch_schema() -> Model:
    return schema_of(DISPATCH_YAML)


# ---------------------------------------------------------------------------
# the plan the language lowers to
# ---------------------------------------------------------------------------


def test_lower_program_structure(dispatch_schema):
    program = lower_program(expand_piecewise(dispatch_schema))

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
    assert program.objective.expression == Sum(Variable('p') * Parameter('cost'), ('generator', 'snapshot')), (
        'the objective carries the sum the file wrote, over the dims it named none of'
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

    The dim algebra and alpha.4 settled the language question: summing over a dim
    the operand does not carry builds a model that solves and is wrong, so it
    is an error rather than the silent identity it once was. ``check_schema``
    raises it for anything entering through ``lps.check``; this pins that
    ``_Lowering.expr`` does not quietly disagree one layer down, which is what it
    used to do — it returned the operand unchanged, and the comment claiming
    eager parity outlived the parity.
    """
    with pytest.raises(DimensionError, match='no-op that builds and solves wrong'):
        _Lowering(dispatch_schema, 't').expr(resolved('sum(load, over=generator)', dispatch_schema))


def test_a_power_lowers_only_where_no_variable_is_under_it(dispatch_schema):
    """roll/shift lower to plan.Translate and binary/integer to variable_type;
    `**` lowers to plan.Power, but only over operands that carry no variable —
    with one under it there is no affine reading and nowhere to go."""
    lowered = _Lowering(dispatch_schema, 't').expr(resolved('cost ** cost', dispatch_schema))
    assert isinstance(lowered, Power), 'a variable-free power has a plan node of its own'

    with pytest.raises(LanguageError, match='over variables'):
        _Lowering(dispatch_schema, 't').expr(resolved('p ** 2', dispatch_schema))


def test_a_binary_variable_lowers_to_a_vtype():
    program = lower_program(
        expand_piecewise(schema_of(DISPATCH_YAML, **{'variables.p.domain': 'binary', 'variables.p.bounds': {}}))
    )
    assert program.variable('p').variable_type == 'binary'


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
    pulled = At(quotient, over='flow', coordinate=('component',), into=('component',))

    assert divisor_parameters(pulled) == frozenset({'rate'})
    assert divisor_parameters(Sum(pulled, ('flow',))) == frozenset({'rate'})


def test_a_quotient_is_found_whole_so_its_two_halves_stay_paired():
    """`divisor_parameters` flattens, and one caller cannot use the flat answer.

    A divisor has to have values wherever the row is built *and the numerator
    exists*, so the eager lane narrows the mask by the variables in that
    quotient's own numerator — which needs the pair, not the union. Two
    quotients in one expression is the case a flattened set gets wrong.
    """
    left = Divide(Variable('x'), Parameter('rate'))
    right = Divide(Variable('y'), Parameter('loss'))

    found = quotients(Sum(left + right, ('flow',)))
    assert [(variables_of(q.numerator), q.divisor) for q in found] == [
        (frozenset({'x'}), Parameter('rate')),
        (frozenset({'y'}), Parameter('loss')),
    ], 'each quotient keeps its own numerator, in the order the expression writes them'
    assert divisor_parameters(Sum(left + right, ('flow',))) == frozenset({'rate', 'loss'}), (
        'the flat answer is still the union of the same walk'
    )


def test_a_lookup_names_the_dimension_its_values_label():
    """Five sites asked this and each walked for it; the plan answers it once.

    Both shapes are here because both had callers: one dimension's maps, for an
    operator that partitions along it, and every map in the program, for
    binding, which reads them all before it knows which are used.
    """
    program = Program(
        (),
        (),
        (),
        None,
        dimensions=(
            DimensionDeclaration('snapshot', (LookupDeclaration('season_of', 'season'),)),
            DimensionDeclaration('generator', (LookupDeclaration('at_bus', 'bus'),)),
        ),
    )

    assert program.dimension('snapshot').targets == {'season_of': 'season'}, (
        'one dimension names its own maps and no other dimension'
    )
    assert program.dimension('generator').targets == {'at_bus': 'bus'}, 'and the same for the second'
    assert program.dimension('nothing_declared').targets == {}, 'a dimension with no maps has none to name'
    assert [(d, lk.name) for d, lk in program.lookups] == [
        ('snapshot', 'season_of'),
        ('generator', 'at_bus'),
    ], 'every map with the dimension it is over, in declaration order'
