"""The plan boundary: a malformed Program is refused before any engine sees it.

The plan is a public IR, so a program built by hand used to die mid-query in
whatever error the compiler's fragments hit first (#1134). `Program.check`
holds it at the boundary, in plan vocabulary; each case here is one invariant
and the sentence it fails with. A valid program passes untouched, and every
lowered program passes by construction — the language refuses each of these
shapes at load, which the differential suite exercises end to end.
"""

from __future__ import annotations

import pytest

from lpspec import plan
from lpspec.errors import LanguageError
from lpspec.relational.engines.polars.engine import PolarsEngine

DIMENSIONS = (
    plan.DimensionDeclaration('g', (plan.LookupDeclaration('zone_of', 'zone'),)),
    plan.DimensionDeclaration('zone'),
    plan.DimensionDeclaration('t'),
)
PARAMETERS = (
    plan.ParameterDeclaration('cost', ('g',)),
    plan.ParameterDeclaration('load', ('zone',)),
    plan.ParameterDeclaration('lead', ('t',)),
    plan.ParameterDeclaration('width', ('g',)),
)
VARIABLES = (
    plan.VariableDeclaration('x', ('g',)),
    plan.VariableDeclaration('u', ('t', 'g')),
)


def constrained(lhs: plan.Expression, dims: tuple[str, ...] = ()) -> plan.Program:
    """A program whose one constraint carries *lhs* — the flaw under test rides in the expression."""
    constraint = plan.ConstraintDeclaration('c', dims, lhs=lhs, sense='<=', rhs=plan.Constant(0.0))
    return plan.Program(PARAMETERS, VARIABLES, (constraint,), None, DIMENSIONS)


def x() -> plan.Expression:
    return plan.Variable('x')


@pytest.mark.parametrize(
    ('program', 'match'),
    [
        pytest.param(
            constrained(plan.Variable('y'), ('g',)),
            "unknown variable 'y'",
            id='a-variable-nothing-declares',
        ),
        pytest.param(
            constrained(plan.Multiply(x(), plan.Parameter('price')), ('g',)),
            "unknown parameter 'price'",
            id='a-parameter-nothing-declares',
        ),
        pytest.param(
            plan.Program((plan.ParameterDeclaration('x', ('g',)), *PARAMETERS), VARIABLES, (), None, DIMENSIONS),
            "'x' is declared twice",
            id='one-name-two-declarations',
        ),
        pytest.param(
            constrained(plan.Sum(x(), over=('t',))),
            "sum over \\['t'\\], which the operand does not span",
            id='a-sum-over-a-dim-the-operand-lacks',
        ),
        pytest.param(
            constrained(plan.GroupSum(x(), over='g', coordinate=('zone_of', 'other'), into=('zone',)), ('zone',)),
            r'2 lookup\(s\) paired with 1 target dimension\(s\)',
            id='a-grouping-whose-tuples-do-not-pair',
        ),
        pytest.param(
            constrained(plan.GroupSum(x(), over='g', coordinate=('zone_of',), into=('t',)), ('t',)),
            "lookup 'zone_of' targets 'zone', not 't'",
            id='a-grouping-into-a-dim-that-is-not-the-target',
        ),
        pytest.param(
            constrained(plan.GroupSum(plan.Parameter('load'), over='g', coordinate=('zone_of',), into=('zone',))),
            "sum\\(by=\\) over 'g', which the operand does not span",
            id='a-grouping-over-a-dim-the-operand-lacks',
        ),
        pytest.param(
            constrained(plan.At(x(), over='g', coordinate=('zone_of',), into=('zone',)), ('g',)),
            "at\\(\\) through \\['zone'\\], which the operand does not span",
            id='a-pullback-through-a-dim-the-operand-lacks',
        ),
        pytest.param(
            constrained(plan.Translate(x(), 't', offset=1), ('g',)),
            "shift\\(\\) along 't', which the operand does not span",
            id='a-translation-along-a-dim-the-operand-lacks',
        ),
        pytest.param(
            constrained(plan.Translate(plan.Variable('u'), 't', offset='lead'), ('t', 'g')),
            "shift\\(\\) distance 'lead' varies along 't'",
            id='an-offset-that-varies-along-the-walked-dim',
        ),
        pytest.param(
            constrained(plan.Window(plan.Variable('u'), 't', width='lead'), ('t', 'g')),
            "sum_back\\(\\) distance 'lead' varies along 't'",
            id='a-width-that-varies-along-the-walked-dim',
        ),
        pytest.param(
            constrained(plan.Multiply(plan.Multiply(x(), x()), x()), ('g',)),
            'a product of degree 3',
            id='a-cubic-product',
        ),
        pytest.param(
            constrained(plan.Divide(x(), x()), ('g',)),
            'the divisor contains variables',
            id='a-divisor-carrying-a-variable',
        ),
        pytest.param(
            constrained(plan.Power(x(), plan.Constant(2.0)), ('g',)),
            'a power over variables',
            id='a-power-over-a-variable',
        ),
        pytest.param(
            plan.Program(
                PARAMETERS,
                (plan.VariableDeclaration('x', ('g',), lower=plan.Variable('x')),),
                (),
                None,
                DIMENSIONS,
            ),
            'unsupported node Variable',
            id='a-bound-carrying-a-variable',
        ),
        pytest.param(
            plan.Program(
                PARAMETERS,
                (plan.VariableDeclaration('x', ('g',), upper=plan.Parameter('nope')),),
                (),
                None,
                DIMENSIONS,
            ),
            "bounds of variable 'x'.*unknown parameter 'nope'",
            id='a-bound-naming-no-parameter',
        ),
        pytest.param(
            plan.Program(
                PARAMETERS,
                (plan.VariableDeclaration('x', ('g',), where=plan.ParameterComparison('nope', '>', 0)),),
                (),
                None,
                DIMENSIONS,
            ),
            "variable 'x'.*unknown parameter 'nope'",
            id='a-mask-naming-no-parameter',
        ),
        pytest.param(
            plan.Program(
                PARAMETERS,
                VARIABLES,
                (),
                plan.ObjectiveDeclaration('min', plan.Sum(plan.Variable('y'), over=('g',))),
                DIMENSIONS,
            ),
            "the objective.*unknown variable 'y'",
            id='an-objective-naming-no-variable',
        ),
        pytest.param(
            plan.Program(
                PARAMETERS,
                VARIABLES,
                (),
                None,
                DIMENSIONS,
                (plan.SosDeclaration('s', 'x', 't', sos_type=1),),
            ),
            "over 't', which variable 'x' is not indexed by",
            id='a-set-over-a-dim-its-variable-lacks',
        ),
        pytest.param(
            plan.Program(PARAMETERS, VARIABLES, (), None, DIMENSIONS, (), {'reach': plan.Parameter('gone')}),
            "named expression 'reach'.*unknown parameter 'gone'",
            id='a-named-expression-naming-no-parameter',
        ),
    ],
)
def test_a_malformed_program_is_refused_in_plan_vocabulary(program: plan.Program, match: str):
    with pytest.raises(LanguageError, match=match):
        program.check()


def test_a_coherent_program_passes():
    """Every construct the checks read, in one valid program — the boundary admits it whole."""
    balance = plan.ConstraintDeclaration(
        'balance',
        ('zone',),
        lhs=plan.GroupSum(plan.Multiply(x(), plan.Parameter('cost')), 'g', ('zone_of',), ('zone',)),
        sense='<=',
        rhs=plan.Parameter('load'),
        where=plan.ParameterDefined('load'),
    )
    ramp = plan.ConstraintDeclaration(
        'ramp',
        ('t', 'g'),
        lhs=plan.Add(plan.Variable('u'), plan.Negate(plan.Translate(plan.Variable('u'), 't', offset='width'))),
        sense='<=',
        rhs=plan.Window(plan.Variable('u'), 't', width=2),
    )
    program = plan.Program(
        PARAMETERS,
        VARIABLES,
        (balance, ramp),
        plan.ObjectiveDeclaration('min', plan.Sum(plan.Multiply(x(), x()), over=('g',))),
        DIMENSIONS,
        (plan.SosDeclaration('s', 'x', 'g', sos_type=2),),
    )
    assert program.check() is None, 'a coherent program is admitted without complaint'


def test_the_engine_holds_every_program_to_the_boundary():
    """`build` checks before it binds — the wiring, probed where deleting it would show.

    A bound naming an unknown parameter used to surface as a `KeyError` from
    the compiler's declaration lookup, mid-build; the boundary refuses it
    first, before any source is read — which is why no sources are needed to
    reach the refusal.
    """
    program = plan.Program(
        PARAMETERS,
        (plan.VariableDeclaration('x', ('g',), upper=plan.Parameter('nope')),),
        (),
        None,
        DIMENSIONS,
    )
    with PolarsEngine() as engine, pytest.raises(LanguageError, match="unknown parameter 'nope'"):
        engine.build(program, {})
