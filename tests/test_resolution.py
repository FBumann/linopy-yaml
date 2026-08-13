"""Name resolution as a pass: one namespace, one answer, both lanes.

The divergences below were all real before resolution moved out of the
backends. Each test names the behaviour each lane used to have.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from lpspec.errors import LanguageError
from lpspec.language.resolution import Namespace, expression_of, where_of
from lpspec.language.validation import validate_expressions
from lpspec.lowering import lower_program
from tests.conftest import DISPATCH_MODEL, schema_of

if TYPE_CHECKING:
    from lpspec.language.model import Model


def _schema(**overrides) -> Model:
    return schema_of(DISPATCH_MODEL, **overrides)


# ---------------------------------------------------------------------------
# the AST that reaches a backend is fully typed
# ---------------------------------------------------------------------------


def test_no_unresolved_name_survives_the_pass():
    from lpspec.language.expression_parser import NameNode

    schema = _schema()
    ast = expression_of('sum(p * cost, over=generator) == load', schema, Namespace.of(schema), 't')

    def walk(node):
        assert not isinstance(node, NameNode), f'unresolved {node}'
        for child in vars(node).values():
            if hasattr(child, '__dataclass_fields__'):
                walk(child)
            elif isinstance(child, (list, dict)):
                for item in child.values() if isinstance(child, dict) else child:
                    if hasattr(item, '__dataclass_fields__'):
                        walk(item)

    walk(ast)


def test_names_are_typed_by_kind():
    from lpspec.language.expression_parser import DimensionNode, ParameterNode, VariableNode

    schema = _schema()
    ast = expression_of('sum(p * cost, over=generator)', schema, Namespace.of(schema), 't')
    assert isinstance(ast.args[0].left, VariableNode)
    assert isinstance(ast.args[0].right, ParameterNode)
    assert isinstance(ast.kwargs['over'], DimensionNode)


def test_dimension_is_not_a_value_in_an_expression():
    schema = _schema()
    with pytest.raises(LanguageError, match='is a dimension'):
        expression_of('p * snapshot', schema, Namespace.of(schema), 't')


# ---------------------------------------------------------------------------
# the divergences resolution removes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('where', 'error', 'match'),
    [
        # Was: scalar-False mask in the eager lane (a model that builds, solves
        # and is silently empty). The model does not survive construction, so
        # lowering never sees it — the earliest place that can tell is the one
        # that tells.
        pytest.param('typo_name > 0', LanguageError, "'typo_name' not found", id='unknown-name'),
        # Was: a parameter comparison in the eager lane, a comparison against
        # the string 'cost' in the relational one.
        pytest.param('p_max > cost', ValueError, 'compares two parameters', id='parameter-vs-parameter'),
        # Was: silently empty. The RHS read as the string 'snapshot', so the
        # mask compared generator coordinates against another dimension's
        # *name*, matched nothing, and the block built with zero rows on both
        # lanes.
        pytest.param('generator == snapshot', ValueError, 'compares against dimension', id='dimension-vs-dimension'),
        pytest.param('p > 0', ValueError, 'built before variables exist', id='a-variable-in-a-where'),
    ],
)
def test_an_ill_formed_where_is_a_load_error(where, error, match):
    with pytest.raises(error, match=match):
        _schema(**{'variables.p.where': where})


def test_string_literal_rhs_still_works():
    """A bare name that is not declared stays a literal — this is how string
    coordinates are compared."""
    schema = _schema(**{'variables.p.where': 'generator == wind'})
    validate_expressions(schema)
    program = lower_program(schema)
    assert program.variables[0].where is not None


# ---------------------------------------------------------------------------
# one flat namespace
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('overrides', 'expected'),
    [
        ({'parameters.snapshot': {'dims': []}}, 'collides with the dimension'),
        ({'variables.load': {'foreach': ['snapshot']}}, 'collides with the parameter'),
        ({'parameters.sum': {'dims': []}}, "collides with the built-in helper 'sum'"),
        ({'dimensions.sum': {'values': [1]}}, "collides with the built-in helper 'sum'"),
        ({'variables.generator': {'foreach': ['snapshot']}}, 'collides with the dimension'),
    ],
)
def test_collisions_are_load_errors(overrides, expected):
    with pytest.raises(ValueError, match=expected):
        _schema(**overrides)


def test_declaring_a_parameter_cannot_change_an_existing_where():
    """The hazard the flat namespace exists to prevent: `where: "snapshot > 0"`
    means the dimension, and declaring a parameter of that name must not
    silently reinterpret it — it is rejected instead."""
    schema = _schema(**{'variables.p.where': 'snapshot > 0'})
    assert where_of('snapshot > 0', Namespace.of(schema), 't').__class__.__name__ == 'DimensionComparisonNode'
    with pytest.raises(ValueError, match='collides with the dimension'):
        _schema(**{'variables.p.where': 'snapshot > 0', 'parameters.snapshot': {'dims': []}})


# ---------------------------------------------------------------------------
# macro formals: the one legitimate scope
# ---------------------------------------------------------------------------


def test_macro_formal_may_shadow_a_parameter():
    schema = _schema(**{'macros.scale': {'args': ['cost'], 'template': 'cost * 2'}})
    validate_expressions(schema)


def test_macro_formal_may_not_shadow_a_dimension():
    with pytest.raises(ValueError, match="formal 'generator' collides with declared dimension"):
        _schema(**{'macros.agg': {'args': ['x'], 'kwargs': ['generator'], 'template': 'sum(x, over=generator)'}})


# ---------------------------------------------------------------------------
# bounds are a narrower language than expressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('bounds', 'match'),
    [
        pytest.param(
            {'lower': 0, 'upper': '2 * p_max'},
            'bounds accept a parameter name or a number, not an expression',
            id='an-expression-is-rejected-with-a-message-that-says-so',
        ),
        pytest.param(
            {'lower': 0, 'upper': 'p_maxx'},
            "'p_maxx' is not a declared parameter",
            id='a-typo-names-the-parameter',
        ),
    ],
)
def test_bounds_reject_what_expressions_allow(bounds, match):
    with pytest.raises(ValueError, match=match):
        _schema(**{'variables.p.bounds': bounds})
