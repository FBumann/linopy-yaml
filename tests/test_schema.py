"""Pydantic-level validation: what a well-formed declaration looks like.

Everything here is decided from the YAML mapping alone — no expressions
parsed, no dims inferred, no data. The rules are mostly of two kinds
(a name that must be declared, a key that must be spelled right), so they are
stated as tables: a new rule is a row, and a rule that silently stops firing
is a row that stops failing.
"""

import pytest

from lpspec.errors import SchemaError
from lpspec.language.model import Model


def test_empty_schema():
    s = Model.model_validate({})
    assert s.dimensions == {}
    assert s.variables == {}


def test_minimal_schema():
    s = Model.model_validate(
        {
            'dimensions': {'x': {'values': [1, 2, 3], 'dtype': 'int'}},
            'parameters': {'a': {'dims': ['x']}},
            'variables': {'v': {'foreach': ['x']}},
        }
    )
    assert 'x' in s.dimensions
    assert s.parameters['a'].dims == ['x']
    assert s.variables['v'].foreach == ['x']


# ---------------------------------------------------------------------------
# a name a declaration uses must be declared
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('section', 'body', 'match'),
    [
        pytest.param('parameters', {'a': {'dims': ['y']}}, "undeclared dimension 'y'", id='parameter-dim'),
        pytest.param('variables', {'v': {'foreach': ['y']}}, "undeclared dimension 'y'", id='variable-foreach'),
        pytest.param(
            'constraints',
            {'c': {'foreach': ['y'], 'expression': 'v == 0'}},
            "undeclared dimension 'y'",
            id='constraint-foreach',
        ),
        pytest.param(
            'variables',
            {'v': {'foreach': ['x'], 'bounds': {'upper': 'nonexistent'}}},
            "'nonexistent' is not a declared parameter",
            id='bound-parameter',
        ),
    ],
)
def test_an_undeclared_name_is_rejected(section, body, match):
    with pytest.raises(SchemaError, match=match):
        Model.model_validate({'dimensions': {'x': {'values': [1], 'dtype': 'int'}}, section: body})


def test_an_omitted_bound_means_unbounded_all_the_way_down():
    """A declaration that omits a bound means unbounded, exactly as in
    ``linopy.Model.add_variables`` — never an implicit ``>= 0``.

    Nothing else pins this: both lanes read the same default, so the
    differential tests agree with each other whatever it is. The second half
    checks the relational lane carries the default through rather than
    re-defaulting it on the way to the plan.
    """
    from lpspec.lowering import _bound_expression

    s = Model.model_validate(
        {'dimensions': {'x': {'values': [1], 'dtype': 'int'}}, 'variables': {'v': {'foreach': ['x']}}}
    )
    bounds = s.variables['v'].bounds

    assert (bounds.lower, bounds.upper) == (float('-inf'), float('inf'))
    assert _bound_expression(bounds.lower).value == float('-inf')
    assert _bound_expression(bounds.upper).value == float('inf')


def test_a_declared_bound_parameter_is_accepted():
    s = Model.model_validate(
        {
            'dimensions': {'x': {'values': [1], 'dtype': 'int'}},
            'parameters': {'p_max': {'dims': ['x']}},
            'variables': {'v': {'foreach': ['x'], 'bounds': {'upper': 'p_max'}}},
        }
    )
    assert s.variables['v'].bounds.upper == 'p_max'


@pytest.mark.parametrize('flag', ['binary', 'integer'])
def test_a_removed_flag_names_its_domain_rewrite(flag):
    body = {'foreach': ['x'], flag: True}
    with pytest.raises(SchemaError, match=f'domain: {flag}'):
        Model.model_validate({'dimensions': {'x': {'values': [1], 'dtype': 'int'}}, 'variables': {'v': body}})


def test_invalid_domain():
    body = {'foreach': ['x'], 'domain': 'boolean'}
    with pytest.raises(SchemaError, match=r'continuous|integer|binary'):
        Model.model_validate({'dimensions': {'x': {'values': [1], 'dtype': 'int'}}, 'variables': {'v': body}})


def test_invalid_sense():
    with pytest.raises(SchemaError, match=r'minimize|maximize'):
        Model.model_validate({'objective': {'sense': 'unknown', 'expression': 'v'}})


# ---------------------------------------------------------------------------
# a misspelled key is a different model, so it is an error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('raw', 'match'),
    [
        pytest.param(
            {'dimenzions': {'x': {'values': [1], 'dtype': 'int'}}},
            "unknown key 'dimenzions' in the top level",
            id='top',
        ),
        pytest.param({'dimensions': {'thing': {'dtypo': 'str'}}}, "unknown key 'dtypo'", id='dimension'),
        pytest.param(
            {
                'dimensions': {'x': {'values': [1], 'dtype': 'int'}},
                'parameters': {'thing': {'dims': ['x'], 'dtyp': 'float'}},
            },
            "unknown key 'dtyp'",
            id='parameter',
        ),
        pytest.param(
            {
                'dimensions': {'x': {'values': [1], 'dtype': 'int'}},
                'macros': {'thing': {'template': 'a + b', 'arg': ['a']}},
            },
            "unknown key 'arg'",
            id='macro',
        ),
        pytest.param(
            {
                'dimensions': {'x': {'values': [1], 'dtype': 'int'}},
                'piecewise': {'thing': {'over': 'x', 'links': [['v', 'p'], ['w', 'q']], 'convx': True}},
            },
            "unknown key 'convx'",
            id='piecewise',
        ),
        pytest.param(
            {
                'dimensions': {'x': {'values': [1], 'dtype': 'int'}},
                'variables': {'v': {'foreach': ['x'], 'bounds': {'lowerr': 0}}},
            },
            "unknown key 'lowerr' in a bounds block",
            id='nested-bounds',
        ),
        pytest.param(
            {
                'dimensions': {'x': {'values': [1], 'dtype': 'int'}},
                'variables': {'v': {'foreach': ['x']}},
                'constraints': {'c': {'foreach': ['x'], 'expresion': 'v >= 0'}},
            },
            "unknown key 'expresion' in a constraint declaration",
            id='nested-constraint',
        ),
    ],
)
def test_an_unknown_key_is_rejected(raw, match):
    """Strictness lives on the shared `_StrictBlock` base, so no model can opt
    out of it by omission — one case per model to prove it."""
    with pytest.raises(SchemaError, match=match):
        Model.model_validate(raw)


@pytest.mark.parametrize(
    ('block', 'match'),
    [
        pytest.param(
            {'foreach': ['x'], 'boundz': {'lower': 0}},
            r"unknown key 'boundz'.*Did you mean 'bounds'",
            id='a-near-miss-is-named',
        ),
        pytest.param(
            {'foreach': ['x'], 'zzzz': 1},
            'Valid keys: bounds, domain, foreach, where',
            id='anything-else-lists-the-valid-keys',
        ),
    ],
)
def test_a_near_miss_is_named_and_anything_else_lists_the_valid_keys(block, match):
    """A misspelled key used to be dropped, leaving the variable unbounded —
    so the message has to be good enough to act on without reading the source."""
    base = {'dimensions': {'x': {'values': [1], 'dtype': 'int'}}}

    with pytest.raises(SchemaError, match=match):
        Model.model_validate({**base, 'variables': {'v': block}})


# ---------------------------------------------------------------------------
# lookups
# ---------------------------------------------------------------------------


def test_coords_under_a_dimension_is_refused_with_the_lookup_rewrite():
    """`coords:` is gone, and the refusal names the top-level `lookups:` rewrite
    — the dedicated message, not the generic unknown-key near miss."""
    with pytest.raises(SchemaError, match="'coords:' under a dimension was removed") as caught:
        Model.model_validate(
            {'dimensions': {'bus': {'values': ['n']}, 'generator': {'values': ['w'], 'coords': ['bus']}}}
        )
    assert 'bus_of: {over: generator, into: bus}' in str(caught.value), (
        'the refusal has to show the lookups: block a coords: file rewrites into'
    )


def test_two_lookups_may_map_one_dimension_onto_one_target():
    s = Model.model_validate(
        {
            'dimensions': {'bus': {'values': ['n']}, 'line': {'values': ['l1']}},
            'lookups': {'from': {'over': 'line', 'into': 'bus'}, 'to': {'over': 'line', 'into': 'bus'}},
        }
    )
    assert s.targeted_of('line') == {'from': 'bus', 'to': 'bus'}


@pytest.mark.parametrize(
    ('lookups', 'match'),
    [
        pytest.param(
            {'gen_bus': {'over': 'generator', 'into': 'bus'}},
            "targets undeclared dimension 'bus'",
            id='target-undeclared',
        ),
        pytest.param(
            {'gen_bus': {'over': 'plant', 'into': 'generator'}},
            "is over undeclared dimension 'plant'",
            id='over-undeclared',
        ),
        pytest.param(
            {'gen_gen': {'over': 'generator', 'into': 'generator'}},
            "maps 'generator' into itself",
            id='target-self',
        ),
        pytest.param(
            {'generator': {'over': 'generator', 'into': 'zone'}},
            "Lookup 'generator' collides with the dimension of the same name",
            id='a-lookup-may-not-take-a-dimensions-name-its-own-dim-included',
        ),
        pytest.param(
            {'zone': {'over': 'generator', 'into': 'zone'}},
            "Lookup 'zone' collides with the dimension of the same name",
            id='a-lookup-may-not-take-its-targets-name',
        ),
        pytest.param(
            {'gen_bus': {'over': 'generator'}},
            "exactly one of 'into:'",
            id='neither-kind',
        ),
        pytest.param(
            {'gen_bus': {'over': 'generator', 'into': 'zone', 'dtype': 'str'}},
            "exactly one of 'into:'",
            id='both-kinds',
        ),
    ],
)
def test_an_ill_formed_lookup_is_rejected(lookups, match):
    with pytest.raises(SchemaError, match=match):
        Model.model_validate(
            {'dimensions': {'generator': {'values': ['w']}, 'zone': {'values': ['z']}}, 'lookups': lookups}
        )
