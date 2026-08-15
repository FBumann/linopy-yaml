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
            'Valid keys: bounds, description, domain, foreach, where',
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


# ---------------------------------------------------------------------------
# `description:` — free text on every declaration kind, never parsed (#222)
# ---------------------------------------------------------------------------


DESCRIBED = {
    'dimensions': {
        'snapshot': {'dtype': 'int', 'values': [0, 1], 'description': 'the operational hours'},
        'bp': {'dtype': 'int', 'values': [0, 1]},
    },
    'lookups': {
        'period': {'over': 'snapshot', 'dtype': 'int', 'description': 'the month a snapshot falls in'},
    },
    'parameters': {
        'load': {'dims': ['snapshot'], 'description': 'demand per hour'},
        'bp_x': {'dims': ['bp']},
        'bp_y': {'dims': ['bp']},
    },
    'variables': {
        'p': {'foreach': ['snapshot'], 'bounds': {'lower': 0, 'upper': 100}, 'description': 'dispatch'},
        'op_cost': {'foreach': ['snapshot'], 'bounds': {'lower': 0}},
    },
    'constraints': {
        'balance': {'foreach': ['snapshot'], 'expression': 'p == load', 'description': 'supply meets demand'},
    },
    'objective': {'expression': 'sum(op_cost, over=snapshot)', 'description': 'operating cost'},
    'expressions': {
        'spend': {'expression': 'sum(op_cost, over=snapshot)', 'description': 'what the horizon costs'},
    },
    'macros': {
        'weighted': {'args': ['a', 'w'], 'template': 'sum(a * w, over=snapshot)', 'description': 'a weighted sum'},
    },
    'piecewise': {
        'cost_curve': {'over': 'bp', 'links': [['p', 'bp_x'], ['op_cost', 'bp_y']], 'description': 'the cost curve'},
    },
    'sos': {
        'pick': {'variable': 'p', 'over': 'snapshot', 'type': 1, 'description': 'at most one hour dispatches'},
    },
}

DECLARATION_KINDS = [
    pytest.param('dimensions', 'snapshot', id='dimension'),
    pytest.param('lookups', 'period', id='lookup'),
    pytest.param('parameters', 'load', id='parameter'),
    pytest.param('variables', 'p', id='variable'),
    pytest.param('constraints', 'balance', id='constraint'),
    pytest.param('expressions', 'spend', id='named-expression'),
    pytest.param('macros', 'weighted', id='macro'),
    pytest.param('piecewise', 'cost_curve', id='piecewise'),
    pytest.param('sos', 'pick', id='sos'),
]


@pytest.mark.parametrize(('section', 'name'), DECLARATION_KINDS)
def test_a_description_survives_loading(section, name):
    schema = Model.model_validate(DESCRIBED)
    block = getattr(schema, section)[name]
    assert block.description == DESCRIBED[section][name]['description'], (
        'the description is part of the Model, so it must reach AST consumers verbatim'
    )


def test_the_objective_carries_one_too():
    """`objective:` is one block rather than a mapping, so it misses the
    parametrization above and would otherwise go unchecked."""
    assert Model.model_validate(DESCRIBED).objective.description == 'operating cost'


def test_an_undescribed_declaration_carries_none():
    assert Model.model_validate(DESCRIBED).variables['op_cost'].description is None, (
        'absent means None, never an empty string'
    )


@pytest.mark.parametrize(
    ('raw', 'match'),
    [
        pytest.param(
            {
                'dimensions': {'x': {'values': [1], 'dtype': 'int'}},
                'variables': {'v': {'foreach': ['x'], 'bounds': {'lower': 0, 'description': 'floor'}}},
            },
            "unknown key 'description' in a bounds block",
            id='bounds',
        ),
        pytest.param(
            {
                **DESCRIBED,
                'piecewise': {
                    'cost_curve': {
                        'over': 'bp',
                        'links': [{'expression': 'p', 'values': 'bp_x', 'description': 'the x axis'}],
                    }
                },
            },
            "unknown key 'description' in a piecewise link",
            id='piecewise-link',
        ),
    ],
)
def test_a_description_on_a_non_declaration_block_is_rejected(raw, match):
    """The key belongs to declarations; the sub-blocks inside one stay closed."""
    with pytest.raises(SchemaError, match=match):
        Model.model_validate(raw)


def test_the_file_itself_is_describable():
    """The one description with no declaration under it: what the model *is*,
    which every example otherwise states in a `#` comment the parser throws
    away."""
    schema = Model.model_validate({'description': 'least-cost dispatch', **DESCRIBED})
    assert schema.description == 'least-cost dispatch'
    assert Model.model_validate(DESCRIBED).description is None, 'absent means None, never an empty string'


def test_a_model_description_survives_a_round_trip():
    schema = Model.model_validate({'description': 'least-cost dispatch', **DESCRIBED})
    assert Model.model_validate(schema.to_dict()).description == 'least-cost dispatch'
    assert 'description' not in Model.model_validate(DESCRIBED).to_dict(), 'None is stripped, as every other default is'


# ---------------------------------------------------------------------------
# a named expression is a string until it has more than one thing to say
# ---------------------------------------------------------------------------


EXPRESSIONS = {
    'dimensions': {'g': {'values': ['a'], 'dtype': 'str'}},
    'parameters': {'rate': {'dims': ['g']}},
    'variables': {'p': {'foreach': ['g']}},
}


def test_a_named_expression_is_written_as_a_bare_string():
    schema = Model.model_validate({**EXPRESSIONS, 'expressions': {'total': 'sum(p, over=g)'}})
    assert schema.expressions['total'].expression == 'sum(p, over=g)'
    assert schema.expressions['total'].description is None


def test_a_named_expression_carrying_a_description_is_written_as_a_mapping():
    body = {'expression': 'sum(p * rate, over=g)', 'description': 'CO2 released'}
    schema = Model.model_validate({**EXPRESSIONS, 'expressions': {'emissions': body}})
    assert schema.expressions['emissions'].description == 'CO2 released'


@pytest.mark.parametrize(
    ('written', 'id_'),
    [
        pytest.param('sum(p, over=g)', 'a-bare-string', id='a-bare-string'),
        pytest.param(
            {'expression': 'sum(p, over=g)', 'description': 'total output'},
            'a-mapping',
            id='a-mapping-with-a-description',
        ),
    ],
)
def test_a_named_expression_round_trips_in_the_form_it_was_written(written, id_):
    """A file that says it in one line gets one line back — the same trade
    `PiecewiseLink` makes for its list form."""
    schema = Model.model_validate({**EXPRESSIONS, 'expressions': {'e': written}})
    assert schema.to_dict()['expressions']['e'] == written, f'{id_} did not survive to_dict'
    assert Model.model_validate(schema.to_dict()).expressions['e'].expression == 'sum(p, over=g)'


def test_an_unknown_key_in_a_named_expression_is_rejected():
    """The mapping form is a block like any other, so it is closed too."""
    body = {'expression': 'sum(p, over=g)', 'describtion': 'typo'}
    with pytest.raises(SchemaError, match=r"unknown key 'describtion' in a named expression"):
        Model.model_validate({**EXPRESSIONS, 'expressions': {'e': body}})
