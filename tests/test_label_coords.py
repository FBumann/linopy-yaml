"""Label-space lookups: structure on a dimension, never an axis.

The declaration rules' split, pinned: everything under ``dimensions:`` is an axis, and a
label a dimension's members carry is a lookup — *groupable* when it targets a
dimension something aggregates into, *label-space* when it owns its values and
is only ever selected on. What these tests hold still: the schema tells the
kinds apart by which field is set, a lookup name joins the flat namespace,
grouping into a label space is refused with the promotion rewrite, ``check``
advises on a dimension nothing uses as an axis, and the bind-time contract
(the column arrives with the index, named after the lookup, single-valued per
label) covers both kinds alike.
"""

from __future__ import annotations

import re
import warnings

import polars as pl
import pytest

import lpspec as lps
from lpspec.errors import DataError, LpspecError, LpspecWarning
from lpspec.language.validation import load_model
from lpspec.typeset import FORMATS, typeset


def _model(objective: str = 'sum(x, over=snapshot)') -> dict:
    return {
        'dimensions': {
            'snapshot': {'dtype': 'int'},
        },
        'lookups': {'period': {'over': 'snapshot', 'dtype': 'int'}},
        'parameters': {'load': {'dims': ['snapshot']}},
        'variables': {'x': {'foreach': ['snapshot'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'c': {'foreach': ['snapshot'], 'expression': 'x >= load'}},
        'objective': {'sense': 'minimize', 'expression': objective},
    }


def _index() -> pl.DataFrame:
    return pl.DataFrame({'snapshot': [0, 1, 2], 'period': [1, 1, 2]})


def _load() -> pl.DataFrame:
    return pl.DataFrame({'snapshot': [0, 1, 2], 'value': [1.0, 2.0, 3.0]})


def test_the_two_lookup_kinds_parse_apart():
    """The field set decides the kind, and each surfaces only as itself."""
    schema = load_model(
        {
            'dimensions': {'bus': {}, 'generator': {}},
            'lookups': {
                'gen_bus': {'over': 'generator', 'into': 'bus'},
                'tech': {'over': 'generator', 'dtype': 'str'},
            },
        }
    )
    assert schema.targeted_of('generator') == {'gen_bus': 'bus'}
    labels = schema.labels_of('generator')
    assert list(labels) == ['tech']
    assert labels['tech'].dtype == 'str'


def test_a_label_space_lookup_puts_nothing_under_dimensions():
    """The file with one axis declares one dimension — the original complaint."""
    schema = load_model(_model())
    assert list(schema.dimensions) == ['snapshot']


def test_a_lookup_joins_the_flat_namespace():
    model = _model()
    model['parameters']['period'] = {'dims': ['snapshot']}
    with pytest.raises(LpspecError, match="Parameter 'period' collides with the lookup"):
        load_model(model)


def test_a_lookup_cannot_take_a_dimensions_name():
    model = _model()
    model['dimensions']['period'] = {'dtype': 'int'}
    with pytest.raises(LpspecError, match="Lookup 'period' collides with the dimension"):
        load_model(model)


def test_grouping_into_a_label_space_is_refused_with_the_promotion_rewrite():
    """The error teaches the promotion, not merely the refusal."""
    with pytest.raises(LpspecError, match="is a label space over 'snapshot'") as caught:
        lps.check(_model('sum(x, by=period)'))
    assert 'period_of: {over: snapshot, into: period}' in str(caught.value), (
        'the refusal has to show the axis-plus-lookup declaration that makes grouping sayable'
    )


def test_a_by_typo_is_offered_only_the_lookups_it_could_have_meant():
    """The suggestion lists groupable lookups, never a label space.

    One store holds both kinds, so which ones ``by=`` accepts is a filter
    rather than a separate dict — and a filter that slipped would offer
    ``period`` as the fix for a typo, the very thing the test above proves
    unsayable.
    """
    model = _model()
    model['dimensions']['bus'] = {'dtype': 'str'}
    model['lookups']['bus_of'] = {'over': 'snapshot', 'into': 'bus'}
    model['constraints']['c'] = {'foreach': ['bus'], 'expression': 'sum(x, by=bus_ov) >= load'}
    with pytest.raises(LpspecError, match=r'by=bus_ov\) does not name a lookup') as caught:
        lps.check(model)
    assert "Lookups: ['bus_of']" in str(caught.value), (
        "the listing offers only what by= accepts — 'period' is a label space and cannot be grouped into"
    )


def test_check_advises_a_label_space_wearing_a_dimensions_clothes():
    """A dim that only serves as a lookup target is advice, not an error."""
    model = _model()
    model['dimensions']['period'] = {'dtype': 'int'}
    model['lookups'] = {'period_of': {'over': 'snapshot', 'into': 'period'}}
    with pytest.warns(LpspecWarning, match='label space, not a dimension'):
        lps.check(model)


def test_check_advises_an_unused_dimension():
    model = _model()
    model['dimensions']['scenario'] = {'dtype': 'str'}
    with pytest.warns(LpspecWarning, match="'scenario' is never used"):
        lps.check(model)


def test_a_dimension_grouped_into_draws_no_advice():
    """`group_by=` lands terms on its target, so the target is an axis even
    when nothing is declared over it — an objective groups and implicitly
    sums, and no warning fires."""
    model = {
        'dimensions': {'bus': {}, 'generator': {}},
        'lookups': {'gen_bus': {'over': 'generator', 'into': 'bus'}},
        'parameters': {'cost': {'dims': ['generator']}},
        'variables': {'p': {'foreach': ['generator'], 'bounds': {'lower': 0, 'upper': 1}}},
        'constraints': {'c': {'foreach': ['generator'], 'expression': 'p <= 1'}},
        'objective': {
            'sense': 'minimize',
            'expression': 'sum(sum(p * cost, by=gen_bus), over=bus)',
        },
    }
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        lps.check(model)


def test_a_clean_model_checks_silently():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        lps.check(_model())


def test_the_lookup_column_arrives_with_the_index():
    with lps.solve(_model(), {'load': _load(), 'snapshot': _index()}) as solution:
        assert solution.objective == pytest.approx(6.0)

    missing = _index().drop('period')
    with pytest.raises(DataError, match='missing declared lookup column'):
        lps.build(_model(), {'load': _load(), 'snapshot': missing})


def test_a_lookup_is_single_valued_per_label():
    doubled = pl.DataFrame({'snapshot': [0, 0, 1, 2], 'period': [1, 2, 1, 2]})
    with pytest.raises(DataError, match='more than one value per label'):
        lps.build(_model(), {'load': _load(), 'snapshot': doubled})


def _unused_target_model(month: dict) -> dict:
    """#488's incremental multi-period shape.

    The flat ``snapshot`` index declares every lookup it will need, but no
    constraint groups into ``month`` yet — only ``period`` is used.
    """
    return {
        'dimensions': {
            'snapshot': {'dtype': 'int'},
            'period': {'dtype': 'int'},
            'month': month,
        },
        'lookups': {
            'period_of': {'over': 'snapshot', 'into': 'period'},
            'month_of': {'over': 'snapshot', 'into': 'month'},
        },
        'parameters': {'cap': {'dims': ['period']}},
        'variables': {'p': {'foreach': ['snapshot'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'budget': {'foreach': ['period'], 'expression': 'sum(p, by=period_of) <= cap'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(p, over=snapshot)'},
    }


def _unused_target_sources() -> dict:
    return {
        'snapshot': pl.DataFrame(
            {'snapshot': [0, 1, 2], 'period_of': [2030, 2030, 2050], 'month_of': ['jan', 'feb', 'jan']}
        ),
        'period': pl.DataFrame({'period': [2030, 2050]}),
        'cap': pl.DataFrame({'period': [2030, 2050], 'value': [5.0, 5.0]}),
    }


@pytest.mark.parametrize(
    ('month', 'extra'),
    [
        pytest.param({'dtype': 'str'}, {'month': pl.DataFrame({'month': ['jan', 'feb']})}, id='index-in-sources'),
        pytest.param({'values': ['jan', 'feb']}, {}, id='values-on-the-declaration'),
    ],
)
def test_a_lookup_may_target_a_dimension_nothing_spans_yet(month, extra):
    """#488: the first build after declaring a lookup, before its constraint exists."""
    with lps.solve(_unused_target_model(month), _unused_target_sources() | extra) as solution:
        assert solution.objective == pytest.approx(10.0), 'each period caps its snapshots at 5, so the model builds'


def test_an_unused_target_still_checks_containment():
    short = {'month': pl.DataFrame({'month': ['jan']})}
    with pytest.raises(DataError, match="not 'month' labels"):
        lps.build(_unused_target_model({'dtype': 'str'}), _unused_target_sources() | short)


def test_an_unused_target_without_an_index_is_refused_with_the_true_reason():
    """The old message blamed missing data the caller may well have supplied (#488)."""
    with pytest.raises(DataError, match='no index of its own') as caught:
        lps.build(_unused_target_model({'dtype': 'str'}), _unused_target_sources())
    assert "Pass an index for 'month'" in str(caught.value), 'the refusal has to say what would satisfy it'


def test_the_legend_names_a_label():
    text = typeset(_model(), FORMATS['markdown'])
    assert 'carrying label' in text
    assert 'period' in text


def test_both_lanes_read_the_same_index():
    """The `period` lookup column arrives with the index on the eager lane too —
    both lanes reach the 6.0 the relational test above asserts.

    The oracle is imported in the body rather than at module scope: every other
    test here is linopy-free and has to keep running on the bare install, so
    this one test skips there instead of failing on a missing pandas.
    """
    from tests.differential import differential
    from tests.oracle import pd

    data = {'load': pd.Series({0: 1.0, 1: 2.0, 2: 3.0}).rename_axis('snapshot')}
    index = {'snapshot': pd.DataFrame({'snapshot': [0, 1, 2], 'period': [1, 1, 2]})}
    with differential(_model(), data | index) as run:
        assert run.oracle == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# where on a lookup (#553): the consumer the label-space kind was missing
# ---------------------------------------------------------------------------


#: A three-line network whose structure is entirely lookups: each line has two
#: endpoints, and `spur` deliberately has an open end so the partial case is
#: reachable. Both kinds appear — `voltage` owns its values, `send`/`recv`
#: target `bus` — so one model covers every lookup predicate.
NETWORK = {
    'dimensions': {'bus': {'dtype': 'str'}, 'line': {'dtype': 'str'}},
    'lookups': {
        'send': {'over': 'line', 'into': 'bus'},
        'recv': {'over': 'line', 'into': 'bus'},
        'voltage': {'over': 'line', 'dtype': 'int'},
    },
    'parameters': {'cap': {'dims': ['line']}, 'price': {'dims': ['line']}},
    'variables': {'f': {'foreach': ['line'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {'ceiling': {'foreach': ['line'], 'expression': 'f <= cap'}},
    'objective': {'sense': 'maximize', 'expression': 'f * price'},
}

LINES = ['ring_a', 'ring_b', 'loop', 'spur']

#: `loop` starts and ends on the same bus; `spur` has no receiving end at all.
NETWORK_SOURCES = {
    'bus': pl.DataFrame({'bus': ['north', 'south']}),
    'line': pl.DataFrame(
        {
            'line': LINES,
            'send': ['north', 'south', 'north', 'north'],
            'recv': ['south', 'north', 'north', None],
            'voltage': [220, 380, 220, 380],
        }
    ),
    'cap': pl.DataFrame({'line': LINES, 'value': [10.0, 20.0, 30.0, 40.0]}),
    'price': pl.DataFrame({'line': LINES, 'value': [1.0, 1.0, 1.0, 1.0]}),
}


@pytest.mark.parametrize(
    ('where', 'kept'),
    [
        pytest.param('voltage == 220', ['loop', 'ring_a'], id='a-label-space-lookup-against-a-literal'),
        pytest.param("send == 'north'", ['loop', 'ring_a', 'spur'], id='a-targeted-lookup-against-a-label'),
        pytest.param('send != recv', ['ring_a', 'ring_b'], id='two-lookups-over-one-dimension'),
        pytest.param('recv', ['loop', 'ring_a', 'ring_b'], id='a-bare-lookup-is-the-partial-case'),
        pytest.param('NOT voltage == 220', ['ring_b', 'spur'], id='negated'),
        pytest.param('voltage == 380 AND send != recv', ['ring_b'], id='conjoined-with-a-pair-comparison'),
    ],
)
def test_a_where_reads_a_lookup(where, kept):
    """The atom #553 asked for, on both kinds and in both shapes.

    `kept` is asserted rather than just a count: a predicate that inverted its
    sense would keep the complement, which is the same size on a symmetric
    case and a different model everywhere.

    `spur`'s null `recv` is the reading law 8 fixes — a comparison over it is
    false, so the bare name keeps exactly the lines that map.
    """
    model = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': where}}}
    with lps.solve(model, NETWORK_SOURCES) as result:
        built = sorted(row['line'] for row in result.primal('f').to_dicts())
    assert built == sorted(kept), f'where: {where!r} built the wrong set of variables'


@pytest.mark.parametrize(
    ('where', 'objective'),
    [
        pytest.param('send != recv', 30.0, id='the-two-ring-lines-survive'),
        pytest.param('NOT send != recv', 70.0, id='negated-over-a-partial-lookup'),
        # The two probes for the eager lane's explicit null exclusion. Only a
        # `!=` reaches it: numpy answers `None != 'north'` with True, so
        # without it the eager lane keeps exactly `spur` — the line that maps
        # nowhere — where the relational lane drops it.
        pytest.param("recv != 'north'", 10.0, id='not-equal-over-a-null-value'),
        pytest.param('recv != send', 30.0, id='not-equal-between-two-lookups'),
    ],
)
def test_a_lookup_where_agrees_with_the_oracle(where, objective):
    """Both lanes, one answer — the differential half of #553.

    A mask reading a lookup is a join on the dim table in the relational lane
    and an array read in the eager one; nothing but this shows they agree on
    which rows survive, since a wrong mask still solves.
    """
    from tests.differential import differential
    from tests.oracle import pd

    model = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': where}}}
    data = {
        'cap': pd.Series([10.0, 20.0, 30.0, 40.0], index=LINES),
        'price': pd.Series([1.0, 1.0, 1.0, 1.0], index=LINES),
    }
    index = {
        'bus': pd.Index(['north', 'south'], name='bus'),
        'line': pd.DataFrame(
            {
                'line': LINES,
                'send': ['north', 'south', 'north', 'north'],
                'recv': ['south', 'north', 'north', None],
                'voltage': [220, 380, 220, 380],
            }
        ),
    }
    with differential(model, data | index) as run:
        assert run.result.objective == pytest.approx(objective), (
            f'where: {where!r} — the two lanes agree on the objective but not on this one'
        )


def test_a_where_on_a_lookup_outside_the_frame_is_refused():
    """A lookup is read on the dim it maps out of, so that dim has to be in the
    frame — otherwise the mask would silently reduce over an unlisted dim."""
    model = {
        **NETWORK,
        'variables': {'f': {'foreach': ['line'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'constraints': {
            'ceiling': {'foreach': ['bus'], 'where': 'voltage == 220', 'expression': 'sum(f, by=send) <= 100'}
        },
    }
    with pytest.raises(LpspecError, match=r"lookup 'voltage', which is over dimension 'line'"):
        load_model(model)


def test_two_lookups_over_different_dims_cannot_be_compared():
    """There is no row carrying both, so the comparison has nothing to test.

    The pair comparison is legal exactly because two lookups over one dim are
    two columns of one index. Drop that condition and this reads as a join
    nobody wrote.
    """
    model = {
        **NETWORK,
        'lookups': {**NETWORK['lookups'], 'zone': {'over': 'bus', 'dtype': 'str'}},
        'variables': {'f': {**NETWORK['variables']['f'], 'where': 'send != zone'}},
    }
    with pytest.raises(LpspecError, match='over different dimensions'):
        load_model(model)


@pytest.mark.parametrize(
    ('extra', 'where'),
    [
        pytest.param(
            {'area': {'over': 'line', 'into': 'zone'}}, 'send != area', id='two-targets-that-are-different-dimensions'
        ),
        pytest.param({}, 'send != voltage', id='a-label-space-against-a-targeted-lookup'),
        pytest.param(
            {'grid': {'over': 'line', 'dtype': 'int'}}, 'voltage != grid', id='two-label-spaces-of-the-same-dtype'
        ),
    ],
)
def test_two_lookups_into_different_label_sets_cannot_be_compared(extra, where):
    """One dimension is necessary but not sufficient — the label sets must match too.

    A bus label is never a zone label and a label space owns its values, so
    the predicate could only mask everything out. It does not even do that
    consistently: the eager lane answers `!=` True at every row while polars
    refuses the Enum mismatch, so both lanes accepted the model and then
    disagreed about it.
    """
    model = {
        **NETWORK,
        'dimensions': {**NETWORK['dimensions'], 'zone': {'dtype': 'str'}},
        'lookups': {**NETWORK['lookups'], **extra},
        'variables': {'f': {**NETWORK['variables']['f'], 'where': where}},
    }
    with pytest.raises(LpspecError, match='map into the same dimension'):
        load_model(model)


def test_a_lookup_comparison_is_checked_against_its_dtype():
    """The same dtype check every other where-comparison gets (#460).

    A lookup's literal is as silent to get wrong as a dimension's: `voltage`
    is an int, so a quoted right-hand side matches nothing rather than
    erroring at run time.
    """
    model = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': "voltage == 'high'"}}}
    with pytest.raises(LpspecError, match=r"has dtype 'int'"):
        load_model(model)


def test_a_targeted_lookup_compares_against_a_label_the_target_lacks():
    """A stranger label masks everything out; it does not raise.

    The where-string rules' reading for every other comparison, and the reason
    the lookup
    column is compared as a string: binding casts it to the target's `Enum`,
    which orders by declaration and *refuses* a label outside it — so without
    the cast back this is a polars error rather than an empty mask.
    """
    model = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': "send == 'atlantis'"}}}
    with lps.build(model, NETWORK_SOURCES) as bound:
        surviving = bound._engine._variables['f'].select(pl.len()).collect().item()
    assert surviving == 0, "a label no bus carries matches nothing, so no 'f' is built"


def test_a_targeted_lookup_orders_bytewise_not_by_declaration():
    """Labels order bytewise, whatever order the dimension declared them.

    Binding casts a lookup column to the target's `Enum`, which orders by
    *declaration*, so an ordering comparison read off it would answer a
    different question — and silently, since both readings return a mask.
    `south` is declared first here precisely so the two disagree.
    """
    sources = {**NETWORK_SOURCES, 'bus': pl.DataFrame({'bus': ['south', 'north']})}
    model = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': "send >= 'south'"}}}
    with lps.solve(model, sources) as result:
        built = sorted(row['line'] for row in result.primal('f').to_dicts())
    assert built == ['ring_b'], "only ring_b sends from 'south'; declaration order would keep the 'north' lines too"


# ---------------------------------------------------------------------------
# a lookup's map, declared in the file
# ---------------------------------------------------------------------------


#: The whole model — no index table, no parameter carrying structure. What a
#: dimension's own `values:` does for labels, `values:` on a lookup does for
#: the map, so a relation small enough to read lives beside the equation that
#: traverses it. `g3` maps to no bus: the partial case, declared by omission.
DECLARED = {
    'dimensions': {
        'generator': {'values': ['g1', 'g2', 'g3']},
        'bus': {'values': ['north', 'south']},
    },
    'lookups': {'gen_bus': {'over': 'generator', 'into': 'bus', 'values': {'g1': 'north', 'g2': 'south'}}},
    'parameters': {'cost': {'dims': ['generator']}, 'load': {'dims': ['bus']}},
    'variables': {'p': {'foreach': ['generator'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'balance': {'foreach': ['bus'], 'expression': 'sum(p, by=gen_bus) >= load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost, over=generator)'},
}

DECLARED_SOURCES = {
    'cost': pl.DataFrame({'generator': ['g1', 'g2', 'g3'], 'value': [1.0, 2.0, 0.5]}),
    'load': pl.DataFrame({'bus': ['north', 'south'], 'value': [5.0, 4.0]}),
}


def test_a_declared_map_needs_no_index_source():
    """The point: a model whose structure is in the file binds without one.

    `g3` is the cheapest generator and contributes nothing, which is what a
    null lookup means — so this also pins that an omitted key is the partial
    case rather than an error or a default.
    """
    with lps.solve(DECLARED, DECLARED_SOURCES) as result:
        built = {row['generator']: row['value'] for row in result.primal('p').to_dicts()}
        assert result.objective == pytest.approx(13.0)
    assert built['g3'] == pytest.approx(0.0), 'a generator on no bus can serve no load, however cheap'


def test_a_declared_map_agrees_with_the_oracle():
    """Both lanes assemble the same index out of the declaration."""
    from tests.differential import differential
    from tests.oracle import pd

    data = {
        'cost': pd.Series([1.0, 2.0, 0.5], index=['g1', 'g2', 'g3']),
        'load': pd.Series([5.0, 4.0], index=['north', 'south']),
    }
    with differential(DECLARED, data) as run:
        assert run.result.objective == pytest.approx(13.0)


#: A dimension bare but for its declared map, so the caller owns its labels and
#: the file owns the relation over them.
MAP_ONLY = {**DECLARED, 'dimensions': {'generator': {}, 'bus': {'values': ['north', 'south']}}}

_LABELS = pl.DataFrame({'generator': ['g1', 'g2', 'g3']})
_LABELS_AND_MAP = pl.DataFrame({'generator': ['g1', 'g2', 'g3'], 'gen_bus': ['south', 'north', None]})


@pytest.mark.parametrize(
    ('model', 'sources', 'names'),
    [
        pytest.param(DECLARED, {'generator': _LABELS}, 'dimensions.generator.values', id='labels-twice'),
        pytest.param(DECLARED, {'generator': ['g1', 'g2', 'g3']}, 'dimensions.generator.values', id='bare-labels'),
        pytest.param(MAP_ONLY, {'generator': _LABELS_AND_MAP}, 'lookups.gen_bus.values', id='the-map-twice'),
    ],
)
def test_a_declared_index_refuses_a_supplied_one(model, sources, names):
    """One fact, one home — for the labels, and for each map over them.

    A precedence rule instead lets the file describe a model the caller does
    not build: the YAML says ``g1`` sits on north, the passed column says south,
    and the file a reviewer reads is not the model that solved.
    """
    with pytest.raises(DataError, match=re.escape(names)):
        lps.solve(model, {**DECLARED_SOURCES, **sources})


def test_a_map_alone_does_not_say_which_labels_exist():
    """A map is a relation over a dimension, never the dimension.

    It may omit members and its key order is whatever someone typed, so reading
    the label set out of it would let an added entry create a member and a
    reordered map re-order the axis that ``shift`` reads positionally. With
    ``generator`` declaring no ``values:`` and nothing supplying them, the map
    over it leaves the dimension without an index, and both lanes say so.
    """
    with pytest.raises(DataError, match='has its maps in the file'):
        lps.solve(MAP_ONLY, DECLARED_SOURCES)


@pytest.mark.parametrize('lane', ['relational', 'eager'])
def test_a_declared_map_keyed_off_the_callers_labels_is_refused(lane):
    """A key matching no label is a typo, and the two sides only meet at bind.

    Where the file declares the labels too, this is decided at load with no data
    at all. Here they are the caller's, so the same law lands later — and land
    it must, because the join that reads the map would otherwise drop the key
    and build a model that solves while placing those terms nowhere.

    Deliberately not symmetric with the test below: a label no map mentions is a
    null, a key no label matches is an error.
    """
    from tests.oracle import lpspec_linopy

    model = {
        **MAP_ONLY,
        'lookups': {'gen_bus': {'over': 'generator', 'into': 'bus', 'values': {'g1': 'north', 'g9': 'south'}}},
    }
    build = lps.solve if lane == 'relational' else lpspec_linopy.build
    with pytest.raises(DataError, match=r"lookup 'gen_bus' declares values for g9"):
        build(model, {**DECLARED_SOURCES, 'generator': _LABELS})


def test_a_declared_map_is_read_against_the_labels_the_caller_brings():
    """The two facts have different authors, which is the shape the split buys.

    The caller says which generators exist and in what order; the file says how
    they sit on buses. ``g3`` is in nobody's map, so it is a generator with a
    null lookup rather than a generator that does not exist — the partial case,
    reachable here with a single map. And the labels arrive unsorted, so a
    result in the caller's order is what proves the map did not impose its own.
    """
    unsorted_labels = pl.DataFrame({'generator': ['g3', 'g1', 'g2']})
    with lps.solve(MAP_ONLY, {**DECLARED_SOURCES, 'generator': unsorted_labels}) as result:
        assert result.objective == pytest.approx(13.0), 'g1 serves north and g2 south, exactly as the file maps them'
        built = [row['generator'] for row in result.primal('p').to_dicts()]
    assert built == ['g3', 'g1', 'g2'], "the label order is the caller's, not the order the map was typed in"


def test_a_declared_index_is_refused_the_same_way_on_the_eager_lane():
    """The refusal is the binding rule, not one lane's reading of it."""
    from tests.oracle import lpspec_linopy

    with pytest.raises(DataError, match=re.escape('dimensions.generator.values')):
        lpspec_linopy.build(DECLARED, {**DECLARED_SOURCES, 'generator': _LABELS})


@pytest.mark.parametrize(
    ('values', 'match'),
    [
        pytest.param({'g1': 'atlantis'}, 'not labels of', id='a-value-the-target-does-not-carry'),
        pytest.param({'g9': 'north'}, 'not labels of', id='a-key-the-dimension-does-not-carry'),
    ],
)
def test_a_declared_map_is_checked_without_data(values, match):
    """Law 2: both sides are in the file, so this is decided at load.

    The same mistake in an index source is a bind-time error, which is later
    and needs the data to hand — declaring the map is what moves it earlier.
    """
    model = {**DECLARED, 'lookups': {'gen_bus': {'over': 'generator', 'into': 'bus', 'values': values}}}
    with pytest.raises(LpspecError, match=match):
        load_model(model)


@pytest.mark.parametrize(
    ('dimensions', 'lookup', 'match'),
    [
        pytest.param(
            {'generator': {'dtype': 'str'}, 'bus': {'values': ['north']}},
            {'over': 'generator', 'into': 'bus', 'values': {12: 'north'}},
            r"key 12 has type int, but dtype is 'str'",
            id='a-key-that-is-not-the-dimensions-dtype',
        ),
        pytest.param(
            {'generator': {'values': ['g1']}, 'bus': {'dtype': 'str'}},
            {'over': 'generator', 'into': 'bus', 'values': {'g1': 12}},
            r"value 12 has type int, but dtype is 'str'",
            id='a-value-that-is-not-the-targets-dtype',
        ),
        pytest.param(
            {'generator': {'values': ['g1']}, 'bus': {'values': ['north']}},
            {'over': 'generator', 'dtype': 'int', 'values': {'g1': 'north'}},
            r"value 'north' has type str, but dtype is 'int'",
            id='a-label-space-value-that-is-not-its-own-dtype',
        ),
    ],
)
def test_a_declared_map_is_checked_against_its_dtypes(dimensions, lookup, match):
    """The guard a dimension's own `values:` has always had, both sides of the map.

    Containment covers the mistyped label only where the *other* side declares
    its labels too, so each case here is one whose other side comes from data
    — and the label space, which targets nothing and so has no containment
    check at all. Left unchecked the last one is a lane disagreement rather
    than a wrong answer: numpy compares the object array and answers, polars
    refuses the dtype outright, on a model both accepted.
    """
    model = {
        **DECLARED,
        'dimensions': dimensions,
        'lookups': {'gen_bus' if lookup.get('into') else 'tech': lookup},
        'constraints': {'balance': {'foreach': ['generator'], 'expression': 'p >= 1'}},
    }
    with pytest.raises(LpspecError, match=match):
        load_model(model)


def test_a_declared_map_may_leave_a_label_unmapped():
    """A null is the partial case, not a mistyped label — the dtype check skips it."""
    model = {**DECLARED, 'lookups': {'gen_bus': {'over': 'generator', 'into': 'bus', 'values': {'g1': None}}}}
    assert load_model(model).lookups['gen_bus'].values == {'g1': None}, 'an explicit null maps nowhere and is legal'


def test_a_label_space_lookup_may_declare_its_values_too():
    """Both kinds take the map, and a `where` reads what it declares."""
    model = {
        'dimensions': {'snapshot': {'dtype': 'int', 'values': [0, 1, 2]}},
        'lookups': {'period': {'over': 'snapshot', 'dtype': 'int', 'values': {0: 1, 1: 1, 2: 2}}},
        'parameters': {'load': {'dims': ['snapshot']}},
        'variables': {'x': {'foreach': ['snapshot'], 'where': 'period == 1', 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'c': {'foreach': ['snapshot'], 'where': 'period == 1', 'expression': 'x >= load'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(x, over=snapshot)'},
    }
    with lps.solve(model, {'load': _load()}) as result:
        built = sorted(row['snapshot'] for row in result.primal('x').to_dicts())
    assert built == [0, 1], 'only period 1 is built, and the map that says so is in the file'
