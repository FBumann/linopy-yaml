"""Label-space lookups: structure on a dimension, never an axis.

The declaration rules' split, pinned: everything under ``dimensions:`` is an axis, and a
label a dimension's members carry is a lookup — *groupable* when it targets a
dimension something aggregates into, *label-space* when it owns its values and
is only ever selected on. What these tests hold still: the schema tells the
kinds apart by which field is set, a lookup name joins the flat namespace,
grouping into a label space is refused with the promotion rewrite, ``check``
advises on a dimension nothing uses as an axis, and the attach-time contract
(the column arrives with the index, named after the lookup, single-valued per
label) covers both kinds alike.
"""

from __future__ import annotations

import re
import warnings

import polars as pl
import pytest
from math_spec import to_spec

import lpspec as lps
from lpspec.errors import DataError, LpspecError, LpspecWarning
from tests.conftest import by_coord


def _spec(objective: str = 'sum(x, over=snapshot)') -> dict:
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
    return pl.DataFrame({'snapshot': [0, 1, 2]})


def _period() -> pl.DataFrame:
    """`period` as a relation, which is the only way a map arrives as data."""
    return pl.DataFrame({'snapshot': [0, 1, 2], 'period': [1, 1, 2]})


def _load() -> pl.DataFrame:
    return pl.DataFrame({'snapshot': [0, 1, 2], 'value': [1.0, 2.0, 3.0]})


def test_the_two_lookup_kinds_parse_apart():
    """The field set decides the kind, and each surfaces only as itself."""
    schema = to_spec(
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
    schema = to_spec(_spec())
    assert list(schema.dimensions) == ['snapshot']


def test_a_lookup_joins_the_flat_namespace():
    spec = _spec()
    spec['parameters']['period'] = {'dims': ['snapshot']}
    with pytest.raises(LpspecError, match="Parameter 'period' collides with the lookup"):
        to_spec(spec)


def test_a_lookup_cannot_take_a_dimensions_name():
    spec = _spec()
    spec['dimensions']['period'] = {'dtype': 'int'}
    with pytest.raises(LpspecError, match="Lookup 'period' collides with the dimension"):
        to_spec(spec)


def test_grouping_into_a_label_space_is_refused_with_the_promotion_rewrite():
    """The error teaches the promotion, not merely the refusal."""
    with pytest.raises(LpspecError, match="is a label space over 'snapshot'") as caught:
        lps.check(_spec('sum(x, by=period)'))
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
    spec = _spec()
    spec['dimensions']['bus'] = {'dtype': 'str'}
    spec['lookups']['bus_of'] = {'over': 'snapshot', 'into': 'bus'}
    spec['constraints']['c'] = {'foreach': ['bus'], 'expression': 'sum(x, by=bus_ov) >= load'}
    with pytest.raises(LpspecError, match=r'by=bus_ov\) does not name a lookup') as caught:
        lps.check(spec)
    assert "Lookups: ['bus_of']" in str(caught.value), (
        "the listing offers only what by= accepts — 'period' is a label space and cannot be grouped into"
    )


def test_check_advises_a_label_space_wearing_a_dimensions_clothes():
    """A dim that only serves as a lookup target is advice, not an error."""
    spec = _spec()
    spec['dimensions']['period'] = {'dtype': 'int'}
    spec['lookups'] = {'period_of': {'over': 'snapshot', 'into': 'period'}}
    with pytest.warns(LpspecWarning, match='label space, not a dimension'):
        lps.check(spec)


def test_check_advises_an_unused_dimension():
    spec = _spec()
    spec['dimensions']['scenario'] = {'dtype': 'str'}
    with pytest.warns(LpspecWarning, match="'scenario' is never used"):
        lps.check(spec)


def test_a_dimension_grouped_into_draws_no_advice():
    """`group_by=` lands terms on its target, so the target is an axis even
    when nothing is declared over it — an objective groups and implicitly
    sums, and no warning fires."""
    spec = {
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
        lps.check(spec)


def test_a_clean_model_checks_silently():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        lps.check(_spec())


def test_a_map_arrives_under_its_own_name():
    with lps.solve(_spec(), {'load': _load(), 'snapshot': _index(), 'period': _period()}) as solution:
        assert solution.objective == pytest.approx(6.0)

    with pytest.raises(DataError, match="no data provided for lookup 'period'"):
        lps.build(_spec(), {'load': _load(), 'snapshot': _index()})


def test_a_lookup_is_single_valued_per_label():
    doubled = pl.DataFrame({'snapshot': [0, 0, 1, 2], 'period': [1, 2, 1, 2]})
    with pytest.raises(DataError, match='more than once'):
        lps.build(_spec(), {'load': _load(), 'snapshot': _index(), 'period': doubled})


def _unused_target_spec(month: dict) -> dict:
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
        'snapshot': pl.DataFrame({'snapshot': [0, 1, 2]}),
        'period_of': pl.DataFrame({'snapshot': [0, 1, 2], 'period': [2030, 2030, 2050]}),
        'month_of': pl.DataFrame({'snapshot': [0, 1, 2], 'month': ['jan', 'feb', 'jan']}),
        'period': pl.DataFrame({'period': [2030, 2050]}),
        'cap': pl.DataFrame({'period': [2030, 2050], 'value': [5.0, 5.0]}),
    }


@pytest.mark.parametrize(
    ('month', 'extra'),
    [
        pytest.param({'dtype': 'str'}, {'month': pl.DataFrame({'month': ['jan', 'feb']})}, id='a-table'),
        pytest.param({'dtype': 'str'}, {'month': ['jan', 'feb']}, id='a-bare-sequence'),
    ],
)
def test_a_lookup_may_target_a_dimension_nothing_spans_yet(month, extra):
    """#488: the first build after declaring a lookup, before its constraint exists."""
    with lps.solve(_unused_target_spec(month), _unused_target_sources() | extra) as solution:
        assert solution.objective == pytest.approx(10.0), 'each period caps its snapshots at 5, so the model builds'


@pytest.mark.parametrize('lane', ['relational', 'eager'])
def test_an_unused_target_still_checks_containment(lane):
    """A map into a dimension no constraint groups by is checked all the same.

    On both lanes, because the check now runs where the map is read rather than
    where each engine holds one — the eager lane never spans `month` either,
    and used to reach this only through its own copy.
    """
    from tests.oracle import lpspec_linopy

    build = lps.build if lane == 'relational' else lpspec_linopy.build
    short = {'month': pl.DataFrame({'month': ['jan']})}
    with pytest.raises(DataError, match="not 'month' labels"):
        build(_unused_target_spec({'dtype': 'str'}), _unused_target_sources() | short)


@pytest.mark.parametrize('lane', ['relational', 'eager'])
def test_an_unused_target_without_an_index_is_refused_with_the_true_reason(lane):
    """The old message blamed missing data the caller may well have supplied (#488)."""
    from tests.oracle import lpspec_linopy

    build = lps.build if lane == 'relational' else lpspec_linopy.build
    with pytest.raises(DataError, match='no index of its own') as caught:
        build(_unused_target_spec({'dtype': 'str'}), _unused_target_sources())
    assert "Pass an index for 'month'" in str(caught.value), 'the refusal has to say what would satisfy it'


def test_both_lanes_read_the_same_index():
    """The `period` relation is read the same way on the eager lane too —
    both lanes reach the 6.0 the relational test above asserts.

    The oracle is imported in the body rather than at module scope: every other
    test here is linopy-free and has to keep running on the bare install, so
    this one test skips there instead of failing on a missing pandas.
    """
    from tests.differential import differential
    from tests.oracle import pd

    data = {'load': pd.Series({0: 1.0, 1: 2.0, 2: 3.0}).rename_axis('snapshot')}
    index = {
        'snapshot': pd.DataFrame({'snapshot': [0, 1, 2]}),
        'period': pd.DataFrame({'snapshot': [0, 1, 2], 'period': [1, 1, 2]}),
    }
    with differential(_spec(), data | index) as run:
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
    'objective': {'sense': 'maximize', 'expression': 'sum(f * price)'},
}

LINES = ['ring_a', 'ring_b', 'loop', 'spur']
SEND = ['north', 'south', 'north', 'north']
RECV = ['south', 'north', 'north']
VOLTAGE = [220, 380, 220, 380]
CAP = [10.0, 20.0, 30.0, 40.0]
PRICE = [1.0, 1.0, 1.0, 1.0]

#: `loop` starts and ends on the same bus; `spur` has no receiving end at all,
#: which `recv` says by having no row for it.
NETWORK_SOURCES = {
    'bus': pl.DataFrame({'bus': ['north', 'south']}),
    'line': pl.DataFrame({'line': LINES}),
    'send': pl.DataFrame({'line': LINES, 'bus': SEND}),
    'recv': pl.DataFrame({'line': LINES[:3], 'bus': RECV}),
    'voltage': pl.DataFrame({'line': LINES, 'voltage': VOLTAGE}),
    'cap': pl.DataFrame({'line': LINES, 'value': CAP}),
    'price': pl.DataFrame({'line': LINES, 'value': PRICE}),
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
    spec = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': where}}}
    with lps.solve(spec, NETWORK_SOURCES) as result:
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

    spec = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': where}}}
    data = {
        'cap': pd.Series(CAP, index=LINES),
        'price': pd.Series(PRICE, index=LINES),
    }
    index = {
        'bus': pd.Index(['north', 'south'], name='bus'),
        'line': pd.DataFrame({'line': LINES}),
        'send': pd.DataFrame({'line': LINES, 'bus': SEND}),
        'recv': pd.DataFrame({'line': LINES[:3], 'bus': RECV}),
        'voltage': pd.DataFrame({'line': LINES, 'voltage': VOLTAGE}),
    }
    with differential(spec, data | index) as run:
        assert run.result.objective == pytest.approx(objective), (
            f'where: {where!r} — the two lanes agree on the objective but not on this one'
        )


def test_a_where_on_a_lookup_outside_the_frame_is_refused():
    """A lookup is read on the dim it maps out of, so that dim has to be in the
    frame — otherwise the mask would silently reduce over an unlisted dim."""
    spec = {
        **NETWORK,
        'variables': {'f': {'foreach': ['line'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'constraints': {
            'ceiling': {'foreach': ['bus'], 'where': 'voltage == 220', 'expression': 'sum(f, by=send) <= 100'}
        },
    }
    with pytest.raises(LpspecError, match=r"lookup 'voltage', which is over dimension 'line'"):
        to_spec(spec)


def test_two_lookups_over_different_dims_cannot_be_compared():
    """There is no row carrying both, so the comparison has nothing to test.

    The pair comparison is legal exactly because two lookups over one dim are
    two columns of one index. Drop that condition and this reads as a join
    nobody wrote.
    """
    spec = {
        **NETWORK,
        'lookups': {**NETWORK['lookups'], 'zone': {'over': 'bus', 'dtype': 'str'}},
        'variables': {'f': {**NETWORK['variables']['f'], 'where': 'send != zone'}},
    }
    with pytest.raises(LpspecError, match='over different dimensions'):
        to_spec(spec)


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
    spec = {
        **NETWORK,
        'dimensions': {**NETWORK['dimensions'], 'zone': {'dtype': 'str'}},
        'lookups': {**NETWORK['lookups'], **extra},
        'variables': {'f': {**NETWORK['variables']['f'], 'where': where}},
    }
    with pytest.raises(LpspecError, match='map into the same dimension'):
        to_spec(spec)


def test_a_lookup_comparison_is_checked_against_its_dtype():
    """The same dtype check every other where-comparison gets (#460).

    A lookup's literal is as silent to get wrong as a dimension's: `voltage`
    is an int, so a quoted right-hand side matches nothing rather than
    erroring at run time.
    """
    spec = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': "voltage == 'high'"}}}
    with pytest.raises(LpspecError, match=r"has dtype 'int'"):
        to_spec(spec)


def test_a_targeted_lookup_compares_against_a_label_the_target_lacks():
    """A stranger label masks everything out; it does not raise.

    The where-string rules' reading for every other comparison, and the reason
    the lookup
    column is compared as a string: attaching casts it to the target's `Enum`,
    which orders by declaration and *refuses* a label outside it — so without
    the cast back this is a polars error rather than an empty mask.
    """
    spec = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': "send == 'atlantis'"}}}
    with lps.build(spec, NETWORK_SOURCES) as model:
        surviving = model._engine._model.variables['f'].frame.select(pl.len()).collect().item()
    assert surviving == 0, "a label no bus carries matches nothing, so no 'f' is built"


def test_a_targeted_lookup_orders_bytewise_not_by_declaration():
    """Labels order bytewise, whatever order the dimension declared them.

    Binding casts a lookup column to the target's `Enum`, which orders by
    *declaration*, so an ordering comparison read off it would answer a
    different question — and silently, since both readings return a mask.
    `south` is declared first here precisely so the two disagree.
    """
    sources = {**NETWORK_SOURCES, 'bus': pl.DataFrame({'bus': ['south', 'north']})}
    spec = {**NETWORK, 'variables': {'f': {**NETWORK['variables']['f'], 'where': "send >= 'south'"}}}
    with lps.solve(spec, sources) as result:
        built = sorted(row['line'] for row in result.primal('f').to_dicts())
    assert built == ['ring_b'], "only ring_b sends from 'south'; declaration order would keep the 'north' lines too"


# ---------------------------------------------------------------------------
# a lookup's map, supplied under its own key
# ---------------------------------------------------------------------------


#: The model the section is written over: one lookup, and the caller holding
#: both its labels and its relation. `g3` maps to no bus — the partial case,
#: spelled by the row that is not there.
GENERATORS = ['g1', 'g2', 'g3']
COST = [1.0, 2.0, 0.5]
LOAD = [5.0, 4.0]

BASE = {
    'dimensions': {'generator': {'dtype': 'str'}, 'bus': {'dtype': 'str'}},
    'lookups': {'gen_bus': {'over': 'generator', 'into': 'bus'}},
    'parameters': {'cost': {'dims': ['generator']}, 'load': {'dims': ['bus']}},
    'variables': {'p': {'foreach': ['generator'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'balance': {'foreach': ['bus'], 'expression': 'sum(p, by=gen_bus) >= load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
}

BASE_SOURCES = {
    'bus': ['north', 'south'],
    'cost': pl.DataFrame({'generator': GENERATORS, 'value': COST}),
    'load': pl.DataFrame({'bus': ['north', 'south'], 'value': LOAD}),
}


_LABELS_AND_MAP = pl.DataFrame({'generator': GENERATORS, 'gen_bus': ['south', 'north', None]})


def test_a_map_is_not_a_column_of_the_index_it_runs_over():
    """The one stray column that is refused rather than filtered away.

    An index may carry anything — attributes, other frameworks' fields — and
    the extras are dropped. A column named after a lookup over that dimension
    is not an extra: it is a map somebody meant to supply, and dropping it
    would build the model they did not write.
    """
    with pytest.raises(DataError, match=re.escape("index for dimension 'generator' carries a 'gen_bus' column")):
        lps.solve(BASE, {**BASE_SOURCES, 'gen_bus': _RELATION, 'generator': _LABELS_AND_MAP})


def test_a_map_alone_does_not_say_which_labels_exist():
    """A map is a relation over a dimension, never the dimension.

    It may omit members and its key order is whatever someone typed, so reading
    the label set out of it would let an added entry create a member and a
    reordered map re-order the axis that ``shift`` reads positionally. With
    nothing supplying ``generator``'s labels, the map over it leaves the
    dimension without an index, and both lanes say so.
    """
    with pytest.raises(DataError, match=re.escape("has its maps (sources['gen_bus'])")):
        lps.solve(BASE, {**BASE_SOURCES, 'gen_bus': _RELATION})


# ---------------------------------------------------------------------------
# A map supplied under the lookup's own key
# ---------------------------------------------------------------------------


#: `gen_bus` takes its own source key, and the two columns say what it is — the
#: dimension it runs over, and the space its values are labels of. `g3` is
#: mapped nowhere by being in no row.
SUPPLIED = BASE

_RELATION = pl.DataFrame({'generator': ['g1', 'g2'], 'bus': ['north', 'south']})
_SUPPLIED_SOURCES = {**BASE_SOURCES, 'generator': GENERATORS, 'gen_bus': _RELATION}


def test_a_supplied_relation_reaches_the_declared_map_without_touching_the_index():
    """The point: a caller adds a map to a dimension whose index is not theirs.

    `g3` is the cheapest generator and contributes nothing, which is what an
    unmapped label means. The index goes in as a bare label list, so nothing
    here rewrites a table someone else generated.
    """
    with lps.solve(SUPPLIED, _SUPPLIED_SOURCES) as result:
        built = by_coord(result, 'p', 'generator')
        assert result.objective == pytest.approx(13.0)
    assert built['g3'] == pytest.approx(0.0), 'a generator in no row of the map is a generator on no bus'


def test_a_supplied_relation_agrees_with_the_oracle():
    """Both lanes read the relation through the one front door, so both see it."""
    from tests.differential import differential
    from tests.oracle import pd

    data = {
        'cost': pd.Series([1.0, 2.0, 0.5], index=['g1', 'g2', 'g3']),
        'load': pd.Series([5.0, 4.0], index=['north', 'south']),
        'bus': ['north', 'south'],
        'generator': ['g1', 'g2', 'g3'],
        'gen_bus': _RELATION,
    }
    with differential(SUPPLIED, data) as run:
        assert run.result.objective == pytest.approx(13.0)


def test_a_partial_map_is_supplied_as_the_rows_it_has():
    """Absence is the absent row here as everywhere else, which is the whole change.

    As a column on the index the same map needs a cell for every label and
    spells "unmapped" as a null — the one place the absence rules read a hole
    as data. Supplied as a relation it says nothing about `g3` at all, and a
    null in it is refused the way a null in a parameter's values is.
    """
    holed = pl.DataFrame({'generator': ['g1', 'g2', 'g3'], 'bus': ['north', 'south', None]})
    with pytest.raises(DataError, match=r"lookup 'gen_bus' carries 1 row\(s\) with a null in 'bus'"):
        lps.solve(SUPPLIED, {**_SUPPLIED_SOURCES, 'gen_bus': holed})


def test_a_label_space_lookup_is_supplied_under_its_own_name():
    """The kind with no target names its column after itself.

    One rule for both kinds — *the space the values are labels of* — which for
    a lookup owning its label space is the lookup. There is no dimension to
    borrow a name from, which is why naming the column after the target cannot
    be the rule.
    """
    spec = {**_spec(), 'dimensions': {'snapshot': {'dtype': 'int'}}}
    spec['constraints']['c']['where'] = 'period == 1'
    sources = {
        'snapshot': [0, 1, 2],
        'load': _load(),
        'period': pl.DataFrame({'snapshot': [0, 1], 'period': [1, 1]}),
    }
    with lps.solve(spec, sources) as result:
        built = sorted(row['snapshot'] for row in result.primal('x').to_dicts())
    assert built == [0, 1, 2], 'the variable is unmasked; only the constraint reads the label space'
    assert result.objective == pytest.approx(3.0), 'snapshot 2 is in no row of the map, so nothing constrains it'


@pytest.mark.parametrize(
    ('relation', 'match'),
    [
        pytest.param(
            pl.DataFrame({'generator': ['g1'], 'gen_bus': ['north']}),
            r"must carry columns \['generator', 'bus'\]",
            id='named-after-itself-not-its-target',
        ),
        pytest.param(
            pl.DataFrame({'bus': ['north']}),
            r"must carry columns \['generator', 'bus'\]",
            id='no-key-column',
        ),
        pytest.param(
            pl.DataFrame({'generator': ['g1', 'g1'], 'bus': ['north', 'south']}),
            r"maps 1 'generator' label\(s\) more than once: g1",
            id='mapped-twice',
        ),
        pytest.param(
            pl.DataFrame({'generator': ['g9'], 'bus': ['north']}),
            r"lookup 'gen_bus' maps g9, which are not labels of 'generator'",
            id='key-is-not-a-label',
        ),
    ],
)
def test_a_supplied_relation_is_held_to_what_a_map_is(relation, match):
    """Single-valued, keyed by labels that exist, and spelled as the pair it is.

    The stray key is the one refusal the column form got for free: a map that
    rides the index cannot name a label the index lacks. Dropping it instead
    would place that generator's terms nowhere while the model built and solved.
    """
    with pytest.raises(DataError, match=match):
        lps.solve(SUPPLIED, {**_SUPPLIED_SOURCES, 'gen_bus': relation})


def test_a_map_with_no_author_at_all_is_refused():
    """Neither is not a spelling of empty: a lookup nothing supplies is missing data.

    The counterpart of a declared parameter with no data, and what replaced
    three checks — a map arriving under its own key is present and
    single-valued by construction, so the transport cannot be short of it.
    """
    with pytest.raises(DataError, match="no data provided for lookup 'gen_bus'"):
        lps.solve(SUPPLIED, {**BASE_SOURCES, 'generator': GENERATORS})


def test_a_supplied_map_does_not_say_which_labels_exist():
    """A relation over a dimension is not the dimension, whoever holds it.

    Reading the label set out of the map would let an omitted row delete a
    member, and `g3` — mapped nowhere and still a generator — is exactly the
    row that would vanish.
    """
    with pytest.raises(DataError, match=re.escape("has its maps (sources['gen_bus'])")):
        lps.solve(SUPPLIED, {**BASE_SOURCES, 'gen_bus': _RELATION})


@pytest.mark.parametrize('lane', ['relational', 'eager'])
def test_a_supplied_relation_is_refused_the_same_way_on_both_lanes(lane):
    """One defect, one sentence: the checks live in the door both lanes enter."""
    from tests.oracle import lpspec_linopy

    build = lps.solve if lane == 'relational' else lpspec_linopy.build
    with pytest.raises(DataError, match=r"maps 1 'generator' label\(s\) more than once"):
        build(
            SUPPLIED,
            {**_SUPPLIED_SOURCES, 'gen_bus': pl.DataFrame({'generator': ['g1', 'g1'], 'bus': ['north', 'south']})},
        )


#: Two maps out of one dimension into the same target — the PyPSA shape, where
#: a line has a sending and a receiving bus. What it is here for: every check
#: over a dimension's maps has to run per map, and one lookup cannot tell a
#: loop that runs once from a loop that runs per name.
TWO_MAPS = {
    'dimensions': {'line': {}, 'bus': {'dtype': 'str'}},
    'lookups': {
        'line_from': {'over': 'line', 'into': 'bus'},
        'line_to': {'over': 'line', 'into': 'bus'},
    },
    'parameters': {'flow_max': {'dims': ['line']}, 'load': {'dims': ['bus']}},
    'variables': {'f': {'foreach': ['line'], 'bounds': {'lower': 0, 'upper': 'flow_max'}}},
    'constraints': {'served': {'foreach': ['bus'], 'expression': 'sum(f, by=line_to) >= load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(f, over=line)'},
}

_TWO_MAP_SOURCES = {
    'line': ['l1', 'l2'],
    'bus': ['north', 'south'],
    'flow_max': pl.DataFrame({'line': ['l1', 'l2'], 'value': [10.0, 10.0]}),
    'load': pl.DataFrame({'bus': ['north', 'south'], 'value': [1.0, 2.0]}),
    'line_from': pl.DataFrame({'line': ['l1', 'l2'], 'bus': ['south', 'north']}),
    'line_to': pl.DataFrame({'line': ['l1', 'l2'], 'bus': ['north', 'south']}),
}


def test_two_maps_into_one_target_each_take_their_own_key():
    """Two relations of identical schema, told apart by the key they arrive under.

    The alternative — one table per ``(over, into)`` pair — has nowhere to put
    the second, which is why the key is the lookup and not the pair.
    """
    with lps.solve(TWO_MAPS, _TWO_MAP_SOURCES) as result:
        assert result.objective == pytest.approx(3.0), 'each line serves the bus line_to sends it to'


def test_the_second_map_is_checked_as_hard_as_the_first():
    """Per-map, not per-dimension: the check runs for `line_from` as for `line_to`."""
    index = pl.DataFrame({'line': ['l1', 'l2'], 'line_from': ['south', 'north']})
    with pytest.raises(DataError, match=re.escape("carries a 'line_from' column")):
        lps.solve(TWO_MAPS, {**_TWO_MAP_SOURCES, 'line': index})


#: A parameter written positionally over a dimension that carries a supplied
#: map. `cost` is a bare list, so which label each number belongs to is the
#: index's row order — the order a map joined onto it must not disturb. `cap`
#: pins the solution to `t = 0` alone, so the objective *is* that label's cost.
POSITIONAL = {
    'dimensions': {'t': {'dtype': 'int'}, 'g': {'dtype': 'str'}},
    'lookups': {'g_of': {'over': 't', 'into': 'g'}},
    'parameters': {'cost': {'dims': ['t']}, 'cap': {'dims': ['t']}},
    'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {'c': {'foreach': ['t'], 'expression': 'x >= cap'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x * cost, over=t)'},
}


def test_a_supplied_map_does_not_reorder_the_index_it_joins_onto():
    """A label's position is its ordinal, and joining a map on may not move it.

    A positional shape is placed against the labels read back off the index
    *after* the map has been joined onto it, so a join free to reorder hands
    every one of these numbers to the wrong label — and `shift`, which reads
    ordinals, moves every coordinate with it. Both lanes then agree on a model
    neither caller wrote, which is why the check is a number here rather than a
    comparison between the two.
    """
    sources = {
        't': pl.DataFrame({'t': [0, 1, 2]}),
        'g': ['n', 's'],
        'g_of': pl.DataFrame({'t': [0, 1, 2], 'g': ['n', 'n', 's']}),
        'cost': [1.0, 10.0, 100.0],
        'cap': pl.DataFrame({'t': [0, 1, 2], 'value': [1.0, 0.0, 0.0]}),
    }
    with lps.solve(POSITIONAL, sources) as result:
        assert result.objective == pytest.approx(1.0), "the first number is the first label's, whatever the join did"
