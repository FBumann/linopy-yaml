"""Inline label coordinates: structure on a dimension, never an axis.

SPEC §2's split, pinned: everything under ``dimensions:`` is an axis, and a
label a dimension's members carry is a coordinate — *targeted* when something
aggregates into it, *inline* when it is only ever selected on. What these
tests hold still: the schema tells the kinds apart by shape, an inline name
joins the flat namespace, grouping into a label is refused with the promotion
rewrite, ``check`` advises on a dimension nothing uses as an axis, and the
bind-time contract (the column arrives with the index, single-valued per
label) covers both kinds alike.
"""

from __future__ import annotations

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
            'snapshot': {'dtype': 'int', 'coords': {'period': {'dtype': 'int'}}},
        },
        'parameters': {'load': {'dims': ['snapshot']}},
        'variables': {'x': {'foreach': ['snapshot'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'c': {'foreach': ['snapshot'], 'expression': 'x >= load'}},
        'objective': {'sense': 'minimize', 'expression': objective},
    }


def _index() -> pl.DataFrame:
    return pl.DataFrame({'snapshot': [0, 1, 2], 'period': [1, 1, 2]})


def _load() -> pl.DataFrame:
    return pl.DataFrame({'snapshot': [0, 1, 2], 'value': [1.0, 2.0, 3.0]})


def test_the_two_coordinate_kinds_parse_apart():
    """The value's shape decides the kind, and each surfaces only as itself."""
    schema = load_model(
        {
            'dimensions': {
                'bus': {},
                'generator': {'coords': {'bus': 'bus', 'tech': {'dtype': 'str'}}},
            },
        }
    )
    block = schema.dimensions['generator']
    assert block.targeted == {'bus': 'bus'}
    assert list(block.labels) == ['tech']
    assert block.labels['tech'].dtype == 'str'


def test_an_inline_coordinate_puts_nothing_under_dimensions():
    """The file with one axis declares one dimension — the original complaint."""
    schema = load_model(_model())
    assert list(schema.dimensions) == ['snapshot']


def test_an_inline_coordinate_joins_the_flat_namespace():
    model = _model()
    model['parameters']['period'] = {'dims': ['snapshot']}
    with pytest.raises(LpspecError, match="Label coordinate 'period' collides with the parameter"):
        load_model(model)


def test_two_dimensions_cannot_carry_one_label_name():
    model = _model()
    model['dimensions']['scenario'] = {'coords': {'period': {'dtype': 'int'}}}
    with pytest.raises(LpspecError, match="'period' collides"):
        load_model(model)


def test_grouping_into_a_label_is_refused_with_the_promotion_rewrite():
    """The error teaches the one-word promotion, not merely the refusal."""
    with pytest.raises(LpspecError, match="is a label on 'snapshot'") as caught:
        lps.check(_model('sum(x, over=snapshot, group_by=period)'))
    assert 'coords: {period: period}' in str(caught.value)


def test_check_advises_a_label_space_wearing_a_dimensions_clothes():
    """A dim that only serves as a coordinate target is advice, not an error."""
    model = _model()
    model['dimensions']['period'] = {'dtype': 'int'}
    model['dimensions']['snapshot']['coords'] = {'period': 'period'}
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
        'dimensions': {
            'bus': {},
            'generator': {'coords': ['bus']},
        },
        'parameters': {'cost': {'dims': ['generator']}},
        'variables': {'p': {'foreach': ['generator'], 'bounds': {'lower': 0, 'upper': 1}}},
        'constraints': {'c': {'foreach': ['generator'], 'expression': 'p <= 1'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(sum(p * cost, over=generator, group_by=bus), over=bus)'},
    }
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        lps.check(model)


def test_a_clean_model_checks_silently():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        lps.check(_model())


def test_an_inline_column_arrives_with_the_index():
    with lps.solve(_model(), {'load': _load(), 'snapshot': _index()}) as solution:
        assert solution.objective == pytest.approx(6.0)

    missing = _index().drop('period')
    with pytest.raises(DataError, match='missing declared coordinate column'):
        lps.build(_model(), {'load': _load(), 'snapshot': missing})


def test_an_inline_coordinate_is_single_valued_per_label():
    doubled = pl.DataFrame({'snapshot': [0, 0, 1, 2], 'period': [1, 2, 1, 2]})
    with pytest.raises(DataError, match='more than one value per label'):
        lps.build(_model(), {'load': _load(), 'snapshot': doubled})


def _unused_target_model(month: dict) -> dict:
    """#488's incremental multi-period shape.

    The flat ``snapshot`` index declares every coordinate it will need, but no
    constraint groups into ``month`` yet — only ``period`` is used.
    """
    return {
        'dimensions': {
            'snapshot': {'dtype': 'int', 'coords': ['period', 'month']},
            'period': {'dtype': 'int'},
            'month': month,
        },
        'parameters': {'cap': {'dims': ['period']}},
        'variables': {'p': {'foreach': ['snapshot'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {
            'budget': {'foreach': ['period'], 'expression': 'sum(p, over=snapshot, group_by=period) <= cap'}
        },
        'objectives': {'o': {'sense': 'maximize', 'expression': 'sum(p, over=snapshot)'}},
    }


def _unused_target_sources() -> dict:
    return {
        'snapshot': pl.DataFrame({'snapshot': [0, 1, 2], 'period': [2030, 2030, 2050], 'month': ['jan', 'feb', 'jan']}),
        'cap': pl.DataFrame({'period': [2030, 2050], 'value': [5.0, 5.0]}),
    }


@pytest.mark.parametrize(
    ('month', 'extra'),
    [
        pytest.param({'dtype': 'str'}, {'month': pl.DataFrame({'month': ['jan', 'feb']})}, id='index-in-sources'),
        pytest.param({'values': ['jan', 'feb']}, {}, id='values-on-the-declaration'),
    ],
)
def test_a_coordinate_may_target_a_dimension_nothing_spans_yet(month, extra):
    """#488: the first build after declaring a coordinate, before its constraint exists."""
    with lps.solve(_unused_target_model(month), _unused_target_sources() | extra) as solution:
        assert solution.objective == pytest.approx(10.0), 'each period caps its snapshots at 5, so the model builds'


def test_an_unused_target_still_checks_containment():
    short = {'month': pl.DataFrame({'month': ['jan']})}
    with pytest.raises(DataError, match="not 'month' coordinates"):
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
    """The inline `period` column arrives with the index on the eager lane too —
    both lanes reach the 6.0 the relational test above asserts.

    The oracle is imported in the body rather than at module scope: every other
    test here is linopy-free and has to keep running on the bare install, so
    this one test skips there instead of failing on a missing pandas.
    """
    from tests.differential import differential
    from tests.oracle import pd

    data = {'load': pd.Series({0: 1.0, 1: 2.0, 2: 3.0}).rename_axis('snapshot')}
    coords = {'snapshot': pd.DataFrame({'snapshot': [0, 1, 2], 'period': [1, 1, 2]})}
    with differential(_model(), data, coords) as run:
        assert run.oracle == pytest.approx(6.0)
