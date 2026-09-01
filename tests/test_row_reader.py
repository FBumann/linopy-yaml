"""``model.row`` — the verb for *this row is wrong and I do not know why*.

`typeset` renders the model as math before any data, and `Result.dual` gives a
row's number without its terms. Neither answers what a named row at a named
coordinate actually says, which is the question a model with a hundred thousand
rows is debugged by.

The claim that makes it worth having is that it reads the **built** row and not
the declared one: a coefficient that data scaled, a term whose variable was
absent, a row a ``where`` removed. Every test here is a case where the file and
the built row differ, because a reader that only agrees with the file would be
`typeset` with extra steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

import lpspec as lps
from lpspec.errors import LpspecError
from tests.conftest import DISPATCH_SPEC, override

if TYPE_CHECKING:
    from lpspec.relational.result import ConstraintRow

DATA = {
    'generator': pl.DataFrame({'generator': ['wind', 'gas']}),
    'p_max': pl.DataFrame({'generator': ['wind', 'gas'], 'value': [40.0, 200.0]}),
    'cost': pl.DataFrame({'generator': ['wind', 'gas'], 'value': [1.0, 50.0]}),
    'snapshot': pl.DataFrame({'snapshot': [0, 1, 2, 3]}),
    'load': pl.DataFrame({'snapshot': [0, 1, 2, 3], 'value': [80.0, 60.0, 100.0, 45.0]}),
}

#: Two variables in one row, and a coefficient that only data knows — the two
#: things a rendered coordinate and a built read exist for.
COMMITMENT: dict[str, Any] = {
    'dimensions': {'t': {'dtype': 'int'}, 'g': {'dtype': 'str'}},
    'parameters': {'p_max': {'dims': ['g']}, 'load': {'dims': ['t']}},
    'variables': {
        'p': {'foreach': ['t', 'g'], 'bounds': {'lower': 0, 'upper': 'p_max'}},
        'u': {'foreach': ['t', 'g'], 'domain': 'binary'},
    },
    'constraints': {
        'commit': {'foreach': ['t', 'g'], 'expression': 'p <= p_max * u'},
        'balance': {'foreach': ['t'], 'expression': 'sum(p, over=g) == load'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
}

COMMITMENT_DATA = {
    't': [0, 1],
    'g': ['wind', 'gas'],
    'p_max': pl.DataFrame({'g': ['wind', 'gas'], 'value': [40.0, 200.0]}),
    'load': pl.DataFrame({'t': [0, 1], 'value': [80.0, 60.0]}),
}


def _terms(row: ConstraintRow) -> list[tuple[str, str, float]]:
    """The row's terms as tuples, for a readable assertion."""
    return list(row.terms.iter_rows())


def test_a_row_is_its_terms_its_comparison_and_its_right_hand_side() -> None:
    """The whole shape, on a row whose right-hand side only the data knows."""
    with lps.build(DISPATCH_SPEC, DATA) as model:
        row = model.row('balance', snapshot=2)

    assert _terms(row) == [('p', '2, wind', 1.0), ('p', '2, gas', 1.0)]
    assert (row.sense, row.rhs) == ('==', 100.0), 'the right-hand side is the bound value, not the parameter name'


def test_printing_a_row_gives_the_line_linopy_gives() -> None:
    """The form this verb is read in, and it is deliberately linopy's.

    Their ``Constraint.print()`` renders ``+1 p[1, wind] + 50 p[1, gas] … >=
    60.0``; a reader arriving from there should not have to learn a second way
    to read a constraint. What is added is the row's own identity on the same
    line, where linopy prints it as a header.
    """
    with lps.build(COMMITMENT, COMMITMENT_DATA) as model:
        printed = str(model.row('commit', t=1, g='gas'))

    assert printed == 'commit[t=1, g=gas]: +1 p[1, gas] -200 u[1, gas] <= 0'


def test_a_row_too_wide_to_spell_out_summarises_instead_of_truncating() -> None:
    """Twelve terms of three hundred are twelve arbitrary ones.

    What a wide row is actually asked is how much of it each declaration
    contributes and whether its coefficients span an order of magnitude the
    solve will pay for — both of which fit the same line. The model here has a
    thousand-fold spread *inside one row*, which is the fault
    ``coefficient_range`` reports per declaration and nothing reported per row.
    """
    generators = [f'g{i}' for i in range(300)]
    spec = {
        'dimensions': {'t': {'dtype': 'int'}, 'g': {'dtype': 'str'}},
        'parameters': {'cost': {'dims': ['g']}, 'load': {'dims': ['t']}},
        'variables': {
            'p': {'foreach': ['t', 'g'], 'bounds': {'lower': 0, 'upper': 100}},
            'slack': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 9}},
        },
        'constraints': {'balance': {'foreach': ['t'], 'expression': 'sum(p * cost, over=g) + slack * 1000 >= load'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
    }
    data = {
        't': [0],
        'g': generators,
        'cost': pl.DataFrame({'g': generators, 'value': [0.001 * (i + 1) for i in range(300)]}),
        'load': pl.DataFrame({'t': [0], 'value': [5.0]}),
    }
    with lps.build(spec, data) as model:
        row = model.row('balance', t=0)

    assert row.terms.height == 301, 'the frame keeps every term whatever the line does'
    assert str(row) == 'balance[t=0]: 301 terms — p: 300 (|coef| 0.001…0.3), slack: 1 (|coef| 1000) >= 5'


def test_a_declaration_whose_coefficients_are_all_one_says_so_once() -> None:
    """A single magnitude prints as itself, not as a range against itself."""
    generators = [f'g{i}' for i in range(30)]
    data = {
        'generator': generators,
        'p_max': pl.DataFrame({'generator': generators, 'value': [10.0] * 30}),
        'cost': pl.DataFrame({'generator': generators, 'value': [1.0] * 30}),
        'snapshot': pl.DataFrame({'snapshot': [0]}),
        'load': pl.DataFrame({'snapshot': [0], 'value': [5.0]}),
    }
    with lps.build(DISPATCH_SPEC, data) as model:
        assert str(model.row('balance', snapshot=0)) == 'balance[snapshot=0]: 30 terms — p: 30 (|coef| 1) == 5'


def test_it_answers_on_a_model_that_was_never_solved() -> None:
    """The reason this is ``Model``'s and not ``Result``'s.

    A model too wrong to solve is exactly the model whose rows need reading,
    so the verb may not require a solver to have run — or even to exist.
    """
    with lps.build(DISPATCH_SPEC, DATA) as model:
        assert model.diagnostics().solves == 0, 'nothing has been solved, and the row still reads'
        assert model.row('balance', snapshot=0).rhs == 80.0


def test_a_row_spanning_two_declarations_names_both() -> None:
    """Why ``coordinate`` is rendered rather than spread across dim columns.

    ``p <= p_max * u`` puts a term from each of two variables in one row. They
    happen to share dims here; a frame schema still cannot promise that, and
    the coefficient on ``u`` is the one the *data* supplied.
    """
    with lps.build(COMMITMENT, COMMITMENT_DATA) as model:
        row = model.row('commit', t=1, g='gas')

    assert _terms(row) == [('p', '1, gas', 1.0), ('u', '1, gas', -200.0)], (
        'p - p_max*u <= 0, with p_max the bound value for gas'
    )
    assert (row.sense, row.rhs) == ('<=', 0.0)


def test_the_coefficient_is_the_one_data_produced_not_the_one_declared() -> None:
    """The claim `typeset` cannot make: the file says ``p_max``, the row says 40."""
    with lps.build(COMMITMENT, COMMITMENT_DATA) as model:
        terms = model.row('commit', t=0, g='wind').terms
    wind = dict(zip(terms['variable'].to_list(), terms['coefficient'].to_list(), strict=True))
    assert wind['u'] == -40.0, "wind's bound, where the same declaration gives gas -200"


def test_a_term_whose_variable_is_absent_is_absent_from_the_row() -> None:
    """A built row, so a masked variable leaves a *shorter* row and shows it.

    This is the failure the verb exists for: the file says the row sums over
    both generators, and the built row has one term. Reading the file cannot
    tell you that; reading the row can.
    """
    spec = override(COMMITMENT, **{'variables.p.where': 'p_max > 100'})
    with lps.build(spec, COMMITMENT_DATA) as model:
        row = model.row('balance', t=0)

    assert _terms(row) == [('p', '0, gas', 1.0)], 'wind is masked out of p, so the balance row lost its term'


def test_a_row_a_where_removed_says_so_rather_than_answering() -> None:
    """The coordinate is legal and the row does not exist — which is the answer."""
    spec = override(COMMITMENT, **{'constraints.commit.where': 'p_max > 100'})
    with lps.build(spec, COMMITMENT_DATA) as model, pytest.raises(LpspecError, match='built no row'):
        model.row('commit', t=0, g='wind')


def test_a_partial_coordinate_is_refused_rather_than_answered_about_one_row() -> None:
    """A verb that answered about the first matching row would be reporting a
    block as if it were a row — the one wrong answer a debugging verb may not
    give."""
    with lps.build(COMMITMENT, COMMITMENT_DATA) as model, pytest.raises(LpspecError, match='declared over'):
        model.row('commit', t=0)


def test_an_unknown_constraint_lists_the_declared_ones() -> None:
    with lps.build(COMMITMENT, COMMITMENT_DATA) as model, pytest.raises(KeyError, match='balance'):
        model.row('nope', t=0, g='wind')


def test_a_closed_model_says_it_was_closed() -> None:
    """And says which row was being asked for, which is what the reader came with.

    ``row()`` keeps a refusal of its own rather than the engine's general one
    for that clause alone, so the clause is what pins it: without this the
    bespoke message could be replaced by the general one and the suite would
    not notice.
    """
    model = lps.build(DISPATCH_SPEC, DATA)
    model.close()
    with pytest.raises(LpspecError, match="no built model to read 'balance' out of"):
        model.row('balance', snapshot=0)


def test_a_update_moves_what_the_row_says() -> None:
    """The row is read off the current build, not the one that was first bound."""
    with lps.build(DISPATCH_SPEC, DATA) as model:
        assert model.row('balance', snapshot=0).rhs == 80.0
        moved = {**DATA, 'load': pl.DataFrame({'snapshot': [0, 1, 2, 3], 'value': [7.0, 7.0, 7.0, 7.0]})}
        assert model.update(moved).row('balance', snapshot=0).rhs == 7.0


def test_the_row_read_is_the_row_the_solver_was_given() -> None:
    """The strongest available check, and the one a hand-written expectation
    cannot make: every term of every row, against the matrix handed to the sink.

    A reader that resolved the row index or the column ranges wrongly would
    still return plausible terms — this is what says they are *that* row's.
    """
    with lps.build(COMMITMENT, COMMITMENT_DATA) as model:
        tables = model._engine._model.tables()
        for name in ('commit', 'balance'):
            block = model._engine._model.constraints[name]
            coordinates = block.frame.collect()
            for offset in range(block.height):
                at = block.start + offset
                given = tables.matrix_block(at, at + 1)
                where = {d: coordinates.item(offset, d) for d in coordinates.columns if d != 'row'}
                read = model.row(name, **where)
                assert read.terms['coefficient'].to_list() == pytest.approx(given['coeff'].to_list()), (
                    f'{name} at {where} does not carry the coefficients the sink was handed for row {at}'
                )
                assert read.terms.height == given.height


#: A coefficient and a right-hand side that ``%g``'s six significant digits
#: cannot tell apart from their neighbours, and a scalar declaration — the
#: cases where the *rendering* is what makes a row readable or not.
PRECISE: dict[str, Any] = {
    'dimensions': {'t': {'dtype': 'int'}, 'g': {'dtype': 'str'}},
    'parameters': {'cost': {'dims': ['g']}, 'load': {'dims': ['t']}},
    'variables': {'p': {'foreach': ['t', 'g'], 'bounds': {'lower': 0}}},
    'constraints': {'balance': {'foreach': ['t'], 'expression': 'sum(p * cost, over=g) >= load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
}

PRECISE_DATA = {
    't': [0],
    'g': ['a', 'b'],
    'cost': pl.DataFrame({'g': ['a', 'b'], 'value': [1.0000001, 12345678.0]}),
    'load': pl.DataFrame({'t': [0], 'value': [12345678.9]}),
}


def test_a_coefficient_prints_every_digit_the_data_gave_it() -> None:
    """A rendering that rounds agrees with the file in exactly the case worth reading.

    ``%g`` stops at six significant digits, which prints ``1.0000001`` as
    ``1`` and two bounds differing in the seventh identically — so the one
    line whose job is *this number is not what you wrote* would say it was.
    """
    with lps.build(PRECISE, PRECISE_DATA) as model:
        printed = str(model.row('balance', t=0))

    assert printed == 'balance[t=0]: +1.0000001 p[0, a] +12345678 p[0, b] >= 12345678.9', (
        'every digit the data carried survives, and a whole coefficient still reads as linopy prints it'
    )


def test_a_row_echoed_at_a_prompt_is_the_line_not_the_frame() -> None:
    """``repr`` is how a row is read in a REPL and in a notebook cell.

    The generated dataclass one puts a multi-line frame inside a single row's
    identity, which is the rendering this verb exists to replace.
    """
    with lps.build(COMMITMENT, COMMITMENT_DATA) as model:
        row = model.row('commit', t=1, g='gas')

    assert repr(row) == str(row) == 'commit[t=1, g=gas]: +1 p[1, gas] -200 u[1, gas] <= 0'


def test_a_row_has_one_spelling_whatever_order_its_coordinate_was_given_in() -> None:
    """One row, one identity: the declaration orders the coordinate, not the caller's keywords."""
    with lps.build(COMMITMENT, COMMITMENT_DATA) as model:
        by_declaration = model.row('commit', t=1, g='gas')
        reversed_kwargs = model.row('commit', g='gas', t=1)

    assert str(by_declaration) == str(reversed_kwargs)
    assert list(reversed_kwargs.coordinate) == ['t', 'g'], "the declaration's dim order, not the call's"


def test_a_declaration_over_no_dims_carries_no_bracket() -> None:
    """``z``, not ``z[]`` — linopy's spelling, and an empty bracket states a
    coordinate that does not exist."""
    spec = {
        'dimensions': {'g': {'dtype': 'str'}},
        'variables': {
            'p': {'foreach': ['g'], 'bounds': {'lower': 0}},
            'z': {'foreach': [], 'bounds': {'lower': 0}},
        },
        'constraints': {'total': {'foreach': [], 'expression': 'sum(p, over=g) + z <= 10'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(p) + z'},
    }
    with lps.build(spec, {'g': ['wind', 'gas']}) as model:
        assert str(model.row('total')) == 'total: +1 p[wind] +1 p[gas] +1 z <= 10'


def test_a_dimension_called_name_is_still_a_coordinate() -> None:
    """``name`` is a legal dimension, and the parameter naming the constraint
    may not take it away — so the constraint is positional."""
    spec = {
        'dimensions': {'name': {'dtype': 'str'}},
        'parameters': {'p_max': {'dims': ['name']}},
        'variables': {'p': {'foreach': ['name'], 'bounds': {'lower': 0}}},
        'constraints': {'cap': {'foreach': ['name'], 'expression': 'p <= p_max'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
    }
    data = {'name': ['wind', 'gas'], 'p_max': pl.DataFrame({'name': ['wind', 'gas'], 'value': [40.0, 200.0]})}
    with lps.build(spec, data) as model:
        assert str(model.row('cap', name='wind')) == 'cap[name=wind]: +1 p[wind] <= 40'


def test_a_coefficient_the_data_made_zero_leaves_no_term() -> None:
    """The third way a built row is shorter than its file, beside a masked
    variable and a masked row.

    What a zero coefficient states, absence already states, so the build
    prunes it (``_without_zeros``) and the row reads the matrix the sink was
    handed — which is the whole of its value, and is why the term is gone
    rather than printed as ``+0``.
    """
    zeroed = {**PRECISE_DATA, 'cost': pl.DataFrame({'g': ['a', 'b'], 'value': [0.0, 2.0]})}
    with lps.build(PRECISE, zeroed) as model:
        row = model.row('balance', t=0)

    assert _terms(row) == [('p', '0, b', 2.0)], 'a zero coefficient is not a term, so `a` is not in the row'


@pytest.mark.parametrize(
    ('coordinate', 'names'),
    [
        pytest.param({'t': '0', 'g': 'wind'}, 'Int64', id='a string against an integer dim'),
        pytest.param({'t': 0, 'g': 1}, 'Enum', id='an integer against a label dim'),
        pytest.param({'t': 0, 'g': 'nope'}, 'Enum', id='a stranger against an Enum'),
    ],
)
def test_a_label_the_dimension_cannot_hold_is_refused_in_our_own_tree(coordinate: dict[str, Any], names: str) -> None:
    """Labels arrive from JSON and CSV as the wrong type, and an ``Enum`` refuses strangers.

    All three are one failure — this is not a label the dimension has — and
    none of them may reach the caller in polars' vocabulary, which names a
    dtype comparison and not the dimension that was misspelled.
    """
    with lps.build(COMMITMENT, COMMITMENT_DATA) as model, pytest.raises(LpspecError, match='not one of its labels'):
        try:
            model.row('commit', **coordinate)
        except LpspecError as refused:
            assert names in str(refused), 'the message names the type the dimension does hold'
            raise
