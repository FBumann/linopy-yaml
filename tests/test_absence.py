"""Absence: what a missing value does to a coefficient, a row and a bound.

Absence is a first-class state here rather than a zero
(docs/reference/language/absence.md), and almost every rule in this file is a
consequence of that one decision: a sparse coefficient is still a coefficient,
a constant side with a hole is refused rather than filled, an empty group is a
zero and not a gap, and a term whose variable is absent takes its row with it.

The pairs matter more than the singles. A sparse coefficient and a sparse
*constant* are the same shape of data and are treated differently on purpose,
and an empty group is a zero while a member with no value is a refusal — so
they are checked next to each other, where the line between them is visible.
"""

from __future__ import annotations

import polars as pl
import pytest

import lpspec as lps
from lpspec.errors import DataError
from tests.conftest import by_coord, override
from tests.differential import RTOL, differential
from tests.oracle import pd

#: A masked variable broadcast onto a wider frame, then reduced back. `p` is
#: over (node, tech); `produces` adds `carrier`; the sum removes `tech`. So the
#: constraint's dims are neither a subset nor a superset of the variable's.
BROADCAST_MASK_SPEC = {
    'dimensions': {
        'node': {'dtype': 'str'},
        'tech': {'dtype': 'str'},
        'carrier': {'dtype': 'str'},
    },
    'parameters': {
        'produces': {'dims': ['tech', 'carrier']},
        'demand': {'dims': ['node', 'carrier']},
        'cost': {'dims': ['tech']},
        'installed': {'coverage': 'masked', 'dims': ['node', 'tech']},
    },
    'variables': {
        'p': {'foreach': ['node', 'tech'], 'where': 'installed > 0', 'bounds': {'lower': 0, 'upper': 'installed'}},
    },
    'constraints': {
        'balance': {'foreach': ['node', 'carrier'], 'expression': 'sum(p * produces, over=tech) == demand'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
}


def _grid(dims, labels, values):
    """The full product of *labels* as a tidy frame, one row per coordinate."""
    frame = pd.MultiIndex.from_product(labels, names=dims).to_frame(index=False)
    return frame.assign(value=values)


SPARSE_COEFFICIENT_SPEC = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'c': {'coverage': 'masked', 'dims': ['t']}, 'w': {'coverage': 'masked', 'dims': ['t']}},
    'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'cap': {'foreach': ['t'], 'expression': 'w * x <= c'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x, over=t)'},
}


#: `w` everywhere, `c` with no row at t=0 — the sparse constant side that is
#: refused unmasked and legal behind a `where`.
SPARSE_CONSTANT_DATA = {'t': [0, 1, 2], 'w': pd.Series({0: 1.0, 1: 1.0, 2: 1.0}), 'c': pd.Series({1: 4.0, 2: 5.0})}


def test_a_sparse_coefficient_is_still_a_zero_coefficient():
    """A tidy parameter table is a compressed dense array — where it can be.

    Supplying rows only where a *coefficient* is nonzero stays the language's
    sparsity idiom (the data-attachment rules). The uncovered coordinate contributes no term, the
    row survives, and both lanes agree about it: nothing was invented, the term
    simply is not there.

    Only the *constant* side lost this reading, and the test below says why.

    """
    data = {'t': [0, 1, 2], 'w': pd.Series({1: 1.0, 2: 1.0}), 'c': pd.Series({0: 0.0, 1: 4.0, 2: 5.0})}
    with differential(SPARSE_COEFFICIENT_SPEC, data, lp=True) as run:
        assert run.result.objective == pytest.approx(10.0 + 4.0 + 5.0, rel=RTOL), (
            't=0 carries `<= 0` with no term: a row that exists and constrains nothing'
        )


def test_a_sparse_constant_side_is_refused_on_both_lanes():
    """The same omission on the constant side is a `DataError`, not a zero.

    There the fill *is* the bound — `w * x <= c` with no `c` row reads `<= 0`,
    which binds rather than vanishing, and the solve reports optimal. Nothing in
    the model said so: a table left sparse is compression, not a claim.

    Refused on both lanes, in the same words, because a rule the eager lane did
    not share would be a parity break rather than a language rule (hard rule 3).
    """
    with (
        pytest.raises(DataError, match="parameter 'c' covers 1 fewer"),
        differential(SPARSE_COEFFICIENT_SPEC, SPARSE_CONSTANT_DATA),
    ):
        pass


def test_a_where_is_the_escape_from_the_constant_side_check():
    """Masking the coordinate answers the question, so it is not refused.

    The check is keyed to the rows a declaration builds, not to the coordinate
    product — the same property that keeps #312's divisor check from becoming a
    wall. Without it the remedy the error names would not work.

    """
    masked = override(SPARSE_COEFFICIENT_SPEC, **{'constraints.cap.where': 'c'})
    with differential(masked, SPARSE_CONSTANT_DATA) as run:
        assert run.result.objective == pytest.approx(10.0 + 4.0 + 5.0, rel=RTOL), (
            't=0 has no row at all, so x runs to its bound there'
        )


#: `south` is a load-only bus: both generators sit on `north`, so the group
#: behind `south`'s constant side has no members at all.
GROUPED_CONSTANT_SPEC = {
    'dimensions': {'generator': {}, 'bus': {'dtype': 'str'}},
    'lookups': {'gen_bus': {'coverage': 'masked', 'over': 'generator', 'into': 'bus'}},
    'parameters': {'capacity': {'coverage': 'masked', 'dims': ['generator']}},
    'variables': {'imports': {'foreach': ['bus'], 'bounds': {'lower': 0, 'upper': 100}}},
    'constraints': {'import_limit': {'foreach': ['bus'], 'expression': 'imports <= sum(capacity, by=gen_bus)'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(imports, over=bus)'},
}


def _grouped_constant_sources(capacity=('g1', 'g2')):
    return {
        'bus': ['north', 'south'],
        'generator': pl.DataFrame({'generator': ['g1', 'g2']}),
        'gen_bus': pl.DataFrame({'generator': ['g1', 'g2'], 'bus': ['north', 'north']}),
        'capacity': pl.DataFrame({'generator': list(capacity), 'value': [3.0, 4.0][: len(capacity)]}),
    }


def test_an_empty_group_on_the_constant_side_is_a_zero_and_not_a_gap():
    """A group with no members holds the empty sum, which is a value.

    The coverage check reads what a constant fragment produced, and a label no
    member maps to is missing from it for the same reason a label whose data was
    omitted is. Only the first is a value: `capacity` covers every generator it
    has, and the operator rules say a group with no members contributes nothing.

    Refusing it named a parameter that was complete, and none of the three
    remedies the message offers existed — there is no row to supply, and the
    `where` that would mask `south` needs a grouped sum, which the predicate
    grammar has no atom for.
    """
    with differential(GROUPED_CONSTANT_SPEC, _grouped_constant_sources(), lp=True) as run:
        assert run.result.objective == pytest.approx(7.0, rel=RTOL), 'north imports up to 3 + 4, south up to nothing'
        built = by_coord(run.result, 'imports', 'bus')

    assert built['south'] == pytest.approx(0.0), "an empty group caps south's imports at the empty sum"


#: The same story with a dim the group does not consume, so the empty label
#: has to be paired with every snapshot rather than standing on its own.
SPANNED_GROUPED_CONSTANT_SPEC = {
    **GROUPED_CONSTANT_SPEC,
    'dimensions': {**GROUPED_CONSTANT_SPEC['dimensions'], 'snapshot': {'dtype': 'int'}},
    'parameters': {'capacity': {'dims': ['snapshot', 'generator']}},
    'variables': {'imports': {'foreach': ['snapshot', 'bus'], 'bounds': {'lower': 0, 'upper': 100}}},
    'constraints': {
        'import_limit': {'foreach': ['snapshot', 'bus'], 'expression': 'imports <= sum(capacity, by=gen_bus)'}
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(sum(imports, over=bus), over=snapshot)'},
}


def test_an_empty_group_spanning_another_dim_is_zero_at_every_coordinate():
    """The empty label is a row per snapshot, not one row.

    A group consumes one dim and the fragment keeps the rest, so the value an
    empty group holds is needed at every coordinate of what is kept. One row
    would leave the others uncovered and refuse the model for the reason the
    zero was written to remove.
    """
    sources = {
        'snapshot': [0, 1],
        'bus': ['north', 'south'],
        'generator': pl.DataFrame({'generator': ['g1', 'g2']}),
        'gen_bus': pl.DataFrame({'generator': ['g1', 'g2'], 'bus': ['north', 'north']}),
        'capacity': pl.DataFrame(
            {'snapshot': [0, 0, 1, 1], 'generator': ['g1', 'g2', 'g1', 'g2'], 'value': [3.0, 4.0, 1.0, 1.0]}
        ),
    }
    with differential(SPANNED_GROUPED_CONSTANT_SPEC, sources, lp=True) as run:
        assert run.result.objective == pytest.approx(7.0 + 2.0, rel=RTOL), 'north takes both snapshots, south neither'
        built = by_coord(run.result, 'imports', 'snapshot', 'bus')

    assert built[(0, 'south')] == pytest.approx(0.0), 'the empty group is zero at every snapshot'
    assert built[(1, 'south')] == pytest.approx(0.0), 'the empty group is zero at every snapshot'


#: Two coordinates at once, so what the reached set is subtracted from is the
#: *product* of the targets: `south` reaches neither technology, and `north`
#: reaches both, so two of the four combinations have no members.
PLURAL_GROUPED_CONSTANT_SPEC = {
    **GROUPED_CONSTANT_SPEC,
    'dimensions': {**GROUPED_CONSTANT_SPEC['dimensions'], 'technology': {'dtype': 'str'}},
    'lookups': {
        'gen_bus': {'coverage': 'masked', 'over': 'generator', 'into': 'bus'},
        'gen_tech': {'over': 'generator', 'into': 'technology'},
    },
    'variables': {'imports': {'foreach': ['bus', 'technology'], 'bounds': {'lower': 0, 'upper': 100}}},
    'constraints': {
        'import_limit': {
            'foreach': ['bus', 'technology'],
            'expression': 'imports <= sum(capacity, by=[gen_bus, gen_tech])',
        }
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(sum(imports, over=bus), over=technology)'},
}


def test_an_empty_combination_of_two_groups_is_a_zero_and_not_a_gap():
    """A combination no member sits at is empty for the reason one label is.

    Grouping through two coordinates lands on a product of targets, and the
    unreached part of that product is what holds the empty sum — so subtracting
    the reached labels of each dim in turn would leave `(north, solar)` looking
    reached when nothing sits there.
    """
    sources = {
        'bus': ['north', 'south'],
        'technology': ['wind', 'solar'],
        'generator': pl.DataFrame({'generator': ['g1', 'g2']}),
        'gen_bus': pl.DataFrame({'generator': ['g1', 'g2'], 'bus': ['north', 'north']}),
        'gen_tech': pl.DataFrame({'generator': ['g1', 'g2'], 'technology': ['wind', 'solar']}),
        'capacity': pl.DataFrame({'generator': ['g1', 'g2'], 'value': [3.0, 4.0]}),
    }
    with differential(PLURAL_GROUPED_CONSTANT_SPEC, sources, lp=True) as run:
        assert run.result.objective == pytest.approx(3.0 + 4.0, rel=RTOL), 'south holds nothing at either technology'
        built = by_coord(run.result, 'imports', 'bus', 'technology')

    assert built[('south', 'wind')] == pytest.approx(0.0), 'no generator sits at (south, wind)'
    assert built[('south', 'solar')] == pytest.approx(0.0), 'no generator sits at (south, solar)'
    assert built[('north', 'solar')] == pytest.approx(4.0), 'one generator sits at (north, solar), and it is g2'


def test_a_member_with_no_value_is_still_refused_through_a_group():
    """The group that is empty *because its data is missing* keeps the refusal.

    Both generators sit on `north`, and dropping `g2`'s row leaves `north`'s
    group with one member and a hole rather than with no members. Nothing
    downstream can tell those two apart once the fragment is built, which is why
    the empty group is written down as a zero where the reason is known — this
    is the case that says the zero is not written down for every absence.
    """
    with (
        pytest.raises(DataError, match="parameter 'capacity' covers 1 fewer"),
        differential(GROUPED_CONSTANT_SPEC, _grouped_constant_sources(capacity=('g1',))),
    ):
        pass


ABSENT_VARIABLE_SPEC = {
    'dimensions': {'f': {'dtype': 'str'}},
    'parameters': {
        'gate': {'coverage': 'masked', 'dims': ['f'], 'dtype': 'bool'},
        'relmax': {'coverage': 'masked', 'dims': ['f']},
        'cost': {'coverage': 'masked', 'dims': ['f']},
    },
    'variables': {
        'x': {'foreach': ['f'], 'bounds': {'lower': 0, 'upper': 100}},
        'size': {'foreach': ['f'], 'where': 'gate', 'bounds': {'lower': 0, 'upper': 50}},
    },
    'constraints': {'envelope': {'foreach': ['f'], 'expression': 'x - relmax * size <= 0'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x * cost, over=f)'},
}


def test_a_term_whose_variable_is_absent_drops_the_row_on_both_lanes():
    """Absence propagates into the comparison; it does not zero the term.

    ``x - relmax * size <= 0`` where ``size`` is masked out used to build
    ``x <= 0`` — a row that silently pinned the flow to zero. Plausible answer,
    no error, which is goal 1 of linopy's v1 convention ("no silent wrong
    answers") and the whole of PyPSA/linopy#712. Under v1 §6 the slot is absent
    and v1 §12 drops the row instead, so ``x`` is left free at ``f=b`` and bounded only
    by its own declaration.

    The oracle is the point: the eager lane gets this from linopy's own v1
    semantics, the relational lane from carrying variable presence apart from
    the term stream. Two independent implementations, one answer.
    """
    data = {
        'f': ['a', 'b'],
        'gate': pd.Series({'a': True}),
        'relmax': pd.Series({'a': 0.5, 'b': 0.5}),
        'cost': pd.Series({'a': 1.0, 'b': 1.0}),
    }
    with differential(ABSENT_VARIABLE_SPEC, data, lp=True) as run:
        x = by_coord(run.result, 'x', 'f')
        assert x['a'] == pytest.approx(25.0, rel=RTOL), 'sized: x <= 0.5 * size, size <= 50'
        assert x['b'] == pytest.approx(100.0, rel=RTOL), 'unsized: the row is gone, so only the bound holds'


#: One rule per block, so the two regimes are two named constraints.
DEFINED_SPEC = {
    'dimensions': {'f': {'dtype': 'str'}},
    'parameters': {
        'gate': {'coverage': 'masked', 'dims': ['f'], 'dtype': 'bool'},
        'relmax': {'coverage': 'masked', 'dims': ['f']},
        'cost': {'coverage': 'masked', 'dims': ['f']},
    },
    'variables': {
        'x': {'foreach': ['f'], 'bounds': {'lower': 0, 'upper': 100}},
        'size': {'foreach': ['f'], 'where': 'gate', 'bounds': {'lower': 0, 'upper': 50}},
    },
    'constraints': {
        'envelope_sized': {'foreach': ['f'], 'where': 'size', 'expression': 'x - relmax * size <= 0'},
        'envelope_unsized': {'foreach': ['f'], 'where': 'NOT size', 'expression': 'x <= 0'},
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(x * cost, over=f)'},
}


def test_a_bare_variable_name_in_a_where_asks_whether_it_exists():
    """The escape hatch for a language where absence drops the row.

    Since a term whose variable is absent takes the row with it, a model that
    wanted the *other* reading — keep the row, treat the term as zero — needs a
    way to say which coordinates those are. A bare parameter name in a ``where``
    already asks "does this have a value here"; a bare variable name asks "does
    this exist here", and the two complementary clauses spell out both cases.

    Without it the only way to write this is a parameter mirroring the
    variable's own mask, which is two sources for one fact and drifts.
    """
    data = {
        'f': ['a', 'b'],
        'gate': pd.Series({'a': True}),
        'relmax': pd.Series({'a': 0.5, 'b': 0.5}),
        'cost': pd.Series({'a': 1.0, 'b': 1.0}),
    }
    with differential(DEFINED_SPEC, data, lp=True) as run:
        x = by_coord(run.result, 'x', 'f')
        assert x['a'] == pytest.approx(25.0, rel=RTOL), 'sized: the envelope binds'
        assert x['b'] == pytest.approx(0.0, abs=1e-9), 'unsized: the complementary clause pins it'


ABSENT_COEFFICIENT_SPEC = {
    'dimensions': {'f': {'dtype': 'str'}},
    'parameters': {'relmax': {'coverage': 'masked', 'dims': ['f']}, 'cost': {'coverage': 'masked', 'dims': ['f']}},
    'variables': {
        'x': {'foreach': ['f'], 'bounds': {'lower': 0, 'upper': 100}},
        'size': {'foreach': ['f'], 'bounds': {'lower': 0, 'upper': 50}},
    },
    'constraints': {'envelope': {'foreach': ['f'], 'expression': 'x - relmax * size <= 0'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x * cost, over=f)'},
}


def test_a_sparse_coefficient_on_the_bound_side_still_pins_the_variable():
    """The half of v1 §6's hazard that survives absence propagation.

    Same expression as ``ABSENT_VARIABLE_SPEC`` above, one operand different:
    the thing missing at ``f=b`` is the *parameter* ``relmax``, not the variable
    ``size``. Absence is a property of variables, so nothing propagates — the
    row is kept, the term is dropped, and ``x <= 0`` is built.

    That is correct and it is the documented reading of a sparse coefficient
    table, but it is the same silently-wrong shape the v1 convention removed
    from the variable side, so the absence rules now name it and this pins the behaviour
    the prose describes. The benign case is
    ``test_a_sparse_coefficient_is_still_a_zero_coefficient``:
    there the zero lands on a coefficient *and* a right-hand side, so the row
    constrains nothing. Here the right-hand side is a literal 0 and the missing
    coefficient was the whole bound.
    """
    data = {
        'f': ['a', 'b'],
        'relmax': pd.Series({'a': 0.5}),  # no row at 'b'
        'cost': pd.Series({'a': 1.0, 'b': 1.0}),
    }
    with differential(ABSENT_COEFFICIENT_SPEC, data, lp=True) as run:
        x = by_coord(run.result, 'x', 'f')
        assert x['a'] == pytest.approx(25.0, rel=RTOL), 'sized: x <= 0.5 * size, size <= 50'
        assert x['b'] == pytest.approx(0.0, abs=1e-9), 'the row survived the missing coefficient and pins x'


SCALAR_MASKED_SPEC = {
    'dimensions': {'f': {'dtype': 'str'}},
    'parameters': {'cost': {'coverage': 'masked', 'dims': ['f']}, 'budget': {'dims': []}},
    'variables': {
        'x': {'foreach': ['f'], 'bounds': {'lower': 0, 'upper': 100}},
        'slack': {'foreach': [], 'where': 'budget > 1000', 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {'cap': {'foreach': [], 'expression': 'sum(x, over=f) - slack <= budget'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x * cost)'},
}


def test_a_masked_out_scalar_variable_drops_the_row_that_uses_it():
    """Law 7 holds at no dimension either (#340).

    Was: a scalar's presence was `select()` over no dims, and polars cannot hold
    rows with no columns — collecting reports (0, 0), so present and absent were
    one frame and nothing downstream could restrict on it. `cap` stayed enforced
    with its term gone, as `sum(x) <= budget` — a constraint the file does not
    contain. The presence frame now carries a marker column instead, and a
    keyless restriction is a cross join.

    Held here as well as in the parity suite because that suite needs the
    ``[linopy]`` extra, and this has to be true on the bare install too.

    """
    data = {'f': ['a', 'b'], 'cost': pl.DataFrame({'f': ['a', 'b'], 'value': [1.0, 2.0]}), 'budget': 120.0}

    with lps.solve(SCALAR_MASKED_SPEC, data) as sol:
        assert sol.dual('cap').height == 0, 'the row is gone, not slackened — a dropped row has no dual'
        assert sol.objective == pytest.approx(300.0), 'unbudgeted, both generators run flat out'


def test_a_mask_survives_a_broadcast_into_a_reduction():
    """`Presence.keyed_by=None` means "keyed by the fragment's dims", and a
    product may *widen* dims — so carrying it through the widening re-read
    `p`'s (node, tech) presence as keyed by (node, tech, carrier) and
    `_propagate_absence` selected a column it never had (#345).

    Unmasked the same model was fine, which is what made it look like a problem
    with the coordinate dim rather than with the mask. The whole benchmark
    `sector` case sat on this.

    """
    data = {
        'node': ['n1', 'n2'],
        'tech': ['t1', 't2'],
        'carrier': ['elec', 'heat'],
        # a tech produces exactly one carrier, which is what makes `produces` sparse
        'produces': _grid(['tech', 'carrier'], [['t1', 't2'], ['elec', 'heat']], [1.0, 0.0, 0.0, 1.0]),
        'demand': _grid(['node', 'carrier'], [['n1', 'n2'], ['elec', 'heat']], [10.0, 20.0, 10.0, 20.0]),
        'cost': pd.Series({'t1': 1.0, 't2': 2.0}),
        'installed': _grid(['node', 'tech'], [['n1', 'n2'], ['t1', 't2']], [100.0] * 4),
    }

    with differential(BROADCAST_MASK_SPEC, data) as run:
        assert run.result.objective == pytest.approx(100.0, rel=RTOL)
