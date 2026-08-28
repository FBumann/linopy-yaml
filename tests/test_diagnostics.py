"""What the build reports about itself: omissions, timings, magnitudes, sparsity.

A diagnostic is a claim about a model the solver never sees. A row that lost
every term is not built, and saying so is the difference between a smaller
problem and a silently different one; a coefficient range is what turns "the
solver struggled" into a named block; a parameter short of its dimensions is
reported rather than judged, because whether the gap is a mistake is the
modeller's call and not the engine's.

They are gathered here rather than beside the assembly they measure because
that is the question a reader arrives with — *what does the build tell me* —
and because a report that survives the model being released is a property of
the report, not of the matrix.
"""

from __future__ import annotations

import polars as pl
import pytest

import lpspec as lps
from lpspec.relational.sinks import SOLVERS
from tests.conftest import SOLVER_VECTOR_LOAD, SOLVER_VECTOR_MODEL
from tests.differential import RTOL, differential


@pytest.mark.parametrize('solver_name', sorted(SOLVERS))
@pytest.mark.parametrize('batch_rows', [1, 2, 7, 100_000], ids=['one', 'two', 'odd', 'whole'])
def test_a_row_with_no_terms_is_not_built_and_is_reported(solver_name, batch_rows):
    """A row that lost every term is not a constraint, and the build says so.

    `where: "t > 0"` leaves `balance` at `t = 0` with nothing to sum. Three
    provenances reach that shape — an absent variable, an empty reduction, a
    missing coefficient — and the language used to answer them differently, so
    the same empty row meant different things depending on how it emptied. The
    rule is now at the level the property lives at: no variable terms, no row.

    **The omission is reported, and that is what makes dropping defensible.**
    An unenforced constraint the caller cannot see is the failure this used to
    guard against by keeping the row; `diagnostics().omissions` answers it without asking
    the solver to carry a comparison nothing can fail.

    Ragged batches because the range loop is where a *surviving* seat would be
    lost — labels are compacted when a row goes, so the dense vector and the
    chunk ranges have to agree about the narrower block. Both solvers, because
    the seating is theirs jointly.
    """
    model = {
        'dimensions': {'t': {'dtype': 'int'}, 'g': {'dtype': 'str'}},
        'parameters': {'load': {'dims': ['t']}},
        'variables': {'p': {'foreach': ['t', 'g'], 'where': 't > 0', 'bounds': {'lower': 0, 'upper': 100}}},
        'constraints': {'balance': {'foreach': ['t'], 'expression': 'sum(p, over=g) == load'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(sum(p, over=g), over=t)'},
    }
    data = {'t': [0, 1, 2], 'g': ['a', 'b'], 'load': pl.DataFrame({'t': [0, 1, 2], 'value': [5.0, 4.0, 6.0]})}
    with lps.build(model, data) as bound:
        tables = bound._engine._model.tables()
        occupied = sorted(set(tables.matrix_block(0, tables.row_count)['row'].to_list()))
        assert occupied == [0, 1], 'the block closes up around the gap'
        assert bound.diagnostics().omissions.to_dicts() == [{'constraint': 'balance', 'rows_not_built': 1}]
        solution = bound._engine.solve(solver_name, batch_rows=batch_rows)
        assert solution.termination_condition == 'optimal'
        assert solution.objective == pytest.approx(4.0 + 6.0, rel=RTOL), 'the two built rows still bind'


def test_omissions_is_empty_when_every_declared_row_is_built():
    """The common case says nothing, so the report is a signal rather than noise."""
    with lps.build(SOLVER_VECTOR_MODEL, SOLVER_VECTOR_LOAD) as bound:
        assert bound.diagnostics().omissions.is_empty()


def test_a_row_a_propagated_absence_deleted_is_reported_too():
    """The other way a declared row goes missing, and it used to go unrecorded (#944).

    A row that loses *all* its terms was always counted. This one keeps three of
    them: ``x`` exists at both coordinates with a bound of its own, and the row
    is deleted because absence travelled out of ``y``. Nothing ever counted it,
    because a restricted row is removed before there is a row to count — so the
    model quietly enforced half of what it declared and said so nowhere.

    Asserted through the objective as well as the report, because the point is
    that the two disagree with each other: `both[b]` reads `x[b] >= 5` and its
    loss is worth 5 of the answer.
    """
    model = {
        'dimensions': {'g': {'dtype': 'str'}},
        'parameters': {'cap': {'dims': ['g']}, 'extra': {'dims': ['g']}},
        'variables': {
            'x': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 'cap'}},
            'y': {'foreach': ['g'], 'where': 'extra', 'bounds': {'lower': 0, 'upper': 0}},
        },
        'constraints': {'both': {'foreach': ['g'], 'expression': 'x + y >= 5'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(x, over=g)'},
    }
    data = {
        'g': ['a', 'b'],
        'cap': pl.DataFrame({'g': ['a', 'b'], 'value': [10.0, 10.0]}),
        'extra': pl.DataFrame({'g': ['a'], 'value': [1.0]}),
    }
    with lps.build(model, data) as bound:
        assert bound.diagnostics().omissions.to_dicts() == [{'constraint': 'both', 'rows_not_built': 1}], (
            'the row absence travelled out of y and deleted is counted'
        )
        assert bound.solve().objective == pytest.approx(5.0, rel=RTOL), (
            'and it is worth 5: only one of the two declared rows is enforced'
        )


def test_diagnostics_say_where_the_time_went(tmp_path):
    """A run that is slower than it should be can say which phase the time went to.

    `timings` is advisory wall time, so nothing here asserts a magnitude —
    only that each phase that ran left a clock, that none ran backwards, and
    that they accumulate across calls the way `solves` counts.
    """
    with lps.build(SOLVER_VECTOR_MODEL, SOLVER_VECTOR_LOAD) as bound:
        built = bound.diagnostics().timings
        assert set(built) == {'bind', 'build'}, (
            'a model only built has spent time binding sources and building frames, nowhere else'
        )
        assert all(seconds >= 0 for seconds in built.values()), 'a wall clock cannot run backwards'

        bound.solve()
        bound.write(tmp_path / 'model.lp')
        ran = bound.diagnostics().timings
        assert set(ran) == {'bind', 'build', 'handoff', 'solve', 'write'}, (
            'a solve adds the hand-off and the solver run, a write adds the file stream'
        )
        assert all(seconds >= 0 for seconds in ran.values()), 'a wall clock cannot run backwards'

        snapshot = dict(ran)
        bound.solve()
        assert bound.diagnostics().timings['solve'] >= ran['solve'], (
            'the clocks accumulate across solves, the way `solves` counts'
        )
        assert ran == snapshot, 'a diagnostics snapshot is its own dict, not a view of the running clocks'


#: Three blocks that differ only in how they are scaled, so the report has
#: something to distinguish. `badly_scaled` spans nine orders of magnitude by
#: itself, `signed` carries `ordinary`'s coefficients negated, and one cost is
#: negative — which is where a signed extreme and a magnitude part company.
SCALING = {
    'dimensions': {'unit': {'dtype': 'str'}},
    'parameters': {'small': {'dims': ['unit']}, 'large': {'dims': ['unit']}, 'cost': {'dims': ['unit']}},
    'variables': {'p': {'foreach': ['unit'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {
        'ordinary': {'foreach': ['unit'], 'expression': 'p * small >= 1'},
        'badly_scaled': {'foreach': ['unit'], 'expression': 'p * large <= 10000000'},
        'signed': {'foreach': ['unit'], 'expression': '0 - p * small >= -100'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost, over=unit)'},
}


SCALING_SOURCES = {
    'unit': ['a', 'b'],
    'small': pl.DataFrame({'unit': ['a', 'b'], 'value': [1.0, 4.0]}),
    'large': pl.DataFrame({'unit': ['a', 'b'], 'value': [1e6, 1e-3]}),
    'cost': pl.DataFrame({'unit': ['a', 'b'], 'value': [2.0, -0.5]}),
}


def test_the_coefficient_range_names_the_block_that_holds_the_outlier():
    """The spread of the matrix, per declaration — which is what a caller can act on.

    A solver prints one ``Matrix range`` for the whole model, and the model is
    too large to open, so the number says a repair is needed and not where. The
    engine builds the matrix a declaration at a time and can say both.

    Magnitudes, not signed extremes: `signed` holds `ordinary`'s coefficients
    negated and is scaled identically, which is the answer a modeller wants and
    the one a signed min/max cannot give.
    """
    with lps.build(SCALING, SCALING_SOURCES) as bound:
        spread = bound.diagnostics().coefficient_range

    assert spread.to_dicts() == [
        {'constraint': 'ordinary', 'smallest': 1.0, 'largest': 4.0},
        {'constraint': 'badly_scaled', 'smallest': 1e-3, 'largest': 1e6},
        {'constraint': 'signed', 'smallest': 1.0, 'largest': 4.0},
    ], 'one row per block in build order, and `signed` is scaled exactly as `ordinary` is'

    worst = spread.select((pl.col('largest') / pl.col('smallest')).max()).item()
    assert worst == pytest.approx(1e9), 'the conditioning number to compare against what the solver reports'


def test_the_objective_range_is_read_beside_the_matrix_and_not_in_it():
    """Costs and coefficients are different faults, so they are different fields.

    `cost` is negative on one unit, which is the whole reason the pair is
    magnitudes: a signed answer here would be ``(-0.5, 2.0)`` and say nothing
    about the four-fold spread it actually has.
    """
    with lps.build(SCALING, SCALING_SOURCES) as bound:
        seen = bound.diagnostics()

    assert seen.objective_range == (0.5, 2.0), 'the objective is read off `obj`, never off the matrix'
    assert 'objective' not in seen.coefficient_range.get_column('constraint').to_list(), (
        'the objective is not a constraint block and does not appear as one'
    )


def test_a_model_with_no_objective_has_no_objective_range():
    """A feasibility model has no costs to be badly scaled, and says so rather than lying with zeros."""
    feasibility = {k: v for k, v in SCALING.items() if k != 'objective'}
    with lps.build(feasibility, SCALING_SOURCES) as bound:
        seen = bound.diagnostics()

    assert seen.objective_range is None, 'no objective is not an objective whose coefficients span nothing'
    assert seen.coefficient_range.height == 3, 'the matrix is still reported — every constraint block is there'


#: A coefficient short of its dims, and a bound that is not. Sparse in a
#: coefficient is the ordinary case — every other position where a missing row
#: has no reading is refused already (a bound, a comparison's constant side),
#: so this is the shape where a lost row goes unreported.
SPARSE_SOURCE = {
    'dimensions': {'g': {'dtype': 'str'}, 't': {'dtype': 'int'}},
    'parameters': {'p_max': {'dims': ['g']}, 'avail': {'dims': ['t', 'g']}},
    'variables': {'p': {'foreach': ['t', 'g'], 'bounds': {'lower': 0, 'upper': 'p_max'}}},
    'constraints': {'capped': {'foreach': ['t', 'g'], 'expression': 'p * avail <= 1'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
}


SPARSE_SOURCES = {
    't': [0, 1],
    'g': ['wind', 'solar', 'gas'],
    'p_max': pl.DataFrame({'g': ['wind', 'solar', 'gas'], 'value': [1.0, 2.0, 3.0]}),
    'avail': pl.DataFrame({'t': [0, 0, 1], 'g': ['wind', 'solar', 'wind'], 'value': [1.0, 1.0, 1.0]}),
}


def test_a_parameter_short_of_its_dims_is_reported_rather_than_judged():
    """Which parameters arrived short, and by how much — not whether they should have.

    A table that lost a row and a `where:` that removed one build the same
    model, so nothing in the answer distinguishes them and nothing here tries
    to: what a missing row *means* is the absence rules', and whether it was
    meant is the caller's. Reporting is the half that can be said without
    taking the data contract.
    """
    with lps.build(SPARSE_SOURCE, SPARSE_SOURCES) as bound:
        short = bound.diagnostics().sparse_parameters

    assert short.to_dicts() == [{'parameter': 'avail', 'coordinates': 6, 'rows': 3, 'missing': 3}], (
        'the complete parameter is not a row here — an empty frame is the useful answer for a dense model'
    )


def test_a_model_whose_parameters_all_span_their_dims_reports_none():
    dense = {
        **SPARSE_SOURCES,
        'avail': pl.DataFrame({'t': [0, 0, 0, 1, 1, 1], 'g': ['wind', 'solar', 'gas'] * 2, 'value': [1.0] * 6}),
    }
    with lps.build(SPARSE_SOURCE, dense) as bound:
        assert bound.diagnostics().sparse_parameters.is_empty(), 'empty is what a complete model reports'


def test_the_sparsity_report_survives_the_model_being_released():
    """Summarised at bind from two counts binding already had, so it outlives
    the frames — the same reason the coefficient range does."""
    with lps.build(SPARSE_SOURCE, SPARSE_SOURCES) as bound:
        held = bound.diagnostics()
    released = bound.diagnostics()

    assert released.sparse_parameters.equals(held.sparse_parameters)


#: A model whose one constraint divides by a parameter: with a value missing
#: the assembly refuses it, which is a raise *after* the bind has succeeded.
UNDEFINED_DIVISOR = {
    'dimensions': {'f': {'dtype': 'str'}},
    'parameters': {'d': {'dims': ['f']}},
    'variables': {'x': {'foreach': ['f'], 'bounds': {'lower': 0, 'upper': 100}}},
    'constraints': {'c': {'foreach': ['f'], 'expression': 'x / d <= 10'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
}

DENSE_DIVISOR = {'f': ['a', 'b'], 'd': pl.DataFrame({'f': ['a', 'b'], 'value': [2.0, 5.0]})}
HALF_A_DIVISOR = {'f': ['a', 'b'], 'd': pl.DataFrame({'f': ['a'], 'value': [2.0]})}


def test_a_build_that_raises_reports_the_bind_it_got_through_and_no_size():
    """The two halves of the report part company exactly where the build stopped.

    A size is written when the model is whole, so there is none — and none of
    the *previous* build's either, which is the point: those numbers described
    a model the engine released before this build started, and reporting them
    would be the half-a-model state a failed build exists to refuse.

    What the bind measured is a different kind of fact. It was taken before
    anything raised and it is still true, and here it is the reason the build
    then raised at all — the parameter it names is the one the divisor lacked.
    """
    with lps.build(UNDEFINED_DIVISOR, DENSE_DIVISOR) as bound:
        built = bound.diagnostics()
        assert (built.columns, built.rows) == (2, 2), 'the model under test builds before it is asked not to'

        with pytest.raises(lps.DataError, match='used as a divisor'):
            bound.rebind(HALF_A_DIVISOR)
        after = bound.diagnostics()

    assert (after.columns, after.rows, after.nonzeros) == (0, 0, 0), (
        "a build that raised reported a size — a partial count, or the released build's"
    )
    assert after.sparse_parameters.to_dicts() == [{'parameter': 'd', 'coordinates': 2, 'rows': 1, 'missing': 1}], (
        'the bind that succeeded is still reported, and it names the gap the assembly then refused'
    )


def test_the_coefficient_range_survives_the_model_being_released():
    """Read off each share as it is built, so it outlives the frames it came from.

    The alternative — a reader over the live matrix — would go dark exactly
    when a caller comes back to a finished run asking why it solved badly.
    """
    with lps.build(SCALING, SCALING_SOURCES) as bound:
        held = bound.diagnostics()
    released = bound.diagnostics()

    assert released.coefficient_range.equals(held.coefficient_range), 'a released model still says how it was scaled'
    assert released.objective_range == held.objective_range


def test_the_largest_magnitude_agrees_with_the_oracle():
    """linopy answers the same question per constraint, and the two must not drift.

    Its `coefficientrange` is a *signed* min/max, so only the larger magnitude
    is comparable — the smaller one is the half this engine adds, and there is
    nothing upstream to check it against.
    """
    with differential(SCALING, SCALING_SOURCES) as run:
        ours = {row['constraint']: row['largest'] for row in run.engine.diagnostics().coefficient_range.to_dicts()}
        theirs = run.model.constraints.coefficientrange

    for name, largest in ours.items():
        expected = max(abs(theirs.loc[name, 'min']), abs(theirs.loc[name, 'max']))
        assert largest == pytest.approx(expected), f"the lanes disagree on the widest coefficient in '{name}'"
