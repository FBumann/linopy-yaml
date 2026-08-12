"""Re-solving one built model with new numbers.

Two claims, and the first is the whole contract: **a rebind answers what a
fresh build answers**. `build(model, sources | change)` is always available as
the reference, so every rung below is checked against it rather than against a
number someone wrote down — the same oracle shape as the two-lane differential
and the Benders monolith check.

The second is that the fast path is *only* a fast path. A rebind that moves a
mask renumbers labels and cannot be pushed onto a loaded solver, so the engine
rebuilds and solves cold; nothing about the answer changes, and `diagnostics().reloads`
is where a driver finds out which happened.
"""

from __future__ import annotations

import polars as pl
import pytest

import lpspec as lps
from lpspec.relational.sinks import SOLVERS

GENERATORS = ['wind', 'solar', 'gas']
SNAPSHOTS = [0, 1, 2, 3]
COORDS = {'snapshot': SNAPSHOTS}


def sources() -> dict[str, pl.DataFrame]:
    """`examples/dispatch.yaml`'s data, small enough to read in a failure."""
    return {
        'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [100.0, 60.0, 200.0]}),
        'cost': pl.DataFrame({'generator': GENERATORS, 'value': [1.0, 2.0, 50.0]}),
        'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [40.0, 80.0, 55.0, 95.0]}),
    }


#: Each rung of the rebind table (docs/api.md), and whether it may keep the
#: loaded solver. `p_max` appears twice on purpose: it gates ``where: p_max > 0``
#: *and* bounds the variable, so whether it is structural is a property of the
#: values and not of where the name appears.
RUNGS = [
    pytest.param({'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [10.0, 20.0, 30.0, 40.0]})}, True, id='rhs'),
    pytest.param({'cost': pl.DataFrame({'generator': GENERATORS, 'value': [9.0, 2.0, 1.0]})}, True, id='objective'),
    pytest.param({'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [80.0, 70.0, 90.0]})}, True, id='bounds'),
    pytest.param({'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [100.0, 60.0, 0.0]})}, False, id='mask'),
]


@pytest.mark.parametrize(('change', 'keeps_the_solver'), RUNGS)
def test_a_rebind_answers_what_a_fresh_build_answers(dispatch_yaml, change, keeps_the_solver):
    """The oracle. Every rung, one assertion: the reference build is the truth.

    Read-back is keyed by coordinate, so this holds even where the rung moved
    every label underneath — which is what makes `rebind` total rather than a
    method that refuses the data it cannot do quickly.
    """
    del keeps_the_solver
    reference = lps.solve(dispatch_yaml, {**sources(), **change}, coords=COORDS)
    with lps.build(dispatch_yaml, sources(), coords=COORDS) as bound:
        bound.solve()
        rebound = bound.rebind(change).solve()

        assert rebound.objective == pytest.approx(reference.objective), 'the rebind reached a different optimum'
        assert rebound.primal('p').equals(reference.primal('p')), 'the rebind laid its values out differently'
        assert rebound.dual('power_balance').equals(reference.dual('power_balance')), (
            'the rebind laid its duals out differently'
        )
    reference.close()


@pytest.mark.parametrize('solver_name', sorted(SOLVERS))
@pytest.mark.parametrize(('change', 'keeps_the_solver'), RUNGS)
def test_only_a_rebind_that_moves_a_label_loads_the_solver_again(dispatch_yaml, change, keeps_the_solver, solver_name):
    """The fast path is taken exactly when the structure held.

    The first solve always loads — there was nothing to keep — so a driver on
    the fast path leaves `diagnostics().reloads` one row long however many times
    round. The rule is the digest's, so it is the same rule for every sink that
    can stay loaded.
    """
    if solver_name == 'gurobi':
        pytest.importorskip('gurobipy', reason='the gurobi sink needs the [gurobi] extra')
    with lps.build(dispatch_yaml, sources(), coords=COORDS) as bound:
        bound.solve(solver_name=solver_name)
        assert bound.diagnostics().reloads.height == 1, 'the first solve has nothing loaded to keep'

        bound.rebind(change).solve(solver_name=solver_name)
        seen = bound.diagnostics()
        expected = 1 if keeps_the_solver else 2
        assert seen.solves == 2, 'both solves are counted whichever path each took'
        assert seen.reloads.height == expected, (
            f'{seen.reloads.to_dicts()} — a rebind that keeps every label pushes values onto '
            f'the loaded solver; one that moves a label has to load the model again'
        )


def test_a_rebind_takes_a_change_at_a_time_and_keeps_the_rest(dispatch_yaml):
    """Partial by construction: what is not named keeps what `build` bound."""
    every = {**sources(), 'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [1.0, 2.0, 3.0, 4.0]})}
    reference = lps.solve(dispatch_yaml, every, coords=COORDS)
    with lps.build(dispatch_yaml, sources(), coords=COORDS) as bound:
        rebound = bound.rebind({'load': every['load']}).solve()
        assert rebound.objective == pytest.approx(reference.objective)
    reference.close()


def test_a_rebind_may_be_repeated_and_each_answer_is_its_own(dispatch_yaml):
    """The loop `rebind` exists for: bind, solve, read, bind again.

    Each iteration's frame is read out before the next rebind, which is the
    discipline a driver keeps anyway — and the reason the values are compared
    here rather than the results.
    """
    with lps.build(dispatch_yaml, sources(), coords=COORDS) as bound:
        served = []
        for scale in (0.5, 1.0, 1.5):
            load = pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [40.0 * scale] * len(SNAPSHOTS)})
            served.append(bound.rebind({'load': load}).solve().primal('p')['value'].sum())

        assert served == sorted(served), f'{served} — more load must dispatch more power'
        assert bound.diagnostics().reloads.height == 1, 'a scaled right-hand side moves no label'


def test_a_result_from_before_a_rebind_refuses_to_read(dispatch_yaml):
    """The one lifetime: a rebind replaces the frames the readers join through.

    Values already read are their own data and stay valid, which is what makes
    "read it out first" a discipline rather than a loss.
    """
    with lps.build(dispatch_yaml, sources(), coords=COORDS) as bound:
        stale = bound.solve()
        kept = stale.primal('p')
        objective = stale.objective

        bound.rebind({'cost': pl.DataFrame({'generator': GENERATORS, 'value': [3.0, 1.0, 2.0]})})

        assert kept.height > 0, 'a frame read before the rebind is its own data'
        assert stale.objective == objective, 'and the outcome needs no model to report'
        for read in (lambda: stale.primal('p'), lambda: stale.dual('power_balance')):
            with pytest.raises(lps.LpspecError, match='the model was rebound'):
                read()


def test_closing_a_stale_result_leaves_the_rebound_model_alone(dispatch_yaml):
    """A result the model outgrew releases its own values and nothing else.

    Otherwise leaving the `with` block of an earlier iteration would take down
    the model the loop is still solving.
    """
    with lps.build(dispatch_yaml, sources(), coords=COORDS) as bound:
        stale = bound.solve()
        bound.rebind({'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [1.0, 2.0, 3.0, 4.0]})})
        stale.close()

        assert bound.solve().primal('p').height > 0, 'the live model is still there to solve'


@pytest.mark.parametrize(
    ('call', 'unknown'),
    [
        pytest.param(lambda bound: bound.rebind({'p_maxx': 1}), 'p_maxx', id='sources'),
        pytest.param(lambda bound: bound.rebind({}, coords={'snapshots': [0]}), 'snapshots', id='coords'),
    ],
)
def test_a_rebind_refuses_a_name_the_model_does_not_declare(dispatch_yaml, call, unknown):
    """A rebind names what changed, so a name nothing reads is the one failure
    a driver cannot see: it re-solves the numbers already bound and reports the
    answer. `build` needs no such check — it binds every declared name or
    fails."""
    with lps.build(dispatch_yaml, sources(), coords=COORDS) as bound, pytest.raises(lps.DataError, match=unknown):
        call(bound)


def test_a_dimension_index_rebinds_as_a_source(dispatch_yaml):
    """A dimension index is a source (SPEC §8), so `rebind` takes it where it
    takes any other — the refusal above is for names the model never declared,
    not for names that happen not to be parameters."""
    with lps.build(dispatch_yaml, sources(), coords=COORDS) as bound:
        change = {'snapshot': [0, 1], 'load': pl.DataFrame({'snapshot': [0, 1], 'value': [5.0, 6.0]})}
        assert bound.rebind(change).solve().primal('p').height > 0, 'a dimension index is a source, and rebinds as one'


def test_a_rebind_can_grow_a_dimension():
    """Appending rows is a rebind — the Benders master, in three lines.

    A cut family is declared once and its members come from data (SPEC §8), so
    an iteration hands over a longer table and the coordinates to match. The
    labels of the rows that were already there do not move, but the model has
    more rows than the solver holds, so it is loaded again.
    """
    master = {
        'dimensions': {'generator': {'values': ['wind', 'gas']}, 'cut': {'dtype': 'int'}},
        'parameters': {
            'invest': {'dims': ['generator']},
            'cut_const': {'dims': ['cut']},
            'cut_slope': {'dims': ['cut', 'generator']},
        },
        'variables': {
            'cap': {'foreach': ['generator'], 'bounds': {'lower': 0, 'upper': 100}},
            'theta': {'foreach': [], 'bounds': {'lower': 0}},
        },
        'constraints': {
            'optimality_cut': {
                'foreach': ['cut'],
                'expression': 'theta >= cut_const + sum(cut_slope * cap, over=generator)',
            }
        },
        'objectives': {'total': {'sense': 'minimize', 'expression': 'cap * invest + theta'}},
    }
    invest = pl.DataFrame({'generator': ['wind', 'gas'], 'value': [90.0, 30.0]})

    def cuts(n: int) -> dict[str, pl.DataFrame]:
        return {
            'cut_const': pl.DataFrame({'cut': list(range(n)), 'value': [500.0 * (i + 1) for i in range(n)]}),
            'cut_slope': pl.DataFrame(
                {
                    'cut': [i for i in range(n) for _ in range(2)],
                    'generator': ['wind', 'gas'] * n,
                    'value': [-5.0 * (i + 1) for i in range(n) for _ in range(2)],
                }
            ),
        }

    reference = lps.solve(master, {'invest': invest, **cuts(3)}, coords={'cut': [0, 1, 2]})
    with lps.build(master, {'invest': invest, **cuts(1)}, coords={'cut': [0]}) as bound:
        bound.solve()
        grown = bound.rebind(cuts(3), coords={'cut': [0, 1, 2]}).solve()

        assert grown.objective == pytest.approx(reference.objective)
        assert grown.primal('cap').equals(reference.primal('cap'))
        assert bound.diagnostics().reloads.height == 2, 'more rows than the solver holds is a load, not a push'
    reference.close()


def test_a_written_file_follows_the_rebind(dispatch_yaml, tmp_path):
    """`write` reads the built model, so it reads the rebound one."""
    with lps.build(dispatch_yaml, sources(), coords=COORDS) as bound:
        bound.write(tmp_path / 'before.lp')
        bound.rebind({'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [7.0, 7.0, 7.0, 7.0]})})
        bound.write(tmp_path / 'after.lp')

    after = (tmp_path / 'after.lp').read_text()
    assert (tmp_path / 'before.lp').read_text() != after
    assert '7' in after


def test_a_rebind_that_cannot_build_leaves_nothing_half_built(dispatch_yaml):
    """The build's own rule, one call later: a failure releases the model.

    A handle holding half a model would answer the next `solve` with a mixture
    of two, which is worse than having nothing to answer with.
    """
    with lps.build(dispatch_yaml, sources(), coords=COORDS) as bound:
        bound.solve()
        with pytest.raises(lps.DataError):
            bound.rebind({'load': pl.DataFrame({'snapshot': [0, 0, 1], 'value': [1.0, 2.0, 3.0]})})

        with pytest.raises(lps.LpspecError, match='no built model to hand over'):
            bound.solve()


def test_diagnostics_report_the_shape_the_solver_was_handed(dispatch_yaml):
    """The size question `check` cannot answer, needing no data where this needs all of it.

    `examples/dispatch.yaml` masks on `p_max > 0`, so the shape is what
    *survived* the mask rather than what the declarations multiply out to —
    which is the whole reason it is read off the built model.
    """
    with lps.build(dispatch_yaml, sources(), coords=COORDS) as bound:
        bound.solve()
        seen = bound.diagnostics()

        assert (seen.columns, seen.rows) == (len(SNAPSHOTS) * len(GENERATORS), len(SNAPSHOTS))
        assert seen.nonzeros == seen.columns, 'one balance row per snapshot, one entry per generator in it'
        assert seen.omissions.is_empty(), 'every declared row reached the solver'

    assert bound.diagnostics().nonzeros == seen.nonzeros, 'a released model still says how big it was'


def test_a_mask_that_removes_a_column_removes_it_from_the_shape(dispatch_yaml):
    """Read off the built model, so a mask that moved moves the counts with it."""
    zeroed = {**sources(), 'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [100.0, 60.0, 0.0]})}
    with lps.build(dispatch_yaml, zeroed, coords=COORDS) as bound:
        assert bound.diagnostics().columns == len(SNAPSHOTS) * (len(GENERATORS) - 1)
