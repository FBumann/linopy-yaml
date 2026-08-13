"""Re-solving one built model with new numbers.

Two claims, and the first is the whole contract: **a rebind answers what a
fresh build answers**. `build(model, sources | change)` is always available as
the reference, so every rung below is checked against it rather than against a
number someone wrote down — the same oracle shape as the two-lane differential
and the Benders monolith check.

The second is that the fast path is *only* a fast path. A rebind that moves a
mask renumbers labels and cannot be pushed onto a loaded solver, so the engine
rebuilds and solves cold; nothing about the answer changes, and
`diagnostics().loads` is where a driver finds out which happened.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

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


#: Which plants may serve which zone, and how well. Every matrix entry of
#: `examples/dispatch.yaml` is a 1 — its only constraint is `sum(p) == load` —
#: and its objective has no constant, so no change to its data can move a
#: coefficient, move one to another column, move one to another row, or move
#: the term that has no column at all. Those are the four things a rebind can
#: move that the example cannot say, and this is the model that says them.
ZONES = ['north', 'south']
PLANTS = ['a', 'b', 'c', 'd']
REACH = {
    'dimensions': {'zone': {'values': ZONES}, 'plant': {'values': PLANTS}},
    'parameters': {
        'reach': {'dims': ['zone', 'plant']},
        'cost': {'dims': ['plant']},
        'demand': {'dims': ['zone']},
        'levy': {'dims': []},
    },
    'variables': {'p': {'foreach': ['plant'], 'bounds': {'lower': 0, 'upper': 100}}},
    'constraints': {'meet': {'foreach': ['zone'], 'expression': 'sum(reach * p, over=plant) >= demand'}},
    #: `levy` is the objective's **constant** — the one term with no column, so
    #: it reaches a solver by neither of the two routes the others take.
    'objectives': {'total': {'sense': 'minimize', 'expression': 'p * cost + levy'}},
}


def reaching(*served: tuple[str, str, float]) -> pl.DataFrame:
    """A `reach` frame. An absent row is a zero coefficient (SPEC §8), so it drops the entry."""
    return pl.DataFrame(
        {'zone': [z for z, _, _ in served], 'plant': [p for _, p, _ in served], 'value': [v for _, _, v in served]},
        schema={'zone': pl.String, 'plant': pl.String, 'value': pl.Float64},
    )


def reach_sources() -> dict[str, pl.DataFrame]:
    """North reached by the two cheap plants, south by the two dear ones."""
    return {
        'reach': reaching(('north', 'a', 1.0), ('north', 'b', 1.0), ('south', 'c', 1.0), ('south', 'd', 1.0)),
        'cost': pl.DataFrame({'plant': PLANTS, 'value': [1.0, 2.0, 3.0, 4.0]}),
        'demand': pl.DataFrame({'zone': ZONES, 'value': [60.0, 30.0]}),
        'levy': pl.DataFrame({'value': [5.0]}),
    }


#: A knapsack, because nothing above declares a discrete variable — and a
#: rebound mixed-integer model re-solves on a solver still holding the last
#: solve's incumbent.
ITEMS = [f'item{i}' for i in range(12)]
KNAPSACK = {
    'dimensions': {'item': {'values': ITEMS}},
    'parameters': {'worth': {'dims': ['item']}, 'weight': {'dims': ['item']}, 'capacity': {'dims': []}},
    'variables': {'take': {'foreach': ['item'], 'binary': True}},
    'constraints': {'fits': {'foreach': [], 'expression': 'sum(weight * take, over=item) <= capacity'}},
    'objectives': {'total': {'sense': 'maximize', 'expression': 'take * worth'}},
}


def knapsack_sources() -> dict[str, pl.DataFrame]:
    return {
        'worth': pl.DataFrame({'item': ITEMS, 'value': [float(7 * i % 13 + 1) for i in range(12)]}),
        'weight': pl.DataFrame({'item': ITEMS, 'value': [float(5 * i % 11 + 1) for i in range(12)]}),
        'capacity': pl.DataFrame({'value': [20.0]}),
    }


def _dispatch(yaml: Any) -> tuple[Any, dict[str, pl.DataFrame], dict[str, Any]]:
    """The example, its data and its coordinates — what the table's own rungs move."""
    return yaml, sources(), COORDS


def _reach(yaml: Any) -> tuple[Any, dict[str, pl.DataFrame], dict[str, Any]]:
    del yaml
    return REACH, reach_sources(), {}


def _knapsack(yaml: Any) -> tuple[Any, dict[str, pl.DataFrame], dict[str, Any]]:
    del yaml
    return KNAPSACK, knapsack_sources(), {}


#: Each rung of the rebind table (docs/api.md): the model it moves, what
#: changes, and whether the loaded solver may be kept. `p_max` appears twice on
#: purpose: it gates ``where: p_max > 0`` *and* bounds the variable, so whether
#: it is structural is a property of the values and not of where the name
#: appears.
#:
#: The four `reach` rungs move **one field of the digest each**, which is what
#: earns them a model of their own: a rung that moved two would still pass with
#: either one dropped. Each changes the answer, so a solver wrongly kept
#: reports a wrong number rather than a lucky one.
RUNGS = [
    pytest.param(
        _dispatch, {'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [10.0, 20.0, 30.0, 40.0]})}, True, id='rhs'
    ),
    pytest.param(
        _dispatch, {'cost': pl.DataFrame({'generator': GENERATORS, 'value': [9.0, 2.0, 1.0]})}, True, id='objective'
    ),
    pytest.param(
        _dispatch, {'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [80.0, 70.0, 90.0]})}, True, id='bounds'
    ),
    pytest.param(
        _dispatch, {'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [100.0, 60.0, 0.0]})}, False, id='mask'
    ),
    pytest.param(_reach, {'levy': pl.DataFrame({'value': [500.0]})}, True, id='objective constant'),
    pytest.param(
        _reach,
        {'reach': reaching(('north', 'a', 0.5), ('north', 'b', 1.0), ('south', 'c', 1.0), ('south', 'd', 1.0))},
        False,
        id='a coefficient moved',
    ),
    pytest.param(
        _reach,
        {'reach': reaching(('north', 'c', 1.0), ('north', 'd', 1.0), ('south', 'a', 1.0), ('south', 'b', 1.0))},
        False,
        id='an entry changed column',
    ),
    pytest.param(
        _reach,
        {'reach': reaching(('north', 'a', 1.0), ('south', 'b', 1.0), ('south', 'c', 1.0), ('south', 'd', 1.0))},
        False,
        id='an entry changed row',
    ),
    pytest.param(_knapsack, {'capacity': pl.DataFrame({'value': [9.0]})}, True, id='integer'),
]


@pytest.fixture(params=sorted(SOLVERS))
def solver_name(request: pytest.FixtureRequest) -> str:
    """Every sink that can stay loaded, skipping one this build cannot run.

    Asked through the sink's own availability rule rather than by naming its
    package here, so a member that grows a second dependency does not also grow
    a second skip.
    """
    if not SOLVERS[request.param].is_available():
        pytest.skip(f'{request.param} is not installed here')
    return str(request.param)


def _priced(schema: Any) -> list[str]:
    """The constraints an answer carries prices for — none, where a variable is discrete."""
    if any(v.binary or v.integer for v in schema.variables.values()):
        return []
    return list(schema.constraints)


@pytest.mark.parametrize(('case', 'change', 'keeps_the_solver'), RUNGS)
def test_a_rebind_answers_what_a_fresh_build_answers(dispatch_yaml, case, change, keeps_the_solver, solver_name):
    """The oracle. Every rung, one assertion: the reference build is the truth.

    Read-back is keyed by coordinate, so this holds even where the rung moved
    every label underneath — which is what makes `rebind` total rather than a
    method that refuses the data it cannot do quickly.

    Over **every** declaration rather than a named one, and over **every** sink
    that can stay loaded: each writes its own push, and a field one of them
    forgets is a confident answer to the model before the rebind.
    """
    del keeps_the_solver
    model, given, coords = case(dispatch_yaml)
    schema = lps.load_model(model)
    reference = lps.solve(model, {**given, **change}, solver_name=solver_name, coords=coords)
    with lps.build(model, given, coords=coords) as bound:
        bound.solve(solver_name=solver_name)
        rebound = bound.rebind(change).solve(solver_name=solver_name)

        assert rebound.objective == pytest.approx(reference.objective), 'the rebind reached a different optimum'
        for name in schema.variables:
            assert rebound.primal(name).equals(reference.primal(name)), f"'{name}' came back laid out differently"
        for name in _priced(schema):
            assert rebound.dual(name).equals(reference.dual(name)), f"'{name}' came back priced differently"
    reference.close()


@pytest.mark.parametrize(('case', 'change', 'keeps_the_solver'), RUNGS)
def test_only_a_rebind_that_moves_a_label_loads_the_solver_again(
    dispatch_yaml, case, change, keeps_the_solver, solver_name
):
    """The fast path is taken exactly when the structure held.

    The first solve always loads — there was nothing to keep — so a driver on
    the fast path leaves `diagnostics().loads` at one however many times round.
    The rule is the digest's, so it is the same rule for every sink that can
    stay loaded.
    """
    model, given, coords = case(dispatch_yaml)
    with lps.build(model, given, coords=coords) as bound:
        bound.solve(solver_name=solver_name)
        assert bound.diagnostics().loads == 1, 'the first solve has nothing loaded to keep'

        bound.rebind(change).solve(solver_name=solver_name)
        seen = bound.diagnostics()
        expected = 1 if keeps_the_solver else 2
        assert seen.solves == 2, 'both solves are counted whichever path each took'
        assert seen.loads == expected, (
            'a rebind that keeps every label pushes values onto the loaded solver; '
            'one that moves a label has to load the model again'
        )


def _structure(model: Any) -> bytes:
    """*model*'s digest, read off it built on the same data."""
    with lps.build(model, reach_sources()) as bound:
        return bound._engine._tables().structure


#: The three fields of the digest **no rung above can reach**: a variable's
#: type, a row's comparison and the objective's direction all come from the
#: YAML, so no change to data moves one. They are hashed anyway — the digest's
#: soundness must not rest on reasoning about which of a model's facts the
#: language lets data touch — so they are pinned where they *are* reachable,
#: one edit of the declaration apart. Each replaces its whole block.
DECLARED = [
    pytest.param(
        {'variables': {'p': {'foreach': ['plant'], 'bounds': {'lower': 0, 'upper': 100}, 'integer': True}}},
        id='a variable type',
    ),
    pytest.param(
        {'constraints': {'meet': {'foreach': ['zone'], 'expression': 'sum(reach * p, over=plant) == demand'}}},
        id="a row's comparison",
    ),
    pytest.param({'objectives': {'total': {'sense': 'maximize', 'expression': 'p * cost + levy'}}}, id='the sense'),
]


@pytest.mark.parametrize('edited', DECLARED)
def test_the_digest_moves_where_a_declaration_moved(edited):
    """What a re-solve may not change, checked directly for want of a rung.

    A solver keeps the model it holds when the digest matches, so a field left
    out of it is a push onto a model that is no longer the one being asked
    about — and every such answer is confident. Two builds one declaration
    apart is the only way to ask about a field the data cannot move.
    """
    assert _structure(REACH) != _structure({**REACH, **edited}), (
        'a model a re-solve may not be pushed onto has to hash differently'
    )


#: The counts, which no edit of a declaration reaches either — and which no
#: *vector* stands in for, the reason below.
COUNTS = [pytest.param('column_count', id='the column count'), pytest.param('row_count', id='the row count')]


@pytest.mark.parametrize('count', COUNTS)
def test_the_digest_reads_the_counts_that_frame_its_vectors(count):
    """The counts say where one hashed vector ends and the next begins.

    Every vector goes in as raw bytes, one after another and with nothing
    between them, so the concatenation alone does not say how it was split: a
    model with one column more and one row fewer offers the digest the same
    bytes in the same order. It is the counts that make the stream mean one
    model, which is why they are not the redundant restatement of five vector
    lengths they look like.

    Asked of the tables rather than of two builds, since a build produces the
    counts and the vectors together and so cannot pose the question.
    """
    with lps.build(REACH, reach_sources()) as bound:
        tables = bound._engine._tables()

    moved = replace(tables, **{count: getattr(tables, count) + 1})
    assert moved.structure != tables.structure, f'{count} is framing, not decoration: the same bytes split elsewhere'


#: One option each sink understands, at two values. The vocabulary is the
#: solver's own — `solver_options` is forwarded verbatim — so there is one word
#: per member rather than one shared word.
LIMITS = {'highs': ('time_limit', 60.0, 120.0), 'gurobi': ('TimeLimit', 60.0, 120.0)}


def test_a_solve_asking_for_other_options_loads_the_model_again(dispatch_yaml, solver_name):
    """Options are recorded at the load, so they are part of what may be kept.

    A solver holds what it was told when it took the model. Keeping it for a
    solve that asked for others would run that solve under the *first* one's
    limits and report the answer as the one asked for — a gap left loose, or a
    time limit that was never the caller's.
    """
    option, first, second = LIMITS[solver_name]
    with lps.build(dispatch_yaml, sources(), coords=COORDS) as bound:
        bound.solve(solver_name=solver_name, solver_options={option: first})
        bound.solve(solver_name=solver_name, solver_options={option: second})
        assert bound.diagnostics().loads == 2, 'a solver told the first options cannot be told others'

        bound.solve(solver_name=solver_name, solver_options={option: second})
        assert bound.diagnostics().loads == 2, 'the same options ask for the model the solver already holds'


def test_a_rebind_takes_a_change_at_a_time_and_keeps_the_rest(dispatch_yaml):
    """Partial by construction: what is not named keeps what `build` bound."""
    every = {**sources(), 'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [1.0, 2.0, 3.0, 4.0]})}
    reference = lps.solve(dispatch_yaml, every, coords=COORDS)
    with lps.build(dispatch_yaml, sources(), coords=COORDS) as bound:
        rebound = bound.rebind({'load': every['load']}).solve()
        assert rebound.objective == pytest.approx(reference.objective)
    reference.close()


def test_a_rebind_may_be_repeated_and_each_answer_is_its_own(dispatch_yaml):
    """The loop `rebind` exists for: bind, solve, read, bind again."""
    with lps.build(dispatch_yaml, sources(), coords=COORDS) as bound:
        served = []
        for scale in (0.5, 1.0, 1.5):
            load = pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [40.0 * scale] * len(SNAPSHOTS)})
            served.append(bound.rebind({'load': load}).solve().primal('p')['value'].sum())

        assert served == sorted(served), f'{served} — more load must dispatch more power'
        assert bound.diagnostics().loads == 1, 'a scaled right-hand side moves no label'


def test_a_result_from_before_a_rebind_keeps_reading(dispatch_yaml):
    """A result owns its read-back, so nothing done to the model expires it.

    The label frames are immutable and shared: a rebind builds new ones
    without touching what earlier results hold, so an old answer stays an
    answer over its own coordinates — a driver keeps any result it still
    wants, at the price of keeping that build's label frames alive.
    """
    with lps.build(dispatch_yaml, sources(), coords=COORDS) as bound:
        before = bound.solve()
        kept = before.primal('p')
        prices = before.dual('power_balance')

        after = bound.rebind({'cost': pl.DataFrame({'generator': GENERATORS, 'value': [3.0, 1.0, 2.0]})}).solve()

        assert after.objective != before.objective, 'reordered costs move the optimum, so the two answers differ'
        assert before.primal('p').equals(kept), 'the old result still reads, and reads its own build'
        assert before.dual('power_balance').equals(prices), 'its duals too'


def test_closing_a_result_never_touches_the_model(dispatch_yaml):
    """`close` releases what the result holds — its values and its hold on the
    label frames — never the model or the solver, which are the handle's to
    close. So a result closed on the way out of a `with` block cannot take
    down the model a loop is still solving, and a sibling result, holding its
    own read-back, keeps reading.
    """
    with lps.build(dispatch_yaml, sources(), coords=COORDS) as bound:
        first = bound.solve()
        sibling = bound.solve()
        first.close()

        assert sibling.primal('p').height > 0, 'a sibling holds its own read-back'
        assert bound.solve().primal('p').height > 0, 'the model is still there to solve'


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
        assert bound.diagnostics().loads == 2, 'more rows than the solver holds is a load, not a push'
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
