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
from typing import Any, NamedTuple

import polars as pl
import pytest

import lpspec as lps
from lpspec.relational.sinks import SOLVERS
from tests.conftest import bindable_on_this_install, override, port_sources

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
    'objective': {'sense': 'minimize', 'expression': 'p * cost + levy'},
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
    'variables': {'take': {'foreach': ['item'], 'domain': 'binary'}},
    'constraints': {'fits': {'foreach': [], 'expression': 'sum(weight * take, over=item) <= capacity'}},
    'objective': {'sense': 'maximize', 'expression': 'take * worth'},
}


def knapsack_sources() -> dict[str, pl.DataFrame]:
    return {
        'worth': pl.DataFrame({'item': ITEMS, 'value': [float(7 * i % 13 + 1) for i in range(12)]}),
        'weight': pl.DataFrame({'item': ITEMS, 'value': [float(5 * i % 11 + 1) for i in range(12)]}),
        'capacity': pl.DataFrame({'value': [20.0]}),
    }


class Rung(NamedTuple):
    """One row of the rebind table: which model, what changes, what may be kept."""

    model: str
    change: dict[str, pl.DataFrame]
    keeps_the_solver: bool


def _case(rung: Rung, dispatch_yaml: Any) -> tuple[Any, dict[str, pl.DataFrame], dict[str, Any]]:
    """The rung's model, its data and its coordinates."""
    return {
        'dispatch': lambda: (dispatch_yaml, sources(), COORDS),
        'reach': lambda: (REACH, reach_sources(), {}),
        'knapsack': lambda: (KNAPSACK, knapsack_sources(), {}),
    }[rung.model]()


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
        Rung('dispatch', {'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [10.0, 20.0, 30.0, 40.0]})}, True),
        id='rhs',
    ),
    pytest.param(
        Rung('dispatch', {'cost': pl.DataFrame({'generator': GENERATORS, 'value': [9.0, 2.0, 1.0]})}, True),
        id='objective',
    ),
    pytest.param(
        Rung('dispatch', {'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [80.0, 70.0, 90.0]})}, True),
        id='bounds',
    ),
    pytest.param(
        Rung('dispatch', {'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [100.0, 60.0, 0.0]})}, False),
        id='mask',
    ),
    pytest.param(Rung('reach', {'levy': pl.DataFrame({'value': [500.0]})}, True), id='objective constant'),
    pytest.param(
        Rung(
            'reach',
            {'reach': reaching(('north', 'a', 0.5), ('north', 'b', 1.0), ('south', 'c', 1.0), ('south', 'd', 1.0))},
            False,
        ),
        id='a coefficient moved',
    ),
    pytest.param(
        Rung(
            'reach',
            {'reach': reaching(('north', 'c', 1.0), ('north', 'd', 1.0), ('south', 'a', 1.0), ('south', 'b', 1.0))},
            False,
        ),
        id='an entry changed column',
    ),
    pytest.param(
        Rung(
            'reach',
            {'reach': reaching(('north', 'a', 1.0), ('south', 'b', 1.0), ('south', 'c', 1.0), ('south', 'd', 1.0))},
            False,
        ),
        id='an entry changed row',
    ),
    pytest.param(Rung('knapsack', {'capacity': pl.DataFrame({'value': [9.0]})}, True), id='integer'),
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


@pytest.fixture
def bound(dispatch_yaml):
    """The example dispatch on its own data, built and open for the test's duration."""
    with lps.build(dispatch_yaml, sources(), coords=COORDS) as model:
        yield model


def _priced(schema: Any) -> list[str]:
    """The constraints an answer carries prices for — none, where a variable is discrete."""
    if any(v.domain != 'continuous' for v in schema.variables.values()):
        return []
    return list(schema.constraints)


@pytest.mark.parametrize('rung', RUNGS)
def test_a_rebind_answers_what_a_fresh_build_answers(dispatch_yaml, rung, solver_name):
    """The oracle. Every rung, one assertion: the reference build is the truth.

    Read-back is keyed by coordinate, so this holds even where the rung moved
    every label underneath — which is what makes `rebind` total rather than a
    method that refuses the data it cannot do quickly.

    Over **every** declaration rather than a named one, and over **every** sink
    that can stay loaded: each writes its own push, and a field one of them
    forgets is a confident answer to the model before the rebind.
    """
    model, given, coords = _case(rung, dispatch_yaml)
    schema = lps.load_model(model)
    with (
        lps.solve(model, {**given, **rung.change}, solver_name=solver_name, coords=coords) as reference,
        lps.build(model, given, coords=coords) as bound,
    ):
        bound.solve(solver_name=solver_name)
        rebound = bound.rebind(rung.change).solve(solver_name=solver_name)

        assert rebound.objective == pytest.approx(reference.objective), 'the rebind reached a different optimum'
        for name in schema.variables:
            assert rebound.primal(name).equals(reference.primal(name)), f"'{name}' came back laid out differently"
        for name in _priced(schema):
            assert rebound.dual(name).equals(reference.dual(name)), f"'{name}' came back priced differently"


# ---------------------------------------------------------------------------
# the same oracle, over models nobody here wrote
# ---------------------------------------------------------------------------

#: One walk over a port's own data. **1.0 first** pins determinism — two builds
#: of one model have to hash alike or no driver ever takes the fast path, and a
#: `rows` frame that came back in a different order each build was exactly that
#: bug — then two scalings, each rebound off the state the last one left, so no
#: step here is a single hop from the build.
WALK = [1.0, 1.25, 0.8]

#: `tsp_mtz` walks nowhere: gr17's branch-and-bound is seconds a solve and a
#: walk takes three of them per model, which is a third of the suite's runtime
#: for one port. The other three discrete ports reach the same paths in a tenth
#: of the time.
TOO_SLOW_TO_WALK = {'tsp_mtz'}

#: `transport_modes` prices two of its eleven connections at 12 — `d1_c1_road`
#: and `d2_c2_rail` — so once the walk scales the stocks the optimum is reached
#: at more than one vertex, and which one a solve lands on is a simplex route
#: rather than an answer. The objective is compared as before; only the primal
#: is not, because there is no single right one to compare against. Book data,
#: so the tie is the source's and not ours to perturb away.
ALTERNATE_OPTIMA = {'transport_modes'}

#: Constraints whose prices the walk compares for layout but not for numbers.
#: An investment optimum sits on a kink of the piecewise-linear value of
#: capacity — the marginal MW is worth more than capex on one side of a load
#: level and less on the other — so how the capacity rent splits across the
#: snapshots binding there is a free dual ray, and a warm-started re-solve
#: legitimately lands on a different split than a cold one. The objective, the
#: layouts and every other price (`balance`, whose degeneracy would move the
#: objective) stay exact.
NONUNIQUE_PRICES: dict[str, set[str]] = {'multi_period': {'within_cap'}}


def _declared(given: dict[str, Any], schema: Any) -> dict[str, Any]:
    """*given* less the names the model never declares.

    `build` binds what it recognises and ignores the rest; `rebind` refuses a
    name it does not know, deliberately, a typo there being a silent re-solve.
    So the two doors disagree about one mapping — `pypsa_kvl`'s data carries a
    `reactance` its model reads through `cycle_incidence` instead — and this is
    what hands both of them the same thing.
    """
    known = {**schema.parameters, **schema.dimensions}
    return {name: value for name, value in given.items() if name in known}


def _scaled(given: dict[str, Any], by: float) -> dict[str, Any]:
    """*given* with every dimensioned real-valued table scaled, and nothing else.

    Dimensioned, because a **scalar** in this corpus is as often a formulation
    constant as a quantity — `tsp_mtz`'s ``n`` is the Miller-Tucker-Zemlin
    bound — and scaling one of those writes a different *model* rather than
    different numbers for the same one. Real-valued for the same reason: an
    integer index or a label is structure.
    """
    scalable = pl.col('value') * by
    return {
        name: value.with_columns(scalable)
        if isinstance(value, pl.DataFrame) and value.schema.get('value') == pl.Float64 and len(value.columns) > 1
        else value
        for name, value in given.items()
    }


def _prices(result: Any, schema: Any) -> dict[str, pl.DataFrame] | None:
    """Every constraint's prices, or ``None`` where this answer carries none.

    Asked of the answer rather than read off the declarations: what leaves
    duals undefined is the solve's business, and `Result.dual` is where it is
    already decided.
    """
    try:
        return {name: result.dual(name) for name in schema.constraints}
    except lps.LpspecError:
        return None


def _laid_out_alike(got: pl.DataFrame, want: pl.DataFrame, *, values: bool, where: str) -> None:
    """*got* is *want*'s frame: same coordinates in the same order, same numbers."""
    assert got.drop('value').equals(want.drop('value')), f'{where}: came back keyed or ordered differently'
    if values:
        assert got['value'].to_list() == pytest.approx(want['value'].to_list()), f'{where}: different numbers'


def test_a_rebind_walk_answers_what_a_fresh_build_answers(port):
    """The oracle again, over ported models, three rebinds deep.

    `build` + `solve` is always available as the reference, so breadth costs
    only the models — and there are ten here that nobody wrote to be a test:
    networks, storage, ramping, unit commitment, a diet, a facility location,
    two transports. The rungs above are models built to move one field of the
    digest; these are models built to be models, walked through three data
    states with the answer checked against a fresh build at every one.

    What is compared is what a rebind may be held to:

    - **The objective and the layout, always.** A read-back that sliced the
      solver's vector wrongly puts the right numbers on the wrong coordinates,
      and a corpus this wide is what finds it.
    - **The numbers, where the answer carries prices** — which is to say where
      the model is continuous. A discrete model's optimum is not unique, so a
      rebound `tsp_mtz` reaches a different tour of the same length; that is
      branch-and-bound's answer and not a rebind's mistake. On the continuous
      ports the two agree to 1e-14 on both sinks, which is a different simplex
      route rather than a different vertex — so `approx` rather than `equals`,
      and the exact form stays on the rungs above, whose optima are unique by
      construction.

    On `highs` alone, the default. What a second sink pushes differently is the
    rungs' question; this one is about the models.
    """
    bindable_on_this_install(port['name'])
    if port['name'] in TOO_SLOW_TO_WALK:
        pytest.skip(f'{port["name"]} is too slow to walk — see TOO_SLOW_TO_WALK')

    schema = lps.load_model(port['model'])
    given = _declared(port_sources(port['name']), schema)

    with lps.build(port['model'], given) as bound:
        bound.solve()
        for step, factor in enumerate(WALK):
            change = _scaled(given, factor)
            where = f'{port["name"]} x{factor}'
            with lps.solve(port['model'], change) as reference:
                got = bound.rebind(change).solve()

                assert got.termination_condition == reference.termination_condition, f'{where}: terminated differently'
                assert got.has_primal == reference.has_primal, f'{where}: one left values and the other did not'
                if reference.has_primal:
                    assert got.objective == pytest.approx(reference.objective), f'{where}: a different optimum'
                    wanted = _prices(reference, schema)
                    assert (_prices(got, schema) is None) == (wanted is None), f'{where}: one is priced and one is not'
                    unique = port['name'] not in ALTERNATE_OPTIMA
                    for name in schema.variables:
                        _laid_out_alike(
                            got.primal(name),
                            reference.primal(name),
                            values=wanted is not None and unique,
                            where=f'{where} {name}',
                        )
                    for name in wanted or {}:
                        exact = name not in NONUNIQUE_PRICES.get(port['name'], set())
                        _laid_out_alike(got.dual(name), wanted[name], values=exact, where=f'{where} {name} price')

            if not step:
                assert bound.diagnostics().loads == 1, (
                    'the same numbers rebound have to hash alike, or no driver ever takes the fast path'
                )


@pytest.mark.parametrize('rung', RUNGS)
def test_only_a_rebind_that_moves_a_label_loads_the_solver_again(dispatch_yaml, rung, solver_name):
    """The fast path is taken exactly when the structure held.

    The first solve always loads — there was nothing to keep — so a driver on
    the fast path leaves `diagnostics().loads` at one however many times round.
    The rule is the digest's, so it is the same rule for every sink that can
    stay loaded.
    """
    model, given, coords = _case(rung, dispatch_yaml)
    with lps.build(model, given, coords=coords) as bound:
        bound.solve(solver_name=solver_name)
        assert bound.diagnostics().loads == 1, 'the first solve has nothing loaded to keep'

        bound.rebind(rung.change).solve(solver_name=solver_name)
        seen = bound.diagnostics()
        expected = 1 if rung.keeps_the_solver else 2
        assert seen.solves == 2, 'both solves are counted whichever path each took'
        assert seen.loads == expected, (
            'a rebind that keeps every label pushes values onto the loaded solver; '
            'one that moves a label has to load the model again'
        )


def _tables(model: Any) -> Any:
    """*model*'s solver tables, read off it built on the reach data."""
    with lps.build(model, reach_sources()) as built:
        return built._engine._tables()


#: The three fields of the digest **no rung above can reach**: a variable's
#: type, a row's comparison and the objective's direction all come from the
#: YAML, so no change to data moves one. They are hashed anyway — the digest's
#: soundness must not rest on reasoning about which of a model's facts the
#: language lets data touch — so they are pinned where they *are* reachable,
#: one edit of the declaration apart.
DECLARED = [
    pytest.param({'variables.p.domain': 'integer'}, id='a variable type'),
    pytest.param({'constraints.meet.expression': 'sum(reach * p, over=plant) == demand'}, id="a row's comparison"),
    pytest.param({'objective.sense': 'maximize'}, id='the sense'),
]


@pytest.mark.parametrize('edited', DECLARED)
def test_the_digest_moves_where_a_declaration_moved(edited):
    """What a re-solve may not change, checked directly for want of a rung.

    A solver keeps the model it holds when the digest matches, so a field left
    out of it is a push onto a model that is no longer the one being asked
    about — and every such answer is confident. Two builds one declaration
    apart is the only way to ask about a field the data cannot move.
    """
    assert _tables(REACH).structure != _tables(override(REACH, **edited)).structure, (
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
    tables = _tables(REACH)
    moved = replace(tables, **{count: getattr(tables, count) + 1})
    assert moved.structure != tables.structure, f'{count} is framing, not decoration: the same bytes split elsewhere'


#: The option name each sink gives a time limit — `solver_options` is forwarded
#: verbatim, so the vocabulary is the solver's own and there is one word per
#: member rather than one shared word.
LIMITS = {'highs': 'time_limit', 'gurobi': 'TimeLimit'}


def test_a_solve_asking_for_other_options_loads_the_model_again(bound, solver_name):
    """Options are recorded at the load, so they are part of what may be kept.

    A solver holds what it was told when it took the model. Keeping it for a
    solve that asked for others would run that solve under the *first* one's
    limits and report the answer as the one asked for — a gap left loose, or a
    time limit that was never the caller's.
    """
    option = LIMITS[solver_name]
    bound.solve(solver_name=solver_name, solver_options={option: 60.0})
    bound.solve(solver_name=solver_name, solver_options={option: 120.0})
    assert bound.diagnostics().loads == 2, 'a solver told the first options cannot be told others'

    bound.solve(solver_name=solver_name, solver_options={option: 120.0})
    assert bound.diagnostics().loads == 2, 'the same options ask for the model the solver already holds'


def test_a_rebind_takes_a_change_at_a_time_and_keeps_the_rest(dispatch_yaml, bound):
    """Partial by construction: what is not named keeps what `build` bound."""
    every = {**sources(), 'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [1.0, 2.0, 3.0, 4.0]})}
    with lps.solve(dispatch_yaml, every, coords=COORDS) as reference:
        rebound = bound.rebind({'load': every['load']}).solve()
        assert rebound.objective == pytest.approx(reference.objective)


def test_a_rebind_may_be_repeated_and_each_answer_is_its_own(bound):
    """The loop `rebind` exists for: bind, solve, read, bind again."""
    served = []
    for scale in (0.5, 1.0, 1.5):
        load = pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [40.0 * scale] * len(SNAPSHOTS)})
        served.append(bound.rebind({'load': load}).solve().primal('p')['value'].sum())

    assert served == sorted(served), f'{served} — more load must dispatch more power'
    assert bound.diagnostics().loads == 1, 'a scaled right-hand side moves no label'


def test_a_result_from_before_a_rebind_keeps_reading(bound):
    """A result owns its read-back, so nothing done to the model expires it.

    The label frames are immutable and shared: a rebind builds new ones
    without touching what earlier results hold, so an old answer stays an
    answer over its own coordinates — a driver keeps any result it still
    wants, at the price of keeping that build's label frames alive.
    """
    before = bound.solve()
    kept = before.primal('p')
    prices = before.dual('power_balance')

    after = bound.rebind({'cost': pl.DataFrame({'generator': GENERATORS, 'value': [3.0, 1.0, 2.0]})}).solve()

    assert after.objective != before.objective, 'reordered costs move the optimum, so the two answers differ'
    assert before.primal('p').equals(kept), 'the old result still reads, and reads its own build'
    assert before.dual('power_balance').equals(prices), 'its duals too'


def test_closing_a_result_never_touches_the_model(bound):
    """`close` releases what the result holds — its values and its hold on the
    label frames — never the model or the solver, which are the handle's to
    close. So a result closed on the way out of a `with` block cannot take
    down the model a loop is still solving, and a sibling result, holding its
    own read-back, keeps reading.
    """
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
def test_a_rebind_refuses_a_name_the_model_does_not_declare(bound, call, unknown):
    """A rebind names what changed, so a name nothing reads is the one failure
    a driver cannot see: it re-solves the numbers already bound and reports the
    answer. `build` needs no such check — it binds every declared name or
    fails."""
    with pytest.raises(lps.DataError, match=unknown):
        call(bound)


def test_a_dimension_index_rebinds_as_a_source(bound):
    """A dimension index is a source (SPEC §8), so `rebind` takes it where it
    takes any other — the refusal above is for names the model never declared,
    not for names that happen not to be parameters."""
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
        'objective': {'sense': 'minimize', 'expression': 'cap * invest + theta'},
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

    with (
        lps.solve(master, {'invest': invest, **cuts(3)}, coords={'cut': [0, 1, 2]}) as reference,
        lps.build(master, {'invest': invest, **cuts(1)}, coords={'cut': [0]}) as bound,
    ):
        bound.solve()
        grown = bound.rebind(cuts(3), coords={'cut': [0, 1, 2]}).solve()

        assert grown.objective == pytest.approx(reference.objective)
        assert grown.primal('cap').equals(reference.primal('cap'))
        assert bound.diagnostics().loads == 2, 'more rows than the solver holds is a load, not a push'


def test_a_written_file_follows_the_rebind(bound, tmp_path):
    """`write` reads the built model, so it reads the rebound one."""
    bound.write(tmp_path / 'before.lp')
    bound.rebind({'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [7.0, 7.0, 7.0, 7.0]})})
    bound.write(tmp_path / 'after.lp')

    after = (tmp_path / 'after.lp').read_text()
    assert (tmp_path / 'before.lp').read_text() != after
    assert '7' in after


def test_a_rebind_that_cannot_build_leaves_nothing_half_built(bound):
    """The build's own rule, one call later: a failure releases the model.

    A handle holding half a model would answer the next `solve` with a mixture
    of two, which is worse than having nothing to answer with.
    """
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


def test_a_cost_falling_to_zero_shrinks_the_objective_and_keeps_the_solver():
    """The objective frame may change height across a rebind. The solver may not.

    A zero cost is pruned, so `obj` holds one row fewer than before — while
    `structure` deliberately does not read `obj`, costs being pushable. The
    column is still *there*; it is `dense_columns` that puts the zero back,
    scattering the sparse frame over the solver's full index. A push that read
    `obj` positionally instead would hand the solver one plant's cost under
    another's name, and every answer after it would be confidently wrong.
    """
    given = reach_sources()
    with lps.build(REACH, given) as bound:
        bound.solve()
        before = bound._engine._obj.height
        assert bound.diagnostics().loads == 1, 'the first solve has nothing loaded to keep'

        zeroed = pl.DataFrame({'plant': PLANTS, 'value': [0.0, 2.0, 3.0, 4.0]})
        rebound = bound.rebind({'cost': zeroed}).solve()

        assert bound._engine._obj.height == before - 1, 'the zero cost should have left the objective frame'
        assert bound.diagnostics().loads == 1, 'a cost is pushed, so a cost falling to zero may not reload'

    with lps.build(REACH, {**given, 'cost': zeroed}) as fresh:
        assert rebound.objective == fresh.solve().objective, (
            'the pushed cost vector disagrees with the one a cold build hands over'
        )
