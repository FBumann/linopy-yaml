"""How a solve starts: the session's warmth, and how to refuse it.

Two levels, and the split is the point.

**The public half** is `solve(warm=...)` and `result.started`. A kept solver
re-solves from wherever its last solve left it (`'session'`); `warm=False`
refuses that **structurally** — the held solver is discarded, so the fresh one
has nothing to start from whatever a member squirrels away — and the answer
says which happened. The observable that makes "cold" more than a claim is the
iteration count: a deliberately cold solve repeats the first solve exactly.

**The sink half** is `Solver.warm_start()` / `warm()`, machinery with no
caller above the family yet (#382). It is tested here because it exists: a
carried basis starts the simplex at the optimum — zero iterations against the
cold solve's hundreds — and every refusal it can check is checked. The reason
it stays below the surface is the refusal in
`test_a_warm_start_for_a_differently_shaped_model_is_refused`: the case that
wants a carry most, a cutting-plane master re-solved after gaining a cut, is a
model that gained a row, and a basis spans the model it was read from.

Iteration counts are deterministic, so none of this needs an idle box.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import polars as pl
import pytest

import charter as lps
from charter.relational.sinks import SOLVERS

# ---------------------------------------------------------------------------
# models: an LP big enough to make the simplex work, and a MIP
# ---------------------------------------------------------------------------

SNAPSHOTS = list(range(40))
GENERATORS = [f'g{i}' for i in range(6)]
DISPATCH = {
    'dimensions': {'snapshot': {'dtype': 'int'}, 'generator': {'values': GENERATORS}},
    'parameters': {
        'p_max': {'dims': ['generator']},
        'cost': {'dims': ['snapshot', 'generator']},
        'load': {'dims': ['snapshot']},
    },
    'variables': {'p': {'foreach': ['snapshot', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}}},
    'constraints': {'balance': {'foreach': ['snapshot'], 'expression': 'sum(p, over=generator) == load'}},
    'objective': {'sense': 'minimize', 'expression': 'p * cost'},
}


def dispatch_sources(snapshots: list[int] = SNAPSHOTS, generators: list[str] = GENERATORS) -> dict[str, pl.DataFrame]:
    """Deterministic data whose per-snapshot costs keep the simplex busy."""
    rng = np.random.default_rng(7)
    return {
        'p_max': pl.DataFrame({'generator': generators, 'value': rng.uniform(50.0, 150.0, len(generators))}),
        'cost': pl.DataFrame(
            {
                'snapshot': [s for s in snapshots for _ in generators],
                'generator': generators * len(snapshots),
                'value': rng.uniform(1.0, 50.0, len(snapshots) * len(generators)),
            }
        ),
        'load': pl.DataFrame({'snapshot': snapshots, 'value': rng.uniform(60.0, 250.0, len(snapshots))}),
    }


#: The same rows over fewer columns — the column-span refusal's case.
DISPATCH_NARROW = {
    **DISPATCH,
    'dimensions': {'snapshot': {'dtype': 'int'}, 'generator': {'values': GENERATORS[:5]}},
}

#: The same columns under more rows — what makes the row-span refusal its own
#: case rather than the column refusal arriving first.
DISPATCH_CAPPED = {
    **DISPATCH,
    'parameters': {**DISPATCH['parameters'], 'cap': {'dims': ['generator']}},
    'constraints': {
        **DISPATCH['constraints'],
        'capped': {'foreach': ['generator'], 'expression': 'sum(p, over=snapshot) <= cap'},
    },
}


def capped_sources() -> dict[str, pl.DataFrame]:
    return {
        **dispatch_sources(),
        'cap': pl.DataFrame({'generator': GENERATORS, 'value': [4000.0] * len(GENERATORS)}),
    }


ITEMS = [f'item{i}' for i in range(12)]
KNAPSACK = {
    'dimensions': {'item': {'values': ITEMS}},
    'parameters': {'worth': {'dims': ['item']}, 'weight': {'dims': ['item']}, 'capacity': {'dims': []}},
    'variables': {'take': {'foreach': ['item'], 'domain': 'binary'}},
    'constraints': {'fits': {'foreach': [], 'expression': 'sum(weight * take, over=item) <= capacity'}},
    'objective': {'sense': 'maximize', 'expression': 'take * worth'},
}


def knapsack_sources(items: list[str] = ITEMS) -> dict[str, pl.DataFrame]:
    return {
        'worth': pl.DataFrame({'item': items, 'value': [float(7 * i % 13 + 1) for i in range(len(items))]}),
        'weight': pl.DataFrame({'item': items, 'value': [float(5 * i % 11 + 1) for i in range(len(items))]}),
        'capacity': pl.DataFrame({'value': [20.0]}),
    }


@pytest.fixture(params=sorted(SOLVERS))
def solver_name(request: pytest.FixtureRequest) -> str:
    """Every sink that can stay loaded, skipping one this build cannot run."""
    if not SOLVERS[request.param].is_available():
        pytest.skip(f'{request.param} is not installed here')
    return str(request.param)


def _tables(model: dict[str, Any], given: dict[str, Any], coords: dict[str, Any]) -> Any:
    """*model*'s solver tables, read off it built on *given*."""
    with lps.build(model, given, coords=coords) as built:
        return built._engine._tables()


#: Each member's own iteration counter — the noise-free observable of warmth.
SIMPLEX_ITERATIONS = {
    'highs': lambda solver: int(solver._handle.getInfo().simplex_iteration_count),
    'gurobi': lambda solver: int(solver._m.IterCount),
}


# ---------------------------------------------------------------------------
# the key claim: a carried basis is warm where a fresh session is cold
# ---------------------------------------------------------------------------


def test_a_carried_basis_starts_at_the_optimum_not_from_scratch(solver_name):
    """The same tables in a fresh session: cold works, warm starts done.

    Sink-level on purpose — the iteration counters are the members' own, so
    this is where warmth is observable rather than inferred from timing.
    """
    tables = _tables(DISPATCH, dispatch_sources(), {'snapshot': SNAPSHOTS})
    member = SOLVERS[solver_name]
    first = member(tables)
    try:
        first.run(tables)
        cold = SIMPLEX_ITERATIONS[solver_name](first)
        ws = first.warm_start()
    finally:
        first.close()

    assert cold > 0, 'the model must make the simplex work, or warmth would be unobservable'
    assert ws is not None and ws.column_statuses is not None, 'an LP solve leaves a basis to carry'

    fresh = member(tables)
    try:
        fresh.warm(ws)
        fresh.run(tables)
        assert SIMPLEX_ITERATIONS[solver_name](fresh) == 0, 'a carried basis starts at the optimum, not from scratch'
    finally:
        fresh.close()


def test_a_carried_basis_answers_what_the_cold_session_answered(solver_name):
    """A carry moves the route, never the answer — the oracle for the primitive."""
    tables = _tables(DISPATCH, dispatch_sources(), {'snapshot': SNAPSHOTS})
    member = SOLVERS[solver_name]
    first = member(tables)
    try:
        cold = first.run(tables)
        ws = first.warm_start()
    finally:
        first.close()
    assert ws is not None

    fresh = member(tables)
    try:
        fresh.warm(ws)
        warm = fresh.run(tables)
        assert warm.objective == pytest.approx(cold.objective), 'the carried basis reached a different optimum'
        assert warm.primal.to_list() == pytest.approx(cold.primal.to_list()), 'and a different vertex'
    finally:
        fresh.close()


# ---------------------------------------------------------------------------
# warm=False: deliberately cold, held to structurally
# ---------------------------------------------------------------------------


def test_warm_false_after_a_warm_session_solves_cold_by_construction(solver_name):
    """`warm=False` discards the held solver, so nothing *can* carry over.

    Structural rather than scrubbed: the object holding the basis and any
    solver-internal state is destroyed, which is why the cold solve repeats
    the first solve's iteration count exactly — and why `loads` ticks, the
    whole model having been transferred again.
    """
    with lps.build(DISPATCH, dispatch_sources(), coords={'snapshot': SNAPSHOTS}) as bound:
        first = bound.solve(solver_name=solver_name)
        iterations = SIMPLEX_ITERATIONS[solver_name](bound._engine._solver)
        assert first.started == 'cold', 'a first solve has nothing to start from'

        again = bound.solve(solver_name=solver_name)
        assert again.started == 'session', 'a kept solver re-solves from wherever it left off'

        cold = bound.solve(solver_name=solver_name, warm=False)
        assert cold.started == 'cold', 'warm=False is a cold start however warm the session was'
        assert cold.objective == pytest.approx(first.objective), 'cold moves the start, never the answer'
        assert SIMPLEX_ITERATIONS[solver_name](bound._engine._solver) == iterations, (
            'a deliberately cold solve repeats the first solve, iteration for iteration'
        )
        assert bound.diagnostics().loads == 2, 'warm=False discards the held solver, so the model loads again'


def test_warm_false_after_a_mip_solve_is_cold_too(solver_name):
    """The structural guarantee covers the MIP state no basis carries.

    An incumbent, a MIP start, cut pools — whatever the member squirrels away
    dies with the discarded solver, with no per-solver scrubbing to forget.
    """
    with lps.build(KNAPSACK, knapsack_sources()) as bound:
        first = bound.solve(solver_name=solver_name)
        cold = bound.solve(solver_name=solver_name, warm=False)

        assert cold.started == 'cold'
        assert cold.objective == pytest.approx(first.objective)
        assert bound.diagnostics().loads == 2, 'the discarded solver is the guarantee, and it shows up here'


# ---------------------------------------------------------------------------
# the MIP arm: no valid basis survives one, the incumbent crosses instead
# ---------------------------------------------------------------------------


def test_a_mixed_integer_solve_carries_an_incumbent_not_a_basis(solver_name):
    tables = _tables(KNAPSACK, knapsack_sources(), {})
    member = SOLVERS[solver_name]
    first = member(tables)
    try:
        before = first.run(tables).objective
        ws = first.warm_start()
    finally:
        first.close()

    assert ws is not None, 'a solved MIP still leaves something to carry'
    assert ws.column_statuses is None and ws.row_statuses is None, 'a solved MIP leaves no valid basis on any solver'
    assert ws.column_values is not None, 'what crosses a MIP rebuild is the incumbent'

    fresh = member(tables)
    try:
        fresh.warm(ws)
        assert fresh.run(tables).objective == pytest.approx(before), 'an incumbent bounds the search, never the answer'
    finally:
        fresh.close()


# ---------------------------------------------------------------------------
# refusals: everything checkable is checked, everything else is the caller's
# ---------------------------------------------------------------------------

SHAPES = [
    pytest.param(
        DISPATCH,
        dispatch_sources(),
        {'snapshot': SNAPSHOTS},
        (DISPATCH_NARROW, dispatch_sources(SNAPSHOTS, GENERATORS[:5]), {'snapshot': SNAPSHOTS}),
        id='a column short',
    ),
    pytest.param(
        DISPATCH,
        dispatch_sources(),
        {'snapshot': SNAPSHOTS},
        (DISPATCH_CAPPED, capped_sources(), {'snapshot': SNAPSHOTS}),
        id='rows extra under the same columns',
    ),
    pytest.param(
        KNAPSACK,
        knapsack_sources(),
        {},
        (KNAPSACK, knapsack_sources(ITEMS[:8]), {'item': ITEMS[:8]}),
        id='an incumbent for other items',
    ),
]


def _read_from(solver_name: str, model: dict[str, Any], given: dict[str, Any], coords: dict[str, Any]) -> Any:
    """The warm start one solve of *model* leaves, the session closed behind it."""
    tables = _tables(model, given, coords)
    session = SOLVERS[solver_name](tables)
    try:
        session.run(tables)
        return session.warm_start()
    finally:
        session.close()


@pytest.mark.parametrize(('model', 'given', 'coords', 'other'), SHAPES)
def test_a_warm_start_for_a_differently_shaped_model_is_refused(solver_name, model, given, coords, other):
    """A basis is positional, so a wrong span is a start about a different model.

    The refusal a cutting-plane master meets on its own rebind: a master that
    gained a row is exactly the second column of this table (#382).
    """
    ws = _read_from(solver_name, model, given, coords)
    other_model, other_given, other_coords = other
    tables = _tables(other_model, other_given, other_coords)
    session = SOLVERS[solver_name](tables)
    try:
        with pytest.raises(lps.CharterError, match='warm start carries'):
            session.warm(ws)
    finally:
        session.close()


def test_a_warm_start_from_another_solver_is_refused(solver_name):
    """Statuses are the reading solver's own encoding; nothing else takes them."""
    ws = _read_from(solver_name, DISPATCH, dispatch_sources(), {'snapshot': SNAPSHOTS})
    assert ws is not None
    tables = _tables(DISPATCH, dispatch_sources(), {'snapshot': SNAPSHOTS})
    session = SOLVERS[solver_name](tables)
    try:
        with pytest.raises(lps.CharterError, match='read from'):
            session.warm(replace(ws, solver='someone_else'))
    finally:
        session.close()


def test_a_solver_that_has_not_run_has_nothing_to_carry(solver_name):
    """Loaded is not solved: there is no basis and no incumbent to read yet."""
    tables = _tables(DISPATCH, dispatch_sources(), {'snapshot': SNAPSHOTS})
    session = SOLVERS[solver_name](tables)
    try:
        assert session.warm_start() is None, 'a model that has not been solved has left nothing behind'
    finally:
        session.close()


class _Refusing:
    """A HiGHS handle refusing the named hint call — the probe for the `_took` guard.

    HiGHS reports refusals by return value and carries on, so a dropped hint
    would silently start cold; nothing reachable from the tables can make the
    real handle refuse a span-checked hint, which is why this stands in.
    """

    def __init__(self, handle: Any, call: str) -> None:
        self._handle = handle
        self._call = call

    def __getattr__(self, name: str) -> Any:
        if name == self._call:
            import highspy

            return lambda hint: highspy.HighsStatus.kError
        return getattr(self._handle, name)


HINTS = [
    pytest.param(DISPATCH, dispatch_sources(), {'snapshot': SNAPSHOTS}, 'setBasis', id='a basis'),
    pytest.param(KNAPSACK, knapsack_sources(), {}, 'setSolution', id='an incumbent'),
]


@pytest.mark.parametrize(('model', 'given', 'coords', 'call'), HINTS)
def test_a_hint_the_solver_refuses_is_loud_not_a_silent_cold_start(model, given, coords, call):
    """The `_took` guard: a refused hint raises instead of solving cold."""
    tables = _tables(model, given, coords)
    member = SOLVERS['highs']
    session = member(tables)
    try:
        session.run(tables)
        ws = session.warm_start()
        assert ws is not None
        session._handle = _Refusing(session._handle, call)
        with pytest.raises(lps.CharterError, match='refused'):
            session.warm(ws)
    finally:
        session.close()
