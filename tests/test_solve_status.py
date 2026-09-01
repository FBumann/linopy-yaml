"""The solve outcome, and linopy as its oracle.

`relational/status.py` copies linopy's status vocabulary spelling for
spelling, and each solver sink copies linopy's own mapping for its solver.
Copies rot. These tests import linopy and compare, so a divergence — ours
drifting, or a linopy release moving — fails here instead of being discovered
by a user who knows one vocabulary and is handed another.

The engine itself never imports linopy (docs/about/architecture.md, hard
rule 2). Tests may — the same oracle arrangement the differential tests use
for the math.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any

import polars as pl
import pytest

import lpspec as lps
from lpspec.errors import NoSolutionError
from lpspec.relational.sinks.solvers.gurobi import _CONDITION_OF_GUROBI_STATUS, _LINOPY_DIVERGENCES
from lpspec.relational.sinks.solvers.highs import _CONDITION_OF_HIGHS_STATUS
from lpspec.relational.sinks.solvers.xpress import _BEYOND_LINOPY, _CONDITION_OF_SOL_STATUS
from lpspec.relational.status import STATUS_TO_TERMINATION_CONDITIONS, SolveStatus
from tests.conftest import CASES

# ---------------------------------------------------------------------------
# linopy as the oracle for the vocabulary
# ---------------------------------------------------------------------------


def test_the_status_rollup_matches_linopy():
    constants = pytest.importorskip('linopy.constants')
    theirs = {
        status.value: {condition.value for condition in conditions}
        for status, conditions in constants.STATUS_TO_TERMINATION_CONDITION_MAP.items()
    }
    assert {k: set(v) for k, v in STATUS_TO_TERMINATION_CONDITIONS.items()} == theirs


def test_the_highs_mapping_matches_linopy():
    assert _linopy_condition_map('Highs', ast.Attribute, 'attr') == _CONDITION_OF_HIGHS_STATUS


def test_the_gurobi_mapping_matches_linopy_where_it_claims_to():
    """The copy, minus three declared exceptions.

    linopy's Gurobi map contradicts Gurobi's own documented status codes in
    three places, each listed in ``_LINOPY_DIVERGENCES`` with its reason.
    Asserted in both directions: everything else still matches, and every
    declared divergence still diverges — so if linopy fixes one, the entry
    has to go.
    """
    theirs = _linopy_condition_map('Gurobi', ast.Constant, 'value')
    assert set(theirs) == set(_CONDITION_OF_GUROBI_STATUS), (
        'linopy and this package no longer cover the same Gurobi status codes'
    )
    for code, condition in theirs.items():
        if code in _LINOPY_DIVERGENCES:
            assert _CONDITION_OF_GUROBI_STATUS[code] != condition, (
                f'linopy now agrees with us on status {code} — drop the entry from _LINOPY_DIVERGENCES'
            )
        else:
            assert _CONDITION_OF_GUROBI_STATUS[code] == condition


def test_the_xpress_mapping_matches_linopy():
    """Copied entry for entry, with nothing claimed as an exception.

    The map is keyed by ``SolStatus`` *value* here and by the enum member
    there, the sink not being allowed to import xpress at module level — so
    the enum is what the two are compared through, and a member renamed
    upstream fails here rather than silently dropping an entry.
    """
    xpress = pytest.importorskip('xpress', reason='the xpress sink needs the [xpress] extra')
    theirs = _linopy_condition_map('Xpress', ast.Attribute, 'attr', ast.Constant, 'value')
    assert theirs, 'linopy no longer spells its Xpress map as SolStatus attributes against strings'
    assert {int(xpress.SolStatus[name]): condition for name, condition in theirs.items()} == _CONDITION_OF_SOL_STATUS


def test_the_xpress_sink_adds_to_linopys_answer_rather_than_contradicting_it():
    """``_BEYOND_LINOPY`` is an addition, so it names no status code.

    Gurobi's ``_LINOPY_DIVERGENCES`` overrides verdicts linopy gives; this
    reads a second axis linopy never looks at, so there is nothing to
    disagree with — and the word it reports still has to be one linopy
    defines.
    """
    assert _BEYOND_LINOPY, 'the sink reads solvestatus for a reason — it belongs in the table'
    every = set().union(*STATUS_TO_TERMINATION_CONDITIONS.values())
    assert set(_CONDITION_OF_SOL_STATUS.values()) <= every
    assert 'internal_solver_error' in every, 'the condition the second axis reports is linopy vocabulary'


def test_every_gurobi_divergence_stays_inside_linopys_vocabulary():
    """Diverging on a verdict is not licence to invent a word for it. Every
    condition this package reports is one linopy also defines, which is what
    keeps `status`, `is_ok` and the rollup meaningful across both."""
    assert set(_CONDITION_OF_GUROBI_STATUS.values()) <= set().union(*STATUS_TO_TERMINATION_CONDITIONS.values())


def _linopy_condition_map(
    solver: str,
    node: type[ast.expr],
    attribute: str,
    value_node: type[ast.expr] | None = None,
    value_attribute: str | None = None,
) -> dict[Any, Any]:
    """linopy's ``CONDITION_MAP`` for *solver*, read out of its source.

    Each solver spells the map differently — HiGHS keys it by
    ``HighsModelStatus`` attributes, Gurobi by integer literals, Xpress by
    ``SolStatus`` attributes against plain strings — so the node type and the
    attribute holding the value are arguments, and the two sides may differ.

    Brittle to a linopy refactor, deliberately: the map is a local inside a
    method, so there is nothing to import, and a copy nobody checks is a copy
    that rots. If linopy moves it, the assertions say so rather than passing
    vacuously.
    """
    solvers = pytest.importorskip('linopy.solvers')
    tree = ast.parse(inspect.getsource(solvers))
    cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == solver), None)
    assert cls is not None, f'linopy no longer has a {solver} solver class — re-verify the copy by hand'
    literals = [
        n.value
        for n in ast.walk(cls)
        if isinstance(n, (ast.AnnAssign, ast.Assign)) and 'CONDITION_MAP' in ast.dump(n)
        if isinstance(n.value, ast.Dict)
    ]
    assert literals, f'linopy no longer defines {solver}.CONDITION_MAP as a dict literal — re-verify by hand'
    values, value_of = value_node or node, value_attribute or attribute
    return {
        getattr(key, attribute): getattr(value, value_of)
        for key, value in zip(literals[0].keys, literals[0].values, strict=True)
        if isinstance(key, node) and isinstance(value, values)
    }


# ---------------------------------------------------------------------------
# what the two axes mean here
# ---------------------------------------------------------------------------


def test_ok_means_values_worth_reading_not_optimality():
    """A run stopped at a time limit still has an incumbent."""
    assert SolveStatus('optimal').is_ok
    assert SolveStatus('time_limit').is_ok
    assert SolveStatus('suboptimal').is_ok
    assert not SolveStatus('infeasible').is_ok
    assert not SolveStatus('unbounded').is_ok


def test_an_infeasible_solve_reports_both_axes_and_a_nan_objective():
    with lps.solve(*CASES['INFEASIBLE']) as solution:
        assert solution.status == 'warning'
        assert solution.termination_condition == 'infeasible'
        assert not solution.is_ok
        assert solution.objective != solution.objective, 'nan, not 0.0'


def test_reading_results_without_a_solution_raises(tmp_path):
    """HiGHS returns a full-length vector of zeros whatever the status, so
    handing it back would be indistinguishable from an answer."""
    with lps.solve(*CASES['INFEASIBLE']) as solution:
        with pytest.raises(NoSolutionError, match='infeasible'):
            solution.primal('p')
        with pytest.raises(NoSolutionError):
            solution.to_parquet(tmp_path)


# ---------------------------------------------------------------------------
# solver options, and the incumbent question they make reachable
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def knapsack():
    """A MIP big enough that HiGHS does not finish it instantly."""
    import random

    random.seed(0)
    n = 60
    weights = [random.randint(10**6, 2 * 10**6) for _ in range(n)]
    spec = {
        'dimensions': {'i': {'dtype': 'int'}, 'one': {'dtype': 'int'}},
        'parameters': {'w': {'dims': ['i']}, 'cap': {'dims': ['one']}},
        'variables': {'x': {'foreach': ['i'], 'domain': 'binary'}},
        'constraints': {'budget': {'foreach': ['one'], 'expression': 'sum(x * w, over=i) <= cap'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x * w, over=i)'},
    }
    sources = {
        'i': list(range(n)),
        'one': [0],
        'w': pl.DataFrame({'i': list(range(n)), 'value': [float(v) for v in weights]}),
        'cap': pl.DataFrame({'one': [0], 'value': [float(sum(weights) // 2)]}),
    }
    return spec, sources


def test_solver_options_reach_the_solver(knapsack):
    """Forwarded verbatim, the way linopy's are. `time_limit=0` is the cheapest
    proof: without it this model solves to optimality."""
    spec, sources = knapsack
    with lps.solve(spec, sources, solver_options={'time_limit': 0.0}) as result:
        assert result.termination_condition == 'time_limit'
    with lps.solve(spec, sources) as result:
        assert result.termination_condition == 'optimal'


def test_a_time_limit_with_no_incumbent_is_ok_but_unreadable(knapsack):
    """The gap `is_ok` alone cannot see, and where we go beyond linopy.

    A MIP stopped before it found any feasible point rolls up to `ok` —
    linopy's `safe_get_solution` would read its zero-filled `col_value` as an
    answer. `has_primal` carries the solver's own verdict instead.
    """
    spec, sources = knapsack
    with lps.solve(spec, sources, solver_options={'time_limit': 0.0}) as result:
        assert result.is_ok, "linopy's rollup says the run was not an error"
        assert not result.has_primal, 'but nothing was found'
        assert result.objective != result.objective, 'nan, not 0.0'
        with pytest.raises(NoSolutionError, match='time_limit'):
            result.primal('x')


def test_an_optimal_solve_is_both_ok_and_readable(knapsack):
    spec, sources = knapsack
    with lps.solve(spec, sources) as result:
        assert result.is_ok
        assert result.has_primal
        assert result.primal('x')['value'].sum() > 0
