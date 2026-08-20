"""The scoping divergences, checked against the oracle lane itself.

`test_resolution.py` checks the language rules. This module checks the thing
that actually mattered: that the *eager* lane now refuses what the relational
lane refuses, in the same place, for the same reason. Before resolution was a
pass, each of these built a model on one lane and raised on the other.
"""

from __future__ import annotations

import datetime

import polars as pl
import pytest
import yaml as pyyaml

import lpspec as lps
from tests.conftest import dispatch_model_path, override
from tests.differential import differential
from tests.oracle import lpspec_linopy, pd  # skips the module without the [linopy] extra


@pytest.mark.parametrize(
    ('where', 'match'),
    [
        pytest.param('typo_name > 0', "'typo_name' not found", id='a-name-nothing-declares'),
        pytest.param('p_max > cost', 'compares two parameters', id='two-parameters-compared'),
        pytest.param('generator == snapshot', 'compares against dimension', id='a-dimension-on-the-right'),
        pytest.param('nonexistent', "'nonexistent' not found", id='a-bare-name-nothing-declares'),
        pytest.param('snapshot', 'bare dimension name is true at every coordinate', id='a-bare-dimension-name'),
    ],
)
def test_both_lanes_refuse_the_same_where(tmp_path, dispatch_model_inputs, where, match):
    data = dispatch_model_inputs
    path = dispatch_model_path(tmp_path, **{'variables.p.where': where})

    with pytest.raises(ValueError, match=match):
        lpspec_linopy.build(path, data)

    with pytest.raises(ValueError, match=match):
        lps.check(path)


def test_both_lanes_refuse_a_comparison_that_carries_no_variable(tmp_path, dispatch_model_inputs):
    """A constraint whose two sides are both constants decides nothing (#1171).

    Was: the relational lane built the model quietly with no such row, while
    the eager lane raised linopy's own `TypeError` at build — one language,
    two answers, and neither of them said what was wrong with the file. It is
    decidable with no data bound, so it is decided where the file is read.
    """
    data = dispatch_model_inputs
    path = dispatch_model_path(tmp_path, **{'constraints.balance.expression': 'p_max <= 1'})

    with pytest.raises(ValueError, match='decides nothing'):
        lpspec_linopy.build(path, data)

    with pytest.raises(ValueError, match='decides nothing'):
        lps.check(path)


#: Where-strings that must build *identically* on both lanes. Chosen to cover
#: every resolved predicate type — see the exhaustiveness test below. The dim
#: comparisons are deliberately always-true: a mask that removes every variable
#: from a constraint row exposes a separate divergence, pinned below.
ACCEPTED = [
    'True',
    'p_max',
    'p_max > 0',
    'snapshot >= 0',
    'snapshot >= index(snapshot, 0)',
    'NOT p_max > 150',
    'p_max > 0 AND snapshot >= 0',
    'p_max > 0 OR snapshot >= 0',
]

#: Predicates this sweep cannot host, with where they are checked instead. The
#: sweep masks ``variables.p.where`` on the dispatch model, and a bare variable
#: name fits neither slot: in a variable's own where it is a self-reference
#: (rejected at load), and on ``balance`` it spans a dim the constraint does not
#: (a DimensionError, correctly — reducing it needs an `all`-reduction, #469).
#: The three lookup predicates fit the slot but not the *model*: dispatch
#: declares no lookup, and giving it one changes a fixture the rest of this
#: file shares. They sweep a network carrying both lookup kinds and a partial
#: one, differentially against the same oracle.
#: Mapped rather than skipped so the coverage guard below still names a test.
COVERED_ELSEWHERE = {
    'VariableDefinedNode': ('tests/test_relational.py::test_a_bare_variable_name_in_a_where_asks_whether_it_exists'),
    'LookupComparisonNode': 'tests/test_label_coords.py::test_a_where_reads_a_lookup',
    'LookupPairComparisonNode': 'tests/test_label_coords.py::test_a_lookup_where_agrees_with_the_oracle',
    'LookupDefinedNode': 'tests/test_label_coords.py::test_a_where_reads_a_lookup',
}


@pytest.mark.parametrize('where', ACCEPTED)
def test_both_lanes_build_the_same_model(tmp_path, dispatch_model_inputs, where):
    """Both lanes agree on *which* model they built, feasible or not.

    A mask that excludes snapshot 0 leaves the balance row unsatisfiable; that
    is not the claim here, and neither lane is asked to make every mask
    feasible.
    """
    data = dispatch_model_inputs
    path = dispatch_model_path(tmp_path, **{'variables.p.where': where})

    m = lpspec_linopy.build(path, data)
    eager_rows = int((m.variables['p'].labels != -1).sum())
    eager_status = m.solve(solver_name='highs')[1]

    with lps.build(path, data) as bound:
        relational_rows = bound._engine._variables['p'].select(pl.len()).collect().item()
        relational_status = bound.solve().termination_condition

    assert eager_rows == relational_rows, f'{where}: {eager_rows} vs {relational_rows} variables'
    assert eager_status == relational_status, f'{where}: {eager_status} vs {relational_status}'


def test_every_resolved_predicate_is_parity_tested():
    """The guard that would have caught the DimDefined hole.

    `DimDefined` shipped in #62 lowering to `plan.BooleanConstant(True)`, which discarded
    the dimension — so unlike `DimensionComparisonNode`, nothing checked it against the frame's
    dims, and a bare dimension name outside `foreach` raised eagerly and built
    relationally. No test touched it. This one fails if any resolved predicate
    is not exercised by ACCEPTED above, so a new node cannot arrive untested.
    """
    from typing import get_args

    from lpspec.language.resolution import Namespace, where_of
    from lpspec.language.where_parser import (
        UnresolvedComparisonNode,
        UnresolvedNameNode,
        UnresolvedPositionNode,
        WhereNode,
    )

    # rewritten by resolution, never evaluated
    unresolved = {UnresolvedNameNode, UnresolvedComparisonNode, UnresolvedPositionNode}
    expected = set(get_args(WhereNode)) - unresolved

    ns = Namespace(('p',), ('p_max', 'cost', 'load'), ('snapshot', 'generator'))
    covered: set[type] = set()

    def walk(node):
        covered.add(type(node))
        for child in vars(node).values():
            if hasattr(child, '__dataclass_fields__'):
                walk(child)

    for where in ACCEPTED:
        walk(where_of(where, ns, 't'))
    covered |= {t for t in expected if t.__name__ in COVERED_ELSEWHERE}

    missing = expected - covered
    assert not missing, (
        f'resolved predicates with no both-lanes test: {sorted(t.__name__ for t in missing)}. '
        f'Add it to ACCEPTED, or to COVERED_ELSEWHERE naming the test that does cover it.'
    )


def test_a_constraint_row_left_with_no_variables(tmp_path, dispatch_model_inputs):
    """A masked *variable* can orphan an unmasked *constraint* row — and both
    lanes now agree that such a row is not built.

    `where: "snapshot > 0"` on `p` leaves `balance` at snapshot 0 with no
    terms. This was an xfail: linopy handed the solver three rows of four while
    the relational lane kept the fourth as `0 == 80` and reported Infeasible —
    one lane answering a question the other refused.

    The rule is now stated at the level the property lives at rather than per
    provenance, so the lanes reach it independently: linopy's own invariant is
    the same one (`labels != -1` and at least one var), which is why it needed
    no shim to agree.

    The omission is asserted too. Dropping a declared row is only defensible
    because the build says it happened.
    """
    data = dispatch_model_inputs
    path = dispatch_model_path(tmp_path, **{'variables.p.where': 'snapshot > 0'})

    m = lpspec_linopy.build(path, data)
    eager_status = m.solve(solver_name='highs')[1]

    with lps.build(path, data) as bound:
        relational_status = bound.solve().termination_condition
        assert bound.diagnostics().omissions.to_dicts() == [{'constraint': 'balance', 'rows_not_built': 1}], (
            'a dropped row has to be reported, or a declared constraint goes quietly unenforced'
        )

    assert eager_status == relational_status


#: A dimension the data leaves with **no members**, and a variable reduced over
#: it. The empty sum is a number, so the row it lands in asserts something about
#: constants alone — the same shape a masked variable leaves behind, reached by
#: the one provenance that removes the term axis itself.
EMPTY_AXIS_MODEL = {
    'dimensions': {'g': {'dtype': 'str'}, 'k': {'dtype': 'int'}},
    'parameters': {'exists': {'dims': ['g', 'k'], 'dtype': 'bool'}},
    'variables': {
        'w': {'foreach': ['g', 'k'], 'where': 'exists', 'bounds': {'lower': 0, 'upper': 1}},
        'p': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {'convex': {'foreach': ['g'], 'expression': 'sum(w, over=k) == 1'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(p, over=g)'},
}


@pytest.mark.parametrize('sense', ['==', '<=', '>='], ids=['equality', 'upper-bound', 'lower-bound'])
def test_a_row_over_a_dimension_with_no_members_is_not_built_on_either_lane(sense: str):
    """The same rule as above, reached where the *dimension* is empty (#1108).

    A reduction over a set with no members is `0`, so `sum(w, over=k) == 1`
    is a row about constants alone and neither lane builds it. Was: the
    relational lane solved, and the eager lane raised linopy's `Both sides of
    the constraint are constant` before any mask could speak — so a component
    library, whose whole shape is one program covering features a given system
    does not use, could not use the oracle lane at all.

    The two senses are one property: the shape decides, not the comparison.
    """
    model = override(EMPTY_AXIS_MODEL, **{'constraints.convex.expression': f'sum(w, over=k) {sense} 1'})
    data = {
        'g': pd.Index(['a', 'b'], name='g'),
        'k': pd.Index([], name='k', dtype='int64'),
        'exists': pd.DataFrame(
            {
                'g': pd.Series([], dtype='object'),
                'k': pd.Series([], dtype='int64'),
                'value': pd.Series([], dtype='bool'),
            }
        ),
    }

    with differential(model, data) as run:
        assert 'convex' not in run.model.constraints, 'a row asserting something about constants only was built'
        assert run.engine.diagnostics().rows == 0, 'the relational lane built one anyway'
        assert run.engine.diagnostics().omissions.to_dicts() == [{'constraint': 'convex', 'rows_not_built': 2}], (
            'a declared row dropped for want of a term is only defensible because the build says it happened'
        )
        assert run.oracle == 20.0, 'both `p` reach their bound — an unbuilt row pins nothing'


BOOL_MASK_MODEL = {
    'dimensions': {'t': {'dtype': 'int', 'values': [0, 1, 2]}},
    'parameters': {'active': {'dims': ['t'], 'dtype': 'bool'}, 'cap': {'dims': ['t']}},
    'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {'floor': {'foreach': ['t'], 'expression': 'x >= cap', 'where': 'active'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x, over=t)'},
}


def test_a_bool_parameter_is_a_mask_on_both_lanes():
    """A bool parameter reads as its own value: true masks in, false masks out,
    and an absent row masks out. Was: the relational lane raised
    `isfinite(BOOLEAN)` at build, and the eager lane read false as true.
    """
    data = {
        'active': pd.Series({0: True, 1: False}),
        'cap': pd.Series({0: 1.0, 1: 1.0, 2: 1.0}),
    }

    with differential(BOOL_MASK_MODEL, data) as run:
        assert run.oracle == 1.0, 'true masks the floor in at t=0 only, so exactly one x sits at its cap'


#: A budget row over no dims, because `sum` reduces the only one away — and a
#: scalar `slack` column and scalar `budget` value beside it, so one model
#: carries the empty coordinate in all three positions it can appear in.
SCALAR_ROW_MODEL = {
    'dimensions': {'f': {'values': ['a', 'b', 'c']}},
    'parameters': {'cost': {'dims': ['f']}, 'budget': {'dims': []}},
    'variables': {
        'x': {'foreach': ['f'], 'bounds': {'lower': 0, 'upper': 100}},
        'slack': {'foreach': [], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {'budget_row': {'foreach': [], 'expression': 'sum(x, over=f) - slack <= budget'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x * cost)'},
}


def test_the_empty_coordinate_builds_on_both_lanes():
    """A scalar row, a scalar column and a scalar value, in one model (#320).

    Was: the eager lane built all three and solved; the relational lane raised
    `constraint 'budget_row' has no dims`, and with that guard gone,
    `variable 'slack' has no dims (scalars: use dims of size 1)`. So the same
    file was two languages, against hard rule 3 — and the hint pointed at the
    dummy dimension the declaration rules now say is never how a scalar is
    written.

    Underneath both guards `_coordinate_product` asserted that no declaration
    arrives dimensionless. A product over nothing has one coordinate, not none.
    """
    data = {'cost': pd.Series({'a': 1.0, 'b': 2.0, 'c': 3.0}), 'budget': 120.0}

    with differential(SCALAR_ROW_MODEL, data) as run:
        assert run.oracle == 360.0
        assert run.result.dual('budget_row').height == 1, 'each claim is one — not zero, and not one per f'
        assert run.result.primal('slack').to_dicts() == [{'value': 10.0}]
        assert run.result.primal('x').sort('f')['value'].to_list() == [0.0, 30.0, 100.0]


@pytest.mark.parametrize(
    ('threshold', 'rows', 'objective'),
    [
        pytest.param(999.0, 0, 600.0, id='masked-out'),
        pytest.param(10.0, 1, 360.0, id='masked-in'),
    ],
)
def test_a_masked_scalar_variable_takes_its_row_with_it(threshold, rows, objective):
    """Absence spreads through arithmetic at no dimension either (#340).

    Was: the relational lane kept `budget_row` and enforced `sum(x) <= budget`
    — a constraint the language says should not exist — because a scalar
    variable's presence was `select()` over no dims, and polars cannot hold a
    frame with rows and no columns. Present and absent collapsed to the same
    `(0, 0)` at the moment presence was built, so nothing downstream could
    restrict on it. Later refused at load instead, which traded a silent wrong
    answer for a loud divergence; this is the answer.

    The two rungs are the whole property: the mask is data, so the *same file*
    must drop the row or keep it depending only on what `budget` turns out to
    be.
    """
    model = override(SCALAR_ROW_MODEL, **{'variables.slack.where': f'budget > {threshold}'})
    data = {'cost': pd.Series({'a': 1.0, 'b': 2.0, 'c': 3.0}), 'budget': 120.0}

    with differential(model, data) as run:
        assert run.oracle == objective
        assert run.result.dual('budget_row').height == rows, 'the row is gone, not slackened: a dropped row has no dual'


DATETIME_MODEL = {
    'dimensions': {'snapshot': {'dtype': 'datetime'}, 'generator': {'dtype': 'str'}},
    'parameters': {'cost': {'dims': ['generator']}, 'load': {'dims': ['snapshot']}},
    'variables': {
        'p': {'foreach': ['snapshot', 'generator'], 'where': "snapshot > '2030-01-02'", 'bounds': {'lower': 0}}
    },
    'constraints': {
        'bal': {
            'foreach': ['snapshot'],
            'where': "snapshot > '2030-01-02'",
            'expression': 'sum(p, over=generator) == load',
        }
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
}


def test_a_datetime_boundary_is_sayable_on_both_lanes(tmp_path):
    """A quoted ISO date in a `where`, which had no spelling at all (#460).

    `snapshot > 2030-01-01` and its quoted form both failed to parse, and
    `snapshot > 0` parsed into a comparison against the *epoch* — so a datetime
    dimension was usable exactly as long as nothing about the model was
    conditional on time. There was no way to name a boundary.
    """
    path = tmp_path / 'm.yaml'
    path.write_text(pyyaml.safe_dump(DATETIME_MODEL))
    days = [datetime.date(2030, 1, d) for d in (1, 2, 3)]
    frames = {
        'cost': pl.DataFrame({'generator': ['wind', 'gas'], 'value': [1.0, 5.0]}),
        'load': pl.DataFrame({'snapshot': days, 'value': [10.0, 20.0, 30.0]}),
        'snapshot': pl.DataFrame({'snapshot': days}),
        'generator': pl.DataFrame({'generator': ['wind', 'gas']}),
    }
    eager_data = {
        'cost': pd.Series({'wind': 1.0, 'gas': 5.0}),
        'load': pd.Series([10.0, 20.0, 30.0], index=pd.Index(days, name='snapshot')),
    }
    eager_data |= {
        'snapshot': pd.Index(days, name='snapshot'),
        'generator': pd.Index(['wind', 'gas'], name='generator'),
    }

    m = lpspec_linopy.build(path, eager_data)
    m.solve(solver_name='highs')
    eager = float(m.objective.value)

    with lps.solve(path, frames) as result:
        relational = result.objective
        assert result.primal('p')['snapshot'].dtype in (pl.Date, pl.Datetime('us')), 'the coordinate keeps its dtype'
        assert result.primal('p').height == 2, 'only the third day survives the boundary'

    assert eager == relational == 30.0
