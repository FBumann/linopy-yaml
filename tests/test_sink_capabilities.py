"""The capability descriptor: what it answers, and what each sink declares.

The probes measure what the libraries do; the claim is tied to that evidence at
the bottom of each probe module. What is left for here is the descriptor's own
arithmetic, which no probe reaches.
"""

from __future__ import annotations

import pytest

from lpspec.relational.sinks import SOLVERS, WRITERS
from lpspec.relational.sinks.capabilities import CAPABILITIES, Capabilities

EMPTY = Capabilities(supports={})

HIGHS_SHAPED = Capabilities(
    supports={'integrality': 'native', 'sos': 'reformulated', 'quadratic_objective': 'native'},
    excludes=(frozenset({'quadratic_objective', 'integrality'}),),
)


def test_a_capability_left_out_is_absent():
    """So a descriptor lists what a sink *can* do, and nothing has to keep the
    full vocabulary in step by hand."""
    assert EMPTY.support('sos') == 'absent'
    assert HIGHS_SHAPED.support('quadratic_constraint') == 'absent'
    assert HIGHS_SHAPED.support('sos') == 'reformulated', 'reformulated is an answer, not a missing native'


def test_missing_names_only_what_is_required_and_absent():
    assert HIGHS_SHAPED.missing(['sos', 'quadratic_objective']) == []
    assert HIGHS_SHAPED.missing(['quadratic_constraint']) == ['quadratic_constraint']


def test_missing_reads_in_vocabulary_order_not_the_callers():
    """A refusal naming two capabilities reads the same way whichever order the
    program's requirements happened to be collected in.

    Both calls pass a *list*, in each of the two orders, since a set has no
    order to disagree with.
    """
    ordered = ['nonconvex_quadratic_objective', 'quadratic_constraint']
    assert EMPTY.missing(ordered) == ordered
    assert EMPTY.missing(ordered[::-1]) == ordered, 'the caller collected them the other way round'


def test_an_exclusion_fires_only_on_the_whole_combination():
    """The case a flat feature set cannot express: both halves supported, the
    pair refused."""
    assert HIGHS_SHAPED.excluded(['quadratic_objective']) is None
    assert HIGHS_SHAPED.excluded(['integrality']) is None
    assert HIGHS_SHAPED.excluded(['quadratic_objective', 'integrality']) == frozenset(
        {'quadratic_objective', 'integrality'}
    )


def test_an_exclusion_fires_inside_a_larger_requirement():
    """A model needing a third thing as well is still the refused pair."""
    assert HIGHS_SHAPED.excluded(['quadratic_objective', 'integrality', 'sos']) is not None


def test_nothing_is_excluded_where_no_exclusion_is_declared():
    assert HIGHS_SHAPED.excluded([]) is None
    assert Capabilities(supports={'integrality': 'native'}).excluded(CAPABILITIES) is None


@pytest.mark.parametrize(
    ('sink', 'capability', 'expected'),
    [
        pytest.param('highs', 'sos', 'reformulated', id='highs-has-no-set-concept'),
        pytest.param('highs', 'quadratic_objective', 'native', id='highs-takes-a-convex-hessian'),
        pytest.param('highs', 'nonconvex_quadratic_objective', 'absent', id='highs-refuses-a-nonconvex-one'),
        pytest.param('highs', 'quadratic_constraint', 'absent', id='highs-has-no-quadratic-row-at-all'),
        pytest.param('gurobi', 'sos', 'native', id='gurobi-branches-on-a-set'),
        pytest.param('gurobi', 'quadratic_objective', 'absent', id='the-gurobi-sink-passes-no-hessian-yet'),
        pytest.param('gurobi', 'quadratic_constraint', 'absent', id='nor-a-quadratic-row'),
    ],
)
def test_the_shipped_solver_table(sink, capability, expected):
    """What each sink can ingest **as shipped**.

    Not the same table as docs/about/benchmarks.md#sink-capabilities, and the
    difference is the point: that page describes the *libraries*, and gurobipy
    takes a Hessian. This sink does not pass it one yet, so its entry is
    `absent` — a descriptor that claimed otherwise would drop the quadratic
    part of an objective and answer a different model's optimum.
    """
    assert SOLVERS[sink].capabilities.support(capability) == expected


def test_only_highs_excludes_a_combination():
    """Gurobi's column has no exclusion, which is what makes it the sink a
    refusal can name."""
    assert SOLVERS['highs'].capabilities.excludes == (
        frozenset({'quadratic_objective', 'integrality'}),
        frozenset({'quadratic_objective', 'sos'}),
    )
    assert SOLVERS['gurobi'].capabilities.excludes == ()


def test_a_set_is_excluded_from_the_hessian_it_would_arrive_beside():
    """The exclusion a reformulation manufactures, and the reason it is
    declared rather than derived at the hand-off: what HiGHS is handed for a
    set *is* binaries, so the pair it refuses is the pair it would be given —
    while the model itself declares no integrality at all."""
    excluded = SOLVERS['highs'].capabilities.excluded(['sos', 'quadratic_objective'])
    assert excluded == frozenset({'quadratic_objective', 'sos'}), (
        'a set and a Hessian reach highs as integrality and a Hessian, which it refuses'
    )
    assert SOLVERS['gurobi'].capabilities.excluded(['sos', 'quadratic_objective']) is None, (
        'gurobi branches on the set instead, so nothing is manufactured'
    )


def test_the_lp_writer_carries_everything():
    """A section is text, so the format has no exclusion either — and its
    capabilities are the *writer's*, not the reader's."""
    capabilities = WRITERS['.lp'].capabilities
    assert capabilities.missing(CAPABILITIES) == []
    assert capabilities.excludes == ()
