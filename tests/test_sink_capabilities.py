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
    program's requirements happened to be collected in."""
    required = {'quadratic_constraint', 'nonconvex_quadratic_objective'}
    assert EMPTY.missing(required) == ['nonconvex_quadratic_objective', 'quadratic_constraint']
    assert EMPTY.missing(list(required)[::-1]) == ['nonconvex_quadratic_objective', 'quadratic_constraint']


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
        pytest.param('gurobi', 'nonconvex_quadratic_objective', 'native', id='gurobi-goes-spatial'),
        pytest.param('gurobi', 'quadratic_constraint', 'native', id='gurobi-takes-a-quadratic-row'),
    ],
)
def test_the_shipped_solver_table(sink, capability, expected):
    """The rows of docs/about/benchmarks.md#sink-capabilities, as declared."""
    assert SOLVERS[sink].capabilities.support(capability) == expected


def test_only_highs_excludes_a_combination():
    """Gurobi's column has no exclusion, which is what makes it the sink a
    refusal can name."""
    assert SOLVERS['highs'].capabilities.excludes == (frozenset({'quadratic_objective', 'integrality'}),)
    assert SOLVERS['gurobi'].capabilities.excludes == ()


def test_the_lp_writer_carries_everything():
    """A section is text, so the format has no exclusion either — and its
    capabilities are the *writer's*, not the reader's."""
    capabilities = WRITERS['.lp'].capabilities
    assert capabilities.missing(CAPABILITIES) == []
    assert capabilities.excludes == ()
