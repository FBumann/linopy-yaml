"""A quadratic objective: the one position the language takes degree 2 in.

The claim is not "it solves" but that **four spellings of one quadratic form
agree**: the engine's one entry per unordered pair, a Hessian's halved form
with its doubled diagonal, the LP section's uniform doubling, and linopy's own
`QuadraticExpression`. A conversion error in any of them still solves and
answers something else, which is why nearly every test here is differential.
"""

from __future__ import annotations

import polars as pl
import pytest
from math_spec import to_latex

import lpspec as lps
from lpspec.errors import LpspecError
from tests.differential import differential

#: Two generators, two variables over them, one row tying them together. Small
#: enough that the optimum is arithmetic: the quadratic cost spreads output
#: evenly where a linear one would fill the cheapest first.
MODEL = {
    'dimensions': {'g': {'dtype': 'str', 'values': ['a', 'b']}},
    'parameters': {'need': {'dims': []}, 'weight': {'dims': ['g']}},
    'variables': {
        'p': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 10}},
        'q': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {'meet': {'foreach': [], 'expression': 'sum(p, over=g) + sum(q, over=g) >= need'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * p, over=g)'},
}

SOURCES = {
    'need': pl.DataFrame({'value': [4.0]}),
    'weight': pl.DataFrame({'g': ['a', 'b'], 'value': [1.0, 3.0]}),
}


def model(expression: str, **patch) -> dict:
    """MODEL with another objective — the axis every test here varies."""
    return {**MODEL, 'objective': {'sense': 'minimize', 'expression': expression}, **patch}


def quad_of(expression: str, sources=None) -> pl.DataFrame:
    """The built model's quadratic stream, exactly as a sink would be handed it.

    Deliberately **not** sorted here: the stream leaves the engine in
    ``(col_l, col_r)`` order as a contract, and a helper that tidied it would
    be the one place no test could tell whether the contract held.
    """
    with lps.build(model(expression), dict(sources or SOURCES)) as bound:
        return bound._engine._model.quad


# ---------------------------------------------------------------------------
# the four spellings agree
# ---------------------------------------------------------------------------


#: Every form here is **convex**, and deliberately: HiGHS solves only convex
#: QPs, so a pure cross term is refused by the oracle lane before it can
#: disagree with anything. Curvature is a property of the *data* and belongs to
#: the nonconvex test below, not to the agreement tests.
@pytest.mark.parametrize(
    'expression',
    [
        pytest.param('sum(p * p, over=g)', id='a-square'),
        pytest.param('sum(p * p + p * q + q * q, over=g)', id='a-cross-term'),
        pytest.param('sum(p * p * weight, over=g)', id='a-square-with-a-parameter-coefficient'),
        pytest.param('sum(p * p + q * q - p * q, over=g)', id='a-form-with-both'),
        pytest.param('sum(p * p, over=g) + sum(q * weight, over=g)', id='quadratic-plus-affine'),
        pytest.param('sum(p * (p + weight), over=g)', id='a-factor-carrying-a-constant-part'),
        pytest.param('sum((p + weight) * p, over=g)', id='the-same-factors-the-other-way-round'),
    ],
)
def test_both_lanes_and_the_lp_file_reach_one_optimum(expression):
    """An off-diagonal Hessian entry that doubled, an LP coefficient that did
    not, a pair counted twice — each still solves, and each moves the optimum."""
    with differential(model(expression), SOURCES, lp=True):
        pass


#: The agreement models above leave ``p`` free at zero, where a *missing* term
#: costs nothing. A floor puts every variable in the objective's way, so a
#: dropped product moves the optimum instead of being multiplied by nothing.
FLOORED = {name: bounds | {'bounds': {'lower': 1, 'upper': 10}} for name, bounds in MODEL['variables'].items()}


@pytest.mark.parametrize(
    ('expression', 'optimum'),
    [
        pytest.param('sum(p * (p + weight), over=g)', 6.0, id='a-constant-part-on-the-right'),
        pytest.param('sum((p + weight) * p, over=g)', 6.0, id='a-constant-part-on-the-left'),
        pytest.param('sum((p + weight) * (q + weight), over=g)', 20.0, id='a-constant-part-on-both-sides'),
    ],
)
def test_a_product_keeps_both_of_its_linear_cross_terms(expression, optimum):
    """``(a + c)(b + d)`` has two mixed products and the model owns both.

    A walk that normalises the variable-carrying side to the left forms one of
    them and drops the other, and *which* one it drops follows the spelling —
    so the oracle here is arithmetic rather than another spelling, which would
    lose the same term and agree. At the floor ``p = q = 1`` with
    ``weight = (1, 3)``: ``Σ p² + p·weight`` is ``2 + 4``, and
    ``Σ (p + weight)(q + weight)`` is ``4 + 16``. Every model still builds and
    still solves with the term missing.
    """
    solved = lps.solve(model(expression, variables=FLOORED), SOURCES).objective
    assert solved == pytest.approx(optimum), f'{expression} lost one of its two mixed products'


def test_a_square_spreads_where_a_linear_cost_would_fill_the_cheapest_first():
    """The answer, not just the agreement — two lanes agreeing on a wrong
    number being what a differential test cannot see. A square puts the
    requirement on the *free* variables until they run out, then splits the
    rest evenly, which a linear objective would never do."""
    with lps.build(model('sum(p * p, over=g)'), SOURCES) as bound:
        result = bound.solve()
        assert result.objective == pytest.approx(0.0), 'the free variables carry it all'

    tight = {**SOURCES, 'need': pl.DataFrame({'value': [24.0]})}
    with lps.build(model('sum(p * p, over=g)'), tight) as bound:
        result = bound.solve()
        assert result.primal('p')['value'].to_list() == pytest.approx([2.0, 2.0]), (
            'a square spreads the remaining 4 evenly across the two columns; a linear cost would '
            'have filled one of them'
        )
        assert result.objective == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# the form the engine hands over
# ---------------------------------------------------------------------------


def test_a_pair_is_stored_once_whichever_order_it_was_written():
    """``x·y`` and ``y·x`` are one entry of one matrix — two rows survive a
    symmetric Hessian and not the LP section, which would carry the term twice
    at half weight."""
    written = quad_of('sum(p * q, over=g)')
    reversed_ = quad_of('sum(q * p, over=g)')
    assert written.equals(reversed_), 'the order the factors were written in is not part of the model'
    assert (written['col_l'] <= written['col_r']).all(), 'a pair is ordered by column index'


def test_the_same_pair_twice_is_summed_rather_than_repeated():
    """``p·p + p·p`` is ``2p²`` — one row, not two. The aggregate runs behind a
    probe, and a sink scatters by pair and would keep the last write."""
    once = quad_of('sum(p * p, over=g)')
    twice = quad_of('sum(p * p + p * p, over=g)')
    assert twice.height == once.height, 'a repeated pair collapses to one row'
    assert twice['coeff'].to_list() == pytest.approx([2 * c for c in once['coeff'].to_list()])


def test_the_stream_leaves_sorted_whatever_order_the_terms_were_written_in():
    """A contract, not tidiness. The stack is fragments in written order, so
    naming the higher-numbered variable first is enough to arrive unsorted —
    and two builds disagreeing makes `structure` read a moved *coefficient* as
    a moved pattern, reloading on every rebind that touches a quadratic
    cost."""
    backwards = quad_of('sum(q * q, over=g) + sum(p * p, over=g)')
    assert backwards['col_l'].to_list() == [0, 1, 2, 3], (
        'the quadratic stream leaves in (col_l, col_r) order however the expression was written'
    )


def test_a_zero_quadratic_coefficient_states_nothing_and_is_dropped():
    """Same rule as the matrix: absence already says it, and a nonzero costs
    the solver a Hessian entry to presolve away."""
    zeroed = {**SOURCES, 'weight': pl.DataFrame({'g': ['a', 'b'], 'value': [0.0, 3.0]})}
    assert quad_of('sum(p * p * weight, over=g)', zeroed).height == 1


def test_the_lp_section_doubles_every_coefficient(tmp_path):
    """The format divides the section by two, so the text is not the model —
    byte-asserted, since a *consistent* doubling error survives a round trip."""
    path = tmp_path / 'model.lp'
    lps.write(model('sum(p * p + p * q * weight, over=g)'), SOURCES, path)
    section = path.read_text().split('+ [')[1].split('] / 2')[0]
    assert '+2.0 x0 ^ 2' in section, 'a squared column doubles, and is spelled ^ 2 rather than x0 * x0'
    assert '+2.0 x0 * x2' in section, 'so does a cross term — uniformly, unlike the Hessian it is written from'


# ---------------------------------------------------------------------------
# absence, shape operators, and the degree ceiling
# ---------------------------------------------------------------------------


def test_a_quadratic_term_is_absent_where_either_factor_is():
    """Masked on one factor only, which is what makes the presence a
    conjunction rather than a copy: the term vanishes where ``q`` does."""
    masked = model('sum(p * p + p * q + q * q, over=g)')
    masked['variables'] = {**MODEL['variables'], 'q': {**MODEL['variables']['q'], 'where': 'weight > 2'}}
    with differential(masked, SOURCES, lp=True):
        pass

    with lps.build(masked, SOURCES) as bound:
        pairs = bound._engine._model.quad
        assert pairs.filter(pl.col('col_l') != pl.col('col_r')).height == 1, (
            "the cross term exists only where 'q' does — one coordinate of two"
        )


def test_absence_under_a_quadratic_term_reaches_its_siblings():
    """The conjunction where it is observable: a reduction sums where the whole
    summand exists, so a coordinate at which ``q`` is absent contributes
    neither the product nor the lone ``p`` beside it. Carrying only the left
    factor's presence keeps that ``p`` — one term heavier, and it still
    solves."""
    masked = model('sum(p * q + p, over=g)')
    masked['variables'] = {**MODEL['variables'], 'q': {**MODEL['variables']['q'], 'where': 'weight > 2'}}
    with lps.build(masked, SOURCES) as bound:
        objective = bound._engine._model.obj
        assert objective.height == 1, (
            "the lone 'p' survives only where 'q' does — a quadratic term is absent wherever "
            'either of its factors is, and that absence reaches the terms summed beside it'
        )


def test_a_pattern_that_moves_reloads_the_solver_rather_than_pushing():
    """The other half of the digest rule: a coefficient that moved is pushed, a
    *pair* that vanished is a different Hessian. Zeroing a weight drops its
    pair — a pattern change wearing a data change."""
    tight = {**SOURCES, 'need': pl.DataFrame({'value': [24.0]})}
    zeroed = {**tight, 'weight': pl.DataFrame({'g': ['a', 'b'], 'value': [0.0, 3.0]})}
    with lps.build(model('sum(p * p * weight, over=g)'), tight) as bound:
        bound.solve()
        bound.rebind(zeroed)
        bound.solve()
        assert bound.diagnostics().loads == 2, 'a pair that vanished is a model to load again'


def test_a_shape_operator_moves_a_quadratic_term_like_any_other():
    """A rewrite moves rows between coordinates and never reads what they
    carry: ``shift`` over a product puts two labels through the remap a linear
    fragment goes through, and both lanes still agree."""
    cyclic = {
        'parameters': {'need': {'dims': []}},
        'dimensions': {'g': {'dtype': 'str', 'values': ['a', 'b']}, 't': {'dtype': 'int', 'values': [0, 1, 2]}},
        'variables': {
            'p': {'foreach': ['g', 't'], 'bounds': {'lower': 0, 'upper': 10}},
            'q': {'foreach': ['g', 't'], 'bounds': {'lower': 0, 'upper': 10}},
        },
        'constraints': {'meet': {'foreach': [], 'expression': 'sum(sum(p, over=g), over=t) >= need'}},
        'objective': {
            'sense': 'minimize',
            'expression': "sum(sum(p * shift(p, over=t, offset=1, edge='wrap'), over=g), over=t) + "
            'sum(sum(p * p, over=g), over=t)',
        },
    }
    with differential(cyclic, {'need': SOURCES['need']}, lp=True):
        pass


def test_degree_three_is_refused_where_degree_two_is_taken():
    """The ceiling is 2 in the objective, not 'nonlinear is fine here'.

    Asked of ``check``, like its sibling below: every factor of ``p * p * p``
    is admissible at every step, so a cubic caught only at the build is one the
    gate let through — and the typesetter, which reads the same verdict, would
    have printed math no lane can build.
    """
    with pytest.raises(LpspecError, match='degree 3'):
        lps.check(model('sum(p * p * p, over=g)'))
    with pytest.raises(LpspecError, match='degree 3'):
        to_latex(model('sum(p * p * p, over=g)'))


def test_two_reductions_may_not_be_multiplied_even_in_the_objective():
    """The one shape "bilinear" hides that is genuinely out.

    ``sum(p) * sum(q)`` is every term of one against every term of the other,
    and the file says nothing about how many that is. It is also exactly where
    linopy's own ``*`` stops — it multiplies a multi-term expression by a
    single-term one and refuses two multi-term ones — so refusing it here is
    what keeps hard rule 3 structural instead of lucky.
    """
    with pytest.raises(LpspecError, match='sums of more than one term'):
        lps.check(model('sum(p, over=g) * sum(q, over=g)'))


def test_a_broadcast_product_is_not_an_outer_product():
    """What the rule above must *not* refuse.

    Factors carrying different dims broadcast, which is the fan-out every
    affine product already pays, and ``x[i] * y[j] * a[i, j]`` is the honest
    general bilinear form the ceiling doc admits. Each factor is one term at
    its coordinate, so the pairing is a join and not a cross product — the
    distinction the rule above turns on.
    """
    wide = {
        'dimensions': {'i': {'dtype': 'str', 'values': ['a', 'b']}, 'j': {'dtype': 'str', 'values': ['u', 'v']}},
        'parameters': {'link': {'dims': ['i', 'j']}},
        'variables': {
            'x': {'foreach': ['i'], 'bounds': {'lower': 0, 'upper': 10}},
            'y': {'foreach': ['j'], 'bounds': {'lower': 0, 'upper': 10}},
        },
        'constraints': {'meet': {'foreach': [], 'expression': 'sum(x, over=i) + sum(y, over=j) >= 4'}},
        'objective': {
            'sense': 'minimize',
            'expression': 'sum(sum(x * y * link, over=i), over=j)',
        },
    }
    lps.check(wide)


# ---------------------------------------------------------------------------
# what the sinks do with it
# ---------------------------------------------------------------------------


def test_a_quadratic_objective_beside_integrality_is_refused_before_the_build():
    """The exclusion, cashed: both halves are decidable with no data, so this
    is a `check` verdict rather than a surprise at `run()`."""
    integral = model('sum(p * p, over=g)')
    integral['variables'] = {**MODEL['variables'], 'p': {**MODEL['variables']['p'], 'domain': 'integer'}}

    with pytest.raises(LpspecError, match='separately and refuses them together'):
        lps.check(integral, sink='highs')
    with pytest.raises(LpspecError, match='separately and refuses them together'):
        lps.solve(integral, SOURCES)


def test_a_nonconvex_objective_is_refused_at_the_solve_and_still_writes(tmp_path):
    """The one capability verdict no data-free check can reach: `check` is
    silent by construction and HiGHS discovers it at `run()`. The error code
    must not reach the caller, and the file must still write."""
    concave = model('-sum(p * p, over=g)')
    lps.check(concave, sink='highs')

    with pytest.raises(LpspecError, match='not positive semidefinite'), lps.build(concave, SOURCES) as bound:
        bound.solve()

    path = tmp_path / 'nonconvex.lp'
    lps.write(concave, SOURCES, path)
    assert '-2.0 x0 ^ 2' in path.read_text(), 'the writer has no opinion about curvature'


def test_a_moved_quadratic_coefficient_is_pushed_rather_than_reloaded():
    """A second `passHessian` replaces `Q` and leaves the LP standing, so the
    *pattern* is structure and the values are not. Both halves: the answer
    moves, and the solver was not loaded twice."""
    heavier = {**SOURCES, 'weight': pl.DataFrame({'g': ['a', 'b'], 'value': [4.0, 12.0]})}
    tight = {**SOURCES, 'need': pl.DataFrame({'value': [24.0]})}
    tighter = {**heavier, 'need': pl.DataFrame({'value': [24.0]})}

    with lps.build(model('sum(p * p * weight, over=g)'), tight) as bound:
        first = bound.solve().objective
        bound.rebind(tighter)
        second = bound.solve().objective
        assert second == pytest.approx(4 * first), 'four times the curvature at the same optimum'
        assert bound.diagnostics().loads == 1, (
            'a coefficient that moved must not reload the solver — only the Hessian pattern is '
            'structure, and this one did not move'
        )
        assert bound.diagnostics().solves == 2
