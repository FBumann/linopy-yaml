"""The compiler is lazy, and this file is the proof.

No solver, no data, not one row read — a plan node goes in and a query comes
out. That is the seam the split bought: checking what an operator does costs a
compile, not a build and a solve.

**This is the only place query *shape* is asserted**, which is what the
hand-built fixture below buys. Every property here can regress while the whole
suite still passes and every model still solves to the right answer, so no
end-to-end test stands in for it:

- ``AGGREGATE`` absent from ``Sum`` and ``GroupSum``. They *project*; duplicates
  collapse once, in the terminal ``SUM(coeff) GROUP BY row, col`` at assembly.
  Make either of them aggregate and the answers stay right while the single
  place duplicates are meant to collapse quietly becomes two.
- ``OVER`` absent from a translation, which joins the dim table twice instead.
  A window function answers correctly and gives up bounded-halo locality.
- the modulo appearing only when a translation wraps.
- a dimension comparison *filtering* a column the frame already carries rather
  than joining to find it, and a constant bound costing no join at all.

The frames are declared as **empty frames with the right schemas**, and that is
the purity claim itself rather than a convenience: a lazy frame is a plan, so a
schema is all it takes to compile one. It cannot be checked any other way —
reach the compiler through the executor and it needs rows, at which point the
demonstration that a schema suffices has evaporated.

The assertions are deliberately about shape, not exact text, so the query
planner stays free to change underneath them.
"""

from __future__ import annotations

import polars as pl
import pytest

from lpspec.errors import LanguageError
from lpspec.relational import plan
from lpspec.relational.engines.polars.binding import BoundSources
from lpspec.relational.engines.polars.compiler import PolarsCompiler

PROGRAM = plan.Program(
    parameters=(
        plan.ParameterDeclaration('cost', ('generator',)),
        plan.ParameterDeclaration('load', ('snapshot',)),
        plan.ParameterDeclaration('available', ('generator',)),
    ),
    variables=(plan.VariableDeclaration('p', ('snapshot', 'generator')),),
    constraints=(),
    objective=plan.ObjectiveDeclaration('min', plan.Variable('p')),
    dimensions=(
        plan.DimensionDeclaration('snapshot'),
        plan.DimensionDeclaration('generator', coordinates=(('bus', 'bus'),)),
        plan.DimensionDeclaration('bus'),
    ),
)

CARDINALITY = {'snapshot': 24, 'generator': 3, 'bus': 2}

DIMENSIONS = {
    'snapshot': pl.LazyFrame(schema={'val': pl.Int64, 'ord': pl.Int64}),
    'generator': pl.LazyFrame(schema={'val': pl.String, 'ord': pl.Int64, 'bus': pl.String}),
    'bus': pl.LazyFrame(schema={'val': pl.String, 'ord': pl.Int64}),
}
PARAMETERS = {
    'cost': pl.LazyFrame(schema={'generator': pl.String, 'value': pl.Float64}),
    'load': pl.LazyFrame(schema={'snapshot': pl.Int64, 'value': pl.Float64}),
    'available': pl.LazyFrame(schema={'generator': pl.String, 'value': pl.Float64}),
}
VARIABLES = {'p': pl.LazyFrame(schema={'snapshot': pl.Int64, 'generator': pl.String, 'var_label': pl.Int64})}


def bound(boolean_parameters: frozenset[str] = frozenset()) -> BoundSources:
    """The data a query is written against — schemas only, no rows.

    Compiling reads nothing, so an empty frame of the right schema is a whole
    fixture (docs/ARCHITECTURE.md's admissibility test).
    """
    return BoundSources(
        parameters=PARAMETERS,
        dimensions=DIMENSIONS,
        cardinality=CARDINALITY,
        boolean_parameters=boolean_parameters,
    )


def compiler(boolean_parameters: frozenset[str] = frozenset()) -> PolarsCompiler:
    return PolarsCompiler(PROGRAM, bound(boolean_parameters), VARIABLES)


def columns(frame: pl.LazyFrame) -> list[str]:
    return frame.collect_schema().names()


def query(frame: pl.LazyFrame) -> str:
    """The query plan as text — what an admissibility judgement is read off."""
    return frame.explain(optimized=False)


def joins(frame: pl.LazyFrame) -> int:
    """How many joins the plan performs.

    Counted on the header polars prints for each one (``INNER JOIN:``,
    ``LEFT JOIN:``); the matching ``END … JOIN`` carries no colon, so this
    counts joins rather than lines mentioning one.
    """
    return query(frame).count('JOIN:')


# ---------------------------------------------------------------------------
# expressions
# ---------------------------------------------------------------------------


def test_a_variable_compiles_to_one_term_fragment_over_its_dims():
    compiled = compiler().expression(plan.Variable('p'), 'test')
    assert len(compiled.terms) == 1
    assert not compiled.consts
    fragment = compiled.terms[0]
    assert fragment.dims == ('snapshot', 'generator')
    assert fragment.is_term
    assert columns(fragment.frame) == ['snapshot', 'generator', 'var_label', 'coeff']


def test_a_parameter_is_a_constant_fragment_not_a_term():
    compiled = compiler().expression(plan.Parameter('cost'), 'test')
    assert not compiled.terms
    assert compiled.consts[0].dims == ('generator',)
    assert not compiled.consts[0].is_term
    assert columns(compiled.consts[0].frame) == ['generator', 'cval']


def test_addition_concatenates_fragments_rather_than_joining():
    """An LP row is a sum of terms, so ``+`` needs no query at all."""
    compiled = compiler().expression(plan.Variable('p') + plan.Variable('p'), 'test')
    assert len(compiled.terms) == 2


def test_multiplying_a_variable_by_a_parameter_joins_on_the_shared_dim():
    compiled = compiler().expression(plan.Multiply(plan.Variable('p'), plan.Parameter('cost')), 'test')
    fragment = compiled.terms[0]
    assert fragment.dims == ('snapshot', 'generator')
    assert columns(fragment.frame) == ['snapshot', 'generator', 'var_label', 'coeff']
    assert 'JOIN' in query(fragment.frame)


def test_a_product_of_two_variable_carrying_factors_is_refused():
    with pytest.raises(LanguageError, match='nonlinear product'):
        compiler().expression(plan.Multiply(plan.Variable('p'), plan.Variable('p')), 'test')


def test_a_divisor_carrying_variables_is_refused():
    with pytest.raises(LanguageError, match='nonlinear quotient'):
        compiler().expression(plan.Divide(plan.Variable('p'), plan.Variable('p')), 'test')


# ---------------------------------------------------------------------------
# shape operators — each rewrites exactly one dim column
# ---------------------------------------------------------------------------


def test_sum_drops_the_dim_it_sums_over_without_aggregating():
    """The aggregate lives in the terminal assembly, not in the fragment —
    which is what keeps the operator pointwise."""
    compiled = compiler().expression(plan.Sum(plan.Variable('p'), ('generator',)), 'test')
    fragment = compiled.terms[0]
    assert fragment.dims == ('snapshot',)
    assert columns(fragment.frame) == ['snapshot', 'var_label', 'coeff']
    assert 'AGGREGATE' not in query(fragment.frame)


def masked_compiler() -> PolarsCompiler:
    """A compiler over two masked variables, so fragments carry presence.

    Two, because a restriction only ever crosses from one fragment to another —
    with a single masked term there is nothing for absence to propagate *to*.
    """
    over = ('snapshot', 'generator')
    where = plan.ParameterComparison('available', '>', 0.0)
    masked = plan.Program(
        parameters=PROGRAM.parameters,
        variables=(
            plan.VariableDeclaration('p', over, where=where),
            plan.VariableDeclaration('q', over, where=where),
        ),
        constraints=(),
        objective=PROGRAM.objective,
        dimensions=PROGRAM.dimensions,
    )
    frames = dict(VARIABLES, q=VARIABLES['p'])
    return PolarsCompiler(masked, bound(), frames)


def test_a_reduction_carries_absence_between_fragments_and_not_into_the_one_it_came_from():
    """`sum(p + q)` sums where each exists, and neither is checked against itself.

    Which is the whole of what "each one's absence says nothing about the
    other" means: the *other*. A fragment's rows and its presence come from one
    frame and are rewritten in step, so restricting a fragment by its own
    coordinates can only return the rows it was given. Under a mask over a
    single term — the ordinary case — that made the pass a semi-join of a
    frame against itself, and no assertion about the answer can see it, since
    the answer is the same frame.
    """
    both = masked_compiler().expression(
        plan.Sum(plan.Add(plan.Variable('p'), plan.Variable('q')), ('generator',)), 'test'
    )
    assert [joins(t.frame) for t in both.terms] == [1, 1]
    assert all('SEMI JOIN' in query(t.frame) for t in both.terms)

    alone = masked_compiler().expression(plan.Sum(plan.Variable('p'), ('generator',)), 'test')
    assert joins(alone.terms[0].frame) == 0


def test_a_reduction_restricts_by_existence_and_does_not_deduplicate():
    """A semi-join asks whether a key occurs, so nothing distinguishes first.

    A distinct on the right of one changes no row: occurring twice is still
    occurring. It costs a hash pass over every coordinate the variable has,
    which on `dispatch/l` was a third of the restriction — and it is invisible
    from the answer, since both plans return the same frame.
    """
    compiled = masked_compiler().expression(
        plan.Sum(plan.Add(plan.Variable('p'), plan.Variable('q')), ('generator',)), 'test'
    )
    assert 'UNIQUE' not in query(compiled.terms[0].frame)


def test_sum_over_an_absent_dim_scales_by_that_dims_cardinality():
    """Eager parity: summing a snapshot-only term over `generator` repeats it."""
    inner = plan.Sum(plan.Variable('p'), ('generator',))
    compiled = compiler().expression(plan.Sum(inner, ('generator',)), 'test')
    assert '3' in query(compiled.terms[0].frame)


def test_sum_swaps_the_source_dim_for_the_target_and_emits_no_aggregate():
    node = plan.GroupSum(plan.Variable('p'), over='generator', coordinate='bus', into='bus')
    fragment = compiler().expression(node, 'test').terms[0]
    assert fragment.dims == ('snapshot', 'bus')
    assert columns(fragment.frame) == ['snapshot', 'bus', 'var_label', 'coeff']
    assert 'AGGREGATE' not in query(fragment.frame)
    assert joins(fragment.frame) == 1


def test_translate_keeps_its_dims_and_joins_the_dim_table_twice():
    """Bounded halo: a row at ord *o* lands at ord *o + by*, no window."""
    fragment = compiler().expression(plan.Translate(plan.Variable('p'), 'snapshot', by=1), 'test').terms[0]
    assert fragment.dims == ('snapshot', 'generator')
    assert columns(fragment.frame) == ['generator', 'snapshot', 'var_label', 'coeff']
    assert joins(fragment.frame) == 2
    assert 'OVER' not in query(fragment.frame)


def test_wrapping_is_modulo_and_acyclic_is_not():
    cyclic = compiler().expression(plan.Translate(plan.Variable('p'), 'snapshot', by=1, wrap=True), 't').terms[0]
    acyclic = compiler().expression(plan.Translate(plan.Variable('p'), 'snapshot', by=1, wrap=False), 't').terms[0]
    assert '%' in query(cyclic.frame)
    assert '%' not in query(acyclic.frame)


def test_a_shape_operator_along_a_dim_the_expression_lacks_is_refused():
    with pytest.raises(LanguageError, match='translation'):
        compiler().expression(plan.Translate(plan.Parameter('cost'), 'snapshot', by=1), 'test')


# ---------------------------------------------------------------------------
# predicates
# ---------------------------------------------------------------------------


def test_a_dimension_comparison_filters_a_column_already_in_the_frame():
    """Pointwise, and free: no table is read to decide it."""
    frame = compiler().frame(('snapshot',), plan.DimensionComparison('snapshot', '>', 0))
    text = query(frame)
    assert 'FILTER' in text
    assert 'JOIN' not in text


def test_a_parameter_predicate_needs_a_left_join():
    frame = compiler().frame(('generator',), plan.ParameterDefined('available'))
    text = query(frame)
    assert 'JOIN' in text
    assert 'FILTER' in text


def test_defined_on_a_boolean_parameter_tests_the_value_not_its_finiteness():
    numeric = query(compiler().frame(('generator',), plan.ParameterDefined('available')))
    boolean = query(compiler(frozenset({'available'})).frame(('generator',), plan.ParameterDefined('available')))
    assert 'is_finite' in numeric
    assert 'is_finite' not in boolean


def test_a_where_parameter_outside_the_frame_dims_is_refused():
    """Otherwise the mask would be reduced over a dim the declaration never named."""
    with pytest.raises(LanguageError, match='outside the foreach dims'):
        compiler().frame(('generator',), plan.ParameterDefined('load'))


# ---------------------------------------------------------------------------
# frames and bounds
# ---------------------------------------------------------------------------


def test_a_frame_cross_joins_its_dim_tables_and_carries_their_ordinals():
    frame = compiler().frame(('snapshot', 'generator'), None)
    assert columns(frame) == ['snapshot', '__ord snapshot__', 'generator', '__ord generator__']
    assert 'CROSS' in query(frame)


def test_an_unmasked_frame_has_nothing_to_filter():
    assert 'FILTER' not in query(compiler().frame(('snapshot', 'generator'), None))


def test_a_parameter_bound_joins_on_the_variable_frame():
    variable = plan.VariableDeclaration('p', ('snapshot', 'generator'), upper=plan.Parameter('cost'))
    bounded = compiler().bounds(VARIABLES['p'], variable)
    assert {'lb', 'ub'} <= set(columns(bounded))
    assert joins(bounded) == 1


def test_a_constant_bound_needs_no_join_at_all():
    bounded = compiler().bounds(VARIABLES['p'], PROGRAM.variables[0])
    assert {'lb', 'ub'} <= set(columns(bounded))
    assert joins(bounded) == 0


def test_a_bound_carrying_a_variable_is_refused():
    variable = plan.VariableDeclaration('p', ('snapshot', 'generator'), upper=plan.Variable('p'))
    with pytest.raises(LanguageError, match='bounds must be variable-free'):
        compiler().bounds(VARIABLES['p'], variable)


# ---------------------------------------------------------------------------
# the variable a fragment names
# ---------------------------------------------------------------------------


def test_a_term_names_its_variable_through_every_operator():
    """The name travels, because absence is read off the variable underneath.

    A fragment carries the coordinates its *variable* exists at beside its own
    rows (`presence`), and which declaration that is has to survive every
    reshaping for the constraint rows to be restricted by it.
    """
    for node in (
        plan.Variable('p'),
        plan.Sum(plan.Variable('p'), ('generator',)),
        plan.GroupSum(plan.Variable('p'), over='generator', coordinate='bus', into='bus'),
        plan.Translate(plan.Variable('p'), 'snapshot', by=1),
        plan.Multiply(plan.Variable('p'), plan.Parameter('cost')),
        plan.Divide(plan.Variable('p'), plan.Parameter('cost')),
        -plan.Variable('p'),
    ):
        assert compiler().expression(node, 'test').terms[0].variable == 'p', node

    assert compiler().expression(plan.Parameter('cost'), 'test').consts[0].variable is None
