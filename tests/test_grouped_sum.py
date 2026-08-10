"""sum: the transport YAML through both backends, and what coordinates buy.

Three-way differential on examples/transport.yaml:
  1. eager lpspec_linopy.build + solve (sum via linopy groupby)
  2. lowered Program -> PolarsExecutor -> the `highs` solver, plus the LP file
  3. hand-built indicator-matrix linopy model (an independent oracle that
     involves no sum at all)

Plus ``examples/monthly_budget.yaml``, which is the same primitive over *time*:
a coordinate on ``snapshot`` groups it into months exactly as a coordinate on
``generator`` groups onto buses. The gallery page quotes its dual and prints
its snapshot index, so a test has to hold both.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import lpspec as lps
from lpspec.errors import DataError, LanguageError
from lpspec.lowering import _lower_expr, lower_program
from lpspec.relational import PolarsExecutor
from lpspec.relational.plan import (
    Add,
    GroupSum,
    Negate,
    Variable,
)
from lpspec.sources import tidy_sources
from tests.conftest import override, resolved, schema_of
from tests.differential import RTOL, differential
from tests.oracle import lpspec_linopy, pd, transport_eager_objective, xr

TRANSPORT_YAML = Path('examples/transport.yaml')


def _inputs(gens, lines, load):
    data = {
        'p_max': gens.set_index('generator')['p_max'],
        'cost': gens.set_index('generator')['cost'],
        'cap': lines.set_index('line')['cap'],
        'neg_cap': -lines.set_index('line')['cap'],
        'load': xr.DataArray.from_series(load.set_index(['snapshot', 'bus'])['value']),
    }
    # the two dims carrying declared coordinates arrive as frames: the label
    # column plus one column per coordinate
    coords = {
        'snapshot': pd.Index(sorted(load['snapshot'].unique()), name='snapshot'),
        'generator': gens[['generator', 'bus']],
        'bus': pd.Index(sorted(load['bus'].unique()), name='bus'),
        'line': lines[['line', 'from_bus', 'to_bus']].rename(columns={'from_bus': 'from', 'to_bus': 'to'}),
    }
    return data, coords


def test_transport_yaml_agrees_with_an_independent_oracle(transport_data):
    gens, lines, load = transport_data
    data, coords = _inputs(gens, lines, load)

    # indicator matrices, no sum involved — an oracle for the oracle
    independent = transport_eager_objective(gens, lines, load)
    assert np.isfinite(independent)

    with differential(TRANSPORT_YAML, data, coords, lp=True) as run:
        assert run.oracle == pytest.approx(independent, rel=RTOL)


# ---------------------------------------------------------------------------
# lowering
# ---------------------------------------------------------------------------


def _flatten(expr):
    if isinstance(expr, Add):
        return _flatten(expr.left) + _flatten(expr.right)
    if isinstance(expr, Negate):
        return _flatten(expr.operand)
    return [expr]


def test_sum_lowers_to_one_node_per_injection_term():
    program = lower_program(schema_of(TRANSPORT_YAML))

    (c,) = program.constraints
    assert c.dims == ('snapshot', 'bus')
    terms = _flatten(c.lhs)
    assert GroupSum(Variable('p'), over='generator', coordinate='bus', into='bus') in terms
    assert GroupSum(Variable('f'), over='line', coordinate='to', into='bus') in terms


@pytest.mark.parametrize(
    ('expression', 'match'),
    [
        # an undeclared dim, or a coordinate the dim does not declare, is caught
        # in resolution before lowering ever sees the call
        ('sum(p, over=nope, group_by=bus)', r'over=nope\) does not name a declared dimension'),
        ('sum(p, over=generator, group_by=nope)', r"group_by=nope\) does not name a coordinate of 'generator'"),
        # a coordinate declared on a *different* dim is not in scope either
        ('sum(p, over=generator, group_by=to)', r"group_by=to\) does not name a coordinate of 'generator'"),
    ],
)
def test_a_name_sum_cannot_resolve_is_refused(expression, match):
    with pytest.raises(LanguageError, match=match):
        resolved(expression, schema_of(TRANSPORT_YAML))


def test_grouping_an_expression_that_lacks_the_dim_is_refused():
    """The names resolve and the arity fits, so what is left is a dim rule:
    lowering raises it by asking `dimensions`, not by restating it."""
    schema = schema_of(TRANSPORT_YAML)
    with pytest.raises(LanguageError, match='but the expression'):
        _lower_expr(resolved('sum(f, over=generator, group_by=bus)', schema), schema, 't')


# ---------------------------------------------------------------------------
# the hole a coordinate closes: a label that is not a coordinate of its target
# ---------------------------------------------------------------------------


def _relationally(data, coords):
    schema = schema_of(TRANSPORT_YAML)
    with PolarsExecutor() as ex:
        ex.build(lower_program(schema), tidy_sources(schema, data, coords))


def test_a_mistyped_coordinate_is_refused_on_both_lanes(transport_data):
    """Before coordinates were declared this built and solved: the mapping's
    value column was promoted to an index unchecked, and the inner join that
    places the terms dropped the generator out of its balance silently."""
    gens, lines, load = transport_data
    bad = gens.copy()
    bad.loc[bad.index[0], 'bus'] = 'nowhere'  # a bus that does not exist
    data, coords = _inputs(bad, lines, load)

    with pytest.raises(DataError, match="not 'bus' coordinates"):
        _relationally(data, coords)
    with pytest.raises(DataError, match="not 'bus' coordinates"):
        lpspec_linopy.build(TRANSPORT_YAML, data=data, coords=coords)


def test_a_coordinate_must_be_single_valued(transport_data):
    """Two rows disagreeing about a generator's bus is a data bug, not a
    silently-picked winner.

    Only the *index* is doubled here. Doubling the generator frame outright
    duplicates the parameters it also feeds, and the keyed-parameter check
    below catches that first — which is correct, and would leave this test
    asserting the wrong message.
    """
    gens, lines, load = transport_data
    other = 's' if gens['bus'].iloc[0] != 's' else 'n'
    data, coords = _inputs(gens, lines, load)
    coords['generator'] = pd.concat([coords['generator'], coords['generator'].head(1).assign(bus=other)])

    with pytest.raises(DataError, match='more than one value'):
        _relationally(data, coords)


def test_a_parameter_carrying_a_coordinate_twice_is_refused(transport_data):
    """A parameter is a function of its dims, so two rows for one coordinate
    has no answer — and the eager lane will not lay such a source out either.

    The relational lane used to resolve it into a sum, silently, which is a
    divergence between two lanes that are supposed to accept the same thing.
    Refusing it is also what lets the assembly skip its terminal aggregate:
    every parameter being keyed is the premise that argument rests on.
    """
    gens, lines, load = transport_data
    doubled = pd.concat([gens, gens.head(1)])

    with pytest.raises(DataError, match="parameter 'p_max' has more than one row"):
        _relationally(*_inputs(doubled, lines, load))


def test_a_coordinate_bearing_dim_needs_an_index_source(transport_data):
    """A coordinate cannot be inferred from the parameters that use the dim —
    inferring it is what would let a typo extend the label space."""
    gens, lines, load = transport_data
    data, coords = _inputs(gens, lines, load)
    del coords['generator']

    with pytest.raises(DataError, match='no index source'):
        _relationally(data, coords)


PARTIAL_YAML = """
dimensions:
  g: {dtype: str}
  item:
    dtype: str
    coords: {grp: g}
parameters:
  cap: {dims: [item]}
  target: {dims: [g]}
variables:
  x:
    foreach: [item]
    bounds: {lower: 0, upper: cap}
constraints:
  meet:
    foreach: [g]
    expression: sum(x, over=item, group_by=grp) >= target
objectives:
  obj:
    sense: minimize
    expression: sum(x, over=item)
"""


def _partial_inputs(grp_labels):
    """`item` carries coordinate `grp`; *grp_labels* is one label per item."""
    items = ['i0', 'i1', 'i2']
    index = pd.DataFrame({'item': items, 'grp': grp_labels})
    return (
        {  # relational sources
            'item': index,
            'g': pd.DataFrame({'g': ['g0']}),
            'cap': pd.DataFrame({'item': items, 'value': [5.0, 5.0, 5.0]}),
            'target': pd.DataFrame({'g': ['g0'], 'value': [3.0]}),
        },
        {  # eager data / coords
            'cap': pd.Series([5.0, 5.0, 5.0], index=pd.Index(items, name='item')),
            'target': pd.Series([3.0], index=pd.Index(['g0'], name='g')),
        },
        {'item': index, 'g': pd.Index(['g0'], name='g')},
    )


def test_a_partial_coordinate_places_its_orphans_nowhere(tmp_path):
    """A null coordinate means "this label is in no group", not "typo".

    Row absence is the language's idiom for "not present" everywhere else —
    an absent parameter row is a structural zero — and a coordinate is the one
    place it used to be an error. `i2` belongs to no group, so `sum`
    places its terms nowhere and only `i0`/`i1` can meet the target of 3.
    """
    path = tmp_path / 'partial.yaml'
    path.write_text(PARTIAL_YAML)
    sources, data, coords = _partial_inputs(['g0', 'g0', None])

    with lps.solve(path, sources) as result:
        assert result.is_ok
        assert result.objective == pytest.approx(3.0)
        # the orphan is still a variable; it just carries no group obligation
        assert result.to_pandas('x').set_index('item')['value']['i2'] == pytest.approx(0.0)

    model = lpspec_linopy.build(path, data=data, coords=coords)
    model.solve(solver_name='highs', output_flag=False)
    assert float(model.objective.value) == pytest.approx(3.0)


BROADCAST_GROUP_SUM = {
    'dimensions': {
        'snapshot': {'dtype': 'int', 'values': [0, 1]},
        'generator': {'dtype': 'str', 'coords': ['bus']},
        'bus': {'dtype': 'str'},
    },
    'parameters': {'w': {'dims': ['generator']}, 'limit': {'dims': ['snapshot', 'bus']}},
    'variables': {'x': {'foreach': ['snapshot'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {
        'cap': {
            'foreach': ['snapshot', 'bus'],
            'expression': 'sum(x * w, over=generator, group_by=bus) <= limit',
        }
    },
    'objectives': {'o': {'sense': 'maximize', 'expression': 'x'}},
}

#: g1 and g2 share a bus, so grouping merges two rows carrying the *same*
#: variable — which is the case a broadcast `over` creates and a `foreach` one
#: cannot.
BROADCAST_SOURCES = {
    'w': pl.DataFrame({'generator': ['g1', 'g2', 'g3'], 'value': [1.0, 2.0, 5.0]}),
    'limit': pl.DataFrame({'snapshot': [0, 0, 1, 1], 'bus': ['b1', 'b2'] * 2, 'value': [9.0, 100.0, 9.0, 100.0]}),
    'generator': pl.DataFrame({'generator': ['g1', 'g2', 'g3'], 'bus': ['b1', 'b2'] * 1 + ['b1']}),
    'bus': pl.DataFrame({'bus': ['b1', 'b2']}),
}


def test_sum_over_a_broadcast_dim_still_collapses_its_terms():
    """The variable does not carry the grouped dim, so a group holds it twice.

    `sum(x * w, over=generator, group_by=bus)` with `x` indexed by snapshot
    alone: `generator` reaches the fragment by broadcast from `w`, so two
    generators on one bus put the *same* `var_label` on one row. Nothing after
    this point can tell them apart — a solver handed a row with a column twice
    is entitled to reject the whole model, and HiGHS does.
    """
    sources = dict(
        BROADCAST_SOURCES,
        generator=pl.DataFrame({'generator': ['g1', 'g2', 'g3'], 'bus': ['b1', 'b1', 'b2']}),
    )
    with lps.build(BROADCAST_GROUP_SUM, sources) as ex:
        tables = ex._tables()
        matrix = tables.matrix_block(0, tables.row_count).sort('row', 'col')
        assert matrix.height == 4, 'a column appears twice on a row'
        assert matrix['coeff'].to_list() == [3.0, 5.0, 3.0, 5.0]  # 1.0 + 2.0 merged

        result = ex.solve()
    assert result.termination_condition == 'optimal'
    assert result.objective == pytest.approx(6.0)  # 3x <= 9 at b1, two snapshots


def test_sum_over_a_foreach_dim_needs_no_such_collapse():
    """The counterpart: when the variable carries the grouped dim, each merged
    row has its own label and there is nothing to add."""
    model = override(
        BROADCAST_GROUP_SUM,
        **{
            'variables.x.foreach': ['snapshot', 'generator'],
            'constraints.cap.expression': 'sum(x * w, over=generator, group_by=bus) <= limit',
        },
    )
    sources = dict(
        BROADCAST_SOURCES,
        generator=pl.DataFrame({'generator': ['g1', 'g2', 'g3'], 'bus': ['b1', 'b1', 'b2']}),
    )
    with lps.build(model, sources) as ex:
        tables = ex._tables()
        matrix = tables.matrix_block(0, tables.row_count).sort('row', 'col')
        # one entry per (row, generator-on-that-bus), not one per bus
        assert matrix.height == 6
        assert ex.solve().termination_condition == 'optimal'


# ---------------------------------------------------------------------------
# the objective's own key — the projection the matrix does not do
# ---------------------------------------------------------------------------

#: `y` is indexed by bus and `w` by snapshot, so `y * w` holds one row per
#: (bus, snapshot) and one *column* per bus. The fragment is legitimately
#: keyed on `(dims…, var_label)`; it is the objective's projection down to
#: `(col, coeff)` that drops the dims and merges those rows.
BROADCAST_OBJECTIVE = {
    'dimensions': {'snapshot': {'dtype': 'int', 'values': [0, 1, 2, 3]}, 'bus': {'dtype': 'str'}},
    'parameters': {'w': {'dims': ['snapshot']}, 'floor': {'dims': ['bus']}},
    'variables': {'y': {'foreach': ['bus'], 'bounds': {'lower': 0, 'upper': 100}}},
    'constraints': {'atleast': {'foreach': ['bus'], 'expression': 'y >= floor'}},
    'objectives': {'c': {'sense': 'minimize', 'expression': 'y * w'}},
}

BROADCAST_OBJECTIVE_SOURCES = {
    # deliberately unequal, so last-write-wins is not the same number as the sum
    'w': pl.DataFrame({'snapshot': [0, 1, 2, 3], 'value': [1.0, 10.0, 100.0, 1000.0]}),
    'floor': pl.DataFrame({'bus': ['b0', 'b1', 'b2'], 'value': [1.0, 2.0, 3.0]}),
    'bus': pl.DataFrame({'bus': ['b0', 'b1', 'b2']}),
}


def test_an_objective_term_carrying_dims_is_still_summed_per_column():
    """A coefficient is the *sum* over the dims the objective projects away.

    `keyed` is about `(dims…, var_label)`. The matrix keeps those dims — a
    constraint row is a function of dims that include the fragment's — so the
    key carries into `(row, col)`. The objective drops them, and a fragment
    that still carries one then holds several rows per column.

    Nothing downstream would fix it: the hand-off scatters with
    `dense[at] = values`, which keeps the last write rather than accumulating,
    so this reads as a plausible answer to a model nobody wrote.
    """
    with lps.build(BROADCAST_OBJECTIVE, BROADCAST_OBJECTIVE_SOURCES) as ex:
        obj = ex._tables().obj.sort('col')
        assert obj.height == 3, 'one row per column, not one per (bus, snapshot)'
        assert obj['coeff'].to_list() == [1111.0] * 3  # sum(w), not w[-1]


def test_the_broadcast_objective_agrees_with_the_eager_lane():
    """The same model end to end, against linopy — 6666.0, not 6000.0."""
    data = {
        'w': pd.Series([1.0, 10.0, 100.0, 1000.0], index=pd.Index([0, 1, 2, 3], name='snapshot')),
        'floor': pd.Series([1.0, 2.0, 3.0], index=pd.Index(['b0', 'b1', 'b2'], name='bus')),
    }
    coords = {'bus': pd.Index(['b0', 'b1', 'b2'], name='bus')}
    with differential(BROADCAST_OBJECTIVE, data, coords, lp=True) as run:
        assert run.oracle == pytest.approx(6666.0)


def test_an_objective_whose_dims_are_all_the_variables_own_still_skips_it():
    """The counterpart, and the reason the test is `label_dims` not `dims`.

    `p * cost` reaches the objective carrying `(snapshot, generator)` — it is
    never wrapped in a `sum` — and both are `p`'s own dims, so `var_label`
    determines them and no column can repeat. Refusing every fragment that
    merely *has* dims would be sound and would re-enable the aggregate on every
    model in `bench/`, which is the whole optimisation (#161).
    """
    model = override(BROADCAST_OBJECTIVE, **{'objectives.c.expression': 'y * floor'})
    with lps.build(model, BROADCAST_OBJECTIVE_SOURCES) as ex:
        obj = ex._tables().obj.sort('col')
        assert obj.height == 3
        assert obj['coeff'].to_list() == [1.0, 2.0, 3.0]  # floor itself, un-summed


# ---------------------------------------------------------------------------
# the same construct, grouping time
# ---------------------------------------------------------------------------

MONTHLY_YAML = Path('examples/monthly_budget.yaml')
MONTHLY_PAGE = Path('docs/models/monthly_budget.md')


def _monthly_sources():
    """Six snapshots over three calendar months, wind capped in the first.

    The `month` column is data prep — one polars expression — which is the
    page's whole point: the language never learns what a calendar is.
    """
    import datetime as dt

    hours = [dt.datetime(2030, 1, 1) + dt.timedelta(days=15 * i) for i in range(6)]
    index = pl.DataFrame({'snapshot': hours}).with_columns(pl.col('snapshot').dt.strftime('%Y-%m').alias('month'))
    months = sorted(set(index['month']))
    gens = ['wind', 'gas']
    return (
        index,
        months,
        {
            'snapshot': index,
            'month': pl.DataFrame({'month': months}),
            'p_max': pl.DataFrame({'generator': gens, 'value': [10.0, 100.0]}),
            'cost': pl.DataFrame({'generator': gens, 'value': [1.0, 50.0]}),
            'load': pl.DataFrame({'snapshot': hours, 'value': [20.0] * 6}),
            'monthly_cap': pl.DataFrame(
                {
                    'month': [m for m in months for _ in gens],
                    'generator': gens * len(months),
                    'value': [5.0 if (m == months[0] and g == 'wind') else 1e4 for m in months for g in gens],
                }
            ),
        },
    )


def test_a_monthly_budget_binds_and_prices_itself():
    """The number the gallery page quotes, held by a test.

    January caps wind at 5 where three snapshots could carry 30, so the cap
    binds and its shadow price is the cost of covering that energy with gas
    instead — 50 against 1. February and March are slack and price at zero,
    which is what distinguishes a binding budget from a decorative one.
    """
    index, _months, sources = _monthly_sources()
    with lps.solve(MONTHLY_YAML, sources) as result:
        assert result.is_ok
        wind = (
            result.primal('p')
            .filter(pl.col('generator') == 'wind')
            .join(index, on='snapshot')
            .group_by('month')
            .agg(pl.col('value').sum())
            .sort('month')
        )
        # 3 snapshots in Jan (capped at 5), 1 in Feb, 2 in Mar — unequal groups
        assert wind['value'].to_list() == pytest.approx([5.0, 10.0, 20.0])

        duals = result.dual('monthly_budget').filter(pl.col('generator') == 'wind').sort('month')
        assert duals['value'].to_list() == pytest.approx([-49.0, 0.0, 0.0])


def test_the_monthly_grouping_is_a_column_and_nothing_else():
    """Re-grouping the same snapshots re-states the budget, model untouched.

    Quarters instead of months: one different column in the snapshot index,
    and the constraint now spans three-month blocks. That is the claim the
    page makes about weeks, seasons and representative periods, checked once.
    """
    index, _months, sources = _monthly_sources()
    quarters = index.with_columns(pl.lit('2030-Q1').alias('month')).select('snapshot', 'month')
    regrouped = {
        **sources,
        'snapshot': quarters,
        'month': pl.DataFrame({'month': ['2030-Q1']}),
        'monthly_cap': pl.DataFrame({'month': ['2030-Q1'] * 2, 'generator': ['wind', 'gas'], 'value': [5.0, 1e4]}),
    }
    with lps.solve(MONTHLY_YAML, regrouped) as result:
        assert result.is_ok
        assert result.dual('monthly_budget').height == 2, 'one row per group, and there is now one group'
        wind = result.primal('p').filter(pl.col('generator') == 'wind')['value'].sum()
        assert wind == pytest.approx(5.0), 'the cap now binds across the whole quarter'


def test_a_mistyped_month_is_a_typo_and_not_a_new_group():
    """Why the target of a coordinate has to be a declared dimension.

    Without one there is nothing to check the snapshot index against, and
    `2030-3` beside `2030-03` would quietly become a fourth group with a budget
    of its own — the model then solves a smaller problem and says nothing. The
    same check catches a generator assigned to a bus that does not exist.
    """
    index, _months, sources = _monthly_sources()
    typo = index.with_columns(
        pl.when(pl.col('month') == '2030-03').then(pl.lit('2030-3')).otherwise(pl.col('month')).alias('month')
    )
    with pytest.raises(DataError, match=r"coordinate 'month' has value\(s\) that are not 'month' coordinates"):
        lps.solve(MONTHLY_YAML, {**sources, 'snapshot': typo})


def test_the_index_the_page_prints_is_the_index_it_solves():
    """The frame printed on the page is the frame these tests build.

    `test_doc_examples.py` sweeps `python` and `yaml` fences and runs neither,
    so a pasted *output* block is the one kind of doc claim nothing checks —
    change the timestamps here and the page would keep printing the old ones.
    Defaults are restored while formatting, so a contributor's `POLARS_FMT_*`
    environment cannot fail this.
    """
    index, _months, _sources = _monthly_sources()
    fences = re.findall(r'^```text\n(.*?)^```', MONTHLY_PAGE.read_text(), re.MULTILINE | re.DOTALL)
    printed = [block for block in fences if block.startswith('shape: (')]
    assert len(printed) == 1, 'the page prints exactly one frame'
    with pl.Config(restore_defaults=True):
        assert printed[0].rstrip('\n') == str(index)
