"""``sos:`` — one construct, three sinks, and one of them without the concept.

The claim has two halves and they need different oracles. That the *language*
means the same thing on both lanes is `differential`'s job as usual. That the
**reformulation** is the same feasible set is not something either lane can
say, since both would be reformulating: so the optimum is enumerated here
(:func:`best`), which is tractable because a set over four options has nine
admissible shapes, and asserted against every sink.

The model is purpose-built for that enumeration: two sites choosing among four
sizes, values and caps varying enough that SOS1, SOS2 and the unrestricted LP
all have *different* optima — a set that does not bind proves nothing about a
formulation of it.
"""

from __future__ import annotations

import itertools
from typing import Any

import polars as pl
import pytest

import lpspec as lps
from lpspec.errors import DataError, LanguageError, LpspecError
from lpspec.relational.sinks import sos as sos_sink
from tests.conftest import solve_lp_file

SITES = ['north', 'south']
SIZES = [0, 1, 2, 3]

#: Value per unit and how much of it there is, by ``(site, size)``. Chosen so
#: the three regimes separate: unrestricted takes everything (39), SOS2 takes
#: an adjacent pair (26), SOS1 takes one member (14).
VALUE = {
    ('north', 0): 1.0, ('north', 1): 3.0, ('north', 2): 2.0, ('north', 3): 2.5,
    ('south', 0): 5.0, ('south', 1): 1.0, ('south', 2): 4.0, ('south', 3): 0.5,
}  # fmt: skip
CAP = {
    ('north', 0): 4.0, ('north', 1): 2.0, ('north', 2): 3.0, ('north', 3): 1.0,
    ('south', 0): 1.0, ('south', 1): 6.0, ('south', 2): 2.0, ('south', 3): 3.0,
}  # fmt: skip


def _table(values: dict[tuple[str, int], float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            'site': [site for site, _ in values],
            'size': [size for _, size in values],
            'value': list(values.values()),
        }
    )


DATA = {'value': _table(VALUE), 'cap': _table(CAP)}

BASE: dict[str, Any] = {
    'dimensions': {'site': {'values': SITES}, 'size': {'dtype': 'int', 'values': SIZES}},
    'parameters': {'value': {'dims': ['site', 'size']}, 'cap': {'dims': ['site', 'size']}},
    'variables': {'take': {'foreach': ['site', 'size'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'objectives': {
        'total': {'sense': 'maximize', 'expression': 'sum(sum(take * value, over=site), over=size)'},
    },
}


def model(sos_type: int, **sos: Any) -> dict[str, Any]:
    """The base model with one set over ``take``'s ``size`` dim."""
    return BASE | {'sos': {'pick': {'variable': 'take', 'over': 'size', 'type': sos_type, **sos}}}


def best(sos_type: int | None, sizes: list[int] = SIZES) -> float:
    """The optimum by enumeration — the oracle neither lane can be.

    Every set is independent (nothing couples the sites), and a set admits a
    known family of nonzero patterns: any single member for SOS1, plus any
    consecutive pair for SOS2. Taking each admitted member at its cap is
    optimal because every value is positive.
    """
    if sos_type is None:
        patterns = [tuple(sizes)]
    else:
        patterns = [(size,) for size in sizes]
        if sos_type == 2:
            patterns += list(itertools.pairwise(sizes))
    return sum(
        max(sum(VALUE[site, size] * CAP[site, size] for size in pattern) for pattern in patterns) for site in SITES
    )


def test_the_three_regimes_have_different_optima():
    """The premise every assertion below rests on, stated as a test.

    A set that does not bind is satisfied by ignoring it, so a model where the
    LP optimum already respects the set would pass every test here with the
    reformulation deleted.
    """
    assert (best(None), best(2), best(1)) == (39.0, 26.0, 14.0), 'the enumeration no longer separates the regimes'


# ---------------------------------------------------------------------------
# the language
# ---------------------------------------------------------------------------


#: A dim the model declares and ``take`` does not carry, so a set over it has
#: no members — the one case needing a declaration the base model lacks.
UNCARRIED = {'dimensions': BASE['dimensions'] | {'other': {'values': ['x']}}}


@pytest.mark.parametrize(
    ('blocks', 'expected', 'also'),
    [
        pytest.param(
            {'pick': {'variable': 'nope', 'over': 'size'}}, "'nope' is not a declared variable", {}, id='no-var'
        ),
        pytest.param({'pick': {'variable': 'take', 'over': 'site2'}}, "undeclared dimension 'site2'", {}, id='no-dim'),
        pytest.param(
            {'pick': {'variable': 'take', 'over': 'other'}},
            "over 'other' is not a dim of variable 'take'",
            UNCARRIED,
            id='a-dim-the-variable-does-not-carry',
        ),
        pytest.param(
            {'pick': {'variable': 'take', 'over': 'size'}, 'again': {'variable': 'take', 'over': 'size', 'type': 2}},
            "already carries the set declared by 'pick'",
            {},
            id='two-sets-over-one-variable',
        ),
        pytest.param(
            {'pick': {'variable': 'take', 'over': 'size', 'type': 3}}, 'sos type must be 1 or 2', {}, id='third-order'
        ),
        pytest.param(
            {'pick': {'variable': 'take', 'over': 'size', 'big_m': 0}},
            'big_m must be a positive, finite number',
            {},
            id='zero-big-m',
        ),
        pytest.param(
            {'pick': {'variable': 'take', 'over': 'size', 'kind': 1}}, "unknown key 'kind'", {}, id='unknown-key'
        ),
    ],
)
def test_a_malformed_set_is_a_load_error_naming_it(blocks, expected, also):
    """Everything decidable without data, in one table.

    ``type: 1`` is filled in where the case is not about it, so each row shows
    only what it is testing.
    """
    filled = {name: {'type': 1} | block for name, block in blocks.items()}
    with pytest.raises(LanguageError, match=expected):
        lps.check(BASE | also | {'sos': filled})


# ---------------------------------------------------------------------------
# what the sinks do with it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('sos_type', [1, 2], ids=['sos1', 'sos2'])
def test_both_lanes_and_the_enumeration_agree(sos_type):
    """The differential claim, plus the one oracle that is nobody's code.

    The harness is imported **inside the test**: importing it is the
    ``[linopy]`` guard, and everything else here is the streaming lane's own
    and has to keep running on the bare install. ``lp=False`` because HiGHS
    reads the written file back and its parser refuses the section, which is
    the capability finding rather than a defect (pinned below).
    """
    from tests.differential import differential
    from tests.oracle import xr

    eager = {
        name: xr.DataArray.from_series(_table(v).to_pandas().set_index(['site', 'size'])['value'])
        for name, v in (('value', VALUE), ('cap', CAP))
    }
    with differential(model(sos_type), eager) as run:
        assert run.result.objective == pytest.approx(best(sos_type)), 'the set does not restrict what it claims to'


@pytest.mark.parametrize('sos_type', [1, 2], ids=['sos1', 'sos2'])
def test_the_reformulated_solution_is_a_member_of_the_set(sos_type):
    """An optimum can be right while the formulation admits shapes it should
    not, so the pattern itself is checked."""
    result = lps.solve(model(sos_type), DATA)
    taken = result.primal('take')
    for site in SITES:
        nonzero = taken.filter((pl.col('site') == site) & (pl.col('value') > 1e-9))['size'].to_list()
        assert len(nonzero) <= sos_type, f'{site} has {len(nonzero)} nonzero members in an SOS{sos_type} set'
        if sos_type == 2 and len(nonzero) == 2:
            assert nonzero[1] - nonzero[0] == 1, f'{site} took two members that are not consecutive'


@pytest.mark.parametrize('sos_type', [1, 2], ids=['sos1', 'sos2'])
def test_the_native_sink_reaches_the_same_optimum(sos_type):
    """Gurobi branches on the set; HiGHS is handed binaries. One answer."""
    pytest.importorskip('gurobipy', reason='the native SOS path needs the [gurobi] extra')
    assert lps.solve(model(sos_type), DATA, 'gurobi').objective == pytest.approx(best(sos_type))


@pytest.mark.parametrize('sos_type', [1, 2], ids=['sos1', 'sos2'])
def test_the_lp_file_carries_the_set_and_a_reader_agrees(sos_type, tmp_path):
    """The section, in label order, and read back by a solver that takes it."""
    path = lps.write(model(sos_type), DATA, tmp_path / 'model.lp')
    text = path.read_text()

    assert '\nsos\n' in text, 'the LP file dropped the set entirely'
    section = text.split('\nsos\n')[1].split('\nend')[0].strip().splitlines()
    assert section == [
        f's0: S{sos_type} :: x0:1 x1:2 x2:3 x3:4',
        f's1: S{sos_type} :: x4:1 x5:2 x6:3 x7:4',
    ], 'the section is not the sets in label order, weighted by declared position'

    gurobipy = pytest.importorskip('gurobipy', reason='no shipped solver but gurobi reads an SOS section')
    with gurobipy.Env(params={'OutputFlag': 0}) as env:
        read = gurobipy.read(str(path), env=env)
        read.optimize()
        assert read.ObjVal == pytest.approx(best(sos_type)), 'the written file is not the model that was built'
        read.dispose()


def test_highs_refuses_the_written_section_which_is_why_it_reformulates(tmp_path):
    """The capability finding itself, pinned rather than described.

    If HiGHS ever grows an SOS concept this fails, and the sink's
    ``sos = 'reformulated'`` is what should change.
    """
    path = lps.write(model(1), DATA, tmp_path / 'model.lp')
    with pytest.raises(AssertionError):
        solve_lp_file(path)


# ---------------------------------------------------------------------------
# the reformulation's own conditions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('bounds', 'expected'),
    [
        pytest.param({'lower': 0}, 'no upper bound and no big_m', id='nothing-to-link-with'),
        pytest.param({'lower': -1, 'upper': 'cap'}, 'negative lower bound', id='a-member-that-can-go-negative'),
    ],
)
def test_a_member_a_big_m_cannot_stand_in_for_is_refused(bounds, expected):
    raw = model(1)
    raw['variables'] = {'take': {'foreach': ['site', 'size'], 'bounds': bounds}}
    with pytest.raises(DataError, match=expected):
        lps.solve(raw, DATA)


def test_a_big_m_stands_in_for_the_missing_bound():
    """What the refusal names as the fix, taken — and the optimum follows it,
    which is what makes `big_m` a statement rather than a knob."""
    raw = model(1, big_m=2.0)
    raw['variables'] = {'take': {'foreach': ['site', 'size'], 'bounds': {'lower': 0}}}
    result = lps.solve(raw, DATA)
    assert result.objective == pytest.approx(2.0 * 3.0 + 2.0 * 5.0), 'the optimum does not follow the declared big-M'


def test_the_tighter_of_the_bound_and_big_m_is_the_coefficient():
    """``M = min(big_m, ub)``, linopy's rule — a looser one is a worse search."""
    with lps.build(model(1, big_m=2.5), DATA) as bound:
        tables = sos_sink.reformulated(bound._engine._tables())
    used = sorted({-coeff for coeff in tables.matrix['coeff'].to_list() if coeff < 0})
    assert used == [1.0, 2.0, 2.5], 'a member whose bound is looser than big_m did not take big_m'


def test_the_refusals_do_not_reach_the_sinks_that_need_neither(tmp_path):
    """An unbounded member is a *reformulation* condition, not a language one."""
    raw = model(1)
    raw['variables'] = {'take': {'foreach': ['site', 'size'], 'bounds': {'lower': 0}}}
    lps.check(raw)
    assert lps.write(raw, DATA, tmp_path / 'unbounded.lp').read_text().count('S1 ::') == 2


# ---------------------------------------------------------------------------
# what a set does to the rest of the model
# ---------------------------------------------------------------------------


def test_a_masked_member_leaves_the_set_and_its_neighbours_adjacent():
    """Membership is the variable's own, so a mask closes the gap it leaves.

    With size 1 masked out at every site, SOS2 admits ``{0, 2}`` — consecutive
    among the members that *exist* — which the unmasked model refuses.
    """
    raw = model(2)
    raw['variables'] = {
        'take': {'foreach': ['site', 'size'], 'bounds': {'lower': 0, 'upper': 'cap'}, 'where': 'size != 1'}
    }
    assert lps.solve(raw, DATA).objective == pytest.approx(best(2, [0, 2, 3]))


def test_the_solution_reads_back_past_the_appended_columns():
    """A declaration's share is a slice, and the binaries land after all of them."""
    result = lps.solve(model(2), DATA)
    taken = result.primal('take')
    assert taken.height == len(SITES) * len(SIZES), 'the read-back took the binaries for members'
    assert taken['site'].to_list() == [s for s in SITES for _ in SIZES]
    assert taken['size'].to_list() == SIZES * len(SITES)


def test_a_reformulated_model_says_why_it_has_no_duals():
    """The model declares no integrality, so the ordinary message would lie."""
    result = lps.solve(model(1), DATA)
    with pytest.raises(LpspecError, match="no SOS concept, so 'pick' reached it as binaries"):
        result.dual('total')


def test_diagnostics_separate_the_built_model_from_what_the_sink_added():
    """Two shapes, because a reformulating sink makes them differ.

    The build is what the file declared; ``sink_*`` is the growth no
    declaration accounts for — a binary per member, a linking row each, and
    one cardinality row per set. Nothing else in a build reports it, so a
    solve larger than the model would otherwise be invisible.
    """
    with lps.build(model(1), DATA) as bound:
        assert (bound.diagnostics().sink_columns, bound.diagnostics().sink_rows) == (0, 0), (
            'nothing has been handed to a sink yet'
        )
        bound.solve()
        report = bound.diagnostics()
        assert (report.columns, report.rows) == (len(SITES) * len(SIZES), 0), 'the model declares no rows of its own'
        assert report.sink_columns == len(SITES) * len(SIZES), 'a binary per member'
        assert report.sink_rows == len(SITES) * len(SIZES) + len(SITES), 'a linking row each, and one row per set'


def test_a_sink_that_takes_the_set_reports_adding_nothing():
    """The counterpart, and the reason the two numbers are separate at all."""
    pytest.importorskip('gurobipy', reason='the native SOS path needs the [gurobi] extra')
    with lps.build(model(1), DATA) as bound:
        bound.solve('gurobi')
        assert (bound.diagnostics().sink_columns, bound.diagnostics().sink_rows) == (0, 0)


def test_a_model_with_no_set_is_handed_over_as_built(tmp_path):
    """And a writer never grows a model, whatever it carries."""
    with lps.build(BASE, DATA) as bound:
        bound.solve()
        bound.write(tmp_path / 'plain.lp')
        assert (bound.diagnostics().sink_columns, bound.diagnostics().sink_rows) == (0, 0)


def test_a_sos2_set_of_one_member_restricts_nothing():
    """A set with one member has no segment, so it has no formulation either.

    The member is left alone rather than linked to a binary that does not
    exist — which is what it would be, at a coefficient of zero, pinning the
    one member of the set to zero and quietly deleting it from the model.
    linopy returns early on the same case.

    Masked down per site, so one set keeps three members and the other has
    one: a set that is dropped whole must also not shift the rows the sets
    after it own.
    """
    raw = model(2)
    raw['parameters'] = raw['parameters'] | {'live': {'dims': ['site', 'size'], 'dtype': 'bool'}}
    raw['variables'] = {'take': {'foreach': ['site', 'size'], 'bounds': {'lower': 0, 'upper': 'cap'}, 'where': 'live'}}
    live = DATA | {
        'live': _table({(site, size): (site == 'north' or size == 0) for site in SITES for size in SIZES}).with_columns(
            pl.col('value').cast(pl.Boolean)
        )
    }
    north = max(
        VALUE['north', a] * CAP['north', a] + VALUE['north', b] * CAP['north', b]
        for a, b in itertools.pairwise([0, 1, 2, 3])
    )
    south = VALUE['south', 0] * CAP['south', 0]
    assert lps.solve(raw, live).objective == pytest.approx(north + south), 'the lone member was pinned to zero'


def test_regrouping_the_members_is_a_different_model_to_a_loaded_solver():
    """A set is structure, and it is the one structure nothing else reports.

    Two binds differ in *nothing a solver was handed* — two columns, the same
    bounds, no rows, no matrix — except which sets those columns fall into:
    two members of one set against one member each of two. A digest that did
    not read the sets would call the second the model it already holds.

    Asserted on the digest rather than through a solver because only a sink
    taking the set *natively* depends on it: the one that reformulates gets a
    different matrix out of the rewrite and reloads either way.
    """
    raw = {
        'dimensions': {'site': {'values': SITES}, 'size': {'dtype': 'int', 'values': [0, 1]}},
        'parameters': {'worth': {'dims': ['site', 'size']}, 'live': {'dims': ['site', 'size'], 'dtype': 'bool'}},
        'variables': {'take': {'foreach': ['site', 'size'], 'bounds': {'lower': 0, 'upper': 1}, 'where': 'live'}},
        'sos': {'pick': {'variable': 'take', 'over': 'size', 'type': 1}},
        'objectives': {'total': {'sense': 'maximize', 'expression': 'sum(sum(take * worth, over=site), over=size)'}},
    }
    worth = _table({('north', 0): 3.0, ('north', 1): 5.0, ('south', 0): 5.0, ('south', 1): 0.0})
    together = {'north': [True, True], 'south': [False, False]}
    apart = {'north': [True, False], 'south': [True, False]}

    def live(mask: dict[str, list[bool]]) -> dict[str, Any]:
        return {'worth': worth, 'live': _table({(s, k): mask[s][k] for s in SITES for k in (0, 1)})}

    with lps.build(raw, live(together)) as bound:
        one_set = bound._engine._tables()
        assert bound.solve().objective == pytest.approx(5.0), 'two members of one set are both nonzero'

        bound.rebind(live(apart))
        two_sets = bound._engine._tables()
        assert (one_set.cols.equals(two_sets.cols), one_set.column_count, one_set.row_count) == (True, 2, 0), (
            'the two binds differ in something other than their sets, so this proves nothing'
        )
        assert two_sets.structure != one_set.structure, 'the digest calls a regrouped set the same model'
        assert bound.solve().objective == pytest.approx(8.0), 'one member each, so both may be nonzero'


def test_a_set_is_typeset_as_the_domain_it_restricts():
    """A consumer that skipped it would print math the model does not build.

    Under *variable domains*, because the set restricts which members of a
    family may be nonzero — not among the constraints, where it would read as
    a row a solver holds.
    """
    rendered = lps.to_markdown(model(2))
    assert r'\in \mathrm{SOS}2' in rendered, 'the set is missing from the rendered model'
    assert rendered.index('take sos') > rendered.index('Variable domains')


def test_a_set_that_runs_along_a_leading_dim_still_arrives_grouped():
    """The one shape where the stream is not already in ``(set, weight)`` order.

    With ``over`` first in the ``foreach``, a set's members are ``|site|``
    apart in label order, so every set interleaves with every other. The sinks
    read a set's edges off neighbouring rows, so an ungrouped stream links
    members to the wrong binaries — and still reaches a plausible optimum.
    """
    raw = model(1)
    raw['variables'] = {'take': {'foreach': ['size', 'site'], 'bounds': {'lower': 0, 'upper': 'cap'}}}
    with lps.build(raw, DATA) as bound:
        sets = bound._engine._tables().sos
        assert sets['set'].to_list() == [0, 0, 0, 0, 1, 1, 1, 1], 'the members of a set did not end up together'
        assert sets['weight'].to_list() == [1, 2, 3, 4] * 2, 'a set is not in weight order'
        assert sets['col'].to_list() == [0, 2, 4, 6, 1, 3, 5, 7], 'a member is not the column its coordinate got'
        assert bound.solve().objective == pytest.approx(best(1))
