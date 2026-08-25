"""The harness measures what it says it measures.

`test_ladder.py` is the measurement; this is the part of it that has to be true
for a number to mean anything. It is fast — nothing here solves above a tiny
rung or builds above `xs` — so it runs on a bare `pytest bench` before anything
is timed.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from bench import conftest as harness
from bench import floor, plot, profile_build, profile_phases, report, tidy, warm_payoff
from bench.arms import ARMS, solved
from bench.arms.lpspec import _tables, checked_sources
from bench.cases import CASES, Shape
from bench.conftest import (
    MIN_ROUNDS,
    _holder_if_alive,
    flag_passed,
    pytest_benchmark_update_machine_info,
    refuse_unless_idle,
    take_lock,
)
from lpspec.relational.engines.polars.engine import _Block
from lpspec.relational.sinks.solvers.base import WarmStart

# ---------------------------------------------------------------------------
# the machine interlock (#705)
# ---------------------------------------------------------------------------


def _dead_pid() -> int:
    """A pid that was alive a moment ago and is not any more."""
    child = subprocess.Popen(['true'])
    child.wait()
    return child.pid


def test_a_second_session_is_refused_naming_the_holder(tmp_path) -> None:
    """The failure this exists for is silent: a concurrent run's file looks
    complete and its numbers are junk (#419 measured noise floors of 176% and
    344%), so the second session has to be refused before anything is timed."""
    lock = tmp_path / 'bench.lock'
    take_lock(lock)
    with pytest.raises(pytest.UsageError, match='another benchmark is running') as caught:
        take_lock(lock)
    assert f'pid {os.getpid()}' in str(caught.value), 'the refusal has to say who holds the machine'
    assert '--i-know-another-is-running' in str(caught.value), 'and how to get past it deliberately'


def test_a_dead_holders_lock_is_evicted(tmp_path) -> None:
    lock = tmp_path / 'bench.lock'
    lock.write_text(f'pid {_dead_pid()}, started 03:14')
    take_lock(lock)
    assert f'pid {os.getpid()}' in lock.read_text(), 'a crashed session must not wedge every one after it'


def test_an_unreadable_lock_counts_as_held(tmp_path) -> None:
    lock = tmp_path / 'bench.lock'
    lock.write_text('garbage')
    assert _holder_if_alive(lock) == 'garbage', 'a lock we cannot interpret is safer read as held than free'


def test_a_busy_machine_is_refused_and_an_idle_one_is_not() -> None:
    with pytest.raises(pytest.UsageError, match='already working'):
        refuse_unless_idle(9.5, 8)
    refuse_unless_idle(7.5, 8)


def test_the_interlock_is_wired_into_session_start(tmp_path) -> None:
    """`take_lock` working proves nothing unless a session actually calls it —
    a feature and its check can coexist for years without meeting (#321 did).

    A real second session, refused: the lock path follows `TMPDIR`, so the
    child looks at a private tempdir holding a lock owned by this live process.
    `CI` is stripped because CI legitimately bypasses the interlock, and the
    lock check runs before the load check, so the refusal asserted here is
    deterministic on a busy machine too.
    """
    (tmp_path / 'lpspec-bench.lock').write_text(f'pid {os.getpid()}, started 03:14')
    env = {k: v for k, v in os.environ.items() if k != 'CI'}
    env['TMPDIR'] = str(tmp_path)
    child = subprocess.run(
        [sys.executable, '-m', 'pytest', 'bench/test_harness.py', '--collect-only', '-q'],
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).resolve().parent.parent,
        check=False,
    )
    assert child.returncode != 0, 'the second session has to refuse to start, not collect quietly'
    assert 'another benchmark is running' in child.stderr + child.stdout


def test_the_fingerprint_carries_the_load_triple(request: pytest.FixtureRequest) -> None:
    machine_info: dict[str, object] = {}
    pytest_benchmark_update_machine_info(request.config, machine_info)
    load = machine_info['load_avg']
    assert isinstance(load, tuple) and len(load) == 3, (
        'the 1/5/15-minute triple is what lets a contaminated file be recognised after the fact'
    )


# ---------------------------------------------------------------------------
# a contaminated minimum is marked rather than published as fact (#797)
# ---------------------------------------------------------------------------


def _timing(arm: str, **over: Any) -> dict[str, Any]:
    """One `timing` record in the shape `bench.results` emits, tight by default."""
    return {
        'record': 'timing',
        'case': 'dispatch',
        'size': 'm',
        'sink': 'lp',
        'arm': arm,
        'wall_seconds': 1.0,
        'median': 2.0,
        'iqr': 0.0,
        'rounds': 9,
        'peak_rss_bytes': 1e9,
        'live_fraction': 1.0,
        'counts': {'columns': 1000, 'rows': 100, 'nonzeros': 1000},
    } | over


def _rendered(**over: Any) -> str:
    return report.table('dispatch', report.best([_timing('lpspec', **over), _timing('linopy')]), 'lp')


def _loop(case: str, arm: str, width: int) -> dict[str, Any]:
    """One `loop` record — the first-vs-steady pair the marginal table reads."""
    return {
        'record': 'loop',
        'case': case,
        'size': 'm',
        'arm': arm,
        'nominal_variables': width,
        'first_build_seconds': 0.1,
        'steady_build_seconds': 0.05,
        'counts': {'columns': width, 'rows': 10, 'nonzeros': width},
    }


# ---------------------------------------------------------------------------
# a hand-written arm is the same model, or it is not an arm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('case_name', ['dispatch', 'transport'])
@pytest.mark.parametrize('dialect', [a for a in sorted(ARMS) if a != 'lpspec'])
def test_a_hand_written_arm_builds_the_same_model(case_name: str, dialect: str) -> None:
    """Every arm but `lpspec` is a model somebody typed twice.

    `lpspec.linopy` could never be a different model — it read the same YAML
    (hard rule 3), which is what made it an oracle. A hand-written dialect has
    no such protection: a transposed index or a load vector read in the wrong
    order builds a *different model* that benchmarks perfectly, and the faster
    it is the more likely someone quotes it.

    So the smallest rung of each case is solved both ways and the objectives
    compared. It is slow for a test — four LPs — and it is the whole reason to
    believe any number these arms produce.
    """
    pytest.importorskip('gurobipy')
    case = CASES[case_name]
    smallest = case.ladder[0].label
    paths = case.data(case.shape(smallest))
    ours = solved('lpspec', case_name, smallest, paths, {})
    theirs = solved(dialect, case_name, smallest, paths, {})
    assert theirs == pytest.approx(ours, rel=1e-9), (
        f'{dialect} solves {case_name}/{smallest} to {theirs}, lpspec to {ours} — not the same model'
    )


# ---------------------------------------------------------------------------
# the entry points still run — every one of these broke unnoticed in one week
# ---------------------------------------------------------------------------


def test_the_report_renders_from_the_committed_results() -> None:
    """`pixi run report` named three files, two of which no run had ever
    written, so it raised `FileNotFoundError` on a clean checkout. The readers
    take the directory now, and this is what says they still find something in
    it."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert report.main([]) == 0
    assert '| variables |' in out.getvalue(), 'the default target is bench/results, and it renders a table'


def test_the_long_table_renders_from_the_committed_results() -> None:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert tidy.main([]) == 0
    lines = out.getvalue().splitlines()
    assert lines[0] == 'run,case,size,sink,arm,phase,variables,metric,value', 'the header is the schema'
    assert len(lines) > 1, 'the committed provenance produces rows, not just a header'


def test_the_profilers_wrap_the_class_that_actually_builds() -> None:
    """Both profilers monkeypatch a private engine class by name, so a refactor
    in `src/` retires them without touching `bench/`. #1245 moved the build
    methods off `PolarsEngine` onto `_Assembly` and both died on the next run —
    a day before anyone looked."""
    import importlib

    from lpspec.relational.engines.polars.engine import _Assembly

    for module_path, class_name, method in profile_build.STEPS:
        module = importlib.import_module(module_path)
        owner = module if class_name is None else getattr(module, class_name, None)
        assert owner is not None, f'profile_build patches {class_name} in {module_path}, which moved'
        assert hasattr(owner, method), f'profile_build patches {class_name or module_path}.{method}, which moved'
    for method in profile_phases.PHASES:
        assert hasattr(_Assembly, method), f'profile_phases patches _Assembly.{method}, which moved'


# ---------------------------------------------------------------------------
# the long table: a row per number, and no holes (bench/tidy.py)
# ---------------------------------------------------------------------------


def _long(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(tidy.measurements(records, run='r'))


def test_a_timing_record_fans_into_one_row_per_metric() -> None:
    rows = _long([_timing('lpspec')])
    assert [r['metric'] for r in rows] == [
        'wall_seconds',
        'peak_rss_bytes',
        'iqr_seconds',
        'median_seconds',
        'rounds',
        'live_fraction',
        'columns',
        'rows',
        'nonzeros',
    ], 'every number the record carries becomes a row, counts last, memray absent because it was not measured'
    assert {r['case'] for r in rows} == {'dispatch'}, 'the dims repeat down the rows — that is what long form is'
    assert {r['arm'] for r in rows} == {'lpspec'}


def test_a_number_the_run_did_not_produce_is_an_absent_row() -> None:
    """A hole is refused here for the same reason the language refuses one in a
    value column: a null would have to mean *something*, and nothing it could
    mean is true of a measurement that was never taken."""
    rows = _long([_timing('lpspec', peak_rss_bytes=None, counts={'columns': 10, 'rows': 1, 'nonzeros': None})])
    assert [r['metric'] for r in rows].count('peak_rss_bytes') == 0, 'no isolate=True, so no rss row at all'
    assert [r['metric'] for r in rows].count('nonzeros') == 0, 'an arm that cannot count nonzeros writes none'
    assert all(r['value'] is not None for r in rows), 'every value column is complete'


def test_the_rebuild_loop_becomes_two_phases() -> None:
    rows = [r for r in _long([_loop('dispatch', 'lpspec', 1200)]) if r['metric'] == 'wall_seconds']
    assert [(r['phase'], r['value']) for r in rows] == [('first', 0.1), ('steady', 0.05)], (
        'first and steady answer different questions, so they are two rows and never one'
    )
    assert {r['sink'] for r in rows} == {''}, 'the rebuild loop has no sink — it stops before one'


def test_the_long_table_and_the_published_table_agree_on_a_cell() -> None:
    """The two renderings read one extraction. If they could disagree, the long
    form would be a second source of truth rather than a second view."""
    record = _timing('lpspec', wall_seconds=0.5)
    wall = next(r['value'] for r in _long([record]) if r['metric'] == 'wall_seconds')
    assert f'{wall:.2f}' in report.table('dispatch', report.best([record]), 'lp')


def test_the_fingerprint_is_long_too() -> None:
    run = {'record': 'run', 'python': '3.12.0', 'versions': {'polars': '1.0', 'gurobipy': None}, 'commits': {}}
    rows = list(tidy.fingerprint([run], run='r'))
    assert [(r['key'], r['value']) for r in rows] == [('python', '3.12.0'), ('version:polars', '1.0')], (
        'a package the environment does not have is absent, not blank — blank would read as a version'
    )


#: Renders the marginal table for three cases of identical width. Run twice
#: under different hash seeds, it is the whole of the determinism check below.
_RENDER_TIED = """
import sys
sys.path.insert(0, %r)
from bench import report
rows = [
    dict(record='loop', case=c, size='m', arm=a, nominal_variables=1200,
         first_build_seconds=0.1, steady_build_seconds=0.05,
         counts={'columns': 1200, 'rows': 10, 'nonzeros': 1200})
    for c in ('fleet', 'nodal', 'profiled') for a in ('lpspec', 'linopy')
]
print(report.marginal(rows))
"""


def test_the_marginal_table_does_not_reshuffle_between_processes() -> None:
    """A published table a re-render reshuffles has a diff that means nothing.

    Ladders tie by construction — `_ladder` grows every case by the same two
    factors, so `fleet`, `nodal` and `profiled` share all six widths — and the
    rows come out of a set, whose iteration order depends on the interpreter's
    hash seed. Sorting on width alone left those ties in whatever order the set
    produced: four renders of `latest.json` gave three different orderings.

    Two *processes* under different `PYTHONHASHSEED`, because that is what
    varies. One process renders the same bytes however wrong the sort is — its
    seed is fixed at startup — so a same-process check would pass on the bug.
    """
    root = str(Path(__file__).resolve().parent.parent)
    out = []
    for seed in ('0', '1'):
        child = subprocess.run(
            [sys.executable, '-c', _RENDER_TIED % root],
            capture_output=True,
            text=True,
            env=os.environ | {'PYTHONHASHSEED': seed},
            check=True,
        )
        out.append(child.stdout)
    assert out[0] == out[1], 'two hash seeds rendered two row orders — a refresh diff would be noise'
    names = [line.split('|')[1].strip() for line in out[0].splitlines() if line.startswith('| ')]
    assert names[1:] == ['fleet', 'nodal', 'profiled'], (
        'tied widths break by name, so the order is stated rather than inherited from a hash'
    )


def test_the_marginal_table_survives_a_file_that_never_measured_lpspec() -> None:
    """`--arms linopy` is a legitimate run, and reporting one used to raise.

    The width was read off `best[(case, size, 'lpspec')]` directly, so a file
    with no lpspec arm in it died with a `KeyError` inside a sort key — before
    printing any of the tables that did carry both arms.
    """
    assert report.marginal([_loop('dispatch', 'linopy', 1200)]) is not None


@pytest.mark.parametrize(
    ('iqr', 'marked'),
    [
        pytest.param(2.0 * (report.SPREAD_BUDGET + 0.01), True, id='spread-past-the-budget-is-marked'),
        pytest.param(2.0 * (report.SPREAD_BUDGET - 0.01), False, id='spread-inside-the-budget-is-not'),
        pytest.param(None, False, id='a-file-written-before-the-spread-was-carried-is-not'),
    ],
)
def test_a_cell_is_marked_by_its_spread_over_its_own_median(iqr: float | None, marked: bool) -> None:
    """The signal is iqr/median: a minimum whose whole distribution is spread had
    no clean round to fall back on, which is the one contamination `min` cannot
    filter out (#797 measured a cell 2.33x wrong)."""
    assert (report.MARK in _rendered(iqr=iqr)) is marked, (
        f'iqr/median of {iqr} against a budget of {report.SPREAD_BUDGET} must {"mark" if marked else "leave"} the cell'
    )


def test_the_note_appears_exactly_where_a_cell_is_marked() -> None:
    assert report._SPREAD_NOTE not in _rendered(), 'a table with nothing to doubt must not carry the warning'
    assert _rendered(iqr=1.9).count(report._SPREAD_NOTE) == 1, (
        'a marked table says once what the mark means, or the mark is decoration'
    )


def test_marking_leaves_the_published_number_alone() -> None:
    """The mark is a doubt about the minimum, not a different statistic: the
    number in a marked cell is the same one an unmarked run would print."""
    clean, dirty = _rendered(), _rendered(iqr=1.9)
    assert '| 1.00 s |' in clean and '| 1.00 s~ |' in dirty, 'the marked cell still prints its minimum'
    assert dirty.removesuffix('\n\n' + report._SPREAD_NOTE).replace(report.MARK, '') == clean, (
        'marking must annotate the table, not restate it'
    )


def test_the_ratio_beside_a_marked_cell_is_marked_too() -> None:
    """A ratio is only as quotable as the two minima it divides, and #797 is a
    ratio that would have flipped from 0.73x to 1.23x on one contaminated arm."""
    assert '| 1.00x~ |' in _rendered(iqr=1.9), 'a ratio drawn from a marked minimum carries the doubt'


def test_a_cell_with_no_number_in_it_is_never_marked() -> None:
    """A ratio needs both arms. One noisy arm and nothing to divide it by leaves
    an em dash, and a mark on that claims doubt about a measurement nobody took."""
    table = report.table('dispatch', report.best([_timing('lpspec', iqr=1.9)]), 'lp')
    assert '| — |' in table, 'the arm that was not measured still renders as absent'
    assert f'—{report.MARK}' not in table, 'an absent measurement cannot be noisy'


def test_a_measurement_without_a_peak_is_skipped_rather_than_divided(tmp_path: Path) -> None:
    """`peak_rss_bytes` is `None` for a run taken without `benchmem(isolate=True)`,
    and the figures divide it — unguarded that is a `TypeError` halfway through
    a render, where a missing point is what it actually is."""
    path = tmp_path / 'results.jsonl'
    records = [_timing('lpspec'), _timing('linopy', size='l', peak_rss_bytes=None)]
    path.write_text('\n'.join(json.dumps(r) for r in records))

    table = plot.best(path, 'lp')
    assert ('dispatch', 'l', 'linopy') not in table['wall'], 'a record with no peak cannot be plotted, so it is dropped'
    assert ('dispatch', 'm', 'lpspec') in table['wall'], 'and the records around it still are'


@pytest.mark.parametrize(
    ('args', 'given', 'expected'),
    [
        pytest.param((), 5, MIN_ROUNDS, id='the-plugin-default-becomes-the-documented-nine'),
        pytest.param(('--benchmark-min-rounds=3',), 3, 3, id='an-explicit-flag-wins'),
        pytest.param(('--benchmark-min-rounds', '3'), 3, 3, id='an-explicit-flag-wins-when-spaced'),
    ],
)
def test_the_rounds_default_is_the_documented_one_and_an_explicit_flag_wins(
    args: tuple[str, ...], given: int, expected: int
) -> None:
    config = SimpleNamespace(
        invocation_params=SimpleNamespace(args=args),
        option=SimpleNamespace(benchmark_min_rounds=given),
    )
    harness.pytest_configure(config)  # pyrefly: ignore[bad-argument-type]
    assert config.option.benchmark_min_rounds == expected, (
        'docs/about/benchmarks.md publishes nine rounds per measurement; a run that asks for another count keeps it'
    )


def test_the_rounds_default_is_silent_where_the_plugin_is_absent() -> None:
    """The CodSpeed job runs this same suite under a plugin with no such option."""
    config = SimpleNamespace(invocation_params=SimpleNamespace(args=()), option=SimpleNamespace())
    harness.pytest_configure(config)  # pyrefly: ignore[bad-argument-type]
    assert not hasattr(config.option, 'benchmark_min_rounds'), 'nothing to set, so nothing is set'


def test_the_rounds_default_is_wired_into_the_session(request: pytest.FixtureRequest) -> None:
    """A default and the session applying it can coexist without meeting (#321 did)."""
    if not hasattr(request.config.option, 'benchmark_min_rounds'):
        pytest.skip('pytest-benchmark is not installed — bench/ runs through `pixi run -e bench`')
    if flag_passed(request.config, '--benchmark-min-rounds'):
        pytest.skip('this session asked for a round count of its own')
    assert request.config.option.benchmark_min_rounds == MIN_ROUNDS, (
        'the documented command has to reproduce the documented method'
    )


@pytest.mark.parametrize('label', [pytest.param(s.label, id=s.label) for s in CASES['declarations'].ladder])
def test_the_generated_declaration_model_is_the_language(label: str, tmp_path: Path) -> None:
    """A model file nobody committed still has to pass the front door.

    Every arm parses the same YAML, so a generated file the validator refuses
    would kill every rung of the sweep at once, and only at run time.
    """
    from math_spec import load_model

    from lpspec.lowering import lower_program

    case = CASES['declarations']
    shape = case.shape(label)
    schema = load_model(str(case.model_path(shape, cache=tmp_path)))
    n = shape.sizes['declaration']
    assert len(schema.variables) == n, 'one variable declaration per unit of the swept count'
    assert len(schema.constraints) == n + 1, 'a capacity constraint per declaration, plus one balance'
    lower_program(schema)


def test_the_generated_declaration_model_builds(tmp_path: Path) -> None:
    """Loading is not building — `sector` once passed `check()` and died in the
    engine (#345). The sweep's own smallest rung is a million variables, so the
    build gate runs on a tiny shape of the same generated model instead.
    """
    import lpspec as lps

    case = CASES['declarations']
    shape = Shape('tiny', {'declaration': 2, 'unit': 8, 'snapshot': 20}, 20 * 16)
    paths = case.write(shape, tmp_path)
    sources = {k: v for k, v in paths.items() if k in ('p_max', 'cost', 'demand', 'unit', 'snapshot')}
    with lps.build(case.model_path(shape, cache=tmp_path), sources) as bound:
        assert bound is not None


def test_the_declaration_rungs_do_not_share_a_cache_key() -> None:
    keys = [s.key for s in CASES['declarations'].ladder]
    assert len(set(keys)) == len(keys), "rungs sharing a cache key would read each other's data and generated model"


def test_the_declaration_sweep_holds_the_model_size_flat() -> None:
    ladder = CASES['declarations'].ladder
    totals = {s.sizes['declaration'] * s.sizes['unit'] * s.sizes['snapshot'] for s in ladder}
    assert len(totals) == 1, 'a rung that moves total variables confounds the declaration axis with model size'
    assert {s.nominal_variables for s in ladder} == totals, (
        'nominal_variables must count every declaration, or live_fraction misreports the sweep'
    )


@pytest.mark.parametrize('name', [pytest.param(n, id=n) for n in sorted(CASES) if CASES[n].generate_model is None])
def test_a_static_case_still_reads_its_committed_model(name: str) -> None:
    case = CASES[name]
    assert case.model is not None and case.model.exists(), 'a static case names a committed YAML file'
    assert case.model_path(case.ladder[0]) == case.model, (
        'model_path must stay the committed file for every case that does not generate one'
    )


def test_the_milp_case_lowers_with_both_variable_types() -> None:
    """`commitment` only measures the vtype stream if the plan actually carries it.

    The ladder's other cases are all-continuous, so a YAML edit that dropped
    `domain: binary` would leave the case measuring dispatch under a MILP's name
    and nothing downstream would notice — every sink handles an all-continuous
    model happily.
    """
    from math_spec import load_model

    from lpspec.lowering import lower_program

    program = lower_program(load_model(str(CASES['commitment'].model)))
    types = {v.name: v.variable_type for v in program.variables}
    assert types == {'u': 'binary', 'p': 'continuous'}, (
        'the MILP case must declare one binary and one continuous variable, or vtype streaming goes unmeasured'
    )


def test_the_floor_builds_the_model_lpspec_builds() -> None:
    """The floor's counts match lpspec's on `transport/xs`, so its headroom claim is about one model.

    Columns, rows and nonzeros are the cheap fingerprint; the objectives are
    compared by the test below. A floor that quietly dropped a term would post
    an unbeatable time for a model nobody built.
    """
    import lpspec as lps

    case = CASES[floor.CASE]
    paths = case.data(case.ladder[0])
    model = floor.arrays(floor.read(paths))

    sources = checked_sources(case, case.ladder[0].label, paths)
    with lps.build(case.model, sources) as bound:
        tables = _tables(bound)
        assert model.column_count == tables.column_count, 'the floor holds a different number of variables'
        assert model.row_count == tables.row_count, 'the floor holds a different number of constraints'
        assert model.nonzeros == tables.matrix.height, 'the floor holds a different coefficient matrix'


def test_a_spliced_basis_reproduces_the_cold_answer() -> None:
    """The measurement's own claim, on the smallest instance it can be made on.

    `warm_payoff.sweep` asserts objective equality step by step; running it
    here is what makes that a gate rather than a claim in a module nothing
    exercises. A carried basis may move the route and never the optimum, so a
    splice that indexed rows wrongly would show up as a different answer.
    """
    run = warm_payoff.sweep(warm_payoff.SIZES['xs'], n_snap=4, steps=8)
    assert len(run.steps) > 1, 'a single rebuild carries nothing, so the splice would go unexercised'
    for i, step in enumerate(run.steps):
        assert step.warm_objective == pytest.approx(step.cold_objective, rel=1e-9), (
            f'step {i}: a carried basis moved the answer'
        )


def test_the_splice_shifts_a_later_declarations_rows() -> None:
    """The whole reason the carry is not a truncation.

    `feasibility_cut` follows `optimality_cut`, so a row gained by the first
    moves every row of the second. Truncating the previous basis to the new
    height would leave the second family reading the first's statuses — right
    only for a model whose growth is all in the last declaration.
    """
    was = {'optimality_cut': _Block(0, 2), 'feasibility_cut': _Block(2, 2)}
    now = {'optimality_cut': _Block(0, 3), 'feasibility_cut': _Block(3, 2)}
    previous = WarmStart(
        solver='highs',
        column_statuses=np.zeros(4, dtype=np.int8),
        row_statuses=np.array([10, 11, 20, 21], dtype=np.int8),
        column_values=None,
    )
    order = ['optimality_cut', 'feasibility_cut']

    carried = warm_payoff.spliced(previous, was, now, order, 5).row_statuses
    assert list(carried) == [10, 11, warm_payoff.BASIC, 20, 21], (
        'the second declaration keeps its own statuses at its new start, and the gained row starts basic'
    )
    assert list(warm_payoff.prefixed(previous, 5).row_statuses) == [10, 11, 20, 21, warm_payoff.BASIC], (
        'the prefix carry is the mistake this splice exists to avoid; it must stay measurably different'
    )


def test_the_floor_and_lpspec_agree_on_the_answer() -> None:
    """`check()` runs, and the two models solve to one objective.

    The counts above are a fingerprint, not the answer — they match for a floor
    that permuted a coefficient. This calls what `--check` calls, which is also
    the only thing that reaches `workloads.objective`: the counts test never
    does, so a signature change there went unnoticed until someone ran the flag
    by hand.
    """
    ours, lpspec = floor.check()

    assert ours == pytest.approx(lpspec, rel=1e-9), (
        f'the floor solves a different model than lpspec: {ours} against {lpspec}'
    )
