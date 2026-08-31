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
import polars as pl
import pytest

from bench import conftest as harness
from bench import floor, plot, profile_build, profile_phases, report, results, tidy, warm_payoff
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
from lpspec.relational.engines.polars.labels import Labelled
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


def _plotted() -> dict[str, Any]:
    """One rung of a panel line, in the shape `plot.series` emits."""
    return {'wall': 1.0, 'lo': 0.9, 'hi': 1.1, 'peak': 0.5, 'vars': 1000}


def _ceiling_record(ladder: str, size: str, **over: Any) -> dict[str, Any]:
    return {
        'record': 'ceiling',
        'arm': 'linopy',
        'case': 'transport',
        'ladder': ladder,
        'sink': 'highs',
        'size': size,
        'budget': 30.0,
        'memory_budget': 6.0,
        'stopped_by': 'time',
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
# a short run cannot replace the published provenance
# ---------------------------------------------------------------------------


def _config(sizes: list[str], destination: str) -> Any:
    return SimpleNamespace(
        invocation_params=SimpleNamespace(args=(f'--benchmark-json={destination}',)),
        getoption=lambda name: sizes if name == '--sizes' else None,
    )


def test_a_short_run_may_not_write_the_committed_results(tmp_path: Path) -> None:
    """`pixi run ladder xs` is a smoke test, and pointed at the committed file it
    replaces every published table's provenance with four measurements —
    silently, in a file whose diff nobody reads closely. That is how I nearly
    lost it: `pixi run ladder --help` does not print help, it starts the run.
    """
    with pytest.raises(harness.pytest.UsageError, match='cannot write'):
        harness.refuse_to_overwrite_the_provenance(_config(['xs'], str(harness.COMMITTED)))


def test_a_short_run_pointed_anywhere_else_is_nobody_business(tmp_path: Path) -> None:
    harness.refuse_to_overwrite_the_provenance(_config(['xs'], str(tmp_path / 'scratch.json')))


def test_narrower_sinks_still_write_the_provenance() -> None:
    """The scheduled run takes one sink per job — both in one job project past
    the ceiling a hosted job is killed at — and each half is still the published
    run. What makes a run a smoke test is leaving out *rungs*, not
    destinations."""
    harness.refuse_to_overwrite_the_provenance(_config(sorted(harness.published_rungs()), str(harness.COMMITTED)))


# ---------------------------------------------------------------------------
# the published selection has one home
# ---------------------------------------------------------------------------


def test_no_workflow_retypes_the_published_selection() -> None:
    """A run that retypes the selection is a number whose fingerprint no longer
    describes it — `bench/README.md` has said so since the harness became
    pytest, and I still wrote a workflow that did it.

    The rule is about the *published* selection, not about pytest: `bench.yml`
    and `codspeed.yml` take deliberately narrower ones to answer *did this pull
    request regress*, and those belong to them. What may not be copied is what
    the page is taken with, which lives in the `ladder` task.
    """
    import tomllib

    root = Path(__file__).resolve().parents[1]
    task = tomllib.loads((root / 'pyproject.toml').read_text())
    cases = ' '.join(task['tool']['pixi']['feature']['bench']['tasks']['ladder']['cmd'].split())
    marker = cases[cases.index('--cases') : cases.index('--sizes')].strip()

    guilty = [w.name for w in sorted((root / '.github' / 'workflows').glob('*.y*ml')) if marker in w.read_text()]
    assert not guilty, f'{guilty} spell out `{marker}`; call `pixi run ladder` so the selection has one home'


def test_the_ci_ladder_defaults_to_the_published_memory_budget() -> None:
    """`ladder-ci` reads `BENCH_MEMORY_BUDGET` so a diagnostic run is a dispatch
    input rather than a commit, and falls back to the published number — which
    therefore exists twice. Drifting them apart makes every scheduled run
    measure a ladder nobody chose."""
    import tomllib

    tasks = tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text())
    tasks = tasks['tool']['pixi']['feature']['bench']['tasks']
    published = next(a['default'] for a in tasks['ladder']['args'] if a['arg'] == 'memory')
    fallback = tasks['ladder-ci']['cmd'].split('BENCH_MEMORY_BUDGET:-', 1)[1].split('}', 1)[0]
    assert fallback == published, f'ladder-ci falls back to {fallback} GB, the published ladder is {published} GB'


def test_the_ci_ladder_covers_every_published_case() -> None:
    """`ladder-ci` runs one pytest per case so no process carries a finished
    case's memory into the next one, which means the case list exists twice —
    in `ladder`'s default and in the loop. A case added to one and not the other
    is a column that silently stops being measured.
    """
    import tomllib

    tasks = tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text())
    tasks = tasks['tool']['pixi']['feature']['bench']['tasks']
    published = next(a['default'] for a in tasks['ladder']['args'] if a['arg'] == 'cases').split()
    asked = next(a['default'] for a in tasks['ladder-ci']['args'] if a['arg'] == 'cases').split()
    assert asked == published, f'ladder-ci runs {asked} and the published ladder is {published}'


def test_a_case_the_box_cannot_hold_leaves_the_others_their_turn() -> None:
    """One pytest per case is half of it; the loop not stopping is the other half.

    `|| exit 1` gave the three cases after a dead one nothing to run, so a run
    that lost `transport` came back with no results at all rather than with the
    three it could still have taken (runs 12 and 16 of the published
    benchmark). The ladder still fails once it has taken what it can —
    `report`, `plot` and the artifact sit behind `!cancelled()` rather than
    behind success, so a partial run is published as a partial run and never
    read as a whole one.
    """
    import tomllib

    tasks = tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text())
    cmd = tasks['tool']['pixi']['feature']['bench']['tasks']['ladder-ci']['cmd']
    body, tail = cmd.split(' do ', 1)[1].split('; done', 1)
    assert 'exit' not in body, f'a dead case must not end the loop, and this body exits: {body.strip()}'
    assert 'exit 1' in tail, 'and the ladder still fails, once the cases it could take are taken'


def test_every_published_case_is_measured_and_uploaded_on_its_own() -> None:
    """A runner that dies skips every step it has left, `if: always()` included.

    So results that live only on the box until one upload at the end are lost
    whole: runs 12, 16, 18 and 19 each measured cases they never handed back.
    One step per case, each uploading before the next can take the box, bounds
    that to the case the box died on — and the step list is where the published
    selection has to be reachable, so a case added to `ladder` and not here is a
    column that silently stops being measured.
    """
    import tomllib

    root = Path(__file__).resolve().parents[1]
    tasks = tomllib.loads((root / 'pyproject.toml').read_text())['tool']['pixi']['feature']['bench']['tasks']
    published = next(a['default'] for a in tasks['ladder']['args'] if a['arg'] == 'cases').split()

    workflow = (root / '.github/workflows/published-benchmark.yml').read_text()
    for sink in ('highs', 'gurobi'):
        for case in published:
            step = f'          sink: {sink}\n          case: {case}\n'
            assert step in workflow, f'{sink}/{case} has no step of its own, so it cannot be uploaded on its own'
    assert workflow.count('uses: ./.github/actions/ladder-case') == 2 * len(published), (
        'one step per case per sink, and no more'
    )

    action = (root / '.github/actions/ladder-case/action.yml').read_text()
    assert 'bash bench/memory-watchdog.sh &' in action, 'each case runs under the watchdog'
    assert 'uses: actions/upload-artifact@v4' in action, 'and hands back what it measured before the next one starts'


def test_the_watchdog_says_something_even_when_nothing_moves() -> None:
    """The high-water line prints only when it moves, so a quiet watchdog and a
    dead one read alike. Run 19 went six minutes in silence and it took an
    orphaned `sleep` in the runner's cleanup to establish it had been sampling
    at all — which is a diagnosis that should not need forensics.
    """
    script = (Path(__file__).resolve().parents[1] / 'bench/memory-watchdog.sh').read_text()
    assert 'BENCH_MEMORY_HEARTBEAT_SECONDS' in script, 'silence has to be distinguishable from death'


def test_the_watchdog_reaches_the_process_that_holds_the_model() -> None:
    """Killing the case by its pytest flags leaves the memory behind.

    `benchmem(isolate=True)` measures in a `multiprocessing` **spawn**, so the
    process holding the model is `python -c 'from multiprocessing.spawn import
    spawn_main…'` and carries none of pytest's arguments. Run 18 killed
    `transport` at 24 GB used, freed nothing, started `storage` on the same full
    box and lost the runner fifteen seconds later — inside the settle the
    watchdog was sleeping through.

    Read off the script rather than run against a real case: this file is
    collected inside every ladder invocation, so a watchdog exercised here would
    kill the ladder running it.
    """
    script = (Path(__file__).resolve().parents[1] / 'bench/memory-watchdog.sh').read_text()
    assert 'multiprocessing.spawn import spawn_main' in script, (
        "the spawned child is what holds the model, and pytest's own flags do not name it"
    )
    assert '-P "$pid"' in script, (
        'children go first — once the pytest is gone the child is reparented and only its own argv is left'
    )
    assert 'available again' in script, (
        'after a kill it watches for the memory to come back rather than sleeping through the seconds '
        'in which the next case starts'
    )


def test_the_reproduction_script_runs_what_the_task_runs() -> None:
    """`bench/reproduce.py` exists so a published number can be re-taken on the
    versions that produced it. A reproduction running a *different* selection
    would be worth less than none, so it reads the task rather than repeating
    it — and this fails if somebody gives it a selection of its own again.
    """
    from bench import reproduce

    selection = ' '.join(reproduce.published())
    for expected in ('--cases dispatch transport storage fleet', '--budget 30', 'bench/results/latest.json'):
        assert expected in selection, f'the reproduction lost `{expected}` from the task definition'
    assert 'PUBLISHED' not in Path(reproduce.__file__).read_text(), 'the selection is read, not repeated'


# ---------------------------------------------------------------------------
# the reproduction environment carries every library the harness measures
# ---------------------------------------------------------------------------


def test_the_lock_pins_every_library_an_arm_needs() -> None:
    """`bench/reproduce.py.lock` is what makes a published number reproducible,
    and the way it stops being that is quietly: an arm is added, the harness
    measures it, and the environment somebody else installs has no idea it
    exists. Two of these resolve from git — lpspec itself, and linopy from a
    branch that moves — so the lock is the only place their commits are written
    down at all.
    """
    locked = (Path(__file__).resolve().parent / 'reproduce.py.lock').read_text()
    for name, module in sorted(ARMS.items()):
        for required in getattr(module, 'REQUIRES', ()):
            assert f'name = "{required}"' in locked, (
                f'the {name} arm needs {required}, which bench/reproduce.py.lock does not pin — '
                f're-run `uv lock --script bench/reproduce.py`'
            )


def test_the_lock_freezes_the_branch_linopy_moves_on() -> None:
    """linopy is installed from `master`. A version string cannot pin that and a
    published number taken against it is otherwise unrepeatable, so the lock has
    to carry the commit."""
    locked = (Path(__file__).resolve().parent / 'reproduce.py.lock').read_text()
    assert 'git = "https://github.com/PyPSA/linopy?rev=master#' in locked, (
        'the lock has to name the linopy commit, not the branch'
    )


# ---------------------------------------------------------------------------
# the width ladder reaches the size ladder's rungs by a different route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('case_name', ['transport', 'storage'])
def test_a_width_rung_matches_its_size_twin_variable_for_variable(case_name: str) -> None:
    """`w10` is `s`, `w1000` is `l` — same variables, same rows, different shape.

    That is the whole point of the second ladder: a library whose cost tracks
    the row count answers the twins the same way, and one that pays for joins or
    for materialising a product does not. If the two drift apart the comparison
    silently becomes two different models, so the multipliers and the
    per-snapshot width have to stay in step.
    """
    ladder = {shape.label: shape for shape in CASES[case_name].ladder}
    for width, size in (('w1', 'xs'), ('w10', 's'), ('w100', 'm'), ('w1000', 'l')):
        assert ladder[width].nominal_variables == ladder[size].nominal_variables, (
            f'{case_name}/{width} is {ladder[width].nominal_variables:,} variables '
            f"against {size}'s {ladder[size].nominal_variables:,} — the ladders have drifted apart"
        )


@pytest.mark.parametrize('case_name', ['transport', 'storage'])
def test_a_width_rung_grows_entities_and_holds_the_snapshots(case_name: str) -> None:
    """The axis that was missing: every other ladder here grows `snapshot` and
    freezes the entity counts, which is why `transport`'s bus x generator
    incidence was 20 x 100 at every rung of it."""
    width = [s for s in CASES[case_name].ladder if s.label.startswith('w')]
    snapshots = {s.sizes['snapshot'] for s in width}
    assert len(snapshots) == 1, f'a width rung must hold the snapshot count fixed, got {sorted(snapshots)}'
    entities = [sum(v for k, v in s.sizes.items() if k != 'snapshot') for s in width]
    assert entities == sorted(entities) and entities[0] * 1000 == entities[-1], (
        f'entities have to grow by the stated factors across {[s.label for s in width]}, got {entities}'
    )


# ---------------------------------------------------------------------------
# an arm stops climbing a ladder it cannot afford (bench/conftest.py)
# ---------------------------------------------------------------------------


def _ceiling(budget: float, selected: tuple[str, ...] = ('xs', 's', 'm', 'l'), memory: float = 0.0) -> Any:
    return harness.Ceiling(budget, {}, selected, memory)


# ---------------------------------------------------------------------------
# an arm stops climbing a ladder whose next rung will not fit the machine
# ---------------------------------------------------------------------------


def test_a_rung_that_projects_over_the_memory_budget_stops_the_ladder() -> None:
    """What the time budget cannot catch. `transport/w100` on linopy takes 51 s
    and 31 GB; over-time leaves a number behind, over-memory leaves the run with
    no runner at all (#1416). The rungs grow tenfold, so 3 GB at `xs` projects
    to 30 GB at `s`."""
    ceiling = _ceiling(0.0, memory=16.0)
    ceiling.record('linopy', 'dispatch', 'xs', 'lp', 1.0, 3e9)
    reason = ceiling.reached('linopy', 'dispatch', 'xs', 'lp')
    assert reason is not None, 'a projection over the memory budget stops the ladder'
    assert '30 GB' in reason and '3 GB' in reason, 'the reason carries the measurement and the projection'


def test_a_measurement_over_the_memory_budget_stops_without_projecting() -> None:
    ceiling = _ceiling(0.0, memory=16.0)
    ceiling.record('linopy', 'dispatch', 'xs', 'lp', 1.0, 31e9)
    reason = ceiling.reached('linopy', 'dispatch', 'xs', 'lp')
    assert reason is not None and 'projects' not in reason, 'the rung itself was over, so there is nothing to project'


def test_a_rung_inside_both_budgets_lets_the_ladder_continue() -> None:
    ceiling = _ceiling(120.0, memory=16.0)
    ceiling.record('lpspec', 'dispatch', 'xs', 'lp', 0.5, 0.2e9)
    assert ceiling.reached('lpspec', 'dispatch', 'xs', 'lp') is None


def test_no_memory_budget_measures_everything() -> None:
    """The default off, so a local run behaves as it did before this existed."""
    ceiling = _ceiling(120.0)
    ceiling.record('linopy', 'dispatch', 'xs', 'lp', 0.5, 900e9)
    assert ceiling.reached('linopy', 'dispatch', 'xs', 'lp') is None


def test_a_rung_that_projects_over_budget_stops_the_ladder() -> None:
    """The rungs grow tenfold, so one measurement settles the next one.

    `dispatch/xs` is 10k variables and `s` is 100k, so a 20 s build at `xs`
    projects to 200 s — and taking that measurement would buy nothing the
    projection has not already said.
    """
    ceiling = _ceiling(120.0)
    ceiling.record('pyomo', 'dispatch', 'xs', 'lp', 20.0)
    reason = ceiling.reached('pyomo', 'dispatch', 'xs', 'lp')
    assert reason is not None, 'a projection over budget stops the ladder'
    assert '200 s' in reason and '20 s' in reason, 'the reason carries the measurement and the projection'


def test_a_rung_inside_budget_lets_the_ladder_continue() -> None:
    ceiling = _ceiling(120.0)
    ceiling.record('lpspec', 'dispatch', 'xs', 'lp', 0.5)
    assert ceiling.reached('lpspec', 'dispatch', 'xs', 'lp') is None, '0.5 s projects to 5 s, well inside 120 s'


def test_a_measurement_over_budget_stops_the_ladder_without_projecting() -> None:
    ceiling = _ceiling(120.0)
    ceiling.record('pyomo', 'dispatch', 'm', 'lp', 300.0)
    reason = ceiling.reached('pyomo', 'dispatch', 'xs', 'lp')
    assert reason is not None and 'projects' not in reason, (
        'a rung that already blew the budget needs no arithmetic about the next one'
    )


def test_the_top_of_the_run_never_projects() -> None:
    """The last rung *this run asked for* has nothing after it.

    Read off the selection rather than off the case: a ladder measured to `l`
    still has `xl` and `2xl` defined behind it, four and twelve times wider, and
    projecting onto a rung nobody asked for stops the climb over work that was
    never going to happen. On the first published run it did exactly that.
    """
    ceiling = _ceiling(120.0, selected=('xs', 's', 'm', 'l'))
    ceiling.record('lpspec', 'dispatch', 'l', 'lp', 100.0)
    assert ceiling.reached('lpspec', 'dispatch', 'l', 'lp') is None, (
        '`l` is the top of this run; `xl` is not being measured and cannot stop it'
    )


def test_a_ceiling_on_one_ladder_leaves_the_other_alone() -> None:
    """A case can carry two ladders, and they ask different questions — `l` is
    the longest model, `w1000` the widest. The first published run ceilinged
    pyomo and `gurobipy-loop` on the size ladder and lost them from the width
    tables entirely, which is a measurement nobody decided not to take.
    """
    ceiling = _ceiling(120.0, selected=('xs', 's', 'm', 'l', 'w1', 'w10', 'w100', 'w1000'))
    ceiling.record('pyomo', 'transport', 'm', 'lp', 20.0)
    assert ceiling.reached('pyomo', 'transport', 'l', 'lp') is not None, 'the size ladder stops'
    assert ceiling.reached('pyomo', 'transport', 'w10', 'lp') is None, 'the width ladder is a separate climb'


def test_the_sidecar_says_which_budget_stopped_each_climb() -> None:
    """The record carries both budgets, so the one that fired has to be on it too.

    Memory is checked first and either can stop a rung, so neither budget's
    value tells a reader which did — and the prose reason is the only other
    place it is said.
    """
    ceiling = _ceiling(30.0, memory=6.0)
    ceiling.record('linopy', 'dispatch', 'm', 'highs', 1.0, 3e9)
    ceiling.record('pyomo', 'dispatch', 'm', 'highs', 20.0, 0.1e9)

    stopped = {row['arm']: row['stopped_by'] for row in ceiling.rows()}
    assert stopped == {'linopy': 'memory', 'pyomo': 'time'}, (
        '3 GB projects past the 6 GB budget and 20 s past the 30 s one, each on its own axis'
    )


@pytest.mark.parametrize(
    ('stopped_by', 'expected'),
    [
        pytest.param('memory', '>6 GB', id='a-memory-stop-prints-the-memory-budget'),
        pytest.param('time', '>30 s', id='a-time-stop-prints-the-time-budget'),
        pytest.param(None, '>30 s', id='a-sidecar-from-before-the-field-prints-seconds'),
    ],
)
def test_a_bound_names_the_budget_that_actually_stopped_the_climb(stopped_by: str | None, expected: str) -> None:
    """Both budgets are on the record and only one of them fired.

    Run 15 of the published benchmark was held to 6 GB and stopped 34 of its 35
    cells on memory — `linopy` on `transport/m` for projecting 8.94 GB after
    0.894 s. Read as seconds, every one of those publishes `>30 s`: not a
    missing number in a table whose job is to be checkable, but a false one.

    `None` is a sidecar written before the harness recorded which budget fired.
    Those runs carried no memory budget at all, so seconds is both the fallback
    and what they enforced.
    """
    ceiling = _ceiling_record('size', 'm')
    ceiling.pop('stopped_by')
    if stopped_by is not None:
        ceiling['stopped_by'] = stopped_by
    assert results.bound_label(ceiling) == expected


def test_both_renderers_read_the_same_bound() -> None:
    """The table and the chart print the same cell, and the wording is one
    function so they cannot come to disagree about it — which they did, each
    formatting `budget` as seconds on its own line."""
    ceiling = _ceiling_record('size', 'm', stopped_by='memory')
    report.CEILINGS[:] = [ceiling]
    taken = {
        ('transport', 'highs', 'linopy'): {r: _plotted() for r in ('xs', 's')},
        ('transport', 'highs', 'lpspec'): {r: _plotted() for r in ('xs', 's', 'm', 'l')},
    }

    charted = plot.panels(taken, [ceiling])['transport — highs']['series']['linopy']['bound']
    tabled = report.over_budget('transport', 'l', 'highs', 'linopy')
    assert tabled == '>6 GB', 'the table names the budget that fired'
    assert charted == [None, None, None, tabled], 'and the chart says the same thing at the same rungs'


def test_a_ceiling_is_per_sink() -> None:
    """Writing an LP file and filling a solver are different costs, and an arm
    that cannot afford one may still afford the other."""
    ceiling = _ceiling(120.0)
    ceiling.record('pyomo', 'dispatch', 'xs', 'lp', 20.0)
    assert ceiling.reached('pyomo', 'dispatch', 'xs', 'highs') is None, 'the other sink was never measured'


def test_no_budget_measures_everything() -> None:
    ceiling = _ceiling(0.0)
    ceiling.record('pyomo', 'dispatch', 'xs', 'lp', 9999.0)
    assert ceiling.reached('pyomo', 'dispatch', 'xs', 'lp') is None, (
        '--budget 0 is the way to take the slow number anyway'
    )


# ---------------------------------------------------------------------------
# a hand-written arm is the same model, or it is not an arm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('case_name', ['dispatch', 'transport', 'storage', 'fleet'])
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
    pytest.importorskip('gurobipy' if dialect.startswith('gurobipy') else dialect)
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


def test_the_run_record_carries_the_cpu_pytest_benchmark_collected(tmp_path: Path) -> None:
    """`platform.processor()` answers `x86_64` on every Linux runner, so the
    record used to say nothing about which box a number came from."""
    doc = {
        'machine_info': {
            'system': 'Linux',
            'release': '6.8.0',
            'machine': 'x86_64',
            'processor': 'x86_64',
            'python_version': '3.12.0',
            'cpu': {'brand_raw': 'AMD EPYC 7763', 'count': 4},
        },
        'benchmarks': [],
    }
    path = tmp_path / 'latest.json'
    path.write_text(json.dumps(doc))
    run = next(r for r in results.records(path) if r['record'] == 'run')
    assert (run['cpu'], run['cores']) == ('AMD EPYC 7763', 4), 'the brand and the core count, not the arch'


def _run(cpu: str, cores: int = 4) -> dict[str, object]:
    return {
        'record': 'run',
        'platform': 'Linux 6.8.0',
        'cpu': cpu,
        'cores': cores,
        'python': '3.12.0',
        'versions': {'polars': '1.0'},
    }


def test_one_machine_prints_as_one_line() -> None:
    assert report.provenance([_run('AMD EPYC 7763')]) == (
        'AMD EPYC 7763, 4 cores (Linux 6.8.0), python 3.12.0 — polars 1.0.'
    )


def test_a_page_merged_from_two_machines_says_so() -> None:
    """The ladder takes one sink per job (#1315), so this is the ordinary case
    the moment two runners draw different CPUs from the pool."""
    line = report.provenance([_run('AMD EPYC 7763'), _run('Intel Xeon Platinum 8370C')])
    assert 'Taken on 2 machines' in line, 'a merged page must not print one machine for rows from two'
    assert 'AMD EPYC 7763' in line and 'Intel Xeon Platinum 8370C' in line, 'both boxes are named'


def test_two_files_from_one_machine_are_not_marked() -> None:
    assert 'machines' not in report.provenance([_run('AMD EPYC 7763'), _run('AMD EPYC 7763')])


def test_a_record_from_before_the_harness_carried_a_machine_still_renders() -> None:
    """A `.jsonl` result is taken verbatim, so the reader meets run records
    written to an older shape and has to render them rather than raise."""
    assert report.provenance([{'record': 'run', 'platform': None}]) == '? (?), python ? — .'
    assert report.provenance([{'record': 'run'}]) == '? (?), python ? — .'


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
    marked = report.density(report.best([_timing('lpspec', size='d100', iqr=1.9), _timing('linopy', size='d100')]))
    assert '| 1.00x~ |' in marked, 'a ratio drawn from a marked minimum carries the doubt'


def test_a_cell_with_no_number_in_it_is_never_marked() -> None:
    """A ratio needs both arms. One noisy arm and nothing to divide it by leaves
    an em dash, and a mark on that claims doubt about a measurement nobody took.

    The rung below is measured on `lpspec` only while the run as a whole
    carries a second arm, which is what leaves an empty cell in the table at
    all now that the columns are whichever arms the run measured.
    """
    rows = report.best([_timing('lpspec', iqr=1.9), _timing('linopy', size='l')])
    table = report.table('dispatch', rows, 'lp')
    assert '| \u2014 |' in table, 'the arm that did not run this rung still renders as absent'
    assert f'\u2014{report.MARK}' not in table, 'an absent measurement cannot be noisy'


_FENCED = """# A page

Prose the numbers are read with.

<!-- bench:results -->

| old | table |

<!-- bench:/results -->

More prose.
"""


def test_a_fenced_block_is_replaced_and_the_prose_around_it_is_not() -> None:
    """The page is a tracked source file: its prose and headings are reviewed in
    a diff like any other code, and only what sits inside a fence is
    mechanical. That is the split `bench.plot` already makes on the chart."""
    written, skipped = report.splice(_FENCED, {'results': '| new | table |'})
    assert skipped == [], 'the page has the fence, so nothing was skipped'
    assert '| new | table |' in written and '| old | table |' not in written
    assert 'Prose the numbers are read with.' in written, 'everything outside the fence survives'
    assert 'More prose.' in written


def test_writing_twice_changes_nothing_the_second_time() -> None:
    once, _ = report.splice(_FENCED, {'results': '| new | table |'})
    assert report.splice(once, {'results': '| new | table |'})[0] == once, 'a re-render has an empty diff'


@pytest.mark.parametrize(
    ('page', 'complaint'),
    [
        pytest.param('<!-- bench:results -->\nhalf a fence\n', 'half a `results` fence', id='unclosed'),
        pytest.param('<!-- bench:/results -->\n<!-- bench:results -->\n', 'closes the results fence', id='inverted'),
    ],
)
def test_a_page_that_cannot_take_the_block_is_refused(page: str, complaint: str) -> None:
    """Refused rather than appended to: a page that quietly grew a second copy
    of every table would look fine in the render and wrong in the diff."""
    with pytest.raises(SystemExit, match=complaint):
        report.splice(page, {'results': '| new | table |'})


def test_a_page_without_a_fence_is_told_so_rather_than_failed() -> None:
    """The tables live on the chart page now, and a page is entitled to host
    only the parts it wants. Named in the return rather than raised, so the
    caller can print what it had nowhere to put — silence would let a renamed
    fence stop updating a table with nobody the wiser."""
    written, skipped = report.splice('# A page\n\nno fence here\n', {'results': '| new | table |'})
    assert skipped == ['results'], 'the fragment is reported, not written'
    assert written == '# A page\n\nno fence here\n', 'and the page is untouched'


def test_an_empty_fragment_never_blanks_the_page() -> None:
    """A results file that rendered nothing would otherwise publish nothing,
    silently — the failure mode `bench/results.py` warns about, where a run
    that died leaves a page that looks merely quiet."""
    with pytest.raises(SystemExit, match='refusing to blank the page'):
        report.splice(_FENCED, {'results': '   '})


def test_the_marginal_table_carries_no_ratio_between_libraries() -> None:
    """A build-only number is not comparable across libraries.

    One that defers materialising its coefficients to its writer spends almost
    nothing in the build and pays at the seam: on `dispatch` at 1M columns
    linopy built in 18.6 ms against lpspec's 33.7 ms and then emitted in 0.64 s
    against 0.44 s, so a ratio drawn here says the opposite of the run it came
    from. The tables that measure to a common artifact carry the ratios.
    """
    table = report.marginal([_loop('dispatch', 'lpspec', 1200), _loop('dispatch', 'linopy', 1200)])
    assert 'lpspec: steady' in table and 'linopy: steady' in table, 'both libraries still get their columns'
    assert '\u00f7' not in table, 'no ratio column here — the build is not the same work in each'
    assert 'not across the row' in table, 'and the table says so where it is read'


def test_a_model_table_shows_the_numbers_and_leaves_the_dividing_to_the_reader() -> None:
    """Five libraries times wall, peak and a ratio each is nineteen columns
    before the dimensions start, and the ratio is the half a reader can do by
    eye from the two numbers beside it. The sweeps keep theirs: they compare at
    one size, with no column of absolutes to read a ratio off.
    """
    rows = report.best([_timing('lpspec'), _timing('linopy')])
    table = report.table('dispatch', rows, 'lp')
    assert 'wall: lpspec' in table and 'wall: linopy' in table, 'every library measured is still a column'
    assert '\u00f7' not in table, 'the per-model table carries no ratio'


def test_a_run_of_one_arm_has_no_ratio_column() -> None:
    """A number divided by itself is not a comparison, and a column of 1.00x
    reads like one."""
    table = report.table('dispatch', report.best([_timing('lpspec')]), 'lp')
    assert 'wall: lpspec' in table, 'the arm that ran is still a column'
    assert '\u00f7' not in table, 'nothing to divide against, so no ratio column at all'


def test_a_measurement_without_a_peak_is_skipped_rather_than_divided(tmp_path: Path) -> None:
    """`peak_rss_bytes` is `None` for a run taken without `benchmem(isolate=True)`,
    and the figures divide it — unguarded that is a `TypeError` halfway through
    a render, where a missing point is what it actually is."""
    path = tmp_path / 'results.jsonl'
    records = [_timing('lpspec'), _timing('linopy', size='l', peak_rss_bytes=None)]
    path.write_text('\n'.join(json.dumps(r) for r in records))

    taken = plot.series(path)
    assert 'l' not in taken.get(('dispatch', 'lp', 'linopy'), {}), (
        'a record with no peak cannot be plotted, so it is dropped'
    )
    assert 'm' in taken[('dispatch', 'lp', 'lpspec')], 'and the records around it still are'


def test_a_ceiling_from_the_width_ladder_does_not_bound_the_size_panel() -> None:
    """A case carries two ladders and a panel plots one of them.

    `series` already drops a width measurement, and the ceiling beside it was
    read anyway: `w100` has no position on an axis of `xs s m l`, so the bound
    that says which rungs the budget stopped indexed a rung the axis does not
    hold. That is a `ValueError` in the middle of a render, and it took run 15
    of the published benchmark after both ladders had finished measuring.

    Latent until an arm is *also* short of a size rung — the memory budget made
    that ordinary, which is why a chart that had always been wrong here started
    failing.

    The width record is second so that reading it would also lose the size one:
    the ceilings are keyed by arm with no ladder in the key, so the wrong
    reading costs a bound that was real as well as raising on one that is not.
    """
    taken = {
        ('transport', 'highs', 'lpspec'): {r: _plotted() for r in ('xs', 's', 'm', 'l')},
        ('transport', 'highs', 'linopy'): {r: _plotted() for r in ('xs', 's', 'm')},
    }
    ceilings = [_ceiling_record('size', 'm'), _ceiling_record('width', 'w100')]

    panel = plot.panels(taken, ceilings)['transport — highs']
    assert panel['series']['linopy']['bound'] == [None, None, None, '>30 s'], (
        'the size ceiling still bounds the rung above it, and the width one says nothing here'
    )


def test_a_ceiling_on_a_rung_no_line_could_plot_bounds_nothing() -> None:
    """The rung a ceiling names is one that arm measured — and `series` drops a
    measurement taken without a peak, so a results set mixing one of those in
    leaves the ceiling pointing at a rung the axis no longer holds. Indexed
    against it that is the same `ValueError` a width ceiling raises, reached by
    the other road.
    """
    taken = {('transport', 'highs', 'linopy'): {r: _plotted() for r in ('xs', 's')}}
    ceilings = [_ceiling_record('size', 'm')]

    panel = plot.panels(taken, ceilings)['transport — highs']
    assert panel['series']['linopy']['bound'] == [None, None], 'no rung is above one the axis does not carry'


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
    from math_spec import to_program, to_spec

    case = CASES['declarations']
    shape = case.shape(label)
    schema = to_spec(str(case.model_path(shape, cache=tmp_path)))
    n = shape.sizes['declaration']
    assert len(schema.variables) == n, 'one variable declaration per unit of the swept count'
    assert len(schema.constraints) == n + 1, 'a capacity constraint per declaration, plus one balance'
    to_program(schema)


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
    from math_spec import to_program, to_spec

    program = to_program(to_spec(str(CASES['commitment'].model)))
    types = {n: v.variable_type for n, v in program.variables.items()}
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
    was = {'optimality_cut': Labelled(pl.LazyFrame(), 0, 2), 'feasibility_cut': Labelled(pl.LazyFrame(), 2, 2)}
    now = {'optimality_cut': Labelled(pl.LazyFrame(), 0, 3), 'feasibility_cut': Labelled(pl.LazyFrame(), 3, 2)}
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
