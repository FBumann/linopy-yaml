"""The harness measures what it says it measures.

`test_ladder.py` is the measurement; this is the part of it that has to be true
for a number to mean anything. It is fast — nothing here solves above a tiny
rung or builds above `xs` — so it runs on a bare `pytest bench` before anything
is timed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from bench import floor
from bench.cases import CASES, Shape
from bench.conftest import _holder_if_alive, pytest_benchmark_update_machine_info, refuse_unless_idle, take_lock
from bench.workloads import _engine, _tables, split_sources


def test_the_default_arm_clears_the_engine_rather_than_leaving_it() -> None:
    """A set-only switch leaks, and a leak here is a confident wrong number.

    One pytest session is one interpreter, so `LPSPEC_ENGINE` set by an arm
    that names an engine outlives that arm. The default arm has to clear it —
    otherwise the first named engine selects itself for every arm after it and
    a two-engine comparison measures one engine against itself, at ratios near
    1.00 that look like a result.

    The old runner spawned a process per measurement and could not have this
    bug; the docstring saying so outlived the runner it described.
    """
    _engine('duckdb')
    assert os.environ.get('LPSPEC_ENGINE') == 'duckdb'

    _engine(None)
    assert 'LPSPEC_ENGINE' not in os.environ, (
        'the default arm left the previous engine selected, so every arm after it measures that one'
    )


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


@pytest.mark.parametrize('label', [pytest.param(s.label, id=s.label) for s in CASES['declarations'].ladder])
def test_the_generated_declaration_model_is_the_language(label: str, tmp_path: Path) -> None:
    """A model file nobody committed still has to pass the front door.

    Both arms parse the same YAML — the linopy arm through
    `lpspec.linopy.build`, the lpspec arm through `lps.build` — so a generated
    file the validator refuses would kill every rung of the sweep at once, and
    only at run time.
    """
    from lpspec.language.validation import load_model
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
    sources = {k: v for k, v in paths.items() if k in ('p_max', 'cost', 'demand')}
    coords = {k: v for k, v in paths.items() if k in ('unit', 'snapshot')}
    with lps.build(case.model_path(shape, cache=tmp_path), sources, coords=coords) as bound:
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
    from lpspec.language.validation import load_model
    from lpspec.lowering import lower_program

    program = lower_program(load_model(str(CASES['commitment'].model)))
    types = {v.name: v.variable_type for v in program.variables}
    assert types == {'u': 'binary', 'p': 'continuous'}, (
        'the MILP case must declare one binary and one continuous variable, or vtype streaming goes unmeasured'
    )


def test_the_floor_builds_the_model_lpspec_builds() -> None:
    """The floor's counts match lpspec's on `transport/xs`, so its headroom claim is about one model.

    Columns, rows and nonzeros are the cheap fingerprint; `--check` compares
    objectives on top and is run by hand. A floor that quietly dropped a term
    would post an unbeatable time for a model nobody built.
    """
    import lpspec as lps

    case = CASES[floor.CASE]
    paths = case.data(case.ladder[0])
    model = floor.arrays(floor.read(paths))

    sources, coords = split_sources(case, case.ladder[0].label, paths)
    with lps.build(case.model, sources, coords=coords) as bound:
        tables = _tables(bound)
        assert model.column_count == tables.column_count, 'the floor holds a different number of variables'
        assert model.row_count == tables.row_count, 'the floor holds a different number of constraints'
        assert model.nonzeros == tables.matrix.height, 'the floor holds a different coefficient matrix'
