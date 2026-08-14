"""The harness measures what it says it measures.

`test_ladder.py` is the measurement; this is the part of it that has to be true
for a number to mean anything. It is fast, needs no data and no rung, so it
runs on a bare `pytest bench` before anything is timed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from bench.conftest import _holder_if_alive, pytest_benchmark_update_machine_info, refuse_unless_idle, take_lock
from bench.workloads import _engine


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
