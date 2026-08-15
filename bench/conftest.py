"""The harness: selection, data, and the parity gate.

This is what `bench/run.py` used to be. It was an argparse front end over three
nested loops, a subprocess protocol and a JSON-line writer — every one of which
pytest already has, and has tested. The mapping is close to line for line:

    for case x size x sink x arm      ->  parametrize (below)
    subprocess.run(_run_case.py)      ->  benchmem(isolate=True)
    --cases/--sizes/--arms/--sinks    ->  the same flags, via pytest_addoption
    --repeat N, collapse by minimum   ->  rounds; `min` is pytest-benchmark's own
    --out results.jsonl               ->  --benchmark-json
    the parity gate, before timing    ->  a session fixture

What is *not* free is the ragged shape: cases have different ladders, and the
density rungs exist on one case only. So the (case, size) axis is built here
rather than taken as a product — a missing rung is skipped, not an error, which
is what makes `--sizes all` and `--sizes d100 d50` both mean something.

**The two axes stay separate params.** `parametrize(('case_name', 'size'), ...)`
gives pytest-benchmem two scalar dims to group and plot by; packing them into
one string id would leave it nothing to read but the id, which it deliberately
does not parse.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from bench.cases import CASES
from bench.workloads import objective

if TYPE_CHECKING:
    from bench.cases import Shape

#: The relative gap two arms' objectives may differ by and still be one model.
GATE_RTOL = 1e-9

#: How `--arms` names map to what actually runs. `duckdb` is not a third lane:
#: it is the lpspec arm with the engine switch a caller has, which is why the
#: harness sets `LPSPEC_ENGINE` in the measured process rather than reaching
#: for a selector only it knows about.
ENGINE = {'lpspec': None, 'duckdb': 'duckdb'}


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup('ladder', 'the lpspec benchmark ladder')
    g.addoption('--cases', nargs='+', default=sorted(CASES), choices=sorted(CASES))
    g.addoption('--sizes', nargs='+', default=['xs', 's', 'm'], help="rung labels, or 'all' for every rung a case has")
    g.addoption('--arms', nargs='+', default=['lpspec', 'linopy'], choices=('lpspec', 'linopy', 'duckdb'))
    g.addoption(
        '--sinks',
        nargs='+',
        default=['lp', 'highs'],
        choices=('lp', 'highs', 'gurobi'),
        help='where each built model goes. `lp` and `highs` by default: the LP file is the '
        "artifact fewest callers want, and it is not the same comparison — HiGHS's own model "
        'is resident in both arms and narrows the gap. `gurobi` is opt-in because it needs the '
        '[gurobi] extra, and it is measured against linopy the same way, through `to_gurobipy()`.',
    )
    g.addoption('--builds', type=int, default=5, help='rebuilds per process in the first-vs-steady pass; 0 skips it')
    g.addoption('--io-api', default='lp-polars')
    g.addoption('--skip-gate', action='store_true', help='measure without checking the arms agree')
    g.addoption(
        '--i-know-another-is-running',
        action='store_true',
        help='start despite the machine lock or a high load average (#705); the numbers are on you',
    )


#: Rounds every measurement gets at least, which is what `docs/benchmarks.md`
#: publishes as the method. pytest-benchmark's own default is 5, and its
#: calibration hands the fewest rounds to the slowest cells — exactly where
#: interference sustained across every round is most likely and a clean round
#: hardest to come by, which is how a minimum ends up 2.33x wrong (#797).
MIN_ROUNDS = 9


def flag_passed(config: pytest.Config, flag: str) -> bool:
    """Whether *flag* was given on the command line, in either `--x=v` or `--x v` form."""
    return any(arg == flag or arg.startswith(f'{flag}=') for arg in config.invocation_params.args)


def pytest_configure(config: pytest.Config) -> None:
    """Hold every measurement to the number of rounds the published method claims.

    A default rather than a fixed value: an explicit ``--benchmark-min-rounds``
    still wins, so a narrow re-take can ask for more. Silent where
    pytest-benchmark is absent — the CodSpeed job runs this same suite under a
    plugin that has no such option.
    """
    if hasattr(config.option, 'benchmark_min_rounds') and not flag_passed(config, '--benchmark-min-rounds'):
        config.option.benchmark_min_rounds = MIN_ROUNDS


#: Machine-global on purpose — `tempfile.gettempdir()`, not the repo: the run
#: this lock exists to refuse comes from *another worktree* (#705), which shares
#: nothing with this one but the machine.
BENCH_LOCK = Path(tempfile.gettempdir()) / 'lpspec-bench.lock'

_TOOK_LOCK = pytest.StashKey[bool]()


def _holder_if_alive(path: Path) -> str | None:
    """The lock's own description of its holder, or None when that process is gone.

    A lock that cannot be read as `pid N, ...` — or whose pid this user may not
    signal — counts as held: the safe reading of a lock we cannot interpret is
    that someone owns it, and the override flag is the way past a wrong guess.
    """
    try:
        content = path.read_text().strip()
    except FileNotFoundError:
        return None
    try:
        os.kill(int(content.removeprefix('pid ').split(',')[0]), 0)
    except ProcessLookupError:
        return None
    except (ValueError, PermissionError):
        pass  # unreadable, or another user's live process: treat the lock as held
    return content


def take_lock(path: Path) -> None:
    """Claim the machine for one benchmark session, or refuse to start.

    ``O_EXCL`` makes the claim atomic, so two sessions racing for a free lock
    resolve to one refusal rather than two holders. A lock whose holder is no
    longer alive is evicted, so a crashed session cannot wedge the next one.

    Raises:
        pytest.UsageError: If another live session holds the lock.
    """
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = _holder_if_alive(path)
        if holder is not None:
            raise pytest.UsageError(
                f'another benchmark is running ({holder}). Benchmarks do not share a machine: '
                f'the numbers would be wrong and would look fine. Wait, or pass '
                f'--i-know-another-is-running. Lock: {path}'
            ) from None
        path.unlink(missing_ok=True)
        take_lock(path)
        return
    os.write(fd, f'pid {os.getpid()}, started {time.strftime("%H:%M")}'.encode())
    os.close(fd)


def refuse_unless_idle(load1: float, cores: int) -> None:
    """Refuse to benchmark a machine that is already working.

    The lock above only catches this harness; the load average catches
    everything else that owns the box.

    Raises:
        pytest.UsageError: If the 1-minute load average exceeds the core count.
    """
    if load1 > cores:
        raise pytest.UsageError(
            f'1-minute load average is {load1:.2f} on {cores} cores — this machine is already '
            f'working, and a benchmark taken now would be wrong and look fine (#419 measured '
            f'noise floors of 176% and 344% at load 25.78 on 8 cores). Wait for idle, or pass '
            f'--i-know-another-is-running.'
        )


def pytest_sessionstart(session: pytest.Session) -> None:
    """One benchmark per machine, refused up front rather than found in the numbers (#705).

    CI exempts itself: the runners in bench.yml and codspeed.yml are
    single-purpose, and CodSpeed counts instructions rather than wall time, so
    the machine-sharing hazard this refuses is a developer-box one — while a
    runner whose background load trips the threshold would fail the job for
    nothing. The load stamp in ``machine_info`` still records what the box was
    doing.
    """
    if session.config.getoption('--i-know-another-is-running') or os.environ.get('CI'):
        return
    take_lock(BENCH_LOCK)
    session.config.stash[_TOOK_LOCK] = True
    refuse_unless_idle(os.getloadavg()[0], os.cpu_count() or 1)


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Release the machine — only a lock this session took, never another holder's."""
    if session.config.stash.get(_TOOK_LOCK, False):
        BENCH_LOCK.unlink(missing_ok=True)


#: Fingerprinted into every result file. The published tables name these, and a
#: number measured against a different polars is a different number.
#:
#: `pytest-benchmem` is one of them: a fix to its isolated pass moves `rss`
#: without a line of lpspec changing, so a result file that does not name the
#: version that measured it cannot be compared across such a release.
#:
#: `gurobipy` and `scipy` for the same reason one level out: the `gurobi` sink
#: is measurable now, and a published ratio through a solver has to say which
#: solver — scipy being what carries the matrix into it.
TRACKED = (
    'lpspec',
    'linopy',
    'highspy',
    'gurobipy',
    'scipy',
    'polars',
    'pandas',
    'numpy',
    'xarray',
    'pyarrow',
    'pytest-benchmem',
)


@pytest.hookimpl(optionalhook=True)
def pytest_benchmark_update_machine_info(config: pytest.Config, machine_info: dict[str, Any]) -> None:
    """Stamp the result file with what was installed when it ran.

    pytest-benchmark already records the machine and — in `commit_info` — the
    commit *and whether the tree was dirty*, which is the fingerprint the old
    runner shelled out to git for. It does not record dependency versions, and
    those are what a published ratio is actually a ratio of.

    The load-average triple goes in beside them (#705): the session refuses to
    start on a busy machine, but a run forced past that — or one the load crept
    up on — leaves a file that looks complete, and the stamp is what lets it be
    recognised as contaminated after the fact.

    ``optionalhook`` because the spec is pytest-benchmark's and that plugin is
    not always installed: the CodSpeed job runs this same suite with only
    pytest-codspeed, and pluggy rejects an implementation whose spec no plugin
    registered — as an INTERNALERROR, before a single test runs.
    """
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for pkg in TRACKED:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    machine_info['versions'] = versions
    machine_info['load_avg'] = os.getloadavg()


def _rungs(config: pytest.Config) -> list[tuple[str, str]]:
    """Every (case, rung) the selection asks for and the case actually has."""
    wanted = config.getoption('--sizes')
    out = []
    for name in config.getoption('--cases'):
        labels = [s.label for s in CASES[name].ladder]
        out += [(name, s) for s in (labels if wanted == ['all'] else wanted) if s in labels]
    return out


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    names = set(metafunc.fixturenames)
    if {'case_name', 'size'} <= names:
        rungs = _rungs(metafunc.config)
        metafunc.parametrize(('case_name', 'size'), rungs, ids=[f'{c}-{s}' for c, s in rungs])
    if 'arm' in names:
        metafunc.parametrize('arm', metafunc.config.getoption('--arms'))
    if 'sink' in names:
        metafunc.parametrize('sink', metafunc.config.getoption('--sinks'))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """`test_rebuild` asks a question CodSpeed's instruments cannot answer.

    Its whole premise is what the *second* build costs, which is ``rounds`` in
    one process — and pytest-codspeed warns that its memory instrument ignores
    rounds and iterations in pedantic mode. Left in, it would report a
    first-vs-steady number measured over a single build: a wrong number under a
    right-sounding name. (It also fails outright, because `filterwarnings` is
    `error` — which is the warning doing its job.)

    Deselected rather than skipped, so the count reads as "not asked here"
    rather than "asked and unanswered".
    """
    if not getattr(config.option, 'codspeed', False):
        return
    dropped = [i for i in items if i.name.startswith('test_rebuild')]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = [i for i in items if not i.name.startswith('test_rebuild')]


@pytest.fixture(scope='session')
def paths() -> Any:
    """``(case, rung) -> parquet paths``, generated once and shared.

    Generation is neither lpspec's work nor stable across machines, so it has
    to sit outside every measured region — and session scope is what makes that
    structural rather than a convention the next test can forget. The files are
    also cached on disk between runs, so a second invocation pays nothing.
    """

    def resolve(case_name: str, size: str) -> dict[str, str]:
        case = CASES[case_name]
        return case.data(case.shape(size))

    return resolve


@pytest.fixture(scope='session')
def io_api(request: pytest.FixtureRequest) -> str:
    """linopy's LP writer backend — a flag rather than a constant because it is
    the one place the two arms' `lp` sink is not literally the same call."""
    return str(request.config.getoption('--io-api'))


@pytest.fixture(scope='session')
def builds(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption('--builds'))


@pytest.fixture(scope='session')
def gate(request: pytest.FixtureRequest) -> Any:
    """The smallest rung of a case, solved on every arm, objectives compared.

    Runs before the first measurement of each case and at most once per case.
    The differential suite proves the two lanes agree on the *language*; it says
    nothing about the data this harness generates, and a performance number
    describing two different models is worse than none.

    Gating *the arms being measured* rather than a fixed pair: an arm that is
    fast because it built a different model is the one result this harness must
    never publish, and a third arm would otherwise be exempt from the check the
    first two answer to.
    """
    checked: dict[str, None] = {}
    skip = request.config.getoption('--skip-gate')
    arms = request.config.getoption('--arms')

    def check(case_name: str, resolve: Any) -> None:
        if skip or case_name in checked:
            return
        checked[case_name] = None
        smallest = CASES[case_name].ladder[0].label
        paths_ = resolve(case_name, smallest)
        objectives = {a: objective(a, case_name, smallest, paths_, ENGINE.get(a)) for a in arms}
        lo, hi = min(objectives.values()), max(objectives.values())
        if abs(hi - lo) / max(abs(lo), 1e-12) > GATE_RTOL:
            raise AssertionError(f'{case_name}/{smallest}: arms disagree on the objective — {objectives}')

    return check


def shape_of(case_name: str, size: str) -> Shape:
    return CASES[case_name].shape(size)
