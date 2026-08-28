"""The harness: selection, and the data every arm reads.

Every moving part is pytest's own: the case x size x sink x arm product is a
`parametrize`, isolation is `benchmem(isolate=True)`, repetition and its
minimum are pytest-benchmark's rounds, and the output is `--benchmark-json`.
Nothing here re-implements a runner.

**There is one arm, so there is no parity gate.** It compared the arms'
objectives at the smallest rung, which is a check with one counterparty — and
with one arm it has none. It returns with the second arm, where it means
something again; until then `bench.floor`'s ``--check`` is the only
same-model check here, and `test_ladder._record` is the arithmetic one that
runs on every measurement.

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
import re
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from bench.arms import ARMS
from bench.cases import CASES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bench.cases import Shape


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup('ladder', 'the lpspec benchmark ladder')
    g.addoption('--cases', nargs='+', default=sorted(CASES), choices=sorted(CASES))
    g.addoption('--sizes', nargs='+', default=['xs', 's', 'm'], help="rung labels, or 'all' for every rung a case has")
    g.addoption('--arms', nargs='+', default=sorted(ARMS), choices=sorted(ARMS))
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
    g.addoption(
        '--budget',
        type=float,
        default=120.0,
        help='seconds a measurement may take before its arm stops climbing that ladder; 0 measures everything',
    )
    g.addoption(
        '--memory-budget',
        type=float,
        default=0.0,
        help='GB a measurement may take before its arm stops climbing that ladder; 0 measures everything',
    )
    g.addoption(
        '--i-know-another-is-running',
        action='store_true',
        help='start despite the machine lock or a high load average (#705); the numbers are on you',
    )


#: Rounds every measurement gets at least, which is what `docs/about/benchmarks.md`
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


#: How a rung label says which ladder it belongs to. `bench/report.py` renders
#: each as its own table for the same reason the budget decides each on its own:
#: `w10` and `s` carry the same variables through different shapes.
_LADDERS = (
    ('width', re.compile(r'w\d+$')),
    ('density', re.compile(r'd\d+$')),
    ('declarations', re.compile(r'n\d+$')),
)


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


#: The file the published tables are drawn from. A run narrower than the ladder
#: may not replace it.
COMMITTED = Path('bench/results/latest.json')


def published_rungs() -> set[str]:
    """The rungs `pixi run ladder` takes, read from the task that defines it."""
    import tomllib

    manifest = Path(__file__).resolve().parents[1] / 'pyproject.toml'
    task = tomllib.loads(manifest.read_text())['tool']['pixi']['feature']['bench']['tasks']['ladder']
    sizes = next(a['default'] for a in task['args'] if a['arg'] == 'sizes')
    return set(sizes.split())


def refuse_to_overwrite_the_provenance(config: pytest.Config) -> None:
    """A short run may not write the file the published tables are drawn from.

    The hazard is not a wasted afternoon, it is a silent one: a smoke test aimed
    at `latest.json` replaces every published table's provenance with four
    measurements, and nothing about the resulting file looks wrong afterwards.
    `bench/README.md` has warned about it in prose since the harness became
    pytest; this is the same sentence where it can be enforced.

    Narrower *sinks* or *libraries* are allowed — the scheduled run takes one
    sink per job, so each half writes a file naming the sink it measured and
    `bench.report` reads the pair. Narrower *rungs* are what makes a run a
    smoke test.

    Raises:
        pytest.UsageError: If the run is missing a published rung and would
            still write the committed file.
    """
    destination = next(
        (arg.split('=', 1)[1] for arg in config.invocation_params.args if arg.startswith('--benchmark-json=')), None
    )
    if not destination or Path(destination).resolve() != (Path.cwd() / COMMITTED).resolve():
        return
    missing = published_rungs() - set(config.getoption('--sizes'))
    if missing:
        raise pytest.UsageError(
            f'this run leaves out {sorted(missing)}, so it cannot write {COMMITTED} — the published '
            f'tables are drawn from that file and a shorter run replaces them with fewer rows, '
            f'silently. Point --benchmark-json somewhere else, or take the whole ladder '
            f'(`pixi run ladder`).'
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
    refuse_to_overwrite_the_provenance(session.config)
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
    'highspy',
    'gurobipy',
    'scipy',
    'polars',
    'pandas',
    'numpy',
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
def builds(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption('--builds'))


#: Where a ladder stopped, and why. Read by `pytest_terminal_summary`, which is
#: the only place a skipped cell can still say something.
CEILINGS = pytest.StashKey[dict]()


class Ceiling:
    """Which arms have stopped climbing which ladders, and the sentence why.

    An arm that builds per entity costs roughly what the rung is wide, and the
    rungs grow tenfold — so one measurement is enough to know the next one is
    out of reach. *That a library is far slower is the finding*; a number
    measured over an hour to say it again is a machine kept busy for nothing.

    The projection is stated in the reason and never recorded as a measurement:
    it is arithmetic on one rung, not a second rung.
    """

    def __init__(self, budget: float, stash: dict, selected: Sequence[str], memory: float = 0.0) -> None:
        self.budget = budget
        self.memory = memory
        self.reasons = stash
        self.selected = selected

    def reached(self, arm: str, case_name: str, size: str, sink: str) -> str | None:
        found = self.reasons.get((arm, case_name, ladder_of(size), sink))
        return found[1] if found else None

    def rows(self) -> list[dict[str, Any]]:
        """Every ceiling as a record, for the file beside the measurements.

        A cell nobody measured leaves no benchmark entry, so without this the
        only trace is a line in the terminal — and the table then prints the
        same em dash for *too slow to measure* as it does for a sink the
        library cannot reach. They are different answers and the reader is
        entitled to both.

        ``stopped_by`` names which of the two budgets did it. Both are on the
        record and only one of them fired, so a renderer reading the wrong one
        publishes a limit the arm was nowhere near.
        """
        return [
            {
                'record': 'ceiling',
                'arm': arm,
                'case': case_name,
                'ladder': ladder,
                'sink': sink,
                'size': size,
                'budget': self.budget,
                'memory_budget': self.memory,
                'stopped_by': stopped_by,
                'reason': reason,
            }
            for (arm, case_name, ladder, sink), (size, reason, stopped_by) in self.reasons.items()
        ]

    def record(
        self,
        arm: str,
        case_name: str,
        size: str,
        sink: str,
        seconds: float | None,
        peak_bytes: float | None = None,
    ) -> None:
        """Take one measurement, and decide whether the next rung is worth taking.

        Two budgets, because a rung can be affordable in one and not the other:
        `transport/w100` on `linopy` takes 51 s and 31 GB, and it was the second
        that took the machine down with it (#1416). Memory is checked first —
        an over-time rung leaves a number behind, an over-memory one leaves the
        run with no runner.
        """
        key = (arm, case_name, ladder_of(size), sink)
        if self.memory and peak_bytes:
            self._memory(key, arm, case_name, size, peak_bytes)
            if key in self.reasons:
                return
        if not self.budget or seconds is None:
            return
        if seconds > self.budget:
            self.reasons[key] = (
                size,
                f'{arm} took {_seconds(seconds)} on {case_name}/{size}, over the {_seconds(self.budget)} budget',
                'time',
            )
            return
        projected = seconds * _growth(case_name, size, self.selected)
        if projected > self.budget:
            self.reasons[key] = (
                size,
                f'{arm} took {_seconds(seconds)} on {case_name}/{size}, so the next rung projects to '
                f'{_seconds(projected)} — over the {_seconds(self.budget)} budget',
                'time',
            )

    def _memory(self, key: tuple, arm: str, case_name: str, size: str, peak_bytes: float) -> None:
        """Stop the ladder when this rung, or the next, does not fit the machine."""
        peak = peak_bytes / 1e9
        if peak > self.memory:
            self.reasons[key] = (
                size,
                f'{arm} took {peak:.3g} GB on {case_name}/{size}, over the {self.memory:.3g} GB budget',
                'memory',
            )
            return
        projected = peak * _growth(case_name, size, self.selected)
        if projected > self.memory:
            self.reasons[key] = (
                size,
                f'{arm} took {peak:.3g} GB on {case_name}/{size}, so the next rung projects to '
                f'{projected:.3g} GB — over the {self.memory:.3g} GB budget',
                'memory',
            )


def _seconds(value: float) -> str:
    """Seconds at a precision that survives both ends: a 0.013 s rung and a 4000 s projection."""
    return f'{value:.3g} s'


def ladder_of(size: str) -> str:
    """Which ladder a rung belongs to — its own, or the size one.

    A case can carry more than one, and they ask different questions: `l` is the
    longest model and `w1000` is the widest. A library that cannot afford the
    top of one may be perfectly able to climb the other, so the budget has to
    decide them separately.
    """
    return next((name for name, pattern in _LADDERS if pattern.match(size)), 'size')


def _growth(case_name: str, size: str, selected: Sequence[str]) -> float:
    """How much wider the next rung *of this run* is. 1.0 when there is none.

    Read off the rungs the run was asked for rather than off everything the case
    defines. A ladder measured to `l` has `xl` and `2xl` behind it, four and
    twelve times wider, and projecting onto a rung nobody asked for stops the
    climb on arithmetic about work that was never going to happen — which is
    what it did on the first published run, taking the width ladder down with
    it because the ceiling outlived the ladder that set it.
    """
    ladder = ladder_of(size)
    rungs = [s for s in CASES[case_name].ladder if s.label in set(selected) and ladder_of(s.label) == ladder]
    labels = [s.label for s in rungs]
    if size not in labels or labels.index(size) + 1 >= len(rungs):
        return 1.0
    here, following = rungs[labels.index(size)], rungs[labels.index(size) + 1]
    return following.nominal_variables / here.nominal_variables if here.nominal_variables else 1.0


@pytest.fixture(scope='session')
def ceiling(request: pytest.FixtureRequest) -> Ceiling:
    stash = request.config.stash.setdefault(CEILINGS, {})
    return Ceiling(
        float(request.config.getoption('--budget')),
        stash,
        request.config.getoption('--sizes'),
        float(request.config.getoption('--memory-budget')),
    )


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    """Every ladder an arm stopped climbing, printed where the run ends.

    A skipped cell leaves no record in the results file — a measurement nobody
    took is an absent row, not a null — so this is the one place the *reason*
    survives the run.
    """
    del exitstatus
    reasons = config.stash.get(CEILINGS, {})
    if not reasons:
        return
    terminalreporter.write_sep('-', 'over budget')
    for key in sorted(reasons):
        where = f' [{key[-1]} sink]' if key[-1] else ''
        terminalreporter.write_line(f'{reasons[key][1]}{where}')
    _write_ceilings(config)


def _write_ceilings(config: pytest.Config) -> None:
    """The ceilings, beside the file the measurements went to.

    Named from `--benchmark-json` rather than fixed, so a scratch run's ceilings
    land with its scratch results and never overwrite the committed ones. Read
    off the command line rather than off `config.option`: pytest-benchmark keeps
    that path in its own storage object, and the attribute a plugin does not
    promise is one that goes missing on an upgrade without failing.
    """
    import json

    reasons = config.stash.get(CEILINGS, {})
    destination = next(
        (arg.split('=', 1)[1] for arg in config.invocation_params.args if arg.startswith('--benchmark-json=')), None
    )
    if not destination or not reasons:
        return
    stash = Ceiling(
        float(config.getoption('--budget')),
        reasons,
        config.getoption('--sizes'),
        float(config.getoption('--memory-budget')),
    )
    path = Path(str(destination)).with_suffix('.ceilings.json')
    path.write_text(json.dumps(stash.rows(), indent=1))


def shape_of(case_name: str, size: str) -> Shape:
    return CASES[case_name].shape(size)
