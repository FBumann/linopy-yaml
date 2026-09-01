"""pytest-benchmark JSON in, flat measurement records out.

`bench/report.py` and `bench/plot.py` render the published tables and the chart
page, and between them they are four hundred lines of formatting that has been
read against the numbers it prints. When the harness became pytest, the cheap
and safe move was to leave every one of those lines alone and change only what
feeds them — so this module speaks the record shape they already read:

    {'record': 'timing', 'case', 'size', 'arm', 'sink',
     'wall_seconds', 'fastest_seconds', 'q1_seconds', 'q3_seconds', 'iqr', 'median', 'rounds',
     'peak_rss_bytes', 'peak_bytes', 'allocations',
     'counts': {...}, 'live_fraction'}
    {'record': 'loop',   'case', 'size', 'arm',
     'first_build_seconds', 'steady_build_seconds'}
    {'record': 'run',    'platform', 'machine', 'cpu', 'cores', 'python', 'versions', 'commits'}
    {'record': 'ceiling','case', 'size', 'sink', 'arm', 'ladder', 'budget', 'memory_budget',
     'stopped_by', 'reason'}

**Where each number comes from.** ``wall_seconds`` is pytest-benchmark's own
``median`` — and it used to be ``min``, on the rule that noise only ever adds.
That rule is right for A/B-ing one engine against itself, where both sides get
the same treatment. It is wrong for a table comparing libraries, twice over:

- **The minimum is a best-of-n and n is not equal.** pytest-benchmark calibrates
  by duration, so a fast cell here took 84 rounds and a slow one took 9. The
  expected minimum of 84 draws is not the expected minimum of 9.
- **The mean is worse.** These distributions have a right tail that belongs to
  the machine rather than to the library: one round in forty of a 20 ms
  measurement took 1.5 s, which drags its mean to 2.9x its median. One
  scheduler hiccup should not set a published number.

The median needs five bad rounds out of nine to move, which is no longer a
hiccup. ``fastest_seconds`` keeps the old number beside it. ``peak_rss_bytes`` is the minimum
of pytest-benchmem's ``rss_bytes`` series, the whole-process high-water mark it
records under ``benchmem(isolate=True)`` and the only memory number honest
across two libraries (see `bench/test_ladder.py`). ``first`` and ``steady`` are
read off the per-round series: round 0 is the cold build, and the minimum of
the rest is what a rolling horizon pays.

``q1_seconds`` and ``q3_seconds`` are the middle half of the same rounds, and
they are what the chart page draws as a band. Not min to max: one nine-round
measurement here read
``[1.18, 1.02, 1.07, 1.02, 1.02, 1.02, 1.06, 1.45, 9.97]``, so its envelope
would have been a single outlying round drawn ten times the height of the
model it belongs to. The quartiles say where the work actually sits, and a
band that overlaps another line's is two numbers this run cannot tell apart.

**The distribution rides along with the minimum.** ``iqr``, ``median`` and
``rounds`` are carried so the report can say whether a published minimum is
trustworthy: a run whose every round was slow prints a clean-looking minimum
and nothing else in the record contradicts it (#797). They are a quality
signal, not a second headline — and ``iqr`` over ``median`` is now a ratio of
the same distribution the tables publish, which it was not while they printed
the minimum.

**The record names the machine, because the ladder no longer runs on one.**
``cpu`` is pytest-benchmark's ``machine_info.cpu.brand_raw`` and ``cores`` its
count — collected all along, dropped here in favour of ``platform.processor()``,
which answers ``x86_64`` on every Linux box there is. The sinks measure in
separate jobs (#1315) and a runner pool mixes CPU models, so without these two
a merged table prints one provenance line for rows taken on two machines.

**A cell nobody measured is a result too.** A library the time budget stopped
leaves no benchmark entry at all, so its ceiling rides in a `.ceilings.json`
beside the measurements and comes back as a `ceiling` record. Without it the
tables print one em dash for *too slow to measure* and the same em dash for
*this library has no HiGHS*, which are different answers.

**One thing the old runner did that this does not: record a failure as a
result.** It caught a child that died, kept the exception line, and the report
rendered it as a cell — which is how `docs/about/benchmarks.md` publishes that the
eager lane runs out of memory at a rung the relational one survives. Under
pytest a dead pass is an error, and a real OOM kills the process before
anything in it can record why. The readers still understand an ``error`` record
and will render one; nothing produces it yet. Until that is resolved, the top
rung is a claim made by a run that failed rather than by a table.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from bench.cases import CASES

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

#: pytest-benchmem's blob inside pytest-benchmark's ``extra_info``.
BENCHMEM = 'benchmem'


def _nominal(case: str | None, size: str | None) -> int | None:
    """The rung's declared width — a property of (case, rung), not of the run.

    Looked up rather than read back out of the result file: it is the x axis of
    every scaling table, and a file written by a run that forgot to record it
    would sort the tables by ``None``.
    """
    try:
        return CASES[case].shape(size).nominal_variables  # pyrefly: ignore[bad-argument-type]
    except (KeyError, TypeError):
        return None


def _commit(info: dict[str, Any]) -> str | None:
    head = (info.get('id') or '')[:7]
    if not head:
        return None
    return f'{head}-dirty' if info.get('dirty') else head


def _benchmem(extra: dict[str, Any], field: str) -> float | None:
    """One pytest-benchmem series, reduced across repeats. ``None`` without `isolate=True`.

    Minimum for every field, the same rule as the wall clock: `rss_bytes` and
    `peak_bytes` are high-water marks that interference only pushes up, and
    `allocations` is a count the measured region owns, so a repeat above the
    smallest one is the harness paying for something the model did not ask for.
    """
    series = (extra.get(BENCHMEM) or {}).get(field)
    return min(float(v) for v in series) if series else None


def _counts(extra: dict[str, Any]) -> dict[str, Any]:
    return {k: extra.get(k) for k in ('columns', 'rows', 'nonzeros')}


def records(path: Path) -> Iterator[dict[str, Any]]:
    """Every measurement in *path*, in the shape the report and the plot read.

    Two formats, because the harness changed and the committed results did
    not. ``.jsonl`` is what the runner before #448 wrote — one record per
    line, already in this shape — and it is still what `bench/results/` holds,
    so the reader takes it verbatim. `.json` is pytest-benchmark's document,
    which the rest of this function unpacks.

    ``versions`` is stamped by ``pytest_benchmark_update_machine_info`` in
    ``bench/conftest.py``. The commit — dirty flag included — is what says
    which working tree produced a number, because an editable install reports
    the version it was synced at rather than the tree that ran.

    ``nominal_variables`` is the x of every scaling table; ``columns`` is what
    survived the mask and is a different number. ``first_build_seconds`` is
    round 0 and never competes with the steady rounds: the two answer
    different questions.
    """
    if path.suffix == '.jsonl':
        for line in path.read_text().splitlines():
            if line.strip():
                yield json.loads(line)
        return

    ceilings = path.with_suffix('.ceilings.json')
    if ceilings.exists():
        yield from json.loads(ceilings.read_text())

    doc = json.loads(path.read_text())
    machine, commit = doc.get('machine_info', {}), doc.get('commit_info', {})
    yield {
        'record': 'run',
        'platform': machine.get('system', '') + ' ' + machine.get('release', ''),
        'machine': machine.get('machine'),
        'cpu': (machine.get('cpu') or {}).get('brand_raw') or machine.get('processor'),
        'cores': (machine.get('cpu') or {}).get('count'),
        'python': machine.get('python_version'),
        'versions': machine.get('versions', {}),
        'commits': {'lpspec': _commit(commit)},
    }

    for b in doc.get('benchmarks', []):
        params, extra, stats = b.get('params') or {}, b.get('extra_info') or {}, b.get('stats') or {}
        common = {
            'case': params.get('case_name'),
            'size': params.get('size'),
            'arm': params.get('arm'),
            'nominal_variables': _nominal(params.get('case_name'), params.get('size')),
        }
        if b['name'].startswith('test_rebuild'):
            series = stats.get('data') or []
            yield {
                **common,
                'record': 'loop',
                'first_build_seconds': series[0] if series else None,
                'steady_build_seconds': min(series[1:]) if len(series) > 1 else None,
                'counts': _counts(extra),
            }
            continue
        yield {
            **common,
            'record': 'timing',
            'sink': params.get('sink'),
            'wall_seconds': stats.get('median'),
            'fastest_seconds': stats.get('min'),
            'q1_seconds': stats.get('q1'),
            'q3_seconds': stats.get('q3'),
            'iqr': stats.get('iqr'),
            'median': stats.get('median'),
            'rounds': stats.get('rounds'),
            'peak_rss_bytes': _benchmem(extra, 'rss_bytes'),
            'peak_bytes': _benchmem(extra, 'peak_bytes'),
            'allocations': _benchmem(extra, 'allocations'),
            'counts': _counts(extra),
            'live_fraction': extra.get('live_fraction'),
        }


def bound_label(ceiling: Mapping[str, Any]) -> str:
    """What a cell above *ceiling* prints — the budget that stopped the climb, in its own unit.

    Two budgets guard a rung and either can stop it (`bench/conftest.py`), so
    which one did is what the cell means: a run held to 6 GB publishes every
    stop as `>30 s` if the seconds are read regardless, naming a limit the arm
    was nowhere near — `linopy` was stopped on `transport/m` for projecting
    8.94 GB after 0.894 s.

    A sidecar written before the harness recorded `stopped_by` carries no
    memory budget either, so seconds is both the fallback and what those runs
    actually enforced.
    """
    if ceiling.get('stopped_by') == 'memory':
        return f'>{ceiling["memory_budget"]:g} GB'
    return f'>{ceiling["budget"]:g} s'


def files(target: Path) -> list[Path]:
    """Every result file under *target*, or *target* itself when it is one.

    The readers took a written-out list of three names, two of which no run had
    ever produced — so `pixi run report` raised ``FileNotFoundError`` on a clean
    checkout, and the list could not be right both before a refresh and after
    it. A directory cannot go stale.

    ``.jsonl`` first and pytest-benchmark's ``.json`` after: the historic files
    are the older measurements, and the readers collapse repeats by minimum in
    the order they are given.

    Nor is ``casualties.json``: it is a list the watchdog appends to when it
    kills a cell that exhausted the machine, and reading it as a run document
    is an `AttributeError` halfway through a render — the same failure the
    sidecar below describes.

    A ``.ceilings.json`` is not a results file — it is the sidecar naming what a
    run refused to measure, and `records` picks it up beside the measurements it
    belongs to. Reading it as a run of its own means parsing a list as a
    document, which is an `AttributeError` halfway through a render.
    """
    if target.is_dir():
        found = sorted(target.glob('*.jsonl')) + sorted(target.glob('*.json'))
        return [p for p in found if not p.name.endswith(('.ceilings.json', 'casualties.json'))]
    return [target]


def load(*paths: Path) -> list[dict[str, Any]]:
    """Flatten several result files, newest last — the readers take as many as given."""
    return [r for p in paths for f in files(p) for r in records(f)]
