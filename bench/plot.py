"""Refresh the chart page's numbers.

    pixi run -e bench python -m bench.plot

The page is a tracked source file, not a build artifact: its markup, prose and
renderer are edited by hand and reviewed in the diff. Only the measurements go
stale, so this rewrites exactly one line of it — the ``const DATA = {...};``
literal — and touches nothing else. Templating the page instead would move the
interesting part (what the bands *say*) into a file nobody opens.

One panel per model and sink, one line per library, log on both axes — which is
the only shape that shows a *slope*, and the slope is the claim. A table can
say a library is behind at one size; only the curve says whether it is falling
further behind or catching up.

**The band around each line is that measurement's own rounds**, minimum to
maximum. It is not a confidence interval and not the spread across models: it
is what the machine did to the same work nine times over, so a line whose band
overlaps another's is two numbers this run cannot tell apart. `bench/report.py`
marks the same doubt with `~` where it exceeds a quarter of the median; here it
is drawn, which is the one thing a table cannot do.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from bench import results as bench_results

#: What the page calls each library. Only lpspec is renamed — the page is about
#: our engine and `polars` is what the reader sees named in the architecture —
#: and anything unlisted keeps the name the harness measured it under.
NAME = {'lpspec': 'polars'}

#: The rungs the page plots, in order. Width rungs are their own axis and are
#: not mixed in: `w10` and `s` are the same size through different shapes, so a
#: single curve through both would read as one model growing.
LADDER = ('xs', 's', 'm', 'l')
_DATA = re.compile(r'^const DATA = .*;$', re.MULTILINE)


def measurements(name: str) -> Path:
    """The results file called *name*, in whichever format is committed."""
    for suffix in ('.json', '.jsonl'):
        path = Path('bench/results') / f'{name}{suffix}'
        if path.exists():
            return path
    raise SystemExit(f'no bench/results/{name}.json — run the ladder first (bench/README.md)')


def series(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """``(case, sink, arm) -> rung -> what one panel line needs at that rung``.

    ``wall`` is the minimum, which is what the tables publish, and the band runs
    from there up to the third quartile — so it *contains* its own line and says
    how much slower the rest of the rounds were.

    Not the first quartile at the bottom: q1 is above the minimum by
    construction, so a q1–q3 band floats above the line it belongs to and reads
    as a drawing mistake, which is what it looked like. Not the maximum at the
    top either — one nine-round measurement here read
    ``[1.18, 1.02, 1.07, 1.02, 1.02, 1.02, 1.06, 1.45, 9.97]``, and a band drawn
    to that outlier is ten times the height of the model it belongs to.

    A measurement taken without `isolate=True` has no peak and is dropped rather
    than plotted as zero.
    """
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in bench_results.records(path):
        if record.get('record') != 'timing' or 'error' in record:
            continue
        if record.get('peak_rss_bytes') is None or record['size'] not in LADDER:
            continue
        key = (record['case'], record.get('sink', 'lp'), record['arm'])
        out.setdefault(key, {})[record['size']] = {
            'wall': record['wall_seconds'],
            'lo': record['wall_seconds'],
            'hi': record.get('q3_seconds') or record['wall_seconds'],
            'peak': record['peak_rss_bytes'] / 1e9,
            'vars': (record.get('counts') or {}).get('columns') or record.get('nominal_variables'),
        }
    return out


def panels(taken: dict[tuple[str, str, str], dict[str, Any]], ceilings: list[dict[str, Any]]) -> dict[str, Any]:
    """One panel per (case, sink): a shared rung axis, and a line per library.

    Shared, with ``null`` where a library has no measurement, because the panel
    feeds both the chart and the table under it. The chart skips a null; the
    table prints what the run actually decided there — ``>30 s`` where the time
    budget stopped that library, an em dash where it simply has no number.

    A library that cannot reach a sink is absent from the panel rather than
    present and empty: `gurobipy` has no HiGHS, and a row of dashes says the
    measurement was missed rather than impossible.
    """
    out: dict[str, Any] = {}
    for (case, sink, arm), rungs in sorted(taken.items()):
        title = f'{case} — {sink}'
        panel = out.setdefault(title, {'case': case, 'sink': sink, 'series': {}, 'rungs': []})
        for rung in LADDER:
            if rung in rungs and rung not in panel['rungs']:
                panel['rungs'].append(rung)
        panel['series'][NAME.get(arm, arm)] = {'arm': arm, 'at': rungs}

    stopped = {(c['case'], c['sink'], c['arm']): c for c in ceilings}
    for panel in out.values():
        order = [r for r in LADDER if r in panel['rungs']]
        panel['rungs'] = order
        panel['vars'] = [next(s['at'][r]['vars'] for s in panel['series'].values() if r in s['at']) for r in order]
        for line in panel['series'].values():
            at, ceiling = line.pop('at'), stopped.get((panel['case'], panel['sink'], line.pop('arm')))
            for key in ('wall', 'lo', 'hi', 'peak'):
                line[key] = [round(at[r][key], 4) if r in at else None for r in order]
            line['bound'] = [
                f'>{ceiling["budget"]:g} s'
                if r not in at and ceiling and order.index(r) > order.index(ceiling['size'])
                else None
                for r in order
            ]
    return out


def main() -> int:
    path = measurements('latest')
    taken = series(path)
    ceilings = [r for r in bench_results.records(path) if r.get('record') == 'ceiling']
    if not taken:
        raise SystemExit('bench/results/latest.json has no plottable measurement — was it run with --benchmark-memory?')
    data = {'panels': panels(taken, ceilings), 'rungs': list(LADDER)}

    page = Path('docs/about/benchmarks-scaling.html')
    text = page.read_text()
    if not _DATA.search(text):
        raise SystemExit(f'{page} has no `const DATA = ...;` line — keep the literal on one line of its own')
    page.write_text(_DATA.sub(lambda _: 'const DATA = ' + json.dumps(data) + ';', text, count=1))
    print(f'{page} refreshed: {len(data["panels"])} panels')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
