"""Refresh the chart page's numbers.

    uv run python -m bench.plot

The page is a tracked source file, not a build artifact: its markup, prose and
renderer are edited by hand and reviewed in the diff. Only the measurements go
stale, so this rewrites exactly one line of it — the ``const DATA = {...};``
literal — and touches nothing else. Templating the page instead would move the
interesting part (what the bands *say*) into a file nobody opens.

Two result files, because the page plots two runs: the ladder (six models to
`l`, through the `highs` sink) and the scaling run (`dispatch` alone out to
120M, through `lp`). A rung either run is missing stops this rather than
drawing a line that skips it, which would read as a measurement.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from bench import results as bench_results

NAME = {'lpspec': 'polars', 'linopy': 'linopy'}
LADDER = ('xs', 's', 'm', 'l')
SCALING = ('xs', 's', 'm', 'l', 'xl', '2xl')
_DATA = re.compile(r'^const DATA = .*;$', re.MULTILINE)


def measurements(name: str) -> Path:
    """The results file called *name*, in whichever format is committed.

    The harness writes `.json` now; `bench/results/` still holds the `.jsonl`
    the previous one wrote. Preferring the new and falling back to the old is
    what lets this run against the tree as it stands rather than only after
    the next full ladder.
    """
    for suffix in ('.json', '.jsonl'):
        path = Path('bench/results') / f'{name}{suffix}'
        if path.exists():
            return path
    raise SystemExit(f'no bench/results/{name}.json — run the ladder first (bench/README.md)')


def best(path: Path, sink: str) -> dict[str, dict[Any, Any]]:
    """``(case, size, arm) -> fastest repeat``. Minimum, because noise only adds.

    A measurement taken without `benchmem(isolate=True)` has no peak, and the
    page plots one — so such a record is dropped here rather than divided by a
    billion halfway through the render. What the reader then gets is `_at`
    naming the point the run is missing, which says what to re-measure.
    """
    out: dict[str, dict[Any, Any]] = {'wall': {}, 'peak': {}, 'cols': {}}
    for r in bench_results.records(path):
        if r.get('record') != 'timing' or 'error' in r or r.get('sink') != sink:
            continue
        if r.get('peak_rss_bytes') is None:
            continue
        k = (r['case'], r['size'], r['arm'])
        out['wall'][k] = min(out['wall'].get(k, 9e99), r['wall_seconds'])
        out['peak'][k] = min(out['peak'].get(k, 9e99), r['peak_rss_bytes'] / 1e9)
        out['cols'][(r['case'], r['size'])] = r['counts']['columns']
    return out


def _at(table: dict[Any, Any], key: Any) -> Any:
    if key not in table:
        raise SystemExit(f'no measurement for {key} — the page plots that point, so the run has to cover it')
    return table[key]


def panel(t: dict[str, dict[Any, Any]], case: str, rungs: tuple[str, ...], arms: tuple[str, ...]) -> dict[str, Any]:
    return {
        'vars': [_at(t['cols'], (case, r)) for r in rungs],
        **{m: {NAME[a]: [round(_at(t[m], (case, r, a)), 3) for r in rungs] for a in arms} for m in ('wall', 'peak')},
    }


def main() -> int:
    ladder = best(measurements('latest'), 'highs')
    scaling = best(measurements('scaling'), 'lp')
    cases = sorted({c for c, _, _ in ladder['wall']})
    data = {
        'scaling': panel(scaling, 'dispatch', SCALING, ('lpspec', 'linopy')),
        'cases': {c: panel(ladder, c, LADDER, ('lpspec', 'linopy')) for c in cases},
        'caseNames': cases,
        'rungs': list(LADDER),
    }

    page = Path('docs/benchmarks-scaling.html')
    text = page.read_text()
    if not _DATA.search(text):
        raise SystemExit(f'{page} has no `const DATA = ...;` line — keep the literal on one line of its own')
    page.write_text(_DATA.sub(lambda _: 'const DATA = ' + json.dumps(data) + ';', text, count=1))
    print(f'{page} refreshed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
