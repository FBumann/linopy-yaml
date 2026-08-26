"""The PyPSA ladder pages: one per rung, in the gallery's shape, from what the parity runner committed.

    pixi run python -m tools.ladder           # rewrite docs/examples/pypsa_ladder.md and docs/examples/pypsa_ladder/*.md
    pixi run python -m tools.ladder --check   # fail if any has drifted

A rung's page is its projected model as math, then `lpspec` beside `PyPSA`:
the projected YAML, the binding that makes its tables from the network, and
the solve — against the PyPSA script that builds the same network and
optimises it. Below: the tables the rung is the first to declare, the rows
lpspec built, and how deep the comparison went. Every fence is a committed
file under ``differential/pypsa/`` (the projection, the script, the tables)
or derived from one (the binding slice, from ``prep.py``); the runner wrote
them from the pinned math-spec and the ``PyPSA parity`` workflow holds them
there. Nothing here runs pypsa.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml
from math_spec import SymbolTable, to_markdown

ROOT = Path(__file__).resolve().parent.parent
LADDER = ROOT / 'differential' / 'pypsa'
RUNGS = LADDER / 'rungs'
PAGES = ROOT / 'docs' / 'examples' / 'pypsa_ladder'
INDEX = ROOT / 'docs' / 'examples' / 'pypsa_ladder.md'
CORPUS_PAGE = 'https://math-spec.readthedocs.io/en/latest/examples/pypsa/'


def stems() -> list[str]:
    return sorted(path.stem for path in RUNGS.glob('rung_*.yaml') if not path.stem.endswith('.symbols'))


def _indent(text: str) -> str:
    return '\n'.join(f'    {line}' if line else '' for line in text.splitlines())


def _title(stem: str) -> str:
    """The rung's own first docstring line — ``Rung 2: storage — …``."""
    first = (RUNGS / f'{stem}.py').read_text().splitlines()[0]
    return first.strip('"').rstrip('.')


def prep_slice(declared: list[str]) -> str:
    """The lines of ``prep.sources`` that make *declared*, with the helpers they call — the rung's binding."""
    text = (LADDER / 'prep.py').read_text()
    body = text[text.index('def sources(') :]
    entries = dict(re.findall(r"^        '(\w+)': (.+?),\n(?=        '|    \})", body, flags=re.DOTALL | re.MULTILINE))
    tail = re.findall(r"^    tables\['(\w+)'\] = (.+)$", body, flags=re.MULTILINE)
    entries.update(tail)
    helpers = {
        m.group(1): m.group(0).rstrip()
        for m in re.finditer(r'^def (_\w+)\(.*?(?=^def |\Z)', text, flags=re.DOTALL | re.MULTILINE)
    }
    lines = [f'    {name!r}: {entries[name]},' for name in declared if name in entries]
    used = sorted({h for h in helpers if any(f'{h}(' in line for line in lines)})
    closure = set(used)
    for h in used:
        closure |= {g for g in helpers if f'{g}(' in helpers[h]}
    return (
        '\n\n\n'.join(helpers[h] for h in sorted(closure))
        + ('\n\n\n' if closure else '')
        + 'n = build()  # the network from the PyPSA tab\n\nsources = {\n'
        + '\n'.join(lines)
        + '\n}'
    )


def lpspec_tab(stem: str, projection: dict, record: dict) -> str:
    declared = [*projection['dimensions'], *projection.get('lookups', {}), *projection['parameters']]
    model = (RUNGS / f'{stem}.yaml').read_text().rstrip()
    binding = prep_slice(declared)
    call = (
        f'{binding}\n\n'
        f"with lps.solve('differential/pypsa/rungs/{stem}.yaml', sources) as solution:\n"
        f'    solution.objective  # {record["parity"]["lpspec_objective"]!r}'
    )
    return (
        '=== "lpspec"\n\n'
        f'{_indent(f"The model, `differential/pypsa/rungs/{stem}.yaml` — the file projected onto what this rung builds:")}\n\n'
        f'{_indent(f"```yaml{chr(10)}{model}{chr(10)}```")}\n\n'
        f'{_indent("The binding — every table the model declares, from the network — and the solve:")}\n\n'
        f'{_indent(f"```python{chr(10)}{call}{chr(10)}```")}\n'
    )


def pypsa_tab(stem: str, record: dict) -> str:
    script = (RUNGS / f'{stem}.py').read_text().rstrip()
    solve = f"n = build()\nn.optimize(solver_name='highs')\nn.objective  # {record['pypsa_objective']!r}"
    return (
        '=== "PyPSA"\n\n'
        f'{_indent(f"The network, `{stem}.py` in the corpus — the spine plus what this rung adds:")}\n\n'
        f'{_indent(f"```python{chr(10)}{script}{chr(10)}```")}\n\n'
        f'{_indent(f"```python{chr(10)}{solve}{chr(10)}```")}\n'
    )


def _tables(stem: str) -> str:
    files = sorted((LADDER / 'tables' / stem).glob('*.csv'))
    if not files:
        return 'Every table this model declares was first declared by a lower rung; its values here are in the binding above.'
    return (
        f'The tables this rung is the first to declare ({len(files)}), as the binding produced them:\n\n'
        + '\n\n'.join(f'`{p.name}`\n\n```csv\n{p.read_text().rstrip()}\n```' for p in files)
    )


def _verdict(record: dict) -> str:
    parity, structural = record['parity'], record['structural']
    prices = parity['prices']
    priced = f'prices agree on {prices["compared"]} rows' if prices['compared'] else f'no prices — {prices["skipped"]}'
    proof = (
        f'**model for model**: {len(structural["equal"])} blocks equal, {len(structural["region"])} documented splits'
        if 'equal' in structural
        else f'objective only — `lpspec.linopy` stops at `{structural["error"]}`'
    )
    rows = '\n'.join(f'| `{name}` | {count} |' for name, count in parity['built_rows'].items() if count)
    return (
        f'> {"✔" if parity["matches"] else "✘"} Verified against pypsa 1.3.0 — objective **{parity["lpspec_objective"]}**'
        f' on both sides; {priced}; {proof}.\n\n'
        f'| block | rows built |\n| --- | ---: |\n{rows}'
    )


def _symbols(stem: str, projection: dict) -> SymbolTable | None:
    """The file's symbol table cut to what the projection declares — a table naming a dropped name is refused."""
    path = RUNGS / f'{stem}.symbols.yaml'
    if not path.exists():
        return None
    raw = yaml.safe_load(path.read_text())
    declared = {
        *projection['dimensions'],
        *projection.get('lookups', {}),
        *projection['parameters'],
        *projection['variables'],
    }
    return SymbolTable.load(
        {
            'notation': raw['notation'],
            'dimensions': {d: v for d, v in raw.get('dimensions', {}).items() if d in projection['dimensions']},
            'names': {n: v for n, v in raw.get('names', {}).items() if n in declared},
        }
    )


def page(stem: str, record: dict) -> str:
    projection = yaml.safe_load((RUNGS / f'{stem}.yaml').read_text())
    math = to_markdown(str(RUNGS / f'{stem}.yaml'), symbols=_symbols(stem, projection), legend=True)
    number = int(stem[5:7])
    return (
        f'# {_title(stem)}\n\n'
        f'<!-- generated by tools/ladder.py from differential/pypsa — do not edit -->\n\n'
        f'One rung of [the PyPSA corpus]({CORPUS_PAGE}#rung-{number}): the file `pypsa.yaml` projected onto what'
        f' this network builds, bound to that network, and held to what PyPSA solves it to.\n\n'
        f'{_verdict(record)}\n\n'
        f'## The model\n\n<details markdown="1">\n<summary>The same model, as math</summary>\n\n{math}\n</details>\n\n'
        f'{lpspec_tab(stem, projection, record)}\n'
        f'{pypsa_tab(stem, record)}\n'
        f'## The data\n\n{_tables(stem)}\n'
    )


def index(stamped: dict) -> str:
    rows = '\n'.join(
        f'| [{_title(s)}](pypsa_ladder/{s}.md) | {stamped[s]["parity"]["lpspec_objective"]} |'
        f' {"✔" if stamped[s]["parity"]["matches"] else "✘"} |'
        f' {"model for model" if "equal" in stamped[s]["structural"] else "objective and prices"} |'
        for s in stems()
    )
    return (
        '# The PyPSA ladder\n\n'
        '<!-- generated by tools/ladder.py from differential/pypsa — do not edit -->\n\n'
        f'[math-spec states PyPSA in one file]({CORPUS_PAGE}), grown a rung at a time; each rung here is that file'
        ' projected onto what its network builds, shown as lpspec builds it beside the PyPSA code that builds the'
        " same network. The `PyPSA parity` workflow regenerates every page's sources from the pinned math-spec on"
        ' each run and fails on a diff.\n\n'
        f'| rung | objective | agrees | proven |\n| --- | ---: | :-: | --- |\n{rows}\n'
    )


def rendered() -> dict[Path, str]:
    stamped = json.loads((LADDER / 'references.json').read_text())
    for s in stems():
        stamped[s]['pypsa_objective'] = stamped[s]['parity']['lpspec_objective']
    return {INDEX: index(stamped), **{PAGES / f'{s}.md': page(s, stamped[s]) for s in stems()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='fail if a committed page has drifted')
    args = ap.parse_args(argv)
    stale = []
    for path, text in rendered().items():
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(path.relative_to(ROOT))
        else:
            path.parent.mkdir(exist_ok=True)
            path.write_text(text)
    if stale:
        print(f'stale: {", ".join(map(str, stale))} — pixi run python -m tools.ladder', file=sys.stderr)
        return 1
    if not args.check:
        print(f'wrote {INDEX.relative_to(ROOT)} and {len(stems())} rung pages')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
