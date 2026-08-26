"""The parity gate: every rung of the PyPSA corpus, as deep as the engines allow.

    python differential/pypsa/parity.py <math-spec checkout>

The corpus is math-spec's — `examples/pypsa.yaml` and its quadratic sibling,
one `rung_*.py` per rung whose `build()` returns the PyPSA network with its
data inline, and `prep.py`, the binding that turns a network into the tables
the file declares. This file is the engine side: bind, build, solve, compare,
and it needs a checkout of that repository at the tag `pyproject.toml` pins,
which is what the `PyPSA parity` workflow hands it. Run with this tree's
lpspec, `pypsa==1.3.0` and `highspy` installed, and the `[linopy]` extra for
the model comparison. No pixi environment carries pypsa, so the way to run it
locally is the workflow's own line, which installs nothing on disk:

    pixi exec -s uv uv run --with-editable ".[linopy]" \
        --with "pypsa==1.3.0" --with "highspy==1.15.1" --with "polars>=1.30" \
        python differential/pypsa/parity.py ../math-spec

Per rung, from the same network, three comparisons:

1. **Model against model** — PyPSA's ``n.optimize.create_model()`` and
   ``lpspec.linopy.build``, label for label: coefficients, sense, right-hand
   side, bounds, integrality, objective terms. No solver, so it covers MIP
   and QP alike. The verdict speaks the index table's words: ``equal`` is
   the one block PyPSA builds — **done**; ``region`` is the same rows from
   several ``where:`` blocks — **split**; ``mismatch`` fails the run. A rung
   whose file `lpspec.linopy` cannot build yet stamps the error instead —
   the upstream hardening this gate waits on — and its proof stops at (2).
2. **One solved objective across the fence** — PyPSA's solve against
   `lpspec.relational`'s, both HiGHS, rtol 1e-9 on the generic spine.
3. **Coverage** — what the relational lane built per block, each
   dimension's size, the tables bound non-empty; and, over the ladder as a
   whole, that every block is built by some rung, every mask is partially
   true somewhere and every parameter is fed by some rung, so an equality is
   never over data that tests nothing.
4. **Prices across the fence** — PyPSA's ``buses_t.marginal_price`` against
   the relational lane's ``Bus_nodal_balance`` duals, per unit of the
   snapshot's objective weighting, which is how PyPSA reports them. A
   mixed-integer rung has no duals on our side and stamps why instead.

Primals are deliberately not compared — an optimum need not be unique — and
counts are not compared separately: they are a strict subset of (1).

The comparison reads linopy's own ``.flat`` export but does not call
``linopy.testing``: those asserts hold the raw datasets equal, and two
builders lay the same model out differently — PyPSA pads absent ``_term``
slots with NaN where lpspec writes -0.0, and term order within a row is the
builder's own. A canonicalizing ``assert`` upstream would shrink this file.
PyPSA's model is built before `lpspec.linopy` is imported: that import flips
linopy's global ``semantics`` option to ``v1`` and PyPSA speaks ``legacy``,
so the option is reset around each PyPSA build.

The stamps are rewritten into `references.json` beside this file on every
run, so the committed certificate is always what the last run of this tree
produced against the pinned corpus; the workflow fails on a diff, which is
how a stale stamp shows.
"""

from __future__ import annotations

import importlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

CORPUS = Path(sys.argv[1] if len(sys.argv) > 1 else 'corpus').resolve()
RUNGS = CORPUS / 'examples' / 'references' / 'pypsa'
RECORDS = Path(__file__).resolve().parent / 'references.json'
sys.path.insert(0, str(RUNGS))

import linopy  # noqa: E402
import math_spec  # noqa: E402
import prep  # noqa: E402  the corpus's own binding

import lpspec as lps  # noqa: E402


def rungs() -> list[str]:
    """Every rung, in ladder order — the scripts beside the corpus's spine."""
    return sorted(path.stem for path in RUNGS.glob('rung_*.py'))


def network(stem: str):
    """The rung's PyPSA network, built by its own script."""
    return importlib.import_module(stem).build()


def model_of(stem: str) -> Path:
    """The file the rung binds: ``MODEL`` in its script where it names one, ``pypsa.yaml`` otherwise."""
    return CORPUS / 'examples' / getattr(importlib.import_module(stem), 'MODEL', 'pypsa.yaml')


def stands_for(description: str | None) -> str:
    """The PyPSA name a declaration's description opens with, in backticks — the declared pages' convention."""
    return re.match(r'`([^`]+)`', description or '').group(1)


def bound(model: Path, n) -> dict[str, object]:
    """`prep.sources` cut to what *model* declares — lpspec refuses a key the model does not take."""
    declared = math_spec.load_model(model)
    names = {*declared.dimensions, *declared.parameters, *declared.lookups}
    return {name: table for name, table in prep.sources(n).items() if name in names}


def built(result, declared) -> tuple[dict[str, int], dict[str, int]]:
    """The labels the relational lane actually built, per file block — masked ones excluded, like PyPSA's records."""
    return (
        {name: len(result.activity(name)) for name in declared.constraints},
        {name: len(result.primal(name)) for name in declared.variables},
    )


def prices(result, n) -> dict[str, object]:
    """`Bus_nodal_balance` duals against PyPSA's `marginal_price`, per (snapshot, bus).

    PyPSA divides the row dual by the objective weighting; so does this. An
    integer variable leaves the lane without duals, and the stamp says so.
    """
    try:
        dual = result.dual('Bus_nodal_balance').to_pandas()
    except lps.LpspecError as error:
        return {'compared': 0, 'skipped': str(error).splitlines()[0][:120]}
    weights = n.snapshot_weightings['objective']
    theirs = n.buses_t.marginal_price
    gaps = [
        abs(row.value / weights[row.snapshot] - float(theirs.at[row.snapshot, row.bus])) for row in dual.itertuples()
    ]
    return {
        'compared': len(gaps),
        'max_abs_diff': round(max(gaps, default=0.0), 12),
        'matches': all(g <= 1e-6 for g in gaps),
    }


def pypsa_model(stem: str):
    """The network's own linopy model, built under the ``legacy`` semantics PyPSA speaks."""
    linopy.options['semantics'] = 'legacy'
    try:
        return network(stem).optimize.create_model()
    finally:
        linopy.options['semantics'] = 'v1'


def _keyed(labels) -> pd.Series:
    """label per coordinate key — dim names dropped, ``snapshot`` first, so the two spellings align."""
    series = labels.to_series()
    index = series.index
    if index.nlevels > 1:
        order = sorted(index.names, key=lambda name: (name != 'snapshot', name))
        series = series.reorder_levels(order).sort_index()
        series.index = pd.Index(series.index.to_flat_index())
    return series


def _label_map(theirs, ours, pairs: dict[str, list[str]]) -> dict[int, int]:
    """Our variable labels to theirs, matched by name pair and coordinate key."""
    mapping: dict[int, int] = {}
    for pypsa_name, our_names in pairs.items():
        their = _keyed(theirs.variables[pypsa_name].labels)
        for our_name in our_names:
            for key, our_label in _keyed(ours.variables[our_name].labels).items():
                their_label = int(their[key])
                if our_label != -1 and their_label != -1:
                    mapping[int(our_label)] = their_label
    return mapping


def _rows(flat: pd.DataFrame, labels, relabel) -> dict:
    """Constraint rows by coordinate key: (sign, rhs, sorted (variable, coefficient) pairs)."""
    terms = defaultdict(list)
    meta = {}
    for row in flat.itertuples():
        terms[row.labels].append((relabel(int(row.vars)), float(row.coeffs)))
        meta[row.labels] = (row.sign, float(row.rhs))
    return {
        key: (*meta[int(label)], tuple(sorted(terms[int(label)])))
        for key, label in _keyed(labels).items()
        if int(label) != -1
    }


def _objective(model, relabel) -> tuple:
    """The objective as a sorted term tuple — quadratic pairs unordered."""
    flat = model.objective.expression.flat
    terms = []
    for row in flat.itertuples():
        if hasattr(row, 'vars1'):
            pair = tuple(sorted((relabel(int(row.vars1)), relabel(int(row.vars2)))))
        else:
            pair = (relabel(int(row.vars)),)
        terms.append((pair, round(float(row.coeffs), 9)))
    return tuple(sorted(terms))


def compare(theirs, ours, declared, gc_kinds: dict[str, str]) -> dict[str, list[str]]:
    """Verdicts: which PyPSA names are model-equal, which are the same region in several blocks, which differ."""
    rows = defaultdict(list)
    for name, block in declared.constraints.items():
        rows[stands_for(block.description)].append(name)
    columns = defaultdict(list)
    for name, block in declared.variables.items():
        columns[stands_for(block.description)].append(name)

    ours_to_theirs = _label_map(theirs, ours, columns)

    def relabel(label: int) -> int:
        if label == -1:
            return -1
        return ours_to_theirs.get(label, -label - 1000)

    verdict: dict[str, list[str]] = {'equal': [], 'region': [], 'mismatch': []}
    for pypsa_name, our_names in columns.items():
        their_kind = pypsa_name in [*theirs.integers, *theirs.binaries]
        ok = all((our_name in [*ours.integers, *ours.binaries]) == their_kind for our_name in our_names)
        bounds_theirs = {int(r.labels): (r.lower, r.upper) for r in theirs.variables[pypsa_name].flat.itertuples()}
        bounds_ours = {}
        for our_name in our_names:
            for r in ours.variables[our_name].flat.itertuples():
                bounds_ours[ours_to_theirs[int(r.labels)]] = (r.lower, r.upper)
        if bounds_ours != bounds_theirs:
            ok = False
        bucket = 'mismatch' if not ok else ('equal' if len(our_names) == 1 else 'region')
        verdict[bucket].append(pypsa_name)

    for pypsa_name, our_names in rows.items():
        their_names = (
            [n for n in theirs.constraints if n.startswith('GlobalConstraint-')]
            if not pypsa_name[0].isupper()
            else ([pypsa_name] if pypsa_name in theirs.constraints else [])
        )
        their_rows: dict = {}
        for their_name in their_names:
            constraint = theirs.constraints[their_name]
            for key, row in _rows(constraint.flat, constraint.labels, lambda x: x).items():
                their_rows[key if their_name == pypsa_name else their_name.removeprefix('GlobalConstraint-')] = row
        our_rows: dict = {}
        for our_name in our_names:
            constraint = ours.constraints[our_name]
            our_rows |= _rows(constraint.flat, constraint.labels, relabel)
        if not pypsa_name[0].isupper():
            typed = {label for label, gc in gc_kinds.items() if gc == pypsa_name}
            their_rows = {key: row for key, row in their_rows.items() if key in typed}
        if our_rows == their_rows:
            verdict['equal' if len(our_names) == 1 else 'region'].append(pypsa_name)
        else:
            verdict['mismatch'].append(pypsa_name)
            for key in sorted({*our_rows, *their_rows}, key=str):
                if our_rows.get(key) != their_rows.get(key):
                    print(
                        f'  {pypsa_name}[{key}]:\n    ours   {our_rows.get(key)}\n    theirs {their_rows.get(key)}',
                        file=sys.stderr,
                    )

    if _objective(ours, relabel) == _objective(theirs, lambda x: x):
        verdict['equal'].append('objective')
    else:
        verdict['mismatch'].append('objective')
    return {kind: sorted(names) for kind, names in verdict.items()}


def lanes(stem: str) -> tuple[dict[str, object], dict[str, object], bool]:
    """One rung through everything: the objective across the fence, the model against the model, the coverage."""
    from lpspec import linopy as lpl

    theirs = pypsa_model(stem)
    n = network(stem)
    gc_kinds = {str(label): str(gc['type']) for label, gc in n.global_constraints.iterrows()}
    status, condition = n.optimize(solver_name='highs')
    assert status == 'ok', f'{stem}: pypsa did not solve — {status} / {condition}'
    model = model_of(stem)
    declared = math_spec.load_model(model)
    tables = bound(model, network(stem))
    result = lps.solve(model, tables)
    assert result.is_ok, f'{stem}: lpspec did not solve — {result.termination_condition}'
    built_rows, built_columns = built(result, declared)
    parity = {
        'lpspec_objective': round(float(result.objective), 6),
        'matches': math.isclose(
            float(result.objective), float(n.objective) + float(n.objective_constant), rel_tol=1e-9, abs_tol=1e-6
        ),
        'model': model.name,
        'built_rows': built_rows,
        'built_columns': built_columns,
        'dims': {name: len(table) for name, table in tables.items() if name in declared.dimensions},
        'bound_nonempty': sorted(name for name, table in tables.items() if len(table)),
        'prices': prices(result, n),
    }
    try:
        ours = lpl.build(model, tables)
    except Exception as error:
        note = f'{type(error).__name__}: {error}'.splitlines()[0][:200]
        return parity, {'error': note}, parity['matches'] and priced(parity)
    verdict = compare(theirs, ours, declared, gc_kinds)
    structural = verdict
    return parity, structural, parity['matches'] and priced(parity) and not verdict['mismatch']


def priced(parity: dict) -> bool:
    """Prices agree, or the lane had none to offer."""
    return parity['prices']['compared'] == 0 or parity['prices']['matches']


def settled(committed: object, fresh: object) -> object:
    """*fresh*, with every number the committed certificate already agrees on left as it stands.

    Rounding stops the last-digit churn; this stops the rest. HiGHS re-solving
    the same model does not return the same bits — the objective moved by one
    ulp and a price residual by 1e-16 between two runs of the same commit — and
    the gate is a byte diff, so without this every re-run rewrites the file and
    reds the job over nothing.

    A number that moves by more than the tolerance is still written, so a red
    diff means a claim changed rather than a rebuild happened. Ints are left
    alone: a count that moved is never noise.
    """
    if isinstance(committed, dict) and isinstance(fresh, dict):
        return {key: settled(committed.get(key), value) for key, value in fresh.items()}
    if isinstance(committed, list) and isinstance(fresh, list) and len(committed) == len(fresh):
        return [settled(was, now) for was, now in zip(committed, fresh, strict=True)]
    if _is_float(committed) and _is_float(fresh):
        return committed if math.isclose(float(committed), float(fresh), rel_tol=1e-9, abs_tol=1e-12) else fresh
    return fresh


def _is_float(value: object) -> bool:
    return isinstance(value, float) and not isinstance(value, bool)


def coverage(stamped: dict[str, dict]) -> list[str]:
    """What the ladder as a whole leaves untested — empty when every block, mask and parameter is exercised.

    A declared block no rung builds is a silent regime; a ``where:`` no rung
    leaves half-true is untested as a mask; a parameter every rung leaves
    empty is data no comparison has ever weighed.
    """
    gaps = []
    by_file: dict[str, list[dict]] = defaultdict(list)
    for stem in sorted(stamped):
        by_file[stamped[stem]['parity']['model']].append(stamped[stem]['parity'])
    for name, stamps in by_file.items():
        declared = math_spec.load_model(CORPUS / 'examples' / name)
        for kind, blocks in (('built_rows', declared.constraints), ('built_columns', declared.variables)):
            for block_name, block in blocks.items():
                counts = [stamp[kind][block_name] for stamp in stamps]
                if not sum(counts):
                    gaps.append(f'{name}: no rung builds {block_name}')
                elif block.where and not any(
                    0 < c < math.prod(stamp['dims'][d] for d in block.foreach)
                    for c, stamp in zip(counts, stamps, strict=True)
                ):
                    gaps.append(f'{name}: {block_name} is always all-or-nothing, so its mask is untested')
        fed = set().union(*(stamp['bound_nonempty'] for stamp in stamps))
        gaps.extend(
            f'{name}: no rung feeds {unfed}' for unfed in sorted({*declared.parameters, *declared.lookups} - fed)
        )
    return gaps


def main() -> int:
    ladder = rungs()
    assert ladder, f'no rung scripts under {RUNGS} — is {CORPUS} a math-spec checkout?'
    committed = json.loads(RECORDS.read_text()) if RECORDS.exists() else {}
    stamped: dict[str, dict] = {}
    broken = []
    for stem in ladder:
        parity, structural, good = lanes(stem)
        was = committed.get(stem, {})
        stamped[stem] = {
            'parity': settled(was.get('parity'), parity),
            'structural': settled(was.get('structural'), structural),
        }
        proof = (
            f'{len(structural["equal"])} equal · {len(structural["region"])} region'
            if 'equal' in structural
            else f'objective only — {structural["error"]}'
        )
        prices_ = parity['prices']
        priced_ = (
            f'prices on {prices_["compared"]} rows' if prices_['compared'] else f'no prices — {prices_["skipped"]}'
        )
        print(f'{stem}: {"MATCH" if parity["matches"] else "DIFFER"} · {priced_} · {proof}')
        if not good:
            broken.append(stem)
    RECORDS.write_text(json.dumps(stamped, indent=2, sort_keys=True) + '\n')
    gaps = coverage(stamped)
    for gap in gaps:
        print(gap, file=sys.stderr)
    if broken or gaps:
        print(f'{len(broken)} rung(s) differ, {len(gaps)} coverage gap(s)', file=sys.stderr)
        return 1
    print('every rung matches PyPSA as deep as the engines allow, and says how deep that is')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
