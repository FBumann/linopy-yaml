"""The parity gate: every rung of the PyPSA corpus, as deep as the engines allow.

    python differential/pypsa/parity.py <math-spec checkout>

The corpus is math-spec's — `examples/pypsa.yaml` and its quadratic sibling,
and one `rung_*.py` per rung whose `build()` returns the PyPSA network with
its data inline. `prep.py` beside this file is the binding: a network becomes
the tables the file declares, every "data prep" parameter computed there.
This file is the rest of the engine side — bind, build, solve, compare — and
it needs a checkout of that repository at the tag `pyproject.toml` pins,
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
5. **Structure** — PyPSA's rows and columns per name, masked labels
   excluded, against what the relational lane built per block — never
   summed, so a PyPSA name split over several blocks differs even where the
   parts add up. A difference is allowed only with a reason in
   ``deviations.yaml``; one recorded nowhere reds the run, and so does a
   reason no rung needs any more.

Primals are deliberately not compared — an optimum need not be unique.

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
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import polars.selectors as cs

CORPUS = Path(sys.argv[1] if len(sys.argv) > 1 else 'corpus').resolve()
RUNGS = CORPUS / 'examples' / 'references' / 'pypsa'
HERE = Path(__file__).resolve().parent
RECORDS = HERE / 'references.json'
TABLES = HERE / 'tables'
PROJECTIONS = HERE / 'rungs'
DEVIATIONS = HERE / 'deviations.yaml'
sys.path.insert(0, str(RUNGS))
sys.path.insert(0, str(HERE))

import linopy  # noqa: E402
import math_spec  # noqa: E402
import prep  # noqa: E402  the binding, beside this file
import projection  # noqa: E402
import yaml  # noqa: E402

import lpspec as lps  # noqa: E402
from lpspec.sources import tidy_sources  # noqa: E402


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


#: Model file -> the tables some lower rung already committed; a table is written once, under the rung that first feeds it.
FIRST: dict[str, set[str]] = defaultdict(set)


def projected(stem: str, model: Path, parity: dict, n) -> Path:
    """Write the rung's projection of *model*, solve it, and hold it to the full file's objective.

    The projection is what the page shows as this rung's model and what its
    tables are cut to; solving it here is what makes it a model rather than
    an excerpt — a cut that lost something load-bearing lands elsewhere than
    PyPSA and reds the run. The rung's script and the file's symbol table are
    copied beside it, so the page can show the network and typeset the math
    with no checkout at hand; the same diff gate holds the copies.
    """
    raw = yaml.safe_load(model.read_text())
    cut = projection.project(raw, parity)
    path = PROJECTIONS / f'{stem}.yaml'
    path.parent.mkdir(exist_ok=True)
    path.write_text(projection.dump(cut))
    shutil.copy(RUNGS / f'{stem}.py', PROJECTIONS / f'{stem}.py')
    symbols = model.parent / 'symbols' / model.name
    if symbols.exists():
        shutil.copy(symbols, PROJECTIONS / f'{stem}.symbols.yaml')
    result = lps.solve(path, bound(path, n))
    assert result.is_ok, f'{stem}: the projection did not solve — {result.termination_condition}'
    assert math.isclose(float(result.objective), parity['lpspec_objective'], rel_tol=1e-9, abs_tol=1e-6), (
        f'{stem}: the projection lands on {result.objective}, the file on {parity["lpspec_objective"]} — the cut lost a term'
    )
    return path


def committed(stem: str, model: str, declared, tables: dict[str, object]) -> None:
    """Write the tables this rung is the first to feed as CSV, rows sorted — the tables the page shows under it.

    Written through :func:`tidy_sources`, so a file holds exactly the tidy
    frame `lps.solve` received, floats rounded to twelve places because a
    ``pow`` differs by an ulp between libms and the gate is a byte diff; the
    workflow's diff gate makes a table that drifts from `prep.sources(build())`
    a red diff. Once per table rather than
    once per rung: a higher rung binds the same table with a row more, and
    committing that copy again would say nothing the page's rung order does not.
    """
    folder = TABLES / stem
    folder.mkdir(parents=True)
    for name, source in tidy_sources(declared, tables).items():
        frame = source.collect() if hasattr(source, 'collect') else source
        if len(frame) and name not in FIRST[model]:
            frame.sort(frame.columns).with_columns(cs.float().round(12)).write_csv(folder / f'{name}.csv')
            FIRST[model].add(name)


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
    """label per coordinate key — dim names dropped, ``snapshot`` first, so the two spellings align.

    Key components are strings, because the labels of a dimension are ints on
    one side and their str spelling on the other (PyPSA numbers its cycles,
    the tables hold every label as text). A dimensionless array — a per-label
    global-constraint row — is its one label at the empty key.
    """
    if not labels.ndim:
        return pd.Series({(): int(labels.item())})
    series = labels.to_series()
    index = series.index
    if index.nlevels > 1:
        order = sorted(index.names, key=lambda name: (name != 'snapshot', name))
        series = series.reorder_levels(order).sort_index()
        series.index = pd.Index([tuple(str(part) for part in key) for key in series.index])
    else:
        series.index = pd.Index([str(key) for key in series.index])
    return series


def _label_map(theirs, ours, pairs: dict[str, list[str]]) -> dict[int, int]:
    """Our variable labels to theirs, matched by name pair and coordinate key."""
    mapping: dict[int, int] = {}
    for pypsa_name, our_names in pairs.items():
        if pypsa_name not in theirs.variables:
            continue
        their = _keyed(theirs.variables[pypsa_name].labels)
        for our_name in our_names:
            for key, our_label in _keyed(ours.variables[our_name].labels).items():
                if int(our_label) == -1 or key not in their.index:
                    continue
                their_label = int(their[key])
                if their_label != -1:
                    mapping[int(our_label)] = their_label
    return mapping


def _rows(flat: pd.DataFrame, labels, relabel) -> dict:
    """Constraint rows by coordinate key: (sign, rhs, sorted (variable, coefficient) pairs).

    Term order within a row is the builder's own, so the pairs are sorted; so
    is the row's orientation — the two builders move the terms to opposite
    sides of the same balance — so a row whose first nonzero coefficient is
    negative is flipped whole, its sense with it. Coefficients and constants
    are rounded to nine places, as the objective's already are: the builders
    reach the same number through different arithmetic and may differ in the
    last ulp.
    """
    terms = defaultdict(list)
    meta = {}
    for row in flat.itertuples():
        terms[row.labels].append((relabel(int(row.vars)), round(float(row.coeffs), 9)))
        meta[row.labels] = (row.sign, round(float(row.rhs), 9))
    rows = {}
    flipped = {'<=': '>=', '>=': '<=', '=': '='}
    for key, label in _keyed(labels).items():
        if int(label) == -1:
            continue
        sign, rhs = meta[int(label)]
        pairs = tuple(sorted(terms[int(label)]))
        lead = next((coeff for _, coeff in pairs if coeff), 0.0)
        if lead < 0:
            sign, rhs, pairs = flipped[sign], -rhs, tuple((var, -coeff) for var, coeff in pairs)
        rows[key] = (sign, rhs, pairs)
    return rows


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


def structure(theirs, declared, gc_kinds: dict[str, str], built_rows: dict, built_columns: dict) -> dict:
    """Row and column counts per PyPSA name, PyPSA's model against what lpspec built — the shape, before the labels.

    PyPSA's counts come off its own linopy model, masked labels excluded;
    ours are the rows and columns built per block, keyed by the PyPSA name
    the block's description opens with and never summed: a PyPSA name is
    matched only by exactly one block with the same count, so a split is a
    difference even where its parts add up to PyPSA's number. A
    global-constraint row is named after its label on PyPSA's side and after
    its type here, so those are matched through the recorded type.
    """
    theirs_rows = {name: int((c.labels != -1).sum()) for name, c in theirs.constraints.items()}
    theirs_columns = {name: int((v.labels != -1).sum()) for name, v in theirs.variables.items()}
    for label, kind in gc_kinds.items():
        theirs_rows[kind] = theirs_rows.get(kind, 0) + theirs_rows.pop(f'GlobalConstraint-{label}', 0)
    ours_rows: dict[str, dict[str, int]] = defaultdict(dict)
    for name, block in declared.constraints.items():
        if built_rows.get(name, 0):
            ours_rows[stands_for(block.description)][name] = built_rows[name]
    ours_columns: dict[str, dict[str, int]] = defaultdict(dict)
    for name, block in declared.variables.items():
        if built_columns.get(name, 0):
            ours_columns[stands_for(block.description)][name] = built_columns[name]

    def table(theirs_side: dict, ours_side: dict) -> dict[str, dict]:
        names = {n for n, c in theirs_side.items() if c} | set(ours_side)
        return {n: {'pypsa': theirs_side.get(n, 0), 'lpspec': ours_side.get(n, {})} for n in sorted(names)}

    return {'rows': table(theirs_rows, ours_rows), 'columns': table(theirs_columns, ours_columns)}


def matched(counts: dict) -> bool:
    """One PyPSA name, one block, one equal count — anything else is a difference."""
    return len(counts['lpspec']) == 1 and next(iter(counts['lpspec'].values())) == counts['pypsa']


def shown(blocks: dict[str, int]) -> str:
    """A block breakdown as the pages and messages print it — ``3+1+4``, never a sum."""
    return '+'.join(str(count) for count in blocks.values()) or '0'


def explained(stem: str, shape: dict, reasons: dict) -> tuple[dict, list[str]]:
    """Every name whose count is not one block equal to PyPSA's, with its recorded reason — and those with none.

    ``deviations.yaml`` maps a PyPSA name to ``{structure: reason}``; a
    difference without a reason is a red run, and so is a reason no rung
    needs any more (checked once over the ladder in ``main``).
    """
    differences, unexplained = {}, []
    for kind in ('rows', 'columns'):
        for name, counts in shape[kind].items():
            if matched(counts):
                continue
            reason = reasons.get(name, {}).get('structure')
            differences[name] = {**counts, 'kind': kind, 'reason': reason}
            if not reason:
                unexplained.append(
                    f'{stem}: {kind} of {name} — pypsa {counts["pypsa"]}, lpspec {shown(counts["lpspec"])}'
                )
    return differences, unexplained


def compare(theirs, ours, declared, gc_kinds: dict[str, str]) -> dict[str, list[str]]:
    """Verdicts: which PyPSA names are model-equal, which are the same region in several blocks, which differ.

    A name absent from a model is its empty set of labels — PyPSA creates
    nothing for a component the network does not carry, and this lane drops a
    block whose every row the data emptied — so a name empty on both sides
    decides nothing and lands in no bucket.
    """
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
        bounds_ours = {}
        for our_name in our_names:
            for r in ours.variables[our_name].flat.itertuples():
                bounds_ours[relabel(int(r.labels))] = (r.lower, r.upper)
        if pypsa_name not in theirs.variables:
            if bounds_ours:
                verdict['mismatch'].append(pypsa_name)
            continue
        bounds_theirs = {int(r.labels): (r.lower, r.upper) for r in theirs.variables[pypsa_name].flat.itertuples()}
        if not bounds_ours and not bounds_theirs:
            continue
        their_kind = pypsa_name in [*theirs.integers, *theirs.binaries]
        ok = all((our_name in [*ours.integers, *ours.binaries]) == their_kind for our_name in our_names)
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
            if our_name not in ours.constraints:
                continue
            constraint = ours.constraints[our_name]
            our_rows |= _rows(constraint.flat, constraint.labels, relabel)
        if not pypsa_name[0].isupper():
            typed = {label for label, gc in gc_kinds.items() if gc == pypsa_name}
            their_rows = {key: row for key, row in their_rows.items() if key in typed}
        if not our_rows and not their_rows:
            continue
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
    shape = structure(theirs, declared, gc_kinds, built_rows, built_columns)
    differences, unexplained = explained(stem, shape, REASONS)
    for line in unexplained:
        print(line, file=sys.stderr)
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
        'structure': {
            'rows': [
                sum(c['pypsa'] for c in shape['rows'].values()),
                sum(sum(c['lpspec'].values()) for c in shape['rows'].values()),
            ],
            'columns': [
                sum(c['pypsa'] for c in shape['columns'].values()),
                sum(sum(c['lpspec'].values()) for c in shape['columns'].values()),
            ],
            'per_name': shape,
            'differences': differences,
        },
    }
    cut = projected(stem, model, parity, n)
    committed(stem, model.name, math_spec.load_model(cut), bound(cut, n))
    try:
        ours = lpl.build(model, tables)
    except Exception as error:
        note = f'{type(error).__name__}: {error}'.splitlines()[0][:200]
        return parity, {'error': note}, parity['matches'] and priced(parity) and shaped(parity)
    verdict = compare(theirs, ours, declared, gc_kinds)
    structural = verdict
    return parity, structural, parity['matches'] and priced(parity) and shaped(parity) and not verdict['mismatch']


def priced(parity: dict) -> bool:
    """Prices agree, or the lane had none to offer."""
    return parity['prices']['compared'] == 0 or parity['prices']['matches']


def shaped(parity: dict) -> bool:
    """Every count that differs has a reason on record."""
    return all(d['reason'] for d in parity['structure']['differences'].values())


REASONS: dict = yaml.safe_load(DEVIATIONS.read_text()) or {} if DEVIATIONS.exists() else {}


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
    for folder in (TABLES, PROJECTIONS):
        if folder.exists():
            shutil.rmtree(folder)
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
        shape_ = parity['structure']
        shaped_ = (
            f'{shape_["rows"][0]} rows, {shape_["columns"][0]} columns'
            if not shape_['differences']
            else f'{len(shape_["differences"])} of {len(shape_["per_name"]["rows"]) + len(shape_["per_name"]["columns"])} names differ'
        )
        print(f'{stem}: {"MATCH" if parity["matches"] else "DIFFER"} · {shaped_} · {priced_} · {proof}')
        if not good:
            broken.append(stem)
    RECORDS.write_text(json.dumps(stamped, indent=2, sort_keys=True) + '\n')
    used = {
        name for s in stamped.values() for name, d in s['parity']['structure']['differences'].items() if d['reason']
    }
    stale = sorted(name for name, entry in REASONS.items() if 'structure' in entry and name not in used)
    gaps = coverage(stamped) + [f'deviations.yaml: {name} records a structure reason no rung needs' for name in stale]
    for gap in gaps:
        print(gap, file=sys.stderr)
    if broken or gaps:
        print(f'{len(broken)} rung(s) differ, {len(gaps)} coverage gap(s)', file=sys.stderr)
        return 1
    print('every rung matches PyPSA as deep as the engines allow, and says how deep that is')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
