#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``transport``: the same LP, hand-written in linopy.

    uv run --script examples/ports/references/linopy/transport.py

A teaching model, so what verifies it is agreement with an independent
formulation, not a published figure — see ``dispatch.py`` next door.

The comparison the page cares about is the nodal balance. The YAML groups by
coordinates it declared on the dimensions — ``sum(p, over=generator,
group_by=bus)`` — where this script has to build the bus x generator and
bus x line incidence matrices itself and multiply through them. Both say
Kirchhoff's current law; one says it as a relation, the other as linear
algebra.
"""

from __future__ import annotations

import json
from pathlib import Path

import linopy
import pandas as pd
import xarray as xr

DATA = Path(__file__).resolve().parents[2] / 'data' / 'transport.json'


def build(data: dict) -> linopy.Model:
    """The instance's tables as a linopy model, row for row."""
    generators = pd.Index(data['generator']['generator'], name='generator')
    lines = pd.Index(data['line']['line'], name='line')
    buses = pd.Index(sorted(set(data['load']['bus'])), name='bus')
    snapshots = pd.Index(sorted(set(data['load']['snapshot'])), name='snapshot')

    p_max = pd.Series(data['p_max']['value'], index=generators)
    cost = pd.Series(data['cost']['value'], index=generators)
    cap = pd.Series(data['cap']['value'], index=lines)
    neg_cap = pd.Series(data['neg_cap']['value'], index=lines)
    load = xr.DataArray(
        pd.DataFrame(data['load']).pivot(index='snapshot', columns='bus', values='value').reindex(columns=buses)
    )

    gen_at = pd.DataFrame(0.0, index=buses, columns=generators)
    for gen, bus in zip(generators, data['generator']['bus'], strict=True):
        gen_at.loc[bus, gen] = 1.0
    flow_in = pd.DataFrame(0.0, index=buses, columns=lines)
    for line, src, dst in zip(lines, data['line']['from'], data['line']['to'], strict=True):
        flow_in.loc[dst, line] += 1.0
        flow_in.loc[src, line] -= 1.0

    m = linopy.Model()
    p = m.add_variables(lower=0, upper=p_max, coords=[snapshots, generators], name='p')
    f = m.add_variables(lower=neg_cap, upper=cap, coords=[snapshots, lines], name='f')
    m.add_constraints(
        (p * xr.DataArray(gen_at)).sum('generator') + (f * xr.DataArray(flow_in)).sum('line') == load,
        name='balance',
    )
    m.add_objective((p * cost).sum())
    return m


def nodal_prices(m: linopy.Model) -> dict[str, list]:
    """The balance dual, tidy: one price per (snapshot, bus)."""
    dual = m.constraints['balance'].dual.transpose('snapshot', 'bus')
    return {
        'snapshot': [int(s) for s in dual.indexes['snapshot'] for _ in dual.indexes['bus']],
        'bus': [str(b) for _ in dual.indexes['snapshot'] for b in dual.indexes['bus']],
        'value': [float(v) for v in dual.values.ravel()],
    }


def main() -> float:
    m = build(json.loads(DATA.read_text()))
    status, condition = m.solve(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    print(f'linopy {linopy.__version__}')
    print(f'objective {float(m.objective.value)!r}')
    print(f'duals {json.dumps({"balance": nodal_prices(m)})}')
    return float(m.objective.value)


if __name__ == '__main__':
    main()
