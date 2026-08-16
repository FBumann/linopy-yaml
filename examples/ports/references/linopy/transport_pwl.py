#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``transport_pwl``: the same model through linopy's own
piecewise formulation. See docs/examples/index.md.

    uv run --script examples/ports/references/linopy/transport_pwl.py

**The independence here is sharper than usual.** Every other reference is
independent of lpspec because it is a different program; this one is
independent of the construct under test. `lpspec`'s ``piecewise:`` block and
linopy's ``add_piecewise_formulation`` are two implementations of the same
λ convex-combination idea, written separately, and this compares them on a
model neither was written for. Nothing here imports lpspec.

The model is GAMS model library ``trnspwl`` — Dantzig's transportation problem
with **economies of scale**: shipping cost grows as ``sqrt(x)`` rather than
linearly, so a big consignment is cheaper per unit. GAMS publishes the model
and its discretisation but not an optimal objective, which is why this script
is what verifies the port rather than a citation.

``sqrt`` is concave and the objective is a minimisation, so the convex-hull
relaxation is **not** valid: it would let the solver ride the chord underneath
the true curve and buy transport cheaper than the model allows. The
formulation therefore needs segment binaries, which is what makes this port a
MILP.

Pinned above to the versions that produced the number in ``references.json``.
linopy is pinned because it *is* the reference here — this script calls its own
``add_piecewise_formulation``, so the formulation is theirs — and xarray is its
data model; pandas is a floor, shaping the input tables and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

import linopy
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / 'data' / 'transport_pwl.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
    """The port's tables as a linopy model, column for column.

    ``tables`` is the same mapping the lpspec call binds as ``sources``.
    ``scaled`` is what the objective is actually charged on — ``sqrt(shipment)``
    read off the discretised curve rather than computed.
    """
    capacity: pd.Series = tables['capacity'].set_index('plant')['value']
    demand: pd.Series = tables['demand'].set_index('market')['value']
    distance: pd.DataFrame = (
        tables['distance']
        .pivot(index='plant', columns='market', values='value')
        .reindex(index=capacity.index)[demand.index]
    )
    cost: pd.DataFrame = distance * tables['freight'] / 1000

    m = linopy.Model()
    shipment = m.add_variables(lower=0, coords=[capacity.index, demand.index], name='shipment')
    scaled = m.add_variables(lower=0, coords=[capacity.index, demand.index], name='scaled')

    m.add_piecewise_formulation(
        (shipment, list(tables['bp_x']['value'])),
        (scaled, list(tables['bp_y']['value'])),
    )

    m.add_constraints(shipment.sum('market') <= capacity, name='within_capacity')
    m.add_constraints(shipment.sum('plant') >= demand, name='meet_demand')
    m.add_objective((scaled * cost).sum())
    return m


def main() -> float:
    m = build(load_tables())
    status, condition = m.solve(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    print(f'linopy {linopy.__version__}')
    print(f'objective {float(m.objective.value)!r}')
    print(m.solution['shipment'].to_series())
    return float(m.objective.value)


if __name__ == '__main__':
    main()
