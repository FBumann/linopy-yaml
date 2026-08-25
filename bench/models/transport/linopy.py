"""`transport` as a linopy user writes it — `examples/ports/references/linopy/transport.py`.

The reviewed idiomatic form (#681), against the ladder's parquet. Its one
modelling decision is argued there and repeated because it is what this case
measures: the YAML groups by the lookups it declared — `sum(p, by=gen_bus)` —
where this builds the bus x generator and bus x line incidence matrices and
multiplies through them. linopy's `groupby` could carry the generator half but
not the flows, since a bus no line enters vanishes from a grouped sum, so the
script keeps one idiom for both halves.

**The incidence matrices are dense**, which is the cost this case exists to
show: the product they multiply through is bus x generator x snapshot, and at
the upper rungs that is the eager lane's materialisation problem rather than a
mistake in the formulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def build(tables: Mapping[str, Any]) -> Any:
    import linopy
    import pandas as pd
    import xarray as xr

    p_max = tables['p_max'].set_index('generator')['value']
    cost = tables['cost'].set_index('generator')['value']
    cap = tables['cap'].set_index('line')['value']
    neg_cap = tables['neg_cap'].set_index('line')['value']
    load = xr.DataArray(tables['load'].pivot(index='snapshot', columns='bus', values='value'))
    snapshots, buses = load.indexes['snapshot'], load.indexes['bus']

    gen_at = pd.DataFrame(0.0, index=buses, columns=p_max.index)
    for gen, bus in zip(tables['gen_bus']['generator'], tables['gen_bus']['bus'], strict=True):
        gen_at.loc[bus, gen] = 1.0
    flow_in = pd.DataFrame(0.0, index=buses, columns=cap.index)
    for line, src, dst in zip(
        tables['line_from']['line'], tables['line_from']['bus'], tables['line_to']['bus'], strict=True
    ):
        flow_in.loc[dst, line] += 1.0
        flow_in.loc[src, line] -= 1.0

    m = linopy.Model()
    p = m.add_variables(lower=0, upper=p_max, coords=[snapshots, p_max.index], name='p')
    f = m.add_variables(lower=neg_cap, upper=cap, coords=[snapshots, cap.index], name='f')
    m.add_constraints(
        (p * xr.DataArray(gen_at)).sum('generator') + (f * xr.DataArray(flow_in)).sum('line') == load,
        name='balance',
    )
    m.add_objective((p * cost).sum())
    return m
