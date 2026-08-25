"""`fleet` as a gurobipy user writes it: twelve dictionaries of variables.

Per-entity twelve times over, which is the cost this case measures. Only the
three quantities the objective prices carry an `obj=`; the other nine are free
to the objective and constrained only by their own row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The twelve, in the order the model file declares them.
QUANTITIES = (
    'p', 'p_up', 'p_down', 'reserve_up', 'reserve_down', 'charge',
    'discharge', 'soc', 'spill', 'curtail', 'import_', 'export_',
)  # fmt: skip

#: The three the objective prices.
PRICED = ('p', 'discharge', 'import_')


def build(env: Any, tables: Mapping[str, Any]) -> Any:
    import gurobipy as gp

    p_max = dict(zip(tables['p_max']['unit'], tables['p_max']['value'], strict=True))
    cost = dict(zip(tables['cost']['unit'], tables['cost']['value'], strict=True))
    demand = dict(zip(tables['demand']['snapshot'], tables['demand']['value'], strict=True))
    snapshots = list(tables['snapshot']['snapshot'])
    units = list(tables['unit']['unit'])

    m = gp.Model(env=env)
    v = {
        name: {(s, u): m.addVar(ub=p_max[u], obj=cost[u] if name in PRICED else 0.0) for s in snapshots for u in units}
        for name in QUANTITIES
    }
    m.addConstrs(
        gp.quicksum(v['p'][s, u] + v['discharge'][s, u] + v['import_'][s, u] for u in units) == demand[s]
        for s in snapshots
    )
    for name in ('p_up', 'p_down'):
        m.addConstrs(v[name][s, u] <= p_max[u] for s in snapshots for u in units)
    for first, second in (('reserve_up', 'reserve_down'), ('spill', 'curtail'), ('import_', 'export_')):
        m.addConstrs(v[first][s, u] + v[second][s, u] <= p_max[u] for s in snapshots for u in units)
    m.addConstrs(
        v['soc'][s, u] + v['charge'][s, u] - v['discharge'][s, u] <= p_max[u] for s in snapshots for u in units
    )
    return m
