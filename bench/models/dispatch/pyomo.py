"""`dispatch` as a pyomo user writes it: a `ConcreteModel` with rules.

The `Set` / `Var(bounds=…)` / `Constraint(rule=…)` / `Objective(expr=…)` form
out of pyomo's own documentation. Bounds arrive through the rule rather than as
a `Param`, which is the shorter of the two idioms and the one that builds
fewer components.

The YAML's `where: p_max > 0` is honoured by leaving the retired generator out
of the index set — the closest pyomo has to a variable that does not exist. On
this ladder it removes nothing: the generator draws `p_max` strictly positive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def build(tables: Mapping[str, Any]) -> Any:
    import pyomo.environ as pyo

    p_max = dict(zip(tables['p_max']['generator'], tables['p_max']['value'], strict=True))
    cost = dict(zip(tables['cost']['generator'], tables['cost']['value'], strict=True))
    load = dict(zip(tables['load']['snapshot'], tables['load']['value'], strict=True))

    m = pyo.ConcreteModel()
    m.snapshots = pyo.Set(initialize=list(tables['snapshot']['snapshot']), ordered=True)
    m.generators = pyo.Set(initialize=[g for g in tables['generator']['generator'] if p_max[g] > 0], ordered=True)
    m.p = pyo.Var(m.snapshots, m.generators, bounds=lambda _m, _s, g: (0.0, p_max[g]))
    m.power_balance = pyo.Constraint(m.snapshots, rule=lambda _m, s: sum(_m.p[s, g] for g in _m.generators) == load[s])
    m.cost = pyo.Objective(expr=sum(cost[g] * m.p[s, g] for s in m.snapshots for g in m.generators), sense=pyo.minimize)
    return m
