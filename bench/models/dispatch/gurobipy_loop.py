"""`dispatch` as a gurobipy user writes it: a variable per (snapshot, generator).

The comprehension-of-`addVar` plus `addConstrs(quicksum(...))` form, which is
what the Gurobi examples and the modelling books use. Costs ride on `obj=`
rather than a `setObjective(quicksum(...))` over every term — both are idiomatic
and this is the cheaper of the two, which is the direction to err in an arm
somebody else's library is being judged by.

No `name=` anywhere: naming is a feature only some arms' models carry, and
`bench/README.md` says why the harness switches it off on every arm that has it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def build(env: Any, tables: Mapping[str, Any]) -> Any:
    import gurobipy as gp

    p_max = dict(zip(tables['p_max']['generator'], tables['p_max']['value'], strict=True))
    cost = dict(zip(tables['cost']['generator'], tables['cost']['value'], strict=True))
    load = dict(zip(tables['load']['snapshot'], tables['load']['value'], strict=True))
    snapshots = list(tables['snapshot']['snapshot'])
    generators = [g for g in tables['generator']['generator'] if p_max[g] > 0]

    m = gp.Model(env=env)
    p = {(s, g): m.addVar(ub=p_max[g], obj=cost[g]) for s in snapshots for g in generators}
    m.addConstrs(gp.quicksum(p[s, g] for g in generators) == load[s] for s in snapshots)
    return m
