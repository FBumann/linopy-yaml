"""`dispatch` through gurobipy's matrix API: one `MVar`, one `addMConstr`.

What a performance-minded gurobipy user writes, and the same seam our own
`gurobi` sink reaches — `addMVar` with `lb`/`ub`/`obj` and a CSR handed to
`addMConstr`. The difference between this arm and ours is therefore *where the
matrix came from*, which is the only thing worth measuring here.

The balance matrix is a block of ones per snapshot, which is
``kron(I(n_snapshot), ones(1, n_generator))`` — built once, in one call, rather
than assembled row by row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping


def build(env: Any, tables: Mapping[str, Any]) -> Any:
    import gurobipy as gp
    from scipy import sparse

    p_max = tables['p_max']['value'].to_numpy()
    cost = tables['cost']['value'].to_numpy()
    load = tables['load']['value'].to_numpy()

    live = p_max > 0
    p_max, cost = p_max[live], cost[live]
    n_snapshot, n_generator = len(load), len(p_max)

    m = gp.Model(env=env)
    x = m.addMVar(
        n_snapshot * n_generator,
        ub=np.tile(p_max, n_snapshot),
        obj=np.tile(cost, n_snapshot),
    )
    balance = sparse.kron(sparse.eye(n_snapshot), np.ones((1, n_generator)), format='csr')
    m.addMConstr(balance, x, '=', load)
    return m
