"""`fleet` through gurobipy's matrix API: one `MVar`, one `addMConstr`.

One snapshot's twelve quantities are twelve blocks of `unit` columns, and the
whole model is that block `kron`ed against the identity — every row here lives
inside a single snapshot, so unlike `storage` there is no off-diagonal.

The block is dense on purpose: `fleet` holds its unit count fixed at fifty, so
it is 301 x 600 whatever the ladder does, and writing it as an array reads as
the model rather than as sparse-matrix plumbing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The twelve, in the order the model file declares them — which is also the
#: column order of every block below.
QUANTITIES = (
    'p', 'p_up', 'p_down', 'reserve_up', 'reserve_down', 'charge',
    'discharge', 'soc', 'spill', 'curtail', 'import_', 'export_',
)  # fmt: skip

PRICED = ('p', 'discharge', 'import_')


def build(env: Any, tables: Mapping[str, Any]) -> Any:
    import gurobipy as gp
    from scipy import sparse

    p_max = tables['p_max']['value'].to_numpy()
    cost = tables['cost']['value'].to_numpy()
    demand = tables['demand']['value'].to_numpy()
    n_snapshot, n_unit = len(demand), len(p_max)
    width = len(QUANTITIES) * n_unit
    at = {name: slice(i * n_unit, (i + 1) * n_unit) for i, name in enumerate(QUANTITIES)}
    unit = np.eye(n_unit)

    def row_block(*names: str) -> np.ndarray:
        """One constraint over `unit`, its terms signed by name."""
        block = np.zeros((n_unit, width))
        for name in names:
            block[:, at[name.lstrip('-')]] = -unit if name.startswith('-') else unit
        return block

    balance = np.zeros((1, width))
    for name in PRICED:
        balance[0, at[name]] = 1.0

    block = np.vstack(
        [
            balance,
            row_block('p_up'),
            row_block('p_down'),
            row_block('reserve_up', 'reserve_down'),
            row_block('soc', 'charge', '-discharge'),
            row_block('spill', 'curtail'),
            row_block('import_', 'export_'),
        ]
    )
    senses = np.array(['='] + ['<'] * (6 * n_unit))
    prices = np.concatenate([cost if name in PRICED else np.zeros(n_unit) for name in QUANTITIES])

    m = gp.Model(env=env)
    x = m.addMVar(
        n_snapshot * width, ub=np.tile(np.tile(p_max, len(QUANTITIES)), n_snapshot), obj=np.tile(prices, n_snapshot)
    )
    m.addMConstr(
        sparse.kron(sparse.eye(n_snapshot), block, format='csr'),
        x,
        np.tile(senses, n_snapshot),
        np.concatenate([np.concatenate([[d], np.tile(p_max, 6)]) for d in demand]),
    )
    return m
