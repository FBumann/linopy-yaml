"""`storage` as a linopy user writes it.

The cyclic state of charge is `.roll(snapshot=1)`, which is what the PyPSA
models this case is modelled on use — an array shifted along its own axis,
where the YAML says `shift(soc, over=snapshot, offset=1, edge='wrap')` and the
relational engine joins the term stream against itself on `snapshot.ord - 1`.
The two spell the same recurrence, and the difference in what it costs is the
reason this case is in the ladder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def build(tables: Mapping[str, Any]) -> Any:
    import linopy

    p_max = tables['p_max'].set_index('generator')['value']
    cost = tables['cost'].set_index('generator')['value']
    load = tables['load'].set_index('snapshot')['value']
    e_max = tables['e_max'].set_index('store')['value']
    p_store = tables['p_store'].set_index('store')['value']
    eta = tables['eta'].set_index('store')['value']

    m = linopy.Model()
    p = m.add_variables(lower=0, upper=p_max, coords=[load.index, p_max.index], name='p')
    charge = m.add_variables(lower=0, upper=p_store, coords=[load.index, p_store.index], name='charge')
    discharge = m.add_variables(lower=0, upper=p_store, coords=[load.index, p_store.index], name='discharge')
    soc = m.add_variables(lower=0, upper=e_max, coords=[load.index, e_max.index], name='soc')

    m.add_constraints(p.sum('generator') + discharge.sum('store') - charge.sum('store') == load, name='power_balance')
    m.add_constraints(soc - soc.roll(snapshot=1) - charge * eta + discharge == 0, name='soc_balance')
    m.add_objective((p * cost).sum())
    return m
