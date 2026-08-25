"""`dispatch` as a linopy user writes it — `examples/ports/references/linopy/dispatch.py`.

That script is the reviewed idiomatic form (#681) and is executed by the docs,
so this is the same model against the ladder's parquet rather than a second
opinion about how linopy should be written.

One deliberate difference from the YAML, and the reference carries it too: the
`where: p_max > 0` mask gives a retired generator no columns at all, where this
keeps them bounded to zero. Same polytope, same objective — and on this ladder
the mask is vacuous anyway, because the generator draws `p_max` strictly
positive.
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

    m = linopy.Model()
    p = m.add_variables(lower=0, upper=p_max, coords=[load.index, p_max.index], name='p')
    m.add_constraints(p.sum('generator') == load, name='power_balance')
    m.add_objective((p * cost).sum())
    return m
