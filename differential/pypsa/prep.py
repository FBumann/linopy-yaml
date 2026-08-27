# SPDX-FileCopyrightText: math-spec Contributors
#
# SPDX-License-Identifier: MIT

"""The prep layer: a PyPSA network as the tables the example models bind.

Every parameter the files mark "data prep" is computed here, beside the plain
renames — the binding half of how lpspec builds the corpus's model, shown on
the ladder page beside the tables it produces. `parity.py` is the caller and
cuts the tables to what each model declares; nothing here imports math_spec
or lpspec — the mapping is pure PyPSA-and-pandas, handed over as polars frames.

Sparseness is meaning: a table row left out is an absent value on the other
side, so the sparse tables here (`*_set` pins, ramp limits, weights) drop
their empty rows instead of shipping fills.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import polars as pl
from pypsa.descriptors import get_switchable_as_dense

if TYPE_CHECKING:
    import pypsa


#: PyPSA component -> the dimension the file declares for it.
DIM = {
    'Generator': 'generator',
    'Link': 'link',
    'Load': 'load',
    'StorageUnit': 'storage_unit',
    'Store': 'store',
    'Line': 'line',
    'GlobalConstraint': 'global_constraint',
}


def static(n: pypsa.Network, component: str, attr: str) -> pd.DataFrame:
    """A static attribute as ``(dim, value)``, one row per component."""
    table = n.static(component)
    values = table[attr].to_numpy() if attr in table.columns else [float('nan')] * len(table)
    return pd.DataFrame({DIM[component]: table.index.astype(str), 'value': values})


def varying(n: pypsa.Network, component: str, attr: str) -> pd.DataFrame:
    """A time-varying attribute as ``(snapshot, dim, value)``, static values broadcast over the snapshots as PyPSA does."""
    dense = get_switchable_as_dense(n, component, attr)
    table = dense.melt(ignore_index=False, var_name=DIM[component]).reset_index(names='snapshot')
    return table.astype({DIM[component]: str, 'value': float})


def lookup(n: pypsa.Network, component: str, attr: str) -> pd.DataFrame:
    """The bus a component's *attr* names, as the lookup the file declares over it; a blank names none."""
    table = n.static(component)
    named = table[attr].astype(str) if attr in table.columns else pd.Series('', index=table.index, dtype=str)
    out = pd.DataFrame({DIM[component]: table.index.astype(str), 'bus': named})
    return out[out['bus'] != '']


def weighting(n: pypsa.Network, column: str) -> pd.DataFrame:
    return pd.DataFrame({'snapshot': n.snapshots, 'value': n.snapshot_weightings[column].to_numpy()})


def _retention(n: pypsa.Network, component: str, dim: str) -> pd.DataFrame:
    losses = n.static(component)['standing_loss']
    hours = n.snapshot_weightings['stores']
    dense = pd.DataFrame({name: (1.0 - loss) ** hours for name, loss in losses.items()}, index=n.snapshots)
    table = dense.melt(ignore_index=False, var_name=dim).reset_index(names='snapshot')
    return table.astype({dim: str, 'value': float})


def _cycle_weights(n: pypsa.Network) -> pd.DataFrame:
    """The KVL rows PyPSA itself writes — ``n.cycle_matrix(apply_weights=True)``, reactance on AC and resistance on DC, times the 1e5 PyPSA scales every cycle row by for conditioning."""
    n.determine_network_topology()
    n.calculate_dependent_values()
    cycles = n.cycle_matrix(apply_weights=True) * 1e5
    rows = [
        {'line': str(name), 'cycle': str(cycle), 'value': float(weight)}
        for (kind, name), weights in cycles.iterrows()
        for cycle, weight in weights.items()
        if kind == 'Line' and weight
    ]
    return pd.DataFrame(rows, columns=['line', 'cycle', 'value']).astype({'value': float})


def _weights(gcs: pd.DataFrame, components: pd.DataFrame, dim: str, value) -> pd.DataFrame:
    """One row per (global constraint, member): *value* returns the weight, or 0/None outside the row's set."""
    rows = [
        {'global_constraint': str(label), dim: str(name), 'value': float(v)}
        for label, gc in gcs.iterrows()
        for name, component in components.iterrows()
        if (v := value(gc, component))
    ]
    return pd.DataFrame(rows, columns=['global_constraint', dim, 'value']).astype({'value': float})


def _typed(n: pypsa.Network, kind: str) -> pd.DataFrame:
    return n.global_constraints[n.global_constraints['type'] == kind]


def _emissions(n: pypsa.Network, gc: pd.Series) -> pd.Series:
    """The nonzero values of the carrier attribute a `primary_energy` row weighs."""
    values = n.carriers[gc['carrier_attribute']]
    return values[values != 0]


def _carrier_list(gc: pd.Series) -> list[str]:
    return [c.strip().strip('[]()') for c in str(gc['carrier_attribute']).split(',')]


def _in_tech_set(gc: pd.Series, component: pd.Series, nominal: str, bus: str) -> bool:
    """PyPSA's membership for a `tech_capacity_expansion_limit` row: extendable, the carrier, and the bus if named."""
    at_bus = not gc.get('bus') or str(component[bus]) == str(gc['bus'])
    return bool(component[f'{nominal}_extendable'] and component['carrier'] == gc['carrier_attribute'] and at_bus)


def _gc_constants(n: pypsa.Network) -> pd.DataFrame:
    """Each row's constant, net of the initial charge PyPSA folds into its side of the row.

    A `primary_energy` or `operational_limit` row counts what its non-cyclic
    storage draws down, so PyPSA adds the initial charge as a constant on the
    variable side; the file keeps the variables and moves it here.
    """
    rows = []
    for label, gc in n.global_constraints.iterrows():
        constant = float(gc['constant'])
        if gc['type'] == 'primary_energy':
            emissions = _emissions(n, gc)
            sus = n.storage_units
            member = sus['carrier'].isin(emissions.index) & ~sus['cyclic_state_of_charge']
            constant -= float(
                (sus.loc[member, 'carrier'].map(emissions) * sus.loc[member, 'state_of_charge_initial']).sum()
            )
            stores = n.stores
            member = stores['carrier'].isin(emissions.index) & ~stores['e_cyclic']
            constant -= float((stores.loc[member, 'carrier'].map(emissions) * stores.loc[member, 'e_initial']).sum())
        if gc['type'] == 'operational_limit':
            sus = n.storage_units
            member = (sus['carrier'] == gc['carrier_attribute']) & ~sus['cyclic_state_of_charge']
            constant -= float(sus.loc[member, 'state_of_charge_initial'].sum())
            stores = n.stores
            member = (stores['carrier'] == gc['carrier_attribute']) & ~stores['e_cyclic']
            constant -= float(stores.loc[member, 'e_initial'].sum())
        rows.append({'global_constraint': str(label), 'value': constant})
    return pd.DataFrame(rows, columns=['global_constraint', 'value']).astype({'value': float})


def _must_stay_up(n: pypsa.Network) -> pd.DataFrame:
    """True while the up time a unit brought into the horizon still binds."""
    rows = []
    for name, g in n.generators.iterrows():
        if not g['committable'] or g['up_time_before'] <= 0:
            continue
        remaining = int(min(g['min_up_time'] - g['up_time_before'], len(n.snapshots)))
        rows.extend({'snapshot': t, 'generator': str(name), 'value': True} for t in n.snapshots[: max(remaining, 0)])
    table = pd.DataFrame(rows, columns=['snapshot', 'generator', 'value'])
    return table.astype({'value': bool})


def sources(n: pypsa.Network) -> dict[str, object]:
    """Every table the example models bind, from one PyPSA network."""
    generators, links, loads = n.generators, n.links, n.loads
    storage_units, stores, lines = n.storage_units, n.stores, n.lines
    applies = generators['committable'] & generators['p_nom_extendable']
    big_m = (generators['p_nom_max'] * get_switchable_as_dense(n, 'Generator', 'p_max_pu').max().clip(lower=1.0))[
        applies
    ]

    tables: dict[str, object] = {
        'snapshot': pl.Series('snapshot', list(n.snapshots), dtype=pl.Int64),
        'bus': pl.Series('bus', list(n.buses.index.astype(str)), dtype=pl.String),
        'generator': pl.Series('generator', list(generators.index.astype(str)), dtype=pl.String),
        'link': pl.Series('link', list(links.index.astype(str)), dtype=pl.String),
        'load': pl.Series('load', list(loads.index.astype(str)), dtype=pl.String),
        'storage_unit': pl.Series('storage_unit', list(storage_units.index.astype(str)), dtype=pl.String),
        'store': pl.Series('store', list(stores.index.astype(str)), dtype=pl.String),
        'line': pl.Series('line', list(lines.index.astype(str)), dtype=pl.String),
        'global_constraint': pl.Series(
            'global_constraint', list(n.global_constraints.index.astype(str)), dtype=pl.String
        ),
        'Generator_bus': lookup(n, 'Generator', 'bus'),
        'Link_bus0': lookup(n, 'Link', 'bus0'),
        'Link_bus1': lookup(n, 'Link', 'bus1'),
        'Load_bus': lookup(n, 'Load', 'bus'),
        'StorageUnit_bus': lookup(n, 'StorageUnit', 'bus'),
        'Store_bus': lookup(n, 'Store', 'bus'),
        'Line_bus0': lookup(n, 'Line', 'bus0'),
        'Line_bus1': lookup(n, 'Line', 'bus1'),
        'snapshot_weightings_objective': weighting(n, 'objective'),
        'snapshot_weightings_stores': weighting(n, 'stores'),
        'snapshot_weightings_generators': weighting(n, 'generators'),
        'Load_p_set': varying(n, 'Load', 'p_set'),
        'Generator_p_nom': static(n, 'Generator', 'p_nom'),
        'Generator_p_nom_extendable': static(n, 'Generator', 'p_nom_extendable'),
        'Generator_p_min_pu': varying(n, 'Generator', 'p_min_pu'),
        'Generator_p_max_pu': varying(n, 'Generator', 'p_max_pu'),
        'Generator_marginal_cost': varying(n, 'Generator', 'marginal_cost'),
        'Generator_p_set': varying(n, 'Generator', 'p_set').dropna(),
        'Generator_p_nom_min': static(n, 'Generator', 'p_nom_min'),
        'Generator_p_nom_max': static(n, 'Generator', 'p_nom_max'),
        'Generator_capital_cost': static(n, 'Generator', 'capital_cost'),
        'Generator_p_nom_set': static(n, 'Generator', 'p_nom_set').dropna(),
        'Generator_e_sum_min': static(n, 'Generator', 'e_sum_min'),
        'Generator_e_sum_max': static(n, 'Generator', 'e_sum_max'),
        'Generator_committable': static(n, 'Generator', 'committable'),
        'Generator_ramp_limit_up': static(n, 'Generator', 'ramp_limit_up').dropna(),
        'Generator_ramp_limit_down': static(n, 'Generator', 'ramp_limit_down').dropna(),
        'Generator_ramp_limit_start_up': static(n, 'Generator', 'ramp_limit_start_up').fillna({'value': 1.0}),
        'Generator_ramp_limit_shut_down': static(n, 'Generator', 'ramp_limit_shut_down').fillna({'value': 1.0}),
        'Generator_min_up_time': static(n, 'Generator', 'min_up_time'),
        'Generator_min_down_time': static(n, 'Generator', 'min_down_time'),
        'Generator_status_initial': pd.DataFrame(
            {
                'generator': generators.index.astype(str),
                'value': (generators['up_time_before'] > 0).astype(int).to_numpy(),
            }
        ),
        'Generator_must_stay_up': _must_stay_up(n),
        'Generator_start_up_cost': static(n, 'Generator', 'start_up_cost'),
        'Generator_shut_down_cost': static(n, 'Generator', 'shut_down_cost'),
        'Generator_stand_by_cost': varying(n, 'Generator', 'stand_by_cost'),
        'Generator_p_nom_mod': static(n, 'Generator', 'p_nom_mod').query('value > 0'),
        'Generator_big_m': pd.DataFrame({'generator': big_m.index.astype(str), 'value': big_m.to_numpy()}),
        'Generator_p_min_pu_nonneg': pd.DataFrame(
            {
                'generator': generators.index.astype(str),
                'value': (get_switchable_as_dense(n, 'Generator', 'p_min_pu') >= 0).all().to_numpy(),
            }
        ),
        'Link_p_nom': static(n, 'Link', 'p_nom'),
        'Link_p_nom_extendable': static(n, 'Link', 'p_nom_extendable'),
        'Link_p_min_pu': varying(n, 'Link', 'p_min_pu'),
        'Link_p_max_pu': varying(n, 'Link', 'p_max_pu'),
        'Link_efficiency': static(n, 'Link', 'efficiency'),
        'Link_marginal_cost': varying(n, 'Link', 'marginal_cost'),
        'Link_p_set': varying(n, 'Link', 'p_set').dropna(),
        'Link_p_nom_min': static(n, 'Link', 'p_nom_min'),
        'Link_p_nom_max': static(n, 'Link', 'p_nom_max'),
        'Link_capital_cost': static(n, 'Link', 'capital_cost'),
        'Link_p_nom_set': static(n, 'Link', 'p_nom_set').dropna(),
        'Link_ramp_limit_up': static(n, 'Link', 'ramp_limit_up').dropna(),
        'Link_ramp_limit_down': static(n, 'Link', 'ramp_limit_down').dropna(),
        'StorageUnit_p_nom': static(n, 'StorageUnit', 'p_nom'),
        'StorageUnit_p_nom_extendable': static(n, 'StorageUnit', 'p_nom_extendable'),
        'StorageUnit_p_min_pu': varying(n, 'StorageUnit', 'p_min_pu'),
        'StorageUnit_p_max_pu': varying(n, 'StorageUnit', 'p_max_pu'),
        'StorageUnit_max_hours': static(n, 'StorageUnit', 'max_hours'),
        'StorageUnit_efficiency_store': static(n, 'StorageUnit', 'efficiency_store'),
        'StorageUnit_efficiency_dispatch': static(n, 'StorageUnit', 'efficiency_dispatch'),
        'StorageUnit_retention': _retention(n, 'StorageUnit', 'storage_unit'),
        'StorageUnit_inflow': varying(n, 'StorageUnit', 'inflow'),
        'StorageUnit_state_of_charge_initial': static(n, 'StorageUnit', 'state_of_charge_initial'),
        'StorageUnit_cyclic_state_of_charge': static(n, 'StorageUnit', 'cyclic_state_of_charge'),
        'StorageUnit_marginal_cost': varying(n, 'StorageUnit', 'marginal_cost'),
        'StorageUnit_marginal_cost_storage': varying(n, 'StorageUnit', 'marginal_cost_storage'),
        'StorageUnit_spill_cost': varying(n, 'StorageUnit', 'spill_cost'),
        'StorageUnit_p_set': varying(n, 'StorageUnit', 'p_set').dropna(),
        'StorageUnit_state_of_charge_set': varying(n, 'StorageUnit', 'state_of_charge_set').dropna(),
        'StorageUnit_p_nom_min': static(n, 'StorageUnit', 'p_nom_min'),
        'StorageUnit_p_nom_max': static(n, 'StorageUnit', 'p_nom_max'),
        'StorageUnit_capital_cost': static(n, 'StorageUnit', 'capital_cost'),
        'StorageUnit_p_nom_set': static(n, 'StorageUnit', 'p_nom_set').dropna(),
        'Store_e_nom': static(n, 'Store', 'e_nom'),
        'Store_e_nom_extendable': static(n, 'Store', 'e_nom_extendable'),
        'Store_e_min_pu': varying(n, 'Store', 'e_min_pu'),
        'Store_e_max_pu': varying(n, 'Store', 'e_max_pu'),
        'Store_retention': _retention(n, 'Store', 'store'),
        'Store_e_initial': static(n, 'Store', 'e_initial'),
        'Store_e_cyclic': static(n, 'Store', 'e_cyclic'),
        'Store_marginal_cost': varying(n, 'Store', 'marginal_cost'),
        'Store_marginal_cost_storage': varying(n, 'Store', 'marginal_cost_storage'),
        'Store_e_set': varying(n, 'Store', 'e_set').dropna(),
        'Store_e_nom_min': static(n, 'Store', 'e_nom_min'),
        'Store_e_nom_max': static(n, 'Store', 'e_nom_max'),
        'Store_capital_cost': static(n, 'Store', 'capital_cost'),
        'Store_e_nom_set': static(n, 'Store', 'e_nom_set').dropna(),
        'Line_s_nom': static(n, 'Line', 's_nom'),
        'Line_s_nom_extendable': static(n, 'Line', 's_nom_extendable'),
        'Line_s_max_pu': varying(n, 'Line', 's_max_pu'),
        'Line_s_nom_min': static(n, 'Line', 's_nom_min'),
        'Line_s_nom_max': static(n, 'Line', 's_nom_max'),
        'Line_capital_cost': static(n, 'Line', 'capital_cost'),
        'Line_s_nom_set': static(n, 'Line', 's_nom_set').dropna(),
        'Line_s_set': varying(n, 'Line', 's_set').dropna(),
        'Line_cycle_weight': _cycle_weights(n),
        'GlobalConstraint_type': static(n, 'GlobalConstraint', 'type').astype({'value': str}),
        'GlobalConstraint_sense': static(n, 'GlobalConstraint', 'sense').astype({'value': str}),
        'GlobalConstraint_constant': _gc_constants(n),
        'snapshot_is_last': pd.DataFrame(
            {'snapshot': n.snapshots, 'value': [0] * (len(n.snapshots) - 1) + [1] if len(n.snapshots) else []}
        ),
        'Generator_marginal_cost_quadratic': varying(n, 'Generator', 'marginal_cost_quadratic'),
        'Link_marginal_cost_quadratic': varying(n, 'Link', 'marginal_cost_quadratic'),
    }

    primary, operational = _typed(n, 'primary_energy'), _typed(n, 'operational_limit')
    volume, expansion_cost = (
        _typed(n, 'transmission_volume_expansion_limit'),
        _typed(n, 'transmission_expansion_cost_limit'),
    )
    tech = _typed(n, 'tech_capacity_expansion_limit')
    tables |= {
        'Generator_primary_energy_weight': _weights(
            primary, generators, 'generator', lambda gc, g: _emissions(n, gc).get(g['carrier'], 0.0) / g['efficiency']
        ),
        'StorageUnit_primary_energy_weight': _weights(
            primary,
            storage_units,
            'storage_unit',
            lambda gc, s: 0.0 if s['cyclic_state_of_charge'] else _emissions(n, gc).get(s['carrier'], 0.0),
        ),
        'Store_primary_energy_weight': _weights(
            primary, stores, 'store', lambda gc, s: 0.0 if s['e_cyclic'] else _emissions(n, gc).get(s['carrier'], 0.0)
        ),
        'Generator_operational_limit_weight': _weights(
            operational, generators, 'generator', lambda gc, g: float(g['carrier'] == gc['carrier_attribute'])
        ),
        'StorageUnit_operational_limit_weight': _weights(
            operational,
            storage_units,
            'storage_unit',
            lambda gc, s: float(s['carrier'] == gc['carrier_attribute'] and not s['cyclic_state_of_charge']),
        ),
        'Store_operational_limit_weight': _weights(
            operational,
            stores,
            'store',
            lambda gc, s: float(s['carrier'] == gc['carrier_attribute'] and not s['e_cyclic']),
        ),
        'Line_volume_weight': _weights(
            volume,
            lines,
            'line',
            lambda gc, c: c['length'] if c['s_nom_extendable'] and c['carrier'] in _carrier_list(gc) else 0.0,
        ),
        'Link_volume_weight': _weights(
            volume,
            links,
            'link',
            lambda gc, c: c['length'] if c['p_nom_extendable'] and c['carrier'] in _carrier_list(gc) else 0.0,
        ),
        'Line_expansion_cost_weight': _weights(
            expansion_cost,
            lines,
            'line',
            lambda gc, c: c['capital_cost'] if c['s_nom_extendable'] and c['carrier'] in _carrier_list(gc) else 0.0,
        ),
        'Link_expansion_cost_weight': _weights(
            expansion_cost,
            links,
            'link',
            lambda gc, c: c['capital_cost'] if c['p_nom_extendable'] and c['carrier'] in _carrier_list(gc) else 0.0,
        ),
        'Generator_tech_capacity_weight': _weights(
            tech, generators, 'generator', lambda gc, c: float(_in_tech_set(gc, c, 'p_nom', 'bus'))
        ),
        'Link_tech_capacity_weight': _weights(
            tech, links, 'link', lambda gc, c: float(_in_tech_set(gc, c, 'p_nom', 'bus0'))
        ),
        'Line_tech_capacity_weight': _weights(
            tech, lines, 'line', lambda gc, c: float(_in_tech_set(gc, c, 's_nom', 'bus0'))
        ),
        'StorageUnit_tech_capacity_weight': _weights(
            tech, storage_units, 'storage_unit', lambda gc, c: float(_in_tech_set(gc, c, 'p_nom', 'bus'))
        ),
        'Store_tech_capacity_weight': _weights(
            tech, stores, 'store', lambda gc, c: float(_in_tech_set(gc, c, 'e_nom', 'bus'))
        ),
    }

    tables['cycle'] = pl.Series('cycle', list(pd.unique(tables['Line_cycle_weight']['cycle'])), dtype=pl.String)
    tables['Link_bus2'] = lookup(n, 'Link', 'bus2')
    with_second_port = tables['Link_bus2']['link']
    tables['Link_efficiency2'] = static(n, 'Link', 'efficiency2').loc[lambda t: t['link'].isin(with_second_port)]

    for name, table in tables.items():
        if isinstance(table, pd.DataFrame):
            lost = {
                column: 'int64' if column == 'snapshot' else 'string'
                for column in table.columns
                if table[column].dtype == object
            }
            tables[name] = pl.from_pandas(table.astype(lost))
    return tables
