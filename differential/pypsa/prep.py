# SPDX-FileCopyrightText: math-spec Contributors
#
# SPDX-License-Identifier: MIT

"""The prep layer: a PyPSA network as the tables the example models bind.

Every parameter the files mark "data prep" is computed here, beside the plain
renames. `parity.py` is the caller and cuts the tables to what each model
declares; nothing here imports math_spec or lpspec — the mapping is pure
PyPSA-and-pandas, handed over as polars frames.

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


def _names(component: pd.DataFrame) -> pd.Index:
    """The component names, without the scenario level a stochastic network puts in front of them."""
    index = component.index
    return index.unique(level='name') if isinstance(index, pd.MultiIndex) else index


def _static(component: pd.DataFrame, attr: str, dim: str, *, sparse: bool = False) -> pd.DataFrame:
    index = component.index
    if isinstance(index, pd.MultiIndex):
        table = pd.DataFrame(
            {
                'scenario': index.get_level_values('scenario').astype(str),
                dim: index.get_level_values('name').astype(str),
                'value': component[attr].to_numpy(),
            }
        )
    else:
        table = pd.DataFrame({dim: index.astype(str), 'value': component[attr].to_numpy()})
    return table.dropna() if sparse else table


def _melt(dense: pd.DataFrame, dim: str) -> pd.DataFrame:
    if isinstance(dense.index, pd.MultiIndex):
        dense = dense.set_axis(pd.Index(list(dense.index), tupleize_cols=False), axis=0)
    if dense.columns.empty:
        return pd.DataFrame(columns=['snapshot', dim, 'value']).astype({dim: str, 'value': float})
    if isinstance(dense.columns, pd.MultiIndex):
        table = dense.melt(ignore_index=False, var_name=['scenario', dim]).reset_index(names='snapshot')
        return table.astype({'scenario': str, dim: str, 'value': float})
    table = dense.melt(ignore_index=False, var_name=dim).reset_index(names='snapshot')
    return table.astype({dim: str, 'value': float})


def _one_scenario(component: pd.DataFrame) -> pd.DataFrame:
    """A static frame as one scenario sees it — for the derived tables that do not vary by scenario."""
    index = component.index
    if not isinstance(index, pd.MultiIndex):
        return component
    if len(index) == 0:
        return component.droplevel('scenario')
    return component.xs(index.get_level_values('scenario')[0], level='scenario')


def _collapse(tables: dict[str, object]) -> dict[str, object]:
    """Drop the scenario column where a table is the same in every scenario; keep it where the data differs."""
    out = {}
    for name, table in tables.items():
        if isinstance(table, pd.DataFrame) and 'scenario' in table.columns:
            keys = [c for c in table.columns if c not in ('scenario', 'value')]
            wide = table.pivot_table(index=keys, columns='scenario', values='value', aggfunc='first') if keys else table.set_index('scenario')['value'].to_frame().T
            if wide.nunique(axis=1, dropna=False).le(1).all():
                table = table[table['scenario'] == table['scenario'].iloc[0]].drop(columns='scenario') if len(table) else table.drop(columns='scenario')
        out[name] = table
    return out


def _varying(n: pypsa.Network, component: str, attr: str, dim: str, *, sparse: bool = False) -> pd.DataFrame:
    table = _flat(n, _melt(get_switchable_as_dense(n, component, attr), dim))
    return table.dropna() if sparse else table


def _lookup(component: pd.DataFrame, attr: str, over: str, into: str) -> pd.DataFrame:
    component = _one_scenario(component)
    table = pd.DataFrame({over: component.index.astype(str), into: component[attr].astype(str)})
    return table[table[into] != '']


def _flat(n: pypsa.Network, table: pd.DataFrame) -> pd.DataFrame:
    """Snapshots as positions where a multi-period network keys them by (period, timestep)."""
    if isinstance(n.snapshots, pd.MultiIndex) and 'snapshot' in table.columns and len(table):
        positions = {key: i for i, key in enumerate(n.snapshots)}
        table = table.assign(snapshot=table['snapshot'].map(positions).astype(int))
    return table


def _weighting(n: pypsa.Network, column: str) -> pd.DataFrame:
    return _flat(n, pd.DataFrame({'snapshot': list(n.snapshots), 'value': n.snapshot_weightings[column].to_numpy()}))


def _retention(n: pypsa.Network, component: str, dim: str) -> pd.DataFrame:
    losses = _one_scenario(n.static(component))['standing_loss']
    hours = n.snapshot_weightings['stores']
    return _flat(n, _melt(pd.DataFrame({name: (1.0 - loss) ** hours for name, loss in losses.items()}, index=n.snapshots), dim))


def _cycle_weights(n: pypsa.Network) -> pd.DataFrame:
    """The KVL basis PyPSA itself solves with — ``n.cycle_matrix(apply_weights=True)``: reactance on AC, resistance on DC."""
    n.determine_network_topology()
    n.calculate_dependent_values()
    C = n.cycle_matrix(apply_weights=True)
    rows = [
        {'line': str(name), 'cycle': str(cycle), 'value': float(weight)}
        for (kind, name), weights in C.iterrows()
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
    for name, g in _one_scenario(n.generators).iterrows():
        if not g['committable'] or g['up_time_before'] <= 0:
            continue
        remaining = int(min(g['min_up_time'] - g['up_time_before'], len(n.snapshots)))
        rows.extend({'snapshot': t, 'generator': str(name), 'value': True} for t in range(max(remaining, 0)))
    table = pd.DataFrame(rows, columns=['snapshot', 'generator', 'value'])
    return table.astype({'value': bool})


def _scenarios(n: pypsa.Network) -> dict[str, object]:
    """The scenario dimension, its weights, and the risk preference PyPSA's CVaR rows read — empty without scenarios."""
    if not n.has_scenarios:
        return {}
    weights = n.scenario_weightings['weight']
    tables: dict[str, object] = {
        'scenario': pl.Series('scenario', [str(s) for s in weights.index], dtype=pl.String),
        'scenario_weight': pd.DataFrame({'scenario': weights.index.astype(str), 'value': weights.to_numpy(dtype=float)}),
    }
    if n.has_risk_preference:
        rp = n.risk_preference
        tables['CVaR_omega'] = pd.DataFrame({'value': [float(rp['omega'])]})
        tables['CVaR_inv_tail'] = pd.DataFrame({'value': [1.0 / (1.0 - float(rp['alpha']))]})
    return tables


def _periods(n: pypsa.Network) -> dict[str, object]:
    """The investment periods PyPSA's multi-period rows read: which period a snapshot is in and weighs, which assets stand in it, and the growth caps."""
    if not isinstance(n.snapshots, pd.MultiIndex):
        return {}
    periods = list(n.investment_periods)
    weights = n.investment_period_weightings['objective']
    generators = _one_scenario(n.generators)
    active = pd.DataFrame({p: n.c.generators.get_active_assets(p) for p in periods})
    first = (active.cumsum(axis=1) == 1) & active
    carriers = n.carriers
    growth = carriers['max_growth']
    return {
        'period': pl.Series('period', [int(p) for p in periods], dtype=pl.Int64),
        'carrier': pl.Series('carrier', list(carriers.index.astype(str)), dtype=pl.String),
        'snapshot_period': pd.DataFrame({'snapshot': range(len(n.snapshots)), 'period': [int(p) for p, _ in n.snapshots]}),
        'period_weight_objective': pd.DataFrame({'period': [int(p) for p in periods], 'value': weights.loc[periods].to_numpy(dtype=float)}),
        'Generator_active': pd.DataFrame(
            [{'snapshot': i, 'generator': str(g), 'value': bool(active.at[g, p])} for i, (p, _) in enumerate(n.snapshots) for g in generators.index]
        ),
        'Generator_capital_weight': pd.DataFrame({'generator': generators.index.astype(str), 'value': (active * weights.loc[periods]).sum(axis=1).to_numpy(dtype=float)}),
        'Generator_first_active': pd.DataFrame(
            [{'period': int(p), 'generator': str(g), 'value': float(first.at[g, p])} for p in periods for g in generators.index]
        ),
        'Generator_carrier': _lookup(generators, 'carrier', 'generator', 'carrier'),
        'Carrier_max_growth': pd.DataFrame({'carrier': growth.index.astype(str), 'value': growth.to_numpy(dtype=float)}).replace([float('inf')], float('nan')).dropna(),
        'Carrier_max_relative_growth': pd.DataFrame({'carrier': carriers.index.astype(str), 'value': carriers['max_relative_growth'].clip(lower=0).to_numpy(dtype=float)}),
    }


def _loss_fan(n: pypsa.Network, segments: int) -> dict[str, pd.DataFrame]:
    """The tangent fan PyPSA's `define_tangent_loss_constraints` builds: the loss at rating, and a slope and offset per segment."""
    n.calculate_dependent_values()
    lines = n.lines
    if lines.empty or not segments:
        empty = pd.DataFrame(columns=['snapshot', 'line', 'value'])
        return {'segment': pd.DataFrame({'segment': []}), 'Line_loss_max': empty, 'Line_loss_slope': empty.assign(segment=[]), 'Line_loss_offset': empty.assign(segment=[])}
    s_max_pu = get_switchable_as_dense(n, 'Line', 's_max_pu')
    s_nom_max = lines.s_nom_max.where(lines.s_nom_extendable, lines.s_nom)
    top = s_max_pu * s_nom_max
    r = lines.r_pu_eff
    upper = _flat(n, _melt(r * top**2, 'line'))
    slope, offset = [], []
    for k in range(1, segments + 1):
        p_k = k / segments * top
        slope.append(_flat(n, _melt(2 * r * p_k, 'line')).assign(segment=k))
        offset.append(_flat(n, _melt(r * p_k**2 - 2 * r * p_k * p_k, 'line')).assign(segment=k))
    return {
        'segment': pd.DataFrame({'segment': range(1, segments + 1)}),
        'Line_loss_max': upper,
        'Line_loss_slope': pd.concat(slope, ignore_index=True),
        'Line_loss_offset': pd.concat(offset, ignore_index=True),
    }


def sources(n: pypsa.Network, *, segments: int = 0) -> dict[str, object]:
    """Every table the example models bind, from one PyPSA network."""
    generators, links, loads = n.generators, n.links, n.loads
    storage_units, stores, lines = n.storage_units, n.stores, n.lines
    one = _one_scenario(generators)
    p_max_pu = get_switchable_as_dense(n, 'Generator', 'p_max_pu')
    if isinstance(p_max_pu.columns, pd.MultiIndex):
        p_max_pu = p_max_pu.T.groupby(level='name').max().T
    big_m = one['p_nom_max'] * p_max_pu.max().clip(lower=1.0)

    tables: dict[str, object] = {
        'snapshot': pl.Series('snapshot', list(range(len(n.snapshots))) if isinstance(n.snapshots, pd.MultiIndex) else list(n.snapshots), dtype=pl.Int64),
        'bus': pl.Series('bus', list(_names(n.buses).astype(str)), dtype=pl.String),
        'generator': pl.Series('generator', list(_names(generators).astype(str)), dtype=pl.String),
        'link': pl.Series('link', list(_names(links).astype(str)), dtype=pl.String),
        'load': pl.Series('load', list(_names(loads).astype(str)), dtype=pl.String),
        'storage_unit': pl.Series('storage_unit', list(_names(storage_units).astype(str)), dtype=pl.String),
        'store': pl.Series('store', list(_names(stores).astype(str)), dtype=pl.String),
        'line': pl.Series('line', list(_names(lines).astype(str)), dtype=pl.String),
        'global_constraint': pl.Series(
            'global_constraint', list(_names(n.global_constraints).astype(str)), dtype=pl.String
        ),
        **_scenarios(n),
        **_periods(n),
        'Generator_bus': _lookup(generators, 'bus', 'generator', 'bus'),
        'Link_bus0': _lookup(links, 'bus0', 'link', 'bus'),
        'Link_bus1': _lookup(links, 'bus1', 'link', 'bus'),
        'Load_bus': _lookup(loads, 'bus', 'load', 'bus'),
        'StorageUnit_bus': _lookup(storage_units, 'bus', 'storage_unit', 'bus'),
        'Store_bus': _lookup(stores, 'bus', 'store', 'bus'),
        'Line_bus0': _lookup(lines, 'bus0', 'line', 'bus'),
        'Line_bus1': _lookup(lines, 'bus1', 'line', 'bus'),
        'snapshot_weightings_objective': _weighting(n, 'objective'),
        'snapshot_weightings_stores': _weighting(n, 'stores'),
        'snapshot_weightings_generators': _weighting(n, 'generators'),
        'Load_p_set': _varying(n, 'Load', 'p_set', 'load'),
        'Generator_p_nom': _static(generators, 'p_nom', 'generator'),
        'Generator_p_nom_extendable': _static(generators, 'p_nom_extendable', 'generator'),
        'Generator_p_min_pu': _varying(n, 'Generator', 'p_min_pu', 'generator'),
        'Generator_p_max_pu': _varying(n, 'Generator', 'p_max_pu', 'generator'),
        'Generator_marginal_cost': _varying(n, 'Generator', 'marginal_cost', 'generator'),
        'Generator_p_set': _varying(n, 'Generator', 'p_set', 'generator', sparse=True),
        'Generator_p_nom_min': _static(generators, 'p_nom_min', 'generator'),
        'Generator_p_nom_max': _static(generators, 'p_nom_max', 'generator'),
        'Generator_capital_cost': _static(generators, 'capital_cost', 'generator'),
        'Generator_p_nom_set': _static(generators, 'p_nom_set', 'generator', sparse=True),
        'Generator_e_sum_min': _static(generators, 'e_sum_min', 'generator'),
        'Generator_e_sum_max': _static(generators, 'e_sum_max', 'generator'),
        'Generator_committable': _static(generators, 'committable', 'generator'),
        'Generator_ramp_limit_up': _static(generators, 'ramp_limit_up', 'generator', sparse=True),
        'Generator_ramp_limit_down': _static(generators, 'ramp_limit_down', 'generator', sparse=True),
        'Generator_ramp_limit_start_up': _static(
            generators.fillna({'ramp_limit_start_up': 1.0}), 'ramp_limit_start_up', 'generator'
        ),
        'Generator_ramp_limit_shut_down': _static(
            generators.fillna({'ramp_limit_shut_down': 1.0}), 'ramp_limit_shut_down', 'generator'
        ),
        'Generator_min_up_time': _static(generators, 'min_up_time', 'generator'),
        'Generator_min_down_time': _static(generators, 'min_down_time', 'generator'),
        'Generator_status_initial': pd.DataFrame(
            {'generator': one.index.astype(str), 'value': (one['up_time_before'] > 0).astype(int).to_numpy()}
        ),
        'Generator_must_stay_up': _must_stay_up(n),
        'Generator_start_up_cost': _static(generators, 'start_up_cost', 'generator'),
        'Generator_shut_down_cost': _static(generators, 'shut_down_cost', 'generator'),
        'Generator_stand_by_cost': _varying(n, 'Generator', 'stand_by_cost', 'generator'),
        'Generator_p_nom_mod': _static(generators[generators['p_nom_mod'] > 0], 'p_nom_mod', 'generator'),
        'Generator_big_m': pd.DataFrame({'generator': one.index.astype(str), 'value': big_m.to_numpy()}),
        'Generator_p_min_pu_nonneg': pd.DataFrame(
            {
                'generator': one.index.astype(str),
                'value': (get_switchable_as_dense(n, 'Generator', 'p_min_pu') >= 0).all().groupby(level='name').all().to_numpy()
                if isinstance(generators.index, pd.MultiIndex)
                else (get_switchable_as_dense(n, 'Generator', 'p_min_pu') >= 0).all().to_numpy(),
            }
        ),
        'Link_p_nom': _static(links, 'p_nom', 'link'),
        'Link_p_nom_extendable': _static(links, 'p_nom_extendable', 'link'),
        'Link_p_min_pu': _varying(n, 'Link', 'p_min_pu', 'link'),
        'Link_p_max_pu': _varying(n, 'Link', 'p_max_pu', 'link'),
        'Link_efficiency': _static(links, 'efficiency', 'link'),
        'Link_marginal_cost': _varying(n, 'Link', 'marginal_cost', 'link'),
        'Link_p_set': _varying(n, 'Link', 'p_set', 'link', sparse=True),
        'Link_p_nom_min': _static(links, 'p_nom_min', 'link'),
        'Link_p_nom_max': _static(links, 'p_nom_max', 'link'),
        'Link_capital_cost': _static(links, 'capital_cost', 'link'),
        'Link_p_nom_set': _static(links, 'p_nom_set', 'link', sparse=True),
        'Link_ramp_limit_up': _static(links, 'ramp_limit_up', 'link', sparse=True),
        'Link_ramp_limit_down': _static(links, 'ramp_limit_down', 'link', sparse=True),
        'StorageUnit_p_nom': _static(storage_units, 'p_nom', 'storage_unit'),
        'StorageUnit_p_nom_extendable': _static(storage_units, 'p_nom_extendable', 'storage_unit'),
        'StorageUnit_p_min_pu': _varying(n, 'StorageUnit', 'p_min_pu', 'storage_unit'),
        'StorageUnit_p_max_pu': _varying(n, 'StorageUnit', 'p_max_pu', 'storage_unit'),
        'StorageUnit_max_hours': _static(storage_units, 'max_hours', 'storage_unit'),
        'StorageUnit_efficiency_store': _static(storage_units, 'efficiency_store', 'storage_unit'),
        'StorageUnit_efficiency_dispatch': _static(storage_units, 'efficiency_dispatch', 'storage_unit'),
        'StorageUnit_retention': _retention(n, 'StorageUnit', 'storage_unit'),
        'StorageUnit_inflow': _varying(n, 'StorageUnit', 'inflow', 'storage_unit'),
        'StorageUnit_state_of_charge_initial': _static(storage_units, 'state_of_charge_initial', 'storage_unit'),
        'StorageUnit_cyclic_state_of_charge': _static(storage_units, 'cyclic_state_of_charge', 'storage_unit'),
        'StorageUnit_marginal_cost': _varying(n, 'StorageUnit', 'marginal_cost', 'storage_unit'),
        'StorageUnit_marginal_cost_storage': _varying(n, 'StorageUnit', 'marginal_cost_storage', 'storage_unit'),
        'StorageUnit_spill_cost': _varying(n, 'StorageUnit', 'spill_cost', 'storage_unit'),
        'StorageUnit_p_set': _varying(n, 'StorageUnit', 'p_set', 'storage_unit', sparse=True),
        'StorageUnit_state_of_charge_set': _varying(
            n, 'StorageUnit', 'state_of_charge_set', 'storage_unit', sparse=True
        ),
        'StorageUnit_p_nom_min': _static(storage_units, 'p_nom_min', 'storage_unit'),
        'StorageUnit_p_nom_max': _static(storage_units, 'p_nom_max', 'storage_unit'),
        'StorageUnit_capital_cost': _static(storage_units, 'capital_cost', 'storage_unit'),
        'StorageUnit_p_nom_set': _static(storage_units, 'p_nom_set', 'storage_unit', sparse=True),
        'Store_e_nom': _static(stores, 'e_nom', 'store'),
        'Store_e_nom_extendable': _static(stores, 'e_nom_extendable', 'store'),
        'Store_e_min_pu': _varying(n, 'Store', 'e_min_pu', 'store'),
        'Store_e_max_pu': _varying(n, 'Store', 'e_max_pu', 'store'),
        'Store_retention': _retention(n, 'Store', 'store'),
        'Store_e_initial': _static(stores, 'e_initial', 'store'),
        'Store_e_cyclic': _static(stores, 'e_cyclic', 'store'),
        'Store_marginal_cost': _varying(n, 'Store', 'marginal_cost', 'store'),
        'Store_marginal_cost_storage': _varying(n, 'Store', 'marginal_cost_storage', 'store'),
        'Store_e_set': _varying(n, 'Store', 'e_set', 'store', sparse=True),
        'Store_e_nom_min': _static(stores, 'e_nom_min', 'store'),
        'Store_e_nom_max': _static(stores, 'e_nom_max', 'store'),
        'Store_capital_cost': _static(stores, 'capital_cost', 'store'),
        'Store_e_nom_set': _static(stores, 'e_nom_set', 'store', sparse=True),
        'Line_s_nom': _static(lines, 's_nom', 'line'),
        'Line_s_nom_extendable': _static(lines, 's_nom_extendable', 'line'),
        'Line_s_max_pu': _varying(n, 'Line', 's_max_pu', 'line'),
        'Line_s_nom_min': _static(lines, 's_nom_min', 'line'),
        'Line_s_nom_max': _static(lines, 's_nom_max', 'line'),
        'Line_capital_cost': _static(lines, 'capital_cost', 'line'),
        'Line_s_nom_set': _static(lines, 's_nom_set', 'line', sparse=True),
        'Line_s_set': _varying(n, 'Line', 's_set', 'line', sparse=True),
        'Line_cycle_weight': _cycle_weights(n),
        'GlobalConstraint_type': _static(n.global_constraints, 'type', 'global_constraint').astype({'value': str}),
        'GlobalConstraint_sense': _static(n.global_constraints, 'sense', 'global_constraint').astype({'value': str}),
        'GlobalConstraint_constant': _gc_constants(n),
        'snapshot_is_last': _flat(
            n, pd.DataFrame({'snapshot': list(n.snapshots), 'value': [0] * (len(n.snapshots) - 1) + [1] if len(n.snapshots) else []})
        ),
        'Generator_marginal_cost_quadratic': _varying(n, 'Generator', 'marginal_cost_quadratic', 'generator'),
        **_loss_fan(n, segments),
        'Generator_partly_tightened': pd.DataFrame(
            {'generator': one.index.astype(str), 'value': (one.start_up_cost == one.shut_down_cost).to_numpy()}
        ),
        'Link_marginal_cost_quadratic': _varying(n, 'Link', 'marginal_cost_quadratic', 'link'),
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
    if 'bus2' not in links.columns:
        links = links.assign(bus2='', efficiency2=1.0)
    tables['Link_bus2'] = _lookup(links, 'bus2', 'link', 'bus')
    tables['Link_efficiency2'] = _static(links[links['bus2'] != ''], 'efficiency2', 'link')

    tables = _collapse(tables)
    for name, table in tables.items():
        if isinstance(table, pd.DataFrame):
            lost = {
                column: 'int64' if column == 'snapshot' else 'string'
                for column in table.columns
                if table[column].dtype == object
            }
            tables[name] = pl.from_pandas(table.astype(lost))
    return tables
