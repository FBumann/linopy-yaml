"""The models we time, and the data that sizes them.

One case = one YAML model + a deterministic data generator + a size ladder.
Cases are chosen so each stresses a *different* SQL shape (docs/ARCHITECTURE.md,
"read the verdict off the SQL"), not to cover the language:

``dispatch``   pointwise bounds + one ``sum`` — raw throughput, and the case
               where a dense eager broadcast is at its best, so our worst ratio.
               Its ``where`` is declared but *vacuous*, which is a measurement
               in itself: the engine pays for a mask that removes nothing.
``commitment`` dispatch with a binary commitment gating every generator — the
               MILP, and the only case whose ``vtype`` stream is not
               all-continuous. Its bottom rung is deliberately tiny: the parity
               gate solves it as a MIP, and the objectives only compare at
               ``GATE_RTOL`` if branch and bound closes the gap exactly.
``nodal``      dispatch over (snapshot, node, tech) where a technology only
               exists at the nodes it is installed at — the sparsity every real
               multi-node model has, and the one axis where the two lanes do
               different *amounts* of work rather than the same work in a
               different order.
``transport``  three ``sum(group_by=)`` joins per row — the mapping-table path, where
               the eager lane has to materialise a bus x generator product.
``storage``    a cyclic ``shift`` recurrence — the only locality class whose cost
               has no eager analogue: xarray shifts an array, we join a term
               stream against itself on ``snapshot.ord - 1``. Held at
               ``dispatch``'s width on ``dispatch``'s snapshot counts, so the two
               ladders differ in exactly one thing: whether a row reaches the
               previous one. ``soc_balance`` carries a row per store against
               ``power_balance``'s one, so the self-join dominates the case
               rather than garnishing it.
``sector``     ``nodal``'s sparse portfolio crossed with dense carriers: ``p`` is
               sparse in (node, tech) while ``shed`` and the balance are dense in
               (node, carrier), and the objective spans both.
``fleet``      the same variable total spread over many declarations rather than
               one large one — the only axis it varies.
``declarations`` ``fleet``'s question as a sweep: one model size, a unit pool
               split into N declarations for several N, so the per-declaration
               cost every other ladder holds fixed is varied on its own axis.
               Its model YAML is generated per rung (``_declarations_model``).
``profiled``   ``nodal``'s ladder with no mask, held at the same cardinalities so
               the two differ in exactly one thing: whether the parameters span
               the variable product or a subset of it. Its availability table has
               a row per variable, which is where "I/O is noise" gets tested.

A rung label counts variables per snapshot across *all* of a case's
declarations, so the ladders read against each other.  ``nominal_variables`` is
the full coordinate product; what survives a mask is measured rather than
assumed (``live`` in the report).

Data is generated once per (case, shape) into a cache directory and both arms
read the same parquet files, so no arm pays a generation cost and neither can
be measured against different numbers. Feasibility is by construction — every
bus serves its own load with no flow, and ``sparse`` sizes its load against the
tightest snapshot — so a solve never fails for a reason the harness invented.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

BENCH_DIR = Path(__file__).resolve().parent
MODELS = BENCH_DIR / 'models'
DEFAULT_CACHE = BENCH_DIR / '.cache'


@dataclass(frozen=True)
class Shape:
    """One rung of a ladder: the dimension cardinalities, and how much survives.

    ``density`` is the fraction of the coordinate product a case's mask keeps.
    It is a rung axis rather than a case, because sparsity is the one place the
    two representations of a mask differ in kind — row absence relationally,
    NaN-padding in a dense array — so it has to be swept, not sampled once.
    Cases with no mask leave it at 1.0.
    """

    label: str
    sizes: dict[str, int]
    nominal_variables: int
    density: float = 1.0

    @property
    def key(self) -> str:
        dims = '-'.join(f'{k}{v}' for k, v in sorted(self.sizes.items()))
        return dims if self.density == 1.0 else f'{dims}-d{self.density:g}'


@dataclass(frozen=True)
class Case:
    name: str
    ladder: tuple[Shape, ...]
    write: Callable[[Shape, Path], dict[str, str]]
    eager_inputs: Callable[[dict[str, str]], tuple[dict[str, Any], dict[str, Any]]]
    model: Path | None = None
    generate_model: Callable[[Shape], str] | None = None

    def model_path(self, shape: Shape, cache: Path = DEFAULT_CACHE) -> Path:
        """The YAML *shape* builds — ``model``, unless the case generates one per rung.

        A generated model is cached beside the rung's data, under ``shape.key``,
        so the two arms — separate processes — read the same file and a rung
        never sees another rung's declarations.
        """
        if self.generate_model is None:
            if self.model is None:
                raise ValueError(f'{self.name}: neither a model file nor a generator — nothing to build')
            return self.model
        dest = cache / self.name / shape.key / 'model.yaml'
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(self.generate_model(shape))
        return dest

    def shape(self, label: str) -> Shape:
        for s in self.ladder:
            if s.label == label:
                return s
        known = ', '.join(s.label for s in self.ladder)
        raise KeyError(f"{self.name}: no size '{label}' (have: {known})")

    def data(self, shape: Shape, cache: Path = DEFAULT_CACHE) -> dict[str, str]:
        """Parquet paths for *shape*, generating them on first use."""
        out = cache / self.name / shape.key
        stamp = out / '.complete'
        if not stamp.exists():
            out.mkdir(parents=True, exist_ok=True)
            paths = self.write(shape, out)
            stamp.write_text('\n'.join(sorted(paths)))
            return paths
        return {p.stem: str(p) for p in sorted(out.glob('*.parquet'))}


def _seed(shape: Shape) -> np.random.Generator:
    """Same shape, same numbers — on any machine, in any arm, forever.

    ``hash()`` is salted per process, so it cannot be used here: the two arms
    run in different processes and must see byte-identical data.
    """
    digest = hashlib.blake2b(shape.key.encode(), digest_size=4).digest()
    return np.random.default_rng(int.from_bytes(digest, 'big'))


def _dump(frames: dict[str, pd.DataFrame], dest: Path) -> dict[str, str]:
    paths = {}
    for name, df in frames.items():
        path = (dest / f'{name}.parquet').absolute()
        df.to_parquet(path, index=False)
        paths[name] = str(path)
    return paths


def _installed_frame(nodes: list[str], techs: list[str], installed: np.ndarray, capacity: np.ndarray) -> pd.DataFrame:
    """The (node, tech) pairs that exist, tidy — the table *is* the sparsity."""
    live = installed.reshape(-1)
    return pd.DataFrame(
        {
            'node': np.repeat(nodes, len(techs))[live],
            'tech': np.tile(techs, len(nodes))[live],
            'value': capacity.reshape(-1)[live],
        }
    )


def _dense_pairs(path: str, index: pd.Series, columns: pd.Series) -> pd.DataFrame:
    """A sparse two-key table reindexed over its full product.

    The eager lane has nowhere to put an absent pair, so this is where
    structural sparsity becomes NaN padding. Done in the open, because doing it
    at all is the point of the sparse cases.
    """
    return (
        pd.read_parquet(path)
        .set_index([index.name, columns.name])['value']
        .unstack()
        .reindex(index=index, columns=columns)
        .fillna(0.0)
    )


# --------------------------------------------------------------------------
# dispatch


def _dispatch_data(shape: Shape, dest: Path) -> dict[str, str]:
    """Parquet for one rung of the ``dispatch`` ladder.

    Load is drawn against the fleet total so it is always feasible and never so
    slack that every generator prices in at zero.
    """
    rng = _seed(shape)
    n_snap, n_gen = shape.sizes['snapshot'], shape.sizes['generator']
    gens = [f'g{i:05d}' for i in range(n_gen)]

    p_max = rng.uniform(50.0, 150.0, n_gen)
    cost = rng.uniform(10.0, 100.0, n_gen)
    load = p_max.sum() * 0.6 * (0.8 + 0.4 * rng.random(n_snap))

    return _dump(
        {
            'p_max': pd.DataFrame({'generator': gens, 'value': p_max}),
            'cost': pd.DataFrame({'generator': gens, 'value': cost}),
            'load': pd.DataFrame({'snapshot': np.arange(n_snap), 'value': load}),
            'generator': pd.DataFrame({'generator': gens}),
            'snapshot': pd.DataFrame({'snapshot': np.arange(n_snap)}),
        },
        dest,
    )


def _dispatch_eager(paths: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The same parquet files, as the linopy lane wants them.

    Reading them is the eager arm's own cost and is timed inside its build
    phase — that is how linopy is actually used, and pretending the data is
    already in memory would flatter it.
    """
    p_max = pd.read_parquet(paths['p_max']).set_index('generator')['value']
    cost = pd.read_parquet(paths['cost']).set_index('generator')['value']
    load = pd.read_parquet(paths['load']).set_index('snapshot')['value']
    data = {'p_max': p_max, 'cost': cost, 'load': load}
    coords = {
        'generator': pd.Index(p_max.index, name='generator'),
        'snapshot': pd.Index(load.index, name='snapshot'),
    }
    return data, coords


# --------------------------------------------------------------------------
# commitment — dispatch's MILP twin, the case whose vtype is not all-continuous


def _commitment_data(shape: Shape, dest: Path) -> dict[str, str]:
    """Parquet for one rung of the ``commitment`` ladder.

    Load is ``dispatch``'s draw, so every snapshot is feasible with the whole
    fleet on. Fix costs are drawn wide and every cost is a distinct float, so
    the optimal commitment is a real choice — an all-on optimum would stream
    the binaries and never branch on them.
    """
    rng = _seed(shape)
    n_snap, n_gen = shape.sizes['snapshot'], shape.sizes['generator']
    gens = [f'g{i:05d}' for i in range(n_gen)]

    p_max = rng.uniform(50.0, 150.0, n_gen)
    cost = rng.uniform(10.0, 100.0, n_gen)
    fix_cost = rng.uniform(100.0, 2000.0, n_gen)
    load = p_max.sum() * 0.6 * (0.8 + 0.4 * rng.random(n_snap))

    return _dump(
        {
            'p_max': pd.DataFrame({'generator': gens, 'value': p_max}),
            'cost': pd.DataFrame({'generator': gens, 'value': cost}),
            'fix_cost': pd.DataFrame({'generator': gens, 'value': fix_cost}),
            'load': pd.DataFrame({'snapshot': np.arange(n_snap), 'value': load}),
            'generator': pd.DataFrame({'generator': gens}),
            'snapshot': pd.DataFrame({'snapshot': np.arange(n_snap)}),
        },
        dest,
    )


def _commitment_eager(paths: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    p_max = pd.read_parquet(paths['p_max']).set_index('generator')['value']
    load = pd.read_parquet(paths['load']).set_index('snapshot')['value']
    data = {
        'p_max': p_max,
        'cost': pd.read_parquet(paths['cost']).set_index('generator')['value'],
        'fix_cost': pd.read_parquet(paths['fix_cost']).set_index('generator')['value'],
        'load': load,
    }
    coords = {
        'generator': pd.Index(p_max.index, name='generator'),
        'snapshot': pd.Index(load.index, name='snapshot'),
    }
    return data, coords


# --------------------------------------------------------------------------
# nodal — a technology portfolio per node, which is where real sparsity comes from

#: Technologies a system might have. Which ones a given node *has* is the mask.
TECHNOLOGIES = (
    'onwind',
    'offwind',
    'solar',
    'hydro',
    'ror',
    'biomass',
    'geothermal',
    'ccgt',
    'ocgt',
    'coal',
    'nuclear',
    'oil',
)


def _portfolios(rng: np.random.Generator, n_node: int, n_tech: int, density: float) -> np.ndarray:
    """Which (node, tech) pairs exist — a boolean node x tech matrix.

    Every node gets at least one technology, or its demand cannot be met and
    the parity gate has no two objectives to compare. The rest are drawn to hit
    the requested density; what is actually achieved is reported, never assumed.
    """
    per_node = max(1, round(density * n_tech))
    installed = np.zeros((n_node, n_tech), dtype=bool)
    for i in range(n_node):
        installed[i, rng.choice(n_tech, size=per_node, replace=False)] = True
    return installed


def _nodal_data(shape: Shape, dest: Path) -> dict[str, str]:
    """Parquet for one rung of the ``nodal`` ladder.

    Every node meets its own demand from its own portfolio: the model has no
    transmission, so feasibility must not depend on the draw. Only installed
    pairs are written — the tidy table *is* the sparsity this case measures.
    """
    rng = _seed(shape)
    n_snap, n_node = shape.sizes['snapshot'], shape.sizes['node']
    techs = list(TECHNOLOGIES[: shape.sizes['tech']])
    nodes = [f'n{i:04d}' for i in range(n_node)]

    installed = _portfolios(rng, n_node, len(techs), shape.density)
    capacity = installed * rng.uniform(200.0, 800.0, (n_node, len(techs)))
    cost = rng.uniform(10.0, 100.0, len(techs))

    at_node = capacity.sum(axis=1)
    demand = at_node[None, :] * 0.5 * (0.8 + 0.4 * rng.random((n_snap, n_node)))

    return _dump(
        {
            'installed': _installed_frame(nodes, techs, installed, capacity),
            'cost': pd.DataFrame({'tech': techs, 'value': cost}),
            'demand': pd.DataFrame(
                {
                    'snapshot': np.repeat(np.arange(n_snap), n_node),
                    'node': nodes * n_snap,
                    'value': demand.reshape(-1),
                }
            ),
            'node': pd.DataFrame({'node': nodes}),
            'tech': pd.DataFrame({'tech': techs}),
            'snapshot': pd.DataFrame({'snapshot': np.arange(n_snap)}),
        },
        dest,
    )


def _nodal_eager(paths: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The same parquet files, as the linopy lane wants them.

    The eager lane cannot hold an absent pair, so ``installed`` is reindexed
    over the full node x tech product: that is what turns structural sparsity
    into NaN padding, and doing it here rather than pretending otherwise is the
    point of the case.
    """
    import xarray as xr

    nodes = pd.read_parquet(paths['node'])['node']
    techs = pd.read_parquet(paths['tech'])['tech']
    cost = pd.read_parquet(paths['cost']).set_index('tech')['value']
    demand = pd.read_parquet(paths['demand'])
    installed = _dense_pairs(paths['installed'], nodes, techs)
    data = {
        'installed': installed,
        'cost': cost,
        'demand': xr.DataArray.from_series(demand.set_index(['snapshot', 'node'])['value']),
    }
    coords = {
        'snapshot': pd.Index(sorted(demand['snapshot'].unique()), name='snapshot'),
        'node': pd.Index(nodes, name='node'),
        'tech': pd.Index(techs, name='tech'),
    }
    return data, coords


# --------------------------------------------------------------------------
# sector


#: What each technology's output arrives as. One carrier per technology, which
#: is what makes the tech x carrier map sparser than the portfolio above it.
CARRIERS = ('electricity', 'heat', 'hydrogen', 'gas', 'transport')


def _sector_data(shape: Shape, dest: Path) -> dict[str, str]:
    """Parquet for one rung of the ``sector`` ladder.

    ``reachable`` is what a node can actually deliver into a carrier. Demand
    exists only where that is nonzero, which is what keeps the model feasible
    on any draw and what makes the demand table sparse in (node, carrier) while
    staying dense in time.
    """
    rng = _seed(shape)
    n_snap, n_node = shape.sizes['snapshot'], shape.sizes['node']
    techs = list(TECHNOLOGIES[: shape.sizes['tech']])
    carriers = list(CARRIERS[: shape.sizes['carrier']])
    nodes = [f'n{i:04d}' for i in range(n_node)]

    installed = _portfolios(rng, n_node, len(techs), shape.density)
    capacity = installed * rng.uniform(200.0, 800.0, (n_node, len(techs)))
    serves = rng.integers(0, len(carriers), len(techs))
    efficiency = rng.uniform(0.3, 0.95, len(techs))

    reachable = np.zeros((n_node, len(carriers)))
    for t, c in enumerate(serves):
        reachable[:, c] += capacity[:, t] * efficiency[t]
    served = reachable > 0
    demand = reachable[None, :, :] * 0.6 * (0.8 + 0.4 * rng.random((n_snap, n_node, len(carriers))))
    live = np.broadcast_to(served, demand.shape).reshape(-1)

    return _dump(
        {
            'installed': _installed_frame(nodes, techs, installed, capacity),
            'produces': pd.DataFrame({'tech': techs, 'carrier': [carriers[c] for c in serves], 'value': efficiency}),
            'cost': pd.DataFrame({'tech': techs, 'value': rng.uniform(10.0, 100.0, len(techs))}),
            'demand': pd.DataFrame(
                {
                    'snapshot': np.repeat(np.arange(n_snap), n_node * len(carriers))[live],
                    'node': np.tile(np.repeat(nodes, len(carriers)), n_snap)[live],
                    'carrier': np.array(carriers * (n_snap * n_node))[live],
                    'value': demand.reshape(-1)[live],
                }
            ),
            'node': pd.DataFrame({'node': nodes}),
            'tech': pd.DataFrame({'tech': techs}),
            'carrier': pd.DataFrame({'carrier': carriers}),
            'snapshot': pd.DataFrame({'snapshot': np.arange(n_snap)}),
        },
        dest,
    )


def _sector_eager(paths: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The same parquet files, as the linopy lane wants them.

    Both sparse tables are reindexed over their full product: the eager lane
    has nowhere else to put an absent pair, and doing it in the open is what
    makes the arms comparable.
    """
    import xarray as xr

    nodes = pd.read_parquet(paths['node'])['node']
    techs = pd.read_parquet(paths['tech'])['tech']
    carriers = pd.read_parquet(paths['carrier'])['carrier']
    demand = pd.read_parquet(paths['demand'])
    installed = _dense_pairs(paths['installed'], nodes, techs)
    produces = (
        pd.read_parquet(paths['produces'])
        .set_index(['tech', 'carrier'])['value']
        .unstack()
        .reindex(index=techs, columns=carriers)
        .fillna(0.0)
    )
    data = {
        'installed': installed,
        'produces': produces,
        'cost': pd.read_parquet(paths['cost']).set_index('tech')['value'],
        'demand': xr.DataArray.from_series(demand.set_index(['snapshot', 'node', 'carrier'])['value']),
    }
    coords = {
        'snapshot': pd.Index(sorted(demand['snapshot'].unique()), name='snapshot'),
        'node': pd.Index(nodes, name='node'),
        'tech': pd.Index(techs, name='tech'),
        'carrier': pd.Index(carriers, name='carrier'),
    }
    return data, coords


# --------------------------------------------------------------------------
# transport


def _transport_data(shape: Shape, dest: Path) -> dict[str, str]:
    """Parquet for one rung of the ``transport`` ladder.

    Generation is dealt round-robin so every bus has some, and the network is a
    ring plus chords, which makes ``from != to`` true by construction. Load is
    sized against what a bus can raise from its own generators alone, so the
    model is feasible whatever the line capacities do.
    """
    rng = _seed(shape)
    n_snap = shape.sizes['snapshot']
    n_gen, n_bus, n_line = shape.sizes['generator'], shape.sizes['bus'], shape.sizes['line']

    buses = [f'b{i:04d}' for i in range(n_bus)]
    gens = [f'g{i:05d}' for i in range(n_gen)]
    gen_bus = [buses[i % n_bus] for i in range(n_gen)]
    p_max = rng.uniform(50.0, 150.0, n_gen)

    lines = [f'l{i:05d}' for i in range(n_line)]
    frm = [buses[i % n_bus] for i in range(n_line)]
    to = [buses[(i % n_bus + 1 + i // n_bus) % n_bus] for i in range(n_line)]

    own = pd.Series(p_max, index=gen_bus).groupby(level=0).sum().reindex(buses).to_numpy()
    load = own[None, :] * 0.5 * (0.8 + 0.4 * rng.random((n_snap, n_bus)))
    snaps = np.repeat(np.arange(n_snap), n_bus)

    return _dump(
        {
            'p_max': pd.DataFrame({'generator': gens, 'value': p_max}),
            'cost': pd.DataFrame({'generator': gens, 'value': rng.uniform(10.0, 100.0, n_gen)}),
            'cap': pd.DataFrame({'line': lines, 'value': rng.uniform(20.0, 80.0, n_line)}),
            'neg_cap': pd.DataFrame({'line': lines, 'value': -rng.uniform(20.0, 80.0, n_line)}),
            'load': pd.DataFrame({'snapshot': snaps, 'bus': buses * n_snap, 'value': load.ravel()}),
            'generator': pd.DataFrame({'generator': gens, 'bus': gen_bus}),
            'line': pd.DataFrame({'line': lines, 'from': frm, 'to': to}),
            'bus': pd.DataFrame({'bus': buses}),
            'snapshot': pd.DataFrame({'snapshot': np.arange(n_snap)}),
        },
        dest,
    )


def _transport_eager(paths: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    import xarray as xr

    gens = pd.read_parquet(paths['generator'])
    lines = pd.read_parquet(paths['line'])
    load = pd.read_parquet(paths['load'])
    data = {
        'p_max': pd.read_parquet(paths['p_max']).set_index('generator')['value'],
        'cost': pd.read_parquet(paths['cost']).set_index('generator')['value'],
        'cap': pd.read_parquet(paths['cap']).set_index('line')['value'],
        'neg_cap': pd.read_parquet(paths['neg_cap']).set_index('line')['value'],
        'load': xr.DataArray.from_series(load.set_index(['snapshot', 'bus'])['value']),
    }
    coords = {
        'snapshot': pd.Index(sorted(load['snapshot'].unique()), name='snapshot'),
        'generator': gens,
        'bus': pd.Index(pd.read_parquet(paths['bus'])['bus'], name='bus'),
        'line': lines,
    }
    return data, coords


# --------------------------------------------------------------------------
# fleet — many declarations rather than one large one


#: The variables `fleet` declares. Named here rather than inline because the
#: model file and the eager arm both have to agree on them, and a mismatch
#: would show up as a parity failure rather than as the typo it is.
FLEET_VARIABLES = (
    'p', 'p_up', 'p_down', 'reserve_up', 'reserve_down', 'charge',
    'discharge', 'soc', 'spill', 'curtail', 'import_', 'export_',
)  # fmt: skip


def _fleet_data(shape: Shape, dest: Path) -> dict[str, str]:
    """Parquet for one rung of the ``fleet`` ladder — and of ``declarations``,
    whose generated models read the same three tables.

    Demand sits where the balance can be met three ways and all three are
    priced, so the optimum is a choice rather than "take the free one".
    """
    rng = _seed(shape)
    n_snap, n_unit = shape.sizes['snapshot'], shape.sizes['unit']
    units = [f'u{i:05d}' for i in range(n_unit)]

    p_max = rng.uniform(50.0, 150.0, n_unit)
    demand = p_max.sum() * 0.6 * (0.8 + 0.4 * rng.random(n_snap))

    return _dump(
        {
            'p_max': pd.DataFrame({'unit': units, 'value': p_max}),
            'cost': pd.DataFrame({'unit': units, 'value': rng.uniform(10.0, 100.0, n_unit)}),
            'demand': pd.DataFrame({'snapshot': np.arange(n_snap), 'value': demand}),
            'unit': pd.DataFrame({'unit': units}),
            'snapshot': pd.DataFrame({'snapshot': np.arange(n_snap)}),
        },
        dest,
    )


def _fleet_eager(paths: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    p_max = pd.read_parquet(paths['p_max']).set_index('unit')['value']
    cost = pd.read_parquet(paths['cost']).set_index('unit')['value']
    demand = pd.read_parquet(paths['demand']).set_index('snapshot')['value']
    data = {'p_max': p_max, 'cost': cost, 'demand': demand}
    coords = {
        'unit': pd.Index(p_max.index, name='unit'),
        'snapshot': pd.Index(demand.index, name='snapshot'),
    }
    return data, coords


# --------------------------------------------------------------------------
# declarations — fleet's question as a sweep


def _declarations_model(shape: Shape) -> str:
    """The ``declarations`` model at this rung's declaration count.

    ``fleet``'s mechanism with N as the swept axis: each declaration gets its
    own variable, its own capacity constraint and its own objective term, and
    one balance ties all of them to the load. Every declaration reads the same
    three parameters over the same (snapshot, unit) product, so the rungs of
    the sweep hold total variables and rows flat and differ *only* in how many
    declarations carry them.
    """
    names = [f'v{i:03d}' for i in range(shape.sizes['declaration'])]
    variables = '\n'.join(f'  {v}: {{foreach: [snapshot, unit], bounds: {{lower: 0, upper: p_max}}}}' for v in names)
    caps = '\n'.join(f'  cap_{v}: {{foreach: [snapshot, unit], expression: "{v} <= p_max"}}' for v in names)
    balance = ' + '.join(f'sum({v}, over=unit)' for v in names)
    objective = ' + '.join(f'sum(sum({v} * cost, over=unit), over=snapshot)' for v in names)
    return f"""# Generated by bench.cases._declarations_model — one file per rung of the
# declaration sweep. fleet's mechanism with the declaration count as the axis.
dimensions:
  snapshot:
    dtype: int
  unit:
    dtype: str

parameters:
  p_max:
    dims: [unit]
  cost:
    dims: [unit]
  demand:
    dims: [snapshot]

variables:
{variables}

constraints:
{caps}
  balance:
    foreach: [snapshot]
    expression: "{balance} == demand"

objective:
  sense: minimize
  expression: "{objective}"
"""


# --------------------------------------------------------------------------
# profiled — the one case whose input is the same order as the model


def _profiled_data(shape: Shape, dest: Path) -> dict[str, str]:
    """Parquet for one rung of the ``profiled`` ladder.

    The point of the case is an availability factor per (snapshot, node, tech),
    so ``availability`` has one row per variable rather than per coordinate of
    some smaller product. It is never zero: this case carries no mask, and a
    zero upper bound would be sparsity by the back door. Demand is half of what
    is available in *that* snapshot, so a draw is feasible without depending on
    a profile that happens to be high somewhere.

    Its label columns are categorical, unlike the other generators': the upper
    rungs run to millions of rows, where two object columns of repeated labels
    would cost more to build than the model does. The parquet is
    dictionary-encoded either way, so nothing about what the arms *read*
    changes.
    """
    rng = _seed(shape)
    n_snap, n_node = shape.sizes['snapshot'], shape.sizes['node']
    techs = list(TECHNOLOGIES[: shape.sizes['tech']])
    n_tech = len(techs)
    nodes = [f'n{i:04d}' for i in range(n_node)]

    capacity = rng.uniform(200.0, 800.0, (n_node, n_tech))
    availability = capacity[None, :, :] * (0.2 + 0.8 * rng.random((n_snap, n_node, n_tech)))
    demand = availability.sum(axis=2) * 0.5

    return _dump(
        {
            'availability': pd.DataFrame(
                {
                    'snapshot': np.repeat(np.arange(n_snap), n_node * n_tech),
                    'node': pd.Categorical.from_codes(
                        np.tile(np.repeat(np.arange(n_node), n_tech), n_snap), categories=pd.Index(nodes)
                    ),
                    'tech': pd.Categorical.from_codes(
                        np.tile(np.arange(n_tech), n_snap * n_node), categories=pd.Index(techs)
                    ),
                    'value': availability.reshape(-1),
                }
            ),
            'cost': pd.DataFrame({'tech': techs, 'value': rng.uniform(10.0, 100.0, n_tech)}),
            'demand': pd.DataFrame(
                {
                    'snapshot': np.repeat(np.arange(n_snap), n_node),
                    'node': nodes * n_snap,
                    'value': demand.reshape(-1),
                }
            ),
            'node': pd.DataFrame({'node': nodes}),
            'tech': pd.DataFrame({'tech': techs}),
            'snapshot': pd.DataFrame({'snapshot': np.arange(n_snap)}),
        },
        dest,
    )


def _profiled_eager(paths: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The same parquet files, as the linopy lane wants them.

    This is the eager lane at its best, and the case exists to let it be: the
    file already carries the layout xarray wants, so ``availability`` reads and
    reshapes rather than aligning. Routing it through ``from_series`` like the
    sparse cases would time a sort the *harness* imposed rather than one the
    shape requires, and would make the comparison dishonest in our favour.
    Order is load-bearing — ``_profiled_data`` writes C-order (snapshot, node,
    tech) — so a permuted file would change the objective and the parity gate
    would kill the run before anything is timed.
    """
    import xarray as xr

    nodes = pd.read_parquet(paths['node'])['node']
    techs = pd.read_parquet(paths['tech'])['tech']
    snapshots = pd.read_parquet(paths['snapshot'])['snapshot']
    demand = pd.read_parquet(paths['demand'])

    values = pd.read_parquet(paths['availability'])['value'].to_numpy()
    availability = xr.DataArray(
        values.reshape(len(snapshots), len(nodes), len(techs)),
        coords={'snapshot': snapshots.to_numpy(), 'node': nodes.to_numpy(), 'tech': techs.to_numpy()},
        dims=['snapshot', 'node', 'tech'],
    )

    data = {
        'availability': availability,
        'cost': pd.read_parquet(paths['cost']).set_index('tech')['value'],
        'demand': xr.DataArray.from_series(demand.set_index(['snapshot', 'node'])['value']),
    }
    coords = {
        'snapshot': pd.Index(snapshots, name='snapshot'),
        'node': pd.Index(nodes, name='node'),
        'tech': pd.Index(techs, name='tech'),
    }
    return data, coords


# --------------------------------------------------------------------------
# storage — the cyclic recurrence, the one shape that reaches sideways


def _storage_data(shape: Shape, dest: Path) -> dict[str, str]:
    """Parquet for one rung of the ``storage`` ladder.

    Load is ``dispatch``'s, for ``dispatch``'s reason: the generators alone
    serve every snapshot, so ``charge == discharge == soc == 0`` satisfies both
    constraints whatever the storage parameters say, and feasibility never
    depends on the half of the model this case exists to measure. The optimum
    uses storage anyway — generator costs spread over an order of magnitude, so
    arbitrage beats the round-trip loss, where a case whose optimum is
    ``soc == 0`` would build the same rows and prove nothing.
    """
    rng = _seed(shape)
    n_snap = shape.sizes['snapshot']
    n_gen, n_store = shape.sizes['generator'], shape.sizes['store']

    gens = [f'g{i:05d}' for i in range(n_gen)]
    stores = [f's{i:05d}' for i in range(n_store)]
    p_max = rng.uniform(50.0, 150.0, n_gen)
    load = p_max.sum() * 0.6 * (0.8 + 0.4 * rng.random(n_snap))
    p_store = rng.uniform(10.0, 40.0, n_store)

    return _dump(
        {
            'p_max': pd.DataFrame({'generator': gens, 'value': p_max}),
            'cost': pd.DataFrame({'generator': gens, 'value': rng.uniform(10.0, 100.0, n_gen)}),
            'load': pd.DataFrame({'snapshot': np.arange(n_snap), 'value': load}),
            'e_max': pd.DataFrame({'store': stores, 'value': p_store * 4.0}),
            'p_store': pd.DataFrame({'store': stores, 'value': p_store}),
            'eta': pd.DataFrame({'store': stores, 'value': rng.uniform(0.85, 0.95, n_store)}),
            'generator': pd.DataFrame({'generator': gens}),
            'store': pd.DataFrame({'store': stores}),
            'snapshot': pd.DataFrame({'snapshot': np.arange(n_snap)}),
        },
        dest,
    )


def _storage_eager(paths: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    p_max = pd.read_parquet(paths['p_max']).set_index('generator')['value']
    load = pd.read_parquet(paths['load']).set_index('snapshot')['value']
    e_max = pd.read_parquet(paths['e_max']).set_index('store')['value']
    data = {
        'p_max': p_max,
        'cost': pd.read_parquet(paths['cost']).set_index('generator')['value'],
        'load': load,
        'e_max': e_max,
        'p_store': pd.read_parquet(paths['p_store']).set_index('store')['value'],
        'eta': pd.read_parquet(paths['eta']).set_index('store')['value'],
    }
    coords = {
        'generator': pd.Index(p_max.index, name='generator'),
        'store': pd.Index(e_max.index, name='store'),
        'snapshot': pd.Index(load.index, name='snapshot'),
    }
    return data, coords


# --------------------------------------------------------------------------


def _ladder(
    sizes: dict[str, int],
    snapshots: Sequence[int],
    per_snapshot: int,
    density: float = 1.0,
) -> tuple[Shape, ...]:
    """A case's rungs, one per entry of *snapshots*.

    ``xs``..``l`` is the published ladder — the range the tables compare across
    cases. ``xl`` and ``2xl`` answer a different question: whether an engine
    that keeps the model resident holds together where one that spills would.
    ``2xl`` is the capability rung, where ``docs/benchmarks.md`` claims a model
    whose dense build cannot fit on the machine still streams out under the
    budget; a rung nothing else survives is the only way to keep testing that
    claim rather than restating it. Every case grows by the same two factors,
    so the top rungs stay comparable with each other rather than each case
    choosing its own idea of "large".
    """
    labels = ('xs', 's', 'm', 'l', 'xl', '2xl')
    return tuple(
        Shape(labels[i], {**sizes, 'snapshot': n}, n * per_snapshot, density)
        for i, n in enumerate(snapshots)
        if i < len(labels)
    )


def _declaration_sweep(pool: int, snapshots: int, counts: Sequence[int]) -> tuple[Shape, ...]:
    """One model size, several declaration counts — rungs named ``n002``/``n008``/…

    The pool of units splits into N declarations of pool/N units each, so total
    variables, rows and snapshots are flat across the sweep and a rung differs
    from its neighbour only in how many declarations carry them. Held at one
    size for ``_density_sweep``'s reason: sweeping both axes at once would
    leave no way to tell a declaration effect from a size effect.
    """
    for n in counts:
        if pool % n:
            raise ValueError(f'a pool of {pool} units does not split into {n} equal declarations')
    return tuple(
        Shape(f'n{n:03d}', {'declaration': n, 'unit': pool // n, 'snapshot': snapshots}, snapshots * pool)
        for n in counts
    )


def _density_sweep(
    sizes: dict[str, int], snapshots: int, per_snapshot: int, densities: Sequence[float]
) -> tuple[Shape, ...]:
    """One model size, several mask densities — rungs named ``d100``/``d30``/…

    Held at one size on purpose: sweeping both axes at once would leave no way
    to tell a density effect from a size effect.
    """
    return tuple(
        Shape(f'd{round(d * 100):02d}', {**sizes, 'snapshot': snapshots}, snapshots * per_snapshot, d)
        for d in densities
    )


CASES: dict[str, Case] = {
    'dispatch': Case(
        name='dispatch',
        model=MODELS / 'dispatch.yaml',
        ladder=_ladder({'generator': 100}, (100, 1_000, 10_000, 100_000, 400_000, 1_200_000), per_snapshot=100),
        write=_dispatch_data,
        eager_inputs=_dispatch_eager,
    ),
    'commitment': Case(
        name='commitment',
        model=MODELS / 'commitment.yaml',
        ladder=_ladder({'generator': 50}, (10, 100, 1_000, 10_000, 40_000, 120_000), per_snapshot=100),
        write=_commitment_data,
        eager_inputs=_commitment_eager,
    ),
    'fleet': Case(
        name='fleet',
        model=MODELS / 'fleet.yaml',
        ladder=_ladder({'unit': 50}, (20, 200, 2_000, 20_000, 80_000, 240_000), per_snapshot=600),
        write=_fleet_data,
        eager_inputs=_fleet_eager,
    ),
    'declarations': Case(
        name='declarations',
        ladder=_declaration_sweep(pool=512, snapshots=2_000, counts=(2, 8, 32, 128)),
        write=_fleet_data,
        eager_inputs=_fleet_eager,
        generate_model=_declarations_model,
    ),
    'nodal': Case(
        name='nodal',
        model=MODELS / 'nodal.yaml',
        ladder=(
            *_ladder(
                {'node': 50, 'tech': 12}, (20, 200, 2_000, 20_000, 80_000, 240_000), per_snapshot=600, density=0.25
            ),
            *_density_sweep({'node': 50, 'tech': 12}, 2_000, 600, (1.0, 0.5, 0.25, 0.083)),
        ),
        write=_nodal_data,
        eager_inputs=_nodal_eager,
    ),
    'sector': Case(
        name='sector',
        model=MODELS / 'sector.yaml',
        ladder=_ladder(
            {'node': 50, 'tech': 12, 'carrier': 5},
            (20, 200, 2_000, 20_000, 80_000, 240_000),
            per_snapshot=850,
            density=0.083,
        ),
        write=_sector_data,
        eager_inputs=_sector_eager,
    ),
    'profiled': Case(
        name='profiled',
        model=MODELS / 'profiled.yaml',
        ladder=_ladder({'node': 50, 'tech': 12}, (20, 200, 2_000, 20_000, 80_000, 240_000), per_snapshot=600),
        write=_profiled_data,
        eager_inputs=_profiled_eager,
    ),
    'transport': Case(
        name='transport',
        model=MODELS / 'transport.yaml',
        ladder=_ladder(
            {'generator': 100, 'bus': 20, 'line': 40}, (70, 700, 7_000, 70_000, 280_000, 840_000), per_snapshot=140
        ),
        write=_transport_data,
        eager_inputs=_transport_eager,
    ),
    'storage': Case(
        name='storage',
        model=MODELS / 'storage.yaml',
        ladder=_ladder(
            {'generator': 40, 'store': 20}, (100, 1_000, 10_000, 100_000, 400_000, 1_200_000), per_snapshot=100
        ),
        write=_storage_data,
        eager_inputs=_storage_eager,
    ),
}
