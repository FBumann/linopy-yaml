# Preparing the data

Every model page's tabs start from the same place: **the instance's tables,
one frame per parameter** — the mapping the lpspec call binds as `sources` and
the reference scripts take as `tables`. Nobody's data is born in that shape.
It is born in files, and this page is the one place the journey from files to
that mapping is spelled out, so the model pages can start where it ends.

## The files

Real instances arrive as entity tables — attributes side by side, the shape a
PyPSA-style CSV folder holds — and tidy time series. The committed instance
for [dispatch](dispatch.md), in exactly that shape:

`examples/ports/data/dispatch/generators.csv`

```csv
generator,p_max,cost
wind,80.0,10.0
solar,0.0,25.0
gas,200.0,50.0
```

`examples/ports/data/dispatch/load.csv`

```csv
snapshot,value
0,60.0
1,120.0
2,180.0
3,90.0
```

## To the tables

One frame per parameter means splitting the entity table's columns out — one
`select` per parameter, and the time series passes through untouched:

```python
import polars as pl

generators = pl.read_csv('examples/ports/data/dispatch/generators.csv')

load = pl.read_csv('examples/ports/data/dispatch/load.csv')

sources = {
    'snapshot': load.select('snapshot').unique(maintain_order=True),
    'p_max': generators.select('generator', pl.col('p_max').alias('value')),
    'cost': generators.select('generator', pl.col('cost').alias('value')),
    'load': load,
}
```

That `sources` is the shared starting point of every tab in this gallery. With
data curated one parquet file per parameter the frames disappear entirely —
`sources = {'p_max': 'p_max.parquet', ...}` — and the engine scans the files
itself.

## What each framework still needs on top

- **lpspec** — nothing. Tidy tables per parameter are its native shape, so the
  `select` calls above are the whole cost of arriving from an entity table,
  and the model pages' calls are three lines.
- **linopy** — coordinate-carrying pandas. The `set_index` lines opening every
  `build()` in the linopy tabs turn each table into an indexed Series, and
  model-shaped data — a dense cost matrix, an incidence matrix — is built by
  hand where the formulation demands it (see
  [transport](transport.md) and [Stigler's diet](stigler_diet.md)).
- **PyPSA** — entity tables, which is to say: the files themselves. Static
  attributes go in side by side (`n.add('Generator', names, bus=…, p_nom=…)`),
  a whole folder in the shape above imports in one call
  (`import_from_csv_folder`), and only time series diverge — PyPSA wants them
  wide, one column per component, which is the `pivot` in every PyPSA tab.

None of the three in-memory shapes is neutral ground — the files are. Each
tab shows its own framework's journey and no one else's, which is what makes
the side-by-sides comparable.

## Already holding another framework's shapes?

<details markdown="1">
<summary>From linopy's shapes — pass them as they are</summary>

An indexed pandas Series *is* a source: index levels bind to dims by name,
so there is nothing to convert. A `DataArray` is one `.to_series()` away —
lpspec reads tables and hands arrays back, never the other way. The
[dispatch](dispatch.md) instance, linopy-style:

```python
import pandas as pd

p_max = pd.Series({'wind': 80.0, 'solar': 0.0, 'gas': 200.0}).rename_axis('generator')
cost = pd.Series({'wind': 10.0, 'solar': 25.0, 'gas': 50.0}).rename_axis('generator')
load = pd.Series([60.0, 120.0, 180.0, 90.0]).rename_axis('snapshot')

sources = {'snapshot': load.index, 'p_max': p_max, 'cost': cost, 'load': load}
```

</details>

<details markdown="1">
<summary>From PyPSA's shapes — one rename, one stack</summary>

An entity column is an indexed Series already, so a static attribute over one
dimension passes with a rename of its index. The wide time series needs its
`stack()` back to tidy and a `reset_index()` after it — a parameter over two
dimensions arrives as a frame carrying both as columns, an index being a pandas
idea the frames underneath have no counterpart for. Here it is mapped from load
names onto buses on the way, the shape [transport](transport.md) binds:

```python
load = (
    n.loads_t.p_set.rename(columns=n.loads.bus)
    .rename_axis(index='snapshot', columns='bus')
    .stack()
    .rename('value')
    .reset_index()
)

sources = {
    'snapshot': load['snapshot'].unique(),
    'bus': load['bus'].unique(),
    'p_max': n.generators['p_nom'].rename_axis('generator'),
    'cost': n.generators['marginal_cost'].rename_axis('generator'),
    'gen_bus': n.generators['bus'].rename_axis('generator').reset_index(),
    'load': load,
}
```

`gen_bus` is the last of those and the one that is not a parameter: a
[lookup](https://energy-models.github.io/math-spec/reference/language/dimensions#lookups) arrives under its own name
as the relation it is, so PyPSA's `bus` column is passed across as it stands
rather than merged into an index.

</details>

---

Back to [all models](index.md)
