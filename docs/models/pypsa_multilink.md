# PyPSA multi-link

One `Link`, one input bus, several output buses, each output derated by its
own efficiency — PyPSA's spelling for a CHP plant, an electrolyser with waste
heat, any conversion with more than one product.

> **✔ Verified against pypsa 1.2.4 (its own linopy 0.9.0)** — objective **1100**, matched to `rtol=1e-09`.

**Beside the ladder, because the feature is the schema.** Every rung varies
what a model *says*; a multi-link varies what a table *is*. PyPSA holds the
relation wide — `bus0`, `bus1`, `bus2`, `efficiency`, `efficiency2`, an empty
`bus2` where a link has only two ends — so every arity the data reaches adds a
column pair to the component itself. Here the relation is an axis: a
`terminal` is one end of one link, `link_of` and `bus_of` are its legs, and a
signed `coefficient` carries its share of the link's draw — `-1` on the input,
`+efficiency` on each output. Arity is the number of rows that name the link,
so the three-ended CHP and the two-ended boiler sit in the same four columns,
and a four-ended link would change nothing but the data.

One decision per link survives the tidying: `p`, what the link draws at its
input — PyPSA's `p0`. The balance walks it out to every terminal with
`at(p, onto=terminal, by=link_of)`, scales each end by its coefficient, and
lands it on that end's bus. Three terminals or two, the expression never says.

The instance is a toy gas-to-energy system: a CHP (gas → 0.4 elec + 0.4 heat,
capacity 50), a boiler (gas → 0.8 heat), an OCGT (gas → 0.5 elec), gas at 10.
The marginal heat unit comes from the boiler and the marginal electric unit
from the OCGT, so the prices are `10/0.8 = 12.5` and `10/0.5 = 20` — and one
unit of gas through the CHP earns `0.4·20 + 0.4·12.5 = 13` against 10, so the
CHP runs at its cap of 50 and the others top up: flows `(50, 20, 40)`, gas
110, objective **1100**.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{B}$ | index $b$ --- `bus` |
| $\mathcal{G}$ | index $g$ --- `generator` with $\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}$ |
| $\mathcal{L}$ | index $l$ --- `link` |
| $\mathcal{T}$ | index $t$ --- `terminal` with $\mathrm{link\_of}: \mathcal{T} \to \mathcal{L},\enspace \mathrm{bus\_of}: \mathcal{T} \to \mathcal{B}$ |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\mathit{gen}^{\mathrm{p,nom}}$ | `gen_p_nom` over $\mathcal{G}$ |
| $\mathit{marginal\_cost}$ | `marginal_cost` over $\mathcal{G}$ |
| $p^{\mathrm{nom}}$ | `p_nom` over $\mathcal{L}$ |
| $\mathit{coefficient}$ | `coefficient` over $\mathcal{T}$ |
| $\mathit{load}$ | `load` over $\mathcal{B}$ |

#### Variables

| Symbol | Meaning |
|---|---|
| $\mathit{gen}$ | `gen` over $\mathcal{G}$ |
| $p$ | `p` over $\mathcal{L}$ |

#### Objective

$$\min \sum_{g \in \mathcal{G}} \mathit{gen}_{g} \cdot \mathit{marginal\_cost}_{g}$$

#### Subject to

**`nodal_balance`**

$$\sum_{g \in \mathcal{G} \thinspace:\thinspace \mathrm{gen\_bus}(g) = b} \mathit{gen}_{g} + \sum_{t \in \mathcal{T} \thinspace:\thinspace \mathrm{bus\_of}(t) = b} p_{\mathrm{link\_of}(t)} \cdot \mathit{coefficient}_{t} = \mathit{load}_{b} \qquad \forall\thinspace b \in \mathcal{B}$$

#### Variable domains

**`gen`**

$$0 \le \mathit{gen}_{g} \le \mathit{gen}^{\mathrm{p,nom}}_{g} \qquad \forall\thinspace g \in \mathcal{G}$$

**`p`**

$$0 \le p_{l} \le p^{\mathrm{nom}}_{l} \qquad \forall\thinspace l \in \mathcal{L}$$

</details>
<!-- math:end -->

The tabs start from [the instance’s tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    # PyPSA multi-link: one Link, one input bus, several output buses, each output
    # derated by its own efficiency. PyPSA spells the relation wide — bus0, bus1,
    # bus2, efficiency, efficiency2, an empty bus2 where a link has no third
    # terminal — and grows a column pair per arity. Here the relation is an axis:
    # a `terminal` is one end of one link, its legs are lookups, and arity is the
    # number of rows that name the link. Optimum 1100.0, from PyPSA itself.

    dimensions:
      bus:
        dtype: str
      generator:
        dtype: str
      link:
        dtype: str
      terminal:
        dtype: str

    lookups:
      gen_bus: {over: generator, into: bus}  # every generator sits on a bus
      link_of: {over: terminal, into: link}  # a terminal is one end of one link
      bus_of: {over: terminal, into: bus}  # the bus that end touches

    parameters:
      gen_p_nom:
        dims: [generator]
      marginal_cost:
        dims: [generator]
      # The Link's own p_nom: a cap on what it draws at its input, p0 in PyPSA.
      p_nom:
        dims: [link]
      # The terminal's share of the link's draw: -1 on the input terminal, and
      # +efficiency on each output — PyPSA's efficiency, efficiency2, … columns,
      # tidied into one weighted incidence.
      coefficient:
        dims: [terminal]
      load:
        dims: [bus]

    variables:
      gen:
        foreach: [generator]
        bounds:
          lower: 0
          upper: gen_p_nom
      # The one decision per link, PyPSA's p: what it draws at its input terminal.
      # Every other terminal's flow is that draw scaled by its coefficient, so it
      # needs no variable of its own.
      p:
        foreach: [link]
        bounds:
          lower: 0
          upper: p_nom

    constraints:
      # `at` walks p out to the link's terminals, the coefficient scales each end,
      # and the group lands it on that end's bus — three terminals or two, the
      # expression never says.
      nodal_balance:
        foreach: [bus]
        expression: >-
          sum(gen, over=generator, group_by=gen_bus)
          + sum(at(p, onto=terminal, by=link_of) * coefficient, over=terminal, group_by=bus_of)
          == load

    objective:
      sense: minimize
      expression: gen * marginal_cost
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/ports/pypsa_multilink.yaml', sources) as solution:
        solution.objective  # 1100.0
        solution.dual('nodal_balance')
    ```

=== "PyPSA"

    The model-building half of `examples/ports/references/pypsa/pypsa_multilink.py`:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
        """The port's tables as a PyPSA network.

        ``tables`` is the same mapping the lpspec call binds as ``sources``; only
        the terminal table changes shape on the way in, pivoted from one row per
        link end into PyPSA's one row per link. The input terminal is the one with
        the negative coefficient — PyPSA fixes its share at -1, so the pivot
        asserts it: a different input share is sayable in rows and not in these
        columns. Each output terminal becomes the link's next port, its
        coefficient the port's efficiency, in the terminal table's row order. A
        link narrower than the instance's widest is padded with ``''`` — PyPSA's
        spelling for a port a link does not have — and a filler efficiency of 1.0
        that no equation reads.
        """
        terminals = tables['terminal'].merge(tables['coefficient'], on='terminal')
        inputs = terminals[terminals['value'] < 0].set_index('link_of')
        assert (inputs['value'] == -1.0).all(), 'PyPSA fixes the input share of a Link at -1; this instance must too'

        outputs = terminals[terminals['value'] > 0].copy()
        outputs['port'] = outputs.groupby('link_of', sort=False).cumcount() + 1
        buses = outputs.pivot(index='link_of', columns='port', values='bus_of')
        efficiencies = outputs.pivot(index='link_of', columns='port', values='value')

        links = pd.DataFrame(index=pd.Index(tables['link']['link'], name='link'))
        links['bus0'] = inputs['bus_of']
        for port in buses.columns:
            links[f'bus{port}'] = buses[port].reindex(links.index).fillna('')
            links['efficiency' if port == 1 else f'efficiency{port}'] = efficiencies[port].reindex(links.index).fillna(1.0)

        n = pypsa.Network()
        n.add('Bus', tables['bus']['bus'])

        generators: pd.DataFrame = tables['generator'].set_index('generator')
        n.add(
            'Generator',
            generators.index,
            bus=generators['gen_bus'],
            p_nom=tables['gen_p_nom'].set_index('generator')['value'],
            marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
        )
        n.add('Link', links.index, p_nom=tables['p_nom'].set_index('link')['value'], **dict(links.items()))

        load: pd.Series = tables['load'].set_index('bus')['value']
        for bus in tables['bus']['bus']:
            n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
        return n
    ```

**The pivot is the argument.** The PyPSA tab spends its first half turning
rows into columns — finding the input, numbering the outputs, padding the
narrow links with `''` and a filler efficiency no equation reads — before a
single component exists. That reshape is not this port being awkward: it is
what the wide schema demands of any tidy source, and it runs in reverse for
any reader whose data starts wide. The lpspec tab binds the terminal table as
it stands.

**What the columns cannot say.** PyPSA fixes the input's share at `-1`, so the
pivot asserts it. In rows that constant is just data — an input coefficient of
`-1.05` would model a link burning 5% of its draw in station load, with no new
column and no new construct. Nothing in this instance uses that; the point is
where the wall sits.
