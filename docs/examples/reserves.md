# reserves

Energy and reserve co-optimization on a two-bus grid: offers are (generator,
market, tranche) triples, reserve zones overlap, and one line dangles. The
model exists to prove a claim — every many-to-many shape the language covers,
in one instance, each one load-bearing.

> **✔ Agrees with hand-written linopy 0.9.0** — objective **915**, matched to `rtol=1e-09`.

## The problem

A relation either **is an axis** — the pair set reified as a dimension whose
legs are lookups — or **is data** weighting one aggregation. Both appear here.
The offer set is the first kind, three-legged:

$$r_o \;\le\; \phi_{\mathrm{tranche\_of}(o)} \cdot \bar p_{\mathrm{gen\_of}(o)} \qquad \forall\, o$$

a per-offer cap assembled by pulling two other dimensions' parameters back
through two legs of one edge set. The zones are the second kind: membership
with a weight is a sparse parameter, and the zone requirement is the
contraction

$$\sum_{g} \sigma_{g,z} \cdot \Big( \sum_{o \,:\, \mathrm{gen\_of}(o) = g} r_o \Big) \;\ge\; \underline{R}_z \qquad \forall\, z$$

— multiply by the incidence table, sum the dimension away. A generator may
back several zones at different weights, which is exactly what no lookup can
say and no lookup needs to.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

Energy and reserve co-optimization on a two-bus grid: an offer is a generator, market and tranche together, reserve zones overlap, and one line dangles. The model exists to prove a claim — every many-to-many shape the language covers, in one instance, each one load-bearing.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{B}$ | index $b$ --- `bus` --- network nodes |
| $\mathcal{G}$ | index $g$ --- `generator` with $\mathrm{gen\_bus}: \mathcal{G} \to \mathcal{B}$ --- generating units, each sitting on one bus |
| $\mathcal{M}$ | index $m$ --- `market` --- reserve markets, each with a requirement to fill |
| $\mathcal{T}$ | index $t$ --- `tranche` --- how fast a reserve has to be deliverable |
| $\mathcal{Z}$ | index $z$ --- `zone` --- reserve zones, which overlap |
| $\mathcal{L}$ | index $l$ --- `line` with $\mathrm{from}: \mathcal{L} \to \mathcal{B},\enspace \mathrm{to}: \mathcal{L} \to \mathcal{B}$ --- transmission lines, which may have an open end |
| $\mathcal{O}$ | index $o$ --- `offer` with $\mathrm{gen\_of}: \mathcal{O} \to \mathcal{G},\enspace \mathrm{market\_of}: \mathcal{O} \to \mathcal{M},\enspace \mathrm{tranche\_of}: \mathcal{O} \to \mathcal{T}$ --- one generator's bid into one market at one tranche |

#### Parameters

| Symbol | Meaning |
|---|---|
| $p^{\mathrm{max}}$ | `p_max` over $\mathcal{G}$ --- installed capacity |
| $\mathit{energy\_cost}$ | `energy_cost` over $\mathcal{G}$ --- cost of one unit of output |
| $\mathit{load}$ | `load` over $\mathcal{B}$ --- demand at each bus |
| $\mathit{cap}$ | `cap` over $\mathcal{L}$ --- forward transmission limit |
| $\mathit{neg\_cap}$ | `neg_cap` over $\mathcal{L}$ --- reverse transmission limit |
| $\mathit{bus}^{\mathrm{cap}}$ | `bus_cap` over $\mathcal{B}$ --- most a bus may export over any one line |
| $\mathit{offer}^{\mathrm{cost}}$ | `offer_cost` over $\mathcal{O}$ --- cost of holding one unit of reserve on an offer |
| $\mathit{req}$ | `req` over $\mathcal{M}$ --- reserve a market has to be filled with |
| $\mathit{tranche}^{\mathrm{frac}}$ | `tranche_frac` over $\mathcal{T}$ --- share of capacity a generator may offer at a tranche |
| $\mathit{zone}^{\mathrm{share}}$ | `zone_share` over $\mathcal{G} \times \mathcal{Z}$ --- how much of a generator's reserve counts towards a zone — a generator may back several zones at a per-zone weight, so this cannot be a lookup over the generator, which is single-valued per label; rows are absent where a generator backs no part of a zone |
| $\mathit{zone}^{\mathrm{req}}$ | `zone_req` over $\mathcal{Z}$ --- reserve a zone has to be covered by |

#### Variables

| Symbol | Meaning |
|---|---|
| $p$ | `p` over $\mathcal{G}$ --- output of a generator |
| $f$ | `f` over $\mathcal{L}$ --- flow on a line, signed towards its `to` bus |
| $r$ | `r` over $\mathcal{O}$ --- reserve held against an offer |

#### Objective

$$\min \sum_{g \in \mathcal{G},\enspace o \in \mathcal{O}} \left( p_{g} \cdot \mathit{energy\_cost}_{g} + r_{o} \cdot \mathit{offer}^{\mathrm{cost}}_{o} \right)$$

#### Subject to

**`balance`**

$$\sum_{g \in \mathcal{G} \thinspace:\thinspace \mathrm{gen\_bus}(g) = b} p_{g} + \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{to}(l) = b} f_{l} - \left( \sum_{l \in \mathcal{L} \thinspace:\thinspace \mathrm{from}(l) = b} f_{l} \right) = \mathit{load}_{b} \qquad \forall\thinspace b \in \mathcal{B}$$

**`export_cap`**

$$f_{l} \le \mathit{bus}^{\mathrm{cap}}_{\mathrm{from}(l)} \qquad \forall\thinspace l \in \mathcal{L}$$

**`requirement`**

$$\sum_{o \in \mathcal{O} \thinspace:\thinspace \mathrm{market\_of}(o) = m} r_{o} \ge \mathit{req}_{m} \qquad \forall\thinspace m \in \mathcal{M}$$

**`headroom`**

$$p_{g} + \sum_{o \in \mathcal{O} \thinspace:\thinspace \mathrm{gen\_of}(o) = g} r_{o} \le p^{\mathrm{max}}_{g} \qquad \forall\thinspace g \in \mathcal{G}$$

**`offer_cap`**

$$r_{o} \le \mathit{tranche}^{\mathrm{frac}}_{\mathrm{tranche\_of}(o)} \cdot p^{\mathrm{max}}_{\mathrm{gen\_of}(o)} \qquad \forall\thinspace o \in \mathcal{O}$$

**`zone_cover`**

$$\sum_{g \in \mathcal{G}} \mathit{zone}^{\mathrm{share}}_{g,z} \cdot \left( \sum_{o \in \mathcal{O} \thinspace:\thinspace \mathrm{gen\_of}(o) = g} r_{o} \right) \ge \mathit{zone}^{\mathrm{req}}_{z} \qquad \forall\thinspace z \in \mathcal{Z}$$

#### Variable domains

**`p`**

$$p_{g} \ge 0 \qquad \forall\thinspace g \in \mathcal{G}$$

**`f`**

$$\mathit{neg\_cap}_{l} \le f_{l} \le \mathit{cap}_{l} \qquad \forall\thinspace l \in \mathcal{L}$$

**`r`**

$$r_{o} \ge 0 \qquad \forall\thinspace o \in \mathcal{O}$$

</details>
<!-- math:end -->

The tabs start from [the instance's tables](data.md) — one frame per parameter.

=== "lpspec"

    ```yaml
    description: >-
      Energy and reserve co-optimization on a two-bus grid: an offer is a
      generator, market and tranche together, reserve zones overlap, and one line
      dangles. The model exists to prove a claim — every many-to-many shape the
      language covers, in one instance, each one load-bearing.

    dimensions:
      bus:
        description: network nodes
        dtype: str
        values: [b1, b2]
      generator:
        description: generating units, each sitting on one bus
        dtype: str
      market:
        description: reserve markets, each with a requirement to fill
        dtype: str
        values: [m1, m2]
      tranche:
        description: how fast a reserve has to be deliverable
        dtype: str
        values: [fast, slow]
      zone:
        description: reserve zones, which overlap
        dtype: str
        values: [z1, z2]
      line:
        description: transmission lines, which may have an open end
        dtype: str
      offer:
        description: one generator's bid into one market at one tranche
        dtype: str

    lookups:
      gen_bus: {over: generator, into: bus, description: "the bus a generator sits on"}
      from: {over: line, into: bus, description: "the bus a line leaves, null where the end is open"}
      to: {over: line, into: bus, description: "the bus a line arrives at, null where the end is open"}
      gen_of: {over: offer, into: generator, description: "the generator behind an offer"}
      market_of: {over: offer, into: market, description: "the market an offer is made into"}
      tranche_of: {over: offer, into: tranche, description: "the tranche an offer is made at"}

    parameters:
      p_max:
        description: installed capacity
        dims: [generator]
      energy_cost:
        description: cost of one unit of output
        dims: [generator]
      load:
        description: demand at each bus
        dims: [bus]
      cap:
        description: forward transmission limit
        dims: [line]
      neg_cap:
        description: reverse transmission limit
        dims: [line]
      bus_cap:
        description: most a bus may export over any one line
        dims: [bus]
      offer_cost:
        description: cost of holding one unit of reserve on an offer
        dims: [offer]
      req:
        description: reserve a market has to be filled with
        dims: [market]
      tranche_frac:
        description: share of capacity a generator may offer at a tranche
        dims: [tranche]
      zone_share:
        description: >-
          how much of a generator's reserve counts towards a zone — a generator may
          back several zones at a per-zone weight, so this cannot be a lookup over
          the generator, which is single-valued per label; rows are absent where a
          generator backs no part of a zone
        dims: [generator, zone]
      zone_req:
        description: reserve a zone has to be covered by
        dims: [zone]

    variables:
      p:
        description: output of a generator
        foreach: [generator]
        bounds:
          lower: 0
      f:
        description: flow on a line, signed towards its `to` bus
        foreach: [line]
        bounds:
          lower: neg_cap
          upper: cap
      r:
        description: reserve held against an offer
        foreach: [offer]
        bounds:
          lower: 0

    expressions:
      reserve_of:
        expression: sum(r, by=gen_of)
        description: all the reserve a generator holds, across every offer it made

    constraints:
      balance:
        description: what is generated at a bus plus what arrives over the lines meets the load there
        foreach: [bus]
        expression: >-
          sum(p, by=gen_bus)
          + sum(f, by=to)
          - sum(f, by=from)
          == load
      export_cap:
        description: a line carries no more than the bus it leaves is allowed to export
        foreach: [line]
        expression: f <= at(bus_cap, by=from)
      requirement:
        description: the offers made into a market fill its requirement
        foreach: [market]
        expression: sum(r, by=market_of) >= req
      headroom:
        description: a generator's output plus the reserve it holds stays inside its capacity
        foreach: [generator]
        expression: p + reserve_of <= p_max
      offer_cap:
        description: >-
          an offer is capped by its tranche's share of its generator's capacity —
          two other dimensions' parameters pulled back through two legs of one edge
          set
        foreach: [offer]
        expression: r <= at(tranche_frac, by=tranche_of) * at(p_max, by=gen_of)
      zone_cover:
        description: the weighted reserve of the generators backing a zone covers its requirement
        foreach: [zone]
        expression: sum(zone_share * reserve_of, over=generator) >= zone_req

    objective:
      sense: minimize
      description: what the energy costs to generate, plus what the reserve costs to hold
      expression: p * energy_cost + r * offer_cost
    ```

    ```python
    # sources: parameter name -> frame or parquet path
    with lps.solve('examples/reserves.yaml', sources) as solution:
        solution.objective  # 915.0
        solution.dual('balance')
    ```

=== "linopy"

    The model-building half of `examples/ports/references/linopy/reserves.py`,
    which states every mapping as a dense incidence matrix (its `indicator`
    helper) and multiplies through by hand — the identical algebra with no
    lpspec construct anywhere near it:

    ```python
    def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
        """The instance's tables as a linopy model, row for row.

        ``tables`` is the same mapping the lpspec call binds as ``sources``.
        """
        series = {
            k: tables[k].set_index(tables[k].columns[0])['value']
            for k in (
                'p_max',
                'energy_cost',
                'load',
                'cap',
                'neg_cap',
                'bus_cap',
                'offer_cost',
                'req',
                'tranche_frac',
                'zone_req',
            )
        }
        buses = pd.Index(series['load'].index, name='bus')
        offers = tables['offer'].set_index('offer')
        zones = pd.Index(series['zone_req'].index, name='zone')

        gen_at = indicator(buses, tables['generator'], 'generator', 'gen_bus')
        line_in = indicator(buses, tables['line'], 'line', 'to')
        line_out = indicator(buses, tables['line'], 'line', 'from')
        offer_gen = indicator(pd.Index(series['p_max'].index, name='generator'), tables['offer'], 'offer', 'gen_of')
        offer_market = indicator(pd.Index(series['req'].index, name='market'), tables['offer'], 'offer', 'market_of')

        zone_at = pd.DataFrame(0.0, index=zones, columns=series['p_max'].index)
        for gen, zone, share in zip(
            tables['zone_share']['generator'], tables['zone_share']['zone'], tables['zone_share']['value'], strict=True
        ):
            zone_at.loc[zone, gen] = share
        zone_at.columns.name = 'generator'

        r_cap = offers['tranche_of'].map(series['tranche_frac']) * offers['gen_of'].map(series['p_max'])
        f_cap = tables['line'].set_index('line')['from'].map(series['bus_cap'])

        m = linopy.Model()
        p = m.add_variables(lower=0, coords=[series['p_max'].index], name='p')
        f = m.add_variables(lower=series['neg_cap'], upper=series['cap'], coords=[series['cap'].index], name='f')
        r = m.add_variables(lower=0, upper=r_cap, coords=[offers.index], name='r')

        m.add_constraints(
            (p * gen_at).sum('generator') + (f * line_in).sum('line') - (f * line_out).sum('line') == series['load'],
            name='balance',
        )
        m.add_constraints(f <= f_cap, name='export_cap')
        m.add_constraints((r * offer_market).sum('offer') >= series['req'], name='requirement')
        reserve_of = (r * offer_gen).sum('offer')
        m.add_constraints(p + reserve_of <= series['p_max'], name='headroom')
        m.add_constraints((reserve_of * xr.DataArray(zone_at)).sum('generator') >= series['zone_req'], name='zone_cover')
        m.add_objective((p * series['energy_cost']).sum() + (r * series['offer_cost']).sum())
        return m
    ```

## What it proves

Present is not proven, so each shape carries the one data mutation that must
move the optimum — held by `tests/test_reserves.py`, alongside the three-way
agreement (both lanes, the written LP file, and the incidence-matrix reference
above; the balance duals are checked too).

| Shape | Where | Idiom | Mutation that moves the optimum |
|---|---|---|---|
| self-relation, used in both directions | lines bus→bus, balance sums through `from` and `to` | edge dimension + leg lookups | — (the balance is every other row's feasibility) |
| parallel edges | `l1`, `l2` both b2→b1 | member identity is the label, not the endpoint pair | drop `l2` → dearer |
| dangling member | `l4`'s `to` is null | a partial lookup: the open end aggregates nowhere | point `l4` at b1 → cheaper |
| pullback through a leg | `f ≤ at(bus_cap, by=from)` | `at()` | uncap the exporting bus → cheaper |
| k-ary edge set | offers carry `gen_of`, `market_of`, `tranche_of` | three legs, one edge dimension | — (structure, pinned by test) |
| duplicate pair | `o1`, `o2` share all three legs | multiplicity is real capacity | drop `o2` → dearer |
| two pullbacks through two legs | the offer cap above | `at() * at()` | `o4` sits exactly at its cap |
| weighted n-to-n membership | `zone_share`, `g2` in both zones at 0.5 / 1.0 | incidence parameter, contracted | zero g2's z2 share → dearer |

Zone `z1` stays slack by design: it holds the *overlap* (g2 at weight 0.5)
while `z2` holds the *bindingness*, so the fractional weight is shown without
double-loading one constraint.

## The optimum, by hand

Energy: b2's cheap surplus exports over `l1` (pinned at 15 by `bus_cap`, not
its own 20) and `l2` (its own 8), so `g3` runs 40 local + 23 export = 63 and
`g1` covers the rest of b1, 47 — cost 785. Reserves: `m1`'s 55 takes both
parallel `g1` offers at their 25 caps (`o2` first at cost 1, then `o1` at 2)
plus 5 of `o3`, whose seat on `g2` is also what closes zone `z2` at exactly
25; `m2`'s 20 is `o4` at its tranche cap — cost 130. Total **915**, nodal
prices 10 at b1 and 5 at b2.
