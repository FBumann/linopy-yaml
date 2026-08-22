# Travelling salesman — MTZ

Visit every city once and come home, as cheaply as possible. The most famous problem in combinatorial optimisation, and the one most often assumed to be out of reach here.

> **✔ Verified against TSPLIB's published optimum** — **2085**, matched to `rtol=1e-09`. Instance `gr17` (Groetschel, 17 cities, explicit distance matrix).

**It is not out of reach.** This page exists because "we can't do TSP" was
plausible enough to be worth testing rather than asserting, and it turned out
to be wrong.

## What genuinely is refused, and why

TSP's textbook formulation — Dantzig–Fulkerson–Johnson — forbids subtours with
one constraint per subset of cities:

$$\sum_{i \in S}\sum_{j \in S} x_{ij} \le |S| - 1 \qquad \text{for every subset } S$$

Every row is linear and there are finitely many, so DFJ is an ordinary MILP.
The question is which parts of it this language can say, and the answer is
narrower than "none":

| | Status |
|---|---|
| DFJ, subsets **written out** | **sayable** — the subsets go in as data, exactly as [KVL's cycle basis](pypsa_kvl.md) does. Verified on an 8-city instance: 246 subsets, one correct tour |
| DFJ, subsets **generated lazily** | **outside** — solve, find violations, add rows, re-solve is an *algorithm*, not a model. Nothing declarative describes it |
| MTZ | sayable, and what this port uses |

So the honest refusal is only the second row. The first is not a language limit
at all: it is 2ⁿ rows, which stops being practical somewhere around twenty
cities — a data-size wall, not a ceiling. That distinction is worth being
precise about: a data-dependent *row count* is not itself a refusal — the
[cycle basis](pypsa_kvl.md) has one and is ordinary — so what rules DFJ out at
scale is the size of the data, not the shape of the language.

**Lazy generation is the thing every serious TSP code actually does**, which is
why "lpspec can express TSP" and "lpspec is a good way to solve a large TSP"
are different sentences, and only the first is true.

## What that leaves

Miller–Tucker–Zemlin, the polynomial alternative: give each city a position in
the tour and require that an arc `i → j` puts `j` later than `i`. O(n²) rows,
every one of them known before the data is read — static, relational, degree 1.
Inside the language, and it always was.

## The model

<!-- math:begin -->
<details markdown="1">
<summary>The same model, as math</summary>

The travelling salesman problem in the Miller-Tucker-Zemlin formulation: visit every city once and come home as cheaply as possible. TSPLIB instance gr17 — 17 cities, explicit distance matrix, published optimum 2085.

#### Sets

| Symbol | Meaning |
|---|---|
| $\mathcal{C}$ | index $c$ — `city` with $\mathrm{as\_from}: \mathcal{C} \to \mathcal{F},\enspace \mathrm{as\_to}: \mathcal{C} \to \mathcal{T}$ — the cities of the tour, each also read as an arc endpoint |
| $\mathcal{F}$ | index $f$ — `from_city` — the city an arc leaves |
| $\mathcal{T}$ | index $t$ — `to_city` — the city an arc arrives at |

#### Parameters

| Symbol | Meaning |
|---|---|
| $\mathit{distance}$ | `distance` over $\mathcal{F} \times \mathcal{T}$ — distance along an arc, with no row on the diagonal — a city has no distance to itself, so no arc variable exists there |
| $n$ | `n` (scalar) — the number of cities, which is the big-M the ordering rows need |

#### Variables

| Symbol | Meaning |
|---|---|
| $\mathit{travel}$ | `travel` over $\mathcal{F} \times \mathcal{T}$ — is this arc on the tour? |
| $u$ | `u` over $\mathcal{C}$ — position of a city in the tour — continuous, because the formulation needs only that the positions be orderable |

#### Objective

$$\min \sum_{f \in \mathcal{F},\enspace t \in \mathcal{T}} \mathit{travel}_{f,t} \cdot \mathit{distance}_{f,t}$$

#### Subject to

**`leave_each_city_once`**

$$\sum_{t \in \mathcal{T}} \mathit{travel}_{f,t} = 1 \qquad \forall\thinspace f \in \mathcal{F}$$

**`enter_each_city_once`**

$$\sum_{f \in \mathcal{F}} \mathit{travel}_{f,t} = 1 \qquad \forall\thinspace t \in \mathcal{T}$$

**`ordering`**

$$\sum_{c \in \mathcal{C} \thinspace:\thinspace \mathrm{as\_from}(c) = f} u_{c} - \left( \sum_{c \in \mathcal{C} \thinspace:\thinspace \mathrm{as\_to}(c) = t} u_{c} \right) + n \cdot \mathit{travel}_{f,t} \le n - 1 \qquad \forall\thinspace f \in \mathcal{F},\enspace t \in \mathcal{T} \thinspace:\thinspace f \neq \text{c01} \wedge t \neq \text{c01}$$

#### Variable domains

**`travel`**

$$\mathit{travel}_{f,t} \in \{0, 1\} \qquad \forall\thinspace f \in \mathcal{F},\enspace t \in \mathcal{T} \thinspace:\thinspace \mathit{distance}_{f,t} \text{ is defined}$$

**`u`**

$$1 \le u_{c} \le 17 \qquad \forall\thinspace c \in \mathcal{C}$$

</details>
<!-- math:end -->

```yaml
description: >-
  The travelling salesman problem in the Miller-Tucker-Zemlin formulation:
  visit every city once and come home as cheaply as possible. TSPLIB instance
  gr17 — 17 cities, explicit distance matrix, published optimum 2085.

dimensions:
  city:
    description: the cities of the tour, each also read as an arc endpoint
    dtype: str
  from_city:
    description: the city an arc leaves
    dtype: str
  to_city:
    description: the city an arc arrives at
    dtype: str

lookups:
  as_from: {over: city, into: from_city}
  as_to: {over: city, into: to_city}

parameters:
  distance:
    description: >-
      distance along an arc, with no row on the diagonal — a city has no
      distance to itself, so no arc variable exists there
    dims: [from_city, to_city]
  n:
    description: the number of cities, which is the big-M the ordering rows need
    dims: []

variables:
  travel:
    description: is this arc on the tour?
    foreach: [from_city, to_city]
    where: distance
    domain: binary
  u:
    description: >-
      position of a city in the tour — continuous, because the formulation
      needs only that the positions be orderable
    foreach: [city]
    bounds:
      lower: 1
      upper: 17

constraints:
  leave_each_city_once:
    foreach: [from_city]
    expression: sum(travel, over=to_city) == 1

  enter_each_city_once:
    foreach: [to_city]
    expression: sum(travel, over=from_city) == 1

  ordering:
    description: >-
      no subtours — if the tour goes from one city to another then the second
      is later in the numbering, and the big-M leaves the row saying nothing
      when it does not. Written for every ordered pair except those touching
      the depot, which anchors the numbering.
    foreach: [from_city, to_city]
    where: "from_city != c01 AND to_city != c01"
    expression: >-
      sum(u, by=as_from)
      - sum(u, by=as_to)
      + n * travel
      <= n - 1

objective:
  sense: minimize
  description: the length of the tour
  expression: sum(travel * distance)
```

**One shape here is worth the whole page.** MTZ needs `u` — the tour position —
at *both ends of the same row*: `u_i − u_j`. A variable indexed by one
dimension appearing twice under two different roles is exactly the kind of
self-join that looks like it should need a primitive.

It does not. Declare the identity map from `city` onto each end of the pair:

```yaml
lookups:
  as_from: {over: city, into: from_city}
  as_to: {over: city, into: to_city}
```

and `sum(u, by=as_from)` becomes a **relabel** rather than a
reduction — the map is one-to-one, so nothing is added up; `u` simply moves
from the `city` axis onto the `from_city` axis. Doing it twice with different
lookups puts the same variable at both ends of one row.

That is `sum(by=)` doing a job it was not designed for and handling it because
[topology is data](pypsa_transport.md): a lookup is a join, and a join
does not care whether it is many-to-one or one-to-one.

**The diagonal takes care of itself.** `distance` has no row where a city meets
itself, `travel`'s `where` is that parameter, and absence spreads — so every
row mentioning a self-arc simply is not built. No `i ≠ j` guard is written
anywhere, because [dimension-to-dimension comparison is not in the
language](https://math-spec.readthedocs.io/en/latest/reference/language/expressions/#where-strings) and here it is not needed.

## What it finds

A single tour of all 17 cities, closing at the start:

```
c01 → c16 → c12 → c09 → c05 → c02 → c10 → c11 → c03
    → c15 → c14 → c17 → c06 → c08 → c07 → c13 → c04 → c01
```

Length **2085**, TSPLIB's published optimum. One tour, not several — which is
the whole thing MTZ is there to guarantee, and worth checking on the primal
rather than trusting the objective, since a subtour-ridden solution would be
*cheaper*, not more expensive.

It solves in about two and a half seconds. MTZ's LP relaxation is famously
weak — that is the price of the formulation being small, and it is why nobody
solves large instances this way.

## What it exercises

`sum(by=)` as a relabel through a one-to-one lookup, a `where`
comparing a dimension against a string label, sparsity standing in for an
`i ≠ j` guard, and `binary` over a two-dimensional index.

No new construct. The honest summary is that the ceiling refuses an
*algorithm*, not a *problem* — and the corpus is a better place to find that
out than an argument.
