# linopy_yaml — Design Specification

**Status:** Draft for discussion
**Audience:** Contributors and collaborators

-----

## Table of Contents

1. [Overview](#1-overview)
1. [Relationship to linopy](#2-relationship-to-linopy)
1. [YAML Schema Reference](#3-yaml-schema-reference)
1. [Data Loading Contract](#4-data-loading-contract)
1. [Expression Language](#5-expression-language)
1. [Where Strings](#6-where-strings)
1. [Built-in Helper Functions](#7-built-in-helper-functions)
1. [Error Handling Philosophy](#8-error-handling-philosophy)
1. [Python API](#9-python-api)
1. [Out of Scope](#10-out-of-scope)
1. [Open Questions](#11-open-questions)

-----

## 1. Overview

`linopy_yaml` is a thin layer on top of [linopy](https://github.com/PyPSA/linopy) that lets users define optimisation problems in YAML rather than Python. A YAML file declares dimensions, parameters, variables, constraints, and an objective. At runtime, the user supplies data (pandas, numpy, or xarray objects) and receives a fully built `linopy.Model` ready to solve.

The core value proposition is **transparency and shareability**. A YAML math definition is readable without knowing Python, can be version-controlled, diffed, and shared with collaborators who don't write optimisation code. It separates *what the problem is* from *how it is built*.

### What it looks like

YAML definition (`dispatch.yaml`):

```yaml
dimensions:
  snapshot:
    dtype: int
  generator:
    values: [wind, solar, gas]

parameters:
  p_max:
    dims: [generator]
    default: null
  load:
    dims: [snapshot]
    default: null
  cost:
    dims: [generator]
    default: null

variables:
  p:
    foreach: [snapshot, generator]
    where: "p_max > 0"
    bounds:
      lower: 0
      upper: p_max

constraints:
  power_balance:
    foreach: [snapshot]
    equations:
      - expression: sum(p, over=generator) == load

objectives:
  total_cost:
    sense: minimize
    equations:
      - expression: sum(p * cost, over=generator)
```

Python call site:

```python
import linopy_yaml
import pandas as pd

m = linopy_yaml.Model.from_yaml(
    "dispatch.yaml",
    data={
        "p_max": pd.Series({"wind": 100, "solar": 60, "gas": 200}),
        "load":  pd.Series([80, 120, 150, 180, 140, 100], name="snapshot"),
        "cost":  pd.Series({"wind": 0, "solar": 0, "gas": 50}),
    },
    coords={
        "snapshot": pd.RangeIndex(6, name="snapshot"),
    },
)

m.solve()
print(m.solution["p"])
```

### Design principles

- **Explicit over inferred.** Dimension coordinates and parameter shapes are declared in the YAML. There is no guessing.
- **Fail early, fail clearly.** All validation happens at load time, before any linopy calls. Errors name the problem and say how to fix it.
- **Linopy-native output.** The result is a standard `linopy.Model`. Nothing is hidden or wrapped. Users can inspect variables, constraints, and the solution exactly as they would with hand-written linopy code.
- **No domain assumptions.** The package knows nothing about energy, transport, or any other domain. It is a general-purpose layer over linopy's API.

-----

## 2. Relationship to linopy

`linopy_yaml` is a **pure consumer of linopy's public API**. It calls `model.add_variables()`, `model.add_constraints()`, and `model.add_objective()` — nothing else. It does not monkey-patch, subclass internal classes, or depend on linopy internals.

The `linopy_yaml.Model` class subclasses `linopy.Model` to attach `from_yaml()` and `extend()` classmethods and to store the parsed math definition and parameter dataset as attributes. All existing linopy behaviour is inherited unchanged.

```
linopy_yaml.Model
    └── linopy.Model          (all solving, variable/constraint storage, etc.)
         └── from_yaml()      (new classmethod — builds the model from YAML + data)
         └── extend()         (new method — adds more math from another YAML file)
         └── .math            (new property — the parsed MathSchema)
         └── .dataset         (new property — the xr.Dataset of parameters)
```

### Why a separate package?

- Keeps linopy's core dependency footprint lean (`pyparsing`, `pydantic` are not needed there).
- Allows independent versioning and iteration without coupling to linopy's release cycle.
- Different stability contracts: linopy's public API is stable; the YAML schema will evolve.

### Dependency surface

| Dependency  | Used for                                                               |
|-------------|------------------------------------------------------------------------|
| `linopy`    | Model, Variable, LinearExpression, add_variables/constraints/objective |
| `xarray`    | DataArrays, Dataset, broadcasting, merge                               |
| `pandas`    | Index objects, Series/DataFrame coercion                               |
| `numpy`     | Array operations, NaN handling                                         |
| `pyparsing` | Expression and where-string parsing                                    |
| `pydantic`  | YAML schema validation                                                 |
| `pyyaml`    | YAML file loading                                                      |

-----

## 3. YAML Schema Reference

A `linopy_yaml` YAML file has five top-level keys. All are optional except that a useful model will have at least `dimensions`, `parameters`, `variables`, and either `constraints` or `objectives`.

```yaml
dimensions:   ...   # master coordinate definitions
parameters:   ...   # named input data with declared shapes
variables:    ...   # decision variables
constraints:  ...   # linear constraints
objectives:   ...   # objective function(s)
```

### 3.1 `dimensions`

Declares the master coordinate index for each dimension. Every dimension referenced anywhere in the YAML must be declared here.

```yaml
dimensions:
  snapshot:
    dtype: int          # optional: float | int | str | datetime. Default: str
    values: null        # optional: list of values, or omit and pass via coords= at load time

  generator:
    values: [wind, solar, gas]
```

**Fields:**

| Field    | Type         | Default | Description                                                                                              |
|----------|--------------|---------|----------------------------------------------------------------------------------------------------------|
| `dtype`  | str          | `str`   | Expected dtype of the index. Used for coercion and validation. One of `float`, `int`, `str`, `datetime`. |
| `values` | list or null | `null`  | Coordinate values. If null, values must be supplied via the `coords=` argument at load time.             |

**Rules:**

- If `values` is null and no `coords` entry is provided at load time, loading raises immediately.
- All dimension names used in `foreach`, `parameters.dims`, `where` strings, or helper function calls must appear in `dimensions`.

### 3.2 `parameters`

Declares all named input data the model expects. Every parameter referenced in variable bounds, constraint expressions, or where strings must be declared here.

```yaml
parameters:
  p_max:
    dims: [generator]         # required: list of declared dimension names
    default: .inf             # optional: scalar default if data not provided
    dtype: float              # optional: float | int | bool | str. Default: float

  efficiency:
    dims: []                  # empty list = scalar parameter
    default: null             # null = no default, data must be provided

  is_storage:
    dims: [generator]
    dtype: bool
    default: false

  load:
    dims: [snapshot]
    default: null
```

**Fields:**

| Field     | Type           | Default  | Description                                                                                               |
|-----------|----------------|----------|-----------------------------------------------------------------------------------------------------------|
| `dims`    | list[str]      | required | Dimensions this parameter is indexed over. Must all be declared in `dimensions`. Empty list means scalar. |
| `default` | scalar or null | `null`   | Scalar default value. If null, the parameter must be provided in `data=`. Supports `.inf` and `-.inf`.    |
| `dtype`   | str            | `float`  | Expected data type. Used for coercion after loading.                                                      |

**Rules:**

- A parameter with `default: null` that is missing from `data=` raises at load time.
- A parameter with a non-null `default` that is missing from `data=` is filled with the scalar default and broadcasts freely over all dimensions.
- Parameters cannot have dims that aren't in `dimensions`.

### 3.3 `variables`

Declares decision variables.

```yaml
variables:
  p:
    foreach: [snapshot, generator]   # required: dimensions to index over
    where: "p_max > 0"               # optional: boolean mask — only create variables where True
    bounds:
      lower: 0                       # number or parameter name. Default: 0
      upper: p_max                   # number or parameter name. Default: inf

  committed:
    foreach: [snapshot, generator]
    binary: true                     # optional: binary variable. Default: false

  unit_count:
    foreach: [generator]
    integer: true                    # optional: integer variable. Default: false
    bounds:
      lower: 0
      upper: 10
```

**Fields:**

| Field          | Type          | Default  | Description                                                                                                               |
|----------------|---------------|----------|---------------------------------------------------------------------------------------------------------------------------|
| `foreach`      | list[str]     | required | Dimension names to iterate over. Each combination is one variable.                                                        |
| `where`        | str or null   | `null`   | Where string (see [Section 6](#6-where-strings)). Variables are only created at coordinates where this evaluates to True. |
| `bounds.lower` | number or str | `0`      | Lower bound. Either a literal number or the name of a declared parameter.                                                 |
| `bounds.upper` | number or str | `inf`    | Upper bound. Either a literal number or the name of a declared parameter.                                                 |
| `binary`       | bool          | `false`  | If true, variable is binary (0/1). Bounds are ignored.                                                                    |
| `integer`      | bool          | `false`  | If true, variable is integer-valued.                                                                                      |

**Rules:**

- `binary` and `integer` cannot both be true.
- If `bounds.lower` or `bounds.upper` is a string, it must be the name of a declared parameter.
- A parameter used as a bound must be broadcastable onto the variable's `foreach` dimensions.

### 3.4 `constraints`

Declares linear constraints. Each constraint is a foreach loop with one or more equation expressions.

```yaml
constraints:
  power_balance:
    foreach: [snapshot]                 # required
    where: null                         # optional
    equations:
      - expression: sum(p, over=generator) == load

  ramp_up:
    foreach: [snapshot, generator]
    where: "snapshot > 0 AND ramp_max"
    equations:
      - expression: p - roll(p, snapshot=1) <= ramp_max

  storage_balance:
    foreach: [snapshot, storage]
    equations:
      - expression: soc == roll(soc, snapshot=1) * (1 - loss) + charge - discharge
        where: "snapshot > 0"           # per-equation where — narrows the foreach mask
      - expression: soc == soc_initial
        where: "snapshot == 0"
```

**Fields:**

| Field                     | Type        | Default  | Description                                                                                                                     |
|---------------------------|-------------|----------|---------------------------------------------------------------------------------------------------------------------------------|
| `foreach`                 | list[str]   | required | Dimensions to iterate over.                                                                                                     |
| `where`                   | str or null | `null`   | Mask applied to all equations in this constraint.                                                                                |
| `equations`               | list        | required | One or more equations. At least one required.                                                                                   |
| `equations[i].expression` | str         | required | The equation string (see [Section 5](#5-expression-language)). Must contain exactly one comparison operator (`<=`, `>=`, `==`). |
| `equations[i].where`      | str or null | `null`   | Additional mask for this equation only. ANDed with the constraint-level `where`.                                                |

**Rules:**

- Each equation produces one named constraint in the linopy model. If a constraint has multiple equations, they are named `constraint_name_0`, `constraint_name_1`, etc. If only one equation, it is named `constraint_name`.
- Each expression must contain exactly one of `<=`, `>=`, `==`. The operator separates LHS from RHS.
- The LHS must involve at least one decision variable (linopy cannot build a constraint that is purely parameter arithmetic).

### 3.5 `objectives`

Declares the objective function. Typically one, but multiple may be defined (only the last one added to the model takes effect unless using `extend()`).

```yaml
objectives:
  total_cost:
    sense: minimize             # minimize | maximize. Default: minimize
    equations:
      - expression: sum(p * cost, over=generator)
```

**Fields:**

| Field                     | Type | Default    | Description                                                                                    |
|---------------------------|------|------------|------------------------------------------------------------------------------------------------|
| `sense`                   | str  | `minimize` | Optimisation direction. One of `minimize`, `maximize`.                                         |
| `equations`               | list | required   | Currently only the first equation is used.                                                     |
| `equations[0].expression` | str  | required   | Arithmetic expression (no comparison operator). Must produce a scalar linopy LinearExpression. |

-----

## 4. Data Loading Contract

### 4.1 Design rationale: why explicit is better than inferred

When building constraints from a YAML like `p <= p_max`, the evaluator needs to know the shape of `p_max`. Without that knowledge, errors surface late — deep inside the expression evaluator with cryptic xarray or linopy messages — rather than early at load time with a clear message pointing to the parameter name and the fix.

We considered several approaches to acquiring this shape information:

**Option A — Infer dims from data only.** Accept any named pandas/xarray object and infer dims from its axes. Works for well-named DataFrames but fails silently for scalars, dicts, and unnamed arrays, and cannot catch missing parameters upfront.

**Option B — Infer dims from math context only.** A parameter that appears in `foreach: [snapshot, generator]` must be broadcastable onto those dims. But "broadcastable" is not "equal to" — a scalar, a 1-D `[generator]` array, and a 2-D `[snapshot, generator]` array are all valid. Math context gives an upper bound on dims, not the exact shape.

**Option C — Explicit declaration in YAML (chosen).** Each parameter declares its `dims` in the YAML. The data loader validates that provided data matches. This eliminates ambiguity, enables immediate and precise error messages, and makes the YAML self-documenting as a data contract.

The tradeoff is verbosity: users must declare every parameter they intend to pass. This is consistent with the overall design philosophy — the YAML is meant to be a complete, readable specification of the model, not just a math shorthand.

### 4.2 Master coordinates

Before any parameter is loaded, a master coordinate index is assembled for every dimension. Sources, in order of precedence:

1. `coords=` kwarg passed to `from_yaml()` — highest priority, overrides everything.
1. `values:` declared in the YAML under `dimensions.dim_name`.
1. If neither is present for a declared dimension, loading raises immediately.

```python
# Values in YAML
dimensions:
  generator:
    values: [wind, solar, gas]

# Values via coords= (overrides YAML values if both present)
m = Model.from_yaml("model.yaml", data={...}, coords={
    "snapshot": pd.date_range("2024-01-01", periods=24, freq="h"),
})
```

The master coordinates are a `dict[str, pd.Index]`. They are passed to linopy's `add_variables()` as the `coords=` argument and used for mask broadcasting.

### 4.3 Accepted input types per parameter

For a parameter declared with `dims: [dim1, dim2]`:

| Python type      | How it is coerced                                                   | Constraints                                                                                 |
|------------------|---------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `int` or `float` | Scalar `xr.DataArray`. Broadcasts freely over all dimensions.       | None.                                                                                       |
| `dict`           | `pd.Series` → `xr.DataArray`. Dict keys become coordinate values.   | Only for 1-D parameters. Dict keys must be a subset of the master coordinate for that dim.  |
| `pd.Series`      | `.to_xarray()`. Index name must match the declared dim.             | Only for 1-D parameters. Index values must be a subset of master coords.                    |
| `pd.DataFrame`   | `.stack()` → `.to_xarray()`. Index name → dim1, column name → dim2. | Only for 2-D parameters. Row/column values must be subsets of master coords.                |
| `xr.DataArray`   | Accepted directly. Dim names validated against declared dims.       | Dim names must be a subset of declared dims. Coord values must be subsets of master coords. |
| `np.ndarray`     | Requires explicit dim information — see below.                      | Must match declared shape exactly.                                                          |
| `list`           | Treated as `np.ndarray`.                                            | Same as ndarray.                                                                            |

**numpy arrays and lists** have no named axes, so the loader cannot determine which dimension each axis corresponds to. If a plain array is provided for a parameter with dims, it is accepted only if it is 0-D (scalar) or 1-D with length matching a single declared dim. Otherwise loading raises with a message asking the user to provide a named pandas or xarray object instead.

### 4.4 Validation rules

Validation happens in this order at load time. Each step fails immediately if its condition is not met.

**Step 1: Dimension coords**

- Every declared dimension has a value source (YAML or `coords=`).
- Error: `"Dimension '{name}' has no values. Declare them under 'dimensions.{name}.values' in the YAML or pass coords={{'{name}': [...]}} to from_yaml()."`

**Step 2: Parameter presence**

- Every parameter with `default: null` is present in `data=`.
- Error: `"Parameter '{name}' is required (no default declared) but was not provided in data."`

**Step 3: Dimension names in provided data**

- For xr.DataArray: all dim names must be a subset of the declared dims for that parameter.
- Error: `"Parameter '{name}' has unexpected dimensions {unexpected}. Declared dims: {declared}."`

**Step 4: Coordinate values**

- All coordinate values present in the provided data must exist in the master coordinate for that dimension. Values not in the master are not silently dropped — they raise.
- Error: `"Parameter '{name}' has values in dimension '{dim}' that are not in the master coordinate: {unknown}.\nMaster '{dim}' coords: {master}"`

**Step 5: Unknown data keys**

- Keys in `data=` that are not declared parameters produce a warning (not an error), since the user may be passing extra data for use in `extend()` later.
- Warning: `"The following data keys are not declared as parameters and will be ignored: {names}"`

### 4.5 What the loader does NOT validate

- Whether the data's values are sensible (no range checks, no NaN warnings).
- Whether a parameter is actually *used* in the math (declared but unused is fine).
- Whether coordinate values in the data are a *complete* cover of the master coordinate. Missing values become NaN in the DataArray, which propagates into the where mask via `.notnull()` checks. This is intentional — sparse data produces sparse variables and constraints.

-----

## 5. Expression Language

Expressions appear in:

- `constraints.equations[i].expression` — must contain exactly one comparison operator
- `objectives.equations[i].expression` — arithmetic only, no comparison
- `variables.bounds.lower` / `bounds.upper` — currently only a name or number; full arithmetic expressions here are a v2 consideration

### 5.1 Syntax

```
expression  ::= arithmetic
             |  arithmetic COMPARATOR arithmetic

arithmetic  ::= atom
             |  unary_op arithmetic
             |  arithmetic binary_op arithmetic
             |  function_call
             |  "(" arithmetic ")"

atom        ::= NUMBER | NAME

unary_op    ::= "+" | "-"
binary_op   ::= "+" | "-" | "*" | "/" | "**"
COMPARATOR  ::= "<=" | ">=" | "=="

function_call ::= NAME "(" arg_list ")"
arg_list      ::= pos_arg ("," pos_arg)* ("," kwarg)*
               |  kwarg ("," kwarg)*
               |  empty
pos_arg       ::= arithmetic
kwarg         ::= NAME "=" (arithmetic | NAME)

NAME   ::= [a-zA-Z][a-zA-Z0-9_]*
NUMBER ::= integer | float | "inf" | ".inf"
```

### 5.2 Operator precedence

Standard mathematical precedence, highest to lowest:

| Priority    | Operators         | Associativity |
|-------------|-------------------|---------------|
| 1 (highest) | `**`              | Right         |
| 2           | `*`, `/`          | Left          |
| 3           | `+`, `-` (binary) | Left          |
| 4 (lowest)  | `+`, `-` (unary)  | Right         |

Parentheses override precedence in the usual way.

### 5.3 Name resolution

When a `NAME` token is encountered during evaluation:

1. Check decision variables (the linopy Model's variable store). If found, return the `linopy.Variable`.
1. Check parameters (the `xr.Dataset`). If found, return the `xr.DataArray`.
1. If neither, raise `NameError` with the name and the lists of available variables and parameters.

This ordering means a variable named `p` shadows a parameter named `p`. In practice, variable and parameter names should not overlap.

### 5.4 Type behaviour of arithmetic

The result type of arithmetic follows linopy's operator overloading:

| LHS                       | Operator           | RHS                          | Result                    |
|---------------------------|--------------------|------------------------------|---------------------------|
| `xr.DataArray`            | `+`, `-`, `*`, `/` | `xr.DataArray`               | `xr.DataArray`            |
| `linopy.Variable`         | `+`, `-`           | `xr.DataArray` or `Variable` | `linopy.LinearExpression` |
| `linopy.Variable`         | `*`                | `xr.DataArray`               | `linopy.LinearExpression` |
| `linopy.LinearExpression` | `+`, `-`           | anything                     | `linopy.LinearExpression` |

Broadcasting follows xarray semantics (dimension-name-based, not shape-based). A scalar DataArray broadcasts freely over any dimension.

### 5.5 Comparison operators

A comparison produces a `(lhs, op, rhs)` tuple consumed by the builder to call `model.add_constraints(lhs, op, rhs)`. The mapping from YAML operators to linopy signs:

| YAML | linopy `sign` argument |
|------|------------------------|
| `==` | `"="`                  |
| `<=` | `"<="`                 |
| `>=` | `">="`                 |

The LHS must be or reduce to a `linopy.LinearExpression`. The RHS must be or reduce to a numeric `xr.DataArray` or scalar. If this is violated, linopy will raise — the expression parser does not pre-validate this.

### 5.6 Examples

```yaml
# Simple capacity constraint
expression: p <= p_max

# Efficiency (parameter * variable)
expression: p_out == p_in * efficiency

# Sum over a dimension
expression: sum(p, over=generator) == load

# Time-coupled (rolling window)
expression: soc == roll(soc, snapshot=1) + charge - discharge

# Arithmetic on both sides
expression: p_in - p_out * (1 - loss) == 0

# Nested arithmetic in function
expression: sum(p * cost, over=generator) == total_cost_var
```

-----

## 6. Where Strings

Where strings produce boolean `xr.DataArray` masks. A `True` value means "include this coordinate combination". They appear on variables (restricting which variables are created) and constraints (restricting which constraints are built).

### 6.1 Syntax

```
where_expr  ::= atom_where
             |  "NOT" where_expr
             |  where_expr "AND" where_expr
             |  where_expr "OR" where_expr
             |  "(" where_expr ")"

atom_where  ::= NAME                          # existence check
             |  NAME COMPARATOR value         # comparison
             |  "True" | "False"              # boolean literals

COMPARATOR  ::= "<=" | ">=" | "==" | "!=" | "<" | ">"
value       ::= NUMBER | NAME_OR_STRING
```

`AND`, `OR`, `NOT` are case-insensitive.

### 6.2 Semantics

**Plain name** (`"p_max"`): Evaluates to True wherever the parameter is defined (non-null) and finite. Equivalent to `p_max.notnull() & (p_max != inf) & (p_max != -inf)`. If the parameter does not exist in the dataset, evaluates to scalar False.

**Comparison** (`"p_max > 0"`): Evaluates the comparison element-wise. NaN values propagate as False.

**Boolean operators**: Standard boolean logic. `AND` has higher precedence than `OR`. `NOT` has highest precedence.

**Boolean literals**: `True` and `False` (case-insensitive). `True` is equivalent to no where string.

### 6.3 Interaction with foreach

The where mask is evaluated against the parameter dataset (which has the master coordinates) and then broadcast onto the `foreach` grid. If a mask dimension is not in `foreach`, it is reduced by `any()` over that dimension before broadcasting.

For variables, the mask is passed directly to linopy's `mask=` argument. For constraints, the mask restricts which constraint rows are built.

### 6.4 Examples

```yaml
# Only create variables where p_max is defined and positive
where: "p_max > 0"

# Only where both parameters are defined
where: "p_max AND ramp_max"

# Exclude a specific snapshot (e.g. for time-coupling constraints)
where: "snapshot > 0"

# Combine conditions
where: "p_max > 0 AND NOT is_must_run"

# Always include
where: null   # omit entirely, or write: "True"
```

-----

## 7. Built-in Helper Functions

Helper functions are called inside expressions to perform operations that cannot be expressed with arithmetic alone.

### 7.1 `sum(array, over=dim)`

Sums an array or expression over a dimension.

```yaml
expression: sum(p, over=generator) == load
expression: sum(p * cost, over=generator)   # arithmetic in positional arg
```

| Argument | Type                  | Description                |
|----------|-----------------------|----------------------------|
| `array`  | arithmetic expression | The expression to sum.     |
| `over`   | dimension name        | The dimension to sum over. |

If the array does not have the named dimension, it is returned unchanged (no error).

Works with both `xr.DataArray` and `linopy.Variable`/`LinearExpression` (calls `.sum(dim)` on the underlying object).

### 7.2 `roll(array, dim=n)`

Shifts an array along a dimension by `n` positions (wrapping). Used for time-coupling constraints where a variable at time `t` depends on its value at `t-1`.

```yaml
# soc at t depends on soc at t-1
expression: soc == roll(soc, snapshot=1) + charge - discharge
```

| Argument | Type             | Description                                      |
|----------|------------------|--------------------------------------------------|
| `array`  | component name   | The array or variable to shift.                  |
| `dim=n`  | keyword, integer | The dimension to roll over and the shift amount. |

The shift is applied with `roll_coords=False` (coordinates stay fixed; values wrap). For time-coupling constraints where the first snapshot should be handled separately, use a `where` string to exclude it:

```yaml
constraints:
  storage_balance:
    foreach: [snapshot, storage]
    equations:
      - expression: soc == roll(soc, snapshot=1) + charge - discharge
        where: "snapshot > 0"
      - expression: soc == soc_initial
        where: "snapshot == 0"
```

### 7.3 Custom helper functions

Register additional helpers at module load time using the `@linopy_yaml.register` decorator:

```python
import linopy_yaml

@linopy_yaml.register("weighted_sum")
def weighted_sum(array, weights, *, over):
    """sum(array * weights, over=dim)"""
    return (array * weights).sum(over)
```

Then use in YAML:

```yaml
expression: weighted_sum(p, duration, over=snapshot) <= energy_budget
```

**Rules for custom helpers:**

- The name must not conflict with built-in helpers.
- Positional arguments receive evaluated values (DataArrays or linopy expressions).
- Keyword arguments receive either evaluated values or bare strings (for dimension names).
- The function must return something that linopy can handle in the context it is used (DataArray for pure parameter operations, LinearExpression if it involves variables).

-----

## 8. Error Handling Philosophy

**Fail at load time, not at evaluation time.** Every error that can be detected before building the linopy model should be detected before building the linopy model. The worst errors are those that surface as opaque xarray or linopy exceptions with no indication of which YAML declaration caused them.

**Name the problem and say how to fix it.** Every error message includes:

1. What went wrong (the specific parameter, dimension, or expression).
1. What the user needs to do to fix it.
1. When helpful, what valid options look like.

### 8.1 Error message templates

**Missing dimension values:**

```
Dimension 'snapshot' has no values.
Declare them under 'dimensions.snapshot.values' in the YAML
or pass coords={'snapshot': [...]} to from_yaml().
```

**Missing required parameter:**

```
Parameter 'load' is required (no default declared) but was not provided in data.
Add 'load' to the data= argument, or declare a default under 'parameters.load.default'.
```

**Parameter with unexpected dims:**

```
Parameter 'p_max' has unexpected dimensions {'carrier'}.
Declared dims: ['generator'].
Either update the declaration or reshape your data.
```

**Parameter coordinate not in master:**

```
Parameter 'p_max' has values in dimension 'generator' that are not in the master coordinate: ['nuclear'].
Master 'generator' coords: ['wind', 'solar', 'gas']
```

**Undeclared dimension in foreach:**

```
Variable 'p' references undeclared dimension 'carrier'.
Declare it under 'dimensions:' in the YAML.
```

**Unknown name in expression:**

```
'p_charge' not found in expression 'p_charge - p_discharge'.
  Variables:  ['p', 'soc']
  Parameters: ['p_max', 'load', 'efficiency']
Check for typos, or ensure 'p_charge' is declared as a variable or parameter.
```

**Unknown helper function:**

```
Unknown helper function 'weighted_sum' in expression 'weighted_sum(p, over=generator)'.
Available built-ins: ['roll', 'sum']
Register custom helpers with @linopy_yaml.register('weighted_sum').
```

-----

## 9. Python API

### 9.1 `Model.from_yaml()`

```python
@classmethod
def from_yaml(
    cls,
    path: str | Path,
    *,
    data: dict[str, Any] | None = None,
    coords: dict[str, Any] | None = None,
) -> Model:
```

**Parameters:**

| Parameter | Type             | Description                                                                                                                                                          |
|-----------|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `path`    | `str` or `Path`  | Path to the YAML file.                                                                                                                                               |
| `data`    | `dict` or `None` | Parameter data. Keys are parameter names as declared in the YAML. See [Section 4.3](#43-accepted-input-types-per-parameter) for accepted value types.                |
| `coords`  | `dict` or `None` | Dimension coordinate values. Keys are dimension names. Values are anything accepted by `pd.Index()`. Overrides `values` declared in the YAML for the same dimension. |

**Returns:** A fully built `linopy_yaml.Model` (subclass of `linopy.Model`). The model has no solver attached; call `.solve()` as normal.

**Raises:** `ValueError` with descriptive message for any validation failure. `pydantic.ValidationError` if the YAML structure is invalid.

### 9.2 `Model.extend()`

Adds additional variables, constraints, and/or objectives from a second YAML file. Useful for building modular models from composable pieces.

```python
def extend(
    self,
    path: str | Path,
    *,
    data: dict[str, Any] | None = None,
) -> None:
```

The second YAML may reference dimensions and parameters already loaded in the model. New parameters introduced in the second YAML can be provided via `data=`. Existing parameters cannot be overridden.

### 9.3 `@linopy_yaml.register(name)`

Decorator to register a custom helper function. Must be called before `Model.from_yaml()`.

```python
import linopy_yaml

@linopy_yaml.register("weighted_sum")
def weighted_sum(array, weights, *, over):
    return (array * weights).sum(over)
```

### 9.4 `Model.math`

Property returning the parsed `MathSchema` object. Provides programmatic access to the YAML definition after loading.

```python
m.math.variables["p"].foreach   # ['snapshot', 'generator']
m.math.parameters["load"].dims  # ['snapshot']
```

### 9.5 `Model.dataset`

Property returning the `xr.Dataset` of all loaded parameters, after coercion and validation.

```python
m.dataset["p_max"]    # xr.DataArray indexed over generator
m.dataset["load"]     # xr.DataArray indexed over snapshot
```

-----

## 10. Out of Scope

**Time series processing.** Resampling, clustering, interpolation, and alignment of time series data are not handled by `linopy_yaml`. Users should preprocess their data before passing it in.

**Data loading from files.** The package does not read CSV, Parquet, NetCDF, or any other file formats. Users load their data into pandas/xarray objects using whatever tools they prefer, then pass those objects to `from_yaml()`.

**Solver configuration.** Solver selection, options, and result parsing are handled by linopy directly. `linopy_yaml` does not wrap `.solve()` or modify solver behaviour.

**Piecewise linear constraints, SOS constraints.** These are linopy features but are not exposed through the YAML interface in v1.

**Multiple objectives / multi-objective optimisation.** Only one objective is added to the linopy model. Defining multiple objectives in YAML is not an error, but only the last one takes effect.

**Schema migrations.** No tooling is provided for migrating YAML files between versions of the schema.

**LaTeX rendering.** A `Model.to_latex(constraint_name)` method is a planned v2 feature, but not committed for v1.

-----

## 11. Open Questions

**Q1: Package name.**
Current candidates: `linopy-math`, `linopy-yaml`, `linopy-declarative`. The name affects discoverability and signals intent. `linopy-math` is closest to Calliope's terminology (which inspired this design) and clearest about what the package adds.

**Q2: Sub-expressions.**
Calliope supports `sub_expressions` in constraints — named intermediate expressions that can be reused across equations. Excluded from v1 to keep the parser scope manageable, but worth considering for v2.

**Q3: Array slicing syntax.**
Calliope supports `p[generator_bus=bus]` — selecting a subset of an array along a dimension. This is needed for network models where a generator is "at" a bus. Probably needed before this package is useful for PyPSA-style models.

**Q4: `bounds` as full expressions.**
Currently `bounds.lower` and `bounds.upper` accept only a number or a parameter name. Supporting full arithmetic expressions (`lower: p_min * 0.9`) would make the schema more consistent. Low-risk addition, could be in v1.

**Q5: Where string dimension comparisons.**
`where: "snapshot > 0"` compares a dimension coordinate to a literal. This is useful but requires the where parser to know which names are dimensions (not parameters). Currently `snapshot` would not be found in the parameter dataset and would return False. Needs a dedicated code path for dimension-coordinate comparisons.

**Q6: Validation strictness.**
Should providing `data=` for a parameter not declared in the YAML be a warning or an error? Currently it's a warning. An error would be stricter (the YAML is the source of truth) but might be annoying in workflows where the same data dict is reused across models.
