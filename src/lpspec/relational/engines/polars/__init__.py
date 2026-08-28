"""The polars engine: plan → frames → `sinks.ModelTables`.

Everything here is engine-private. The contract is either side of it —
`math_spec.program` going in, `relational/sinks/tables.py` coming out — and
nothing outside this package may reach past those two.

Split out when the question of a second engine was priced (the duckdb lane
this one replaced — `docs/about/benchmarks.md`): with one engine the boundary
between *what a model is* and *how it is built* was real but invisible, and a
reader had to know which of the eleven modules under `relational/` were which.
"""
