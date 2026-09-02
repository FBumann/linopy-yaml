"""The polars engine: plan → frames → `sinks.Tables`.

Everything here is engine-private. The contract is either side of it —
`math_spec.program` going in, `relational/sinks/tables.py` coming out — and
nothing outside this package may reach past those two.
"""
