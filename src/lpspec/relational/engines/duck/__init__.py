"""The duckdb engine: plan → duckdb relations → `sinks.ModelTables`.

**The default engine**, and what `LPSPEC_ENGINE` unset selects. Not routed to:
there is no routing here, only a choice, and the two engines answer the same
YAML with the same numbers (`tests/test_engine_parity.py`). `LPSPEC_ENGINE=polars`
is the other one.

**On the committed ladder it wins nothing**: slower than the polars engine on
every rung and heavier on most. `bench/duckdb-spike.md` carries the measurement,
its method and its provenance, and is the only place the figures live — read its
§7 as the record of the *decision* rather than as the current cost, since it
prices the out-of-tree engine this one was ported from.

**Default anyway, and deliberately so.** Being behind is what makes it the
engine worth having under the instruments: as the default, CodSpeed, `bench.yml`
and the unflagged CI pass all measure it without being asked to, which is the
only way that gap closes or is shown not to.

Where the difference comes from, and the part that is a contract rather than an
engine: `cols` is positional, so its rows must leave in label order. On the
polars engine that order falls out of the build for free; here it is an
`ORDER BY` unless the label frame is a rectangle, which
:meth:`~lpspec.relational.engines.duck.executor.DuckExecutor._bounds_run`
is what exploits.

What the ladder cannot say is what happens above it: every rung fits in RAM, so
the argument this engine was built on — a model that does not — is untested
rather than refuted. What it *did* settle about a declared memory ceiling is in
`docs/ROADMAP.md`.

It **needs pyarrow**, which the polars engine does not: duckdb and polars hand
frames to each other through Arrow. It does *not* need pandas — pyarrow imports
pandas only when pandas is already installed, which is easy to mistake for a
requirement in a development environment. Both are runtime dependencies, since
this is the engine a bare install gets; `tests/test_api.py` pins the pandas
half and the narrower polars-engine claim.
"""

from lpspec.relational.engines.duck.compiler import DuckCompiler
from lpspec.relational.engines.duck.executor import DuckExecutor

__all__ = ['DuckCompiler', 'DuckExecutor']
