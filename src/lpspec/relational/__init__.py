"""**Internal:** relational LP construction — the logical plan and its engine.

The public interface of the package is YAML (see ``lpspec.api``).
Constructing Programs in Python is not supported API; a stable plan API may be
offered later.

This subpackage is the relational lane described in docs/ARCHITECTURE.md. It
must not import the eager builder — the typed AST (and, in phase 2, hand-built
plans) is the only contract with the rest of the package. Engine dependencies
(polars, highspy) are imported lazily so the core package stays lean.

**Two layers, and the directory says which is which.** ``plan.py``,
``sinks/``, ``status.py``, ``chunking.py``, ``result.py`` and ``frames.py`` are
the contract: what a model *is*, what an engine answers to, what a sink reads.
``engines/`` holds implementations of that contract, one per directory.

Nothing is re-exported here. Every consumer imports from the module that
owns the name — ``engines/polars/engine`` for the engine, ``result`` for what
a solve returns — so the import site says which layer the caller is reaching
into, and no contract module has to name an implementation.
"""
