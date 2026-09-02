"""**Internal:** relational LP construction — the streaming lane.

The public interface of the package is YAML (see ``lpspec.api``); constructing
programs in Python is not supported API.

**Two layers, and the directory says which is which.** ``sinks/``,
``status.py`` and ``result.py`` are the contract: what an engine answers to and
what a sink reads. What a model *is* is upstream of both — ``math_spec.program``
— which is why no module here declares it. ``engines/`` holds implementations
of that contract, one per directory. A **solver's** own package is imported
inside the function that calls it, so one a caller has not installed never
reaches their import path.

Nothing is re-exported here. Every consumer imports from the module that owns
the name, so the import site says which layer the caller is reaching into, and
no contract module has to name an implementation.
"""
