"""Engines: implementations of the contract one level up.

There is no ``Engine`` base class to subclass — the contract is the two types
that cross the boundary, ``plan.Program`` going in and ``sinks.ModelTables``
coming out, and an engine is whatever turns one into the other. One ships
(``polars``).

The package exists so the boundary is a directory rather than a convention:
``tests/test_architecture.py`` reads membership off the path, and the fence
names no engine — which is what keeps a second one from arriving as a special
case, and what makes everything above here answerable by any of them.
"""
