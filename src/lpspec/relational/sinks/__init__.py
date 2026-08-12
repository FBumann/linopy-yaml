"""Sinks: how a built model leaves the engine. See README.md.

**Two families.** A *solver* takes the tables and runs them (``solvers/``,
chosen by name); a *writer* renders them to a file (``writers/``, chosen by
suffix). They are directories rather than a convention, so
``tests/test_architecture.py`` reads membership off the path.

``tables.py`` is what both read, and neither family imports the other.
"""

from lpspec.relational.sinks.solvers import SOLVERS, Solver, loaded, solver
from lpspec.relational.sinks.tables import ModelTables
from lpspec.relational.sinks.writers import PLANNED_WRITERS, WRITERS, writer

__all__ = [
    'PLANNED_WRITERS',
    'SOLVERS',
    'WRITERS',
    'ModelTables',
    'Solver',
    'loaded',
    'solver',
    'writer',
]
