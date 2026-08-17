"""Sinks: how a built model leaves the engine. See README.md.

**Two families.** A *solver* takes the tables and runs them (``solvers/``,
chosen by name); a *writer* renders them to a file (``writers/``, chosen by
suffix). They are directories rather than a convention, so
``tests/test_architecture.py`` reads membership off the path.

``tables.py`` is what both read, and neither family imports the other.
``capabilities.py`` is what both *declare*, and the three functions below are
where a caller's model meets those declarations — the only place the two
families are asked one question together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lpspec.errors import (
    LpspecError,
    sink_reformulates_message,
    sink_refuses_combination_message,
    sink_refuses_message,
    unknown_name_message,
)
from lpspec.relational.sinks import capabilities as caps
from lpspec.relational.sinks.solvers import SOLVERS, Solver, ingestible, loaded, solver
from lpspec.relational.sinks.tables import ModelTables
from lpspec.relational.sinks.writers import WRITERS, writer

if TYPE_CHECKING:
    from lpspec.relational import plan

__all__ = [
    'SOLVERS',
    'WRITERS',
    'ModelTables',
    'Solver',
    'ingestible',
    'loaded',
    'refusal',
    'relaxations',
    'sink_capabilities',
    'solver',
    'writer',
]


def sink_capabilities(name: str) -> caps.Capabilities:
    """What the sink called *name* can ingest — a solver name, or a suffix.

    **Answered without importing the solver**, a capability being a declared
    table rather than a probe: one environment can check a repository of models
    against every sink they will eventually be solved on.

    Raises:
        LpspecError: A name belonging to neither family.
    """
    if name in SOLVERS:
        return SOLVERS[name].capabilities
    if (suffix := name.lower()) in WRITERS:
        return WRITERS[suffix].capabilities
    raise LpspecError(unknown_name_message('sink', name, (*SOLVERS, *WRITERS)))


def _takes(program: plan.Program, name: str) -> bool:
    """Whether the sink called *name* would take *program* as it stands."""
    table = sink_capabilities(name)
    needed = caps.required(program)
    return not table.missing(needed) and table.excluded(needed) is None


def _takers(program: plan.Program, exclude: str) -> list[str]:
    """Every sink but *exclude* that would take *program* — a refusal's third clause.

    Asked without building a message, or it would reach back into
    :func:`refusal` and ask every sink about every other one.
    """
    return [name for name in (*SOLVERS, *WRITERS) if name != exclude and _takes(program, name)]


def refusal(program: plan.Program, name: str) -> str | None:
    """Why the sink called *name* cannot take *program*, or ``None``.

    The refusal contract is **the construct, the sink, and the sinks that do
    take it** — without that third clause an optional check only moves the
    surprise earlier for whoever thought to ask. Two shapes, since they have
    different remedies: a capability the sink lacks outright, and a pair it has
    both halves of and refuses together.
    """
    table = sink_capabilities(name)
    needed = caps.required(program)
    if missing := table.missing(needed):
        return sink_refuses_message(name, missing, _takers(program, name))
    if combination := table.excluded(needed):
        return sink_refuses_combination_message(name, sorted(combination), _takers(program, name))
    return None


def relaxations(program: plan.Program, name: str) -> list[str]:
    """What the sink called *name* would rewrite to take *program*.

    Not refusals — the model solves — but it answers a question slightly
    different from the one asked, which is worth saying *before* it is read.

    In :data:`~lpspec.relational.sinks.capabilities.CAPABILITIES` order, for
    :meth:`~lpspec.relational.sinks.capabilities.Capabilities.missing`'s reason:
    a sink rewriting two of them reads the same way twice.
    """
    table = sink_capabilities(name)
    needed = caps.required(program)
    declared = any(v.variable_type != 'continuous' for v in program.variables)
    return [
        sink_reformulates_message(
            name,
            c,
            integrality_added=c in caps.REWRITTEN_AS_INTEGRALITY and not declared,
        )
        for c in caps.CAPABILITIES
        if c in needed and table.support(c) == 'reformulated'
    ]
