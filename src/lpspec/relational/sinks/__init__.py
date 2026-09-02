"""Sinks: how a built model leaves the engine. See README.md.

**Two families.** A *solver* takes the tables and runs them (``solvers/``,
chosen by name); a *writer* renders them to a file (``writers/``, chosen by
suffix). They are directories rather than a convention, so
``tests/test_architecture.py`` reads membership off the path.

``tables.py`` is what both read, and neither family imports the other.
``capabilities.py`` is what both *declare*, and the functions below are where
a caller's model meets those declarations — the only place the two families are
asked one question together, which is why ``ingestible`` is here rather than in
``solvers/``: what a sink cannot take is refused by naming the sinks that can,
and those live in both families.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lpspec.errors import LpspecError, unknown_name_message
from lpspec.relational.sinks import capabilities as caps
from lpspec.relational.sinks import sos
from lpspec.relational.sinks.capabilities import spelled
from lpspec.relational.sinks.solvers import SOLVERS, Solver, loaded, solver
from lpspec.relational.sinks.tables import Tables
from lpspec.relational.sinks.writers import WRITERS, writer

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from math_spec import program

__all__ = [
    'SOLVERS',
    'WRITERS',
    'Solver',
    'Tables',
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


def _blocker(name: str, needed: Collection[caps.Capability]) -> Callable[[Sequence[str]], str] | None:
    """Why the sink called *name* refuses capabilities *needed*, or ``None``.

    The one home for what "takes" means, so the refusal and the takers it
    names cannot disagree. What comes back is the message short of its third
    clause — a function of the takers — because naming them means asking this
    of every other sink first.
    """
    table = sink_capabilities(name)
    if missing := table.missing(needed):
        return lambda takers: _sink_refuses_message(name, missing, takers)
    if combination := table.excluded(needed):
        return lambda takers: _sink_refuses_combination_message(name, sorted(combination), takers)
    return None


def refusal(program: program.Program, name: str) -> str | None:
    """Why the sink called *name* cannot take *program*, or ``None``.

    The refusal contract is **the construct, the sink, and the sinks that do
    take it** — without that third clause an optional check only moves the
    surprise earlier for whoever thought to ask. Two shapes, since they have
    different remedies: a capability the sink lacks outright, and a pair it has
    both halves of and refuses together.
    """
    needed = caps.required(program)
    if (refuses := _blocker(name, needed)) is None:
        return None
    return refuses([other for other in (*SOLVERS, *WRITERS) if other != name and _blocker(other, needed) is None])


def _instead(takers: Sequence[str]) -> str:
    """The third clause of the refusal contract: who *does* take it."""
    if not takers:
        return 'No sink this build has takes it.'
    return f'Sinks that do take it: {", ".join(sorted(takers))}.'


def _sink_refuses_combination_message(sink: str, combination: Sequence[str], takers: Sequence[str]) -> str:
    """A sink that has both halves of a pair and refuses them together.

    Its own sentence, because a caller reading "it cannot take a quadratic
    objective" of a sink whose documentation says it can would conclude the
    message is wrong.
    """
    return (
        f'the {sink!r} sink takes {spelled(combination)} separately and refuses them together, '
        f'which is a limit of that solver rather than of the model. {_instead(takers)}'
    )


def _sink_refuses_message(sink: str, missing: Sequence[str], takers: Sequence[str]) -> str:
    """A sink asked for a capability it does not have at all."""
    return (
        f'the {sink!r} sink cannot take {spelled(missing)}: it has no such concept, so there is '
        f'nothing to hand the model to. {_instead(takers)}'
    )


def relaxations(program: program.Program, name: str) -> list[str]:
    """What the sink called *name* would rewrite to take *program*.

    Not refusals — the model solves — but it answers a question slightly
    different from the one asked, which is worth saying *before* it is read.

    In :data:`~lpspec.relational.sinks.capabilities.CAPABILITIES` order, for
    :meth:`~lpspec.relational.sinks.capabilities.Capabilities.missing`'s reason:
    a sink rewriting two of them reads the same way twice.
    """
    table = sink_capabilities(name)
    needed = caps.required(program)
    return [
        _sink_reformulates_message(
            name,
            c,
            integrality_added=c in caps.REWRITTEN_AS_INTEGRALITY and 'integrality' not in needed,
        )
        for c in caps.CAPABILITIES
        if c in needed and table.support(c) == 'reformulated'
    ]


def _sink_reformulates_message(sink: str, capability: str, *, integrality_added: bool) -> str:
    """A sink meeting a capability by rewriting the model into one it takes.

    *integrality_added* is the one consequence derivable here rather than
    assumed per capability: a model that declared no integrality and reaches
    the solver mixed-integer comes back without duals.
    """
    cost = (
        ' The model declared no integrality of its own and reaches the solver mixed-integer, '
        'so it will come back without duals.'
        if integrality_added
        else ''
    )
    return (
        f'the {sink!r} sink has no native support for {spelled([capability])} and '
        f'will take it reformulated, so what reaches the solver is not what the file declares.{cost}'
    )


def ingestible(name: str, tables: Tables, program: program.Program | None = None) -> Tables:
    """*tables* in the form the named solver can take it — sets included.

    The one place a capability is acted on, and it is the *family*'s rather
    than a member's: a solver that cannot ingest a special-ordered set is
    handed :func:`~lpspec.relational.sinks.sos.reformulated` tables, so no
    ``_load`` has to know the model ever carried one, and everything that
    reads a solve back — the span check, the label slices — sees the one model
    the solver actually holds.

    Asked before the load rather than inside it because an update compares the
    *ingested* digest: a big-M is a matrix coefficient by then, so a bound
    that moved one is a model to load again rather than numbers to push.

    *program* is what the refusal is decided on, and it is optional only
    because a caller composing tables by hand has no plan to hand over: given
    one, a model this sink cannot take is refused **here**, before the load,
    with the same sentence ``check(spec, sink=...)`` would have given hours
    earlier. Without it the refusal falls to the solver, which reports it as
    an error code from inside a library.

    Only ``reformulated`` is rewritten, so the rewrite and the refusal read the
    same cell: a sink is never handed a rewrite of a construct it declared it
    has no concept of.

    Returns:
        *tables* itself where nothing has to change, which is every model
        declaring no sets.

    Raises:
        LpspecError: A *program* carrying a construct this sink has no concept
            of, or a combination it refuses.
    """
    if program is not None and (refused := refusal(program, name)) is not None:
        raise LpspecError(refused)
    if tables.sos.height and sink_capabilities(name).support('sos') == 'reformulated':
        return sos.reformulated(tables)
    return tables
