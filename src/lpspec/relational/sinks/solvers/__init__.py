"""The solver family: one class per solver, holding one model. See ../README.md.

One module per solver, **named for the solver** — nothing here is named for the
mechanism, because every member uses the same one. Each defines a
:class:`~lpspec.relational.sinks.solvers.base.Solver` subclass named for it,
plus ``build_<name>``, the load-only seam `bench/` measures.
``tests/test_architecture.py`` checks all of that off the path.

What a solver holds between solves, and the rule for keeping it, is
:mod:`~lpspec.relational.sinks.solvers.base` — the one module a member may read
besides ``tables.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lpspec.errors import LpspecError
from lpspec.relational.sinks.solvers.base import Solver
from lpspec.relational.sinks.solvers.gurobi import Gurobi
from lpspec.relational.sinks.solvers.highs import Highs
from lpspec.relational.sinks.solvers.xpress import Xpress

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from lpspec.relational.sinks.tables import Tables

__all__ = ['SOLVERS', 'Solver', 'loaded', 'solver']

#: Every solver a caller may name. Closed: a dict literal, not a registry
#: something installed can add to.
SOLVERS: Mapping[str, type[Solver]] = {
    'highs': Highs,
    'gurobi': Gurobi,
    'xpress': Xpress,
}


def solver(name: str) -> type[Solver]:
    """The solver called *name*, or why this build cannot give you it.

    Both refusals land where the sink is resolved — before the build — which is
    what makes naming a sink nothing can serve cost no model. What to do about
    a missing package is the *member's* sentence
    (:attr:`~lpspec.relational.sinks.solvers.base.Solver.unavailable_message`),
    whether a solver ships or needs an extra being its own fact.

    Raises:
        LpspecError: A name outside the closed set.
        ModuleNotFoundError: A name inside it whose package this environment
            does not have.
    """
    try:
        found = SOLVERS[name]
    except KeyError:
        raise LpspecError(f'unknown solver {name!r} — this build solves with {", ".join(sorted(SOLVERS))}.') from None
    if not found.is_available():
        raise ModuleNotFoundError(
            f'{name} is a solver this build knows, but its package is not installed here. {found.unavailable_message}'
        )
    return found


def loaded(
    held: Solver | None,
    name: str,
    tables: Tables,
    solver_options: Mapping[str, Any] | None = None,
) -> Solver:
    """The solver to run *tables* on — *held* where it may keep what it holds.

    The whole of "reuse or load again": a caller keeps the solver it is
    handed and nothing else.

    *held* is kept exactly when it is the named class holding a model that
    differs from this one in nothing but numbers — same
    :attr:`~lpspec.relational.sinks.tables.Tables.structure`, same
    options, both recorded at its load — and then the new numbers are pushed
    onto it. The digest is the correctness floor: a model whose structure
    moved is a different model wearing the same labels, and pushing values
    onto it would answer a question nobody asked.

    A solver being replaced is closed here. It holds memory — and for one of
    them a licence — that no frame in this process accounts for, so leaving it
    to the collector would be leaving it to chance. *name* is resolved first,
    so a caller who named nothing this build has does not first pay a release.
    """
    wanted = solver(name)
    if held is not None:
        if type(held) is wanted and held.keeps(tables, solver_options):
            held.push(tables)
            return held
        held.close()
    return wanted(tables, None, solver_options)
