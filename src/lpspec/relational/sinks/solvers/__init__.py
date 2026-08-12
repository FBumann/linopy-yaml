"""The solver family: the tables in, an answer out. See ../README.md.

One module per solver, **named for the solver** — nothing here is named for
the mechanism, because every member uses the same one. Each answers::

    (tables, batch_rows, solver_options) -> (status, objective, primal, dual)

plus a ``build_<solver>`` that loads the model and stops, which is the seam
`bench/` measures. ``tests/test_architecture.py`` checks that off the path.

A solver may also define a **session** — that hand-off held open, so a second
solve of a rebound model pushes values onto the model it already holds. That
one is optional, and :data:`SESSIONS` is the list of who has it: a driver that
re-solves is faster where it exists and correct where it does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lpspec.errors import LpspecError
from lpspec.relational.sinks.solvers.gurobi import solve_gurobi
from lpspec.relational.sinks.solvers.highs import HighsSession, solve_highs

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any

    import polars as pl

    from lpspec.relational.sinks.tables import ModelTables
    from lpspec.relational.status import SolveStatus

    Answer = tuple[SolveStatus, float, pl.Series | None, pl.Series | None]
    Solve = Callable[[ModelTables, int | None, Mapping[str, Any] | None], Answer]

__all__ = ['SESSIONS', 'SOLVERS', 'session', 'solver']

#: Every solver a caller may name, and **closed** — a dict literal, not a
#: registry something installed can add to. Which solver runs is the caller's
#: choice at the call, so an installed package able to change what
#: ``solver_name='x'`` resolves to would be hard rule 5 one level down.
SOLVERS: Mapping[str, Solve] = {
    'highs': solve_highs,
    'gurobi': solve_gurobi,
}

#: Those of :data:`SOLVERS` that can stay loaded between solves. What a
#: session answers is ``../README.md``'s table, and ``highs.HighsSession`` is
#: the member that does. A subset, and the uneven capability Track 3 (#472)
#: exists to declare: absence costs a driver the warm basis and nothing else,
#: so it is read where the model is handed over rather than asked about at
#: the call.
SESSIONS: Mapping[str, type[Any]] = {
    'highs': HighsSession,
}


def solver(name: str) -> Solve:
    """The solver called *name*, or an error listing every alternative."""
    try:
        return SOLVERS[name]
    except KeyError:
        raise LpspecError(
            f'unknown solver {name!r} — this build solves with {", ".join(sorted(SOLVERS))}. '
            'HiGHS ships with the package and is the default; gurobi needs the [gurobi] extra.'
        ) from None


def session(name: str) -> type[Any] | None:
    """How *name* stays loaded between solves, or ``None`` if it cannot.

    Never an error: which solver was named is :func:`solver`'s question, and a
    second answer to it here could disagree.
    """
    return SESSIONS.get(name)
