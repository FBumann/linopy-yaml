"""The solver family: one class per solver, holding one model. See ../README.md.

One module per solver, **named for the solver** — nothing here is named for the
mechanism, because every member uses the same one. Each defines a
:class:`~lpspec.relational.sinks.solvers.base.Solver` subclass named for it,
plus two functions: ``solve_<name>``, the one-shot spelling that loads, runs
and releases; and ``build_<name>``, the load-only seam `bench/` measures.
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

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ['SOLVERS', 'Solver', 'solver']

#: Every solver a caller may name, and **closed** — a dict literal, not a
#: registry something installed can add to. Which solver runs is the caller's
#: choice at the call, so an installed package able to change what
#: ``solver_name='x'`` resolves to would be hard rule 5 one level down.
#:
#: The *class*, not a function: a solver is a loaded model with a lifecycle
#: (:class:`~lpspec.relational.sinks.solvers.base.Solver`), and ``solve_<name>``
#: is that lifecycle walked once and thrown away rather than a second thing
#: this could hold.
SOLVERS: Mapping[str, type[Solver]] = {
    'highs': Highs,
    'gurobi': Gurobi,
}


def solver(name: str) -> type[Solver]:
    """The solver called *name*, or an error listing every alternative."""
    try:
        return SOLVERS[name]
    except KeyError:
        raise LpspecError(
            f'unknown solver {name!r} — this build solves with {", ".join(sorted(SOLVERS))}. '
            'HiGHS ships with the package and is the default; gurobi needs the [gurobi] extra.'
        ) from None
