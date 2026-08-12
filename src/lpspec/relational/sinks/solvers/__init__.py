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

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from lpspec.relational.sinks.tables import ModelTables

__all__ = ['SOLVERS', 'Solver', 'loaded', 'solver']

#: Every solver a caller may name, and **closed** — a dict literal, not a
#: registry something installed can add to. Which solver runs is the caller's
#: choice at the call, so an installed package able to change what
#: ``solver_name='x'`` resolves to would be hard rule 5 one level down.
#:
#: The *class*, not a function: a solver is a loaded model with a lifecycle
#: (:class:`~lpspec.relational.sinks.solvers.base.Solver`), and a one-shot
#: solve is that lifecycle walked once and thrown away rather than a second
#: thing this could hold.
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


def loaded(
    held: Solver | None,
    name: str,
    model: ModelTables,
    batch_rows: int | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> Solver:
    """The solver to run *model* on — *held* where it may keep what it holds.

    The whole of "reuse or load again", in the family that owns both halves —
    a caller keeps the solver it is handed and nothing else, where the same
    decision spread across the engine meant the engine tracking which sink was
    loaded and closing it at the right moment.

    *held* is kept exactly when it is the named class holding a model that
    differs from this one in nothing but numbers — same
    :attr:`~lpspec.relational.sinks.tables.ModelTables.structure`, same
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
        keeps = held._options == dict(solver_options or {}) and held._structure == model.structure
        if type(held) is wanted and keeps:
            held.push(model)
            return held
        held.close()
    return wanted(model, batch_rows, solver_options)
