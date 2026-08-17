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
from lpspec.relational.sinks import sos
from lpspec.relational.sinks.solvers.base import Solver
from lpspec.relational.sinks.solvers.gurobi import Gurobi
from lpspec.relational.sinks.solvers.highs import Highs
from lpspec.relational.sinks.solvers.xpress import Xpress

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from lpspec.relational.sinks.tables import ModelTables

#: ``WarmStart`` is deliberately absent. The carry it describes has no caller
#: above the family yet (#382), so it stays where the machinery is —
#: ``solvers.base`` — rather than reading as a surface something may use.
__all__ = ['SOLVERS', 'Solver', 'ingestible', 'loaded', 'solver']

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
        raise LpspecError(
            f'unknown solver {name!r} — this build solves with {", ".join(sorted(SOLVERS))}. '
            'HiGHS ships with the package and is the default; gurobi and xpress need their own extras.'
        ) from None
    if not found.is_available():
        raise ModuleNotFoundError(
            f'{name} is a solver this build knows, but its package is not installed here. {found.unavailable_message}'
        )
    return found


def ingestible(name: str, model: ModelTables) -> ModelTables:
    """*model* in the form the named solver can take it — sets included.

    The one place a capability is acted on, and it is the *family*'s rather
    than a member's: a solver that cannot ingest a special-ordered set is
    handed :func:`~lpspec.relational.sinks.sos.reformulated` tables, so no
    ``_load`` has to know the model ever carried one, and everything that
    reads a solve back — the span check, the label slices — sees the one model
    the solver actually holds.

    Asked before the load rather than inside it because a rebind compares the
    *ingested* digest: a big-M is a matrix coefficient by then, so a bound
    that moved one is a model to load again rather than numbers to push.

    Only ``reformulated`` is rewritten here. ``absent`` is a refusal rather
    than a silent rewrite, and one the caller is told about before the solve
    (#89) — a sink that cannot take a set and is handed one anyway would
    otherwise be given a model nobody chose.

    Returns:
        *model* itself where nothing has to change, which is every model
        declaring no sets.
    """
    if model.sos.height and solver(name).capabilities.support('sos') == 'reformulated':
        return sos.reformulated(model)
    return model


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
