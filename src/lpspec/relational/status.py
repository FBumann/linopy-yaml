"""What a solve returned, on two axes.

Copied spelling for spelling from ``linopy.constants``, so anyone arriving from
linopy need not learn a second vocabulary for the same facts;
`tests/test_solve_status.py` asserts the tables still match, making drift a
test failure rather than a discovery. Nothing here imports linopy — the engine
may not (hard rule 2) — but the test does.

The two axes stay separate: ``termination_condition`` is what the solver said,
``status`` what it means for the caller. **ok does not mean optimal** — a run
stopped at a time limit with an incumbent is ``ok``, there being values worth
reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SolverStatusName = Literal['ok', 'warning', 'error', 'aborted', 'unknown']

TerminationConditionName = Literal[
    'unknown',
    'optimal',
    'time_limit',
    'iteration_limit',
    'terminated_by_limit',
    'suboptimal',
    'imprecise',
    'unbounded',
    'infeasible',
    'infeasible_or_unbounded',
    'other',
    'internal_solver_error',
    'error',
    'user_interrupt',
    'resource_interrupt',
    'licensing_problems',
]

#: Which termination conditions roll up to which coarse status.
STATUS_TO_TERMINATION_CONDITIONS: dict[str, frozenset[str]] = {
    'ok': frozenset({'optimal', 'time_limit', 'iteration_limit', 'terminated_by_limit', 'suboptimal', 'imprecise'}),
    'warning': frozenset({'infeasible', 'infeasible_or_unbounded', 'unbounded', 'other'}),
    'error': frozenset({'internal_solver_error', 'error'}),
    'aborted': frozenset({'user_interrupt', 'resource_interrupt', 'licensing_problems'}),
    'unknown': frozenset({'unknown'}),
}


def status_of(termination_condition: str) -> str:
    """The coarse status a termination condition rolls up to."""
    for status, conditions in STATUS_TO_TERMINATION_CONDITIONS.items():
        if termination_condition in conditions:
            return status
    return 'unknown'


@dataclass(frozen=True)
class SolveStatus:
    """The outcome of a solve, on both axes plus the solver's own wording."""

    termination_condition: str
    #: Exactly what the solver called it, for a message a user can search for.
    solver_wording: str = ''
    #: Whether the solver reports an actual primal, which the termination
    #: condition does not tell you — see :attr:`is_readable`.
    has_primal: bool = True

    @property
    def status(self) -> str:
        return status_of(self.termination_condition)

    @property
    def is_ok(self) -> bool:
        """linopy's rollup: the run is not an error, an abort or a refusal.

        Kept exactly as linopy defines it, because it is shared vocabulary.
        It is *not* the question "can I read values" — see
        :attr:`is_readable`.
        """
        return self.status == 'ok'

    @property
    def is_readable(self) -> bool:
        """Whether there are primal values to read.

        Deliberately beyond linopy, whose ``safe_get_solution`` gates on
        ``is_ok``: a MIP stopped at a time limit **before finding any
        incumbent** is ``ok``, and its zero-filled ``col_value`` would be read
        as an answer (#115).

        ``optimal`` always has a primal. Every other ``ok`` condition means
        "stopped early", and whether an incumbent exists is a separate fact
        only the solver knows.
        """
        return self.is_ok and self.has_primal
