"""The run half of the exception hierarchy, and the whole of it re-exported.

The spec half — :class:`LanguageError` and what derives from it, decidable at
load time with no data bound — belongs to ``math_spec`` and is re-exported here,
so one ``except`` clause covers the package. The run half is defined here:
:class:`DataError` is a fine file with the wrong thing attached to it,
:class:`LaneError` a file one lane cannot build, :class:`NoSolutionError` a
solve with nothing to read back.

A message lives here only where both lanes raise it. One raiser keeps its
message beside itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from math_spec import (
    DimensionError,
    LanguageError,
    MathSpecError,
    PiecewiseExpansionError,
    SchemaError,
    did_you_mean,
)

#: The root, under the name callers catch it by. An alias and not a subclass:
#: `except lps.LpspecError` has to catch a `LanguageError`.
LpspecError = MathSpecError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class LpspecWarning(UserWarning):
    """Advice from ``check``: the spec loads and solves, and reads wrong.

    A warning rather than an error because the reading may be deliberate — a
    spec part-written declares what its expressions have not reached yet.
    """


class LaneError(LpspecError):
    """A lane cannot **build** a spec it accepts — the other one can.

    Not a :class:`LanguageError`: the file is sayable, lowers, and reaches an
    answer by the other route, so the fix is which lane runs it rather than
    what the file says.
    """


class DataError(LpspecError):
    """Data bound to a valid spec is missing or the wrong shape."""


class NoSolutionError(LpspecError):
    """The solve returned no values to read — infeasible, unbounded, errored.

    Its own class because the caller's response differs: a scenario sweep
    catches this and records the outcome, where a :class:`LanguageError` means
    the file needs editing.
    """


__all__ = [
    'DataError',
    'DimensionError',
    'LaneError',
    'LanguageError',
    'LpspecError',
    'LpspecWarning',
    'NoSolutionError',
    'PiecewiseExpansionError',
    'SchemaError',
    'did_you_mean',
]


def uncovered_constant_message(names: str, missing: int, subject: str) -> str:
    """Why a constant side may not be sparse.

    ``x <= cap`` with ``cap`` missing becomes ``x <= 0``: the most binding row
    expressible, built and solved and reported optimal. Which of the three
    exits is right depends on what was meant, so none is guessed.
    """
    return (
        f"{subject}: parameter '{names}' covers {missing} fewer coordinates than the rows "
        f'built here. A missing row is read as 0, and on the constant side that zero is a '
        f'bound rather than an absence — the row still exists, and it binds.\n'
        f'  Supply the missing rows, if the value is what was meant.\n'
        f'  Mask them out with a where, if the row should not exist there.\n'
        f'  Drop the declaration, if the spec has no such quantity at all.'
    )


def sparse_divisor_message(name: str, missing: int) -> str:
    """Why a divisor may not be sparse.

    Divisor position is the one place the absence rules' zero fill has no
    identity to fall back on: 0 divides by zero, 1 silently rescales, and
    dropping the term rewrites what the row asserts.
    """
    return (
        f"parameter '{name}' is used as a divisor but covers {missing} fewer "
        f'coordinates than it is indexed over. A missing row means a zero '
        f'coefficient everywhere else, and zero is not a divisor: the term '
        f'would drop and the constraint would silently stop constraining.\n'
        f'  Supply the missing rows, or mask the coordinates out with a where.'
    )


def null_bounds_message(name: str, rows: int) -> str:
    """A bound with no value.

    The absence rules' zero is a coefficient, never a bound. Both exits are
    named because they build different models — supplying the value bounds the
    variable, masking removes it from every row and from the solution.
    """
    return (
        f"variable '{name}': {rows} rows have NULL bounds — a bound parameter is missing "
        f'values for some coordinates. The two ways out build different models, so the '
        f'language will not pick one:\n'
        f'  supply the value           the variable exists there, bounded (`inf` is a value)\n'
        f'  where: "<the parameter>"   the variable does not exist there at all'
    )


def position_out_of_range_message(name: str, op: str, position: int, at: int, cardinality: int) -> str:
    """A ``position(dim)`` boundary naming no coordinate of the dimension."""
    return (
        f'where: position({name}) {op} {position} names position {at} of '
        f"'{name}', which has {cardinality} coordinate(s). A boundary that "
        f'names no coordinate leaves the rows it was to seed unseeded.'
    )


def short_groups_message(name: str, by: str, op: str, position: int, short: Sequence[str]) -> str:
    """A grouped ``position(dim, by=)`` boundary that some group is too short to reach."""
    return (
        f'where: position({name}, by={by}) {op} {position} names position '
        f'{position} within each group, and {len(short)} of them are shorter than '
        f'that: {list(short[:5])}. A boundary that names no coordinate leaves the rows it '
        f'was to seed unseeded.'
    )


def unknown_name_message(kind: str, name: str, known: Iterable[str]) -> str:
    r"""``unknown <kind> '<name>'``, plus the near miss or the declared set.

    Single-line on purpose: these are raised as ``KeyError``, whose ``str`` is
    the *repr* of its argument, so a newline reaches the reader as a literal
    ``\n``. Untruncated, because a caller reading a solution back by name has
    no other way to discover what the model built.

    A prefix hit lists the whole family rather than a near miss — ``piecewise:``
    expands one block into several constraints, and naming one sibling implies
    the others do not exist.
    """
    candidates = sorted(known)

    family = [c for c in candidates if c.startswith(f'{name}_')]
    if family:
        return (
            f"unknown {kind} '{name}': no declaration has that name, but "
            f'{len(family)} begin with it — {", ".join(family)}.'
        )

    return f"unknown {kind} '{name}'. {did_you_mean(name, candidates)}"
