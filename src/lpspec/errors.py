"""The run half of the exception hierarchy, and the whole of it re-exported.

The split that matters is **the model versus the run**. The model half —
:class:`LanguageError` and what derives from it, decidable at load time with no
data bound — belongs to ``math_spec`` and is re-exported here, so one
``except`` clause still covers the package and a caller keeps saying
``lps.LanguageError``. What is defined here is the run: :class:`DataError` is a
fine file with the wrong thing bound to it, :class:`LaneError` a file one lane
cannot build, :class:`NoSolutionError` a solve with nothing to read back.

**Not a leaf any more.** This module imports the language, because the root of
the hierarchy has to live upstream of every class that extends it. The engine
still names only this module and ``frames.py``; what it now costs to import is
the language package behind them (docs/about/architecture.md, hard rule 2).

**A message lives here only where two modules raise it** — the cross-lane
wordings, which have to be one sentence and not two. One raiser keeps its
message beside itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from math_spec import (
    DimensionError,
    LanguageError,
    MathSpecError,
    PiecewiseExpansionError,
    SchemaError,
    did_you_mean,
    schema_error,
)

#: The root, under the name callers have always caught it by. An alias and not a
#: subclass: `except lps.LpspecError` has to keep catching a `LanguageError`, and
#: it only does while the two names are one class. The language owns the root
#: because a base cannot live downstream of what extends it, and that is now
#: literally true — it lives in the other package.
LpspecError = MathSpecError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class LpspecWarning(UserWarning):
    """Advice from ``check``: the model loads and solves, and reads wrong.

    A warning rather than an error because the reading may be deliberate — a
    model part-written declares what its expressions have not reached yet.
    """


# ---------------------------------------------------------------------------
# The run is the problem — the model was fine
# ---------------------------------------------------------------------------


class LaneError(LpspecError):
    """A lane cannot **build** a model it accepts — the other one can.

    Not a :class:`LanguageError`: the file is sayable, lowers, and reaches an
    answer by the other route, so the fix is which lane runs it rather than
    what the file says. Its own class because that is the difference a caller
    running both acts on.
    """


class DataError(LpspecError):
    """Data bound to a valid model is missing or the wrong shape."""


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
    'schema_error',
]


def uncovered_constant_message(names: str, missing: int, subject: str) -> str:
    """Why a constant side may not be sparse — one wording, both lanes.

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
        f'  Drop the declaration, if the model has no such quantity at all.'
    )


def sparse_divisor_message(name: str, missing: int) -> str:
    """Why a divisor may not be sparse — one wording, both lanes.

    Divisor position is the one place the absence rules' zero fill has no identity to
    fall back on: 0 divides by zero, 1 silently rescales, and dropping the term
    rewrites what the row asserts.
    """
    return (
        f"parameter '{name}' is used as a divisor but covers {missing} fewer "
        f'coordinates than it is indexed over. A missing row means a zero '
        f'coefficient everywhere else, and zero is not a divisor: the term '
        f'would drop and the constraint would silently stop constraining.\n'
        f'  Supply the missing rows, or mask the coordinates out with a where.'
    )


def coordinates_shown(dims: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Coordinates as a refusal prints them: ``f='b'; f='c'``.

    Both lanes format the offenders they found here rather than each in its own
    frame vocabulary, because ``tests/test_data_parity.py`` compares the two
    messages as strings. Values arrive as python natives — a numpy scalar reprs
    as ``np.str_('b')`` under numpy 2 and would break that comparison.
    """
    return '; '.join(', '.join(f'{d}={v!r}' for d, v in zip(dims, row, strict=True)) for row in rows)


def holes_in_values_message(name: str, holes: int, shown: str) -> str:
    """A row carrying no value — one wording, both lanes.

    In long form the absence of a value is the absence of the row, so a hole is
    the one encoding that says both at once: the row claims the coordinate and
    the value denies it. Refused rather than read as either, because the two
    readings build different models — a hole read as a row is a zero
    coefficient, a hole read as no row is what the absence rules then govern.

    NaN is named beside null because pandas has no other spelling: a ``None``
    in a float column arrives as NaN, so a message naming only null would send
    half its readers hunting for something their frame cannot hold.
    """
    at = f': {shown}' if shown else ''
    return (
        f"parameter '{name}' carries {holes} row(s) with no value — null or NaN{at}. "
        f'In long form the absence of a value is the absence of the row, and such a row '
        f'says the coordinate exists and denies it in the same breath.\n'
        f'  Drop them     polars .drop_nulls("value").drop_nans("value"), pandas .dropna(subset=["value"])\n'
        f'  Supply them   if a number was what was meant'
    )


def null_bounds_message(name: str, rows: int) -> str:
    """A bound with no value — one wording, both lanes.

    The absence rules' zero is a coefficient, never a bound: unbounded is not
    bounded-at-zero. Both exits are named because they build different models —
    supplying the value bounds the variable, masking removes it from every row
    and from the solution — so naming one would choose for the caller.
    """
    return (
        f"variable '{name}': {rows} rows have NULL bounds — a bound parameter is missing "
        f'values for some coordinates. The two ways out build different models, so the '
        f'language will not pick one:\n'
        f'  supply the value           the variable exists there, bounded (`inf` is a value)\n'
        f'  where: "<the parameter>"   the variable does not exist there at all'
    )


def wrong_value_dtype_message(name: str, declared: str, arrived: str) -> str:
    """A column that is not the type its declaration claims — one wording, both lanes.

    Refused rather than read, because the declaration is not decoration: it
    decides what a ``where`` comparison is checked against, whether the name
    may stand where an operator reads a position, and what a bare ``where`` on
    it means. A column that disagrees makes the file describe a model the data
    does not build.

    *arrived* is named in the language's own four words rather than in polars'
    or numpy's, which is what lets both lanes reach this sentence — and reads
    as the declaration the caller would have to write instead.
    """
    return (
        f"parameter '{name}' is declared '{declared}' and its values arrived as '{arrived}'. "
        f'A declared dtype is a claim about the values, and it is checked here — the file '
        f'says what the column is, or the column is not bound.\n'
        f'  Cast the column to {declared}, if the declaration is what you meant\n'
        f'  Or declare what the data has: {{dtype: {arrived}}}'
    )


#: How a capability reads in a sentence. The identifiers are the descriptor's
#: vocabulary and are not what a modeller calls these things, and a refusal is
#: read by whoever hit it rather than by whoever wrote the table.
_SPELLED = {
    'integrality': 'binary or integer variables',
    'sos': 'special-ordered sets (`sos:`)',
    'quadratic_objective': 'a quadratic objective',
    'nonconvex_quadratic_objective': 'a nonconvex quadratic objective',
    'quadratic_constraint': 'a quadratic constraint',
}


def spelled(capabilities: Sequence[str]) -> str:
    """Capabilities as a refusal names them, wherever one is worded."""
    return ', '.join(_SPELLED.get(c, c) for c in capabilities)


def lane_cannot_build_message(lane: str, missing: Sequence[str]) -> str:
    """A construct the language accepts and one *lane* cannot construct.

    Hard rule 3's amendment, worded. It names the other lane rather than a
    rewrite, there being nothing wrong with the model — the sink refusal
    (:func:`lpspec.relational.sinks._sink_refuses_message`) one level up.
    """
    return (
        f'the {lane} lane cannot build {spelled(missing)}, and no reformulation of it is exact. '
        f'The language accepts it and the streaming lane builds it, so this is a limit of the '
        f'lane rather than of the model.\n'
        f'Build it with lps.build()/lps.solve() instead, and ask check(model, sink=...) which '
        f'solver will take it — gurobi does, and an .lp file carries it to anything that does.'
    )


def duplicate_coordinate_message(name: str, shown: str, dims: list[str]) -> str:
    """More than one value for one coordinate — one wording, both lanes.

    A parameter is a function of its dims, so two rows for one coordinate has
    no answer the language could pick. Both lanes refuse before the duplicate
    reaches xarray, whose own `ValueError` names neither the parameter nor the
    repair — the opaque exception the error rules exist to prevent (#351).
    """
    return (
        f"parameter '{name}' has more than one row for a coordinate: {shown}. "
        f'A parameter is a function of its dims, so which value applies is undefined — '
        f'aggregate the source to one row per {dims} before binding it.'
    )


def index_without_its_label_column_message(dim: str, available: Sequence[str]) -> str:
    """An index table carrying everything but the labels — one wording, both lanes.

    The column is named after the dimension because that is what a lookup
    column is named after too: an index is a table about one dimension, and
    nothing else in it would say which column the members are.
    """
    return (
        f"index for dimension '{dim}' is a table without a '{dim}' column (has "
        f'{list(available)}). The label column is named after the dimension.'
    )


def no_index_source_message(dim: str) -> str:
    """A dimension with no index — one wording, both lanes.

    The index is what says which labels exist, so a parameter carrying a label
    it does not hold is a typo rather than a definition. Inferring the labels
    from the parameters instead would make that distinction unavailable, which
    is why there is no fallback to describe here.
    """
    return (
        f"dimension '{dim}' has no index: declare dimensions.{dim}.values in the model, "
        f"or pass a table of its labels under key '{dim}'. The index is what says which "
        f'labels exist — without one a mistyped label is indistinguishable from a new one.'
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
