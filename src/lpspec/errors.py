"""The exception hierarchy, so that one ``except`` clause covers the package.

Before this module the package raised four unrelated ``ValueError``
subclasses and a great deal of bare ``ValueError``, which left a caller no
way to say "this model is the problem" without also catching every
``ValueError`` pandas or pydantic might raise on the way past.

The split that matters is **the model versus the run**:

* :class:`LanguageError` — the file says something the language does not
  accept. Nothing about the data would change the outcome; it is decidable at
  load time, and ``lps.check()`` raises exactly these.
* :class:`DataError` — the file is fine; what was bound to it is not. An
  unbound source, a column that does not carry the declared dims.

Everything subclasses :class:`LpspecError`, which subclasses ``ValueError`` —
so code that catches ``ValueError`` today keeps working.

``model.py``'s field validators raise plain ``ValueError``, because pydantic
collects those into its own ``ValidationError`` and a custom class does not
survive the trip. :func:`schema_error` turns one back into a
:class:`SchemaError` at the API boundary, so a caller sees one tree rather than
two — the class was always named for exactly that case ("unknown key, bad
dtype") and simply was not wired to it.

Deliberately dependency-free: the relational engine imports this module and
nothing else from the package (docs/ARCHITECTURE.md, hard rule 2).
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


class LpspecError(ValueError):
    """Base class for every error this package raises on purpose."""


# ---------------------------------------------------------------------------
# The model is the problem — decidable without data
# ---------------------------------------------------------------------------


class LanguageError(LpspecError):
    """The model is not sayable in the language, or does not obey its rules."""


class SchemaError(LanguageError):
    """**The declarations themselves are wrong**, before any expression is read.

    An unknown key, a bad ``dtype``, a duplicate YAML key, two objectives, a
    version this reader does not know. Distinct from a bare
    :class:`LanguageError`, which is declarations that are fine saying
    something the language rejects — an undeclared name in an expression, a
    dim rule, degree 2.

    Every failure of schema validation arrives as this, including the ones
    pydantic raises: :func:`schema_error` unwraps its ``ValidationError`` so a
    caller sees one tree rather than two.
    """


class DimensionError(LanguageError):
    """A dim-set rule was violated. Raised at load time, before any data."""


class PiecewiseExpansionError(LanguageError):
    """A piecewise block references something that doesn't exist or collides."""


# ---------------------------------------------------------------------------
# The run is the problem — the model was fine
# ---------------------------------------------------------------------------


class DataError(LpspecError):
    """Data bound to a valid model is missing or the wrong shape."""


class NoSolutionError(LpspecError):
    """The solve returned no values to read — infeasible, unbounded, errored.

    Neither the model nor the data was wrong; the answer is that there is no
    answer. It has its own class because the caller's response differs: a
    scenario sweep catches this and records the outcome, where a
    :class:`LanguageError` means the file needs editing.
    """


__all__ = [
    'DataError',
    'DimensionError',
    'LanguageError',
    'LpspecError',
    'NoSolutionError',
    'PiecewiseExpansionError',
    'SchemaError',
]


def did_you_mean(name: str, known: Iterable[str], *, label: str = 'Declared') -> str:
    """The repair clause for an unrecognised name: the near miss, or the set.

    Three places ask it — an unknown declaration name, an unknown YAML key, an
    unknown symbol-table entry — and each frames it with a sentence of its own.
    Only the clause is shared, because only the clause is the same question.
    """
    candidates = sorted(known)
    near = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
    if near:
        return f"Did you mean '{near[0]}'?"
    return f'{label}: {", ".join(candidates) or "nothing"}.'


def sparse_divisor_message(name: str, missing: int) -> str:
    """Why a divisor may not be sparse — one wording, both lanes.

    Everywhere else a missing parameter row is a zero coefficient (SPEC §6), and
    a zeroed term is a term that does not participate: the row survives and
    still says something. In divisor position there is no fill that preserves
    the constraint — 0 divides by zero, 1 silently rescales, and dropping the
    term rewrites what the row asserts — so the language refuses rather than
    picking one. That is v1's own argument for not filling on a caller's behalf,
    at the one position where it has no identity to fall back on.
    """
    return (
        f"parameter '{name}' is used as a divisor but covers {missing} fewer "
        f'coordinates than it is indexed over. A missing row means a zero '
        f'coefficient everywhere else, and zero is not a divisor: the term '
        f'would drop and the constraint would silently stop constraining.\n'
        f'  Supply the missing rows, or mask the coordinates out with a where.'
    )


def null_bounds_message(name: str, rows: int) -> str:
    """A bound with no value — one wording, both lanes.

    A missing row is a zero coefficient in a product (SPEC §6) and nothing at
    all in a bound: unbounded is not bounded-at-zero, and guessing either way
    changes which solutions exist. So both lanes refuse, and both say so while
    the model is still being built rather than letting the gap reach a sink.

    Naming both exits is the point. They are not two spellings of one repair —
    supplying the value keeps the variable and bounds it, masking removes the
    variable from every row and from the solution. A message that named only one
    would be choosing the model on the caller's behalf, which is the thing the
    refusal exists to avoid.
    """
    return (
        f"variable '{name}': {rows} rows have NULL bounds — a bound parameter is missing "
        f'values for some coordinates. The two ways out build different models, so the '
        f'language will not pick one:\n'
        f'  supply the value           the variable exists there, bounded (`inf` is a value)\n'
        f'  where: "<the parameter>"   the variable does not exist there at all'
    )


def duplicate_coordinate_message(name: str, shown: str, dims: list[str]) -> str:
    """More than one value for one coordinate — one wording, both lanes.

    A parameter is a function of its dims, so two rows for one coordinate has
    no answer the language could pick. Both lanes refuse; the eager one used to
    let the duplicate reach xarray, which raised a bare `ValueError` from its
    index machinery — the opaque exception with no pointer back to a
    declaration that §9 exists to prevent (#351).
    """
    return (
        f"parameter '{name}' has more than one row for a coordinate: {shown}. "
        f'A parameter is a function of its dims, so which value applies is undefined — '
        f'aggregate the source to one row per {dims} before binding it.'
    )


def unknown_labels_message(name: str, dim: str, strangers: list[object], known: list[object]) -> str:
    """A source label the dimension does not have — one wording, both lanes.

    Distinct from sparsity, which is ordinary: a *missing* row reads as zero
    (SPEC §8), but a row that is present and unaddressable is a typo, and §2
    already draws that line for coordinates — "null means no group; an unknown
    non-null value is a typo". The consequence of not saying so is the row
    vanishing in the join that places it, and the coordinate it was meant for
    falling back on the zero it never asked for (#350).

    Only asked where the dimension's labels come from somewhere else. A
    dimension derived *from* the parameters cannot have a stranger in it,
    because the union of what arrived is the definition.
    """
    shown = ', '.join(repr(s) for s in strangers[:5])
    more = f' (and {len(strangers) - 5} more)' if len(strangers) > 5 else ''
    return (
        f"parameter '{name}' has label(s) in dimension '{dim}' that are not coordinates "
        f'of it: {shown}{more}.\n'
        f'  {dim} has: {sorted(str(k) for k in known)[:10]}\n'
        f'A missing row is a zero coefficient, but a label that is not a coordinate is a '
        f'typo: its row joins nothing, so the coordinate it was meant for silently reads '
        f'as absent. Fix the label, or declare it as a coordinate.'
    )


def unknown_name_message(kind: str, name: str, known: Iterable[str]) -> str:
    """``unknown <kind> '<name>'``, plus the near miss or the declared set.

    The same shape as the loader's unknown-key error, deliberately: a reader who
    has met one has met both, and there were already two copies of this idiom in
    the tree before this one.

    Written for #298's positional names (`ramp_0`, `ramp_1`) and kept after they
    were removed, because the shape outlived the cause: `piecewise:` still
    expands one block into several constraints, and a rule split by regime is
    conventionally `x` and `x_initial`. What changed is the wording — "named by
    position" would now be a claim about a surface that no longer exists.

    Single-line on purpose. These are raised as ``KeyError``, whose ``str`` is
    the *repr* of its argument, so a newline arrives at the reader as a literal
    ``\\n``. The list is not truncated for the same reason the loader does not
    truncate: the answer is usually in it, and a caller reading a solution back
    by name has no other way to discover what the model actually built.

    When one name expanded into several, nearest-match is unhelpful — it picks
    one sibling and implies the others do not exist — so a prefix hit lists
    the whole family instead.
    """
    candidates = sorted(known)

    family = [c for c in candidates if c.startswith(f'{name}_')]
    if family:
        return (
            f"unknown {kind} '{name}': no declaration has that name, but "
            f'{len(family)} begin with it — {", ".join(family)}.'
        )

    return f"unknown {kind} '{name}'. {did_you_mean(name, candidates)}"


def schema_error(exc: Any, context: str = '') -> SchemaError:
    """A pydantic ``ValidationError`` as one of ours.

    Pydantic wraps whatever a validator raises, so a class of our own cannot
    reach the caller from inside the model — the envelope arrives instead,
    carrying ``input_value=`` dumps and a link to pydantic's docs that mean
    nothing to someone who wrote a YAML file. Unwrapping here is what lets
    ``except LpspecError`` cover the model, which is what the API promises.

    The messages themselves are kept: they were written for this audience and
    only the envelope is discarded.
    """
    lines = []
    for error in exc.errors():
        message = str(error.get('msg', '')).removeprefix('Value error, ')
        where = '.'.join(str(part) for part in error.get('loc', ()))
        lines.append(f'{where}: {message}' if where else message)
    body = '\n'.join(lines) or str(exc)
    return SchemaError(f'{context}: {body}' if context else body)
