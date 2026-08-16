"""The exception hierarchy, so that one ``except`` clause covers the package.

The split that matters is **the model versus the run**:
:class:`LanguageError` is the file saying something the language does not
accept — decidable at load time, and what ``lps.check()`` raises;
:class:`DataError` is a fine file with the wrong thing bound to it. Everything
subclasses :class:`LpspecError`, which subclasses ``ValueError``.

``model.py``'s field validators raise plain ``ValueError``, since pydantic
collects those into its own ``ValidationError`` and a custom class does not
survive the trip; :func:`schema_error` turns one back at the API boundary.

Deliberately dependency-free: the relational engine imports this module and
nothing else from the package (docs/about/architecture.md, hard rule 2).
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class LpspecError(ValueError):
    """Base class for every error this package raises on purpose."""


class LpspecWarning(UserWarning):
    """Advice from ``check``: the model loads and solves, and reads wrong.

    A warning rather than an error because the reading may be deliberate — a
    model part-written declares what its expressions have not reached yet.
    """


# ---------------------------------------------------------------------------
# The model is the problem — decidable without data
# ---------------------------------------------------------------------------


class LanguageError(LpspecError):
    """The model is not sayable in the language, or does not obey its rules."""


class SchemaError(LanguageError):
    """**The declarations themselves are wrong**, before any expression is read.

    An unknown key, a bad ``dtype``, a duplicate YAML key, a
    version this reader does not know — as against a bare
    :class:`LanguageError`, which is sound declarations saying something the
    language rejects (an undeclared name, a dim rule, degree 2).
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

    Its own class because the caller's response differs: a scenario sweep
    catches this and records the outcome, where a :class:`LanguageError` means
    the file needs editing.
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

    Only the clause is shared — an unknown declaration, an unknown YAML key and
    an unknown symbol-table entry each frame it with a sentence of their own.
    """
    candidates = sorted(known)
    near = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
    if near:
        return f"Did you mean '{near[0]}'?"
    return f'{label}: {", ".join(candidates) or "nothing"}.'


def unknown_source_keys_message(keys: Iterable[str], known: Iterable[str]) -> str:
    """A source key naming nothing the file declares — one wording, both doors.

    Refused rather than ignored, and ``rebind`` is where the reason was settled
    first: a name it does not recognise is a typo, and ignoring one there is a
    silent re-solve of the numbers you meant to replace. Binding owes the same
    answer — a dump carrying more than a model uses is filtered at the call,
    where the caller can see what was dropped.
    """
    unknown = sorted(keys)
    lead = f'source key {unknown[0]!r} names' if len(unknown) == 1 else f'source keys {unknown} name'
    return (
        f'{lead} neither a parameter nor a dimension this model declares. '
        f'{did_you_mean(unknown[0], known, label="Declared")} Pass only what the '
        f'model takes — a table carrying more than that is filtered here, not bound.'
    )


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


def no_duals_message(discrete: Sequence[str], termination_condition: str, sets: Sequence[str] = ()) -> str:
    """Why a solve that *did* leave values still has no duals.

    Integrality is decidable from the model, and naming the variable is
    actionable where "the solver reported none" is not.

    *sets* are the special-ordered sets a sink without the concept turned into
    binaries. They come first because a model that declared none of its own
    integrality would otherwise be told it is mixed-integer with nothing named
    — and because the fix is a different one: another sink, not a different
    model.
    """
    if sets:
        names = ', '.join(f"'{n}'" for n in sets)
        return (
            f'duals are undefined for a mixed-integer model, and this sink has no SOS concept, so '
            f'{names} reached it as binaries. Solve with a sink that takes a set natively (gurobi) '
            f'to keep the LP, or drop the set to price the relaxation.'
        )
    if discrete:
        names = ', '.join(f"'{n}'" for n in discrete)
        return (
            f'duals are undefined for a mixed-integer model: {names} '
            f'{"is" if len(discrete) == 1 else "are"} not continuous. '
            f'Drop the integrality to price the LP relaxation instead.'
        )
    return (
        f'the solver returned no dual solution, though the solve terminated '
        f'{termination_condition!r}. Duals come from a simplex basis, which a '
        f'run stopped short of one does not have.'
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


def unknown_labels_message(name: str, dim: str, strangers: list[object], known: list[object]) -> str:
    """A source label the dimension does not have — one wording, both lanes.

    Distinct from sparsity: a *missing* row reads as zero (the data-binding rules), but a row
    that is present and unaddressable is a typo, the line the declaration rules
    already draw for coordinates (#350).

    Only asked where the dimension's labels come from somewhere else — one
    derived *from* the parameters cannot have a stranger in it, the union of
    what arrived being its definition.
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


def dense_array_message(name: str) -> str:
    """A dense array where a table belongs — one wording, both lanes.

    An ``xarray.DataArray`` is recognisable and has ``__len__``, so left alone
    it would be read positionally on one lane and directly on the other. Both
    refuse it: this package reads tables — rows under named columns, an index
    being a column wearing a hat — and hands arrays back rather than taking
    them.
    """
    return (
        f"parameter '{name}': an xarray.DataArray is not a source. lpspec reads tables — "
        f'rows under named columns — and hands arrays back rather than taking them. Pass '
        f'array.to_series(), whose index binds by name on both lanes, or '
        f'array.to_series().reset_index() for a tidy frame. Result.to_dataarray() is the '
        f'way back out.'
    )


def lookups_need_an_index_message(dim: str, lookups: list[str], got: str) -> str:
    """A dimension carrying lookups and no index to read them from — one wording, both lanes.

    A lookup is a *column* of its dimension's index, so unlike labels it cannot
    be inferred from the parameters that happen to span the dimension: they
    carry the label, never what it maps to.
    """
    return (
        f"dimension '{dim}' carries lookups {sorted(lookups)} but has no index source "
        f"(got {got}). Pass one under key '{dim}' — a parquet path, or any table "
        f'carrying columns {[dim, *sorted(lookups)]}. A lookup cannot be inferred from '
        f'the parameters that happen to use the dimension: they carry the label, not '
        f'what it maps to.'
    )


def missing_lookup_columns_message(dim: str, missing: list[str], available: list[str]) -> str:
    """An index that is present and short of a declared lookup — one wording, both lanes."""
    return f"index for dimension '{dim}' is missing declared lookup column(s) {sorted(missing)} (has {available})"


def declared_index_also_supplied_message(dim: str, declares: str, where: str) -> str:
    """A dimension's index declared in the file and supplied by the caller — both lanes.

    Refused rather than resolved by precedence: a declaration says the file owns
    the fact, and any rule picking a winner lets the file describe a model the
    caller does not build.
    """
    return (
        f"dimension '{dim}' has its index in the file ({declares}) and is also supplied "
        f'under {where}. Exactly one of the two may say what its labels are: drop {where} '
        f'to keep the declaration, or remove {declares} to let the data decide.'
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


def schema_error(exc: Any) -> LanguageError:
    """A pydantic ``ValidationError`` as one of ours, keeping the class.

    Pydantic wraps whatever a validator raises, so our own class cannot reach
    the caller from inside the model — but the original survives under
    ``ctx['error']``, so a :class:`DimensionError` comes back one. Anything
    else, including several errors at once, is a :class:`SchemaError`.
    """
    errors = exc.errors()
    lines = []
    for error in errors:
        message = str(error.get('msg', '')).removeprefix('Value error, ')
        where = '.'.join(str(part) for part in error.get('loc', ()))
        lines.append(f'{where}: {message}' if where else message)
    text = '\n'.join(lines) or str(exc)

    if len(errors) == 1:
        original = errors[0].get('ctx', {}).get('error')
        if isinstance(original, LanguageError):
            return type(original)(text)
    return SchemaError(text)
