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
    from collections.abc import Iterable, Mapping, Sequence


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


def curve_with_a_hole_message(block: str, name: str, shown: str, expected: int, found: int, points: str | None) -> str:
    """A piecewise curve supplied at some of its coordinates — one wording, both lanes.

    A missing row is the one shape whose two readings are both wrong here. The
    absence rules read it as a zero coefficient, which puts a breakpoint at the
    origin that the file never declared; read as a shorter curve it would need
    the weights to shrink with it, which is what ``points:`` says and a bare
    table cannot.

    Which is why the way out depends on whether the block already has a mask.
    Without one the reader wants to hear that a curve may declare its length;
    with one they have said it, and the disagreement is the news — the mask
    claims a breakpoint the values do not carry.
    """
    remedy = (
        f"  Shorten it    '{points}' claims this breakpoint, so either it is one row too long "
        f'or the value is missing\n'
        f'  Or supply it  a value everywhere the mask says the curve runs'
        if points
        else (
            "  Say how far   points: a mask over the curve, true up to each one's last "
            'breakpoint\n'
            '  Or supply it  a value at every coordinate of the axis\n'
            '  Or write it   where the *arity* is data, the λ formulation states it '
            'directly (issue #1101)'
        )
    )
    return (
        f"piecewise '{block}': parameter '{name}' has no value at {shown} — {found} of the "
        f'{expected} coordinates it needs. Every breakpoint the block builds gets a weight, so '
        f'a missing row is not a shorter curve: read as a zero coefficient it is a breakpoint '
        f'at the origin, and the answer mixes onto it.\n{remedy}'
    )


def curve_mask_is_not_contiguous_message(block: str, points: str, over: str, shown: str) -> str:
    """A curve whose breakpoints are not consecutive — one wording, both lanes.

    Where a curve *starts* does not matter: every row that reads the mask asks
    for a predecessor or for an end, and all of those are the curve's own. A
    gap matters twice over — a chord would join across it, and a domain row
    would sit inside the curve rather than at its edge. Both build, and neither
    says what the file does.
    """
    at = f' at {shown}' if shown else ''
    return (
        f"piecewise '{block}': the breakpoints '{points}' marks along '{over}'{at} are not "
        f'consecutive — there is a gap in them, or nothing is marked at all. A curve may sit '
        f'anywhere along the axis, on breakpoints that follow one another.\n'
        f'  Close the gap   a curve is its points and the ones between them\n'
        f'  A curve of none is not a curve: leave the block to the members that have one'
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
        f'array.to_series().reset_index() for a tidy frame, whose columns bind by name on '
        f'both lanes. Result.to_dataarray() is the way back out.'
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


def lookup_not_single_valued_message(dim: str, offenders: Mapping[str, int]) -> str:
    """A label with two lookup values — one wording, both lanes.

    Every offending lookup is named rather than the first, so the rest are not
    found one build at a time. A null counts as a value: a label mapped
    nowhere in one row and somewhere in another does not have one answer
    either, and reading the two as agreeing is what let a member fall out of
    the group that was to hold it.
    """
    listed = '; '.join(f"'{name}' ({count} label(s))" for name, count in sorted(offenders.items()))
    return (
        f"dimension '{dim}' carries more than one value per label for lookup(s): "
        f'{listed}. A lookup is single-valued per label — reduce the source to '
        f'one row per {dim}, or model the relation as a parameter instead.'
    )


def lookup_values_are_not_labels_message(dim: str, lookup: str, target: str, values: Sequence[Any]) -> str:
    """A lookup value naming no label of the dimension it targets — one wording, both lanes.

    A *null* is not one: the label belongs to no group, which is the
    row-absence idiom the rest of the language uses. Only a value that is
    present and unknown is a typo, and that one drops terms in the join that
    places them rather than raising anywhere.
    """
    shown = ', '.join(repr(v) for v in values)
    return (
        f"dimension '{dim}' lookup '{lookup}' has value(s) that are not "
        f"'{target}' labels: {shown}. Every value must be a declared "
        f"'{target}' label — otherwise sum(by={lookup}) drops "
        f'those terms in the join that places them, and the model builds and '
        f'solves without them.'
    )


def multi_indexed_series_message(name: str, dims: Sequence[str]) -> str:
    """A pandas Series carrying its dims in a MultiIndex — one wording, both lanes.

    The index depth is a second statement of how many dimensions the parameter
    has, and it is the caller's rather than the file's, so the two can disagree
    with nothing able to tell which was meant. Columns cannot: a tidy frame
    says it once, in the vocabulary the engine already reads.
    """
    tidy = [*dims, 'value']
    return (
        f"parameter '{name}': a pandas Series with a MultiIndex is not a source. An index is "
        f'a pandas idea with no counterpart in the frames both lanes build, and its depth is a '
        f"second claim about what '{name}' is over. Pass a tidy frame carrying {tidy} — "
        f'series.reset_index() is the whole change.'
    )


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


def map_keys_are_not_labels_message(dim: str, lookup: str, strays: Sequence[str], labels: Sequence[str]) -> str:
    """A declared map keyed by something the caller's index does not carry — both lanes.

    The same law ``Model._declared_lookup_errors`` decides at load where the
    dimension declares its own labels, arriving later because these labels do
    not exist until the caller supplies them. Refused rather than dropped: a key
    matching no label is a typo, and the join that reads the map would silently
    place its terms nowhere.
    """
    shown = ', '.join(strays[:5]) + (' …' if len(strays) > 5 else '')
    return (
        f"lookup '{lookup}' declares values for {shown}, which are not labels of '{dim}'. "
        f"'{dim}' takes its labels from the data here, and they are "
        f'{list(labels[:8])}{" …" if len(labels) > 8 else ""}. A map maps the labels that '
        f'exist — a key matching none of them would place its terms nowhere, so it is a typo '
        f'on one side or a label missing from the other.'
    )


def declared_map_needs_labels_message(dim: str, maps: Iterable[str]) -> str:
    """A dimension whose maps the file declares and whose labels nothing does — both lanes.

    Its own wording rather than the missing-index one, because the fix is not
    "pass a table carrying these lookup columns": those columns are the file's,
    and passing them is refused. Only the labels are wanted.
    """
    declared = ', '.join(f'lookups.{n}.values' for n in sorted(maps))
    return (
        f"dimension '{dim}' has its maps in the file ({declared}) but nothing says which of its "
        f'labels exist. A map is a relation over a dimension, not the dimension itself — it may '
        f'omit members, and its key order is arbitrary. Declare dimensions.{dim}.values, or pass '
        f"the labels under key '{dim}': the declared maps are read against them, and a label no "
        f'map mentions gets a null.'
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
