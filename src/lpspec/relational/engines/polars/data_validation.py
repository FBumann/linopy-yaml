"""Is the bound data usable? One place, for this lane.

The split this module makes explicit (#351): **spec validation**
(`math_spec.validation`) is everything decidable from the file alone and is
where `check()` happens; **data validation** is here — is it there, can it be
read, is it single-valued per coordinate, are its labels real. The two
positions where law 8 grants no default (a divisor, a bound) stay with the
assembly, needing the matrix.

Every function is a pure question over frames and declarations, holding no
engine state, so what counts as usable data can be read without following the
build.

**Scoped to this lane on purpose.** These take tidy polars frames; the eager
lane reads pandas/xarray natively, so it keeps its own checks in
`linopy/loader.py` rather than paying a copy of every parameter to adapt.
The lanes share the *wording* (`lpspec/errors.py`) and the *contract* —
`tests/test_data_parity.py` asserts they reach the same verdict on the same bad
data, which is what keeps the duplication honest.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import polars as pl

from lpspec.errors import (
    DataError,
    coordinates_shown,
    duplicate_coordinate_message,
    holes_in_values_message,
    wrong_value_dtype_message,
)

if TYPE_CHECKING:
    from math_spec import program

#: The dimension frames a check reads labels out of, by dimension name. Only the
#: ones already built: a dimension derived *from* the parameters is not here when
#: a parameter is checked, and has nothing to answer.
Dimensions = Mapping[str, pl.LazyFrame]


def check_one_row_per_coordinate(p: program.ParameterDeclaration, frame: pl.LazyFrame, dimensions: Dimensions) -> None:
    """A parameter is a function of its dims: one row per coordinate.

    A parameter with **no dims** has exactly one coordinate, so the rule reads
    as "exactly one row" — the case where breaking it is least visible, since a
    dimensionless parameter broadcasts by joining on nothing, which is correct
    for one row and a silent row multiplication for two: duplicate columns for
    one variable in a bound, duplicate mask rows in a where (#166).

    Labels are checked here too, against dimensions that have an index of their
    own; one derived *from* the parameters is not built yet and would have
    nothing to answer, the union of what arrived being its definition (#350).

    Every cheap question runs in one pass over the source. *Naming* an offender
    costs a pass of its own — the duplicate ``group_by`` being the single most
    expensive step of a large build — so those run only on a path about to
    raise. The aggregate names use ``#`` so they cannot collide with a dim's.
    ``.implode()`` on the membership test says "this whole collection" where
    ``is_in`` against a bare Series is ambiguous and deprecated in polars.
    """
    if not p.dims:
        rows = frame.select(pl.len()).collect().item()
        if rows != 1:
            raise DataError(
                f"parameter '{p.name}' is declared with no dims, which means one value "
                f'broadcast everywhere — but its source has {rows} rows. '
                f'Declare the dims it is indexed by, or reduce the source to a single row.'
            )
        return

    known = {d: dimensions[d].select('val').collect()['val'] for d in p.dims if d in dimensions}
    answers = (
        frame.select(
            pl.struct(p.dims).is_duplicated().any().alias('#duplicated'),
            *(pl.col(d).is_in(labels.implode()).all().alias(f'#known {d}') for d, labels in known.items()),
        )
        .collect()
        .row(0, named=True)
    )

    for d, labels in known.items():
        if not answers[f'#known {d}']:
            strangers = frame.filter(~pl.col(d).is_in(labels.implode())).select(pl.col(d).unique()).collect()
            raise DataError(_unknown_labels_message(p.name, d, strangers[d].to_list(), labels.to_list()))

    if not answers['#duplicated']:
        return
    duplicated = frame.group_by(p.dims).agg(pl.len().alias('#rows')).filter(pl.col('#rows') > 1).head(3).collect()
    shown = '; '.join(
        ', '.join(f'{d}={row[d]!r}' for d in p.dims) + f' ({row["#rows"]} rows)'
        for row in duplicated.iter_rows(named=True)
    )
    raise DataError(duplicate_coordinate_message(p.name, shown, list(p.dims)))


def _unknown_labels_message(name: str, dim: str, strangers: list[object], known: list[object]) -> str:
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


def check_values_are_present(p: program.ParameterDeclaration, frame: pl.LazyFrame) -> None:
    """Every row carries a value: a null or a NaN is refused rather than read.

    One aggregate on the happy path, over a column the binder has already
    collected. Naming the offenders costs a second pass, and runs only where
    the first found something — the same shape the checks above take.

    The NaN half of the question is asked only of a float column: ``is_nan``
    raises on the others, which have no NaN to hold. A null makes the whole
    disjunction true on its own, ``is_nan`` being null there rather than false.
    """
    value = pl.col('value')
    holed = value.is_null() | value.is_nan() if frame.collect_schema()['value'].is_float() else value.is_null()
    holes = int(frame.select(holed.sum()).collect().item())
    if not holes:
        return
    offenders = frame.filter(holed).select(p.dims).head(3).collect().rows() if p.dims else ()
    raise DataError(holes_in_values_message(p.name, holes, coordinates_shown(p.dims, offenders)))


#: The column each declared dtype *is*, in polars types. Pinned to the
#: language's ``PARAMETER_DTYPES`` by ``tests/test_architecture.py`` — a test
#: rather than an import, because the engine may not reach the language.
_COLUMNS: Mapping[str, tuple[type[pl.DataType], ...]] = {
    'float': (pl.Float32, pl.Float64),
    'int': (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64),
    'bool': (pl.Boolean,),
    'str': (pl.String, pl.Categorical, pl.Enum),
}

#: What each declared dtype accepts. ``int`` serving ``float`` is the one
#: widening: whole numbers *are* numbers, it is the only conversion between two
#: declared types that loses nothing, and it is what real instance data looks
#: like. Every other direction is refused — including a float column under
#: ``int``, which is what makes a fractional offset unrepresentable rather than
#: checked.
ACCEPTED_VALUE_TYPES: Mapping[str, tuple[type[pl.DataType], ...]] = {
    **_COLUMNS,
    'float': _COLUMNS['float'] + _COLUMNS['int'],
}


def check_value_dtype(p: program.ParameterDeclaration, frame: pl.LazyFrame) -> None:
    """The bound column is the type the declaration claims.

    A schema comparison rather than a scan: what the file declares is a
    property of the column, so nothing here reads a value. That is also what
    makes it cheap enough to run on every parameter.

    Asked *after* the holes, so a column of nothing but nulls — which polars
    types ``Null``, no dtype at all — is told it has no values rather than that
    it has the wrong kind of them.
    """
    column = frame.collect_schema()['value']
    if column in ACCEPTED_VALUE_TYPES[p.dtype]:
        return
    arrived = next((name for name, types in _COLUMNS.items() if column in types), str(column))
    raise DataError(wrong_value_dtype_message(p.name, p.dtype, arrived))
