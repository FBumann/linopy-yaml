"""Bind runtime data to a lowered program — the one door both lanes enter.

The language says what a parameter *is* — its dims, its dtype — and never where
its values come from. This is the other half: what the caller passed (parquet
paths, any table exposing the Arrow PyCapsule protocol, or a plain-Python
shape) becomes the tidy frames both lanes read by name, and every question
about whether that data is usable — is it there, does it carry the declared
columns, is it single-valued per coordinate, are its labels real, are its values
present and of the declared type — is asked here, once.

The guards that need the numbers rather than the shapes are
:mod:`lpspec.curves`, which :func:`tidy_sources` calls on the way through.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from lpspec.curves import derive_curve_sources, validate_curve_extent, validate_piecewise_data
from lpspec.errors import DataError, did_you_mean
from lpspec.frames import as_frame, is_dense_array, is_multi_indexed

if TYPE_CHECKING:
    from math_spec.program import LookupDeclaration, ParameterDeclaration, Program


def attachable(program: Program) -> dict[str, Any]:
    """Every name data may be attached to — declared parameters, dimensions and lookups, one flat namespace.

    A parameter a ``piecewise:`` expansion emitted is not one: it carries a
    derivation saying how it is filled, and
    :func:`~lpspec.curves.derive_curve_sources` fills it, so a caller neither
    can nor need supply it.
    """
    return {
        **{name: p for name, p in program.parameters.items() if p.derivation is None},
        **program.dimensions,
        **{lk.name: lk for _, lk in program.lookups},
    }


def tidy_sources(program: Program, data: Mapping[str, object]) -> dict[str, pl.LazyFrame]:
    """Read the caller's ``sources`` into the frames both lanes build against.

    Every source comes back as an in-memory :class:`polars.LazyFrame`: a
    parameter as tidy ``(dims…, value)``, a dimension's index as the table it
    arrived as with the labels under the dimension's own name, a lookup as the
    ``(over, lookup)`` relation holding one row per label it maps. Dimensions
    are read first, because the plain-Python parameter shapes :func:`_spread`
    accepts are spread over their labels; a ``piecewise:`` block's derived
    parameters are filled next (:func:`derive_curve_sources`), before the loop
    that reads the caller's own.

    Args:
        program: The lowered spec.
        data: Parameter, dimension and lookup names to the caller's tables.

    Raises:
        DataError: A key naming nothing the spec declares; a declared
            dimension, lookup or parameter with no data; a source no reader
            accepts or short of the columns its declaration needs; a parameter
            with two rows for one coordinate, a label its dimension lacks, a
            null or NaN value, or a column of another type than it declares.
    """
    known = attachable(program)
    if unknown := set(data) - set(known):
        raise DataError(unknown_source_keys_message(unknown, known))

    _check_lookup_sources(program, data)
    sources: dict[str, pl.LazyFrame] = {}
    for dname, declared in program.dimensions.items():
        if dname in data:
            sources[dname] = _index(data[dname], dname, declared.dtype)
        elif authors := [f'sources[{n!r}]' for n in declared.maps if n in data]:
            raise DataError(_declared_map_needs_labels_message(dname, authors))
    sources |= _lookup_relations(program, data, sources)

    sources = derive_curve_sources(program, sources, data)
    for pname, pdef in program.parameters.items():
        if pname in sources or pdef.derivation is not None:
            continue
        if pname not in data:
            raise DataError(f"no data provided for parameter '{pname}'")
        sources[pname] = _parameter_frame(pname, pdef, data[pname], sources)
    curved = _curve_shaped(program)
    for pname, pdef in program.parameters.items():
        if pname in sources:
            curve_shaped = pname in curved or pdef.derivation is not None
            sources[pname] = _checked_parameter(pname, pdef, sources[pname], sources, curve_shaped)

    validate_curve_extent(program, sources)
    validate_piecewise_data(program, sources)

    for dname in program.dimensions:
        if dname not in sources:
            raise DataError(no_index_source_message(dname))
    return sources


def _curve_shaped(program: Program) -> frozenset[str]:
    """Every parameter whose completeness a ``piecewise:`` block already decides.

    A block's values parameters and the ``points:`` mask naming how far each
    curve runs. The language refuses ``coverage:`` on these — the block owns
    the shape of its curve — so the door must not ask them for it either;
    :func:`~lpspec.curves.validate_curve_extent` holds them to ``points:``.
    """
    from math_spec.program import Contiguous

    names: set[str] = set()
    for block in program.piecewise.values():
        names.update(block.breakpoints)
        names.update(c.mask for c in block.checks if isinstance(c, Contiguous) and c.mask)
    return frozenset(names)


def unknown_source_keys_message(keys: Iterable[str], known: Iterable[str]) -> str:
    """A source key naming nothing the file declares — a typo, refused rather than ignored."""
    unknown = sorted(keys)
    lead = f'source key {unknown[0]!r} names' if len(unknown) == 1 else f'source keys {unknown} name'
    return (
        f'{lead} neither a parameter, a dimension nor a lookup this spec declares. '
        f'{did_you_mean(unknown[0], known, label="Declared")} Pass only what the '
        f'spec takes — a table carrying more than that is filtered here, not attached.'
    )


def no_index_source_message(dim: str) -> str:
    """A dimension with no index — the labels have no other home."""
    return (
        f"dimension '{dim}' has no index: pass its labels under key '{dim}' — a table "
        f'carrying that column, a parquet path, or a bare sequence of them. The index is what '
        f'says which labels exist, and without one a mistyped label is indistinguishable from '
        f'a new one.'
    )


def _declared_map_needs_labels_message(dim: str, authors: Iterable[str]) -> str:
    """A dimension whose maps have an author and whose labels have none."""
    declared = ', '.join(sorted(authors))
    return (
        f"dimension '{dim}' has its maps ({declared}) but nothing says which of its "
        f'labels exist. A map is a relation over a dimension, not the dimension itself — it may '
        f'omit members, and its key order is arbitrary. Pass the labels under key '
        f"'{dim}': the maps are read against them, and a label no "
        f'map mentions gets a null.'
    )


# ---------------------------------------------------------------------------
# dimensions and lookups
# ---------------------------------------------------------------------------


def _index(source: object, dim: str, dtype: str) -> pl.LazyFrame:
    """One dimension's index, read once and held in memory.

    Raises:
        DataError: A table with no column named after the dimension, or labels
            no frame can be made of.
    """
    table = pl.scan_parquet(source) if isinstance(source, (str, Path)) else as_frame(source, (dim,))
    table = table if table is not None else _labels_frame(dim, source, dtype)
    available = table.collect_schema().names()
    if dim not in available:
        raise DataError(
            f"index for dimension '{dim}' is a table without a '{dim}' column (has "
            f'{list(available)}). The label column is named after the dimension.'
        )
    return table.collect().lazy()


#: The declared dimension dtypes as the column an index becomes. Read only
#: when there are no labels to infer from.
_DECLARED: dict[str, pl.DataType] = {
    'int': pl.Int64(),
    'float': pl.Float64(),
    'str': pl.String(),
    'datetime': pl.Datetime('us'),
}


def _labels_frame(dim: str, values: object, dtype: str) -> pl.LazyFrame:
    """A one-column index frame from a plain sequence of labels.

    **An empty index takes the dimension's declared dtype.** polars infers
    ``Null`` from no labels, and a ``Null`` key joins against nothing — so a
    parameter with the right dtype and no rows would fail to attach against
    the dimension it belongs to. An empty index is what a driver that grows
    one starts from.
    """
    try:
        labels: list[Any] = list(values)  # pyrefly: ignore[bad-argument-type]  — `values` is whatever a caller passed
        if not labels:
            return pl.LazyFrame(schema={dim: _DECLARED[dtype]})
        return pl.LazyFrame({dim: labels})
    except (TypeError, pl.exceptions.PolarsError) as exc:
        raise DataError(
            f"index for dimension '{dim}': cannot read labels out of "
            f'{type(values).__name__} — pass a sequence of labels, a table '
            f'polars can read with a {dim!r} column, or a parquet path'
        ) from exc


def _check_lookup_sources(program: Program, data: Mapping[str, object]) -> None:
    """Refuse a lookup nothing supplies, and a lookup column carried on an index.

    The second is refused rather than filtered, unlike every other stray
    column: it is a map somebody meant to supply under its own key.
    """
    for over, lk in program.lookups:
        if lk.name not in data:
            raise DataError(_unsupplied_lookup_message(lk.name, over, lk.target or lk.name))

    for dim in program.dimensions:
        if dim not in data:
            continue
        carried = _column_names(data[dim], dim)
        for name in program.dimensions[dim].maps:
            if name in carried:
                raise DataError(
                    f"index for dimension '{dim}' carries a '{name}' column, and '{name}' is a lookup "
                    f"over '{dim}'. A map is supplied under its own key, not as a column of the index "
                    f'it runs over: pass it as sources[{name!r}], a table of the rows it maps.'
                )


def _unsupplied_lookup_message(lookup: str, over: str, space: str) -> str:
    """A lookup nothing gives a map for — the counterpart of a parameter with no data."""
    return (
        f"no data provided for lookup '{lookup}'. Pass it under key '{lookup}' as a table with "
        f"columns ['{over}', '{space}'] — one row per '{over}' label it maps, and no row for a "
        f'label it does not.'
    )


def _column_names(source: Any, dim: str) -> frozenset[str]:
    """What a supplied index carries, or nothing where it is a bare label sequence."""
    table = pl.scan_parquet(source) if isinstance(source, (str, Path)) else as_frame(source, (dim,))
    return frozenset(table.collect_schema().names()) if table is not None else frozenset()


def _lookup_relations(
    program: Program, data: Mapping[str, object], indices: Mapping[str, pl.LazyFrame]
) -> dict[str, pl.LazyFrame]:
    """Every lookup's map as the ``(over, lookup)`` relation both lanes read.

    Rows only where the map is defined — a label it leaves out simply has none.
    The keys are checked against ``over``'s labels and, for a lookup with a
    target, the values against the target's: a stray on either side would
    place terms nowhere, silently.

    Raises:
        DataError: A relation short of either column, carrying a null in one,
            mapping a label twice, keyed by a label its dimension lacks, or
            holding a value that is not a label of the dimension it targets.
    """
    relations: dict[str, pl.LazyFrame] = {}
    for over, lk in program.lookups:
        rows = _read_relation(data[lk.name], lk.name, over, lk.target or lk.name)
        _check_keys_are_labels(rows, lk.name, over, _labels_of(over, indices[over]))
        _check_map_covers_its_labels(rows, lk, over, _labels_of(over, indices[over]))
        if lk.target is not None:
            if lk.target not in indices:
                raise DataError(
                    f"dimension '{over}' lookup '{lk.name}' targets '{lk.target}', which nothing in this "
                    f"spec spans and which has no index of its own, so the lookup's values have no label "
                    f"set to be checked against. Pass an index for '{lk.target}' under that key in "
                    f'sources, or remove the lookup.'
                )
            _check_values_are_labels(rows, over, lk.name, lk.target, _labels_of(lk.target, indices[lk.target]))
        relations[lk.name] = rows
    return relations


def _check_values_are_labels(rows: pl.LazyFrame, over: str, lookup: str, target: str, labels: pl.Series) -> None:
    """Refuse a map holding a value *target* does not have as a label.

    Offenders keep their own type — a python native off polars, never a numpy
    scalar — because the message reprs them.
    """
    known = set(labels.to_list())
    seen: dict[Any, None] = {v: None for v in rows.select(lookup).collect()[lookup].to_list() if v not in known}
    if seen:
        shown = ', '.join(repr(v) for v in list(seen)[:5])
        raise DataError(
            f"dimension '{over}' lookup '{lookup}' has value(s) that are not "
            f"'{target}' labels: {shown}. Every value must be a declared "
            f"'{target}' label — otherwise sum(by={lookup}) drops "
            f'those terms in the join that places them, and the model builds and '
            f'solves without them.'
        )


def _labels_of(dim: str, index: pl.LazyFrame) -> pl.Series:
    """One dimension's labels, for a check that is about to run against them."""
    return index.select(dim).collect()[dim]


def _check_keys_are_labels(rows: pl.LazyFrame, lookup: str, over: str, labels: pl.Series) -> None:
    """Refuse a map keyed by anything *over* does not have as a label.

    A label no map mentions is the partial case and simply has no row; a key
    naming no label is a typo.
    """
    known = set(labels.to_list())
    keys = rows.select(over).collect()[over].to_list()
    if strays := sorted(str(x) for x in keys if x not in known):
        shown = ', '.join(strays[:5]) + (' …' if len(strays) > 5 else '')
        spelled = [str(x) for x in labels.to_list()]
        raise DataError(
            f"lookup '{lookup}' maps {shown}, which are not labels of '{over}'. "
            f"'{over}' takes its labels from the data here, and they are "
            f'{spelled[:8]}{" …" if len(spelled) > 8 else ""}. A map maps the labels that '
            f'exist — a key matching none of them would place its terms nowhere, so it is a typo '
            f'on one side or a label missing from the other.'
        )


def _check_map_covers_its_labels(rows: pl.LazyFrame, lk: LookupDeclaration, over: str, labels: pl.Series) -> None:
    """Refuse a ``total`` map that leaves a label of *over* out.

    The declaration is what separates a deliberate open end from a port nobody
    wired: both are a label with no row, and they build the same model. Only a
    map that groups has this to answer for — a label space is selected on, and
    the language refuses ``coverage:`` on one.
    """
    if lk.target is None or lk.coverage != 'total':
        return
    mapped = set(rows.select(over).collect()[over].to_list())
    if missing := sorted(str(x) for x in labels.to_list() if x not in mapped):
        shown = ', '.join(missing[:5]) + (' …' if len(missing) > 5 else '')
        raise DataError(
            f"lookup '{lk.name}' must map every label of '{over}', and {len(missing)} have no row: "
            f'{shown}. A label the map leaves out belongs to no group, so sum(by='
            f'{lk.name}) places its terms nowhere and the model solves without them. Add the '
            f"missing row(s), or declare `coverage: masked` on '{lk.name}' if the gap is meant."
        )


def _read_relation(source: object, lookup: str, over: str, space: str) -> pl.LazyFrame:
    """One supplied relation, read and held to the rules a map has."""
    table = pl.scan_parquet(source) if isinstance(source, (str, Path)) else as_frame(source, (over, space))
    if table is None:
        raise DataError(
            f"lookup '{lookup}': cannot adapt {type(source).__name__} to a table — pass any "
            f"table polars can read with columns ['{over}', '{space}'] (polars, pyarrow, "
            f'pandas), or a parquet path.'
        )
    available = table.collect_schema().names()
    if any(c not in available for c in (over, space)):
        raise DataError(
            f"lookup '{lookup}' is supplied as a relation and must carry columns "
            f"['{over}', '{space}'] (has {list(available)}). '{over}' is the dimension it runs over "
            f"and '{space}' is what its values are labels of."
        )
    rows = table.select(over, pl.col(space).alias(lookup)).collect()

    holes = rows.filter(pl.col(over).is_null() | pl.col(lookup).is_null())
    if holes.height:
        shown = coordinates_shown([over], holes.select(over).head(5).rows())
        at = f': {shown}' if shown else ''
        raise DataError(
            f"lookup '{lookup}' carries {holes.height} row(s) with a null in '{space}'{at}. A map is "
            f'partial by leaving a label out, not by mapping it to nothing — drop the row and the '
            f'label is unmapped, which is what every operator reading the lookup already means by it.'
        )

    twice = rows.group_by(over).len().filter(pl.col('len') > 1).sort(over)
    if twice.height:
        offenders = [str(x) for x in twice[over]]
        shown = ', '.join(offenders[:5]) + (' …' if len(offenders) > 5 else '')
        raise DataError(
            f"lookup '{lookup}' maps {len(offenders)} '{over}' label(s) more than once: {shown}. "
            f'A lookup is single-valued, so each label it maps takes exactly one row.'
        )

    return rows.lazy()


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


def _parameter_frame(
    name: str, p: ParameterDeclaration, obj: object, sources: Mapping[str, pl.LazyFrame]
) -> pl.LazyFrame:
    """The caller's object for one parameter as a lazy frame, whatever shape it took.

    Raises:
        DataError: A shape neither a table reader nor :func:`_spread` accepts.
    """
    if isinstance(obj, (str, Path)):
        return pl.scan_parquet(obj)
    if is_dense_array(obj):
        raise DataError(
            f"parameter '{name}': an xarray.DataArray is not a source. lpspec reads tables — "
            f'rows under named columns — and hands arrays back rather than taking them. Pass '
            f'array.to_series().reset_index() for a tidy frame, whose columns attach by name on '
            f'both lanes. Result.to_dataarray() is the way back out.'
        )
    if is_multi_indexed(obj):
        raise DataError(
            f"parameter '{name}': a pandas Series with a MultiIndex is not a source. An index is "
            f'a pandas idea with no counterpart in the frames both lanes build, and its depth is a '
            f"second claim about what '{name}' is over. Pass a tidy frame carrying {[*p.dims, 'value']} — "
            f'series.reset_index() is the whole change.'
        )
    table = as_frame(obj, p.dims)
    return table if table is not None else _spread(name, obj, p.dims, sources)


def _spread(name: str, obj: object, dims: Sequence[str], sources: Mapping[str, pl.LazyFrame]) -> pl.LazyFrame:
    """A parameter written as plain Python, spread over the dims it declares.

    Three shapes a hand-written spec reaches for and no table library
    produces: a ``{label: value}`` map, a sequence in the dimension's own label
    order, and one number standing for every coordinate. A bool stays boolean
    rather than widening to float: a mask's truthiness is read off the column
    type.

    Raises:
        DataError: A shape that does not fit the declared dims, a sequence
            whose length does not match, or a dimension whose labels nothing
            supplies — a positional shape cannot be placed without them.
    """
    if isinstance(obj, Mapping):
        if len(dims) != 1:
            raise DataError(_wrong_rank(name, 'a dict maps one label to one value', dims))
        return pl.LazyFrame({dims[0]: list(obj.keys()), 'value': list(obj.values())})

    if isinstance(obj, bool):
        return _broadcast(name, pl.lit(obj, dtype=pl.Boolean), dims, sources)
    if isinstance(obj, (int, float)):
        return _broadcast(name, pl.lit(float(obj), dtype=pl.Float64), dims, sources)

    if hasattr(obj, '__len__') and not isinstance(obj, (str, bytes)):
        if len(dims) != 1:
            raise DataError(_wrong_rank(name, 'a sequence runs along one dimension', dims))
        labels = _labels(name, dims[0], sources)
        values = list(obj)  # pyrefly: ignore[bad-argument-type]  — narrowed by the __len__ test
        if len(values) != len(labels):
            raise DataError(
                f"parameter '{name}': {len(values)} values against {len(labels)} "
                f"'{dims[0]}' labels. A sequence is positional, so it must have "
                f'one entry per label, in the order the index declares them.'
            )
        return pl.LazyFrame({dims[0]: labels, 'value': values})

    raise DataError(
        f"parameter '{name}': cannot adapt {type(obj).__name__} to a tidy "
        f'table — pass any table polars can read with columns '
        f'{[*dims, "value"]} (polars, pyarrow, pandas), a parquet path, or the '
        f'plain-Python shapes: a dict, a sequence, or one number.'
    )


def _wrong_rank(name: str, said: str, dims: Sequence[str]) -> str:
    """One wording for a plain-Python shape against the dims it cannot cover."""
    return (
        f"parameter '{name}': {said}, and '{name}' is over {dims}. Pass a table "
        f'with columns {[*dims, "value"]} instead.'
    )


def _broadcast(name: str, value: pl.Expr, dims: Sequence[str], sources: Mapping[str, pl.LazyFrame]) -> pl.LazyFrame:
    """One number over every coordinate of *dims* — a cross join, since nothing downstream broadcasts."""
    frame = pl.LazyFrame({'__one__': [0]})
    for dim in dims:
        frame = frame.join(pl.LazyFrame({dim: _labels(name, dim, sources)}), how='cross')
    return frame.drop('__one__').with_columns(value.alias('value'))


def _labels(name: str, dim: str, sources: Mapping[str, pl.LazyFrame]) -> list[Any]:
    """*dim*'s labels, in index order, for a shape that has none of its own.

    Raises:
        DataError: Nothing supplies the labels. A positional shape carries
            none, and no lane reads them off the parameters.
    """
    source = sources.get(dim)
    if source is None:
        raise DataError(
            f"parameter '{name}' is written positionally over '{dim}', so it says what the "
            f'values are but not what they are labelled — and nothing else supplies an index '
            f"for '{dim}'. Pass '{dim}': [...] in sources, or pass '{name}' as a table "
            f"carrying its own '{dim}' column."
        )
    return source.select(dim).unique(maintain_order=True).collect()[dim].to_list()


def _checked_parameter(
    name: str,
    p: ParameterDeclaration,
    table: pl.LazyFrame,
    sources: Mapping[str, pl.LazyFrame],
    curved: bool = False,
) -> pl.LazyFrame:
    """*table* held to what its declaration claims, collected once.

    The collect is the one model-sized materialisation on the way in; a
    parquet path is streamed through it.

    Raises:
        DataError: The frame lacks a declared dim or ``value``, holds two rows
            for one coordinate or a label its dimension lacks, carries a null
            or NaN value, or types the column differently from the declaration.
    """
    wanted = [*p.dims, 'value']
    available = table.collect_schema().names()
    if missing := set(wanted) - set(available):
        raise DataError(
            f"source for parameter '{name}' is missing columns {sorted(missing)} "
            f"(need dims {list(p.dims)} plus 'value'; has {available}). Rename them to "
            f'the declared dims, or drop the index names to attach positionally.'
        )
    frame = table.select(wanted).collect(engine='streaming')
    _check_one_row_per_coordinate(name, p, frame, sources)
    if not curved:
        _check_covers_its_coordinates(name, p, frame, sources)
    _check_values_are_present(name, p, frame)
    _check_value_dtype(name, p, frame)
    return frame.lazy()


def _check_one_row_per_coordinate(
    name: str, p: ParameterDeclaration, frame: pl.DataFrame, sources: Mapping[str, pl.LazyFrame]
) -> None:
    """A parameter is a function of its dims: one row per coordinate, every label a real one.

    Labels are checked against the dimensions whose index has been read; one
    still missing is refused once every source is in. A parameter with no dims has exactly one coordinate, so the rule reads as
    "exactly one row" — and a second row would silently multiply every row it
    broadcasts into. Every cheap question runs in one pass; naming an offender
    costs a pass of its own and runs only on a path about to raise.
    """
    if not p.dims:
        if frame.height != 1:
            raise DataError(
                f"parameter '{name}' is declared with no dims, which means one value "
                f'broadcast everywhere — but its source has {frame.height} rows. '
                f'Declare the dims it is indexed by, or reduce the source to a single row.'
            )
        return

    known = {d: _labels_of(d, sources[d]) for d in p.dims if d in sources}
    answers = frame.select(
        pl.struct(p.dims).is_duplicated().any().alias('#duplicated'),
        *(pl.col(d).is_in(labels.implode()).all().alias(f'#known {d}') for d, labels in known.items()),
    ).row(0, named=True)

    for d, labels in known.items():
        if not answers[f'#known {d}']:
            strangers = frame.filter(~pl.col(d).is_in(labels.implode())).select(pl.col(d).unique())[d].to_list()
            shown = ', '.join(repr(s) for s in strangers[:5])
            more = f' (and {len(strangers) - 5} more)' if len(strangers) > 5 else ''
            raise DataError(
                f"parameter '{name}' has label(s) in dimension '{d}' that are not coordinates "
                f'of it: {shown}{more}.\n'
                f'  {d} has: {sorted(str(k) for k in labels.to_list())[:10]}\n'
                f'A missing row is a zero coefficient, but a label that is not a coordinate is a '
                f'typo: its row joins nothing, so the coordinate it was meant for silently reads '
                f'as absent. Fix the label, or declare it as a coordinate.'
            )

    if not answers['#duplicated']:
        return
    duplicated = frame.group_by(p.dims).agg(pl.len().alias('#rows')).filter(pl.col('#rows') > 1).head(3)
    shown = '; '.join(
        ', '.join(f'{d}={row[d]!r}' for d in p.dims) + f' ({row["#rows"]} rows)'
        for row in duplicated.iter_rows(named=True)
    )
    raise DataError(
        f"parameter '{name}' has more than one row for a coordinate: {shown}. "
        f'A parameter is a function of its dims, so which value applies is undefined — '
        f'aggregate the source to one row per {list(p.dims)} before attaching it.'
    )


def _check_covers_its_coordinates(
    name: str, p: ParameterDeclaration, frame: pl.DataFrame, sources: Mapping[str, pl.LazyFrame]
) -> None:
    """Refuse a ``total`` table short of a coordinate its dims reach.

    The counts settle it: :func:`_check_one_row_per_coordinate` has already
    refused duplicates and strangers, so the height *is* the number of
    coordinates covered and no pass over the source is needed. Naming the
    missing ones costs the grid and runs only on the path about to raise.

    A dim whose index has not been read is left alone — :func:`tidy_sources`
    refuses that once every source is in, and a reach computed without it would
    be a different number. A curve's values parameter is left alone too: the
    block owns how far its curve runs, which is why the language refuses
    ``coverage:`` on one, and :func:`validate_curve_extent` holds it to
    ``points:`` instead. So is a parameter the expansion derived: how far it
    runs is the block's answer, and no file declared it at all.
    """
    if p.coverage != 'total' or not p.dims or any(d not in sources for d in p.dims):
        return
    labels = {d: _labels_of(d, sources[d]) for d in p.dims}
    reach = math.prod(len(s) for s in labels.values())
    if frame.height >= reach:
        return

    grid = pl.LazyFrame({p.dims[0]: labels[p.dims[0]]})
    for d in p.dims[1:]:
        grid = grid.join(pl.LazyFrame({d: labels[d]}), how='cross')
    absent = grid.join(frame.lazy().select(p.dims), on=list(p.dims), how='anti').head(3).collect()
    raise DataError(
        f"parameter '{name}' must carry every coordinate its dims reach, and its source has "
        f'{frame.height} row(s) for {reach} coordinate(s). Missing: '
        f'{coordinates_shown(p.dims, absent.rows())}{" …" if reach - frame.height > 3 else ""}. '
        f'A complete table is what `coverage: total` claims, so a row lost in preparation would '
        f'otherwise read as a deliberate mask and build a different model. Supply the missing '
        f"row(s), or declare `coverage: masked` on '{name}' if the gap is meant."
    )


def _check_values_are_present(name: str, p: ParameterDeclaration, frame: pl.DataFrame) -> None:
    """Every row carries a value: a null or a NaN is refused rather than read.

    In long form the absence of a value is the absence of the row, so a hole
    claims the coordinate and denies it at once; the two readings build
    different models. NaN is named beside null because pandas spells both NaN.
    """
    value = pl.col('value')
    holed = value.is_null() | value.is_nan() if frame.schema['value'].is_float() else value.is_null()
    holes = int(frame.select(holed.sum()).item())
    if not holes:
        return
    shown = coordinates_shown(p.dims, frame.filter(holed).select(p.dims).head(3).rows()) if p.dims else ''
    at = f': {shown}' if shown else ''
    raise DataError(
        f"parameter '{name}' carries {holes} row(s) with no value — null or NaN{at}. "
        f'In long form the absence of a value is the absence of the row, and such a row '
        f'says the coordinate exists and denies it in the same breath.\n'
        f'  Drop them     polars .drop_nulls("value").drop_nans("value"), pandas .dropna(subset=["value"])\n'
        f'  Supply them   if a number was what was meant'
    )


def coordinates_shown(dims: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Coordinates as a refusal prints them: ``f='b'; f='c'``."""
    return '; '.join(', '.join(f'{d}={v!r}' for d, v in zip(dims, row, strict=True)) for row in rows)


#: The column each declared dtype *is*, in polars types.
_COLUMNS: Mapping[str, tuple[type[pl.DataType], ...]] = {
    'float': (pl.Float32, pl.Float64),
    'int': (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64),
    'bool': (pl.Boolean,),
    'str': (pl.String, pl.Categorical, pl.Enum),
}

#: What each declared dtype accepts. ``int`` serving ``float`` is the one
#: widening: it is the only conversion between two declared types that loses
#: nothing. A float column under ``int`` is refused, which is what makes a
#: fractional offset unrepresentable rather than checked.
ACCEPTED_VALUE_TYPES: Mapping[str, tuple[type[pl.DataType], ...]] = {
    **_COLUMNS,
    'float': _COLUMNS['float'] + _COLUMNS['int'],
}


def _check_value_dtype(name: str, p: ParameterDeclaration, frame: pl.DataFrame) -> None:
    """The attached column is the type the declaration claims.

    Asked after the holes, so a column of nothing but nulls — which polars
    types ``Null`` — is told it has no values rather than the wrong kind.
    """
    column = frame.schema['value']
    if column in ACCEPTED_VALUE_TYPES[p.dtype]:
        return
    arrived = next((name for name, types in _COLUMNS.items() if column in types), str(column))
    raise DataError(
        f"parameter '{name}' is declared '{p.dtype}' and its values arrived as '{arrived}'. "
        f'A declared dtype is a claim about the values, and it is checked here — the file '
        f'says what the column is, or the column is not attached.\n'
        f'  Cast the column to {p.dtype}, if the declaration is what you meant\n'
        f'  Or declare what the data has: {{dtype: {arrived}}}'
    )
