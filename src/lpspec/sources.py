"""Bind runtime data to a validated schema.

The language says what a parameter *is* — its dims, its dtype — and never where
its values come from. This is the other half: what the caller passed (parquet
paths, or any table exposing the Arrow PyCapsule protocol) becomes the tidy
frames the engine reads by name. The shapes themselves are recognised in
:mod:`lpspec.frames`, so no dataframe library beyond the engine's
own is a dependency of either lane.

Not lowering, which turns an AST into a plan and touches no data; this touches
only data and knows nothing about expressions.

The guards that need the numbers rather than the shapes are
:mod:`lpspec.curves`, which :func:`tidy_sources` calls on the way through.

Every ``*_message`` helper here is worded once and raised by both lanes, so a
caller fixing a defect reads one sentence whichever lane found it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from lpspec.curves import (
    derive_curve_edges,
    derive_curve_masks,
    validate_curve_extent,
    validate_piecewise_data,
)
from lpspec.errors import (
    DataError,
    coordinates_shown,
    did_you_mean,
    index_without_its_label_column_message,
)
from lpspec.frames import TidySource, as_frame, is_dense_array, is_multi_indexed, labels_frame, scan

if TYPE_CHECKING:
    from math_spec import Model


def bindable(schema: Model) -> dict[str, Any]:
    """Every name data may be bound to — parameters, dimensions and lookups, one flat namespace."""
    return {**schema.parameters, **schema.dimensions, **schema.lookups}


def tidy_sources(schema: Model, data: Mapping[str, object]) -> dict[str, TidySource]:
    """Adapt the caller's ``sources`` mapping to engine sources.

    Every in-memory source becomes a tidy :class:`polars.LazyFrame` with columns
    ``(dims…, value)``; parquet paths pass through untouched for the engine to
    scan directly. Normalising here rather than at the engine is what lets the
    piecewise curvature guard see every in-memory shape alike.

    **Dimensions are resolved first** — ownership included, so a map with two
    authors is refused before either author's table is read — because the
    plain-Python parameter shapes :func:`_spread` accepts are meaningless
    without the labels they are spread over. A ``piecewise:`` block's derived
    flags come next, before the parameter loop asks for data no caller can
    have (:func:`derive_curve_edges`).

    Whether a *parameter* source carries the columns its declaration needs is
    *not* asked here — binding asks it of every source, path or frame.

    A **lookup** comes back under its own name too, as the ``(over, lookup)``
    relation :func:`lookup_relations` checked — the rows it maps and no others.

    Raises:
        DataError: A key naming nothing the model declares, a declared parameter
            with no data, or one bound to something neither a tidy table nor
            :func:`_spread` can read.
    """
    known = bindable(schema)
    if unknown := set(data) - set(known):
        raise DataError(_unknown_source_keys_message(unknown, known))

    indices = dimension_sources(schema, data)
    sources: dict[str, TidySource] = {
        dname: polars_index(source, dname, schema.dimensions[dname].dtype) for dname, source in indices.items()
    }
    sources |= lookup_relations(schema, data, sources)

    sources = derive_curve_edges(schema, derive_curve_masks(schema, sources, data), data)

    for pname, pdef in schema.parameters.items():
        if pname in sources:
            continue
        if pname not in data:
            raise DataError(f"no data provided for parameter '{pname}'")
        obj = data[pname]
        if isinstance(obj, (str, Path)):
            sources[pname] = obj
            continue
        if is_dense_array(obj):
            raise DataError(_dense_array_message(pname))
        if is_multi_indexed(obj):
            raise DataError(_multi_indexed_series_message(pname, pdef.dims))
        table = as_frame(obj, pdef.dims)
        sources[pname] = table if table is not None else _spread(pname, obj, pdef.dims, sources)

    validate_curve_extent(schema, sources)
    validate_piecewise_data(schema, sources)

    return sources


def _dense_array_message(name: str) -> str:
    """A dense array where a table belongs.

    An ``xarray.DataArray`` has ``__len__``, so left alone it would be read
    positionally on one lane and directly on the other.
    """
    return (
        f"parameter '{name}': an xarray.DataArray is not a source. lpspec reads tables — "
        f'rows under named columns — and hands arrays back rather than taking them. Pass '
        f'array.to_series().reset_index() for a tidy frame, whose columns bind by name on '
        f'both lanes. Result.to_dataarray() is the way back out.'
    )


def _multi_indexed_series_message(name: str, dims: Sequence[str]) -> str:
    """A pandas Series carrying its dims in a MultiIndex.

    The index depth is a second statement of the parameter's arity, and the
    two can disagree with nothing able to tell which was meant.
    """
    tidy = [*dims, 'value']
    return (
        f"parameter '{name}': a pandas Series with a MultiIndex is not a source. An index is "
        f'a pandas idea with no counterpart in the frames both lanes build, and its depth is a '
        f"second claim about what '{name}' is over. Pass a tidy frame carrying {tidy} — "
        f'series.reset_index() is the whole change.'
    )


def _unknown_source_keys_message(keys: Iterable[str], known: Iterable[str]) -> str:
    """A source key naming nothing the file declares.

    Refused rather than ignored: a name nothing recognises is a typo, and
    ignoring one silently re-solves the numbers it meant to replace.
    """
    unknown = sorted(keys)
    lead = f'source key {unknown[0]!r} names' if len(unknown) == 1 else f'source keys {unknown} name'
    return (
        f'{lead} neither a parameter, a dimension nor a lookup this model declares. '
        f'{did_you_mean(unknown[0], known, label="Declared")} Pass only what the '
        f'model takes — a table carrying more than that is filtered here, not bound.'
    )


def dimension_sources(schema: Model, data: Mapping[str, object]) -> dict[str, object]:
    """Which object supplies each dimension's index — the rule, in one place.

    A key in *data*, or the ``values:`` the YAML declares, never both
    (:func:`check_index_ownership`). What comes back is what the caller or the
    file passed and nothing more — a path, a table, a sequence of labels —
    because each lane reads it into its own frame library.

    A dimension nothing supplies an index for is simply absent: which
    dimensions *need* one is the caller's question. A map does not travel with
    the source — both spellings become the same relation in
    :func:`lookup_relations`, read against the labels this returns.

    Raises:
        DataError: A dimension whose index two authors claim, or whose maps the
            file declares and whose labels nothing does.
    """
    check_index_ownership(schema, data)

    sources: dict[str, object] = {}
    for dname in schema.dimensions:
        if dname in data:
            sources[dname] = data[dname]
        elif authors := map_authors(schema, data, dname):
            raise DataError(_declared_map_needs_labels_message(dname, authors))

    return sources


def _declared_map_needs_labels_message(dim: str, authors: Iterable[str]) -> str:
    """A dimension whose maps have an author and whose labels have none.

    Its own wording rather than the missing-index one: only the labels are
    wanted — the lookup columns belong to whoever already supplies the map.
    """
    declared = ', '.join(sorted(authors))
    return (
        f"dimension '{dim}' has its maps ({declared}) but nothing says which of its "
        f'labels exist. A map is a relation over a dimension, not the dimension itself — it may '
        f'omit members, and its key order is arbitrary. Pass the labels under key '
        f"'{dim}': the maps are read against them, and a label no "
        f'map mentions gets a null.'
    )


def map_authors(schema: Model, data: Mapping[str, object], dimension: str) -> list[str]:
    """Where each map over *dimension* comes from, spelled as the reader wrote it.

    Only used to refuse a dimension whose maps have an author and whose labels
    have none, which is why it names the *author* rather than the lookup: the
    two spellings want different fixes and neither is "pass the column".
    """
    return [f'sources[{n!r}]' for n in sorted(schema.lookups) if n in data and schema.lookups[n].over == dimension]


def lookup_relations(
    schema: Model, data: Mapping[str, object], indices: Mapping[str, TidySource]
) -> dict[str, pl.LazyFrame]:
    """Every lookup's map as the ``(over, lookup)`` relation both lanes read.

    One shape whichever author supplied it: the file's ``values:`` becomes the
    same two columns the caller's own key carries, so nothing downstream asks
    where a map came from. Rows only where the map is defined — a label it
    leaves out simply has none. The value column is renamed to the lookup, the
    name every lane reads a map by.

    **Both label columns are checked here**, against the indices this is handed:
    the keys against ``over``'s, and — for a lookup with a target — the values
    against that dimension's.

    Returns:
        ``{lookup name: (over, lookup) frame}`` for every lookup the model
        declares, since one with no map at all is refused before this runs.

    Raises:
        DataError: A relation short of either column, carrying a null in one,
            mapping a label twice, keyed by a label its dimension lacks, or
            holding a value that is not a label of the dimension it targets.
    """
    relations: dict[str, pl.LazyFrame] = {}
    for name, lookup in sorted(schema.lookups.items()):
        over = lookup.over
        rows = _read_relation(data[name], name, over, lookup.into or name)
        said = 'maps'
        _check_keys_are_labels(rows, name, over, _labels_of(over, indices[over]), said)
        if (target := lookup.into) is not None:
            if target not in indices:
                raise DataError(_lookup_target_without_labels_message(over, name, target))
            _check_values_are_labels(rows, over, name, target, _labels_of(target, indices[target]))
        relations[name] = rows
    return relations


def _lookup_target_without_labels_message(dim: str, lookup: str, target: str) -> str:
    """A lookup whose target has no label set to check against.

    Its own wording because the target dimension may be one no constraint
    spans, so nothing else in the model would ask for its index at all.
    """
    return (
        f"dimension '{dim}' lookup '{lookup}' targets '{target}', which nothing in this model "
        f"spans and which has no index of its own, so the lookup's values have no label set to "
        f"be checked against. Pass an index for '{target}' under that key in sources, or remove "
        f'the lookup.'
    )


def _check_values_are_labels(rows: pl.LazyFrame, over: str, lookup: str, target: str, labels: pl.Series) -> None:
    """Refuse a map holding a value *target* does not have as a label.

    The check that makes ``sum(by=)`` safe: such a value is dropped by the join
    that places its terms, so the model builds and solves with them silently
    missing. Offenders keep their own type — a python native off polars, never
    a numpy scalar — because the message reprs them.
    """
    known = set(labels.to_list())
    seen: dict[Any, None] = {v: None for v in rows.select(lookup).collect()[lookup].to_list() if v not in known}
    if seen:
        raise DataError(_lookup_values_are_not_labels_message(over, lookup, target, list(seen)[:5]))


def _lookup_values_are_not_labels_message(dim: str, lookup: str, target: str, values: Sequence[Any]) -> str:
    """A lookup value naming no label of the dimension it targets.

    A *null* is not one — the label belongs to no group, the row-absence idiom
    the rest of the language uses. Only a value present and unknown is a typo.
    """
    shown = ', '.join(repr(v) for v in values)
    return (
        f"dimension '{dim}' lookup '{lookup}' has value(s) that are not "
        f"'{target}' labels: {shown}. Every value must be a declared "
        f"'{target}' label — otherwise sum(by={lookup}) drops "
        f'those terms in the join that places them, and the model builds and '
        f'solves without them.'
    )


def _labels_of(dim: str, index: TidySource) -> pl.Series:
    """One dimension's labels, for a check that is about to run against them."""
    return scan(index).select(dim).collect()[dim]


def _check_keys_are_labels(rows: pl.LazyFrame, lookup: str, over: str, labels: pl.Series, said: str) -> None:
    """Refuse a map keyed by anything *over* does not have as a label.

    The asymmetry is the point. A label no map mentions is the **partial case**
    and simply has no row; a key naming no label is a **typo**, and dropping it
    would place its terms nowhere while the model built and solved.

    *said* is how the map got here, because the fix differs and the law does
    not: a key in the file is edited there, a row in a table is dropped.
    """
    known = set(labels.to_list())
    keys = rows.select(over).collect()[over].to_list()
    if strays := sorted(str(x) for x in keys if x not in known):
        raise DataError(
            _map_keys_are_not_labels_message(over, lookup, strays, [str(x) for x in labels.to_list()], said)
        )


def _map_keys_are_not_labels_message(
    dim: str, lookup: str, strays: Sequence[str], labels: Sequence[str], said: str
) -> str:
    """A map keyed by something the caller's index does not carry.

    The law ``Model._declared_lookup_errors`` decides at load, arriving later
    because these labels do not exist until the caller supplies them. *said* is
    how the map got here — declared in the file, or supplied as its own
    relation — because the fix differs and the law does not.
    """
    shown = ', '.join(strays[:5]) + (' …' if len(strays) > 5 else '')
    return (
        f"lookup '{lookup}' {said} {shown}, which are not labels of '{dim}'. "
        f"'{dim}' takes its labels from the data here, and they are "
        f'{list(labels[:8])}{" …" if len(labels) > 8 else ""}. A map maps the labels that '
        f'exist — a key matching none of them would place its terms nowhere, so it is a typo '
        f'on one side or a label missing from the other.'
    )


def _read_relation(source: object, lookup: str, over: str, space: str) -> pl.LazyFrame:
    """One supplied relation, read and held to the rules a map has.

    Collected here rather than left lazy: both checks below read every row, and
    a scan re-read per pass is what #273 came from.
    """
    table = pl.scan_parquet(source) if isinstance(source, (str, Path)) else as_frame(source, (over, space))
    if table is None:
        raise DataError(
            f"lookup '{lookup}': cannot adapt {type(source).__name__} to a table — pass any "
            f"table polars can read with columns ['{over}', '{space}'] (polars, pyarrow, "
            f'pandas), or a parquet path.'
        )
    available = table.collect_schema().names()
    if any(c not in available for c in (over, space)):
        raise DataError(_lookup_relation_columns_message(lookup, over, space, available))
    rows = table.select(over, pl.col(space).alias(lookup)).collect()

    holes = rows.filter(pl.col(over).is_null() | pl.col(lookup).is_null())
    if holes.height:
        shown = coordinates_shown([over], holes.select(over).head(5).rows())
        raise DataError(_lookup_relation_holes_message(lookup, space, holes.height, shown))

    twice = rows.group_by(over).len().filter(pl.col('len') > 1).sort(over)
    if twice.height:
        raise DataError(_lookup_relation_not_single_valued_message(lookup, over, [str(x) for x in twice[over]]))

    return rows.lazy()


def _lookup_relation_columns_message(lookup: str, over: str, space: str, available: Sequence[str]) -> str:
    """A supplied lookup relation short of one of its two columns.

    Both names are the declaration's, so the message spells the pair rather
    than asking the reader to derive it.
    """
    return (
        f"lookup '{lookup}' is supplied as a relation and must carry columns "
        f"['{over}', '{space}'] (has {list(available)}). '{over}' is the dimension it runs over "
        f"and '{space}' is what its values are labels of."
    )


def _lookup_relation_holes_message(lookup: str, column: str, holes: int, shown: str) -> str:
    """A supplied lookup relation with a null in it.

    A partial map is rows for the labels it maps and no row for the rest, so a
    null says the label is mapped and unmapped at once.
    """
    at = f': {shown}' if shown else ''
    return (
        f"lookup '{lookup}' carries {holes} row(s) with a null in '{column}'{at}. A map is "
        f'partial by leaving a label out, not by mapping it to nothing — drop the row and the '
        f'label is unmapped, which is what every operator reading the lookup already means by it.'
    )


def _lookup_relation_not_single_valued_message(lookup: str, over: str, offenders: Sequence[str]) -> str:
    """A supplied lookup relation giving one label two values.

    Refused rather than resolved: a second row would multiply the label's
    terms through the join that reads it.
    """
    shown = ', '.join(offenders[:5]) + (' …' if len(offenders) > 5 else '')
    return (
        f"lookup '{lookup}' maps {len(offenders)} '{over}' label(s) more than once: {shown}. "
        f'A lookup is single-valued, so each label it maps takes exactly one row.'
    )


def polars_index(source: object, dim: str, dtype: str) -> pl.LazyFrame:
    """One dimension's index, read once — the only read of one in the package.

    A lane wanting another library **converts this frame** rather than reading
    the source a second time, which is what makes a caller's choice of
    dataframe invisible past this line. A parquet path stays lazy, so the
    engine still hands the file to the query it builds.

    Raises:
        DataError: A table with no column named after the dimension, or labels
            no frame can be made of.
    """
    table = pl.scan_parquet(source) if isinstance(source, (str, Path)) else as_frame(source, (dim,))
    table = table if table is not None else labels_frame(dim, source, dtype)
    available = table.collect_schema().names()
    if dim not in available:
        raise DataError(index_without_its_label_column_message(dim, available))
    return table


def _spread(name: str, obj: object, dims: list[str], sources: Mapping[str, object]) -> pl.LazyFrame:
    """A parameter written as plain Python, spread over the dims it declares.

    Three shapes a hand-written model reaches for and no table library
    produces: a ``{label: value}`` map, a sequence in the dimension's own label
    order, and one number standing for every coordinate. Each is dense by
    construction, so each is materialised here — which is the cost of writing a
    parameter out in Python rather than handing over a table, and the reason a
    value that really is constant is better declared ``dims: []``.

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


def _wrong_rank(name: str, said: str, dims: list[str]) -> str:
    """One wording for a plain-Python shape against the dims it cannot cover."""
    return (
        f"parameter '{name}': {said}, and '{name}' is over {dims}. Pass a table "
        f'with columns {[*dims, "value"]} instead.'
    )


def _broadcast(name: str, value: pl.Expr, dims: list[str], sources: Mapping[str, object]) -> pl.LazyFrame:
    """One number over every coordinate of *dims*.

    The cross join is what makes it a table; nothing downstream can broadcast,
    a parameter frame carrying one row per coordinate by contract.
    """
    frame = pl.LazyFrame({'__one__': [0]})
    for dim in dims:
        frame = frame.join(pl.LazyFrame({dim: _labels(name, dim, sources)}), how='cross')
    return frame.drop('__one__').with_columns(value.alias('value'))


def _labels(name: str, dim: str, sources: Mapping[str, object]) -> list[Any]:
    """*dim*'s labels, in index order, for a shape that has none of its own.

    Read back off the dimension source this call already built, which is why
    the dimensions are resolved first.

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
    return scan(source).select(dim).unique(maintain_order=True).collect()[dim].to_list()


def check_index_ownership(schema: Model, data: Mapping[str, object]) -> None:
    """Refuse an index nothing supplies, and a lookup column smuggled onto one.

    Two facts, each with one home, and the home is the data: **the labels** —
    which members exist, and in what order — come from the caller's ``d``
    source, and **each lookup's map** from the lookup's own source key. The
    file declares that they exist and what type they are; what they *are* is
    the data's to say.

    A map is deliberately *not* a claim on the labels: it is a partial
    relation over the dimension, free to omit members and written in whatever
    key order someone typed. A sparse map over a
    caller's label set is therefore the one index with two authors — one per
    fact — and it is the shape this check exists to keep honest.

    An index carrying a column named after a lookup over it is refused rather
    than filtered: every other stray column is a dump's extra, and this one is
    a map someone meant to supply.

    Raises:
        DataError: Naming the dimension or the lookup, and both authors — or
            neither, where a map has none.
    """
    for name, lookup in sorted(schema.lookups.items()):
        if name not in data:
            raise DataError(_unsupplied_lookup_message(name, lookup.over, lookup.into or name))

    for dim in schema.dimensions:
        if dim not in data:
            continue
        carried = _column_names(data[dim], dim)
        for name, lookup in sorted(schema.lookups.items()):
            if lookup.over == dim and name in carried:
                raise DataError(_lookup_column_on_an_index_message(dim, name))


def _lookup_column_on_an_index_message(dim: str, lookup: str) -> str:
    """An index carrying a column named after a lookup over it.

    Refused rather than filtered away, unlike every other stray column: this
    one is a map somebody meant to supply.
    """
    return (
        f"index for dimension '{dim}' carries a '{lookup}' column, and '{lookup}' is a lookup "
        f"over '{dim}'. A map is supplied under its own key, not as a column of the index it "
        f'runs over: pass it as sources[{lookup!r}], a table of the rows it maps.'
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
    table = scan(source) if isinstance(source, (str, Path)) else as_frame(source, (dim,))
    return frozenset(table.collect_schema().names()) if table is not None else frozenset()
