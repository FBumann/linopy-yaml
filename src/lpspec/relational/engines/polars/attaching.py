"""What a caller's ``sources`` become: the frames the engine reads by name.

The door (:func:`~lpspec.sources.tidy_sources`) has already read and checked
every source; this gives each the shape the query is written against, and
encodes the string dimensions. Everything downstream reads
:class:`AttachedSources` and nothing else.

**It is frozen, and that is the point.** Written once by the passes below,
then read to construct the compiler and the labeller — unlike the one registry
that is deliberately *live*, the variable frames, which appear as declarations
build and which a constraint compiled afterwards has to see.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping

    from math_spec import program

#: Scratch column carrying a source row's position while first-occurrence
#: order is computed. The spaces make it unrepresentable as a declared name, so
#: it cannot collide with a column the caller's index already has.
_ROW_POSITION = '__row position__'


@dataclass(frozen=True)
class AttachedSources:
    """The data a program is built against, after attaching.

    ``parameters`` are tidy ``(dims…, value)``; ``dimensions`` are
    ``(val, ord)``; ``lookups`` are ``(over, lookup)``, one row per label the
    map is defined at and none for the rest, so "this label maps nowhere" is a
    row that is not there and every operator reading one inherits that from
    its join.

    ``cardinality`` and ``parameter_rows`` are frame heights, cached because
    deriving them later means collecting the frame again.
    """

    parameters: Mapping[str, pl.LazyFrame]
    dimensions: Mapping[str, pl.LazyFrame]
    lookups: Mapping[str, pl.LazyFrame]
    cardinality: Mapping[str, int]
    parameter_rows: Mapping[str, int]

    def is_enum_encoded(self, dim: str) -> bool:
        """Whether *dim* was given an ``Enum`` — answered where the encoding is decided."""
        return self.dimensions[dim].collect_schema()['val'] == pl.Enum


def attach(program: program.Program, sources: Mapping[str, pl.LazyFrame]) -> AttachedSources:
    """Shape the door's frames into what *program* is written against.

    Dimensions first, then lookups, then parameters, then the encoding: a
    dimension's ``Enum`` is built from its labels, and every frame that
    carries the dimension is re-encoded against it.
    """
    dimensions = {d: _ordinal_frame(d, sources[d]).collect() for d in program.dimensions}
    lookups = {name: sources[name].collect() for d in program.dimensions for name in program.dimension(d).maps}
    parameters = {name: sources[name].collect() for name in program.parameters}

    enums = {d: pl.Enum(f['val']) for d, f in dimensions.items() if f.schema['val'] == pl.String}
    for d, enum in enums.items():
        dimensions[d] = dimensions[d].with_columns(pl.col('val').cast(enum))
    for d in program.dimensions:
        targets = program.dimension(d).targets
        for name in program.dimension(d).maps:
            casts = [pl.col(d).cast(enums[d])] if d in enums else []
            target = targets.get(name)
            casts += [pl.col(name).cast(enums[target])] if target in enums else []
            if casts:
                lookups[name] = lookups[name].with_columns(casts)
    for name, p in program.parameters.items():
        frame = _plain_strings(parameters[name], p.dims)
        casts = [pl.col(d).cast(enums[d]) for d in p.dims if d in enums]
        parameters[name] = frame.with_columns(casts) if casts else frame

    return AttachedSources(
        parameters={name: f.lazy() for name, f in parameters.items()},
        dimensions={d: f.lazy() for d, f in dimensions.items()},
        lookups={name: f.lazy() for name, f in lookups.items()},
        cardinality={d: f.height for d, f in dimensions.items()},
        parameter_rows={name: f.height for name, f in parameters.items()},
    )


def _ordinal_frame(d: str, index: pl.LazyFrame) -> pl.LazyFrame:
    """A dimension's ``(val, ord)`` from its index.

    Ordinals follow the source's own order — a label's position is the row it
    first appears at — so a translation moves by position exactly as the eager
    lane does, even for string labels.
    """
    return (
        index.select(d)
        .with_row_index(_ROW_POSITION)
        .group_by(d)
        .agg(pl.col(_ROW_POSITION).min())
        .sort(_ROW_POSITION)
        .with_row_index('ord')
        .select(pl.col(d).alias('val'), pl.col('ord').cast(pl.Int64))
    )


def _plain_strings(frame: pl.DataFrame, dims: tuple[str, ...]) -> pl.DataFrame:
    """Dim columns as plain strings, whatever encoding the source used.

    A dictionary-encoded source carries a writer's own dictionary; decoding it
    first is what lets the strict cast into the dimension's ``Enum`` land.
    """
    categorical = [d for d, dtype in frame.schema.items() if d in dims and dtype in (pl.Categorical, pl.Enum)]
    if not categorical:
        return frame
    return frame.with_columns(pl.col(d).cast(pl.String) for d in categorical)
