"""The ``mps_file`` sink: the model as MPS text.

The format the other half of the world reads, and it differs from
:mod:`~lpspec.relational.sinks.writers.lp_file` in one way that shapes the
whole module: **MPS is column-major.** It hands a reader each column with its
whole column of the matrix, where LP walks the matrix by row. So this is the
one writer that sorts — CSR is row-major, and no engine frame holds a column
index — and the sort is what its peak is spent on.

The names are the LP writer's, so the two files describe one model to a reader
holding both: ``x0`` a column, ``c0`` a row, ``s0`` a set.

**Every section is written in label order**, for #109's reason.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, TYPE_CHECKING

import numpy as np
import polars as pl

from lpspec.relational import chunking
from lpspec.relational.sinks.capabilities import Capabilities
from lpspec.relational.sinks.tables import SENSE_CODES
from lpspec.relational.sinks.writers.base import chunk_key, digits, number, sink

if TYPE_CHECKING:
    import numpy.typing as npt

    from lpspec.relational.sinks.tables import Tables


#: The sections this writer emits, and nothing beyond them. Where
#: :data:`~lpspec.relational.sinks.writers.lp_file.LP_FILE_CAPABILITIES` carries
#: every construct because a section is text, MPS spells a quadratic term in an
#: extension section this writer does not write, so a model that needs one is
#: refused by name rather than written without it.
MPS_FILE_CAPABILITIES = Capabilities(supports={'integrality': 'native', 'sos': 'native'})


#: How MPS spells each comparison, read off the engine's own vocabulary so a
#: sense added there raises here at import rather than being written as
#: whatever the table last held.
_MPS_SENSE = {sense: {'<=': 'L', '>=': 'G', '==': 'E'}[sense] for sense in SENSE_CODES}

#: What an integer column is wrapped in. The name field is a constant because
#: nothing reads it — a marker is positional — and a counter would be one more
#: thing to keep identical between two writes of one model.
_MARKER = "    MARKER 'MARKER' '{}'"

#: Nonzeros per column chunk — a chunk's rendered lines live in memory until it
#: is sunk, so this bounds the writer's peak rather than its speed.
#:
#: **A quarter of :data:`~lpspec.relational.sinks.writers.lp_file.EMIT_BUDGET`,
#: because a nonzero costs more text here.** An MPS entry names its column on
#: every line where an LP term names it once per row, so the same nonzero count
#: holds several times the bytes and the twin budget bounded nothing: the wider
#: setting rendered a 2M-entry model in one chunk (#1102). Narrower than this
#: starts costing wall — the ladder turns at about this width and is 3.4x by
#: 20,000 — so it is the last value that is free.
EMIT_BUDGET = 500_000


def write_mps_file(tables: Tables, path: str | Path) -> None:
    """Write the model as MPS text.

    ``COLUMNS`` streams a column range at a time off the sorted matrix; every
    other section streams straight off the frame it renders, sorting nothing.
    """
    path = Path(path)
    entries, starts = _column_major(tables)

    with open(path, 'wb') as f:
        f.write(b'NAME\n')
        if tables.objective_sense == 'maximize':
            f.write(b'OBJSENSE\n    MAX\n')

        f.write(b'ROWS\n N  obj\n')
        sink(_row_lines(tables), f)

        f.write(b'COLUMNS\n')
        width = tables.matrix.height / max(1, tables.column_count)
        for lo, hi in chunking.ranges(tables.column_count, EMIT_BUDGET, width):
            owned = entries.slice(int(starts[lo]), int(starts[hi] - starts[lo]))
            sink(_column_lines(tables, lo, hi, owned), f)

        f.write(b'RHS\n')
        if tables.objective_constant:
            f.write(f'    rhs obj {-tables.objective_constant!r}\n'.encode())
        sink(_rhs_lines(tables), f)

        f.write(b'BOUNDS\n')
        _write_bounds(tables, f)

        if tables.sos.height:
            f.write(b'SOS\n')
            sink(_set_lines(tables), f)

        f.write(b'ENDATA\n')


def _column_major(tables: Tables) -> tuple[pl.DataFrame, npt.NDArray[np.int64]]:
    """The matrix in ``(col, row)`` order, and where each column's entries begin.

    This module's own CSR, by column — computed rather than asked of the
    engine, which holds no column index and would pay for one on every build
    to serve this one writer. The offsets are what let the ranges above slice
    instead of filtering the matrix once per chunk.

    The sort is the format's, not a choice: a column's entries have to reach
    consecutive lines. It is what this writer holds that the LP writer does
    not, and it is a floor rather than the peak: what dominates is a chunk's
    rendered lines, which is :data:`EMIT_BUDGET`'s to bound (#1102).
    """
    entries = tables.matrix_block(0, tables.row_count).sort('col', 'row')
    counts = np.bincount(entries['col'].to_numpy(), minlength=tables.column_count)
    return entries, np.concatenate(([0], np.cumsum(counts)))


def _row_lines(tables: Tables) -> pl.LazyFrame:
    """One ``ROWS`` entry per constraint row, after the objective's ``N``."""
    return tables.rows.lazy().select(
        pl.concat_str(
            pl.lit(' '),
            pl.col('sense').replace_strict(_MPS_SENSE, return_dtype=pl.String),
            pl.lit('  c'),
            digits(pl.col('row')),
        )
    )


def _rhs_lines(tables: Tables) -> pl.LazyFrame:
    """Each row's right-hand side, in row order."""
    return tables.rows.lazy().select(
        pl.concat_str(pl.lit('    rhs c'), digits(pl.col('row')), pl.lit(' '), number(pl.col('rhs')))
    )


def _column_lines(tables: Tables, lo: int, hi: int, entries: pl.DataFrame) -> pl.LazyFrame:
    """Every ``COLUMNS`` line for columns ``[lo, hi)``, one sorted stream.

    The LP writer's key trick, transposed: a column's lines occupy ``slots``
    consecutive keys — the integer marker, its objective coefficient, each
    matrix entry at its row index, the closing marker — so one sort settles
    both the column order and the order within a column.

    **Every column gets an objective line, coefficient or not.** A column MPS
    never names is a column the reader does not have, where LP declares them
    all in its bounds section; this is where the two formats put the same fact.
    """
    slots = tables.row_count + 3

    def _key(within: pl.Expr) -> pl.Expr:
        return chunk_key(pl.col('col'), lo, slots, within)

    columns = (
        tables.cols.lazy()
        .slice(lo, hi - lo)
        .with_row_index('col', offset=lo)
        .with_columns(pl.col('col').cast(pl.Int64))
    )
    integral = columns.filter(pl.col('vtype') != 'continuous')
    name = pl.concat_str(pl.lit('    x'), digits(pl.col('col')))
    cost = columns.join(tables.obj.lazy().with_columns(pl.col('col').cast(pl.Int64)), on='col', how='left').select(
        _key(pl.lit(1, dtype=pl.Int64)),
        pl.concat_str(name, pl.lit(' obj '), number(pl.col('coeff').fill_null(0.0))).alias('line'),
    )
    terms = (
        entries.lazy()
        .with_columns(pl.col('col').cast(pl.Int64))
        .select(
            _key(pl.col('row').cast(pl.Int64) + 2),
            pl.concat_str(name, pl.lit(' c'), digits(pl.col('row')), pl.lit(' '), number(pl.col('coeff'))).alias(
                'line'
            ),
        )
    )
    markers = [
        integral.select(_key(pl.lit(at, dtype=pl.Int64)), pl.lit(_MARKER.format(word)).alias('line'))
        for at, word in ((0, 'INTORG'), (slots - 1, 'INTEND'))
    ]
    return pl.concat([*markers, cost, terms]).sort('key').select('line')


def _write_bounds(tables: Tables, f: IO[bytes]) -> None:
    """Every column's lower bound, then every column's upper.

    Two passes rather than one interleaved section, and both parts of that are
    deliberate. Interleaving would need a sort, and this section is one line
    per *column* — the sort would hold the whole of it rendered, where two
    passes hold none. Reading it back, a reader that has already been given
    every lower bound cannot apply the MPS rule that an ``UP`` below zero
    implies an unbounded lower one, which is the one place the format guesses.
    """
    for keyword, unbounded, column in (('LO', 'MI', 'lb'), ('UP', 'PL', 'ub')):
        name = pl.concat_str(pl.lit(' bnd x'), digits(pl.col('col')))
        sink(
            tables.cols.lazy()
            .with_row_index('col')
            .select(
                pl.when(pl.col(column).is_infinite())
                .then(pl.concat_str(pl.lit(f' {unbounded}'), name))
                .otherwise(pl.concat_str(pl.lit(f' {keyword}'), name, pl.lit(' '), number(pl.col(column))))
            ),
            f,
        )


def _set_lines(tables: Tables) -> pl.LazyFrame:
    """Each special-ordered set as its header line and one line per member.

    The stream arrives grouped by set and ascending in weight, so a set's
    header belongs at its first member's position and nothing has to be
    gathered: the key is the member's own index, doubled to leave the header a
    place to sit.

    Written even where the reader may refuse it, for the LP writer's reason.
    """
    members = tables.sos.lazy().with_row_index('ord').with_columns(pl.col('ord').cast(pl.Int64))
    headers = (
        members.group_by('set', maintain_order=True)
        .agg(pl.col('type').first(), pl.col('ord').min())
        .select(
            (pl.col('ord') * 2).alias('key'),
            pl.concat_str(pl.lit(' S'), digits(pl.col('type')), pl.lit(' s'), digits(pl.col('set'))).alias('line'),
        )
    )
    lines = members.select(
        (pl.col('ord') * 2 + 1).alias('key'),
        pl.concat_str(pl.lit('    x'), digits(pl.col('col')), pl.lit(' '), digits(pl.col('weight'))).alias('line'),
    )
    return pl.concat([headers, lines]).sort('key').select('line')
