"""The ``lp_file`` sink: the model as LP text.

Portability, debugging, and the differential oracle. Every section is a lazy
frame sunk straight into the open file, so the rendered text is polars' to
stream and no byte is written twice.

**Every section is written in label order.** A solver does not care, but a
reader diffing two LP files does, and so does anyone checking that a model
builds the same bytes twice (#109).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from lpspec.relational.sinks.capabilities import Capabilities
from lpspec.relational.sinks.tables import SENSE_CODES
from lpspec.relational.sinks.writers.base import chunk_key, digits, number, sink

if TYPE_CHECKING:
    from lpspec.relational.sinks.tables import Tables


#: A section is text, so this format excludes no combination and curvature
#: costs it nothing — and every construct the language can reach is a section
#: this writer emits, quadratic rows included. What a descriptor declares is
#: what :func:`write_lp_file` **emits**, on the rule the gurobi sink's states:
#: an entry for a section nothing writes would hand back a file missing the
#: rows that make the model what it is, which is why `.mps` declares none.
#:
#: What no descriptor promises is that the solver reading the file back parses
#: what was written — that is a property of a *reader*, and HiGHS's refuses two
#: of these sections (docs/about/benchmarks.md#sink-capabilities).
LP_FILE_CAPABILITIES = Capabilities(
    supports={
        'integrality': 'native',
        'sos': 'native',
        'quadratic_objective': 'native',
        'nonconvex_quadratic_objective': 'native',
        'quadratic_constraint': 'native',
    }
)


#: How the LP format spells each comparison. Derived from the engine's own
#: vocabulary rather than written out, so a sense added there reaches the file
#: or raises here, instead of being rendered as whatever the loop last saw.
#: The format differs on one word: it writes an equality as ``=``.
_LP_SENSE = {sense: '=' if sense == '==' else sense for sense in SENSE_CODES}

#: Nonzeros per constraint chunk. A chunk's rendered lines live in memory until
#: it is sunk, so this is the knob that bounds the writer's peak rather than its
#: speed: chunking at this width takes most of the constraint section out of
#: peak for no change in the bytes written. Wider costs memory for nothing; much
#: narrower pays per-chunk overhead on every range (#189).
EMIT_BUDGET = 2_000_000


def write_lp_file(tables: Tables, path: str | Path) -> None:
    """Write the model as LP text.

    ``cols`` is positional, so the bounds section's index is added inside the
    streamed pipeline. The constraint section goes out one row range at a time,
    since a chunk's rendered lines are held until it is sunk.
    """
    path = Path(path)
    objective = tables.obj.lazy().sort('col').select(_term(pl.col('coeff'), pl.col('col')))
    bounds = (
        tables.cols.lazy()
        .with_row_index('col')
        .select(
            pl.concat_str(
                _bound(pl.col('lb'), '-infinity').alias('lb'),
                pl.lit(' <= x').alias('open'),
                digits(pl.col('col')),
                pl.lit(' <= ').alias('close'),
                _bound(pl.col('ub'), '+infinity').alias('ub'),
            )
        )
    )

    with open(path, 'wb') as f:
        f.write((b'max' if tables.objective_sense == 'maximize' else b'min') + b'\n\nobj:\n')
        if tables.objective_constant:
            f.write(f'{tables.objective_constant:+.17g}\n'.encode())
        sink(objective, f)
        if tables.quad.height:
            f.write(b'+ [\n')
            sink(_quadratic_terms(tables), f)
            f.write(b'] / 2\n')

        f.write(b'\ns.t.\n\n')
        for block in tables.row_blocks(EMIT_BUDGET):
            sink(_constraint_lines(tables, block.lo, block.hi, tables.matrix_block(block.lo, block.hi)), f)
        for row, pairs in tables.quadratic_blocks():
            sink(_quadratic_row_lines(tables, row, pairs), f)

        f.write(b'\nbounds\n')
        sink(bounds, f)

        for variable_type, keyword in (('binary', 'binary'), ('integer', 'general')):
            chosen = tables.cols.lazy().with_row_index('col').filter(pl.col('vtype') == variable_type)
            if chosen.select(pl.len()).collect().item() == 0:
                continue
            f.write(f'\n{keyword}\n'.encode())
            sink(chosen.select(pl.concat_str(pl.lit('x'), digits(pl.col('col')))), f)

        if tables.sos.height:
            f.write(b'\nsos\n')
            sink(_set_lines(tables), f)

        f.write(b'\nend\n')


def _quadratic_row_lines(tables: Tables, row: int, pairs: pl.DataFrame) -> pl.LazyFrame:
    r"""One quadratic constraint, linear part then bracketed quadratic part.

    ``c7: +1 x0 + [ 2 x0 * x1 ] >= 4``. **Not** halved, unlike the objective's
    section: the format divides only that one by two, an asymmetry of the
    format rather than of ours.

    Written a row at a time, after the linear rows and still in label order —
    the quadratic rows *are* the tail. Gathering one row's lines is the ``sos``
    section's trade, and it leaves the linear path's streamed interleave alone.
    """
    entries = tables.matrix_block(row, row + 1)
    header = pl.LazyFrame({'line': [f'c{row}:']})
    linear = entries.lazy().sort('col').select(_term(pl.col('coeff'), pl.col('col')).alias('line'))
    opened = pl.LazyFrame({'line': ['+ [']})
    quadratic = pairs.lazy().select(_pair(pl.col('coeff')).alias('line'))
    sense = tables.rows.filter(pl.col('row') == row)
    closed = pl.LazyFrame({'line': [']' + ' ' + _LP_SENSE[sense.item(0, 'sense')] + ' ' + str(sense.item(0, 'rhs'))]})
    return pl.concat([header, linear, opened, quadratic, closed])


def _quadratic_terms(tables: Tables) -> pl.LazyFrame:
    r"""The objective's quadratic part, one ``+2 x3 * x7`` line per pair.

    **The section is divided by two, so every coefficient here is doubled.**
    The format writes :math:`[\;\cdot\;] / 2`, which is the Hessian convention
    wearing text: a term the model states as :math:`q\,x_i x_j` is written
    ``2q``, on the diagonal and off it alike, and the reader halves it back.
    The uniformity is worth stating because it is *not* the Hessian's own rule
    — there the diagonal doubles and the off-diagonal does not, since a
    symmetric matrix holds an off-diagonal pair twice.

    A pair arrives ordered, summed and deduplicated
    (:meth:`~lpspec.relational.engines.polars.engine.PolarsEngine._objective_quadratic`),
    so nothing here sorts — the same contract the ``sos`` section reads its
    groups off.
    """
    return tables.quad.lazy().select(_pair(pl.col('coeff') * 2))


def _pair(coeff: pl.Expr) -> pl.Expr:
    """One quadratic pair as ``+2 x3 * x7`` — or ``x3 ^ 2`` for a squared column.

    ``^ 2`` is the format's spelling and no parser accepts ``x3 * x3``. The
    objective section doubles *coeff* and the constraint section does not,
    which is the one asymmetry between its two callers.
    """
    return pl.concat_str(
        *_signed(coeff),
        pl.lit(' x'),
        digits(pl.col('col_l')),
        pl.when(pl.col('col_l') == pl.col('col_r'))
        .then(pl.lit(' ^ 2'))
        .otherwise(pl.concat_str(pl.lit(' * x'), digits(pl.col('col_r')))),
    )


def _set_lines(tables: Tables) -> pl.LazyFrame:
    """Each special-ordered set as one ``s0: S2 :: x3:1 x4:2`` line.

    linopy's spelling of the section, so a file this writes and a file the
    eager lane writes are read by the same parsers.

    **The one section gathered rather than interleaved.** A set's members have
    to reach one line, where the constraint section sorts one row per output
    line instead; what makes that affordable is that a set is a handful of
    members and a model declares far fewer sets than rows. Order is the
    stream's own, and ``maintain_order`` is what keeps a group's line the same
    bytes twice.

    Written even where the reader may refuse it: HiGHS has no SOS concept and
    its parser says so, which is the honest outcome for a solver that cannot
    answer the question.
    """
    return (
        tables.sos.lazy()
        .group_by('set', maintain_order=True)
        .agg(
            pl.col('type').first(),
            pl.concat_str(pl.lit('x'), digits(pl.col('col')), pl.lit(':'), digits(pl.col('weight')))
            .str.join(' ')
            .alias('members'),
        )
        .select(
            pl.concat_str(
                pl.lit('s'),
                digits(pl.col('set')),
                pl.lit(': S'),
                digits(pl.col('type')),
                pl.lit(' :: '),
                pl.col('members'),
            )
        )
    )


def _constraint_lines(tables: Tables, lo: int, hi: int, entries: pl.DataFrame) -> pl.LazyFrame:
    """Every constraint line for rows ``[lo, hi)``, one sorted stream.

    One row per *output line*, interleaved by sorting, so nothing gathers a
    row's terms into a string list first — a ``group_by('row')`` into a list
    column and an explode measured 3x this on ``sector/m`` emit (#520). *entries*
    is the
    chunk's slice of the matrix from :meth:`Tables.matrix_block`, and the
    anti-join gives a termless row the line a solver still needs to parse.

    **The order is one integer, and the only other column.** A row's lines
    occupy ``slots`` consecutive keys — header, placeholder, each term at its
    column index, sense — so one sort settles both the row order and the order
    within a row, which is what #109 pins.

    The terms are sorted although they arrive sorted: the union subsumes the
    order and the bytes are identical without it, but the union sort merges
    pre-ordered runs rather than permuting them, and dropping it costs emit on
    every case measured (#520).
    """
    slots = tables.cols.height + 3

    def _key(within: pl.Expr) -> pl.Expr:
        return chunk_key(pl.col('row'), lo, slots, within)

    rows = tables.rows.lazy().filter(pl.col('row').is_between(lo, hi, closed='left'))
    matrix = entries.lazy()
    header = rows.select(
        _key(pl.lit(0, dtype=pl.Int64)),
        pl.concat_str(pl.lit('c').alias('c'), digits(pl.col('row')), pl.lit(':').alias('colon')).alias('line'),
    )
    placeholder = rows.join(matrix.select('row'), on='row', how='anti').select(
        _key(pl.lit(1, dtype=pl.Int64)),
        pl.lit('+0 x0').alias('line'),
    )
    terms = matrix.sort('row', 'col').select(
        _key(pl.col('col').cast(pl.Int64) + 2),
        _term(pl.col('coeff'), pl.col('col')).alias('line'),
    )
    footer = rows.select(
        _key(pl.lit(slots - 1, dtype=pl.Int64)),
        pl.concat_str(
            pl.col('sense').replace_strict(_LP_SENSE, return_dtype=pl.String),
            pl.lit(' '),
            number(pl.col('rhs')),
        ).alias('line'),
    )
    return pl.concat([header, placeholder, terms, footer]).sort('key').select('line')


def _term(coeff: pl.Expr, col: pl.Expr) -> pl.Expr:
    """One ``+1.5 x7`` term, allocated once.

    Chaining ``+`` would make each of the four pieces its own pass over a
    full-width string column.
    """
    return pl.concat_str(*_signed(coeff), pl.lit(' x'), digits(col))


def _signed(value: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """A coefficient, sign always explicit — the LP format needs the ``+``.

    Two pieces rather than one finished string: the cast already carries the
    ``-``, so only a non-negative value needs a sign glued on and the sign
    column stays one character wide. Rendering ``abs()`` under a ``when``
    instead would render the magnitude at full width in both arms to discard
    one.

    Zero is spelled out rather than cast because ``-0.0`` is ``>= 0``: it takes
    the ``+`` arm while the cast renders ``-0.0``, giving ``+-0.0``, which no LP
    parser accepts. Any negative coefficient times a zero parameter reaches it.
    """
    return (
        pl.when(value >= 0).then(pl.lit('+')).otherwise(pl.lit('')).alias('sign'),
        pl.when(value == 0).then(pl.lit('0.0')).otherwise(number(value)).alias('magnitude'),
    )


def _bound(value: pl.Expr, infinite: str) -> pl.Expr:
    """A bound, with the LP format's own spelling for an unbounded one."""
    return pl.when(value.is_infinite()).then(pl.lit(infinite)).otherwise(number(value))
