"""The ``lp_file`` sink: the model as LP text.

Portability, debugging, and the differential oracle. Every section is a lazy
frame sunk straight into the open file, so the rendered text is polars' to
stream and no byte is written twice.

Numbers go through polars' float cast, which round-trips exactly: the text a
solver reads back is the double the engine computed.

**Every section is written in label order.** A solver does not care, but a
reader diffing two LP files does, and so does anyone checking that a model
builds the same bytes twice (#109).
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from lpspec.relational.sinks.tables import ModelTables


def _sink(frame: pl.LazyFrame, f: IO[bytes]) -> None:
    """Append a one-column frame to *f*, one raw line per row.

    A CSV writer with the CSV switched off: no header, no quoting, so the bytes
    on disk are exactly the strings the frame holds.

    The frame goes into the file the caller is already holding rather than into
    a part file to be concatenated afterwards. Sections are produced in the
    order the LP format wants them, so there is nothing to reorder — and a
    concatenation pass would read and rewrite the whole file, which at these
    sizes costs more than producing it did. polars writes through the handle's
    own buffer, so a ``f.write()`` between two sinks lands between them.
    """
    # `maintain_order` is polars' default and is what #109 rests on, so it is
    # stated rather than inherited: the parameter is documented as unstable,
    # and a default that flips would make the bytes non-reproducible silently.
    frame.sink_csv(f, include_header=False, quote_style='never', maintain_order=True)


def write_lp_file(model: ModelTables, path: str | Path) -> None:
    """Write the model as LP text."""

    path = Path(path)
    objective = model.obj.lazy().sort('col').select(_term(pl.col('coeff'), pl.col('col')))
    # `cols` is positional, so the index is added inside the streamed pipeline
    # rather than sorted out of a column the model carried all along.
    bounds = (
        model.cols.lazy()
        .with_row_index('col')
        .select(
            pl.concat_str(
                _bound(pl.col('lb'), '-infinity').alias('lb'),
                pl.lit(' <= x').alias('open'),
                _digits(pl.col('col')),
                pl.lit(' <= ').alias('close'),
                _bound(pl.col('ub'), '+infinity').alias('ub'),
            )
        )
    )

    with open(path, 'wb') as f:
        f.write((b'min' if model.objective_sense == 'min' else b'max') + b'\n\nobj:\n')
        if model.objective_constant:
            f.write(f'{model.objective_constant:+.17g}\n'.encode())
        _sink(objective, f)

        f.write(b'\ns.t.\n\n')
        _sink(_constraint_lines(model), f)

        f.write(b'\nbounds\n')
        _sink(bounds, f)

        for variable_type, keyword in (('binary', 'binary'), ('integer', 'general')):
            chosen = model.cols.lazy().with_row_index('col').filter(pl.col('vtype') == variable_type)
            if chosen.select(pl.len()).collect().item() == 0:
                continue
            f.write(f'\n{keyword}\n'.encode())
            _sink(chosen.select(pl.concat_str(pl.lit('x'), _digits(pl.col('col')))), f)

        f.write(b'\nend\n')


def _constraint_lines(model: ModelTables) -> pl.LazyFrame:
    """The whole constraint section, one output line per row of the frame.

    A constraint is a header, its terms one per line, then its comparison and
    right-hand side. So the terms are gathered into a list per row, the header
    and footer are put on either side of it, and the list is exploded back into
    lines.

    **Both orderings are what #109 pins**: a model must write the same bytes
    twice. Within a row, the terms come out in column order because the matrix
    arrives in it (:class:`~lpspec.relational.sinks.tables.ModelTables`) and a
    list keeps the order it was built in. The rows are sorted **after** the
    join, since a join hands groups back in whatever order it finishes them.

    A row with no terms still needs a line a solver can parse, which is what
    the placeholder is: the left join leaves it null, and nothing else can.
    """
    terms = model.matrix.lazy().group_by('row').agg(_term(pl.col('coeff'), pl.col('col')).alias('terms'))
    return (
        model.rows.lazy()
        .join(terms, on='row', how='left')
        .sort('row')
        .select(
            pl.concat_list(
                pl.concat_str(pl.lit('c'), _digits(pl.col('row')), pl.lit(':')),
                pl.col('terms').fill_null(pl.lit(['+0 x0'], dtype=pl.List(pl.String))),
                pl.concat_str(pl.col('sense').replace({'==': '='}), pl.lit(' '), _number(pl.col('rhs'))),
            ).alias('line')
        )
        # never empty — every row has a header and a footer at least
        .explode('line', empty_as_null=False)
    )


def _term(coeff: pl.Expr, col: pl.Expr) -> pl.Expr:
    """One ``+1.5 x7`` term.

    Built by a single ``concat_str`` rather than by chaining ``+``. Every ``+``
    is its own pass allocating its own full-width string column, and a term has
    four pieces; this way the line is allocated once.
    """
    return pl.concat_str(*_signed(coeff), pl.lit(' x'), _digits(col))


def _number(value: pl.Expr) -> pl.Expr:
    """A float as LP text."""

    return value.cast(pl.String)


def _signed(value: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """A coefficient, sign always explicit — the LP format needs the ``+``.

    Two pieces for ``concat_str`` rather than one finished string, because the
    cast already carries the ``-``: only a non-negative value needs a sign
    glued on, and the sign column is one character wide however large the
    model. Deciding the sign and then rendering ``abs()`` would render the
    magnitude in both arms of the ``when``, at full width, to discard one.

    ``-0.0`` is why zero is spelled out rather than cast: it is ``>= 0``, so it
    takes the ``+`` arm while the cast still renders ``-0.0``, giving
    ``+-0.0``, which no LP parser accepts. It is reachable from any negative
    coefficient times a zero parameter, so it is a real file, not a curiosity.
    """
    return (
        pl.when(value >= 0).then(pl.lit('+')).otherwise(pl.lit('')).alias('sign'),
        pl.when(value == 0).then(pl.lit('0.0')).otherwise(_number(value)).alias('magnitude'),
    )


def _bound(value: pl.Expr, infinite: str) -> pl.Expr:
    """A bound, with the LP format's own spelling for an unbounded one."""

    return pl.when(value.is_infinite()).then(pl.lit(infinite)).otherwise(_number(value))


def _digits(value: pl.Expr) -> pl.Expr:
    """An index as text — never in scientific notation, whatever its size."""

    return value.cast(pl.Int64).cast(pl.String)
