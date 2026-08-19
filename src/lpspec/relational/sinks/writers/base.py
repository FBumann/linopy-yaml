"""What a writer **is**: three decisions every format makes the same way.

How a section is appended, how a float is rendered, and how an index is. One
home each, because a second copy of any of them drifts into a file one solver
reads and another does not.

The family base for :mod:`~lpspec.relational.sinks.solvers.base`'s reason, one
family over: it renders no format of its own, so it cannot carry one across —
and it is what stops the alternative, one writer importing the other to share
a rule.
"""

from __future__ import annotations

from typing import IO

import polars as pl

__all__ = ['digits', 'number', 'sink']


def sink(frame: pl.LazyFrame, f: IO[bytes]) -> None:
    """Append a one-column frame to *f*, one raw line per row.

    A CSV writer with the CSV switched off, straight into the handle the caller
    holds: polars writes through its buffer, so an ``f.write()`` between two
    sinks lands between them and no concatenation pass rereads the file.

    ``maintain_order`` is polars' default, stated rather than inherited because
    the parameter is documented as unstable and a flipped default would make
    the bytes non-reproducible in silence (#109).
    """
    frame.sink_csv(f, include_header=False, quote_style='never', maintain_order=True)


def number(value: pl.Expr) -> pl.Expr:
    """A float as text.

    Polars' cast is shortest-*round-trip* rather than merely shortest, so the
    double a solver reads back is the double the engine computed. That is what
    makes emit affordable: it is almost entirely float-to-text, and a cast is
    far cheaper than a format string.
    """
    return value.cast(pl.String)


def digits(value: pl.Expr) -> pl.Expr:
    """An index as text — never in scientific notation, whatever its size."""
    return value.cast(pl.Int64).cast(pl.String)
