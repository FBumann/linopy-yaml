"""The writer family: the tables in, a file out. See ../README.md.

One module per format, chosen by the output's **suffix** — the caller names an
output, not a writer, because a file's format is a property of the file. Each
answers ``(tables, path) -> None``, and streams.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lpspec.relational.sinks.writers.lp_file import write_lp_file

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from lpspec.relational.sinks.tables import ModelTables

    Write = Callable[[ModelTables, Path], None]

__all__ = ['WRITERS', 'writer']

#: What can be written today, by suffix. Closed, for
#: :data:`~lpspec.relational.sinks.solvers.SOLVERS`' reason.
WRITERS: Mapping[str, Write] = {'.lp': write_lp_file}


def writer(suffix: str) -> Write:
    """The writer for *suffix*.

    Raises:
        ValueError: A format nothing writes; the message lists what can be.
    """
    if suffix in WRITERS:
        return WRITERS[suffix]
    supported = ', '.join(sorted(WRITERS))
    raise ValueError(f'unsupported output format {suffix!r} — supported: {supported}')
