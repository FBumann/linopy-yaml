"""The writer family: the tables in, a file out. See ../README.md.

One module per format, chosen by the output's **suffix** — the caller names an
output, not a writer, because a file's format is a property of the file. Each
answers ``(tables, path) -> None``, and streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lpspec.relational.sinks.writers.lp_file import LP_FILE_CAPABILITIES, write_lp_file
from lpspec.relational.sinks.writers.mps_file import MPS_FILE_CAPABILITIES, write_mps_file

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from lpspec.relational.sinks.capabilities import Capabilities
    from lpspec.relational.sinks.tables import ModelTables

    Write = Callable[[ModelTables, Path], None]

__all__ = ['WRITERS', 'Writer', 'writer']


@dataclass(frozen=True)
class Writer:
    """One format: how to render it, and what it can carry.

    Together rather than in twin dicts keyed alike, where a format added to one
    and not the other would answer a capability question with a ``KeyError``.
    """

    write: Write
    capabilities: Capabilities


#: What can be written today, by suffix. Closed, for
#: :data:`~lpspec.relational.sinks.solvers.SOLVERS`' reason.
WRITERS: Mapping[str, Writer] = {
    '.lp': Writer(write_lp_file, LP_FILE_CAPABILITIES),
    '.mps': Writer(write_mps_file, MPS_FILE_CAPABILITIES),
}


def writer(suffix: str) -> Writer:
    """The writer for *suffix*.

    Raises:
        ValueError: A format nothing writes; the message lists what can be.
    """
    if suffix in WRITERS:
        return WRITERS[suffix]
    supported = ', '.join(sorted(WRITERS))
    raise ValueError(f'unsupported output format {suffix!r} — supported: {supported}')
