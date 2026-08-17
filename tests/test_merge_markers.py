"""No file in the tree carries a conflict marker.

`examples/README.md` shipped three nested `<<<<<<< HEAD` blocks through three
squash merges before anyone noticed, because nothing reads that file. The one
test that does read it — `test_the_prose_counts_the_ports_there_are` — asks
whether its sentence is *in* the file, which stays true with markers wrapped
around it.

Every port PR edits the same handful of lines in the same handful of files, so
a rebase-and-continue loop is normal here, and `git add -A` is one keystroke
away from committing whatever the merge left behind. This is the check that
makes that loud.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent

#: A marker git writes, anchored at the start of a line. The trailing space is
#: load-bearing: it keeps this file's own prose, and a markdown `=======` rule,
#: from matching.
MARKER = re.compile(r'^(<<<<<<< |>>>>>>> |=======$)', re.MULTILINE)


def _tracked() -> list[Path]:
    files = subprocess.run(
        ['git', 'ls-files', '-z'], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split('\0')
    return [REPO / f for f in files if f]


@pytest.mark.parametrize('path', _tracked(), ids=lambda p: str(p.relative_to(REPO)))
def test_no_file_carries_a_conflict_marker(path: Path) -> None:
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return  # binary, or a path git tracks that the checkout does not hold

    if path == Path(__file__):
        return  # this module names the markers it looks for

    found = MARKER.search(text)
    assert not found, (
        f'{path.relative_to(REPO)} line {text[: found.start()].count(chr(10)) + 1} is a conflict '
        f'marker: {found.group(0)!r} — a merge was committed unresolved'
    )
