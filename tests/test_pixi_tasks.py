"""A pixi task's ``cmd`` is a shell script, so its newlines are what separate
one command from the next — they are not the line wrapping they look like.

A task written across lines for readability therefore runs as several commands,
and the shape that hides it is a long invocation whose first line is already
valid on its own::

    cmd = \"\"\"
    pytest bench --benchmark-memory
      --cases dispatch transport storage fleet
      --budget 30 --benchmark-json=bench/results/latest.json
    \"\"\"

That ran a full default benchmark for four and a half hours, then reported
``--cases: command not found`` — the selection, the budget and the results file
were three separate commands that never reached pytest. Nothing above it
noticed: the run passed 260 tests, and the flags it dropped are exactly the
ones that would have made it write anything down.

A line ending in an operator (``&&``, ``|``) is genuinely continued and is
fine. What this refuses is a line that ends in the middle of an argument list.
"""

import tomllib
from pathlib import Path

import pytest

CONTINUATIONS = ('&&', '||', '|', '\\', '(', '{', ';')


def tasks() -> dict[str, str]:
    """Every pixi task in the file, ``name`` to ``cmd``, features included."""
    config = tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text())
    pixi = config['tool']['pixi']
    tables = [pixi.get('tasks', {})] + [f.get('tasks', {}) for f in pixi.get('feature', {}).values()]
    return {
        name: body['cmd'] if isinstance(body, dict) else body
        for table in tables
        for name, body in table.items()
        if isinstance(body, str) or 'cmd' in body
    }


@pytest.mark.parametrize('name', sorted(tasks()), ids=str)
def test_a_task_wrapped_across_lines_is_still_one_command(name: str) -> None:
    lines = [line.strip() for line in tasks()[name].strip().splitlines() if line.strip()]
    dangling = [line for line in lines[:-1] if not line.endswith(CONTINUATIONS)]
    assert not dangling, (
        f'`{name}` wraps onto the next line after {dangling} — pixi runs each line as its own '
        f'command, so everything below it is dropped from the invocation and then executed as a '
        f'program. End the line with `\\` to wrap it, or with `&&` to mean a second command.'
    )
