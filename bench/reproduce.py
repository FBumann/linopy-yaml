# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "lpspec[gurobi,linopy] @ git+https://github.com/fluxopt/lpspec@v0.0.1-alpha.258",
#   "pyomo>=6.7",
#   "pytest==9.1.1",
#   "pytest-benchmem>=0.5",
#   "pyarrow>=16",
# ]
# ///
"""Re-run the published comparison on the versions that produced it.

    uv run --locked bench/reproduce.py            # the published selection
    uv run --locked bench/reproduce.py --sizes xs # a smaller look

**Why this file exists at all.** `pixi.lock` is not committed, and two of the
libraries the benchmark measures are installed from git — lpspec itself, and
linopy from `master`, a branch that moves. So "the versions that produced a
number" was recorded only inside the results file, after the fact, in a form
nobody could install. `bench/reproduce.py.lock` beside this script freezes every
one of them, git commits included, and `--locked` refuses to run if the
resolution has drifted.

**It reads the selection rather than repeating it.** `pixi run ladder` in
`pyproject.toml` is where the cases, rungs, sinks and libraries are written
down; this script pulls that string out of the manifest, so a reproduction
cannot quietly run a different comparison than the one being reproduced.

**It drives the harness rather than repeating it.** The models, the data
generators and the rungs live in `bench/`; a standalone script that rebuilt them
would be a second definition of every model, free to disagree with the one being
measured. So this needs the repository checked out, which anyone reproducing a
number needs anyway — the rival formulations are in it.

**The published numbers predate this file and cannot be reproduced from it.**
They were taken against `lpspec 0.0.1a61.dev3+gf319cd10f` and a linopy built
from a branch, neither of which resolves from an index. That is the argument for
the lock rather than an objection to it: the next published run is taken through
this script, and then the pins and the page describe the same environment.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def published() -> list[str]:
    """The selection `pixi run ladder` takes, read out of `pyproject.toml`.

    Read rather than repeated. This script exists so that a number can be
    reproduced on the versions that produced it, and a reproduction that ran a
    *different selection* would be worth less than no reproduction at all — so
    the selection has one home, and it is the task definition every other caller
    uses.

    The task's argument defaults are the published values; a narrower run passes
    them on the command line and is a smoke test rather than a table.
    """
    import tomllib

    manifest = Path(__file__).resolve().parent.parent / 'pyproject.toml'
    task = tomllib.loads(manifest.read_text())['tool']['pixi']['feature']['bench']['tasks']['ladder']
    command = ' '.join(task['cmd'].split())
    for argument in task['args']:
        command = command.replace('{{ ' + argument['arg'] + ' }}', argument['default'])
    return command.split()[1:]


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parent.parent
    command = [sys.executable, '-m', 'pytest', '-q', *published(), *argv]
    print(f'$ {" ".join(command)}\n', flush=True)
    return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
