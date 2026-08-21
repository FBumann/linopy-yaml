"""Delete a guard, run the tests, and record whether anything went red.

    pixi run python -m tools.mutate src/lpspec/sources.py:98-99
    pixi run python -m tools.mutate 'src/a.py:98-99#the empty-parameter guard' src/b.py:88
    pixi run python -m tools.mutate src/a.py:98-99 --tests tests/test_relational.py

A *mutation table* is what AGENTS.md asks for beside a correctness guard: delete
the guard, run the suite, and write down whether it was caught. A guard the
suite survives is one no test can reach, and the table is the evidence that each
one is reachable — which a green suite, on its own, never is.

Run by hand that table can lie three ways, and each of the three is a run that
happened here:

**A stale ``__pycache__`` keeps the mutant running after it was restored.** The
staleness check is mtime plus size, so a same-length patch put back inside the
same second leaves the interpreter on the old bytecode, and the *next* mutation
measures a tree nobody is looking at.

**A restore held in a variable leaves the mutant in the tree.** A run killed by
a timeout or a Ctrl-C never reaches its ``finally``; #974 committed a mutated
sink that way, and the following run read it as the baseline.

**Uncommitted work has nothing to restore to.** ``git checkout --`` puts back
what was committed, so an edit that was never committed would be destroyed
rather than restored. Untracked files are not at risk and are not counted.

So this refuses to start unless every tracked file is committed, restores each with
``git checkout --`` rather than from memory, drops ``__pycache__`` on both sides
of every mutation, and ends by asserting the tree is clean again — printed, so a
reader of the output knows it was checked.

Rows print as they finish, so a killed run still leaves the ones it got through.
For the two-column form — the same mutations against a base branch — run the
tool once in each worktree and put the columns side by side; nothing here needs
to know about the other branch.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: ``path:line``, ``path:start-end``, either with ``#`` and a label after it.
_SPEC = re.compile(r'^(?P<path>[^:]+):(?P<start>\d+)(?:-(?P<end>\d+))?(?:#(?P<label>.*))?$')

#: pytest's exit code for a run that collected nothing.
_NO_TESTS = 5

#: How far a label read off the source itself is allowed to run.
_LABEL_WIDTH = 60


@dataclass(frozen=True)
class Guard:
    """One mutation: the lines to delete, and what to call them in the table."""

    path: Path
    start: int
    end: int
    label: str

    @property
    def where(self) -> str:
        span = f'{self.start}-{self.end}' if self.end != self.start else f'{self.start}'
        return f'{self.path.name}:{span}'


class SpecError(ValueError):
    """A guard specification the tool cannot act on."""


def parse_guard(text: str, root: Path | None = None) -> Guard:
    """Read one ``path:start-end#label`` argument.

    Args:
        text: The argument as typed.
        root: Repository root the path is resolved against; ``REPO`` when
            omitted, read at call time so a caller may move it.

    Returns:
        The guard, with a label read off the first mutated line where the
        argument gave none.

    Raises:
        SpecError: If the argument is not that shape, names no file, or names
            lines the file does not have.
    """
    root = REPO if root is None else root
    match = _SPEC.match(text)
    if match is None:
        raise SpecError(f'{text!r} is not path:line, path:start-end, or either with #label')
    path = root / match['path']
    if not path.is_file():
        raise SpecError(f'{match["path"]} is not a file')
    start = int(match['start'])
    end = int(match['end']) if match['end'] else start
    lines = path.read_text().splitlines()
    if not 1 <= start <= end <= len(lines):
        raise SpecError(f'{match["path"]} has {len(lines)} lines, so {start}-{end} is not in it')
    label = (match['label'] or '').strip() or _read_label(lines[start - 1])
    return Guard(path=path, start=start, end=end, label=label)


def _read_label(line: str) -> str:
    """A label off the first mutated line, for an argument that gave none."""
    stripped = line.strip()
    shown = stripped if len(stripped) <= _LABEL_WIDTH else f'{stripped[:_LABEL_WIDTH].rstrip()}…'
    return f'drop `{shown}`'


def mutate(source: str, guard: Guard) -> tuple[str, str]:
    """The file with the guard's lines gone, and how they had to go.

    Deleting a guard that is a whole statement leaves the file parsing. Deleting
    one that is the *body* of an ``if`` or a ``with`` does not, so the body
    becomes ``pass`` instead — which is the same mutation for the suite's
    purposes and is reported as its own form so a reader is not misled.

    Args:
        source: The file's current text.
        guard: The lines to remove.

    Returns:
        The mutated text, and ``deleted`` or ``pass``.

    Raises:
        SpecError: If neither form parses, which means the range straddles
            something no mutation of it can leave valid.
    """
    lines = source.splitlines(keepends=True)
    before, cut, after = lines[: guard.start - 1], lines[guard.start - 1 : guard.end], lines[guard.end :]
    deleted = ''.join(before + after)
    if _parses(deleted):
        return deleted, 'deleted'
    indent = cut[0][: len(cut[0]) - len(cut[0].lstrip())]
    passed = ''.join([*before, f'{indent}pass\n', *after])
    if _parses(passed):
        return passed, 'pass'
    raise SpecError(f'{guard.where} parses neither deleted nor replaced by `pass` — narrow the range')


def _parses(source: str) -> bool:
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(['git', '-C', str(REPO), *args], capture_output=True, text=True, check=False)


def dirty_paths() -> list[str]:
    """Uncommitted changes to *tracked* files, which are the ones at risk.

    Untracked files are deliberately not counted. The restore is
    ``git checkout -- <path>``, which cannot touch a file git does not track, so
    a scratch note or an editor leftover is not something this run could destroy
    — and refusing to start because one exists guards nothing while making the
    tool unusable in a working tree anyone is actually working in.
    """
    return [line[3:] for line in _git('status', '--porcelain', '--untracked-files=no').stdout.splitlines()]


def _drop_bytecode() -> None:
    """Every ``__pycache__`` in the tree, ``.git`` and the virtualenv aside.

    Restoring a file is not enough on its own: the interpreter's staleness check
    is mtime plus size, and a same-length restore inside the same second passes
    it, so the mutant would keep running. The sweep is the whole tree rather
    than the mutated package because an importer caches too, and a package list
    is one more thing to keep in step with the layout.
    """
    for cache in REPO.rglob('__pycache__'):
        if any(part in {'.git', '.venv'} for part in cache.parts):
            continue
        shutil.rmtree(cache, ignore_errors=True)


def command(tests: list[str]) -> list[str]:
    """The pytest invocation, with ``-n auto`` only where xdist is installed.

    The bare-install job has no xdist, and pytest answers an unknown option with
    a usage error — exit 4, having run nothing — which a mutation row must never
    be built on. So the flag is added where it can be used and left off where it
    cannot, rather than assumed from the environment the tool was written in.
    """
    parallel = ['-n', 'auto'] if importlib.util.find_spec('xdist') else []
    return [sys.executable, '-m', 'pytest', '-q', '-x', *parallel, '-p', 'no:cacheprovider', *tests]


def _run_tests(tests: list[str]) -> str:
    """Run the suite against the tree as it stands, and judge what came back.

    Stops at the first failure: a guard is caught the moment anything goes red,
    and the count only has to be exact when nothing does.

    Returns:
        The table cell — see :func:`verdict` for what each one means.
    """
    completed = subprocess.run(command(tests), cwd=REPO, capture_output=True, text=True, check=False)
    return verdict(completed.returncode, _summary(completed.stdout, completed.stderr))


def verdict(returncode: int, summary: str) -> str:
    """What one run of the suite says about the guard that was deleted.

    Nothing here is read off the exit code alone, because three different
    outcomes share one. Under ``-x`` with xdist a failure interrupts the session
    and pytest exits 2 rather than 1; a collection error exits 2 as well and
    prints ``1 error`` having run no test at all; and a ``--tests`` path that
    matches nothing exits 5. A code-only reading calls all three "caught" and
    certifies a guard that was never reached, which is the one thing this tool
    exists not to do.

    So a catch requires a *failure*. An error is not one: a mutation that breaks
    an import errors in collection, and a fixture that errors is the suite
    giving up rather than disagreeing. Both are reported as proving nothing —
    under-claiming, which costs a second look, where over-claiming costs the
    guarantee the table is for.

    Args:
        returncode: pytest's exit code.
        summary: Its last line, which carries the counts.

    Returns:
        The markdown cell for this mutation's row.
    """
    if returncode == 0:
        return summary
    if returncode == _NO_TESTS or 'no tests ran' in summary:
        return f'**nothing ran** — {summary}; this row proves nothing'
    if 'failed' in summary:
        return '**caught**'
    if 'error' in summary:
        return (
            f'**errored, not failed** — {summary}; the suite gave up rather than disagreed, so this row proves nothing'
        )
    return f'**pytest exited {returncode}** — {summary}; this row proves nothing'


def _summary(stdout: str, stderr: str = '') -> str:
    """pytest's last non-empty line, which carries the counts.

    Falls back to stderr, because that is where a run that never started says
    why: a usage error prints nothing at all on stdout, and a row reading "no
    output" hides the one sentence that would fix it.
    """
    for stream in (stdout, stderr):
        lines = [line.strip() for line in stream.splitlines() if line.strip()]
        if lines:
            return lines[-1].strip('= ')
    return 'no output on either stream'


def _row(guard: Guard, form: str, result: str) -> str:
    shown = f'`{guard.where}`' if form == 'deleted' else f'`{guard.where}`, body → `pass`'
    return f'| {guard.label} ({shown}) | {result} |'


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('guards', nargs='+', metavar='path:start-end[#label]', help='the guards to delete, in turn')
    ap.add_argument('--tests', nargs='+', default=['tests'], help='what to run for each mutation (default: tests)')
    args = ap.parse_args(argv)

    if dirty := dirty_paths():
        listed = '\n  '.join(dirty)
        print(f'tracked files have uncommitted changes, so there is nothing to restore to — commit first:\n  {listed}')
        return 2
    try:
        guards = [parse_guard(text) for text in args.guards]
    except SpecError as exc:
        print(str(exc))
        return 2

    print('| mutation | result |')
    print('|---|---|')
    for guard in guards:
        original = guard.path.read_text()
        try:
            mutated, form = mutate(original, guard)
        except SpecError as exc:
            print(f'| {guard.label} (`{guard.where}`) | **not applied** — {exc} |')
            continue
        try:
            _drop_bytecode()
            guard.path.write_text(mutated)
            _drop_bytecode()
            result = _run_tests(args.tests)
        finally:
            _git('checkout', '--', str(guard.path.relative_to(REPO)))
            _drop_bytecode()
        print(_row(guard, form, result), flush=True)

    if left := dirty_paths():
        print(f'\n**a tracked file was left modified** — {", ".join(left)}')
        return 1
    print('\nTree clean after the run.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
