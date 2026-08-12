"""The decomposition example must keep reaching the monolith, or it is not evidence.

`examples/benders/run.py` is the claim that lpspec can *express* Benders — four
files, cuts as data, no engine change. A claim like that is worth exactly as
much as the check behind it, so the check is the one the algorithm itself
provides: the decomposed answer has to equal the answer the monolith gives on
the same sources, and the run prints both.

Committed output rather than an assertion on a number, for the reason
`test_walkthrough.py` gives: a page that shows output is making a promise about
what a reader will see, and a diff is how that promise stays true. Regenerate
with ``--update-golden`` when the story legitimately changes.
"""

from __future__ import annotations

import contextlib
import difflib
import importlib.util
import io
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).parent.parent / 'examples' / 'benders' / 'run.py'
GOLDEN = EXAMPLE.with_name('run.out')


@pytest.fixture(scope='module')
def output() -> str:
    spec = importlib.util.spec_from_file_location('benders_example', EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['benders_example'] = module
    spec.loader.exec_module(module)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        module.main()
    del sys.modules['benders_example']
    return buffer.getvalue()


def test_the_decomposition_reaches_the_monolith(output: str) -> None:
    """The oracle, stated as the example prints it.

    Asserted separately from the golden because it is the *claim*: a golden
    file would keep passing if the difference drifted, so long as it drifted
    identically every run.
    """
    assert 'difference: 0.0e+00' in output, output


def test_the_example_matches_its_committed_output(output: str, pytestconfig: pytest.Config) -> None:
    if pytestconfig.getoption('--update-golden'):
        GOLDEN.write_text(output)
        pytest.skip(f'rewrote {GOLDEN.name} from this run')
    expected = GOLDEN.read_text()
    if output != expected:
        diff = '\n'.join(difflib.unified_diff(expected.splitlines(), output.splitlines(), 'committed', 'this run'))
        pytest.fail(f'the example no longer prints what the docs show:\n{diff}')
