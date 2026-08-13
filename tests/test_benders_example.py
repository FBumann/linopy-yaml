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

import pytest

from tests.conftest import EXAMPLES_DIR, assert_golden, run_example

EXAMPLE = EXAMPLES_DIR / 'benders' / 'run.py'
GOLDEN = EXAMPLE.with_name('run.out')


@pytest.fixture(scope='module')
def output() -> str:
    return run_example(EXAMPLE, 'benders_example')


def test_the_decomposition_reaches_the_monolith(output: str) -> None:
    """The oracle, stated as the example prints it.

    Asserted separately from the golden because it is the *claim*: a golden
    file would keep passing if the difference drifted, so long as it drifted
    identically every run.
    """
    assert 'difference: 0.0e+00' in output, output


def test_the_example_matches_its_committed_output(output: str, pytestconfig: pytest.Config) -> None:
    assert_golden(output, GOLDEN, pytestconfig, drifted='the example no longer prints what the docs show:')
