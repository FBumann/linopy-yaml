"""The architecture walkthrough must keep running, or it stops being true.

``examples/walkthrough.py`` calls the real pipeline stage by stage, so any
signature change in the modules it narrates breaks this test rather than
leaving a plausible-looking script that no longer matches the code.

Running is the weaker half. The script also *claims* things — that the macro is
gone by stage 3, that ``var_p`` has 18 rows and not 24, that the degree-1
ceiling is caught with no data bound. A script that merely executes proves none
of them: 18 could silently become 31 and CI would stay green. So its whole
output is committed as ``examples/walkthrough.out`` and compared line for line.
When the pipeline legitimately changes, regenerate it —

    pixi run pytest tests/test_walkthrough.py --update-golden

— and the diff of that file is the review artifact: exactly how the
architecture's story changed, in the same PR that changed it.
"""

from __future__ import annotations

import pytest

from tests.conftest import EXAMPLES_DIR, assert_golden, run_example

WALKTHROUGH = EXAMPLES_DIR / 'walkthrough.py'
GOLDEN = EXAMPLES_DIR / 'walkthrough.out'


@pytest.fixture(scope='module')
def output() -> str:
    """One run of the whole pipeline, shared by both tests."""
    return run_example(WALKTHROUGH, 'walkthrough')


def test_walkthrough_matches_golden(output: str, pytestconfig: pytest.Config) -> None:
    assert_golden(
        output,
        GOLDEN,
        pytestconfig,
        drifted='the walkthrough narrates something the pipeline no longer does.\n'
        'If this run is the correct story, regenerate the golden file:\n'
        '    pixi run pytest tests/test_walkthrough.py --update-golden\n',
    )


def test_walkthrough_claims_hold(output: str) -> None:
    """The golden file catches *any* change; this names the ones that matter.

    Redundant with the diff above by construction, and kept anyway: when one of
    these breaks, the failure says which architectural property lapsed instead
    of pointing at a line number.
    """
    for stage in range(1, 8):
        assert f'[{stage}]' in output, f'stage {stage} did not run'

    assert 'weighted_sum' in output and "FunctionCallNode(name='sum'" in output, 'the macro expanded away'
    assert 'row absence' in output and 'not 24' in output, 'a mask removes rows, not values'
    assert 'ok (optimal)' in output
    assert 'degree 3' in output, 'the ceiling still bites — the objective takes 2 and no more'
    assert 'caught by check()' in output, 'and with no data bound, so CI can run it'
