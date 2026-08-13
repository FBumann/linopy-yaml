"""The two `solve_over` examples must keep making their claims, or they are not evidence.

`examples/rolling/` and `examples/myopic/` are the two coupled strategies —
windows over one dimension, and periods that hand a fleet forward. Each is a
claim that the driver composes into the thing people actually run, and a claim
is worth what the check behind it is worth. So each has one *substantive*
assertion here, apart from its golden:

- rolling: lookahead really does close the myopia gap, and enough of it reaches
  full foresight exactly
- myopic: the fleet each period starts from is the one the last period ended
  with, which is the promise `carry` makes

Committed output for the rest, for the reason `test_benders_example.py` gives:
a page that shows output is promising what a reader will see, and a diff is how
that promise stays true. Regenerate with ``--update-golden``.

The scenario sweep has no example of its own — it is the uncoupled case, so
`docs/api.md`'s three lines are the whole of it and there is nothing an example
would add.
"""

from __future__ import annotations

import pytest

from tests.conftest import EXAMPLES_DIR, assert_golden, run_example

STRATEGIES = ['rolling', 'myopic']


@pytest.fixture(scope='module')
def outputs() -> dict[str, str]:
    return {name: run_example(EXAMPLES_DIR / name / 'run.py', f'{name}_example') for name in STRATEGIES}


def test_lookahead_closes_the_myopia_gap(outputs: dict[str, str]) -> None:
    """Asserted apart from the golden because it is the *point*.

    A golden alone would keep passing if every number drifted together — and
    in particular if the gap went to zero because storage stopped being used,
    which is the degenerate model this example was rewritten to stop being.
    """
    lines = [line for line in outputs['rolling'].splitlines() if line.startswith('rolling')]
    gaps = [float(line.rsplit('+', 1)[1].rstrip('% ')) for line in lines]
    peaks = [float(line.split('peak soc')[1].split()[0]) for line in lines]

    assert gaps[0] > 0, 'a window with no lookahead must pay for its myopia'
    assert gaps == sorted(gaps, reverse=True), 'more lookahead must never cost more'
    assert gaps[-1] == 0.0, 'enough lookahead must reach full foresight'
    assert all(peak > 0 for peak in peaks), (
        'the store must cycle in every schedule — a gap that came from storage '
        'going unused would be arithmetic, not myopia'
    )


def test_the_pathway_inherits_each_fleet(outputs: dict[str, str]) -> None:
    """The example asserts this internally; this is the check that it ran at all."""
    assert 'each period starts from the fleet the last one left' in outputs['myopic']


@pytest.mark.parametrize('name', STRATEGIES)
def test_the_example_matches_its_committed_output(
    name: str, outputs: dict[str, str], pytestconfig: pytest.Config
) -> None:
    assert_golden(
        outputs[name],
        EXAMPLES_DIR / name / 'run.out',
        pytestconfig,
        drifted=f'the {name} example no longer prints what the docs show:',
    )
