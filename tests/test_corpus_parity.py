"""Every referenced model, built on both lanes.

``test_ports.py`` asks whether the relational lane reaches an optimum somebody
else published. This module asks the second question of the same corpus —
whether the eager linopy lane builds the same model — and it is the same corpus
because the data is already there: ``port_sources`` hands both lanes the same
tidy frames, so a model added to ``references.json`` is swept here the day it
lands rather than when someone remembers a glob.

Per model the claim is the strong one, three routes at once: the eager
objective, the relational objective, and the objective HiGHS reaches re-reading
the written LP file. ``test_ports.py`` supplies the fourth from outside, so a
model green in both modules has agreed with a published optimum four ways.

Importing ``tests.differential`` is the ``[linopy]`` guard, which is why this is
a module of its own rather than three more tests in ``test_ports.py``: that one
is linopy-free and pandas-free on purpose, and runs on the bare-install job.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.conftest import PORT_REFERENCES, port_model, port_sources
from tests.differential import differential

#: What the eager lane cannot build yet, keyed by model, valued by the issue
#: that owns it and the error it raises today. Strict, so the day a fix lands
#: these XPASS, the suite goes red, and the entry comes out in the same PR.
LANE_BUGS: dict[str, tuple[str, type[Exception]]] = {
    'osemosys_utopia': ('#894 — linopy has no objective-constant slot', ValueError),
}


def _case(name: str) -> Any:
    reason, raises = LANE_BUGS.get(name, (None, None))
    marks = [pytest.mark.xfail(reason=reason, raises=raises, strict=True)] if reason else []
    return pytest.param(name, marks=marks, id=name)


@pytest.mark.parametrize('name', [_case(n) for n in sorted(PORT_REFERENCES)])
def test_both_lanes_and_the_lp_file_reach_one_objective(name: str) -> None:
    """The harness is the whole assertion: it builds both lanes and re-solves the LP.

    Every port's ``sources`` already carries each dimension's own index table,
    which is what both lanes read.
    """
    with differential(port_model(name), port_sources(name), lp=True):
        pass  # a deliberate no-op: the harness asserted everything on the way in
