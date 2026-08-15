"""The notebook pages must keep running, and keep claiming true things.

``docs/interactive.ipynb`` teaches the three loops a session actually has —
rebind, grow a coordinate set, patch the spec — and ``docs/lifecycle.ipynb``
aims them at linopy's `fix` / `relax` / `remove`. Every cell is a real call, so a
signature change breaks this test rather than leaving a page that reads fine and
errors in a reader's kernel.

Running is the weaker half, as with ``test_walkthrough.py``. The prose also
*claims* things: that the rebind loop loaded one model for three solves, that
growing an axis loads a second, that a pin moves bounds rather than labels. A
page that executed but had stopped doing any of that would still be green here
without these assertions, and would teach the wrong loop.

Cells are exec'd in order in one namespace rather than run through a kernel:
the property under test is that the notebook works top to bottom, and a kernel
would add jupyter_client and ipykernel to the dev group to prove the same thing.
The site does run it on one — ``execute: true`` in mkdocs.yml — so a build is
the second place this would fail, several minutes later and only on a push.
"""

from __future__ import annotations

import contextlib
import io
import json
from typing import TYPE_CHECKING, Any

import pytest

from tests.conftest import EXAMPLES_DIR

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip('IPython', reason='the notebook displays through IPython, which the bare install lacks')

DOCS_DIR = EXAMPLES_DIR.parent / 'docs'
LOOPS = DOCS_DIR / 'interactive.ipynb'
LIFECYCLE = DOCS_DIR / 'lifecycle.ipynb'


def run(notebook: Path) -> tuple[dict[str, Any], str]:
    """One top-to-bottom run: the namespace it ends with, and what it printed.

    Runs from ``docs/`` because that is where the pages sit, and so where both a
    reader's kernel and mkdocs-jupyter's start them — which is what makes
    ``../examples/dispatch.yaml`` resolve.
    """
    document = json.loads(notebook.read_text())
    namespace: dict[str, Any] = {'__name__': '__notebook__'}
    printed = io.StringIO()
    with contextlib.chdir(DOCS_DIR), contextlib.redirect_stdout(printed):
        for cell in document['cells']:
            if cell['cell_type'] == 'code':
                exec(compile(''.join(cell['source']), str(notebook), 'exec'), namespace)
    return namespace, printed.getvalue()


@pytest.fixture(scope='module')
def session() -> tuple[dict[str, Any], str]:
    return run(LOOPS)


@pytest.fixture(scope='module')
def lifecycle() -> tuple[dict[str, Any], str]:
    return run(LIFECYCLE)


@pytest.mark.parametrize('notebook', [LOOPS, LIFECYCLE], ids=lambda p: p.name)
def test_the_tree_copy_has_no_outputs(notebook: Path) -> None:
    """A committed output is an unreviewable diff, and one this test would not check."""
    document = json.loads(notebook.read_text())
    stored = [cell for cell in document['cells'] if cell.get('outputs') or cell.get('execution_count') is not None]
    assert not stored, f'{notebook.name}: {len(stored)} cell(s) carry stored output — clear them before committing'


def test_the_rebind_loop_stays_on_the_fast_path(session: tuple[dict[str, Any], str]) -> None:
    namespace, _ = session
    warm = namespace['warm']
    assert (warm.loads, warm.solves) == (1, 3), 'the notebook says three answers came off one loaded model'


def test_growing_a_coordinate_set_loads_again(session: tuple[dict[str, Any], str]) -> None:
    namespace, _ = session
    assert namespace['grown'].loads == 2, 'the notebook says new coordinates cost a reload, and why that is fine'
    assert namespace['schedule'].height == 36, 'twelve snapshots against three generators'


def test_a_rebind_answers_what_a_fresh_build_answers(session: tuple[dict[str, Any], str]) -> None:
    namespace, _ = session
    assert namespace['rebound'] == pytest.approx(namespace['fresh']), (
        'the equality the notebook offers as the oracle for a loop that looks wrong'
    )


def test_the_added_constraint_changes_the_answer(session: tuple[dict[str, Any], str]) -> None:
    namespace, _ = session
    assert namespace['ramped'] > namespace['base'], (
        'the ramp limit has to bind — a structural edit with no effect teaches nothing'
    )


def test_pinning_a_variable_stays_on_the_fast_path(lifecycle: tuple[dict[str, Any], str]) -> None:
    """The claim that makes a fix worth spelling as bounds rather than as a row."""
    namespace, _ = lifecycle
    assert namespace['pinning'].loads == 1, 'a pin writes bounds, so the solver keeps the model it has loaded'
    assert namespace['held'] > namespace['unpinned'], 'holding gas at 60 has to cost something, or it pins nothing'


def test_a_refused_edit_says_what_is_wrong(session: tuple[dict[str, Any], str]) -> None:
    _, printed = session
    assert 'does not name a declared dimension' in printed, 'the load-time error is the notebook error message'


def test_the_session_leaves_a_file(session: tuple[dict[str, Any], str]) -> None:
    """``to_yaml`` on the patched spec is what the reader diffs against the model."""
    import yaml

    import charter as lps

    namespace, _ = session
    patched = lps.load_model(namespace['spec'])
    written = patched.to_yaml()
    assert lps.load_model(yaml.safe_load(written)).to_dict() == patched.to_dict(), (
        'the review copy has to reload as the model it was written from'
    )
    assert 'ramp_up' in written and 'ramp_up' not in (EXAMPLES_DIR / 'dispatch.yaml').read_text(), (
        'and to differ from the file on disk — that difference is what the reader commits'
    )


def test_integrality_is_a_declaration_and_costs_the_duals(lifecycle: tuple[dict[str, Any], str]) -> None:
    """The relax page's claim: a domain edit is a rebuild, and a MILP has no prices."""
    namespace, printed = lifecycle
    assert namespace['milp'].has_primal, 'the integer solve still answers'
    assert 'duals are undefined for a mixed-integer model' in printed, 'and says why it cannot be priced'
    assert namespace['relaxed'].dual('power_balance').height == 6, 'the continuous declaration prices every row'


def test_removing_a_constraint_moves_the_answer(lifecycle: tuple[dict[str, Any], str]) -> None:
    namespace, _ = lifecycle
    assert namespace['with_ramp'] > namespace['without_ramp'], 'popping the key has to give the objective back'
