"""The interactive notebook must keep running, and keep claiming true things.

``examples/interactive.ipynb`` teaches the three loops a session actually has —
rebind, grow a coordinate set, patch the spec — and every one of them is a real
call, so a signature change breaks this test rather than leaving a notebook that
reads fine and errors in a reader's kernel.

Running is the weaker half, as with ``test_walkthrough.py``. The prose also
*claims* things: that the rebind loop loaded one model for three solves, that
growing an axis loads a second, that the ramp limit changes the answer. A
notebook that executed but had stopped doing any of that would still be green
here without these assertions, and would teach the wrong loop.

Cells are exec'd in order in one namespace rather than run through a kernel:
the property under test is that the notebook works top to bottom, and a kernel
would add jupyter_client and ipykernel to the dev group to prove the same thing.
"""

from __future__ import annotations

import contextlib
import io
import json
from typing import Any

import pytest

from tests.conftest import EXAMPLES_DIR

pytest.importorskip('IPython', reason='the notebook displays through IPython, which the bare install lacks')

NOTEBOOK = EXAMPLES_DIR / 'interactive.ipynb'


def cells(kind: str) -> list[str]:
    document = json.loads(NOTEBOOK.read_text())
    return [''.join(cell['source']) for cell in document['cells'] if cell['cell_type'] == kind]


@pytest.fixture(scope='module')
def session() -> tuple[dict[str, Any], str]:
    """One top-to-bottom run: the namespace it ends with, and what it printed.

    Runs from ``examples/`` because that is where a reader opens it and why the
    notebook says ``dispatch.yaml`` rather than a path.
    """
    namespace: dict[str, Any] = {'__name__': '__notebook__'}
    printed = io.StringIO()
    with contextlib.chdir(EXAMPLES_DIR), contextlib.redirect_stdout(printed):
        for source in cells('code'):
            exec(compile(source, str(NOTEBOOK), 'exec'), namespace)
    return namespace, printed.getvalue()


def test_the_tree_copy_has_no_outputs() -> None:
    """A committed output is an unreviewable diff, and one this test would not check."""
    document = json.loads(NOTEBOOK.read_text())
    stored = [cell for cell in document['cells'] if cell.get('outputs') or cell.get('execution_count') is not None]
    assert not stored, f'{len(stored)} cell(s) carry stored output — clear them before committing'


def test_the_rebind_loop_stays_on_the_fast_path(session: tuple[dict[str, Any], str]) -> None:
    namespace, _ = session
    warm = namespace['warm']
    assert (warm.loads, warm.solves) == (1, 3), 'the notebook says three answers came off one loaded model'


def test_growing_a_coordinate_set_loads_again(session: tuple[dict[str, Any], str]) -> None:
    namespace, _ = session
    assert namespace['grown'].loads == 2, 'the notebook says new coordinates cost a reload, and why that is fine'
    assert namespace['answer'].primal('p').height == 36, 'twelve snapshots against three generators'


def test_a_rebind_answers_what_a_fresh_build_answers(session: tuple[dict[str, Any], str]) -> None:
    namespace, _ = session
    assert namespace['sweep'][90.0] == pytest.approx(namespace['fresh']), (
        'the equality the notebook offers as the oracle for a loop that looks wrong'
    )


def test_the_added_constraint_changes_the_answer(session: tuple[dict[str, Any], str]) -> None:
    namespace, _ = session
    assert namespace['ramped'] > namespace['base'], (
        'the ramp limit has to bind — a structural edit with no effect teaches nothing'
    )


def test_a_refused_edit_says_what_is_wrong(session: tuple[dict[str, Any], str]) -> None:
    _, printed = session
    assert 'does not name a declared dimension' in printed, 'the load-time error is the notebook error message'


def test_the_session_leaves_a_file(session: tuple[dict[str, Any], str]) -> None:
    """``to_yaml`` on the patched spec is what the reader diffs against the model."""
    import yaml

    import lpspec as lps

    namespace, _ = session
    patched = lps.load_model(namespace['spec'])
    written = patched.to_yaml()
    assert lps.load_model(yaml.safe_load(written)).to_dict() == patched.to_dict(), (
        'the review copy has to reload as the model it was written from'
    )
    assert 'ramp_up' in written and 'ramp_up' not in (EXAMPLES_DIR / 'dispatch.yaml').read_text(), (
        'and to differ from the file on disk — that difference is what the reader commits'
    )
