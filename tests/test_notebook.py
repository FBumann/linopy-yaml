"""The notebook pages must keep running, and keep claiming true things.

``docs/interactive.ipynb`` teaches the three loops a session actually has —
update, grow a coordinate set, patch the spec — plus the verb that reads a built
row back when one of them lands wrong, and ``docs/lifecycle.ipynb``
aims them at linopy's `fix` / `relax` / `remove`. Every cell is a real call, so a
signature change breaks this test rather than leaving a page that reads fine and
errors in a reader's kernel.

Running is the weaker half, as with ``test_walkthrough.py``. The prose also
*claims* things: that the update loop loaded one model for three solves, that
growing an axis loads a second, that a pin moves bounds rather than labels,
that a masked-out generator leaves the balance row a term short. A
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
from math_spec import to_spec

from tests.conftest import EXAMPLES_DIR

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip('IPython', reason='the notebook displays through IPython, which the bare install lacks')

DOCS_DIR = EXAMPLES_DIR.parent / 'docs'
LOOPS = DOCS_DIR / 'interactive.ipynb'
LIFECYCLE = DOCS_DIR / 'lifecycle.ipynb'
REGION = DOCS_DIR / 'region.ipynb'


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


@pytest.fixture(scope='module')
def region() -> tuple[dict[str, Any], str]:
    matplotlib = pytest.importorskip('matplotlib', reason='the region page draws, which is the [plot] extra')
    matplotlib.use('Agg')
    return run(REGION)


@pytest.mark.parametrize('notebook', [LOOPS, LIFECYCLE, REGION], ids=lambda p: p.name)
def test_the_tree_copy_has_no_outputs(notebook: Path) -> None:
    """A committed output is an unreviewable diff, and one this test would not check."""
    document = json.loads(notebook.read_text())
    stored = [cell for cell in document['cells'] if cell.get('outputs') or cell.get('execution_count') is not None]
    assert not stored, f'{notebook.name}: {len(stored)} cell(s) carry stored output — clear them before committing'


def test_the_update_loop_stays_on_the_fast_path(session: tuple[dict[str, Any], str]) -> None:
    namespace, _ = session
    reused = namespace['reused']
    assert (reused.loads, reused.solves) == (1, 3), 'the notebook says three answers came off one loaded model'


def test_growing_a_coordinate_set_loads_again(session: tuple[dict[str, Any], str]) -> None:
    namespace, _ = session
    assert namespace['grown'].loads == 2, 'the notebook says new coordinates cost a reload, and why that is fine'
    assert namespace['schedule'].height == 36, 'twelve snapshots against three generators'


def test_a_update_answers_what_a_fresh_build_answers(session: tuple[dict[str, Any], str]) -> None:
    namespace, _ = session
    assert namespace['updated'] == pytest.approx(namespace['fresh']), (
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


def test_a_masked_generator_leaves_the_balance_row_short(session: tuple[dict[str, Any], str]) -> None:
    """The debugging claim: the file names three generators and the built row carries two."""
    namespace, _ = session
    fleet = namespace['fleet'].row('power_balance', snapshot=3)
    short = namespace['short'].row('power_balance', snapshot=3)
    assert fleet.terms['coordinate'].to_list() == ['3, wind', '3, solar', '3, gas'], (
        'the fleet as declared puts all three generators in the balance row'
    )
    assert short.terms['coordinate'].to_list() == ['3, wind', '3, solar'], (
        "a capacity of zero deletes gas's column, so the row that built has no term to carry it"
    )
    assert namespace['answer'].termination_condition == 'infeasible', (
        'and the page needs a model that actually stopped having an answer'
    )


def test_the_session_leaves_a_file(session: tuple[dict[str, Any], str]) -> None:
    """``to_yaml`` on the patched spec is what the reader diffs against the model."""
    import yaml

    namespace, _ = session
    patched = to_spec(namespace['spec'])
    written = patched.to_yaml()
    assert to_spec(yaml.safe_load(written)).to_dict() == patched.to_dict(), (
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


# --------------------------------------------------------------------------
# docs/region.ipynb
# --------------------------------------------------------------------------


def test_the_region_page_reads_its_edges_off_the_data(region: tuple[dict[str, Any], str]) -> None:
    """The prose walks six vertices and names the constraint behind each edge."""
    namespace, _ = region
    free = namespace['free']
    assert free.pieces == (), 'the first trace leaves the binaries free'
    assert free.vertices.rows() == [
        (36.0, 40.0),
        (120.0, 40.0),
        (112.0, 48.0),
        (48.0, 88.0),
        (40.0, 90.0),
        (36.0, 86.0),
    ], 'the six vertices the page reads off: the two loads, the boiler corner, three well edges, the peaker cap'


def test_the_region_page_puts_the_optimum_on_the_floor_and_off_the_corner(region: tuple[dict[str, Any], str]) -> None:
    """The claim that makes the page's first picture worth drawing: the CHP's ratio, not the load, sets the heat."""
    namespace, _ = region
    assert namespace['optimum'] == (40.0, 40.0), 'the power load binds and the CHP dumps four units of heat'


def test_the_region_page_finds_the_tight_hour(region: tuple[dict[str, Any], str]) -> None:
    namespace, _ = region
    stacked = namespace['stacked']
    assert stacked.columns == ['hour', 'vertex', 'heat', 'power'], 'the long form, an hour column prepended'
    sliver = stacked.filter(stacked['hour'] == 2)
    assert sliver.height == 3, 'hour 2 leaves a triangle, which is the sliver the prose points at'
    assert (sliver['heat'].max(), sliver['power'].max()) == (86.4, 68.0), 'and how far that sliver reaches'


def test_the_region_page_shows_what_the_hull_hides(region: tuple[dict[str, Any], str]) -> None:
    """Five states meet hour 0, and the hull's long well edge belongs to none of them."""
    namespace, printed = region
    each = namespace['each']
    assert '5 of 8 combinations can meet the loads in hour 0' in printed, 'the page counts the states it draws'
    assert [len(piece.vertices) for piece in each.pieces] == [4, 2, 5, 4, 7], (
        'a box without the CHP, the CHP alone as a segment, two pairs, and all three'
    )
    assert each.vertices.rows() == namespace['free'].vertices.rows(), 'the hull of the pieces is what free traced'
    all_on = each.pieces[-1]
    assert all_on.fixed == {
        'running[t=0, unit=chp]': 1,
        'running[t=0, unit=boiler]': 1,
        'running[t=0, unit=peaker]': 1,
    }, 'the last combination counted is every unit on'
    inside = {(112.0, 48.0), (48.0, 88.0)}
    assert inside.isdisjoint(all_on.vertices.rows()), (
        'the well edge the page says no state reaches is not a vertex of the all-on piece'
    )
    assert namespace['chp_and_boiler'].label.endswith('running[t=0, unit=peaker]=0'), (
        'the piece drawn alone is the one with the peaker off'
    )


def test_the_region_page_breaks_the_model_on_purpose(region: tuple[dict[str, Any], str]) -> None:
    namespace, printed = region
    assert 'the feasible region is unbounded toward (+1·heat, +0·power)' in printed, (
        'dropping the capacity rows is caught at the first direction nothing caps'
    )
    stiff = namespace['stiff_each']
    with_boiler = [piece for piece in stiff.pieces if piece.fixed['running[t=0, unit=boiler]'] == 1]
    assert with_boiler and all(piece.vertices['heat'].min() >= 40.0 for piece in with_boiler), (
        'a boiler that cannot idle makes every state it runs in start at 40 of heat'
    )
    flat_out = set(namespace['free'].vertices.rows()) - {(36.0, 40.0)}
    assert flat_out <= set(stiff.vertices.rows()), (
        'while the hull keeps every corner but the load corner, which is why it would not have said'
    )
