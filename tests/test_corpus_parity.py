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

import json
from typing import Any

import polars as pl
import pytest

from lpspec.errors import LaneError
from lpspec.language import expand_piecewise, load_model
from tests.conftest import PORT_REFERENCES, PORTS_DIR, port_model, port_sources
from tests.differential import differential

#: The instance files the ports bind, for a test that needs a column the model
#: does not declare — ``port_sources`` filters those out, as it should.
PORTS_DATA = PORTS_DIR / 'data'

#: What the eager lane accepts and cannot build, keyed by model. `LaneError` is
#: the point of the pair: it pins the xfail to *this* refusal, where the bare
#: `ValueError` it used to name was satisfied by any bug that raised one.
#: Strict, so the day linopy grows an objective-constant slot these XPASS, the
#: suite goes red, and the entry comes out with the check in the same PR.
LANE_GAPS: dict[str, str] = {
    'osemosys_utopia': '#894 — linopy has no objective-constant slot',
}


def _case(name: str) -> Any:
    reason = LANE_GAPS.get(name)
    marks = [pytest.mark.xfail(reason=reason, raises=LaneError, strict=True)] if reason else []
    return pytest.param(name, marks=marks, id=name)


@pytest.mark.parametrize('name', [_case(n) for n in sorted(PORT_REFERENCES)])
def test_both_lanes_and_the_lp_file_reach_one_objective(name: str) -> None:
    """The harness is the whole assertion: it builds both lanes and re-solves the LP.

    Every port's ``sources`` already carries each dimension's own index table,
    which is what both lanes read.

    **A model whose expansion declares a set skips the file leg**, read off the
    expanded schema rather than a list so it cannot drift — ``method: sos2``
    emits an ``sos:`` block the file never wrote. HiGHS reads no SOS section,
    from an LP file or an MPS one — five LP spellings and the standard MPS
    header each come back ``kError`` with an empty model — so the re-solve would
    be checking the reader. What the writer put there is checked as bytes in
    ``test_lp_text``.
    """
    declares_a_set = bool(expand_piecewise(load_model(port_model(name))).sos)
    with differential(port_model(name), port_sources(name), lp=not declares_a_set) as run:
        _same_matrix(name, run)
        _eager_matches_the_recorded_duals(name, run)


def _same_matrix(name: str, run: Any) -> None:
    """The two lanes wrote the same coefficients, not merely the same shape.

    The strongest cross-lane claim available, and the one duals cannot make: an
    LP with alternative optima has many optimal primal *and* dual solutions, so
    comparing answers is comparing which vertex a solver happened to reach. The
    matrix has no such freedom — one model, one set of coefficients.

    Canonical rather than positional. Each constraint becomes the sorted multiset
    of its rows, each row the sorted multiset of its coefficients, so the
    comparison survives the two lanes numbering rows and columns differently —
    which they do, each labelling in its own declaration order.

    Structure is compared exactly and values approximately, because the two
    lanes reach a coefficient from the same data by a different order of
    operations: ``pypsa_ac_dc`` writes ``-0.556229726`` where the other writes
    ``-0.556229727``. How many terms a row has is a fact about the model; its
    last bit is not.
    """
    tables = run.engine._tables()
    for constraint, block in run.engine._constraint_blocks.items():
        if not block.height:
            continue
        got = _canonical(tables.matrix_block(block.start, block.start + block.height))
        want = _eager_matrix(run.model, constraint)
        assert [len(r) for r in got] == [len(r) for r in want], (
            f'{name}.{constraint}: the lanes wrote a different number of terms per row'
        )
        flat, expected = [c for r in got for c in r], [c for r in want for c in r]
        assert flat == pytest.approx(expected, rel=1e-9, abs=1e-12), (
            f'{name}.{constraint}: the lanes wrote different coefficients'
        )


def _canonical(matrix: pl.DataFrame) -> list[tuple[float, ...]]:
    """``(row, col, coeff)`` as a sorted multiset of sorted coefficient rows."""
    rows = matrix.group_by('row').agg(pl.col('coeff').sort()).get_column('coeff').to_list()
    return sorted(tuple(r) for r in rows)


def _eager_matrix(eager: Any, constraint: str) -> list[tuple[float, ...]]:
    """The same, off linopy's dense arrays — duplicate terms collapsed first.

    linopy stores ``x + 2 * x`` as two entries where the relational lane sums
    them into one, so an uncollapsed comparison reports a difference that is not
    one: on ``genx_piecewise_fuel`` it is 3408 entries against 2376.
    """
    import numpy as np

    c = eager.constraints[constraint]
    labels = np.asarray(c.labels).reshape(-1)
    variables = np.asarray(c.vars).reshape(len(labels), -1)
    coefficients = np.asarray(c.coeffs).reshape(len(labels), -1)

    rows = []
    for i, label in enumerate(labels):
        if label < 0:
            continue
        collapsed: dict[int, float] = {}
        for column, coefficient in zip(variables[i], coefficients[i], strict=True):
            if column >= 0:
                collapsed[int(column)] = collapsed.get(int(column), 0.0) + float(coefficient)
        rows.append(tuple(sorted(v for v in collapsed.values() if round(v, 12) != 0)))
    return sorted(rows)


def _eager_matches_the_recorded_duals(name: str, run: Any) -> None:
    """The eager lane against the price somebody else published, where there is one.

    ``test_ports`` asks this of the relational lane and cannot ask it here: it is
    linopy-free on purpose, for the bare-install job. So the second half of the
    claim lives in this module, where the oracle is already built — and until it
    did, the eager lane's duals were compared against nothing at all.

    Against the *recording* rather than against the other lane, because two lanes
    need not agree on a dual: an LP with alternative optima has many, and which
    one HiGHS returns depends on the order the rows reach it (see
    ``differential``). A recorded dual is a claim that this instance has a unique
    one, so both lanes owe it the same answer.
    """
    _check_recorded_duals(name, PORT_REFERENCES[name], run)


def _check_recorded_duals(name: str, entry: dict[str, Any], run: Any) -> None:
    """*entry*'s recorded duals against the eager lane, split out so a probe can pass a wrong one."""
    recorded = entry.get('duals')
    if not recorded:
        return
    for constraint, table in recorded.items():
        want = pl.DataFrame(table)
        dims = [c for c in want.columns if c != 'value']
        got = _tidy(run.model.constraints[constraint].dual, dims, want)
        want = want.with_columns(pl.col(d).cast(got.schema[d]) for d in dims).sort(dims)
        assert got[dims].equals(want[dims]), f'{name}.{constraint}: the eager dual is keyed differently'
        assert got['value'].to_list() == pytest.approx(want['value'].to_list(), rel=entry['rtol'], abs=1e-9), (
            f'{name}.{constraint}: the eager lane disagrees with {entry["provenance"]}'
        )


def _tidy(dual: Any, dims: list[str], like: pl.DataFrame) -> pl.DataFrame:
    """An eager dual as ``(dims…, value)``, keyed and sorted like *like*."""
    if not dims:
        return pl.DataFrame({'value': [float(dual.values.reshape(-1)[0])]})
    series = dual.to_series().dropna()
    keys = list(series.index)
    columns = {d: [k[i] if isinstance(k, tuple) else k for k in keys] for i, d in enumerate(dims)}
    frame = pl.DataFrame({**columns, 'value': [float(v) for v in series.to_numpy()]})
    return frame.with_columns(pl.col(d).cast(like.schema[d]) for d in dims).sort(dims)


def test_the_eager_dual_check_would_notice_a_wrong_price() -> None:
    """The probe the mutation table asked for.

    Deleting the comparison above leaves the suite green, because the comparison
    *is* the assertion — nothing else reads the eager lane's duals. So the guard
    needs a case that fails on purpose: a recording one entry away from the
    truth, which the check must refuse.

    ``monthly_budget`` because its dual is a short vector with distinct values,
    so a single perturbed entry cannot coincide with another and pass by luck.
    """
    name = 'monthly_budget'
    recorded = PORT_REFERENCES[name].get('duals')
    assert recorded, f'{name} is the probe because it records duals — give the probe another model'

    constraint, table = next(iter(recorded.items()))
    wrong = {**table, 'value': [v + 1.0 for v in table['value']]}
    entry = {**PORT_REFERENCES[name], 'duals': {constraint: wrong}}

    with differential(port_model(name), port_sources(name)) as run, pytest.raises(AssertionError, match=constraint):
        _check_recorded_duals(name, entry, run)


#: PyPSA's secant loss mode on ``pypsa_losses``' own network — its default, where
#: the ported tangent mode is deprecated. From
#: ``examples/ports/references/pypsa/pypsa_losses.py``'s ``secant_objective``,
#: pypsa 1.2.4, under the tolerances that script records.
SECANT_OBJECTIVE = 24432.20488089685

#: The loss each lossy line settles on, ``(snapshot, line)`` in sorted order.
#: The objective alone pins only the half-planes that bind, so this pins where
#: on the approximated curve the model sits — and the instance carries three
#: low-demand snapshots so that the flows reach the *early* segments too. With
#: the busy snapshots alone only the top two of five bound, and the other three
#: coefficients could be anything.
SECANT_LOSSES = [
    4.319999999999999,
    1.6236570501210394,
    4.319999999999999,
    1.2607240286010466,
    4.319999999999999,
    2.034868195461712,
    0.05571402370778852,
    0.031231008229264477,
    0.3029679080268904,
    0.14940704839558197,
    1.4593292560812914,
    0.753085939715055,
]


def test_the_two_loss_approximations_are_one_model() -> None:
    """PyPSA's other loss mode is the same rows with different numbers in them.

    ``pypsa_losses`` ports the tangent approximation. The secant one — PyPSA's
    **default**, the tangents being deprecated — emits the identical shape: one
    half-plane per segment per sign of the flow, ``loss ± slope * f >= offset``.
    Only the coefficients differ, and how many of them there are.

    So it gets no model file of its own. A second YAML byte-identical to the
    first would assert a second model that does not exist; binding *this* model
    to the other instance is the claim, and it is the stronger one — the two
    approximations are one thing the language says once.

    The coefficients are **dumped from PyPSA**, not re-derived here. Where its
    breakpoints fall — and even how many there are — comes out of a tolerance
    heuristic that is its business, so reproducing it would put their algorithm
    in this repository, where a change on their side becomes a failure on ours.
    ``pypsa_losses.py::secant_coefficients`` is the provenance, the same footing
    as every other committed instance.

    Through ``differential``, so the secant instance is held to everything the
    corpus holds a port to: both lanes, the written LP file, and the coefficient
    matrices agreeing entry for entry.
    """
    sources = dict(port_sources('pypsa_losses'))
    sources |= {
        name: pl.DataFrame(table)
        for name, table in json.loads((PORTS_DATA / 'pypsa_losses_secants.json').read_text()).items()
    }

    with differential(port_model('pypsa_losses'), sources, lp=True) as run:
        assert run.result.objective == pytest.approx(SECANT_OBJECTIVE, rel=1e-9), (
            'the secant instance reaches the number PyPSA reaches under its own default mode'
        )
        loss = run.result.primal('loss').sort(['snapshot', 'line'])['value'].to_list()
        assert loss == pytest.approx(SECANT_LOSSES, rel=1e-9), (
            'and sits where PyPSA sits on the approximated curve, line by line'
        )
