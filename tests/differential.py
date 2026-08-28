"""The both-lanes harness: one model, two backends, one answer.

The differential test is this project's central claim — the same YAML must
mean the same thing on the eager linopy lane and on the streaming relational
one (docs/about/architecture.md, hard rule 3). Twelve tests made that claim by hand, in
seven files, each rebuilding the same fifteen lines: build eagerly, solve,
take the objective, re-parse the schema, lower it, bind sources, execute,
compare.

What the repetition cost was not correctness but *evenness*. Every copy
compared objectives and checked the status, but only five of the twelve also
wrote the LP file and re-solved it, and nothing recorded why the other seven
skipped that third opinion — so the strength of the claim varied with which
file you happened to be reading. Here it is one ``lp=True``, and a test that
does not ask for it is visibly choosing not to.

Importing this module is the ``[linopy]`` guard: it reaches the oracle
through ``tests.oracle``, so a bare install skips every module that uses the
harness at collection time, with no filename list to maintain.

Usage — the engine stays open for the length of the ``with`` block, so
per-variable primal checks live inside it::

    with differential(NONCONVEX_YAML, sources, lp=True) as run:
        assert run.result.to_pandas('op_cost') ...
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from math_spec import to_program

from lpspec.relational.engines.polars.engine import PolarsEngine
from lpspec.sources import tidy_sources
from tests.conftest import raw_of, schema_of, solve_written_file
from tests.oracle import linopy, lpspec_linopy

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from lpspec.relational.engines.polars.engine import Result

#: Both lanes hand the same numbers to the same solver, so they must agree to
#: solver precision, not to a fudge factor. One tolerance, one place.
RTOL = 1e-9


class NoFiniteAnswerError(AssertionError):
    """The fixture admits no finite optimum, so neither lane is on trial.

    An ``AssertionError`` because that is what it was and what every caller
    that does not catch it still wants: a failure naming the fixture. A class
    of its own because a caller generating its models — ``test_expression_sweep``
    — must tell "this data has no answer" from "the lanes disagree", and was
    doing it by matching the message text.
    """


@dataclass
class Agreement:
    """What the two lanes produced, for tests that assert past the objective."""

    oracle: float
    """The eager objective — the number both lanes had to reach."""

    model: linopy.Model
    """The eager model, for structural assertions (labels, masks, solution)."""

    result: Result
    """The relational solution; live until the ``with`` block exits."""

    engine: PolarsEngine
    lp: Path | None = None
    """The written LP file, when ``lp=True`` — already checked to agree."""


@contextmanager
def differential(
    model: str | Path | dict[str, Any],
    sources: Mapping[str, Any],
    *,
    lp: bool = False,
) -> Iterator[Agreement]:
    """Build ``model`` on both lanes with the same inputs; assert they agree.

    ``model`` is a ``Path`` to a file, the YAML text itself, or a raw dict —
    the eager lane only takes paths, so text and dicts are written to a
    temporary file here rather than in every caller.

    **Duals are not compared here, and cannot be.** An LP with alternative
    optima has many optimal dual solutions, and the two lanes hand HiGHS the
    same rows in a different order, so it lands on a different basis:
    ``genx_piecewise_fuel`` agrees on the objective to nine decimals, differs in
    2 of 72 entries of one primal, and in 12 of 48 entries of one dual. A
    lane-to-lane dual assertion would therefore be false rather than merely
    strict. What *is* checkable is a dual against a recording made from an
    instance designed to have a unique one, which is ``test_ports`` and
    ``test_corpus_parity``'s job rather than this harness's.

    Set ``lp=True`` to also write and re-solve the LP file, the third opinion.
    HiGHS reads that file, so a model carrying ``sos:`` must not ask for it:
    HiGHS has no SOS concept and its parser refuses the section outright, which
    is the same fact ``reformulate_sos='auto'`` handles on the eager side — a
    no-op for every model that declares no set, and what lets the oracle solve
    one that does.
    """
    schema = schema_of(model)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        path = model if isinstance(model, Path) else _write(work / 'model.yaml', model)

        m = lpspec_linopy.build(path, dict(sources))
        m.solve(solver_name='highs', output_flag=False, reformulate_sos='auto')
        oracle = float(m.objective.value)
        if not np.isfinite(oracle):
            raise NoFiniteAnswerError('the eager oracle is infeasible or unbounded — fix the data, not the tolerance')

        program = to_program(schema)
        with PolarsEngine() as engine:
            engine.build(program, tidy_sources(schema, dict(sources)))
            result = engine.solve()
            assert result.is_ok, f'the relational lane reached no solution: {result.status}'
            assert result.objective == pytest.approx(oracle, rel=RTOL), (
                f'the lanes disagree on the objective — relational {result.objective}, eager {oracle}'
            )
            _same_shape(engine.diagnostics(), m)

            lp_path = None
            if lp:
                lp_path = work / 'model.lp'
                engine.write(lp_path)
                assert solve_written_file(lp_path) == pytest.approx(oracle, rel=RTOL), (
                    f'the written {lp_path.name} re-solves to a different objective than both lanes reached'
                )

            yield Agreement(oracle=oracle, model=m, result=result, engine=engine, lp=lp_path)


def _same_shape(diagnostics: Any, eager: Any) -> None:
    """The two lanes built the same *model*, not merely the same answer.

    An objective, a dual vector and a re-solved LP file are all invariant to a
    column that cannot move: a variable pinned to ``[0, 0]``, or a row that is
    true whatever the solver does. So a lane could materialise either and every
    other assertion here would still pass — which is not hypothetical, it is
    how a first draft of ``absence: zero`` shipped an extra column per absent
    coordinate on the eager lane with the whole suite green.

    Counts rather than a set comparison: the two lanes name their columns
    differently by design (labels against a ``(name, coordinate)`` index), and
    the claim worth making is that the same declarations produced the same
    number of them.
    """
    assert diagnostics.columns == eager.nvars, (
        f'the lanes disagree on how many columns this model has — relational {diagnostics.columns}, eager {eager.nvars}'
    )
    assert diagnostics.rows == eager.ncons, (
        f'the lanes disagree on how many rows this model has — relational {diagnostics.rows}, eager {eager.ncons}'
    )


def _write(path: Path, model: str | dict[str, Any]) -> Path:
    import yaml as pyyaml

    path.write_text(model if isinstance(model, str) else pyyaml.safe_dump(raw_of(model)))
    return path
