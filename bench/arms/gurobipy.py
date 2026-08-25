"""gurobipy, both dialects — the model is hand-written, in `bench/models/<case>/`.

One runtime for two arms: `gurobipy-loop` and `gurobipy-matrix` differ only in
which formulation module they call, and everything around that call — reading
the parquet, the environment, the seam, the counts — is the same and belongs
here rather than twice in the models.

**The seam is `update()`, and it is inside the clock.** gurobipy defers every
`addVar` and `addConstr` until the model is flushed, so timing the calls alone
measures a queue and not a model. `build_gurobi` on our own arm ends with the
same call, which is what makes the two comparable.

**`OutputFlag` goes off at `Env` construction**, not after: set later, the
licence banner has already been written, and it lands inside the measurement.

**Counts are read after the clock stops** — touching `NumVars` forces the
update this arm has just paid for deliberately.

There is one sink. An LP file would measure Gurobi's writer rather than ours,
and this arm cannot reach HiGHS at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bench.arms import Counts

#: Where this arm can hand a model over.
SINKS = ('gurobi',)


class Prepared(NamedTuple):
    """What the timed verbs need. The parquet is *not* read here — reading it is
    this arm's own cost, exactly as it is every other arm's."""

    dialect: str
    case_name: str
    paths: dict[str, str]


def formulation(case_name: str, dialect: str) -> Any:
    """The case's model in *dialect*, or None where nobody has written one.

    A case package names what it holds in `FORMULATIONS`; a case that holds
    nothing has no package at all, which is the same answer.
    """
    import importlib

    try:
        case = importlib.import_module(f'bench.models.{case_name}')
    except ModuleNotFoundError:
        return None
    return getattr(case, 'FORMULATIONS', {}).get(dialect)


def prepare(dialect: str, case_name: str, size: str, paths: dict[str, str], options: Mapping[str, Any]) -> Prepared:
    del size, options
    return Prepared(dialect, case_name, dict(paths))


def _built(prepared: Prepared) -> tuple[Any, Any]:
    """The environment and a flushed model — every timed verb's whole body."""
    import gurobipy as gp
    import polars as pl

    tables = {name: pl.read_parquet(path) for name, path in prepared.paths.items()}
    env = gp.Env(params={'OutputFlag': 0})
    model = formulation(prepared.case_name, prepared.dialect).build(env, tables)
    model.update()
    return env, model


def _counts(model: Any) -> Counts:
    return {'columns': model.NumVars, 'rows': model.NumConstrs, 'nonzeros': model.NumNZs}


def _release(env: Any, model: Any) -> None:
    """Both, or the next round's peak is this round's high-water mark as well."""
    model.dispose()
    env.dispose()


def build_and_emit(sink: str, prepared: Prepared) -> Counts:
    """Build the model and flush it — for this arm those are one act.

    There is no second hand-off to time: a populated `gurobipy.Model` is what
    `addVar` and `addMConstr` produce directly, where our own arm reaches it by
    handing a built matrix to `build_gurobi`. That difference is the
    measurement.
    """
    del sink
    env, model = _built(prepared)
    try:
        return _counts(model)
    finally:
        _release(env, model)


def build_only(prepared: Prepared) -> Counts:
    """The same work: this arm has no sink to leave out."""
    env, model = _built(prepared)
    try:
        return _counts(model)
    finally:
        _release(env, model)


def objective(prepared: Prepared) -> float:
    """Solve, and return what the arms are checked against."""
    env, model = _built(prepared)
    try:
        model.optimize()
        if model.Status != 2:
            raise RuntimeError(f'gurobi finished with status {model.Status}, not optimal')
        return float(model.ObjVal)
    finally:
        _release(env, model)
