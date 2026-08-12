"""The runner: bind data to a YAML model and execute it. Not a modeling API.

Math is defined in YAML only — there is no Python API for constructing models,
and the logical plan is internal. Four verbs: ``check``, ``build`` (YAML +
sources → live executor), ``solve`` and ``write``.

This is the product path (docs/ARCHITECTURE.md): validated at load time,
lowered to the plan, executed relationally. linopy exists only in the optional
compatibility/oracle layer (``import lpspec.linopy``).

Example::

    import lpspec as lps

    result = lps.solve(
        'model.yaml',
        {'p_max': 'p_max.parquet', 'load': 'load.parquet'},
        coords={'snapshot': range(8760)},
    )
    result.objective
    result.primal('p')  # tidy polars.DataFrame (coords..., value)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lpspec.errors import LpspecWarning
from lpspec.language.validation import load_model
from lpspec.lowering import advice, lower_program
from lpspec.relational.engines.polars.executor import PolarsExecutor
from lpspec.relational.sinks import solver, writer
from lpspec.sources import tidy_sources

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lpspec.language.model import Model
    from lpspec.relational.result import Result

#: Re-exported: parsing and validating a model is the *language's* job, and a
#: consumer that binds no data (``typeset``) must be able to reach it without
#: reaching the runner. Callers keep saying ``lps.load_model``.
__all__ = ['build', 'check', 'load_model', 'solve', 'write']


def check(model: str | Path | dict[str, Any] | Model) -> Model:
    """Compile-check a model without data: parse, expand, validate, lower.

    Lowering needs no sources, so this works on a bare YAML file — the CI
    verb for model repositories.

    Advice that stops short of an error — a declared dimension nothing uses as
    an axis, say — is issued as :class:`~lpspec.errors.LpspecWarning`, here
    and only here: ``check`` is the explicit gate, so the advice never costs a
    build or a solve.

    Returns:
        The validated schema.

    Raises:
        LanguageError: If the model uses a construct outside the streaming
            language.
        ValueError: If the schema or an expression does not parse.
    """
    schema = load_model(model)
    program = lower_program(schema)
    for note in advice(program):
        warnings.warn(note, LpspecWarning, stacklevel=2)
    return schema


def build(
    model: str | Path | dict[str, Any] | Model,
    sources: Mapping[str, Any],
    *,
    coords: dict[str, Any] | None = None,
) -> PolarsExecutor:
    """Build *model* on the relational engine and return the executor.

    One build can feed more than one sink: call ``ex.solve()`` and
    ``ex.write(path)`` on the same object.

    Args:
        model: The YAML file, its parsed mapping, or a loaded :class:`Model`.
        sources: Parameter names to parquet paths or in-memory tables, and
            optionally dimension names to index tables.
        coords: Dimension labels, where neither *sources* nor the YAML
            declares them.

    Raises:
        LanguageError: If the model uses a construct outside the streaming
            language — the message names the construct and its context.
    """
    schema = load_model(model)
    program = lower_program(schema)
    ex = PolarsExecutor()
    try:
        ex.build(program, tidy_sources(schema, dict(sources), coords))
    except BaseException:
        ex.close()
        raise
    return ex


def solve(
    model: str | Path | dict[str, Any] | Model,
    sources: Mapping[str, Any],
    solver_options: Mapping[str, Any] | None = None,
    solver_name: str = 'highs',
    **build_kwargs: Any,
) -> Result:
    """Build and solve in one call.

    Which solver serves the solve is the *caller's* decision, linopy's
    spelling: the same file solves the same model either way, so nothing in
    the YAML names one.

    The sink is resolved before the build, as ``write`` checks the suffix
    first: a caller who named a sink nothing can serve should not pay for a
    model.

    Args:
        model: As :func:`build` takes it.
        sources: As :func:`build` takes them.
        solver_options: Forwarded verbatim, in that solver's vocabulary.
            Build options stay separate and never reach the solver.
        solver_name: ``highs``, which ships with the package, or ``gurobi``,
            needing the ``[gurobi]`` extra.
        **build_kwargs: Passed on to :func:`build`.

    Returns:
        The solution, with the executor still attached — its label frames
        back ``result.primal(...)``. Nothing has to be released, though
        ``result.close()`` drops a large model early.
    """
    solver(solver_name)
    ex = build(model, sources, **build_kwargs)
    try:
        return ex.solve(solver_options=solver_options, solver_name=solver_name)
    except BaseException:
        ex.close()
        raise


def write(
    model: str | Path | dict[str, Any] | Model,
    sources: Mapping[str, Any],
    out: str | Path,
    **build_kwargs: Any,
) -> Path:
    """Build and stream the model to a file; format from the suffix.

    Which formats exist is the writer family's answer, not a branch here — this
    verb owns *when* to build. The suffix is checked **before** the build, so a
    caller who named a format nothing can write does not pay for a model first.

    Args:
        model: As :func:`build` takes it.
        sources: As :func:`build` takes them.
        out: Destination path; its suffix picks the format.
        **build_kwargs: Passed on to :func:`build`.

    Returns:
        The path written.
    """
    out = Path(out)
    writer(out.suffix.lower())
    with build(model, sources, **build_kwargs) as ex:
        ex.write(out)
    return out
