"""The runner: bind data to a YAML model and execute it. Not a modeling API.

Math is defined in YAML only — there is no Python API for constructing models,
and the logical plan is internal. Four verbs: ``check``, ``build`` (YAML +
sources → a :class:`BoundModel`), ``solve`` and ``write``.

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
from typing import TYPE_CHECKING, Any, Literal

from lpspec.errors import DataError, LpspecWarning
from lpspec.language.validation import load_model
from lpspec.lowering import advice, lower_program
from lpspec.relational.engines.polars.executor import PolarsExecutor
from lpspec.relational.sinks import solver, writer
from lpspec.sources import tidy_sources

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lpspec.language.model import Model
    from lpspec.relational.result import Diagnostics, Result

#: Re-exported: parsing and validating a model is the *language's* job, and a
#: consumer that binds no data (``typeset``) must be able to reach it without
#: reaching the runner. Callers keep saying ``lps.load_model``.
__all__ = ['build', 'check', 'load_model', 'solve', 'write']


def check(model: str | Path | dict[str, Any] | Model) -> Model:
    """Compile-check a model without data: parse, expand, validate, lower.

    Lowering needs no sources, so this works on a bare YAML file — the CI
    verb for model repositories. Raises :class:`LanguageError` when the model
    uses a construct outside the streaming language, ``ValueError`` for
    schema/expression problems. Returns the validated schema.

    Advice that stops short of an error — a declared dimension nothing uses as
    an axis, say — is issued as :class:`~lpspec.errors.LpspecWarning`, here
    and only here: ``check`` is the explicit gate, so the advice never costs a
    build or a solve.
    """
    schema = load_model(model)
    program = lower_program(schema)
    for note in advice(program):
        warnings.warn(note, LpspecWarning, stacklevel=2)
    return schema


class BoundModel:
    """A model with your data bound to it — what :func:`build` returns.

    Three nouns, each arrow adding one thing: a ``Model`` is the math,
    a ``BoundModel`` is the math with your data, a ``Result`` is one answer.

        ``load_model`` → ``Model`` → ``build`` → ``BoundModel`` → ``solve`` → ``Result``

    Named for what it holds rather than for what built it. The engine
    underneath is the swappable half of the package, and which one ran is not
    something a top-level verb's return type should say.

    One build feeds more than one sink — :meth:`solve` and :meth:`write` on the
    same object — :meth:`rebind` puts new numbers on it without re-reading the
    YAML or re-lowering the plan, and :meth:`diagnostics` says what it did.
    **Nothing has to be released**: the built model is frames this process owns,
    and :meth:`close` hands a large one back early.
    """

    def __init__(
        self,
        schema: Model,
        sources: Mapping[str, Any],
        coords: Mapping[str, Any] | None = None,
    ) -> None:
        self._schema = schema
        self._program = lower_program(schema)
        self._sources = dict(sources)
        self._coords = dict(coords or {})
        self._engine = PolarsExecutor()
        self._fill()

    def _fill(self) -> None:
        """Build the frames from whatever is bound now.

        A failure leaves nothing behind, which is also what makes a rebind that
        raises leave a closed handle rather than a stale one: the half-built
        model is released and the exception is the caller's.
        """
        try:
            self._engine.build(self._program, tidy_sources(self._schema, dict(self._sources), self._coords))
        except BaseException:
            self._engine.close()
            raise

    def rebind(
        self,
        sources: Mapping[str, Any],
        *,
        coords: Mapping[str, Any] | None = None,
    ) -> BoundModel:
        """Same model, new numbers. Returns this object, so a driver can chain.

        *sources* names only what changed — the rest keeps what :func:`build`
        bound — and it may name a dimension index as well as a parameter, which
        is how a coordinate set grows::

            bound.rebind({'cap_hat': capacity}).solve()

        **Total.** There is no shape of new data it refuses and no capability to
        query first: what a value change can cost is the *fast path*, never the
        answer. Reading back is keyed by coordinate, so a rebind that moves a
        mask — a parameter a ``where`` compares against — renumbers labels
        under a caller who cannot tell, and the engine rebuilds and solves cold
        instead of pushing values onto a loaded solver. Which one ran is
        :attr:`~lpspec.relational.result.Diagnostics.loads`.

        The answer is the reference build's, always: ``bound.rebind(x)`` solves
        what ``build(model, sources | x)`` solves, and that equality is what
        ``tests/test_rebind.py`` asserts rather than assuming.

        **Results from before it stop reading.** A rebind replaces the label
        frames every reader joins through, so read out what you need first —
        which is what a Benders loop does anyway, taking its duals before it
        moves the cut table.
        """
        self._sources.update(_known(sources, {**self._schema.parameters, **self._schema.dimensions}, 'sources'))
        self._coords.update(_known(coords or {}, self._schema.dimensions, 'coords'))
        self._fill()
        return self

    def solve(
        self,
        solver_name: str = 'highs',
        *,
        solver_options: Mapping[str, Any] | None = None,
        batch_rows: int | None = None,
    ) -> Result:
        """Sink the built model straight into a solver and solve it.

        ``solver_name`` picks the sink and ``solver_options`` is forwarded
        verbatim in that solver's vocabulary; both are the *caller's* choice at
        the call, no YAML file being able to express either. ``batch_rows`` is
        the hand-off budget in elements.

        A solver that can stay loaded is kept between calls, so a rebound model
        re-solves from the basis the last one ended on.
        """
        return self._engine.solve(solver_name, solver_options=solver_options, batch_rows=batch_rows)

    def write(self, path: str | Path) -> None:
        """Sink the built model to a file; the **suffix** picks the writer."""
        self._engine.write(path)

    def diagnostics(self) -> Diagnostics:
        """What this build and its solves did that the answer does not show.

        Answerable after :meth:`close`: every field is a count or a small frame
        the engine keeps, not a read of the model it releases.
        """
        return self._engine.diagnostics()

    def close(self) -> None:
        """Release the built model, and any solver still holding it."""
        self._engine.close()

    def __enter__(self) -> BoundModel:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False


def _known(given: Mapping[str, Any], declared: Mapping[str, Any], where: str) -> Mapping[str, Any]:
    """*given*, or an error naming what *declared* does not hold.

    A rebind that names nothing re-solves the same numbers and reports it
    as an answer, which is the one failure a driver cannot see. ``build``
    does not ask this — it binds every declared name or fails — where a
    rebind is *partial* by construction and so has to.
    """
    unknown = sorted(set(given) - set(declared))
    if unknown:
        raise DataError(
            f'rebind: {where} names {unknown}, which this model does not declare — '
            f'it has {sorted(declared)}. A rebind names what changed, so a name nothing '
            f'reads would silently re-solve the numbers already bound.'
        )
    return given


def build(
    model: str | Path | dict[str, Any] | Model,
    sources: Mapping[str, Any],
    *,
    coords: dict[str, Any] | None = None,
) -> BoundModel:
    """Bind *sources* to *model* and build it — the model with your data on it.

    ``sources`` maps parameter names to parquet paths or in-memory tables (and
    optionally dimension names to index tables). One build can feed more than
    one sink: call ``bound.solve()`` and ``bound.write(path)`` on the same
    object, and ``bound.rebind(...)`` to put new numbers on it.

    Raises
    ------
    LanguageError
        If the model uses a construct outside the streaming language —
        the message names the construct and its context.
    """
    return BoundModel(load_model(model), sources, coords)


def solve(
    model: str | Path | dict[str, Any] | Model,
    sources: Mapping[str, Any],
    solver_name: str = 'highs',
    *,
    solver_options: Mapping[str, Any] | None = None,
    **build_kwargs: Any,
) -> Result:
    """Build and solve in one call.

    ``solver_name`` picks the sink — ``highs``, which ships with the package,
    or ``gurobi``, needing the ``[gurobi]`` extra. linopy's spelling, and the
    *caller's* decision: the same file solves the same model either way, so
    nothing in the YAML names a solver. ``solver_options`` is forwarded
    verbatim in that solver's vocabulary; build options stay separate, never
    reaching the solver.

    The built model stays attached to the returned :class:`Result`, whose label
    frames back ``result.primal(...)``; nothing has to be released, though
    ``result.close()`` drops a large model early. A caller who will solve the
    same model again with new numbers wants :func:`build` and ``rebind``, this
    being the one-shot spelling.

    The sink is resolved before the build, as ``write`` checks the suffix
    first: a caller who named a sink nothing can serve should not pay for a
    model.
    """
    solver(solver_name)
    ex = build(model, sources, **build_kwargs)
    try:
        return ex.solve(solver_name, solver_options=solver_options)
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
    """
    out = Path(out)
    writer(out.suffix.lower())
    with build(model, sources, **build_kwargs) as ex:
        ex.write(out)
    return out
