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
from dataclasses import dataclass
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

    import polars as pl

    from lpspec.language.model import Model
    from lpspec.relational.result import Result

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


@dataclass(frozen=True)
class Diagnostics:
    """What a build and its solves did that the answer does not show.

    One accessor rather than a reader per fact, so that watching a build stays
    one question — and because these answer *what is this model*, where the
    handle's own methods answer *what do I do with it*
    (docs/ARCHITECTURE.md, "the Python surface").

    **Advisory, both of them.** Nothing about an answer depends on either, and
    a caller who branches on one has made this engine's bookkeeping part of
    their model. They are here to be read when a loop is slower or smaller than
    it should be.
    """

    #: The shape the solver was handed: columns, rows, and matrix entries.
    #: What ``check`` cannot answer, needing no data where this needs all of
    #: it, and the thing to report when a model is bigger than its author
    #: expected — a broadcast that multiplied rows shows up here first.
    columns: int
    rows: int
    nonzeros: int

    #: ``(constraint, rows_not_built)`` — every row that lost all its terms and
    #: so was not built (SPEC §6). Empty for a model whose every declared row
    #: reached the solver. Counts rather than coordinates: the label of an
    #: unbuilt row does not exist.
    omissions: pl.DataFrame

    #: How many times this model has been solved. The denominator ``reloads``
    #: is read against: one load in one solve is a cold start, one load in
    #: twenty-five is a driver on the fast path.
    solves: int

    #: ``(solve, reason)`` — the solves that loaded the model from scratch
    #: instead of pushing values onto a solver that already held it. One row on
    #: a driver taking the fast path, being the first solve, which had nothing
    #: to keep; a row per iteration on one that is not, which is the difference
    #: between "lpspec is slow" and "this model masks on a parameter that
    #: varies".
    reloads: pl.DataFrame


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
        program: Any,
        sources: Mapping[str, Any],
        coords: Mapping[str, Any] | None = None,
    ) -> None:
        self._schema = schema
        self._program = program
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
        :attr:`Diagnostics.reloads`.

        The answer is the reference build's, always: ``bound.rebind(x)`` solves
        what ``build(model, sources | x)`` solves, and that equality is what
        ``tests/test_rebind.py`` asserts rather than assuming.

        **Results from before it stop reading.** A rebind replaces the label
        frames every reader joins through, so read out what you need first —
        which is what a Benders loop does anyway, taking its duals before it
        moves the cut table.
        """
        self._sources.update(self._known(sources, 'sources'))
        self._coords.update(self._known(coords or {}, 'coords'))
        self._fill()
        return self

    def _known(self, given: Mapping[str, Any], where: str) -> Mapping[str, Any]:
        """*given*, or an error naming what the model does not declare.

        A rebind that names nothing re-solves the same numbers and reports it
        as an answer, which is the one failure a driver cannot see. ``build``
        does not ask this — it binds every declared name or fails — where a
        rebind is *partial* by construction and so has to.
        """
        declared = (
            self._schema.dimensions if where == 'coords' else {**self._schema.parameters, **self._schema.dimensions}
        )
        unknown = sorted(set(given) - set(declared))
        if unknown:
            raise DataError(
                f'rebind: {where} names {unknown}, which this model does not declare — '
                f'it has {sorted(declared)}. A rebind names what changed, so a name nothing '
                f'reads would silently re-solve the numbers already bound.'
            )
        return given

    def solve(
        self,
        batch_rows: int | None = None,
        solver_options: Mapping[str, Any] | None = None,
        solver_name: str = 'highs',
    ) -> Result:
        """Sink the built model straight into a solver and solve it.

        ``solver_name`` picks the sink and ``solver_options`` is forwarded
        verbatim in that solver's vocabulary; both are the *caller's* choice at
        the call, no YAML file being able to express either. ``batch_rows`` is
        the hand-off budget in elements.

        A solver that can stay loaded is kept between calls, so a rebound model
        re-solves from the basis the last one ended on.
        """
        return self._engine.solve(batch_rows, solver_options, solver_name)

    def write(self, path: str | Path) -> None:
        """Sink the built model to a file; the **suffix** picks the writer."""
        self._engine.write(path)

    def diagnostics(self) -> Diagnostics:
        """What this build and its solves did that the answer does not show.

        Answerable after :meth:`close`: every field is a count or a small frame
        this keeps, not a read of the model it releases.
        """
        return Diagnostics(
            columns=self._engine._n_cols,
            rows=self._engine._n_rows,
            nonzeros=self._engine._n_entries,
            omissions=self._engine.omissions(),
            solves=self._engine.solves,
            reloads=self._engine.reloads(),
        )

    def close(self) -> None:
        """Release the built model, and any solver still holding it."""
        self._engine.close()

    def __enter__(self) -> BoundModel:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False


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
    schema = load_model(model)
    return BoundModel(schema, lower_program(schema), sources, coords)


def solve(
    model: str | Path | dict[str, Any] | Model,
    sources: Mapping[str, Any],
    solver_options: Mapping[str, Any] | None = None,
    solver_name: str = 'highs',
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
    """
    out = Path(out)
    writer(out.suffix.lower())
    with build(model, sources, **build_kwargs) as ex:
        ex.write(out)
    return out
