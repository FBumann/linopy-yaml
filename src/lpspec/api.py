"""The runner: bind data to a YAML model and execute it. Not a modeling API.

Math is defined in YAML only — there is no Python API for constructing models,
and the logical plan is internal. Four verbs: ``check``, ``build`` (YAML +
sources → a :class:`BoundModel`), ``solve`` and ``write``.

This is the relational lane (docs/about/architecture.md): validated at load
time, lowered to the plan, executed relationally. The same file builds as a
``linopy.Model`` through ``lpspec.linopy``, on the same call and the same
sources — which lane a caller wants is theirs to pick, and this one needs no
optional extra.

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
from lpspec.lowering import advice, expression_thunks, lower_program
from lpspec.relational.engines.polars.engine import PolarsEngine
from lpspec.relational.sinks import solver, writer
from lpspec.sources import tidy_sources

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lpspec.language.model import Model
    from lpspec.relational.result import Diagnostics, Keep, Result

#: Re-exported: parsing and validating a model is the *language's* job, and a
#: consumer that binds no data (``typeset``) must be able to reach it without
#: reaching the runner. Callers keep saying ``lps.load_model``.
__all__ = ['build', 'check', 'load_model', 'solve', 'write']


def check(model: str | Path | dict[str, Any] | Model) -> Model:
    """Parse, expand, validate and lower a model; bind no data.

    Args:
        model: A YAML path, a mapping, or a loaded :class:`Model`.

    Returns:
        The validated schema.

    Raises:
        LanguageError: A construct outside the streaming language.
        ValueError: A schema or expression that does not parse.

    Warns:
        LpspecWarning: Advice short of an error — a declared dimension nothing
            uses as an axis, say. Issued here and nowhere else.
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

    One build feeds any number of sinks — :meth:`solve` and :meth:`write` on
    the same object — :meth:`rebind` puts new numbers on it without re-reading
    the YAML or re-lowering the plan, and :meth:`diagnostics` says what it did.
    Nothing has to be released; :meth:`close` hands a large model back early.
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
        self._engine = PolarsEngine()
        self._fill()

    def _fill(self) -> None:
        """Build the frames from whatever is bound now.

        A failure leaves nothing behind, which is also what makes a rebind that
        raises leave a closed handle rather than a stale one: the half-built
        model is released and the exception is the caller's.
        """
        try:
            self._engine.build(
                self._program,
                tidy_sources(self._schema, dict(self._sources), self._coords),
                expressions=expression_thunks(self._schema),
            )
        except BaseException:
            self._engine.close()
            raise

    def rebind(
        self,
        sources: Mapping[str, Any],
        *,
        coords: Mapping[str, Any] | None = None,
    ) -> BoundModel:
        """Put new numbers on the same model, in place.

        ::

            bound.rebind({'cap_hat': capacity}).solve()

        Any new data is accepted: ``bound.rebind(x)`` answers what
        ``build(model, sources | x)`` answers, whatever changed. What a change
        costs is the fast path, never the answer — data that moves a mask
        renumbers labels, so the model is rebuilt and solved cold instead of
        pushed onto a loaded solver, and
        :attr:`~lpspec.relational.result.Diagnostics.loads` says which ran.

        Results taken before the rebind keep reading: each owns the frames it
        reads, and a rebind builds new ones rather than touching those. What
        retaining one costs is its build's label frames staying alive until it
        is dropped or :meth:`~lpspec.relational.result.Result.close` is called.

        Args:
            sources: Only what changed; the rest keeps what :func:`build`
                bound. A dimension index as well as a parameter, which is how
                a coordinate set grows.
            coords: Dimension labels, likewise only where they changed.

        Returns:
            This object, so a driver can chain.

        Raises:
            DataError: A name the model does not declare — a rebind that named
                nothing would silently re-solve the numbers already bound.
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
        keep: Keep = 'solver',
    ) -> Result:
        """Hand the built model to a solver and solve it.

        A solver that can stay loaded is kept between calls, so a rebound
        model skips the hand-off and only its numbers are pushed. Whether the
        *work* that solver did is kept too is *keep*, and it is off by
        default: a solver given a run to resume may forgo preparation it would
        otherwise do, which on the sinks that ship has been worth a large
        multiple in both directions (#815), and only a caller knows which way
        their model goes. How much this solve actually kept is its
        :attr:`~lpspec.relational.result.Result.kept`.

        Args:
            solver_name: ``highs``, which ships with the package, or
                ``gurobi``, which needs the ``[gurobi]`` extra.
            solver_options: Forwarded to the solver verbatim, in its own
                vocabulary (``{'time_limit': 60}``).
            keep: How much of the session this solve may keep — one of
                :data:`~lpspec.relational.result.KEEPS`. ``solver``, the
                default, reuses the solver holding the model and discards the
                work it did; ``progress`` keeps that work too, which is what
                an iterating driver moving one step at a time wants;
                ``nothing`` keeps neither, which is what timing a build or
                comparing against a cold baseline needs and what no solver
                option can promise. A preference: a model whose structure
                moved is loaded again whatever was asked.

        Returns:
            The solution, holding this model.

        Raises:
            LpspecError: A solver name nothing serves, one this environment
                cannot run, or a *keep* outside
                :data:`~lpspec.relational.result.KEEPS`.
        """
        return self._engine.solve(solver_name, solver_options=solver_options, keep=keep)

    def write(self, path: str | Path) -> None:
        """Stream the built model to *path*, in the format its suffix names.

        Raises:
            ValueError: A suffix nothing writes.
        """
        self._engine.write(path)

    def diagnostics(self) -> Diagnostics:
        """What this build and its solves did that the answer does not show.

        Answerable after :meth:`close`: every field is a count, a clock or a
        small frame the engine keeps, not a read of the model it releases.
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

    Args:
        model: A YAML path, a mapping, or a loaded :class:`Model`.
        sources: Parameter names to parquet paths or in-memory tables, and
            optionally dimension names to index tables.
        coords: Dimension labels neither *sources* nor the YAML carries.

    Returns:
        The bound model. It feeds any number of sinks — ``bound.solve()`` and
        ``bound.write(path)`` on the same object — and ``bound.rebind(...)``
        puts new numbers on it.

    Raises:
        LanguageError: A construct outside the streaming language.
        DataError: A source that is missing, unreadable, or the wrong shape.
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
    """Build *model* and solve it in one call.

    The one-shot spelling: a caller who will solve the same model again with
    new numbers wants :func:`build` and :meth:`BoundModel.rebind`.

    There is no ``keep`` here and no room for one — this builds the model it
    solves, so the solve is the first of that model's life and
    :attr:`~lpspec.relational.result.Result.kept` is always ``nothing``.
    Choosing what to keep is :meth:`BoundModel.solve`, where a previous solve
    exists to keep something of.

    Args:
        model: A YAML path, a mapping, or a loaded :class:`Model`.
        sources: As :func:`build` takes them.
        solver_name: ``highs``, which ships with the package, or ``gurobi``,
            which needs the ``[gurobi]`` extra.
        solver_options: Forwarded to the solver verbatim, in its own
            vocabulary (``{'time_limit': 60}``).
        **build_kwargs: Passed to :func:`build`.

    Returns:
        The solution, self-contained: it owns the frames it reads, so the built
        model and the solver are released before this returns and there is
        nothing to manage. ``result.close()`` drops its own hold early.

    Raises:
        LpspecError: A solver name nothing serves — checked before the build.
    """
    solver(solver_name)
    bound = build(model, sources, **build_kwargs)
    try:
        return bound.solve(solver_name, solver_options=solver_options)
    finally:
        bound.close()


def write(
    model: str | Path | dict[str, Any] | Model,
    sources: Mapping[str, Any],
    out: str | Path,
    **build_kwargs: Any,
) -> Path:
    """Build *model* and stream it to a file, in the format *out*'s suffix names.

    Args:
        model: A YAML path, a mapping, or a loaded :class:`Model`.
        sources: As :func:`build` takes them.
        out: Where to write; ``.lp`` is what ships.
        **build_kwargs: Passed to :func:`build`.

    Returns:
        The path written.

    Raises:
        ValueError: A suffix nothing writes — checked before the build.
    """
    out = Path(out)
    writer(out.suffix.lower())
    with build(model, sources, **build_kwargs) as bound:
        bound.write(out)
    return out
