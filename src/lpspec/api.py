"""The runner: bind data to a YAML spec and execute it. Not a modeling API.

Math is defined in YAML only — there is no Python API for constructing specs,
and the logical plan is internal. Four verbs: ``check``, ``build`` (YAML +
sources → a :class:`Model`), ``solve`` and ``write``.

This is the relational lane (docs/about/architecture.md): validated at load
time, lowered to the plan, executed relationally. The same file builds as a
``linopy.Model`` through ``lpspec.linopy``, on the same call and the same
sources — which lane a caller wants is theirs to pick, and this one needs no
optional extra.

Example::

    import lpspec as lps

    result = lps.solve(
        'spec.yaml',
        {'p_max': 'p_max.parquet', 'load': 'load.parquet', 'snapshot': range(8760)},
    )
    result.objective
    result.primal('p')  # tidy polars.DataFrame (coords..., value)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from math_spec import advice, to_program

from lpspec.errors import DataError, LpspecError, LpspecWarning, lane_cannot_build_message
from lpspec.relational import sinks
from lpspec.relational.engines.polars.engine import PolarsEngine
from lpspec.relational.sinks import solver, writer
from lpspec.relational.sinks.capabilities import Capabilities, required
from lpspec.sources import bindable, tidy_sources

if TYPE_CHECKING:
    from collections.abc import Mapping

    from math_spec import Spec
    from math_spec.program import Program

    from lpspec.relational.result import ConstraintRow, Diagnostics, Keep, Result

#: Anything the verbs here take as the spec: a YAML path, a mapping, or a
#: spec the language has already read — a ``Spec`` from
#: :func:`math_spec.to_spec`, or a ``Program`` from :func:`check`. Each is
#: handed straight to :func:`math_spec.to_program`, which is where a file
#: becomes the plan this package builds rows from.
Buildable: TypeAlias = 'str | Path | dict[str, Any] | Spec | Program'

__all__ = ['Buildable', 'build', 'check', 'solve', 'write']


#: What each **lane** can build, beside what each sink can ingest. Nothing is
#: *solved* on a lane, but the question is the same one, and hard rule 3's
#: amendment (docs/about/architecture.md) is this constant: both lanes accept the same language, and one
#: cannot build a quadratic constraint —
#: ``linopy.Model.add_constraints`` refuses a ``QuadraticExpression`` outright
#: and no reformulation of it is exact.
#:
#: Declared here rather than in ``lpspec.linopy`` because that needs an extra
#: this build may not have, and read as *data* so ``check`` answers without it.
#: What the gap costs is the differential oracle for that one construct, which
#: is why it should stay one entry long.
LANES: Mapping[str, Capabilities] = {
    'linopy': Capabilities(
        supports={
            'integrality': 'native',
            'sos': 'native',
            'quadratic_objective': 'native',
            'nonconvex_quadratic_objective': 'native',
        }
    )
}


def _portability(program: Program, sink: str) -> tuple[str | None, list[str]]:
    """Why *sink* cannot take *program*, and what it would rewrite if it can.

    The one place a lane is told apart from a sink: what a sink refuses or
    reformulates is ``relational.sinks``' business, and it may not know a lane
    exists (docs/about/architecture.md, hard rule 2). A lane rewrites nothing —
    everything it supports it builds natively — so its second answer is empty.
    """
    relaxed: list[str] = []
    if (lane := LANES.get(sink)) is not None:
        missing = lane.missing(required(program))
        return (lane_cannot_build_message(sink, missing) if missing else None), relaxed
    refused = sinks.refusal(program, sink)
    return refused, relaxed if refused else sinks.relaxations(program, sink)


def check(spec: Buildable, sink: str | None = None) -> Program:
    """Parse, expand, validate and lower a spec; bind no data.

    With *sink*, also: **will that sink take it?** The two are separate axes
    (docs/about/ceiling.md) — whether a spec is sayable is solver-independent,
    where it can land is not — so bare ``check`` stays silent about
    portability. Most specs never leave the common subset, and a default that
    warned about a sink nobody named would be noise on every one of them.

    **Named expressions are lowered here and nowhere else.** They are thunks
    until read, so an error inside one would otherwise wait for a reader —
    which hard rule 2 (docs/about/architecture.md) refuses. This is the verb that can afford to look.

    A capability question is answered off a declared table with no data bound,
    so it costs no build and needs no solver installed: a repository of specs
    can be checked in CI against every sink they will eventually be solved on.
    The solver-independent advice is issued either way — a sink that refuses is
    the answer to the second question, and naming one must not cost the first.

    Args:
        spec: A YAML path, a mapping, or anything :func:`math_spec.to_program`
            already takes — a ``Spec`` from ``math_spec.to_spec``, or a
            ``Program`` from an earlier call to this.
        sink: A solver name (``highs``, ``gurobi``, ``xpress``), an output
            suffix (``.lp``, ``.mps``), or a lane (``linopy``). ``None`` asks
            only whether the spec is sayable.

    Returns:
        The lowered program: what a build reads rows off, and what every verb
        here takes back without parsing the file again. It is the language's
        own type — typeset it, or read its declarations, through
        :mod:`math_spec`.

    Raises:
        LanguageError: A construct outside the streaming language.
        LpspecError: A *sink* that cannot take this spec, or a name belonging
            to no sink.
        ValueError: A schema or expression that does not parse.

    Warns:
        LpspecWarning: Advice short of an error — a declared dimension nothing
            uses as an axis, a variable the objective drives to infinity with
            nothing to stop it, a construct the named sink takes only
            reformulated. Issued here and nowhere else.
    """
    program = to_program(spec)
    notes = [str(note) for note in advice(program)]
    refused: str | None = None
    relaxed: list[str] = []
    if sink is not None:
        refused, relaxed = _portability(program, sink)
    for note in (*notes, *relaxed):
        warnings.warn(note, LpspecWarning, stacklevel=2)
    if refused is not None:
        raise LpspecError(refused)
    return program


class Model:
    """A spec with your data bound to it — what :func:`build` returns.

    Three nouns, each arrow adding one thing: a ``Program`` is the math,
    a ``Model`` is the math with your data, a ``Result`` is one answer.

        ``check`` → ``Program`` → ``build`` → ``Model`` → ``solve`` → ``Result``

    One build feeds any number of sinks — :meth:`solve` and :meth:`write` on
    the same object — :meth:`rebind` puts new numbers on it without re-reading
    the YAML or re-lowering the plan, and :meth:`diagnostics` says what it did.
    Nothing has to be released; :meth:`close` hands a large model back early.
    """

    def __init__(self, spec: Buildable, sources: Mapping[str, Any]) -> None:
        self._program = to_program(spec)
        self._sources = dict(sources)
        self._engine = PolarsEngine()
        self._fill()

    def _fill(self) -> None:
        """Build the frames from whatever is bound now.

        A failure leaves nothing behind, which is also what makes a rebind that
        raises leave a closed handle rather than a stale one: the half-built
        model is released and the exception is the caller's.
        """
        try:
            self._engine.build(self._program, tidy_sources(self._program, dict(self._sources)))
        except BaseException:
            self._engine.close()
            raise

    def rebind(self, sources: Mapping[str, Any]) -> Model:
        """Put new numbers on the same model, in place.

        ::

            model.rebind({'cap_hat': capacity}).solve()

        Any new data is accepted: ``model.rebind(x)`` answers what
        ``build(spec, sources | x)`` answers, whatever changed. What a change
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
                bound. A dimension's labels as well as a parameter, which is
                how a coordinate set grows.

        Returns:
            This object, so a driver can chain.

        Raises:
            DataError: A name the spec does not declare — a rebind that named
                nothing would silently re-solve the numbers already bound.
        """
        _refuse_unknown(sources, bindable(self._program))
        self._sources.update(sources)
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
            LpspecError: A construct the format has no section for. What each
                one carries is :func:`check`'s ``sink=`` answer, hours earlier.
        """
        self._engine.write(path)

    def row(self, name: str, /, **coordinate: Any) -> ConstraintRow:
        """One built constraint row at one coordinate — its terms, sense and right-hand side.

        The verb for *this row is wrong and I do not know why*. ``to_latex``
        and its siblings render the spec as math before any data, and
        :meth:`~lpspec.relational.result.Result.dual` gives a row's number
        without its terms; this gives the row the build actually produced, at
        the coordinate you name.

        Reads the **built** model and needs no solve, so it answers on a model
        that never reached a solver — and it is the built row, so a term whose
        variable was absent is missing from it and a row a ``where`` masked out
        is not there at all. That is the point: it shows what the model says
        rather than what the file appears to say.

        Args:
            name: A declared constraint. Positional, so that a dimension may
                be called ``name`` and still be named in *coordinate*.
            coordinate: One label per dim of that declaration, all of them —
                a partial coordinate names a set of rows rather than one.

        Returns:
            The terms as ``(variable, coordinate, coefficient)``, beside the
            comparison and the right-hand side.

        Raises:
            KeyError: No constraint is called *name*.
            LpspecError: The coordinate names the wrong dims, matches no row
                the build produced, or the model has been closed.

        Example:
            >>> print(model.row('balance', snapshot=1))  # doctest: +SKIP
            balance[snapshot=1]: +1 p[1, wind] +50 p[1, gas] >= 60
        """
        return self._engine.row(name, coordinate)

    def diagnostics(self) -> Diagnostics:
        """What this build and its solves did that the answer does not show.

        Answerable after :meth:`close`, and after a build that raised: every
        field is a count, a clock or a small frame the engine keeps, not a read
        of the model it releases. A raise leaves the sizes at zero — they are
        taken once a model is whole — and everything measured before it stands.
        """
        return self._engine.diagnostics()

    def close(self) -> None:
        """Release the built model, and any solver still holding it."""
        self._engine.close()

    def __enter__(self) -> Model:
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        self.close()
        return False


def _refuse_unknown(given: Mapping[str, Any], declared: Mapping[str, Any]) -> None:
    """Refuse a rebind naming anything *declared* does not hold.

    A rebind that names nothing re-solves the same numbers and reports it
    as an answer, which is the one failure a driver cannot see. ``build``
    does not ask this — it binds every declared name or fails — where a
    rebind is *partial* by construction and so has to.
    """
    unknown = sorted(set(given) - set(declared))
    if unknown:
        raise DataError(
            f'rebind: sources names {unknown}, which this spec does not declare — '
            f'it has {sorted(declared)}. A rebind names what changed, so a name nothing '
            f'reads would silently re-solve the numbers already bound.'
        )


def build(spec: Buildable, sources: Mapping[str, Any]) -> Model:
    """Bind *sources* to *spec* and build it — the model with your data on it.

    Args:
        spec: As :func:`check` takes it.
        sources: Parameter names to parquet paths or in-memory tables, and
            dimension names to their labels — an index table, a parquet path,
            or a bare sequence — wherever the YAML declares none.

    Returns:
        The built model. It feeds any number of sinks — ``model.solve()`` and
        ``model.write(path)`` on the same object — and ``model.rebind(...)``
        puts new numbers on it.

    Raises:
        LanguageError: A construct outside the streaming language.
        DataError: A source that is missing, unreadable, or the wrong shape.
    """
    return Model(spec, sources)


def solve(
    spec: Buildable,
    sources: Mapping[str, Any],
    solver_name: str = 'highs',
    *,
    solver_options: Mapping[str, Any] | None = None,
) -> Result:
    """Build *spec* and solve it in one call.

    The one-shot spelling: a caller who will solve the same spec again with
    new numbers wants :func:`build` and :meth:`Model.rebind`.

    There is no ``keep`` here and no room for one — this builds the model it
    solves, so the solve is the first of that model's life and
    :attr:`~lpspec.relational.result.Result.kept` is always ``nothing``.
    Choosing what to keep is :meth:`Model.solve`, where a previous solve
    exists to keep something of.

    Args:
        spec: As :func:`check` takes it.
        sources: As :func:`build` takes them.
        solver_name: ``highs``, which ships with the package, or ``gurobi``,
            which needs the ``[gurobi]`` extra.
        solver_options: Forwarded to the solver verbatim, in its own
            vocabulary (``{'time_limit': 60}``).

    Returns:
        The solution, self-contained: it owns the frames it reads, so the built
        model and the solver are released before this returns and there is
        nothing to manage. ``result.close()`` drops its own hold early.

    Raises:
        LpspecError: A solver name nothing serves — checked before the build.
    """
    solver(solver_name)
    model = build(spec, sources)
    try:
        return model.solve(solver_name, solver_options=solver_options)
    finally:
        model.close()


def write(
    spec: Buildable,
    sources: Mapping[str, Any],
    out: str | Path,
) -> Path:
    """Build *spec* and stream it to a file, in the format *out*'s suffix names.

    Args:
        spec: As :func:`check` takes it.
        sources: As :func:`build` takes them.
        out: Where to write; ``.lp`` and ``.mps`` are what ship.

    Returns:
        The path written.

    Raises:
        ValueError: A suffix nothing writes — checked before the build.
        LpspecError: A construct the format has no section for, which is
            ``check(spec, sink=out.suffix)``'s answer with no data bound.
    """
    out = Path(out)
    writer(out.suffix.lower())
    with build(spec, sources) as model:
        model.write(out)
    return out
