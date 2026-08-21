"""The linopy lane: a YAML model built as a ``linopy.Model``.

Requires the ``[linopy]`` extra (linopy, xarray).

One language, two lanes: the same file either binds relationally and solves
through :mod:`lpspec.api`, or is constructed here as a ``linopy.Model`` the
caller then owns. Which one to take is the caller's choice and nothing else
differs — both accept *exactly* the same language, which is what makes the
differential tests an oracle rather than a comparison of dialects.

Two functions — a producer and a reader — and both are **pure**: YAML goes in,
a model or a value comes out, and nothing is retained. No accessor on the
model, no session, no state. A file's meaning never depends on what was loaded
before it (docs/about/architecture.md, hard rule 5), so every file declares the
parameters it uses and the caller supplies their data per call — the reader
included, which is why :func:`expression` takes ``sources`` again rather than
remembering what :func:`build` saw::

    from lpspec import linopy as lpspec_linopy

    m = lpspec_linopy.build('model.yaml', {...})
    m.solve(...)
    lpspec_linopy.expression(m, 'model.yaml', 'co2', {...})

The same model on the other lane, which streams::

    import lpspec as lps

    with lps.solve('model.yaml', {...}) as result:
        result.primal('p')

This lane **constructs**; it does not attach. Math for a ``linopy.Model``
something else built has no verb here (#845) — a file is valid alone, and one
that referenced variables it does not declare was the single exception.

**Importing this module sets** ``linopy.options['semantics'] = 'v1'``. This
lane speaks v1 and the option is global, so importing is what sets it.
linopy's ``legacy`` default fills every absent slot with 0, where the
relational lane drops the row (the absence rules) — left alone the two lanes answer
the same YAML 25.0 against 125.0 on a masked-variable model, a wrong answer
rather than a wrong error.

Writing global state on import is a real cost, a process importing this module
having its own linopy arithmetic changed too. Scoping it per call is what
linopy's context manager cannot do: ``__exit__`` calls ``reset()``, restoring
*all* options to their defaults rather than their prior values, so it would
silently discard a caller's ``display_max_rows``. Between a documented global
and a hand-rolled save/restore around every entry point, the global is the one
a reader can find. Unguarded, the declared linopy floor being a version that
has the option.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import linopy
    import xarray
except ModuleNotFoundError as exc:
    msg = 'The linopy compatibility layer requires the [linopy] extra: pip install "lpspec[linopy]"'
    raise ModuleNotFoundError(msg) from exc


from lpspec._notes import note
from lpspec.errors import unknown_name_message
from lpspec.language import ComparisonNode, Namespace, expand_piecewise, expression_of, load_model
from lpspec.linopy.builder import EvaluationContext, _eval_ast, build_model
from lpspec.linopy.loader import dimension_coords, load_parameters
from lpspec.lowering import lower_expression, lower_program
from lpspec.sources import tidy_sources, validate_curve_extent, validate_piecewise_data

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lpspec.language import Model

linopy.options['semantics'] = 'v1'

__all__ = ['build', 'expression']


def build(model: str | Path | dict[str, Any] | Model, sources: Mapping[str, Any]) -> linopy.Model:
    """Bind *sources* to *model* and build it as a ``linopy.Model``.

    :func:`lpspec.build`'s signature, and deliberately: which lane builds a
    file is the caller's choice, so the call cannot differ.

    Args:
        model: A YAML path, a mapping, or a loaded :class:`~lpspec.language.model.Model`.
        sources: Parameter names to parquet paths or in-memory tables, and
            dimension names to their labels — an index table, a parquet path,
            or a bare sequence — wherever the YAML declares none.

    Returns:
        A model carrying every declaration the file makes.

    Raises:
        LanguageError: A construct the language does not accept — the same
            verdict :func:`lpspec.check` gives, reached through the same
            lowering pass, so neither lane accepts a file the other refuses.
        DataError: A source that is missing, unreadable, or the wrong shape.
    """
    with note(f'while loading {_named(model)}'):
        original = load_model(model)
        schema = expand_piecewise(original)
        lower_program(original)

        tidy = tidy_sources(original, sources)
        validate_curve_extent(original, tidy)
        master_coords, dim_coords = dimension_coords(schema, tidy)
        dataset = load_parameters(schema, tidy, master_coords)
        validate_piecewise_data(original, dataset)

        built = linopy.Model()
        build_model(built, schema, dataset, master_coords, dim_coords)

    return built


def expression(
    built: linopy.Model,
    model: str | Path | dict[str, Any] | Model,
    name: str,
    sources: Mapping[str, Any],
) -> xarray.DataArray:
    """Evaluate named expression *name* of *model* at *built*'s solution.

    The eager lane's half of readable expressions — the streaming lane spells
    it ``result.expression(name)``. Pure like :func:`build`: nothing was
    retained there, so the same *sources* the model was built with are passed
    again, the declared expression is evaluated on the model, and
    linopy's native ``.solution`` is the answer.

    Args:
        built: A solved model carrying this file's variables.
        model: The file declaring the expression, as :func:`build` takes it.
        name: A name declared under ``expressions:`` — never an expression
            string.
        sources: As :func:`build` takes them.

    Returns:
        The expression's value over its own dims, as an ``xarray.DataArray``
        (0-dimensional for a variable-free scalar expression).

    Raises:
        KeyError: No named expression called *name*.
        LanguageError: A construct the language does not accept, in the file or
            in the expression — the latter lowered here rather than at
            :func:`build`, exactly as ``result.expression`` lowers at the read.
        DataError: A source that does not fit the file.
    """
    with note(f"while reading named expression '{name}' from {_named(model)}"):
        original = load_model(model)
        schema = expand_piecewise(original)
        if name not in schema.expressions:
            raise KeyError(
                unknown_name_message('named expression', name, schema.expressions)
                + ' expression() takes a name declared under expressions:, never an expression string.'
            )
        lower_program(original)
        lower_expression(schema, name)
        tidy = tidy_sources(original, sources)
        master_coords, dim_coords = dimension_coords(schema, tidy)
        dataset = load_parameters(schema, tidy, master_coords)
        ns = Namespace.of(schema)
        ast = expression_of(schema.expressions[name].expression, schema, ns, f"named expression '{name}'")
        assert not isinstance(ast, ComparisonNode), 'load-time validation refuses a comparison in a named expression'
        value = _eval_ast(ast, EvaluationContext(built, dataset, master_coords, schema, ns, dim_coords))
        if hasattr(value, 'solution'):
            return value.solution
        if isinstance(value, xarray.DataArray):
            return value
        return xarray.DataArray(float(value))


def _named(model: str | Path | dict[str, Any] | Model) -> str:
    """What to call *model* in an error note.

    A path names itself; a mapping or an already-loaded schema has no name, and
    saying so beats printing a dict into a traceback.
    """
    return f"YAML '{model}'" if isinstance(model, (str, Path)) else 'the model passed in'
