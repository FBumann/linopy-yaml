"""The linopy lane: a YAML spec built as a ``linopy.Model``.

Requires the ``[linopy]`` extra (linopy, xarray).

One language, two lanes: the same file either attaches relationally and solves
through :mod:`lpspec.api`, or is constructed here as a ``linopy.Model`` the
caller then owns. Both accept *exactly* the same language, which is what makes
the differential tests an oracle rather than a comparison of dialects.

Two functions — a producer and a reader — and both are **pure**: YAML goes in,
a model or a value comes out, and nothing is retained, which is why
:func:`expression` takes ``sources`` again rather than remembering what
:func:`build` saw::

    from lpspec import linopy as lpspec_linopy

    m = lpspec_linopy.build('spec.yaml', {...})
    m.solve(...)
    lpspec_linopy.expression(m, 'spec.yaml', 'co2', {...})

The same spec on the other lane, which streams::

    import lpspec as lps

    with lps.solve('spec.yaml', {...}) as result:
        result.primal('p')

**Importing this module sets** ``linopy.options['semantics'] = 'v1'``. This
lane speaks v1 and the option is global: linopy's ``legacy`` default fills every
absent slot with 0, where the relational lane drops the row, so left alone the
two lanes answer the same YAML with different numbers. linopy's own context
manager cannot scope it — its ``__exit__`` resets *every* option to its default
rather than its prior value.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import linopy
    import xarray
except ModuleNotFoundError as exc:
    msg = 'The linopy lane requires the [linopy] extra: pip install "lpspec[linopy]"'
    raise ModuleNotFoundError(msg) from exc


from math_spec import to_program

from lpspec.errors import unknown_name_message
from lpspec.linopy._notes import note
from lpspec.linopy.builder import _eval, build_model
from lpspec.linopy.loader import dimension_coords, load_parameters
from lpspec.linopy.where import EvaluationContext
from lpspec.sources import tidy_sources

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lpspec.lanes import Buildable

linopy.options['semantics'] = 'v1'

__all__ = ['build', 'expression']


def build(spec: Buildable, sources: Mapping[str, Any]) -> linopy.Model:
    """Bind *sources* to *spec* and build it as a ``linopy.Model``.

    :func:`lpspec.build`'s signature: which lane builds a file is the caller's
    choice, so the call cannot differ.

    Args:
        spec: As :func:`lpspec.check` takes it.
        sources: Parameter names to parquet paths or in-memory tables, and
            dimension names to their labels — an index table, a parquet path,
            or a bare sequence.

    Returns:
        A model carrying every declaration the file makes.

    Raises:
        LanguageError: A construct the language does not accept — the same
            verdict :func:`lpspec.check` gives, reached through the same
            lowering pass, so neither lane accepts a file the other refuses.
        DataError: A source that is missing, unreadable, or the wrong shape.
    """
    with note(f'while loading {_named(spec)}'):
        program = to_program(spec)

        tidy = tidy_sources(program, sources)
        master_coords, dim_coords = dimension_coords(program, tidy)
        dataset = load_parameters(program, tidy, master_coords)

        built = linopy.Model()
        build_model(built, program, dataset, master_coords, dim_coords)

    return built


def expression(
    built: linopy.Model,
    spec: Buildable,
    name: str,
    sources: Mapping[str, Any],
) -> xarray.DataArray:
    """Evaluate named expression *name* of *spec* at *built*'s solution.

    Args:
        built: A solved model carrying this file's variables.
        spec: The file declaring the expression, as :func:`build` takes it.
        name: A name declared under ``expressions:`` — never an expression
            string.
        sources: As :func:`build` takes them.

    Returns:
        The expression's value over its own dims, as an ``xarray.DataArray``
        (0-dimensional for a variable-free scalar expression).

    Raises:
        KeyError: No named expression called *name*.
        LanguageError: A construct the language does not accept, in the file or
            in the expression.
        DataError: A source that does not fit the file.
    """
    with note(f"while reading named expression '{name}' from {_named(spec)}"):
        program = to_program(spec)
        if name not in program.named_expressions:
            raise KeyError(
                unknown_name_message('named expression', name, program.named_expressions)
                + ' expression() takes a name declared under expressions:, never an expression string.'
            )
        expression = program.named_expressions[name]
        tidy = tidy_sources(program, sources)
        master_coords, dim_coords = dimension_coords(program, tidy)
        dataset = load_parameters(program, tidy, master_coords)
        value = _eval(expression, EvaluationContext(dataset, master_coords, built, dim_coords, program))
        if hasattr(value, 'solution'):
            return value.solution
        if isinstance(value, xarray.DataArray):
            return value
        return xarray.DataArray(float(value))


def _named(spec: Buildable) -> str:
    """What to call *spec* in an error note.

    A path names itself; a mapping or an already-loaded schema has no name, and
    saying so beats printing a dict into a traceback.
    """
    return f"YAML '{spec}'" if isinstance(spec, (str, Path)) else 'the spec passed in'
