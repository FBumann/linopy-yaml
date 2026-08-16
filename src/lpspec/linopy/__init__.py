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
included, which is why :func:`expression` takes ``data=`` again rather than
remembering what :func:`build` saw::

    from lpspec import linopy as lpspec_linopy

    m = lpspec_linopy.build('model.yaml', data={...})
    m.solve(...)
    lpspec_linopy.expression(m, 'model.yaml', 'co2', data={...})

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
from typing import Any

try:
    import linopy
    import xarray
except ModuleNotFoundError as exc:
    msg = 'The linopy compatibility layer requires the [linopy] extra: pip install "lpspec[linopy]"'
    raise ModuleNotFoundError(msg) from exc


from lpspec._notes import note
from lpspec.errors import unknown_name_message
from lpspec.language.expression_parser import ComparisonNode
from lpspec.language.piecewise import expand_piecewise
from lpspec.language.resolution import Namespace, expression_of
from lpspec.language.validation import load_model
from lpspec.linopy.builder import EvaluationContext, _eval_ast, build_model
from lpspec.linopy.loader import build_dim_coords, build_master_coords, load_parameters
from lpspec.sources import validate_piecewise_data

linopy.options['semantics'] = 'v1'

__all__ = ['build', 'expression']


def build(
    path: str | Path,
    *,
    data: dict[str, Any] | None = None,
    coords: dict[str, Any] | None = None,
) -> linopy.Model:
    """Build a ``linopy.Model`` from a YAML math definition.

    Args:
        path: Path to the YAML file.
        data: Parameter data, keyed by the names the YAML declares.
        coords: Dimension coordinate values. Overrides ``values:`` declared
            in the YAML.

    Returns:
        A model carrying every declaration the file makes.

    Raises:
        LanguageError: A file the language does not accept — its structure,
            its declarations or its expressions.
        DataError: A file that is fine, and data that does not fit it.
    """
    path = Path(path)
    with note(f"while loading YAML '{path}'"):
        original = load_model(path)
        schema = expand_piecewise(original)

        master_coords = build_master_coords(schema, coords)
        dim_coords = build_dim_coords(schema, coords, master_coords)
        dataset = load_parameters(schema, data, master_coords)
        validate_piecewise_data(original, dataset)

        model = linopy.Model()
        build_model(model, schema, dataset, master_coords, dim_coords)

    return model


def expression(
    model: linopy.Model,
    path: str | Path,
    name: str,
    *,
    data: dict[str, Any] | None = None,
    coords: dict[str, Any] | None = None,
) -> xarray.DataArray:
    """Evaluate named expression *name* of *path* at *model*'s solution.

    The eager lane's half of readable expressions — the streaming lane spells
    it ``result.expression(name)``. Pure like :func:`build`: nothing was
    retained there, so the same *data* and *coords* the model was
    built with are passed again, the declared expression is evaluated on the
    model, and linopy's native ``.solution`` is the answer.

    Args:
        model: A solved model carrying this file's variables.
        path: Path to the YAML file declaring the expression.
        name: A name declared under ``expressions:`` — never an expression
            string.
        data: Parameter data, as :func:`build` takes it.
        coords: Dimension coordinate values, as :func:`build` takes them.

    Returns:
        The expression's value over its own dims, as an ``xarray.DataArray``
        (0-dimensional for a variable-free scalar expression).

    Raises:
        KeyError: No named expression called *name*.
        LanguageError: A file the language does not accept.
        DataError: Data that does not fit the file.
    """
    path = Path(path)
    with note(f"while reading named expression '{name}' from YAML '{path}'"):
        schema = expand_piecewise(load_model(path))
        if name not in schema.expressions:
            raise KeyError(
                unknown_name_message('named expression', name, schema.expressions)
                + ' expression() takes a name declared under expressions:, never an expression string.'
            )
        master_coords = build_master_coords(schema, coords)
        dim_coords = build_dim_coords(schema, coords, master_coords)
        dataset = load_parameters(schema, data, master_coords)
        ns = Namespace.of(schema)
        ast = expression_of(schema.expressions[name].expression, schema, ns, f"named expression '{name}'")
        assert not isinstance(ast, ComparisonNode), 'load-time validation refuses a comparison in a named expression'
        value = _eval_ast(ast, EvaluationContext(model, dataset, master_coords, schema, ns, dim_coords))
        if hasattr(value, 'solution'):
            return value.solution
        if isinstance(value, xarray.DataArray):
            return value
        return xarray.DataArray(float(value))
