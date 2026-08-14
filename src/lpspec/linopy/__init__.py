"""Compatibility shim: YAML math onto a ``linopy.Model``.

Requires the ``[linopy]`` extra (linopy, xarray).

The product path is YAML → AST → streaming engine; linopy is not on it. This
module exists for two narrow jobs:

1. **Python math that the language cannot say.** Build (or extend) a model in
   linopy, where arbitrary Python is available.
2. **Parity checking.** Every language feature is differentially tested by
   running the same YAML + data through both this and the streaming engine.
   That is only meaningful because both accept *exactly* the same language —
   there is no construct that works here and not there.

Three functions — two producers and one reader — and all are **pure**: YAML
goes in, a model or a value comes out, and nothing is retained. No accessor
on the model, no session, no state. A file's meaning never depends on what was
loaded before it (docs/ARCHITECTURE.md, hard rule 5), so every file declares
the parameters it uses and the caller supplies their data per call — the
reader included, which is why :func:`expression` takes ``data=`` again rather
than remembering what :func:`build` saw::

    from lpspec import linopy as lpspec_linopy

    m = lpspec_linopy.build('model.yaml', data={...})
    lpspec_linopy.extend(m, 'ramp_constraint.yaml', data={...})
    m.solve(...)
    lpspec_linopy.expression(m, 'model.yaml', 'co2', data={...})

For models declared entirely in YAML, use the native API — it streams::

    import lpspec as lps

    with lps.solve('model.yaml', {...}) as result:
        result.primal('p')

**Importing this module sets** ``linopy.options['semantics'] = 'v1'``. This
lane speaks v1 and the option is global, so importing is what sets it.
linopy's ``legacy`` default fills every absent slot with 0, where the
relational lane drops the row (SPEC §6, §7) — left alone the two lanes answer
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

import warnings
from pathlib import Path
from typing import Any

try:
    import linopy
    import pandas as pd
    import xarray
except ModuleNotFoundError as exc:
    msg = 'The linopy compatibility layer requires the [linopy] extra: pip install "lpspec[linopy]"'
    raise ModuleNotFoundError(msg) from exc


from lpspec._notes import note
from lpspec.errors import LanguageError, unknown_name_message
from lpspec.language.expression_parser import ComparisonNode
from lpspec.language.piecewise import expand_piecewise
from lpspec.language.resolution import Namespace, expression_of
from lpspec.language.validation import load_model
from lpspec.linopy.builder import EvaluationContext, _eval_ast, build_model
from lpspec.linopy.loader import (
    build_dim_coords,
    build_master_coords,
    dim_index_of,
    load_parameters,
)
from lpspec.sources import validate_piecewise_data

linopy.options['semantics'] = 'v1'

__all__ = ['build', 'expression', 'extend']


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


def extend(
    model: linopy.Model,
    path: str | Path,
    *,
    data: dict[str, Any] | None = None,
    coords: dict[str, Any] | None = None,
) -> None:
    """Add variables, constraints, and/or an objective from YAML to *model*.

    Expressions may reference variables *model* already carries; every
    parameter they use is declared in this file and supplied in this call. A
    referenced dimension takes its labels from ``coords``, then from the
    model's existing variables, then from this file's ``values:``.

    Args:
        model: Extended in place.
        path: Path to the YAML file.
        data: Parameter data, keyed by the names the YAML declares.
        coords: Dimension coordinate values.

    Raises:
        LanguageError: A file the language does not accept, or ``values:``
            for a dim the model already carries with other labels.
        DataError: A dimension nothing resolves, or parameter data that does
            not fit the file.
    """
    path = Path(path)
    with note(f"while extending with YAML '{path}'"):
        known_variables = _variable_dims(model)
        original = load_model(path, known_variables=known_variables)
        schema = expand_piecewise(original, known_variables=known_variables)

        existing_coords = _infer_coords(model)
        if coords is not None:
            existing_coords.update({k: dim_index_of(v, k) for k, v in coords.items()})

        for dim_name, dim_def in schema.dimensions.items():
            if dim_def.values is None or dim_name not in existing_coords:
                continue
            declared = pd.Index(dim_def.values, name=dim_name)
            existing = existing_coords[dim_name]
            if not declared.equals(existing):
                msg = (
                    f"Extension declares dimension '{dim_name}' with values "
                    f'that differ from the existing model.\n'
                    f'  Existing: {list(existing)}\n'
                    f'  Declared: {list(declared)}\n'
                    f"Either omit 'values:' for '{dim_name}' in the "
                    f'extension, or make them match.'
                )
                raise LanguageError(msg)

        master_coords = build_master_coords(schema, existing_coords)
        dim_coords = build_dim_coords(schema, coords, master_coords)
        dataset = load_parameters(schema, data, master_coords)
        validate_piecewise_data(original, dataset)

        build_model(model, schema, dataset, master_coords, dim_coords)


def expression(
    model: linopy.Model,
    path: str | Path,
    name: str,
    *,
    data: dict[str, Any] | None = None,
    coords: dict[str, Any] | None = None,
) -> Any:
    """Evaluate named expression *name* of *path* at *model*'s solution.

    The eager lane's half of readable expressions — the streaming lane spells
    it ``result.expression(name)``. Pure like the other two verbs: nothing was
    retained by :func:`build`, so the same *data* and *coords* the model was
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
        ns = Namespace.of(schema, [str(v) for v in model.variables])
        ast = expression_of(schema.expressions[name], schema, ns, f"named expression '{name}'")
        assert not isinstance(ast, ComparisonNode), 'load-time validation refuses a comparison in a named expression'
        value = _eval_ast(ast, EvaluationContext(model, dataset, master_coords, schema, ns, dim_coords))
        if hasattr(value, 'solution'):
            return value.solution
        if isinstance(value, xarray.DataArray):
            return value
        return xarray.DataArray(float(value))


def _variable_dims(model: linopy.Model) -> dict[str, list[str]]:
    """The dims of every variable on *model*, as language names.

    An extension is deliberately not valid alone — it references variables the
    model already has — so these travel in as validation context and the file
    is checked against the namespace it will run in. linopy's dims are
    ``Hashable``; the language's are names.
    """
    return {n: [str(d) for d in model.variables[n].dims] for n in model.variables}


def _infer_coords(model: linopy.Model) -> dict[str, pd.Index]:
    """Union the coordinates of every variable on ``model``, keyed by dim.

    Delegates to ``model.variables.indexes``, linopy's public API for the
    per-dimension union of coordinates across all variables. linopy warns when
    variables carry non-aligned coords and performs an outer join; that outer
    join is exactly the union wanted here, so the warning is suppressed rather
    than answered.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='Coordinates across variables not equal',
            category=UserWarning,
        )
        return dict(model.variables.indexes)
