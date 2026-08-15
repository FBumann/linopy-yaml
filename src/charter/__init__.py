"""Declarative optimisation: YAML math on a streaming engine.

Models build relationally on polars and stream to the solver — see
docs/ARCHITECTURE.md. linopy is not imported at runtime; it serves as the
differential-test oracle and as an opt-in compatibility shim
(``from charter import linopy as charter_linopy``).

Example::

    import charter as lps

    result = lps.solve('model.yaml', {'p_max': 'p_max.parquet', 'load': 'load.parquet'})
    result.objective
    result.primal('p')  # tidy polars.DataFrame
    result.to_dataarray('p')  # labelled, for array post-processing

``__version__`` reads the installed metadata: the git tag is the source of
truth, and hatch-vcs bakes it in at build time. A source tree with nothing
installed reads ``0.0.0``.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _installed_version

from charter.api import build, check, load_model, solve, write
from charter.errors import (
    CharterError,
    DataError,
    DimensionError,
    LanguageError,
    PiecewiseExpansionError,
    SchemaError,
)
from charter.language.model import Model
from charter.strategy import EachCoordinate, EachWindow, solve_over
from charter.typeset import SymbolTable, to_latex, to_markdown, to_typst

__all__ = [
    'CharterError',
    'DataError',
    'DimensionError',
    'EachCoordinate',
    'EachWindow',
    'LanguageError',
    'Model',
    'PiecewiseExpansionError',
    'SchemaError',
    'SymbolTable',
    'build',
    'check',
    'load_model',
    'solve',
    'solve_over',
    'to_latex',
    'to_markdown',
    'to_typst',
    'write',
]

try:
    __version__ = _installed_version('charter')
except _PackageNotFoundError:
    __version__ = '0.0.0'
