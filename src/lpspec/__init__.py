"""Declarative optimisation: YAML math on a streaming engine.

Models build relationally on polars and stream to the solver — see
docs/about/architecture.md. linopy is not imported here; with the ``[linopy]``
extra it is the second lane a file can be built on, and the differential-test
oracle (``from lpspec import linopy as lpspec_linopy``).

Example::

    import lpspec as lps

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

from math_spec import Model

from lpspec.api import build, check, load_model, solve, write
from lpspec.errors import (
    DataError,
    DimensionError,
    LaneError,
    LanguageError,
    LpspecError,
    PiecewiseExpansionError,
    SchemaError,
)
from lpspec.strategy import EachCoordinate, EachWindow, solve_over

__all__ = [
    'DataError',
    'DimensionError',
    'EachCoordinate',
    'EachWindow',
    'LaneError',
    'LanguageError',
    'LpspecError',
    'Model',
    'PiecewiseExpansionError',
    'SchemaError',
    'build',
    'check',
    'load_model',
    'solve',
    'solve_over',
    'write',
]

try:
    __version__ = _installed_version('lpspec')
except _PackageNotFoundError:
    __version__ = '0.0.0'
