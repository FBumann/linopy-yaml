"""Declarative optimisation: YAML math on a streaming engine.

Specs build relationally on polars and stream to the solver — see
docs/about/architecture.md. linopy is not imported here; with the ``[linopy]``
extra it is the second lane a file can be built on, and the differential-test
oracle (``from lpspec import linopy as lpspec_linopy``).

Example::

    import lpspec as lps

    result = lps.solve('spec.yaml', {'p_max': 'p_max.parquet', 'load': 'load.parquet'})
    result.objective
    result.primal('p')  # tidy polars.DataFrame
    result.to_dataarray('p')  # labelled, for array post-processing

``__version__`` reads the installed metadata: the git tag is the source of
truth, and hatch-vcs bakes it in at build time. A source tree with nothing
installed reads ``0.0.0``.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _installed_version

from lpspec.api import Model, build, check, solve, write
from lpspec.errors import (
    DataError,
    DimensionError,
    LaneError,
    LanguageError,
    LpspecError,
    LpspecWarning,
    NoSolutionError,
    PiecewiseExpansionError,
    SchemaError,
)
from lpspec.projection import Region, project
from lpspec.relational.result import Result
from lpspec.strategy import EachCoordinate, EachWindow, Runs, solve_over

__all__ = [
    'DataError',
    'DimensionError',
    'EachCoordinate',
    'EachWindow',
    'LaneError',
    'LanguageError',
    'LpspecError',
    'LpspecWarning',
    'Model',
    'NoSolutionError',
    'PiecewiseExpansionError',
    'Region',
    'Result',
    'Runs',
    'SchemaError',
    'build',
    'check',
    'project',
    'solve',
    'solve_over',
    'write',
]

try:
    __version__ = _installed_version('lpspec')
except _PackageNotFoundError:
    __version__ = '0.0.0'
