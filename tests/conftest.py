"""Shared fixtures and schema helpers for lpspec tests.

Everything here is linopy-free *and pandas-free at import*, so it loads on a
bare install. On a bare install (no [linopy] extra) the eager/oracle modules
skip themselves: they reach the oracle through ``tests.oracle``, whose
``importorskip`` guard fires at collection. There is no list of filenames to
keep in sync here — a module that needs the extra says so by importing it. The
differential harness lives in ``tests.differential`` for the same reason:
importing it *is* the guard.

pandas follows the same discipline one level down. It is no longer a runtime
dependency (it ships with the ``[linopy]`` extra, for the oracle and for
``Result.to_pandas``), so a fixture that hands out pandas objects imports it in
its own body: requesting the fixture is what asks for the dependency, and the
bare job never requests it. ``dispatch_inputs`` and ``dispatch_frame_inputs``
are the same numbers in the two shapes — the oracle lane is pandas-native, the
engine is frame-native, and the module constants are the single source of both.
"""

from __future__ import annotations

import contextlib
import copy
import difflib
import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pytest
import yaml as pyyaml

from lpspec.language._yaml import parse_yaml, read_yaml
from lpspec.language.validation import load_model
from tools import constructs

if TYPE_CHECKING:
    from lpspec.language.model import Model

EXAMPLES_DIR = Path(__file__).parent.parent / 'examples'

#: The referenced models — the ports somebody else published an optimum for,
#: plus the teaching models that carry a hand-written reference implementation
#: — with their data and the number each should reach. Shared because two
#: modules ask different questions of one corpus: ``test_ports.py`` whether we
#: reach the outside answer, ``test_rebind.py`` whether a rebind reaches the
#: answer a fresh build does.
PORTS_DIR = EXAMPLES_DIR / 'ports'
PORT_REFERENCES: dict[str, dict[str, Any]] = json.loads((PORTS_DIR / 'references.json').read_text())


def bindable_on_this_install(name: str) -> None:
    """Skip the referenced models the bare install cannot bind.

    ``piecewise`` declares ``method: convex``, whose curvature guard needs
    xarray until issue #27 makes it numpy-only. The guard runs at bind, so
    ``lps.check`` stays exercised on every install and only the data-touching
    tests skip.
    """
    if name == 'piecewise':
        pytest.importorskip('xarray', reason="piecewise's convex curvature guard needs xarray until #27")


def port_sources(name: str) -> dict[str, Any]:
    """One JSON per port, filtered to what its model declares.

    The file carries what the upstream framework dumped — `pypsa_kvl` ships a
    `reactance` the ported model reads through `cycle_incidence` instead, and
    `pypsa_ac_dc` six more of that kind. Keeping them is the point: they are the
    provenance of the instance. Binding refuses a name the model does not
    declare, so the filter belongs here, where a dump becomes a call.
    """
    data = json.loads((PORTS_DIR / 'data' / f'{name}.json').read_text())
    tables = {k: pl.DataFrame(v) if isinstance(v, dict) else v for k, v in data.items()}
    model = PORTS_DIR / f'{name}.yaml'
    schema = load_model(model if model.exists() else EXAMPLES_DIR / f'{name}.yaml')
    known = {**schema.parameters, **schema.dimensions}
    return {k: v for k, v in tables.items() if k in known}


@pytest.fixture(params=sorted(PORT_REFERENCES), ids=str)
def port(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Each referenced model in turn: its name, its file, and what it should reach.

    A port's model file lives in ``examples/ports/``; a teaching model with a
    reference implementation keeps its file in ``examples/``, where the guide
    and the gallery already point.
    """
    model = PORTS_DIR / f'{request.param}.yaml'
    if not model.exists():
        model = EXAMPLES_DIR / f'{request.param}.yaml'
    return {'name': request.param, 'model': model} | PORT_REFERENCES[request.param]


#: Every model in the repo, ports included — ``constructs.models()`` is the one
#: list the gallery and the docs already build from, so a model added anywhere
#: is covered the day it lands rather than when someone remembers a glob.
MODEL_PATHS = [p for _, p in constructs.models()]

#: The dispatch model as a dict, for tests that need to mutate a declaration
#: rather than read a file. Deliberately the same math as
#: ``examples/dispatch.yaml`` so a reader who knows one knows the other; use
#: :func:`override` to vary it.
DISPATCH_MODEL: dict[str, Any] = {
    'dimensions': {'snapshot': {'dtype': 'int'}, 'generator': {'values': ['wind', 'gas']}},
    'parameters': {
        'p_max': {'dims': ['generator']},
        'cost': {'dims': ['generator']},
        'load': {'dims': ['snapshot']},
    },
    'variables': {'p': {'foreach': ['snapshot', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}}},
    'constraints': {'balance': {'foreach': ['snapshot'], 'expression': 'sum(p, over=generator) == load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost, over=generator)'},
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        '--update-golden',
        action='store_true',
        default=False,
        help='rewrite committed golden output (examples/*.out) from this run instead of asserting on it',
    )


# ---------------------------------------------------------------------------
# building schemas to test against
# ---------------------------------------------------------------------------


def override(base: dict[str, Any], **patch: Any) -> dict[str, Any]:
    """A deep copy of ``base`` with dotted paths replaced.

    ``override(DISPATCH_MODEL, **{'variables.p.where': 'p_max > 0'})``. Missing
    intermediate keys are created, so this both edits an existing declaration
    and adds a new one — which is what makes a whole family of "the base model
    but for one thing" tests a one-liner each.
    """
    raw = copy.deepcopy(base)
    for dotted, value in patch.items():
        node = raw
        *parents, leaf = dotted.split('.')
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return raw


def schema_of(source: str | Path | dict[str, Any], **patch: Any) -> Model:
    """A ``Model`` from a YAML path, YAML text, or a raw dict.

    ``Path`` means a file, ``str`` means the YAML itself — the distinction is
    the type, never a guess about the content. ``**patch`` applies
    :func:`override` first, which is how a test says "this example, but with
    ``**`` in the objective".
    """
    raw = raw_of(source)
    return load_model(override(raw, **patch) if patch else raw)


def raw_of(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """The parsed mapping behind a path / YAML text / dict, unvalidated."""
    if isinstance(source, dict):
        return source
    return read_yaml(source) if isinstance(source, Path) else parse_yaml(source)


def solve_lp_file(path: Path | str) -> float:
    """Objective HiGHS reaches reading the written LP file back from disk.

    The third opinion in a differential: the ``highs`` solver builds the model
    through the HiGHS API, this one round-trips it through text, and a sink
    that writes a wrong file is otherwise invisible. Lives here rather than in
    ``tests.differential`` because highspy is a core dependency — a bare
    install must still be able to check the LP sink.
    """
    import highspy

    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    h.readModel(str(path))
    h.run()
    assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    return h.getInfo().objective_function_value


def by_coord(result: Any, name: str, *dims: str) -> dict[Any, float]:
    """A variable's primal as ``{coordinate: value}`` — tuple keys past one dim.

    One ``primal`` call, then one zip — and that is the whole reason this is a
    function. ``primal`` is a label join and promises row *order* but not that
    two separate calls line up column-wise, so the idiom has to read the frame
    once and pair its columns in a single pass. A dozen tests do this and the
    caveat was written down at one of them; here it applies to all by
    construction.
    """
    frame = result.primal(name)
    columns = [frame[dim] for dim in dims]
    keys = columns[0] if len(dims) == 1 else zip(*columns, strict=True)
    return dict(zip(keys, frame['value'], strict=True))


def resolved(text, schema):
    """Parse + expand + resolve — exactly what a backend receives.

    Tests that call `_lower_expr` or `evaluate_where` directly must go through
    this: a raw `parse_expression` result still holds NameNodes, and both
    backends now assert those never reach them (resolution.py).
    """
    from lpspec.language.resolution import Namespace, expression_of

    return expression_of(text, schema, Namespace.of(schema), 't')


# ---------------------------------------------------------------------------
# examples as evidence
# ---------------------------------------------------------------------------


def run_example(path: Path, name: str) -> str:
    """Import a script as module ``name`` and capture what its ``main()`` prints.

    ``StringIO`` is not a tty, so any banners come out unstyled — the same
    plain text a shell redirect into a golden file would produce.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            module.main()
    finally:
        del sys.modules[name]
    return buffer.getvalue()


def assert_golden(output: str, golden: Path, pytestconfig: pytest.Config, drifted: str) -> None:
    """``output`` matches the committed golden, or ``--update-golden`` rewrites it.

    Args:
        output: What this run printed.
        golden: The committed file to compare against or rewrite.
        pytestconfig: The fixture, for the ``--update-golden`` flag.
        drifted: What a mismatch means for this example — opens the failure
            message, above the diff, and should say how to regenerate.
    """
    if pytestconfig.getoption('--update-golden'):
        golden.write_text(output)
        pytest.skip(f'rewrote {golden.name} from this run')
    expected = golden.read_text()
    if output != expected:
        diff = '\n'.join(difflib.unified_diff(expected.splitlines(), output.splitlines(), 'committed', 'this run'))
        pytest.fail(f'{drifted}\n{diff}', pytrace=False)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


@pytest.fixture
def dispatch_yaml() -> Path:
    return EXAMPLES_DIR / 'dispatch.yaml'


@pytest.fixture
def dispatch_model_inputs():
    """``DISPATCH_MODEL``'s data as pandas — what the parity modules feed both lanes."""
    import pandas as pd

    data = {
        'p_max': pd.Series({'wind': 100.0, 'gas': 200.0}),
        'cost': pd.Series({'wind': 0.0, 'gas': 50.0}),
        'load': pd.Series([80.0] * 4, index=pd.RangeIndex(4, name='snapshot')),
    }
    return data, {'snapshot': pd.RangeIndex(4, name='snapshot')}


def dispatch_model_path(directory: Path, **patch: Any) -> Path:
    """``DISPATCH_MODEL``, varied and written to disk — the eager lane only takes a path."""
    path = directory / 'model.yaml'
    path.write_text(pyyaml.safe_dump(override(DISPATCH_MODEL, **patch)))
    return path


#: Generators of ``examples/dispatch.yaml``, and the snapshot count. Distinct
#: costs, so the optimal vertex is unique and primals are comparable across
#: lanes.
DISPATCH_GENERATORS = ('wind', 'solar', 'gas')
DISPATCH_P_MAX = (100.0, 60.0, 200.0)
DISPATCH_COST = (1.0, 2.0, 50.0)
DISPATCH_SNAPSHOTS = 48


def _dispatch_load() -> np.ndarray:
    """The load series both shapes below carry — one draw, one seed."""
    rng = np.random.default_rng(3)
    return (rng.uniform(0.2, 0.8, DISPATCH_SNAPSHOTS) * sum(DISPATCH_P_MAX)).round(3)


@pytest.fixture
def dispatch_inputs():
    """Dispatch data as pandas — the shape the linopy oracle takes.

    Pairs with :func:`dispatch_frame_inputs`: same numbers, and a test picks
    the shape by which lane it exercises. Importing pandas here rather than at
    module scope is what keeps this file loadable on a bare install.
    """
    import pandas as pd

    data = {
        'p_max': pd.Series(dict(zip(DISPATCH_GENERATORS, DISPATCH_P_MAX, strict=True))),
        'cost': pd.Series(dict(zip(DISPATCH_GENERATORS, DISPATCH_COST, strict=True))),
        'load': pd.Series(_dispatch_load(), index=pd.RangeIndex(DISPATCH_SNAPSHOTS, name='snapshot')),
    }
    coords = {'snapshot': pd.RangeIndex(DISPATCH_SNAPSHOTS, name='snapshot')}
    return data, coords


@pytest.fixture
def dispatch_frame_inputs():
    """The same data as tidy frames — the shape the engine documents.

    Tests that assert the native API's behaviour use this one, so they stay
    runnable with no dataframe library beyond the engine's own installed.
    """
    import polars as pl

    generators = list(DISPATCH_GENERATORS)
    data = {
        'p_max': pl.DataFrame({'generator': generators, 'value': list(DISPATCH_P_MAX)}),
        'cost': pl.DataFrame({'generator': generators, 'value': list(DISPATCH_COST)}),
        'load': pl.DataFrame({'snapshot': list(range(DISPATCH_SNAPSHOTS)), 'value': _dispatch_load()}),
    }
    coords = {'snapshot': range(DISPATCH_SNAPSHOTS)}
    return data, coords


@pytest.fixture
def commitment_inputs():
    """Data for the unit-commitment MILP in ``tests.test_milp``.

    Here rather than beside the model because two modules need it: the MILP
    itself, and the duals refusal — a mixed-integer model is the case that has
    no dual solution to give.
    """
    import pandas as pd

    rng = np.random.default_rng(5)
    n_s = 24
    p_max = pd.Series({'coal': 120.0, 'gas': 80.0, 'peaker': 60.0})
    data = {
        'p_max': p_max,
        'cost': pd.Series({'coal': 10.0, 'gas': 30.0, 'peaker': 90.0}),
        'fix_cost': pd.Series({'coal': 400.0, 'gas': 150.0, 'peaker': 20.0}),
        'load': pd.Series(
            (rng.uniform(0.3, 0.9, n_s) * p_max.sum()).round(1),
            index=pd.RangeIndex(n_s, name='snapshot'),
        ),
    }
    coords = {
        'snapshot': pd.RangeIndex(n_s, name='snapshot'),
        'generator': pd.Index(p_max.index, name='generator'),
    }
    return data, coords


@pytest.fixture
def transport_data():
    """A four-bus network whose data is feasible by construction.

    Generation is dealt round-robin so every bus has some locally, the topology
    is a ring plus one chord so every bus is reachable, and loads sit below each
    bus's local capacity — feasible even with zero flow. The cost spread still
    makes cross-bus flows optimal, so the network is not decoration.
    """
    import pandas as pd

    rng = np.random.default_rng(11)
    n_s, n_b, n_g, n_l = 24, 4, 9, 5
    buses = [f'b{i}' for i in range(n_b)]
    gens = pd.DataFrame(
        {
            'generator': [f'g{i}' for i in range(n_g)],
            'bus': [buses[i % n_b] for i in range(n_g)],
            'p_max': rng.uniform(80, 150, n_g).round(3),
            'cost': rng.uniform(5, 100, n_g).round(3),
        }
    )
    pairs = [(buses[i], buses[(i + 1) % n_b]) for i in range(n_b)] + [(buses[0], buses[2])]
    lines = pd.DataFrame(
        {
            'line': [f'l{i}' for i in range(n_l)],
            'from_bus': [a for a, _ in pairs],
            'to_bus': [b for _, b in pairs],
            'cap': rng.uniform(60, 120, n_l).round(3),
        }
    )
    local_cap = gens.groupby('bus')['p_max'].sum().reindex(buses).to_numpy()
    factors = rng.uniform(0.3, 0.8, (n_s, n_b))
    load = pd.DataFrame(
        {
            'snapshot': np.repeat(np.arange(n_s), n_b),
            'bus': buses * n_s,
            'value': (factors * local_cap).round(3).ravel(),
        }
    )
    return gens, lines, load
