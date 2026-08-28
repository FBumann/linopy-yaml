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
from math_spec import curvature_required, to_spec

from lpspec.relational.sinks import SOLVERS
from lpspec.sources import bindable

# The language's own tests own these (#1150); the noqa marks the two this file
# re-exports without using, so the forty-odd tests on the other side of the cut
# keep importing all four from one place.
from tests.fixtures import (  # noqa: F401
    DISPATCH_MODEL,
    override,
    raw_of,
    schema_of,
)
from tools import constructs

if TYPE_CHECKING:
    from collections.abc import Sequence

EXAMPLES_DIR = Path(__file__).parent.parent / 'examples'

#: The referenced models — the ports somebody else published an optimum for,
#: plus the teaching models that carry a hand-written reference implementation
#: — with their data and the number each should reach. Shared because two
#: modules ask different questions of one corpus: ``test_ports.py`` whether we
#: reach the outside answer, ``test_rebind.py`` whether a rebind reaches the
#: answer a fresh build does.
PORTS_DIR = EXAMPLES_DIR / 'ports'
PORT_REFERENCES: dict[str, dict[str, Any]] = constructs.REFERENCES


def bindable_on_this_install(name: str) -> None:
    """Skip the referenced models the bare install cannot bind.

    A ``method:`` whose exactness depends on the curve's *shape* — ``convex``
    and ``lp`` — is guarded against the breakpoint values, and that guard needs
    xarray until issue #27 makes it numpy-only. Read off the model rather than
    listed by name, so a third such model is covered the day it lands. The
    guard runs at bind, so ``lps.check`` stays exercised on every install and
    only the data-touching tests skip.
    """
    schema = to_spec(port_model(name))
    if any(curvature_required(pw) is not None for pw in schema.piecewise.values()):
        pytest.importorskip('xarray', reason=f"{name}'s curvature guard needs xarray until #27")


def relation(over: str, into: str, labels: Sequence[Any], values: Sequence[Any]) -> pl.DataFrame:
    """A lookup's map as the table it is supplied under its own key.

    Takes the column form these fixtures used to carry — one value per label,
    `None` where the label maps nowhere — and returns the relation: the rows it
    maps, and no row for the rest. The tests that pin the transport itself
    write their relations out literally; this is for the many where the map is
    a prop rather than the subject.
    """
    rows = [(a, b) for a, b in zip(labels, values, strict=True) if b is not None]
    return pl.DataFrame({over: [a for a, _ in rows], into: [b for _, b in rows]})


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
    schema = to_spec(model if model.exists() else EXAMPLES_DIR / f'{name}.yaml')
    return {k: v for k, v in tables.items() if k in bindable(schema)}


def port_model(name: str) -> Path:
    """The file behind a referenced model's name.

    A port's model file lives in ``examples/ports/``; a teaching model with a
    reference implementation keeps its file in ``examples/``, where the guide
    and the gallery already point.
    """
    model = PORTS_DIR / f'{name}.yaml'
    return model if model.exists() else EXAMPLES_DIR / f'{name}.yaml'


@pytest.fixture(params=sorted(PORT_REFERENCES), ids=str)
def port(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Each referenced model in turn: its name, its file, and what it should reach."""
    return {'name': request.param, 'model': port_model(request.param)} | PORT_REFERENCES[request.param]


#: Every model in the repo, ports included — ``constructs.models()`` is the one
#: list the gallery and the docs already build from, so a model added anywhere
#: is covered the day it lands rather than when someone remembers a glob.
MODEL_PATHS = [p for _, p in constructs.models()]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        '--update-golden',
        action='store_true',
        default=False,
        help='rewrite committed golden output (examples/*.out) from this run instead of asserting on it',
    )
    parser.addoption(
        '--sweep-depth',
        type=int,
        default=2,
        help='how deep tests/test_expression_sweep.py enumerates; 3 is minutes and has its own CI job',
    )
    parser.addoption(
        '--sweep-shard',
        default='0/1',
        help='run the i-th of n equal strides of the sweep, as `i/n`; the CI job takes one per matrix leg',
    )


# ---------------------------------------------------------------------------
# building schemas to test against
# ---------------------------------------------------------------------------


def solve_written_file(path: Path | str) -> float:
    """Objective HiGHS reaches reading a written model back from disk.

    The third opinion in a differential: the ``highs`` solver builds the model
    through the HiGHS API, this one round-trips it through text, and a sink
    that writes a wrong file is otherwise invisible. The format is the path's,
    HiGHS reading both of the ones that ship. Lives here rather than in
    ``tests.differential`` because highspy is a core dependency — a bare
    install must still be able to check the writers.
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
    """``DISPATCH_MODEL``'s data as pandas, index included — one mapping, both lanes."""
    import pandas as pd

    return {
        'p_max': pd.Series({'wind': 100.0, 'gas': 200.0}),
        'cost': pd.Series({'wind': 0.0, 'gas': 50.0}),
        'load': pd.Series([80.0] * 4, index=pd.RangeIndex(4, name='snapshot')),
        'snapshot': pd.RangeIndex(4, name='snapshot'),
        'generator': ['wind', 'gas'],
    }


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

    return {
        'p_max': pd.Series(dict(zip(DISPATCH_GENERATORS, DISPATCH_P_MAX, strict=True))),
        'cost': pd.Series(dict(zip(DISPATCH_GENERATORS, DISPATCH_COST, strict=True))),
        'load': pd.Series(_dispatch_load(), index=pd.RangeIndex(DISPATCH_SNAPSHOTS, name='snapshot')),
        'snapshot': pd.RangeIndex(DISPATCH_SNAPSHOTS, name='snapshot'),
        'generator': list(DISPATCH_GENERATORS),
    }


@pytest.fixture
def dispatch_frame_inputs():
    """The same data as tidy frames — the shape the engine documents.

    Tests that assert the native API's behaviour use this one, so they stay
    runnable with no dataframe library beyond the engine's own installed.
    """
    generators = list(DISPATCH_GENERATORS)
    return {
        'p_max': pl.DataFrame({'generator': generators, 'value': list(DISPATCH_P_MAX)}),
        'cost': pl.DataFrame({'generator': generators, 'value': list(DISPATCH_COST)}),
        'load': pl.DataFrame({'snapshot': list(range(DISPATCH_SNAPSHOTS)), 'value': _dispatch_load()}),
        'snapshot': pl.DataFrame({'snapshot': range(DISPATCH_SNAPSHOTS)}),
        'generator': pl.DataFrame({'generator': generators}),
    }


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
    return data | {
        'snapshot': pd.RangeIndex(n_s, name='snapshot'),
        'generator': pd.Index(p_max.index, name='generator'),
    }


# ---------------------------------------------------------------------------
# the law fixture: one model, one masked dimension
# ---------------------------------------------------------------------------
#
# `test_arithmetic_laws.py` states the laws a reader should know, chosen by
# hand; `test_expression_sweep.py` sweeps every spelling at a bounded depth.
# The second is evidence about the first only while both are written over the
# *same* model — otherwise a law holding there and a sweep agreeing here are
# two unrelated facts — which is the second importer that brings it here.

#: The two dimensions every expression in those two files is written over.
LAW_DIMS = {'f': {'dtype': 'str'}, 't': {'dtype': 'int'}}


def law_data() -> dict[str, Any]:
    """``gate`` masks ``y`` at ``f=b``; ``w`` is a dense coefficient.

    Every interesting law is conditional on whether absence is in play, so the
    fixture keeps one masked variable and one total one, and one coefficient
    that is not a variable at all. The labels of both dimensions ride along:
    they are data now, and every law is written over the same two.
    """
    import pandas as pd

    return {
        'f': ['a', 'b'],
        't': [0, 1],
        'gate': pd.Series({'a': True}),
        'w': pd.Series({'a': 2.0, 'b': 3.0}),
    }


def law_model(
    expression: str,
    *,
    foreach: list[str],
    objective: str = 'sum(x)',
    also: dict | None = None,
) -> dict:
    """A model whose only variable content is *expression*, in a binding row.

    Args:
        expression: The constraint the model exists to state.
        foreach: The dimensions the row is repeated over. Required rather than
            defaulted: a row repeated across a dimension its expression does
            not carry is refused, so the caller that built the expression is
            the one that knows.
        objective: What is maximised, unless the caller needs the row to bind
            against something else.
        also: A second named constraint, for the cases that need two rules —
            which are two blocks rather than two entries in a list (#298).
    """
    return {
        'dimensions': dict(LAW_DIMS),
        'parameters': {'gate': {'dims': ['f'], 'dtype': 'bool'}, 'w': {'dims': ['f']}},
        'variables': {
            'x': {'foreach': ['f', 't'], 'bounds': {'lower': 0, 'upper': 100}},
            'y': {'foreach': ['f', 't'], 'where': 'gate', 'bounds': {'lower': 0, 'upper': 50}},
        },
        'constraints': {'c': {'foreach': foreach, 'expression': expression}, **(also or {})},
        'objective': {'sense': 'maximize', 'expression': objective},
    }


def masked_operand_model(constraint: str, expression: str, *, grouped: bool = False, masked: bool = True) -> dict:
    """The probe behind the shift and window edge cases, over one masked operand.

    ``level`` is masked where ``usable`` says so (or not at all, under
    ``masked=False`` — the operand that reaches the operator with no presence
    frame of its own), and ``take`` is capped only by *expression*'s row. The
    1000x penalty on ``level`` is the knowledge: it makes "row dropped" and
    "row built and binding" separable from the objective alone, rather than
    only from a row count. ``grouped`` adds the ``season_of`` lookup the
    partitioned walks read.
    """
    model: dict[str, Any] = {
        'dimensions': {'t': {'dtype': 'int'}},
        'parameters': {'usable': {'dims': ['t']}},
        'variables': {
            'level': {'foreach': ['t'], 'where': 'usable > 0', 'bounds': {'lower': 0, 'upper': 10}},
            'take': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 10}},
        },
        'constraints': {constraint: {'foreach': ['t'], 'expression': expression}},
        'objective': {'sense': 'maximize', 'expression': 'sum(take, over=t) - 1000 * sum(level, over=t)'},
    }
    if grouped:
        model['dimensions']['season'] = {'dtype': 'str'}
        model['lookups'] = {'season_of': {'over': 't', 'into': 'season'}}
    if not masked:
        del model['parameters']
        del model['variables']['level']['where']
    return model


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


def recomputed_row_values(engine, result) -> Any:
    """Every row's left-hand side at the solution, recomputed from the model.

    ``Ax`` for a linear row and ``xᵀQx + Ax`` for a quadratic one, out of the
    built frames and the primal — and nothing else the solver produced, which
    is what makes agreement with ``result.activity`` a check of the whole chain
    rather than a tautology. For a quadratic row it stands in for the oracle
    the linopy lane cannot provide.

    Scattered rather than ``reduceat``-ed: a purely quadratic row owns no
    linear entries, and ``reduceat`` repeats the previous row on an empty span.
    """
    tables = engine._model.tables()
    x = np.zeros(tables.column_count)
    for name, block in engine._model.variables.items():
        x[block.start : block.start + block.height] = result.primal(name)['value'].to_numpy()

    values = np.zeros(tables.row_count)
    spans = np.diff(tables.row_starts)
    rows = np.repeat(np.arange(tables.row_count), spans)
    np.add.at(values, rows, tables.matrix['coeff'].to_numpy() * x[tables.matrix['col'].to_numpy()])

    quadratic = tables.qmatrix
    if quadratic.height:
        pairs = quadratic['coeff'].to_numpy() * x[quadratic['col_l'].to_numpy()] * x[quadratic['col_r'].to_numpy()]
        np.add.at(values, quadratic['row'].to_numpy(), pairs)
    return values


#: A three-period model whose solver vector is exactly three long — shared
#: because the hand-off tests and the timing tests both need one that small.
#: Three columns and three rows, the smallest model whose solution vector has a
#: length worth disagreeing about — and, every declared row being built, the
#: control for the omissions report.
SOLVER_VECTOR_MODEL = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'load': {'dims': ['t']}},
    'variables': {'x': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'meet': {'foreach': ['t'], 'expression': 'x >= load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x, over=t)'},
}


SOLVER_VECTOR_LOAD = {'t': [0, 1, 2], 'load': pl.DataFrame({'t': [0, 1, 2], 'value': [1.0, 2.0, 3.0]})}


# ---------------------------------------------------------------------------
# the sink cases: what a second solver has to earn
# ---------------------------------------------------------------------------

LP = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'load': {'dims': ['t']}, 'price': {'dims': ['t']}},
    'variables': {'p': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 100}}},
    'constraints': {'meet': {'foreach': ['t'], 'expression': 'p >= load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * price, over=t)'},
}

#: A convex quadratic objective — the third convention for one form, so two
#: sinks agreeing is what says the conversion is right rather than consistent.
QP = {
    'dimensions': {'g': {'dtype': 'str'}},
    'parameters': {'need': {'dims': []}, 'toll': {'dims': ['g']}},
    'variables': {
        'p': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 10}},
        'q': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {'meet': {'foreach': [], 'expression': 'sum(p, over=g) + sum(q, over=g) >= need'}},
    #: A linear term beside the quadratic one, deliberately: ``setMObjective``
    #: sets the *whole* objective, so a hand-off that passed only ``Q`` would
    #: drop the linear half — and a purely quadratic case could not tell.
    'objective': {'sense': 'minimize', 'expression': 'sum(p * p + p * q + q * q + q * toll, over=g)'},
}

QP_SOURCES = {
    'g': ['a', 'b'],
    'need': pl.DataFrame({'value': [24.0]}),
    'toll': pl.DataFrame({'g': ['a', 'b'], 'value': [1.0, 7.0]}),
}

#: Maximisation *and* an objective constant, which are the two things the
#: sink states outside the frames: ``ModelSense`` and ``ObjCon``.
MAX = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'cap': {'dims': ['t']}},
    'variables': {'p': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'lim': {'foreach': ['t'], 'expression': 'p <= cap'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(p, over=t) + 5'},
}

MIP = {
    'dimensions': {'i': {'dtype': 'int'}, 'one': {'dtype': 'int'}},
    'parameters': {'w': {'dims': ['i']}, 'cap': {'dims': ['one']}},
    'variables': {'x': {'foreach': ['i'], 'domain': 'binary'}},
    'constraints': {'budget': {'foreach': ['one'], 'expression': 'sum(x * w, over=i) <= cap'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x * w, over=i)'},
}

INFEASIBLE = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'load': {'dims': ['t']}},
    'variables': {'p': {'foreach': ['t'], 'bounds': {'lower': 0, 'upper': 1}}},
    'constraints': {'meet': {'foreach': ['t'], 'expression': 'p == load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
}

#: Each case is the ``(model, data)`` pair a call site unpacks:
#: ``lps.solve(*CASES['MIP'])``.
CASES: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    'LP': (
        LP,
        {
            't': [0, 1, 2],
            'load': pl.DataFrame({'t': [0, 1, 2], 'value': [1.0, 2.0, 3.0]}),
            'price': pl.DataFrame({'t': [0, 1, 2], 'value': [10.0, 20.0, 30.0]}),
        },
    ),
    'MAX': (MAX, {'t': [0, 1], 'cap': pl.DataFrame({'t': [0, 1], 'value': [3.0, 4.0]})}),
    'MIP': (
        MIP,
        {
            'i': [0, 1, 2],
            'one': [0],
            'w': pl.DataFrame({'i': [0, 1, 2], 'value': [2.0, 3.0, 4.0]}),
            'cap': pl.DataFrame({'one': [0], 'value': [5.0]}),
        },
    ),
    'INFEASIBLE': (INFEASIBLE, {'t': [0], 'load': pl.DataFrame({'t': [0], 'value': [99.0]})}),
    'QP': (QP, QP_SOURCES),
}


def assert_agrees_with_highs(solver_name: str, case: str, variable: str, constraint: str, *, has_duals: bool) -> None:
    """The claim a second solver has to earn, on all four quantities.

    Coordinates as well as values, since a sink that loaded the columns in a
    different order would still reach the same objective on these models — and
    duals under ``maximize``, where a sign convention could differ and nothing
    else in the suite would notice. Activity is the quantity every member
    reaches through its own door — HiGHS reads its own ``row_value``, the
    others subtract slack from the right-hand side — so agreement is two
    solvers *and* two derivations; it holds on the MIP too, being gated on
    ``has_primal`` alone where duals are not.
    """
    import lpspec as lps

    with lps.solve(*CASES[case]) as highs, lps.solve(*CASES[case], solver_name=solver_name) as other:
        assert other.termination_condition == highs.termination_condition
        assert other.objective == pytest.approx(highs.objective)

        expected, got = highs.primal(variable), other.primal(variable)
        assert got.columns == expected.columns
        assert got.drop('value').equals(expected.drop('value'))
        assert got['value'].to_list() == pytest.approx(expected['value'].to_list())

        assert other.activity(constraint)['value'].to_list() == pytest.approx(
            highs.activity(constraint)['value'].to_list()
        )
        if has_duals:
            assert other.dual(constraint)['value'].to_list() == pytest.approx(highs.dual(constraint)['value'].to_list())


def assert_infeasible_reports_both_axes(solver_name: str) -> None:
    """The status pair, and the solver's own word for it where a user reads it."""
    import lpspec as lps
    from lpspec.errors import NoSolutionError

    with lps.solve(*CASES['INFEASIBLE'], solver_name=solver_name) as solution:
        assert solution.status == 'warning'
        assert solution.termination_condition == 'infeasible'
        assert not solution.has_primal
        assert solution.objective != solution.objective, 'nan, not 0.0'
        with pytest.raises(NoSolutionError, match='INFEASIBLE'):
            solution.primal('p')


#: A knapsack, because nothing above declares a discrete variable. Shared
#: because two modules need a MILP: a rebound one re-solves on a solver still
#: holding the last solve's incumbent, and a solved one leaves no valid basis
#: for a warm start to carry.
ITEMS = [f'item{i}' for i in range(12)]
KNAPSACK = {
    'dimensions': {'item': {'dtype': 'str'}},
    'parameters': {'worth': {'dims': ['item']}, 'weight': {'dims': ['item']}, 'capacity': {'dims': []}},
    'variables': {'take': {'foreach': ['item'], 'domain': 'binary'}},
    'constraints': {'fits': {'foreach': [], 'expression': 'sum(weight * take, over=item) <= capacity'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(take * worth)'},
}


def knapsack_sources(items: list[str] = ITEMS) -> dict[str, pl.DataFrame]:
    return {
        'item': pl.DataFrame({'item': items}),
        'worth': pl.DataFrame({'item': items, 'value': [float(7 * i % 13 + 1) for i in range(len(items))]}),
        'weight': pl.DataFrame({'item': items, 'value': [float(5 * i % 11 + 1) for i in range(len(items))]}),
        'capacity': pl.DataFrame({'value': [20.0]}),
    }


@pytest.fixture(params=sorted(SOLVERS))
def solver_name(request: pytest.FixtureRequest) -> str:
    """Every sink that can stay loaded, skipping one this build cannot run.

    Asked through the sink's own availability rule rather than by naming its
    package here, so a member that grows a second dependency does not also grow
    a second skip.
    """
    if not SOLVERS[request.param].is_available():
        pytest.skip(f'{request.param} is not installed here')
    return str(request.param)
