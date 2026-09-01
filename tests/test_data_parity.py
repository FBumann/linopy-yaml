"""Malformed *data*, checked for the same verdict on both lanes.

`test_resolution_parity.py` does this for the language: the same YAML is
accepted or refused identically, because hard rule 3 says both lanes accept
exactly the same language. Nothing said the same about the same **data**, and
the checks are written twice — 15 `DataError` sites in `relational/engines/polars/engine.py`
against 16 in `linopy/loader.py`, with only the wording of two of them shared
(#351). Two of the cases below diverged when this table was written:

- an unknown label was **accepted** by the relational lane, worth two thirds of
  the objective on the model here (#350);
- a duplicated coordinate row raised `DataError` relationally and a bare
  `ValueError` eagerly — which the error rules name as the failure mode to avoid, "an
  opaque xarray or solver exception with no pointer back to a YAML
  declaration".

A third pair diverged on a hole in a value column, one lane refusing it as an
undefined divisor that was not there while the other read it as a missing row.

The table is the contract the decoupling in #351 has to preserve. It is also
what makes "decoupled" mean something: without it, the next divergence lands
the way these did — silently, and found by accident.

**Each case carries data twice**, once per lane's preferred shape. That is not
duplication for its own sake: the relational lane adapts everything to tidy
polars frames, the eager lane reads pandas/xarray natively because that is what
linopy wants, and the point of the table is that two *representations* of the
same mistake get the same answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest
import yaml as pyyaml

import lpspec as lps
from lpspec.errors import DataError
from tests.oracle import lpspec_linopy, pd  # skips the module without the [linopy] extra

if TYPE_CHECKING:
    from pathlib import Path

#: One dimension, one variable, a coefficient and a bound — the smallest model
#: that has somewhere for each kind of bad data to go wrong.
SPEC = {
    'dimensions': {'f': {'dtype': 'str'}},
    'parameters': {'cost': {'dims': ['f']}, 'cap': {'dims': ['f']}},
    'variables': {'x': {'foreach': ['f'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {'k': {'foreach': ['f'], 'expression': 'x <= cap'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x * cost)'},
}

ACCEPTED = 'accepted'


def _tidy(**cols: list[Any]) -> pl.DataFrame:
    return pl.DataFrame(cols)


def _written(tmp_path: Path, spec: dict) -> Path:
    """*spec* on disk, because the eager lane only takes a path."""
    path = tmp_path / 'model.yaml'
    path.write_text(pyyaml.safe_dump(spec))
    return path


def both_lanes_refuse(path: Path | str, sources: dict, match: str) -> str:
    """Both doors refuse *sources* with one sentence, returned for the cases that pin more of it."""
    with pytest.raises(DataError, match=match) as relational:
        lps.build(path, sources).close()
    with pytest.raises(DataError, match=match) as eager:
        lpspec_linopy.build(path, sources)
    assert str(relational.value) == str(eager.value), 'one defect, one sentence'
    return str(relational.value)


@dataclass(frozen=True)
class Case:
    """One malformed (or valid) attachment, in both representations."""

    label: str
    relational: dict[str, Any]
    eager: dict[str, Any]
    #: `DataError` for a refusal both lanes owe the caller, or `ACCEPTED`.
    verdict: type[Exception] | str


def _cases() -> list[Case]:
    index = {'f': ['a', 'b']}
    good_r = {**index, 'cost': _tidy(f=['a', 'b'], value=[1.0, 2.0]), 'cap': _tidy(f=['a', 'b'], value=[5.0, 5.0])}
    good_e = {**index, 'cost': pd.Series({'a': 1.0, 'b': 2.0}), 'cap': pd.Series({'a': 5.0, 'b': 5.0})}
    return [
        Case('valid', good_r, good_e, ACCEPTED),
        Case(
            'parameter missing entirely',
            {**index, 'cost': good_r['cost']},
            {**index, 'cost': good_e['cost']},
            DataError,
        ),
        Case(
            'bound parameter sparse',
            {**good_r, 'cap': _tidy(f=['a'], value=[5.0])},
            {**good_e, 'cap': pd.Series({'a': 5.0})},
            DataError,  # a missing bound has no reading, so law 8 refuses rather than guessing
        ),
        Case(
            'coefficient sparse',
            {**good_r, 'cost': _tidy(f=['a'], value=[1.0])},
            {**good_e, 'cost': pd.Series({'a': 1.0})},
            # The ordinary case: a missing row is a zero coefficient (the data-attachment rules).
            ACCEPTED,
        ),
        Case(
            'duplicated coordinate row',
            {**good_r, 'cost': _tidy(f=['a', 'a', 'b'], value=[1.0, 9.0, 2.0])},
            {**good_e, 'cost': pd.Series([1.0, 9.0, 2.0], index=pd.Index(['a', 'a', 'b'], name='f'))},
            # Which value applies is undefined, so neither lane may pick one.
            DataError,
        ),
        Case(
            'label the dimension does not have',
            {**good_r, 'cost': _tidy(f=['a', 'zz'], value=[1.0, 2.0])},
            {**good_e, 'cost': pd.Series({'a': 1.0, 'zz': 2.0})},
            # Present and unaddressable is a typo, not sparsity (#350).
            DataError,
        ),
        Case(
            'a null value',
            {**good_r, 'cost': _tidy(f=['a', 'b'], value=[1.0, None])},
            {**good_e, 'cost': pd.Series({'a': 1.0, 'b': None})},
            # A row claiming the coordinate while its value denies it says both at once.
            DataError,
        ),
        Case(
            'a NaN value',
            {**good_r, 'cost': _tidy(f=['a', 'b'], value=[1.0, float('nan')])},
            {**good_e, 'cost': pd.Series({'a': 1.0, 'b': float('nan')})},
            # The same hole, in the only spelling pandas has for one.
            DataError,
        ),
        Case(
            'a hole in a bound',
            {**good_r, 'cap': _tidy(f=['a', 'b'], value=[5.0, None])},
            {**good_e, 'cap': pd.Series({'a': 5.0, 'b': None})},
            # Refused at attach, where `null_bounds_message` caught it at assembly.
            DataError,
        ),
    ]


CASES = _cases()


@pytest.fixture(scope='module')
def spec_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The eager lane only takes a path, so the model has to hit disk once."""
    path = tmp_path_factory.mktemp('data-parity') / 'm.yaml'
    path.write_text(pyyaml.safe_dump(SPEC))
    return path


def _verdict_relational(path: Path, data: dict[str, Any]) -> type[Exception] | str:
    try:
        lps.solve(path, data)
    except DataError:
        return DataError
    return ACCEPTED


def _verdict_eager(path: Path, data: dict[str, Any]) -> type[Exception] | str:
    try:
        m = lpspec_linopy.build(path, data)
        m.solve(solver_name='highs', output_flag=False)
    except DataError:
        return DataError
    return ACCEPTED


@pytest.mark.parametrize('case', CASES, ids=lambda c: c.label)
def test_both_lanes_reach_the_same_verdict(case: Case, spec_path: Path):
    """And it is the verdict the table names, not merely the same one.

    Asserting agreement alone would pass on two lanes that are both wrong,
    which is the failure this table exists to catch rather than to reproduce.
    """
    relational = _verdict_relational(spec_path, case.relational)
    eager = _verdict_eager(spec_path, case.eager)

    assert relational == case.verdict, f'{case.label}: relational lane'
    assert eager == case.verdict, f'{case.label}: eager lane'


def test_the_table_covers_both_verdicts():
    """A guard on the guard: a table that had drifted to all-accepted, or to
    all-refused, would still pass every assertion above.
    """
    verdicts = {c.verdict for c in CASES}
    assert verdicts == {ACCEPTED, DataError}, f'expected both verdicts to be exercised; got {verdicts}'


def test_a_hole_is_named_where_it_sits_rather_than_as_a_divisor(spec_path: Path):
    """`x * cost` divides by nothing, and the message used to say it did.

    The relational lane read a null coefficient in the assembled matrix as an
    undefined divisor, which is the only way one used to arise — so a hole in
    an ordinary coefficient printed `parameter ''`, naming no parameter at all,
    while the eager lane read the same hole as a missing row and solved. Both
    now refuse it at attach, in one sentence, before anything is assembled.
    """
    index = {'f': ['a', 'b']}
    holed = {**index, 'cost': _tidy(f=['a', 'b'], value=[1.0, None]), 'cap': _tidy(f=['a', 'b'], value=[5.0, 5.0])}
    eager = {**index, 'cost': pd.Series({'a': 1.0, 'b': None}), 'cap': pd.Series({'a': 5.0, 'b': 5.0})}

    with pytest.raises(DataError, match="parameter 'cost'") as relational_error:
        lps.build(spec_path, holed).close()
    with pytest.raises(DataError, match="parameter 'cost'") as eager_error:
        lpspec_linopy.build(spec_path, eager)

    assert 'divisor' not in str(relational_error.value), (
        'the message names the hole, not a divisor the model has not got'
    )
    assert "f='b'" in str(relational_error.value), 'and names the coordinate the hole sits at'
    assert str(relational_error.value) == str(eager_error.value), 'one defect, one sentence'


@pytest.mark.parametrize(
    'holed',
    [
        pytest.param(float('nan'), id='a-scalar'),
        pytest.param([1.0, float('nan')], id='a-sequence'),
        pytest.param({'a': 1.0, 'b': None}, id='a-dict'),
        pytest.param(_tidy(f=['a', 'b'], value=[1.0, None]), id='a-tidy-frame'),
    ],
)
def test_a_hole_is_refused_in_every_shape_a_source_arrives_in(spec_path: Path, holed: Any):
    """One source object, both lanes — these four shapes are nobody's dialect.

    Each stops being a list of supplied rows at a different line: a dict and a
    sequence are spread over the master coordinates, a scalar is broadcast, a
    tidy frame is unstacked. The eager lane asks its question at four sites for
    that reason, and a guard no test reaches is a guard that rots.
    """
    sources = {'f': ['a', 'b'], 'cost': holed, 'cap': _tidy(f=['a', 'b'], value=[5.0, 5.0])}

    both_lanes_refuse(spec_path, sources, match='no value')


def test_a_hole_in_a_scalar_parameter_is_refused_on_both_lanes(tmp_path: Path):
    """`dims: []` attaches one value, and one value that is a hole is still a hole.

    A scalar is the shape where reading a hole as a row would be least visible:
    it broadcasts everywhere, so one unsupplied number reaches every
    coordinate. The eager lane takes its own branch for it — one row, no index
    to unstack — which is why the question is asked there separately.
    """
    spec = {
        'dimensions': {'f': {'dtype': 'str'}},
        'parameters': {'rate': {'dims': []}},
        'variables': {'x': {'foreach': ['f'], 'bounds': {'lower': 0, 'upper': 1}}},
        'constraints': {'k': {'foreach': ['f'], 'expression': 'x <= 1'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x * rate)'},
    }
    path = _written(tmp_path, spec)
    sources = {'f': ['a', 'b'], 'rate': _tidy(value=[None])}

    both_lanes_refuse(path, sources, match='no value')


#: A model reading a parameter as a position, which is what made the declared
#: dtype load-bearing before it was checked.
POSITION_SPEC = {
    'dimensions': {'g': {'dtype': 'str'}, 't': {'dtype': 'int'}},
    'parameters': {'lead': {'dims': ['g'], 'dtype': 'int'}, 'demand': {'dims': ['g', 't']}},
    'variables': {'x': {'foreach': ['g', 't'], 'bounds': {'lower': 0}}},
    'constraints': {'c': {'foreach': ['g', 't'], 'expression': 'shift(x, over=t, offset=lead, edge=0) >= demand'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x)'},
}

_DEMAND = _tidy(g=['a', 'a', 'a'], t=[0, 1, 2], value=[1.0, 2.0, 3.0])


def _position_sources(lead: Any) -> dict[str, Any]:
    return {'t': [0, 1, 2], 'g': ['a'], 'lead': _tidy(g=['a'], value=[lead]), 'demand': _DEMAND}


def test_an_int_declaration_takes_no_float_column_so_a_fraction_cannot_arrive(tmp_path: Path):
    """What used to be a value scan is now unrepresentable.

    `by=1.5` once built exactly what `by=1` builds, on both lanes, so the
    differential suite agreed with itself on the wrong model. The repair is not
    a scan for fractions: an `int` declaration takes an integer column, and an
    integer column has no fraction to hold.
    """
    path = _written(tmp_path, POSITION_SPEC)

    both_lanes_refuse(path, _position_sources(1.5), match="declared 'int'")
    with lps.solve(path, _position_sources(1)) as run:
        assert run.is_ok, 'and an integer column is the ordinary case'


def test_whole_numbers_serve_a_float_declaration(tmp_path: Path):
    """The one widening, and the only mismatch the shipped corpus contains.

    `transport_dantzig` and `transport_pwl` declare `capacity` and `demand`
    `float` and supply whole numbers; refusing that would cost two ports a cast
    that protects nothing, since an integer is a number.
    """
    spec = {
        'dimensions': {'g': {'dtype': 'str'}},
        'parameters': {'cost': {'dims': ['g'], 'dtype': 'float'}},
        'variables': {'x': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 1}}},
        'constraints': {'k': {'foreach': [], 'expression': 'sum(x, over=g) <= 9'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(x * cost)'},
    }
    path = _written(tmp_path, spec)
    integral = {'g': ['a', 'b'], 'cost': _tidy(g=['a', 'b'], value=[1, 2])}

    with lps.solve(path, integral) as run:
        assert run.is_ok, 'an integer column serves a float declaration'
    assert lpspec_linopy.build(path, integral) is not None, 'and does so on both lanes'


#: A flag, and the three ways a source may spell one. Only the boolean column
#: satisfies `dtype: bool`, and what the mask means no longer depends on which
#: spelling arrived.
FLAG_SPEC = {
    'dimensions': {'g': {'dtype': 'str'}},
    'parameters': {'active': {'dims': ['g'], 'dtype': 'bool'}},
    'variables': {'x': {'foreach': ['g'], 'where': 'active', 'bounds': {'lower': 0, 'upper': 1}}},
    'constraints': {'k': {'foreach': [], 'expression': 'sum(x, over=g) <= 9'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
}


@pytest.mark.parametrize(
    ('spelling', 'column'),
    [
        pytest.param('boolean', _tidy(g=['a', 'b'], value=[True, False]), id='a-boolean-column'),
        pytest.param('1/0 ints', _tidy(g=['a', 'b'], value=[1, 0]), id='a-1-0-int-column'),
        pytest.param('1.0/0.0 floats', _tidy(g=['a', 'b'], value=[1.0, 0.0]), id='a-1-0-float-column'),
    ],
)
def test_a_flag_masks_by_its_declaration_rather_than_by_its_storage(tmp_path: Path, spelling: str, column: Any):
    """`where: active` used to mask only where the source happened to store booleans.

    A 1/0 column read as "defined", which is true of every row, so the same
    flags in a different spelling built a model with nothing masked out — no
    error, on either lane. Now the declaration decides, and a column that is
    not what it declares does not attach at all.
    """
    path = _written(tmp_path, FLAG_SPEC)
    sources = {'g': ['a', 'b'], 'active': column}

    if spelling == 'boolean':
        with lps.solve(path, sources) as run:
            assert run.objective == pytest.approx(1.0), 'the inactive column is masked away'
        return
    both_lanes_refuse(path, sources, match="declared 'bool'")


def test_a_bare_where_on_a_string_parameter_asks_whether_it_has_a_row(tmp_path: Path):
    """It used to reach polars' `is_finite`, which strings do not have.

    `InvalidOperationError: is_finite operation not supported for dtype str` is
    the opaque exception the error rules exist to prevent, and the declaration
    that answers it was already in the file.
    """
    spec = {
        'dimensions': {'g': {'dtype': 'str'}},
        'parameters': {'fuel': {'dims': ['g'], 'dtype': 'str'}},
        'variables': {'x': {'foreach': ['g'], 'where': 'fuel', 'bounds': {'lower': 0, 'upper': 1}}},
        'constraints': {'k': {'foreach': [], 'expression': 'sum(x, over=g) <= 9'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
    }
    path = _written(tmp_path, spec)
    sources = {'g': ['a', 'b'], 'fuel': _tidy(g=['a'], value=['gas'])}

    with lps.solve(path, sources) as run:
        assert run.objective == pytest.approx(1.0), 'defined is having a row, and only `a` has one'


#: A lookup-carrying dimension: the one index a parameter table cannot stand in
#: for, since it carries the label and never what the label maps to.
LOOKUP_SPEC = {
    'dimensions': {'g': {}, 'b': {'dtype': 'str'}},
    'lookups': {'gen_bus': {'over': 'g', 'into': 'b'}},
    'parameters': {'p_max': {'dims': ['g']}},
    'variables': {'x': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 'p_max'}}},
    'constraints': {'k': {'foreach': ['b'], 'expression': 'sum(x, by=gen_bus) <= 10'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
}

_P_MAX = {'p_max': _tidy(g=['w', 's'], value=[5.0, 5.0])}
_INDEX = {'g': _tidy(g=['w', 's']), 'b': _tidy(b=['n', 'e'])}
_MAP = {'gen_bus': _tidy(g=['w', 's'], b=['n', 'e'])}


@pytest.mark.parametrize(
    ('sources', 'match'),
    [
        pytest.param({**_P_MAX, **_MAP}, 'has its maps', id='a-map-and-no-labels'),
        pytest.param({**_P_MAX, **_INDEX}, 'no data provided for lookup', id='an-index-and-no-map'),
        pytest.param(
            {**_P_MAX, **_INDEX, 'gen_bus': _tidy(g=['w', 's'], gen_bus=['n', 'e'])},
            r"must carry columns \['g', 'b'\]",
            id='a-map-named-after-itself-and-not-its-target',
        ),
        pytest.param(
            {**_P_MAX, **_MAP, 'g': _tidy(gg=['w', 's'])},
            "without a 'g' column",
            id='an-index-without-the-label-column',
        ),
        pytest.param(
            {**_P_MAX, **_INDEX, 'gen_bus': _tidy(g=['w', 's'], b=['n', 'zz'])},
            'not .b. labels',
            id='a-lookup-value-that-is-no-label-of-its-target',
        ),
        pytest.param(
            {**_P_MAX, **_INDEX, 'gen_bus': _tidy(g=['w', 'w', 's'], b=['n', 'e', 'e'])},
            'more than once',
            id='a-lookup-with-two-values-for-one-label',
        ),
        pytest.param(
            {**_P_MAX, **_INDEX, 'gen_bus': _tidy(g=['w', 's'], b=[None, 'e'])},
            "null in 'b'",
            id='a-lookup-mapping-a-label-to-nothing',
        ),
        pytest.param(
            {**_P_MAX, **_MAP, 'g': _tidy(g=['w', 's'], gen_bus=['n', 'e'])},
            "is a lookup over 'g'",
            id='a-map-carried-on-the-index-it-runs-over',
        ),
    ],
)
def test_a_lookup_defect_reads_the_same_on_both_lanes(tmp_path, sources, match):
    """One wording, not two — the same rule `no_index_source_message` follows.

    The first two were written twice and drifted: the relational lane named the
    `sources` key and the eager one a separate argument, for one defect a caller
    fixes the same way whichever lane they were on. The rest were the same
    duplication one function further in, where each lane read the index itself
    — and with a map arriving under its own key they are one check in the door
    both lanes enter, which is what makes the parity structural here rather
    than tested into place.
    """
    path = _written(tmp_path, LOOKUP_SPEC)

    both_lanes_refuse(path, sources, match=match)


def test_an_index_a_declared_map_is_read_against_is_checked_before_the_read(tmp_path):
    """The one shape where the front door, not the engine, owes the sentence.

    A map the file declares is read against the caller's labels while the
    sources are adapted, which is upstream of every attacher — so an index
    without its label column reached polars there and came back as
    `ColumnNotFoundError: unable to find column "g"`, the opaque exception the
    error rules exist to prevent, on a lane whose attacher has the right sentence
    for it two calls later.
    """
    spec = {**LOOKUP_SPEC, 'lookups': {'gen_bus': {'over': 'g', 'into': 'b'}}}
    path = _written(tmp_path, spec)
    sources = {**_P_MAX, **_MAP, 'b': _tidy(b=['n', 'e']), 'g': _tidy(gg=['w', 's'])}

    both_lanes_refuse(path, sources, match="without a 'g' column")


def test_a_lookup_a_label_holds_twice_is_refused_before_it_can_drop_a_row(tmp_path):
    """Counting nulls is what makes the refusal reach the case that costs an answer.

    pandas `nunique()` skips nulls where polars `n_unique()` counts them, so a
    label carrying a null in one row and a real value in another read as
    single-valued on the eager lane — and the null won, that row being the
    first. The member then belonged to no group, its terms left the constraint
    that was to hold them, and the model solved: 8.0 against the 3.0 both lanes
    give the same index deduplicated.
    """
    spec = {
        **LOOKUP_SPEC,
        'dimensions': {'g': {}, 'b': {'dtype': 'str'}},
        'constraints': {'k': {'foreach': ['b'], 'expression': 'sum(x, by=gen_bus) <= 3'}},
    }
    path = _written(tmp_path, spec)
    clean = {**_P_MAX, **_INDEX, 'gen_bus': _tidy(g=['w', 's'], b=['n', 'n'])}
    holed = {**_P_MAX, **_INDEX, 'gen_bus': _tidy(g=['w', 'w', 's'], b=[None, 'n', 'n'])}

    with lps.solve(path, clean) as run:
        assert run.objective == pytest.approx(3.0), 'both members are on the bus, and the bus caps them'
    built = lpspec_linopy.build(path, clean)
    built.solve(solver_name='highs', output_flag=False)
    assert float(built.objective.value) == pytest.approx(3.0), 'and the eager lane agrees where the index is clean'

    both_lanes_refuse(path, holed, match="null in 'b'")


def test_a_dimension_index_is_a_table_on_both_lanes(tmp_path):
    """And it may arrive under `sources`, which is where the relational lane looks first.

    The eager lane took its own argument and required a pandas frame, so an index
    a caller passed the way the runner documents — a polars table under the
    dimension's own key — was invisible to one of two lanes.
    """
    path = _written(tmp_path, LOOKUP_SPEC)
    sources = {**_P_MAX, **_INDEX, **_MAP}

    with lps.solve(path, sources) as relational:
        assert relational.is_ok
    built = lpspec_linopy.build(path, sources)
    assert set(built.variables['x'].coords['g'].to_numpy()) == {'w', 's'}, 'the eager lane read the same index'


def test_a_dimension_index_may_be_a_parquet_path_without_pyarrow(tmp_path, monkeypatch):
    """The `[linopy]` extra ships pandas and xarray, and nothing says it ships pyarrow.

    The eager lane read an index path with `polars.read_parquet().to_pandas()`,
    which wants pyarrow for anything Arrow-backed — so the way the runner
    documents passing an index, a path under the dimension's own key, raised
    `ModuleNotFoundError: No module named 'pyarrow'` on a supported install.
    Blocked here rather than assumed absent, the extra's own resolution being
    free to bring it in.
    """
    import sys

    path = _written(tmp_path, LOOKUP_SPEC)
    index = tmp_path / 'g.parquet'
    _tidy(g=['w', 's']).write_parquet(index)
    sources = {**_P_MAX, **_MAP, 'b': _tidy(b=['n', 'e']), 'g': str(index)}

    monkeypatch.setitem(sys.modules, 'pyarrow', None)
    with lps.solve(path, sources) as relational:
        assert relational.is_ok
    built = lpspec_linopy.build(path, sources)
    assert set(built.variables['x'].coords['g'].to_numpy()) == {'w', 's'}, 'the eager lane read the same path'


#: The same shape one column over: a lookup whose *target* is the temporal
#: dimension, rather than the dimension the index is of.
TEMPORAL_LOOKUP_SPEC = {
    'dimensions': {'g': {}, 'd': {'dtype': 'datetime'}},
    'lookups': {'day_of': {'over': 'g', 'into': 'd'}},
    'parameters': {'p_max': {'dims': ['g']}, 'cap': {'dims': ['d']}},
    'variables': {'x': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 'p_max'}}},
    'constraints': {'k': {'foreach': ['d'], 'expression': 'sum(x, by=day_of) <= cap'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x, over=g)'},
}


@pytest.mark.parametrize('library', ['pandas', 'polars', 'a parquet path'])
def test_a_lookup_into_a_temporal_dimension_is_one_instant_on_both_lanes(tmp_path, library):
    """A lookup's values are labels of the dimension it targets, in that spelling.

    The label column is canonicalised against the declared `dtype: datetime`
    because a `datetime.date` and a `datetime64` are one instant that compares
    unequal — and a targeted lookup's *values* are labels of a dimension too,
    so the same instant reaches the same containment check by the other route.
    Read without it, whichever library the caller brought the index in decided
    whether every value looked like a stranger: refused eagerly through pandas
    and accepted through polars, for one index saying one thing.

    Both members map to the same day, so the cap that day carries binds them
    together — a group that quietly lost a member would leave `p_max` the only
    ceiling and answer 10.0.

    A pandas frame of `datetime.date` and a polars `pl.Date` are the two
    spellings a caller writes by hand. A nanosecond one is a third and is not
    settled here: `datetime[ns]` reaches the relational lane's own join as a
    key it will not match, and out of pandas both lanes refuse it alike — see
    the follow-ups on #1076.
    """
    import datetime

    days = [datetime.date(2030, 1, 1), datetime.date(2030, 1, 2)]
    path = _written(tmp_path, TEMPORAL_LOOKUP_SPEC)
    day_of = _tidy(g=['w', 's'], d=[days[0], days[0]])
    if library == 'a parquet path':
        day_of.write_parquet(tmp_path / 'day_of.parquet')
    sources = {
        **_P_MAX,
        'cap': _tidy(d=days, value=[3.0, 7.0]),
        'd': days,
        'g': ['w', 's'],
        'day_of': {
            'pandas': lambda: pd.DataFrame({'g': ['w', 's'], 'd': [days[0], days[0]]}),
            'polars': lambda: day_of,
            'a parquet path': lambda: str(tmp_path / 'day_of.parquet'),
        }[library](),
    }

    with lps.solve(path, sources) as run:
        assert run.objective == pytest.approx(3.0), 'one day, one cap, both members under it'
    built = lpspec_linopy.build(path, sources)
    built.solve(solver_name='highs', output_flag=False)
    assert float(built.objective.value) == pytest.approx(3.0), 'and the eager lane groups them the same way'


def test_a_stray_lookup_value_reads_the_same_over_an_int_labelled_target(tmp_path):
    """One sentence, and the labels in it spelled as the caller wrote them.

    The eager lane took its offending values off a pandas frame and printed
    them as they came, so an `int` dimension read back `np.int64(99)` where the
    relational lane said `99` — one defect, two sentences again, and invisible
    to a table whose every label is a string.
    """
    spec = {
        **LOOKUP_SPEC,
        'dimensions': {'g': {}, 'b': {'dtype': 'int'}},
    }
    path = _written(tmp_path, spec)
    sources = {**_P_MAX, **_INDEX, 'gen_bus': _tidy(g=['w', 's'], b=[1, 99])}

    sentence = both_lanes_refuse(path, sources, match=r'not .b. labels')
    assert '99.' in sentence, 'the label as the caller wrote it, not as numpy holds it'


def test_a_multi_indexed_series_is_refused_on_both_lanes(tmp_path):
    """The one pandas shape with no counterpart in the frames underneath.

    An index carries the dims a parameter is over *and* how many of them there
    are, and the second is a claim the declaration already makes — so the two
    can disagree with nothing able to say which was meant. A depth check would
    catch the disagreement; refusing the shape removes it, and a tidy frame
    says the same thing in the vocabulary the other five accepted shapes use.

    Refused at `tidy_sources`, which is the one door both lanes enter by, so
    neither can drift a second wording for it.
    """
    path = _written(tmp_path, LOOKUP_SPEC)
    deep = pd.MultiIndex.from_tuples([('w', 0), ('s', 0)], names=['g', 'k'])
    sources = {'p_max': pd.Series([5.0, 5.0], index=deep), **_INDEX, **_MAP}

    sentence = both_lanes_refuse(path, sources, match='MultiIndex is not a source')
    assert "['g', 'value']" in sentence, 'and it names the tidy frame the caller should pass'


def test_a_series_shallower_than_the_declared_dims_is_refused_on_both_lanes(tmp_path):
    """The other half of the same disagreement, once a MultiIndex cannot arrive.

    A Series is one index level deep, so it says the parameter has one dim. A
    declaration of any other arity contradicts it, and the two shapes that are
    also one level deep — a dict and a sequence — have said so all along.
    `_series_to_frame` answered by handing pandas the declared dims as names for
    that one level, which raised `ValueError: Length of new names must be 1, got
    2` — naming neither the parameter nor the repair, on the lane whose sibling
    shapes had the sentence to hand.

    The index is deliberately *unnamed*: a named one attaches by its own name and
    goes wrong further downstream, where the missing column is what gets
    reported.
    """
    spec = {
        'dimensions': {'g': {'dtype': 'str'}, 'b': {'dtype': 'str'}},
        'parameters': {'p_max': {'dims': ['g', 'b']}},
        'variables': {'x': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 1}}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
    }
    path = _written(tmp_path, spec)
    sources = {'p_max': pd.Series([5.0, 5.0], index=pd.Index(['w', 's']))}

    sentence = both_lanes_refuse(path, sources, match='runs along one dimension')
    assert "['g', 'b', 'value']" in sentence, 'and it names the table that carries both dims'


def test_a_source_key_the_model_does_not_declare_is_refused_on_both_lanes(tmp_path):
    """Ignoring it is a silent fallback, which is the one thing we do not do.

    `update` settled this first — a name it does not recognise is a typo, and
    ignoring one there is a silent re-solve — and attaching owes the same answer.
    It is also what catches a misspelled *index* key, whose labels would
    otherwise fall back to derivation and change only their order.

    The cost is that a driver attaching one bag of data to several models says
    which slice each takes; `examples/benders/run.py` is that, in one line.
    """
    path = _written(tmp_path, SPEC)
    good = {'cost': _tidy(f=['a', 'b'], value=[1.0, 2.0]), 'cap': _tidy(f=['a', 'b'], value=[5.0, 5.0])}
    typo = {**good, 'csot': good['cost']}

    both_lanes_refuse(path, typo, match="Did you mean 'cost'")


def test_an_entity_table_is_a_dimension_index_columns_and_all(tmp_path):
    """Why an undeclared *column* is ignored where an undeclared *key* is not.

    The columns a table must carry are exact and total — every dim, plus
    `value` — so a misspelled one is a missing one and is refused. Nothing can
    hide in the extras, and the extras are the point: a framework hands over
    `generators` with its index and its attributes in one table.

    One column is not an extra. A column named after a lookup over the
    dimension is a map somebody meant to supply, and dropping it silently would
    build the model they did not write — so it is the one stray that is
    refused, naming the key it belongs under.
    """
    spec = {
        'dimensions': {'g': {}, 'b': {'dtype': 'str'}},
        'lookups': {'gen_bus': {'over': 'g', 'into': 'b'}},
        'parameters': {'cap': {'dims': ['g']}},
        'variables': {'x': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'constraints': {'k': {'foreach': ['b'], 'expression': 'sum(x, by=gen_bus) <= 100'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
    }
    path = _written(tmp_path, spec)
    generators = _tidy(g=['w', 's'], cap=[10.0, 20.0], note=['a', 'b'])
    sources = {
        'g': generators,
        'b': _tidy(b=['n', 'e']),
        'gen_bus': _tidy(g=['w', 's'], b=['n', 'e']),
        'cap': _tidy(g=['w', 's'], value=[10.0, 20.0]),
    }

    with lps.solve(path, sources) as result:
        assert result.objective == pytest.approx(30.0)

    with pytest.raises(DataError, match=r"missing columns \['g'\]"):
        lps.build(path, {**sources, 'cap': _tidy(gg=['w', 's'], value=[10.0, 20.0])}).close()

    carried = _tidy(g=['w', 's'], cap=[10.0, 20.0], gen_bus=['n', 'e'])
    with pytest.raises(DataError, match=r"index for dimension 'g' carries a 'gen_bus' column"):
        lps.build(path, {**sources, 'g': carried}).close()
