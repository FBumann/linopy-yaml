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
MODEL = {
    'dimensions': {'f': {'values': ['a', 'b']}},
    'parameters': {'cost': {'dims': ['f']}, 'cap': {'dims': ['f']}},
    'variables': {'x': {'foreach': ['f'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {'k': {'foreach': ['f'], 'expression': 'x <= cap'}},
    'objective': {'sense': 'maximize', 'expression': 'x * cost'},
}

ACCEPTED = 'accepted'


def _tidy(**cols: list[Any]) -> pl.DataFrame:
    return pl.DataFrame(cols)


@dataclass(frozen=True)
class Case:
    """One malformed (or valid) binding, in both representations."""

    label: str
    relational: dict[str, Any]
    eager: dict[str, Any]
    #: `DataError` for a refusal both lanes owe the caller, or `ACCEPTED`.
    verdict: type[Exception] | str


def _cases() -> list[Case]:
    good_r = {'cost': _tidy(f=['a', 'b'], value=[1.0, 2.0]), 'cap': _tidy(f=['a', 'b'], value=[5.0, 5.0])}
    good_e = {'cost': pd.Series({'a': 1.0, 'b': 2.0}), 'cap': pd.Series({'a': 5.0, 'b': 5.0})}
    return [
        Case('valid', good_r, good_e, ACCEPTED),
        Case(
            'parameter missing entirely',
            {'cost': good_r['cost']},
            {'cost': good_e['cost']},
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
            # The ordinary case: a missing row is a zero coefficient (the data-binding rules).
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
            # Refused at bind, where `null_bounds_message` caught it at assembly.
            DataError,
        ),
    ]


CASES = _cases()


@pytest.fixture(scope='module')
def model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The eager lane only takes a path, so the model has to hit disk once."""
    path = tmp_path_factory.mktemp('data-parity') / 'm.yaml'
    path.write_text(pyyaml.safe_dump(MODEL))
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
def test_both_lanes_reach_the_same_verdict(case: Case, model_path: Path):
    """And it is the verdict the table names, not merely the same one.

    Asserting agreement alone would pass on two lanes that are both wrong,
    which is the failure this table exists to catch rather than to reproduce.
    """
    relational = _verdict_relational(model_path, case.relational)
    eager = _verdict_eager(model_path, case.eager)

    assert relational == case.verdict, f'{case.label}: relational lane'
    assert eager == case.verdict, f'{case.label}: eager lane'


def test_the_table_covers_both_verdicts():
    """A guard on the guard: a table that had drifted to all-accepted, or to
    all-refused, would still pass every assertion above.
    """
    verdicts = {c.verdict for c in CASES}
    assert verdicts == {ACCEPTED, DataError}, f'expected both verdicts to be exercised; got {verdicts}'


def test_a_hole_is_named_where_it_sits_rather_than_as_a_divisor(model_path: Path):
    """`x * cost` divides by nothing, and the message used to say it did.

    The relational lane read a null coefficient in the assembled matrix as an
    undefined divisor, which is the only way one used to arise — so a hole in
    an ordinary coefficient printed `parameter ''`, naming no parameter at all,
    while the eager lane read the same hole as a missing row and solved. Both
    now refuse it at bind, in one sentence, before anything is assembled.
    """
    holed = {'cost': _tidy(f=['a', 'b'], value=[1.0, None]), 'cap': _tidy(f=['a', 'b'], value=[5.0, 5.0])}
    eager = {'cost': pd.Series({'a': 1.0, 'b': None}), 'cap': pd.Series({'a': 5.0, 'b': 5.0})}

    with pytest.raises(DataError, match="parameter 'cost'") as relational_error:
        lps.build(model_path, holed).close()
    with pytest.raises(DataError, match="parameter 'cost'") as eager_error:
        lpspec_linopy.build(model_path, eager)

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
def test_a_hole_is_refused_in_every_shape_a_source_arrives_in(model_path: Path, holed: Any):
    """One source object, both lanes — these four shapes are nobody's dialect.

    Each stops being a list of supplied rows at a different line: a dict and a
    sequence are spread over the master coordinates, a scalar is broadcast, a
    tidy frame is unstacked. The eager lane asks its question at four sites for
    that reason, and a guard no test reaches is a guard that rots.
    """
    sources = {'cost': holed, 'cap': _tidy(f=['a', 'b'], value=[5.0, 5.0])}

    with pytest.raises(DataError, match='no value'):
        lps.build(model_path, sources).close()
    with pytest.raises(DataError, match='no value'):
        lpspec_linopy.build(model_path, sources)


def test_a_hole_in_a_scalar_parameter_is_refused_on_both_lanes(tmp_path: Path):
    """`dims: []` binds one value, and one value that is a hole is still a hole.

    A scalar is the shape where reading a hole as a row would be least visible:
    it broadcasts everywhere, so one unsupplied number reaches every
    coordinate. The eager lane takes its own branch for it — one row, no index
    to unstack — which is why the question is asked there separately.
    """
    model = {
        'dimensions': {'f': {'values': ['a', 'b']}},
        'parameters': {'rate': {'dims': []}},
        'variables': {'x': {'foreach': ['f'], 'bounds': {'lower': 0, 'upper': 1}}},
        'constraints': {'k': {'foreach': ['f'], 'expression': 'x <= 1'}},
        'objective': {'sense': 'maximize', 'expression': 'x * rate'},
    }
    path = tmp_path / 'scalar.yaml'
    path.write_text(pyyaml.safe_dump(model))
    sources = {'rate': _tidy(value=[None])}

    with pytest.raises(DataError, match='no value'):
        lps.build(path, sources).close()
    with pytest.raises(DataError, match='no value'):
        lpspec_linopy.build(path, sources)


#: The two places the language reads a parameter's *values* as positions rather
#: than as arithmetic. Both take `dtype: int` at load, and neither looked at
#: what the column actually carried.
POSITION_MODELS = {
    'shift(offset=lead)': {
        'dimensions': {'g': {'values': ['a']}, 't': {'dtype': 'int'}},
        'parameters': {'lead': {'dims': ['g'], 'dtype': 'int'}, 'demand': {'dims': ['g', 't']}},
        'variables': {'x': {'foreach': ['g', 't'], 'bounds': {'lower': 0}}},
        'constraints': {'c': {'foreach': ['g', 't'], 'expression': 'shift(x, over=t, offset=lead, edge=0) >= demand'}},
        'objective': {'sense': 'minimize', 'expression': 'x'},
    },
    'sum_back(within=lead)': {
        'dimensions': {'g': {'values': ['a']}, 't': {'dtype': 'int'}},
        'parameters': {'lead': {'dims': ['g'], 'dtype': 'int'}, 'demand': {'dims': ['g', 't']}},
        'variables': {'x': {'foreach': ['g', 't'], 'bounds': {'lower': 0}}},
        'constraints': {'c': {'foreach': ['g', 't'], 'expression': 'sum_back(x, over=t, within=lead) >= demand'}},
        'objective': {'sense': 'minimize', 'expression': 'x'},
    },
}

_DEMAND = _tidy(g=['a', 'a', 'a'], t=[0, 1, 2], value=[1.0, 2.0, 3.0])


def _position_sources(lead: float) -> dict[str, Any]:
    return {'t': [0, 1, 2], 'lead': _tidy(g=['a'], value=[lead]), 'demand': _DEMAND}


@pytest.mark.parametrize('spelling', sorted(POSITION_MODELS), ids=lambda s: s.split('(')[0])
@pytest.mark.parametrize('lead', [1.5, 2.6], ids=['rounds-down', 'rounds-down-further'])
def test_a_fractional_position_is_refused_on_both_lanes(tmp_path: Path, spelling: str, lead: float):
    """It used to truncate, on both lanes, to the same wrong model.

    `by=1.5` built exactly what `by=1` builds and `by=2.6` what `by=2` does, so
    the differential suite compared two lanes that agreed — the one failure
    mode a second implementation cannot catch. The declaration promising
    `dtype: int` is checked at load; that the column kept the promise is a
    question only data can answer.
    """
    path = tmp_path / 'position.yaml'
    path.write_text(pyyaml.safe_dump(POSITION_MODELS[spelling]))

    with pytest.raises(DataError, match='not whole numbers') as relational:
        lps.build(path, _position_sources(lead)).close()
    with pytest.raises(DataError, match='not whole numbers') as eager:
        lpspec_linopy.build(path, _position_sources(lead))

    assert spelling in str(relational.value), 'the message quotes the call that reads it'
    assert str(relational.value) == str(eager.value), 'one defect, one sentence'


@pytest.mark.parametrize('spelling', sorted(POSITION_MODELS), ids=lambda s: s.split('(')[0])
def test_a_whole_number_in_a_float_column_is_still_a_position(tmp_path: Path, spelling: str):
    """`2.0` lands on a coordinate exactly as `2` does, and pandas has no other dtype for it.

    The guard is against a value with something after the point, not against
    the column's storage — a float offset column is what a pandas-shaped
    source gives you for a whole number, and refusing it would refuse the
    ordinary case.
    """
    path = tmp_path / 'position.yaml'
    path.write_text(pyyaml.safe_dump(POSITION_MODELS[spelling]))

    with lps.solve(path, _position_sources(2.0)) as floating, lps.solve(path, _position_sources(2)) as integral:
        assert floating.objective == integral.objective, 'a whole number is a whole number'


#: A lookup-carrying dimension: the one index a parameter table cannot stand in
#: for, since it carries the label and never what the label maps to.
LOOKUP_MODEL = {
    'dimensions': {'g': {}, 'b': {'values': ['n', 'e']}},
    'lookups': {'gen_bus': {'over': 'g', 'into': 'b'}},
    'parameters': {'p_max': {'dims': ['g']}},
    'variables': {'x': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 'p_max'}}},
    'constraints': {'k': {'foreach': ['b'], 'expression': 'sum(x, by=gen_bus) <= 10'}},
    'objective': {'sense': 'maximize', 'expression': 'x'},
}

_P_MAX = {'p_max': _tidy(g=['w', 's'], value=[5.0, 5.0])}


@pytest.mark.parametrize(
    ('sources', 'match'),
    [
        pytest.param(_P_MAX, 'carries lookups', id='no-index-at-all'),
        pytest.param(
            {**_P_MAX, 'g': _tidy(g=['w', 's'])},
            'missing declared lookup column',
            id='an-index-without-the-lookup-column',
        ),
    ],
)
def test_a_lookup_index_defect_reads_the_same_on_both_lanes(tmp_path, sources, match):
    """One wording, not two — the same rule `no_index_source_message` follows.

    These two were written twice and drifted: the relational lane named the
    `sources` key and the eager one a separate argument, for one defect a caller
    fixes the same way whichever lane they were on.
    """
    path = tmp_path / 'lookup.yaml'
    path.write_text(pyyaml.safe_dump(LOOKUP_MODEL))

    with pytest.raises(DataError, match=match) as relational:
        lps.build(path, sources).close()
    with pytest.raises(DataError, match=match) as eager:
        lpspec_linopy.build(path, sources)

    assert str(relational.value) == str(eager.value), 'one defect, one sentence'


def test_a_dimension_index_is_a_table_on_both_lanes(tmp_path):
    """And it may arrive under `sources`, which is where the relational lane looks first.

    The eager lane took its own argument and required a pandas frame, so an index
    a caller passed the way the runner documents — a polars table under the
    dimension's own key — was invisible to one of two lanes.
    """
    path = tmp_path / 'lookup.yaml'
    path.write_text(pyyaml.safe_dump(LOOKUP_MODEL))
    sources = {**_P_MAX, 'g': _tidy(g=['w', 's'], gen_bus=['n', 'e'])}

    with lps.solve(path, sources) as relational:
        assert relational.is_ok
    built = lpspec_linopy.build(path, sources)
    assert set(built.variables['x'].coords['g'].to_numpy()) == {'w', 's'}, 'the eager lane read the same index'


def test_a_source_key_the_model_does_not_declare_is_refused_on_both_lanes(tmp_path):
    """Ignoring it is a silent fallback, which is the one thing we do not do.

    `rebind` settled this first — a name it does not recognise is a typo, and
    ignoring one there is a silent re-solve — and binding owes the same answer.
    It is also what catches a misspelled *index* key, whose labels would
    otherwise fall back to derivation and change only their order.

    The cost is that a driver binding one bag of data to several models says
    which slice each takes; `examples/benders/run.py` is that, in one line.
    """
    path = tmp_path / 'extra.yaml'
    path.write_text(pyyaml.safe_dump(MODEL))
    good = {'cost': _tidy(f=['a', 'b'], value=[1.0, 2.0]), 'cap': _tidy(f=['a', 'b'], value=[5.0, 5.0])}
    typo = {**good, 'csot': good['cost']}

    with pytest.raises(DataError, match="Did you mean 'cost'") as relational:
        lps.build(path, typo).close()
    with pytest.raises(DataError, match="Did you mean 'cost'") as eager:
        lpspec_linopy.build(path, typo)

    assert str(relational.value) == str(eager.value), 'one defect, one sentence'


def test_an_entity_table_is_a_dimension_index_columns_and_all(tmp_path):
    """Why an undeclared *column* is ignored where an undeclared *key* is not.

    The columns a table must carry are exact and total — every dim, plus
    `value` — so a misspelled one is a missing one and is refused. Nothing can
    hide in the extras, and the extras are the point: a framework hands over
    `generators` with its index, its lookups and its attributes in one table.
    """
    model = {
        'dimensions': {'g': {}, 'b': {'values': ['n', 'e']}},
        'lookups': {'gen_bus': {'over': 'g', 'into': 'b'}},
        'parameters': {'cap': {'dims': ['g']}},
        'variables': {'x': {'foreach': ['g'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'constraints': {'k': {'foreach': ['b'], 'expression': 'sum(x, by=gen_bus) <= 100'}},
        'objective': {'sense': 'maximize', 'expression': 'x'},
    }
    path = tmp_path / 'entity.yaml'
    path.write_text(pyyaml.safe_dump(model))
    generators = _tidy(g=['w', 's'], gen_bus=['n', 'e'], cap=[10.0, 20.0], note=['a', 'b'])
    sources = {'g': generators, 'cap': _tidy(g=['w', 's'], value=[10.0, 20.0])}

    with lps.solve(path, sources) as result:
        assert result.objective == pytest.approx(30.0)

    with pytest.raises(DataError, match=r"missing columns \['g'\]"):
        lps.build(path, {**sources, 'cap': _tidy(gg=['w', 's'], value=[10.0, 20.0])}).close()
