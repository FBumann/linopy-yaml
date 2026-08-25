"""Eager evaluation of the language's built-in operators, on xarray and linopy.

Each entry point takes an operand that is already a value — an ``xr.DataArray``
for a parameter, a linopy ``Variable`` or ``LinearExpression`` for anything
carrying one — and returns the same. Nothing here reads the schema, the AST or
the model: ``builder.py`` evaluates the operands and the keywords, and calls in.

The operator *names* are the language (``math_spec``); these evaluations are
this lane's private business, mirrored on the relational side by lowering cases
rather than by shared code. :data:`OPERATORS` must name exactly the language's
set, which ``tests/test_architecture.py`` checks — a name one lane implements
and the other does not is precisely the divergence that would make the
differential tests a comparison of dialects.

Each entry point comes first and its own machinery after.
"""

from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from math_spec import EDGE_WRAP

from lpspec.errors import LanguageError
from lpspec.linopy import absence

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping

    import pandas as pd


def _operator_sum(array: Any, *, over: str | None = None) -> Any:
    """Sum *array* over dimension *over*, or over all of them where none is named.

    A DataArray and a linopy expression both carry ``dims`` and both take the
    dim positionally, so there is one branch: if the array does not have the
    named dimension, it is returned unchanged.
    """
    if over is None:
        return array.sum()
    if over in getattr(array, 'dims', ()):
        return array.sum(over)
    return array


def operator_grouped_sum(
    array: Any, mappings: tuple[Any, ...], *, into: tuple[str, ...], labels: Mapping[str, pd.Index]
) -> Any:
    """Sum *array* through declared lookups, producing dimensions *into*.

    YAML: ``sum(p, by=gen_bus)`` or ``sum(p, by=[gen_bus, gen_tech])``.
    *mappings* are the lookups' values as one-dimensional arrays over the dim
    being grouped, from ``EvaluationContext.dim_coords``; that dim is summed
    out and *into* holds the group labels, one dim per lookup.

    A null lookup value says the label belongs to no group, so its terms
    contribute nowhere. linopy refuses to group by NaN at all, so those members
    are dropped before grouping rather than after — and with several lookups a
    member missing *any* of them belongs to no group at all.

    Several groups go through the coordinate-name grouper rather than the
    single-array one, because that is the form whose result is one dim per key
    rather than one stacked ``group`` dim.

    *labels* holds each target's declared index, and the result is reindexed
    onto them: a groupby yields only the labels some member actually points at,
    in xarray's sort order, so without this a label no member reaches is
    missing and a declared order that is not sorted is lost. Either one makes
    linopy v1 refuse the next combination with a coordinate mismatch, since it
    aligns on membership *and* order. Lookup values are validated against their
    target's labels when they are loaded, so this only ever adds a label, never
    drops a term.
    """
    mappings = _checked_mappings('sum(by=)', mappings, into)
    present = _present(mappings)
    dim = str(mappings[0].dims[0])
    if not bool(present.all()):
        mask = present.to_numpy()
        mappings = tuple(m.isel({dim: mask}) for m in mappings)
        array = array.isel({dim: mask})
    if not (isinstance(array, xr.DataArray) or hasattr(array, 'groupby')):
        raise _unsupported('sum(by=)', array)
    attached = array.assign_coords({target: (dim, m.to_numpy()) for target, m in zip(into, mappings, strict=True)})
    summed = attached.groupby(list(into)).sum()
    return _reindexed(summed, into=into, labels=labels)


def operator_at(array: Any, mappings: tuple[Any, ...], *, into: tuple[str, ...]) -> Any:
    """Read *array* through declared lookups — the adjoint of a group.

    YAML: ``at(on, by=component)``. *mappings* are the same one-dimensional
    arrays ``sum`` takes; grouping sums *along* them, this indexes *through*
    them, so the operand must carry every dim in ``into`` and the result
    carries the mappings' own dim.

    xarray's vectorised selection is the pullback exactly — one ``into`` label
    read once per fine label pointing at it, and with several lookups one
    *tuple* of labels read once — so the fan-out is the indexer's doing rather
    than a broadcast arranged here.

    A null lookup value reads nothing and its row is absent, the same reading
    ``sum`` gives a null group. It cannot be selected — there is no ``into``
    label to read — so it is dropped from the indexer and the result is put
    back over the whole dim, the missing positions filled with the operand's
    own **absence** rather than a zero — which is what ``reindex`` does untold,
    on all three operand shapes. Absence is what the absence rules ask for
    here: it propagates and takes the row with it, the same mechanism the
    default ``shift`` edge relies on, where a zero would leave a row asserting
    ``x <= 0`` at a coordinate the model said nothing about. Putting the dim
    back is also what keeps the operand combinable at all — linopy v1 aligns on
    membership, so a result short of a label refuses the next arithmetic
    outright, which is how #897 surfaced.
    """
    mappings = _checked_mappings('at()', mappings, into)
    if not (isinstance(array, xr.DataArray) or hasattr(array, 'sel')):
        raise _unsupported('at()', array)

    present = _present(mappings)
    if bool(present.all()):
        return array.sel(dict(zip(into, mappings, strict=True)))

    dim = str(mappings[0].dims[0])
    kept = present.to_numpy()
    picked = array.sel(dict(zip(into, (m.isel({dim: kept}) for m in mappings), strict=True)))
    return picked.reindex({dim: mappings[0][dim]})


def _operator_shift(array: Any, *, over: str, offset: float, edge: str | float | None = None, by: Any = None) -> Any:
    """Translate *array* along one dimension — the value at *t - offset*.

    YAML: ``shift(soc, over=snapshot, offset=1)``. ``edge`` carries all three
    policies so no two keywords can disagree: ``edge='wrap'`` is cyclic and
    vacates nothing, a number is what the vacated positions contribute, and
    omitting it leaves them **absent**, which propagates and drops the row.
    Nothing is done to the result in that default case — linopy v1 already
    gives that answer (#289).

    A DataArray shift always fills, absence not being representable in data, so
    lowering refuses a bare shift over a variable-free operand and that branch
    is only reached under a numeric ``edge=``.

    ``offset`` arrives as an array where the model named a parameter — an offset
    that differs per entity, which is a gather rather than a shift and is
    :func:`_gather_by_offset`.
    """
    if by is not None:
        return _gather_in_groups(
            array,
            over,
            _per_group(offset, by),
            groups=by,
            wrap=edge == EDGE_WRAP,
            fill=None if isinstance(edge, str) else edge,
        )
    if isinstance(offset, xr.DataArray) and offset.ndim:
        return _gather_by_offset(
            array,
            over,
            offset,
            wrap=edge == EDGE_WRAP,
            fill=None if isinstance(edge, str) else edge,
            card=int(array.sizes[over]),
        )
    amount = _translation(over, offset)
    if edge == EDGE_WRAP:
        if isinstance(array, xr.DataArray):
            return array.roll(amount, roll_coords=False)
        if hasattr(array, 'roll'):
            return array.roll(amount)
        raise _unsupported("shift(edge='wrap')", array)
    if isinstance(edge, str):
        msg = f'shift(edge={edge!r}) reached the evaluator: only {EDGE_WRAP!r} or a number resolve.'
        raise AssertionError(msg)
    fill = edge
    if isinstance(array, xr.DataArray):
        return array.shift(amount, fill_value=fill if fill is not None else np.nan)
    if hasattr(array, 'shift'):
        shifted = array.shift(amount)
        return (
            shifted if fill is None else absence.vacated(shifted, array, over, _off_the_axis(array, over, offset), fill)
        )
    raise _unsupported('shift()', array)


def _operator_sum_back(array: Any, *, over: str, within: Any, edge: str | None = None, by: Any = None) -> Any:
    """Sum *array* over a trailing window along one dimension.

    YAML: ``sum_back(started, over=snapshot, within=min_up)``. The result at
    *t* is the sum from *t - within + 1* through *t*, so a width of 1 is the
    operand itself and ``edge='wrap'`` lets the window reach around the axis.

    Written as a sum of scalar gathers, one per position of the widest window
    the data asks for. That bound is read from data, which is only sound
    because it decides how many *terms* are added rather than what the plan
    does — the same reading under which cardinality is data's.

    A position the window cannot reach contributes a **zero**, never an
    absence. linopy counts absence among the things that propagate, so an
    unreachable lag added to a reachable one would annihilate the whole row
    (v1 §4) — which is right for a shift, whose vacated slot really is
    unknown, and wrong here: a window at the first position is short, not
    empty. That covers a masked slot the window reaches too: absence is not a
    term, and a reduction is where absence stops (the absence reference).

    Which leaves the window that reaches **nothing** — every position it spans
    masked away, the whole of it where the width is 1. A zero there would build
    a row about constants alone, so the fill is paired with the positions any
    lag actually reached, and a window that reached none of them keeps no row
    (#1059, #1060).

    ``by=`` stops the window at each group's edge, and it is the same gather
    one lag at a time: :func:`_gather_in_groups` reads each position's peer
    inside its own group, so a lag reaching past the group's start is the
    unreachable position it already is at the axis edge. A coordinate the
    lookup places nowhere is in no group and reaches nothing, itself included,
    which is the one way a window loses a row.

    A width declared over the group's own dim is read through the lookup first
    (:func:`_per_group`): left as it came it would broadcast the comparison
    ``within > lag`` onto that dim and hand the constraint a dimension the
    language says a window does not have.
    """
    card = int(array.sizes[over])
    within = _per_group(within, by) if by is not None else within
    widest = int(np.max(np.asarray(within))) if isinstance(within, xr.DataArray) else int(within)
    widest = min(widest, card)
    terms: list[Any] = []
    reached: list[Any] = []
    for lag in range(widest):
        lagged = (
            _gather_in_groups(array, over, lag, groups=by, wrap=edge == EDGE_WRAP, fill=None)
            if by is not None
            else _gather_by_offset(array, over, lag, wrap=edge == EDGE_WRAP, fill=None, card=card)
        )
        live, term = ~lagged.isnull(), absence.filled(lagged, 0.0)
        if isinstance(within, xr.DataArray):
            live, term = live & (within > lag), term * (within > lag).astype(float)
        terms.append(term)
        reached.append(live)
    return reduce(operator.add, terms).where(reduce(operator.or_, reached))


def _checked_mappings(call: str, mappings: tuple[Any, ...], into: tuple[str, ...]) -> tuple[Any, ...]:
    """*mappings* renamed to the dims they target, refusing a shape no lane has.

    Each arrives as the lookup's values over the dim being grouped; renaming
    it to its target is what makes the group's own name the dim that comes
    out, and it is the one thing both arities need.
    """
    renamed = []
    for mapping, target in zip(mappings, into, strict=True):
        if not isinstance(mapping, xr.DataArray):
            msg = f'{call} lookup must be an array (got {type(mapping).__name__}). Usage: {call[:-2]}(expr, by=lookup)'
            raise TypeError(msg)
        if mapping.ndim != 1:
            msg = f'{call} mapping must have exactly one dimension, got {list(mapping.dims)}'
            raise LanguageError(msg)
        renamed.append(mapping.rename(target))
    return tuple(renamed)


def _present(mappings: tuple[Any, ...]) -> Any:
    """The members every mapping has a value for, as a boolean over their dim."""
    keep = mappings[0].notnull()
    for mapping in mappings[1:]:
        keep = keep & mapping.notnull()
    return keep


def _reindexed(summed: Any, *, into: tuple[str, ...], labels: Mapping[str, pd.Index]) -> Any:
    """*summed* over exactly the declared labels, empty groups filled with an empty sum.

    A grouped parameter is a plain ``DataArray``, where the empty sum is 0. A
    grouped expression is a ``LinearExpression``, whose empty term is spelled
    per-variable — linopy's own ``_fill_value`` cannot be used, its ``const:
    nan`` propagates through the arithmetic that follows and poisons the row.

    The unstacked multi-key groupby has already invented the combinations no
    member lands on, and filled them with linopy's ``_fill_value`` — the same
    ``const: nan``, which reindex never sees because those labels are present.
    So the fill comes first and the reindex second, and a (bus, technology)
    pair nothing sits at is an empty sum for the same reason a bus nothing sits
    on is. No other NaN reaches here: a grouped sum zeroes its members' NaN
    constants, and a hole in the data is refused at bind (#1001).
    """
    index = {d: labels[d] for d in into}
    if hasattr(summed, 'const'):
        fill = {'vars': -1, 'coeffs': 0.0, 'const': 0.0}
        return summed.fillna({'const': 0.0}).reindex(index, fill_value=fill)
    return summed.fillna(0.0).reindex(index, fill_value=0)


def _translation(over: str, by: float) -> Mapping[Hashable, int]:
    """The ``{dim: n}`` mapping xarray and linopy both take."""
    if int(by) != by:
        msg = f'shift() by must be an integer, got {by!r}'
        raise TypeError(msg)
    return {over: int(by)}


def _gather_by_offset(array: Any, over: str, offset: Any, *, wrap: bool, fill: float | None, card: int) -> Any:
    """Translate *array* along *over* by an offset that differs per entity.

    A scalar shift is one call; a per-entity one is a **gather**: every output
    position reads a source position of its own, so the index is an array over
    the offset's dims and *over* rather than a number.

    Selection is by *label* rather than by ordinal, because that is what linopy
    passes through to its own labels — the positions are turned back into
    coordinate values here, which also keeps a non-integer axis (a datetime
    snapshot) working for free.

    Out-of-range positions are clipped so the gather stays on the axis, then
    emptied again by ``where``, so an edge means the same thing it does for a
    scalar shift: absent by default, and :func:`~lpspec.linopy.absence.vacated` fills it where the
    model asked. Under ``wrap`` nothing is out of range and the modulo is the
    whole of it.
    """
    labels = np.asarray(array.indexes[over])
    ordinal = xr.DataArray(np.arange(card), coords={over: labels}, dims=[over])
    source = (ordinal - offset).astype(int)

    def gathered(ordinals: Any) -> Any:
        # The indexer carries no coordinate, so the result comes back with the
        # axis unlabelled; the output position *t* still means "at t", so the
        # original labels go back on before anything is combined with it.
        picked = array.sel({over: _labelled(labels, ordinals, over)})
        return picked.assign_coords({over: labels})

    if wrap:
        return gathered(source % card)
    inside = ((source >= 0) & (source < card)).assign_coords({over: labels})
    moved = gathered(source.clip(0, card - 1)).where(inside)
    return moved if fill is None else absence.vacated(moved, array, over, ~inside, fill)


def _per_group(offset: Any, groups: Any) -> Any:
    """*offset* at every coordinate, where it is declared over the group's own dim.

    One lag per group — a construction lead time that differs by period, the
    thing a ``(period, timestep)`` model writes as an offset over ``period``
    because ``period`` is not the axis it walks (#1161). On a flat axis the
    group *is* the lookup's value, so the lag a coordinate moves by is its
    group's, read through that lookup: the pullback ``at()`` already is.

    Every other offset is returned as it came — a number, or an array over dims
    the operand carries, which the gather reads without help.

    The group's label is dropped rather than ridden along: a pullback leaves
    what it read through as a coordinate, and a constraint built from one is
    then reported as carrying a dimension the language says a shift does not
    have.
    """
    target = getattr(groups, 'name', None)
    if not isinstance(offset, xr.DataArray) or target not in offset.dims:
        return offset
    return operator_at(offset, (groups,), into=(str(target),)).drop_vars(str(target))


def _gather_in_groups(array: Any, over: str, offset: Any, *, groups: Any, wrap: bool, fill: float | None) -> Any:
    """Translate *array* inside each group *groups* makes, not along the axis.

    The neighbour of a coordinate is the one *offset* back among the coordinates
    sharing its group, so the gather is by a source ordinal computed per group:
    a position past the group's start is vacated where the axis edge would
    vacate, and under *wrap* it comes round to that group's own last.

    *offset* is a number, or an array where the model named a parameter — per
    entity, per group (:func:`_per_group`), or both — so the source ordinal is
    computed from the within-group position rather than looked up member by
    member, and carries the offset's own dims.

    A coordinate the lookup sends nowhere belongs to no group, so it reaches
    nothing — the null reading a partial lookup gets everywhere else. That is
    not the same as reaching *off* a group's edge, which is what a policy
    speaks for, so the two are tracked apart and only the second is filled
    (#1061). Its lag is a null too, and a comparison against one is False,
    which lands it outside every group by the same arithmetic.

    The relational lane computes the identical map as a rank over the dim
    table joined back on ``(group, position)``.
    """
    labels = np.asarray(array.indexes[over])
    keys = np.asarray(groups.sel({over: labels}).values, dtype=object)

    peers: dict[object, list[int]] = {}
    within = np.zeros(len(labels), dtype=int)
    grouped = np.zeros(len(labels), dtype=bool)
    for k, key in enumerate(keys):
        if key is None or key != key:  # nan: what a partial lookup leaves
            continue
        grouped[k] = True
        beside = peers.setdefault(key, [])
        within[k] = len(beside)
        beside.append(k)

    order = {key: g for g, key in enumerate(peers)}
    widest = max((len(beside) for beside in peers.values()), default=1)
    roster = np.zeros((max(len(peers), 1), widest), dtype=int)
    for key, beside in peers.items():
        roster[order[key], : len(beside)] = beside
    belongs = np.array([order.get(key, 0) for key in keys], dtype=int)
    span = np.array([len(peers[key]) if held else 1 for key, held in zip(keys, grouped, strict=True)], dtype=int)

    axis = {over: labels}
    size = xr.DataArray(span, coords=axis, dims=[over])
    reached = xr.DataArray(within, coords=axis, dims=[over]) - offset
    if wrap:
        reached = reached % size
    inside = xr.DataArray(grouped, coords=axis, dims=[over]) & (reached >= 0) & (reached < size)

    def peer(group: Any, position: Any) -> Any:
        """The label ordinal sitting at *position* of the group at *group*."""
        return roster[group, position]

    source = xr.apply_ufunc(peer, xr.DataArray(belongs, coords=axis, dims=[over]), reached.where(inside, 0).astype(int))
    gathered = array.sel({over: _labelled(labels, source, over)}).assign_coords({over: labels}).where(inside)
    if fill is None:
        return gathered
    return absence.vacated(gathered, array, over, xr.DataArray(grouped, coords=axis, dims=[over]) & ~inside, fill)


def _off_the_axis(array: Any, over: str, offset: float) -> Any:
    """Which positions along *over* a scalar shift of *offset* leaves vacated.

    The source a position reads, off both ends, so one expression covers a
    shift in either direction — the same verdict :func:`_gather_by_offset`
    reaches with ``inside`` negated.
    """
    labels = np.asarray(array.indexes[over])
    source = xr.DataArray(np.arange(len(labels)), coords={over: labels}, dims=[over]) - int(offset)
    return (source < 0) | (source >= len(labels))


def _labelled(labels: Any, ordinals: Any, over: str) -> Any:
    """*ordinals* as the coordinate labels they stand for, keeping their dims.

    Carries no coordinates of its own: an indexer that keeps the axis's own
    coordinate asserts the values it holds *are* that axis, and after a gather
    they are not — which xarray reports as a size conflict rather than a
    mislabelling.
    """
    return xr.DataArray(labels[ordinals.transpose(*ordinals.dims).values], dims=ordinals.dims)


def _unsupported(call: str, array: Any) -> TypeError:
    """One wording for an operand shape an operator cannot take.

    Reached only from a hand-built call: every operator's operands come from
    ``_eval_ast``, so a lane running the language proper never sees this.
    """
    return TypeError(f"{call} does not support type '{type(array).__name__}'.")


#: Eager evaluation of every name in ``operators.BUILTIN_NAMES``. The two must
#: agree exactly — enforced by ``tests/test_architecture.py``, because a name
#: one lane implements and the other does not is precisely the divergence
#: that would make the differential tests a comparison of dialects.
OPERATORS: dict[str, Callable[..., Any]] = {
    'sum': _operator_sum,
    'at': operator_at,
    'shift': _operator_shift,
    'sum_back': _operator_sum_back,
}
