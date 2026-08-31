"""Model builder: logical plan + data → linopy Model.

**One section per kind of translation**, in the order a build performs them:
the four declarations (``Variables``, ``Special-ordered sets``,
``Constraints``, ``Objectives``) each ending in the ``model.add_*`` call they
exist to make, then ``Plan evaluation`` for what an expression is worth.

**The input is a** :class:`~math_spec.program.Program`, the same one the relational
engine executes. Every name is already typed, every operator call is already
its own node, and every ``where:`` is already a predicate, so this module
resolves nothing and cannot disagree with the other lane about what a file
means — the two read one plan and differ only in what they build from it.

Two questions a build asks are answered beside it rather than here, because
neither needs the plan or the model: ``operators.py`` evaluates a built-in
once its operands are values, and ``where.py`` turns a predicate into
the boolean array a declaration is masked by. The four positions an absent
value is spelled differently in are ``absence.py``, called qualified from here.

*Which* linopy call each construct becomes is the table in
``docs/about/linopy.md`` — one page rather than a comment per site, and
``tests/test_docs_site.py`` holds it to the operator list.
"""

from __future__ import annotations

import functools
import operator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, assert_never

import numpy as np
from math_spec import EDGE_WRAP, program

from lpspec.errors import (
    DataError,
    LaneError,
    lane_cannot_build_message,
    null_bounds_message,
    unbound_lookup_message,
)
from lpspec.linopy import absence
from lpspec.linopy._notes import note
from lpspec.linopy.coverage import check_constant_side_covers, check_divisors_cover, gaps_under
from lpspec.linopy.operators import OPERATORS, operator_at, operator_grouped_sum
from lpspec.linopy.where import WhereContext, as_linopy_mask, evaluate_where

if TYPE_CHECKING:
    import linopy
    import pandas as pd
    import xarray as xr

_SIGN_MAP = {'==': '=', '<=': '<=', '>=': '>='}


@dataclass(frozen=True)
class EvaluationContext(WhereContext):
    """Everything expression evaluation needs, beyond the data itself.

    :class:`~lpspec.linopy.where.WhereContext` plus the program, which is what
    a variable's absence rule and a lookup's target dimension are read off.
    Extend this rather than adding parameters to ``_eval`` and every
    operator-facing seam.
    """

    #: Narrowed from the base: a build always has the model it populates.
    model: linopy.Model
    program: program.Program = field(kw_only=True)


def build_model(
    model: linopy.Model,
    program: program.Program,
    dataset: xr.Dataset,
    master_coords: dict[str, pd.Index],
    dim_coords: dict[str, dict[str, xr.DataArray]] | None = None,
) -> None:
    """Populate a linopy Model from a lowered program and loaded parameters.

    This mutates *model* in-place, adding variables, constraints and the
    objective as declared in *program*. Nothing is re-checked here: a program
    is trusted by construction, ``to_program`` having decided every rule the
    language can decide without data.
    """
    ctx = EvaluationContext(dataset, master_coords, model, dim_coords or {}, program=program)
    _build_variables(ctx)
    _build_sos(ctx)
    _build_constraints(ctx)
    _build_objective(ctx)


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------


def _build_variables(ctx: EvaluationContext) -> None:
    for name, vdef in ctx.program.variables.items():
        with note(f"while building variable '{name}'"):
            coords = {d: ctx.master_coords[d] for d in vdef.dims}
            mask = evaluate_where(vdef.where, ctx)

            _check_bounds_are_defined(name, vdef, ctx.dataset, mask)

            ctx.model.add_variables(
                lower=_bound(vdef.lower, ctx.dataset),
                upper=_bound(vdef.upper, ctx.dataset),
                coords=coords,
                name=name,
                mask=as_linopy_mask(mask),
                binary=vdef.variable_type == 'binary',
                integer=vdef.variable_type == 'integer',
            )


def _check_bounds_are_defined(name: str, vdef: program.VariableDeclaration, dataset: xr.Dataset, mask: Any) -> None:
    """Refuse a bound with no value, at build, as the native lane does.

    Otherwise the NaN travels into linopy and surfaces two phases later from
    inside its IO layer — ``Continuous Variable x contains nan's in field(s)
    ['upper']``, raised at solve or write, naming neither the YAML nor the fix,
    from a ``build()`` that had already returned.

    Checked against the variable's own mask: a coordinate the variable does not
    occupy needs no bound, and supplying data only where it exists is the
    ordinary idiom.
    """
    missing = sum(gaps_under(dataset[name], mask) for name in sorted(program.parameters_of(vdef.lower, vdef.upper)))
    if missing:
        raise DataError(null_bounds_message(name, missing))


def _bound(bound: program.ExpressionNode, dataset: xr.Dataset) -> Any:
    """A bound as linopy takes it: the literal, or the named parameter's array.

    Read raw rather than through :func:`absence.coefficient`, which is the
    whole point of the position: the absence rules' zero is a coefficient and
    never a bound, so a gap here has to survive to
    :func:`_check_bounds_are_defined` instead of being filled in.
    """
    if isinstance(bound, program.Constant):
        return bound.value
    if isinstance(bound, program.Parameter):
        return dataset[bound.name]
    msg = f'bounds accept a number or a parameter, and lowering builds nothing else — got {type(bound).__name__}'
    raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Special-ordered sets
# ---------------------------------------------------------------------------


def _build_sos(ctx: EvaluationContext) -> None:
    """Attach every ``sos:`` block to the variable it names.

    linopy holds a set the same way the language declares one — a variable, a
    dimension of it, a type — so this is the block handed over, not a
    formulation rebuilt. Which is the point of copying its decomposition:
    the eager lane is the oracle, and a set it had to *reformulate* to accept
    would be an oracle for a different model.

    It runs before the constraints because a set is a property of the
    variable, so it belongs beside the declaration rather than after
    everything that uses it.
    """
    for name, sos in ctx.program.sos.items():
        with note(f"while building sos '{name}'"):
            ctx.model.add_sos_constraints(
                ctx.model.variables[sos.variable],
                sos_type=sos.sos_type,
                sos_dim=sos.over,
                big_m=sos.big_m,
            )


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def _refuse_quadratic(p: program.Program) -> None:
    """A quadratic constraint is sayable and this lane cannot build one.

    Hard rule 3's amendment in the one place it bites: both lanes accept the
    same language, and ``linopy.Model.add_constraints`` refuses a
    ``QuadraticExpression`` outright. Refused *here*, in the language's own
    words and before linopy is asked, so the caller gets a sentence naming the
    lane that does build it rather than somebody else's
    ``NotImplementedError`` (:data:`lpspec.api.LANES`).

    Asked of the program rather than of each row, because the wall is the
    lane's and not one declaration's: no row is worth building before the
    answer is known.
    """
    if 'constraint' in p.footprint.quadratic:
        raise LaneError(lane_cannot_build_message('linopy', ['quadratic_constraint']))


def _build_constraints(ctx: EvaluationContext) -> None:
    _refuse_quadratic(ctx.program)
    for name, row in ctx.program.constraints.items():
        with note(f"while building constraint '{name}'"):
            mask = evaluate_where(row.where, ctx)
            context = f"constraint '{name}'"

            check_divisors_cover(context, (row.lhs, row.rhs), ctx, mask)
            check_constant_side_covers(context, row, ctx, mask)

            lhs = _eval(row.lhs, ctx)
            rhs = _eval(row.rhs, ctx)
            if _term_free(lhs) and _term_free(rhs):
                continue

            ctx.model.add_constraints(lhs, _SIGN_MAP[row.sense], rhs, name=name, mask=as_linopy_mask(mask))


def _term_free(side: Any) -> bool:
    """Whether *side* has nowhere for a variable term to sit.

    A bare ``Variable`` is a term; a ``LinearExpression`` over an empty axis has
    a term dimension of length zero; anything else is data.

    Both sides term-free is a constraint the *data* emptied: a dimension with no
    members reduces away to a number, so every row asserts something about
    constants alone, which the absence rules say is not a row and the relational
    lane builds none (#1108). The expression that names no variable to begin
    with cannot arrive here — validation refuses it while reading the file
    (#1171) — so this reads as data, not as a modelling mistake.
    """
    if hasattr(side, 'to_linexpr'):
        return False
    return getattr(side, 'nterm', 0) == 0


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------


def _build_objective(ctx: EvaluationContext) -> None:
    """Build the declared objective, if any, onto the model.

    An objective has no ``where``, so its divisor check runs with no row mask
    — the numerator's own presence is the only thing that can excuse a gap.

    The expression is scalar by the time it gets here, ``check_schema`` having
    refused one carrying dims. That is what lets it be evaluated like any
    other: an eager ``+`` broadcasts its operands, so a lane that added the
    terms of ``x[i] * a[i] + y[j] * b[j]`` and summed afterwards would count
    each once per coordinate of the other (#197) — and now nothing has to,
    because the file said where each sum ends.

    The one position where a product of two variables is legal. Nothing here
    branches on that: lowering applied the ceiling before the plan existed, and
    linopy's ``*`` answers with a ``QuadraticExpression`` on its own.
    """
    odef = ctx.program.objective
    if odef is None:
        return
    with note('while building the objective'):
        check_divisors_cover('the objective', (odef.expression,), ctx, None)

        expr = _eval(odef.expression, ctx)
        _refuse_an_objective_constant(expr)

        ctx.model.add_objective(expr, overwrite=True, sense=_LINOPY_SENSE[odef.sense])


#: The program's objective sense in linopy's own spelling. Every sink names the
#: direction its own way — ``min`` here, ``MAXIMIZE`` in gurobi, a keyword byte
#: in an LP file — and each translates at its own edge, so the one spelling the
#: tree carries is the language's.
_LINOPY_SENSE: dict[program.ObjectiveSense, str] = {'minimize': 'min', 'maximize': 'max'}


#: The one construct this lane accepts and cannot build, as the sentence a user
#: reads. Module-level so the test that pins the wording has something to name.
OBJECTIVE_CONSTANT_IS_A_LANE_GAP = (
    "the objective carries a constant term, and this lane cannot build one: linopy's objective "
    'takes no constant — there is no `objective_constant` slot anywhere in the package, which is '
    'why PyPSA carries its own out of band. The relational lane builds it and returns the right '
    'number, so the model is sayable and only this lane is short: run it with `lpspec.solve` / '
    '`lpspec.build`. Dropping the constant here is refused deliberately — it would answer a '
    'different model, and this lane is the differential oracle, so every test on such a file '
    "would be calibrated to the shortened objective's number (#894)."
)


def _refuse_an_objective_constant(expr: Any) -> None:
    """Refuse an objective this lane cannot build, before linopy is asked.

    linopy's own refusal names neither the file nor the other lane, and arrives
    from a setter two frames down. A check rather than a `try`, because the
    upstream message is not a contract and a nonzero constant is the whole of
    what it means.
    """
    const = getattr(expr, 'const', None)
    if const is not None and bool(np.any(np.asarray(const) != 0)):
        raise LaneError(OBJECTIVE_CONSTANT_IS_A_LANE_GAP)


# ---------------------------------------------------------------------------
# Plan evaluation
# ---------------------------------------------------------------------------


def _eval(node: program.ExpressionNode, ctx: EvaluationContext) -> Any:
    """One plan node as a linopy term, an array, or a number.

    One node kind per branch, and each is a line: a variable is its linopy
    term, a parameter its filled array, arithmetic the Python operator linopy
    overloads, and an operator its function in ``operators.py``.

    Shorter than the walk it replaced by every branch that existed to refuse
    something: a plan carries no unresolved name, no bare keyword and no edge
    policy in a value position, because lowering either typed it or refused the
    file. What is left is what a build actually evaluates.
    """
    if isinstance(node, program.Constant):
        return node.value

    if isinstance(node, program.Variable):
        return absence.variable_term(ctx.model.variables[node.name], ctx.program.variable(node.name).absence)

    if isinstance(node, program.Parameter):
        return absence.coefficient(ctx.dataset[node.name])

    if isinstance(node, program.Negate):
        return -_eval(node.operand, ctx)

    if isinstance(node, program.Add):
        return _eval(node.left, ctx) + _eval(node.right, ctx)

    if isinstance(node, program.Multiply):
        return _eval(node.left, ctx) * _eval(node.right, ctx)

    if isinstance(node, program.Divide):
        return _eval(node.numerator, ctx) / _eval(node.divisor, ctx)

    if isinstance(node, program.Power):
        return _eval(node.base, ctx) ** _eval(node.exponent, ctx)

    if isinstance(node, program.Sum):
        summed = _eval(node.operand, ctx)
        for dimension in node.over:
            summed = OPERATORS['sum'](summed, over=dimension)
        return summed

    if isinstance(node, program.GroupSum):
        return operator_grouped_sum(
            _eval(node.operand, ctx),
            _lookup_arrays(node.over, node.coordinate, ctx),
            into=node.into,
            labels=ctx.master_coords,
        )

    if isinstance(node, program.At):
        return operator_at(_eval(node.operand, ctx), _lookup_arrays(node.over, node.coordinate, ctx), into=node.into)

    if isinstance(node, program.Translate):
        return OPERATORS['shift'](
            _eval(node.operand, ctx),
            over=node.dimension,
            offset=_amount(node.offset, ctx),
            edge=_edge(node),
            by=_partition(node, ctx),
        )

    if isinstance(node, program.Window):
        return OPERATORS['sum_back'](
            _eval(node.operand, ctx),
            over=node.dimension,
            within=_amount(node.width, ctx),
            edge=_edge(node),
            by=_partition(node, ctx),
        )

    if isinstance(node, program.Cases):
        return _cases(node, ctx)

    assert_never(node)


def _in_region(value: Any, mask: xr.DataArray) -> Any:
    """*value* where the region holds, and a hard zero everywhere else.

    A **fill**, not a multiplication. Multiplying would carry the value's own
    absence out of the region that owns it: the ``otherwise`` of a commitment
    file shifts with no ``edge=`` and so has nothing at the first snapshot,
    which times a false mask is still nothing rather than zero, and the row
    the other regions do cover would be unmade by a region that does not
    claim it. Inside the mask absence still stands, because a region empty
    where it applies genuinely leaves the quantity with no value there.

    A bare number is the one value with no absence to protect, and no
    ``where`` to reach it by, so there the mask multiplies.
    """
    if hasattr(value, 'to_linexpr'):
        value = value.to_linexpr()
    if hasattr(value, 'where'):
        return value.where(mask, 0)
    return mask * value


def _cases(node: program.Cases, ctx: EvaluationContext) -> Any:
    """A value defined by region, as the regions added.

    The regions are disjoint and total — the language proved that before any
    data bound — so each one filled with zero outside itself and the lot added
    gives every coordinate exactly one region's value, and no coordinate two.
    Neither an order nor a tie-break is needed, and none is taken.
    """
    filled = (_in_region(_eval(region.value, ctx), evaluate_where(region.when, ctx)) for region in node.regions)
    return functools.reduce(operator.add, filled)


def _amount(amount: int | str, ctx: EvaluationContext) -> Any:
    """An offset or a width: the number, or the integer parameter naming it.

    Read through :func:`absence.coefficient` like any other parameter — a step
    nobody supplied is a step of nothing, which is what a zero offset means.
    """
    return absence.coefficient(ctx.dataset[amount]) if isinstance(amount, str) else amount


def _edge(node: program.Translate | program.Window) -> str | float | None:
    """The edge policy as ``operators.py`` spells it.

    The plan has already decided between the two, so this is a spelling and not
    a choice: wrapping is the keyword, a fill is the number itself, and neither
    is the acyclic default. ``EDGE_WRAP`` rather than the literal it happens to
    equal — the constant is the language's, and a copy of its value here would
    keep working until the day it stopped.
    """
    if node.wrap:
        return EDGE_WRAP
    return getattr(node, 'fill', None)


def _partition(node: program.Translate | program.Window, ctx: EvaluationContext) -> Any:
    """The lookup a windowed operator may not reach across, as its values.

    One lookup — plural is refused at load — so the single array is the whole
    of it, and it is read off the dim store rather than the parameter dataset
    for the reason :func:`_lookup_arrays` gives.

    **Named for the dimension its values are labels of**, not for itself: an
    amount declared over the group's own dim is read through this array by
    :func:`~lpspec.linopy.operators._per_group`, which pairs the two by that
    name. The plan says which dimension that is, so the name is looked up
    rather than assumed.
    """
    if node.partition is None:
        return None
    array = _lookup_arrays(node.dimension, (node.partition,), ctx)[0]
    return array.rename(ctx.program.dimension(node.dimension).targets[node.partition])


def _lookup_arrays(over: str, names: tuple[str, ...], ctx: EvaluationContext) -> tuple[Any, ...]:
    """The declared lookups *names* as arrays over the dimension they are over.

    Looked up rather than evaluated as operands: a lookup lives on the
    dimension, not in the parameter dataset. One array per name, in the
    order the plan wrote them, so the caller can pair each with the dim it
    targets.
    """
    arrays = []
    for name in names:
        try:
            arrays.append(ctx.dim_coords[over][name])
        except KeyError:
            raise DataError(unbound_lookup_message(name, over)) from None
    return tuple(arrays)
