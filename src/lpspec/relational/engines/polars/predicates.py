"""A ``where:`` mask as a query: which rows of a coordinate product survive.

The plan's predicate nodes in, a boolean expression out — and the frame the
walk had to join parameters onto to build it, since a mask reads values the
product does not carry. Two returns rather than one because the joins happen
*during* the walk: the condition is built first and the frame read after.

A closed vocabulary of its own — comparisons against a parameter, a dimension
label, a position along a dimension, a lookup, and the three connectives — so
it is a module rather than a method. It takes the
:class:`~lpspec.relational.engines.polars.compiler.PolarsCompiler` as an
argument and holds nothing.

:class:`Carrier` lives here too, and the bounds walk imports it: both walks
that read parameters build an expression over columns they are joining on as
they go, and this is the larger of the two.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from math_spec import program

from lpspec.errors import DataError, LanguageError

if TYPE_CHECKING:
    import datetime
    from collections.abc import Callable

    from polars._typing import JoinStrategy

    from lpspec.relational.engines.polars.compiler import PolarsCompiler


class Carrier:
    """A frame a walk joins onto, each attachment made at most once.

    Both walks that read parameters — the mask (:func:`compile_predicate`)
    and the bounds (:meth:`~lpspec.relational.engines.polars.compiler.PolarsCompiler.bounds`)
    — build an expression over
    columns they are joining on as they go, so the frame and the set of aliases
    already attached travel together rather than as two ``nonlocal``s and an
    ``if alias not in joined`` at every site.
    """

    def __init__(self, frame: pl.LazyFrame) -> None:
        self.frame = frame
        self._attached: set[str] = set()

    def once(self, alias: str, attach: Callable[[pl.LazyFrame, str], pl.LazyFrame]) -> str:
        """Join *attach* onto the frame under *alias*, unless it already is.

        Returns:
            *alias*, so a caller reads the column it just made sure of.
        """
        if alias not in self._attached:
            self.frame = attach(self.frame, alias)
            self._attached.add(alias)
        return alias


def defined(col: pl.Expr, dtype: str) -> pl.Expr:
    """What a bare parameter name in a ``where`` asks of *col*.

    Three readings, and the declaration picks: a ``bool`` is its own answer, a
    ``str`` is defined wherever the table has a row, and a number has to be
    finite as well. Read off the declaration rather than the column, which is
    the same thing since attaching refuses a column that is not what the file
    declared — and unlike the column it cannot be ``is_finite`` over strings,
    which polars refuses outright.
    """
    if dtype == 'bool':
        return col.is_not_null() & col.cast(pl.Boolean)
    if dtype == 'str':
        return col.is_not_null()
    return col.is_not_null() & col.is_finite()


def compile_predicate(
    compiler: PolarsCompiler, frame: pl.LazyFrame, pred: program.WhereNode, dims: tuple[str, ...]
) -> tuple[pl.LazyFrame, pl.Expr]:
    """``(frame with the mask's parameters joined, boolean expression)``.

    Walking joins the parameters, so the condition is built first and the
    frame read after — one expression would return the pre-walk frame.

    **A name the mask is certain of is joined rather than left-joined**,
    and a certain variable is semi-joined and never read
    (:func:`_certain_parameters`). An atom over a missing value reads as
    false either way, so the strategies differ only in *where* the row is
    dropped, and the inner join saves the width of the product it is
    dropped from — a few percent of a pipeline on the direct path, and
    nothing measurable behind a semi-join's key product (#520).

    ``VariableDefinedNode`` is the one atom answered by a join rather than a
    column test — existence lives in the variable's own frame — keyed by
    dims the dim rule has already checked are inside this frame.

    No join here maintains order: consumers verify where they read
    (:func:`labels.in_position_order`), so a shuffle costs a sort
    downstream at worst, never a wrong label.
    """
    certain = _certain_parameters(pred)
    carrier = Carrier(frame)

    def join_param(param: str) -> str:
        how: JoinStrategy = 'inner' if param in certain else 'left'
        return carrier.once(
            f'__where {param}__',
            lambda f, alias: compiler.parameter_join(f, param, dims, alias, f"where-parameter '{param}'", how),
        )

    def refuse_outside_foreach(reading: str, dimension: str) -> None:
        """A mask reading a dim the frame does not span — the plan's refusal, asserted here.

        Reducing a mask over an unlisted dim would admit a row wherever *any*
        coordinate of it satisfied the mask. The language refuses it at load,
        before a plan exists to carry it, so the frame planner states it as the
        invariant it now is.
        """
        assert dimension in dims, f'where-comparison on {reading} is outside the foreach dims {list(dims)}'

    def join_ordinal(dimension: str) -> str:
        refuse_outside_foreach(f"dimension '{dimension}'", dimension)
        return carrier.once(
            f'__where ord {dimension}__',
            lambda f, alias: f.join(
                compiler.data.dimensions[dimension].select(pl.col('val').alias(dimension), pl.col('ord').alias(alias)),
                on=dimension,
                how='left',
            ),
        )

    def join_group_offset(p: program.DimensionPositionNode) -> str:
        """One column: the row's ordinal minus its own group's target ordinal."""
        refuse_outside_foreach(f"dimension '{p.name}'", p.name)
        table = compiler.partitioned(p.name, str(p.by))
        _refuse_short_groups(p, table)
        group = pl.col(str(p.by))
        within = pl.col('ord').rank('ordinal').over(group).cast(pl.Int64) - 1
        size = pl.len().over(group).cast(pl.Int64)
        target = pl.lit(p.position) if p.position >= 0 else size + p.position
        offset = within - target
        return carrier.once(
            f'__where ord {p.name} by {p.by}__',
            lambda f, alias: f.join(
                table.select(pl.col('val').alias(p.name), offset.alias(alias)),
                on=p.name,
                how='left',
            ),
        )

    def join_lookup(lookup: str, over: str) -> str:
        refuse_outside_foreach(f"lookup '{lookup}' reading dimension '{over}'", over)
        return carrier.once(
            f'__where lookup {lookup}__',
            lambda f, alias: f.join(
                compiler.data.lookups[lookup].select(pl.col(over), pl.col(lookup).alias(alias)),
                on=over,
                how='left',
            ),
        )

    def walk(p: program.WhereNode) -> pl.Expr:
        if isinstance(p, program.ParameterComparisonNode):
            return _compare(pl.col(join_param(p.name)), p.op, p.value)
        if isinstance(p, program.DimensionComparisonNode):
            refuse_outside_foreach(f"dimension '{p.name}'", p.name)
            return _compare(_dimension_column(p.name, p.value), p.op, p.value)
        if isinstance(p, program.DimensionPositionNode):
            if p.by is not None:
                return falsy_if_null(_COLUMN_COMPARISONS[p.op](pl.col(join_group_offset(p)), pl.lit(0)))
            at = _position_ordinal(p, compiler.data.cardinality[p.name])
            return _COLUMN_COMPARISONS[p.op](pl.col(join_ordinal(p.name)), pl.lit(at))
        if isinstance(p, program.LookupComparisonNode):
            column = pl.col(join_lookup(p.name, p.over))
            if isinstance(p.value, str):
                column = column.cast(pl.String)
            return _compare(column, p.op, p.value)
        if isinstance(p, program.LookupPairComparisonNode):
            left = pl.col(join_lookup(p.name, p.over))
            right = pl.col(join_lookup(p.other, p.over))
            return _COLUMN_COMPARISONS[p.op](left, right)
        if isinstance(p, program.LookupDefinedNode):
            return pl.col(join_lookup(p.name, p.over)).is_not_null()
        if isinstance(p, program.ParameterDefinedNode):
            return defined(pl.col(join_param(p.name)), compiler.program.parameter(p.name).dtype)
        if isinstance(p, program.VariableDefinedNode):
            on = list(compiler.program.variable(p.name).dims)
            coordinates = compiler.variables[p.name].frame.select(*on)
            if p.name in certain:
                carrier.once(f'__where defined {p.name}__', lambda f, _: f.join(coordinates, on=on, how='semi'))
                return pl.lit(value=True)
            flag = carrier.once(
                f'__where defined {p.name}__',
                lambda f, alias: f.join(
                    coordinates.unique().with_columns(pl.lit(value=True).alias(alias)), on=on, how='left'
                ),
            )
            return falsy_if_null(pl.col(flag))
        if isinstance(p, program.BooleanLiteralNode):
            return pl.lit(value=p.value)
        if isinstance(p, program.AndNode):
            return walk(p.left) & walk(p.right)
        if isinstance(p, program.OrNode):
            return walk(p.left) | walk(p.right)
        if isinstance(p, program.NotNode):
            return ~falsy_if_null(walk(p.operand))
        raise LanguageError(f'unsupported predicate node {type(p).__name__}')

    condition = walk(pred)
    return carrier.frame, condition


def _certain_parameters(pred: program.WhereNode) -> frozenset[str]:
    """Names whose absence alone makes the whole mask false.

    A row those names have no value for is one the filter would drop anyway, so
    the join may drop it first. Only ``And`` descends — under ``Or`` or ``Not``
    an absent value can still leave the mask true, and dropping the row there
    is a wrong model rather than a slow one, which is why the fallthrough
    answers nothing rather than raising on a node it does not know.
    """
    if isinstance(pred, program.AndNode):
        return _certain_parameters(pred.left) | _certain_parameters(pred.right)
    if isinstance(pred, (program.ParameterComparisonNode, program.ParameterDefinedNode)):
        return frozenset({pred.name})
    if isinstance(pred, program.VariableDefinedNode):
        return frozenset({pred.name})
    return frozenset()


def _refuse_short_groups(p: program.DimensionPositionNode, table: pl.LazyFrame) -> None:
    """Refuse a position no coordinate of some group occupies.

    The ungrouped counterpart is :func:`_position_ordinal`, and the reason is
    the same one construct-wide: a boundary clause that silently seeds no row
    leaves that group's recurrence unanchored. Grouping only multiplies the
    chance — one short period is enough — so it is checked per group, which
    costs one pass over the table the mask is about to join anyway.

    *table* is :meth:`_Compiler._partitioned`'s, so a coordinate in no group is
    not in it and no group of ``None`` can be counted short.
    """
    needed = p.position + 1 if p.position >= 0 else -p.position
    sizes = table.select(pl.col(str(p.by))).group_by(str(p.by)).len().collect()
    short = sorted(str(g) for g, n in sizes.iter_rows() if n < needed)
    if short:
        msg = (
            f'where: position({p.name}, by={p.by}) {p.op} {p.position} names position '
            f'{p.position} within each group, and {len(short)} of them are shorter than '
            f'that: {short[:5]}. A boundary that names no coordinate leaves the rows it '
            f'was to seed unseeded.'
        )
        raise DataError(msg)


def falsy_if_null(condition: pl.Expr) -> pl.Expr:
    """*condition* with null read as false.

    A missing parameter row must exclude the coordinate rather than
    propagate. Masks are row absence.
    """
    return condition.fill_null(value=False)


def _position_ordinal(p: program.DimensionPositionNode, cardinality: int) -> int:
    """*p*'s position as an ordinal into a dimension of *cardinality* labels.

    A negative position counts from the end. Out of range is an error rather
    than a predicate matching nothing: a boundary clause that silently seeds
    no row leaves the recurrence unanchored, which is the failure this
    construct exists to make impossible.
    """
    at = p.position + cardinality if p.position < 0 else p.position
    if not 0 <= at < cardinality:
        raise DataError(
            f'where: position({p.name}) {p.op} {p.position} names position {at} of '
            f"'{p.name}', which has {cardinality} coordinate(s). A boundary that "
            f'names no coordinate leaves the rows it was to seed unseeded.'
        )
    return at


def _dimension_column(dimension: str, value: float | str | datetime.date) -> pl.Expr:
    """The column a where-comparison on *dimension* reads.

    A string label is compared in ``String`` space, undoing attaching's ``Enum``:
    The where-string rules order labels bytewise and read an unknown label as
    matching nothing,
    where an ``Enum`` orders by declaration and refuses strangers.
    """
    column = pl.col(dimension)
    return column.cast(pl.String) if isinstance(value, str) else column


#: The comparison operators, evaluated column against column — the one table,
#: so a seventh operator added to :data:`program.PredicateOperator` fails here
#: rather than falling through a second copy.
_COLUMN_COMPARISONS: dict[program.PredicateOperator, Callable[[pl.Expr, pl.Expr], pl.Expr]] = {
    '==': lambda left, right: left == right,
    '!=': lambda left, right: left != right,
    '<': lambda left, right: left < right,
    '<=': lambda left, right: left <= right,
    '>': lambda left, right: left > right,
    '>=': lambda left, right: left >= right,
}


def _compare(column: pl.Expr, op: program.PredicateOperator, value: float | str | datetime.date) -> pl.Expr:
    """One where-comparison. A string, a float and a date are all literals here."""
    return _COLUMN_COMPARISONS[op](column, pl.lit(value))
