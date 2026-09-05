"""Reading a built model back: one constraint row, a solve's frames, a named expression."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import polars as pl
from math_spec import program

from lpspec.errors import DataError, LpspecError, sparse_divisor_message, unknown_name_message
from lpspec.relational.engines.polars import labels
from lpspec.relational.engines.polars.assembly import absence_restrictions
from lpspec.relational.engines.polars.fragments import join_on
from lpspec.relational.result import ConstraintRow

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lpspec.relational.engines.polars.assembly import BuiltModel
    from lpspec.relational.engines.polars.attaching import AttachedSources
    from lpspec.relational.engines.polars.compiler import PolarsCompiler
    from lpspec.relational.engines.polars.fragments import TermFragment
    from lpspec.relational.result import Result

#: Scratch columns of the expression reader. The spaces make them
#: unrepresentable as declared names.
SOLUTION = '__solution value__'
_EXPRESSION_ROW = '__expression row__'


def row(model: BuiltModel, name: str, coordinate: Mapping[str, Any]) -> ConstraintRow:
    """One built constraint row, spelled back out. See :meth:`~lpspec.api.Model.row`.

    Three positional takes against frames the build already keeps, and no scan
    of the matrix: the constraint's own coordinate frame carries the global row
    index, ``row_starts`` says where that row's entries lie, and each
    variable's frame carries the global column index its terms point at.

    Raises:
        KeyError: No constraint of that name.
        LpspecError: A coordinate the declaration cannot name, or one it built
            no row at.
    """
    if name not in model.constraints:
        raise KeyError(unknown_name_message('constraint', name, sorted(model.constraints)))

    at, ordered = _row_index(model, name, coordinate)
    starts = model.matrix_starts
    entries = model.matrix.slice(int(starts[at]), int(starts[at + 1] - starts[at]))
    stated = model.rows.slice(at, 1)
    return ConstraintRow(
        name=name,
        coordinate=ordered,
        terms=_named_terms(model, entries),
        sense=str(stated.item(0, 'sense')),
        rhs=float(stated.item(0, 'rhs')),
    )


def _row_index(model: BuiltModel, name: str, coordinate: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    """The global row index constraint *name* built at *coordinate*, and that coordinate in dim order.

    The coordinate has to name **every** dim of the declaration: a partial
    one matches a set of rows, and a verb that quietly answered about the
    first of them would be reporting one row as if it were the block.

    Raises:
        LpspecError: The coordinate names dims the declaration does not,
            holds a label the dimension cannot, or matches no row the build
            produced — a row masked out by ``where`` or dropped for having no
            terms.
    """
    dims = model.program.constraints[name].dims
    if set(coordinate) != set(dims):
        raise LpspecError(
            f"constraint '{name}' is declared over {list(dims)}, and a row is read at all of them "
            f'— got {sorted(coordinate)}. A row is one coordinate: name every dim once, and no dim '
            'the declaration does not have.'
        )
    frame = model.constraints[name].frame
    schema = frame.collect_schema()
    ordered = {d: coordinate[d] for d in dims}
    predicates = [pl.col(d) == _label(name, d, v, schema[d]) for d, v in ordered.items()]
    found = frame.filter(predicates).collect() if predicates else frame.collect()
    if not found.height:
        raise LpspecError(
            f"constraint '{name}' built no row at {ordered}. Either a `where` masked the "
            'coordinate out, every term it had was absent, or the labels are not ones the '
            'dimension holds — and which of those it is, is what diagnostics() reports as an '
            'omission.'
        )
    return int(found.item(0, 'row')), ordered


def _label(name: str, dim: str, value: Any, dtype: pl.DataType) -> pl.Expr:
    """*value* as a literal of *dim*'s own type, or a refusal naming what it is not.

    The cast **is** the check: a string against an integer dim and a stranger
    against an ``Enum`` are one failure, and neither reaches polars as a
    comparison it can only report in its own vocabulary.
    """
    try:
        return pl.lit(pl.Series([value], dtype=dtype).item(0), dtype=dtype)
    except (pl.exceptions.PolarsError, TypeError, OverflowError) as refused:
        raise LpspecError(
            f"constraint '{name}' is declared over '{dim}', which holds {dtype}, and {value!r} is "
            f'not one of its labels. Read the row at a label the dimension has.'
        ) from refused


def _named_terms(model: BuiltModel, entries: pl.DataFrame) -> pl.DataFrame:
    """``(variable, coordinate, coefficient)`` for one row's matrix entries.

    Each declaration owns a contiguous, dense run of column indices, so which
    variable a term belongs to is a range test, and a term's place in its own
    declaration's frame is its label minus that block's start — a positional
    take, not a search.

    ``coordinate`` is rendered rather than spread across dim columns because
    one row's terms may come from variables with *different* dims. It carries
    the labels alone, in the declaration's dim order — linopy's ``p[1, wind]``
    bracket. The terms leave in the order the entries arrived in, which is the
    solver's own column order.
    """
    wanted = entries['col'].to_numpy()
    named = []
    for variable, held in model.variables.items():
        inside = wanted[(wanted >= held.start) & (wanted < held.start + held.height)]
        if not inside.size:
            continue
        dims = model.program.variable(variable).dims
        at = pl.Series('#position', inside - held.start, dtype=pl.UInt32)
        picked = held.frame.select(pl.col('var_label'), *(pl.col(d) for d in dims)).select(pl.all().gather(at))
        rendered = pl.concat_str([pl.col(d).cast(pl.String) for d in dims], separator=', ') if dims else pl.lit('')
        named.append(
            picked.select(
                pl.col('var_label').alias('col'),
                pl.lit(variable).alias('variable'),
                rendered.alias('coordinate'),
            ).collect()
        )
    labelled = (
        pl.concat(named)
        if named
        else pl.DataFrame(schema={'col': pl.Int64, 'variable': pl.String, 'coordinate': pl.String})
    )
    return (
        entries.with_columns(pl.col('col').cast(pl.Int64))
        .join(labelled.with_columns(pl.col('col').cast(pl.Int64)), on='col', how='left', maintain_order='left')
        .select('variable', 'coordinate', pl.col('coeff').alias('coefficient'))
    )


def laid_out(
    attached: AttachedSources, held: labels.Labelled, dims: tuple[str, ...], values: pl.Series
) -> pl.LazyFrame:
    """One declaration's coordinates in label order, beside its share of *values*.

    The order was never lost: :func:`labels.frame` hands back a
    label-ascending frame, and the solver's vector is positional in the same
    index. The share is attached as a column rather than concatenated as a
    frame, so a mismatched length raises instead of padding with nulls.

    **Dim columns leave in ``String``**, where the build holds them as
    ``pl.Enum``: a returned frame is something a caller joins against their
    own data, and polars refuses ``Enum`` against ``String``.
    """
    labelled = held.frame.select(*dims).with_columns(held.share(values))
    return labelled.with_columns(pl.col(d).cast(pl.String) for d in string_dims(attached, dims))


def string_dims(attached: AttachedSources, dims: Sequence[str]) -> list[str]:
    """Those of *dims* attaching encoded as ``Enum`` — its string ones."""
    return [d for d in dims if attached.is_enum_encoded(d)]


def binding(model: BuiltModel, result: Result, tolerance: float) -> pl.DataFrame:
    """Every bound and row *result* sits on — ``(kind, name, dims…, side)``.

    A variable is on its ``lower`` or ``upper`` bound, a row on the side its
    sense closes — ``lower`` for ``>=``, ``upper`` for ``<=``, ``equal`` for
    ``==`` — when the value is within *tolerance*, relative to the bound, of
    it. An infinite bound is never sat on. Dims come back as the union of
    every declaration's, null where a declaration does not carry one.
    """
    frames: list[pl.DataFrame] = []
    for name in model.program.variables:
        held = model.variables[name]
        lower, upper = held.share(model.cols['lb']), held.share(model.cols['ub'])
        frames += _sides('variable', name, result.primal(name), lower, upper, tolerance)
    for name in model.program.constraints:
        held = model.constraints[name]
        sense = held.share(model.rows['sense']).cast(pl.String)
        rhs = held.share(model.rows['rhs'])
        lower = pl.when(sense.is_in(['>=', '=='])).then(rhs).otherwise(-math.inf)
        upper = pl.when(sense.is_in(['<=', '=='])).then(rhs).otherwise(math.inf)
        bounds = pl.select(lower=lower, upper=upper)
        frames += _sides('constraint', name, result.activity(name), bounds['lower'], bounds['upper'], tolerance)
    schema = {'kind': pl.String, 'name': pl.String, 'side': pl.String}
    return pl.concat([pl.DataFrame(schema=schema), *frames], how='diagonal')


def _sides(
    kind: str, name: str, values: pl.DataFrame, lower: pl.Series, upper: pl.Series, tolerance: float
) -> list[pl.DataFrame]:
    """The rows of *values* sitting on *lower* or *upper*, each side its own frame, ``equal`` where the two coincide."""
    dims = [d for d in values.columns if d != 'value']
    valued = values.with_columns(lower.alias('__lower'), upper.alias('__upper'))
    on_lower = (pl.col('value') - pl.col('__lower')).abs() <= tolerance * (1 + pl.col('__lower').abs())
    on_upper = (pl.col('value') - pl.col('__upper')).abs() <= tolerance * (1 + pl.col('__upper').abs())
    side = (
        pl.when(pl.col('__lower') == pl.col('__upper'))
        .then(pl.lit('equal'))
        .when(on_lower & pl.col('__lower').is_finite())
        .then(pl.lit('lower'))
        .when(on_upper & pl.col('__upper').is_finite())
        .then(pl.lit('upper'))
        .otherwise(None)
    )
    tight = valued.with_columns(side.alias('side')).filter(pl.col('side').is_not_null())
    if tight.is_empty():
        return []
    return [tight.select(pl.lit(kind).alias('kind'), pl.lit(name).alias('name'), *dims, 'side')]


def expression_dims(name: str, expr: program.ExpressionNode, compiler: PolarsCompiler) -> tuple[str, ...]:
    """The dims named expression *expr* comes back over, in declaration order.

    What :func:`expression_frame` lays a value out over, answered without a
    primal: the union of what the expression's fragments carry, which is the
    frame the language proved it spans.
    """
    compiled = compiler.expression(expr, f"named expression '{name}'")
    return _fragment_dims(compiler, (*compiled.terms, *compiled.consts))


def _fragment_dims(compiler: PolarsCompiler, fragments: Sequence[TermFragment]) -> tuple[str, ...]:
    union = {d for p in fragments for d in p.dims}
    return tuple(d for d in compiler.program.dimensions if d in union)


def expression_frame(
    name: str, expr: program.ExpressionNode, compiler: PolarsCompiler, values: pl.LazyFrame
) -> pl.DataFrame:
    """Named expression *expr* evaluated at the primal *values* — ``(dims…, value)``.

    A value is ``sum(coeff · value)`` over the expression's term stream plus
    its constant part: each fragment from :meth:`PolarsCompiler.expression` is
    joined to the solver's primal vector (a term) or taken as it is (a
    constant part), aggregated to its own dims, and accumulated over the
    expression's coordinate product the way a constraint's right-hand side is.

    The frame answers the way a constraint over the same expression would: a
    coordinate a parameter does not cover contributes zero, a coordinate where
    a term's variable is absent has no row, and a variable-free expression is
    one row of ``value``. Dims come back in declaration order and rows in
    label order over those dims.

    Raises:
        DataError: A divisor with no value where the expression divides —
            checked before any sum can read the null as zero.
    """
    context = f"named expression '{name}'"
    compiled = compiler.expression(expr, context)
    fragments = (*compiled.terms, *compiled.consts)
    dims = _fragment_dims(compiler, fragments)

    divisors = sorted(program.divisor_parameters(expr))
    if divisors:
        counts = pl.collect_all([p.frame.select(pl.col(p.value_column).null_count()) for p in fragments])
        undefined = sum(count.item() for count in counts)
        if undefined:
            raise DataError(f'{context}: {sparse_divisor_message(", ".join(divisors), undefined)}')

    restrictions = absence_restrictions(list(compiled.terms))
    carrier = labels.frame(compiler, dims, None, _EXPRESSION_ROW, 0, restrictions).lazy()

    total = pl.lit(0.0, dtype=pl.Float64)
    for i, p in enumerate(fragments):
        column = f'__piece {i}__'
        if p.kind != 'const':
            valued = p.frame.join(values, on='var_label', how='left').select(
                *p.dims, (pl.col('coeff') * pl.col(SOLUTION)).alias(column)
            )
        else:
            valued = p.frame.select(*p.dims, pl.col('cval').alias(column))
        aggregated = (
            valued.group_by(p.dims).agg(pl.col(column).sum()) if p.dims else valued.select(pl.col(column).sum())
        )
        carrier = join_on(carrier, aggregated, p.dims, 'left')
        total = total + pl.col(column).fill_null(0.0)

    out = carrier.select(_EXPRESSION_ROW, *dims, total.alias('value')).collect(engine='streaming')
    ordered = labels.in_position_order(out, _EXPRESSION_ROW).drop(_EXPRESSION_ROW)
    return ordered.with_columns(pl.col(d).cast(pl.String) for d in string_dims(compiler.data, dims))
