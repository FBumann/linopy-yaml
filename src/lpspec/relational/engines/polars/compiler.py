"""Logical plan → polars. Lazy: nothing is read, nothing is executed.

`lowering.py` compiles the AST to a plan; this compiles the plan to a query, so
docs/about/architecture.md's admissibility test is a ``.explain()`` away. An identifier is
a value here, never syntax.

Column conventions, relied on by the engine:

===================  ==========================================
frame                columns
===================  ==========================================
dimension table      ``val``, ``ord``, plus declared lookups
parameter table      ``dims…``, ``value``
variable frame       ``dims…``, ``var_label``
term fragment        ``dims…``, ``var_label``, ``coeff``
quad fragment        ``dims…``, ``var_label``, ``var_label_2``, ``coeff``
const fragment       ``dims…``, ``cval``
===================  ==========================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, NoReturn

import polars as pl

from lpspec.errors import (
    DataError,
    LaneError,
    LanguageError,
    LpspecError,
)
from lpspec.relational import plan

if TYPE_CHECKING:
    import datetime
    from collections.abc import Callable, Mapping, Sequence

    from polars._typing import JoinStrategy, MaintainOrderJoin

    from lpspec.relational.engines.polars.binding import BoundSources


#: Scratch columns. The spaces make them unrepresentable as declared names, so
#: they cannot collide with a dimension or lookup the model already has.
_RHS = '__rhs value__'
#: The per-entity offset, joined in beside the ordinal it moves.
_OFFSET = '__offset'
_LAG = '__lag'
_WIDTH = '__width'
_ORD_IN = '__ord in__'
_ORD_OUT = '__ord out__'
_POS = '__pos in group__'
_SPAN = '__group size__'

#: Carries the single row of the empty coordinate product. Polars cannot hold a
#: frame with one row and no columns — collecting one reports ``(0, 0)`` — so the
#: unit needs a column to exist in, and every path drops it by selecting the
#: dims and the label instead.
UNIT = '__unit__'

#: The same trick for a *scalar* declaration's presence frame. It is distinct
#: from :data:`UNIT` rather than reusing it because the two meet: the empty
#: coordinate product already carries ``UNIT``, and the restriction below cross
#: joins a presence into it.
PRESENT = '__present__'


@dataclass(frozen=True)
class Presence:
    """Where the *variable* under a fragment exists, and what keys it.

    Not which rows the fragment's frame has. A fragment loses rows for two
    unrelated reasons and a constraint row reacts to only one: a **masked
    variable** is genuinely absent, a **sparse parameter** is a compressed
    dense array whose missing rows mean a zero coefficient (the data-binding rules).
    Multiplied together the frame cannot tell them apart, so the variable's
    coordinates ride alongside.

    ``keyed_by`` is ``None`` where the frame is keyed by the fragment's own
    dims — the implied key survives the dims being rewritten downstream (a
    product broadcasts, a sum drops) where a stated one would silently become
    a claim about columns the frame never had. Only an acyclic ``shift``
    states it: the coordinates it vacates are an edge along *one* dimension,
    so the restriction is one column wide however many dims the fragment
    carries — where keying it by the fragment's dims would materialise the
    whole coordinate product to name an edge.
    """

    frame: pl.LazyFrame
    keyed_by: tuple[str, ...] | None = None

    def keys(self, fragment_dims: tuple[str, ...]) -> tuple[str, ...]:
        """The columns this presence restricts by, for a fragment over *fragment_dims*."""
        return self.keyed_by if self.keyed_by is not None else fragment_dims

    def restrict(self, frame: pl.LazyFrame, on: Sequence[str]) -> pl.LazyFrame:
        """Keep only the rows of *frame* this presence admits.

        Keyed by *on* this is a semi-join. A **scalar** declaration has no
        key, its presence being at most one row saying only whether it
        exists, so the question becomes a cross join: every row survives a
        present scalar and none survives an absent one — absence spreading
        through arithmetic (the operator rules) at no dimension.
        """
        if on:
            return frame.join(self.frame.select(list(on)), on=list(on), how='semi')
        return frame.join(self.frame.select(PRESENT), how='cross').drop(PRESENT)


#: What a fragment is a piece *of*. ``term`` and ``quad`` differ only in how
#: many label columns the coefficient multiplies, which is why the shape
#: operators read :attr:`TermFragment.carried` rather than branching.
Kind = Literal['term', 'quad', 'const']


@dataclass(frozen=True)
class TermFragment:
    """One additive piece of a compiled expression.

    Terms yield ``(dims…, var_label, coeff)``, quadratic terms
    ``(dims…, var_label, var_label_2, coeff)`` and const parts
    ``(dims…, cval)``. An LP row *is* a sum of pieces, so every shape operator
    rewrites one.
    """

    dims: tuple[str, ...]
    frame: pl.LazyFrame
    kind: Kind

    presences: tuple[Presence, ...] = ()
    """Where the variables under this fragment exist — see :class:`Presence`.

    Empty is nothing to report: a constant fragment has no variable, and a
    reduction clears it, ``sum`` skipping absent slots rather than propagating
    them (v1 ``convention.rst`` §13).

    A **tuple**, because a quadratic term stands on two variables and is absent
    where either is. Joining two differently-keyed coordinate sets into one
    frame would materialise a product to say what both halves already say, so
    they travel side by side and each consumer applies them in turn.
    """

    @property
    def value_column(self) -> str:
        """``coeff`` where a variable is under it, ``cval`` otherwise."""
        return value_column(self.kind)

    @property
    def carried(self) -> list[str]:
        """The non-dim columns a projection has to keep."""
        return carried_columns(self.kind)


def _carries_cases(expression: plan.Expression) -> bool:
    """Whether a cased expression appears anywhere under *expression*."""
    if isinstance(expression, plan.Cases):
        return True
    return any(_carries_cases(child) for child in plan.children(expression))


def _refuse_a_cased_operand(context: str, position: str) -> NoReturn:
    """Refuse a cased expression where this lane needs a single fragment.

    ``a / b`` inverts *one* fragment and ``a ** b`` raises one to another. A
    cased expression compiles to one fragment per arm — disjoint, so their sum
    is the value — and there is no single frame to invert or exponentiate
    without first folding the arms together, which is a join per arm to
    reconstruct what the partition already guarantees.

    Asked of the *node* rather than of the fragment count, which is what tells
    this apart from an operand that **adds**: that one the language already
    refuses at load, and the assertion downstream is its plan-boundary
    backstop.

    A language error would be the wrong class here: the file is inside the
    language and the eager lane evaluates it, so this is the relational lane's
    shortfall and says so.
    """
    raise LaneError(
        f'in {context}: {position} is a cased expression, and this lane cannot build that. Each case '
        f'compiles to its own frame, and dividing or exponentiating needs one — the arms would have to '
        f'be folded into a single frame first. Inline the case you mean, or name the quotient as its '
        f'own cased expression so the division happens inside each arm. The eager lane builds the file '
        f'as written, so only this lane is short — run it with `lpspec.linopy.build`.'
    )


def _refuse_a_fragment_without_the_dims(p: TermFragment, dims: list[str], context: str, operator: str) -> NoReturn:
    """Refuse a fragment an operator cannot act on, in the right class.

    Two different failures share this shape and must not share a class. A
    **constant part** lacking the dims is a file the language accepts and the
    eager lane builds — `check` passes, so `LanguageError` would be a lie — and
    it is reachable from ordinary YAML wherever a scalar is added beside a term
    (#1137). A **term** lacking them is not reachable that way: `dims_of` gives
    every term the foreach dims at load, so reaching here means the plan is
    malformed, which is the lane's own business and stays `LanguageError`
    pending #1134.

    *operator* is the surface spelling, not the plan node: the reader wrote
    ``sum(by=…)``, and ``GroupSum`` is a word their file does not contain.
    """
    if p.kind == 'const':
        raise LaneError(
            f'in {context}: {operator} acts along {dims}, which a constant part of the expression '
            f'does not carry, and this lane cannot build that. A constant part compiles to its own '
            f'frame, so a fragment with no rows for {dims} has no slots for the operator to act on — '
            f'and under a mask, which slots those are is known only to the rows. Declare the parameter '
            f'over {dims} and supply it there: the model is the same and the number is unchanged. '
            f'The eager lane builds the file as written, so only this lane is short — run it with '
            f'`lpspec.linopy.build` (#1137).'
        )
    raise LanguageError(f'in {context}: {operator} along {dims} but the expression has dims {list(p.dims)}')


#: The label columns each kind carries, in the order a projection keeps them.
_LABELS: dict[Kind, list[str]] = {'term': ['var_label'], 'quad': ['var_label', 'var_label_2'], 'const': []}


def value_column(kind: Kind) -> str:
    """The value column a fragment of this kind carries.

    A free function as well as a :class:`TermFragment` property because
    :func:`_join_mul` names the columns of the fragment it is *building*, whose
    kind need not be either operand's.
    """
    return 'cval' if kind == 'const' else 'coeff'


def carried_columns(kind: Kind) -> list[str]:
    """The non-dim columns a projection of this fragment kind has to keep."""
    return [*_LABELS[kind], value_column(kind)]


class _Carrier:
    """A frame a walk joins onto, each attachment made at most once.

    Both walks that read parameters — the mask (:meth:`PolarsCompiler._predicate`)
    and the bounds (:meth:`PolarsCompiler.bounds`) — build an expression over
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


@dataclass(frozen=True)
class CompiledExpression:
    """An expression as fragments: variable terms, quadratic terms, a constant part.

    Three tuples rather than one keyed by kind, because every consumer wants a
    different subset of them and wants it named: a constraint row takes terms
    and constants and refuses quadratics outright, the objective takes all
    three, and the reader of a named expression takes the affine two.
    """

    terms: tuple[TermFragment, ...]
    consts: tuple[TermFragment, ...]
    quads: tuple[TermFragment, ...] = ()


def _defined(col: pl.Expr, dtype: str) -> pl.Expr:
    """What a bare parameter name in a ``where`` asks of *col*.

    Three readings, and the declaration picks: a ``bool`` is its own answer, a
    ``str`` is defined wherever the table has a row, and a number has to be
    finite as well. Read off the declaration rather than the column, which is
    the same thing since binding refuses a column that is not what the file
    declared — and unlike the column it cannot be ``is_finite`` over strings,
    which polars refuses outright.
    """
    if dtype == 'bool':
        return col.is_not_null() & col.cast(pl.Boolean)
    if dtype == 'str':
        return col.is_not_null()
    return col.is_not_null() & col.is_finite()


@dataclass(frozen=True)
class PolarsCompiler:
    """Turn plan nodes into polars queries over the model's tidy frames.

    ``data`` is everything binding produced, frozen. ``variables`` is
    deliberately outside it — the engine's own dict, not a copy, because a
    variable frame appears while its declaration is built and a constraint
    compiled afterwards has to see it.
    """

    program: plan.Program
    data: BoundSources
    variables: Mapping[str, pl.LazyFrame]

    # ------------------------------------------------------------------
    # frames — the masked coordinate product a declaration is instantiated over
    # ------------------------------------------------------------------

    @property
    def name_dims(self) -> dict[str, tuple[str, ...]]:
        """The dims each name in a where is read through.

        Parameters by their ``dims`` and variables by their ``foreach``. One
        flat mapping, because the language has one flat namespace and the two
        cannot collide.
        """
        named: dict[str, tuple[str, ...]] = {p.name: p.dims for p in self.program.parameters}
        named.update({v.name: v.dims for v in self.program.variables})
        return named

    def frame(self, dims: tuple[str, ...], where: plan.Predicate | None) -> pl.LazyFrame:
        """The masked coordinate product over *dims*.

        Labels, plus the ordinals a caller sorts by so labels follow
        declaration order.

        **A mask that has to join restricts by semi-join, not by value join.**
        The predicate reads only its own dims, so it is evaluated over *their*
        product and the full product is semi-joined against the truth set: the
        mask's parameter columns never touch the full product, and a semi-join
        leaves the left side's row order alone where a value join + filter does
        not — which keeps labelling's verify-then-sort a verify.

        Four shapes stay on the direct filter path, which is pointwise and
        keeps order too: a predicate that joins nothing (``_predicate`` hands
        the carrier back unchanged), one reading no frame dim, one reading dims
        outside the frame (so errors name the full frame), and one reading
        **every** frame dim — where the truth set is as wide as the product and
        the semi-join would build it twice to save no width. `sector`'s balance
        mask is that shape, and the semi-join there was a measurable share of
        the whole pipeline (#520).
        """
        out = self._coordinate_product(dims)
        if where is None:
            return out
        carrier, condition = self._predicate(out, where, dims)
        if carrier is out:
            return out.filter(_falsy_if_null(condition))
        touched = predicate_dims(where, self.name_dims)
        on = tuple(d for d in dims if d in touched)
        if on and len(on) < len(dims) and touched <= set(dims):
            keyed, keyed_condition = self._predicate(self._coordinate_product(on), where, on)
            surviving = keyed.filter(_falsy_if_null(keyed_condition)).select(*on)
            return out.join(surviving, on=list(on), how='semi')
        return carrier.filter(_falsy_if_null(condition))

    def _coordinate_product(self, dims: tuple[str, ...]) -> pl.LazyFrame:
        """Cross join of the dim tables: labels and ordinals, nothing else.

        **Folded in reverse, then projected back.** polars' streaming engine
        walks a cross join right-major, so folding backwards makes the product
        arrive in declaration row-major order — label order, which is what lets
        labelling and ``cols`` be read positionally instead of sorted (#433).
        :func:`labels.frame` verifies that rather than trusting it, so the fold
        decides speed, never correctness; the projection restores the column
        order the fold reversed.

        The empty product is one *real* row carrying only :data:`UNIT`: a
        ``where`` on a scalar declaration filters this frame, and nothing
        survives a filter.
        """
        out: pl.LazyFrame | None = None
        for d in reversed(dims):
            table = self.data.dimensions[d].select(pl.col('val').alias(d), pl.col('ord').alias(ordinal(d)))
            out = table if out is None else out.join(table, how='cross')
        if out is None:
            return pl.LazyFrame({UNIT: [0]})
        return out.select(*(c for d in dims for c in (d, ordinal(d))))

    def parameter_join(
        self,
        frame: pl.LazyFrame,
        param: str,
        frame_dims: tuple[str, ...],
        alias: str,
        subject: str,
        how: JoinStrategy = 'left',
        maintain_order: MaintainOrderJoin | None = None,
    ) -> pl.LazyFrame:
        """Join *param* onto *frame*, its value column renamed to *alias*.

        A parameter carrying a dim the frame lacks would be reduced over it,
        widening a mask or picking an arbitrary bound, so that is refused;
        *subject* is the caller's word for the declaration to name.

        *how* is ``left`` for a bound, where a missing value is a fact to
        report rather than a row to drop. ``inner`` is :meth:`_predicate`'s
        story.

        *maintain_order* is asked for only by the bounds, which become ``cols``
        and are read in order. Asking the join for that order costs an order of
        magnitude less than sorting the same frame afterwards (#433), so it is
        passed deliberately rather than defaulted on; every other consumer
        verifies order where it reads.
        """
        declaration = self.program.parameter(param)
        extra = set(declaration.dims) - set(frame_dims)
        if extra:
            raise LanguageError(f'{subject} has dims {sorted(extra)} outside the foreach dims {list(frame_dims)}')
        table = self.data.parameters[param].rename({'value': alias})
        if not declaration.dims:
            return frame.join(table, how='cross', maintain_order=maintain_order)
        return frame.join(table, on=list(declaration.dims), how=how, maintain_order=maintain_order)

    # ------------------------------------------------------------------
    # predicates (where masks — row absence)
    # ------------------------------------------------------------------

    def _predicate(
        self, frame: pl.LazyFrame, pred: plan.Predicate, dims: tuple[str, ...]
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

        ``VariableDefined`` is the one atom answered by a join rather than a
        column test — existence lives in the variable's own frame — keyed by
        dims the dim rule has already checked are inside this frame.

        No join here maintains order: consumers verify where they read
        (:func:`labels.in_position_order`), so a shuffle costs a sort
        downstream at worst, never a wrong label.
        """
        certain = _certain_parameters(pred)
        carrier = _Carrier(frame)

        def join_param(param: str) -> str:
            how: JoinStrategy = 'inner' if param in certain else 'left'
            return carrier.once(
                f'__where {param}__',
                lambda f, alias: self.parameter_join(f, param, dims, alias, f"where-parameter '{param}'", how),
            )

        def join_ordinal(dimension: str) -> str:
            if dimension not in dims:
                raise LanguageError(
                    f"where-comparison on dimension '{dimension}' is outside the foreach dims "
                    f'{list(dims)} — reducing a mask over an unlisted dim is not supported'
                )
            return carrier.once(
                f'__where ord {dimension}__',
                lambda f, alias: f.join(
                    self.data.dimensions[dimension].select(pl.col('val').alias(dimension), pl.col('ord').alias(alias)),
                    on=dimension,
                    how='left',
                ),
            )

        def join_group_offset(p: plan.DimensionPosition) -> str:
            """One column: the row's ordinal minus its own group's target ordinal."""
            if p.dimension not in dims:
                raise LanguageError(
                    f"where-comparison on dimension '{p.dimension}' is outside the foreach dims "
                    f'{list(dims)} — reducing a mask over an unlisted dim is not supported'
                )
            table = self._partitioned(p.dimension, str(p.by))
            _refuse_short_groups(p, table)
            group = pl.col(str(p.by))
            within = pl.col('ord').rank('ordinal').over(group).cast(pl.Int64) - 1
            size = pl.len().over(group).cast(pl.Int64)
            target = pl.lit(p.position) if p.position >= 0 else size + p.position
            offset = within - target
            return carrier.once(
                f'__where ord {p.dimension} by {p.by}__',
                lambda f, alias: f.join(
                    table.select(pl.col('val').alias(p.dimension), offset.alias(alias)),
                    on=p.dimension,
                    how='left',
                ),
            )

        def join_lookup(lookup: str, over: str) -> str:
            if over not in dims:
                raise LanguageError(
                    f"where-comparison on lookup '{lookup}' reads dimension '{over}', which is "
                    f'outside the foreach dims {list(dims)} — reducing a mask over an unlisted '
                    f'dim is not supported'
                )
            return carrier.once(
                f'__where lookup {lookup}__',
                lambda f, alias: f.join(
                    self.data.lookups[lookup].select(pl.col(over), pl.col(lookup).alias(alias)),
                    on=over,
                    how='left',
                ),
            )

        def walk(p: plan.Predicate) -> pl.Expr:
            if isinstance(p, plan.ParameterComparison):
                return _compare(pl.col(join_param(p.parameter)), p.op, p.value)
            if isinstance(p, plan.DimensionComparison):
                if p.dimension not in dims:
                    raise LanguageError(
                        f"where-comparison on dimension '{p.dimension}' is outside the foreach dims "
                        f'{list(dims)} — reducing a mask over an unlisted dim is not supported'
                    )
                return _compare(_dimension_column(p.dimension, p.value), p.op, p.value)
            if isinstance(p, plan.DimensionPosition):
                if p.by is not None:
                    return _falsy_if_null(_COLUMN_COMPARISONS[p.op](pl.col(join_group_offset(p)), pl.lit(0)))
                at = _position_ordinal(p, self.data.cardinality[p.dimension])
                return _COLUMN_COMPARISONS[p.op](pl.col(join_ordinal(p.dimension)), pl.lit(at))
            if isinstance(p, plan.LookupComparison):
                column = pl.col(join_lookup(p.lookup, p.over))
                if isinstance(p.value, str):
                    column = column.cast(pl.String)
                return _compare(column, p.op, p.value)
            if isinstance(p, plan.LookupPairComparison):
                left = pl.col(join_lookup(p.lookup, p.over))
                right = pl.col(join_lookup(p.other, p.over))
                return _COLUMN_COMPARISONS[p.op](left, right)
            if isinstance(p, plan.LookupDefined):
                return pl.col(join_lookup(p.lookup, p.over)).is_not_null()
            if isinstance(p, plan.ParameterDefined):
                return _defined(pl.col(join_param(p.parameter)), self.program.parameter(p.parameter).dtype)
            if isinstance(p, plan.VariableDefined):
                on = list(self.program.variable(p.variable).dims)
                coordinates = self.variables[p.variable].select(*on)
                if p.variable in certain:
                    carrier.once(f'__where defined {p.variable}__', lambda f, _: f.join(coordinates, on=on, how='semi'))
                    return pl.lit(value=True)
                flag = carrier.once(
                    f'__where defined {p.variable}__',
                    lambda f, alias: f.join(
                        coordinates.unique().with_columns(pl.lit(value=True).alias(alias)), on=on, how='left'
                    ),
                )
                return _falsy_if_null(pl.col(flag))
            if isinstance(p, plan.BooleanConstant):
                return pl.lit(value=p.value)
            if isinstance(p, plan.And):
                return walk(p.left) & walk(p.right)
            if isinstance(p, plan.Or):
                return walk(p.left) | walk(p.right)
            if isinstance(p, plan.Not):
                return ~_falsy_if_null(walk(p.operand))
            raise LanguageError(f'unsupported predicate node {type(p).__name__}')

        condition = walk(pred)
        return carrier.frame, condition

    # ------------------------------------------------------------------
    # bounds
    # ------------------------------------------------------------------

    def bounds(self, frame: pl.LazyFrame, v: plan.VariableDeclaration) -> pl.LazyFrame:
        """*frame* with ``lb``/``ub`` columns for variable *v*.

        Joins and arithmetic are one object, so a bound cannot be evaluated
        against a frame missing what it reads.
        """
        carrier = _Carrier(frame)

        def attach_bound(f: pl.LazyFrame, alias: str, name: str) -> pl.LazyFrame:
            aligned = self._aligned_bound(f, name, v, alias)
            if aligned is not None:
                return aligned
            subject = f"bound parameter '{name}' of variable '{v.name}'"
            return self.parameter_join(f, name, v.dims, alias, subject, maintain_order='left')

        def walk(e: plan.Expression) -> pl.Expr:
            if isinstance(e, plan.Constant):
                return pl.lit(float(e.value), dtype=pl.Float64)
            if isinstance(e, plan.Parameter):
                alias = carrier.once(f'__bound {e.name}__', lambda f, a: attach_bound(f, a, e.name))
                return pl.col(alias).cast(pl.Float64)
            if isinstance(e, plan.Negate):
                return -walk(e.operand)
            if isinstance(e, plan.Add):
                return walk(e.left) + walk(e.right)
            if isinstance(e, plan.Multiply):
                return walk(e.left) * walk(e.right)
            raise LanguageError(
                f"unsupported node {type(e).__name__} in bounds of variable '{v.name}' "
                f'(bounds must be variable-free arithmetic over Constant/Parameter)'
            )

        lower, upper = walk(v.lower), walk(v.upper)
        return carrier.frame.with_columns(lower.alias('lb'), upper.alias('ub'))

    def _aligned_bound(
        self, frame: pl.LazyFrame, param: str, v: plan.VariableDeclaration, alias: str
    ) -> pl.LazyFrame | None:
        """*frame* with *param* attached **by position**, or ``None`` to join.

        A bound dense over the whole variable product — a profile per node per
        technology per hour, the ordinary shape in energy modelling — costs a
        full-size join against a full-size coordinate product here, where the
        eager lane gets it free from array position — on `profiled` that join
        was most of the build (#511).

        Position means the coordinate here too. The label frame is row-major
        over the product in each dimension's own index order (*not*
        lexicographic dim order, which is why sorting by the dim columns does
        not match), so each parameter row's slot is computed from its own
        labels' ordinals and its value scattered there. **The table's row order
        is nothing**: which slot a row lands in is decided by the coordinate it
        carries, and ``_scattered`` refuses a product any slot of which nothing
        wrote.

        **Wrong bounds are a wrong model with no error**, so this is refused
        unless all three hold, each a fact already computed:

        * the parameter's dims are exactly the variable's, in the same order —
          fewer broadcast, more is already refused, a different order is a
          different row-major walk
        * the variable declares no ``where`` — a mask makes the label frame a
          subset of the product and position stops lining up
        * the parameter is dense over that product, its height equal to the
          product of the cardinalities binding cached

        Duplicate coordinates would break density without changing the height,
        and are refused before this by ``check_one_row_per_coordinate``.
        """
        declaration = self.program.parameter(param)
        if v.where is not None or tuple(declaration.dims) != tuple(v.dims) or not v.dims:
            return None

        cards = [self.data.cardinality[d] for d in v.dims]
        expected = math.prod(cards)
        table = self.data.parameters[param]
        if table.select(pl.len()).collect().item() != expected:
            return None

        stride = 1
        strides: list[int] = []
        for card in reversed(cards):
            strides.insert(0, stride)
            stride *= card
        position = sum(
            (self.ordinal_of(d) * step for d, step in zip(v.dims, strides, strict=True)),
            start=pl.lit(0, dtype=pl.Int64),
        )
        pairs = table.select(position.alias('__at__'), pl.col('value')).collect(engine='streaming')
        return frame.with_columns(pl.Series(alias, _scattered(pairs['__at__'], pairs['value'], expected)))

    def ordinal_of(self, dim: str) -> pl.Expr:
        """A *dim* value column as that dimension's ordinal.

        **Free for a string dimension**, which is most of them: binding encodes
        those as an ``Enum`` over the labels in ordinal order
        (``_Binder.encode_dimensions``), so the physical code already *is* the
        ordinal. Every other dtype pays a dictionary built from the dimension
        table — one entry per label, not per row.
        """
        column = pl.col(dim)
        if self.data.is_enum_encoded(dim):
            return column.to_physical().cast(pl.Int64)
        labels = self.data.dimensions[dim].select('val').collect()['val']
        return column.replace_strict({value: at for at, value in enumerate(labels)}, return_dtype=pl.Int64)

    # ------------------------------------------------------------------
    # expressions → fragments
    # ------------------------------------------------------------------

    def expression(self, expr: plan.Expression, context: str, *, quadratic: bool = False) -> CompiledExpression:
        """Compile an expression into term, quadratic and const fragments.

        *quadratic* is the position's ceiling, passed by the caller that knows
        it: the objective can hold a product of two variables and a constraint
        row cannot. The language has already refused what it refuses
        (``language/degree.py``), so this is the **plan-boundary backstop** —
        a degree-2 node arriving by any other route dies here rather than
        becoming a term whose second variable is silently dropped.

        No join in the walk maintains order, on evidence rather than by
        omission: the mul join's ``maintain_order`` holds the label order on
        some shapes and loses it on others differing only in data, so the tax
        lands unpredictably and `profiled/l`'s objective phase triples paying
        it for nothing. Every consumer re-derives or verifies order where it
        reads (:meth:`PolarsEngine._build_objective`'s docstring keeps the
        numbers).
        """

        def product(a: CompiledExpression, b: CompiledExpression) -> CompiledExpression:
            """``a * b``, distributed over both operands' fragment lists.

            Every pairing is formed and each is formed once, **including both
            mixed products**: where the two factors each carry a variable and a
            constant part, ``a.terms`` against ``b.consts`` and ``b.terms``
            against ``a.consts`` are different terms of the model, and dropping
            either answers something else. Degree 3 is refused rather than
            represented — a quadratic fragment times a term has nowhere to put
            the third label.
            """
            if (a.quads and b.terms) or (b.quads and a.terms) or (a.quads and b.quads):
                raise LanguageError(
                    f'in {context}: a product of degree 3 or more. A sink takes a quadratic form; '
                    f'nothing takes a cubic one.'
                )
            if a.terms and b.terms and not quadratic:
                raise LanguageError(f'nonlinear product in {context}: both factors contain variables')
            quads = tuple(_join_quad(t, u) for t in a.terms for u in b.terms)
            quads += tuple(_join_mul(q, c, 'quad') for q in a.quads for c in b.consts)
            quads += tuple(_join_mul(q, c, 'quad') for q in b.quads for c in a.consts)
            terms = tuple(_join_mul(t, c, t.kind) for t in a.terms for c in b.consts)
            terms += tuple(_join_mul(t, c, t.kind) for t in b.terms for c in a.consts)
            consts = tuple(_join_mul(x, c, 'const') for x in a.consts for c in b.consts)
            return CompiledExpression(terms, consts, quads)

        def quotient(a: CompiledExpression, b: CompiledExpression) -> CompiledExpression:
            """``a / b``, where *b* is one variable-free factor.

            That it is *one* is ``degree.check_binary``'s answer, given at load
            with no data bound, so a divisor that adds never reaches a plan. A
            **cased** divisor is refused a node earlier, in :func:`ev`, where
            the operand is still an expression and can be told apart from one
            that adds.
            """
            if b.terms or b.quads:
                raise LanguageError(f'nonlinear quotient in {context}: the divisor contains variables')
            assert len(b.consts) == 1, 'a divisor that adds is refused at load'
            inv = b.consts[0]
            terms = tuple(_join_mul(t, inv, t.kind, divide=True) for t in a.terms)
            quads = tuple(_join_mul(q, inv, 'quad', divide=True) for q in a.quads)
            consts = tuple(_join_mul(x, inv, 'const', divide=True) for x in a.consts)
            return CompiledExpression(terms, consts, quads)

        def power(a: CompiledExpression, b: CompiledExpression) -> CompiledExpression:
            """``a ** b``, where neither side carries a variable.

            The language refuses one that does (``language/degree.py``), so
            this is the plan-boundary backstop the other operators keep: a node
            arriving by any other route dies here rather than folding a
            variable's coefficient into a base.
            """
            if a.terms or a.quads or b.terms or b.quads:
                raise LanguageError(f'in {context}: a power over variables — `**` takes neither side variable')
            assert len(a.consts) == 1 and len(b.consts) == 1, 'a base or exponent that adds is refused at load'
            return CompiledExpression((), (_join_pow(a.consts[0], b.consts[0]),))

        def ev(e: plan.Expression) -> CompiledExpression:
            if isinstance(e, plan.Constant):
                frame = pl.LazyFrame({'cval': [float(e.value)]}, schema={'cval': pl.Float64})
                return CompiledExpression((), (TermFragment((), frame, 'const'),))
            if isinstance(e, plan.Parameter):
                return CompiledExpression((), (self._parameter_fragment(e.name),))
            if isinstance(e, plan.Variable):
                return CompiledExpression((self._variable_fragment(e.name),), ())
            if isinstance(e, plan.Negate):
                return _map_fragments(ev(e.operand), _negate)
            if isinstance(e, plan.Add):
                a, b = ev(e.left), ev(e.right)
                return CompiledExpression(a.terms + b.terms, a.consts + b.consts, a.quads + b.quads)
            if isinstance(e, plan.Multiply):
                return product(ev(e.left), ev(e.right))
            if isinstance(e, plan.Divide):
                if _carries_cases(e.divisor):
                    _refuse_a_cased_operand(context, 'a divisor')
                return quotient(ev(e.numerator), ev(e.divisor))
            if isinstance(e, plan.Power):
                if _carries_cases(e.base) or _carries_cases(e.exponent):
                    _refuse_a_cased_operand(context, 'a base or an exponent')
                return power(ev(e.base), ev(e.exponent))
            if isinstance(e, plan.Sum):
                inner = _propagate_absence(ev(e.operand))
                return _map_fragments(inner, lambda p: self._sum_fragment(p, e.over, context))
            if isinstance(e, plan.GroupSum):
                inner = _propagate_absence(ev(e.operand))
                return _map_fragments(inner, lambda p: self._group_fragment(p, e, context))
            if isinstance(e, plan.At):
                return _map_fragments(ev(e.operand), lambda p: self._at_fragment(p, e, context))
            if isinstance(e, plan.Translate):
                return _map_fragments(ev(e.operand), lambda p: self._translate_fragment(p, e, context))
            if isinstance(e, plan.Window):
                inner = _propagate_absence(ev(e.operand))
                return _map_fragments(inner, lambda p: self._window_fragment(p, e, context))
            if isinstance(e, plan.Cases):
                return self._cases(e, ev)
            raise LanguageError(f'unsupported expression node {type(e).__name__} in {context}')

        return ev(expr)

    def _parameter_fragment(self, name: str) -> TermFragment:
        """A parameter as a constant part, keyed by its declared dims.

        One row per coordinate, which the engine enforces by refusing a
        duplicated one.
        """
        dims = self.program.parameter(name).dims
        frame = self.data.parameters[name].select(*dims, pl.col('value').cast(pl.Float64).alias('cval'))
        return TermFragment(dims, frame, 'const')

    def _variable_fragment(self, name: str) -> TermFragment:
        """A variable as a term with unit coefficients.

        Presence is what makes absence *propagate*, and it is attached only
        where the declaration asks for it — decided before any data is read.
        Two declarations carry none: an unmasked variable, which exists at every
        coordinate of its foreach and could restrict nothing, and one declaring
        ``absence: zero``, whose missing coordinates hold a quantity that *is*
        zero rather than one with no value. Both then leave the term simply
        absent from the rows it does not reach, which is the same arithmetic —
        only the second had a choice about it.

        ``keyed_by`` is stated rather than left to its ``None`` default,
        because dims are rewritten downstream while the presence frame is not
        — the hazard :class:`Presence` names.
        """
        declaration = self.program.variable(name)
        dims = declaration.dims
        frame = self.variables[name].select(*dims, 'var_label', pl.lit(1.0, dtype=pl.Float64).alias('coeff'))
        propagates = declaration.where is not None and declaration.absence == 'undefined'
        presences = (Presence(self._presence(name, dims), dims),) if propagates else ()
        return TermFragment(dims, frame, 'term', presences=presences)

    def _presence(self, name: str, dims: tuple[str, ...]) -> pl.LazyFrame:
        """The coordinates a masked variable exists at.

        A **scalar** declaration has none, and ``select()`` over no dims is the
        empty frame polars cannot represent, so the marker column carries the
        one bit left: whether the row is there at all. It is renamed from a
        real column, never a ``pl.lit()`` — a select of literals alone is
        length 1 whatever it selects from, so an absent scalar would come back
        present.
        """
        frame = self.variables[name]
        if dims:
            return frame.select(*dims)
        return frame.select(pl.col('var_label').alias(PRESENT))

    # ------------------------------------------------------------------
    # shape operators — one dim rewritten per fragment
    # ------------------------------------------------------------------

    def _sum_fragment(self, p: TermFragment, over: tuple[str, ...], context: str) -> TermFragment:
        """Drop the summed dims — **not an aggregate**.

        The rows that carried them stay and collapse in the terminal
        ``sum(coeff)`` at assembly. Constructed rather than ``replace``d so
        ``presence`` is *dropped*: v1 §13 reads a reduction as skipping absent
        slots, so summing over a partly-masked dim reports nothing.
        """
        missing = [d for d in over if d not in p.dims]
        if missing and p.kind == 'const':
            _refuse_a_fragment_without_the_dims(p, missing, context, f'sum(over={list(over)})')
        keep = tuple(d for d in p.dims if d not in over)
        scale = math.prod(self.data.cardinality[d] for d in missing)
        frame = p.frame.select(*keep, *p.carried)
        if scale != 1:
            frame = frame.with_columns(pl.col(p.value_column) * scale)
        return TermFragment(keep, frame, p.kind)

    def _group_fragment(self, p: TermFragment, g: plan.GroupSum, context: str) -> TermFragment:
        """Relabel dim ``over`` to ``into`` through declared coordinates.

        No aggregate either: the dim table holds one row per label and its
        coordinates were checked for containment at build time, so the join
        neither duplicates nor drops a term, and rows landing on one ``into``
        are added by the terminal aggregate as ``Sum``'s are. A group is a sum,
        so v1 §13 applies and this constructs rather than ``replace``s — see
        :meth:`_sum_fragment`.

        Grouping through several coordinates costs nothing extra here: they
        ride the same dim table and the same single join, which is why the
        surface is a list rather than a composition of calls.
        """
        if g.over not in p.dims:
            _refuse_a_fragment_without_the_dims(p, [g.over], context, f'sum(by=) over {g.over!r}')
        grouped = self._remap_fragment(p, g, consumed=(g.over,), produced=g.into)
        if p.kind != 'const':
            return grouped
        return replace(grouped, frame=pl.concat([grouped.frame, self._empty_groups(grouped, g)]))

    def _mapping(self, over: str, coordinate: tuple[str, ...], into: tuple[str, ...]) -> pl.LazyFrame:
        """The ``(over, into…)`` table a group or a pullback joins against.

        One relation per coordinate, met on ``over`` by **inner** joins: a
        label some coordinate does not map has no row in that relation and so
        none here, which is what "reaches no slot" means for the whole tuple.
        Reading several at once therefore costs joins and no null bookkeeping —
        the tuple exists exactly where every coordinate does.

        The pairing is materialised before any frame is looked up, so a node
        built by hand with tuples of different lengths is refused by ``zip``
        rather than by a missing name (``plan`` is a public IR).
        """
        pairs = list(zip(coordinate, into, strict=True))
        mapping, *rest = (self.data.lookups[c].select(pl.col(over), pl.col(c).alias(i)) for c, i in pairs)
        for other in rest:
            mapping = mapping.join(other, on=over, how='inner')
        return mapping

    def _partitioned(self, dim: str, lookup: str) -> pl.LazyFrame:
        """*dim*'s ``(val, ord, lookup)``, only for labels the map places in a group.

        The inner join is where "this coordinate is in no group" now comes
        from: it has no row in the relation, so it has none here, and every
        rank, span and neighbour computed below sees only labels that are in
        one. What used to be a null group is an absent row.
        """
        rows = self.data.lookups[lookup].select(pl.col(dim).alias('val'), pl.col(lookup))
        return self.data.dimensions[dim].join(rows, on='val', how='inner')

    def _empty_groups(self, p: TermFragment, g: plan.GroupSum) -> pl.LazyFrame:
        """The ``into`` combinations no member maps to, as constant rows worth zero.

        A group with no members contributes nothing, so on a constant side it
        holds a *value* — the empty sum — and not a hole. The two are the same
        missing row to :meth:`PolarsEngine._build_constraint`'s coverage check,
        which reads what the fragment produced and cannot see why a label is
        absent, so the value is written down here where the reason is known
        (#1026).

        Several coordinates land on a *product* of targets, and a combination
        no member sits at is empty for the reason one unreached label is — so
        what the reached set is subtracted from is that product.

        Only for a constant part: an empty group contributes no *term*, and a
        row left with no terms is not built at all.
        """
        universe = self.data.dimensions[g.into[0]].select(pl.col('val').alias(g.into[0]))
        for target in g.into[1:]:
            labels = self.data.dimensions[target].select(pl.col('val').alias(target))
            universe = universe.join(labels, how='cross')
        reached = self._mapping(g.over, g.coordinate, g.into).select(*g.into)
        empty = universe.join(reached, on=list(g.into), how='anti')
        spanned = [d for d in p.dims if d not in g.into]
        if spanned:
            empty = p.frame.select(spanned).unique().join(empty, how='cross')
        return empty.with_columns(pl.lit(0.0, dtype=pl.Float64).alias('cval')).select(*p.dims, *p.carried)

    def _at_fragment(self, p: TermFragment, a: plan.At, context: str) -> TermFragment:
        """Spread ``into`` back out over ``over`` — the adjoint of a group.

        The same mapping table as :meth:`_group_fragment`, joined on the other
        columns, so the join **fans out**: one row per ``into`` tuple lands on
        every ``over`` sharing it. Still one equi-join against a table the
        frame holds, so the locality class does not move.

        A pullback duplicates a label — the same ``var_label`` at every fine
        coordinate of its component — so a later reduction can bring two copies
        into one row, where the terminal aggregate adds them.

        Unlike the group it shares that join with, it **reports absence**:
        pointwise, so what the fine coordinate has is whatever the coarse slot
        it reads has, and a slot with nothing has to take the row with it.
        """
        absent = [d for d in a.into if d not in p.dims]
        if absent:
            raise LanguageError(f'in {context}: At through {absent} but the expression has dims {list(p.dims)}')
        remapped = self._remap_fragment(p, a, consumed=a.into, produced=(a.over,))
        return replace(remapped, presences=self._pulled_back_presences(p, a))

    def _pulled_back_presences(self, p: TermFragment, a: plan.At) -> tuple[Presence, ...]:
        """Where a pullback's variables exist, keyed by the fine dim they now span.

        Two absences reach the fine coordinate and :meth:`_remap_fragment`'s
        inner join swallows both — the operand's own, where the variable under
        it was masked away, and the **lookup's**, where the map has no row for
        the label and there is no slot to read (the absence rules list it among
        the four constructs that create one). Unreported, the term merely vanishes and
        its row survives to assert something the model never said: `x <= 0`
        where the model said nothing (#968).

        Reading through several coordinates at once, a label reaches its slot
        only where *every* one of them maps, which is what :meth:`_mapping`'s
        inner joins already say.

        A total lookup over an operand with nothing to report yields nothing
        rather than a restriction admitting everything, so a model with no
        absence in it does not pay for the machinery that carries one. A
        quadratic fragment stands on two variables, so each of its presences is
        pulled back on its own.

        The key is stated rather than left implied because a later product
        widens the fragment's dims while this frame keeps the one column that
        matters — the hazard :class:`Presence` names.
        """
        reachable = self._mapping(a.over, a.coordinate, a.into)
        if not p.presences:
            total = self.data.cardinality[a.over] == reachable.select(pl.len()).collect().item()
            return () if total else (Presence(reachable.select(a.over), (a.over,)),)

        def pulled(presence: Presence) -> Presence:
            keys = presence.keys(p.dims)
            if not keys:
                return Presence(presence.restrict(reachable.select(a.over), keys), (a.over,))
            carries_targets = all(i in keys for i in a.into)
            source, keys = (
                (presence.frame, keys) if carries_targets else (self._widen(presence.frame, keys, p.dims), p.dims)
            )
            kept = tuple(k for k in keys if k not in a.into)
            return Presence(source.join(reachable, on=list(a.into), how='inner').select(*kept, a.over), (*kept, a.over))

        return tuple(pulled(x) for x in p.presences)

    def _remap_fragment(
        self, p: TermFragment, node: plan.GroupSum | plan.At, *, consumed: tuple[str, ...], produced: tuple[str, ...]
    ) -> TermFragment:
        """Trade dims *consumed* for *produced* through *node*'s coordinates.

        The mapping table is :meth:`_mapping` — the declared coordinates' own
        relations, keyed by ``over`` and named for the dims they target — and
        the rewrite is a single inner equi-join on *consumed*. A group consumes
        ``over`` (:meth:`_group_fragment`); an ``At`` reads the same table
        backwards (:meth:`_at_fragment`). Written once so the adjoints cannot
        drift: a change to how the mapping joins is a change to both.

        One of the two sides is always a single dim — a group consumes the one
        the coordinates are over, a pullback produces it — so exactly one join
        happens whatever the arity.
        """
        dropped = set(consumed)
        keep = tuple(x for x in p.dims if x not in dropped)
        mapping = self._mapping(node.over, node.coordinate, node.into)
        frame = p.frame.join(mapping, on=list(consumed), how='inner').select(*keep, *produced, *p.carried)
        return TermFragment((*keep, *produced), frame, p.kind)

    def _cases(
        self, e: plan.Cases, compile_value: Callable[[plan.Expression], CompiledExpression]
    ) -> CompiledExpression:
        """A cased value as the concatenation of its arms' restricted fragments.

        A :class:`CompiledExpression` is already a *sum* of fragments, and the
        arms partition the frame — so restricting each arm's fragments to the
        coordinates its ``when`` claims makes that sum a selection. Nothing
        downstream learns it was a case: an arm's terms are terms, and the
        terminal ``sum(coeff)`` adds one contribution per coordinate because
        only one arm ever reaches it.

        This is what the partition check buys. Without it two arms could reach
        one coordinate and the sum would silently be their total.
        """
        terms: list[TermFragment] = []
        consts: list[TermFragment] = []
        quads: list[TermFragment] = []
        for arm in e.arms:
            compiled = compile_value(arm.value)
            for into, fragments in ((terms, compiled.terms), (consts, compiled.consts), (quads, compiled.quads)):
                into.extend(f for p in fragments if (f := self._restricted(p, arm, e)) is not None)
        return CompiledExpression(tuple(terms), tuple(consts), tuple(quads))

    def _restricted(self, p: TermFragment, arm: plan.CaseArm, e: plan.Cases) -> TermFragment | None:
        """*p* widened to the whole declared frame, then cut to its arm's mask.

        Widening first is what makes the cut mean anything: an arm may be a
        scalar where its ``when`` is not — ``1`` for every non-committable unit
        — and a scalar fragment has no coordinate for the mask to remove.

        **To the whole of** ``e.dims``, not merely as far as the mask reaches.
        A cased expression's dims *are* its declared ``foreach`` — that is what
        ``dims_of`` answers upstream — so a fragment narrower than that is a
        fragment lying about the quantity's shape. Broadcasting the missing dims
        at the row join happens to give the right rows for a constraint, and the
        wrong answer everywhere the dim has to be *there*: ``sum(over=)`` finds
        no slots to act on, and reading the value back drops the axis the
        declaration promised.

        **A constant keeps a row outside its arm, holding zero; a term does
        not.** Both say "this arm contributes nothing here", but they are read
        by different halves of the engine. Constants are left-joined onto the
        rows and a missing one is a *gap* — the coverage check cannot tell an
        arm that does not apply from a parameter row nobody supplied, so the
        arm says so with a zero. Terms are inner-joined and summed, where an
        absent row already contributes nothing: a zero there would be an
        explicit zero coefficient handed to the solver for every arm that does
        not apply. Within the arm a missing value is still missing, and still
        reported.

        **An arm's absence is confined to the arm too**, which is
        :meth:`_arm_presence`'s job — a shift that vacates the first position
        says nothing about a coordinate its arm never claimed.

        ``None`` where the mask is dimensionless and false, which is the one
        arm with no coordinates to carry it.
        """
        touched = predicate_dims(arm.when, self.name_dims)
        claimed_dims = tuple(d for d in e.dims if d in touched)
        if not claimed_dims:
            claims_something = self.frame((), arm.when).select(pl.len()).collect().item()
            return p if claims_something else None
        needed = e.dims
        claimed = self.frame(claimed_dims, arm.when).select(*claimed_dims)
        frame = p.frame
        for d in needed:
            if d not in p.dims:
                frame = frame.join(self.data.dimensions[d].select(pl.col('val').alias(d)), how='cross')
        inside = frame.select(*needed, *p.carried).join(claimed, on=list(claimed_dims), how='semi')
        if p.kind != 'const':
            confined = tuple(self._arm_presence(x, x.keys(p.dims), claimed_dims, claimed, e) for x in p.presences)
            return TermFragment(needed, inside, p.kind, presences=confined)
        outside = (
            self._coordinate_product(needed)
            .select(*needed)
            .join(claimed, on=list(claimed_dims), how='anti')
            .with_columns(pl.lit(0.0, dtype=pl.Float64).alias(p.value_column))
        )
        return TermFragment(needed, pl.concat([inside, outside], how='vertical'), p.kind)

    def _arm_presence(
        self,
        x: Presence,
        keys: tuple[str, ...],
        claimed_dims: tuple[str, ...],
        claimed: pl.LazyFrame,
        e: plan.Cases,
    ) -> Presence:
        """*x*, widened so it admits every coordinate its arm does not claim.

        A presence deletes constraint rows, and it is collected across *all* a
        row's terms — so an arm's left unrestricted would speak for coordinates
        another arm defines. ``shift(soc, over=snapshot, offset=1)`` vacates the
        first snapshot, and under `cases:` that arm is masked off the first
        snapshot anyway: the row there belongs to the arm reading the initial
        level, and would be deleted by an edge that says nothing about it.

        So the admitted set is *the value exists* **or** *this arm does not
        claim the coordinate*, keyed by both spans at once — the one place a
        presence widens past the edge it names, and only for an arm that has
        one.
        """
        span = tuple(d for d in e.dims if d in keys or d in claimed_dims)
        product = self._coordinate_product(span).select(*span)
        unclaimed = product.join(claimed, on=list(claimed_dims), how='anti')
        if not keys:
            # a masked scalar variable: present everywhere or nowhere, so the
            # arm's own rows survive only in the first case
            exists = product if x.frame.select(pl.len()).collect().item() else product.clear()
        else:
            exists = product.join(x.frame, on=list(keys), how='semi')
        return Presence(pl.concat([exists, unclaimed], how='vertical').unique(), span)

    def _window_fragment(self, p: TermFragment, s: plan.Window, context: str) -> TermFragment:
        """A one-to-many remap of the dim through its ord.

        A row at *o* contributes at every ``o + lag`` for ``lag`` inside the
        window, so the terms land on each output position that can see them and
        the terminal ``sum(coeff)`` at assembly adds them up — the same trick
        :meth:`_sum_fragment` relies on, which is why this needs no aggregate.

        The lag table is built to the widest window the data asks for; a named
        width then keeps only the lags that entity reaches. Every join is still
        on a dim-table key or the width's own dims, so the reach stays a lookup
        and the locality class is the one :meth:`_translate_fragment` has.

        Unlike a shift this vacates nothing: the window at the first position
        is short rather than empty, since it always contains that position
        itself. So an operand with no presence gains none.
        """
        if s.dimension not in p.dims:
            _refuse_a_fragment_without_the_dims(p, [s.dimension], context, f'sum_back(over={s.dimension!r})')
        card = self.data.cardinality[s.dimension]
        table = self.data.dimensions[s.dimension]
        incoming = table.select(pl.col('val').alias(s.dimension), pl.col('ord').alias(_ORD_IN))
        outgoing = table.select(pl.col('val').alias(s.dimension), pl.col('ord').alias(_ORD_OUT))

        width_name = s.width if isinstance(s.width, str) else None
        width_dims: tuple[str, ...] = ()
        if width_name is not None:
            width_dims = tuple(self.program.parameter(width_name).dims)
            widest = int(self.data.parameters[width_name].select(pl.col('value').max()).collect().item() or 0)
        else:
            assert not isinstance(s.width, str)
            widest = s.width
        lags = pl.LazyFrame({_LAG: pl.Series(range(min(widest, card)), dtype=pl.Int64)})

        moved = pl.col(_ORD_IN) + pl.col(_LAG)
        if s.wrap:
            moved = moved % card

        def remap(source: pl.LazyFrame, carried: list[str], source_dims: tuple[str, ...] | None = None) -> pl.LazyFrame:
            kept = [d for d in (source_dims if source_dims is not None else p.dims) if d != s.dimension]
            walked = source.join(incoming, on=s.dimension, how='inner').drop(s.dimension).join(lags, how='cross')
            if width_name is not None:
                walked = walked.join(
                    self.data.parameters[width_name].select(*width_dims, pl.col('value').cast(pl.Int64).alias(_WIDTH)),
                    on=list(width_dims),
                    how='inner',
                ).filter(pl.col(_LAG) < pl.col(_WIDTH))
            return (
                walked.with_columns(moved.alias(_ORD_OUT))
                .join(outgoing, on=_ORD_OUT, how='inner')
                .select(*kept, s.dimension, *carried)
            )

        def travelled(presence: Presence) -> Presence:
            keyed_by, source = presence.keyed_by, presence.frame
            if keyed_by is not None and s.dimension not in keyed_by:
                source, keyed_by = self._widen(source, keyed_by, p.dims), None
            return Presence(remap(source, [], keyed_by).unique(), keyed_by)

        frame = remap(p.frame, p.carried)
        return TermFragment(p.dims, frame, p.kind, presences=tuple(travelled(x) for x in p.presences))

    def _translate_fragment(self, p: TermFragment, s: plan.Translate, context: str) -> TermFragment:
        """A pointwise remap of the dim through its ord.

        A row at *o* contributes at ``(o + by) % card``.

        Both joins are on a dim-table key, so the row count is unchanged and an
        out-of-range ordinal does not join. No window function; bounded-halo
        locality. The operand's *presences* are :func:`travelled_presences` below.

        Every fill over a *constant* is written, ``0`` included (#551): the
        arithmetic is unchanged, but the slot now has a value, so asking for
        zero stops being indistinguishable from having nothing. Over a *term*
        there is nothing to write — ``edge=0`` on a variable means the vacated
        slot contributes no term at all (the operator rules), where a zero-coefficient
        entry would be a matrix nonzero standing for a term that is not there.
        Lowering refuses every other numeric edge over a variable.
        """
        if s.dimension not in p.dims:
            _refuse_a_fragment_without_the_dims(p, [s.dimension], context, f'shift(over={s.dimension!r})')
        card = self.data.cardinality[s.dimension]
        others = [d for d in p.dims if d != s.dimension]
        table = self.data.dimensions[s.dimension]
        if s.partition is not None:
            # Inside a group the neighbour is decided by position within *that*
            # group, so both joins read a rank rather than the axis-wide `ord`
            # and a wrap closes on the group's own size. A coordinate the map
            # places nowhere is not in this table at all and joins to nothing,
            # which is what it reaches everywhere else.
            table = self._partitioned(s.dimension, s.partition)
            grouped = pl.col(s.partition)
            table = table.with_columns(
                (pl.col('ord').rank('ordinal').over(grouped) - 1).cast(pl.Int64).alias(_POS),
                pl.len().over(grouped).cast(pl.Int64).alias(_SPAN),
            )
        position = _POS if s.partition is not None else 'ord'
        group_cols = [pl.col(s.partition)] if s.partition is not None else []
        incoming = table.select(
            pl.col('val').alias(s.dimension),
            pl.col(position).alias(_ORD_IN),
            *group_cols,
            *([pl.col(_SPAN)] if s.partition is not None else []),
        )
        outgoing = table.select(pl.col('val').alias(s.dimension), pl.col(position).alias(_ORD_OUT), *group_cols)

        named_offset = isinstance(s.offset, str)
        if named_offset:
            moved = pl.col(_ORD_IN) + pl.col(_OFFSET)
        else:
            assert not isinstance(s.offset, str)
            moved = pl.col(_ORD_IN) + s.offset
        if s.wrap:
            span = pl.col(_SPAN) if s.partition is not None else pl.lit(card)
            moved = (moved % span + span) % span

        def remap(source: pl.LazyFrame, carried: list[str], source_dims: tuple[str, ...] | None = None) -> pl.LazyFrame:
            """*source*, with ``s.dimension`` moved by ``s.offset``.

            ``source_dims`` is the caller's, because a presence frame need not
            carry the fragment's: an acyclic shift's presence speaks only about
            the dim it vacated, so projecting the fragment's dims onto it asks
            for columns it never had.
            """
            kept = [d for d in (source_dims if source_dims is not None else p.dims) if d != s.dimension]
            walked = source.join(incoming, on=s.dimension, how='inner').drop(s.dimension)
            if named_offset:
                # A per-entity offset is one more equi-join, on keys the frame
                # already carries — the same locality class as a literal one,
                # since the reach is still a lookup on the dim table.
                offsets, keys = self._offsets(s)
                walked = walked.join(offsets, on=keys, how='inner')
            landing = [_ORD_OUT, s.partition] if s.partition is not None else [_ORD_OUT]
            return (
                walked.with_columns(moved.alias(_ORD_OUT))
                .join(outgoing, on=landing, how='inner')
                .select(*kept, s.dimension, *carried)
            )

        def travelled_presences() -> tuple[Presence, ...]:
            """Where the variable exists after the shift, and what keys it.

            An existing presence **travels**: the coordinate set goes through
            the same map the rows did, and the inner join drops whatever the
            edge vacated. Under a fill the vacated positions go back in
            (:meth:`_vacated`) — a filled slot counts as present. A narrow
            presence is widened first when the shift moves a dim it is silent
            about, since there is no column to remap otherwise and joining the
            row set on a column it never had is #546; :meth:`_widen` changes no
            answer, and what comes back is full-width.

            An operand with **no** presence gets one: nothing was absent before
            and the acyclic edge now is, where without this the vacated slot
            would merely fail to join and the row would survive with its term
            quietly gone (#239, #289). It is keyed by the one dimension it
            speaks about, which is what :attr:`Presence.keyed_by` is
            for — keying it by the fragment's dims would materialise the whole
            coordinate product to name an edge, which costs a fifth again of
            build on a wide ramp (#520), a shape no case in `bench/` covers.
            """
            if not p.presences:
                if s.wrap or s.fill is not None:
                    # A policy speaks about a group's edge, and a coordinate in
                    # no group has none — it is absent under every policy, there
                    # being nothing to come round from or to fill from.
                    return () if s.partition is None else (Presence(self._grouped(s), (s.dimension,)),)
                return (Presence(self._edge(s, card, vacated=False), (s.dimension, *self.offset_dims(s))),)
            return tuple(travelled(x) for x in p.presences)

        def travelled(presence: Presence) -> Presence:
            source, keyed_by = presence.frame, presence.keyed_by
            if keyed_by is not None and not {s.dimension, *self.offset_dims(s)}.issubset(keyed_by):
                source, keyed_by = self._widen(source, keyed_by, p.dims), None
            moved_presence = remap(source, [], keyed_by)
            if s.wrap or s.fill is None:
                return Presence(moved_presence, keyed_by)
            vacated = self._vacated(presence, p.dims, s, card, others)
            return Presence(pl.concat([moved_presence, vacated], how='vertical_relaxed').unique())

        frame = remap(p.frame, p.carried)
        if not s.wrap and s.fill is not None and p.kind == 'const':
            frame = pl.concat([frame, self._filled_edge(s, card, others, s.fill)], how='vertical_relaxed')
        return replace(p, frame=frame, presences=travelled_presences())

    def _widen(self, presence: pl.LazyFrame, have: tuple[str, ...], want: tuple[str, ...]) -> pl.LazyFrame:
        """*presence* over every dim in *want*, saying the same thing.

        A presence frame is silent about the dims it omits, which reads as
        "present at all of them" — so the widening is a cross join with those
        dimensions' own tables, and it changes no answer.
        """
        for d in want:
            if d not in have:
                presence = presence.join(self.data.dimensions[d].select(pl.col('val').alias(d)), how='cross')
        return presence.select(*want)

    def _filled_edge(self, s: plan.Translate, card: int, others: list[str], fill: float) -> pl.LazyFrame:
        """``(dims…, cval=fill)`` at every coordinate the shift vacated.

        Dense over *others*, not over the rows the operand happened to carry:
        the eager lane shifts an array already reindexed to the master
        coordinates, so a fill appearing only where the parameter was
        non-sparse would be a second answer to the same question.

        Only a *truthy* fill gets here, ``fill=0`` needing no rows at all. A
        nonzero fill reaches a translation only over a variable-free operand,
        so this is always the const branch and never invents a ``var_label``.
        """
        edge = self._edge(s, card, vacated=True)
        keyed = self.offset_dims(s)
        for d in others:
            if d in keyed:
                continue
            edge = edge.join(self.data.dimensions[d].select(pl.col('val').alias(d)), how='cross')
        return edge.with_columns(pl.lit(fill, dtype=pl.Float64).alias('cval')).select(*others, s.dimension, 'cval')

    def offset_dims(self, s: plan.Translate) -> tuple[str, ...]:
        """The dims a per-entity offset varies over — empty where it is a number.

        An edge is keyed by them as well as by the translated dimension: how
        far back a row reaches decides which rows have nothing to reach, so
        under a named offset the two entities of one coordinate need not agree
        about whether it is the edge.

        The grouped dimension is not among them: a lag per group varies along
        the translated dimension itself, which the edge is already keyed by.
        """
        if not isinstance(s.offset, str):
            return ()
        grouped = self._grouped_into(s)
        return tuple(d for d in self.program.parameter(s.offset).dims if d != grouped)

    def _grouped_into(self, s: plan.Translate) -> str | None:
        """The dimension the partition groups into, where the offset is over it.

        ``None`` for every other shift — a numeric offset, an unpartitioned
        one, or one over dims the operand carries itself.
        """
        if s.partition is None or not isinstance(s.offset, str):
            return None
        targeted = {lk.name: lk.target for lk in self.program.dimension(s.dimension).lookups}
        target = targeted[s.partition]
        return target if target in self.program.parameter(s.offset).dims else None

    def _offsets(self, s: plan.Translate) -> tuple[pl.LazyFrame, list[str]]:
        """A named offset's values and the keys a frame reads them by.

        A **per-group** offset is declared over the dimension the partition
        groups into, and no frame carries a column of it: what travels with a
        coordinate is the lookup's own value, so the offset is read under the
        lookup's name and one equi-join lands each group's own lag (#1161).
        A coordinate the map places nowhere is in no partitioned table and
        joins to nothing, which is what it reaches everywhere else.
        """
        assert isinstance(s.offset, str)
        grouped = self._grouped_into(s)
        dims = self.program.parameter(s.offset).dims
        keys = [str(s.partition) if d == grouped else d for d in dims]
        frame = self.data.parameters[s.offset].select(
            *(pl.col(d).alias(key) for d, key in zip(dims, keys, strict=True)),
            pl.col('value').cast(pl.Int64).alias(_OFFSET),
        )
        return frame, keys

    def _edge(self, s: plan.Translate, card: int, *, vacated: bool) -> pl.LazyFrame:
        """The coordinates an acyclic shift vacates, or keeps.

        Exact complements, so one filter negated rather than two conditions to
        keep in step: a fill and the presence set it implies must not disagree
        about which coordinates the edge is. One column wide under a numeric
        offset, an edge then being vacated for *every* combination of the other
        dims; under a named one, one column per :meth:`offset_dims` as well.

        Under a partition the edge is **each group's**, counted along the same
        within-group rank the translation itself walks: a coordinate reaches
        outside its own group exactly where it would have reached outside the
        axis. A coordinate in no group is neither — it is absent, the reading
        :meth:`_grouped` gives it, so it is dropped here rather than counted as
        an edge a policy could speak for (#1061). A per-group offset reaches it
        by the lookup rather than by a cross join, one lag standing for the
        whole group.
        """
        table = self.data.dimensions[s.dimension]
        if s.partition is not None:
            grouped = pl.col(s.partition)
            table = self._partitioned(s.dimension, s.partition).with_columns(
                (pl.col('ord').rank('ordinal').over(grouped) - 1).cast(pl.Int64).alias(_POS),
                pl.len().over(grouped).cast(pl.Int64).alias(_SPAN),
            )
            position, span = pl.col(_POS), pl.col(_SPAN)
        else:
            position, span = pl.col('ord'), pl.lit(card, dtype=pl.Int64)
        dims = self.offset_dims(s)
        if isinstance(s.offset, str):
            offsets, keys = self._offsets(s)
            on = [key for key in keys if key == s.partition]
            table = table.join(offsets, on=on, how='inner') if on else table.join(offsets, how='cross')
            offset = pl.col(_OFFSET)
        else:
            offset = pl.lit(s.offset, dtype=pl.Int64)
        source = position - offset
        reaches = (source % span + span) % span if s.wrap else source
        outside = (reaches < 0) | (reaches >= span)
        return table.filter(outside if vacated else ~outside).select(pl.col('val').alias(s.dimension), *dims)

    def _grouped(self, s: plan.Translate) -> pl.LazyFrame:
        """The labels the partition lookup actually places in a group.

        The rest belong to none, so a translation reaches nothing for them and
        their rows are not built — the reading ``sum(by=)`` gives a label the
        map has no row for, and the one an edge policy cannot speak about.
        """
        return self._partitioned(s.dimension, str(s.partition)).select(pl.col('val').alias(s.dimension))

    def _vacated(
        self, presence: Presence, dims: tuple[str, ...], s: plan.Translate, card: int, others: list[str]
    ) -> pl.LazyFrame:
        """The edge positions ``shift`` leaves with nothing to move in.

        Reached only under ``fill=0``, which is the whole of what ``fill`` does
        here: back in the presence set they are present-with-no-term, a zero
        contribution and a surviving row. Left out, absence propagates and the
        row drops — linopy v1's reading of ``.shift()``.

        Only the ``shift`` edge qualifies; a coordinate the variable's own mask
        removed is genuinely absent and remapping already dropped it. So the
        edge is crossed with the other-dim combinations the variable actually
        has, one vacated row each.

        The incoming presence is widened to *others* first, since a narrowly
        keyed one — a pullback's, an earlier shift's — is silent about the
        columns this reads and asking for them is #546 all over again.
        """
        edge = self._edge(s, card, vacated=True)
        if not others:
            return edge
        have = presence.keys(dims)
        source = presence.frame if all(d in have for d in others) else self._widen(presence.frame, have, dims)
        keys = [d for d in self.offset_dims(s) if d in others]
        rows = source.select(*others).unique()
        return rows.join(edge, on=keys, how='inner') if keys else rows.join(edge, how='cross')

    # ------------------------------------------------------------------
    # assembly helpers used by the engine
    # ------------------------------------------------------------------

    @staticmethod
    def constant_scalar(p: TermFragment) -> pl.LazyFrame:
        """The const fragment summed per coordinate: ``(dims…, cval)``."""
        if not p.dims:
            return p.frame.select(pl.col('cval').sum())
        return p.frame.group_by(p.dims).agg(pl.col('cval').sum())


def ordinal(dim: str) -> str:
    """The frame column carrying *dim*'s position in its declared order."""
    return f'__ord {dim}__'


def predicate_dims(where: plan.Predicate, name_dims: Mapping[str, tuple[str, ...]]) -> frozenset[str]:
    """Which dims *where* reads.

    A parameter is read through its own dims, a variable through its foreach,
    a dimension comparison through the dim it names, and a constant reads
    nothing.

    Raises:
        LanguageError: A predicate this function does not know. One that
            forgot to answer here would silently mis-restrict or mislabel a
            model — :meth:`PolarsCompiler.frame`'s semi-join and the label
            planner's factored prefix both read this.
    """
    if isinstance(where, plan.BooleanConstant):
        return frozenset()
    if isinstance(where, (plan.DimensionComparison, plan.DimensionPosition)):
        return frozenset({where.dimension})
    if isinstance(where, (plan.LookupComparison, plan.LookupPairComparison, plan.LookupDefined)):
        return frozenset({where.over})
    if isinstance(where, (plan.ParameterComparison, plan.ParameterDefined)):
        dims = frozenset(name_dims.get(where.parameter, ()))
        value = getattr(where, 'value', None)
        if isinstance(value, str) and value in name_dims:
            dims |= frozenset(name_dims[value])
        return dims
    if isinstance(where, plan.VariableDefined):
        return frozenset(name_dims.get(where.variable, ()))
    if isinstance(where, (plan.And, plan.Or)):
        return predicate_dims(where.left, name_dims) | predicate_dims(where.right, name_dims)
    if isinstance(where, plan.Not):
        return predicate_dims(where.operand, name_dims)
    raise LanguageError(
        f'{type(where).__name__} is a predicate the mask planner does not know how to read; '
        'add it to predicate_dims before using it in a where'
    )


def _certain_parameters(pred: plan.Predicate) -> frozenset[str]:
    """Names whose absence alone makes the whole mask false.

    A row those names have no value for is one the filter would drop anyway, so
    the join may drop it first. Only ``And`` descends — under ``Or`` or ``Not``
    an absent value can still leave the mask true, and dropping the row there
    is a wrong model rather than a slow one, which is why the fallthrough
    answers nothing rather than raising on a node it does not know.
    """
    if isinstance(pred, plan.And):
        return _certain_parameters(pred.left) | _certain_parameters(pred.right)
    if isinstance(pred, (plan.ParameterComparison, plan.ParameterDefined)):
        return frozenset({pred.parameter})
    if isinstance(pred, plan.VariableDefined):
        return frozenset({pred.variable})
    return frozenset()


def _refuse_short_groups(p: plan.DimensionPosition, table: pl.LazyFrame) -> None:
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
            f'where: position({p.dimension}, by={p.by}) {p.op} {p.position} names position '
            f'{p.position} within each group, and {len(short)} of them are shorter than '
            f'that: {short[:5]}. A boundary that names no coordinate leaves the rows it '
            f'was to seed unseeded.'
        )
        raise DataError(msg)


def _falsy_if_null(condition: pl.Expr) -> pl.Expr:
    """*condition* with null read as false.

    A missing parameter row must exclude the coordinate rather than
    propagate. Masks are row absence.
    """
    return condition.fill_null(value=False)


def _position_ordinal(p: plan.DimensionPosition, cardinality: int) -> int:
    """*p*'s position as an ordinal into a dimension of *cardinality* labels.

    A negative position counts from the end. Out of range is an error rather
    than a predicate matching nothing: a boundary clause that silently seeds
    no row leaves the recurrence unanchored, which is the failure this
    construct exists to make impossible.
    """
    at = p.position + cardinality if p.position < 0 else p.position
    if not 0 <= at < cardinality:
        raise DataError(
            f'where: position({p.dimension}) {p.op} {p.position} names position {at} of '
            f"'{p.dimension}', which has {cardinality} coordinate(s). A boundary that "
            f'names no coordinate leaves the rows it was to seed unseeded.'
        )
    return at


def _dimension_column(dimension: str, value: float | str | datetime.date) -> pl.Expr:
    """The column a where-comparison on *dimension* reads.

    A string label is compared in ``String`` space, undoing binding's ``Enum``:
    The where-string rules order labels bytewise and read an unknown label as
    matching nothing,
    where an ``Enum`` orders by declaration and refuses strangers.
    """
    column = pl.col(dimension)
    return column.cast(pl.String) if isinstance(value, str) else column


#: Column-against-column comparison, for the one predicate whose both sides are
#: structure. Kept apart from :func:`_compare`, which takes a literal: polars
#: needs no ``pl.lit`` here and a shared helper would have to branch on which.
_COLUMN_COMPARISONS: dict[plan.ComparisonOperator, Callable[[pl.Expr, pl.Expr], pl.Expr]] = {
    '==': lambda left, right: left == right,
    '!=': lambda left, right: left != right,
    '<': lambda left, right: left < right,
    '<=': lambda left, right: left <= right,
    '>': lambda left, right: left > right,
    '>=': lambda left, right: left >= right,
}


def _compare(column: pl.Expr, op: plan.ComparisonOperator, value: float | str | datetime.date) -> pl.Expr:
    """One where-comparison. A string, a float and a date are all literals here."""
    literal = pl.lit(value)
    match op:
        case '==':
            return column == literal
        case '!=':
            return column != literal
        case '<':
            return column < literal
        case '<=':
            return column <= literal
        case '>':
            return column > literal
        case '>=':
            return column >= literal


def _propagate_absence(compiled: CompiledExpression) -> CompiledExpression:
    """Restrict every fragment to where the *whole* expression exists.

    Addition is fragment concatenation, so ``x + size`` is two independent
    streams — right at row level, where the engine intersects the presences,
    but a **reduction** consumes the expression before any row exists. That is
    the difference between ``sum(x + size, over=f)``, which sums where the
    summand exists, and ``sum(x, over=f) + sum(size, over=f)``, which sums each
    operand over its own domain and reads the absent ``size`` as a zero (the
    absence and operator rules).

    Applied only where the key columns are dims the fragment carries: a
    restriction naming a dim a fragment lacks cannot speak about it.

    **Which operators need it is decided by their fan-in.** ``Sum`` and
    ``GroupSum`` are many-to-one and ``Window`` is one-to-many, so an output row
    mixes several input slots: the row-level intersection at assembly can say
    the *row* survives, never which of the slots behind it did, and a constant
    read from an absent slot is already inside the total by then. ``At`` and
    ``Translate`` are one-to-one — one output, one input — so the row either
    survives or does not, and the intersection is exactly right without a pass
    here. ``Window`` was missing from this list and the lanes disagreed about a
    constant at a masked slot (#1142); the fan-in is the rule, not the list.

    **A fragment is never restricted by its own presence.** Its rows and its
    presence are built from one frame and rewritten in step — a product joins
    the rows and leaves the coordinates, a translation remaps both, a fill adds
    to both — so the rows are inside the coordinates by construction and the
    join could only return them all. Under a mask over a single term, the
    ordinary case, that made the pass a semi-join of a frame against itself,
    which is measurable on any model large enough to care (#413).

    The presence frame is not deduplicated first: a semi-join asks whether a
    key occurs, and occurring twice is still occurring, so the distinct changes
    no row and costs a hash pass over every coordinate the variable has.
    """
    absent = [(p, x) for p in (*compiled.terms, *compiled.quads, *compiled.consts) for x in p.presences]
    if not absent:
        return compiled

    def restrict(p: TermFragment) -> TermFragment:
        frame = p.frame
        for source, presence in absent:
            if source is p:
                continue
            on = list(presence.keys(source.dims))
            if all(d in p.dims for d in on):
                frame = presence.restrict(frame, on)
        return p if frame is p.frame else replace(p, frame=frame)

    return _map_fragments(compiled, restrict)


def _map_fragments(
    compiled: CompiledExpression,
    rewrite: Callable[[TermFragment], TermFragment],
) -> CompiledExpression:
    """Apply *rewrite* to every fragment, keeping the kinds apart.

    Rewriting one fragment at a time is what pointwise and bounded-halo
    locality mean; a node needing them together is global, and rejected at
    lowering. A quadratic fragment goes through the same rewrites as a linear
    one and for the same reason — a shape operator moves rows between
    coordinates and never looks at what the row *carries*, which is why the
    operators project through ``carried`` rather than naming columns.
    """
    return CompiledExpression(
        tuple(rewrite(p) for p in compiled.terms),
        tuple(rewrite(p) for p in compiled.consts),
        tuple(rewrite(p) for p in compiled.quads),
    )


def _negate(p: TermFragment) -> TermFragment:
    return replace(p, frame=p.frame.with_columns(-pl.col(p.value_column)))


def _join_mul(a: TermFragment, c: TermFragment, kind: Kind, divide: bool = False) -> TermFragment:
    """``a * c`` (or ``a / c``) where *c* is a const fragment.

    Joins on shared dims, broadcasts the rest. The right-hand value is renamed
    first: both sides may carry ``cval``, and a suffix collision would multiply
    a column by itself. The dims *c* contributes are broadcast, so the label
    says nothing about them.

    A divide joins **left**, so a coordinate the divisor has no value for
    yields a *null* coefficient instead of silently dropping the term. The row
    may still be masked out downstream, taking the null with it: the question
    is not whether the divisor is dense but whether it is defined where the
    model divides by it.

    *c* is variable-free, so it contributes no absence: a sparse coefficient
    zeroes a term, it does not unmake the variable underneath it. The output
    dims may be wider than ``a.dims``, which is why the presence key travels
    with the fragment rather than being re-derived from dims here (#345).
    """
    shared = [d for d in a.dims if d in c.dims]
    out_dims = a.dims + tuple(d for d in c.dims if d not in a.dims)
    right = c.frame.rename({'cval': _RHS})
    how = 'left' if divide else 'inner'
    joined = a.frame.join(right, on=shared, how=how) if shared else a.frame.join(right, how='cross')

    value, rhs = pl.col(a.value_column), pl.col(_RHS)
    combined = value / rhs if divide else value * rhs
    out = value_column(kind)
    frame = joined.with_columns(combined.alias(out)).select(*out_dims, *carried_columns(kind))
    return replace(a, dims=out_dims, frame=frame, kind=kind)


def _join_pow(a: TermFragment, b: TermFragment) -> TermFragment:
    """``a ** b``, both const fragments — one const fragment out.

    :func:`_join_mul`'s shape with ``pow`` in place of ``*``, and the same
    reason for renaming the right-hand value first: both sides carry ``cval``.
    An **inner** join, unlike divide's left: an exponent with no value at a
    coordinate is not a division by a hole, it is a factor the model never
    stated, and a null base or exponent would poison the coefficient it
    multiplies rather than reporting anything.
    """
    shared = [d for d in a.dims if d in b.dims]
    out_dims = a.dims + tuple(d for d in b.dims if d not in a.dims)
    right = b.frame.rename({'cval': _RHS})
    joined = a.frame.join(right, on=shared, how='inner') if shared else a.frame.join(right, how='cross')
    frame = joined.with_columns(pl.col('cval').pow(pl.col(_RHS)).alias('cval')).select(
        *out_dims, *carried_columns('const')
    )
    return TermFragment(out_dims, frame, 'const')


def _join_quad(a: TermFragment, b: TermFragment) -> TermFragment:
    """``a * b`` where both carry a variable — one quadratic fragment.

    A join on the dims the two share, so a quadratic term costs what a linear
    one does: aligned is an equi-join, broadcast joins on the coarser side, and
    the cross join is refused a level up (``language/degree.py``).

    The second label is renamed on the way in, since both sides carry
    ``var_label`` and a suffix collision would pair a variable with itself —
    which ``p * p`` makes a *legal* fragment, so no error would catch it.

    Nothing is canonicalised here: which of ``x * y`` and ``y * x`` a pair is
    depends on column labels, which fragments do not carry until the engine
    places them (:meth:`PolarsEngine._build_objective`).
    """
    shared = [d for d in a.dims if d in b.dims]
    out_dims = a.dims + tuple(d for d in b.dims if d not in a.dims)
    right = b.frame.rename({'var_label': 'var_label_2', 'coeff': _RHS})
    joined = a.frame.join(right, on=shared, how='inner') if shared else a.frame.join(right, how='cross')
    frame = joined.with_columns((pl.col('coeff') * pl.col(_RHS)).alias('coeff')).select(
        *out_dims, *carried_columns('quad')
    )
    return TermFragment(out_dims, frame, 'quad', presences=a.presences + b.presences)


def _scattered(at: pl.Series, values: pl.Series, size: int) -> Any:
    """*values* moved to the positions *at* names, one pass, order checked."""
    import numpy as np

    indices = at.to_numpy()
    written = np.zeros(size, dtype=bool)
    written[indices] = True
    if not written.all():
        msg = 'a parameter passed the density gate but does not cover the coordinate product'
        raise LpspecError(msg)

    out = np.empty(size, dtype=np.float64)
    out[indices] = values.to_numpy()
    return out
