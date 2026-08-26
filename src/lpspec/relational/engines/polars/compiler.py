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
from typing import TYPE_CHECKING, Any

import polars as pl

from lpspec.errors import (
    LanguageError,
    LpspecError,
)
from lpspec.relational import plan
from lpspec.relational.engines.polars.fragments import (
    PRESENT,
    CompiledExpression,
    Presence,
    TermFragment,
    join_mul,
    join_on,
    join_pow,
    join_quad,
    map_fragments,
    negate,
    propagate_absence,
    refuse_a_fragment_without_the_dims,
)
from lpspec.relational.engines.polars.predicates import (
    Carrier,
    compile_predicate,
    falsy_if_null,
    predicate_dims,
)
from lpspec.relational.engines.polars.reindex import translate_fragment, window_fragment

if TYPE_CHECKING:
    from collections.abc import Mapping

    from polars._typing import JoinStrategy, MaintainOrderJoin

    from lpspec.relational.engines.polars.binding import BoundSources
    from lpspec.relational.engines.polars.labels import Labelled


#: Carries the single row of the empty coordinate product. Polars cannot hold a
#: frame with one row and no columns — collecting one reports ``(0, 0)`` — so the
#: unit needs a column to exist in, and every path drops it by selecting the
#: dims and the label instead.
UNIT = '__unit__'


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
    variables: Mapping[str, Labelled]

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
        carrier, condition = compile_predicate(self, out, where, dims)
        if carrier is out:
            return out.filter(falsy_if_null(condition))
        touched = predicate_dims(where, self.name_dims)
        on = tuple(d for d in dims if d in touched)
        if on and len(on) < len(dims) and touched <= set(dims):
            keyed, keyed_condition = compile_predicate(self, self._coordinate_product(on), where, on)
            surviving = keyed.filter(falsy_if_null(keyed_condition)).select(*on)
            return out.join(surviving, on=list(on), how='semi')
        return carrier.filter(falsy_if_null(condition))

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
        return join_on(frame, table, declaration.dims, how, maintain_order)

    # ------------------------------------------------------------------
    # predicates (where masks — row absence)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # bounds
    # ------------------------------------------------------------------

    def bounds(self, frame: pl.LazyFrame, v: plan.VariableDeclaration) -> pl.LazyFrame:
        """*frame* with ``lb``/``ub`` columns for variable *v*.

        Joins and arithmetic are one object, so a bound cannot be evaluated
        against a frame missing what it reads.
        """
        carrier = Carrier(frame)

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

        A bound dense over the whole variable product — the ordinary shape in
        energy modelling — costs a full-size join against a full-size
        coordinate product here, where the eager lane gets it free from array
        position; on `profiled` that join was most of the build (#511). Each
        parameter row's slot is computed from its own labels' ordinals
        (:func:`labels.row_major`'s layout) and its value scattered there —
        the table's row order is nothing, and ``_scattered`` refuses a product
        any slot of which nothing wrote.

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
            quads = tuple(join_quad(t, u) for t in a.terms for u in b.terms)
            quads += tuple(join_mul(q, c, 'quad') for q in a.quads for c in b.consts)
            quads += tuple(join_mul(q, c, 'quad') for q in b.quads for c in a.consts)
            terms = tuple(join_mul(t, c, t.kind) for t in a.terms for c in b.consts)
            terms += tuple(join_mul(t, c, t.kind) for t in b.terms for c in a.consts)
            consts = tuple(join_mul(x, c, 'const') for x in a.consts for c in b.consts)
            return CompiledExpression(terms, consts, quads)

        def quotient(a: CompiledExpression, b: CompiledExpression) -> CompiledExpression:
            """``a / b``, where *b* is one variable-free factor.

            That it is *one* is ``degree.check_binary``'s answer, given at load
            with no data bound, so a divisor that adds never reaches a plan.
            """
            if b.terms or b.quads:
                raise LanguageError(f'nonlinear quotient in {context}: the divisor contains variables')
            assert len(b.consts) == 1, 'a divisor that adds is refused at load'
            inv = b.consts[0]
            terms = tuple(join_mul(t, inv, t.kind, divide=True) for t in a.terms)
            quads = tuple(join_mul(q, inv, 'quad', divide=True) for q in a.quads)
            consts = tuple(join_mul(x, inv, 'const', divide=True) for x in a.consts)
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
            return CompiledExpression((), (join_pow(a.consts[0], b.consts[0]),))

        def ev(e: plan.Expression) -> CompiledExpression:
            if isinstance(e, plan.Constant):
                frame = pl.LazyFrame({'cval': [float(e.value)]}, schema={'cval': pl.Float64})
                return CompiledExpression((), (TermFragment((), frame, 'const'),))
            if isinstance(e, plan.Parameter):
                return CompiledExpression((), (self._parameter_fragment(e.name),))
            if isinstance(e, plan.Variable):
                return CompiledExpression((self._variable_fragment(e.name),), ())
            if isinstance(e, plan.Negate):
                return map_fragments(ev(e.operand), negate)
            if isinstance(e, plan.Add):
                a, b = ev(e.left), ev(e.right)
                return CompiledExpression(a.terms + b.terms, a.consts + b.consts, a.quads + b.quads)
            if isinstance(e, plan.Multiply):
                return product(ev(e.left), ev(e.right))
            if isinstance(e, plan.Divide):
                return quotient(ev(e.numerator), ev(e.divisor))
            if isinstance(e, plan.Power):
                return power(ev(e.base), ev(e.exponent))
            if isinstance(e, plan.Sum):
                inner = propagate_absence(ev(e.operand))
                return map_fragments(inner, lambda p: self._sum_fragment(p, e.over, context))
            if isinstance(e, plan.GroupSum):
                inner = propagate_absence(ev(e.operand))
                return map_fragments(inner, lambda p: self._group_fragment(p, e, context))
            if isinstance(e, plan.At):
                return map_fragments(ev(e.operand), lambda p: self._at_fragment(p, e, context))
            if isinstance(e, plan.Translate):
                return map_fragments(ev(e.operand), lambda p: translate_fragment(self, p, e, context))
            if isinstance(e, plan.Window):
                inner = propagate_absence(ev(e.operand))
                return map_fragments(inner, lambda p: window_fragment(self, p, e, context))
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
        frame = self.variables[name].frame.select(*dims, 'var_label', pl.lit(1.0, dtype=pl.Float64).alias('coeff'))
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
        frame = self.variables[name].frame
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
            refuse_a_fragment_without_the_dims(p, missing, context, f'sum(over={list(over)})')
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
            refuse_a_fragment_without_the_dims(p, [g.over], context, f'sum(by=) over {g.over!r}')
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

    def partitioned(self, dim: str, lookup: str) -> pl.LazyFrame:
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
        inner join swallows both — the operand's own, and the **lookup's**,
        where the map has no row for the label. Unreported, the term merely
        vanishes and its row survives to assert `x <= 0` where the model said
        nothing (#968).

        A total lookup over an operand with nothing to report yields nothing
        rather than a restriction admitting everything, so a model with no
        absence in it does not pay for the machinery that carries one. The key
        is stated rather than left implied because a later product widens the
        fragment's dims while this frame keeps the one column that matters —
        the hazard :class:`Presence` names.
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
                (presence.frame, keys) if carries_targets else (self.widen(presence.frame, keys, p.dims), p.dims)
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

    def widen(self, presence: pl.LazyFrame, have: tuple[str, ...], want: tuple[str, ...]) -> pl.LazyFrame:
        """*presence* over every dim in *want*, saying the same thing.

        A presence frame is silent about the dims it omits, which reads as
        "present at all of them" — so the widening is a cross join with those
        dimensions' own tables, and it changes no answer.
        """
        for d in want:
            if d not in have:
                presence = presence.join(self.data.dimensions[d].select(pl.col('val').alias(d)), how='cross')
        return presence.select(*want)


def ordinal(dim: str) -> str:
    """The frame column carrying *dim*'s position in its declared order."""
    return f'__ord {dim}__'


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
