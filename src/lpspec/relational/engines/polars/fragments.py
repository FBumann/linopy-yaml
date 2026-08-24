"""The algebra of additive pieces: what an expression compiles *to*.

An LP row is a sum of pieces, so a compiled expression is a list of fragments
and every shape operator rewrites one. This module is that vocabulary and the
arithmetic over it — product, quotient, power, negation, and the absence rule
that decides which rows a reduction is allowed to see.

It holds no state and reads no data: everything here takes fragments and
returns fragments, which is what lets
:mod:`~lpspec.relational.engines.polars.compiler` be about *which* query a plan
node becomes rather than about what a term is.

Column conventions, relied on by the engine:

===================  ==========================================
frame                columns
===================  ==========================================
term fragment        ``dims…``, ``var_label``, ``coeff``
quad fragment        ``dims…``, ``var_label``, ``var_label_2``, ``coeff``
const fragment       ``dims…``, ``cval``
===================  ==========================================
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, NoReturn

import polars as pl

from lpspec.errors import LaneError, LanguageError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


#: The right-hand operand's value while a join holds both. The spaces make it
#: unrepresentable as a declared name, so it cannot collide with a dimension or
#: lookup the model already has.
_RHS = '__rhs value__'

#: Carries the one bit a *scalar* declaration's presence frame has to say:
#: whether the declaration exists at all. Polars cannot hold a frame with rows
#: and no columns, and a scalar presence has no dim to be keyed by, so the
#: restriction becomes a cross join against this marker.
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


def refuse_a_fragment_without_the_dims(p: TermFragment, dims: list[str], context: str, operator: str) -> NoReturn:
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
    :func:`join_mul` names the columns of the fragment it is *building*, whose
    kind need not be either operand's.
    """
    return 'cval' if kind == 'const' else 'coeff'


def carried_columns(kind: Kind) -> list[str]:
    """The non-dim columns a projection of this fragment kind has to keep."""
    return [*_LABELS[kind], value_column(kind)]


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


def constant_scalar(p: TermFragment) -> pl.LazyFrame:
    """The const fragment summed per coordinate: ``(dims…, cval)``."""
    if not p.dims:
        return p.frame.select(pl.col('cval').sum())
    return p.frame.group_by(p.dims).agg(pl.col('cval').sum())


def propagate_absence(compiled: CompiledExpression) -> CompiledExpression:
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

    return map_fragments(compiled, restrict)


def map_fragments(
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


def negate(p: TermFragment) -> TermFragment:
    return replace(p, frame=p.frame.with_columns(-pl.col(p.value_column)))


def join_mul(a: TermFragment, c: TermFragment, kind: Kind, divide: bool = False) -> TermFragment:
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


def join_pow(a: TermFragment, b: TermFragment) -> TermFragment:
    """``a ** b``, both const fragments — one const fragment out.

    :func:`join_mul`'s shape with ``pow`` in place of ``*``, and the same
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


def join_quad(a: TermFragment, b: TermFragment) -> TermFragment:
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
