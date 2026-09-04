"""The feasible region on two axes: what a model *can* do, drawn rather than solved.

A model's feasible set has one dimension per column, and nothing draws that.
What a modeller checks is its shadow on two quantities they can name — heat
against power for a plant, storage level against spill — which is a convex
polygon for a linear model, and each of its vertices is the answer to one
solve: maximise the two quantities weighted by a direction, and the optimum
is the vertex that direction points at. So the region is found by solving
the same built model along a sequence of directions, which is what
:func:`project` does, on the fast path the whole way: only two costs change
between solves, so the solver keeps the model and carries its basis.

The polygon comes back as a :class:`Region`: its vertices as a frame, and
:meth:`Region.plot` to fill it on a matplotlib axes, which is the ``[plot]``
extra rather than the engine's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl
from math_spec import DimensionError, program, to_program, to_spec

from lpspec.api import Model
from lpspec.errors import LpspecError, NoSolutionError, unknown_name_message
from lpspec.relational.sinks import solver

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from math_spec import Spec
    from matplotlib.axes import Axes

__all__ = ['Region', 'project']

#: The names the probing model adds to the caller's, each refused where the
#: file already declares one — one flat namespace, and a quiet override would
#: solve a different model.
_AXES = ('x_axis', 'y_axis')
_DIRECTIONS = ('x_direction', 'y_direction')
_SELECTIONS = ('x_selection', 'y_selection')

#: The four directions every trace starts from — enough to enclose a bounded
#: region, and each one the probe a caller asks first: how far does it go.
_COMPASS = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))

#: One direction's probe is one solve, so a region that keeps producing
#: vertices past this many is not converging — noise the tolerance does not
#: absorb — and stopping is better than a driver that never returns.
_MOST_SOLVES = 1000

Point = tuple[float, float]

_NEEDS_THE_EXTRA = (
    'matplotlib ships with the [plot] extra rather than with the engine, so this build cannot draw: '
    'pip install "lpspec[plot]". A region needs nothing added to be read as it stands — its vertices '
    'are a polars frame, two columns any plotting library fills a polygon from.'
)


@dataclass(frozen=True)
class Region:
    """What :func:`project` hands back: the feasible region on two quantities, as a polygon.

    Attributes:
        x: The quantity on the horizontal axis.
        y: The quantity on the vertical axis.
        vertices: The polygon's vertices as ``(x, y)`` columns named after the
            two quantities, counter-clockwise from the lowest-leftmost. A
            region that is a segment has two rows and a single point one.
    """

    x: str
    y: str
    vertices: pl.DataFrame

    def plot(self, ax: Axes | None = None, **style: Any) -> Axes:
        """Fill the region on a matplotlib axes, and return the axes.

        A polygon is filled and outlined, a segment drawn as a line, a point
        as a marker; the axes are labelled with the two quantities. Anything
        the picture should say beyond that — the optimum on it, a second
        region beside it — is a call on the axes that comes back.

        Args:
            ax: Where to draw; a new figure's axes where none is given.
            style: Forwarded to matplotlib's ``fill``, ``plot`` or
                ``scatter``, whichever the region's shape calls for — a
                ``color``, an ``alpha``, a ``label`` for a legend.

        Raises:
            ModuleNotFoundError: On an install without matplotlib, naming the extra.
        """
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(_NEEDS_THE_EXTRA) from exc

        if ax is None:
            _, ax = plt.subplots()
        xs, ys = self.vertices[self.x].to_list(), self.vertices[self.y].to_list()
        if len(xs) == 1:
            ax.scatter(xs, ys, **style)
        elif len(xs) == 2:
            ax.plot(xs, ys, **style)
        else:
            style.setdefault('alpha', 0.3)
            (patch,) = ax.fill(xs, ys, **style)
            ax.plot([*xs, xs[0]], [*ys, ys[0]], color=patch.get_facecolor(), alpha=1.0)
        ax.set_xlabel(self.x)
        ax.set_ylabel(self.y)
        return ax


def project(
    spec: str | Path | dict[str, Any] | Spec,
    sources: Mapping[str, Any],
    *,
    x: str,
    y: str,
    at: Mapping[str, Any] | None = None,
    solver_name: str = 'highs',
    solver_options: Mapping[str, Any] | None = None,
    tolerance: float = 1e-6,
) -> Region:
    """Trace the feasible region of *spec* on two of its quantities.

    ::

        region = lps.project('plant.yaml', sources, x='heat', y='power', at={'t': 5})
        region.vertices  # (heat, power), one row per vertex
        region.plot()  # filled, on a matplotlib axes

    The region is every ``(x, y)`` some feasible solution reaches, which the
    objective plays no part in: the file's is set aside and the solve is
    driven by a direction instead. For a continuous model the polygon is
    exact — every vertex is a solve, and an edge is kept only once a solve
    along its outward normal finds nothing beyond it. A model with integer
    or binary variables gives the **convex hull** of its region, since each
    solve still returns an extreme point of it; what the hull encloses may
    have holes it cannot show.

    *at* fixes coordinates, and the rest is summed: with ``at={'t': 5}`` a
    quantity over ``(t, unit)`` is read at that hour and totalled over the
    units, and with no *at* it is totalled over everything. A quantity you
    want at one coordinate of every dim is one to name in *at*; a quantity
    you want summed over a dim is one to declare summed, as an expression of
    its own.

    Args:
        spec: A YAML path, a mapping, or a loaded ``Spec`` — the file, since
            the probe rewrites its objective. A ``Program`` from
            :func:`~lpspec.api.check` has already been lowered and is
            refused; pass what it was lowered from.
        sources: As :func:`~lpspec.api.build` takes them.
        x: A declared variable or named expression, the horizontal axis.
        y: The same for the vertical axis.
        at: Dimension names to one label each, fixing where the two
            quantities are read. Every dim named must be one both carry.
        solver_name: As :meth:`~lpspec.api.Model.solve` takes it.
        solver_options: As :meth:`~lpspec.api.Model.solve` takes them.
        tolerance: How far past an edge a solve must reach, relative to the
            region's scale, to count as a vertex rather than as the solver's
            own noise.

    Returns:
        The region — its vertices as a frame, and a ``plot`` for the picture.

    Raises:
        KeyError: *x* or *y* names nothing declared.
        NoSolutionError: The model is infeasible, so there is no region.
        LpspecError: The region is unbounded along some direction, a dim in
            *at* one of the quantities does not carry, a name the probe adds
            already declared, or a solve that stopped without a solution.
    """
    if isinstance(spec, program.Program):
        raise LpspecError(
            'project takes the spec as the path or mapping it was written as, not a Program: the '
            'probe rewrites the objective, which a lowered program no longer carries as a file.'
        )
    if x == y:
        raise LpspecError(f"project needs two different quantities; both axes are '{x}'.")
    solver(solver_name)
    declared = to_spec(spec).to_dict()
    at = dict(at or {})
    probe = _probing_spec(declared, x, y, at)
    data = {**sources, **dict(zip(_DIRECTIONS, _COMPASS[0], strict=True))}
    for selection in _SELECTIONS:
        if at:
            data[selection] = pl.DataFrame({**{d: [label] for d, label in at.items()}, 'value': [1.0]})

    with Model(probe, data) as model:
        _refuse_dims_at_does_not_reach(model, x, y, at)

        def support(direction: Point) -> Point:
            solving = dict(zip(_DIRECTIONS, direction, strict=True))
            result = model.update(solving).solve(solver_name, solver_options=solver_options, keep='progress')
            with result:
                _refuse_without_a_vertex(result, direction, x, y)
                return result.expression(_AXES[0]).item(), result.expression(_AXES[1]).item()

        vertices = _trace(support, tolerance)
    return Region(x, y, pl.DataFrame({x: [p[0] for p in vertices], y: [p[1] for p in vertices]}))


def _probing_spec(declared: dict[str, Any], x: str, y: str, at: Mapping[str, Any]) -> dict[str, Any]:
    """The caller's model with the objective replaced by a direction over the two axes.

    Ordinary declarations, all of them — two scalar weights, two named
    expressions, and one selection parameter per axis where *at* fixes a
    coordinate — so the probe is a file the language validates, typesets and
    builds like any other, on both lanes.
    """
    variables = declared.get('variables') or {}
    expressions = declared.get('expressions') or {}
    for name in (x, y):
        if name not in variables and name not in expressions:
            raise KeyError(unknown_name_message('variable or named expression', name, [*variables, *expressions]))
    dimensions = declared.get('dimensions') or {}
    if strangers := [d for d in at if d not in dimensions]:
        raise LpspecError(
            f'at names {strangers}, which the spec declares no dimension for. Declared: {sorted(dimensions)}.'
        )
    taken = {
        name: kind
        for kind in ('parameters', 'variables', 'expressions', 'constraints', 'lookups', 'macros')
        for name in (declared.get(kind) or {})
    }
    if clashes := [name for name in (*_AXES, *_DIRECTIONS, *_SELECTIONS) if name in taken]:
        raise LpspecError(
            f'project adds {clashes} to the model to probe it, and the spec already declares '
            f'{", ".join(f"{n} under {taken[n]}:" for n in clashes)}. Rename the declaration.'
        )
    scalar: dict[str, list[str]] = {'dims': []}
    parameters = {**(declared.get('parameters') or {}), **{d: dict(scalar) for d in _DIRECTIONS}}
    axes: dict[str, str] = {}
    for axis, selection, quantity in zip(_AXES, _SELECTIONS, (x, y), strict=True):
        if _is_scalar(declared, quantity):
            if at:
                raise LpspecError(
                    f"at names {list(at)}, which '{quantity}' does not carry: it is read over no dims. "
                    f'Drop at, or name a quantity that varies along it.'
                )
            axes[axis] = quantity
        elif at:
            parameters[selection] = {'dims': list(at)}
            axes[axis] = f'sum({selection} * {quantity})'
        else:
            axes[axis] = f'sum({quantity})'
    return {
        **declared,
        'parameters': parameters,
        'expressions': {**expressions, **axes},
        'objective': {
            'sense': 'maximize',
            'expression': f'{_DIRECTIONS[0]} * {_AXES[0]} + {_DIRECTIONS[1]} * {_AXES[1]}',
        },
    }


def _is_scalar(declared: dict[str, Any], quantity: str) -> bool:
    """Whether *quantity* carries no dims — asked of the language, whose objective takes nothing else.

    A variable's dims are written in its ``foreach``; a named expression's
    fall out of its body, and the rule that decides them is the language's.
    Lowering the spec with the quantity as its objective asks that rule
    directly: an objective must carry no dims, so it lowers exactly when the
    quantity is a scalar.
    """
    try:
        to_program({**declared, 'objective': {'sense': 'maximize', 'expression': quantity}})
    except DimensionError:
        return False
    return True


def _refuse_dims_at_does_not_reach(model: Model, x: str, y: str, at: Mapping[str, Any]) -> None:
    """Refuse an *at* naming a dim one quantity does not carry.

    The probe multiplies the selection into the quantity, and a product over
    a dim only the selection carries broadcasts rather than selects — the
    same number at every coordinate, which reads as a region and is not one.
    A variable's dims are its ``foreach``; an expression's fall out of its
    body, which the built model is the first to hold.
    """
    if not at:
        return
    program_ = model._program  # the driver reads what its own build holds, which no verb hands out
    for name in (x, y):
        variable = program_.variables.get(name)
        dims = variable.dims if variable is not None else model._engine.expression_dims(name)
        if missing := [d for d in at if d not in dims]:
            raise LpspecError(
                f"at names {missing}, which '{name}' does not carry: it is read over {list(dims) or 'no dims'}. "
                f'Name only dims both quantities carry, or declare an expression that does carry it.'
            )


def _refuse_without_a_vertex(result: Any, direction: Point, x: str, y: str) -> None:
    """Why a probe found no vertex, in the caller's terms rather than the solver's."""
    if result.has_primal:
        return
    where = f'({direction[0]:+.3g}·{x}, {direction[1]:+.3g}·{y})'
    condition = result.termination_condition
    if condition == 'infeasible':
        raise NoSolutionError('the model is infeasible, so it has no feasible region to trace.')
    if condition in ('unbounded', 'infeasible_or_unbounded'):
        raise LpspecError(
            f'the feasible region is unbounded toward {where}: nothing in the model caps that '
            f'direction. Bound the variables it runs along, then trace it again.'
        )
    raise LpspecError(
        f'the solve toward {where} stopped {condition!r} with no solution to read, so the region '
        f'cannot be traced from it.'
    )


def _trace(support: Any, tolerance: float) -> list[Point]:
    """Every vertex of the region, by probing until no edge has anything beyond it.

    *support* answers one direction with the point the region reaches
    farthest along it. The four compass points come first; from then on the
    polygon is the convex hull of everything found so far, and each of its
    edges is probed along its outward normal once. A probe reaching past the
    edge, by more than *tolerance* at the region's scale, is a new vertex and
    the hull is retaken; one that does not settles the edge. It ends when
    every edge is settled, which a polygon with finitely many vertices does.
    """
    points: list[Point] = []
    settled: set[tuple[Point, Point]] = set()
    solves = 0

    def found(point: Point) -> None:
        if all(not _near(point, p, tolerance) for p in points):
            points.append(point)

    for direction in _COMPASS:
        found(support(direction))
        solves += 1
    while True:
        hull = _hull(points)
        pending = [edge for edge in _edges(hull) if edge not in settled]
        if not pending:
            return hull
        if solves >= _MOST_SOLVES:
            raise LpspecError(
                f'the region kept producing vertices for {_MOST_SOLVES} solves without settling, '
                f'which is solver noise rather than geometry. Raise tolerance above {tolerance}.'
            )
        a, b = pending[0]
        normal = _outward(a, b)
        reached = support(normal)
        solves += 1
        gain = _dot(normal, reached) - _dot(normal, a)
        if gain > tolerance * _scale(points) and all(not _near(reached, p, tolerance) for p in points):
            points.append(reached)
        else:
            settled.add((a, b))


def _edges(hull: Sequence[Point]) -> list[tuple[Point, Point]]:
    """The directed edges of a counter-clockwise polygon — both ways along a segment, none for a point."""
    if len(hull) < 2:
        return []
    return [(hull[i], hull[(i + 1) % len(hull)]) for i in range(len(hull))]


def _hull(points: Sequence[Point]) -> list[Point]:
    """The convex hull, counter-clockwise from the lowest-leftmost point, collinear points dropped."""
    ordered = sorted(set(points))
    if len(ordered) < 3:
        return ordered

    def half(sequence: Sequence[Point]) -> list[Point]:
        chain: list[Point] = []
        for p in sequence:
            while len(chain) > 1 and _cross(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
        return chain

    lower, upper = half(ordered), half(ordered[::-1])
    return lower[:-1] + upper[:-1]


def _outward(a: Point, b: Point) -> Point:
    """The unit normal pointing out of a counter-clockwise polygon across the edge *a* to *b*."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    return dy / length, -dx / length


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _scale(points: Sequence[Point]) -> float:
    """What a tolerance is relative to: the region's own size, never below one."""
    return max(1.0, *(abs(c) for p in points for c in p))


def _near(a: Point, b: Point, tolerance: float) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= tolerance * max(1.0, abs(a[0]), abs(a[1]))
