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

What comes back is a :class:`Region` of four tidy frames — the vertices, what
each piece pinned, which bound or row each edge sits on, and where the
model's own optimum lands — and :meth:`Region.plot` to fill it on a
matplotlib axes, which is the ``[plot]`` extra rather than the engine's.

A binary makes the region a union of polygons rather than one, and a solve
along a direction only ever finds the hull of the union. ``binaries='each'``
fixes every combination of the binaries in turn — a pair of rows per binary
whose right-hand sides are data, so each combination is a push onto the
loaded solver — and traces the region each leaves as a piece of the whole.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import polars as pl
from math_spec import DimensionError, program, to_program, to_spec

from lpspec.api import Model
from lpspec.errors import LpspecError, NoSolutionError, unknown_name_message
from lpspec.relational.sinks import solver

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from math_spec import Spec
    from matplotlib.axes import Axes

    from lpspec.relational.result import Result

__all__ = ['Region', 'project']

#: The names the probing model adds to the caller's, each refused where the
#: file already declares one — one flat namespace, and a quiet override would
#: solve a different model.
_AXES = ('x_axis', 'y_axis')
_DIRECTIONS = ('x_direction', 'y_direction')
_SELECTIONS = ('x_selection', 'y_selection')
_OBJECTIVE_WEIGHT = 'objective_weight'

#: The four directions every trace starts from — enough to enclose a bounded
#: region, and each one the probe a caller asks first: how far does it go.
_COMPASS = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))

#: What pins one binary to a value: two rows, ``b >= at_least`` and
#: ``b <= at_most``, both sides data — a fix is bounds (docs/lifecycle.ipynb),
#: and a bound that is data moves on the fast path.
_PIN_ROWS = (('{b}_pinned_low', '{b} >= {b}_at_least'), ('{b}_pinned_high', '{b} <= {b}_at_most'))
_PIN_PARAMETERS = ('{b}_at_least', '{b}_at_most')

#: Every combination of the binaries is traced, so their count is an exponent:
#: past this many the trace is thousands of solves, and *at* is the way to ask
#: about fewer.
_MOST_PINNED = 10

#: One direction's probe is one solve, so a region that keeps producing
#: vertices past this many is not converging — noise the tolerance does not
#: absorb — and stopping is better than a driver that never returns.
_MOST_SOLVES = 1000

Point = tuple[float, float]
Binaries = Literal['free', 'each']

_NEEDS_THE_EXTRA = (
    'matplotlib ships with the [plot] extra rather than with the engine, so this build cannot draw: '
    'pip install "lpspec[plot]". A region needs nothing added to be read as it stands — its vertices '
    'are a polars frame, two columns any plotting library fills a polygon from.'
)


@dataclass(frozen=True)
class Region:
    """What :func:`project` hands back: the feasible region on two quantities, as tidy frames.

    Every frame keeps its schema whether the binaries were left free or
    traced one combination at a time: a region traced free is one piece,
    numbered ``0``, that pinned nothing.

    Attributes:
        x: The quantity on the horizontal axis.
        y: The quantity on the vertical axis.
        vertices: ``(piece, vertex, x, y)`` — every vertex of every piece,
            counter-clockwise from each piece's lowest-leftmost, ``vertex``
            counting from zero. A piece that is a segment has two rows and
            one that is a point one.
        hull: ``(vertex, x, y)`` — the region as one polygon: the piece
            itself where the binaries were free, the hull of the pieces where
            they were traced apart, which is what a free trace would give.
        pieces: ``(piece, variable, dims…, value)`` — what each piece pinned,
            one row per binary column, the coordinate as typed columns. No
            rows where the binaries were free.
        edges: ``(piece, edge, kind, name, dims…, side)`` — what bounds each
            edge: every variable bound and constraint row the solver sat on
            at both of the edge's ends, at the coordinates *at* named. Edge
            ``i`` runs from vertex ``i`` to the next; a segment has one edge.
            Binaries are left out, being always on a bound.
        optimum: ``(piece, x, y)`` — where the model *as written* lands, and
            in which piece; no rows where the spec declares no objective or
            the solve as written found no solution.
    """

    x: str
    y: str
    vertices: pl.DataFrame
    hull: pl.DataFrame
    pieces: pl.DataFrame
    edges: pl.DataFrame
    optimum: pl.DataFrame

    def label(self, piece: int) -> str:
        """The combination *piece* pinned, in one line — ``running[chp]=1, running[boiler]=1, running[peaker]=0``.

        A dim every row of a variable agrees on — the hour *at* fixed — is
        left out, so the label says what varies between the pieces; a label
        along a numbered dim keeps the dim's name, ``running[t=3]=1``.
        """
        pinned = self.pieces.filter(pl.col('piece') == piece)
        parts = []
        for (variable,), rows in pinned.group_by('variable', maintain_order=True):
            whole = self.pieces.filter(pl.col('variable') == variable)
            dims = [
                d
                for d in rows.columns
                if d not in ('piece', 'variable', 'value') and rows[d].null_count() == 0 and whole[d].n_unique() > 1
            ]
            for row in rows.iter_rows(named=True):
                coordinate = ', '.join(row[d] if isinstance(row[d], str) else f'{d}={row[d]}' for d in dims)
                parts.append(f'{variable}[{coordinate}]={row["value"]}' if dims else f'{variable}={row["value"]}')
        return ', '.join(parts)

    def plot(self, ax: Axes | None = None, *, optimum: bool = True, **style: Any) -> Axes:
        """Fill the region on a matplotlib axes, and return the axes.

        A polygon is filled and outlined, a segment drawn as a line, a point
        as a marker; the axes are labelled with the two quantities. Pieces
        traced apart are drawn each in its own colour under its
        :meth:`label`, so ``ax.legend()`` names the combinations. The
        model's own optimum is marked where there is one. Anything the
        picture should say beyond that is a call on the axes that comes back.

        Args:
            ax: Where to draw; a new figure's axes where none is given.
            optimum: Whether to mark :attr:`optimum`.
            style: Forwarded to matplotlib's ``fill``, ``plot`` or
                ``scatter``, whichever the shape calls for — a ``color``, an
                ``alpha``, a ``label`` for a legend.

        Raises:
            ModuleNotFoundError: On an install without matplotlib, naming the extra.
        """
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(_NEEDS_THE_EXTRA) from exc

        if ax is None:
            _, ax = plt.subplots()
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, (piece, polygon) in enumerate(self.vertices.group_by('piece', maintain_order=True)):
            own = (
                {'color': colours[i % len(colours)], 'label': self.label(piece[0])}
                if not self.pieces.is_empty()
                else {}
            )
            _draw(ax, polygon[self.x].to_list(), polygon[self.y].to_list(), {**own, **style})
        if optimum and not self.optimum.is_empty():
            ax.scatter(self.optimum[self.x], self.optimum[self.y], color='black', zorder=3, label='the optimum')
        ax.set_xlabel(self.x)
        ax.set_ylabel(self.y)
        return ax


def _draw(ax: Axes, xs: list[float], ys: list[float], style: dict[str, Any]) -> None:
    """One polygon, filled and outlined — or the line or marker a flat one is."""
    if len(xs) == 1:
        ax.scatter(xs, ys, **style)
    elif len(xs) == 2:
        ax.plot(xs, ys, **style)
    else:
        style.setdefault('alpha', 0.3)
        (patch,) = ax.fill(xs, ys, **style)
        ax.plot([*xs, xs[0]], [*ys, ys[0]], color=patch.get_facecolor(), alpha=1.0)


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
    binaries: Binaries = 'free',
) -> Region:
    """Trace the feasible region of *spec* on two of its quantities.

    ::

        region = lps.project('plant.yaml', sources, x='heat', y='power', at={'t': 5})
        region.vertices  # (piece, vertex, heat, power)
        region.edges  # which bound or row each edge sits on
        region.plot()  # filled, the optimum marked, on a matplotlib axes

    The region is every ``(x, y)`` some feasible solution reaches, which the
    objective plays no part in: the file's is set aside and the solve is
    driven by a direction instead. For a continuous model the polygon is
    exact — every vertex is a solve, and an edge is kept only once a solve
    along its outward normal finds nothing beyond it. The model's own
    optimum is one more solve, marked on the region it sits in.

    A binary makes the region a union of polygons, one per combination, and
    with the binaries **free** a solve along a direction finds only the
    **convex hull** of that union: each solve still returns an extreme point
    of it, so the hull is exact, and what it encloses may have holes it
    cannot show. ``binaries='each'`` traces the pieces instead: every
    combination of the binary columns *at* reaches — all of them, with no
    *at* — is pinned in turn and its region traced, so a plant's on/off
    states come back as the separate polygons they are. An infeasible
    combination is left out rather than raised — the first solve, with every
    binary free, is the one that says whether there is a region at all. An
    ``integer`` variable is never pinned, so a piece holding one is, again,
    a hull.

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
            own noise; and how close to a bound a value must sit to count as
            on it. A vertex is rounded to its decimals, so the solver's
            noise below it does not reach the frame either.
        binaries: ``free``, the hull of whatever the binaries allow, or
            ``each``, one piece per combination of the binary columns *at*
            reaches.

    Returns:
        The region, as its four frames and a ``plot``.

    Raises:
        KeyError: *x* or *y* names nothing declared.
        NoSolutionError: The model is infeasible, so there is no region.
        LpspecError: The region is unbounded along some direction, a dim in
            *at* one of the quantities does not carry, a name the probe adds
            already declared, a solve that stopped without a solution,
            ``binaries='each'`` on a model with no binary or with more binary
            columns at *at* than a trace of every combination can afford.
    """
    if isinstance(spec, program.Program):
        raise LpspecError(
            'project takes the spec as the path or mapping it was written as, not a Program: the '
            'probe rewrites the objective, which a lowered program no longer carries as a file.'
        )
    if x == y:
        raise LpspecError(f"project needs two different quantities; both axes are '{x}'.")
    if binaries not in ('free', 'each'):
        raise LpspecError(f"binaries is 'free' or 'each', not {binaries!r}.")
    solver(solver_name)
    declared = to_spec(spec).to_dict()
    at = dict(at or {})
    probe = _probing_spec(declared, x, y, at)
    data = {**sources, **dict(zip(_DIRECTIONS, _COMPASS[0], strict=True)), _OBJECTIVE_WEIGHT: 0.0}
    for selection in _SELECTIONS:
        if at:
            data[selection] = pl.DataFrame({**{d: [label] for d, label in at.items()}, 'value': [1.0]})
    pinned: list[str] = _binary_variables(declared) if binaries == 'each' else []
    binary = [name for name, v in (declared.get('variables') or {}).items() if v.get('domain') == 'binary']
    if pinned:
        probe = _pinned_spec(probe, pinned)
        for b in pinned:
            data.update(zip((p.format(b=b) for p in _PIN_PARAMETERS), (0.0, 1.0), strict=True))
    added = [row.format(b=b) for b in pinned for row, _ in _PIN_ROWS]

    with Model(probe, data) as model:
        _refuse_dims_at_does_not_reach(model, x, y, at)
        tight: dict[Point, pl.DataFrame] = {}

        def solve(direction: Point, weight: float = 0.0) -> Result:
            solving = {**dict(zip(_DIRECTIONS, direction, strict=True)), _OBJECTIVE_WEIGHT: weight}
            return model.update(solving).solve(solver_name, solver_options=solver_options, keep='progress')

        def read(result: Result) -> Point:
            return _snap((result.expression(_AXES[0]).item(), result.expression(_AXES[1]).item()), tolerance)

        def support(direction: Point) -> Point:
            with solve(direction) as result:
                _refuse_without_a_vertex(result, direction, x, y)
                point = read(result)
                if point not in tight:
                    tight[point] = _binding(model, result, tolerance, at, [*added, *binary])
                return point

        def traced() -> tuple[list[Point], dict[Point, pl.DataFrame]]:
            tight.clear()
            polygon = _trace(support, tolerance)
            return polygon, dict(tight)

        columns: list[_Column] = []
        traces: list[tuple[list[Point], dict[Point, pl.DataFrame]]] = []
        assignments: list[tuple[int, ...]] = []
        if not pinned:
            traces.append(traced())
            assignments.append(())
        else:
            with solve(_COMPASS[0]) as first:
                _refuse_without_a_vertex(first, _COMPASS[0], x, y)
                columns = _pinned_columns(first, pinned, at)
            for assignment in itertools.product((0, 1), repeat=len(columns)):
                model.update(_pins(columns, assignment))
                try:
                    traces.append(traced())
                except NoSolutionError:
                    continue
                assignments.append(assignment)
            model.update(_pins(columns))

        optimum = _optimum(solve, read, declared, columns, assignments, (x, y))

    dims = list(to_program(declared).dimensions)
    vertices = pl.concat(
        _frame(x, y, polygon).select(
            pl.lit(i, dtype=pl.Int64).alias('piece'), pl.int_range(pl.len(), dtype=pl.Int64).alias('vertex'), pl.all()
        )
        for i, (polygon, _) in enumerate(traces)
    )
    hull = _frame(x, y, _hull([p for polygon, _ in traces for p in polygon])).select(
        pl.int_range(pl.len(), dtype=pl.Int64).alias('vertex'), pl.all()
    )
    return Region(
        x,
        y,
        vertices,
        hull,
        _pieces_frame(columns, assignments, dims),
        _edges_frame(traces, dims),
        optimum,
    )


def _frame(x: str, y: str, vertices: Sequence[Point]) -> pl.DataFrame:
    return pl.DataFrame(
        {x: [p[0] for p in vertices], y: [p[1] for p in vertices]}, schema={x: pl.Float64, y: pl.Float64}
    )


def _in_dimension_order(
    frame: pl.DataFrame, leading: Sequence[str], dims: Sequence[str], trailing: Sequence[str]
) -> pl.DataFrame:
    """*frame* with its dim columns between *leading* and *trailing*, in the program's own order."""
    return frame.select(*leading, *(d for d in dims if d in frame.columns), *trailing)


def _pieces_frame(
    columns: Sequence[_Column], assignments: Sequence[Sequence[int]], dims: Sequence[str]
) -> pl.DataFrame:
    """What each piece pinned — ``(piece, variable, dims…, value)``, the coordinates as typed columns."""
    empty = pl.DataFrame(schema={'piece': pl.Int64, 'variable': pl.String, 'value': pl.Int64})
    frames = [
        column.coordinates.slice(column.row, 1).select(
            pl.lit(i, dtype=pl.Int64).alias('piece'),
            pl.lit(column.variable).alias('variable'),
            pl.all(),
            pl.lit(value, dtype=pl.Int64).alias('value'),
        )
        for i, assignment in enumerate(assignments)
        for column, value in zip(columns, assignment, strict=True)
    ]
    return _in_dimension_order(pl.concat([empty, *frames], how='diagonal'), ['piece', 'variable'], dims, ['value'])


def _edges_frame(traces: Sequence[tuple[list[Point], dict[Point, pl.DataFrame]]], dims: Sequence[str]) -> pl.DataFrame:
    """What bounds each edge of each piece — the bounds and rows sat on at both of its ends."""
    empty = pl.DataFrame(
        schema={'piece': pl.Int64, 'edge': pl.Int64, 'kind': pl.String, 'name': pl.String, 'side': pl.String}
    )
    frames = []
    for i, (polygon, tight) in enumerate(traces):
        for j, (a, b) in enumerate(_polygon_edges(polygon)):
            common = _both(tight[a], tight[b])
            frames.append(
                common.select(
                    pl.lit(i, dtype=pl.Int64).alias('piece'), pl.lit(j, dtype=pl.Int64).alias('edge'), pl.all()
                )
            )
    return _in_dimension_order(
        pl.concat([empty, *frames], how='diagonal'), ['piece', 'edge', 'kind', 'name'], dims, ['side']
    )


def _both(one: pl.DataFrame, other: pl.DataFrame) -> pl.DataFrame:
    """The rows *one* and *other* share — counted rather than joined, since each may carry dims the other lacks."""
    stacked = pl.concat([one, other], how='diagonal_relaxed')
    return stacked.group_by(stacked.columns, maintain_order=True).len().filter(pl.col('len') == 2).drop('len')


def _polygon_edges(polygon: Sequence[Point]) -> list[tuple[Point, Point]]:
    """Edge ``i`` from vertex ``i`` to the next — one for a segment, none for a point."""
    if len(polygon) < 2:
        return []
    if len(polygon) == 2:
        return [(polygon[0], polygon[1])]
    return [(polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon))]


def _binding(
    model: Model, result: Result, tolerance: float, at: Mapping[str, Any], excluded: Sequence[str]
) -> pl.DataFrame:
    """The bounds and rows *result* sits on, at the coordinates *at* names, *excluded* names left out.

    A row at another hour is on a bound because the solver parked it there,
    not because it bounds this hour's region; a row that carries none of the
    dims *at* names is kept, since it reaches every coordinate.
    """
    tight = model._engine.binding(
        result, tolerance
    )  # the driver reads what its own build holds, as strategy does the runner
    tight = tight.filter(~pl.col('name').is_in(list(excluded)))
    for dim, label in at.items():
        if dim in tight.columns:
            tight = tight.filter(pl.col(dim).is_null() | (pl.col(dim) == label))
    return tight


def _optimum(
    solve: Callable[[Point, float], Result],
    read: Callable[[Result], Point],
    declared: dict[str, Any],
    columns: Sequence[_Column],
    assignments: Sequence[Sequence[int]],
    axes: tuple[str, str],
) -> pl.DataFrame:
    """Where the model as written lands — ``(piece, x, y)``, or no rows where it has no objective or no solution."""
    objective = declared.get('objective')
    schema = {'piece': pl.Int64, axes[0]: pl.Float64, axes[1]: pl.Float64}
    if objective is None:
        return pl.DataFrame(schema=schema)
    weight = -1.0 if objective.get('sense', 'minimize') == 'minimize' else 1.0
    with solve((0.0, 0.0), weight) as result:
        if not result.has_primal:
            return pl.DataFrame(schema=schema)
        point = read(result)
        piece = 0
        if columns:
            found = tuple(_value_at(result, column) for column in columns)
            piece = list(assignments).index(found)
    return pl.DataFrame({'piece': [piece], axes[0]: [point[0]], axes[1]: [point[1]]}, schema=schema)


def _value_at(result: Result, column: _Column) -> int:
    """The binary *column*'s value in *result*, read at its coordinate."""
    primal = result.primal(column.variable)
    coordinate = column.coordinates.slice(column.row, 1)
    return round(primal.join(coordinate, on=coordinate.columns, how='semi').item(0, 'value'))


@dataclass(frozen=True)
class _Column:
    """One binary column to pin: which variable, the rows of its coordinate frame it is."""

    variable: str
    coordinates: pl.DataFrame
    row: int


def _binary_variables(declared: dict[str, Any]) -> list[str]:
    binary = [name for name, v in (declared.get('variables') or {}).items() if v.get('domain') == 'binary']
    if not binary:
        raise LpspecError(
            "binaries='each' pins every combination of the binary variables, and the spec declares none. "
            "Drop it: with nothing to pin, the region is the one 'free' traces."
        )
    return binary


def _probing_spec(declared: dict[str, Any], x: str, y: str, at: Mapping[str, Any]) -> dict[str, Any]:
    """The caller's model with the objective replaced by a direction over the two axes.

    Ordinary declarations, all of them — three scalar weights, two named
    expressions, and one selection parameter per axis where *at* fixes a
    coordinate — so the probe is a file the language validates, typesets and
    builds like any other, on both lanes. The file's own objective stays in
    as a third term under a weight of zero, so the optimum is one more solve
    on the same loaded model.
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
    taken = _taken(declared)
    if clashes := [name for name in (*_AXES, *_DIRECTIONS, *_SELECTIONS, _OBJECTIVE_WEIGHT) if name in taken]:
        raise LpspecError(
            f'project adds {clashes} to the model to probe it, and the spec already declares '
            f'{", ".join(f"{n} under {taken[n]}:" for n in clashes)}. Rename the declaration.'
        )
    scalar: dict[str, list[str]] = {'dims': []}
    parameters = {**(declared.get('parameters') or {}), **{d: dict(scalar) for d in (*_DIRECTIONS, _OBJECTIVE_WEIGHT)}}
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
    terms = [f'{_DIRECTIONS[0]} * {_AXES[0]}', f'{_DIRECTIONS[1]} * {_AXES[1]}']
    if (objective := declared.get('objective')) is not None:
        terms.append(f'{_OBJECTIVE_WEIGHT} * ({objective["expression"]})')
    return {
        **declared,
        'parameters': parameters,
        'expressions': {**expressions, **axes},
        'objective': {'sense': 'maximize', 'expression': ' + '.join(terms)},
    }


def _taken(declared: dict[str, Any]) -> dict[str, str]:
    """Every name the spec declares, to the block it is declared under."""
    return {
        name: kind
        for kind in ('parameters', 'variables', 'expressions', 'constraints', 'lookups', 'macros')
        for name in (declared.get(kind) or {})
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


def _pinned_spec(probe: dict[str, Any], pinned: Sequence[str]) -> dict[str, Any]:
    """*probe* with a pair of rows per binary holding it between two data values.

    Free is ``0 <= b <= 1``, which changes nothing; a combination sets both
    sides to the same value at the columns it pins. Both sides are data, so
    moving between combinations is a right-hand side pushed onto the loaded
    solver rather than a mask moved and a model reloaded.
    """
    taken = _taken(probe)
    added = [p.format(b=b) for b in pinned for p in (*_PIN_PARAMETERS, *(row for row, _ in _PIN_ROWS))]
    if clashes := [name for name in added if name in taken]:
        raise LpspecError(
            f"binaries='each' adds {clashes} to the model to pin its binaries, and the spec already "
            f'declares {", ".join(f"{n} under {taken[n]}:" for n in clashes)}. Rename the declaration.'
        )
    parameters = dict(probe['parameters'])
    constraints = dict(probe.get('constraints') or {})
    for b in pinned:
        dims = list(probe['variables'][b]['foreach'])
        for parameter in _PIN_PARAMETERS:
            parameters[parameter.format(b=b)] = {'dims': dims}
        for row, expression in _PIN_ROWS:
            constraints[row.format(b=b)] = {'foreach': dims, 'expression': expression.format(b=b)}
    return {**probe, 'parameters': parameters, 'constraints': constraints}


def _pinned_columns(first: Result, pinned: Sequence[str], at: Mapping[str, Any]) -> list[_Column]:
    """Every binary column *at* reaches, read off the first solve's primal.

    The primal is where a variable's coordinates are known: its ``where``
    has been applied, so a column a mask removed is not here to pin, and the
    labels come back typed as the dimension declares them. A dim *at* names
    that the variable carries selects the label; one it does not carry
    leaves every label in.
    """
    columns: list[_Column] = []
    for b in pinned:
        coordinates = first.primal(b).drop('value')
        chosen = coordinates.with_row_index('__row__')
        for dim, label in at.items():
            if dim in coordinates.columns:
                chosen = chosen.filter(pl.col(dim) == label)
        columns += [_Column(b, coordinates, row) for row in chosen['__row__'].to_list()]
    if len(columns) > _MOST_PINNED:
        raise LpspecError(
            f"binaries='each' would trace every combination of {len(columns)} binary columns, which is "
            f'{2 ** len(columns)} regions; the most it traces is {2**_MOST_PINNED}. Name coordinates in at '
            f'to ask about fewer.'
        )
    return columns


def _pins(columns: Sequence[_Column], assignment: Sequence[int] | None = None) -> dict[str, pl.DataFrame]:
    """The two bound tables per pinned variable, holding *assignment* — every column free where there is none."""
    tables: dict[str, pl.DataFrame] = {}
    for column in columns:
        for parameter, free in zip(_PIN_PARAMETERS, (0.0, 1.0), strict=True):
            tables.setdefault(
                parameter.format(b=column.variable), column.coordinates.with_columns(pl.lit(free).alias('value'))
            )
    if assignment is None:
        return tables
    for column, value in zip(columns, assignment, strict=True):
        for parameter in _PIN_PARAMETERS:
            name = parameter.format(b=column.variable)
            tables[name] = tables[name].with_columns(
                pl.when(pl.int_range(pl.len()) == column.row)
                .then(float(value))
                .otherwise(pl.col('value'))
                .alias('value')
            )
    return tables


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


def _refuse_without_a_vertex(result: Result, direction: Point, x: str, y: str) -> None:
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


def _trace(support: Callable[[Point], Point], tolerance: float) -> list[Point]:
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
    """The directed edges to probe — both ways along a segment, none for a point."""
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


def _snap(point: Point, tolerance: float) -> Point:
    """The point rounded to the tolerance's own decimals — two values it cannot tell apart print the same."""
    decimals = max(0, math.ceil(-math.log10(tolerance)))
    return round(point[0], decimals), round(point[1], decimals)


def _scale(points: Sequence[Point]) -> float:
    """What a tolerance is relative to: the region's own size, never below one."""
    return max(1.0, *(abs(c) for p in points for c in p))


def _near(a: Point, b: Point, tolerance: float) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= tolerance * max(1.0, abs(a[0]), abs(a[1]))
