"""Equilibrate a built model, so a solver sees one scale rather than the modeller's units.

A variable's unit is a **column scaling**: it multiplies that variable's
objective coefficient and its matrix column by the same factor. So a modeller
repairing a badly scaled model by redeclaring capacity in kW is choosing a
column factor by hand, and cannot shrink the objective's spread without growing
the matrix's. This module chooses those factors instead, and the model file
keeps saying MW.

**The objective is equilibrated as one more row of the matrix**, which is the
whole of why this works. Equilibrating the matrix alone moves the spread into
the costs — measured, on models whose costs started perfectly scaled — and the
costs are where capacity-expansion models actually hurt (#997).

**An integer column is never scaled.** ``s·x'`` is integral only for integral
``s``, so those columns pin at 1 and whatever spread they carry survives. On a
model that is mostly integer, that is most of the spread.

The factors are chosen by Ruiz equilibration in log space: alternately centre
every row's largest and smallest log-magnitude on zero, then every column's.
What no diagonal scaling can remove is the part of the spread that does not
separate into a per-row and a per-column factor — for any four nonzeros forming
a two-by-two, the cross-ratio ``(a_ij·a_kl)/(a_il·a_kj)`` is invariant, so a
cross-ratio of ``1e9`` leaves a floor of ``√1e9``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    import numpy.typing as npt

#: How many alternating row/column passes an equilibration takes. Ruiz
#: converges geometrically and the models this exists for are within a few
#: percent of their floor by five; the sweeps are the cost, so there is no
#: tolerance loop to stop early on and nothing to tune.
SWEEPS = 5


@dataclass(frozen=True)
class Scaling:
    """The factors one equilibration chose, and the inverse they oblige.

    Every vector a solve hands back is in the scaled model's units, so nothing
    may reach a caller without passing through the method for it here. The
    forward application is the engine's — it owns the frames — and this owns
    the way back, so the two cannot drift apart.
    """

    #: ``s_j``, per column: what the solver's value for column ``j`` is
    #: multiplied by to recover the model's own. ``1.0`` for an integer column
    #: and for a column no equilibration reached.
    columns: npt.NDArray[np.float64]

    #: ``r_i``, per row: what row ``i``'s coefficients and right-hand side were
    #: multiplied by.
    rows: npt.NDArray[np.float64]

    #: What the objective row was multiplied by — the one factor that is a
    #: scalar, the objective being one row however many costs it holds.
    objective: float

    def primal(self, values: pl.Series) -> pl.Series:
        """*values* in the model's own units — ``x = s ∘ x'``."""
        return _over_the_model(values, self.columns)

    def dual(self, values: pl.Series) -> pl.Series:
        """*values* in the model's own units — ``y = (r ∘ y') / r_obj``.

        Both factors: a row's scaling moves its dual, and scaling the objective
        moves every dual with it.
        """
        return _over_the_model(values, self.rows / self.objective)

    def activity(self, values: pl.Series) -> pl.Series:
        """*values* in the model's own units — ``act = act' / r``.

        The reciprocal of what a dual takes: a row's activity is its left-hand
        side, which the row's own factor multiplied.
        """
        return _over_the_model(values, 1.0 / self.rows)

    def objective_value(self, value: float) -> float:
        """*value* in the model's own money — the objective row's factor, undone."""
        return value / self.objective


def equilibrate(
    row: npt.NDArray[np.int64],
    col: npt.NDArray[np.int64],
    coeff: npt.NDArray[np.float64],
    obj_col: npt.NDArray[np.int64],
    obj_coeff: npt.NDArray[np.float64],
    integral: npt.NDArray[np.bool_],
    n_rows: int,
    n_cols: int,
) -> Scaling:
    """Factors that bring *coeff* and *obj_coeff* to a common scale.

    The objective enters as row ``n_rows``, which is what makes the two
    equilibrate against each other rather than one at the other's expense.

    Args:
        row: Row index per matrix entry.
        col: Column index per matrix entry.
        coeff: The matrix entry itself; only its magnitude is read.
        obj_col: Column index per objective coefficient.
        obj_coeff: The objective coefficient itself.
        integral: Per column, whether it may not be scaled.
        n_rows: Rows in the matrix, the objective's own row excluded.
        n_cols: Columns in the model.

    Returns:
        The factors, ready to apply and to undo. An empty matrix and an empty
        objective give the identity rather than an error — a model with no
        entries is scaled correctly by leaving it alone.
    """
    joint_row = np.concatenate([row, np.full(len(obj_col), n_rows, dtype=np.int64)])
    joint_col = np.concatenate([col, obj_col])
    magnitude = np.abs(np.concatenate([coeff, obj_coeff]))
    if not magnitude.size:
        return Scaling(np.ones(n_cols), np.ones(n_rows), 1.0)

    log = np.log(magnitude)
    r, c = np.zeros(n_rows + 1), np.zeros(n_cols)
    for _ in range(SWEEPS):
        r += _centre(log + r[joint_row] + c[joint_col], joint_row, n_rows + 1)
        c += np.where(integral, 0.0, _centre(log + r[joint_row] + c[joint_col], joint_col, n_cols))
    return Scaling(np.exp(c), np.exp(r[:n_rows]), float(np.exp(r[n_rows])))


def _over_the_model(values: pl.Series, factors: npt.NDArray[np.float64]) -> pl.Series:
    """*values* times *factors*, leaving whatever a sink appended past them alone.

    A solver with no SOS concept is handed binaries and linking rows past the
    model's own (:func:`~lpspec.relational.sinks.sos.reformulated`), so the
    vector it answers with can be longer than the model has columns or rows.
    Those entries are the reformulation's and were never scaled — multiplying
    them by a factor belonging to a different column is the bug this exists to
    not have.
    """
    if values.len() == len(factors):
        return values * factors
    return pl.concat([values.slice(0, len(factors)) * factors, values.slice(len(factors))])


def _centre(value: npt.NDArray[np.float64], index: npt.NDArray[np.int64], groups: int) -> npt.NDArray[np.float64]:
    """Per group, the step that puts its largest and smallest *value* either side of zero.

    *value* carries the factors chosen so far, so what comes back is an
    increment and not the factor itself — computing it off the bare magnitudes
    would be an assignment, and adding that to what is already there scales the
    model twice over.

    Zero for a group with no entry at all: an empty row's extremes are
    ``±inf``, whose midpoint is not a number, and a row the matrix never
    reaches has no scale to be wrong about.
    """
    high = np.full(groups, -np.inf)
    low = np.full(groups, np.inf)
    np.maximum.at(high, index, value)
    np.minimum.at(low, index, value)
    with np.errstate(invalid='ignore'):  # an empty group is `-inf + inf`, filtered on the next line
        step = -0.5 * (high + low)
    return np.where(np.isfinite(step), step, 0.0)
