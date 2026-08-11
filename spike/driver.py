"""What the second driver would look like, written against `solve_over`'s vocabulary.

`benders.py` proves the algorithm runs. `walls.py` shows `solve_over` cannot
carry it. This asks the next question: **is the sibling a small thing?**

The shape it needs, from the walls:

- an axis that is *not* known up front — each step is produced from the last
  step's answer (wall 1)
- a stopping rule, since a fold visits every slice and this stops on a bound
  (wall 5)
- state that *appends* rather than replaces, because a cut set grows (wall 3)

It deliberately borrows what already exists rather than inventing beside it:
`Runs`' idea of frames keyed by step, and nothing else. The question this file
answers is whether that borrowing is comfortable or forced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class Step:
    """One iteration's answer, and what it added to the state."""

    index: int
    objective: float
    bound: float
    gap: float


@dataclass
class Iteration:
    """What an iterative driver returns: the steps, and the state they built.

    `Runs` keys frames by *slice* because a sweep's slices are siblings. Here
    they are a sequence, so the key is the step index and the interesting frame
    is the one the steps *accumulated* — the cut table — which no sweep has an
    equivalent of.
    """

    steps: list[Step] = field(default_factory=list)
    state: dict[str, pl.DataFrame] = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        return bool(self.steps) and self.steps[-1].gap <= 0.0

    def history(self) -> pl.DataFrame:
        return pl.DataFrame([vars(step) for step in self.steps])


def iterate(
    step: Callable[[int, dict[str, pl.DataFrame]], tuple[Step, dict[str, pl.DataFrame]]],
    *,
    state: dict[str, pl.DataFrame],
    until: Callable[[Step], bool],
    limit: int = 50,
) -> Iteration:
    """Run *step* until *until* says stop, threading the state it grows.

    The whole of the sibling, and it is eleven lines: a loop, a state dict, a
    predicate. Everything hard about Benders — what a cut *is*, how the duals
    become one, which plan is master — lives in the caller's `step`, where it
    belongs, because those are properties of the algorithm and not of the driver.

    Note what is *not* here. No axis: the steps are not slices of anything, so
    there is nothing to partition. No `carry` mapping parameters to variables:
    the state is frames the caller assembles, because a cut is arithmetic over
    duals and `carry`'s copy-only rule exists precisely to keep arithmetic in
    the YAML where the oracle can see it. That rule is right for a fold and
    cannot hold here, which is the sharpest evidence the two are siblings and
    not one thing with a flag.
    """
    run = Iteration(state=dict(state))
    for index in range(limit):
        taken, run.state = step(index, run.state)
        run.steps.append(taken)
        if until(taken):
            return run
    raise AssertionError(f'no convergence in {limit} steps')
