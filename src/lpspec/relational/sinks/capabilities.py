"""What a sink can ingest — the axis that is not the ceiling.

The ceiling is about streamability and is solver-independent
(docs/about/ceiling.md); what a *sink* can take is separate, and conflating the
two let one solver's limits read as architectural law. One descriptor per sink,
so a construct the language says and a sink cannot take is a refusal naming
both rather than a ``kError`` from inside a library.

Four shapes, each forced by docs/about/benchmarks.md#sink-capabilities rather
than chosen:

- **Three-valued.** ``reformulated`` is an answer, not a missing ``native``: a
  set reaches HiGHS as binaries and linking rows and the model still solves, at
  the cost of the duals an LP would have returned.
- **Exclusions.** HiGHS takes a Hessian, takes integrality, and refuses the
  pair — which linopy's flat ``frozenset`` reports as MIQP available.
- **Some entries are data-time.** Convexity is a property of coefficients, so
  ``check`` cannot answer it (rule 2). ``nonconvex_quadratic_objective`` is
  declared anyway: the sink that discovers it at solve time reads the sinks
  that would have taken it off this table.
- **A descriptor describes the sink as shipped, not the library it wraps.** A
  solver that takes a Hessian through an entry point this package does not call
  cannot ingest a quadratic objective *here*, and saying otherwise would drop
  the quadratic part and answer a different model's optimum. An entry moves
  when the hand-off lands, so the benchmarks table may be ahead of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, get_args

from lpspec.relational import plan

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

#: What a model may need a sink to have. ``indicator`` and ``semi-continuous``
#: are absent deliberately: they have rows in the benchmarks table and no
#: spelling in the language (#220, #383), so an entry would be a fact nothing
#: can consult.
Capability = Literal[
    'integrality',
    'sos',
    'quadratic_objective',
    'nonconvex_quadratic_objective',
    'quadratic_constraint',
]

#: How a sink satisfies one capability. ``reformulated`` means the model is
#: rewritten into what the sink does take — a worse relaxation rather than a
#: refusal, which a caller needs to know before choosing a sink.
Support = Literal['native', 'reformulated', 'absent']

CAPABILITIES: tuple[Capability, ...] = get_args(Capability)

#: Those whose ``reformulated`` rewrite is binaries and linking rows
#: (:func:`~lpspec.relational.sinks.sos.reformulated`): a sink promising such
#: a rewrite must take integrality to perform it, and a model that declared
#: none reaches that sink mixed-integer, so it comes back without the duals an
#: LP would have returned.
REWRITTEN_AS_INTEGRALITY: frozenset[Capability] = frozenset({'sos'})


@dataclass(frozen=True)
class Capabilities:
    """One sink's answer for every capability, and the pairs it refuses.

    Attributes:
        supports: What the sink does with each capability it has; a name left
            out is ``absent``, so a descriptor lists only what it *can* do.
        excludes: Sets of capabilities it has individually and refuses together.
    """

    supports: Mapping[Capability, Support]
    excludes: tuple[frozenset[Capability], ...] = ()

    def __post_init__(self) -> None:
        """Take a read-only copy of *supports*, which a sink holds as a ``ClassVar``.

        A plain ``dict`` behind a frozen field is process-wide mutable state
        one attribute lookup away, and a test that reached it would change what
        every later sink answers.
        """
        object.__setattr__(self, 'supports', MappingProxyType(dict(self.supports)))

    def support(self, capability: Capability) -> Support:
        """What this sink does with *capability* — ``absent`` where it says nothing."""
        return self.supports.get(capability, 'absent')

    def missing(self, required: Collection[Capability]) -> list[Capability]:
        """Those of *required* this sink cannot take at all.

        In :data:`CAPABILITIES` order rather than the caller's, so a refusal
        naming two of them reads the same way twice.
        """
        return [c for c in CAPABILITIES if c in required and self.support(c) == 'absent']

    def excluded(self, required: Collection[Capability]) -> frozenset[Capability] | None:
        """The first conjunction *required* contains that this sink refuses.

        Returns:
            The excluded set, or ``None``. Each member is one the sink supports
            on its own; one it simply lacks is :meth:`missing`'s answer.
        """
        for combination in self.excludes:
            if combination <= set(required):
                return combination
        return None


def required(program: plan.Program, /) -> frozenset[Capability]:
    """What *program* needs a sink to have, decided with no data bound.

    Exactly what the model declares, so a refusal built on it names constructs
    the reader can find in their own file. What a *rewrite* then costs is the
    performing sink's own fact and is declared there: HiGHS excludes a set
    beside a Hessian because what it is handed for one is binaries, and a
    requirement derived here instead would refuse the model for integrality it
    never stated.

    Only what rule 2 can decide appears here, so convexity never does.
    """
    needed: set[Capability] = set()
    if any(v.variable_type != 'continuous' for v in program.variables):
        needed.add('integrality')
    if program.objective is not None and plan.is_quadratic(program.objective.expression):
        needed.add('quadratic_objective')
    if any(plan.declares_quadratic(c) for c in program.constraints):
        needed.add('quadratic_constraint')
    if program.sos:
        needed.add('sos')
    return frozenset(needed)
