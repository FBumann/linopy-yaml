"""What sits above both lanes: what a spec may arrive as, and what each lane can build.

Neither fact belongs to a lane. ``Buildable`` is what every verb in the
package takes; ``LANES`` is read by ``check`` with no extra installed and by
the eager lane when it refuses, so it is data here rather than a property of
a lane that may not be importable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

from lpspec.relational.sinks.capabilities import Capabilities

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from math_spec import Spec
    from math_spec.program import Program

#: Anything a verb takes as the spec: a YAML path, a mapping, or a spec the
#: language has already read — a ``Spec`` from :func:`math_spec.to_spec`, or a
#: ``Program`` from :func:`lpspec.check`. Each is handed straight to
#: :func:`math_spec.to_program`.
Buildable: TypeAlias = 'str | Path | dict[str, Any] | Spec | Program'

#: What each **lane** can build, beside what each sink can ingest: both lanes
#: accept the same language, and one cannot build a quadratic constraint —
#: ``linopy.Model.add_constraints`` refuses a ``QuadraticExpression`` outright
#: and no reformulation of it is exact.
LANES: Mapping[str, Capabilities] = {
    'linopy': Capabilities(
        supports={
            'integrality': 'native',
            'sos': 'native',
            'quadratic_objective': 'native',
            'nonconvex_quadratic_objective': 'native',
        },
    ),
}
