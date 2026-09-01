# SPDX-FileCopyrightText: math-spec Contributors
#
# SPDX-License-Identifier: MIT

"""The coverage question the block-level one cannot reach, as a rule with no data in it.

Its own module because it is *pure* — stamps and a program in, sentences out —
and `parity.py` cannot be imported without pypsa. A rule nothing can test
without a network is a rule nobody tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from math_spec.program import conjuncts

if TYPE_CHECKING:
    from collections.abc import Mapping


def untested_conjuncts(name: str, program: Any, stamps: list[Mapping[str, Any]]) -> list[str]:
    """Every conjunct of every mask true somewhere on the ladder and false somewhere.

    The half the block-level coverage cannot reach. A mask is exercised as a
    whole the moment one of its conjuncts varies, so `committable AND
    p_nom_mod > 0` passes while `committable` is true at every coordinate of
    every rung — and a regime guarded by that conjunct alone would be missing
    with nothing to say so. That is the shape the negative-`p_min_pu` gap had,
    found by hand (math-spec#312); this is the sweep that would have found it.

    Each rung records one character per conjunct — ``t`` held everywhere, ``f``
    nowhere, ``b`` at some coordinates, ``-`` no frame at all. A conjunct is
    exercised when the ladder has it true somewhere (``t`` or ``b``) and false
    somewhere (``f`` or ``b``), which one ``b`` satisfies alone and two rungs
    disagreeing satisfy between them.

    A conjunct of a block no rung builds is not reported: that block is already
    a louder gap one line up, and saying it twice buries it.
    """
    gaps = []
    for block_name, block in (program.constraints | program.variables).items():
        if getattr(block, 'where', None) is None:
            continue
        seen = [stamp['conjuncts'][block_name] for stamp in stamps if block_name in stamp.get('conjuncts', {})]
        for position, conjunct in enumerate(conjuncts(block.where)):
            marks = {verdicts[position] for verdicts in seen if position < len(verdicts)} - {'-'}
            if not marks:
                continue
            if not marks & {'t', 'b'}:
                gaps.append(f'{name}: {block_name} is never true at conjunct {position} — {conjunct!r}')
            elif not marks & {'f', 'b'}:
                gaps.append(f'{name}: {block_name} is never false at conjunct {position} — {conjunct!r}')
    return gaps
