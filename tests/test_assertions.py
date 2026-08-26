"""The assertion-message rule, checked rather than reviewed.

AGENTS.md asks for the claim in the message that prints. Reviewed by eye that
held at 59% of the 1,583 assertions here, and the distribution said age rather
than disagreement — 5% in the oldest file, 96% in the newest. A rule at 59% with
no way to say whether that is bad is unenforceable in both directions.

**100% is the wrong target.** ``assert result.status == 'optimal'`` needs no
sentence, and demanding one produces exactly the restatement AGENTS.md tells you
to cut. So the rule checked here is narrower than the one written there, and
covers only assertions whose *claim is not visible in the expression*:

- **a literal collection** — ``== ['high', 'low', 'mid']`` claims an order and a
  completeness, and neither is on the line;
- **a count** — ``len(rows) == 4`` never says why four;
- **a tolerance** — ``approx(x, rel=1e-9)`` is a chosen precision;
- **an absence** — ``assert not offenders`` says what is empty, never why it
  must be.

**A ceiling, not a floor, and not a backfill.** 247 assertions were in breach
when this landed, and adding 247 sentences in one pass would produce 247
restatements rather than 247 claims. So the number is pinned: a new one turns
the suite red, and the count comes down as the files are touched for other
reasons. It is the same instrument the expression sweep's lane-refusal ceiling
is, for the same reason — a rule nobody can measure is a rule that drifts.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent

#: Assertions in breach when the rule was mechanised. A ratchet: this may fall,
#: and a PR that raises it is adding an assertion whose claim is not written
#: down. Lower it in the PR that lowers the count.
IN_BREACH = 222


def _unwritten_claim(node: ast.Assert) -> str | None:
    """Why this assertion's claim is not visible in its expression, if it is not."""
    test = node.test
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return 'an absence'
    if not isinstance(test, ast.Compare):
        return None
    for side in (test.left, *test.comparators):
        if isinstance(side, (ast.List, ast.Set, ast.Tuple)) and side.elts:
            return 'a literal collection'
        if isinstance(side, ast.Dict) and side.keys:
            return 'a literal collection'
        if isinstance(side, ast.Call):
            if isinstance(side.func, ast.Name) and side.func.id == 'len':
                return 'a count'
            if 'approx' in ast.dump(side.func):
                return 'a tolerance'
    return None


def _in_breach() -> list[str]:
    """``path:line — why`` for every assertion the narrowed rule wants a message on."""
    found = []
    for path in sorted(TESTS.rglob('*.py')):
        tree = ast.walk(ast.parse(path.read_text()))
        found += [
            f'{path.relative_to(TESTS.parent)}:{node.lineno} — {why}'
            for node in tree
            if isinstance(node, ast.Assert) and node.msg is None and (why := _unwritten_claim(node))
        ]
    return found


def test_an_assertion_whose_claim_is_not_on_the_line_carries_a_message() -> None:
    """The ratchet holds in both directions, off one pass over the tree.

    Above the constant, an assertion was added whose claim is not written
    down; below it, the count fell and the ceiling did not — and a ceiling
    nobody lowers is a ceiling that stops meaning anything, since the next
    unmessaged assertion lands under the old headroom unremarked.
    """
    breach = _in_breach()
    assert len(breach) <= IN_BREACH, (
        f'{len(breach)} assertions state a claim their expression does not carry, above the {IN_BREACH} '
        f'this ratchet was set at — put the claim in the message, since a message is what prints. '
        f'In breach, in full:\n  ' + '\n  '.join(breach)
    )
    assert len(breach) == IN_BREACH, (
        f'{len(breach)} assertions are in breach but IN_BREACH says {IN_BREACH} — set it to {len(breach)} '
        f'in this PR, so the headroom the check leaves is the headroom the tree actually has'
    )
