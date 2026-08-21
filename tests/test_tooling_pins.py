"""Two versions written down twice each. This is what keeps each pair equal.

ruff first. The version lives in ``pyproject.toml`` (``ruff==`` in the dev
group, which is what CI installs and runs) and again in
``.pre-commit-config.yaml`` (the ``ruff-pre-commit``
rev, which is what the hook installs into its own isolated environment).
Nothing makes them agree on its own.

Dependabot manages both, but in *separate* PRs — it groups within an ecosystem
and never across one, so a ruff release produces one PR against the dev group
and another against the hook rev. Merge either alone and the formatter that
runs on commit is a different version from the one that gates the branch. That
shows up as a commit that was clean locally and fails CI, or the reverse, and
the cause is two files nobody thought to read together.

So: fail here instead, on the merge that introduced the skew, naming both
files. The fix is always to land the other PR.

Then the dependency floors. ``[project.dependencies]`` declares each one as a
lower bound, and the ``floors`` pixi environment pins the same package to that
exact version — the environment the ``ci`` job runs the suite in to prove the
bound is real rather than decorative. The version is therefore written twice in
one file, and this is what stops the copies drifting: raise a floor and the pin
has to move with it, which is the whole of the fix.

A direct reference (``math-spec @ git+…``) is the exception and is excluded
from both sides: it is already exact, so there is no bound to prove, and the
repository it points at claims its own floors.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: The `rev:` on the ruff-pre-commit repo block, e.g. `v0.16.0`.
_HOOK_REV = re.compile(r'ruff-pre-commit\s*\n\s*rev:\s*(\S+)')


def _pinned_in_pyproject() -> str:
    groups = tomllib.loads((REPO / 'pyproject.toml').read_text())['dependency-groups']
    pins = [spec for spec in groups['dev'] if spec.startswith('ruff==')]
    assert len(pins) == 1, f'expected exactly one `ruff==` pin in the dev group, found {pins}'
    return pins[0].removeprefix('ruff==')


def _pinned_in_pre_commit() -> str:
    match = _HOOK_REV.search((REPO / '.pre-commit-config.yaml').read_text())
    assert match is not None, 'no `rev:` found for the ruff-pre-commit repo'
    # the hook tags releases as `vX.Y.Z`; the package is `X.Y.Z`
    return match[1].removeprefix('v')


def test_ruff_is_the_same_version_in_ci_and_in_the_hook():
    pyproject, pre_commit = _pinned_in_pyproject(), _pinned_in_pre_commit()
    assert pyproject == pre_commit, (
        f'ruff is {pyproject} in pyproject.toml but {pre_commit} in .pre-commit-config.yaml — '
        f'the hook and CI would disagree about formatting. Dependabot bumps these in two '
        f'separate PRs; land the other one, or match them by hand.'
    )


#: `polars>=1.30` -> ('polars', '1.30'). A runtime dependency declared as a bare
#: lower bound is a claim the `floors` environment has to pin to prove.
_FLOOR = re.compile(r'^([A-Za-z0-9._-]+)>=([0-9][0-9a-zA-Z.]*)$')

#: `math-spec @ git+https://…@v0.0.0-alpha.4`. A direct reference is already
#: exact, so it has no lower bound to prove and nothing to pin — it must stay
#: out of the floors environment rather than be pinned twice.
_DIRECT = re.compile(r'^([A-Za-z0-9._-]+) @ \S+$')


def _declared_floors() -> dict[str, str]:
    declared = tomllib.loads((REPO / 'pyproject.toml').read_text())['project']['dependencies']
    unparsed = [spec for spec in declared if not (_FLOOR.match(spec) or _DIRECT.match(spec))]
    assert not unparsed, (
        f'{unparsed} is neither a bare `name>=version` lower bound nor a direct reference. A '
        f'runtime dependency written any other way has no floor for the `floors` environment to '
        f'pin, so teach this pattern the new shape rather than leaving the dependency unchecked.'
    )
    return {match[1]: match[2] for spec in declared if (match := _FLOOR.match(spec))}


def _pinned_in_the_floors_environment() -> dict[str, str]:
    pixi = tomllib.loads((REPO / 'pyproject.toml').read_text())['tool']['pixi']
    pinned = pixi['feature']['floors']['pypi-dependencies']
    # `lpspec` is the project under test and `pytest` is the runner; neither is a
    # dependency whose floor is being claimed.
    return {name: spec for name, spec in pinned.items() if isinstance(spec, str) and name != 'pytest'}


def test_the_floors_environment_pins_every_declared_lower_bound():
    declared, pinned = _declared_floors(), _pinned_in_the_floors_environment()
    assert pinned == {name: f'=={floor}' for name, floor in declared.items()}, (
        f'the `floors` pixi environment pins {pinned}, but `[project.dependencies]` declares '
        f'{declared}. That environment exists to prove each declared lower bound is real, which it '
        f'only does while it installs exactly those versions and nothing the project does not '
        f'declare — both directions, so a dependency added without a pin fails here too.'
    )
