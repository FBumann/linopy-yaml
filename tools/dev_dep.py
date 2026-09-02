"""Install a local checkout of an unreleased dependency into the envs that run code.

    pixi run --no-install python -m tools.dev_dep ../path/to/math-spec

A plain ``pixi run`` re-syncs an environment to the locked pin every time, so a
hand-installed build is reverted under you — the symptom is a symbol that was
there a moment ago going missing. This installs the checkout into the two
environments that execute code, ``default`` and ``docs`` (the docs build runs
``docs/interactive.ipynb``), after which every task must be run with
``pixi run --no-install`` to keep the build in place.

``floors`` is left alone deliberately: it tests the shipped shape at the pinned
floors, so an override there would test the wrong thing.

See AGENTS.md, "Developing against an unreleased math-spec", for the whole loop
— including the pin that stays released and the ``pyrefly`` interpreter flag.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

#: The environments that import and run the dependency. ``floors`` is excluded
#: on purpose — see the module docstring.
ENVS = ('default', 'docs')


def install(dependency: Path) -> None:
    """Install *dependency* into each environment in :data:`ENVS`, via uv."""
    for env in ENVS:
        interpreter = Path('.pixi/envs') / env / 'bin' / 'python'
        if not interpreter.exists():
            raise SystemExit(f"no '{env}' environment yet — run `pixi install` first")
        command = ['pixi', 'exec', '-s', 'uv', 'uv', 'pip', 'install', '--python', str(interpreter), str(dependency)]
        subprocess.run(command, check=True)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('usage: python -m tools.dev_dep <path-to-local-checkout>')
    dependency = Path(sys.argv[1]).expanduser().resolve()
    if not dependency.exists():
        raise SystemExit(f'no such path: {dependency}')
    install(dependency)
    print(f'installed {dependency} into {", ".join(ENVS)} — now run tasks with `pixi run --no-install <task>`')


if __name__ == '__main__':
    main()
