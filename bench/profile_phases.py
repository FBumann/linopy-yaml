"""Attribute build wall time to phases, in seconds you can compare to a real run.

``profile_build.py`` answers *which query*; this answers *which phase, and how
much of the build is it*. The two are complements and the difference is the
instrument: that one wraps every ``LazyFrame.collect``, which identifies queries
precisely and adds enough Python per call that its own docstring says not to
quote its seconds. This one wraps three methods per build, so what it prints is
comparable to ``bench/`` — at the cost of saying nothing about what is inside
them.

    pixi run -e bench python -m bench.profile_phases profiled l
    pixi run -e bench python -m bench.profile_phases transport l --rounds 15

**Everything that is not the build is hoisted out of the loop.** Timing
``lps.build`` repeatedly measures a YAML parse, a lowering pass, a parquet read
and the assembly at once — and the parquet read drags the page cache in with it,
which is why repeated runs of identical code spread 12-55% and nothing smaller
than a rewrite shows up above the noise. Parsing and lowering happen once here,
and binding happens once and is then *reused*: ``BoundSources`` is frozen by
contract, so handing the same one to every round is legitimate and leaves the
assembly alone in the measurement. Measured spread drops to a few percent, which
is what makes a 10% change visible at all.

Binding is not skipped, it is *separated*: the first pass runs it for real, and
the difference between the two minima is what reading and validating the sources
costs. On ``profiled/l`` that is a third of the build and on ``dispatch/l`` it
rounds to nothing, which is the kind of thing a single number hides.
"""

from __future__ import annotations

import argparse
import collections
import statistics
import time
from typing import Any

from bench import cases as bench_cases

#: Wrapped per build rather than per collect. Three calls of overhead against a
#: few hundred milliseconds of work is why these seconds mean something.
PHASES = ('_build_variable', '_build_constraint', '_build_objective')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m bench.profile_phases')
    parser.add_argument('case', choices=sorted(bench_cases.CASES))
    parser.add_argument('size')
    parser.add_argument('--rounds', type=int, default=9, help='timed builds per arm; the minimum is reported')
    args = parser.parse_args(argv)

    from math_spec import load_model

    from lpspec.lowering import lower_program
    from lpspec.relational.engines.polars import engine as executor_module
    from lpspec.relational.engines.polars.engine import PolarsEngine, _Assembly
    from lpspec.sources import tidy_sources

    spent: dict[str, list[float]] = collections.defaultdict(list)
    for name in PHASES:
        setattr(_Assembly, name, _timed(name, getattr(_Assembly, name), spent))

    case = bench_cases.CASES[args.case]
    shape = case.shape(args.size)
    schema = load_model(str(case.model_path(shape)))
    program = lower_program(schema)
    sources = tidy_sources(schema, dict(case.data(shape)))

    real_bind = executor_module.bind
    cached: list[Any] = []

    def bind_once(program_: Any, sources_: Any) -> Any:
        if not cached:
            cached.append(real_bind(program_, sources_))
        return cached[0]

    def one() -> float:
        """One build, timed. Called once untimed to fill the bind cache."""
        engine = PolarsEngine()
        started = time.perf_counter()
        engine.build(program, sources)
        elapsed = time.perf_counter() - started
        engine.close()
        return elapsed

    full = [one() for _ in range(args.rounds)]

    executor_module.bind = bind_once
    one()
    spent.clear()
    assembly = [one() for _ in range(args.rounds)]
    executor_module.bind = real_bind

    print(f'\n{args.case}/{args.size}: {args.rounds} rounds, minimum reported\n')
    _line('build, binding included', full)
    _line('build, binding reused', assembly)
    print(f'  {"binding":28} {(min(full) - min(assembly)) * 1000:8.1f} ms')
    print()
    for phase, times in sorted(spent.items(), key=lambda kv: -sum(kv[1])):
        calls = len(times) // args.rounds
        print(f'  {phase:28} {sum(times) / args.rounds * 1000:8.1f} ms   over {calls} call(s)')
    return 0


def _timed(name: str, method: Any, spent: dict[str, list[float]]) -> Any:
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return method(self, *args, **kwargs)
        finally:
            spent[name].append(time.perf_counter() - started)

    return wrapped


def _line(label: str, times: list[float]) -> None:
    low = min(times)
    spread = (max(times) / low - 1) * 100
    print(f'  {label:28} {low * 1000:8.1f} ms   median {statistics.median(times) * 1000:7.1f}   spread {spread:4.1f}%')


if __name__ == '__main__':
    raise SystemExit(main())
