# About

Why the package is shaped the way it is, what it costs, where it came from and
where it is going. Nothing here is needed to write or run a model — that is
[the guide](../guide.md), [the API](../reference/api.md) and
[the language](https://math-spec.readthedocs.io/en/latest/reference/language/).

| | |
|---|---|
| [Architecture](architecture.md) | the thesis, the hard rules, the lane from YAML to solver, and the module map |
| [The ceiling](https://math-spec.readthedocs.io/en/latest/about/ceiling/) | what the language deliberately cannot say, and what that buys |
| [Decomposition](decomposition.md) | Benders as evidence: an algorithm the language does not contain, written as a loop over models it does |
| [Benchmarks](benchmarks.md) | measured build and solve cost against linopy, with the method and how to reproduce it |
| [Relationship to linopy](linopy.md) | not a runtime dependency, the differential oracle, and the second lane |
| [Prior art and credit](prior-art.md) | the work this is derived from, and how to cite it |
| [Roadmap](roadmap.md) | where it is going, and what it will not become |
| [Changelog](changelog.md) | every release |

Contributing — the test loop, the CI gates, how to add a solver or a port — is
[CONTRIBUTING.md](https://github.com/fluxopt/lpspec/blob/main/CONTRIBUTING.md)
and [AGENTS.md](https://github.com/fluxopt/lpspec/blob/main/AGENTS.md) in the
repository.
