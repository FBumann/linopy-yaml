# Prior art and credit

**[Calliope](https://github.com/calliope-project/calliope) (Apache-2.0) is
where this surface comes from.** Declaring solver-ready math as a reviewable
YAML file, a block per component, `foreach:` for the dim list, a `where:`
string over `AND`/`OR`/`NOT`, `bounds:`, `active:` — their design, down to
parsing the strings with pyparsing rather than `eval`. Our `expressions` is
their `global_expressions`, our `piecewise` their `piecewise_constraints`. What
is ours is the semantics underneath: one `expression` per block, macros that
take arguments, a schema closed at every level, and the absence and degree-1
laws ([SPEC §0](../SPEC.md#0-the-laws)). Their math is also the corpus we score
coverage against, which is a different thing from a specification we match
([SPEC §11](../SPEC.md#11-out-of-scope)).

**[linopy](https://github.com/PyPSA/linopy) (MIT) is the vocabulary, the oracle
and the denominator.** Where a concept is already theirs we copy the spelling
rather than invent a second one; every language feature is differentially
tested against a linopy build; every ratio on the [benchmarks](../benchmarks.md)
page is charter ÷ linopy. The three relationships are
[one page](linopy.md). The ported models in [the gallery](../models/index.md)
and their reference optima are **PyPSA**'s.

No code from any of them is vendored, so none of this is a licence obligation —
it is stated because a debt only the author knows about is one the next reader
has to rediscover. Neither project has reviewed this one; mistakes in the
comparisons above are ours.

If you cite this project, cite the two it is built on:

```bibtex
@article{Pfenninger2018,
  doi = {10.21105/joss.00825}, year = {2018}, publisher = {The Open Journal},
  volume = {3}, number = {29}, pages = {825},
  author = {Stefan Pfenninger and Bryn Pickering},
  title = {Calliope: a multi-scale energy systems modelling framework},
  journal = {Journal of Open Source Software}
}

@article{Hofmann2023,
  doi = {10.21105/joss.04823}, year = {2023}, publisher = {The Open Journal},
  volume = {8}, number = {84}, pages = {4823},
  author = {Fabian Hofmann},
  title = {Linopy: Linear optimization with n-dimensional labeled variables},
  journal = {Journal of Open Source Software}
}
```
