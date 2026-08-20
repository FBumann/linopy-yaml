"""Expansion means the same on both lanes.

The rules for `macros:` and `expressions:` are the language's and live with
it (`tests/language/test_expansion.py`); *that the two lanes agree about what
they expanded to* is a claim about two consumers, so it is asserted here.
One end-to-end case carries it: both constructs expand to core AST before
dispatch, so if the lanes agree here they agree at all.
"""

from __future__ import annotations

import numpy as np

from tests.differential import differential

#: The same model `tests/language/test_expansion.py` expands, copied rather
#: than imported: that module travels with the language and this one does not,
#: so an import here would be a reference across the cut.
EXPANSION_YAML = """
dimensions:
  snapshot: {dtype: int}
  generator: {values: [wind, solar, gas]}
parameters:
  p_max: {dims: [generator]}
  cost: {dims: [generator]}
  load: {dims: [snapshot]}
expressions:
  total_generation: sum(p, over=generator)
macros:
  weighted_sum:
    args: [array, weights]
    kwargs: [over]
    template: sum(array * weights, over=over)
variables:
  p:
    foreach: [snapshot, generator]
    where: "p_max > 0"
    bounds: {lower: 0, upper: p_max}
constraints:
  balance:
    foreach: [snapshot]
    expression: total_generation == load
objective:
  sense: minimize
  expression: sum(weighted_sum(p, cost, over=generator))
"""


def test_a_macro_and_a_named_expression_mean_the_same_on_both_lanes():
    """One end-to-end test carries the whole feature: both constructs expand to
    core AST before dispatch, so if the lanes agree here they agree at all.

    `tests.oracle` is imported in the body so the [linopy] extra gates this one
    test rather than the module.
    """
    from tests.oracle import pd

    rng = np.random.default_rng(5)
    n_s = 24
    data = {
        'p_max': pd.Series({'wind': 100.0, 'solar': 60.0, 'gas': 200.0}),
        'cost': pd.Series({'wind': 1.0, 'solar': 2.0, 'gas': 50.0}),
        'load': pd.Series(
            (rng.uniform(0.2, 0.8, n_s) * 360.0).round(3),
            index=pd.RangeIndex(n_s, name='snapshot'),
        ),
    }
    index = {'snapshot': pd.RangeIndex(n_s, name='snapshot')}

    with differential(EXPANSION_YAML, data | index):
        pass  # agreement on the objective is the whole assertion
