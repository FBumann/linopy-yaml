---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# lpspec

**Self-documenting optimisation models — at any scale.**

Write the math in YAML, bind data at runtime, solve.

[![PyPI](https://img.shields.io/pypi/v/lpspec)](https://pypi.org/project/lpspec/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[Run a model](guide.md){ .md-button .md-button--primary }
[Browse the examples](examples/index.md){ .md-button }

</div>

---

<div class="landing" markdown>

<div class="grid cards" markdown>

-   :material-file-document-outline: __Declarative math__

    ---

    Readable without knowing the implementation, and self-contained: no Python
    state changes what a file means. It diffs cleanly in review and travels as a
    research artefact.

-   :material-grid: __Sparse by construction__

    ---

    A mask is an absent row, never a NaN in a dense array — a model pays for the
    variables it has, not for its coordinate product. Labels *are* the solver's
    own row and column indices.

-   :material-alert-octagon-outline: __Fail early, fail loud__

    ---

    Every expression, `where` string and even *uncalled* macro template is
    parsed and name-checked before a single source is bound. Errors name the
    problem and its rewrite.

-   :material-fence: __A finite language, with a priced way out__

    ---

    The ceiling is a closure (relational ∩ local), not a feature race.
    Genuinely unsayable math goes in an `escape:` island — visible in the file,
    billed before it runs.

-   :material-speedometer: __Straight to the solver__

    ---

    YAML and data in, a populated solver out, no LP file in between: 2–4x faster
    than linopy on four of five benchmark cases, lower peak memory on
    all five. [The numbers](about/benchmarks.md)

-   :material-check-decagram-outline: __Checked against somebody else__

    ---

    Every ported model matches an optimum this project did not compute — GAMS,
    PyPSA, OSeMOSYS, OR-Library, TSPLIB — objectives *and*, where the reference
    records them, shadow prices. [The corpus](examples/index.md)

</div>

{%
   include-markdown "../README.md"
   start="<!--flow-start-->"
   end="<!--flow-end-->"
%}

## The whole thing, in one model

{%
   include-markdown "../README.md"
   start="<!--model-start-->"
   end="<!--model-end-->"
%}

### And that file says, exactly this

Generated from the YAML above — no data, no solver, no second source of truth.
Only the notation is a choice, and **How** shows the one that was made here.

<!-- home-math:begin -->
=== "The math"

    Least-cost dispatch of a generator fleet against an hourly load.

    #### Sets

    | Symbol | Meaning |
    |---|---|
    | $\mathcal{S}$ | index $s$ --- `snapshot` --- dispatch periods |
    | $\mathcal{G}$ | index $g$ --- `generator` --- generating units |

    #### Parameters

    | Symbol | Meaning |
    |---|---|
    | $\bar p$ | `p_max` over $\mathcal{G}$ --- installed capacity |
    | $\ell$ | `load` over $\mathcal{S}$ --- demand to be met |
    | $c$ | `cost` over $\mathcal{G}$ --- marginal cost |

    #### Variables

    | Symbol | Meaning |
    |---|---|
    | $p$ | `p` over $\mathcal{S} \times \mathcal{G}$ --- output of a generator in a snapshot |

    #### Objective

    $$\min \sum_{s \in \mathcal{S},\enspace g \in \mathcal{G}} p_{s,g} \cdot c_{g}$$

    #### Subject to

    **`power_balance`**

    $$\sum_{g \in \mathcal{G}} p_{s,g} = \ell_{s} \qquad \forall\thinspace s \in \mathcal{S}$$

    #### Variable domains

    **`p`**

    $$0 \le p_{s,g} \le \bar p_{g} \qquad \forall\thinspace s \in \mathcal{S},\enspace g \in \mathcal{G} \thinspace:\thinspace \bar p_{g} > 0$$

=== "LaTeX"

    ```latex
    \noindent Least-cost dispatch of a generator fleet against an hourly load.

    \paragraph{Sets}
    \begin{description}
    \item[$\mathcal{S}$] index $s$ --- \texttt{snapshot} --- dispatch periods
    \item[$\mathcal{G}$] index $g$ --- \texttt{generator} --- generating units
    \end{description}

    \paragraph{Parameters}
    \begin{description}
    \item[$\bar p$] \texttt{p\_max} over $\mathcal{G}$ --- installed capacity
    \item[$\ell$] \texttt{load} over $\mathcal{S}$ --- demand to be met
    \item[$c$] \texttt{cost} over $\mathcal{G}$ --- marginal cost
    \end{description}

    \paragraph{Variables}
    \begin{description}
    \item[$p$] \texttt{p} over $\mathcal{S} \times \mathcal{G}$ --- output of a generator in a snapshot
    \end{description}

    \paragraph{Objective}
    \begin{align}
     && \min & \sum_{s \in \mathcal{S},\ g \in \mathcal{G}} p_{s,g} \cdot c_{g}
    \end{align}

    \paragraph{Subject to}
    \begin{align}
    \text{power\_balance} && \sum_{g \in \mathcal{G}} p_{s,g} & = \ell_{s} && \forall\, s \in \mathcal{S}
    \end{align}

    \paragraph{Variable domains}
    \begin{align}
    \text{p} && 0 \le p_{s,g} & \le \bar p_{g} && \forall\, s \in \mathcal{S},\ g \in \mathcal{G} \,:\, \bar p_{g} > 0
    \end{align}
    ```

=== "How"

    ```python
    import math_spec as ms

    symbols = {
        'notation': 'latex',
        'dimensions': {
            'snapshot': {'index': 's', 'set': '\\mathcal{S}'},
            'generator': {'index': 'g', 'set': '\\mathcal{G}'},
        },
        'names': {
            'cost': 'c',
            'load': '\\ell',
            'p_max': '\\bar p',
        },
    }

    ms.to_latex('dispatch.yaml', symbols=symbols)  # amsmath align
    ms.to_typst('dispatch.yaml')  # compiles without a TeX toolchain
    ms.to_markdown('dispatch.yaml')  # renders as-is on GitHub
    ```

    `symbols` is optional — drop it and the same model prints as
    $\mathit{load}_t$, $p^{\mathrm{max}}_g$. A dict, a YAML path or a
    `SymbolTable`; a key naming nothing in the model is an error, not a symbol that
    silently never applies. Every spelling is printed verbatim — `notation` says
    which language they are, and a render in the other one refuses.

    Or from a shell, where the table is that same YAML on disk and `--standalone`
    emits a document that compiles rather than a fragment to `\input`:

    ```bash
    python -m math_spec latex dispatch.yaml --symbols dispatch.symbols.yaml
    python -m math_spec typst dispatch.yaml --standalone -o dispatch.typ
    ```

    The renderer is [math-spec](https://math-spec.readthedocs.io/en/latest/reference/typeset/)'s,
    and reads the same file this page solves.
<!-- home-math:end -->

### Then you solve it

{%
   include-markdown "../README.md"
   start="<!--solve-start-->"
   end="<!--quickstart-end-->"
%}

## Where to next

<div class="grid cards" markdown>

-   :material-school: __Run a model__

    ---

    A file and your tables to an answer you can read back — check, build,
    solve, and what the engine will *not* do.

    [:octicons-arrow-right-24: The guide](guide.md)

-   :material-table-arrow-right: __Your data__

    ---

    The journey from the files an instance arrives in to one frame per
    parameter — and what binding accepts, refuses, and says when it refuses.

    [:octicons-arrow-right-24: Preparing the data](examples/data.md) ·
    [The contract](reference/data.md)

-   :material-view-gallery-outline: __Models__

    ---

    Every model in the repo, what each exercises, and which ones are checked
    against an optimum from elsewhere.

    [:octicons-arrow-right-24: The gallery](examples/index.md)

-   :material-book-open-page-variant: __Language reference__

    ---

    What a YAML file may contain, and what it means — ten rules, ten
    declaration keys, one closed set of operators.

    [:octicons-arrow-right-24: The language](https://math-spec.readthedocs.io/en/latest/reference/language/)

-   :material-code-braces: __Python API__

    ---

    Bind data, build, solve, read the answer back — and sweep the same model
    over scenarios or a rolling horizon.

    [:octicons-arrow-right-24: The API](reference/api.md) ·
    [Sweeps](reference/sweeps.md) ·
    [Typeset](https://math-spec.readthedocs.io/en/latest/reference/typeset/)

-   :material-source-branch: __Why it is shaped this way__

    ---

    The hard rules, the expressive ceiling, the measured cost, the module map —
    and what we have decided never to build.

    [:octicons-arrow-right-24: About](about/index.md)

</div>

```bash
pip install lpspec  # the relational engine (polars, highspy)
pip install "lpspec[linopy]"  # adds linopy + xarray + pandas: the lane, the
                              # oracle, and to_pandas / to_dataarray
pip install "lpspec[gurobi]"  # adds the gurobi sink: solver_name='gurobi'
pip install "lpspec[xpress]"  # adds the xpress sink: solver_name='xpress'
```

!!! warning "Alpha, pre-1.0"

    {%
       include-markdown "../README.md"
       start="<!--status-start-->"
       end="<!--status-end-->"
    %}

</div>
