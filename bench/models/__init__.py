"""One directory per case: the model in every dialect that can express it.

`bench/cases.py` holds what a case *is* — its ladder, its cardinalities, the
parquet its generator writes. Here is what it *says*, once per language that
says it: `model.yaml` for lpspec, and one module per hand-written dialect.

A case with no alternative formulation is just its `model.yaml`; the two that
have one carry a `FORMULATIONS` map naming what they hold. Written out rather
than discovered by scanning, so a misnamed module is an import error rather
than an arm that quietly measures nothing.
"""


def formulation(case_name: str, dialect: str):
    """The case's model in *dialect*, or None where nobody has written one.

    A case package names what it holds in `FORMULATIONS`; a case that holds
    nothing has no package at all, which is the same answer. What `build`
    takes is the *arm's* contract rather than a shared one — gurobipy hands its
    formulations an `Env`, because an environment is arm-level knowledge and a
    model file should not be creating one.
    """
    import importlib

    try:
        case = importlib.import_module(f'bench.models.{case_name}')
    except ModuleNotFoundError:
        return None
    return getattr(case, 'FORMULATIONS', {}).get(dialect)
