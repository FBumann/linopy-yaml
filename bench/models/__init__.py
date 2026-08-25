"""One directory per case: the model in every dialect that can express it.

`bench/cases.py` holds what a case *is* — its ladder, its cardinalities, the
parquet its generator writes. Here is what it *says*, once per language that
says it: `model.yaml` for lpspec, and one module per hand-written dialect.

A case with no alternative formulation is just its `model.yaml`; the two that
have one carry a `FORMULATIONS` map naming what they hold. Written out rather
than discovered by scanning, so a misnamed module is an import error rather
than an arm that quietly measures nothing.
"""
