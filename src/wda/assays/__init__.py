"""Assays (§8.5). One module per assay, plus a one-page estimand spec beside it.

The tier of every assay is fixed at Freeze-1 in :mod:`wda.assays.registry`; each analysis
entry point calls ``require_tier`` before producing a result.
"""

from wda.assays import flexible_generalization, primary_interaction, probe_swap, registry

__all__ = ["flexible_generalization", "primary_interaction", "probe_swap", "registry"]
