"""Exploratory assay: probe-swap (§8.5).

Registered as **exploratory** and reportable only with the direct final-token substitution
baseline attached (tao-hpu). Rationale: external review found probe-swap weak, and the
direct-substitution baseline is the key alternative explanation — a swap effect that the
baseline reproduces is not evidence about a workspace.

Paper reference values (Appendix B, n = 90): J-space component 61%, raw swap 60%,
non-J 28%, non-J with clamp 6%. Descriptive only.

Upstream ``data/experiments/probe-swap.json`` supplies the item set. Note (verified against
the predecessor's vendored copy) that this file lives under upstream ``data/experiments/``
and is a *separate causal-swap benchmark* of 90 items — not part of
``data/evaluations/``. Fact F1's "5/90" figure refers to this set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from wda.assays.registry import require_tier
from wda.errors import ProtocolViolation
from wda.intervene.patch import SwapOutcome, require_direct_substitution_baseline, swap_depth_separation
from wda.phases import AssayTier

ASSAY_NAME = "probe_swap"

#: Appendix B, n = 90. Descriptive references, never targets.
PAPER_REFERENCE = {
    "jspace_component": 0.61,
    "raw_swap": 0.60,
    "non_j": 0.28,
    "non_j_clamped": 0.06,
    "n": 90,
}


class ProbeSwapError(ProtocolViolation):
    """The assay was run without a mandatory paired control."""


@dataclass
class ProbeSwapResult:
    n: int
    mean_target_delta: float
    mean_random_delta: float
    mean_direct_substitution_delta: float
    positive_control_rate: float
    depth_separation: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": "wda/probe_swap/1",
            "assay": ASSAY_NAME,
            "tier": AssayTier.EXPLORATORY.value,
            "n": self.n,
            "mean_target_delta": self.mean_target_delta,
            "mean_matched_norm_random_delta": self.mean_random_delta,
            "mean_direct_substitution_delta": self.mean_direct_substitution_delta,
            "positive_control_rate": self.positive_control_rate,
            "depth_separation": self.depth_separation,
            "paper_reference": dict(PAPER_REFERENCE),
            "interpretation_limit": (
                "exploratory: reported, never inferred from, and not multiplicity-corrected "
                "(§8.5, §9.6). A swap effect the direct-substitution baseline reproduces is "
                "not evidence about a workspace."
            ),
        }


def analyze(
    outcomes: Sequence[SwapOutcome],
    *,
    intermediate_depths: Optional[Sequence[float]] = None,
    answer_depths: Optional[Sequence[float]] = None,
) -> ProbeSwapResult:
    require_tier(ASSAY_NAME, AssayTier.EXPLORATORY)
    if not outcomes:
        raise ProbeSwapError("no swap outcomes supplied")
    for outcome in outcomes:
        require_direct_substitution_baseline(outcome)

    target = np.array([o.target_logit_delta for o in outcomes], dtype=float)
    random = np.array([o.matched_norm_random_delta for o in outcomes], dtype=float)
    direct = np.array([float(o.direct_substitution_delta) for o in outcomes], dtype=float)
    separation = (
        swap_depth_separation(intermediate_depths, answer_depths)
        if intermediate_depths is not None and answer_depths is not None
        else None
    )
    return ProbeSwapResult(
        n=len(outcomes),
        mean_target_delta=float(target.mean()),
        mean_random_delta=float(random.mean()),
        mean_direct_substitution_delta=float(direct.mean()),
        positive_control_rate=float(np.mean([o.positive_control_holds() for o in outcomes])),
        depth_separation=separation,
    )


__all__ = ["ASSAY_NAME", "PAPER_REFERENCE", "ProbeSwapError", "ProbeSwapResult", "analyze"]
