"""Secondary confirmatory assay: flexible generalization (multi-fact editing).

This is the one paper claim that reproduced cleanly in external review (Neel Nanda's
review found rhyme-planning and mental-arithmetic did not reproduce, probe-swap was weak,
false positives were common — but multi-fact editing / flexible generalization replicated).
It is therefore registered as **secondary confirmatory** and interpreted only if the
primary endpoint is material, under Holm correction (§8.5, §9.6).

Paper reference values (Appendix B): 76/192 at ``alpha = 1``, 101/192 at ``alpha = 2``.
Those are calibration references, not targets: matching them is not a success criterion and
missing them is not a failure of this study.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from wda.assays.registry import require_tier
from wda.errors import ProtocolViolation
from wda.phases import AssayTier

ASSAY_NAME = "flexible_generalization"

#: Appendix B reference values. Descriptive only.
PAPER_REFERENCE = {"alpha_1": (76, 192), "alpha_2": (101, 192)}


class FlexibleGeneralizationError(ProtocolViolation):
    """The assay was run without its mandatory arms."""


@dataclass
class EditArm:
    """One intervention arm over the same edit set."""

    name: str
    successes: int
    n: int

    @property
    def rate(self) -> float:
        return self.successes / self.n if self.n else float("nan")

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "successes": self.successes, "n": self.n, "rate": self.rate}


@dataclass
class FlexibleGeneralizationResult:
    arms: Dict[str, EditArm]
    alpha: float
    contrast: float
    contrast_name: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": "wda/flexible_generalization/1",
            "assay": ASSAY_NAME,
            "tier": AssayTier.SECONDARY_CONFIRMATORY.value,
            "alpha": self.alpha,
            "arms": {name: arm.to_dict() for name, arm in sorted(self.arms.items())},
            "contrast": self.contrast,
            "contrast_name": self.contrast_name,
            "paper_reference": {k: list(v) for k, v in PAPER_REFERENCE.items()},
            "paper_reference_note": (
                "Appendix B values are calibration references, not targets. Matching them "
                "is not a success criterion; missing them is not a failure."
            ),
        }


REQUIRED_ARMS = ("jspace_edit", "layer_matched_random", "matched_norm_perturbation")


def analyze(arms: Mapping[str, EditArm], *, alpha: float) -> FlexibleGeneralizationResult:
    """Contrast the J-space edit arm against its controls over the same edit set."""
    require_tier(ASSAY_NAME, AssayTier.SECONDARY_CONFIRMATORY)
    missing = sorted(set(REQUIRED_ARMS) - set(arms))
    if missing:
        raise FlexibleGeneralizationError(
            f"missing mandatory arms {missing}; §7 requires the random-direction and "
            "matched-norm controls alongside every intervention."
        )
    sizes = {arm.n for arm in arms.values()}
    if len(sizes) != 1:
        raise FlexibleGeneralizationError(
            f"the arms use different edit-set sizes {sorted(sizes)}; the contrast is only "
            "defined over a common set."
        )
    contrast = arms["jspace_edit"].rate - arms["layer_matched_random"].rate
    return FlexibleGeneralizationResult(
        arms=dict(arms),
        alpha=alpha,
        contrast=contrast,
        contrast_name="jspace_edit - layer_matched_random",
    )


__all__ = [
    "ASSAY_NAME",
    "EditArm",
    "FlexibleGeneralizationError",
    "FlexibleGeneralizationResult",
    "PAPER_REFERENCE",
    "REQUIRED_ARMS",
    "analyze",
]
