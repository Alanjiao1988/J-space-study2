"""Decision rule (§9.2) and Holm correction (§9.6).

Three outcomes, frozen at Freeze-1:

===================  =========================================================
Material difference  the **95%** CI of ``D`` lies entirely above ``+δ`` or
                     entirely below ``−δ``
Equivalent           the **90%** CI of ``D`` lies entirely inside ``[−δ, +δ]``
Inconclusive         everything else
===================  =========================================================

"Inconclusive" is a legitimate third outcome and may not be re-adjudicated (SC3): no extra
samples, no endpoint switch, no change to ``δ``.

Every entry point here is guarded by :func:`wda.governance.blind.guard_decision` — Phase A
has zero evidential weight and may not compute or inspect ``D`` or ``G`` (§10, §16).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from wda.governance import blind


def _finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")


class Outcome(str, Enum):
    MATERIAL = "material_difference"
    EQUIVALENT = "equivalent"
    INCONCLUSIVE = "inconclusive"

    @property
    def stopping_condition(self) -> Optional[str]:
        """§12 mapping from a Phase-C outcome to its stopping condition."""
        return {
            Outcome.MATERIAL: None,  # Phase D may start
            Outcome.EQUIVALENT: "SC2",
            Outcome.INCONCLUSIVE: "SC3",
        }[self]


@dataclass(frozen=True)
class Interval:
    """A two-sided confidence interval at a stated level."""

    low: float
    high: float
    level: float

    def __post_init__(self) -> None:
        for name in ("low", "high", "level"):
            _finite(getattr(self, name), name)
        if self.high < self.low:
            raise ValueError(f"interval bounds inverted: [{self.low}, {self.high}]")
        if not 0.0 < self.level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {self.level}")

    def contained_in(self, low: float, high: float) -> bool:
        return self.low >= low and self.high <= high

    def entirely_above(self, threshold: float) -> bool:
        return self.low > threshold

    def entirely_below(self, threshold: float) -> bool:
        return self.high < threshold

    def to_dict(self) -> Dict[str, float]:
        return {"low": self.low, "high": self.high, "level": self.level}


@dataclass(frozen=True)
class Decision:
    """The adjudicated result for one endpoint."""

    endpoint: str
    point_estimate: float
    delta: float
    ci95: Interval
    ci90: Interval
    outcome: Outcome
    rationale: str
    stopping_condition: Optional[str] = None
    direction: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "endpoint": self.endpoint,
            "point_estimate": self.point_estimate,
            "delta": self.delta,
            "ci95": self.ci95.to_dict(),
            "ci90": self.ci90.to_dict(),
            "outcome": self.outcome.value,
            "direction": self.direction,
            "rationale": self.rationale,
            "stopping_condition": self.stopping_condition,
        }

    def sentence(self) -> str:
        """The only permitted phrasing (§4)."""
        return (
            f"For {self.endpoint}, the difference is D = {self.point_estimate:+.4f} "
            f"(95% CI [{self.ci95.low:+.4f}, {self.ci95.high:+.4f}]; "
            f"90% CI [{self.ci90.low:+.4f}, {self.ci90.high:+.4f}]; delta = {self.delta:.2f}), "
            f"judged {self.outcome.value.replace('_', ' ')}."
        )


def decide(
    point_estimate: float,
    ci95: Interval,
    ci90: Interval,
    delta: float,
    *,
    endpoint: str = "D",
) -> Decision:
    """Apply the frozen §9.2 rule. Never call this during Phase A."""
    blind.guard_decision(f"wda.stats.decision.decide(endpoint={endpoint!r})")

    _finite(point_estimate, "point_estimate")
    _finite(delta, "delta")
    if delta <= 0:
        raise ValueError("delta must be positive; it is fixed a priori at Freeze-1 (§9.1)")
    if abs(ci95.level - 0.95) > 1e-9:
        raise ValueError(f"the material-difference test requires a 95% CI, got {ci95.level}")
    if abs(ci90.level - 0.90) > 1e-9:
        raise ValueError(f"the equivalence test requires a 90% CI, got {ci90.level}")
    if ci90.low < ci95.low - 1e-12 or ci90.high > ci95.high + 1e-12:
        raise ValueError(
            "the 90% CI is not nested inside the 95% CI; the two intervals were not "
            "produced by the same resampling distribution"
        )

    above = ci95.entirely_above(delta)
    below = ci95.entirely_below(-delta)
    equivalent = ci90.contained_in(-delta, delta)

    if (above or below) and equivalent:  # pragma: no cover - impossible for nested CIs
        raise AssertionError(
            "material and equivalent both fired; the intervals are mutually inconsistent"
        )

    if above or below:
        direction = "positive" if above else "negative"
        outcome = Outcome.MATERIAL
        rationale = (
            f"the 95% CI lies entirely {'above +' if above else 'below -'}{delta:.2f}"
        )
    elif equivalent:
        direction = None
        outcome = Outcome.EQUIVALENT
        rationale = f"the 90% CI lies entirely inside [-{delta:.2f}, +{delta:.2f}]"
    else:
        direction = None
        outcome = Outcome.INCONCLUSIVE
        rationale = (
            "neither the material-difference nor the equivalence criterion is met; "
            "SC3 applies — do not add samples, change the endpoint, or change delta"
        )

    return Decision(
        endpoint=endpoint,
        point_estimate=point_estimate,
        delta=delta,
        ci95=ci95,
        ci90=ci90,
        outcome=outcome,
        rationale=rationale,
        stopping_condition=outcome.stopping_condition,
        direction=direction,
    )


# --------------------------------------------------------------------------------------
# §9.6 multiplicity
# --------------------------------------------------------------------------------------


def holm(pvalues: Mapping[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, object]]:
    """Holm step-down correction over the *secondary confirmatory* family only (§8.5).

    Exploratory assays are neither corrected nor inferred from, so they must not be passed
    in here.
    """
    blind.guard_decision("wda.stats.decision.holm")
    _finite(alpha, "alpha")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not pvalues:
        return {}
    for name, p in pvalues.items():
        _finite(p, f"p-value for {name!r}")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-value for {name!r} out of range: {p}")

    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: Dict[str, Dict[str, object]] = {}
    running_max = 0.0
    still_rejecting = True
    for index, (name, p) in enumerate(ordered):
        threshold = alpha / (m - index)
        adjusted = min(1.0, max(running_max, (m - index) * p))
        running_max = adjusted
        if still_rejecting and p > threshold:
            still_rejecting = False
        out[name] = {
            "p_raw": p,
            "p_holm": adjusted,
            "threshold": threshold,
            "rank": index + 1,
            "reject": bool(still_rejecting),
        }
    return out


@dataclass
class DecisionReport:
    """Everything §17 item 3 requires, for one phase."""

    phase: str
    seal_root_sha256: str
    primary: Decision
    secondary: Dict[str, Decision] = field(default_factory=dict)
    secondary_holm: Dict[str, Dict[str, object]] = field(default_factory=dict)
    exploratory: Dict[str, Dict[str, object]] = field(default_factory=dict)
    feasibility_results: List[Dict[str, object]] = field(default_factory=list)

    def interpret_secondary(self) -> bool:
        """§8.5 — secondary results are interpreted only if the primary is material."""
        return self.primary.outcome is Outcome.MATERIAL

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": "wda/decision_report/1",
            "phase": self.phase,
            "seal_root_sha256": self.seal_root_sha256,
            "primary": self.primary.to_dict(),
            "primary_sentence": self.primary.sentence(),
            "secondary_interpretable": self.interpret_secondary(),
            "secondary": {k: v.to_dict() for k, v in sorted(self.secondary.items())},
            "secondary_holm": self.secondary_holm,
            "exploratory": self.exploratory,
            "exploratory_note": "reported, never inferred from; not multiplicity-corrected (§9.6)",
            "feasibility_results": self.feasibility_results,
            "stopping_condition": self.primary.stopping_condition,
        }


#: §4 — strings that must never appear in a report generated from a Decision.
FORBIDDEN_PHRASES: Tuple[str, ...] = (
    "reasoning-specific",
    "reasoning specific",
    "reasoning-distilled model class",
    "scale trend",
    "scaling trend",
    "distillation caused",
    "caused by distillation",
    "externalization confirmed",
    "externalization is confirmed",
    "j-space exists",
    "j-space does not exist",
    "first reproduction",
)


def check_phrasing(text: str) -> List[str]:
    """Return the forbidden phrases (§4) present in ``text``. Empty means clean."""
    lowered = text.lower()
    return [phrase for phrase in FORBIDDEN_PHRASES if phrase in lowered]


def assert_permitted_phrasing(text: str) -> None:
    hits = check_phrasing(text)
    if hits:
        raise ValueError(
            f"report text contains forbidden phrasing (§4): {hits}. "
            "Only the D = ... (CI ...), judged ... form is permitted."
        )


__all__ = [
    "Decision",
    "DecisionReport",
    "FORBIDDEN_PHRASES",
    "Interval",
    "Outcome",
    "assert_permitted_phrasing",
    "check_phrasing",
    "decide",
    "holm",
]
