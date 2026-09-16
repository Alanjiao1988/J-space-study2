"""Primary assay: the ablation x response-format interaction (§5, §8.5, §9).

This is the **only** gate-bearing analysis. It consumes a ``TrialFrame`` built from
``log.jsonl`` and produces the ``D`` decision under the frozen §9.2 rule.

Design cells per model::

                       clean            ablated
    C_direct           acc_dc           acc_da
    C_frozen           acc_fc           acc_fa

    G_primary = (acc_dc - acc_da) - (acc_fc - acc_fa)
    D         = G_primary(treatment) - G_primary(comparator)

Every accuracy is the equal-fit ITT composite (§9.4). The bootstrap draws two
fits independently within each model, then one shared item sample across all
models/fits. Only one template ID per item is currently supported (§9.5).

During Phase A, :func:`analyze` is guarded before any statistics are computed,
not merely when the final decision is made (§16).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

from wda.assays.registry import require_tier
from wda.conditions.envelopes import C_DIRECT, C_FROZEN
from wda.errors import ProtocolViolation
from wda.governance import blind
from wda.intervene.controls import assert_controls_complete
from wda.phases import AssayTier
from wda.scoring.parse import TripleEndpoint
from wda.stats import bootstrap as bs
from wda.stats.decision import Decision, decide

ASSAY_NAME = "primary_interaction"


class PrimaryAssayError(ProtocolViolation):
    """The primary assay was run on an incomplete or mis-shaped design."""


@dataclass
class CellSummary:
    """Per (model, condition, ablation state) triple endpoint, for §17 item 4."""

    model_role: str
    condition: str
    ablation_state: str
    endpoint: TripleEndpoint

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_role": self.model_role,
            "condition": self.condition,
            "ablation_state": self.ablation_state,
            **self.endpoint.to_dict(),
        }


@dataclass
class PrimaryResult:
    d_estimate: float
    decision: Decision
    g_primary: Dict[str, float]
    deltas: Dict[str, Dict[str, float]]
    bootstrap: Dict[str, object]
    cells: List[CellSummary] = field(default_factory=list)
    feasibility_results: List[Dict[str, object]] = field(default_factory=list)
    controls_reported: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": "wda/primary_result/1",
            "assay": ASSAY_NAME,
            "tier": AssayTier.PRIMARY.value,
            "D": self.d_estimate,
            "G_primary": self.g_primary,
            "deltas": self.deltas,
            "bootstrap": self.bootstrap,
            "decision": self.decision.to_dict(),
            "sentence": self.decision.sentence(),
            "cells": [c.to_dict() for c in self.cells],
            "feasibility_results": self.feasibility_results,
            "controls_reported": sorted(self.controls_reported),
        }


def assert_design_complete(
    frame: bs.TrialFrame, *, treatment: str = "treatment", comparator: str = "comparator"
) -> None:
    """Require every item in all four cells of each of the two real fits per model."""
    try:
        bs.validate_primary_design(frame, treatment=treatment, comparator=comparator)
    except (bs.BootstrapViolation, ValueError) as exc:
        raise PrimaryAssayError(str(exc)) from exc


def analyze(
    frame: bs.TrialFrame,
    *,
    delta: float,
    n_resamples: int = 10_000,
    seed: int = 0,
    treatment: str = "treatment",
    comparator: str = "comparator",
    controls_reported: Sequence[str] = (),
    feasibility_results: Sequence[Mapping[str, object]] = (),
    cells: Sequence[CellSummary] = (),
) -> PrimaryResult:
    """Run the primary analysis and adjudicate ``D`` under §9.2."""
    blind.guard_decision("wda.assays.primary_interaction.analyze")
    require_tier(ASSAY_NAME, AssayTier.PRIMARY)
    assert_design_complete(frame, treatment=treatment, comparator=comparator)
    assert_controls_complete(controls_reported)

    def statistic(f: bs.TrialFrame) -> float:
        return bs.primary_endpoint(f, treatment=treatment, comparator=comparator)

    result = bs.hierarchical_bootstrap(
        frame, statistic, n_resamples=n_resamples, seed=seed
    )
    decision = decide(
        result.estimate,
        result.interval(0.95),
        result.interval(0.90),
        delta,
        endpoint="D (J-space ablation x response-format interaction, 14B pair)",
    )
    return PrimaryResult(
        d_estimate=result.estimate,
        decision=decision,
        g_primary={
            treatment: bs.g_primary(frame, treatment),
            comparator: bs.g_primary(frame, comparator),
        },
        deltas={
            role: {
                C_DIRECT: bs.delta(frame, role, C_DIRECT),
                C_FROZEN: bs.delta(frame, role, C_FROZEN),
            }
            for role in (treatment, comparator)
        },
        bootstrap=result.to_dict(),
        cells=list(cells),
        feasibility_results=[dict(f) for f in feasibility_results],
        controls_reported=list(controls_reported),
    )


__all__ = ["ASSAY_NAME", "CellSummary", "PrimaryAssayError", "PrimaryResult", "analyze", "assert_design_complete"]
