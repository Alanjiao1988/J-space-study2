"""Assay stratification (§8.5).

Exactly one assay is gate-bearing. Everything else is either secondary confirmatory
(Holm-corrected, and interpreted **only** if the primary is material) or exploratory
(reported, never inferred from). Two assays are excluded outright because external
replication failed on them.

Registering an assay here fixes its tier at Freeze-1. :func:`require_tier` is the guard a
runner calls before writing a result, so a tier cannot drift at analysis time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

from wda.errors import ProtocolViolation
from wda.phases import AssayTier


class AssayViolation(ProtocolViolation):
    """§8.5 — an assay was used at a tier other than the one registered."""


@dataclass(frozen=True)
class AssaySpec:
    name: str
    tier: AssayTier
    estimand: str
    rationale: str
    required_controls: Tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "tier": self.tier.value,
            "estimand": self.estimand,
            "rationale": self.rationale,
            "required_controls": list(self.required_controls),
            "note": self.note,
        }


REGISTRY: Mapping[str, AssaySpec] = {
    # ---------------------------------------------------------------- primary (gate) ---
    "primary_interaction": AssaySpec(
        name="primary_interaction",
        tier=AssayTier.PRIMARY,
        estimand="D = G_primary(treatment) - G_primary(comparator) on the generated item bank",
        rationale="§8.5 — the only gate-bearing assay.",
        required_controls=(
            "layer_matched_random",
            "matched_norm_perturbation",
            "clamp_complement",
            "general_damage",
            "skip_clean_top10",
            "logit_lens_floor",
            "shuffled_corpus_lens",
        ),
    ),
    # ------------------------------------------------- secondary confirmatory (Holm) ---
    "gsm8k_interaction": AssaySpec(
        name="gsm8k_interaction",
        tier=AssayTier.SECONDARY_CONFIRMATORY,
        estimand="D on the GSM8K held-out split",
        rationale="§8.5 — comparability with the paper's GSM8K result.",
        required_controls=("layer_matched_random", "general_damage"),
        note="Contamination risk is real; this is an anchor, not evidence about the item bank.",
    ),
    "flexible_generalization": AssaySpec(
        name="flexible_generalization",
        tier=AssayTier.SECONDARY_CONFIRMATORY,
        estimand="multi-fact edit success rate under J-space vs control interventions",
        rationale="§2 — the one paper claim that replicated cleanly in external review.",
        required_controls=("layer_matched_random", "matched_norm_perturbation"),
    ),
    # ------------------------------------------------------------------ exploratory ---
    "g_gen": AssaySpec(
        name="g_gen",
        tier=AssayTier.EXPLORATORY,
        estimand="G_gen = Delta(C_direct) - Delta(C_gen)",
        rationale="§5 — confounded by generation length, exposure count and self-correction.",
    ),
    "c_direct_prefill": AssaySpec(
        name="c_direct_prefill",
        tier=AssayTier.EXPLORATORY,
        estimand="G_primary under the <think></think> prefill variant",
        rationale="§5 — a compliance probe for fact F4, not a condition of the design.",
    ),
    "probe_swap": AssaySpec(
        name="probe_swap",
        tier=AssayTier.EXPLORATORY,
        estimand="target-answer logit change under a lens-coordinate swap",
        rationale="§8.5 — reportable only with the direct-substitution baseline attached.",
        required_controls=("direct_substitution_baseline", "matched_norm_perturbation"),
    ),
    "verbal_report": AssaySpec(
        name="verbal_report",
        tier=AssayTier.EXPLORATORY,
        estimand="verbal-report swap top-5 hit rate",
        rationale="§2 — no strength sweep exists for verbal report; the sweep belongs to "
        "verbal introspection.",
    ),
    "directed_modulation": AssaySpec(
        name="directed_modulation",
        tier=AssayTier.EXPLORATORY,
        estimand="hit rate contrasted across group_kind",
        rationale="§8.5 exploratory.",
        required_controls=("no_instruction_baseline",),
    ),
    "selectivity_language": AssaySpec(
        name="selectivity_language",
        tier=AssayTier.EXPLORATORY,
        estimand="explicit - automatic label hit rate",
        rationale="§8.5 — n = 8 is too small to support inference.",
    ),
    "selectivity_linecount": AssaySpec(
        name="selectivity_linecount",
        tier=AssayTier.EXPLORATORY,
        estimand="condition contrast over the eleven passages",
        rationale="§8.5 — n = 11 is too small to support inference.",
    ),
    "capacity": AssaySpec(
        name="capacity",
        tier=AssayTier.EXPLORATORY,
        estimand="band-min lens rank <= k at list positions, related vs unrelated",
        rationale="§2 — the core contrast is related vs unrelated; the four-block variant "
        "is a sub-variant, not the headline.",
        required_controls=("control_words",),
    ),
    "ignition": AssaySpec(
        name="ignition",
        tier=AssayTier.EXPLORATORY,
        estimand="alpha-sweep of the J-restricted component vs full activation",
        rationale="§2 — the contrast is J-restricted component vs full activation.",
    ),
    "dual_task": AssaySpec(
        name="dual_task",
        tier=AssayTier.EXPLORATORY,
        estimand="interference = single-task - dual-task reachability",
        rationale="§8.5 exploratory.",
    ),
    # -------------------------------------------------------------------- excluded ---
    "rhyme_planning": AssaySpec(
        name="rhyme_planning",
        tier=AssayTier.EXCLUDED,
        estimand="(not run)",
        rationale="§8.5 — not reproduced in external review (Neel Nanda).",
    ),
    "mental_arithmetic": AssaySpec(
        name="mental_arithmetic",
        tier=AssayTier.EXCLUDED,
        estimand="(not run)",
        rationale="§8.5 — not reproduced in external review (Neel Nanda).",
    ),
}


def spec(name: str) -> AssaySpec:
    if name not in REGISTRY:
        raise AssayViolation(
            f"{name!r} is not a registered assay. Registered: {sorted(REGISTRY)}."
        )
    return REGISTRY[name]


def require_tier(name: str, tier: AssayTier) -> AssaySpec:
    """Guard: raise unless ``name`` is registered at ``tier``."""
    found = spec(name)
    if found.tier is AssayTier.EXCLUDED:
        raise AssayViolation(
            f"{name!r} is excluded from this study: {found.rationale}. It may not be run."
        )
    if found.tier is not tier:
        raise AssayViolation(
            f"{name!r} is registered as {found.tier.value}, not {tier.value}. Tiers are "
            "fixed at Freeze-1 and may not be changed at analysis time (§8.5)."
        )
    return found


def by_tier(tier: AssayTier) -> Tuple[str, ...]:
    return tuple(sorted(name for name, s in REGISTRY.items() if s.tier is tier))


def snapshot() -> Dict[str, object]:
    return {
        "schema": "wda/assay_registry/1",
        "primary": list(by_tier(AssayTier.PRIMARY)),
        "secondary_confirmatory": list(by_tier(AssayTier.SECONDARY_CONFIRMATORY)),
        "exploratory": list(by_tier(AssayTier.EXPLORATORY)),
        "excluded": list(by_tier(AssayTier.EXCLUDED)),
        "specs": {name: s.to_dict() for name, s in sorted(REGISTRY.items())},
    }


__all__ = ["REGISTRY", "AssaySpec", "AssayViolation", "by_tier", "require_tier", "snapshot", "spec"]
