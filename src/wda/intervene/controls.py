"""Controls (§7) — all mandatory, none optional.

From the paper
--------------
================================  ===================================================
``matched_norm_perturbation``     rules out any equal-norm arbitrary perturbation
``layer_matched_random``          k random directions in the same layer band; the
                                  ablation's negative control
``clamp_complement``              clamp the complementary component to its clean value,
                                  to test whether the residual effect is J-mediated
``position_control``              selectivity of the injection
``shuffled_position_null``        null for the autocorrelation statistic
``control_words``                 capacity zero point (words absent from the prompt)
``no_instruction_baseline``       directed-modulation zero point
================================  ===================================================

Added by the community / this protocol
--------------------------------------
================================  ===================================================
``shuffled_corpus_lens``          the fit corpus must justify itself (§6.1)
``direct_substitution_baseline``  tao-hpu; the key alternative explanation for probe-swap
``logit_lens_floor``              the paper's own statement makes this the floor
``general_damage``                if the J-space ablation's output KL is not separable
                                  from the random-direction ablation's, the strength tier
                                  is void
================================  ===================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from wda.errors import ProtocolViolation
from wda.intervene.ablate import project_out, removed_norm


class ControlViolation(ProtocolViolation):
    """§7 — a mandatory control was skipped, or a control failed its own contract."""


# --------------------------------------------------------------------------------------
# Random-direction controls
# --------------------------------------------------------------------------------------


def random_orthonormal_directions(d_model: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """``k`` random orthonormal directions in ``R^d`` — the layer-matched random control."""
    if k == 0:
        return np.zeros((0, d_model), dtype=np.float64)
    if k > d_model:
        raise ControlViolation(f"cannot draw {k} orthonormal directions in {d_model} dimensions")
    q, _ = np.linalg.qr(rng.standard_normal((d_model, k)))
    return q.T


def layer_matched_random_ablation(
    hidden: np.ndarray, k: int, *, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Ablate ``k`` random directions at the same layer. Returns ``(hidden', directions)``."""
    rng = np.random.default_rng(seed)
    dirs = random_orthonormal_directions(len(hidden), k, rng)
    return project_out(hidden, dirs), dirs


def matched_norm_perturbation(
    hidden: np.ndarray, target_norm: float, *, seed: int
) -> np.ndarray:
    """Perturb ``hidden`` by a random vector of exactly ``target_norm`` (§7).

    This is the "equal-norm arbitrary perturbation" control: it removes the same amount of
    vector magnitude as the ablation did, in a direction with no J-space meaning.
    """
    rng = np.random.default_rng(seed)
    h = np.asarray(hidden, dtype=np.float64)
    noise = rng.standard_normal(h.shape)
    norm = np.linalg.norm(noise)
    if norm == 0:  # pragma: no cover - measure-zero
        raise ControlViolation("degenerate random perturbation")
    return h + noise * (float(target_norm) / norm)


def assert_norm_matched(
    ablated_from: np.ndarray,
    ablated_to: np.ndarray,
    control_from: np.ndarray,
    control_to: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> float:
    """Phase-0 fixture ``random_control_norm_matched``: the two removals match in norm."""
    a = float(np.linalg.norm(np.asarray(ablated_from, float) - np.asarray(ablated_to, float)))
    b = float(np.linalg.norm(np.asarray(control_from, float) - np.asarray(control_to, float)))
    gap = abs(a - b)
    if gap > tolerance:
        raise ControlViolation(
            f"norm mismatch between the J-space ablation ({a:.8g}) and its random control "
            f"({b:.8g}); gap {gap:.3g} exceeds the registered tolerance {tolerance:g}."
        )
    return gap


# --------------------------------------------------------------------------------------
# Complementary-component clamp
# --------------------------------------------------------------------------------------


def clamp_complement(
    hidden_ablated: np.ndarray, hidden_clean: np.ndarray, directions: np.ndarray
) -> np.ndarray:
    """Clamp the component **outside** ``directions`` to its clean-pass value (§7).

    If an effect survives this clamp, it is not mediated by the complementary component;
    if it disappears, the residual effect was not J-mediated.
    """
    clean = np.asarray(hidden_clean, dtype=np.float64)
    ablated = np.asarray(hidden_ablated, dtype=np.float64)
    dirs = np.atleast_2d(np.asarray(directions, dtype=np.float64))
    if dirs.size == 0:
        return clean.copy()
    q, _ = np.linalg.qr(dirs.T)
    inside = q @ (q.T @ ablated)
    outside_clean = clean - q @ (q.T @ clean)
    return inside + outside_clean


# --------------------------------------------------------------------------------------
# General-damage check (§7, added by this protocol)
# --------------------------------------------------------------------------------------


def kl_divergence(p: np.ndarray, q: np.ndarray, *, eps: float = 1e-12) -> float:
    """``KL(p || q)`` in nats, over a shared vocabulary."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("distributions must share a shape")
    p = p / p.sum()
    q = q / q.sum()
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask] + eps) - np.log(q[mask] + eps))))


@dataclass
class GeneralDamageResult:
    """Is the J-space ablation distinguishable from a random-direction ablation? (§7)"""

    kl_jspace: np.ndarray
    kl_random: np.ndarray
    separable: bool
    statistic: float
    threshold: float
    top1_agreement_jspace: Optional[float] = None
    top1_agreement_random: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": "wda/general_damage/1",
            "mean_kl_jspace": float(np.mean(self.kl_jspace)),
            "mean_kl_random": float(np.mean(self.kl_random)),
            "separable": self.separable,
            "statistic": self.statistic,
            "threshold": self.threshold,
            "top1_agreement_jspace": self.top1_agreement_jspace,
            "top1_agreement_random": self.top1_agreement_random,
            "verdict": (
                "strength tier usable"
                if self.separable
                else "strength tier VOID — the J-space ablation is indistinguishable from "
                "a layer-matched random ablation on output KL (§7)"
            ),
        }

    def require(self) -> "GeneralDamageResult":
        if not self.separable:
            raise ControlViolation(self.to_dict()["verdict"])
        return self


def general_damage_check(
    clean_probs: Sequence[np.ndarray],
    jspace_probs: Sequence[np.ndarray],
    random_probs: Sequence[np.ndarray],
    *,
    threshold: float = 0.80,
    pretraining_top1_agreement: Optional[Tuple[float, float]] = None,
) -> GeneralDamageResult:
    """Compare output-KL distributions of the two ablations (§7).

    ``threshold`` is the AUC-style separability floor: the fraction of (j-space, random)
    pairs whose KL ordering is consistent. At 0.5 the two are indistinguishable and the
    strength tier is void.
    """
    kl_j = np.array([kl_divergence(a, c) for a, c in zip(jspace_probs, clean_probs)])
    kl_r = np.array([kl_divergence(a, c) for a, c in zip(random_probs, clean_probs)])
    if kl_j.size == 0 or kl_r.size == 0:
        raise ControlViolation("the general-damage check requires both ablation arms")
    # Common-language effect size (equivalently, the Mann-Whitney AUC).
    comparisons = kl_j[:, None] - kl_r[None, :]
    auc = float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)
    statistic = max(auc, 1.0 - auc)
    agree_j, agree_r = pretraining_top1_agreement or (None, None)
    return GeneralDamageResult(
        kl_jspace=kl_j,
        kl_random=kl_r,
        separable=statistic >= threshold,
        statistic=statistic,
        threshold=threshold,
        top1_agreement_jspace=agree_j,
        top1_agreement_random=agree_r,
    )


# --------------------------------------------------------------------------------------
# Mandatory-control registry
# --------------------------------------------------------------------------------------

#: Every control §7 marks mandatory. A run that reports fewer than these is incomplete.
MANDATORY_CONTROLS: Tuple[str, ...] = (
    "matched_norm_perturbation",
    "layer_matched_random",
    "clamp_complement",
    "answer_vs_intermediate_swap_depth",
    "position_control",
    "no_instruction_baseline",
    "shuffled_position_null",
    "control_words",
    "skip_clean_top10",
    "shuffled_corpus_lens",
    "direct_substitution_baseline",
    "logit_lens_floor",
    "general_damage",
)


def assert_controls_complete(reported: Iterable[str]) -> None:
    """Raise if any §7 mandatory control is missing from a phase report."""
    missing = sorted(set(MANDATORY_CONTROLS) - set(reported))
    if missing:
        raise ControlViolation(
            f"§7 requires every control to be reported; missing: {missing}. "
            "A report without them is incomplete (§17 item 5)."
        )


__all__ = [
    "ControlViolation",
    "GeneralDamageResult",
    "MANDATORY_CONTROLS",
    "assert_controls_complete",
    "assert_norm_matched",
    "clamp_complement",
    "general_damage_check",
    "kl_divergence",
    "layer_matched_random_ablation",
    "matched_norm_perturbation",
    "random_orthonormal_directions",
]
