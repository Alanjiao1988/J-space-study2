"""Workspace band location (§6.3) — a frozen automatic rule; no human in the loop.

Four statistics are computed per layer on the **calibration** model:

(a) ``top_k_hit_rate``      J-lens top-``k`` readout contains the model's own top-1 token
(b) ``excess_kurtosis``     excess kurtosis of the readout distribution
(c) ``top1_autocorrelation``  top-1 token agreement across adjacent positions, measured
                            against a position-shuffled null
(d) ``effective_dimension``  participation ratio of the Jacobian spectrum

The band runs from the onset of the kurtosis rise to the onset of the sharp rise in
next-token accuracy.

**Repair of the predecessor's rule defect (fact F6').** The predecessor's registered rule
was "the longest run at or above half of the maximum" — scale-free by construction, with
*no absolute floor and no minimum length*. Under it a band could be, and was, emitted at
length 1, and before the matched-norm null was added the gpt2 negative control produced a
band as well. This rule therefore adds four requirements that the predecessor's lacked:

1. :attr:`BandRule.absolute_floor` — an absolute readrate floor, not a relative one;
2. :attr:`BandRule.min_length` — a minimum band length;
3. :attr:`BandRule.null_margin` — every band layer must beat the matched-norm random-lens
   null by a stated margin;
4. :attr:`BandRule.reject_right_censored` — a band whose upper edge is the last layer is
   right-censored by the end of the network and is refused rather than reported.

A rule that cannot produce a band returns ``admissible=False`` with the reasons listed.
That is a legitimate result; it is not repaired by relaxing the rule (§11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from wda.errors import BandRuleViolation


# --------------------------------------------------------------------------------------
# (a)-(d): the four statistics
# --------------------------------------------------------------------------------------


def top_k_hit_rate(readout_probs: np.ndarray, model_top1: np.ndarray, k: int) -> float:
    """(a) Fraction of positions whose J-lens top-``k`` contains the model's top-1 token."""
    probs = np.atleast_2d(np.asarray(readout_probs, dtype=float))
    top1 = np.asarray(model_top1, dtype=int).reshape(-1)
    if probs.shape[0] != top1.shape[0]:
        raise ValueError("readout_probs and model_top1 disagree on the number of positions")
    if k < 1:
        raise ValueError("k must be >= 1")
    k = min(k, probs.shape[1])
    topk = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([top1[i] in set(topk[i].tolist()) for i in range(len(top1))]))


def excess_kurtosis(readout_logits: np.ndarray) -> float:
    """(b) Excess kurtosis of the readout, averaged over positions.

    A workspace-like readout is heavy-tailed: probability mass concentrates on a few
    tokens rather than spreading over the vocabulary.
    """
    x = np.atleast_2d(np.asarray(readout_logits, dtype=float))
    mean = x.mean(axis=1, keepdims=True)
    centred = x - mean
    var = np.mean(centred**2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        kurt = np.mean(centred**4, axis=1) / np.where(var > 0, var**2, np.nan) - 3.0
    return float(np.nanmean(kurt))


def top1_autocorrelation(top1_by_position: Sequence[int], lag: int = 1) -> float:
    """(c) Fraction of position pairs at ``lag`` whose J-lens top-1 token agrees."""
    seq = np.asarray(list(top1_by_position), dtype=int)
    if len(seq) <= lag:
        return float("nan")
    return float(np.mean(seq[:-lag] == seq[lag:]))


def position_shuffled_null(
    top1_by_position: Sequence[int], *, lag: int = 1, n_draws: int = 200, seed: int = 0
) -> Tuple[float, float]:
    """(c) Null for :func:`top1_autocorrelation`: shuffle positions, recompute (§7)."""
    rng = np.random.default_rng(seed)
    seq = np.asarray(list(top1_by_position), dtype=int)
    draws = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        draws[i] = top1_autocorrelation(rng.permutation(seq), lag=lag)
    return float(np.nanmean(draws)), float(np.nanstd(draws))


def effective_dimension(jacobian: np.ndarray) -> float:
    """(d) Participation ratio of the singular-value spectrum: ``(sum s)^2 / sum s^2``."""
    s = np.linalg.svd(np.asarray(jacobian, dtype=np.float64), compute_uv=False)
    denom = float(np.sum(s**2))
    if denom <= 0:
        return 0.0
    return float(np.sum(s) ** 2 / denom)


def identity_energy(jacobian: np.ndarray) -> float:
    """Share of ``J``'s energy explained by the best scaled identity ``alpha * I``.

    Not a band statistic, but recorded alongside them: fact F6' found identity energy
    0.749 with ``alpha ~ 1.0`` in the late layers, and the protocol's §2 verification
    records that late-layer ``J -> I`` is *expected motor-layer behaviour*, not a
    pathology. Reporting it prevents that finding being re-discovered as a surprise.
    """
    j = np.asarray(jacobian, dtype=np.float64)
    n = j.shape[0]
    alpha = float(np.trace(j) / n)
    total = float(np.sum(j**2))
    if total <= 0:
        return 0.0
    return float((alpha**2 * n) / total)


# --------------------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------------------


@dataclass
class LayerStatistics:
    """Per-layer statistics, all indexed by ``layers``."""

    layers: Tuple[int, ...]
    top_k_hit_rate: np.ndarray
    excess_kurtosis: np.ndarray
    top1_autocorrelation: np.ndarray
    autocorrelation_null: np.ndarray
    effective_dimension: np.ndarray
    next_token_accuracy: np.ndarray
    random_lens_null_hit_rate: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        n = len(self.layers)
        for name in (
            "top_k_hit_rate",
            "excess_kurtosis",
            "top1_autocorrelation",
            "autocorrelation_null",
            "effective_dimension",
            "next_token_accuracy",
        ):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.shape != (n,):
                raise ValueError(f"{name} has shape {arr.shape}, expected ({n},)")
            setattr(self, name, arr)
        if self.random_lens_null_hit_rate is not None:
            arr = np.asarray(self.random_lens_null_hit_rate, dtype=float)
            if arr.shape != (n,):
                raise ValueError("random_lens_null_hit_rate has the wrong shape")
            self.random_lens_null_hit_rate = arr


@dataclass(frozen=True)
class BandRule:
    """The frozen §6.3 rule. Values are set in ``configs/freeze1/band_rule.json``."""

    #: (1) absolute readrate floor — the repair the predecessor's scale-free rule lacked.
    absolute_floor: float = 0.05
    #: relative floor, retained from the predecessor rule as an *additional* requirement.
    relative_floor: float = 0.50
    #: (2) minimum admissible band length in layers.
    min_length: int = 4
    #: (3) margin by which each band layer must beat the matched-norm random-lens null.
    null_margin: float = 0.02
    #: (4) refuse a band whose upper edge is the final layer (right-censored).
    reject_right_censored: bool = True
    #: kurtosis rise onset: first layer whose excess kurtosis exceeds this quantile of the
    #: per-model kurtosis profile *and* is increasing.
    kurtosis_onset_quantile: float = 0.60
    #: next-token-accuracy sharp rise: the layer of the largest first difference.
    accuracy_rise_min_jump: float = 0.05
    #: band layers must also beat the position-shuffled autocorrelation null by this much.
    autocorrelation_margin: float = 0.05

    def to_dict(self) -> Dict[str, object]:
        return {
            "absolute_floor": self.absolute_floor,
            "relative_floor": self.relative_floor,
            "min_length": self.min_length,
            "null_margin": self.null_margin,
            "reject_right_censored": self.reject_right_censored,
            "kurtosis_onset_quantile": self.kurtosis_onset_quantile,
            "accuracy_rise_min_jump": self.accuracy_rise_min_jump,
            "autocorrelation_margin": self.autocorrelation_margin,
        }


@dataclass
class BandResult:
    admissible: bool
    band: Tuple[int, ...]
    reasons: List[str] = field(default_factory=list)
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": "wda/band_result/1",
            "admissible": self.admissible,
            "band": list(self.band),
            "length": len(self.band),
            "reasons": list(self.reasons),
            "diagnostics": dict(self.diagnostics),
        }

    def require(self) -> Tuple[int, ...]:
        if not self.admissible:
            raise BandRuleViolation(
                "the automatic band rule produced no admissible band: "
                + "; ".join(self.reasons)
                + ". The rule may not be relaxed to produce one (§11)."
            )
        return self.band


def _kurtosis_onset(stats: LayerStatistics, rule: BandRule) -> Optional[int]:
    kurt = stats.excess_kurtosis
    finite = kurt[np.isfinite(kurt)]
    if finite.size == 0:
        return None
    threshold = float(np.quantile(finite, rule.kurtosis_onset_quantile))
    for idx in range(len(kurt) - 1):
        if kurt[idx] >= threshold and kurt[idx + 1] >= kurt[idx]:
            return idx
    return None


def _accuracy_rise_onset(stats: LayerStatistics, rule: BandRule, start: int) -> Optional[int]:
    acc = stats.next_token_accuracy
    if len(acc) < 2:
        return None
    diffs = np.diff(acc)
    candidates = [
        i + 1 for i in range(start, len(diffs)) if diffs[i] >= rule.accuracy_rise_min_jump
    ]
    if not candidates:
        return None
    # The sharpest rise at or after the kurtosis onset marks the motor transition.
    best = max(candidates, key=lambda i: diffs[i - 1])
    return best


def locate_band(stats: LayerStatistics, rule: BandRule) -> BandResult:
    """Apply the frozen §6.3 rule. Returns a result; never silently widens the band."""
    reasons: List[str] = []
    diagnostics: Dict[str, object] = {"rule": rule.to_dict()}

    lower_idx = _kurtosis_onset(stats, rule)
    if lower_idx is None:
        return BandResult(False, (), ["no kurtosis rise onset found"], diagnostics)
    upper_idx = _accuracy_rise_onset(stats, rule, lower_idx)
    if upper_idx is None:
        return BandResult(
            False,
            (),
            [f"no next-token-accuracy rise of >= {rule.accuracy_rise_min_jump} after the "
             f"kurtosis onset at layer {stats.layers[lower_idx]}"],
            diagnostics,
        )

    idx_range = list(range(lower_idx, upper_idx))
    diagnostics["kurtosis_onset_layer"] = int(stats.layers[lower_idx])
    diagnostics["accuracy_rise_layer"] = int(stats.layers[upper_idx])

    if not idx_range:
        return BandResult(False, (), ["the accuracy rise precedes the kurtosis onset"], diagnostics)

    hit = stats.top_k_hit_rate
    observed_max = float(np.nanmax(hit)) if np.isfinite(hit).any() else 0.0
    relative_bar = rule.relative_floor * observed_max
    diagnostics["observed_max_hit_rate"] = observed_max
    diagnostics["relative_bar"] = relative_bar

    kept: List[int] = []
    for i in idx_range:
        layer = int(stats.layers[i])
        if hit[i] < rule.absolute_floor:
            reasons.append(
                f"layer {layer}: hit rate {hit[i]:.4f} below the absolute floor "
                f"{rule.absolute_floor} (repair 1 of the F6' rule defect)"
            )
            continue
        if hit[i] < relative_bar:
            reasons.append(f"layer {layer}: hit rate {hit[i]:.4f} below the relative bar {relative_bar:.4f}")
            continue
        if stats.random_lens_null_hit_rate is not None:
            margin = hit[i] - float(stats.random_lens_null_hit_rate[i])
            if margin < rule.null_margin:
                reasons.append(
                    f"layer {layer}: beats the matched-norm random-lens null by only "
                    f"{margin:.4f} < {rule.null_margin} (repair 3)"
                )
                continue
        ac_margin = float(stats.top1_autocorrelation[i] - stats.autocorrelation_null[i])
        if ac_margin < rule.autocorrelation_margin:
            reasons.append(
                f"layer {layer}: top-1 autocorrelation exceeds the position-shuffled null "
                f"by only {ac_margin:.4f} < {rule.autocorrelation_margin}"
            )
            continue
        kept.append(i)

    if not kept:
        reasons.insert(0, "no layer in the candidate interval survived the floors")
        return BandResult(False, (), reasons, diagnostics)

    # Longest contiguous run among the surviving layers.
    runs: List[List[int]] = [[kept[0]]]
    for i in kept[1:]:
        if i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    best_run = max(runs, key=len)
    band = tuple(int(stats.layers[i]) for i in best_run)
    diagnostics["candidate_runs"] = [[int(stats.layers[i]) for i in run] for run in runs]

    if len(band) < rule.min_length:
        reasons.append(
            f"longest admissible run is {len(band)} layer(s) {list(band)}, below the minimum "
            f"length {rule.min_length} (repair 2 of the F6' rule defect: the predecessor "
            "rule had no minimum length and emitted a length-1 band)"
        )
        return BandResult(False, (), reasons, diagnostics)

    if rule.reject_right_censored and band[-1] == int(stats.layers[-1]):
        reasons.append(
            "the band's upper edge is the final layer, so it is right-censored by the end "
            "of the network (repair 4); refused rather than reported"
        )
        return BandResult(False, (), reasons, diagnostics)

    return BandResult(True, band, reasons, diagnostics)


def band_halves(band: Sequence[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Split a band into its first and second halves.

    Phase-B criterion 3 requires the real lens to be *no worse than* the logit lens on the
    **first half** of the band, where the paper claims the two diverge — and explicitly
    does not require superiority across all layers (§10 Phase B).
    """
    band = tuple(int(x) for x in band)
    mid = (len(band) + 1) // 2
    return band[:mid], band[mid:]


__all__ = [
    "BandResult",
    "BandRule",
    "LayerStatistics",
    "band_halves",
    "effective_dimension",
    "excess_kurtosis",
    "identity_energy",
    "locate_band",
    "position_shuffled_null",
    "top1_autocorrelation",
    "top_k_hit_rate",
]
