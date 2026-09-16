"""Synthetic/engineering subspace fixtures; NOT qualified J-space experiments.

The SVD interpretation below is unverified against the scientific token-J definition.
No published ablation/patch API exists in the pinned upstream. Scientific callers must
call ``require_scientific_interventions`` and remain blocked pending validation.

At **every layer of the workspace band** and **every position of window W**:

1. take the ``k`` J-lens directions with the strongest projection under the *clean* pass;
2. **exclude** any direction whose lens readout token appears in the clean pass's output
   top-10 — so the ablation never touches the content about to be emitted (§7);
3. zero the residual stream's projection onto the surviving directions.

``k``, the band and the strength tier are fixed at Freeze-2; ``skip_clean_top = 10`` and the
exclusion rule itself are frozen at Freeze-1.

Strength tiers are defined **by layer band** (§5, Appendix B "three layer-band tiers"), not
by a scaling coefficient: ``light`` ablates the first third of the band, ``medium`` the
first two thirds, ``heavy`` the whole band. A tier is only usable if the general-damage
check can separate it from the layer-matched random control (§7); see
:mod:`wda.intervene.controls`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from wda.errors import ProtocolViolation
from wda.intervene import precision
from wda.lens.fit import LensBundle, rms_norm
from wda.intervene.subspace import (
    SubspaceProjector,
    canonical_svd,
    floating_vector,
    require_scientific_interventions,
)

#: Frozen at Freeze-1 (§15.3).
SKIP_CLEAN_TOP = 10


class AblationViolation(ProtocolViolation):
    """§5 — an ablation was requested outside the frozen definition."""


class Strength(str, Enum):
    """§5 strength tiers, defined by layer band."""

    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"

    def layers(self, band: Sequence[int]) -> Tuple[int, ...]:
        band = tuple(int(x) for x in band)
        if not band:
            raise AblationViolation("the band is empty; locate it with wda.lens.band first")
        n = len(band)
        if self is Strength.LIGHT:
            return band[: max(1, n // 3)]
        if self is Strength.MEDIUM:
            return band[: max(1, (2 * n) // 3)]
        return band


@dataclass(frozen=True)
class AblationSpec:
    """§15.3 ablation configuration."""

    band: Tuple[int, ...]
    k: int
    strength: Strength = Strength.MEDIUM
    skip_clean_top: int = SKIP_CLEAN_TOP
    precision_dtype: str = precision.REQUIRED_DTYPE
    control: str = "layer_matched_random"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.k < 0:
            raise AblationViolation("k must be non-negative (k = 0 is the no-op fixture)")
        if self.skip_clean_top != SKIP_CLEAN_TOP:
            raise AblationViolation(
                f"skip_clean_top is frozen at {SKIP_CLEAN_TOP} (§5, §15.3); "
                f"got {self.skip_clean_top}"
            )
        if self.precision_dtype != precision.REQUIRED_DTYPE:
            raise AblationViolation(
                f"precision must be {precision.REQUIRED_DTYPE} (fact F7); got {self.precision_dtype}"
            )

    @property
    def active_layers(self) -> Tuple[int, ...]:
        return self.strength.layers(self.band)

    def to_dict(self) -> Dict[str, object]:
        return {
            "band": list(self.band),
            "k": self.k,
            "strength": self.strength.value,
            "skip_clean_top": self.skip_clean_top,
            "precision": self.precision_dtype,
            "control": self.control,
            "seed": self.seed,
            "active_layers": list(self.active_layers),
        }

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()


# --------------------------------------------------------------------------------------
# Direction selection
# --------------------------------------------------------------------------------------


def direction_readout_tokens(
    bundle: LensBundle,
    layer: int,
    directions: np.ndarray,
    unembed: np.ndarray,
) -> np.ndarray:
    """Synthetic positive-orientation readout; not a scientific skip-rule definition."""
    j = np.asarray(bundle.matrix(layer), dtype=np.float64)
    dirs = np.atleast_2d(np.asarray(directions, dtype=np.float64))
    projected = dirs @ j.T
    if not len(dirs):
        return np.empty(0, dtype=int)
    # Bound the temporary vocabulary matrix instead of materialising d_model x vocab.
    tokens = []
    for start in range(0, len(projected), 8):
        normed = rms_norm(projected[start : start + 8])
        logits = normed @ np.asarray(unembed, dtype=np.float64).T
        tokens.extend(np.argmax(logits, axis=1).tolist())
    return np.asarray(tokens, dtype=int)


@dataclass(frozen=True)
class SyntheticDirectionCache:
    """One explicit layer snapshot, reused across positions; never a scientific provider."""

    layer: int
    directions: np.ndarray
    positive_tokens: np.ndarray
    negative_tokens: np.ndarray

    @classmethod
    def prepare(cls, bundle: LensBundle, layer: int, unembed: np.ndarray):
        _, singular, vt = canonical_svd(np.asarray(bundle.matrix(layer), dtype=np.float64))
        tolerance = np.finfo(vt.dtype).eps * max(bundle.matrix(layer).shape) * singular[0]
        vt = vt[singular > tolerance]
        positive = direction_readout_tokens(bundle, layer, vt, unembed)
        negative = direction_readout_tokens(bundle, layer, -vt, unembed)
        for array in (vt, positive, negative):
            array.setflags(write=False)
        return cls(layer, vt, positive, negative)


def select_ablation_directions(
    bundle: LensBundle,
    layer: int,
    hidden: np.ndarray,
    unembed: np.ndarray,
    *,
    k: int,
    clean_top_tokens: Iterable[int],
    skip_clean_top: int = SKIP_CLEAN_TOP,
    cache: Optional[SyntheticDirectionCache] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Return synthetic ``k`` admissible SVD directions, never a qualified experiment.

    Candidates are ranked by the magnitude of the clean-pass projection. A candidate whose
    lens readout token is among the clean pass's top-``skip_clean_top`` tokens is skipped
    and the next-strongest is taken, so exactly ``k`` directions are ablated whenever that
    many admissible ones exist.
    """
    h = floating_vector(hidden)
    if k < 0 or skip_clean_top != SKIP_CLEAN_TOP:
        raise AblationViolation("k must be nonnegative and skip_clean_top is frozen at 10")
    if k == 0:
        return np.zeros((0, len(hidden)), dtype=h.dtype), {
            "requested_k": 0,
            "selected": 0,
            "skipped_tokens": [],
            "noop": True,
            "qualification": "synthetic/engineering_only",
        }
    skip = set(int(t) for t in list(clean_top_tokens)[:skip_clean_top])
    cache = cache or SyntheticDirectionCache.prepare(bundle, layer, unembed)
    if cache.layer != layer or cache.directions.shape[1] != h.size:
        raise AblationViolation("direction cache does not match layer/hidden dimension")
    vt = cache.directions
    coeffs = vt @ h
    order = np.argsort(-np.abs(coeffs), kind="stable")
    chosen: List[int] = []
    skipped: List[Dict[str, int]] = []
    for rank, idx in enumerate(order):
        if len(chosen) == k:
            break
        token = int(cache.positive_tokens[idx])
        negative_token = int(cache.negative_tokens[idx])
        # Both orientations are excluded: skip decisions cannot flip with an SVD sign.
        if token in skip or negative_token in skip:
            token = token if token in skip else negative_token
            skipped.append({"direction_rank": int(rank), "readout_token": token})
            continue
        chosen.append(int(idx))

    receipt = {
        "requested_k": k,
        "selected": len(chosen),
        "skipped_for_clean_top": skipped,
        "skip_clean_top": skip_clean_top,
        "noop": False,
        "exhausted": len(chosen) < k,
        "qualification": "synthetic/engineering_only",
        "skip_interpretation": "both_svd_orientations_unvalidated",
    }
    return vt[chosen], receipt


# --------------------------------------------------------------------------------------
# The intervention
# --------------------------------------------------------------------------------------


def project_out(hidden: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Zero the projection of ``hidden`` onto the rows of ``directions``.

    The rows are treated as a subspace basis and orthonormalised first, so the result is
    exact even if the rows are not perfectly orthogonal after a numerical SVD.
    """
    h = floating_vector(hidden)
    return SubspaceProjector.prepare(directions, h.size, dtype=h.dtype).remove(h)


def removed_norm(hidden: np.ndarray, directions: np.ndarray) -> float:
    """L2 norm of the component removed — the quantity a matched-norm control must match."""
    h = np.asarray(hidden, dtype=np.float64)
    return float(np.linalg.norm(h - project_out(h, directions)))


@dataclass
class AblationReceipt:
    """Per-(layer, position) record; aggregated into ``log.jsonl`` (§15.4)."""

    layer: int
    position: int
    selected: int
    removed_norm: float
    residual_projection: float
    detail: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "layer": self.layer,
            "position": self.position,
            "selected": self.selected,
            "removed_norm": self.removed_norm,
            "residual_projection": self.residual_projection,
            "detail": dict(self.detail),
        }


def ablate_at(
    bundle: LensBundle,
    layer: int,
    position: int,
    hidden: np.ndarray,
    unembed: np.ndarray,
    spec: AblationSpec,
    clean_top_tokens: Iterable[int],
    *,
    cache: Optional[SyntheticDirectionCache] = None,
) -> Tuple[np.ndarray, AblationReceipt]:
    """Synthetic fixture only. Returns ``(hidden', receipt)``; not a scientific API."""
    directions, sel = select_ablation_directions(
        bundle,
        layer,
        hidden,
        unembed,
        k=spec.k,
        clean_top_tokens=clean_top_tokens,
        skip_clean_top=spec.skip_clean_top,
        cache=cache,
    )
    ablated = project_out(hidden, directions)
    residual = (
        float(np.max(np.abs(np.atleast_2d(directions) @ ablated))) if directions.size else 0.0
    )
    receipt = AblationReceipt(
        layer=int(layer),
        position=int(position),
        selected=int(sel["selected"]),
        removed_norm=float(np.linalg.norm(np.asarray(hidden, float) - ablated)),
        residual_projection=residual,
        detail=sel,
    )
    return ablated, receipt


def ablate_window(
    bundle: LensBundle,
    hidden_by_layer_position: Dict[Tuple[int, int], np.ndarray],
    unembed: np.ndarray,
    spec: AblationSpec,
    clean_top_tokens_by_position: Dict[int, Sequence[int]],
) -> Tuple[Dict[Tuple[int, int], np.ndarray], List[AblationReceipt]]:
    """Synthetic window fixture with one prepared direction cache per active layer."""
    out: Dict[Tuple[int, int], np.ndarray] = {}
    receipts: List[AblationReceipt] = []
    active = set(spec.active_layers)
    caches = {}
    for (layer, position), hidden in sorted(hidden_by_layer_position.items()):
        if layer not in active:
            out[(layer, position)] = floating_vector(hidden).copy()
            continue
        top_tokens = clean_top_tokens_by_position.get(position, ())
        if spec.k and layer not in caches:
            caches[layer] = SyntheticDirectionCache.prepare(bundle, layer, unembed)
        ablated, receipt = ablate_at(
            bundle, layer, position, hidden, unembed, spec, top_tokens, cache=caches.get(layer)
        )
        out[(layer, position)] = ablated
        receipts.append(receipt)
    return out, receipts


__all__ = [
    "AblationReceipt",
    "AblationSpec",
    "AblationViolation",
    "SKIP_CLEAN_TOP",
    "Strength",
    "SyntheticDirectionCache",
    "ablate_at",
    "ablate_window",
    "direction_readout_tokens",
    "project_out",
    "removed_norm",
    "require_scientific_interventions",
    "select_ablation_directions",
]
