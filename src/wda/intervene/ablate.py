"""J-space ablation (§5, frozen elements).

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
    """The lens readout token of each candidate direction — used by the skip rule (§5)."""
    j = np.asarray(bundle.matrix(layer), dtype=np.float64)
    dirs = np.atleast_2d(np.asarray(directions, dtype=np.float64))
    projected = dirs @ j.T
    normed = np.stack([rms_norm(row) for row in projected])
    logits = normed @ np.asarray(unembed, dtype=np.float64).T
    return np.argmax(logits, axis=1)


def select_ablation_directions(
    bundle: LensBundle,
    layer: int,
    hidden: np.ndarray,
    unembed: np.ndarray,
    *,
    k: int,
    clean_top_tokens: Iterable[int],
    skip_clean_top: int = SKIP_CLEAN_TOP,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Return the ``k`` admissible directions at one (layer, position), plus a receipt.

    Candidates are ranked by the magnitude of the clean-pass projection. A candidate whose
    lens readout token is among the clean pass's top-``skip_clean_top`` tokens is skipped
    and the next-strongest is taken, so exactly ``k`` directions are ablated whenever that
    many admissible ones exist.
    """
    if k == 0:
        return np.zeros((0, len(hidden)), dtype=np.float64), {
            "requested_k": 0,
            "selected": 0,
            "skipped_tokens": [],
            "noop": True,
        }
    skip = set(int(t) for t in list(clean_top_tokens)[:skip_clean_top])
    j = np.asarray(bundle.matrix(layer), dtype=np.float64)
    _, _, vt = np.linalg.svd(j, full_matrices=False)
    coeffs = vt @ np.asarray(hidden, dtype=np.float64)
    order = np.argsort(-np.abs(coeffs))

    tokens = direction_readout_tokens(bundle, layer, vt[order], unembed)
    chosen: List[int] = []
    skipped: List[Dict[str, int]] = []
    for rank, idx in enumerate(order):
        if len(chosen) == k:
            break
        token = int(tokens[rank])
        if token in skip:
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
    h = np.asarray(hidden, dtype=np.float64)
    dirs = np.atleast_2d(np.asarray(directions, dtype=np.float64))
    if dirs.size == 0:
        return h.copy()
    q, _ = np.linalg.qr(dirs.T)
    return h - q @ (q.T @ h)


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
) -> Tuple[np.ndarray, AblationReceipt]:
    """Apply the frozen ablation at one (layer, position). Returns ``(hidden', receipt)``."""
    directions, sel = select_ablation_directions(
        bundle,
        layer,
        hidden,
        unembed,
        k=spec.k,
        clean_top_tokens=clean_top_tokens,
        skip_clean_top=spec.skip_clean_top,
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
    """Apply the ablation across the active layers x window-W positions (§5)."""
    out: Dict[Tuple[int, int], np.ndarray] = {}
    receipts: List[AblationReceipt] = []
    active = set(spec.active_layers)
    for (layer, position), hidden in sorted(hidden_by_layer_position.items()):
        if layer not in active:
            out[(layer, position)] = np.asarray(hidden, dtype=np.float64).copy()
            continue
        top_tokens = clean_top_tokens_by_position.get(position, ())
        ablated, receipt = ablate_at(
            bundle, layer, position, hidden, unembed, spec, top_tokens
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
    "ablate_at",
    "ablate_window",
    "direction_readout_tokens",
    "project_out",
    "removed_norm",
    "select_ablation_directions",
]
