"""Lens-coordinate patching (§14 ``intervene/patch.py``).

Read the residual stream in the lens's own coordinate system (``V^+`` from the SVD of
``J_l``), substitute the donor's coordinates on the selected components, then write back.
This is the operation behind probe-swap and the verbal-report swap, and it is the only
place a *donor* representation is allowed to enter a recipient forward pass.

Two paired controls travel with every swap and are not optional (§7):

* ``direct_substitution_baseline`` — tao-hpu's alternative explanation: simply substituting
  the donor's final-position token distribution reproduces much of the apparent effect.
  A probe-swap result reported without it is uninterpretable.
* ``answer_vs_intermediate_swap_depth`` — the paper's own check that the intermediate
  quantity is not smuggling the answer: the intermediate swap should take effect at a
  measurably shallower depth than the answer swap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from wda.errors import ProtocolViolation
from wda.lens.fit import LensBundle


class PatchViolation(ProtocolViolation):
    """§7 — a swap was run without its mandatory paired control."""


@dataclass
class LensBasis:
    """Cached SVD of one layer's Jacobian: ``J = U S V^T``."""

    layer: int
    u: np.ndarray
    s: np.ndarray
    vt: np.ndarray

    @classmethod
    def from_bundle(cls, bundle: LensBundle, layer: int) -> "LensBasis":
        j = np.asarray(bundle.matrix(layer), dtype=np.float64)
        u, s, vt = np.linalg.svd(j, full_matrices=False)
        return cls(layer=int(layer), u=u, s=s, vt=vt)

    @property
    def rank(self) -> int:
        return int(self.s.size)

    def coordinates(self, hidden: np.ndarray) -> np.ndarray:
        """``c = V^+ h`` — the residual stream read in lens coordinates."""
        return self.vt @ np.asarray(hidden, dtype=np.float64)

    def reconstruct(self, coordinates: np.ndarray) -> np.ndarray:
        """``h = V c`` — write lens coordinates back into the residual stream."""
        return self.vt.T @ np.asarray(coordinates, dtype=np.float64)

    def top_components(self, n: int) -> np.ndarray:
        """Indices of the ``n`` components with the largest singular values."""
        return np.argsort(-self.s)[:n]

    def j_component(self, hidden: np.ndarray, components: Sequence[int]) -> np.ndarray:
        """The J-restricted part of ``hidden`` on ``components`` (§Fig-29 style contrast)."""
        idx = np.asarray(list(components), dtype=int)
        basis = self.vt[idx]
        return basis.T @ (basis @ np.asarray(hidden, dtype=np.float64))

    def non_j_component(self, hidden: np.ndarray, components: Sequence[int]) -> np.ndarray:
        return np.asarray(hidden, dtype=np.float64) - self.j_component(hidden, components)


def swap_lens_coordinates(
    recipient: np.ndarray,
    donor: np.ndarray,
    basis: LensBasis,
    components: Sequence[int],
) -> np.ndarray:
    """Replace the recipient's lens coordinates on ``components`` with the donor's.

    Everything outside ``components`` is left bit-identical to the recipient, so the swap
    is confined to the J-space subspace by construction.
    """
    idx = np.asarray(list(components), dtype=int)
    if idx.size == 0:
        return np.asarray(recipient, dtype=np.float64).copy()
    if idx.max(initial=-1) >= basis.rank:
        raise PatchViolation(f"component index {int(idx.max())} exceeds lens rank {basis.rank}")
    c_recipient = basis.coordinates(recipient)
    c_donor = basis.coordinates(donor)
    c_out = c_recipient.copy()
    c_out[idx] = c_donor[idx]
    return basis.reconstruct(c_out)


def permute_singular_coordinates(
    hidden: np.ndarray, basis: LensBasis, permutation: Sequence[int]
) -> np.ndarray:
    """Apply a permutation ``sigma`` to the lens coordinates and write back.

    Used as a structure-preserving null: the coordinate magnitudes are unchanged, only
    their assignment to lens directions is scrambled.
    """
    perm = np.asarray(list(permutation), dtype=int)
    if perm.shape != (basis.rank,) or sorted(perm.tolist()) != list(range(basis.rank)):
        raise PatchViolation("permutation must be a bijection over the lens components")
    return basis.reconstruct(basis.coordinates(hidden)[perm])


def random_permutation(basis: LensBasis, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(basis.rank)


# --------------------------------------------------------------------------------------
# Mandatory paired controls
# --------------------------------------------------------------------------------------


@dataclass
class SwapOutcome:
    """One swap trial and the controls it must be reported with."""

    target_logit_delta: float
    matched_norm_random_delta: float
    direct_substitution_delta: Optional[float] = None
    swap_depth: Optional[float] = None
    detail: Dict[str, object] = field(default_factory=dict)

    def positive_control_holds(self, *, min_margin: float = 0.0) -> bool:
        """Phase-0 fixture 5 / Phase-B criterion 4: the swap moves the target, the null does not."""
        return (
            self.target_logit_delta > min_margin
            and self.matched_norm_random_delta <= min_margin
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "target_logit_delta": self.target_logit_delta,
            "matched_norm_random_delta": self.matched_norm_random_delta,
            "direct_substitution_delta": self.direct_substitution_delta,
            "swap_depth": self.swap_depth,
            "positive_control_holds": self.positive_control_holds(),
            "detail": dict(self.detail),
        }


def require_direct_substitution_baseline(outcome: SwapOutcome) -> SwapOutcome:
    """§7 / §8.5 — probe-swap must be reported with the direct-substitution baseline."""
    if outcome.direct_substitution_delta is None:
        raise PatchViolation(
            "probe-swap results must be accompanied by the direct final-token substitution "
            "baseline (tao-hpu). Without it the swap effect and simple output substitution "
            "are not distinguishable (§7)."
        )
    return outcome


def swap_depth_separation(
    intermediate_depths: Sequence[float], answer_depths: Sequence[float]
) -> Dict[str, float]:
    """``answer_vs_intermediate_swap_depth`` control (§7).

    The paper reports the intermediate-quantity swap taking effect roughly 17% of the
    network's depth earlier than the answer swap. A separation at or below zero is
    consistent with the intermediate vector simply carrying the answer.
    """
    inter = np.asarray(list(intermediate_depths), dtype=float)
    ans = np.asarray(list(answer_depths), dtype=float)
    if inter.size == 0 or ans.size == 0:
        raise PatchViolation("both swap-depth arms are required")
    separation = float(np.mean(ans) - np.mean(inter))
    return {
        "mean_intermediate_depth": float(np.mean(inter)),
        "mean_answer_depth": float(np.mean(ans)),
        "separation": separation,
        "intermediate_is_earlier": separation > 0.0,
        "paper_reference_separation": 0.17,
    }


__all__ = [
    "LensBasis",
    "PatchViolation",
    "SwapOutcome",
    "permute_singular_coordinates",
    "random_permutation",
    "require_direct_substitution_baseline",
    "swap_depth_separation",
    "swap_lens_coordinates",
]
