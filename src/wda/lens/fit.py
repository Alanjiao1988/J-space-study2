"""Synthetic NumPy JVP fitting utilities, not the pinned upstream fit adapter.

These estimators support engineering tests only. The upstream implementation uses
exact row-batched VJPs, not this stochastic helper; see :mod:`wda.lens.upstream`.

Per subject the protocol requires **five** readouts:

===================  ==================================================================
``real`` x 2         two independent fits on *disjoint* corpus samples with different
                     seeds; both are always run and paired-averaged (§6.2)
``shuffled``         same corpus size, token order permuted within each prompt; the
                     community control that ``amaljithkuttamath`` failed (§7)
``logit``            ``J_l = I`` — the paper's own statement that the logit lens
                     "captures much of the workspace-like structure" makes this the
                     floor the real lens must clear (§7)
``tuned``            optional additional baseline
===================  ==================================================================

Frozen fit-scale floors (§6.1): ``n_prompts >= 200`` and ``skip_first >= 16``. The
`amaljithkuttamath` replication used ``n=25`` / ``skip_first=4`` and its shuffled lens was
*not worse* than the real one on all six evaluations; those floors exist to keep this
project out of that regime. The exact scale above the floor is a Phase-A calibration
object.

The Jacobian ``J_l = E_{t, t'>=t, prompt}[ d h_final,t' / d h_l,t ]`` is estimated
stochastically: for isotropic Gaussian probes ``u``, ``E[(J u) u^T] = J``, so ``m`` JVPs
give an unbiased rank-``m`` estimate. ``m`` is recorded in the fit metadata — a rank-``m``
estimate is *not* the exact Jacobian and the difference must never be elided in a report.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from wda.errors import ProtocolViolation

#: §6.1 frozen floors.
MIN_PROMPTS = 200
MIN_SKIP_FIRST = 16

LENS_KINDS = ("real", "shuffled", "logit", "tuned")


class LensFitViolation(ProtocolViolation):
    """§6.1 — a fit was requested below the frozen floors, or with overlapping corpora."""


@dataclass(frozen=True)
class CorpusSample:
    """A disjoint slice of the fitting corpus."""

    corpus_id: str
    prompt_ids: Tuple[str, ...]
    seed: int

    def digest(self) -> str:
        payload = json.dumps(
            {"corpus_id": self.corpus_id, "prompt_ids": list(self.prompt_ids), "seed": self.seed},
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def __len__(self) -> int:
        return len(self.prompt_ids)


@dataclass(frozen=True)
class LensSpec:
    """Everything that determines a fit. Hashed into ``FREEZE-2.json``."""

    lens_id: str
    kind: str
    model_key: str
    n_prompts: int
    skip_first: int
    n_probes: int
    seq_len: int = 128
    seed: int = 0
    corpus_id: str = "fit_corpus_v1"
    layers: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if self.kind not in LENS_KINDS:
            raise LensFitViolation(f"unknown lens kind {self.kind!r}; expected one of {LENS_KINDS}")
        if self.kind in ("real", "shuffled", "tuned"):
            if self.n_prompts < MIN_PROMPTS:
                raise LensFitViolation(
                    f"n_prompts={self.n_prompts} is below the frozen floor of {MIN_PROMPTS} "
                    "(§6.1). The amaljithkuttamath replication failed at n=25."
                )
            if self.skip_first < MIN_SKIP_FIRST:
                raise LensFitViolation(
                    f"skip_first={self.skip_first} is below the frozen floor of "
                    f"{MIN_SKIP_FIRST} (§6.1)."
                )
            if self.n_probes < 1:
                raise LensFitViolation("n_probes must be >= 1 for a stochastic Jacobian estimate")

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> Dict[str, object]:
        return {
            "lens_id": self.lens_id,
            "kind": self.kind,
            "model_key": self.model_key,
            "n_prompts": self.n_prompts,
            "skip_first": self.skip_first,
            "n_probes": self.n_probes,
            "seq_len": self.seq_len,
            "seed": self.seed,
            "corpus_id": self.corpus_id,
            "layers": list(self.layers) if self.layers else None,
        }


@dataclass
class LensBundle:
    """A fitted lens: one ``d x d`` matrix per layer, plus provenance."""

    spec: LensSpec
    jacobians: Dict[int, np.ndarray]
    sample: Optional[CorpusSample] = None
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def layers(self) -> List[int]:
        return sorted(self.jacobians)

    @property
    def d_model(self) -> int:
        first = self.jacobians[self.layers[0]]
        return int(first.shape[0])

    def matrix(self, layer: int) -> np.ndarray:
        if layer not in self.jacobians:
            raise KeyError(f"layer {layer} was not fitted; fitted layers: {self.layers}")
        return self.jacobians[layer]

    def digest(self) -> str:
        """Content hash of the fitted matrices, written into ``FREEZE-2.json``."""
        h = hashlib.sha256()
        h.update(self.spec.digest().encode("utf-8"))
        for layer in self.layers:
            h.update(str(layer).encode("utf-8"))
            h.update(np.ascontiguousarray(self.jacobians[layer], dtype=np.float32).tobytes())
        return h.hexdigest()

    def to_meta_dict(self) -> Dict[str, object]:
        return {
            "spec": self.spec.to_dict(),
            "layers": self.layers,
            "d_model": self.d_model,
            "sample_digest": self.sample.digest() if self.sample else None,
            "lens_digest": self.digest(),
            "meta": dict(self.meta),
        }


# --------------------------------------------------------------------------------------
# Corpus handling
# --------------------------------------------------------------------------------------


def disjoint_samples(
    prompt_ids: Sequence[str],
    *,
    n_prompts: int,
    seeds: Tuple[int, int] = (11, 23),
    corpus_id: str = "fit_corpus_v1",
) -> Tuple[CorpusSample, CorpusSample]:
    """Draw the two **non-overlapping** real-corpus samples required by §6.1."""
    if len(prompt_ids) < 2 * n_prompts:
        raise LensFitViolation(
            f"corpus has {len(prompt_ids)} prompts; two disjoint samples of {n_prompts} "
            f"need at least {2 * n_prompts}."
        )
    if seeds[0] == seeds[1]:
        raise LensFitViolation("the two real-corpus fits must use different seeds (§6.1)")
    rng = np.random.default_rng(seeds[0])
    order = rng.permutation(len(prompt_ids))
    first = tuple(prompt_ids[i] for i in order[:n_prompts])
    second = tuple(prompt_ids[i] for i in order[n_prompts : 2 * n_prompts])
    overlap = set(first) & set(second)
    if overlap:  # pragma: no cover - defensive
        raise LensFitViolation(f"samples overlap on {len(overlap)} prompts")
    return (
        CorpusSample(corpus_id, first, seeds[0]),
        CorpusSample(corpus_id, second, seeds[1]),
    )


def shuffle_within_prompt(token_ids: Sequence[int], seed: int) -> List[int]:
    """Permute token order **within** a prompt — the shuffled-corpus control (§6.1)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(list(token_ids))
    return [int(x) for x in rng.permutation(arr)]


def assert_corpora_disjoint(a: CorpusSample, b: CorpusSample) -> None:
    overlap = set(a.prompt_ids) & set(b.prompt_ids)
    if overlap:
        raise LensFitViolation(
            f"the two real-corpus fits share {len(overlap)} prompts; §6.1 requires disjoint "
            "sampling so that the lens level of the bootstrap is meaningful."
        )


# --------------------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------------------


def estimate_jacobian(
    jvp: Callable[[np.ndarray], np.ndarray],
    d_model: int,
    *,
    n_probes: int,
    seed: int,
    dtype=np.float32,
) -> np.ndarray:
    """Unbiased rank-``n_probes`` estimate of ``J`` from Jacobian-vector products.

    ``E[(J u) u^T] = J`` for isotropic Gaussian ``u``. ``jvp(u)`` must return ``J u``
    accumulated over ``t' >= t`` and over prompts, i.e. it already carries the expectation
    in the definition of ``J_l``.
    """
    if n_probes < 1:
        raise LensFitViolation("n_probes must be >= 1")
    rng = np.random.default_rng(seed)
    acc = np.zeros((d_model, d_model), dtype=np.float64)
    for _ in range(n_probes):
        u = rng.standard_normal(d_model)
        acc += np.outer(np.asarray(jvp(u), dtype=np.float64), u)
    return (acc / n_probes).astype(dtype)


def identity_lens(d_model: int, layers: Iterable[int], model_key: str) -> LensBundle:
    """The logit-lens baseline: ``J_l = I`` for every layer (§6.1)."""
    layers = tuple(sorted(set(int(l) for l in layers)))
    spec = LensSpec(
        lens_id="logit_lens",
        kind="logit",
        model_key=model_key,
        n_prompts=0,
        skip_first=0,
        n_probes=0,
        layers=layers,
    )
    eye = np.eye(d_model, dtype=np.float32)
    return LensBundle(
        spec=spec,
        jacobians={layer: eye.copy() for layer in layers},
        meta={
            "note": (
                "logit-lens baseline. The paper states it 'captures much of the "
                "workspace-like structure', so the real lens must clear it on the first "
                "half of the band (§6.1, Phase-B criterion 3)."
            )
        },
    )


def readout(
    bundle: LensBundle,
    layer: int,
    hidden: np.ndarray,
    unembed: np.ndarray,
    *,
    norm: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """``softmax(W_U . norm(J_l h_l))`` (§6.1). Returns a probability vector."""
    j = bundle.matrix(layer)
    projected = j @ np.asarray(hidden, dtype=j.dtype)
    if norm is not None:
        projected = norm(projected)
    logits = np.asarray(unembed, dtype=np.float64) @ projected.astype(np.float64)
    logits -= logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def rms_norm(
    x: np.ndarray, eps: float = 1e-6, *, weight: Optional[np.ndarray] = None
) -> np.ndarray:
    """Synthetic RMSNorm, optionally with learned gamma and an explicit epsilon.

    Real-model readout must use the model's own final norm and lm_head. In
    particular R1 and Qwen need not share epsilon (1e-5 versus 1e-6).
    """
    x = np.asarray(x)
    if x.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("RMSNorm expects float32 or float64")
    if not np.isfinite(x).all() or eps <= 0 or not np.isfinite(eps):
        raise ValueError("RMSNorm requires finite input and positive finite epsilon")
    normed = x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    if weight is not None:
        gamma = np.asarray(weight, dtype=x.dtype)
        if gamma.shape != (x.shape[-1],) or not np.isfinite(gamma).all():
            raise ValueError("RMSNorm weight must match the hidden dimension and be finite")
        normed = normed * gamma
    return normed


def top_k_directions(
    bundle: LensBundle,
    layer: int,
    hidden: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic SVD components; NOT qualified scientific token-J directions.

    Returns ``(directions, coefficients)`` where ``directions`` is ``k x d`` and
    orthonormal-by-construction (the right singular vectors of ``J_l``), and
    ``coefficients`` holds the signed projection of ``hidden`` onto each.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    j = np.asarray(bundle.matrix(layer), dtype=np.float64)
    # Right singular vectors span the input directions J actually reads.
    from wda.intervene.subspace import canonical_svd

    _, _, vt = canonical_svd(j)
    coeffs = vt @ np.asarray(hidden, dtype=np.float64)
    order = np.argsort(-np.abs(coeffs))[:k]
    return vt[order], coeffs[order]


__all__ = [
    "CorpusSample",
    "LENS_KINDS",
    "LensBundle",
    "LensFitViolation",
    "LensSpec",
    "MIN_PROMPTS",
    "MIN_SKIP_FIRST",
    "assert_corpora_disjoint",
    "disjoint_samples",
    "estimate_jacobian",
    "identity_lens",
    "readout",
    "rms_norm",
    "shuffle_within_prompt",
    "top_k_directions",
]
