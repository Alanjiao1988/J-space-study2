"""Numerical precision (fact F7, Phase-0 fixture ``noop_returns_zero``).

The predecessor measured a bf16 reduction-order offset of 0.11-0.48 logits — the same
order of magnitude as the effects under measurement — and concluded that *any* bf16
patching result without a demonstrated bit-exact no-op is uninterpretable.

Consequences enforced here:

* every intervention runs with fp32 accumulation;
* before a run's first scientific trial, and again on every resume, the no-op path must
  reproduce the clean logits **bit-exactly**;
* the per-trial ``noop_bitexact`` flag in ``log.jsonl`` (§15.4) is produced by
  :func:`bitexact` and is never inferred.

Everything here works on plain arrays so the fixture runs without a GPU; the torch-facing
helpers import torch lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from wda.errors import PrecisionViolation

#: The only accumulation dtype permitted for an intervention (fact F7).
REQUIRED_DTYPE = "float32"

#: Dtypes whose reduction order makes an ablation result uninterpretable.
FORBIDDEN_DTYPES = frozenset({"bfloat16", "float16"})


def bitexact(a: Any, b: Any) -> bool:
    """True iff the two arrays are **bit-for-bit** identical (not merely close)."""
    left = np.ascontiguousarray(_to_numpy(a))
    right = np.ascontiguousarray(_to_numpy(b))
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return left.tobytes() == right.tobytes()


def max_abs_deviation(a: Any, b: Any) -> float:
    left = _to_numpy(a).astype(np.float64)
    right = _to_numpy(b).astype(np.float64)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} vs {right.shape}")
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    detach = getattr(x, "detach", None)
    if detach is not None:  # torch.Tensor
        return detach().cpu().numpy()
    return np.asarray(x)


def dtype_name(x: Any) -> str:
    if isinstance(x, np.ndarray):
        return str(x.dtype)
    dt = getattr(x, "dtype", None)
    return str(dt).replace("torch.", "") if dt is not None else "unknown"


def require_fp32(x: Any, *, context: str = "intervention") -> None:
    """Raise unless ``x`` accumulates in fp32 (fact F7)."""
    name = dtype_name(x)
    if name in FORBIDDEN_DTYPES:
        raise PrecisionViolation(
            f"{context} received a {name} tensor. The predecessor measured a bf16 "
            "reduction-order offset of 0.11-0.48 logits, the same order as the effects "
            "under measurement (fact F7); ablation must run in fp32 or under a "
            "demonstrated bit-exact proof."
        )
    if name != REQUIRED_DTYPE:
        raise PrecisionViolation(
            f"{context} requires {REQUIRED_DTYPE} accumulation, got {name}."
        )


@dataclass
class NoOpReceipt:
    """Result of the ``noop_returns_zero`` conformance fixture (§10 Phase 0)."""

    bitexact: bool
    max_abs_deviation: float
    dtype: str
    context: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": "wda/noop_receipt/1",
            "fixture": "noop_returns_zero",
            "bitexact": self.bitexact,
            "max_abs_deviation": self.max_abs_deviation,
            "dtype": self.dtype,
            "context": self.context,
        }

    def require(self) -> "NoOpReceipt":
        if not self.bitexact:
            raise PrecisionViolation(
                f"the no-op path is not bit-exact (max |deviation| = "
                f"{self.max_abs_deviation:.6g}, dtype {self.dtype}). Fixture "
                "'noop_returns_zero' fails; the run is refused (§10, SC0/SC4)."
            )
        return self


def check_noop(clean_logits: Any, noop_logits: Any, *, context: str = "ablation k=0") -> NoOpReceipt:
    """Phase-0 fixture ``noop_returns_zero``: ``k = 0`` must reproduce the clean logits."""
    require_fp32(clean_logits, context=f"{context} (clean)")
    require_fp32(noop_logits, context=f"{context} (no-op)")
    return NoOpReceipt(
        bitexact=bitexact(clean_logits, noop_logits),
        max_abs_deviation=max_abs_deviation(clean_logits, noop_logits),
        dtype=dtype_name(clean_logits),
        context=context,
    )


def force_deterministic_torch() -> Dict[str, object]:
    """Set the torch flags that make reductions reproducible. No-op without torch."""
    try:  # pragma: no cover - exercised only on a machine with torch
        import os

        import torch
    except ImportError:
        return {"torch": False}
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "torch": True,
        "version": torch.__version__,
        "deterministic_algorithms": True,
        "allow_tf32": False,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


__all__ = [
    "FORBIDDEN_DTYPES",
    "NoOpReceipt",
    "REQUIRED_DTYPE",
    "bitexact",
    "check_noop",
    "dtype_name",
    "force_deterministic_torch",
    "max_abs_deviation",
    "require_fp32",
]
