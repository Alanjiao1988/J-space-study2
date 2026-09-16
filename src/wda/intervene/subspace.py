"""Rank-aware NumPy subspace utilities for synthetic/engineering fixtures only.

An SVD basis is not a validated implementation of the paper's token-J directions.
Sign fixing removes a numerical ambiguity; it does not resolve degenerate singular
spaces or establish scientific equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def floating_vector(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("expected a one-dimensional float32 or float64 vector")
    if not np.isfinite(array).all():
        raise ValueError("vector must be finite")
    return array


def canonical_svd(matrix: np.ndarray):
    """Orient each singular pair by its largest-magnitude right-vector entry."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("matrix must be finite and two-dimensional")
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    if len(s):
        signs = np.sign(vt[np.arange(len(s)), np.argmax(np.abs(vt), axis=1)])
        signs[signs == 0] = 1
        u = u * signs
        vt = vt * signs[:, None]
    return u, s, vt


@dataclass(frozen=True)
class SubspaceProjector:
    """Prepared row basis; discard numerical null directions instead of completing QR."""

    rows: np.ndarray

    @classmethod
    def prepare(cls, directions: np.ndarray, dimension: int, *, dtype=np.float32):
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("subspace precision must be float32 or float64")
        rows = np.asarray(directions, dtype=dtype)
        if rows.size == 0:
            if rows.shape not in ((0,), (0, dimension)):
                raise ValueError("empty basis has an incompatible dimension")
            return cls(np.empty((0, dimension), dtype=dtype))
        rows = np.atleast_2d(rows)
        if rows.ndim != 2 or rows.shape[1] != dimension or not np.isfinite(rows).all():
            raise ValueError("basis must be finite with one column per hidden dimension")
        _, singular, vt = canonical_svd(rows)
        tolerance = np.finfo(dtype).eps * max(rows.shape) * singular[0]
        basis = np.ascontiguousarray(vt[singular > tolerance], dtype=dtype)
        basis.setflags(write=False)
        return cls(basis)

    @property
    def rank(self) -> int:
        return self.rows.shape[0]

    def remove(self, hidden: np.ndarray) -> np.ndarray:
        hidden = floating_vector(hidden)
        if self.rows.shape[1] != hidden.size:
            raise ValueError("basis and hidden dimensions differ")
        if not self.rank:
            return hidden.copy()
        rows = self.rows.astype(hidden.dtype, copy=False)
        return hidden - rows.T @ (rows @ hidden)


def require_scientific_interventions() -> None:
    """Fail closed until the scientific direction/skip/swap definition is verified."""
    from wda.errors import ProtocolViolation

    raise ProtocolViolation(
        "not_qualified: scientific direction provider requires validation. Canonical "
        "top-k token-J directions versus SVD components is unresolved; the pinned "
        "upstream exposes fit/apply, not ablate/patch. Synthetic subspace utilities "
        "must not enter an experiment or stand in for upstream probe-swap."
    )
