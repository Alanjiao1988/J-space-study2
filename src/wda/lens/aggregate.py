"""Dual-lens aggregation (§6.2).

Both real-corpus lenses are **always** run. The main estimate is their paired average, and
the lens fit is a level of the hierarchical bootstrap (§9.5).

Post-hoc selection of "the stronger set" is forbidden and is enforced, not merely
documented: :func:`select_stronger` exists only to raise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from wda.errors import AggregationViolation
from wda.lens.fit import LensBundle, assert_corpora_disjoint


@dataclass
class PairedLenses:
    """The two independent real-corpus fits for one subject."""

    lens_a: LensBundle
    lens_b: LensBundle

    def __post_init__(self) -> None:
        for name, bundle in (("lens_a", self.lens_a), ("lens_b", self.lens_b)):
            if bundle.spec.kind != "real":
                raise AggregationViolation(
                    f"{name} has kind {bundle.spec.kind!r}; §6.2 pairs the two *real-corpus* fits"
                )
        if self.lens_a.spec.model_key != self.lens_b.spec.model_key:
            raise AggregationViolation("the paired lenses were fitted on different models")
        if self.lens_a.spec.seed == self.lens_b.spec.seed:
            raise AggregationViolation("the two real-corpus fits must use different seeds (§6.1)")
        if self.lens_a.sample is not None and self.lens_b.sample is not None:
            assert_corpora_disjoint(self.lens_a.sample, self.lens_b.sample)
        if self.lens_a.layers != self.lens_b.layers:
            raise AggregationViolation("the paired lenses were fitted on different layer sets")

    @property
    def lens_ids(self) -> Tuple[str, str]:
        return (self.lens_a.spec.lens_id, self.lens_b.spec.lens_id)

    def mean_jacobian(self, layer: int) -> np.ndarray:
        """Paired average of the two Jacobians at ``layer`` (§6.2)."""
        return 0.5 * (
            np.asarray(self.lens_a.matrix(layer), dtype=np.float64)
            + np.asarray(self.lens_b.matrix(layer), dtype=np.float64)
        )

    def paired_mean(self, per_lens: Mapping[str, float]) -> float:
        """Paired average of a scalar computed separately on each lens."""
        missing = set(self.lens_ids) - set(per_lens)
        if missing:
            raise AggregationViolation(
                f"paired_mean is missing values for {sorted(missing)}; §6.2 requires both "
                "lenses to be run — a missing lens may not be dropped."
            )
        a, b = self.lens_ids
        return 0.5 * (float(per_lens[a]) + float(per_lens[b]))

    def agreement(self, layer: int) -> float:
        """Cosine similarity of the two fits at ``layer`` — a fit-stability diagnostic."""
        a = np.asarray(self.lens_a.matrix(layer), dtype=np.float64).ravel()
        b = np.asarray(self.lens_b.matrix(layer), dtype=np.float64).ravel()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return float("nan")
        return float(np.dot(a, b) / denom)

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": "wda/paired_lenses/1",
            "lens_ids": list(self.lens_ids),
            "model_key": self.lens_a.spec.model_key,
            "layers": self.lens_a.layers,
            "digests": {
                self.lens_a.spec.lens_id: self.lens_a.digest(),
                self.lens_b.spec.lens_id: self.lens_b.digest(),
            },
            "aggregation": "paired mean of both real-corpus fits (§6.2)",
            "post_hoc_selection": "forbidden",
        }


def select_stronger(*_args, **_kwargs):
    """Always raises. §6.2 forbids post-hoc selection between the two real lenses."""
    raise AggregationViolation(
        "post-hoc selection of the stronger real-corpus lens is forbidden (§6.2). Both "
        "lenses are always run; the main estimate is their paired average and the lens fit "
        "is a level of the hierarchical bootstrap (§9.5)."
    )


def bootstrap_lens_ids(paired: PairedLenses) -> Tuple[str, str]:
    """The ``lens_id`` values that must appear in ``log.jsonl`` for the lens level."""
    return paired.lens_ids


__all__ = ["PairedLenses", "bootstrap_lens_ids", "select_stronger"]
