"""Shared fixtures for the test suite.

Nothing here loads a model. The Phase-0 conformance fixtures that *require* a checkpoint
(``lens_readout_matches_reference``, ``known_intermediate_positive``) are implemented
against synthetic tensors here and marked ``needs_model`` in their model-facing form, so the
suite runs before Freeze-1 on a machine with no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from wda.lens.fit import LensBundle, LensSpec
from wda.stats.bootstrap import ABLATED, CLEAN, C_DIRECT, C_FROZEN, TrialFrame


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def d_model() -> int:
    return 24


@pytest.fixture
def vocab() -> int:
    return 40


@pytest.fixture
def synthetic_bundle(rng: np.random.Generator, d_model: int) -> LensBundle:
    """A small, well-conditioned stand-in for a fitted lens."""
    spec = LensSpec(
        lens_id="real_a",
        kind="real",
        model_key="calib_qwen25_7b_instruct",
        n_prompts=256,
        skip_first=16,
        n_probes=128,
        seed=11,
        layers=(4, 5, 6, 7),
    )
    jacobians = {}
    for layer in spec.layers:
        base = rng.standard_normal((d_model, d_model))
        # Make the spectrum decay so the "top-k directions" ordering is meaningful.
        u, _, vt = np.linalg.svd(base)
        s = np.linspace(3.0, 0.2, d_model)
        jacobians[layer] = (u * s) @ vt
    return LensBundle(spec=spec, jacobians=jacobians)


@pytest.fixture
def unembed(rng: np.random.Generator, vocab: int, d_model: int) -> np.ndarray:
    return rng.standard_normal((vocab, d_model))


@pytest.fixture
def hidden(rng: np.random.Generator, d_model: int) -> np.ndarray:
    return rng.standard_normal(d_model)


def build_frame(
    *,
    n_items: int = 60,
    lenses=("real_a", "real_b"),
    p_treat=(0.90, 0.40, 0.90, 0.80),
    p_comp=(0.90, 0.55, 0.90, 0.85),
    seed: int = 0,
) -> TrialFrame:
    """Synthetic primary design with known cell probabilities.

    Cell order is ``(direct_clean, direct_ablated, frozen_clean, frozen_ablated)``. Every
    item appears in all four cells of both models, mirroring the paired structure §9.3
    assumes.
    """
    rng = np.random.default_rng(seed)
    records = []
    cells = (
        (C_DIRECT, CLEAN, 0),
        (C_DIRECT, ABLATED, 1),
        (C_FROZEN, CLEAN, 2),
        (C_FROZEN, ABLATED, 3),
    )
    for lens in lenses:
        for i in range(n_items):
            item = f"item-{i:04d}"
            for role, probs in (("treatment", p_treat), ("comparator", p_comp)):
                for condition, state, idx in cells:
                    records.append(
                        {
                            "lens_id": lens,
                            "item_id": item,
                            "template_id": "arith_chain_v1",
                            "model_role": role,
                            "condition": condition,
                            "ablation_state": state,
                            "parsed": True,
                            "correct": bool(rng.random() < probs[idx]),
                        }
                    )
    return TrialFrame.from_records(records)


@pytest.fixture
def make_frame():
    return build_frame


@pytest.fixture
def synthetic_frame() -> TrialFrame:
    return build_frame(n_items=8, seed=42)
