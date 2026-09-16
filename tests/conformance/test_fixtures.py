"""Synthetic tests for Phase-0 fixture mechanics, not model-facing qualification.

Five fixtures, all of which must pass before Phase A. Before the freeze a failure may be
repaired and the fixtures re-run; after Freeze-2 any failure voids the assay (SC4).

    1. noop_returns_zero
    2. projection_removed
    3. random_control_norm_matched
    4. lens_readout_matches_reference
    5. known_intermediate_positive
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from wda.errors import PrecisionViolation
from wda.intervene import controls, precision
from wda.intervene.ablate import (
    AblationSpec,
    Strength,
    ablate_at,
    project_out,
    select_ablation_directions,
)
from wda.intervene.patch import LensBasis, SwapOutcome, swap_lens_coordinates
from wda.lens.fit import readout, rms_norm
from wda.paths import configs_dir


@pytest.fixture(scope="module")
def fixture_config() -> dict:
    with (configs_dir() / "freeze1" / "fixtures.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)["fixtures"]


# ======================================================================== fixture 1 ===


class TestNoopReturnsZero:
    """``k = 0`` (or strength 0) must reproduce the clean logits **bit-exactly** in fp32."""

    def test_zero_k_selects_nothing(self, synthetic_bundle, unembed, hidden):
        dirs, receipt = select_ablation_directions(
            synthetic_bundle, 4, hidden, unembed, k=0, clean_top_tokens=[]
        )
        assert dirs.shape[0] == 0
        assert receipt["noop"] is True

    def test_noop_is_bit_exact(self, synthetic_bundle, unembed, hidden):
        spec = AblationSpec(band=(4, 5, 6, 7), k=0)
        ablated, _ = ablate_at(synthetic_bundle, 4, 0, hidden, unembed, spec, [])
        clean_logits = (unembed @ rms_norm(hidden)).astype(np.float32)
        noop_logits = (unembed @ rms_norm(ablated)).astype(np.float32)
        receipt = precision.check_noop(clean_logits, noop_logits)
        assert receipt.require().bitexact is True
        assert receipt.max_abs_deviation == 0.0

    def test_bf16_is_refused(self, fixture_config):
        assert fixture_config["noop_returns_zero"]["tolerance"] == 0

        class _FakeBf16:
            dtype = "torch.bfloat16"

        with pytest.raises(PrecisionViolation, match="bf16 reduction-order|bfloat16"):
            precision.require_fp32(_FakeBf16())

    def test_nonbitexact_noop_raises(self):
        clean = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        drifted = clean.copy()
        drifted[0] = np.float32(1.0000001)
        receipt = precision.check_noop(clean, drifted)
        assert receipt.bitexact is False
        with pytest.raises(PrecisionViolation, match="not bit-exact"):
            receipt.require()


# ======================================================================== fixture 2 ===


class TestProjectionRemoved:
    """After ablation the residual projection onto every ablated direction is ~0."""

    def test_projection_is_zero(self, synthetic_bundle, unembed, hidden, fixture_config):
        tol = fixture_config["projection_removed"]["tolerance"]
        spec = AblationSpec(band=(4, 5, 6, 7), k=5)
        ablated, receipt = ablate_at(synthetic_bundle, 5, 0, hidden, unembed, spec, [])
        assert receipt.selected == 5
        assert receipt.residual_projection < tol

    def test_projection_is_idempotent(self, synthetic_bundle, unembed, hidden):
        dirs, _ = select_ablation_directions(
            synthetic_bundle, 5, hidden, unembed, k=4, clean_top_tokens=[]
        )
        once = project_out(hidden, dirs)
        twice = project_out(once, dirs)
        assert np.allclose(once, twice, atol=1e-12)

    def test_skip_rule_excludes_clean_top_tokens(self, synthetic_bundle, unembed, hidden):
        """§5 — a direction whose readout token is in the clean top-10 is never ablated."""
        from wda.intervene.ablate import direction_readout_tokens

        j = synthetic_bundle.matrix(6)
        _, _, vt = np.linalg.svd(j, full_matrices=False)
        coeffs = vt @ hidden
        order = np.argsort(-np.abs(coeffs))
        tokens = direction_readout_tokens(synthetic_bundle, 6, vt[order], unembed)
        banned = int(tokens[0])

        dirs, receipt = select_ablation_directions(
            synthetic_bundle, 6, hidden, unembed, k=3, clean_top_tokens=[banned]
        )
        assert receipt["selected"] == 3
        assert any(s["readout_token"] == banned for s in receipt["skipped_for_clean_top"])
        chosen_tokens = direction_readout_tokens(synthetic_bundle, 6, dirs, unembed)
        assert banned not in set(int(t) for t in chosen_tokens)


# ======================================================================== fixture 3 ===


class TestRandomControlNormMatched:
    """The random-direction control removes the same vector norm as the ablation."""

    def test_matched_norm_perturbation_matches(self, synthetic_bundle, unembed, hidden, fixture_config):
        tol = fixture_config["random_control_norm_matched"]["tolerance"]
        dirs, _ = select_ablation_directions(
            synthetic_bundle, 4, hidden, unembed, k=6, clean_top_tokens=[]
        )
        ablated = project_out(hidden, dirs)
        removed = float(np.linalg.norm(hidden - ablated))
        perturbed = controls.matched_norm_perturbation(hidden, removed, seed=0)
        gap = controls.assert_norm_matched(hidden, ablated, hidden, perturbed, tolerance=tol)
        assert gap <= tol

    def test_layer_matched_random_removes_k_dimensions(self, hidden, d_model):
        ablated, dirs = controls.layer_matched_random_ablation(hidden, 6, seed=3)
        assert dirs.shape == (6, d_model)
        assert np.allclose(dirs @ ablated, 0.0, atol=1e-10)

    def test_mismatched_norm_raises(self, hidden):
        with pytest.raises(controls.ControlViolation, match="norm mismatch"):
            controls.assert_norm_matched(hidden, hidden * 0.5, hidden, hidden * 0.9)

    def test_clamp_restores_the_complement(self, synthetic_bundle, unembed, hidden):
        dirs, _ = select_ablation_directions(
            synthetic_bundle, 4, hidden, unembed, k=5, clean_top_tokens=[]
        )
        ablated = project_out(hidden, dirs)
        clamped = controls.clamp_complement(ablated, hidden, dirs)
        q, _ = np.linalg.qr(dirs.T)
        # Inside the subspace the clamp keeps the ablated (zero) component ...
        assert np.allclose(q.T @ clamped, 0.0, atol=1e-10)
        # ... and outside it restores the clean value exactly.
        assert np.allclose(clamped - q @ (q.T @ clamped), hidden - q @ (q.T @ hidden), atol=1e-10)


# ======================================================================== fixture 4 ===


class TestLensReadoutMatchesReference:
    """Local fit vs the public community lens: Jaccard over top-25 must clear the bar.

    The real model-facing comparison remains unqualified. These tests cover the
    metric and the explicit refusal of unsupported execution, not a passed Phase 0.
    """

    @staticmethod
    def jaccard_topk(p: np.ndarray, q: np.ndarray, k: int) -> float:
        a = set(np.argsort(-p)[:k].tolist())
        b = set(np.argsort(-q)[:k].tolist())
        return len(a & b) / len(a | b)

    def test_identical_readouts_have_jaccard_one(self, synthetic_bundle, unembed, hidden):
        probs = readout(synthetic_bundle, 4, hidden, unembed, norm=rms_norm)
        assert self.jaccard_topk(probs, probs, 25) == 1.0

    def test_threshold_is_registered(self, fixture_config):
        spec = fixture_config["lens_readout_matches_reference"]
        assert spec["metric"] == "Jaccard over top-25 tokens"
        assert 0.0 < spec["threshold"] < 1.0
        assert spec["reference_lens"]

    def test_disjoint_readouts_fail_the_bar(self, fixture_config):
        threshold = fixture_config["lens_readout_matches_reference"]["threshold"]
        probs = np.arange(100, dtype=float)
        reversed_probs = probs[::-1]
        assert self.jaccard_topk(probs, reversed_probs, 25) == 0.0
        assert self.jaccard_topk(probs, reversed_probs, 25) < threshold

    def test_missing_reference_interface_is_explicit(self):
        from wda.lens.upstream import QualificationPending, reference_readout_fixture
        with pytest.raises(QualificationPending, match="not_qualified"):
            reference_readout_fixture()


# ======================================================================== fixture 5 ===


class TestKnownIntermediatePositive:
    """An intermediate swap must move the target logit where a matched-norm null does not.

    Fact F8: the predecessor project never obtained a passing positive control on the
    J-lens line, which is why its negative results were strong about its own execution and
    weak about the method. This fixture is the guard against repeating that.
    """

    def test_synthetic_donor_has_larger_effect_than_synthetic_null(
        self, synthetic_bundle, unembed, rng, d_model
    ):
        basis = LensBasis.from_bundle(synthetic_bundle, 4)
        components = basis.top_components(6)
        target_token = 7
        jacobian = synthetic_bundle.matrix(4)

        recipient = rng.standard_normal(d_model)
        # A donor the lens maps onto the target token: solve J h = u_target, then scale it
        # so its coordinates dominate the six swapped components.
        raw = np.linalg.lstsq(jacobian, unembed[target_token], rcond=None)[0]
        donor = raw / np.linalg.norm(raw) * np.linalg.norm(recipient) * 10.0

        def target_logit(h: np.ndarray) -> float:
            return float(unembed[target_token] @ rms_norm(jacobian @ h))

        swapped = swap_lens_coordinates(recipient, donor, basis, components)
        delta_swap = target_logit(swapped) - target_logit(recipient)

        # The null is a *distribution*, not a single draw: 64 matched-norm perturbations.
        removed = float(np.linalg.norm(swapped - recipient))
        null_deltas = np.array(
            [
                target_logit(controls.matched_norm_perturbation(recipient, removed, seed=s))
                - target_logit(recipient)
                for s in range(64)
            ]
        )

        outcome = SwapOutcome(
            target_logit_delta=delta_swap,
            matched_norm_random_delta=float(null_deltas.mean()),
            direct_substitution_delta=0.0,
        )
        assert delta_swap > 0.0
        assert delta_swap > float(np.quantile(null_deltas, 0.95))
        assert outcome.to_dict()["target_logit_delta"] == delta_swap

    def test_positive_predicate_requires_nonpositive_random_delta(self):
        assert SwapOutcome(1.0, 0.0).positive_control_holds()
        assert not SwapOutcome(1.0, 0.01).positive_control_holds()
        assert not SwapOutcome(0.0, -1.0).positive_control_holds()

    def test_swap_leaves_the_complement_untouched(self, synthetic_bundle, rng, d_model):
        basis = LensBasis.from_bundle(synthetic_bundle, 5)
        components = basis.top_components(4)
        recipient = rng.standard_normal(d_model)
        donor = rng.standard_normal(d_model)
        swapped = swap_lens_coordinates(recipient, donor, basis, components)

        c_before = basis.coordinates(recipient)
        c_after = basis.coordinates(swapped)
        untouched = [i for i in range(basis.rank) if i not in set(components.tolist())]
        assert np.allclose(c_after[untouched], c_before[untouched], atol=1e-10)

    def test_min_items_is_registered(self, fixture_config):
        spec = fixture_config["known_intermediate_positive"]
        assert spec["min_items"] >= 20
        assert "probe-swap.json" in spec["source"]

    def test_missing_positive_control_interface_is_explicit(self):
        from wda.lens.upstream import QualificationPending, known_intermediate_fixture
        with pytest.raises(QualificationPending, match="not_qualified"):
            known_intermediate_fixture()


# ===================================================================== exit criteria ===


def test_all_five_fixtures_are_registered(fixture_config):
    """§10 — the Phase-0 exit gate is exactly these five, no more and no fewer."""
    assert set(fixture_config) == {
        "noop_returns_zero",
        "projection_removed",
        "random_control_norm_matched",
        "lens_readout_matches_reference",
        "known_intermediate_positive",
    }
