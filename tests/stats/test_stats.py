"""Statistics tests on synthetic data: decision rule (§9.2), bootstrap (§9.5), power (§9.3)."""

from __future__ import annotations

import numpy as np
import pytest

from wda.stats import power as pw
from wda.stats.bootstrap import (
    ABLATED,
    CLEAN,
    C_DIRECT,
    C_FROZEN,
    BootstrapViolation,
    TrialFrame,
    g_primary,
    hierarchical_bootstrap,
    primary_endpoint,
    trial_bootstrap,
)
from wda.stats.decision import (
    Interval,
    Outcome,
    assert_permitted_phrasing,
    check_phrasing,
    decide,
    holm,
)

DELTA = 0.10


# ============================================================================ decision ===


class TestDecisionRule:
    def test_material_positive(self):
        d = decide(0.25, Interval(0.15, 0.35, 0.95), Interval(0.17, 0.33, 0.90), DELTA)
        assert d.outcome is Outcome.MATERIAL
        assert d.direction == "positive"
        assert d.stopping_condition is None

    def test_material_negative(self):
        d = decide(-0.25, Interval(-0.35, -0.15, 0.95), Interval(-0.33, -0.17, 0.90), DELTA)
        assert d.outcome is Outcome.MATERIAL
        assert d.direction == "negative"

    def test_equivalent_maps_to_sc2(self):
        d = decide(0.01, Interval(-0.11, 0.13, 0.95), Interval(-0.08, 0.09, 0.90), DELTA)
        assert d.outcome is Outcome.EQUIVALENT
        assert d.stopping_condition == "SC2"

    def test_inconclusive_maps_to_sc3(self):
        d = decide(0.06, Interval(-0.05, 0.17, 0.95), Interval(-0.02, 0.14, 0.90), DELTA)
        assert d.outcome is Outcome.INCONCLUSIVE
        assert d.stopping_condition == "SC3"
        assert "do not add samples" in d.rationale

    def test_ci_exactly_at_delta_is_not_material(self):
        """The boundary is strict: 'entirely above +delta' excludes touching it."""
        d = decide(0.20, Interval(0.10, 0.30, 0.95), Interval(0.12, 0.28, 0.90), DELTA)
        assert d.outcome is Outcome.INCONCLUSIVE

    def test_ci_exactly_inside_delta_is_equivalent(self):
        d = decide(0.0, Interval(-0.12, 0.12, 0.95), Interval(-0.10, 0.10, 0.90), DELTA)
        assert d.outcome is Outcome.EQUIVALENT

    def test_wrong_ci_levels_are_refused(self):
        with pytest.raises(ValueError, match="95% CI"):
            decide(0.2, Interval(0.1, 0.3, 0.90), Interval(0.12, 0.28, 0.90), DELTA)
        with pytest.raises(ValueError, match="90% CI"):
            decide(0.2, Interval(0.1, 0.3, 0.95), Interval(0.12, 0.28, 0.99), DELTA)

    def test_non_nested_intervals_are_refused(self):
        with pytest.raises(ValueError, match="not nested"):
            decide(0.2, Interval(0.15, 0.25, 0.95), Interval(0.10, 0.30, 0.90), DELTA)

    def test_delta_must_be_positive(self):
        with pytest.raises(ValueError, match="delta must be positive"):
            decide(0.2, Interval(0.1, 0.3, 0.95), Interval(0.12, 0.28, 0.90), 0.0)

    def test_sentence_uses_only_permitted_phrasing(self):
        d = decide(0.25, Interval(0.15, 0.35, 0.95), Interval(0.17, 0.33, 0.90), DELTA)
        assert_permitted_phrasing(d.sentence())
        assert "judged material difference" in d.sentence()

    @pytest.mark.parametrize(
        "text",
        [
            "this is a reasoning-specific effect",
            "the scale trend is clear",
            "distillation caused the difference",
            "externalization confirmed",
            "J-space exists in open models",
            "first reproduction of the result",
        ],
    )
    def test_forbidden_phrasing_is_detected(self, text):
        assert check_phrasing(text)
        with pytest.raises(ValueError, match="forbidden phrasing"):
            assert_permitted_phrasing(text)


class TestHolm:
    def test_step_down_order_and_rejection(self):
        out = holm({"a": 0.001, "b": 0.04, "c": 0.30}, alpha=0.05)
        assert out["a"]["rank"] == 1 and out["a"]["reject"] is True
        assert out["b"]["threshold"] == pytest.approx(0.025)
        assert out["b"]["reject"] is False   # 0.04 > 0.05/2
        assert out["c"]["reject"] is False   # step-down stops after the first failure

    def test_adjusted_pvalues_are_monotone(self):
        out = holm({"a": 0.01, "b": 0.02, "c": 0.03})
        adjusted = [out[k]["p_holm"] for k in ("a", "b", "c")]
        assert adjusted == sorted(adjusted)

    def test_out_of_range_pvalue_is_refused(self):
        with pytest.raises(ValueError, match="out of range"):
            holm({"a": 1.5})


# =========================================================================== bootstrap ===


class TestTrialFrame:
    def test_itt_rejects_inconsistent_verdicts_without_rewriting_them(self):
        with pytest.raises(ValueError, match="correct but not parseable"):
            TrialFrame.from_records(
                [
                    {
                        "lens_id": "real_a",
                        "item_id": "i1",
                        "template_id": "t",
                        "model_role": "treatment",
                        "condition": C_DIRECT,
                        "ablation_state": CLEAN,
                        "parsed": False,
                        "correct": True,
                    }
                ]
            )

    def test_estimands_recover_the_generating_probabilities(self, make_frame):
        frame = make_frame(n_items=4000, seed=7)
        # G = (0.90-0.40) - (0.90-0.80) = 0.40 for the treatment
        assert g_primary(frame, "treatment") == pytest.approx(0.40, abs=0.03)
        # G = (0.90-0.55) - (0.90-0.85) = 0.30 for the comparator
        assert g_primary(frame, "comparator") == pytest.approx(0.30, abs=0.03)
        assert primary_endpoint(frame) == pytest.approx(0.10, abs=0.04)


class TestHierarchicalBootstrap:
    def test_flat_trial_bootstrap_is_forbidden(self):
        with pytest.raises(BootstrapViolation, match="flat trial-level bootstrap is forbidden"):
            trial_bootstrap()

    def test_lens_must_be_the_outermost_level(self, make_frame):
        frame = make_frame(n_items=10)
        with pytest.raises(BootstrapViolation, match="outermost bootstrap level"):
            hierarchical_bootstrap(frame, n_resamples=5, levels=("item_id", "lens_id"))

    def test_intervals_are_nested_and_cover_the_estimate(self, make_frame):
        frame = make_frame(n_items=80, seed=3)
        result = hierarchical_bootstrap(frame, n_resamples=300, seed=1)
        ci95, ci90 = result.interval(0.95), result.interval(0.90)
        assert ci95.low <= ci90.low <= ci90.high <= ci95.high
        assert ci95.low <= result.estimate <= ci95.high

    def test_result_feeds_the_decision_rule(self, make_frame):
        frame = make_frame(n_items=200, p_treat=(0.95, 0.20, 0.95, 0.92),
                           p_comp=(0.95, 0.90, 0.95, 0.92), seed=11)
        result = hierarchical_bootstrap(frame, n_resamples=400, seed=2)
        d = decide(result.estimate, result.interval(0.95), result.interval(0.90), DELTA)
        assert d.outcome is Outcome.MATERIAL

    def test_narrow_effect_yields_equivalence(self, make_frame):
        frame = make_frame(n_items=400, p_treat=(0.90, 0.50, 0.90, 0.70),
                           p_comp=(0.90, 0.50, 0.90, 0.70), seed=5)
        result = hierarchical_bootstrap(frame, n_resamples=400, seed=4)
        d = decide(result.estimate, result.interval(0.95), result.interval(0.90), DELTA)
        assert d.outcome in (Outcome.EQUIVALENT, Outcome.INCONCLUSIVE)

    def test_resampling_preserves_item_pairing(self, make_frame):
        """All four cells of a drawn item must travel together (§9.3)."""
        from wda.stats.bootstrap import LEVELS, _resample_indices

        frame = make_frame(n_items=12, seed=0)
        index = _resample_indices(frame, np.random.default_rng(0), LEVELS)
        drawn = frame.take(index)
        for item in set(drawn.item_id.tolist()):
            mask = drawn.item_id == item
            # 2 models x 2 conditions x 2 states = 8 rows per item copy.
            assert mask.sum() % 8 == 0


# =============================================================================== power ===


class TestPower:
    def test_rough_unpaired_bound_reproduces_the_protocol_number(self):
        """§9.3's 'roughly n ~ 800 items per cell' must be derivable, not asserted."""
        bound = rb = pw.rough_unpaired_bound(0.10)
        assert bound["unit_variance_G"] == pytest.approx(1.0)
        assert bound["unit_variance_D"] == pytest.approx(2.0)
        assert 780 <= bound["n_per_model_for_G"] <= 800
        # Treating the two checkpoints as independent doubles it; the protocol's number is
        # the G-level bound and the honest D-level requirement is larger.
        assert rb["n_per_model_for_D"] == pytest.approx(2 * rb["n_per_model_for_G"], rel=0.01)

    def test_pairing_reduces_the_required_n(self):
        p = pw.CellProbabilities(0.90, 0.50, 0.90, 0.80)
        unpaired = pw.contrast_variance(pw.covariance_from_correlation(p, 0.0))
        paired = pw.contrast_variance(pw.covariance_from_correlation(p, 0.5))
        assert paired < unpaired
        assert pw.required_n_conventional(0.10, paired) < pw.required_n_conventional(0.10, unpaired)

    def test_power_is_monotone_in_n(self):
        var = 2.0
        powers = [pw.power_conventional(n, 0.10, var) for n in (200, 400, 800, 1600)]
        assert powers == sorted(powers)

    def test_material_rule_is_stricter_than_the_conventional_test(self):
        var = 2.0
        n = 2000
        assert pw.power_material(n, 0.10, 0.10, var) < pw.power_conventional(n, 0.10, var)

    def test_no_finite_n_detects_an_effect_equal_to_delta(self):
        assert pw.required_n_material(0.10, 0.10, 2.0) is None
        assert pw.required_n_material(0.25, 0.10, 2.0) is not None

    def test_equivalence_power_peaks_at_zero_effect(self):
        var = 2.0
        at_zero = pw.power_equivalence(4000, 0.0, 0.10, var)
        off_centre = pw.power_equivalence(4000, 0.08, 0.10, var)
        assert at_zero > off_centre

    def test_plan_is_serialisable_and_reports_the_operative_n(self):
        p = pw.CellProbabilities(0.90, 0.50, 0.90, 0.80)
        cov = pw.covariance_from_correlation(p, 0.4)
        plan = pw.plan(
            cov, cov, delta=0.10, d_true=0.20,
            basis="material", cross_model_assumption="independent",
        )
        assert plan["n_operative"] >= plan["n_conventional"]
        assert plan["achieved_power_conventional"] >= 0.80

    def test_simulation_matches_the_analytic_formula(self):
        treat = pw.CellProbabilities(0.90, 0.30, 0.90, 0.85)
        comp = pw.CellProbabilities(0.90, 0.80, 0.90, 0.85)
        out = pw.simulate_power(treat, comp, 400, delta=0.10, rho=0.3, n_sims=120, seed=1)
        assert out["d_true"] == pytest.approx(0.50, abs=1e-9)
        assert abs(out["empirical_power_material"] - out["analytic_power_material"]) < 0.15
