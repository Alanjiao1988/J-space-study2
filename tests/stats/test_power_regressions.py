"""Analytically checkable synthetic power and covariance cases; no model calls."""

import json
import math

import numpy as np
import pytest
from scipy.stats import norm

from wda.governance.blind import phase_a_blinding
from wda.stats import power as pw


COV = np.eye(4) * 0.25


def test_independent_and_correlated_model_variances_have_known_values():
    assert pw.contrast_variance(COV) == 1
    assert pw.unit_variance_of_d(COV, COV) == 2
    assert pw.unit_variance_of_d(COV, COV, cross_model_cov=COV / 2) == 1
    # Shared-bank cross-covariance need not reduce contrast variance.
    assert pw.unit_variance_of_d(COV, COV, cross_model_cov=-COV / 2) == 3
    assert pw.unit_variance_of_d(COV, COV, cross_model_cov=COV) == 0


def test_valid_nonsymmetric_cross_block_is_allowed_but_joint_is_symmetric():
    cross = np.zeros((4, 4))
    cross[0, 1] = 0.05
    assert pw.unit_variance_of_d(COV, COV, cross_model_cov=cross) == pytest.approx(2.1)


@pytest.mark.parametrize(
    "cov",
    [
        np.eye(3), np.ones(4), np.full((4, 4), np.nan), np.full((4, 4), np.inf),
        np.eye(4) * -1, np.array([[0.25, 0.3, 0, 0], [0.3, 0.25, 0, 0],
                                 [0, 0, 0.25, 0], [0, 0, 0, 0.25]]),
        np.array([[0.25, 0.1, 0, 0], [0, 0.25, 0, 0], [0, 0, 0.25, 0], [0, 0, 0, 0.25]]),
        np.full((4, 4), "0.25"), np.eye(4, dtype=bool),
    ],
)
def test_invalid_covariances_are_rejected_everywhere(cov):
    with pytest.raises(ValueError):
        pw.contrast_variance(cov)
    with pytest.raises(ValueError):
        pw.unit_variance_of_d(cov, COV)
    with pytest.raises(ValueError):
        pw.unit_variance_of_d(COV, cov)
    with pytest.raises(ValueError):
        pw.plan(cov, COV)


@pytest.mark.parametrize("contrast", [np.ones(3), np.ones((4, 1)), [1, -1, np.nan, 1],
                                    [1, -1, np.inf, 1], ["1", "-1", "-1", "1"]])
def test_invalid_contrast_arrays(contrast):
    with pytest.raises(ValueError):
        pw.contrast_variance(COV, contrast)


@pytest.mark.parametrize(
    "cross",
    [
        np.eye(3), np.full((4, 4), np.nan), np.full((4, 4), np.inf),
        np.eye(4) * 0.5,
        # Indefinite in a direction orthogonal to c: D variance alone would miss it.
        np.ones((4, 4)) * 0.125,
    ],
)
def test_invalid_joint_covariance_is_rejected_not_clamped(cross):
    with pytest.raises(ValueError, match="cross"):
        pw.unit_variance_of_d(COV, COV, cross_model_cov=cross)
    with pytest.raises(ValueError):
        pw.plan(COV, COV, cross_model_cov=cross)


def test_transparent_normal_power_cases():
    z95 = norm.ppf(0.975)
    # n=100, variance=1 implies SE=.1, effect=.2 implies standardized shift=2.
    conventional = norm.sf(z95 - 2) + norm.cdf(-z95 - 2)
    material = norm.sf(z95 - 1) + norm.cdf(-z95 - 3)
    assert pw.power_conventional(100, 0.2, 1) == pytest.approx(conventional)
    assert pw.power_material(100, 0.2, 0.1, 1) == pytest.approx(material)
    assert pw.power_material(100, -0.2, 0.1, 1) == pytest.approx(material)
    assert conventional > material
    assert pw.power_conventional(100, 0, 1) == pytest.approx(0.05)
    # n=400, variance=1 => SE=.05, centered 90% CI inside +/- .1.
    z90 = norm.ppf(0.95)
    assert pw.power_equivalence(400, 0, 0.1, 1) == pytest.approx(2 * norm.cdf(2 - z90) - 1)


def test_material_boundary_is_about_point_zero_two_five_not_zero():
    assert pw.power_material(2000, 0.1, 0.1, 2) == pytest.approx(0.025, abs=1e-12)
    assert pw.power_material(2000, -0.1, 0.1, 2) == pytest.approx(0.025, abs=1e-12)
    z = norm.ppf(0.975)
    assert pw.power_material(1, 0.1, 0.1, 1) == pytest.approx(0.025 + norm.cdf(-z - 0.2))
    assert pw.required_n_material(0.1, 0.1, 2, power=0.8) is None
    assert pw.required_n_material(0.1, 0.1, 2, power=0.01) == 1


def test_minimum_integer_n_is_exact_and_positive():
    assert pw.required_n_conventional(0.1, 1) == 785
    assert pw.required_n_conventional(0.1, 2) == 1570
    assert pw.required_n_material(0.2, 0.1, 1) == 785
    n = pw.required_n_material(0.25, 0.1, 2)
    assert n == 698
    assert pw.power_material(n - 1, 0.25, 0.1, 2) < 0.8
    assert pw.power_material(n, 0.25, 0.1, 2) >= 0.8


def test_n_max_is_a_real_limit_even_when_below_1024():
    assert pw.required_n_material(0.25, 0.1, 2, n_max=697) is None
    assert pw.required_n_material(0.25, 0.1, 2, n_max=698) == 698
    assert pw.required_n_material(0.25, 0.1, 2, n_max=1) is None


def test_zero_variance_obeys_degenerate_ci_rules_and_never_selects_n_zero():
    assert pw.required_n_conventional(0.1, 0) == 1
    assert pw.required_n_material(0.2, 0.1, 0) == 1
    assert pw.required_n_material(0.1, 0.1, 0) is None
    assert pw.power_conventional(1, 0, 0) == 0
    assert pw.power_conventional(1, -0.1, 0) == 1
    assert pw.power_material(1, 0.1, 0.1, 0) == 0
    assert pw.power_material(1, -0.2, 0.1, 0) == 1
    assert pw.power_equivalence(1, 0.1, 0.1, 0) == 1
    assert pw.power_equivalence(1, 0.2, 0.1, 0) == 0


def test_plan_defaults_do_not_select_a_scientific_n_or_claim_material_power():
    plan = pw.plan(COV, COV)
    assert plan["delta"] == 0.10
    assert plan["basis"] is None
    assert plan["status"] == "unresolved"
    assert plan["n_operative"] is None
    assert plan["achieved_power_conventional"] is None
    assert plan["achieved_power_material"] is None
    assert "single-model calibration" in " ".join(plan["unresolved"])
    json.dumps(plan, allow_nan=False)


@pytest.mark.parametrize("d_true", [None, 0.0, 0.1, -0.1])
def test_unattainable_material_goal_never_falls_back_to_conventional_n(d_true):
    plan = pw.plan(
        COV, COV, basis="material", d_true=d_true, cross_model_assumption="independent"
    )
    assert plan["n_conventional"] == 1570
    assert plan["n_material"] is None
    assert plan["n_operative"] is None
    assert plan["status"] == "unresolved"
    json.dumps(plan, allow_nan=False)


def test_a_material_alternative_does_not_fabricate_cross_model_covariance():
    plan = pw.plan(COV, COV, basis="material", d_true=0.25)
    assert plan["n_material"] == 698
    assert plan["n_operative"] is None
    assert plan["cross_model_covariance_basis"] == "unjustified_independence_scenario_only"


def test_explicit_material_and_conventional_scenarios_are_distinct():
    material = pw.plan(
        COV, COV, basis="material", d_true=0.25, cross_model_assumption="independent"
    )
    assert material["status"] == "conditional_scenario"
    assert material["n_operative"] == material["n_material"] == 698
    assert material["achieved_power_material"] >= 0.8
    assert material["achieved_power_conventional"] < 0.8
    conventional = pw.plan(
        COV, COV, basis="conventional", d_true=0.1, cross_model_cov=np.zeros((4, 4))
    )
    assert conventional["n_operative"] == 1570
    assert conventional["achieved_power_conventional"] >= 0.8
    assert conventional["achieved_power_material"] == pytest.approx(0.025, abs=1e-10)
    assert conventional["basis"] == "conventional"


def test_plan_budget_exhaustion_stays_unresolved():
    plan = pw.plan(
        COV, COV, basis="material", d_true=0.25, cross_model_assumption="independent", n_max=697
    )
    assert plan["n_operative"] is None
    assert "unattainable" in " ".join(plan["unresolved"])
    conventional = pw.plan(
        COV, COV, basis="conventional", cross_model_assumption="independent", n_max=100
    )
    assert conventional["n_operative"] is None


def test_variance_and_conditional_planning_remain_available_while_blinded():
    with phase_a_blinding():
        assert pw.unit_variance_of_d(COV, COV) == 2
        result = pw.plan(COV, COV)
        assert result["n_operative"] is None


@pytest.mark.parametrize("n", [0, -1, 1.5, True, np.nan, np.inf, "2"])
@pytest.mark.parametrize("function", ["power_conventional", "power_material", "power_equivalence"])
def test_invalid_item_counts(function, n):
    kwargs = {"n": n, "unit_variance": 1}
    kwargs.update({"effect": 0.2} if function == "power_conventional" else {"d_true": 0.2, "delta": 0.1})
    with pytest.raises(ValueError, match="n"):
        getattr(pw, function)(**kwargs)


@pytest.mark.parametrize("variance", [-1, np.nan, np.inf, -np.inf, True, "1"])
def test_invalid_unit_variance_at_all_power_entry_points(variance):
    calls = [
        lambda: pw.power_conventional(100, 0.2, variance),
        lambda: pw.required_n_conventional(0.2, variance),
        lambda: pw.power_material(100, 0.2, 0.1, variance),
        lambda: pw.required_n_material(0.1, 0.1, variance),
        lambda: pw.power_equivalence(100, 0.0, 0.1, variance),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="unit_variance"):
            call()


@pytest.mark.parametrize("value", [0, 1, -0.1, 1.1, np.nan, np.inf, True, "0.8"])
def test_invalid_probability_arguments(value):
    calls = [
        lambda: pw.power_conventional(100, 0.2, 1, alpha=value),
        lambda: pw.required_n_conventional(0.2, 1, alpha=value),
        lambda: pw.required_n_conventional(0.2, 1, power=value),
        lambda: pw.power_material(100, 0.2, 0.1, 1, ci_level=value),
        lambda: pw.required_n_material(0.1, 0.1, 1, ci_level=value),
        lambda: pw.required_n_material(0.1, 0.1, 1, power=value),
        lambda: pw.power_equivalence(100, 0.0, 0.1, 1, ci_level=value),
        lambda: pw.plan(COV, COV, alpha=value),
        lambda: pw.plan(COV, COV, power=value),
        lambda: pw.rough_unpaired_bound(alpha=value),
        lambda: pw.rough_unpaired_bound(power=value),
    ]
    for call in calls:
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("value", [0, -0.1, np.nan, np.inf, True, "0.1"])
def test_invalid_delta_arguments(value):
    for call in [
        lambda: pw.power_material(100, 0.2, value, 1),
        lambda: pw.required_n_material(0.1, value, 1),
        lambda: pw.power_equivalence(100, 0, value, 1),
        lambda: pw.plan(COV, COV, delta=value),
        lambda: pw.rough_unpaired_bound(value),
    ]:
        with pytest.raises(ValueError, match="delta"):
            call()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, "0.1"])
def test_invalid_effect_arguments(value):
    for call in [
        lambda: pw.power_conventional(100, value, 1),
        lambda: pw.required_n_conventional(value, 1),
        lambda: pw.power_material(100, value, 0.1, 1),
        lambda: pw.required_n_material(value, 0.1, 1),
        lambda: pw.power_equivalence(100, value, 0.1, 1),
        lambda: pw.plan(COV, COV, d_true=value),
    ]:
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("value", [0, -1, 1.5, True, np.nan, "10"])
def test_invalid_n_max_even_for_unattainable_effect(value):
    with pytest.raises(ValueError, match="n_max"):
        pw.required_n_material(0.1, 0.1, 1, n_max=value)
    with pytest.raises(ValueError, match="n_max"):
        pw.plan(COV, COV, n_max=value)


@pytest.mark.parametrize("rho", [-1, -0.34, 1, np.nan, np.inf, True, "0.5"])
def test_invalid_covariance_correlation(rho):
    with pytest.raises(ValueError, match="rho"):
        pw.covariance_from_correlation(pw.CellProbabilities(0.5, 0.5, 0.5, 0.5), rho)


@pytest.mark.parametrize("p", [-0.1, 1.1, np.nan, np.inf, True, "0.5"])
def test_invalid_marginal_probabilities(p):
    with pytest.raises(ValueError):
        pw.CellProbabilities(p, 0.5, 0.5, 0.5)


def test_plan_rejects_ambiguous_or_contradictory_basis_arguments():
    with pytest.raises(ValueError, match="basis"):
        pw.plan(COV, COV, basis="auto")
    with pytest.raises(ValueError, match="cross_model_assumption"):
        pw.plan(COV, COV, cross_model_assumption="calibration")
    with pytest.raises(ValueError, match="not both"):
        pw.plan(COV, COV, cross_model_cov=COV / 2, cross_model_assumption="independent")


@pytest.mark.parametrize(
    "overrides",
    [{"n": 1}, {"n": 2.5}, {"n_sims": 0}, {"n_sims": True}, {"seed": -1},
     {"seed": 1.5}, {"delta": np.nan}, {"rho": -0.5}, {"rho": np.nan}],
)
def test_simulator_validates_arguments_before_sampling(overrides):
    p = pw.CellProbabilities(0.5, 0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        pw.simulate_power(p, p, **({"n": 2, "n_sims": 2} | overrides))


def test_copula_variance_uses_bernoulli_not_latent_gaussian_correlation():
    p = pw.CellProbabilities(0.5, 0.5, 0.5, 0.5)
    out = pw.simulate_power(p, p, 20, rho=0.5, n_sims=2, seed=1)
    # For p=.5, Cov(B_i,B_j)=arcsin(rho)/(2*pi). c has zero sum.
    expected = 2 * (1 - 4 * math.asin(0.5) / (2 * math.pi))
    assert out["unit_variance_D"] == pytest.approx(expected, abs=1e-4)
