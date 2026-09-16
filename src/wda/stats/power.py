"""Transparent normal-approximation power planning (§9.2, §9.3).

The four-cell contrast is ``c = (1, -1, -1, 1)``. Supplied covariance matrices
must describe the per-item outcomes used by the estimator (including the
equal-weight fit average where appropriate). ``Var(D_hat) = unit_variance / n``.
These analytic formulas condition on the fitted lenses: they do not identify
or account for a between-fit variance floor in a two-fit bootstrap. Merely
increasing the item count cannot remove such a floor.

Conventional power tests H0: D=0. It is NOT the probability of the frozen
material decision, which requires the 95% CI beyond +/-delta (delta remains
0.10). At |D_true|=delta and positive variance the material probability
approaches 0.025, not zero (the opposite tail adds a small probability at
finite n). An 80% material target requires a specified |D*|>delta.

Phase-A calibration on one model cannot identify cross-model covariance.
Omitting cross_model_cov in the variance helper is an explicit mathematical
independence scenario, NOT a measured or universally conservative bound.
``plan`` now requires an explicit ``basis`` ("conventional" or "material")
and an explicit cross-model assumption or supplied joint covariance before
returning n_operative. Missing/unattainable material goals never fall back to
conventional n. All plans remain conditional scenarios, not authorization to
set Freeze-2 or to increase n after observing results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from wda.governance import blind

CONTRAST: Tuple[float, float, float, float] = (1.0, -1.0, -1.0, 1.0)
CELL_ORDER: Tuple[str, str, str, str] = (
    "direct_clean", "direct_ablated", "frozen_clean", "frozen_ablated",
)


def _real(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _positive(value: float, name: str) -> float:
    value = _real(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _probability(value: float, name: str) -> float:
    value = _real(value, name)
    if not 0 < value < 1:
        raise ValueError(f"{name} must be in (0, 1)")
    return value


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _variance(value: float) -> float:
    value = _real(value, "unit_variance")
    if value < 0:
        raise ValueError("unit_variance must be nonnegative")
    return value


def _array(value: np.ndarray, shape: tuple, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a numeric array of shape {shape}")
    array = array.astype(float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _covariance(value: np.ndarray, size: int = 4, name: str = "covariance") -> np.ndarray:
    cov = _array(value, (size, size), name)
    if not np.allclose(cov, cov.T, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    if (np.diag(cov) < 0).any():
        raise ValueError(f"{name} must be PSD with nonnegative diagonal")
    symmetric = cov / 2 + cov.T / 2
    eigenvalues = np.linalg.eigvalsh(symmetric)
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if not np.isfinite(eigenvalues).all() or eigenvalues.min() < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite (PSD)")
    return symmetric


def _quadratic(cov: np.ndarray, contrast: np.ndarray) -> float:
    value = float(contrast @ cov @ contrast)
    if not math.isfinite(value) or value < 0:
        raise ValueError("non-finite or negative contrast variance; no variance is clamped to zero")
    return value


def _se(n: int, unit_variance: float) -> float:
    # Avoid underflow from dividing the variance before taking its square root.
    return math.sqrt(unit_variance) / math.sqrt(n)


def _critical(tail_probability: float) -> float:
    value = float(stats.norm.isf(tail_probability))
    if not math.isfinite(value):
        raise ValueError("tail probability is too small for finite normal quantiles")
    return value


@dataclass(frozen=True)
class CellProbabilities:
    """Hypothetical marginal ITT probabilities in four within-model cells."""

    direct_clean: float
    direct_ablated: float
    frozen_clean: float
    frozen_ablated: float

    def __post_init__(self) -> None:
        for name in CELL_ORDER:
            value = _real(getattr(self, name), name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be a probability in [0, 1]")

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in CELL_ORDER], dtype=float)

    def g(self) -> float:
        blind.guard_decision("wda.stats.power.CellProbabilities.g")
        return float(np.dot(CONTRAST, self.as_array()))


def covariance_from_correlation(p: CellProbabilities, rho: float) -> np.ndarray:
    """A PSD covariance scenario, not a measured covariance or Bernoulli joint law.

    PSD alone does not establish that these marginals/correlations are jointly
    attainable by Bernoulli outcomes. Use measured or justified covariances for
    scientific planning.
    """
    if not isinstance(p, CellProbabilities):
        raise TypeError("p must be CellProbabilities")
    rho = _real(rho, "rho")
    if not -1 / 3 <= rho < 1:
        raise ValueError("rho must be in [-1/3, 1) for a PSD four-cell common correlation")
    sd = np.sqrt(p.as_array() * (1 - p.as_array()))
    corr = np.full((4, 4), rho)
    np.fill_diagonal(corr, 1.0)
    return _covariance(corr * np.outer(sd, sd))


def contrast_variance(cov: np.ndarray, contrast: Sequence[float] = CONTRAST) -> float:
    """Per-item contrast variance cSc; covariance and contrast must be finite."""
    return _quadratic(_covariance(cov), _array(contrast, (4,), "contrast"))


def unit_variance_of_d(
    cov_treatment: np.ndarray,
    cov_comparator: np.ndarray,
    *,
    cross_model_cov: Optional[np.ndarray] = None,
) -> float:
    """Variance of the between-model contrast under a valid joint covariance.

    The cross block need not itself be symmetric. The full 8x8 block matrix
    must be symmetric PSD. None means the *assumed* zero cross-covariance
    scenario; shared items do not imply that this is conservative.
    """
    treatment = _covariance(cov_treatment, name="cov_treatment")
    comparator = _covariance(cov_comparator, name="cov_comparator")
    cross = (
        np.zeros((4, 4)) if cross_model_cov is None
        else _array(cross_model_cov, (4, 4), "cross_model_cov")
    )
    joint = _covariance(
        np.block([[treatment, cross], [cross.T, comparator]]),
        size=8, name="cross-model joint covariance",
    )
    c = np.asarray(CONTRAST)
    return _quadratic(joint, np.concatenate([c, -c]))


def power_conventional(
    n: int, effect: float, unit_variance: float, *, alpha: float = 0.05
) -> float:
    """Two-sided normal-test power against H0: D=0, not material-decision power."""
    n = _integer(n, "n")
    effect = _real(effect, "effect")
    unit_variance = _variance(unit_variance)
    alpha = _probability(alpha, "alpha")
    z = _critical(alpha / 2)
    se = _se(n, unit_variance)
    if se == 0:
        return float(effect != 0)
    shift = abs(effect) / se
    return float(stats.norm.sf(z - shift) + stats.norm.cdf(-z - shift))


def required_n_conventional(
    effect: float, unit_variance: float, *, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Smallest positive integer n meeting conventional power."""
    effect = _positive(effect, "effect")
    unit_variance = _variance(unit_variance)
    alpha = _probability(alpha, "alpha")
    power = _probability(power, "power")
    lo, hi = 1, 1
    while power_conventional(hi, effect, unit_variance, alpha=alpha) < power:
        lo, hi = hi + 1, hi * 2
        if hi > np.iinfo(np.int64).max:
            raise ValueError("required n exceeds the supported integer range")
    while lo < hi:
        mid = (lo + hi) // 2
        if power_conventional(mid, effect, unit_variance, alpha=alpha) >= power:
            hi = mid
        else:
            lo = mid + 1
    return lo


def power_material(
    n: int, d_true: float, delta: float, unit_variance: float, *, ci_level: float = 0.95
) -> float:
    """Probability the CI lies strictly beyond +/-delta.

    At |D_true|=delta and positive variance, this is (1-ci_level)/2 plus
    the opposite tail: approximately 0.025 for a 95% CI, not zero.
    """
    n = _integer(n, "n")
    d_true = _real(d_true, "d_true")
    delta = _positive(delta, "delta")
    unit_variance = _variance(unit_variance)
    ci_level = _probability(ci_level, "ci_level")
    z = _critical((1 - ci_level) / 2)
    se = _se(n, unit_variance)
    if se == 0:
        return float(abs(d_true) > delta)
    upper = stats.norm.sf(z + (delta - d_true) / se)
    lower = stats.norm.cdf(-z - (delta + d_true) / se)
    return float(upper + lower)


def required_n_material(
    d_true: float,
    delta: float,
    unit_variance: float,
    *,
    ci_level: float = 0.95,
    power: float = 0.80,
    n_max: int = 2_000_000,
) -> Optional[int]:
    """Smallest n meeting material power, or None if unattainable by n_max.

    With a 95% CI and an 80% target, |D_true|<=delta cannot suffice.
    Lower targets still use the actual probability, including the boundary tail.
    """
    d_true = _real(d_true, "d_true")
    delta = _positive(delta, "delta")
    unit_variance = _variance(unit_variance)
    ci_level = _probability(ci_level, "ci_level")
    power = _probability(power, "power")
    n_max = _integer(n_max, "n_max")
    if power_material(1, d_true, delta, unit_variance, ci_level=ci_level) >= power:
        return 1
    if abs(d_true) <= delta:
        return None
    lo, hi = 1, 1
    while power_material(hi, d_true, delta, unit_variance, ci_level=ci_level) < power:
        if hi == n_max:
            return None
        lo, hi = hi + 1, min(hi * 2, n_max)
    while lo < hi:
        mid = (lo + hi) // 2
        if power_material(mid, d_true, delta, unit_variance, ci_level=ci_level) >= power:
            hi = mid
        else:
            lo = mid + 1
    return lo


def power_equivalence(
    n: int, d_true: float, delta: float, unit_variance: float, *, ci_level: float = 0.90
) -> float:
    """Probability the CI lies inside [-delta, +delta], including its boundary."""
    n = _integer(n, "n")
    d_true = _real(d_true, "d_true")
    delta = _positive(delta, "delta")
    unit_variance = _variance(unit_variance)
    ci_level = _probability(ci_level, "ci_level")
    z = _critical((1 - ci_level) / 2)
    se = _se(n, unit_variance)
    if se == 0:
        return float(abs(d_true) <= delta)
    lo = z + (-delta - d_true) / se
    hi = -z + (delta - d_true) / se
    if hi <= lo:
        return 0.0
    return float(stats.norm.cdf(hi) - stats.norm.cdf(lo))


def rough_unpaired_bound(
    delta: float = 0.10, *, alpha: float = 0.05, power: float = 0.80
) -> Dict[str, object]:
    """Audit the ~800 G-level number as a conventional independence scenario.

    Eight mutually independent p=0.5 cells imply variance_G=1 and variance_D=2.
    Neither number is a general bound when arbitrary correlations are allowed.
    """
    delta = _positive(delta, "delta")
    alpha = _probability(alpha, "alpha")
    power = _probability(power, "power")
    cov = np.eye(4) * 0.25
    var_g = contrast_variance(cov)
    var_d = unit_variance_of_d(cov, cov)
    return {
        "basis": "conventional",
        "assumptions": {
            "p_per_cell": 0.5,
            "within_item_correlation": 0.0,
            "cross_model_correlation": 0.0,
            "delta": delta, "alpha": alpha, "power": power,
        },
        "unit_variance_G": var_g,
        "unit_variance_D": var_d,
        "n_per_model_for_G": required_n_conventional(delta, var_g, alpha=alpha, power=power),
        "n_per_model_for_D": required_n_conventional(delta, var_d, alpha=alpha, power=power),
        "n_operative": None,
        "note": (
            "A conventional H0:D=0 independence scenario, not 80% material power and not "
            "a universal variance bound. Single-model calibration cannot identify "
            "cross-model covariance. No scientific sample size is selected here."
        ),
    }


def plan(
    cov_treatment: np.ndarray,
    cov_comparator: np.ndarray,
    *,
    delta: float = 0.10,
    d_true: Optional[float] = None,
    cross_model_cov: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    power: float = 0.80,
    basis: Optional[str] = None,
    cross_model_assumption: Optional[str] = None,
    n_max: int = 2_000_000,
) -> Dict[str, object]:
    """Conditional planning scenario; defaults deliberately leave n_operative unset.

    ``basis`` must explicitly select "conventional" or "material".
    With no supplied cross block, ``cross_model_assumption="independent"``
    explicitly opts into the zero-cross-covariance scenario; it is not inferred
    from single-model calibration. Supplied covariance must represent the
    per-item equal-fit means, not unaggregated per-fit rows.
    """
    delta = _positive(delta, "delta")
    alpha = _probability(alpha, "alpha")
    power = _probability(power, "power")
    n_max = _integer(n_max, "n_max")
    if d_true is not None:
        d_true = _real(d_true, "d_true")
    if basis not in (None, "conventional", "material"):
        raise ValueError("basis must be explicitly 'conventional' or 'material', or None")
    if cross_model_assumption not in (None, "independent"):
        raise ValueError("cross_model_assumption must be 'independent' or None")
    if cross_model_cov is not None and cross_model_assumption is not None:
        raise ValueError("supply either cross_model_cov or an independence assumption, not both")
    unit_var = unit_variance_of_d(cov_treatment, cov_comparator, cross_model_cov=cross_model_cov)
    conventional = required_n_conventional(delta, unit_var, alpha=alpha, power=power)
    material = (
        required_n_material(d_true, delta, unit_var, power=power, n_max=n_max)
        if d_true is not None else None
    )
    unresolved = []
    if basis is None:
        unresolved.append("an explicit conventional or material planning basis is required")
    if cross_model_cov is None and cross_model_assumption is None:
        unresolved.append("cross-model covariance/assumption is unidentified by single-model calibration")
    if basis == "material":
        if d_true is None or abs(d_true) <= delta:
            unresolved.append("material planning requires a justified |D*| > delta")
        elif material is None:
            unresolved.append("material target is unattainable within n_max")
    if basis == "conventional" and conventional > n_max:
        unresolved.append("conventional target is unattainable within n_max")
    operative = None if unresolved else (material if basis == "material" else conventional)
    return {
        "schema": "wda/power_plan/2",
        "basis": basis,
        "status": "unresolved" if unresolved else "conditional_scenario",
        "unresolved": unresolved,
        "delta": delta, "alpha": alpha, "material_ci_level": 0.95,
        "target_power": power,
        "cross_model_covariance_basis": (
            "supplied_joint_covariance" if cross_model_cov is not None
            else "assumed_independent" if cross_model_assumption == "independent"
            else "unjustified_independence_scenario_only"
        ),
        "unit_variance_D": unit_var,
        "n_conventional": conventional,
        "d_true_for_material": d_true,
        "n_material": material,
        "n_operative": operative,
        "achieved_power_conventional": (
            power_conventional(operative, delta, unit_var, alpha=alpha) if operative is not None else None
        ),
        "achieved_power_material": (
            power_material(operative, d_true, delta, unit_var)
            if operative is not None and d_true is not None else None
        ),
        "note": (
            "Normal approximation conditional on fits; n counts shared items per model. "
            "Covariances must describe equal-fit per-item means. The between-fit variance "
            "floor is not identified here. Single-model calibration cannot identify "
            "cross-model covariance. This scenario does not set Freeze-2 or authorize "
            "additional samples after an inconclusive result."
        ),
    }


def _copula_covariance(p: CellProbabilities, rho: float) -> np.ndarray:
    probabilities = p.as_array()
    thresholds = stats.norm.ppf(probabilities)
    cov = np.diag(probabilities * (1 - probabilities))
    for i in range(4):
        for j in range(i):
            if probabilities[i] in (0, 1) or probabilities[j] in (0, 1) or rho == 0:
                value = 0.0
            else:
                # Latent Gaussian rho is not the observed Bernoulli correlation.
                joint = stats.multivariate_normal.cdf(
                    [thresholds[i], thresholds[j]], cov=[[1, rho], [rho, 1]]
                )
                value = joint - probabilities[i] * probabilities[j]
            cov[i, j] = cov[j, i] = value
    return _covariance(cov)


def simulate_power(
    p_treatment: CellProbabilities,
    p_comparator: CellProbabilities,
    n: int,
    *,
    delta: float = 0.10,
    rho: float = 0.5,
    n_sims: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """Synthetic Gaussian-copula Bernoulli check, conditional on independent models.

    The analytic comparator uses the copula's induced Bernoulli covariance,
    not the latent Gaussian correlation as a Bernoulli correlation.
    """
    blind.guard_decision("wda.stats.power.simulate_power")
    if not isinstance(p_treatment, CellProbabilities) or not isinstance(p_comparator, CellProbabilities):
        raise TypeError("simulation marginals must be CellProbabilities")
    n = _integer(n, "n", minimum=2)
    n_sims = _integer(n_sims, "n_sims")
    seed = _integer(seed, "seed", minimum=0)
    delta = _positive(delta, "delta")
    rho = _real(rho, "rho")
    if not -1 / 3 < rho < 1:
        raise ValueError("simulation rho must be in (-1/3, 1) for a positive-definite correlation")
    rng = np.random.default_rng(seed)
    unit_var = unit_variance_of_d(
        _copula_covariance(p_treatment, rho), _copula_covariance(p_comparator, rho)
    )
    d_true = p_treatment.g() - p_comparator.g()
    corr = np.full((4, 4), rho)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    thresholds_t = stats.norm.ppf(p_treatment.as_array())
    thresholds_c = stats.norm.ppf(p_comparator.as_array())
    c = np.asarray(CONTRAST)
    material = equivalent = 0
    z95, z90 = _critical(0.025), _critical(0.05)
    for _ in range(n_sims):
        xt = (rng.standard_normal((n, 4)) @ chol.T < thresholds_t).astype(float)
        xc = (rng.standard_normal((n, 4)) @ chol.T < thresholds_c).astype(float)
        d_hat = float(c @ xt.mean(axis=0) - c @ xc.mean(axis=0))
        # Validate covariances, then use direct contrast variance to avoid cancellation.
        contrast_variance(np.cov(xt, rowvar=False))
        contrast_variance(np.cov(xc, rowvar=False))
        se = math.sqrt((np.var(xt @ c, ddof=1) + np.var(xc @ c, ddof=1)) / n)
        if d_hat - z95 * se > delta or d_hat + z95 * se < -delta:
            material += 1
        if d_hat - z90 * se >= -delta and d_hat + z90 * se <= delta:
            equivalent += 1
    return {
        "d_true": d_true, "unit_variance_D": unit_var,
        "empirical_power_material": material / n_sims,
        "empirical_power_equivalence": equivalent / n_sims,
        "analytic_power_material": power_material(n, d_true, delta, unit_var),
        "analytic_power_equivalence": power_equivalence(n, d_true, delta, unit_var),
    }


__all__ = [
    "CELL_ORDER", "CONTRAST", "CellProbabilities", "contrast_variance",
    "covariance_from_correlation", "plan", "power_conventional", "power_equivalence",
    "power_material", "required_n_conventional", "required_n_material", "rough_unpaired_bound",
    "simulate_power", "unit_variance_of_d",
]
