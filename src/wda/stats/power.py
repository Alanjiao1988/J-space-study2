"""Power and sample size (§9.3).

``D`` is a difference (between checkpoints) of a difference (between response formats) of a
difference (ablated vs clean) of paired proportions. Within one model the same item appears
in all four cells (2 conditions x 2 ablation states), so ``G`` is a linear contrast

.. code-block::

    c = (+1, -1, -1, +1)   over   (direct_clean, direct_ablated, frozen_clean, frozen_ablated)

evaluated on a per-item Bernoulli vector ``X_i``. Hence ``Var(G_hat) = cSc / n`` where ``S``
is the 4x4 covariance of ``X_i``. That covariance is *measured in Phase A on the calibration
model* (§10 Phase A) and written into Freeze-2; the helpers here let the operator (a) derive
the pre-registered rough bound before Phase A and (b) recompute ``n`` from the measured
covariance afterwards.

Two power definitions are provided because §9.3 states a conventional target while §9.2
adjudicates with a shifted-CI rule, which is strictly harder:

``power_conventional``
    Two-sided alpha, ``H0: D = 0``, alternative ``|D| = delta``. This is the §9.3 target.
``power_material``
    ``P(the 95% CI of D lies entirely beyond +/-delta)`` — the actual §9.2 criterion.
    It is zero at ``|D_true| = delta`` and only becomes usable for ``|D_true| > delta``.

The operator should size the run with the larger of the two requirements and record which
was used. Nothing here may be re-run after Freeze-2 to justify a larger ``n`` (§11).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

#: The frozen contrast over the four within-model cells (§5).
CONTRAST: Tuple[float, float, float, float] = (1.0, -1.0, -1.0, 1.0)

CELL_ORDER: Tuple[str, str, str, str] = (
    "direct_clean",
    "direct_ablated",
    "frozen_clean",
    "frozen_ablated",
)


@dataclass(frozen=True)
class CellProbabilities:
    """Marginal ITT accuracy in each of the four within-model cells."""

    direct_clean: float
    direct_ablated: float
    frozen_clean: float
    frozen_ablated: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.direct_clean, self.direct_ablated, self.frozen_clean, self.frozen_ablated],
            dtype=float,
        )

    def g(self) -> float:
        return float(np.dot(CONTRAST, self.as_array()))

    def __post_init__(self) -> None:
        for name in CELL_ORDER:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a probability, got {value}")


def covariance_from_correlation(p: CellProbabilities, rho: float) -> np.ndarray:
    """Build a plausible 4x4 covariance from marginals plus one common correlation.

    Used *only* for the pre-Phase-A rough bound. After Phase A the empirical covariance
    from the calibration run replaces it.
    """
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1), got {rho}")
    sd = np.sqrt(p.as_array() * (1.0 - p.as_array()))
    corr = np.full((4, 4), rho, dtype=float)
    np.fill_diagonal(corr, 1.0)
    cov = corr * np.outer(sd, sd)
    eig = np.linalg.eigvalsh(cov)
    if eig.min() < -1e-12:
        raise ValueError(f"rho={rho} yields a non-PSD covariance (min eigenvalue {eig.min():.3e})")
    return cov


def contrast_variance(cov: np.ndarray, contrast: Sequence[float] = CONTRAST) -> float:
    """``cSc`` — the per-item variance of ``G_hat`` before dividing by ``n``."""
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (4, 4):
        raise ValueError(f"covariance must be 4x4, got {cov.shape}")
    c = np.asarray(contrast, dtype=float)
    value = float(c @ cov @ c)
    if value < 0:  # pragma: no cover - defensive
        raise ValueError("negative contrast variance; the covariance is not PSD")
    return value


def unit_variance_of_d(
    cov_treatment: np.ndarray,
    cov_comparator: np.ndarray,
    *,
    cross_model_cov: Optional[np.ndarray] = None,
) -> float:
    """Per-item variance of ``D_hat`` (multiply by ``1/n`` for the sampling variance).

    ``cross_model_cov`` is the 4x4 cross-covariance between the treatment and comparator
    cells for the *same* item. Because both checkpoints see the same item bank, this is
    generally positive and reduces ``Var(D)``. Leaving it ``None`` treats the two models as
    independent, which is the conservative choice used for the pre-registered bound.
    """
    var = contrast_variance(cov_treatment) + contrast_variance(cov_comparator)
    if cross_model_cov is not None:
        c = np.asarray(CONTRAST, dtype=float)
        var -= 2.0 * float(c @ np.asarray(cross_model_cov, dtype=float) @ c)
    return max(var, 0.0)


# --------------------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------------------


def _z(prob: float) -> float:
    return float(stats.norm.ppf(prob))


def power_conventional(
    n: int, effect: float, unit_variance: float, *, alpha: float = 0.05
) -> float:
    """§9.3 target: two-sided ``alpha``, ``H0: D = 0``, alternative ``|D| = effect``."""
    if n <= 0:
        raise ValueError("n must be positive")
    se = math.sqrt(unit_variance / n)
    if se == 0:
        return 1.0
    z_a = _z(1.0 - alpha / 2.0)
    return float(stats.norm.sf(z_a - abs(effect) / se) + stats.norm.cdf(-z_a - abs(effect) / se))


def required_n_conventional(
    effect: float, unit_variance: float, *, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Smallest ``n`` (items per model) meeting the §9.3 conventional target."""
    if effect <= 0:
        raise ValueError("effect must be positive")
    z_a = _z(1.0 - alpha / 2.0)
    z_b = _z(power)
    n = ((z_a + z_b) ** 2) * unit_variance / (effect**2)
    n_int = int(math.ceil(n))
    while power_conventional(n_int, effect, unit_variance, alpha=alpha) < power:  # pragma: no cover
        n_int += 1
    return n_int


def power_material(
    n: int, d_true: float, delta: float, unit_variance: float, *, ci_level: float = 0.95
) -> float:
    """``P(the CI of D lies entirely beyond +/-delta)`` — the actual §9.2 criterion.

    Zero when ``|d_true| <= delta``: the rule is designed so that a true effect merely equal
    to the minimum meaningful difference cannot be declared material.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    se = math.sqrt(unit_variance / n)
    if se == 0:
        return 1.0 if abs(d_true) > delta else 0.0
    z = _z(1.0 - (1.0 - ci_level) / 2.0)
    # Upper side: D_hat - z*se > delta ; lower side: D_hat + z*se < -delta.
    upper = stats.norm.sf((delta + z * se - d_true) / se)
    lower = stats.norm.cdf((-delta - z * se - d_true) / se)
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
    """Smallest ``n`` at which the §9.2 material rule attains ``power``.

    Returns ``None`` when ``|d_true| <= delta``, where no finite ``n`` suffices.
    """
    if abs(d_true) <= delta:
        return None
    lo, hi = 1, 1024
    while hi < n_max and power_material(hi, d_true, delta, unit_variance, ci_level=ci_level) < power:
        lo, hi = hi, hi * 2
    if power_material(hi, d_true, delta, unit_variance, ci_level=ci_level) < power:  # pragma: no cover
        return None
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
    """``P(the 90% CI of D lies entirely inside [-delta, +delta])`` — the §9.2 equivalence rule."""
    if n <= 0:
        raise ValueError("n must be positive")
    se = math.sqrt(unit_variance / n)
    if se == 0:
        return 1.0 if abs(d_true) < delta else 0.0
    z = _z(1.0 - (1.0 - ci_level) / 2.0)
    # Need D_hat - z*se >= -delta and D_hat + z*se <= delta.
    lo = (-delta + z * se - d_true) / se
    hi = (delta - z * se - d_true) / se
    if hi <= lo:
        return 0.0
    return float(stats.norm.cdf(hi) - stats.norm.cdf(lo))


# --------------------------------------------------------------------------------------
# Pre-registered rough bound (§9.3) and the Phase-A recomputation
# --------------------------------------------------------------------------------------


def rough_unpaired_bound(
    delta: float = 0.10, *, alpha: float = 0.05, power: float = 0.80
) -> Dict[str, object]:
    """Reproduce the §9.3 "rough unpaired upper bound" so the number is auditable.

    Worst case is ``p = 0.5`` in every cell with **no** within-item pairing
    (``rho = 0``), giving a per-item contrast variance of ``4 x 0.25 = 1.0`` for ``G``.
    """
    worst = CellProbabilities(0.5, 0.5, 0.5, 0.5)
    cov = covariance_from_correlation(worst, 0.0)
    var_g = contrast_variance(cov)  # == 1.0
    var_d_independent = unit_variance_of_d(cov, cov)  # == 2.0
    return {
        "assumptions": {
            "p_per_cell": 0.5,
            "within_item_correlation": 0.0,
            "cross_model_correlation": 0.0,
            "delta": delta,
            "alpha": alpha,
            "power": power,
        },
        "unit_variance_G": var_g,
        "unit_variance_D": var_d_independent,
        "n_per_model_for_G": required_n_conventional(delta, var_g, alpha=alpha, power=power),
        "n_per_model_for_D": required_n_conventional(
            delta, var_d_independent, alpha=alpha, power=power
        ),
        "note": (
            "The G-level number reproduces the protocol's 'n ~ 800 items per cell' bound. "
            "The D-level number is the honest requirement when the two checkpoints are "
            "treated as independent; pairing across the shared item bank reduces it. The "
            "operative n is fixed in Freeze-2 from the Phase-A measured covariance."
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
) -> Dict[str, object]:
    """Full sample-size plan from Phase-A covariances. Written verbatim into Freeze-2."""
    unit_var = unit_variance_of_d(cov_treatment, cov_comparator, cross_model_cov=cross_model_cov)
    conventional = required_n_conventional(delta, unit_var, alpha=alpha, power=power)
    material = (
        required_n_material(d_true, delta, unit_var, power=power) if d_true is not None else None
    )
    operative = conventional if material is None else max(conventional, material)
    return {
        "schema": "wda/power_plan/1",
        "delta": delta,
        "alpha": alpha,
        "target_power": power,
        "unit_variance_D": unit_var,
        "n_conventional": conventional,
        "d_true_for_material": d_true,
        "n_material": material,
        "n_operative": operative,
        "achieved_power_conventional": power_conventional(operative, delta, unit_var, alpha=alpha),
        "achieved_power_material": (
            power_material(operative, d_true, delta, unit_var) if d_true is not None else None
        ),
        "note": (
            "n is the number of items per model; each item contributes four paired cells. "
            "Fixed at Freeze-2 and never increased afterwards (SC3, §11)."
        ),
    }


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
    """Monte-Carlo check of the analytic formulas using a Gaussian-copula Bernoulli model."""
    rng = np.random.default_rng(seed)
    cov_t = covariance_from_correlation(p_treatment, rho)
    cov_c = covariance_from_correlation(p_comparator, rho)
    unit_var = unit_variance_of_d(cov_t, cov_c)
    d_true = p_treatment.g() - p_comparator.g()

    corr = np.full((4, 4), rho)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    thresholds_t = stats.norm.ppf(p_treatment.as_array())
    thresholds_c = stats.norm.ppf(p_comparator.as_array())
    c = np.asarray(CONTRAST)

    material = 0
    equivalent = 0
    z95 = _z(0.975)
    z90 = _z(0.95)
    for _ in range(n_sims):
        zt = rng.standard_normal((n, 4)) @ chol.T
        zc = rng.standard_normal((n, 4)) @ chol.T
        xt = (zt < thresholds_t).astype(float)
        xc = (zc < thresholds_c).astype(float)
        d_hat = float(c @ xt.mean(axis=0) - c @ xc.mean(axis=0))
        se = math.sqrt(
            (contrast_variance(np.cov(xt, rowvar=False)) + contrast_variance(np.cov(xc, rowvar=False)))
            / n
        )
        if d_hat - z95 * se > delta or d_hat + z95 * se < -delta:
            material += 1
        if d_hat - z90 * se >= -delta and d_hat + z90 * se <= delta:
            equivalent += 1
    return {
        "d_true": d_true,
        "unit_variance_D": unit_var,
        "empirical_power_material": material / n_sims,
        "empirical_power_equivalence": equivalent / n_sims,
        "analytic_power_material": power_material(n, d_true, delta, unit_var),
        "analytic_power_equivalence": power_equivalence(n, d_true, delta, unit_var),
    }


__all__ = [
    "CELL_ORDER",
    "CONTRAST",
    "CellProbabilities",
    "contrast_variance",
    "covariance_from_correlation",
    "plan",
    "power_conventional",
    "power_equivalence",
    "power_material",
    "required_n_conventional",
    "required_n_material",
    "rough_unpaired_bound",
    "simulate_power",
    "unit_variance_of_d",
]
