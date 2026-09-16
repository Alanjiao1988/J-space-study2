"""Model-stratified, paired bootstrap for the primary design (§6.2, §9.5).

Each model has exactly two independent real fits, and both models/fits evaluate
the same item bank in all four primary cells. Fits are drawn independently
*within each model*, two draws per model; one shared item draw is then used
across all models and fit evaluations. Fit names need not match between models.
Real-corpus provenance must be verified upstream: a trial's ID alone cannot
establish how its lens was fitted.

The estimator is the equal-weight mean of the two per-fit ITT cell accuracies,
not a pooled row mean. Missing or duplicate trials are errors; unparseable
attempts remain in the denominator as incorrect. Only one template ID per item
is supported. Multi-template/target weighting and truncated hierarchies are
unsupported rather than silently assigned an estimand.

The public bootstrap signature is retained, but ``levels`` must equal LEVELS
and ``n_resamples >= 2``. Custom statistics receive complete paired frames whose
fit/item IDs identify draw slots (repeated sources get distinct slot IDs).
Point estimators may select a subset of conditions for G_gen or Delta; the
bootstrap itself accepts only the two-model, four-cell primary design.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from wda.errors import ProtocolViolation
from wda.governance import blind
from wda.stats.decision import Interval

CLEAN = "clean"
ABLATED = "ablated"
C_DIRECT = "C_direct"
C_FROZEN = "C_frozen"
C_GEN = "C_gen"
LEVELS: Tuple[str, ...] = ("lens_id", "item_id", "template_id")
_ID_COLUMNS = ("lens_id", "item_id", "template_id", "model_role", "condition", "ablation_state")
_COLUMNS = (*_ID_COLUMNS, "parsed", "correct")
_CONTRAST = np.array([1.0, -1.0, -1.0, 1.0])


class BootstrapViolation(ProtocolViolation):
    """An incomplete design or inadmissible resampling scheme was requested."""


@dataclass
class TrialFrame:
    """Strictly typed trial columns; design completeness is checked at estimation.

    ``from_records`` requires every column and rejects duplicate keys. ``take``
    can also represent repeated rows for resampling diagnostics; such rows are
    not a valid estimation frame until assigned distinct paired draw slots.
    """

    lens_id: np.ndarray
    item_id: np.ndarray
    template_id: np.ndarray
    model_role: np.ndarray
    condition: np.ndarray
    ablation_state: np.ndarray
    parsed: np.ndarray
    correct: np.ndarray

    def __post_init__(self) -> None:
        for name in _COLUMNS:
            values = np.asarray(getattr(self, name), dtype=object)
            if values.ndim != 1:
                raise ValueError(f"column {name!r} must be one-dimensional")
            if name in _ID_COLUMNS:
                if any(not isinstance(v, str) or not v.strip() for v in values):
                    raise ValueError(f"column {name!r} requires nonempty string IDs, without coercion")
                values = values.astype(str)
            else:
                if any(not isinstance(v, (bool, np.bool_)) for v in values):
                    raise ValueError(f"column {name!r} requires boolean values, without coercion")
                values = values.astype(bool)
            setattr(self, name, values)
        n = len(self.item_id)
        if n == 0:
            raise ValueError("no trial records supplied")
        for name in _COLUMNS:
            if len(getattr(self, name)) != n:
                raise ValueError(f"column {name!r} has length {len(getattr(self, name))}, expected {n}")
        if (self.correct & ~self.parsed).any():
            raise ValueError("found trials marked correct but not parseable; ITT requires incorrect")

    def __len__(self) -> int:
        return len(self.item_id)

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, object]]) -> "TrialFrame":
        rows = list(records)
        if not rows:
            raise ValueError("no trial records supplied")
        for i, row in enumerate(rows):
            missing = set(_COLUMNS) - row.keys()
            if missing:
                raise ValueError(f"trial {i} is missing required keys: {sorted(missing)}")
        frame = cls(**{name: [row[name] for row in rows] for name in _COLUMNS})
        _row_index(frame)
        return frame

    def take(self, index: np.ndarray) -> "TrialFrame":
        return TrialFrame(**{name: getattr(self, name)[index] for name in _COLUMNS})


def _row_index(frame: TrialFrame) -> dict:
    index = {}
    for i, key in enumerate(zip(*(getattr(frame, name) for name in _ID_COLUMNS))):
        if key in index:
            raise BootstrapViolation(f"duplicate trial key {key!r}")
        index[key] = i
    return index


def _design_index(
    frame: TrialFrame,
    conditions: Sequence[str],
    *,
    model_roles: Sequence[str] | None = None,
    exact_conditions: bool = False,
) -> dict:
    # Revalidate mutable arrays before doing any estimation.
    frame.__post_init__()
    lookup = _row_index(frame)
    roles = tuple(sorted(set(frame.model_role))) if model_roles is None else tuple(model_roles)
    if not roles or len(set(roles)) != len(roles) or set(roles) != set(frame.model_role):
        raise BootstrapViolation("missing or unexpected model roles; models must be distinct")
    if exact_conditions and set(frame.condition) != set(conditions):
        raise BootstrapViolation("primary design requires exactly C_direct and C_frozen; empty cells or extra conditions")
    if set(frame.ablation_state) - {CLEAN, ABLATED}:
        raise BootstrapViolation("unsupported ablation state; expected clean or ablated")
    items = sorted(set(frame.item_id))
    templates = {}
    fits = {role: sorted(set(frame.lens_id[frame.model_role == role])) for role in roles}
    for role, names in fits.items():
        if len(names) != 2:
            detail = "fewer than two lens fits" if len(names) < 2 else "more than two lens fits"
            raise BootstrapViolation(
                f"{detail} for {role!r}; exactly two real fits per model are required"
            )
    for item, template in zip(frame.item_id, frame.template_id):
        if item in templates and templates[item] != template:
            raise BootstrapViolation(
                f"unsupported multi-template shape for item {item!r}; only one template ID "
                "per item is supported; nested template/target weighting is not specified"
            )
        templates[item] = template
    cells = [(condition, state) for condition in conditions for state in (CLEAN, ABLATED)]
    rows = np.empty((len(roles), 2, len(items), len(cells)), dtype=np.intp)
    for mi, role in enumerate(roles):
        for fi, fit in enumerate(fits[role]):
            for ii, item in enumerate(items):
                for ci, (condition, state) in enumerate(cells):
                    key = (fit, item, templates[item], role, condition, state)
                    if key not in lookup:
                        raise BootstrapViolation(
                            f"missing paired cell {key!r}; every item/model/fit/condition/state "
                            "must be present in the shared bank; empty cells may not be dropped"
                        )
                    rows[mi, fi, ii, ci] = lookup[key]
    return {"rows": rows, "model_roles": roles, "lens_keys": fits, "item_keys": items}


def validate_primary_design(
    frame: TrialFrame, *, treatment: str = "treatment", comparator: str = "comparator"
) -> None:
    """Validate every paired key, including two real-fit IDs per model."""
    _design_index(
        frame, (C_DIRECT, C_FROZEN),
        model_roles=(treatment, comparator), exact_conditions=True,
    )


def _cell_means(frame: TrialFrame, index: Mapping[str, object]) -> np.ndarray:
    # First average items within each fit, then give both fits weight 1/2.
    return frame.correct[index["rows"]].mean(axis=2).mean(axis=1)


def _model_cells(frame: TrialFrame, role: str, conditions: Sequence[str]) -> np.ndarray:
    index = _design_index(frame, conditions)
    if role not in index["model_roles"]:
        raise BootstrapViolation(f"missing model role {role!r}")
    return _cell_means(frame, index)[index["model_roles"].index(role)]


def delta(frame: TrialFrame, model_role: str, condition: str) -> float:
    """Equal-fit ``Delta(c) = acc_clean(c) - acc_ablated(c)``."""
    blind.guard_decision("wda.stats.bootstrap.delta")
    clean, ablated = _model_cells(frame, model_role, (condition,))
    return float(clean - ablated)


def g_primary(frame: TrialFrame, model_role: str) -> float:
    """Equal-fit ``G_primary = Delta(C_direct) - Delta(C_frozen)``."""
    blind.guard_decision("wda.stats.bootstrap.g_primary")
    return float(_model_cells(frame, model_role, (C_DIRECT, C_FROZEN)) @ _CONTRAST)


def g_gen(frame: TrialFrame, model_role: str) -> float:
    """Equal-fit ``G_gen = Delta(C_direct) - Delta(C_gen)``; not primary."""
    blind.guard_decision("wda.stats.bootstrap.g_gen")
    return float(_model_cells(frame, model_role, (C_DIRECT, C_GEN)) @ _CONTRAST)


def primary_endpoint(
    frame: TrialFrame, *, treatment: str = "treatment", comparator: str = "comparator"
) -> float:
    """``D = G_primary(treatment) - G_primary(comparator)``."""
    blind.guard_decision("wda.stats.bootstrap.primary_endpoint")
    index = _design_index(
        frame, (C_DIRECT, C_FROZEN),
        model_roles=(treatment, comparator), exact_conditions=True,
    )
    g = _cell_means(frame, index) @ _CONTRAST
    return float(g[0] - g[1])


def _validate_levels(levels: Sequence[str]) -> Tuple[str, ...]:
    if tuple(levels) != LEVELS:
        raise BootstrapViolation(
            f"unsupported levels {levels!r}; the outermost bootstrap level is the lens "
            f"fit within model, and the complete hierarchy must be {LEVELS!r}"
        )
    return LEVELS


def _integer(value: int, name: str, minimum: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite(value: float, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise BootstrapViolation(f"{name} must be a finite real number")


def _build_hierarchy(frame: TrialFrame, levels: Sequence[str]) -> dict:
    _validate_levels(levels)
    roles = tuple(sorted(set(frame.model_role)))
    if len(roles) != 2:
        raise BootstrapViolation("the primary bootstrap requires exactly two models")
    return _design_index(
        frame, (C_DIRECT, C_FROZEN), model_roles=roles, exact_conditions=True
    )


def _resample_from_hierarchy(index: Mapping[str, object], rng: np.random.Generator) -> np.ndarray:
    """Draw a complete (model, fit draw, shared item draw, cell) index."""
    rows = index["rows"]
    n_models, _, n_items, n_cells = rows.shape
    fits = rng.integers(0, 2, size=(n_models, 2))
    items = rng.integers(0, n_items, size=n_items)
    return rows[
        np.arange(n_models)[:, None, None, None],
        fits[:, :, None, None],
        items[None, None, :, None],
        np.arange(n_cells)[None, None, None, :],
    ]


def _resample_indices(frame: TrialFrame, rng: np.random.Generator, levels: Sequence[str]) -> np.ndarray:
    """Private diagnostic: flattened source indices, preserving paired multiplicities."""
    return _resample_from_hierarchy(_build_hierarchy(frame, levels), rng).ravel()


def _resampled_frame(frame: TrialFrame, rows: np.ndarray) -> TrialFrame:
    sampled = frame.take(rows.ravel())
    n_models, n_fits, n_items, n_cells = rows.shape
    sampled.lens_id = np.tile(
        np.repeat([f"fit_draw_{i}" for i in range(n_fits)], n_items * n_cells), n_models
    )
    sampled.item_id = np.tile(
        np.repeat([f"item_draw_{i}" for i in range(n_items)], n_cells), n_models * n_fits
    )
    return sampled


@dataclass
class BootstrapResult:
    estimate: float
    replicates: np.ndarray
    n_resamples: int
    levels: Tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        _integer(self.n_resamples, "n_resamples", 2)
        _integer(self.seed, "seed", 0)
        _validate_levels(self.levels)
        _finite(self.estimate, "estimate")
        values = np.asarray(self.replicates)
        if values.dtype.kind not in "fiu" or values.shape != (self.n_resamples,):
            raise BootstrapViolation("replicates must be a numeric vector of length n_resamples")
        if not np.isfinite(values).all():
            raise BootstrapViolation("non-finite bootstrap replicates; no replicate may be dropped")
        self.replicates = values.astype(float)

    def interval(self, level: float) -> Interval:
        """Percentile interval from every replicate, without finite-value filtering."""
        blind.guard_decision("wda.stats.bootstrap.BootstrapResult.interval")
        self.__post_init__()
        _finite(level, "level")
        if not 0 < level < 1:
            raise ValueError("level must be in (0, 1)")
        tail = (1.0 - level) / 2.0
        low, high = np.quantile(self.replicates, [tail, 1.0 - tail])
        return Interval(low=float(low), high=float(high), level=float(level))

    def to_dict(self) -> Dict[str, object]:
        blind.guard_decision("wda.stats.bootstrap.BootstrapResult.to_dict")
        ci95, ci90 = self.interval(0.95), self.interval(0.90)
        se = float(np.std(self.replicates, ddof=1))
        _finite(se, "bootstrap standard error")
        return {
            "schema": "wda/bootstrap_result/2",
            "estimate": float(self.estimate),
            "n_resamples": int(self.n_resamples),
            "levels": list(self.levels),
            "resampling": "independent fits within model; shared items across all fit draws",
            "fit_draws_per_model": 2,
            "aggregation": "equal-weight mean of per-fit ITT cell accuracies",
            "seed": int(self.seed),
            "ci95": ci95.to_dict(),
            "ci90": ci90.to_dict(),
            "se": se,
        }


def hierarchical_bootstrap(
    frame: TrialFrame,
    statistic=primary_endpoint,
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
    levels: Sequence[str] = LEVELS,
) -> BootstrapResult:
    """Bootstrap two independent real fits per model over one shared item draw."""
    blind.guard_decision("wda.stats.bootstrap.hierarchical_bootstrap")
    levels = _validate_levels(levels)
    _integer(n_resamples, "n_resamples", 2)
    _integer(seed, "seed", 0)
    index = _build_hierarchy(frame, levels)
    estimate = statistic(frame)
    _finite(estimate, "estimate")
    rng = np.random.default_rng(seed)
    replicates = np.empty(n_resamples, dtype=float)
    for r in range(n_resamples):
        rows = _resample_from_hierarchy(index, rng)
        if statistic is primary_endpoint:
            means = frame.correct[rows].mean(axis=2).mean(axis=1)
            effects = dict(zip(index["model_roles"], means @ _CONTRAST))
            value = effects["treatment"] - effects["comparator"]
        else:
            value = statistic(_resampled_frame(frame, rows))
        _finite(value, f"replicate {r}")
        replicates[r] = value
    return BootstrapResult(float(estimate), replicates, n_resamples, levels, seed)


def trial_bootstrap(*_args, **_kwargs):
    """Always raises: §9.5 forbids a flat trial-level bootstrap."""
    blind.guard_decision("wda.stats.bootstrap.trial_bootstrap")
    raise BootstrapViolation(
        "a flat trial-level bootstrap is forbidden (§9.5). Use the complete model-stratified "
        "hierarchical_bootstrap with paired cells and a shared item bank."
    )


__all__ = [
    "ABLATED", "BootstrapResult", "BootstrapViolation", "CLEAN", "C_DIRECT", "C_FROZEN",
    "C_GEN", "LEVELS", "TrialFrame", "delta", "g_gen", "g_primary", "hierarchical_bootstrap",
    "primary_endpoint", "trial_bootstrap", "validate_primary_design",
]
