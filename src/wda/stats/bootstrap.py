"""Hierarchical bootstrap (§9.5).

Resampling levels, outermost first::

    lens fit  ->  item  ->  template / target (when the assay has them)

A flat trial-level bootstrap is **forbidden**: the four cells of a single item (2 conditions
x 2 ablation states) are paired, and resampling trials independently destroys exactly the
pairing that gives ``D`` its precision (§9.3). :func:`trial_bootstrap` therefore exists only
to raise.

The estimands (§5)::

    Delta(c, m) = acc_unablated(c, m) - acc_ablated(c, m)
    G_primary(m) = Delta(C_direct, m) - Delta(C_frozen, m)
    D            = G_primary(treatment) - G_primary(comparator)

Accuracy is always the ITT composite (§9.4): an unparseable output counts as incorrect and
is never dropped from the denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np

from wda.errors import ProtocolViolation
from wda.stats.decision import Interval

CLEAN = "clean"
ABLATED = "ablated"
C_DIRECT = "C_direct"
C_FROZEN = "C_frozen"
C_GEN = "C_gen"

#: Level order is frozen; §9.5 forbids reordering or omitting the lens level.
LEVELS: Tuple[str, ...] = ("lens_id", "item_id", "template_id")


class BootstrapViolation(ProtocolViolation):
    """§9.5 — an inadmissible resampling scheme was requested."""


@dataclass
class TrialFrame:
    """Columnar view of ``log.jsonl`` (§15.4), restricted to the fields the estimand needs."""

    lens_id: np.ndarray
    item_id: np.ndarray
    template_id: np.ndarray
    model_role: np.ndarray
    condition: np.ndarray
    ablation_state: np.ndarray
    parsed: np.ndarray
    correct: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.item_id)
        for name in (
            "lens_id",
            "template_id",
            "model_role",
            "condition",
            "ablation_state",
            "parsed",
            "correct",
        ):
            if len(getattr(self, name)) != n:
                raise ValueError(f"column {name!r} has length {len(getattr(self, name))}, expected {n}")
        bad = np.asarray(self.correct, dtype=bool) & ~np.asarray(self.parsed, dtype=bool)
        if bad.any():
            raise ValueError(
                "found trials marked correct but not parseable; under ITT (§9.4) an "
                "unparseable output is incorrect by construction"
            )

    def __len__(self) -> int:
        return int(len(self.item_id))

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, object]]) -> "TrialFrame":
        rows = list(records)
        if not rows:
            raise ValueError("no trial records supplied")

        def col(name: str, default=None, dtype=object):
            values = [r.get(name, default) for r in rows]
            return np.asarray(values, dtype=dtype)

        parsed = col("parsed", False, bool)
        correct = col("correct", False, bool)
        return cls(
            lens_id=col("lens_id", "lens_unspecified", object).astype(str),
            item_id=col("item_id", dtype=object).astype(str),
            template_id=col("template_id", "template_none", object).astype(str),
            model_role=col("model_role", dtype=object).astype(str),
            condition=col("condition", dtype=object).astype(str),
            ablation_state=col("ablation_state", dtype=object).astype(str),
            parsed=np.asarray(parsed, dtype=bool),
            # ITT composite: unparseable -> incorrect, enforced here rather than trusted.
            correct=np.asarray(correct, dtype=bool) & np.asarray(parsed, dtype=bool),
        )

    def take(self, index: np.ndarray) -> "TrialFrame":
        return TrialFrame(
            lens_id=self.lens_id[index],
            item_id=self.item_id[index],
            template_id=self.template_id[index],
            model_role=self.model_role[index],
            condition=self.condition[index],
            ablation_state=self.ablation_state[index],
            parsed=self.parsed[index],
            correct=self.correct[index],
        )


# --------------------------------------------------------------------------------------
# Point estimates
# --------------------------------------------------------------------------------------


def _cell_accuracy(frame: TrialFrame, model_role: str, condition: str, state: str) -> float:
    mask = (
        (frame.model_role == model_role)
        & (frame.condition == condition)
        & (frame.ablation_state == state)
    )
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    return float(frame.correct[mask].mean())


def delta(frame: TrialFrame, model_role: str, condition: str) -> float:
    """``Delta(c) = acc_unablated(c) - acc_ablated(c)`` (§5)."""
    return _cell_accuracy(frame, model_role, condition, CLEAN) - _cell_accuracy(
        frame, model_role, condition, ABLATED
    )


def g_primary(frame: TrialFrame, model_role: str) -> float:
    """``G_primary = Delta(C_direct) - Delta(C_frozen)`` (§5)."""
    return delta(frame, model_role, C_DIRECT) - delta(frame, model_role, C_FROZEN)


def g_gen(frame: TrialFrame, model_role: str) -> float:
    """``G_gen = Delta(C_direct) - Delta(C_gen)`` — secondary/exploratory only (§5)."""
    return delta(frame, model_role, C_DIRECT) - delta(frame, model_role, C_GEN)


def primary_endpoint(
    frame: TrialFrame, *, treatment: str = "treatment", comparator: str = "comparator"
) -> float:
    """``D = G_primary(treatment) - G_primary(comparator)`` (§5)."""
    return g_primary(frame, treatment) - g_primary(frame, comparator)


# --------------------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------------------


def _group_index(keys: np.ndarray, within: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key in np.unique(keys[within]):
        out[str(key)] = np.flatnonzero(within & (keys == key))
    return out


def _build_hierarchy(frame: "TrialFrame", levels: Sequence[str]) -> Dict[str, object]:
    """Precompute the nested lens -> item -> template row index.

    Built once per :func:`hierarchical_bootstrap` call rather than once per resample; the
    resulting structure is what the resampler draws from, so the sampling distribution is
    identical to rebuilding it each time but the cost is paid once.
    """
    order = np.lexsort((frame.template_id, frame.item_id, frame.lens_id))
    lens_sorted = frame.lens_id[order]
    item_sorted = frame.item_id[order]
    tpl_sorted = frame.template_id[order]

    hierarchy: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    start = 0
    n = len(order)
    while start < n:
        lens = lens_sorted[start]
        lens_end = start + int(np.searchsorted(lens_sorted[start:], lens, side="right"))
        items: Dict[str, Dict[str, np.ndarray]] = {}
        i = start
        while i < lens_end:
            item = item_sorted[i]
            item_end = i + int(np.searchsorted(item_sorted[i:lens_end], item, side="right"))
            templates: Dict[str, np.ndarray] = {}
            t = i
            while t < item_end:
                tpl = tpl_sorted[t]
                tpl_end = t + int(np.searchsorted(tpl_sorted[t:item_end], tpl, side="right"))
                templates[str(tpl)] = order[t:tpl_end]
                t = tpl_end
            items[str(item)] = templates
            i = item_end
        hierarchy[str(lens)] = items
        start = lens_end

    lens_keys = sorted(hierarchy)
    return {
        "hierarchy": hierarchy,
        "lens_keys": lens_keys,
        "item_keys": {lens: sorted(hierarchy[lens]) for lens in lens_keys},
        "levels": tuple(levels),
    }


def _resample_from_hierarchy(index: Mapping[str, object], rng: np.random.Generator) -> np.ndarray:
    """Draw one hierarchical resample from a precomputed index."""
    hierarchy = index["hierarchy"]  # type: ignore[index]
    lens_keys: List[str] = index["lens_keys"]  # type: ignore[assignment]
    item_keys: Dict[str, List[str]] = index["item_keys"]  # type: ignore[assignment]
    levels: Tuple[str, ...] = index["levels"]  # type: ignore[assignment]

    chosen: List[np.ndarray] = []
    # Level 1: lens fits, with replacement (§6.2 - the lens fit is a bootstrap level).
    for li in rng.integers(0, len(lens_keys), size=len(lens_keys)):
        lens = lens_keys[int(li)]
        items = item_keys[lens]
        if "item_id" not in levels:  # pragma: no cover - defensive
            chosen.extend(hierarchy[lens][item].get(t) for item in items for t in hierarchy[lens][item])
            continue
        if not items:  # pragma: no cover - defensive
            continue
        # Level 2: items, with replacement. All cells of a drawn item travel together,
        # which is what preserves the pairing the power calculation assumes (§9.3).
        for ii in rng.integers(0, len(items), size=len(items)):
            templates = hierarchy[lens][items[int(ii)]]
            tpl_keys = list(templates)
            if "template_id" not in levels or len(tpl_keys) <= 1:
                chosen.extend(templates[t] for t in tpl_keys)
                continue
            # Level 3: template / target, when the assay defines more than one.
            for ti in rng.integers(0, len(tpl_keys), size=len(tpl_keys)):
                chosen.append(templates[tpl_keys[int(ti)]])

    if not chosen:  # pragma: no cover - defensive
        raise BootstrapViolation("hierarchical resample produced no rows")
    return np.concatenate(chosen)


def _resample_indices(frame: "TrialFrame", rng: np.random.Generator, levels: Sequence[str]) -> np.ndarray:
    """Draw one hierarchical resample, returning row indices into ``frame``."""
    return _resample_from_hierarchy(_build_hierarchy(frame, levels), rng)


@dataclass
class BootstrapResult:
    estimate: float
    replicates: np.ndarray
    n_resamples: int
    levels: Tuple[str, ...]
    seed: int

    def interval(self, level: float) -> Interval:
        """Percentile interval at ``level`` (e.g. ``0.95``)."""
        finite = self.replicates[np.isfinite(self.replicates)]
        if finite.size == 0:  # pragma: no cover - defensive
            raise BootstrapViolation("all bootstrap replicates were non-finite")
        tail = (1.0 - level) / 2.0
        low, high = np.quantile(finite, [tail, 1.0 - tail])
        return Interval(low=float(low), high=float(high), level=level)

    def to_dict(self) -> Dict[str, object]:
        ci95 = self.interval(0.95)
        ci90 = self.interval(0.90)
        return {
            "schema": "wda/bootstrap_result/1",
            "estimate": self.estimate,
            "n_resamples": self.n_resamples,
            "levels": list(self.levels),
            "seed": self.seed,
            "ci95": ci95.to_dict(),
            "ci90": ci90.to_dict(),
            "se": float(np.nanstd(self.replicates, ddof=1)),
        }


def hierarchical_bootstrap(
    frame: TrialFrame,
    statistic=primary_endpoint,
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
    levels: Sequence[str] = LEVELS,
) -> BootstrapResult:
    """Resample ``frame`` at the frozen levels and evaluate ``statistic`` on each replicate."""
    levels = tuple(levels)
    if levels[0] != "lens_id":
        raise BootstrapViolation(
            "the outermost bootstrap level must be the lens fit (§6.2, §9.5); "
            f"got {levels!r}"
        )
    unknown = set(levels) - set(LEVELS)
    if unknown:
        raise BootstrapViolation(f"unknown bootstrap levels {sorted(unknown)}")

    rng = np.random.default_rng(seed)
    estimate = float(statistic(frame))
    index = _build_hierarchy(frame, levels)
    replicates = np.empty(n_resamples, dtype=float)
    for r in range(n_resamples):
        rows = _resample_from_hierarchy(index, rng)
        replicates[r] = float(statistic(frame.take(rows)))
    return BootstrapResult(
        estimate=estimate,
        replicates=replicates,
        n_resamples=n_resamples,
        levels=levels,
        seed=seed,
    )


def trial_bootstrap(*_args, **_kwargs):
    """Always raises. §9.5 forbids a flat trial-level bootstrap."""
    raise BootstrapViolation(
        "a flat trial-level bootstrap is forbidden (§9.5). The four cells of an item are "
        "paired; resampling trials independently discards that pairing. Use "
        "hierarchical_bootstrap(levels=('lens_id', 'item_id', 'template_id'))."
    )


__all__ = [
    "ABLATED",
    "BootstrapResult",
    "BootstrapViolation",
    "CLEAN",
    "C_DIRECT",
    "C_FROZEN",
    "C_GEN",
    "LEVELS",
    "TrialFrame",
    "delta",
    "g_gen",
    "g_primary",
    "hierarchical_bootstrap",
    "primary_endpoint",
    "trial_bootstrap",
]
