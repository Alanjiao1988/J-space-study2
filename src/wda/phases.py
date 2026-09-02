"""Phases (§10) and model roles (§15.1).

The phase order is meaningful: a seal stage is admissible only if every phase that must
precede it has an entry in ``runs/``. The enums are string-valued so they serialise
directly into ``log.jsonl`` and ``config.json``.
"""

from __future__ import annotations

from enum import Enum


class Phase(str, Enum):
    """Execution phases in protocol order."""

    FREEZE_1 = "freeze1"
    PHASE_0 = "phase0"
    PHASE_A = "phaseA"
    PHASE_B = "phaseB"
    FREEZE_2 = "freeze2"
    PHASE_C = "phaseC"
    PHASE_D = "phaseD"

    @property
    def order(self) -> int:
        return _PHASE_ORDER[self]

    def precedes(self, other: "Phase") -> bool:
        return self.order < other.order


_PHASE_ORDER = {
    Phase.FREEZE_1: 0,
    Phase.PHASE_0: 1,
    Phase.PHASE_A: 2,
    Phase.PHASE_B: 3,
    Phase.FREEZE_2: 4,
    Phase.PHASE_C: 5,
    Phase.PHASE_D: 6,
}

#: Phases in which model weights are actually loaded and forward passes are run.
MODEL_CALLING_PHASES = frozenset(
    {Phase.PHASE_0, Phase.PHASE_A, Phase.PHASE_B, Phase.PHASE_C, Phase.PHASE_D}
)

#: Phases whose outputs carry evidential weight for RQ1 / RQ2.
EVIDENTIAL_PHASES = frozenset({Phase.PHASE_C, Phase.PHASE_D})


class Role(str, Enum):
    """§8 model roles."""

    TREATMENT = "treatment"
    COMPARATOR = "comparator"
    PARENT_ANCHOR = "parent_anchor"
    CALIBRATION = "calibration"


#: Roles that Phase A blinding (§16) must refuse to hand out.
SUBJECT_ROLES = frozenset({Role.TREATMENT, Role.COMPARATOR})


class Stratum(str, Enum):
    """§8.3 replication strata. Never pooled, never trended."""

    PRIMARY_14B = "primary_14b"
    MATH_PARENT_7B = "math_parent_7b"
    GENERAL_PARENT_32B = "general_parent_32b"
    CALIBRATION = "calibration"


class AssayTier(str, Enum):
    """§8.5 assay stratification."""

    PRIMARY = "primary"
    SECONDARY_CONFIRMATORY = "secondary_confirmatory"
    EXPLORATORY = "exploratory"
    EXCLUDED = "excluded"
