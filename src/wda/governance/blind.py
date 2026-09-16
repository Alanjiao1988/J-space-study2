"""Phase-A blinding (§10, §16).

Two prohibitions are enforced in-process:

* the registry must refuse to hand back ``treatment`` / ``comparator`` entries, and
* :mod:`wda.stats.decision` must refuse to run at all — Phase A may not compute or look at
  any ``D`` or any ``G``.

Parameter selection during Phase A is permitted to consult only structural statistics
(§6.3), the causal positive control (Phase-0 fixture 5), and the general-damage check.
Those three surfaces are declared here so that a violation is a hard error rather than a
matter of discipline.

Usage::

    from wda.governance import blind

    with blind.phase_a_blinding():
        ...  # calibration work; resolving a subject or calling decide() raises

The state is process-global by design: a blinded phase must not be escapable by handing an
object to another module.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional

from wda.errors import BlindingViolation
from wda.phases import SUBJECT_ROLES, Phase, Role

WITHHELD_ROLES = frozenset({*SUBJECT_ROLES, Role.PARENT_ANCHOR})

_LOCK = threading.RLock()

#: Environment variable that re-establishes blinding in a freshly spawned worker process.
BLIND_ENV_VAR = "WDA_BLINDING"

#: Surfaces Phase A is allowed to consult when selecting parameters (§10 Phase A).
PERMITTED_CALIBRATION_SURFACES = frozenset(
    {
        "structural_statistics",  # §6.3 (a)-(d)
        "causal_positive_control",  # Phase-0 fixture 5
        "general_damage_check",  # §7 community-added control
        "parseability",  # §9.4 endpoint 1 - operational feasibility, not an effect
        "paired_variance",  # §9.3 variance for the power calculation only
    }
)

#: Surfaces that are *never* admissible while blinded.
FORBIDDEN_CALIBRATION_SURFACES = frozenset(
    {"D", "G", "G_primary", "G_gen", "delta_accuracy", "primary_endpoint"}
)


@dataclass
class _BlindState:
    active: bool = False
    phase: Optional[Phase] = None
    withheld_roles: frozenset = field(default_factory=frozenset)
    violations: List[Dict[str, str]] = field(default_factory=list)


_STATE = _BlindState()


def _env_blinded() -> bool:
    return os.environ.get(BLIND_ENV_VAR, "").strip().lower() in {"1", "true", "on", "phasea"}


def is_active() -> bool:
    """``True`` while blinding is in force (in-process flag or inherited env var)."""
    with _LOCK:
        return _STATE.active or _env_blinded()


def withheld_roles() -> frozenset:
    """Roles the registry must refuse. Empty when blinding is off."""
    with _LOCK:
        if _STATE.active:
            return _STATE.withheld_roles
    return WITHHELD_ROLES if _env_blinded() else frozenset()


def violations() -> List[Dict[str, str]]:
    """Recorded violation attempts, for the phase report (§17 item 8)."""
    with _LOCK:
        return list(_STATE.violations)


def _record(kind: str, detail: str) -> None:
    with _LOCK:
        _STATE.violations.append(
            {
                "kind": kind,
                "detail": detail,
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "phase": _STATE.phase.value if _STATE.phase else "unknown",
            }
        )


def guard_decision(caller: str) -> None:
    """Raise if a decision-rule computation is attempted while blinded (§16)."""
    if is_active():
        _record("decision_while_blind", caller)
        raise BlindingViolation(
            f"{caller} was called while Phase-A blinding is active. Phase A has zero "
            "evidential weight and may not compute or inspect D or G (§10, §16)."
        )


def guard_model(role: Role, key: str) -> None:
    """Refuse and record non-calibration access BEFORE trusting a phase argument."""
    if is_active() and role is not Role.CALIBRATION:
        _record("model_while_blind", key)
        raise BlindingViolation(
            f"blinding is active: role {role.value!r} is withheld; cannot resolve {key!r}"
        )


def guard_surface(surface: str) -> None:
    """Raise if a calibration decision consults a forbidden surface (§10 Phase A)."""
    if not is_active():
        return
    if surface in FORBIDDEN_CALIBRATION_SURFACES or surface not in PERMITTED_CALIBRATION_SURFACES:
        _record("forbidden_surface", surface)
        raise BlindingViolation(
            f"surface {surface!r} may not inform Phase-A parameter selection. "
            f"Permitted surfaces: {sorted(PERMITTED_CALIBRATION_SURFACES)}."
        )


def guard_phase(phase: Phase) -> None:
    """Raise if a phase that carries evidential weight is entered while blinded."""
    if is_active() and phase in (Phase.PHASE_B, Phase.PHASE_C, Phase.PHASE_D):
        _record("evidential_phase_while_blind", phase.value)
        raise BlindingViolation(
            f"cannot enter {phase.value} while Phase-A blinding is active; "
            "exit the blinding context first and re-seal."
        )


@contextmanager
def phase_a_blinding(*, propagate_to_subprocesses: bool = True) -> Iterator[_BlindState]:
    """Activate Phase-A blinding for the duration of the block."""
    with _LOCK:
        if _STATE.active:
            raise BlindingViolation("blinding is already active; nested activation is refused")
        _STATE.active = True
        _STATE.phase = Phase.PHASE_A
        _STATE.withheld_roles = WITHHELD_ROLES
    previous_env = os.environ.get(BLIND_ENV_VAR)
    if propagate_to_subprocesses:
        os.environ[BLIND_ENV_VAR] = "1"
    try:
        yield _STATE
    finally:
        with _LOCK:
            _STATE.active = False
            _STATE.phase = None
            _STATE.withheld_roles = frozenset()
        if propagate_to_subprocesses:
            if previous_env is None:
                os.environ.pop(BLIND_ENV_VAR, None)
            else:
                os.environ[BLIND_ENV_VAR] = previous_env


def blinding_receipt() -> Dict[str, object]:
    """Receipt embedded in the Phase-A ``results.json`` (§17)."""
    return {
        "schema": "wda/blinding_receipt/1",
        "active": is_active(),
        "withheld_roles": sorted(r.value for r in withheld_roles()),
        "permitted_surfaces": sorted(PERMITTED_CALIBRATION_SURFACES),
        "forbidden_surfaces": sorted(FORBIDDEN_CALIBRATION_SURFACES),
        "violations": violations(),
    }


def _main(argv: Optional[list] = None) -> int:  # pragma: no cover - CLI
    print(json.dumps(blinding_receipt(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(_main())
