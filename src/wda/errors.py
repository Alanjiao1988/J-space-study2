"""Exception hierarchy.

Every exception here corresponds to a governance rule in ``PROTOCOL_v1.1.md``. They are
deliberately *not* subclasses of anything catchable-by-accident such as ``ValueError``:
a protocol violation must never be silently swallowed by a generic ``except`` in a runner.
"""

from __future__ import annotations


class ProtocolViolation(Exception):
    """Base class: an action forbidden by the frozen protocol was attempted."""


class UnregisteredModelError(ProtocolViolation):
    """§15.1 — only checkpoints listed in the registry may be loaded."""


class DeniedModelError(ProtocolViolation):
    """§15.1 — the checkpoint is explicitly denied (e.g. the 1.5B, fact F1)."""


class PhaseViolation(ProtocolViolation):
    """§15.1 — the checkpoint may not be loaded in the current phase."""


class UnlockedRevisionError(ProtocolViolation):
    """§8.1 — a checkpoint must be pinned to a resolved revision before it is loaded."""


class BlindingViolation(ProtocolViolation):
    """§16 — Phase A touched a subject model, or computed D / G."""


class SealMismatch(ProtocolViolation):
    """§16 — the working tree does not match the sealed hash manifest."""


class RerunRefused(ProtocolViolation):
    """§11 — a scientific re-run, or an unregistered failure code, was requested."""


class ConformanceFailure(ProtocolViolation):
    """§10 Phase 0 / SC4 — a conformance fixture failed."""


class PrecisionViolation(ProtocolViolation):
    """§10 fixture ``noop_returns_zero`` / fact F7 — required fp32 accumulation absent."""


class BandRuleViolation(ProtocolViolation):
    """§6.3 — the automatic band rule produced an inadmissible band."""


class AggregationViolation(ProtocolViolation):
    """§6.2 — post-hoc selection between the two real-corpus lenses was attempted."""


class StoppingCondition(ProtocolViolation):
    """§12 — a registered stopping condition fired. Carries its identifier."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
