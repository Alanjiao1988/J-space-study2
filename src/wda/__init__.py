"""Fixed-Checkpoint Workspace-Dependence Audit.

Implementation of ``PROTOCOL_v1.1.md``. Nothing in this package may be changed after the
Freeze-1 seal without invalidating ``FREEZE-1.json`` (see :mod:`wda.governance.seal`).
"""

from wda.errors import (
    AggregationViolation,
    BandRuleViolation,
    BlindingViolation,
    ConformanceFailure,
    DeniedModelError,
    PhaseViolation,
    PrecisionViolation,
    ProtocolViolation,
    RerunRefused,
    SealMismatch,
    UnlockedRevisionError,
    UnregisteredModelError,
)
from wda.phases import Phase, Role, Stratum

__version__ = "1.1.0"
PROTOCOL_VERSION = "v1.1"

__all__ = [
    "PROTOCOL_VERSION",
    "Phase",
    "Role",
    "Stratum",
    "AggregationViolation",
    "BandRuleViolation",
    "BlindingViolation",
    "ConformanceFailure",
    "DeniedModelError",
    "PhaseViolation",
    "PrecisionViolation",
    "ProtocolViolation",
    "RerunRefused",
    "SealMismatch",
    "UnlockedRevisionError",
    "UnregisteredModelError",
    "__version__",
]
