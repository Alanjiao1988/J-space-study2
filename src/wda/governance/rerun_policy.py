"""Re-run policy (§11, inheriting fact F13).

Two layers:

* **Infrastructure re-runs (permitted, pre-registered)** — hardware fault, OOM, killed
  process, checkpoint checksum failure. The trigger codes, the maximum attempt count and
  the recovery mode are frozen here at Freeze-1; every attempt is logged.
* **Scientific re-runs (forbidden)** — re-running because the result was unwelcome, or
  changing seed / band / k / δ / sample size / endpoint. Always refused.

An exit code that is not a registered infrastructure code may never re-run the same
``run_id``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional

from wda.errors import RerunRefused


@dataclass(frozen=True)
class FailureCode:
    code: str
    description: str
    recovery: str
    max_attempts: int


#: Frozen at Freeze-1. Adding a code after the seal changes ``src/wda/**`` and therefore
#: breaks the manifest — which is the intended behaviour.
INFRASTRUCTURE_CODES: Mapping[str, FailureCode] = {
    "IF-GPU-FAULT": FailureCode(
        "IF-GPU-FAULT",
        "CUDA/driver fault, XID error, or device fell off the bus.",
        "restart the identical run_id on a fresh device; no configuration change",
        3,
    ),
    "IF-OOM": FailureCode(
        "IF-OOM",
        "Out of device or host memory during a forward/backward pass.",
        "restart with a smaller micro-batch; the batch size must not alter numerics "
        "(verified by the bit-exact no-op check before the run resumes)",
        3,
    ),
    "IF-HARD-KILL": FailureCode(
        "IF-HARD-KILL",
        "Process killed by the scheduler/OS (SIGKILL, preemption, node drain).",
        "resume from the last flushed trial index in log.jsonl",
        5,
    ),
    "IF-CHECKPOINT-DIGEST": FailureCode(
        "IF-CHECKPOINT-DIGEST",
        "Checkpoint file digest did not match the locked revision.",
        "re-download at the locked revision and re-verify; never substitute a revision",
        2,
    ),
    "IF-STORAGE": FailureCode(
        "IF-STORAGE",
        "I/O error writing run artefacts (disk full, mount lost).",
        "restore storage and resume from the last flushed trial index",
        3,
    ),
    "IF-CPU-FALLBACK": FailureCode(
        "IF-CPU-FALLBACK",
        "Pre-registered canary: device unavailable, execution continued on CPU in fp32.",
        "continue on CPU; the bit-exact no-op fixture must still pass (fact F13 canary)",
        1,
    ),
}

#: Reasons that are always refused. Matching is substring-based and case-insensitive so
#: that a paraphrase in a runner does not slip through.
FORBIDDEN_REASON_MARKERS = (
    "result",
    "unexpected",
    "disappoint",
    "not significant",
    "insignificant",
    "underpower",
    "different seed",
    "new seed",
    "reseed",
    "change band",
    "another band",
    "change k",
    "tune k",
    "change delta",
    "adjust delta",
    "more samples",
    "add samples",
    "increase n",
    "other endpoint",
    "switch endpoint",
    "try again",
    "one more",
    "looks wrong",
)


@dataclass
class RerunLedger:
    """Per-``run_id`` attempt ledger, serialised into ``runs/<run_id>/log.jsonl``."""

    run_id: str
    attempts: List[Dict[str, object]] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for attempt in self.attempts:
            code = str(attempt["code"])
            out[code] = out.get(code, 0) + 1
        return out

    def record(self, code: str, detail: str) -> Dict[str, object]:
        entry = {
            "run_id": self.run_id,
            "code": code,
            "detail": detail,
            "attempt": self.counts().get(code, 0) + 1,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.attempts.append(entry)
        return entry

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": "wda/rerun_ledger/1",
            "run_id": self.run_id,
            "attempts": list(self.attempts),
            "counts": self.counts(),
        }


def classify(reason: str) -> Optional[str]:
    """Return the forbidden marker found in ``reason``, or ``None``."""
    lowered = reason.lower()
    for marker in FORBIDDEN_REASON_MARKERS:
        if marker in lowered:
            return marker
    return None


def authorize(ledger: RerunLedger, code: str, reason: str) -> Dict[str, object]:
    """Authorise one infrastructure re-run of ``ledger.run_id``, or raise.

    Raises
    ------
    RerunRefused
        If the code is unregistered, the attempt budget is exhausted, or the stated reason
        matches a scientific-re-run marker.
    """
    if code not in INFRASTRUCTURE_CODES:
        raise RerunRefused(
            f"{code!r} is not a registered infrastructure failure code. Registered codes: "
            f"{sorted(INFRASTRUCTURE_CODES)}. Any other exit code may not re-run "
            f"run_id={ledger.run_id!r} (§11)."
        )
    marker = classify(reason)
    if marker is not None:
        raise RerunRefused(
            f"the stated reason matches the forbidden marker {marker!r}; this is a "
            "scientific re-run and is refused unconditionally (§11). Report the result "
            "as it stands — SC2/SC3 are legitimate outcomes."
        )
    spec = INFRASTRUCTURE_CODES[code]
    used = ledger.counts().get(code, 0)
    if used >= spec.max_attempts:
        raise RerunRefused(
            f"{code} has already been used {used}/{spec.max_attempts} times for "
            f"run_id={ledger.run_id!r}; the pre-registered budget is exhausted."
        )
    entry = ledger.record(code, reason)
    entry["recovery"] = spec.recovery
    entry["max_attempts"] = spec.max_attempts
    return entry


def policy_snapshot() -> Dict[str, object]:
    """Frozen policy, embedded into every ``config.json``."""
    return {
        "schema": "wda/rerun_policy/1",
        "infrastructure_codes": {
            code: {
                "description": spec.description,
                "recovery": spec.recovery,
                "max_attempts": spec.max_attempts,
            }
            for code, spec in sorted(INFRASTRUCTURE_CODES.items())
        },
        "scientific_reruns": "forbidden (§11)",
        "forbidden_reason_markers": list(FORBIDDEN_REASON_MARKERS),
    }


def _main(argv: Optional[list] = None) -> int:  # pragma: no cover - CLI
    print(json.dumps(policy_snapshot(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(_main())
