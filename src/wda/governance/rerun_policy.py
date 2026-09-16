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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from wda.errors import RerunRefused
from wda.governance.artifacts import (
    EvidenceError, check_ref, create_json, digest, identifier, local_path, read_json,
)


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

#: An additional diagnostic guard, NOT the authorization boundary. Immutable identity
#: and a registered persisted failure event below are authoritative.
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
    """Create-only on-disk attempt ledger; reconstruction cannot reset the budget.

    ``root`` is the run directory; ``identity`` is the immutable state.json
    identity. Authorization also requires a runtime-generated failure reference
    and the unchanged proposed identity. Free text never establishes eligibility.
    """

    run_id: str
    root: Optional[Path] = None
    identity: Optional[dict] = None

    @property
    def attempts(self) -> List[Dict[str, object]]:
        if self.root is None or self.identity is None:
            raise RerunRefused("rerun authorization requires a durable run identity and ledger")
        identifier(self.run_id)
        folder = local_path(self.root, "reruns", file=False)
        if not folder.exists():
            return []
        entries = []
        counts: Dict[str, int] = {}
        for index, path in enumerate(sorted(folder.iterdir()), 1):
            local_path(self.root, path.relative_to(self.root).as_posix())
            if path.name != f"{index:06d}.json":
                raise RerunRefused("rerun ledger is malformed or incomplete")
            item = read_json(path)
            code = item.get("code")
            counts[code] = counts.get(code, 0) + 1
            if (
                item.get("schema") != "wda/rerun-attempt/2"
                or item.get("identity") != self.identity or item.get("run_id") != self.run_id
                or code not in INFRASTRUCTURE_CODES or item.get("attempt") != counts[code]
                or item.get("sha256") != digest({k: v for k, v in item.items() if k != "sha256"})
            ):
                raise RerunRefused("rerun ledger identity/count/hash mismatch")
            check_ref(self.root, item.get("failure"))
            entries.append(item)
        return entries

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for attempt in self.attempts:
            code = str(attempt["code"])
            out[code] = out.get(code, 0) + 1
        return out

    def record(self, code: str, detail: str, failure: dict) -> Dict[str, object]:
        attempts = self.attempts
        spec = INFRASTRUCTURE_CODES[code]
        entry = {
            "schema": "wda/rerun-attempt/2",
            "run_id": self.run_id,
            "identity": self.identity,
            "code": code,
            "detail": detail,
            "attempt": self.counts().get(code, 0) + 1,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "failure": failure,
            "recovery": spec.recovery,
            "max_attempts": spec.max_attempts,
        }
        entry["sha256"] = digest(entry)
        try:
            create_json(self.root / "reruns" / f"{len(attempts) + 1:06d}.json", entry)
        except FileExistsError as exc:
            raise RerunRefused("concurrent rerun attempt; no second authorization was issued") from exc
        return entry

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": "wda/rerun_ledger/2",
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


def authorize(
    ledger: RerunLedger, code: str, reason: str, *,
    failure: Optional[dict] = None, proposed_identity: Optional[dict] = None,
) -> Dict[str, object]:
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
    if proposed_identity is None or proposed_identity != ledger.identity:
        raise RerunRefused("scientific re-run: immutable config/seal identity may not change")
    try:
        if ledger.root is None:
            raise EvidenceError("missing durable ledger")
        incident = read_json(check_ref(ledger.root, failure))
        if (
            incident.get("schema") != "wda/runtime-failure/1"
            or incident.get("code") != code or incident.get("retryable") is not True
            or incident.get("identity") != ledger.identity
            or incident.get("run_id") != ledger.run_id
            or not failure["path"].startswith("failures/")
        ):
            raise EvidenceError("code does not match an eligible recorded runtime failure")
        if any(item["failure"] == failure for item in ledger.attempts):
            raise EvidenceError("this failure already consumed its single restart authorization")
    except EvidenceError as exc:
        raise RerunRefused(str(exc)) from exc
    spec = INFRASTRUCTURE_CODES[code]
    used = ledger.counts().get(code, 0)
    if used >= spec.max_attempts:
        raise RerunRefused(
            f"{code} has already been used {used}/{spec.max_attempts} times for "
            f"run_id={ledger.run_id!r}; the pre-registered budget is exhausted."
        )
    return ledger.record(code, reason, failure)


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
