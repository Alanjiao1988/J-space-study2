"""Run scaffolding — the ``runs/<phase>/<run_id>/`` layout (§14, §15.4, §16).

Every run produces exactly four files::

    runs/<phase>/<run_id>/config.json    frozen configuration + registry + policy snapshots
    runs/<phase>/<run_id>/seal.json      the verified seal header (seal precedes the run)
    runs/<phase>/<run_id>/log.jsonl      one line per trial, per §15.4
    runs/<phase>/<run_id>/results.json   endpoints, controls, decisions, stopping condition

:class:`Run` is the only sanctioned entry point. Constructing one:

* verifies the seal and refuses to start on mismatch (§16);
* refuses to enter an evidential phase while Phase-A blinding is active;
* records the re-run ledger, so an unregistered failure code cannot restart the run (§11).
"""

from __future__ import annotations

import getpass
import json
import platform
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from wda.assays import registry as assay_registry
from wda.errors import ProtocolViolation
from wda.governance import blind, rerun_policy, seal
from wda.models import registry as model_registry
from wda.paths import runs_dir, write_json_lf
from wda.phases import EVIDENTIAL_PHASES, Phase

SCHEMA = "wda/run/1"


class RunViolation(ProtocolViolation):
    """A run was started or written in a way the protocol forbids."""


def make_run_id(phase: Phase, label: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in label)
    return f"{phase.value}-{safe}-{stamp}"


def environment_snapshot() -> Dict[str, object]:
    """Recorded so an infrastructure re-run can be told apart from a different machine."""
    try:
        user = getpass.getuser()
    except Exception:  # pragma: no cover - depends on the host
        user = "unknown"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "user": user,
    }


@dataclass
class Run:
    """One sealed, logged execution."""

    phase: Phase
    run_id: str
    config: Dict[str, Any]
    root: Path
    seal_stage: str
    seal_root_sha256: str
    ledger: rerun_policy.RerunLedger
    _log_handle: Optional[Any] = field(default=None, repr=False)

    # ---------------------------------------------------------------- construction ---
    @classmethod
    def start(
        cls,
        phase: Phase,
        label: str,
        *,
        config: Optional[Mapping[str, Any]] = None,
        seal_stage: Optional[str] = None,
        base_dir: Optional[Path] = None,
    ) -> "Run":
        """Verify the seal, create the run directory, and write ``config.json``/``seal.json``."""
        stage = seal_stage or ("freeze2" if phase in EVIDENTIAL_PHASES else "freeze1")

        if phase in EVIDENTIAL_PHASES:
            blind.guard_phase(phase)

        root_sha = seal.require(stage)
        sealed = seal.load(stage)

        run_id = make_run_id(phase, label)
        base = Path(base_dir) if base_dir else runs_dir()
        root = base / phase.value / run_id
        if root.exists():
            raise RunViolation(
                f"{root} already exists. A run_id is create-only; re-running it requires a "
                "registered infrastructure failure code (§11)."
            )
        root.mkdir(parents=True)

        payload: Dict[str, Any] = {
            "schema": SCHEMA,
            "run_id": run_id,
            "phase": phase.value,
            "label": label,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seal_stage": stage,
            "seal_root_sha256": root_sha,
            "environment": environment_snapshot(),
            "registry": model_registry.registry_snapshot(),
            "assays": assay_registry.snapshot(),
            "rerun_policy": rerun_policy.policy_snapshot(),
            "blinding": blind.blinding_receipt(),
            "config": dict(config or {}),
        }
        write_json_lf(root / "config.json", payload)
        write_json_lf(
            root / "seal.json",
            {
                k: sealed.get(k)
                for k in ("schema", "stage", "file_count", "root_sha256", "sealed_at", "note")
            },
        )
        return cls(
            phase=phase,
            run_id=run_id,
            config=payload,
            root=root,
            seal_stage=stage,
            seal_root_sha256=root_sha,
            ledger=rerun_policy.RerunLedger(run_id=run_id),
        )

    # ------------------------------------------------------------------- logging ---
    def log_trial(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        """Append one §15.4 trial line. The seal hash is stamped, never accepted."""
        required = {
            "run_id",
            "phase",
            "model_role",
            "condition",
            "ablation_state",
            "lens_id",
            "item_id",
            "template_id",
            "target",
            "output_tokens",
            "parsed",
            "correct",
            "kl_vs_clean",
            "noop_bitexact",
            "seal_hash",
        }
        missing = sorted(required - set(record))
        if missing:
            raise RunViolation(f"trial record is missing §15.4 fields: {missing}")
        line = dict(record)
        line["run_id"] = self.run_id
        line["phase"] = self.phase.value
        line["seal_hash"] = self.seal_root_sha256
        if self._log_handle is None:
            self._log_handle = (self.root / "log.jsonl").open("a", encoding="utf-8", newline="\n")
        self._log_handle.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")
        self._log_handle.flush()
        return line

    def read_trials(self) -> List[Dict[str, Any]]:
        path = self.root / "log.jsonl"
        if not path.exists():
            return []
        out: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if raw:
                    out.append(json.loads(raw))
        return out

    # ------------------------------------------------------------------ re-runs ---
    def authorize_rerun(self, code: str, reason: str) -> Dict[str, object]:
        """§11 — only registered infrastructure codes, with a non-scientific reason."""
        return rerun_policy.authorize(self.ledger, code, reason)

    # ------------------------------------------------------------------ results ---
    def finish(
        self,
        results: Mapping[str, Any],
        *,
        stopping_condition: Optional[str] = None,
    ) -> Path:
        """Re-verify the seal, then write ``results.json``.

        Re-verification at the end is deliberate: it proves that no file covered by the
        manifest changed *while* the run was executing.
        """
        seal.require(self.seal_stage)
        payload = {
            "schema": "wda/results/1",
            "run_id": self.run_id,
            "phase": self.phase.value,
            "seal_stage": self.seal_stage,
            "seal_root_sha256": self.seal_root_sha256,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stopping_condition": stopping_condition,
            "rerun_ledger": self.ledger.to_dict(),
            "blinding": blind.blinding_receipt(),
            "results": dict(results),
        }
        path = self.root / "results.json"
        write_json_lf(path, payload)
        self.close()
        return path

    def close(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


__all__ = ["Run", "RunViolation", "SCHEMA", "environment_snapshot", "make_run_id"]
