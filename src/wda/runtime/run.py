"""Create-only scientific run records with explicit qualification prerequisites.

An engineering canary is not a scientific Run and cannot produce a qualifying
receipt. Local JSON integrity does not attest hardware execution; the sealed
model-facing runner is responsible for the provenance of its captured evidence.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from wda.errors import ProtocolViolation
from wda.governance import blind, rerun_policy, seal
from wda.governance.artifacts import (
    EvidenceError, canonical, check_ref, create_json, digest, identifier,
    local_path, read_json, reference,
)
from wda.governance.evidence import (
    model_binding, validate_parameters, validate_phase0, validate_phase_b,
)
from wda.intervene.subspace import require_scientific_interventions
from wda.paths import repo_root, runs_dir
from wda.phases import Phase

SCHEMA = "wda/run/2"
MODEL_PHASES = {Phase.PHASE_0, Phase.PHASE_A, Phase.PHASE_B, Phase.PHASE_C, Phase.PHASE_D}


class RunViolation(ProtocolViolation):
    """An incomplete, altered or unauthorized scientific run was requested."""


def make_run_id(phase: Phase, label: str) -> str:
    identifier(label)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{phase.value}-{label}-{stamp}-{uuid.uuid4().hex[:8]}"


def environment_snapshot() -> dict:
    return {"python": sys.version.split()[0], "platform": platform.platform()}


def _expected_stage(phase: Phase) -> str:
    if phase not in MODEL_PHASES:
        raise RunViolation("only model-calling phases can create a scientific Run")
    return "freeze2" if phase in {Phase.PHASE_C, Phase.PHASE_D} else "freeze1"


def _prerequisites(root: Path, phase: Phase, config: dict) -> None:
    if phase is Phase.PHASE_A:
        if not blind.is_active():
            raise RunViolation("Phase A requires active process blinding")
        completed_receipt(root, config.get("phase0"), Phase.PHASE_0)
    elif phase is Phase.PHASE_B:
        prior = completed_receipt(root, config.get("phase_a"), Phase.PHASE_A)
        validate_parameters(root, config.get("parameters"))
        if config["parameters"] != prior["evidence"]["parameters"]:
            raise RunViolation("Phase-B parameters differ from the completed calibration")
    elif phase in {Phase.PHASE_C, Phase.PHASE_D}:
        # A token-direction provider and a confirmation runner are not implemented
        # merely by having a statistics module or a JSON PASS field.
        require_scientific_interventions()


def _validate_evidence(root: Path, value: dict, config: dict, files: set[str]) -> None:
    phase = Phase(config["phase"])
    if phase is Phase.PHASE_0:
        validate_phase0(root, value, config, files)
    elif phase is Phase.PHASE_A:
        if value.get("schema") != "wda/phase-a-evidence/1":
            raise EvidenceError("Phase A requires a registered calibration evidence object")
        validate_parameters(root, value.get("parameters"))
        if value.get("phase0") != config["config"].get("phase0"):
            raise EvidenceError("Phase-A evidence must bind the same Phase-0 receipt")
        completed_receipt(root, value["phase0"], Phase.PHASE_0)
        for name in ("structural_statistics", "causal_positive_control", "general_damage_check",
                     "paired_variance", "parseability"):
            artifact = read_json(check_ref(root, value.get(name)))
            if artifact.get("kind") != "model" or artifact.get("model_key") != config["config"]["model_key"]:
                raise EvidenceError(f"{name} must be a calibration-model observation")
            if artifact.get("revision") != config["config"]["revision"]:
                raise EvidenceError(f"{name} has the wrong calibration revision")
    elif phase is Phase.PHASE_B:
        validate_phase_b(root, value, config, files)
    else:
        require_scientific_interventions()


def completed_receipt(root: Path, ref: Any, phase: Phase) -> dict:
    """Validate a completed result, its immutable run config and all prerequisite evidence."""
    path = check_ref(root, ref)
    result = read_json(path)
    if (
        result.get("schema") != "wda/results/2" or result.get("phase") != phase.value
        or result.get("status") != "PASS" or result.get("kind") != "model"
    ):
        raise EvidenceError("a completed model-facing PASS receipt for the expected phase is required")
    config_path = check_ref(root, result.get("run_config"))
    if config_path.parent != path.parent or config_path.name != "config.json":
        raise EvidenceError("result and immutable config must belong to the same run directory")
    config = read_json(config_path)
    if config.get("schema") != SCHEMA or config.get("phase") != phase.value:
        raise EvidenceError("result configuration has the wrong schema/phase")
    if result.get("identity") != {
        "config_sha256": digest(config), "seal_root_sha256": config.get("seal_root_sha256")
    } or result.get("run_id") != config.get("run_id"):
        raise EvidenceError("result/config identity mismatch")
    stage = _expected_stage(phase)
    if config.get("seal_stage") != stage or seal.require(stage, root=root) != config.get("seal_root_sha256"):
        raise EvidenceError("receipt does not match the current verified stage seal")
    run_root = path.parent
    ledger = rerun_policy.RerunLedger(config["run_id"], root=run_root, identity=result["identity"])
    authorized = {attempt["failure"]["path"] for attempt in ledger.attempts}
    for failure in (run_root / "failures").glob("*.json"):
        if failure.relative_to(run_root).as_posix() not in authorized:
            raise EvidenceError("run contains an unresolved runtime failure")
    files = {entry["path"] for entry in seal.load(stage, root=root)["files"]}
    _validate_evidence(root, result.get("evidence", {}), config, files)
    return result


@dataclass
class Run:
    phase: Phase
    run_id: str
    config: dict
    root: Path
    seal_stage: str
    seal_root_sha256: str
    ledger: rerun_policy.RerunLedger
    identity: dict
    repository: Path
    _closed: bool = False

    @classmethod
    def start(
        cls, phase: Phase, label: str, *, config: Mapping[str, Any] | None = None,
        seal_stage: str | None = None, base_dir: Path | None = None,
    ) -> "Run":
        stage = _expected_stage(phase)
        blind.guard_phase(phase)
        if seal_stage is not None and seal_stage != stage:
            raise RunViolation(f"{phase.value} requires {stage}; stage override refused")
        root = repo_root()
        root_sha = seal.require(stage, root=root)
        settings = dict(config or {})
        binding = model_binding(root, settings, phase)
        settings.update(binding)
        _prerequisites(root, phase, settings)
        # The model configuration itself must be a registered artifact, not
        # unchecked keyword arguments that change after sealing.
        candidate = settings.get("config_artifact")
        declared = read_json(check_ref(root, candidate))
        files = {entry["path"] for entry in seal.load(stage, root=root)["files"]}
        runtime_links = {"config_artifact", "phase0", "phase_a"}
        if phase is Phase.PHASE_B:
            runtime_links.add("parameters")
        if candidate["path"] not in files or declared != {k: v for k, v in settings.items() if k not in runtime_links}:
            raise RunViolation("run configuration must match its sealed config artifact")
        run_id = make_run_id(phase, label)
        base = Path(base_dir).absolute() if base_dir else runs_dir()
        if not base.resolve().is_relative_to(root.resolve()):
            raise RunViolation("run records must be inside the repository")
        run_root = local_path(root, (base / phase.value / run_id).relative_to(root).as_posix(), file=False)
        payload = {
            "schema": SCHEMA, "kind": "model", "run_id": run_id, "phase": phase.value,
            "label": label, "started_at": datetime.now(timezone.utc).isoformat(),
            "seal_stage": stage, "seal_root_sha256": root_sha,
            "environment": environment_snapshot(), "blinding": blind.blinding_receipt(),
            "rerun_policy": rerun_policy.policy_snapshot(), "config": settings,
        }
        canonical(payload)
        identity = {"config_sha256": digest(payload), "seal_root_sha256": root_sha}
        try:
            run_root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise RunViolation("run_id is create-only") from exc
        create_json(run_root / "config.json", payload)
        create_json(run_root / "seal.json", seal.load(stage, root=root))
        (run_root / "log.jsonl").touch(exist_ok=False)
        ledger = rerun_policy.RerunLedger(run_id, root=run_root, identity=identity)
        return cls(phase, run_id, payload, run_root, stage, root_sha, ledger, identity, root)

    def _live(self, *, allow_failures: bool = False) -> None:
        if self._closed or (self.root / "results.json").exists():
            raise RunViolation("finished or closed run is not writable")
        if digest(read_json(self.root / "config.json")) != self.identity["config_sha256"]:
            raise RunViolation("immutable run configuration changed")
        if seal.require(self.seal_stage, root=self.repository) != self.seal_root_sha256:
            raise RunViolation("run seal identity changed")
        blind.guard_phase(self.phase)
        if self.phase is Phase.PHASE_A and not blind.is_active():
            raise RunViolation("Phase-A blinding must remain active")
        if not allow_failures:
            handled = {a["failure"]["path"] for a in self.ledger.attempts}
            for path in (self.root / "failures").glob("*.json"):
                if path.relative_to(self.root).as_posix() not in handled:
                    raise RunViolation("unresolved runtime failure; run cannot continue")

    def log_trial(self, record: Mapping[str, Any]) -> dict:
        self._live()
        required = {
            "model_role", "condition", "ablation_state", "lens_id", "item_id", "template_id",
            "target", "output_tokens", "parsed", "correct", "kl_vs_clean", "noop_bitexact",
        }
        if not required <= record.keys():
            raise RunViolation(f"missing trial fields: {sorted(required - record.keys())}")
        line = dict(record)
        for key, value in (("run_id", self.run_id), ("phase", self.phase.value), ("seal_hash", self.seal_root_sha256)):
            if key in line and line[key] != value:
                raise RunViolation(f"trial {key} does not match this run")
            line[key] = value
        if (
            type(line["parsed"]) is not bool or type(line["correct"]) is not bool
            or line["correct"] and not line["parsed"] or line["noop_bitexact"] is not True
        ):
            self.record_failure("INVALID_TRIAL", "Invalid parsed/correct/no-op evidence.", retryable=False)
            raise RunViolation("trial requires consistent boolean endpoints and a demonstrated bit-exact no-op")
        from wda.governance.evidence import check_noop
        check_noop(self.repository, line.get("noop_evidence"), self.config)
        if line["condition"] not in {"C_direct", "C_frozen", "C_gen", "C_direct_prefill"}:
            raise RunViolation("unregistered condition")
        if line["ablation_state"] not in {"clean", "ablated"}:
            raise RunViolation("unregistered ablation state")
        from wda.models.registry import resolve
        entry = resolve(self.config["config"]["model_key"], self.phase)
        if line["model_role"] != entry.role.value:
            raise RunViolation("trial model role does not match the loaded checkpoint")
        if not isinstance(line["output_tokens"], list) or any(type(i) is not int or i < 0 for i in line["output_tokens"]):
            raise RunViolation("output_tokens must contain nonnegative integer IDs")
        kl = line["kl_vs_clean"]
        if type(kl) not in (int, float) or not math.isfinite(kl) or kl < 0:
            raise RunViolation("kl_vs_clean must be finite and nonnegative")
        key = tuple(line[name] for name in ("model_role", "condition", "ablation_state", "lens_id", "item_id", "template_id", "target"))
        for name in ("lens_id", "item_id", "template_id"):
            identifier(line[name])
        lock = self.root / "journal.lock"
        try:
            guard = lock.open("xb")
        except FileExistsError as exc:
            raise RunViolation("another writer owns the run journal") from exc
        try:
            for old in self.read_trials():
                old_key = tuple(old[name] for name in ("model_role", "condition", "ablation_state", "lens_id", "item_id", "template_id", "target"))
                if key == old_key:
                    raise RunViolation("duplicate trial key; no scientific rerun allowed")
            with (self.root / "log.jsonl").open("ab") as stream:
                stream.write(canonical(line) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            guard.close()
            lock.unlink()
        return line

    def read_trials(self) -> list[dict]:
        result = []
        for raw in (self.root / "log.jsonl").read_bytes().splitlines(keepends=True):
            if not raw.endswith(b"\n"):
                raise RunViolation("incomplete journal record; recovery requires a recorded incident")
            value = json.loads(raw)
            canonical(value)
            result.append(value)
        return result

    def record_failure(self, code: str, detail: str, *, retryable: bool) -> dict:
        if self._closed or (self.root / "results.json").exists():
            raise RunViolation("cannot record an incident on a closed run")
        if retryable and code not in rerun_policy.INFRASTRUCTURE_CODES:
            raise RunViolation("only a registered infrastructure code can be retryable")
        relative = f"failures/{uuid.uuid4().hex}.json"
        create_json(self.root / relative, {
            "schema": "wda/runtime-failure/1", "run_id": self.run_id, "identity": self.identity,
            "code": code, "detail": detail, "retryable": retryable,
            "at": datetime.now(timezone.utc).isoformat(),
        })
        return reference(self.root, relative)

    def authorize_rerun(self, code: str, reason: str, *, failure: dict) -> dict:
        self._live(allow_failures=True)
        return rerun_policy.authorize(
            self.ledger, code, reason, failure=failure, proposed_identity=self.identity,
        )

    def finish(self, results: Mapping[str, Any], *, stopping_condition: str | None = None) -> Path:
        self._live()
        evidence = dict(results)
        if stopping_condition is not None:
            raise RunViolation("failed/stopped runs must retain incident records, not a PASS receipt")
        files = {e["path"] for e in seal.load(self.seal_stage, root=self.repository)["files"]}
        _validate_evidence(self.repository, evidence, self.config, files)
        config_rel = (self.root / "config.json").relative_to(self.repository).as_posix()
        path = create_json(self.root / "results.json", {
            "schema": "wda/results/2", "kind": "model", "status": "PASS",
            "run_id": self.run_id, "phase": self.phase.value, "identity": self.identity,
            "run_config": reference(self.repository, config_rel), "evidence": evidence,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        self.close()
        return path

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, _traceback) -> None:
        if exc is not None and not self._closed:
            self.record_failure("UNHANDLED_EXCEPTION", type(exc).__name__, retryable=False)
        self.close()
