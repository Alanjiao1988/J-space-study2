"""Synthetic lifecycle tests, never actual Phase-0 model evidence."""

import json

import pytest

from wda.errors import BlindingViolation, RerunRefused, SealMismatch
from wda.governance import blind
from wda.governance.artifacts import EvidenceError, create_json, reference
from wda.phases import Phase
from wda.runtime.run import Run, RunViolation, completed_receipt


def noop_receipt(run):
    # Artificial raw captures test binding and byte checking only.
    for name in ("clean.f32", "noop.f32"):
        (run.root / name).write_bytes(b"\0\0\x80\x3f")
    return {
        "schema": "wda/noop/1", "run_id": run.run_id, "seal_root_sha256": run.seal_root_sha256,
        "model_key": run.config["config"]["model_key"], "revision": "a" * 40,
        "dtype": "float32", "shape": [1],
        "clean": reference(run.repository, (run.root / "clean.f32").relative_to(run.repository).as_posix()),
        "noop": reference(run.repository, (run.root / "noop.f32").relative_to(run.repository).as_posix()),
    }


def trial(run):
    return {
        "model_role": "calibration", "condition": "C_direct", "ablation_state": "clean",
        "lens_id": "toy-a", "item_id": "toy-1", "template_id": "toy", "target": None,
        "output_tokens": [7], "parsed": True, "correct": False, "kl_vs_clean": 0.0,
        "noop_bitexact": True, "noop_evidence": noop_receipt(run),
    }


def test_start_requires_seal(tmp_path, monkeypatch):
    monkeypatch.setenv("WDA_REPO_ROOT", str(tmp_path))
    with pytest.raises(SealMismatch):
        Run.start(Phase.PHASE_0, "toy")


def test_stage_override_cannot_start_confirmation(sealed_tree):
    with pytest.raises(RunViolation, match="override"):
        Run.start(Phase.PHASE_C, "toy", seal_stage="freeze1")


def test_phase_a_requires_blinding_and_phase0(sealed_tree, phase0_config):
    with pytest.raises(RunViolation, match="blinding"):
        Run.start(Phase.PHASE_A, "toy", config=phase0_config)
    with blind.phase_a_blinding(), pytest.raises(EvidenceError):
        Run.start(Phase.PHASE_A, "toy", config=phase0_config)


def test_blinding_precedes_evidential_stage(sealed_tree):
    with blind.phase_a_blinding(), pytest.raises(BlindingViolation):
        Run.start(Phase.PHASE_C, "toy")


def test_start_requires_sealed_settings(sealed_tree, phase0_config):
    phase0_config["seed"] = 999
    with pytest.raises(RunViolation, match="sealed config"):
        Run.start(Phase.PHASE_0, "toy", config=phase0_config)


def test_start_and_trial_are_create_only(sealed_tree, phase0_config, monkeypatch):
    monkeypatch.setattr("wda.runtime.run.make_run_id", lambda *_: "fixed-id")
    run = Run.start(Phase.PHASE_0, "toy", config=phase0_config)
    record = trial(run)
    line = run.log_trial(record)
    assert line["run_id"] == run.run_id and line["seal_hash"] == run.seal_root_sha256
    with pytest.raises(RunViolation, match="duplicate"):
        run.log_trial(record)
    with pytest.raises(RunViolation, match="create-only"):
        Run.start(Phase.PHASE_0, "toy", config=phase0_config)
    for name in ("config.json", "seal.json", "log.jsonl"):
        assert b"\r" not in (run.root / name).read_bytes()


def test_false_noop_stops_subsequent_work(sealed_tree, phase0_config):
    run = Run.start(Phase.PHASE_0, "toy", config=phase0_config)
    row = trial(run)
    row["noop_bitexact"] = False
    with pytest.raises(RunViolation):
        run.log_trial(row)
    with pytest.raises(RunViolation, match="unresolved"):
        run.log_trial(trial(run))
    assert len(list((run.root / "failures").glob("*.json"))) == 1


def test_config_change_blocks_run(sealed_tree, phase0_config):
    run = Run.start(Phase.PHASE_0, "toy", config=phase0_config)
    path = run.root / "config.json"
    config = json.loads(path.read_text())
    config["label"] = "tampered"
    path.write_text(json.dumps(config))
    with pytest.raises(RunViolation, match="configuration changed"):
        run.log_trial(trial(run))


def test_finish_rejects_string_pass_without_evidence(sealed_tree, phase0_config):
    run = Run.start(Phase.PHASE_0, "toy", config=phase0_config)
    with pytest.raises(EvidenceError):
        run.finish({"fixtures": "all pass"})
    assert not (run.root / "results.json").exists()


def test_unsealed_code_change_blocks_finish(sealed_tree, phase0_config):
    run = Run.start(Phase.PHASE_0, "toy", config=phase0_config)
    (sealed_tree / "src/wda/runner.py").write_text("changed")
    with pytest.raises(SealMismatch):
        run.finish({})


def test_structured_failure_authorizes_one_identical_retry(sealed_tree, phase0_config):
    run = Run.start(Phase.PHASE_0, "toy", config=phase0_config)
    failure = run.record_failure("IF-HARD-KILL", "scheduler preemption", retryable=True)
    entry = run.authorize_rerun("IF-HARD-KILL", "scheduler preemption", failure=failure)
    assert entry["attempt"] == 1
    with pytest.raises(RerunRefused, match="already consumed"):
        run.authorize_rerun("IF-HARD-KILL", "scheduler preemption", failure=failure)


def test_finished_engineering_receipt_is_not_qualification(governance_tree):
    path = governance_tree / "runs/engineering/toy/results.json"
    create_json(path, {"schema": "wda/results/2", "kind": "engineering", "status": "PASS", "phase": "phase0"})
    ref = reference(governance_tree, path.relative_to(governance_tree).as_posix())
    with pytest.raises(EvidenceError, match="model-facing"):
        completed_receipt(governance_tree, ref, Phase.PHASE_0)
