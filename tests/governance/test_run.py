"""Run scaffolding tests (§14, §15.4, §16).

The run is the point where seal, blinding and the re-run policy have to hold together, so
these are governance tests rather than plumbing tests.
"""

from __future__ import annotations

import json

import pytest

from wda.errors import BlindingViolation, RerunRefused, SealMismatch
from wda.governance import blind, seal
from wda.phases import Phase
from wda.runtime.run import Run, RunViolation


@pytest.fixture
def sealed_tree(tmp_path, monkeypatch):
    """A minimal repository tree with a valid Freeze-1 seal."""
    root = tmp_path / "repo"
    (root / "configs" / "freeze1").mkdir(parents=True)
    (root / "src" / "wda").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "PROTOCOL_v1.1.md").write_text("protocol", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]", encoding="utf-8")
    (root / "configs" / "freeze1" / "decision.json").write_text('{"delta": 0.1}', encoding="utf-8")
    (root / "src" / "wda" / "m.py").write_text("x = 1\n", encoding="utf-8")
    (root / "tests" / "t.py").write_text("y = 2\n", encoding="utf-8")
    seal.write("freeze1", root=root)
    monkeypatch.setenv("WDA_REPO_ROOT", str(root))
    return root


def make_record(**overrides):
    record = {
        "run_id": "placeholder",
        "phase": "placeholder",
        "model_role": "treatment",
        "condition": "C_direct",
        "ablation_state": "ablated",
        "lens_id": "real_a",
        "item_id": "confirm-3s-abc123",
        "template_id": "arith_chain_v1",
        "target": None,
        "output_tokens": [16, 17],
        "parsed": True,
        "correct": False,
        "kl_vs_clean": 0.0412,
        "noop_bitexact": True,
        "seal_hash": "not-this-one",
    }
    record.update(overrides)
    return record


class TestRun:
    def test_start_writes_config_and_seal(self, sealed_tree):
        run = Run.start(Phase.PHASE_0, "conformance", config={"k": 0})
        assert (run.root / "config.json").exists()
        assert (run.root / "seal.json").exists()
        payload = json.loads((run.root / "config.json").read_text(encoding="utf-8"))
        assert payload["config"] == {"k": 0}
        assert payload["seal_root_sha256"] == run.seal_root_sha256
        # The registry, assay tiers and re-run policy are captured with the run.
        assert payload["registry"]["denylist"]
        assert payload["assays"]["primary"] == ["primary_interaction"]
        assert payload["rerun_policy"]["scientific_reruns"].startswith("forbidden")
        run.close()

    def test_start_refuses_without_a_seal(self, tmp_path, monkeypatch):
        root = tmp_path / "unsealed"
        root.mkdir()
        (root / "PROTOCOL_v1.1.md").write_text("p", encoding="utf-8")
        (root / "pyproject.toml").write_text("[project]", encoding="utf-8")
        monkeypatch.setenv("WDA_REPO_ROOT", str(root))
        with pytest.raises(SealMismatch, match="has not been sealed"):
            Run.start(Phase.PHASE_0, "x")

    def test_start_refuses_after_the_tree_changes(self, sealed_tree):
        (sealed_tree / "src" / "wda" / "m.py").write_text("x = 2\n", encoding="utf-8")
        with pytest.raises(SealMismatch, match="changed"):
            Run.start(Phase.PHASE_0, "x")

    def test_evidential_phase_is_refused_while_blinded(self, sealed_tree):
        with blind.phase_a_blinding():
            with pytest.raises(BlindingViolation, match="cannot enter"):
                Run.start(Phase.PHASE_C, "confirmation", seal_stage="freeze1")

    def test_phase_a_run_records_the_blinding_receipt(self, sealed_tree):
        with blind.phase_a_blinding():
            run = Run.start(Phase.PHASE_A, "calibration")
            payload = json.loads((run.root / "config.json").read_text(encoding="utf-8"))
            assert payload["blinding"]["active"] is True
            assert sorted(payload["blinding"]["withheld_roles"]) == ["comparator", "treatment"]
            run.close()

    def test_log_stamps_run_id_and_seal_hash(self, sealed_tree):
        run = Run.start(Phase.PHASE_0, "x")
        line = run.log_trial(make_record())
        assert line["run_id"] == run.run_id
        assert line["phase"] == Phase.PHASE_0.value
        assert line["seal_hash"] == run.seal_root_sha256   # the caller's value is overwritten
        assert run.read_trials()[0]["item_id"] == "confirm-3s-abc123"
        run.close()

    def test_log_refuses_a_record_missing_required_fields(self, sealed_tree):
        run = Run.start(Phase.PHASE_0, "x")
        incomplete = make_record()
        del incomplete["noop_bitexact"]
        with pytest.raises(RunViolation, match="noop_bitexact"):
            run.log_trial(incomplete)
        run.close()

    def test_finish_reverifies_the_seal(self, sealed_tree):
        run = Run.start(Phase.PHASE_0, "x")
        run.log_trial(make_record())
        (sealed_tree / "src" / "wda" / "m.py").write_text("x = 99\n", encoding="utf-8")
        with pytest.raises(SealMismatch):
            run.finish({"ok": True})

    def test_finish_writes_results_with_the_ledger(self, sealed_tree):
        run = Run.start(Phase.PHASE_0, "x")
        run.authorize_rerun("IF-HARD-KILL", "node preempted by the scheduler")
        path = run.finish({"fixtures": "all pass"}, stopping_condition=None)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["results"] == {"fixtures": "all pass"}
        assert payload["rerun_ledger"]["counts"] == {"IF-HARD-KILL": 1}
        assert payload["seal_root_sha256"] == run.seal_root_sha256

    def test_run_artifacts_are_lf_only(self, sealed_tree):
        run = Run.start(Phase.PHASE_0, "x")
        run.log_trial(make_record())
        path = run.finish({"ok": True})
        for name in ("config.json", "seal.json", "log.jsonl"):
            assert b"\r" not in (run.root / name).read_bytes(), name
        assert b"\r" not in path.read_bytes()

    def test_scientific_rerun_is_refused_through_the_run(self, sealed_tree):
        run = Run.start(Phase.PHASE_0, "x")
        with pytest.raises(RerunRefused, match="scientific re-run"):
            run.authorize_rerun("IF-OOM", "the result was not significant, try again")
        run.close()

    def test_run_id_is_create_only(self, sealed_tree, monkeypatch):
        monkeypatch.setattr("wda.runtime.run.make_run_id", lambda phase, label: "fixed-id")
        first = Run.start(Phase.PHASE_0, "x")
        first.close()
        with pytest.raises(RunViolation, match="create-only"):
            Run.start(Phase.PHASE_0, "x")
