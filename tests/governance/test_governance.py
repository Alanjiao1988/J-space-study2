"""Governance tests: registry gating, Phase-A blinding, seal, re-run policy (§11, §15.1, §16)."""

from __future__ import annotations

import json

import pytest

from wda.errors import (
    BlindingViolation,
    DeniedModelError,
    PhaseViolation,
    RerunRefused,
    SealMismatch,
    UnlockedRevisionError,
    UnregisteredModelError,
)
from wda.governance import blind, rerun_policy, seal
from wda.models import registry
from wda.phases import Phase, Role, Stratum


# ============================================================================ registry ===


class TestRegistry:
    def test_only_registered_checkpoints_resolve(self):
        with pytest.raises(UnregisteredModelError, match="not registered"):
            registry.resolve("mistralai/Mistral-7B-v0.1", Phase.PHASE_B, require_revision=False)

    def test_the_1p5b_is_denied_with_fact_f1_as_the_reason(self):
        """F1 / §13 — the 1.5B is not a subject and nothing may be said about it."""
        with pytest.raises(DeniedModelError, match="fact F1"):
            registry.resolve(
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", Phase.PHASE_B, require_revision=False
            )

    def test_calibration_model_is_refused_in_evidential_phases(self):
        with pytest.raises(PhaseViolation, match="phase0, phaseA"):
            registry.resolve("calib_qwen25_7b_instruct", Phase.PHASE_C, require_revision=False)

    def test_subjects_are_refused_in_phase_a(self):
        with pytest.raises(PhaseViolation):
            registry.resolve("r1_distill_qwen_14b", Phase.PHASE_A, require_revision=False)

    def test_replication_tier_is_phase_d_only(self):
        with pytest.raises(PhaseViolation):
            registry.resolve("r1_distill_qwen_32b", Phase.PHASE_C, require_revision=False)
        entry = registry.resolve("r1_distill_qwen_32b", Phase.PHASE_D, require_revision=False)
        assert entry.stratum is Stratum.GENERAL_PARENT_32B

    def test_unlocked_revision_blocks_loading(self):
        locked = registry.load_revision_lock()
        if "r1_distill_qwen_14b" in locked:  # pragma: no cover - only after the operator locks
            pytest.skip("revision already locked by the operator")
        with pytest.raises(UnlockedRevisionError, match="no locked revision"):
            registry.resolve("r1_distill_qwen_14b", Phase.PHASE_B)

    def test_repo_id_and_key_both_resolve(self):
        by_key = registry.resolve("qwen25_14b_instruct", Phase.PHASE_B, require_revision=False)
        by_repo = registry.resolve("Qwen/Qwen2.5-14B-Instruct", Phase.PHASE_B, require_revision=False)
        assert by_key.key == by_repo.key

    def test_primary_pair_is_the_named_14b_pair(self):
        pair = registry.subjects(Stratum.PRIMARY_14B)
        assert pair[Role.TREATMENT].repo_id == "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        assert pair[Role.COMPARATOR].repo_id == "Qwen/Qwen2.5-14B-Instruct"

    def test_parent_anchor_is_descriptive_only(self):
        anchor = registry.REGISTRY["qwen25_14b_base"]
        assert anchor.role is Role.PARENT_ANCHOR
        assert anchor.descriptive_only is True

    def test_relocking_a_revision_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(registry, "revision_lock_path", lambda: tmp_path / "lock.json")
        registry.write_revision_lock({"qwen25_14b_instruct": "a" * 40})
        registry.write_revision_lock({"qwen25_14b_instruct": "a" * 40})  # idempotent
        with pytest.raises(UnlockedRevisionError, match="already locked"):
            registry.write_revision_lock({"qwen25_14b_instruct": "b" * 40})


# ============================================================================ blinding ===


class TestBlinding:
    def test_subjects_are_withheld_while_blinded(self):
        with blind.phase_a_blinding():
            with pytest.raises(BlindingViolation, match="blinding is active"):
                registry.resolve("r1_distill_qwen_14b", Phase.PHASE_B, require_revision=False)
        # released afterwards
        registry.resolve("r1_distill_qwen_14b", Phase.PHASE_B, require_revision=False)

    def test_calibration_model_stays_available(self):
        with blind.phase_a_blinding():
            entry = registry.resolve("calib_qwen25_7b_instruct", Phase.PHASE_A, require_revision=False)
            assert entry.role is Role.CALIBRATION

    def test_decision_rule_is_refused_while_blinded(self):
        from wda.stats.decision import Interval, decide

        with blind.phase_a_blinding():
            with pytest.raises(BlindingViolation, match="zero \\s*evidential weight|evidential"):
                decide(0.2, Interval(0.15, 0.25, 0.95), Interval(0.16, 0.24, 0.90), 0.10)

    def test_holm_is_refused_while_blinded(self):
        from wda.stats.decision import holm

        with blind.phase_a_blinding():
            with pytest.raises(BlindingViolation):
                holm({"a": 0.01})

    def test_forbidden_surfaces_are_refused(self):
        with blind.phase_a_blinding():
            blind.guard_surface("structural_statistics")
            blind.guard_surface("general_damage_check")
            for bad in ("D", "G_primary", "delta_accuracy"):
                with pytest.raises(BlindingViolation, match="may not inform"):
                    blind.guard_surface(bad)

    def test_evidential_phase_is_refused_while_blinded(self):
        with blind.phase_a_blinding():
            for phase in (Phase.PHASE_B, Phase.PHASE_C, Phase.PHASE_D):
                with pytest.raises(BlindingViolation, match="cannot enter"):
                    blind.guard_phase(phase)

    def test_nested_blinding_is_refused(self):
        with blind.phase_a_blinding():
            with pytest.raises(BlindingViolation, match="already active"):
                with blind.phase_a_blinding():
                    pass

    def test_env_var_reestablishes_blinding_in_a_child_process(self, monkeypatch):
        monkeypatch.setenv(blind.BLIND_ENV_VAR, "1")
        assert blind.is_active() is True
        assert blind.withheld_roles() == frozenset({Role.TREATMENT, Role.COMPARATOR})

    def test_violations_are_recorded(self):
        with blind.phase_a_blinding():
            with pytest.raises(BlindingViolation):
                blind.guard_surface("D")
            assert any(v["kind"] == "forbidden_surface" for v in blind.violations())


# ================================================================================ seal ===


class TestSeal:
    def test_write_then_verify_round_trips(self, tmp_path):
        root = tmp_path
        (root / "PROTOCOL_v1.1.md").write_text("protocol", encoding="utf-8")
        (root / "pyproject.toml").write_text("[project]", encoding="utf-8")
        (root / "configs" / "freeze1").mkdir(parents=True)
        (root / "configs" / "freeze1" / "a.json").write_text("{}", encoding="utf-8")
        (root / "src" / "wda").mkdir(parents=True)
        (root / "src" / "wda" / "m.py").write_text("x = 1\n", encoding="utf-8")
        (root / "tests").mkdir()
        (root / "tests" / "t.py").write_text("y = 2\n", encoding="utf-8")

        manifest = seal.write("freeze1", root=root)
        assert manifest["file_count"] == 5
        assert seal.verify("freeze1", root=root)["ok"] is True

    def test_a_changed_file_breaks_the_seal(self, tmp_path):
        root = tmp_path
        (root / "PROTOCOL_v1.1.md").write_text("protocol", encoding="utf-8")
        (root / "pyproject.toml").write_text("[project]", encoding="utf-8")
        seal.write("freeze1", root=root)
        (root / "PROTOCOL_v1.1.md").write_text("tampered", encoding="utf-8")
        with pytest.raises(SealMismatch, match="changed"):
            seal.verify("freeze1", root=root)
        assert seal.diff("freeze1", root=root)["changed"] == ["PROTOCOL_v1.1.md"]

    def test_an_added_file_breaks_the_seal(self, tmp_path):
        root = tmp_path
        (root / "PROTOCOL_v1.1.md").write_text("p", encoding="utf-8")
        (root / "pyproject.toml").write_text("[project]", encoding="utf-8")
        seal.write("freeze1", root=root)
        (root / "src" / "wda").mkdir(parents=True)
        (root / "src" / "wda" / "extra.py").write_text("z = 3\n", encoding="utf-8")
        with pytest.raises(SealMismatch, match="added"):
            seal.verify("freeze1", root=root)

    def test_resealing_a_changed_tree_is_refused(self, tmp_path):
        root = tmp_path
        (root / "PROTOCOL_v1.1.md").write_text("p", encoding="utf-8")
        (root / "pyproject.toml").write_text("[project]", encoding="utf-8")
        seal.write("freeze1", root=root)
        (root / "PROTOCOL_v1.1.md").write_text("changed", encoding="utf-8")
        with pytest.raises(SealMismatch, match="may not be re-sealed"):
            seal.write("freeze1", root=root)

    def test_missing_seal_is_a_mismatch(self, tmp_path):
        (tmp_path / "PROTOCOL_v1.1.md").write_text("p", encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text("[project]", encoding="utf-8")
        with pytest.raises(SealMismatch, match="has not been sealed"):
            seal.verify("freeze1", root=tmp_path)

    def test_skip_env_var_cannot_bypass_the_check(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WDA_SKIP_SEAL", "1")
        with pytest.raises(SealMismatch, match="not bypassable"):
            seal.require("freeze1", root=tmp_path)

    def test_repo_freeze1_seal_verifies(self):
        """The committed FREEZE-1.json must match the working tree."""
        assert seal.verify("freeze1")["ok"] is True

    def test_sealed_files_contain_no_cr_bytes(self):
        """Line endings are load-bearing: the seal hashes bytes, .gitattributes pins LF.

        A CR byte in a sealed file means the manifest was computed on platform-specific
        bytes and would fail verification after a checkout elsewhere.
        """
        from wda.paths import repo_root

        root = repo_root()
        offenders = [
            entry["path"]
            for entry in seal.build_manifest("freeze1", root=root)["files"]
            if b"\r" in (root / entry["path"]).read_bytes()
        ]
        assert offenders == []

    def test_seal_writer_emits_lf(self, tmp_path):
        root = tmp_path
        (root / "PROTOCOL_v1.1.md").write_text("p", encoding="utf-8")
        (root / "pyproject.toml").write_text("[project]", encoding="utf-8")
        seal.write("freeze1", root=root)
        assert b"\r" not in (root / "FREEZE-1.json").read_bytes()

    def test_revision_lock_writer_emits_lf(self, tmp_path, monkeypatch):
        monkeypatch.setattr(registry, "revision_lock_path", lambda: tmp_path / "lock.json")
        registry.write_revision_lock({"qwen25_14b_instruct": "c" * 40})
        assert b"\r" not in (tmp_path / "lock.json").read_bytes()


# ========================================================================= rerun policy ===


class TestRerunPolicy:
    def test_registered_infrastructure_code_is_authorized(self):
        ledger = rerun_policy.RerunLedger("phaseC-x")
        entry = rerun_policy.authorize(ledger, "IF-OOM", "device OOM at trial 812")
        assert entry["attempt"] == 1
        assert "micro-batch" in entry["recovery"]

    def test_unregistered_code_is_refused(self):
        ledger = rerun_policy.RerunLedger("phaseC-x")
        with pytest.raises(RerunRefused, match="not a registered infrastructure failure code"):
            rerun_policy.authorize(ledger, "EXIT-1", "process exited non-zero")

    @pytest.mark.parametrize(
        "reason",
        [
            "the result was not significant, retry",
            "try again with a different seed",
            "band looks wrong, change band",
            "add samples to increase n",
            "switch endpoint to the conditional accuracy",
        ],
    )
    def test_scientific_reruns_are_always_refused(self, reason):
        ledger = rerun_policy.RerunLedger("phaseC-x")
        with pytest.raises(RerunRefused, match="scientific re-run"):
            rerun_policy.authorize(ledger, "IF-HARD-KILL", reason)

    def test_attempt_budget_is_enforced(self):
        ledger = rerun_policy.RerunLedger("phaseC-x")
        for _ in range(2):
            rerun_policy.authorize(ledger, "IF-CHECKPOINT-DIGEST", "digest mismatch on download")
        with pytest.raises(RerunRefused, match="budget is exhausted"):
            rerun_policy.authorize(ledger, "IF-CHECKPOINT-DIGEST", "digest mismatch on download")

    def test_cpu_fallback_canary_is_registered(self):
        """Fact F13 lineage: the CPU-recovery canary is a pre-registered infrastructure path."""
        assert "IF-CPU-FALLBACK" in rerun_policy.INFRASTRUCTURE_CODES
        assert rerun_policy.INFRASTRUCTURE_CODES["IF-CPU-FALLBACK"].max_attempts == 1

    def test_policy_snapshot_is_serialisable(self):
        json.dumps(rerun_policy.policy_snapshot())
