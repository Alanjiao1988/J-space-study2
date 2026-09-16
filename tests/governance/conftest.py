"""Synthetic repository fixtures. These bytes are not model execution evidence."""

import json
from pathlib import Path

import pytest

from wda.governance import seal
from wda.governance.artifacts import reference


@pytest.fixture
def governance_tree(tmp_path_factory, monkeypatch):
    source = Path(__file__).resolve().parents[2]
    root = tmp_path_factory.mktemp("r")
    for rel in seal.MANDATORY:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        original = source / rel
        if original.exists() and rel.startswith("configs/"):
            path.write_bytes(original.read_bytes())
        else:
            path.write_text("{}\n" if path.suffix == ".json" else "synthetic fixture\n",
                            encoding="utf-8", newline="\n")
    for rel in ("src/wda/__init__.py", "src/wda/runner.py", "tests/test_synthetic.py"):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic fixture, never executed as a model runner.\n", encoding="utf-8")
    (root / "requirements.lock.json").write_text(json.dumps({
        "schema": "wda/dependencies.lock/1", "python": "3.13.15",
        "packages": {"numpy": "2.2.0", "scipy": "1.15.0", "pytest": "8.0.0", "setuptools": "80.0.0"},
    }), encoding="utf-8")
    (root / "configs/freeze1/calibration_revision.lock.json").write_text(json.dumps({
        "schema": "wda/model_revisions.lock/1", "revisions": {
            "calib_qwen25_7b_instruct": {
                "repo_id": "Qwen/Qwen2.5-7B-Instruct", "revision": "a" * 40,
                "locked_at": "2026-09-16T00:00:00Z",
            }
        },
    }), encoding="utf-8")
    vendor_rel = f"third_party/jacobian-lens/{seal.APPROVED_UPSTREAM}"
    (root / "third_party/jacobian-lens/PINNED_COMMIT.txt").write_text(seal.APPROVED_UPSTREAM, encoding="utf-8")
    files = {}
    for rel in seal.UPSTREAM_REQUIRED:
        path = root / vendor_rel / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic upstream inventory fixture\n", encoding="utf-8")
        ref = reference(root, f"{vendor_rel}/{rel}")
        files[rel] = {k: ref[k] for k in ("sha256", "size")}
    (root / vendor_rel / "PROVENANCE.json").write_text(json.dumps({
        "schema": "wda/upstream_provenance/1",
        "upstream": {"repository": seal.UPSTREAM_REPOSITORY, "commit": seal.APPROVED_UPSTREAM, "license": "Apache-2.0"},
        "authorship": {"modifications": "none"}, "file_count": len(files), "files": files,
    }), encoding="utf-8")
    settings = {
        "model_key": "calib_qwen25_7b_instruct", "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a" * 40,
    }
    (root / "configs/freeze1/phase0_run.json").write_text(json.dumps(settings), encoding="utf-8")
    monkeypatch.setenv("WDA_REPO_ROOT", str(root))
    return root


@pytest.fixture
def sealed_tree(governance_tree):
    seal.write("freeze1", root=governance_tree)
    return governance_tree


@pytest.fixture
def phase0_config(governance_tree):
    path = "configs/freeze1/phase0_run.json"
    return {**json.loads((governance_tree / path).read_text(encoding="utf-8")),
            "config_artifact": reference(governance_tree, path)}
