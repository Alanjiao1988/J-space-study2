"""Fail-closed Freeze-1/2 seals (schema ``wda/seal/2``).

All metadata, including ``extra``, note, timestamps and sizes, is canonically
hashed. Verification independently enumerates the mandatory surface; the
manifest cannot choose its own exclusions. Freeze-1 deliberately excludes the
later subject lock and Freeze-2 directory, but pins its own calibration lock.

See ``CONTRACTS.md`` beside this module for the mandatory local artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

from wda.errors import ProtocolViolation, SealMismatch
from wda.governance.artifacts import (
    EvidenceError, canonical, check_ref, collect_refs, create_json, digest,
    hex_digest, local_path, read_json, sha256_file,
)
from wda.paths import repo_root

SCHEMA = "wda/seal/2"
APPROVED_UPSTREAM = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
UPSTREAM_REPOSITORY = "https://github.com/anthropics/jacobian-lens.git"
EXCLUDED_PARTS = frozenset({"__pycache__", ".pytest_cache", ".git", ".venv", "venv"})
STAGE_PATTERNS = {
    "freeze1": ("src", "tests", "tools", "references", "configs"),
    "freeze2": ("src", "tests", "tools", "references", "configs"),
}
MANDATORY = (
    "PROTOCOL_v1.1.md", "pyproject.toml", "requirements.lock.json",
    "configs/freeze1/band_rule.json", "configs/freeze1/decision.json",
    "configs/freeze1/envelopes.json", "configs/freeze1/fixtures.json",
    "configs/freeze1/instrument.json", "configs/freeze1/calibration_revision.lock.json",
    "configs/phase_a/search_space.json",
    "references/predecessor_index.md", "references/predecessor_facts.md",
    "references/paper_reference_values.md", "references/literature_verification.md",
    "tools/vendor_upstream.py", "third_party/jacobian-lens/PINNED_COMMIT.txt",
    "third_party/jacobian-lens/README.md",
)
FREEZE2_FILES = (
    "parameters.json", "power.json", "lens_digests.json", "item_bank.json", "qualification.json",
)
UPSTREAM_REQUIRED = (
    "LICENSE", "data/evaluations/README.md", "data/evaluations/lens-eval-multihop.json",
    "data/evaluations/lens-eval-order-ops.json", "data/experiments/README.md",
    "data/experiments/probe-swap.json",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stage(stage: str) -> str:
    if stage not in STAGE_PATTERNS:
        raise SealMismatch(f"invalid seal stage {stage!r}; expected freeze1 or freeze2")
    return stage


def _walk(root: Path, directory: str) -> set[str]:
    base = local_path(root, directory, file=False)
    if not base.is_dir():
        raise EvidenceError(f"required directory is absent: {directory}")
    out: set[str] = set()
    for current, dirs, files in os.walk(base, followlinks=False):
        for name in list(dirs):
            rel = (Path(current) / name).relative_to(root).as_posix()
            local_path(root, rel, file=False)
            if name in EXCLUDED_PARTS:
                dirs.remove(name)
        for name in files:
            rel = (Path(current) / name).relative_to(root).as_posix()
            local_path(root, rel)
            out.add(rel)
    return out


def _validate_dependencies(root: Path) -> None:
    """Exact version manifest, not a claim that the host environment matches it."""
    lock = read_json(local_path(root, "requirements.lock.json"))
    if lock.get("schema") != "wda/dependencies.lock/1":
        raise EvidenceError("requirements.lock.json requires wda/dependencies.lock/1")
    import re

    if not isinstance(lock.get("python"), str) or not re.fullmatch(r"\d+\.\d+\.\d+", lock["python"]):
        raise EvidenceError("dependency lock requires an exact Python version")
    packages = lock.get("packages")
    if not isinstance(packages, dict) or not {"numpy", "scipy", "pytest", "setuptools"} <= packages.keys():
        raise EvidenceError("dependency lock must pin numpy, scipy, pytest and setuptools")
    for name, version in packages.items():
        if not isinstance(name, str) or not isinstance(version, str) or not re.fullmatch(
            r"\d+(?:\.\d+)*(?:[a-zA-Z0-9.+_-]*)", version
        ):
            raise EvidenceError(f"dependency must have a literal resolved version: {name!r}")


def _upstream_files(root: Path) -> set[str]:
    prefix = "third_party/jacobian-lens"
    pin = local_path(root, prefix + "/PINNED_COMMIT.txt").read_text(encoding="utf-8").strip()
    if pin != APPROVED_UPSTREAM:
        raise EvidenceError("upstream pin is not the protocol-approved commit")
    vendor = f"{prefix}/{pin}"
    provenance = read_json(local_path(root, vendor + "/PROVENANCE.json"))
    if provenance.get("schema") != "wda/upstream_provenance/1" or provenance.get("upstream") != {
        "repository": UPSTREAM_REPOSITORY, "commit": pin, "license": "Apache-2.0",
    } or provenance.get("authorship") != {"modifications": "none"}:
        raise EvidenceError("upstream provenance does not bind the approved repository/commit")
    files = provenance.get("files")
    if not isinstance(files, dict) or not set(UPSTREAM_REQUIRED) <= files.keys():
        raise EvidenceError("upstream provenance omits registered prompt sets/license")
    if type(provenance.get("file_count")) is not int or provenance["file_count"] != len(files):
        raise EvidenceError("invalid upstream provenance file_count")
    for rel, entry in files.items():
        if not isinstance(entry, dict) or set(entry) != {"sha256", "size"}:
            raise EvidenceError("malformed upstream file entry")
        check_ref(root, {"path": f"{vendor}/{rel}", **entry})
    actual = _walk(root, vendor)
    expected = {f"{vendor}/{rel}" for rel in files} | {vendor + "/PROVENANCE.json"}
    if actual != expected:
        raise EvidenceError("upstream provenance file list is incomplete or has extra files")
    return expected


def _surface(root: Path, stage: str) -> set[str]:
    from wda.models.registry import load_revision_lock, validate_revision_lock

    required = set(MANDATORY)
    if stage == "freeze2":
        required |= {f"configs/freeze2/{name}" for name in FREEZE2_FILES}
        required |= {"configs/model_revisions.lock.json", "FREEZE-1.json"}
    for rel in sorted(required):
        path = local_path(root, rel)
        if path.stat().st_size == 0:
            raise EvidenceError(f"required artifact is empty: {rel}")
        if path.suffix == ".json":
            read_json(path)
    _validate_dependencies(root)
    calibration = validate_revision_lock(
        read_json(local_path(root, "configs/freeze1/calibration_revision.lock.json"))
    )
    if set(calibration) != {"calib_qwen25_7b_instruct"}:
        raise EvidenceError("Freeze-1 requires exactly the calibration revision")
    # During Freeze-1 later locks are neither trusted nor required.
    if stage == "freeze2":
        load_revision_lock(root=root)
    files = set(required)
    for directory in STAGE_PATTERNS[stage]:
        found = _walk(root, directory)
        if directory in {"src", "tests"} and not any(p.endswith(".py") for p in found):
            raise EvidenceError(f"{directory} must contain scientific source/tests")
        if stage == "freeze1":
            found = {
                p for p in found
                if not p.startswith("configs/freeze2/")
                and p != "configs/model_revisions.lock.json"
            }
        files |= found
    files |= _upstream_files(root)
    # Hash registered external-to-config local references as well as their descriptors.
    for rel in sorted(files.copy()):
        if rel.startswith("configs/") and rel.endswith(".json"):
            files |= collect_refs(root, read_json(local_path(root, rel)))
    if stage == "freeze2":
        verify("freeze1", root=root)
        from wda.governance.evidence import validate_freeze2

        files |= validate_freeze2(root)
    return files


def _root_hash(manifest: dict) -> str:
    return digest({key: value for key, value in manifest.items() if key != "root_sha256"})


def _validate_manifest(manifest: dict, stage: str, root: Path) -> None:
    required = {"schema", "stage", "protocol", "file_count", "files", "extra", "root_sha256"}
    if set(manifest) - (required | {"sealed_at", "note"}) or not required <= manifest.keys():
        raise EvidenceError("seal has missing or unknown schema fields")
    if manifest["schema"] != SCHEMA or manifest["stage"] != stage:
        raise EvidenceError("seal schema/stage mismatch (old candidates must be archived)")
    if manifest["protocol"] != "PROTOCOL_v1.1.md" or not isinstance(manifest["extra"], dict):
        raise EvidenceError("invalid seal protocol/extra metadata")
    for key in ("sealed_at", "note"):
        if key in manifest and (not isinstance(manifest[key], str) or not manifest[key]):
            raise EvidenceError(f"invalid seal {key}")
    entries = manifest["files"]
    if not isinstance(entries, list) or not entries:
        raise EvidenceError("seal requires a nonempty files list")
    if type(manifest["file_count"]) is not int or manifest["file_count"] != len(entries):
        raise EvidenceError("seal file_count does not match files")
    paths = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size"}:
            raise EvidenceError("invalid seal file entry")
        hex_digest(entry["sha256"])
        if type(entry["size"]) is not int or entry["size"] < 0:
            raise EvidenceError("invalid seal file size")
        local_path(root, entry["path"], file=False)
        paths.append(entry["path"])
    if paths != sorted(set(paths)) or len({p.casefold() for p in paths}) != len(paths):
        raise EvidenceError("seal files must be sorted, unique, non-aliased paths")
    hex_digest(manifest["root_sha256"])
    if manifest["root_sha256"] != _root_hash(manifest):
        raise EvidenceError("seal root hash does not cover its canonical metadata/files")


def build_manifest(
    stage: str, *, root: Optional[Path] = None, extra: Optional[Mapping[str, object]] = None,
) -> dict:
    stage = _stage(stage)
    root = Path(root or repo_root()).absolute()
    try:
        files = [
            {"path": rel, "sha256": sha256_file(local_path(root, rel)),
             "size": local_path(root, rel).stat().st_size}
            for rel in sorted(_surface(root, stage))
        ]
        manifest = {
            "schema": SCHEMA, "stage": stage, "protocol": "PROTOCOL_v1.1.md",
            "file_count": len(files), "files": files, "extra": dict(extra or {}),
        }
        manifest["root_sha256"] = _root_hash(manifest)
        _validate_manifest(manifest, stage, root)
        return manifest
    except (ProtocolViolation, OSError, TypeError, ValueError) as exc:
        raise SealMismatch(f"cannot build {stage}: {exc}") from exc


def write(
    stage: str, *, root: Optional[Path] = None, extra: Optional[Mapping[str, object]] = None,
    note: str = "",
) -> dict:
    root = Path(root or repo_root()).absolute()
    stage = _stage(stage)
    path = local_path(root, f"FREEZE-{stage[-1]}.json", file=False)
    if path.exists():
        try:
            verify(stage, root=root)
            previous = load(stage, root=root)
            if extra is not None and previous["extra"] != extra:
                raise SealMismatch("extra metadata changed")
            if note and previous.get("note") != note:
                raise SealMismatch("note changed")
            return previous
        except SealMismatch as exc:
            raise SealMismatch(f"{path.name} may not be re-sealed: {exc}") from exc
    manifest = build_manifest(stage, root=root, extra=extra)
    manifest["sealed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["note"] = note or f"Seal for {stage}; content integrity, not execution attestation."
    manifest["root_sha256"] = _root_hash(manifest)
    create_json(path, manifest)
    return manifest


def load(stage: str, *, root: Optional[Path] = None) -> dict:
    root = Path(root or repo_root()).absolute()
    stage = _stage(stage)
    try:
        path = local_path(root, f"FREEZE-{stage[-1]}.json", file=False)
        if not path.exists():
            raise EvidenceError(f"{path.name} is absent; the stage has not been sealed")
        manifest = read_json(path)
        _validate_manifest(manifest, stage, root)
        return manifest
    except (EvidenceError, OSError) as exc:
        raise SealMismatch(str(exc)) from exc


def _delta(sealed: dict, current: dict) -> dict:
    before = {e["path"]: (e["sha256"], e["size"]) for e in sealed["files"]}
    after = {e["path"]: (e["sha256"], e["size"]) for e in current["files"]}
    return {
        "added": sorted(after.keys() - before.keys()),
        "removed": sorted(before.keys() - after.keys()),
        "changed": sorted(p for p in before.keys() & after.keys() if before[p] != after[p]),
    }


def diff(stage: str, *, root: Optional[Path] = None) -> dict:
    return _delta(load(stage, root=root), build_manifest(stage, root=root))


def verify(stage: str, *, root: Optional[Path] = None) -> dict:
    sealed = load(stage, root=root)
    current = build_manifest(stage, root=root, extra=sealed["extra"])
    delta = _delta(sealed, current)
    if any(delta.values()):
        raise SealMismatch(f"seal mismatch for {stage}: {delta}; run refused")
    return {
        "stage": stage, "ok": True, "root_sha256": sealed["root_sha256"],
        "file_count": sealed["file_count"],
    }


def require(stage: str, *, root: Optional[Path] = None) -> str:
    if os.environ.get("WDA_SKIP_SEAL"):
        raise SealMismatch("WDA_SKIP_SEAL is set; the seal check is not bypassable")
    return verify(stage, root=root)["root_sha256"]


def _main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("write", "verify", "diff", "show"))
    parser.add_argument("--stage", default="freeze1", choices=sorted(STAGE_PATTERNS))
    args = parser.parse_args(argv)
    result = {"write": write, "verify": verify, "diff": diff, "show": load}[args.cmd](args.stage)
    print(canonical(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
