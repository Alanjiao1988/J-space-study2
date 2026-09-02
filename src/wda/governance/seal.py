"""Seal: hash manifests for Freeze-1 and Freeze-2 (§16).

``FREEZE-1.json`` covers the protocol document, ``configs/freeze1/*``, ``src/wda/**``,
``tests/**`` and the item-generator source. ``FREEZE-2.json`` additionally covers
``configs/freeze2/*``, the fitted-lens file hashes, the item-bank seed and the subject
revisions.

Every run entry point calls :func:`verify` and refuses to start on mismatch. The seal is
written **before** the first model call (fact F13).

CLI::

    python -m wda.governance.seal write  --stage freeze1
    python -m wda.governance.seal verify --stage freeze1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from wda.errors import SealMismatch
from wda.paths import repo_root, seal_path, write_json_lf

SCHEMA = "wda/seal/1"

#: Files whose bytes are hashed for each stage. Globs are relative to the repo root.
STAGE_PATTERNS: Mapping[str, Sequence[str]] = {
    "freeze1": (
        "PROTOCOL_v1.1.md",
        "configs/freeze1/**/*.json",
        "configs/freeze1/**/*.md",
        "src/wda/**/*.py",
        "src/wda/**/*.md",
        "tests/**/*.py",
        "pyproject.toml",
    ),
    # Freeze-2 is a superset: the Freeze-1 surface must still match, plus the parameters
    # chosen in Phase A and the instrument state qualified in Phase B.
    "freeze2": (
        "PROTOCOL_v1.1.md",
        "configs/freeze1/**/*.json",
        "configs/freeze1/**/*.md",
        "configs/freeze2/**/*.json",
        "configs/model_revisions.lock.json",
        "src/wda/**/*.py",
        "src/wda/**/*.md",
        "tests/**/*.py",
        "pyproject.toml",
    ),
}

#: Directories excluded from every stage (build artefacts, caches, run outputs).
EXCLUDED_PARTS = frozenset({"__pycache__", ".pytest_cache", ".git", ".venv", "venv", "build", "dist"})


@dataclass(frozen=True)
class SealEntry:
    path: str
    sha256: str
    size: int

    def to_dict(self) -> Dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _iter_stage_files(root: Path, patterns: Iterable[str]) -> List[Path]:
    seen: Dict[str, Path] = {}
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            if EXCLUDED_PARTS & set(path.relative_to(root).parts):
                continue
            rel = path.relative_to(root).as_posix()
            seen[rel] = path
    return [seen[rel] for rel in sorted(seen)]


def build_manifest(
    stage: str,
    *,
    root: Optional[Path] = None,
    extra: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Compute the manifest for ``stage`` from the current working tree."""
    stage = stage.lower()
    if stage not in STAGE_PATTERNS:
        raise ValueError(f"unknown stage {stage!r}; expected one of {sorted(STAGE_PATTERNS)}")
    root = root or repo_root()
    entries = [
        SealEntry(
            path=path.relative_to(root).as_posix(),
            sha256=sha256_file(path),
            size=path.stat().st_size,
        )
        for path in _iter_stage_files(root, STAGE_PATTERNS[stage])
    ]
    files = [entry.to_dict() for entry in entries]
    # The root hash is a hash of the (path, sha256) list, so reordering or renaming a file
    # changes it even when the file contents are unchanged.
    spine = "\n".join(f"{e.path}\t{e.sha256}" for e in entries).encode("utf-8")
    manifest: Dict[str, object] = {
        "schema": SCHEMA,
        "stage": stage,
        "protocol": "PROTOCOL_v1.1.md",
        "file_count": len(files),
        "root_sha256": sha256_bytes(spine),
        "files": files,
    }
    if extra:
        manifest["extra"] = dict(extra)
    return manifest


def write(
    stage: str,
    *,
    root: Optional[Path] = None,
    extra: Optional[Mapping[str, object]] = None,
    note: str = "",
) -> Dict[str, object]:
    """Write ``FREEZE-{1,2}.json``. Refuses to overwrite a differing existing seal."""
    root = root or repo_root()
    manifest = build_manifest(stage, root=root, extra=extra)
    manifest["sealed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["note"] = note or f"Seal for {stage}; written before the first model call (F13)."

    path = root / seal_path(stage).name
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("root_sha256") != manifest["root_sha256"]:
            raise SealMismatch(
                f"{path.name} already exists with root_sha256={previous.get('root_sha256')!r} "
                f"but the working tree hashes to {manifest['root_sha256']!r}. A frozen stage "
                "may not be re-sealed; scientific re-runs are forbidden (§11)."
            )
        # Idempotent re-write: keep the original timestamp.
        manifest["sealed_at"] = previous.get("sealed_at", manifest["sealed_at"])
    write_json_lf(path, manifest)
    return manifest


def load(stage: str, *, root: Optional[Path] = None) -> Dict[str, object]:
    root = root or repo_root()
    path = root / seal_path(stage).name
    if not path.exists():
        raise SealMismatch(f"{path.name} is absent; the stage has not been sealed (§16).")
    return json.loads(path.read_text(encoding="utf-8"))


def diff(stage: str, *, root: Optional[Path] = None) -> Dict[str, List[str]]:
    """Return ``{'added': [...], 'removed': [...], 'changed': [...]}`` against the seal."""
    root = root or repo_root()
    sealed = load(stage, root=root)
    current = build_manifest(stage, root=root)
    sealed_map = {e["path"]: e["sha256"] for e in sealed.get("files", [])}
    current_map = {e["path"]: e["sha256"] for e in current["files"]}
    added = sorted(set(current_map) - set(sealed_map))
    removed = sorted(set(sealed_map) - set(current_map))
    changed = sorted(p for p in set(sealed_map) & set(current_map) if sealed_map[p] != current_map[p])
    return {"added": added, "removed": removed, "changed": changed}


def verify(stage: str, *, root: Optional[Path] = None) -> Dict[str, object]:
    """Verify the working tree against the seal, or raise :class:`SealMismatch`."""
    root = root or repo_root()
    sealed = load(stage, root=root)
    current = build_manifest(stage, root=root)
    if sealed.get("root_sha256") == current["root_sha256"]:
        return {
            "stage": stage,
            "ok": True,
            "root_sha256": current["root_sha256"],
            "file_count": current["file_count"],
        }
    delta = diff(stage, root=root)
    raise SealMismatch(
        f"seal mismatch for {stage}: "
        f"added={delta['added']} removed={delta['removed']} changed={delta['changed']}. "
        "The run is refused (§16)."
    )


def require(stage: str, *, root: Optional[Path] = None) -> str:
    """Entry-point guard. Returns the verified ``root_sha256`` for the trial log."""
    if os.environ.get("WDA_SKIP_SEAL"):
        raise SealMismatch(
            "WDA_SKIP_SEAL is set. The seal check is not bypassable; unset it (§16)."
        )
    result = verify(stage, root=root)
    return str(result["root_sha256"])


def _main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze seal (§16).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, help_text in (
        ("write", "compute and write the seal"),
        ("verify", "verify the working tree against the seal"),
        ("diff", "show added/removed/changed files vs the seal"),
        ("show", "print the stored seal header"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--stage", default="freeze1", choices=sorted(STAGE_PATTERNS))

    args = parser.parse_args(argv)
    if args.cmd == "write":
        manifest = write(args.stage)
        print(
            json.dumps(
                {k: manifest[k] for k in ("stage", "file_count", "root_sha256", "sealed_at")},
                indent=2,
            )
        )
        return 0
    if args.cmd == "verify":
        print(json.dumps(verify(args.stage), indent=2))
        return 0
    if args.cmd == "diff":
        print(json.dumps(diff(args.stage), indent=2))
        return 0
    sealed = load(args.stage)
    print(
        json.dumps(
            {k: sealed.get(k) for k in ("stage", "file_count", "root_sha256", "sealed_at", "note")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(_main())
