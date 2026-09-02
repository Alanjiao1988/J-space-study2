#!/usr/bin/env python
"""Vendor the pinned upstream ``jacobian-lens`` commit, unmodified.

Clones into ``third_party/jacobian-lens/<commit>/`` and writes a ``PROVENANCE.json`` with a
digest per file. The clone is never modified and is not tracked in git; only the pin and
the vendoring policy are committed.

Usage::

    python tools/vendor_upstream.py            # clone at the pinned commit
    python tools/vendor_upstream.py --verify    # re-hash an existing clone
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = "https://github.com/anthropics/jacobian-lens.git"
LICENSE = "Apache-2.0"

ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT / "third_party" / "jacobian-lens"
PIN_FILE = VENDOR_DIR / "PINNED_COMMIT.txt"


def pinned_commit() -> str:
    commit = PIN_FILE.read_text(encoding="utf-8").strip()
    if len(commit) != 40 or not all(c in "0123456789abcdef" for c in commit):
        raise SystemExit(f"{PIN_FILE} does not contain a full 40-character commit digest")
    return commit


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_provenance(target: Path, commit: str) -> Path:
    files = {}
    for path in sorted(target.rglob("*")):
        if path.is_file() and ".git" not in path.parts and path.name != "PROVENANCE.json":
            files[path.relative_to(target).as_posix()] = {
                "sha256": sha256(path),
                "size": path.stat().st_size,
            }
    payload = {
        "schema": "wda/upstream_provenance/1",
        "upstream": {"repository": REPO, "commit": commit, "license": LICENSE},
        "authorship": {"modifications": "none"},
        "vendored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file_count": len(files),
        "files": files,
        "policy": "read-only; never modified; not tracked in git (see third_party/jacobian-lens/README.md)",
    }
    out = target / "PROVENANCE.json"
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return out


def clone(commit: str, *, force: bool) -> Path:
    target = VENDOR_DIR / commit
    if target.exists():
        if not force:
            print(f"{target} already exists; use --force to re-clone", file=sys.stderr)
            return target
        shutil.rmtree(target)
    target.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet"], cwd=target, check=True)
    subprocess.run(["git", "remote", "add", "origin", REPO], cwd=target, check=True)
    subprocess.run(["git", "fetch", "--quiet", "--depth", "1", "origin", commit], cwd=target, check=True)
    subprocess.run(["git", "checkout", "--quiet", "FETCH_HEAD"], cwd=target, check=True)
    shutil.rmtree(target / ".git", ignore_errors=True)
    return target


def verify(commit: str) -> int:
    target = VENDOR_DIR / commit
    manifest_path = target / "PROVENANCE.json"
    if not manifest_path.exists():
        print(f"no PROVENANCE.json at {target}", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad = []
    for rel, entry in manifest["files"].items():
        path = target / rel
        if not path.exists() or sha256(path) != entry["sha256"]:
            bad.append(rel)
    if bad:
        print(f"upstream clone has been modified: {bad}", file=sys.stderr)
        return 1
    print(f"OK: {manifest['file_count']} files match PROVENANCE.json at {commit}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="re-hash an existing clone")
    parser.add_argument("--force", action="store_true", help="re-clone over an existing directory")
    args = parser.parse_args()

    commit = pinned_commit()
    if args.verify:
        return verify(commit)
    target = clone(commit, force=args.force)
    manifest = write_provenance(target, commit)
    print(f"vendored {REPO}@{commit[:12]} -> {target}")
    print(f"provenance: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
