"""Small, offline primitives for governance records.

An ArtifactRef is a repository-relative POSIX path, a SHA-256 and a byte size.
References are content checks, not attestations that a model actually ran. No
symlinks, junctions, absolute paths, duplicate JSON keys or non-finite JSON are
accepted. Writers are create-only and flush before returning.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, TypedDict

from wda.errors import ProtocolViolation


class EvidenceError(ProtocolViolation):
    """Missing, malformed or inconsistent local governance evidence."""


class ArtifactRef(TypedDict):
    path: str
    sha256: str
    size: int


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"not canonical JSON: {exc}") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hex_digest(value: Any, length: int = 64) -> str:
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise EvidenceError(f"expected exactly {length} lowercase hexadecimal characters")
    return value


def identifier(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", value) is None:
        raise EvidenceError(f"invalid identifier: {value!r}")
    if value.endswith(".") or value.split(".")[0].upper() in {
        "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        raise EvidenceError(f"unsafe identifier: {value!r}")
    return value


def local_path(root: Path, relative: Any, *, file: bool = True) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise EvidenceError(f"invalid repository-relative path: {relative!r}")
    parts = relative.split("/")
    if PurePosixPath(relative).is_absolute() or any(p in {"", ".", ".."} for p in parts):
        raise EvidenceError(f"path escapes or aliases the root: {relative!r}")
    root = Path(root).absolute()
    # Check ancestors too: resolving a symlinked repository would hide its provenance.
    for ancestor in (root, *root.parents):
        if ancestor.is_symlink() or getattr(ancestor, "is_junction", lambda: False)():
            raise EvidenceError(f"symlink/junction is not a governance root: {ancestor}")
    path = root
    for part in parts:
        # File names (e.g. __init__.py) are less restrictive than run identifiers.
        if part.endswith((".", " ")) or any(c in part for c in '<>"|?*\0') or any(ord(c) < 32 for c in part):
            raise EvidenceError(f"unsafe artifact path component: {part!r}")
        if part.split(".")[0].upper() in {
            "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }:
            raise EvidenceError(f"reserved artifact path component: {part!r}")
        path = path / part
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise EvidenceError(f"symlink/junction is not a sealed artifact: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise EvidenceError(f"path escapes root: {relative}")
    if file and not path.is_file():
        raise EvidenceError(f"required local file is absent: {relative}")
    return path


def _pairs(pairs: list) -> dict:
    out: dict = {}
    for key, value in pairs:
        if key in out:
            raise EvidenceError(f"duplicate JSON key: {key!r}")
        out[key] = value
    return out


def read_json(path: Path) -> dict:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                EvidenceError(f"non-finite JSON value: {value}")
            ),
        )
    except (OSError, ValueError) as exc:
        raise EvidenceError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"expected JSON object: {path}")
    canonical(value)
    return value


def create_json(path: Path, value: Mapping[str, Any]) -> Path:
    payload = canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def reference(root: Path, relative: str) -> ArtifactRef:
    path = local_path(root, relative)
    return {"path": relative, "sha256": sha256_file(path), "size": path.stat().st_size}


def check_ref(root: Path, ref: Any) -> Path:
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256", "size"}:
        raise EvidenceError("ArtifactRef requires exactly path, sha256, size")
    hex_digest(ref["sha256"])
    if type(ref["size"]) is not int or ref["size"] < 1:
        raise EvidenceError("artifact size must be a positive integer")
    path = local_path(root, ref["path"])
    if path.stat().st_size != ref["size"] or sha256_file(path) != ref["sha256"]:
        raise EvidenceError(f"artifact digest/size mismatch: {ref['path']}")
    return path


def collect_refs(root: Path, value: Any) -> set[str]:
    """Validate every nested ArtifactRef and return its file paths."""
    out: set[str] = set()
    if isinstance(value, dict):
        if "path" in value and ("sha256" in value or "size" in value):
            check_ref(root, value)
            out.add(value["path"])
        else:
            for child in value.values():
                out.update(collect_refs(root, child))
    elif isinstance(value, list):
        for child in value:
            out.update(collect_refs(root, child))
    return out
