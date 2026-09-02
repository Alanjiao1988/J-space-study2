"""Repository path resolution.

Everything is anchored on the repository root so that runs are reproducible regardless of
the working directory a runner was launched from.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

_MARKERS = ("PROTOCOL_v1.1.md", "pyproject.toml")


def write_text_lf(path: Path, text: str) -> Path:
    """Write ``text`` with **LF** line endings on every platform.

    Line endings are load-bearing: the seal hashes working-tree bytes, and
    ``.gitattributes`` pins the repository to LF. If a writer used the platform default,
    a file produced on Windows would hash differently from the same file checked out on
    Linux and every run would be refused. See ``.gitattributes`` for the full rationale.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def write_json_lf(path: Path, payload: Any, *, indent: int = 2, sort_keys: bool = False) -> Path:
    """JSON + trailing newline, always LF."""
    return write_text_lf(
        path,
        json.dumps(payload, indent=indent, sort_keys=sort_keys, ensure_ascii=False) + "\n",
    )

def repo_root() -> Path:
    """Return the repository root.

    Honours ``WDA_REPO_ROOT`` (used by tests that build a scratch tree), otherwise walks
    up from this file looking for the protocol document.
    """
    override = os.environ.get("WDA_REPO_ROOT")
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if all((candidate / marker).exists() for marker in _MARKERS):
            return candidate
    # Installed (non-editable) fallback: two levels above ``src/wda``.
    return here.parents[2]


def configs_dir() -> Path:
    return repo_root() / "configs"


def runs_dir() -> Path:
    return repo_root() / "runs"


def reports_dir() -> Path:
    return repo_root() / "reports"


def references_dir() -> Path:
    return repo_root() / "references"


def third_party_dir() -> Path:
    return repo_root() / "third_party"


def seal_path(stage: str) -> Path:
    """``FREEZE-1.json`` / ``FREEZE-2.json``."""
    stage = stage.lower()
    if stage not in {"freeze1", "freeze2"}:
        raise ValueError(f"unknown seal stage {stage!r}; expected 'freeze1' or 'freeze2'")
    return repo_root() / f"FREEZE-{stage[-1]}.json"


def revision_lock_path() -> Path:
    """Resolved-and-locked checkpoint revisions (§8.1)."""
    return configs_dir() / "model_revisions.lock.json"
