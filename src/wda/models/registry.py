"""Model registry (§8, §15.1).

Only the checkpoints named in the protocol may be loaded, only in the phases the protocol
allows, and only at a *resolved and locked* revision. Every other request raises.

Three independent gates apply to :func:`resolve`:

1. **Denylist** — fact F1: the 1.5B checkpoint is not a subject and never may be.
2. **Phase gate** — the calibration model is Phase-0/A only; subjects are Phase-B onward.
3. **Blinding** — during Phase A, :mod:`wda.governance.blind` additionally withholds every
   ``treatment`` / ``comparator`` entry (§16), on top of the phase gate.

Revisions are deliberately *not* hard-coded here. §8.1 requires the new repository to
re-resolve and lock them; the resolved digests live in
``configs/model_revisions.lock.json`` and are hashed into the Freeze-2 manifest.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, Mapping, Optional

from wda.errors import (
    DeniedModelError,
    PhaseViolation,
    UnlockedRevisionError,
    UnregisteredModelError,
)
from wda.paths import revision_lock_path, write_json_lf
from wda.phases import Phase, Role, Stratum

_ALL_MODEL_PHASES = (Phase.PHASE_0, Phase.PHASE_A, Phase.PHASE_B, Phase.PHASE_C, Phase.PHASE_D)


@dataclass(frozen=True)
class ModelEntry:
    """One registered checkpoint."""

    key: str
    repo_id: str
    role: Role
    stratum: Stratum
    allowed_phases: frozenset
    note: str = ""
    #: Populated from the lock file; ``None`` means "not yet resolved".
    revision: Optional[str] = None
    #: ``True`` only for the anchor, which is described but never enters an estimand.
    descriptive_only: bool = False

    def with_revision(self, revision: Optional[str]) -> "ModelEntry":
        return ModelEntry(
            key=self.key,
            repo_id=self.repo_id,
            role=self.role,
            stratum=self.stratum,
            allowed_phases=self.allowed_phases,
            note=self.note,
            revision=revision,
            descriptive_only=self.descriptive_only,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "repo_id": self.repo_id,
            "role": self.role.value,
            "stratum": self.stratum.value,
            "allowed_phases": sorted(p.value for p in self.allowed_phases),
            "revision": self.revision,
            "descriptive_only": self.descriptive_only,
            "note": self.note,
        }


def _entry(
    key: str,
    repo_id: str,
    role: Role,
    stratum: Stratum,
    phases: Iterable[Phase],
    note: str = "",
    descriptive_only: bool = False,
) -> ModelEntry:
    return ModelEntry(
        key=key,
        repo_id=repo_id,
        role=role,
        stratum=stratum,
        allowed_phases=frozenset(phases),
        note=note,
        descriptive_only=descriptive_only,
    )


#: §8.1 subjects, §8.2 calibration model, §8.3 replication tiers.
REGISTRY: Mapping[str, ModelEntry] = {
    # --- §8.2 calibration (Phase 0 / A only; disjoint from the subjects) --------------
    "calib_qwen25_7b_instruct": _entry(
        "calib_qwen25_7b_instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        Role.CALIBRATION,
        Stratum.CALIBRATION,
        (Phase.PHASE_0, Phase.PHASE_A),
        note="§8.2 calibration model; a community lens exists for conformance fixture 4.",
    ),
    # --- §8.1 primary pair (Phase B onward) -------------------------------------------
    "r1_distill_qwen_14b": _entry(
        "r1_distill_qwen_14b",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        Role.TREATMENT,
        Stratum.PRIMARY_14B,
        (Phase.PHASE_B, Phase.PHASE_C),
        note="§8.1 treatment. Predecessor registered 1df85071...; re-resolve and lock here.",
    ),
    "qwen25_14b_instruct": _entry(
        "qwen25_14b_instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        Role.COMPARATOR,
        Stratum.PRIMARY_14B,
        (Phase.PHASE_B, Phase.PHASE_C),
        note="§8.1 comparator.",
    ),
    "qwen25_14b_base": _entry(
        "qwen25_14b_base",
        "Qwen/Qwen2.5-14B",
        Role.PARENT_ANCHOR,
        Stratum.PRIMARY_14B,
        (Phase.PHASE_B,),
        note="§8.1 shared parent; descriptive anchor only, never enters the primary estimand.",
        descriptive_only=True,
    ),
    # --- §8.3 stratified replication (Phase D only) -----------------------------------
    "r1_distill_qwen_7b": _entry(
        "r1_distill_qwen_7b",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        Role.TREATMENT,
        Stratum.MATH_PARENT_7B,
        (Phase.PHASE_D,),
        note="§8.3 Math-parent tier. Independent report; never pooled with the 14B tier.",
    ),
    "qwen25_math_7b_instruct": _entry(
        "qwen25_math_7b_instruct",
        "Qwen/Qwen2.5-Math-7B-Instruct",
        Role.COMPARATOR,
        Stratum.MATH_PARENT_7B,
        (Phase.PHASE_D,),
        note="§8.3 Math-parent tier comparator.",
    ),
    "r1_distill_qwen_32b": _entry(
        "r1_distill_qwen_32b",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        Role.TREATMENT,
        Stratum.GENERAL_PARENT_32B,
        (Phase.PHASE_D,),
        note="§8.3 general-parent tier.",
    ),
    "qwen25_32b_instruct": _entry(
        "qwen25_32b_instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        Role.COMPARATOR,
        Stratum.GENERAL_PARENT_32B,
        (Phase.PHASE_D,),
        note="§8.3 general-parent tier comparator.",
    ),
}

#: §8.3 / fact F1 — explicitly refused, with the reason carried into the error message.
DENYLIST: Mapping[str, str] = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": (
        "fact F1: on the upstream evaluations the 1.5B scored 2/93, 2/55, 5/90 and the lens "
        "was never read; §8.3 states it is not a subject and §13 forbids any statement about it."
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": (
        "not registered in §8; a different parent lineage would break the shared-parent design."
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": (
        "not registered in §8; a different parent lineage would break the shared-parent design."
    ),
}

_REPO_ID_INDEX = {entry.repo_id: entry.key for entry in REGISTRY.values()}


# --------------------------------------------------------------------------------------
# Revision locking (§8.1)
# --------------------------------------------------------------------------------------


def load_revision_lock() -> Dict[str, Dict[str, str]]:
    path = revision_lock_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("revisions", {})


def write_revision_lock(revisions: Mapping[str, str], *, note: str = "") -> Dict[str, object]:
    """Write ``configs/model_revisions.lock.json``.

    Existing entries are never silently overwritten: re-locking a key to a different digest
    raises, because a subject's revision is part of the Freeze-2 seal.
    """
    existing = load_revision_lock()
    merged: Dict[str, Dict[str, str]] = dict(existing)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for key, revision in revisions.items():
        if key not in REGISTRY:
            raise UnregisteredModelError(f"cannot lock unknown registry key {key!r}")
        if not isinstance(revision, str) or len(revision) < 7:
            raise ValueError(f"revision for {key!r} must be a commit digest, got {revision!r}")
        prior = existing.get(key)
        if prior is not None and prior["revision"] != revision:
            raise UnlockedRevisionError(
                f"{key!r} is already locked to {prior['revision']!r}; re-locking to "
                f"{revision!r} would invalidate the seal. Scientific re-runs are forbidden (§11)."
            )
        merged[key] = {
            "repo_id": REGISTRY[key].repo_id,
            "revision": revision,
            "locked_at": prior["locked_at"] if prior else now,
        }
    payload = {
        "schema": "wda/model_revisions.lock/1",
        "note": note or "§8.1 resolved-and-locked checkpoint revisions.",
        "revisions": merged,
    }
    write_json_lf(revision_lock_path(), payload, sort_keys=True)
    return payload


# --------------------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------------------


def _blinded_roles() -> frozenset:
    # Imported lazily so that ``blind`` may import the registry's enums without a cycle.
    from wda.governance import blind

    return blind.withheld_roles()


def resolve(
    key: str,
    phase: Phase,
    *,
    require_revision: bool = True,
) -> ModelEntry:
    """Return the registered entry for ``key``, or raise.

    Parameters
    ----------
    key:
        Registry key, or the full ``repo_id`` (both are accepted so that call sites and
        config files may use whichever is clearer).
    phase:
        The phase the caller is executing in.
    require_revision:
        When ``True`` (the default, and mandatory for any real model load) the checkpoint
        must already be locked in ``configs/model_revisions.lock.json``.
    """
    if key in DENYLIST:
        raise DeniedModelError(f"{key!r} is on the denylist — {DENYLIST[key]}")
    lookup = key if key in REGISTRY else _REPO_ID_INDEX.get(key, "")
    if not lookup:
        known = ", ".join(sorted(REGISTRY))
        raise UnregisteredModelError(
            f"{key!r} is not registered in §8. Registered keys: {known}. "
            "Loading an unregistered checkpoint is a protocol violation."
        )
    entry = REGISTRY[lookup]

    if phase not in entry.allowed_phases:
        allowed = ", ".join(sorted(p.value for p in entry.allowed_phases))
        raise PhaseViolation(
            f"{entry.key!r} ({entry.role.value}) may only be loaded in [{allowed}], "
            f"not in {phase.value!r}."
        )

    withheld = _blinded_roles()
    if entry.role in withheld:
        # Deferred import keeps ``errors`` the single source of the exception type.
        from wda.errors import BlindingViolation

        raise BlindingViolation(
            f"blinding is active: role {entry.role.value!r} is withheld, so {entry.key!r} "
            "cannot be resolved. §10 Phase A must not touch a subject model."
        )

    revision = load_revision_lock().get(entry.key, {}).get("revision")
    if require_revision and not revision:
        raise UnlockedRevisionError(
            f"{entry.key!r} has no locked revision. Run "
            f"`python -m wda.models.registry lock --key {entry.key} --revision <sha>` "
            "before loading it (§8.1)."
        )
    return entry.with_revision(revision)


def subjects(stratum: Stratum = Stratum.PRIMARY_14B) -> Dict[Role, ModelEntry]:
    """The treatment/comparator pair for a stratum (the anchor is excluded)."""
    out: Dict[Role, ModelEntry] = {}
    for entry in REGISTRY.values():
        if entry.stratum is stratum and entry.role in (Role.TREATMENT, Role.COMPARATOR):
            out[entry.role] = entry
    return out


def registry_snapshot() -> Dict[str, object]:
    """Serialisable registry state, embedded into every ``config.json``."""
    locked = load_revision_lock()
    return {
        "schema": "wda/registry_snapshot/1",
        "entries": [
            REGISTRY[key].with_revision(locked.get(key, {}).get("revision")).to_dict()
            for key in sorted(REGISTRY)
        ],
        "denylist": dict(sorted(DENYLIST.items())),
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Model registry (§8, §15.1).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="print the registry snapshot")

    lock = sub.add_parser("lock", help="lock a checkpoint revision (§8.1)")
    lock.add_argument("--key", required=True)
    lock.add_argument("--revision", required=True, help="resolved commit digest on the Hub")

    args = parser.parse_args(argv)
    if args.cmd == "list":
        print(json.dumps(registry_snapshot(), indent=2, ensure_ascii=False))
        return 0
    payload = write_revision_lock({args.key: args.revision})
    print(json.dumps(payload["revisions"][args.key], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(_main())
