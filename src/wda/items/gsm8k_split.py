"""GSM8K held-out split (§8.4 secondary set).

GSM8K is the **secondary** anchor only. It exists so that ``G_primary`` can be reported on
the same benchmark the paper used, and it never enters the primary endpoint: the
R1-Distill checkpoints were fine-tuned on 800k reasoning traces that plausibly cover
GSM8K-style problems, so contamination is a live risk and the primary set is generated
instead (§8.4).

The split is a deterministic hash partition, so it is reproducible from the seed alone and
does not depend on the order of the source file. No network access happens here: the
operator places ``gsm8k_test.jsonl`` (fields ``question`` and ``answer``) under
``configs/freeze1/data/`` and its digest enters the seal.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from wda.errors import ProtocolViolation
from wda.items.generator import Item

SPLIT_SALT = "wda/gsm8k-split/v1"

#: Fraction assigned to the held-out (used) partition.
HELD_OUT_FRACTION = 0.5

_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+)")
_CALC_RE = re.compile(r"<<[^>]*>>")


class GSM8KError(ProtocolViolation):
    """§8.4 — the GSM8K source could not be used as registered."""


@dataclass(frozen=True)
class GSM8KRecord:
    index: int
    question: str
    rationale: str
    answer_int: int

    def to_item(self, split: str) -> Item:
        digest = hashlib.sha256(self.question.encode("utf-8")).hexdigest()[:12]
        return Item(
            item_id=f"gsm8k-{split}-{digest}",
            generator_version="gsm8k/official",
            seed=0,
            split=split,
            # GSM8K does not label step counts; 0 marks "not stratified by construction".
            n_steps=0,
            question=self.question,
            rationale=self.rationale,
            answer_int=self.answer_int,
            template_id="gsm8k_official",
        )


def parse_answer(answer_field: str) -> Tuple[str, int]:
    """Split the official ``answer`` field into (rationale, integer answer)."""
    match = _ANSWER_RE.search(answer_field)
    if not match:
        raise GSM8KError(f"no '#### <answer>' marker in: {answer_field[:80]!r}")
    value = int(match.group(1).replace(",", ""))
    rationale = _CALC_RE.sub("", answer_field[: match.start()]).strip()
    return rationale, value


def load(path: Path) -> List[GSM8KRecord]:
    """Load ``gsm8k_test.jsonl`` (one ``{"question", "answer"}`` object per line)."""
    path = Path(path)
    if not path.exists():
        raise GSM8KError(
            f"{path} is absent. Place the GSM8K test split there and record its digest in "
            "the seal; this module never downloads it (§16)."
        )
    records: List[GSM8KRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rationale, value = parse_answer(payload["answer"])
            records.append(
                GSM8KRecord(
                    index=index,
                    question=payload["question"].strip(),
                    rationale=rationale,
                    answer_int=value,
                )
            )
    if not records:
        raise GSM8KError(f"{path} contained no records")
    return records


def _bucket(question: str, seed: int) -> float:
    digest = hashlib.sha256(f"{SPLIT_SALT}|{seed}|{question}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def split(
    records: Sequence[GSM8KRecord],
    *,
    seed: int,
    held_out_fraction: float = HELD_OUT_FRACTION,
) -> Dict[str, List[GSM8KRecord]]:
    """Deterministic hash partition into ``held_out`` and ``reserved``.

    Only ``held_out`` is ever used. ``reserved`` is retained unopened so that a later
    question — should one ever be authorised — has an untouched partition available.
    """
    if not 0.0 < held_out_fraction < 1.0:
        raise GSM8KError("held_out_fraction must lie strictly between 0 and 1")
    held_out: List[GSM8KRecord] = []
    reserved: List[GSM8KRecord] = []
    for record in records:
        (held_out if _bucket(record.question, seed) < held_out_fraction else reserved).append(record)
    return {"held_out": held_out, "reserved": reserved}


def to_items(records: Iterable[GSM8KRecord], *, split_name: str = "gsm8k_held_out") -> List[Item]:
    return [record.to_item(split_name) for record in records]


def manifest(path: Path, parts: Dict[str, List[GSM8KRecord]], seed: int) -> Dict[str, object]:
    return {
        "schema": "wda/gsm8k_split/1",
        "source": str(path),
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "salt": SPLIT_SALT,
        "seed": seed,
        "counts": {name: len(part) for name, part in sorted(parts.items())},
        "tier": "secondary_confirmatory",
        "note": (
            "Secondary anchor only (§8.4, §8.5). Interpreted only if the primary endpoint "
            "is material, and then under Holm correction (§9.6)."
        ),
    }


__all__ = [
    "GSM8KError",
    "GSM8KRecord",
    "HELD_OUT_FRACTION",
    "SPLIT_SALT",
    "load",
    "manifest",
    "parse_answer",
    "split",
    "to_items",
]
