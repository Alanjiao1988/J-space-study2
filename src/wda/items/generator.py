"""Programmatic item bank (§8.4 primary set, §15.5 record schema).

Specification, frozen at Freeze-1:

* 2-4 step arithmetic word problems;
* a deterministic solver gives the unique integer answer;
* every item carries a **templated correct rationale** for ``C_frozen`` — supplied by the
  generator, never by the model under test;
* answers are 1-3 digit integers;
* items are stratified by step count;
* the generator seed and a version hash enter Freeze-2;
* the exploration and confirmation rounds use **disjoint** item sets from different seeds.

Rationale for a generated bank rather than GSM8K as the primary set: the R1-Distill
checkpoints were trained on 800k reasoning samples that very plausibly cover GSM8K-style
problems, so contamination is a real risk. GSM8K is retained as the *secondary* anchor
(see :mod:`wda.items.gsm8k_split`) purely for comparability with the paper.

CLI::

    python -m wda.items.generator --split explore --n 40 --out runs/phase0/items.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from wda.errors import ProtocolViolation

GENERATOR_VERSION = "wda-items/1.0.0"

#: §8.4 — answers are 1-3 digit integers.
MIN_ANSWER = 1
MAX_ANSWER = 999

#: §8.4 — 2 to 4 reasoning steps, stratified.
STEP_COUNTS: Tuple[int, ...] = (2, 3, 4)

#: Disjoint seeds. The confirmation seed is *not* used before Freeze-2.
SPLIT_SEEDS: Dict[str, int] = {"explore": 20260902, "confirm": 20260903}


class ItemGenerationError(ProtocolViolation):
    """§8.4 — the generator could not satisfy a frozen constraint."""


_NAMES = (
    "Ana", "Bo", "Cara", "Dev", "Elin", "Femi", "Gita", "Hugo",
    "Ivy", "Jonas", "Kira", "Luca", "Mina", "Nils", "Olu", "Pia",
)
_OBJECTS = (
    ("marbles", "marble"), ("stickers", "sticker"), ("apples", "apple"),
    ("coins", "coin"), ("pencils", "pencil"), ("cards", "card"),
    ("beads", "bead"), ("stamps", "stamp"),
)
_CONTAINERS = ("boxes", "bags", "trays", "jars", "crates")


@dataclass(frozen=True)
class Step:
    """One solver step: a sentence, a rationale line, and the running value."""

    sentence: str
    rationale: str
    value: int


@dataclass(frozen=True)
class Item:
    """§15.5 item record."""

    item_id: str
    generator_version: str
    seed: int
    split: str
    n_steps: int
    question: str
    rationale: str
    answer_int: int
    answer_token_ids: Optional[Tuple[int, ...]] = None
    single_token: Optional[bool] = None
    template_id: str = "arith_chain_v1"

    def to_dict(self) -> Dict[str, object]:
        return {
            "item_id": self.item_id,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "split": self.split,
            "n_steps": self.n_steps,
            "question": self.question,
            "rationale": self.rationale,
            "answer_int": self.answer_int,
            "answer_token_ids": list(self.answer_token_ids) if self.answer_token_ids else None,
            "single_token": self.single_token,
            "template_id": self.template_id,
        }

    def with_tokenization(self, token_ids: Sequence[int]) -> "Item":
        ids = tuple(int(t) for t in token_ids)
        return Item(
            item_id=self.item_id,
            generator_version=self.generator_version,
            seed=self.seed,
            split=self.split,
            n_steps=self.n_steps,
            question=self.question,
            rationale=self.rationale,
            answer_int=self.answer_int,
            answer_token_ids=ids,
            single_token=len(ids) == 1,
            template_id=self.template_id,
        )


# --------------------------------------------------------------------------------------
# Step constructors — each is a pure function of (rng, name, plural, singular, value)
# --------------------------------------------------------------------------------------


def _add(rng, name, plural, singular, value) -> Optional[Step]:
    n = int(rng.integers(2, 40))
    return Step(
        sentence=f"Then {name} buys {n} more {plural}.",
        rationale=f"{value} + {n} = {value + n}",
        value=value + n,
    )


def _subtract(rng, name, plural, singular, value) -> Optional[Step]:
    if value <= 2:
        return None
    n = int(rng.integers(1, max(2, min(value - 1, 30))))
    return Step(
        sentence=f"Then {name} gives away {n} {plural}.",
        rationale=f"{value} - {n} = {value - n}",
        value=value - n,
    )


def _multiply(rng, name, plural, singular, value) -> Optional[Step]:
    if value < 1 or value > MAX_ANSWER // 2:
        return None
    factor = int(rng.integers(2, 5))
    if value * factor > MAX_ANSWER:
        return None
    return Step(
        sentence=f"Then {name} ends up with {factor} times that many {plural}.",
        rationale=f"{value} x {factor} = {value * factor}",
        value=value * factor,
    )


def _divide(rng, name, plural, singular, value) -> Optional[Step]:
    divisors = [d for d in (2, 3, 4, 5) if value % d == 0 and value // d >= MIN_ANSWER]
    if not divisors:
        return None
    d = int(rng.choice(divisors))
    container = str(rng.choice(_CONTAINERS))
    return Step(
        sentence=(
            f"Then {name} splits the {plural} evenly into {d} {container} "
            f"and keeps just one {container[:-1]}."
        ),
        rationale=f"{value} / {d} = {value // d}",
        value=value // d,
    )


_STEP_FNS: Tuple[Callable, ...] = (_add, _subtract, _multiply, _divide)


def _make_item(rng: np.random.Generator, n_steps: int, split: str, seed: int, index: int) -> Optional[Item]:
    name = str(rng.choice(_NAMES))
    plural, singular = _OBJECTS[int(rng.integers(0, len(_OBJECTS)))]
    start = int(rng.integers(6, 60))

    steps: List[Step] = []
    value = start
    attempts = 0
    while len(steps) < n_steps and attempts < 40:
        attempts += 1
        fn = _STEP_FNS[int(rng.integers(0, len(_STEP_FNS)))]
        step = fn(rng, name, plural, singular, value)
        if step is None or not (MIN_ANSWER <= step.value <= MAX_ANSWER):
            continue
        steps.append(step)
        value = step.value

    if len(steps) != n_steps:
        return None
    if not MIN_ANSWER <= value <= MAX_ANSWER:
        return None

    question = (
        f"{name} starts with {start} {plural}. "
        + " ".join(s.sentence for s in steps)
        + f" How many {plural} does {name} have now?"
    )
    # Soft admissibility: an item whose question literally contains its own answer is
    # solvable by copying, so it is redrawn rather than reported as a generator fault.
    if str(value) in question:
        return None
    rationale_lines = [f"Step {i + 1}: {s.rationale}." for i, s in enumerate(steps)]
    rationale = "\n".join(rationale_lines) + f"\nSo the answer is {value}."

    payload = f"{GENERATOR_VERSION}|{split}|{seed}|{n_steps}|{index}|{question}"
    item_id = f"{split}-{n_steps}s-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    item = Item(
        item_id=item_id,
        generator_version=GENERATOR_VERSION,
        seed=seed,
        split=split,
        n_steps=n_steps,
        question=question,
        rationale=rationale,
        answer_int=value,
    )
    verify(item)
    return item


# --------------------------------------------------------------------------------------
# Deterministic solver / verifier
# --------------------------------------------------------------------------------------


def solve_rationale(rationale: str) -> int:
    """Re-evaluate the rationale arithmetically — the item's own independent check.

    The rationale is the object ``C_frozen`` places in the context, so it must be *proved*
    correct rather than assumed: a wrong frozen rationale would silently turn the primary
    control into a different treatment.
    """
    value: Optional[int] = None
    for line in rationale.splitlines():
        line = line.strip().rstrip(".")
        if not line.startswith("Step"):
            continue
        if ":" not in line:
            raise ItemGenerationError(f"unparseable rationale step: {line!r}")
        expr = line.split(":", 1)[1].strip()
        parts = expr.split("=")
        if len(parts) != 2:
            raise ItemGenerationError(f"unparseable rationale step: {line!r}")
        left, right = parts
        try:
            right_value = int(right.strip())
            tokens = left.strip().split()
            if len(tokens) != 3:
                raise ItemGenerationError(f"unparseable rationale step: {line!r}")
            a, op, b = int(tokens[0]), tokens[1], int(tokens[2])
        except ValueError as exc:
            raise ItemGenerationError(f"unparseable rationale step: {line!r}") from exc
        if op not in {"+", "-", "x", "/"}:
            raise ItemGenerationError(f"unknown operator {op!r} in: {line!r}")
        if op == "/" and a % b != 0:
            raise ItemGenerationError(f"non-exact division in rationale: {line!r}")
        computed = {"+": a + b, "-": a - b, "x": a * b, "/": a // b if b else None}[op]
        if computed != right_value:
            raise ItemGenerationError(f"rationale arithmetic is wrong: {line!r}")
        if value is not None and a != value:
            raise ItemGenerationError(f"rationale chain is broken at: {line!r}")
        value = computed
    if value is None:
        raise ItemGenerationError("rationale contains no steps")
    return value


def verify(item: Item) -> Item:
    """Assert every frozen §8.4 constraint on a single item."""
    if item.n_steps not in STEP_COUNTS:
        raise ItemGenerationError(f"n_steps={item.n_steps} outside {STEP_COUNTS}")
    if not MIN_ANSWER <= item.answer_int <= MAX_ANSWER:
        raise ItemGenerationError(f"answer {item.answer_int} outside [{MIN_ANSWER}, {MAX_ANSWER}]")
    if solve_rationale(item.rationale) != item.answer_int:
        raise ItemGenerationError(f"{item.item_id}: rationale does not derive the stated answer")
    if str(item.answer_int) not in item.rationale:
        raise ItemGenerationError(f"{item.item_id}: rationale does not state the answer")
    if str(item.answer_int) in item.question:
        # A question that literally contains its own answer is solvable by copying.
        raise ItemGenerationError(f"{item.item_id}: the answer appears verbatim in the question")
    return item


# --------------------------------------------------------------------------------------
# Bank generation
# --------------------------------------------------------------------------------------


def generate(
    n: int,
    *,
    split: str = "explore",
    seed: Optional[int] = None,
    step_counts: Sequence[int] = STEP_COUNTS,
) -> List[Item]:
    """Generate a step-stratified bank of ``n`` items with unique questions."""
    if split not in SPLIT_SEEDS:
        raise ItemGenerationError(f"unknown split {split!r}; expected {sorted(SPLIT_SEEDS)}")
    seed = SPLIT_SEEDS[split] if seed is None else seed
    rng = np.random.default_rng(seed)

    per_stratum = [n // len(step_counts)] * len(step_counts)
    for i in range(n - sum(per_stratum)):
        per_stratum[i] += 1

    items: List[Item] = []
    seen_questions = set()
    for n_steps, target in zip(step_counts, per_stratum):
        made = 0
        guard = 0
        while made < target:
            guard += 1
            if guard > 400 * max(target, 1):  # pragma: no cover - defensive
                raise ItemGenerationError(
                    f"could not generate {target} items with {n_steps} steps under the "
                    "frozen constraints"
                )
            item = _make_item(rng, n_steps, split, seed, len(items))
            if item is None or item.question in seen_questions:
                continue
            seen_questions.add(item.question)
            items.append(item)
            made += 1
    return items


def assert_splits_disjoint(a: Iterable[Item], b: Iterable[Item]) -> None:
    """§8.4 — the exploration and confirmation rounds must not share items."""
    qa = {item.question for item in a}
    qb = {item.question for item in b}
    overlap = qa & qb
    if overlap:
        raise ItemGenerationError(
            f"exploration and confirmation banks share {len(overlap)} question(s); §8.4 "
            "requires disjoint sets generated from different seeds."
        )


def bank_digest(items: Sequence[Item]) -> str:
    """Content hash of a bank — enters ``FREEZE-2.json`` together with the seed."""
    h = hashlib.sha256()
    h.update(GENERATOR_VERSION.encode("utf-8"))
    for item in sorted(items, key=lambda i: i.item_id):
        h.update(json.dumps(item.to_dict(), sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def bank_manifest(items: Sequence[Item]) -> Dict[str, object]:
    strata: Dict[int, int] = {}
    for item in items:
        strata[item.n_steps] = strata.get(item.n_steps, 0) + 1
    answers = [item.answer_int for item in items]
    return {
        "schema": "wda/item_bank/1",
        "generator_version": GENERATOR_VERSION,
        "split": items[0].split if items else None,
        "seed": items[0].seed if items else None,
        "n_items": len(items),
        "strata": {str(k): v for k, v in sorted(strata.items())},
        "answer_range": [min(answers), max(answers)] if answers else None,
        "bank_digest": bank_digest(items),
    }


def write_jsonl(items: Sequence[Item], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for item in items:
            fh.write(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def read_jsonl(path: Path) -> List[Item]:
    out: List[Item] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            token_ids = payload.get("answer_token_ids")
            out.append(
                Item(
                    item_id=payload["item_id"],
                    generator_version=payload["generator_version"],
                    seed=payload["seed"],
                    split=payload["split"],
                    n_steps=payload["n_steps"],
                    question=payload["question"],
                    rationale=payload["rationale"],
                    answer_int=payload["answer_int"],
                    answer_token_ids=tuple(token_ids) if token_ids else None,
                    single_token=payload.get("single_token"),
                    template_id=payload.get("template_id", "arith_chain_v1"),
                )
            )
    return out


def _main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Programmatic item bank (§8.4).")
    parser.add_argument("--split", default="explore", choices=sorted(SPLIT_SEEDS))
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.split == "confirm" and args.out is not None:
        # §10 / §16: the confirmation bank is used for the first time *after* the Freeze-2
        # seal. Materialising it earlier would put an unsealed confirmation set on disk.
        from wda.paths import seal_path

        if not seal_path("freeze2").exists():
            raise SystemExit(
                "refusing to write the confirmation item bank before FREEZE-2.json exists. "
                "The confirmation seed is used for the first time after the Freeze-2 seal "
                "(§10, §16). Use --split explore for the exploration round, or omit --out "
                "to print the manifest without materialising the bank."
            )

    items = generate(args.n, split=args.split, seed=args.seed)
    manifest = bank_manifest(items)
    if args.out:
        write_jsonl(items, args.out)
        manifest["path"] = str(args.out)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(_main())


__all__ = [
    "GENERATOR_VERSION",
    "Item",
    "ItemGenerationError",
    "MAX_ANSWER",
    "MIN_ANSWER",
    "SPLIT_SEEDS",
    "STEP_COUNTS",
    "assert_splits_disjoint",
    "bank_digest",
    "bank_manifest",
    "generate",
    "read_jsonl",
    "solve_rationale",
    "verify",
    "write_jsonl",
]
