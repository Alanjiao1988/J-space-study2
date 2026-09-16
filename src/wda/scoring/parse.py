"""Answer parsing and the triple endpoint (§9.4).

Three endpoints are reported **per condition**, never pooled:

1. ``parseability`` — the fraction of outputs that yield a legal answer surface;
2. ``conditional`` — accuracy on the parseable subset, always labelled as conditional;
3. ``ITT composite`` — an unparseable output counts as **incorrect**. This is the only
   endpoint that enters the primary estimand.

If any condition's parseability falls below the Freeze-1 threshold, that is recorded as a
**feasibility result**: it is reported, the denominator is not reduced, and nothing is
repaired (§9.4).

Fact F4 is why this module exists in this shape. Under the predecessor's raw-direct route
the 7B and 14B checkpoints produced 240/240 unparseable outputs, the 7B opening with token
``151649`` (``</think>``) and the 14B with ``32313`` (``Okay``). A pipeline that silently
dropped unparseable rows would have reported an accuracy computed on an empty denominator.

The complete answer surface must match ``[+-]?[0-9]+``, allowing whitespace only
around it. Signs are significant; prose, punctuation and Unicode digits are not
answers. For ``C_gen`` this grammar applies to the entire suffix of the final
literal answer cue, not to a number extracted from that suffix.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from wda.errors import ProtocolViolation


class Verdict(str, Enum):
    """Per-trial outcome. Nothing is ever dropped; every trial gets exactly one verdict."""

    CORRECT = "CORRECT"
    INCORRECT_WRONG_VALUE = "INCORRECT_WRONG_VALUE"
    UNPARSEABLE_NO_INTEGER = "UNPARSEABLE_NO_INTEGER"
    UNPARSEABLE_MULTIPLE_INTEGERS = "UNPARSEABLE_MULTIPLE_INTEGERS"
    UNPARSEABLE_REASONING_SPAN = "UNPARSEABLE_REASONING_SPAN"
    UNPARSEABLE_EMPTY = "UNPARSEABLE_EMPTY"
    UNPARSEABLE_INVALID_SURFACE = "UNPARSEABLE_INVALID_SURFACE"

    @property
    def parsed(self) -> bool:
        return self in (Verdict.CORRECT, Verdict.INCORRECT_WRONG_VALUE)


PARSEABLE_VERDICTS = frozenset({Verdict.CORRECT, Verdict.INCORRECT_WRONG_VALUE})

#: Fact F4: the surfaces the R1-Distill checkpoints actually emitted when asked to answer
#: directly. Recognised explicitly so that a compliance failure is *classified*, not merely
#: counted as "no integer found".
REASONING_SPAN_MARKERS: Tuple[str, ...] = ("<think>", "</think>")


class ScoringViolation(ProtocolViolation):
    """§9.4 — an endpoint was computed in a way the protocol forbids."""


@dataclass(frozen=True)
class ParseResult:
    verdict: Verdict
    value: Optional[int]
    surface: str

    @property
    def parsed(self) -> bool:
        return self.verdict.parsed

    @property
    def correct(self) -> bool:
        return self.verdict is Verdict.CORRECT

    def to_dict(self) -> Dict[str, object]:
        return {
            "verdict": self.verdict.value,
            "value": self.value,
            "parsed": self.parsed,
            "correct": self.correct,
            "surface": self.surface,
        }


_SIGNED_ASCII_INTEGER = re.compile(r"[+-]?[0-9]+")


def _validate_inputs(generated_text: str, expected: int) -> None:
    if not isinstance(generated_text, str):
        raise TypeError("generated_text must be a string")
    if isinstance(expected, bool) or not isinstance(expected, Integral):
        raise TypeError("expected must be a signed integer, not a coerced string or boolean")


def _parse_surface(text: str, expected: int) -> ParseResult:
    if not text:
        return ParseResult(Verdict.UNPARSEABLE_EMPTY, None, text)
    if any(marker in text for marker in REASONING_SPAN_MARKERS):
        return ParseResult(Verdict.UNPARSEABLE_REASONING_SPAN, None, text)
    if _SIGNED_ASCII_INTEGER.fullmatch(text) is None:
        # Diagnostic only: never select or coerce a number from an invalid surface.
        count = sum(_SIGNED_ASCII_INTEGER.fullmatch(token) is not None for token in text.split())
        verdict = (
            Verdict.UNPARSEABLE_MULTIPLE_INTEGERS
            if count > 1
            else Verdict.UNPARSEABLE_INVALID_SURFACE
        )
        return ParseResult(verdict, None, text)
    value = int(text)
    verdict = Verdict.CORRECT if value == expected else Verdict.INCORRECT_WRONG_VALUE
    return ParseResult(verdict, value, text)


def parse_direct_answer(generated_text: str, expected: int) -> ParseResult:
    """Parse a ``C_direct`` / ``C_frozen`` continuation (§5: "final integer only").

    Only a complete signed ASCII integer is legal; no prose or digit extraction.
    """
    _validate_inputs(generated_text, expected)
    return _parse_surface(generated_text.strip(), expected)


def parse_generated_cot(generated_text: str, expected: int, *, answer_cue: str = "Final answer:") -> ParseResult:
    """Parse only the exact signed-integer suffix after the final literal answer cue."""
    _validate_inputs(generated_text, expected)
    if not isinstance(answer_cue, str) or not answer_cue.strip():
        raise ValueError("answer_cue must be a nonempty literal string")
    text = generated_text.strip()
    if not text:
        return ParseResult(Verdict.UNPARSEABLE_EMPTY, None, text)
    idx = text.rfind(answer_cue)
    if idx < 0:
        return ParseResult(Verdict.UNPARSEABLE_NO_INTEGER, None, text)
    tail = text[idx + len(answer_cue) :].strip()
    return _parse_surface(tail, expected)


# --------------------------------------------------------------------------------------
# The triple endpoint
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TripleEndpoint:
    """§9.4 — the three endpoints for one (model, condition, ablation state) cell."""

    n: int
    n_parsed: int
    n_correct: int
    parseability_threshold: float

    def __post_init__(self) -> None:
        for name in ("n", "n_parsed", "n_correct"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ScoringViolation(f"{name} must be a nonnegative integer count")
            object.__setattr__(self, name, int(value))
        if not self.n_correct <= self.n_parsed <= self.n:
            raise ScoringViolation("counts must satisfy 0 <= n_correct <= n_parsed <= n")
        threshold = self.parseability_threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not math.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
        ):
            raise ScoringViolation("parseability_threshold must be finite and in [0, 1]")
        object.__setattr__(self, "parseability_threshold", float(threshold))

    @property
    def parseability(self) -> float:
        return self.n_parsed / self.n if self.n else float("nan")

    @property
    def conditional_accuracy(self) -> float:
        """Accuracy on the parseable subset. **Always a conditional quantity.**"""
        return self.n_correct / self.n_parsed if self.n_parsed else float("nan")

    @property
    def itt_accuracy(self) -> float:
        """Unparseable counts as incorrect. The only endpoint entering the estimand."""
        return self.n_correct / self.n if self.n else float("nan")

    @property
    def feasibility_result(self) -> bool:
        """§9.4 — parseability below the Freeze-1 threshold is a reportable outcome."""
        return self.n > 0 and self.parseability < self.parseability_threshold

    def to_dict(self) -> Dict[str, object]:
        """JSON-safe endpoints; undefined rates are null, never JSON NaN."""
        return {
            "n": self.n,
            "n_parsed": self.n_parsed,
            "n_correct": self.n_correct,
            "parseability": self.parseability if self.n else None,
            "conditional_accuracy": self.conditional_accuracy if self.n_parsed else None,
            "conditional_accuracy_label": "conditional on parseability — not an ITT quantity",
            "itt_accuracy": self.itt_accuracy if self.n else None,
            "parseability_threshold": self.parseability_threshold,
            "feasibility_result": self.feasibility_result,
        }


def summarize(
    results: Iterable[ParseResult], *, parseability_threshold: float
) -> TripleEndpoint:
    """Compute the triple endpoint. The denominator is never reduced (§9.4)."""
    rows = list(results)
    n = len(rows)
    n_parsed = sum(1 for r in rows if r.parsed)
    n_correct = sum(1 for r in rows if r.correct)
    if n_correct > n_parsed:  # pragma: no cover - defensive
        raise ScoringViolation("more correct than parseable results; the verdicts are inconsistent")
    return TripleEndpoint(
        n=n,
        n_parsed=n_parsed,
        n_correct=n_correct,
        parseability_threshold=parseability_threshold,
    )


def verdict_breakdown(results: Iterable[ParseResult]) -> Dict[str, int]:
    """Counts per verdict — the diagnostic that made fact F4 legible."""
    counts: Dict[str, int] = {v.value: 0 for v in Verdict}
    for result in results:
        counts[result.verdict.value] += 1
    return counts


def feasibility_report(
    per_condition: Mapping[str, TripleEndpoint]
) -> List[Dict[str, object]]:
    """§17 item 6 — the list of conditions that fell below the parseability threshold."""
    out: List[Dict[str, object]] = []
    for condition, endpoint in sorted(per_condition.items()):
        if endpoint.feasibility_result:
            out.append(
                {
                    "condition": condition,
                    "parseability": endpoint.parseability,
                    "threshold": endpoint.parseability_threshold,
                    "n": endpoint.n,
                    "disposition": (
                        "feasibility result: reported as-is; the denominator is not reduced "
                        "and no prefill or parser change is applied (§9.4)"
                    ),
                }
            )
    return out


def to_trial_record(
    result: ParseResult,
    *,
    run_id: str,
    phase: str,
    model_role: str,
    condition: str,
    ablation_state: str,
    lens_id: str,
    item_id: str,
    template_id: str,
    target: Optional[str],
    output_tokens: Sequence[int],
    kl_vs_clean: Optional[float],
    noop_bitexact: Optional[bool],
    seal_hash: str,
) -> Dict[str, object]:
    """One ``log.jsonl`` line, exactly per §15.4."""
    return {
        "run_id": run_id,
        "phase": phase,
        "model_role": model_role,
        "condition": condition,
        "ablation_state": ablation_state,
        "lens_id": lens_id,
        "item_id": item_id,
        "template_id": template_id,
        "target": target,
        "output_tokens": list(output_tokens),
        "parsed": result.parsed,
        "correct": result.correct,
        "verdict": result.verdict.value,
        "kl_vs_clean": kl_vs_clean,
        "noop_bitexact": noop_bitexact,
        "seal_hash": seal_hash,
    }


__all__ = [
    "PARSEABLE_VERDICTS",
    "REASONING_SPAN_MARKERS",
    "ParseResult",
    "ScoringViolation",
    "TripleEndpoint",
    "Verdict",
    "feasibility_report",
    "parse_direct_answer",
    "parse_generated_cot",
    "summarize",
    "to_trial_record",
    "verdict_breakdown",
]
