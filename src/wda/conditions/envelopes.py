"""Response-format conditions and window W (§5, §15.2).

Four envelopes; all use the model's **native chat template**.

=====================  ===========================================================
``C_direct``           question + "final integer only"
``C_frozen``           question + a **pre-frozen correct rationale supplied by the
                       generator** (never model-generated) + "final integer only".
                       This is the primary control.
``C_gen``              question + "reason step by step, then give the final
                       integer" — secondary, comparable to the paper's GSM8K setup
``C_direct_prefill``   exploratory only: ``C_direct`` with a ``<think></think>``
                       prefill, to probe the fact-F4 compliance failure
=====================  ===========================================================

**Window W (frozen).** The last ``L_p`` prompt positions (the answer-cue suffix) plus every
generated answer position. Because ``L_p`` and ``max_new_tokens`` are identical for
``C_direct`` and ``C_frozen``, the two conditions receive **exactly the same number of
ablation exposures**; the only difference between them is whether a correct rationale is
present in the context. :func:`assert_window_parity` enforces this rather than assuming it,
because the whole interpretation of ``G_primary`` rests on it.

**Compliance (fact F4).** The R1-Distill checkpoints may refuse to answer directly and open
a ``<think>`` span. The primary conditions therefore use **no prefill**; parseability is
reported per condition, and a rate below the Freeze-1 threshold is recorded as a
*feasibility result*, not repaired (§9.4).

Following the predecessor's ``study4f_interfaces.py``, all surface handling is exact string
construction — never regex assembly — so that a mis-anchored pattern is structurally
impossible rather than merely unlikely.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from wda.errors import ProtocolViolation

C_DIRECT = "C_direct"
C_FROZEN = "C_frozen"
C_GEN = "C_gen"
C_DIRECT_PREFILL = "C_direct_prefill"

#: Only these two enter G_primary (§5).
PRIMARY_CONDITIONS: Tuple[str, str] = (C_DIRECT, C_FROZEN)

WINDOW_RULE = "last_L_p_prompt_plus_generated"


class EnvelopeViolation(ProtocolViolation):
    """§5 — an envelope broke a frozen element of the condition definition."""


@dataclass(frozen=True)
class Envelope:
    """§15.2 condition configuration."""

    condition: str
    system: str
    answer_cue: str
    L_p: int
    max_new_tokens: int
    chat_template: str = "native"
    rationale_source: Optional[str] = None
    prefill: Optional[str] = None
    instruction: str = ""
    ablation_window: str = WINDOW_RULE
    exploratory: bool = False

    def __post_init__(self) -> None:
        if self.chat_template != "native":
            raise EnvelopeViolation("§5 requires the model's native chat template in all conditions")
        if self.ablation_window != WINDOW_RULE:
            raise EnvelopeViolation(f"the window rule is frozen as {WINDOW_RULE!r}")
        if self.L_p < 1:
            raise EnvelopeViolation("L_p must be >= 1")
        if self.max_new_tokens < 1:
            raise EnvelopeViolation("max_new_tokens must be >= 1")
        if self.condition == C_FROZEN and self.rationale_source != "generator":
            raise EnvelopeViolation(
                "C_frozen's rationale must come from the item generator, never from the "
                "model under test (§5)."
            )
        if self.condition in PRIMARY_CONDITIONS and self.prefill is not None:
            raise EnvelopeViolation(
                "the primary conditions use no prefill (§5); C_direct_prefill is an "
                "exploratory arm and must be declared exploratory=True."
            )
        if self.condition == C_DIRECT_PREFILL and not self.exploratory:
            raise EnvelopeViolation("C_direct_prefill is exploratory only (§5, §8.5)")

    @property
    def window_size(self) -> int:
        """``|W|`` in tokens: the ablation-exposure count."""
        return self.L_p + self.max_new_tokens

    def to_dict(self) -> Dict[str, object]:
        return {
            "condition": self.condition,
            "chat_template": self.chat_template,
            "system": self.system,
            "rationale_source": self.rationale_source,
            "prefill": self.prefill,
            "answer_cue": self.answer_cue,
            "L_p": self.L_p,
            "max_new_tokens": self.max_new_tokens,
            "ablation_window": self.ablation_window,
            "exploratory": self.exploratory,
        }

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()


# --------------------------------------------------------------------------------------
# The frozen envelopes (Freeze-1 values live in configs/freeze1/envelopes.json)
# --------------------------------------------------------------------------------------

DIRECT_SYSTEM = "Answer with the final integer only."
GEN_SYSTEM = "Reason step by step, then give the final integer."
ANSWER_CUE = "Final answer:"

#: Frozen at Freeze-1. ``L_p = 3`` matches the tokenised length of the answer cue on the
#: Qwen tokenizer family; :func:`assert_cue_length` re-verifies it against the real
#: tokenizer before the first model call rather than trusting the constant.
DEFAULT_L_P = 3
DEFAULT_MAX_NEW_TOKENS = 8


def build_envelopes(
    *, l_p: int = DEFAULT_L_P, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
) -> Dict[str, Envelope]:
    """The four registered envelopes with a shared ``L_p`` / ``max_new_tokens``."""
    common = {"answer_cue": ANSWER_CUE, "L_p": l_p, "max_new_tokens": max_new_tokens}
    return {
        C_DIRECT: Envelope(
            condition=C_DIRECT,
            system=DIRECT_SYSTEM,
            rationale_source=None,
            **common,
        ),
        C_FROZEN: Envelope(
            condition=C_FROZEN,
            system=DIRECT_SYSTEM,
            rationale_source="generator",
            **common,
        ),
        C_GEN: Envelope(
            condition=C_GEN,
            system=GEN_SYSTEM,
            rationale_source=None,
            answer_cue=ANSWER_CUE,
            L_p=l_p,
            # §5: C_gen ablates *all* generated positions, aligned with the paper's GSM8K
            # setup. Its exposure count therefore differs from the primary pair by design,
            # which is exactly why G_gen is secondary and not the endpoint.
            max_new_tokens=512,
        ),
        C_DIRECT_PREFILL: Envelope(
            condition=C_DIRECT_PREFILL,
            system=DIRECT_SYSTEM,
            rationale_source=None,
            prefill="<think>\n\n</think>",
            exploratory=True,
            **common,
        ),
    }


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


@dataclass
class RenderedPrompt:
    """A rendered condition, ready for the chat template."""

    condition: str
    messages: Tuple[Dict[str, str], ...]
    prefill: Optional[str]
    answer_cue: str
    item_id: str

    def user_content(self) -> str:
        return next(m["content"] for m in self.messages if m["role"] == "user")

    def to_dict(self) -> Dict[str, object]:
        return {
            "condition": self.condition,
            "messages": [dict(m) for m in self.messages],
            "prefill": self.prefill,
            "answer_cue": self.answer_cue,
            "item_id": self.item_id,
        }


def render(envelope: Envelope, question: str, *, item_id: str, rationale: Optional[str] = None) -> RenderedPrompt:
    """Build the message list for one item under one envelope.

    ``C_frozen`` requires a generator-supplied rationale; supplying one to any other
    condition raises, because that would silently turn the control into the treatment.
    """
    if envelope.condition == C_FROZEN:
        if not rationale:
            raise EnvelopeViolation("C_frozen requires the generator's frozen rationale (§5)")
        body = f"{question}\n\n{rationale}\n\n{envelope.answer_cue}"
    else:
        if rationale:
            raise EnvelopeViolation(
                f"a rationale was supplied to {envelope.condition!r}; only C_frozen may "
                "carry one (§5)."
            )
        body = f"{question}\n\n{envelope.answer_cue}"
    return RenderedPrompt(
        condition=envelope.condition,
        messages=(
            {"role": "system", "content": envelope.system},
            {"role": "user", "content": body},
        ),
        prefill=envelope.prefill,
        answer_cue=envelope.answer_cue,
        item_id=item_id,
    )


# --------------------------------------------------------------------------------------
# Window W
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """The ablation window for one trial (§5)."""

    prompt_positions: Tuple[int, ...]
    generated_positions: Tuple[int, ...]
    rule: str = WINDOW_RULE

    @property
    def positions(self) -> Tuple[int, ...]:
        return self.prompt_positions + self.generated_positions

    def __len__(self) -> int:
        return len(self.positions)

    def to_dict(self) -> Dict[str, object]:
        return {
            "rule": self.rule,
            "prompt_positions": list(self.prompt_positions),
            "generated_positions": list(self.generated_positions),
            "size": len(self),
        }


def window(envelope: Envelope, prompt_length: int) -> Window:
    """Window W: the last ``L_p`` prompt positions plus every generated position."""
    if prompt_length < envelope.L_p:
        raise EnvelopeViolation(
            f"prompt of {prompt_length} tokens is shorter than L_p={envelope.L_p}; "
            "the answer-cue suffix does not fit."
        )
    prompt_positions = tuple(range(prompt_length - envelope.L_p, prompt_length))
    generated = tuple(range(prompt_length, prompt_length + envelope.max_new_tokens))
    return Window(prompt_positions=prompt_positions, generated_positions=generated)


def assert_window_parity(a: Envelope, b: Envelope) -> int:
    """§5 — ``C_direct`` and ``C_frozen`` must present identical ablation exposure.

    Returns the shared window size. Raises if the two envelopes differ in ``L_p`` or
    ``max_new_tokens``, which would confound ``G_primary`` with exposure count.
    """
    for env in (a, b):
        if env.condition not in PRIMARY_CONDITIONS:
            raise EnvelopeViolation(
                f"window parity is a requirement on the primary pair; {env.condition!r} "
                "is not one of them."
            )
    if a.condition == b.condition:
        raise EnvelopeViolation("window parity compares the two distinct primary conditions")
    if a.L_p != b.L_p or a.max_new_tokens != b.max_new_tokens:
        raise EnvelopeViolation(
            f"window mismatch: {a.condition} has (L_p={a.L_p}, max_new={a.max_new_tokens}) "
            f"but {b.condition} has (L_p={b.L_p}, max_new={b.max_new_tokens}). The two "
            "primary conditions must receive the same number of ablation exposures (§5)."
        )
    if a.ablation_window != b.ablation_window:
        raise EnvelopeViolation("the two primary conditions use different window rules")
    return a.window_size


def assert_cue_length(
    tokenize: Callable[[str], Sequence[int]], envelope: Envelope
) -> int:
    """Verify ``L_p`` against the real tokenizer before the first model call.

    Fact F3 is the reason this exists: the predecessor discovered only at execution time
    that its registered content surfaces tokenised to two tokens rather than one, which
    made the registered rule unimplementable as written.
    """
    n = len(list(tokenize(envelope.answer_cue)))
    if n != envelope.L_p:
        raise EnvelopeViolation(
            f"answer cue {envelope.answer_cue!r} tokenises to {n} tokens but L_p is "
            f"{envelope.L_p}. Fix L_p in configs/freeze1 *before* Freeze-1, or choose a cue "
            "whose tokenisation matches; this may not be adjusted after the seal (§11)."
        )
    return n


def envelopes_snapshot(envelopes: Dict[str, Envelope]) -> Dict[str, object]:
    return {
        "schema": "wda/envelopes/1",
        "window_rule": WINDOW_RULE,
        "primary_conditions": list(PRIMARY_CONDITIONS),
        "primary_window_size": assert_window_parity(
            envelopes[C_DIRECT], envelopes[C_FROZEN]
        ),
        "envelopes": {name: env.to_dict() for name, env in sorted(envelopes.items())},
        "digests": {name: env.digest() for name, env in sorted(envelopes.items())},
    }


__all__ = [
    "ANSWER_CUE",
    "C_DIRECT",
    "C_DIRECT_PREFILL",
    "C_FROZEN",
    "C_GEN",
    "DEFAULT_L_P",
    "DEFAULT_MAX_NEW_TOKENS",
    "Envelope",
    "EnvelopeViolation",
    "PRIMARY_CONDITIONS",
    "RenderedPrompt",
    "WINDOW_RULE",
    "Window",
    "assert_cue_length",
    "assert_window_parity",
    "build_envelopes",
    "envelopes_snapshot",
    "render",
    "window",
]
