"""Native-template position and measured-exposure receipts, without model imports.

No assistant prefill is added to the primary conditions. Native assistant headers
often follow the user-message cue; this is reported rather than relabelled as a cue.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional

from wda.conditions.envelopes import Envelope, EnvelopeViolation, render, window


def token_digest(ids) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class NativePrompt:
    condition: str
    text: str
    input_ids: tuple[int, ...]
    cue_positions: tuple[int, ...]
    trailing_positions: tuple[int, ...]
    cue_is_trailing_suffix: bool
    template_sha256: str

    def receipt(self) -> dict:
        return {
            "condition": self.condition,
            "prompt_length": len(self.input_ids),
            "prompt_text_sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
            "prompt_tokens_sha256": token_digest(self.input_ids),
            "chat_template_sha256": self.template_sha256,
            "cue_positions": list(self.cue_positions),
            "trailing_L_p_positions": list(self.trailing_positions),
            "trailing_L_p_token_ids": [self.input_ids[p] for p in self.trailing_positions],
            "cue_is_trailing_suffix": self.cue_is_trailing_suffix,
            "scientific_window": "qualified_boundary_only" if self.cue_is_trailing_suffix else "not_qualified",
        }


def render_native(
    tokenizer: Any, envelope: Envelope, question: str, *, item_id: str,
    rationale: Optional[str] = None,
) -> NativePrompt:
    """Tokenize the actual native conversation and locate its cue with exact offsets."""
    prompt = render(envelope, question, item_id=item_id, rationale=rationale)
    if prompt.prefill is not None:
        raise EnvelopeViolation("engineering canary does not execute exploratory prefills")
    messages = [dict(m) for m in prompt.messages]
    template = tokenizer.get_chat_template()
    if not isinstance(template, str) or not template:
        raise EnvelopeViolation("a nonempty native chat template is required")
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if not isinstance(text, str) or not isinstance(ids, (list, tuple)) or not ids:
        raise EnvelopeViolation("native template must return nonempty text and a flat token list")
    ids = tuple(int(i) for i in ids)
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except (TypeError, NotImplementedError, ValueError) as exc:
        raise EnvelopeViolation("native position verification requires exact tokenizer offsets") from exc
    if tuple(encoded["input_ids"]) != ids:
        raise EnvelopeViolation("native template token IDs differ from offset-mapped encoding")
    start = text.rfind(envelope.answer_cue)
    if start < 0:
        raise EnvelopeViolation("native chat template dropped the registered answer cue")
    end = start + len(envelope.answer_cue)
    offsets = encoded["offset_mapping"]
    if len(offsets) != len(ids):
        raise EnvelopeViolation("native tokenizer offsets do not align with input IDs")
    cue_positions = tuple(i for i, (a, b) in enumerate(offsets) if a < end and b > start)
    trailing = tuple(range(max(0, len(ids) - envelope.L_p), len(ids)))
    exact = (
        cue_positions == trailing and len(cue_positions) == envelope.L_p
        and text.endswith(envelope.answer_cue)
        and offsets[cue_positions[0]][0] == start and offsets[cue_positions[-1]][1] == end
    )
    return NativePrompt(
        envelope.condition, text, ids, cue_positions, trailing, exact,
        hashlib.sha256(template.encode("utf-8")).hexdigest(),
    )


def require_answer_cue_suffix(prompt: NativePrompt) -> None:
    if not prompt.cue_is_trailing_suffix:
        raise EnvelopeViolation(
            "not_qualified: native template's final L_p tokens are not the registered "
            "answer cue. Moving the cue into an assistant prefill would change the "
            "primary no-prefill condition; resolve and seal the window definition first."
        )


def exposure_receipt(
    envelope: Envelope, prompt_length: int, *, generated_token_count: int,
    processed_generated_count: int, stop_reason: str, layer_count: int,
) -> dict:
    """Count emitted tokens separately from positions actually seen by decoder hooks."""
    if (
        type(generated_token_count) is not int or type(processed_generated_count) is not int
        or type(layer_count) is not int or layer_count < 1
        or not 0 <= processed_generated_count <= generated_token_count <= envelope.max_new_tokens
        or stop_reason not in ("eos", "max_new_tokens", "forward_only")
    ):
        raise EnvelopeViolation("invalid measured exposure counts or stop reason")
    measured = window(
        envelope, prompt_length, processed_generated_count=processed_generated_count
    )
    return {
        "condition": envelope.condition,
        "prompt_length": prompt_length,
        "generated_token_count": generated_token_count,
        "processed_generated_count": processed_generated_count,
        "stop_reason": stop_reason,
        "prompt_exposures_per_layer": len(measured.prompt_positions),
        "generation_exposures_per_layer": len(measured.generated_positions),
        "layer_count": layer_count,
        "actual_position_layer_exposures": len(measured) * layer_count,
        "nominal_window_upper_bound": envelope.window_size,
        "window": measured.to_dict(),
        "padding_added": False,
    }


def paired_exposure(left: dict, right: dict) -> dict:
    keys = (
        "generated_token_count", "prompt_exposures_per_layer",
        "generation_exposures_per_layer", "actual_position_layer_exposures",
    )
    differences = {key: left[key] - right[key] for key in keys}
    return {
        "left_condition": left["condition"], "right_condition": right["condition"],
        "left_minus_right": differences,
        "exposure_mismatch": any(differences.values()),
        "policy": "report_mismatch_no_padding_no_exclusion_no_scientific_endpoint",
    }
