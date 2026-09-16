"""FP32 HF identity-hook canary. No nonzero intervention or scientific endpoint.

Torch is imported only inside execution functions. CPU toy models are legitimate
engineering fixtures but must never be labelled calibration-model evidence.
"""

from __future__ import annotations

import hashlib
import time

from wda.conditions.native import exposure_receipt
from wda.intervene import precision
from wda.lens.upstream import QualificationPending


def tensor_sha256(tensor) -> str:
    data = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(data.dtype).encode())
    digest.update(str(data.shape).encode())
    digest.update(data.tobytes())
    return digest.hexdigest()


def model_fp32_receipt(model) -> dict:
    parameters = list(model.named_parameters())
    if not parameters:
        raise QualificationPending("model has no parameters to verify")
    count = 0
    devices = set()
    for name, parameter in parameters:
        precision.require_fp32(parameter, context=f"model parameter {name}")
        if parameter.requires_grad:
            raise QualificationPending("upstream adapter must freeze model weight gradients")
        count += parameter.numel()
        devices.add(str(parameter.device))
    for name, buffer in model.named_buffers():
        if buffer.is_floating_point():
            precision.require_fp32(buffer, context=f"model buffer {name}")
    return {
        "parameter_dtype": "float32", "parameter_count": count,
        "parameter_bytes": count * 4, "devices": sorted(devices),
        "weights_require_grad": False,
    }


def _hidden(output):
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise QualificationPending("unsupported decoder block output; expected Tensor or tensor-first tuple")


def exercise_noop(model, adapter, prompt, envelope, *, max_prompt_tokens: int) -> dict:
    """Greedy clean/noop pair with measured hook positions and native EOS stopping.

    The initial forward predicts the first emitted token. Thus N emitted tokens
    normally produce N-1 processed generation positions, not N hook exposures.
    No model accuracy, G, D, fitted lens, or positive-control score is computed.
    """
    import torch

    model_fp32_receipt(model)
    if model.training:
        raise QualificationPending("canary requires eval mode")
    length = len(prompt.input_ids)
    if length > max_prompt_tokens:
        raise QualificationPending("prompt exceeds the sealed cap; silent truncation is forbidden")
    if adapter.n_layers != len(adapter.layers) or adapter.n_layers < 2:
        raise QualificationPending("upstream adapter layer layout is inconsistent")
    source_layers = tuple(range(adapter.n_layers - 1))
    eos = getattr(model.generation_config, "eos_token_id", None)
    eos_ids = set(eos if isinstance(eos, (tuple, list)) else ([] if eos is None else [eos]))
    if not eos_ids:
        raise QualificationPending("native generation EOS token IDs must be explicit")
    started = time.monotonic()

    def decode(noop):
        handles, events, logits_cpu, output_tokens = [], [], [], []
        final_activation = {}
        call_positions = []
        expected_seq_len = 0
        processed_count = 0
        hook_calls = {layer: 0 for layer in source_layers}

        def hook(layer):
            def identity(_module, _inputs, output):
                hidden = _hidden(output)
                precision.require_fp32(hidden, context=f"layer {layer} hook")
                if tuple(hidden.shape) != (1, expected_seq_len, adapter.d_model):
                    raise QualificationPending(f"unexpected hidden shape at layer {layer}")
                hook_calls[layer] += 1
                events.append({
                    "layer": layer, "shape": list(hidden.shape), "dtype": str(hidden.dtype),
                    "device": str(hidden.device), "positions": list(call_positions),
                })
                return output
            return identity

        def final_hook(_module, _inputs, output):
            final_activation["hidden"] = _hidden(output)
            return output

        if noop:
            for layer in source_layers:
                handles.append(adapter.layers[layer].register_forward_hook(hook(layer)))
            handles.append(adapter.layers[-1].register_forward_hook(final_hook))
        try:
            current = torch.tensor([prompt.input_ids], dtype=torch.long, device=adapter.input_device)
            past = None
            stop = "max_new_tokens"
            readout_checks = []
            with torch.inference_mode():
                for step in range(envelope.max_new_tokens):
                    expected_seq_len = current.shape[1]
                    call_positions[:] = (
                        list(range(length - envelope.L_p, length))
                        if step == 0 and envelope.condition != "C_gen"
                        else ([] if step == 0 else [length + step - 1])
                    )
                    if step:
                        processed_count += 1
                    outputs = model(
                        input_ids=current, past_key_values=past, use_cache=True,
                        attention_mask=torch.ones(
                            (1, length + step), dtype=torch.long, device=adapter.input_device
                        ),
                        return_dict=True,
                    )
                    logits = outputs.logits
                    precision.require_fp32(logits, context="real forward logits")
                    if not torch.isfinite(logits).all():
                        raise QualificationPending("nonfinite canary logits")
                    logits_cpu.append(logits.detach().cpu().clone())
                    if noop:
                        readout_checks.append(
                            precision.check_noop(
                                logits, adapter.unembed(final_activation["hidden"]),
                                context="true final norm + lm_head versus HF logits",
                            ).require().to_dict()
                        )
                    token = int(logits[0, -1].argmax().item())
                    output_tokens.append(token)
                    if token in eos_ids:
                        stop = "eos"
                        break
                    if step + 1 < envelope.max_new_tokens:
                        past = outputs.past_key_values
                        if past is None:
                            raise QualificationPending(
                                "model did not return a cache; repeated prefill must not "
                                "be silently counted as one prompt exposure"
                            )
                        current = torch.tensor([[token]], device=adapter.input_device)
            if noop and any(count != len(output_tokens) for count in hook_calls.values()):
                raise QualificationPending("one or more required layer hooks failed to run")
            measured = exposure_receipt(
                envelope, length, generated_token_count=len(output_tokens),
                processed_generated_count=processed_count, stop_reason=stop,
                layer_count=len(source_layers),
            )
            if noop and sum(len(event["positions"]) for event in events) != measured["actual_position_layer_exposures"]:
                raise QualificationPending("observed hook calls differ from the exposure receipt")
            return logits_cpu, output_tokens, measured, events, readout_checks
        finally:
            for handle in handles:
                handle.remove()

    clean = decode(False)
    noop = decode(True)
    if clean[1] != noop[1] or len(clean[0]) != len(noop[0]):
        raise QualificationPending("clean and noop greedy sequences differ")
    checks = [
        precision.check_noop(a, b, context=f"identity hook forward step {i}").require().to_dict()
        for i, (a, b) in enumerate(zip(clean[0], noop[0]))
    ]
    return {
        "schema": "wda/engineering_noop/1",
        "qualification": "engineering_only_not_scientific_phase0",
        "native_prompt": prompt.receipt(),
        "model": model_fp32_receipt(model),
        "n_layers": adapter.n_layers, "d_model": adapter.d_model,
        "source_layers": list(source_layers), "target_layer": adapter.n_layers - 1,
        "noop_checks": checks, "true_final_norm_readout_checks": noop[4],
        "clean_logits_sha256": [tensor_sha256(t) for t in clean[0]],
        "noop_logits_sha256": [tensor_sha256(t) for t in noop[0]],
        "output_token_ids": clean[1],
        "exposure": noop[2], "hook_events": noop[3],
        "elapsed_seconds": time.monotonic() - started,
        "reference_readout": "not_run", "known_intermediate_positive": "not_run",
    }
