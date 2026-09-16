"""Toy-only adapter and exposure tests; no registered checkpoint is loaded."""

from types import SimpleNamespace

import numpy as np
import pytest

from wda.conditions.envelopes import C_DIRECT, C_FROZEN, C_GEN, build_envelopes, window
from wda.conditions.native import NativePrompt, exposure_receipt, paired_exposure, require_answer_cue_suffix
from wda.errors import ProtocolViolation
from wda.intervene.ablate import project_out
from wda.intervene.subspace import SubspaceProjector, canonical_svd, require_scientific_interventions
from wda.lens.fit import rms_norm
from wda.lens.upstream import QualificationPending, known_intermediate_fixture, reference_readout_fixture


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_noop_preserves_dtype_and_bytes(dtype):
    vector = np.array([-0.0, 1.25, 1234.5], dtype=dtype)
    output = project_out(vector, np.empty((0, 3), dtype=dtype))
    assert output.dtype == vector.dtype
    assert output.tobytes() == vector.tobytes()


def test_duplicate_basis_does_not_invent_an_extra_direction():
    projector = SubspaceProjector.prepare(np.array([[1, 0, 0], [2, 0, 0]]), 3)
    assert projector.rank == 1
    result = projector.remove(np.array([2, 3, 4], dtype=np.float32))
    np.testing.assert_array_equal(result, [0, 3, 4])


def test_zero_basis_removes_nothing():
    projector = SubspaceProjector.prepare(np.zeros((2, 3)), 3)
    assert projector.rank == 0


def test_svd_orientation_does_not_change_reconstruction():
    matrix = np.array([[1.2, -0.2], [-0.7, 1.0]])
    u, s, vt = canonical_svd(matrix)
    np.testing.assert_allclose((u * s) @ vt, matrix)
    assert (vt[np.arange(2), np.argmax(np.abs(vt), axis=1)] >= 0).all()


def test_rmsnorm_includes_per_dimension_gamma():
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    gamma = np.array([2, 3], dtype=np.float32)
    expected = x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + 1e-5) * gamma
    np.testing.assert_allclose(rms_norm(x, 1e-5, weight=gamma), expected)


def test_generated_cot_never_ablates_prompt():
    env = build_envelopes()[C_GEN]
    result = window(env, 30, processed_generated_count=4)
    assert result.prompt_positions == ()
    assert result.generated_positions == (30, 31, 32, 33)
    assert result.rule == "generated_only" and result.observed


def test_equal_budgets_do_not_hide_eos_exposure_mismatch():
    env = build_envelopes()
    first = exposure_receipt(env[C_DIRECT], 30, generated_token_count=2,
                             processed_generated_count=1, stop_reason="eos", layer_count=27)
    second = exposure_receipt(env[C_FROZEN], 60, generated_token_count=8,
                              processed_generated_count=7, stop_reason="max_new_tokens", layer_count=27)
    result = paired_exposure(first, second)
    assert result["exposure_mismatch"] is True
    assert not first["padding_added"] and not second["padding_added"]
    assert first["actual_position_layer_exposures"] == (3+1)*27


def test_native_header_is_not_mislabeled_as_answer_cue():
    prompt = NativePrompt(C_DIRECT, "Final answer:<assistant>", (1,2,3,4,5),
                          (1,2), (2,3,4), False, "a"*64)
    with pytest.raises(ProtocolViolation, match="not_qualified"):
        require_answer_cue_suffix(prompt)


@pytest.mark.parametrize("call", [
    require_scientific_interventions, reference_readout_fixture, known_intermediate_fixture
])
def test_missing_real_fixtures_cannot_return_pass(call):
    with pytest.raises(ProtocolViolation, match="not_qualified"):
        call()


@pytest.mark.needs_torch
def test_toy_fp32_identity_hooks_preserve_logits():
    torch = pytest.importorskip("torch")
    from wda.lens.model_adapter import exercise_noop

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 4)
            self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
            self.norm = torch.nn.LayerNorm(4)
            self.lm_head = torch.nn.Linear(4, 8)
            self.generation_config = SimpleNamespace(eos_token_id=7)
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)

        def forward(self, input_ids, **kwargs):
            h = self.embed(input_ids)
            for layer in self.layers:
                h = layer(h)
            return SimpleNamespace(logits=self.lm_head(self.norm(h)), past_key_values=("toy",))

    torch.manual_seed(8)
    model = Toy()
    adapter = SimpleNamespace(
        layers=model.layers, n_layers=3, d_model=4, input_device="cpu",
        unembed=lambda h: model.lm_head(model.norm(h)),
    )
    prompt = NativePrompt(C_DIRECT, "synthetic", (1,2,3,4), (1,2,3), (1,2,3), True, "a"*64)
    result = exercise_noop(model, adapter, prompt, build_envelopes(max_new_tokens=3)[C_DIRECT],
                          max_prompt_tokens=8)
    assert result["qualification"] == "engineering_only_not_scientific_phase0"
    assert all(check["bitexact"] for check in result["noop_checks"])
    assert result["clean_logits_sha256"] == result["noop_logits_sha256"]
    assert result["known_intermediate_positive"] == "not_run"
