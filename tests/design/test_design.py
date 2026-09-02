"""Design-layer tests: envelopes and window W (§5), the triple endpoint (§9.4),
the band rule (§6.3), dual-lens aggregation (§6.2), the item bank (§8.4),
the assay registry (§8.5) and the run scaffolding (§15.4)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from wda.assays import registry as assays
from wda.assays.primary_interaction import PrimaryAssayError, assert_design_complete
from wda.conditions.envelopes import (
    C_DIRECT,
    C_DIRECT_PREFILL,
    C_FROZEN,
    C_GEN,
    Envelope,
    EnvelopeViolation,
    assert_cue_length,
    assert_window_parity,
    build_envelopes,
    envelopes_snapshot,
    render,
    window,
)
from wda.errors import AggregationViolation, BandRuleViolation
from wda.items import generator as gen
from wda.lens.aggregate import PairedLenses, select_stronger
from wda.lens.band import (
    BandRule,
    LayerStatistics,
    band_halves,
    effective_dimension,
    excess_kurtosis,
    identity_energy,
    locate_band,
    position_shuffled_null,
    top1_autocorrelation,
    top_k_hit_rate,
)
from wda.lens.fit import (
    LensBundle,
    LensFitViolation,
    LensSpec,
    assert_corpora_disjoint,
    disjoint_samples,
    estimate_jacobian,
    identity_lens,
    shuffle_within_prompt,
)
from wda.phases import AssayTier
from wda.scoring.parse import (
    TripleEndpoint,
    Verdict,
    feasibility_report,
    parse_direct_answer,
    parse_generated_cot,
    summarize,
    verdict_breakdown,
)


# ========================================================== §5 envelopes and window W ===


class TestEnvelopes:
    def test_primary_conditions_have_identical_window_size(self):
        envs = build_envelopes()
        size = assert_window_parity(envs[C_DIRECT], envs[C_FROZEN])
        assert size == envs[C_DIRECT].window_size == envs[C_FROZEN].window_size

    def test_mismatched_windows_are_refused(self):
        a = Envelope(condition=C_DIRECT, system="s", answer_cue="Final answer:", L_p=3, max_new_tokens=8)
        b = Envelope(
            condition=C_FROZEN,
            system="s",
            answer_cue="Final answer:",
            L_p=3,
            max_new_tokens=16,
            rationale_source="generator",
        )
        with pytest.raises(EnvelopeViolation, match="same number of ablation exposures"):
            assert_window_parity(a, b)

    def test_c_frozen_rationale_must_come_from_the_generator(self):
        with pytest.raises(EnvelopeViolation, match="item generator"):
            Envelope(
                condition=C_FROZEN,
                system="s",
                answer_cue="Final answer:",
                L_p=3,
                max_new_tokens=8,
                rationale_source="model",
            )

    def test_primary_conditions_forbid_prefill(self):
        with pytest.raises(EnvelopeViolation, match="no prefill"):
            Envelope(
                condition=C_DIRECT,
                system="s",
                answer_cue="Final answer:",
                L_p=3,
                max_new_tokens=8,
                prefill="<think></think>",
            )

    def test_prefill_variant_must_be_exploratory(self):
        with pytest.raises(EnvelopeViolation, match="exploratory only"):
            Envelope(
                condition=C_DIRECT_PREFILL,
                system="s",
                answer_cue="Final answer:",
                L_p=3,
                max_new_tokens=8,
                prefill="<think>\n\n</think>",
                exploratory=False,
            )

    def test_non_native_template_is_refused(self):
        with pytest.raises(EnvelopeViolation, match="native chat template"):
            Envelope(
                condition=C_DIRECT,
                system="s",
                answer_cue="Final answer:",
                L_p=3,
                max_new_tokens=8,
                chat_template="raw",
            )

    def test_render_places_the_rationale_only_in_c_frozen(self):
        envs = build_envelopes()
        item = gen.generate(3)[0]
        frozen = render(envs[C_FROZEN], item.question, item_id=item.item_id, rationale=item.rationale)
        assert item.rationale in frozen.user_content()

        direct = render(envs[C_DIRECT], item.question, item_id=item.item_id)
        assert item.rationale not in direct.user_content()
        assert direct.user_content().endswith("Final answer:")

        with pytest.raises(EnvelopeViolation, match="only C_frozen"):
            render(envs[C_DIRECT], item.question, item_id=item.item_id, rationale=item.rationale)

    def test_c_frozen_without_a_rationale_is_refused(self):
        envs = build_envelopes()
        with pytest.raises(EnvelopeViolation, match="frozen rationale"):
            render(envs[C_FROZEN], "q", item_id="i")

    def test_window_is_the_cue_suffix_plus_generation(self):
        env = build_envelopes()[C_DIRECT]
        w = window(env, prompt_length=100)
        assert w.prompt_positions == (97, 98, 99)
        assert w.generated_positions == tuple(range(100, 108))
        assert len(w) == env.L_p + env.max_new_tokens

    def test_short_prompt_is_refused(self):
        env = build_envelopes()[C_DIRECT]
        with pytest.raises(EnvelopeViolation, match="shorter than L_p"):
            window(env, prompt_length=2)

    def test_cue_length_is_verified_against_a_tokenizer(self):
        """Fact F3 precedent: verify the surface tokenises as registered, before running."""
        env = build_envelopes()[C_DIRECT]
        assert assert_cue_length(lambda s: [1, 2, 3], env) == 3
        with pytest.raises(EnvelopeViolation, match="tokenises to 5 tokens"):
            assert_cue_length(lambda s: [1, 2, 3, 4, 5], env)

    def test_c_gen_ablates_all_generated_positions(self):
        envs = build_envelopes()
        assert envs[C_GEN].max_new_tokens > envs[C_DIRECT].max_new_tokens

    def test_snapshot_is_serialisable(self):
        json.dumps(envelopes_snapshot(build_envelopes()))


# ============================================================== §9.4 triple endpoint ===


class TestScoring:
    def test_exact_single_integer_parses(self):
        r = parse_direct_answer(" 42", 42)
        assert r.verdict is Verdict.CORRECT and r.value == 42

    def test_wrong_value_is_parsed_but_incorrect(self):
        r = parse_direct_answer("41", 42)
        assert r.parsed is True and r.correct is False

    def test_multiple_integers_are_unparseable(self):
        """A rule that picks one of several could absorb a compliance failure silently."""
        r = parse_direct_answer("first 12 then 42", 42)
        assert r.verdict is Verdict.UNPARSEABLE_MULTIPLE_INTEGERS
        assert r.correct is False

    def test_reasoning_span_is_classified_not_merely_missing(self):
        """Fact F4: the R1-Distill checkpoints open <think> instead of answering."""
        r = parse_direct_answer("<think>\nlet me work this out", 42)
        assert r.verdict is Verdict.UNPARSEABLE_REASONING_SPAN

    def test_empty_output_is_unparseable(self):
        assert parse_direct_answer("   ", 42).verdict is Verdict.UNPARSEABLE_EMPTY

    def test_cot_answer_is_taken_after_the_final_cue(self):
        text = "Step 1: 2+2 = 4. Final answer: 4"
        assert parse_generated_cot(text, 4).correct is True
        assert parse_generated_cot("no cue here 4", 4).verdict is Verdict.UNPARSEABLE_NO_INTEGER

    def test_itt_keeps_unparseable_in_the_denominator(self):
        results = [parse_direct_answer(s, 7) for s in ("7", "7", "<think>", "  ")]
        ep = summarize(results, parseability_threshold=0.80)
        assert ep.n == 4 and ep.n_parsed == 2 and ep.n_correct == 2
        assert ep.parseability == 0.5
        assert ep.conditional_accuracy == 1.0     # conditional: misleadingly perfect
        assert ep.itt_accuracy == 0.5             # ITT: the endpoint that counts
        assert ep.feasibility_result is True

    def test_all_unparseable_gives_zero_itt_not_a_division_error(self):
        """The F4 scenario: 240/240 unparseable must yield 0.0, not NaN or a crash."""
        results = [parse_direct_answer("<think>", 7) for _ in range(240)]
        ep = summarize(results, parseability_threshold=0.80)
        assert ep.itt_accuracy == 0.0
        assert np.isnan(ep.conditional_accuracy)
        assert ep.feasibility_result is True

    def test_feasibility_report_lists_failing_conditions(self):
        good = TripleEndpoint(n=100, n_parsed=100, n_correct=80, parseability_threshold=0.80)
        bad = TripleEndpoint(n=100, n_parsed=10, n_correct=9, parseability_threshold=0.80)
        report = feasibility_report({C_DIRECT: bad, C_FROZEN: good})
        assert [r["condition"] for r in report] == [C_DIRECT]
        assert "denominator is not reduced" in report[0]["disposition"]

    def test_verdict_breakdown_covers_every_verdict(self):
        counts = verdict_breakdown([parse_direct_answer("7", 7)])
        assert set(counts) == {v.value for v in Verdict}
        assert counts["CORRECT"] == 1


# =================================================================== §6.3 band rule ===


def make_stats(
    *,
    layers=tuple(range(28)),
    hit_peak: float = 0.6,
    null_level: float = 0.01,
    rise_at: int = 20,
) -> LayerStatistics:
    n = len(layers)
    hit = np.zeros(n)
    kurt = np.zeros(n)
    acc = np.zeros(n)
    for i in range(n):
        if i >= 13:
            hit[i] = hit_peak * min(1.0, (i - 12) / 8.0)
        kurt[i] = 0.0 if i < 12 else (i - 12) * 2.0
        acc[i] = 0.02 if i < rise_at + 4 else 0.6
    acc[rise_at + 4] = 0.6
    return LayerStatistics(
        layers=tuple(layers),
        top_k_hit_rate=hit,
        excess_kurtosis=kurt,
        top1_autocorrelation=np.full(n, 0.5),
        autocorrelation_null=np.full(n, 0.1),
        effective_dimension=np.linspace(20, 4, n),
        next_token_accuracy=acc,
        random_lens_null_hit_rate=np.full(n, null_level),
    )


class TestBandRule:
    def test_a_clean_profile_yields_an_admissible_band(self):
        result = locate_band(make_stats(), BandRule())
        assert result.admissible is True
        assert len(result.band) >= BandRule().min_length
        assert result.require() == result.band

    def test_absolute_floor_rejects_a_uniformly_weak_profile(self):
        """Repair 1: the predecessor rule was scale-free and had no absolute floor."""
        weak = make_stats(hit_peak=0.02)   # everything below the 0.05 absolute floor
        result = locate_band(weak, BandRule())
        assert result.admissible is False
        assert any("absolute floor" in r for r in result.reasons)

    def test_relative_rule_alone_would_have_accepted_the_weak_profile(self):
        """Demonstrates why repair 1 is needed: relative-only still fires on noise."""
        weak = make_stats(hit_peak=0.02)
        relative_only = BandRule(absolute_floor=0.0, null_margin=0.0)
        assert locate_band(weak, relative_only).admissible is True
        assert locate_band(weak, BandRule()).admissible is False

    def test_minimum_length_rejects_a_length_one_band(self):
        """Repair 2: the predecessor rule emitted a band of length 1."""
        stats = make_stats()
        result = locate_band(stats, BandRule(min_length=99))
        assert result.admissible is False
        assert any("below the minimum" in r for r in result.reasons)

    def test_null_margin_rejects_a_profile_the_random_lens_matches(self):
        """Repair 3: adding the matched-norm null is what lifted the predecessor's blocker."""
        stats = make_stats(null_level=0.99)
        result = locate_band(stats, BandRule())
        assert result.admissible is False
        assert any("random-lens null" in r for r in result.reasons)

    def test_right_censored_band_is_refused(self):
        """Repair 4: a band ending at the last layer is censored by the end of the network."""
        n = 24
        stats = LayerStatistics(
            layers=tuple(range(n)),
            top_k_hit_rate=np.linspace(0.0, 0.9, n),
            excess_kurtosis=np.linspace(0.0, 10.0, n),
            top1_autocorrelation=np.full(n, 0.5),
            autocorrelation_null=np.full(n, 0.1),
            effective_dimension=np.linspace(20, 4, n),
            next_token_accuracy=np.linspace(0.0, 0.9, n),
            random_lens_null_hit_rate=np.zeros(n),
        )
        result = locate_band(stats, BandRule(min_length=2, accuracy_rise_min_jump=0.0))
        if result.admissible:
            assert result.band[-1] != n - 1
        else:
            assert result.reasons

    def test_inadmissible_band_raises_on_require(self):
        result = locate_band(make_stats(hit_peak=0.02), BandRule())
        with pytest.raises(BandRuleViolation, match="may not be relaxed"):
            result.require()

    def test_band_halves_split_for_the_logit_lens_criterion(self):
        first, second = band_halves((10, 11, 12, 13, 14))
        assert first == (10, 11, 12) and second == (13, 14)

    def test_statistics_behave_as_documented(self):
        rng = np.random.default_rng(0)
        peaked = np.zeros((4, 100))
        peaked[:, 0] = 10.0
        assert excess_kurtosis(peaked) > excess_kurtosis(rng.standard_normal((4, 100)))

        probs = np.eye(5)[[0, 1, 2]]
        assert top_k_hit_rate(probs, [0, 1, 2], 1) == 1.0
        assert top_k_hit_rate(probs, [4, 4, 4], 1) == 0.0

        assert top1_autocorrelation([1, 1, 1, 1]) == 1.0
        mean, _ = position_shuffled_null([1, 2, 3, 4, 5], n_draws=50)
        assert 0.0 <= mean <= 1.0

        assert effective_dimension(np.eye(8)) == pytest.approx(8.0)
        assert identity_energy(np.eye(8)) == pytest.approx(1.0)
        assert identity_energy(np.diag([1.0, -1.0])) < 0.5


# ================================================ §6.1 / §6.2 lens fit and aggregation ===


class TestLens:
    def test_fit_scale_floors_are_enforced(self):
        with pytest.raises(LensFitViolation, match="below the frozen floor of 200"):
            LensSpec("l", "real", "m", n_prompts=25, skip_first=16, n_probes=64)
        with pytest.raises(LensFitViolation, match="skip_first=4"):
            LensSpec("l", "real", "m", n_prompts=400, skip_first=4, n_probes=64)

    def test_logit_lens_has_no_scale_requirement(self):
        bundle = identity_lens(8, (1, 2), "m")
        assert bundle.spec.kind == "logit"
        assert np.allclose(bundle.matrix(1), np.eye(8))

    def test_disjoint_samples_do_not_overlap(self):
        prompts = [f"p{i}" for i in range(500)]
        a, b = disjoint_samples(prompts, n_prompts=200)
        assert_corpora_disjoint(a, b)
        assert a.seed != b.seed

    def test_too_small_corpus_is_refused(self):
        with pytest.raises(LensFitViolation, match="need at least"):
            disjoint_samples([f"p{i}" for i in range(100)], n_prompts=200)

    def test_shuffled_corpus_permutes_within_a_prompt(self):
        tokens = list(range(20))
        shuffled = shuffle_within_prompt(tokens, seed=1)
        assert sorted(shuffled) == tokens
        assert shuffled != tokens

    def test_stochastic_estimator_is_unbiased(self):
        rng = np.random.default_rng(0)
        true_j = rng.standard_normal((6, 6))
        est = estimate_jacobian(lambda u: true_j @ u, 6, n_probes=20000, seed=1, dtype=np.float64)
        assert np.allclose(est, true_j, atol=0.08)

    def test_paired_lenses_average_both_fits(self, synthetic_bundle):
        other = LensBundle(
            spec=LensSpec("real_b", "real", "calib_qwen25_7b_instruct", 256, 16, 128, seed=23,
                          layers=synthetic_bundle.spec.layers),
            jacobians={k: v * 3.0 for k, v in synthetic_bundle.jacobians.items()},
        )
        paired = PairedLenses(synthetic_bundle, other)
        assert np.allclose(paired.mean_jacobian(4), synthetic_bundle.matrix(4) * 2.0)
        assert paired.paired_mean({"real_a": 1.0, "real_b": 3.0}) == 2.0
        assert paired.agreement(4) == pytest.approx(1.0)

    def test_same_seed_pair_is_refused(self, synthetic_bundle):
        clone = LensBundle(
            spec=LensSpec("real_b", "real", "calib_qwen25_7b_instruct", 256, 16, 128, seed=11,
                          layers=synthetic_bundle.spec.layers),
            jacobians=dict(synthetic_bundle.jacobians),
        )
        with pytest.raises(AggregationViolation, match="different seeds"):
            PairedLenses(synthetic_bundle, clone)

    def test_dropping_a_lens_is_refused(self, synthetic_bundle):
        other = LensBundle(
            spec=LensSpec("real_b", "real", "calib_qwen25_7b_instruct", 256, 16, 128, seed=23,
                          layers=synthetic_bundle.spec.layers),
            jacobians=dict(synthetic_bundle.jacobians),
        )
        paired = PairedLenses(synthetic_bundle, other)
        with pytest.raises(AggregationViolation, match="both\\s+lenses to be run"):
            paired.paired_mean({"real_a": 1.0})

    def test_post_hoc_lens_selection_always_raises(self):
        with pytest.raises(AggregationViolation, match="post-hoc selection"):
            select_stronger()


# ============================================================== §8.4 the item bank ===


class TestItemBank:
    def test_bank_is_step_stratified(self):
        items = gen.generate(30)
        counts = {}
        for item in items:
            counts[item.n_steps] = counts.get(item.n_steps, 0) + 1
        assert set(counts) == set(gen.STEP_COUNTS)
        assert max(counts.values()) - min(counts.values()) <= 1

    def test_every_rationale_derives_its_stated_answer(self):
        for item in gen.generate(60):
            assert gen.solve_rationale(item.rationale) == item.answer_int
            gen.verify(item)

    def test_answers_are_one_to_three_digits(self):
        for item in gen.generate(60):
            assert gen.MIN_ANSWER <= item.answer_int <= gen.MAX_ANSWER
            assert 1 <= len(str(item.answer_int)) <= 3

    def test_the_answer_never_appears_in_the_question(self):
        for item in gen.generate(60):
            assert str(item.answer_int) not in item.question

    def test_generation_is_deterministic_from_the_seed(self):
        assert gen.bank_digest(gen.generate(24)) == gen.bank_digest(gen.generate(24))

    def test_explore_and_confirm_banks_are_disjoint(self):
        explore = gen.generate(40, split="explore")
        confirm = gen.generate(40, split="confirm")
        gen.assert_splits_disjoint(explore, confirm)
        assert explore[0].seed != confirm[0].seed

    def test_a_broken_rationale_is_detected(self):
        item = gen.generate(3)[0]
        tampered = gen.Item(
            item_id=item.item_id,
            generator_version=item.generator_version,
            seed=item.seed,
            split=item.split,
            n_steps=item.n_steps,
            question=item.question,
            rationale=item.rationale.replace("Step 1:", "Step 1: 1 + 1 = 3 ;"),
            answer_int=item.answer_int,
        )
        with pytest.raises(gen.ItemGenerationError):
            gen.verify(tampered)

    def test_round_trip_through_jsonl(self, tmp_path):
        items = gen.generate(12)
        path = gen.write_jsonl(items, tmp_path / "items.jsonl")
        assert gen.bank_digest(gen.read_jsonl(path)) == gen.bank_digest(items)

    def test_manifest_records_the_digest_and_strata(self):
        manifest = gen.bank_manifest(gen.generate(12))
        assert manifest["n_items"] == 12
        assert manifest["bank_digest"]
        json.dumps(manifest)

    def test_cli_refuses_to_materialise_the_confirm_bank_before_freeze2(self, tmp_path):
        """§10/§16 — the confirmation seed is used for the first time after Freeze-2."""
        from wda.paths import seal_path

        if seal_path("freeze2").exists():  # pragma: no cover - only after Phase B
            pytest.skip("Freeze-2 already sealed")
        with pytest.raises(SystemExit, match="before FREEZE-2.json exists"):
            gen._main(["--split", "confirm", "--n", "6", "--out", str(tmp_path / "c.jsonl")])
        # Printing the manifest without materialising the bank stays allowed.
        assert gen._main(["--split", "confirm", "--n", "6"]) == 0


# =========================================================== §8.5 the assay registry ===


class TestAssayRegistry:
    def test_exactly_one_primary_assay(self):
        assert assays.by_tier(AssayTier.PRIMARY) == ("primary_interaction",)

    def test_excluded_assays_cannot_be_run(self):
        for name in ("rhyme_planning", "mental_arithmetic"):
            with pytest.raises(assays.AssayViolation, match="excluded"):
                assays.require_tier(name, AssayTier.EXPLORATORY)

    def test_tier_cannot_be_changed_at_analysis_time(self):
        with pytest.raises(assays.AssayViolation, match="registered as exploratory"):
            assays.require_tier("probe_swap", AssayTier.PRIMARY)

    def test_probe_swap_requires_the_direct_substitution_baseline(self):
        assert "direct_substitution_baseline" in assays.spec("probe_swap").required_controls

    def test_unknown_assay_is_refused(self):
        with pytest.raises(assays.AssayViolation, match="not a registered assay"):
            assays.spec("made_up")

    def test_snapshot_is_serialisable(self):
        json.dumps(assays.snapshot())

    def test_each_assay_module_has_an_estimand_spec(self):
        from wda.paths import repo_root

        for name in ("primary_interaction", "flexible_generalization", "probe_swap"):
            assert (repo_root() / "src" / "wda" / "assays" / f"{name}.md").exists()


# ================================================= §9.4 primary design completeness ===


class TestPrimaryDesign:
    def test_incomplete_design_is_refused(self, synthetic_frame):
        partial = synthetic_frame.take(np.flatnonzero(synthetic_frame.condition != C_FROZEN))
        with pytest.raises(PrimaryAssayError, match="empty cells"):
            assert_design_complete(partial)

    def test_single_lens_is_refused(self, make_frame):
        frame = make_frame(n_items=4, lenses=("real_a",))
        with pytest.raises(PrimaryAssayError, match="fewer than two lens fits"):
            assert_design_complete(frame)
