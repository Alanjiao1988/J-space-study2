"""Offline regressions for signed exact answer surfaces and ITT summaries."""

import json

import numpy as np
import pytest

from wda.scoring.parse import (
    ScoringViolation,
    TripleEndpoint,
    Verdict,
    parse_direct_answer,
    parse_generated_cot,
    summarize,
)


@pytest.mark.parametrize(
    "surface,expected",
    [("42", 42), ("+42", 42), ("-42", -42), ("\t -0042 \n", -42),
     ("0", 0), ("-0", 0), ("+000", 0)],
)
def test_exact_signed_ascii_integer(surface, expected):
    result = parse_direct_answer(surface, expected)
    assert result.correct and result.parsed
    assert result.value == expected


@pytest.mark.parametrize("surface,expected", [("-42", 42), ("42", -42), ("+42", -42)])
def test_sign_is_part_of_the_value(surface, expected):
    result = parse_direct_answer(surface, expected)
    assert result.verdict is Verdict.INCORRECT_WRONG_VALUE
    assert result.parsed and not result.correct


@pytest.mark.parametrize(
    "surface",
    [
        "I cannot answer 42", "The answer is 42", "abc42xyz", "42.", "42 dollars",
        "$42", "4.2e1", "42/1", "4,200", "4 2", "- 42", "+ 42", "--42", "+-42",
        "42\nnot really", "42 Final answer:", "٤٢", "４２", "²", "−42", "＋42",
        "\u200b42", "\\boxed{42}", "<think>42</think>", "42 <eos>",
    ],
)
def test_invalid_surface_is_never_digit_extracted(surface):
    result = parse_direct_answer(surface, 42)
    assert not result.parsed and not result.correct
    assert result.value is None


def test_multiple_integer_diagnostic_does_not_select_an_answer():
    result = parse_direct_answer("first 12 then 42", 42)
    assert result.verdict is Verdict.UNPARSEABLE_MULTIPLE_INTEGERS
    assert result.value is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Step 1: 42 - 84 = -42. Final answer: -42", -42),
        ("Final answer: -41\nCorrection. Final answer:\t-42\n", -42),
        ("<think>42</think>\nFinal answer: +42", 42),
    ],
)
def test_c_gen_requires_complete_final_cue_suffix(text, expected):
    result = parse_generated_cot(text, expected)
    assert result.correct and result.value == expected


@pytest.mark.parametrize(
    "text",
    [
        "Final answer: I cannot answer 42", "Final answer: 42.",
        "Final answer: 42\nExplanation", "Final answer: ４２",
        "Final answer: 42\nFinal answer: no answer",
        "Final answer: 42\nFinal answer:", "Final answer: 4 42",
        "final answer: 42", "The answer is 42",
    ],
)
def test_c_gen_never_falls_back_to_an_earlier_number(text):
    result = parse_generated_cot(text, 42)
    assert not result.parsed and result.value is None


def test_c_gen_sign_and_literal_custom_cue():
    assert not parse_generated_cot("Final answer: -42", 42).correct
    assert parse_generated_cot("work\nANSWER=-42", -42, answer_cue="ANSWER=").correct


@pytest.mark.parametrize("cue", ["", " ", None, 42])
def test_invalid_answer_cue(cue):
    with pytest.raises(ValueError, match="answer_cue"):
        parse_generated_cot("Final answer: 42", 42, answer_cue=cue)


@pytest.mark.parametrize("expected", [True, 42.0, "42", np.nan, None])
@pytest.mark.parametrize("parser", [parse_direct_answer, parse_generated_cot])
def test_expected_integer_cannot_be_coerced(parser, expected):
    with pytest.raises(TypeError, match="expected"):
        parser("42", expected)


def test_all_attempts_remain_in_itt_denominator():
    results = [
        parse_direct_answer(surface, 42)
        for surface in ("42", "-42", "I cannot answer 42", "", "４２")
    ]
    endpoint = summarize(results, parseability_threshold=0.8)
    assert (endpoint.n, endpoint.n_parsed, endpoint.n_correct) == (5, 2, 1)
    assert endpoint.parseability == 0.4
    assert endpoint.conditional_accuracy == 0.5
    assert endpoint.itt_accuracy == 0.2
    assert endpoint.feasibility_result


def test_undefined_conditional_rate_serializes_as_null_not_nan():
    endpoint = summarize(
        [parse_direct_answer("<think>", 42) for _ in range(240)],
        parseability_threshold=0.8,
    )
    payload = json.loads(json.dumps(endpoint.to_dict(), allow_nan=False))
    assert payload["n"] == 240
    assert payload["itt_accuracy"] == 0
    assert payload["conditional_accuracy"] is None
    assert payload["feasibility_result"] is True


def test_empty_endpoint_has_no_success_shaped_rates():
    payload = summarize([], parseability_threshold=0.8).to_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["n"] == 0
    assert payload["parseability"] is None
    assert payload["conditional_accuracy"] is None
    assert payload["itt_accuracy"] is None


@pytest.mark.parametrize(
    "counts",
    [(-1, 0, 0), (1, -1, 0), (1, 0, -1), (1, 2, 1), (2, 1, 2),
     (1.0, 1, 1), (True, 1, 1), (1, np.nan, 0), (1, 1, "1")],
)
def test_invalid_endpoint_counts(counts):
    with pytest.raises(ScoringViolation, match="count|integer"):
        TripleEndpoint(*counts, parseability_threshold=0.8)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf, -np.inf, True, "0.8", None])
def test_invalid_parseability_threshold(threshold):
    with pytest.raises(ScoringViolation, match="threshold"):
        TripleEndpoint(1, 1, 1, threshold)


def test_numpy_counts_and_threshold_serialize_and_threshold_boundary_is_strict():
    endpoint = TripleEndpoint(np.int64(10), np.int64(8), np.int64(7), np.float64(0.8))
    json.dumps(endpoint.to_dict(), allow_nan=False)
    assert not endpoint.feasibility_result
