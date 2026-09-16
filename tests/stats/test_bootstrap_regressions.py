"""Synthetic complete-bank, dual-fit bootstrap and blinding regressions."""

import json

import numpy as np
import pytest

from wda.assays.primary_interaction import PrimaryAssayError, analyze, assert_design_complete
from wda.errors import BlindingViolation
from wda.governance.blind import phase_a_blinding
from wda.stats import bootstrap as bs
from wda.stats import power as pw
from wda.stats.decision import Interval, decide, holm


def records(n_items=5, *, unique_ids=True, both_models=False):
    rows = []
    for role in ("treatment", "comparator"):
        for fit in ("a", "b"):
            for i in range(n_items):
                for condition in (bs.C_DIRECT, bs.C_FROZEN):
                    for state in (bs.CLEAN, bs.ABLATED):
                        rows.append({
                            "model_role": role,
                            "lens_id": f"{role}_{fit}" if unique_ids else f"real_{fit}",
                            "item_id": f"item_{i}",
                            "template_id": f"template_{i}",
                            "condition": condition,
                            "ablation_state": state,
                            "parsed": True,
                            "correct": (
                                (role == "treatment" or both_models) and fit == "a"
                                and condition == bs.C_DIRECT and state == bs.CLEAN
                            ),
                        })
    return rows


def frame(**kwargs):
    return bs.TrialFrame.from_records(records(**kwargs))


def test_separate_fit_ids_never_omit_a_model_or_drop_replicates():
    result = bs.hierarchical_bootstrap(frame(), n_resamples=200, seed=0)
    assert result.estimate == 0.5
    assert result.replicates.shape == (200,)
    assert np.isfinite(result.replicates).all()
    assert set(result.replicates) == {0.0, 0.5, 1.0}
    json.dumps(result.to_dict(), allow_nan=False)


def test_equal_fit_estimator_and_bootstrap_match_known_binomial_fit_draws():
    f = frame(n_items=7)
    assert bs.g_primary(f, "treatment") == 0.5
    assert bs.delta(f, "treatment", bs.C_DIRECT) == 0.5
    result = bs.hierarchical_bootstrap(f, n_resamples=200, seed=23)
    rng = np.random.default_rng(23)
    expected = []
    # Bootstrap model order is comparator, treatment; each fit is equally likely.
    for _ in range(200):
        fits = rng.integers(0, 2, size=(2, 2))
        rng.integers(0, 7, size=7)
        expected.append(float(np.mean(fits[1] == 0)))
    np.testing.assert_array_equal(result.replicates, expected)


def test_fit_draws_are_independent_even_when_names_match_between_models():
    shared = bs.hierarchical_bootstrap(frame(unique_ids=False, both_models=True), n_resamples=200, seed=9)
    unique = bs.hierarchical_bootstrap(frame(both_models=True), n_resamples=200, seed=9)
    np.testing.assert_array_equal(shared.replicates, unique.replicates)
    assert shared.estimate == 0
    assert (shared.replicates > 0).any() and (shared.replicates < 0).any()


def test_every_draw_preserves_both_models_two_fit_slots_and_the_shared_item_draw():
    f = frame(n_items=9)
    hierarchy = bs._build_hierarchy(f, bs.LEVELS)
    rng = np.random.default_rng(8)
    for _ in range(25):
        indices = bs._resample_from_hierarchy(hierarchy, rng)
        assert indices.shape == (2, 2, 9, 4)
        source_items = f.item_id[indices]
        for mi, role in enumerate(hierarchy["model_roles"]):
            assert np.all(f.model_role[indices[mi]] == role)
            for fi in range(2):
                assert len(set(f.lens_id[indices[mi, fi]].ravel())) == 1
                np.testing.assert_array_equal(source_items[mi, fi], source_items[0, 0])
                for ii in range(9):
                    pairs = set(zip(f.condition[indices[mi, fi, ii]], f.ablation_state[indices[mi, fi, ii]]))
                    assert pairs == {
                        (bs.C_DIRECT, bs.CLEAN), (bs.C_DIRECT, bs.ABLATED),
                        (bs.C_FROZEN, bs.CLEAN), (bs.C_FROZEN, bs.ABLATED),
                    }
        sampled = bs._resampled_frame(f, indices)
        bs.validate_primary_design(sampled)
        assert len(set(sampled.item_id)) == 9
        assert len(set(sampled.lens_id)) == 2


def test_shared_item_resampling_preserves_identical_model_contrasts_exactly():
    rows = records(n_items=9)
    for row in rows:
        row["correct"] = (
            int(row["item_id"].split("_")[-1]) % 2 == 0
            and row["condition"] == bs.C_DIRECT and row["ablation_state"] == bs.CLEAN
        )
    result = bs.hierarchical_bootstrap(bs.TrialFrame.from_records(rows), n_resamples=200)
    assert result.estimate == 0
    np.testing.assert_array_equal(result.replicates, np.zeros(200))


def test_custom_statistic_receives_valid_draw_slots_and_matches_builtin():
    f = frame()
    calls = []

    def statistic(sample):
        bs.validate_primary_design(sample)
        calls.append(len(sample))
        return bs.primary_endpoint(sample)

    custom = bs.hierarchical_bootstrap(f, statistic, n_resamples=15, seed=3)
    builtin = bs.hierarchical_bootstrap(f, n_resamples=15, seed=3)
    np.testing.assert_array_equal(custom.replicates, builtin.replicates)
    assert calls == [len(f)] * 16


def test_itt_includes_unparseable_attempts_in_each_fit():
    rows = records()
    for row in rows:
        if row["lens_id"] == "treatment_a" and row["item_id"] == "item_0":
            row["parsed"] = row["correct"] = False
    assert bs.primary_endpoint(bs.TrialFrame.from_records(rows)) == pytest.approx(0.4)


@pytest.mark.parametrize("key", list(records(n_items=1)[0]))
def test_no_defaults_for_missing_trial_fields(key):
    row = records(n_items=1)[0]
    del row[key]
    with pytest.raises(ValueError, match="missing required keys"):
        bs.TrialFrame.from_records([row])


@pytest.mark.parametrize("key", ["parsed", "correct"])
@pytest.mark.parametrize("value", ["False", "True", 0, 1, 0.0, None, np.nan, np.inf])
def test_boolean_columns_are_not_coerced(key, value):
    row = records(n_items=1)[0]
    row[key] = value
    with pytest.raises(ValueError, match="boolean"):
        bs.TrialFrame.from_records([row])


@pytest.mark.parametrize("value", [None, 1, "", "  ", np.nan, np.inf])
def test_identifiers_require_nonempty_strings(value):
    row = records(n_items=1)[0]
    row["lens_id"] = value
    with pytest.raises(ValueError, match="nonempty string"):
        bs.TrialFrame.from_records([row])


def test_duplicate_trial_keys_are_rejected_on_ingest_and_estimation():
    rows = records()
    with pytest.raises(bs.BootstrapViolation, match="duplicate"):
        bs.TrialFrame.from_records(rows + [rows[0]])
    f = frame()
    repeated = f.take(np.r_[np.arange(len(f)), 0])
    with pytest.raises(bs.BootstrapViolation, match="duplicate"):
        bs.primary_endpoint(repeated)


@pytest.mark.parametrize("remove", ["cell", "fit_item", "model_item", "fit", "model", "condition", "state"])
def test_missing_cells_and_unequal_fit_row_counts_are_rejected(remove):
    rows = records()

    def omit(row):
        if remove == "cell":
            return row == rows[0]
        if remove == "fit_item":
            return row["lens_id"] == "treatment_a" and row["item_id"] == "item_0"
        if remove == "model_item":
            return row["model_role"] == "comparator" and row["item_id"] == "item_0"
        return row[{"fit": "lens_id", "model": "model_role", "condition": "condition", "state": "ablation_state"}[remove]] == {
            "fit": "treatment_a", "model": "comparator", "condition": bs.C_FROZEN, "state": bs.ABLATED
        }[remove]

    incomplete = bs.TrialFrame.from_records([row for row in rows if not omit(row)])
    with pytest.raises(bs.BootstrapViolation, match="missing|two|empty"):
        bs.primary_endpoint(incomplete)
    with pytest.raises(PrimaryAssayError):
        assert_design_complete(incomplete)
    with pytest.raises(bs.BootstrapViolation):
        bs.hierarchical_bootstrap(incomplete, n_resamples=2)


def test_two_fit_ids_globally_does_not_mean_two_per_model():
    rows = [
        row for row in records(unique_ids=False)
        if row["lens_id"] == ("real_a" if row["model_role"] == "treatment" else "real_b")
    ]
    f = bs.TrialFrame.from_records(rows)
    assert len(set(f.lens_id)) == 2
    with pytest.raises(bs.BootstrapViolation, match="exactly two real fits per model"):
        bs.primary_endpoint(f)


def test_extra_fit_cannot_change_the_equal_fit_estimand():
    rows = records()
    extra = [dict(row, lens_id="treatment_c") for row in rows if row["lens_id"] == "treatment_a"]
    with pytest.raises(bs.BootstrapViolation, match="more than two"):
        bs.primary_endpoint(bs.TrialFrame.from_records(rows + extra))


def test_multi_template_shape_fails_clearly():
    rows = records()
    extra = [dict(rows[0], template_id="another_template")]
    with pytest.raises(bs.BootstrapViolation, match="unsupported multi-template shape"):
        bs.hierarchical_bootstrap(bs.TrialFrame.from_records(rows + extra), n_resamples=2)


def test_mutated_nonfinite_or_multidimensional_columns_cannot_reach_statistics():
    f = frame()
    f.correct = np.full(len(f), np.nan)
    with pytest.raises(ValueError, match="boolean"):
        bs.primary_endpoint(f)
    f = frame()
    f.parsed = f.parsed[:, None]
    with pytest.raises(ValueError, match="one-dimensional"):
        bs.primary_endpoint(f)


@pytest.mark.parametrize("levels", [(), ("lens_id",), ("lens_id", "item_id"),
                                   ("item_id", "lens_id", "template_id"),
                                   ("lens_id", "item_id", "item_id"), (*bs.LEVELS, "target")])
def test_unsupported_hierarchies_are_rejected_before_statistics(levels):
    with pytest.raises(bs.BootstrapViolation, match="unsupported levels"):
        bs.hierarchical_bootstrap(None, lambda _: pytest.fail("statistic called"), levels=levels)


@pytest.mark.parametrize("n", [0, 1, -1, 2.5, True, np.nan, "2"])
def test_invalid_n_resamples_before_statistics(n):
    with pytest.raises(ValueError, match="n_resamples"):
        bs.hierarchical_bootstrap(None, lambda _: pytest.fail("statistic called"), n_resamples=n)


@pytest.mark.parametrize("seed", [-1, 0.5, True, np.nan, None])
def test_invalid_seed_before_statistics(seed):
    with pytest.raises(ValueError, match="seed"):
        bs.hierarchical_bootstrap(None, lambda _: pytest.fail("statistic called"), seed=seed)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_estimates_and_replicates_never_get_filtered(bad):
    with pytest.raises(bs.BootstrapViolation, match="estimate.*finite"):
        bs.hierarchical_bootstrap(frame(), lambda _: bad, n_resamples=2)
    values = iter([0.5, 0.5, bad])
    with pytest.raises(bs.BootstrapViolation, match="replicate 1.*finite"):
        bs.hierarchical_bootstrap(frame(), lambda _: next(values), n_resamples=2)
    with pytest.raises(bs.BootstrapViolation, match="non-finite bootstrap replicates"):
        bs.BootstrapResult(0, np.array([0, bad]), 2, bs.LEVELS, 0)


def test_modified_results_cannot_silently_drop_nonfinite_replicates():
    result = bs.BootstrapResult(0, np.zeros(3), 3, bs.LEVELS, 0)
    result.replicates[1] = np.nan
    with pytest.raises(bs.BootstrapViolation, match="no replicate may be dropped"):
        result.interval(0.95)


@pytest.mark.parametrize("level", [0, 1, -1, np.nan, np.inf, True])
def test_invalid_interval_levels(level):
    result = bs.BootstrapResult(0, np.zeros(3), 3, bs.LEVELS, 0)
    with pytest.raises((ValueError, bs.BootstrapViolation)):
        result.interval(level)


@pytest.mark.parametrize(
    "entry",
    [
        lambda: bs.delta(None, "treatment", bs.C_DIRECT),
        lambda: bs.g_primary(None, "treatment"),
        lambda: bs.g_gen(None, "treatment"),
        lambda: bs.primary_endpoint(None),
        lambda: bs.hierarchical_bootstrap(None, lambda _: pytest.fail("statistic called")),
        lambda: bs.trial_bootstrap(),
        lambda: analyze(None, delta=np.nan),
        lambda: pw.CellProbabilities(0.5, 0.5, 0.5, 0.5).g(),
        lambda: pw.simulate_power(None, None, -1),
        lambda: bs.BootstrapResult(0, np.zeros(2), 2, bs.LEVELS, 0).interval(0.95),
        lambda: bs.BootstrapResult(0, np.zeros(2), 2, bs.LEVELS, 0).to_dict(),
        lambda: decide(np.nan, None, None, np.nan),
    ],
)
def test_blinding_fires_at_the_entry_point_before_any_data_or_statistic(entry):
    with phase_a_blinding():
        with pytest.raises(BlindingViolation):
            entry()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_decision_inputs_are_errors_not_inconclusive(value):
    for kwargs in ({"low": value}, {"high": value}, {"level": value}):
        with pytest.raises(ValueError, match="finite"):
            Interval(**({"low": -0.1, "high": 0.1, "level": 0.95} | kwargs))
    with pytest.raises(ValueError, match="finite"):
        decide(value, Interval(-0.2, 0.2, 0.95), Interval(-0.1, 0.1, 0.90), 0.1)
    with pytest.raises(ValueError, match="finite"):
        decide(0, Interval(-0.2, 0.2, 0.95), Interval(-0.1, 0.1, 0.90), value)
    with pytest.raises(ValueError, match="finite"):
        holm({"a": value})
