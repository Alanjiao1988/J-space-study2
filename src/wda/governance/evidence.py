"""Typed model-facing receipts used by stage gates, never inferred from pytest.

The trust boundary is the sealed runner: JSON and local hashes cannot attest
execution independently of that runner. A commit-looking string, a PASS label,
unit tests, and an engineering run are not sufficient. See CONTRACTS.md.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, TypedDict

from wda.governance.artifacts import (
    ArtifactRef, EvidenceError, check_ref, collect_refs, digest, hex_digest,
    identifier, local_path, read_json,
)
from wda.phases import Phase

CALIBRATION = "calib_qwen25_7b_instruct"
PRIMARY = ("r1_distill_qwen_14b", "qwen25_14b_instruct")
FIXTURES = frozenset({
    "noop_returns_zero", "projection_removed", "random_control_norm_matched",
    "lens_readout_matches_reference", "known_intermediate_positive",
})
CRITERIA = frozenset({
    "readout_reliability", "corpus_necessity", "vs_logit_lens",
    "causal_positive_control", "general_damage",
})


class Execution(TypedDict):
    kind: str
    model_key: str
    repo_id: str
    revision: str
    runner: ArtifactRef
    model_manifest: ArtifactRef


class NoopEvidence(TypedDict):
    schema: str
    run_id: str
    seal_root_sha256: str
    model_key: str
    revision: str
    dtype: str
    shape: list[int]
    clean: ArtifactRef
    noop: ArtifactRef


class CheckEvidence(TypedDict):
    status: str
    artifact: ArtifactRef


class Phase0Evidence(TypedDict):
    schema: str
    execution: Execution
    fixtures: dict[str, CheckEvidence]
    noop: NoopEvidence


class PhaseBEvidence(TypedDict):
    schema: str
    execution: Execution
    parameters_sha256: str
    lenses: dict[str, dict]
    criteria: dict[str, CheckEvidence]
    noop: NoopEvidence


def number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if type(value) not in (float, int) or not math.isfinite(value):
        raise EvidenceError(f"{name} must be finite numeric data, not a boolean")
    if minimum is not None and value < minimum:
        raise EvidenceError(f"{name} must be >= {minimum}")
    return value


def _object(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise EvidenceError(f"{name} requires an object")
    return value


def model_binding(root: Path, config: dict, phase: Phase) -> dict:
    from wda.models import registry

    key = config.get("model_key")
    if not isinstance(key, str):
        raise EvidenceError("science config requires model_key")
    entry = registry.resolve(key, phase, require_revision=False)
    locked = registry.load_revision_lock(root=root).get(entry.key)
    if not locked or config.get("revision") != locked["revision"]:
        raise EvidenceError("science config must match the exact locked model revision")
    if phase in {Phase.PHASE_0, Phase.PHASE_A} and entry.key != CALIBRATION:
        raise EvidenceError("Phase 0/A can only use the calibration model")
    return {"model_key": entry.key, "repo_id": entry.repo_id, "revision": locked["revision"]}


def check_execution(root: Path, execution: Any, config: dict, sealed_files: set[str]) -> dict:
    execution = _object(execution, "model execution")
    if execution.get("kind") != "model":
        raise EvidenceError("only actual model-facing execution can qualify; not synthetic/skipped")
    binding = model_binding(root, config["config"], Phase(config["phase"]))
    if any(execution.get(k) != v for k, v in binding.items()):
        raise EvidenceError("execution model/revision differs from immutable run configuration")
    runner = check_ref(root, execution.get("runner"))
    if execution["runner"]["path"] not in sealed_files or runner.suffix != ".py":
        raise EvidenceError("execution runner must be source covered by the run's seal")
    manifest = read_json(check_ref(root, execution.get("model_manifest")))
    if manifest.get("schema") != "wda/loaded-model/1" or manifest.get("kind") != "model":
        raise EvidenceError("model execution requires a loaded-model artifact manifest")
    if any(manifest.get(k) != v for k, v in binding.items()):
        raise EvidenceError("loaded-model manifest has a different model/revision")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise EvidenceError("loaded-model manifest cannot qualify from arbitrary revision strings")
    paths = [check_ref(root, ref) for ref in files]
    if len(set(paths)) != len(paths) or not any(p.name == "config.json" for p in paths):
        raise EvidenceError("loaded-model files require unique files and model config.json")
    if not any(p.suffix in {".safetensors", ".bin"} for p in paths):
        raise EvidenceError("loaded-model manifest must bind local checkpoint bytes")
    return binding


def check_noop(root: Path, receipt: Any, config: dict) -> None:
    """Compare actual raw, contiguous little-endian float32 bytes (not allclose).

    The two files must be distinct captures from the clean/no-op executions,
    nonempty and exactly ``4 * product(shape)`` bytes. The sealed runner is
    responsible for capturing real logits rather than fabricating either file.
    """
    receipt = _object(receipt, "no-op receipt")
    if receipt.get("schema") != "wda/noop/1" or receipt.get("dtype") != "float32":
        raise EvidenceError("no-op requires wda/noop/1 raw float32 evidence")
    for key in ("run_id", "seal_root_sha256"):
        if receipt.get(key) != config[key]:
            raise EvidenceError(f"no-op evidence has the wrong {key}")
    for key in ("model_key", "revision"):
        if receipt.get(key) != config["config"].get(key):
            raise EvidenceError(f"no-op evidence has the wrong {key}")
    shape = receipt.get("shape")
    if not isinstance(shape, list) or not shape or any(type(n) is not int or n < 1 for n in shape):
        raise EvidenceError("no-op shape must be positive integer dimensions")
    clean = check_ref(root, receipt.get("clean"))
    noop = check_ref(root, receipt.get("noop"))
    if clean == noop or clean.stat().st_size != 4 * math.prod(shape):
        raise EvidenceError("no-op requires distinct float32 captures with the declared shape")
    if clean.read_bytes() != noop.read_bytes():
        raise EvidenceError("no-op is not bit-exact; assay is failed")
    import struct

    if any(not math.isfinite(v[0]) for v in struct.iter_unpack("<f", clean.read_bytes())):
        raise EvidenceError("non-finite no-op logits do not demonstrate a valid execution")


def _observation(root: Path, check: Any, name: str, binding: dict) -> dict:
    check = _object(check, name)
    if check.get("status") != "PASS":
        raise EvidenceError(f"{name} has no PASS receipt")
    observation = read_json(check_ref(root, check.get("artifact")))
    if observation.get("schema") != "wda/fixture-observation/1":
        raise EvidenceError(f"{name} requires a measured observation artifact")
    if observation.get("kind") != "model" or observation.get("passed") is not True:
        raise EvidenceError(f"{name} cannot use skipped, synthetic or failing observations")
    if observation.get("fixture_id") != name or any(
        observation.get(k) != binding[k] for k in ("model_key", "revision")
    ):
        raise EvidenceError(f"{name} observation model/revision/check mismatch")
    return _object(observation.get("metrics"), f"{name}.metrics")


def _positive(metrics: dict, minimum: int, margin: float) -> None:
    items = metrics.get("items")
    if not isinstance(items, list) or len(items) < minimum:
        raise EvidenceError(f"positive control requires at least {minimum} model-facing items")
    seen = set()
    for item in items:
        item = _object(item, "positive control item")
        key = identifier(item.get("item_id"))
        if key in seen:
            raise EvidenceError("duplicate positive-control item")
        seen.add(key)
        if number(item.get("target_logit_delta"), "target_logit_delta") <= margin or number(
            item.get("matched_norm_random_delta"), "matched_norm_random_delta"
        ) > margin:
            raise EvidenceError("registered positive-control direction failed")


def validate_phase0(root: Path, evidence: Any, config: dict, sealed_files: set[str]) -> set[str]:
    evidence = _object(evidence, "Phase-0 evidence")
    if evidence.get("schema") != "wda/phase0-evidence/1":
        raise EvidenceError("Phase 0 requires a completed model-facing five-fixture receipt")
    binding = check_execution(root, evidence.get("execution"), config, sealed_files)
    if binding["model_key"] != CALIBRATION:
        raise EvidenceError("Phase 0 cannot qualify a subject checkpoint")
    checks = _object(evidence.get("fixtures"), "fixtures")
    if set(checks) != FIXTURES:
        raise EvidenceError("Phase 0 must cover exactly all five registered fixtures")
    observed = {name: _observation(root, checks[name], name, binding) for name in FIXTURES}
    check_noop(root, evidence.get("noop"), config)
    rules = read_json(local_path(root, "configs/freeze1/fixtures.json"))["fixtures"]
    for name, metric in (
        ("projection_removed", "max_abs_projection"),
        ("random_control_norm_matched", "max_abs_norm_error"),
    ):
        if number(observed[name].get(metric), metric, minimum=0) > rules[name]["tolerance"]:
            raise EvidenceError(f"registered {name} tolerance failed")
    jaccard = number(observed["lens_readout_matches_reference"].get("jaccard_top25"), "jaccard")
    if not rules["lens_readout_matches_reference"]["threshold"] <= jaccard <= 1:
        raise EvidenceError("registered reference-lens readout threshold failed")
    rule = rules["known_intermediate_positive"]
    _positive(observed["known_intermediate_positive"], rule["min_items"], rule["min_margin"])
    return collect_refs(root, evidence)


def validate_lenses(root: Path, lenses: Any) -> None:
    lenses = _object(lenses, "lenses")
    expected = {"real_a": "real", "real_b": "real", "shuffled": "shuffled", "logit": "logit"}
    if set(lenses) != set(expected):
        raise EvidenceError("each subject requires real_a, real_b, shuffled and logit lenses")
    paths = set()
    for key, kind in expected.items():
        lens = _object(lenses[key], key)
        if lens.get("kind") != kind:
            raise EvidenceError(f"wrong lens kind for {key}")
        hex_digest(lens.get("lens_digest"))
        path = check_ref(root, lens.get("artifact"))
        if path in paths:
            raise EvidenceError("distinct lenses cannot reuse the same artifact")
        paths.add(path)


def validate_phase_b(root: Path, evidence: Any, config: dict, sealed_files: set[str]) -> set[str]:
    evidence = _object(evidence, "Phase-B evidence")
    if evidence.get("schema") != "wda/phase-b-evidence/1":
        raise EvidenceError("Phase B requires wda/phase-b-evidence/1")
    binding = check_execution(root, evidence.get("execution"), config, sealed_files)
    if binding["model_key"] not in PRIMARY:
        raise EvidenceError("primary qualification requires a named primary subject")
    if evidence.get("parameters_sha256") != digest(config["config"].get("parameters")):
        raise EvidenceError("Phase-B qualification must bind the immutable parameter configuration")
    validate_lenses(root, evidence.get("lenses"))
    checks = _object(evidence.get("criteria"), "criteria")
    if set(checks) != CRITERIA:
        raise EvidenceError("Phase B requires exactly the five registered criteria")
    rules = read_json(local_path(root, "configs/freeze1/instrument.json"))["phase_b_qualification"]
    observed = {name: _observation(root, checks[name], name, binding) for name in CRITERIA}
    # All criterion observations bind the exact four lens artifacts, not just role names.
    for name, check in checks.items():
        artifact = read_json(check_ref(root, check["artifact"]))
        if artifact.get("lenses_sha256") != digest(evidence["lenses"]):
            raise EvidenceError(f"{name} observation is not bound to the qualified lenses")
    reliability = observed["readout_reliability"]
    for key in ("real_a", "real_b"):
        recall = number(reliability.get(key), key)
        if not rules["readout_reliability"]["threshold"] <= recall <= 1:
            raise EvidenceError("readout reliability criterion failed")
    # Aggregation is declared by the frozen protocol: paired mean of BOTH real lenses.
    for name, margin_key in (("corpus_necessity", "shuffled"), ("vs_logit_lens", "logit")):
        values = observed[name]
        real = sum(number(values.get(key), key) for key in ("real_a", "real_b")) / 2
        control = number(values.get(margin_key), margin_key)
        if any(not 0 <= number(values.get(k), k) <= 1 for k in ("real_a", "real_b", margin_key)):
            raise EvidenceError("held-out recall values must be probabilities")
        if real - control < rules[name]["margin"]:
            raise EvidenceError(f"{name} registered margin failed")
        if name == "vs_logit_lens" and values.get("band_portion") != "first_half":
            raise EvidenceError("logit comparison must concern the first half of the band")
    _positive(
        observed["causal_positive_control"], rules["causal_positive_control"]["min_items"],
        read_json(local_path(root, "configs/freeze1/fixtures.json"))["fixtures"]
        ["known_intermediate_positive"]["min_margin"],
    )
    separation = number(observed["general_damage"].get("separability"), "separability")
    if not rules["general_damage"]["separability_threshold"] <= separation <= 1:
        raise EvidenceError("general-damage separability criterion failed")
    if observed["general_damage"].get("strength") != config["config"]["parameters"]["strength"]:
        raise EvidenceError("general-damage receipt has the wrong strength tier")
    check_noop(root, evidence.get("noop"), config)
    return collect_refs(root, evidence)


def validate_parameters(root: Path, parameters: Any) -> None:
    parameters = _object(parameters, "parameters")
    if parameters.get("schema") != "wda/freeze2/parameters/1":
        raise EvidenceError("parameters require wda/freeze2/parameters/1")
    space = read_json(local_path(root, "configs/phase_a/search_space.json"))["search_space"]
    for key in ("k", "strength", "n_prompts", "skip_first", "n_probes"):
        value = parameters.get(key)
        if type(value) is not type(space[key][0]) or value not in space[key]:
            raise EvidenceError(f"parameter {key} is outside the sealed Phase-A search space")
    band = parameters.get("band")
    minimum = read_json(local_path(root, "configs/freeze1/band_rule.json"))["min_length"]
    if not isinstance(band, list) or len(band) < minimum or any(
        type(layer) is not int or layer < 0 for layer in band
    ) or band != list(range(band[0], band[-1] + 1)):
        raise EvidenceError("band must be a contiguous registered-length integer band")


def validate_freeze2(root: Path) -> set[str]:
    """Validate the five files named in configs/freeze2/README.md plus run lineage."""
    from wda.runtime.run import completed_receipt
    from wda.models.registry import load_revision_lock

    prefix = "configs/freeze2/"
    parameters = read_json(local_path(root, prefix + "parameters.json"))
    validate_parameters(root, parameters)
    power = read_json(local_path(root, prefix + "power.json"))
    if type(power.get("n")) is not int or power["n"] < 1:
        raise EvidenceError("power requires a positive integer operative n")
    number(power.get("unit_variance_D"), "unit_variance_D", minimum=0)
    covariance = power.get("paired_covariance")
    if not isinstance(covariance, list) or not covariance:
        raise EvidenceError("power requires the measured paired_covariance")
    for row in covariance:
        if not isinstance(row, list) or len(row) != len(covariance):
            raise EvidenceError("paired_covariance must be a square numeric matrix")
        for value in row:
            number(value, "paired_covariance")
    bank = read_json(local_path(root, prefix + "item_bank.json"))
    if bank.get("schema") != "wda/item_bank/1" or bank.get("split") != "confirm":
        raise EvidenceError("item bank must be the confirmation bank manifest")
    if type(bank.get("seed")) is not int or not isinstance(bank.get("generator_version"), str):
        raise EvidenceError("item bank requires seed and generator_version")
    hex_digest(bank.get("bank_digest"))
    check_ref(root, bank.get("artifact"))
    if bank.get("n_items") != power["n"]:
        raise EvidenceError("item bank count must match the operative power n")
    lenses = read_json(local_path(root, prefix + "lens_digests.json"))
    if lenses.get("schema") != "wda/lens_digests/1" or set(lenses.get("subjects", {})) != set(PRIMARY):
        raise EvidenceError("lens digests require both named primary subjects")
    qualification = read_json(local_path(root, prefix + "qualification.json"))
    if qualification.get("schema") != "wda/qualification/1" or set(
        qualification.get("subjects", {})
    ) != set(PRIMARY):
        raise EvidenceError("qualification requires both primary Phase-B PASS receipts")
    phase_a = completed_receipt(root, qualification.get("phase_a"), Phase.PHASE_A)
    if phase_a["evidence"].get("parameters") != parameters:
        raise EvidenceError("Freeze-2 parameters differ from the completed blinded Phase A")
    locked = load_revision_lock(root=root)
    refs: set[str] = set()
    for key in PRIMARY:
        if key not in locked:
            raise EvidenceError(f"Freeze-2 requires the named subject revision: {key}")
        subject = lenses["subjects"][key]
        if subject.get("revision") != locked[key]["revision"]:
            raise EvidenceError("lens revision differs from locked subject revision")
        validate_lenses(root, subject.get("lenses"))
        result = completed_receipt(root, qualification["subjects"][key], Phase.PHASE_B)
        proof = result["evidence"]
        if proof["execution"]["model_key"] != key or proof["execution"]["revision"] != subject["revision"]:
            raise EvidenceError("Phase-B receipt does not bind this exact subject/revision")
        if proof["lenses"] != subject["lenses"] or proof["parameters_sha256"] != digest(parameters):
            raise EvidenceError("Phase-B receipt does not bind the exact lenses/parameters")
        refs |= collect_refs(root, proof)
    for value in (parameters, power, bank, lenses, qualification, phase_a["evidence"]):
        refs |= collect_refs(root, value)
    return refs
