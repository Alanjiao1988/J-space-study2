# Execution boundaries

The September 2 `FREEZE-1.json` is a preserved **candidate**, schema `wda/seal/1`.
It cannot authorize execution of the repaired code. No active schema-2 seal has
been issued. Do not delete and recreate the old candidate to bypass verification.

## Seals and artifacts

Schema `wda/seal/2` hashes canonical manifest metadata, including `extra`, as well
as every covered file's path, size and SHA-256. `seal.MANDATORY` lists the required
surface. Missing files are errors, not empty globs:

- Protocol, scientific code/tests/tools/configs, reference corrections.
- `requirements.lock.json`: `wda/dependencies.lock/1`, exact `python`, and a
  `packages` mapping of exact versions. Host compatibility needs a separate check.
- Calibration revision in `configs/freeze1/calibration_revision.lock.json`;
  later subject pins go in `configs/model_revisions.lock.json`.
- Verified upstream pin, manifest and local vendored bytes.

An `ArtifactRef` contains exactly `path`, `sha256`, `size`. The path is relative
to the repository and uses portable POSIX spelling in JSON. Escapes, symlinks,
junctions and non-finite JSON are rejected. Hashes prove content identity, not
that a model was executed. The model-facing runner is the provenance boundary.

## Runs

`Run.start` requires a complete stage seal, a locked model/revision and a
`config_artifact` referencing an identical sealed configuration. It never
accepts a Phase-C/Phase-D override to Freeze-1.
Runtime links to completed `phase0`/`phase_a` receipts are added at run creation
and covered by the immutable run-config hash, not predicted before the earlier
run occurs. Phase-B calibrated `parameters` must match completed Phase A;
they are not silently chosen by a caller after sealing.

- Phase 0 can use only the registered calibration checkpoint.
- Phase A requires active process blinding and a completed model-facing Phase-0
  receipt covering all five fixtures. Unit tests, skipped checks and engineering
  canaries cannot supply that receipt.
- Phase B requires completed Phase A and the identical calibrated parameters.
- Phase C/D remain explicitly blocked by the unresolved scientific direction
  provider. A statistics implementation alone is not an experiment runner.

`wda/results/2` links the immutable `wda/run/2` config, identity, expected phase
and measured evidence. Validation functions in `evidence.py` check the fixture
metrics, revision bindings, no-op bytes, prerequisites and lens artifacts.
Phase-A evidence uses `wda/phase-a-evidence/1`, `parameters`, `phase0` and
ArtifactRefs for structural statistics, positive control, general damage,
paired variance and parseability.

Results and incidents are create-only. Trial keys are unique. A true
`noop_bitexact` boolean is insufficient: the trial must bind distinct raw
little-endian float32 clean/no-op captures with equal finite bytes. An unresolved
runtime incident prevents continued scientific logging or qualification.
Registered infrastructure retries consume one durable incident reference and
must retain the identical config/seal identity. This does not implement
automatic process recovery or justify micro-batch/numerical changes.

## Still unqualified

- Token-J scientific direction selection versus synthetic SVD components.
- Community lens file schema and estimator/normalization compatibility.
- Actual upstream probe-swap positive control and paired controls.
- Native-template answer-cue boundary and realized exposure comparability.
- Model-facing run assembly, dependency lock and runtime enforcement on a GPU.

Synthetic tests exercise failures and byte/shape invariants. Passing them does
not change the project phase, approve thresholds, select a confirmation seed,
or certify these instruments.
