# `runs/` — execution records

One directory per phase; one directory per run inside it. Created by
`wda.runtime.run.Run.start`, which verifies the seal before the directory exists.

```
runs/<phase>/<run_id>/
├── config.json    frozen configuration + registry, assay, re-run and blinding snapshots
├── seal.json      the verified seal header (the seal always precedes the first call)
├── log.jsonl      one line per trial, §15.4
└── results.json   endpoints, controls, decision, stopping condition
```

`run_id` is `<phase>-<label>-<UTC timestamp>` and is **create-only**: re-running the same
`run_id` requires an authorised infrastructure failure code (§11).

## `log.jsonl` line (§15.4)

```json
{
  "run_id": "...", "phase": "phaseC", "model_role": "treatment",
  "condition": "C_frozen", "ablation_state": "ablated",
  "lens_id": "real_a", "item_id": "confirm-3s-...", "template_id": "arith_chain_v1",
  "target": null, "output_tokens": [16, 17], "parsed": true, "correct": false,
  "verdict": "INCORRECT_WRONG_VALUE",
  "kl_vs_clean": 0.0412, "noop_bitexact": true, "seal_hash": "..."
}
```

`noop_bitexact` is produced by `wda.intervene.precision.check_noop` and is never inferred
(fact F7). `seal_hash` is stamped by the runner, never accepted from the caller.

## Phase directories

| Directory | Contents | Evidential weight |
|---|---|---|
| `phase0/` | conformance fixtures | none — no scientific inference |
| `phaseA/` | blinded calibration on `Qwen2.5-7B-Instruct` | **zero**, by construction |
| `phaseB/` | instrument qualification on the subjects, held-out | gating only |
| `phaseC/` | the confirmation round | the primary result |
| `phaseD/` | stratified replication, only if Phase C is material | reported per tier, never pooled |

Runs are committed deliberately by the operator. Scratch space goes in
`runs/**/scratch/`, which is git-ignored.
