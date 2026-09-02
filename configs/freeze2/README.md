# `configs/freeze2/` — written after Phase B, before Phase C

This directory is **empty until Phase B completes.** Freeze-2 is sealed after instrument
qualification and before the first confirmation-round model call (§10, §16).

## What must land here

| File | Contents | Source |
|---|---|---|
| `parameters.json` | band, `k`, strength tier, fit scale (`n_prompts`, `skip_first`, `n_probes`) | Phase A, on the calibration model only |
| `power.json` | the measured paired covariance, `unit_variance_D`, and the operative `n` | `wda.stats.power.plan` |
| `lens_digests.json` | `lens_digest` for every fitted lens (2 real, 1 shuffled, 1 logit) | `LensBundle.digest()` |
| `item_bank.json` | confirmation-round seed, generator version, and `bank_digest` | `wda.items.generator.bank_manifest` |
| `qualification.json` | the five Phase-B criteria with pass/fail per subject | Phase B report |

`configs/model_revisions.lock.json` (the resolved subject revisions) is also covered by the
Freeze-2 manifest.

## Rules

- Nothing here may be written before Phase B has produced a qualification verdict for
  **both** subjects. If either fails, SC1 fires and there is no Phase C to configure.
- The confirmation-round item seed is used for the first time **after** this seal. The
  exploration and confirmation banks are disjoint (§8.4).
- After the seal, none of these values may change. A parameter change is a scientific
  re-run and is forbidden (§11); a fixture failure after Freeze-2 voids the assay (SC4).

Regenerate and verify with:

```bash
python -m wda.governance.seal write  --stage freeze2
python -m wda.governance.seal verify --stage freeze2
```
