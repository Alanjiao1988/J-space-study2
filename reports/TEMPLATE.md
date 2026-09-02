# Phase report template (§17)

> Copy to `reports/<phase>_report.md`. Every numbered section is required; a section with
> nothing to report says "none", never nothing at all.

---

# `<phase>` report — Fixed-Checkpoint Workspace-Dependence Audit

## 1. Phase and seal

| | |
|---|---|
| Phase | `<phase0 / phaseA / phaseB / phaseC / phaseD>` |
| Seal stage | `<freeze1 / freeze2>` |
| `root_sha256` | `<from FREEZE-N.json>` |
| Sealed at | `<timestamp>` |
| Seal verified at run start | yes / no |
| Seal re-verified at run end | yes / no |
| Run ids | `<runs/<phase>/<run_id> …>` |

## 2. Subjects and revisions

| Role | `repo_id` | Locked revision | Phases allowed |
|---|---|---|---|
| treatment | | | |
| comparator | | | |
| parent anchor (descriptive only) | | | |
| calibration (Phase 0/A only) | | | |

State explicitly whether Phase-A blinding was active and whether any blinding violation was
recorded (`blinding_receipt.violations`).

## 3. Endpoints — point estimate, CI, decision

| Endpoint | Tier | Estimate | 95% CI | 90% CI | δ | Outcome |
|---|---|---|---|---|---|---|
| `D` | primary | | | | 0.10 | material / equivalent / inconclusive |
| … | secondary | | | | | (interpreted only if the primary is material) |
| … | exploratory | | | | | reported, not inferred from |

The permitted sentence, verbatim from `Decision.sentence()`:

> For `<endpoint>`, the difference is `D = …` (95% CI […, …]; 90% CI […, …]; delta = 0.10),
> judged `<outcome>`.

Run `wda.stats.decision.assert_permitted_phrasing` over this whole report before publishing.

## 4. Triple endpoint table (§9.4)

Per model × condition × ablation state. **Never collapse the three columns.**

| Model | Condition | State | n | parseability | conditional acc. *(conditional)* | ITT acc. |
|---|---|---|---|---|---|---|
| | | clean | | | | |
| | | ablated | | | | |

Also give the verdict breakdown (`CORRECT`, `INCORRECT_WRONG_VALUE`,
`UNPARSEABLE_REASONING_SPAN`, `UNPARSEABLE_NO_INTEGER`,
`UNPARSEABLE_MULTIPLE_INTEGERS`, `UNPARSEABLE_EMPTY`) per cell.

## 5. All controls (§7)

Every control below is mandatory; `assert_controls_complete` refuses a report missing any.

| Control | Result | Verdict |
|---|---|---|
| matched-norm perturbation | | |
| layer-matched random ablation | | |
| complementary-component clamp | | |
| answer vs intermediate swap depth | | |
| position control | | |
| no-instruction baseline | | |
| shuffled-position null | | |
| control words | | |
| skip clean top-10 | | |
| shuffled-corpus lens | | |
| direct final-token substitution baseline | | |
| logit-lens floor | | |
| **general damage (output KL, J-space vs random)** | mean KL … vs …; separability … | tier usable / **tier VOID** |

Report the general-damage KL for **every strength tier used**. A tier whose KL is not
separable from the layer-matched random ablation is void and its results must be withdrawn.

## 6. Feasibility results (§9.4)

| Condition | Model | parseability | threshold | Disposition |
|---|---|---|---|---|

"None" if every condition cleared the threshold. Never reduce a denominator to remove one.

## 7. Infrastructure re-run log (§11)

| `run_id` | Code | Attempt | Budget | Reason | Recovery |
|---|---|---|---|---|---|

Scientific re-runs: **0**, always. If a re-run was refused, record the refusal here.

## 8. Stopping conditions triggered (§12)

| Code | Triggered | Consequence |
|---|---|---|
| SC0 | | Phase-0 fixtures could not pass before the freeze → self-built implementation not credible; stop |
| SC1 | | a subject failed Phase-B qualification → that checkpoint is out; if it is the treatment, RQ1 closes permanently |
| SC2 | | Phase C judged equivalent → report the negative; RQ2 does not start |
| SC3 | | Phase C judged inconclusive → report and stop; no extra samples, no endpoint change, no δ change |
| SC4 | | a fixture failed after Freeze-2 → the assay is void; stop; do not patch |
| SC5 | | a new finding appeared after the freeze → accept only a BLOCKING that names which inference it invalidates |

## 9. Claim check

Confirm in one line each:

- [ ] No `reasoning-specific`, training-type, model-class or scale-trend language.
- [ ] No claim that distillation **caused** anything.
- [ ] No claim that externalization is confirmed — it remains one candidate explanation of `G_primary`.
- [ ] No claim that J-space exists or does not exist.
- [ ] No statement of any kind about the 1.5B checkpoint.
- [ ] No "first reproduction in an open model".
- [ ] No treatment of a predecessor instrument failure as evidence about J-space.
- [ ] `assert_permitted_phrasing` run over the full report text: clean.
