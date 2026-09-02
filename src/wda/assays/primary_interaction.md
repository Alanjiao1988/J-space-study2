# Estimand spec — `primary_interaction`

**Tier:** PRIMARY — the sole gate-bearing assay (§8.5).
**Phase:** C (confirmation). Phase A must never touch it (§16).

## Estimand

| Symbol | Definition |
|---|---|
| `acc(m, c, s)` | ITT accuracy for model `m`, condition `c`, ablation state `s ∈ {clean, ablated}` (§9.4) |
| `Δ(m, c)` | `acc(m, c, clean) − acc(m, c, ablated)` |
| `G_primary(m)` | `Δ(m, C_direct) − Δ(m, C_frozen)` |
| **`D`** | `G_primary(treatment) − G_primary(comparator)` |

`treatment = DeepSeek-R1-Distill-Qwen-14B`, `comparator = Qwen2.5-14B-Instruct`.

## Design

Eight cells; every item appears in all four cells of each model, so the contrast is paired.

|  | `C_direct` clean | `C_direct` ablated | `C_frozen` clean | `C_frozen` ablated |
|---|---|---|---|---|
| treatment | ✓ | ✓ | ✓ | ✓ |
| comparator | ✓ | ✓ | ✓ | ✓ |

`assert_design_complete` refuses an incomplete design: a missing cell may not be imputed or
dropped.

## What makes the contrast interpretable

`C_direct` and `C_frozen` share `L_p` and `max_new_tokens`, so window W has **identical
token length** in both — the same number of ablation exposures. The only difference is
whether a correct, generator-supplied rationale is present in the context.
`wda.conditions.envelopes.assert_window_parity` enforces this; without it, `G_primary`
would confound the rationale with exposure count.

## Accuracy definition

ITT composite only (§9.4): an unparseable output is incorrect and stays in the denominator.
`parseability` and `conditional accuracy` are reported alongside but never substituted in.
Fact F4 (240/240 unparseable under the predecessor's raw-direct route) is the reason.

## Uncertainty

Hierarchical bootstrap, levels `lens fit → item → template` (§9.5). A flat trial-level
bootstrap is refused by `wda.stats.bootstrap.trial_bootstrap`.

## Decision (§9.2)

| Outcome | Rule | Stopping condition |
|---|---|---|
| Material difference | 95% CI of `D` entirely beyond `±δ` | — (Phase D may start) |
| Equivalent | 90% CI of `D` inside `[−δ, +δ]` | SC2 |
| Inconclusive | otherwise | SC3 |

`δ = 0.10`, fixed *a priori* at Freeze-1 (§9.1). SC3 does **not** permit adding samples,
changing the endpoint, or changing `δ`.

## Mandatory controls

`layer_matched_random`, `matched_norm_perturbation`, `clamp_complement`, `general_damage`,
`skip_clean_top10`, `logit_lens_floor`, `shuffled_corpus_lens` — enforced by
`assert_controls_complete` (§7).

## Permitted claim

> "Between checkpoints X and Y, the difference in G is `D = …` (95% CI …; 90% CI …),
> judged material / equivalent / inconclusive."

Nothing about training type, model class, scale, causation, or the existence of J-space
(§4, §13).
