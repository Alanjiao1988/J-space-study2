# Literature verification (§2 of the protocol)

Frozen at Freeze-1. Reproduced here so that a claim can be checked without re-reading the
protocol, and so that the two "cannot confirm" rows stay visible — they are the reason the
ablation, patching and sparse-decomposition modules are self-built engineering rather than
a thin wrapper over upstream.

| Claim | Verdict | Evidence |
|---|---|---|
| The paper's subjects are the Claude family | Confirmed | Sonnet 4.5 by default; Haiku 4.5 / Opus 4.5 corroborating; Opus 4.6 in places |
| Late-layer `J_ℓ → I` is expected behaviour | Confirmed | The paper: the logit lens *is* `J_ℓ = I`, "reasonable in later layers because of the residual connection"; at the motor transition "`J_ℓ` approaches the identity" |
| The logit lens captures most workspace structure | Confirmed | The paper: "quite useful in practice… captures much of the workspace-like structure… with somewhat lower reliability (particularly in earlier layers)" |
| GSM8K CoT is more robust to ablation than direct | Confirmed | The paper's ablation battery; explained as "externalised onto paper" |
| The capacity core contrast is related vs unrelated | Confirmed | Fig. 31B; the four-block variant is a sub-variant (Fig. 31E/F) |
| The ignition contrast is the J-restricted component vs full activation | Confirmed | Fig. 29B/C |
| Verbal report has no strength sweep | Confirmed | The sweep belongs to verbal *introspection* (Fig. 7) |
| **Upstream code contains patch / ablate interfaces** | **Cannot confirm** | The public README shows only `jlens.fit`, `JacobianLens.from_pretrained`, `lens.apply`. Ablation, patching and gradient-pursuit decomposition must be **implemented here**, with a conformance fixture. |
| **Upstream `data/` contains GSM8K or a direct/CoT runner** | **No** | `data/experiments/` is a prompt set. No GSM8K, no scoring script. |
| The DeepSeek distillation ladder's parents | Confirmed (official model table) | 1.5B ← Qwen2.5-Math-1.5B · 7B ← Qwen2.5-Math-7B · **14B ← Qwen2.5-14B** · 32B ← Qwen2.5-32B; all fine-tuned on 800k R1 samples |
| Neel Nanda's external review is mixed | Confirmed | rhyme-planning and mental-arithmetic did **not** reproduce; probe-swap weak; false positives common; **multi-fact editing (flexible generalization) reproduced cleanly** |
| `tao-hpu/jspace-replication` | Confirmed | Adds a direct final-token substitution baseline, amplitude-matched random controls, and a 1.7B–14B ladder |
| `amaljithkuttamath/jlens-replication` | **Confirmed (the v1 record of this row was wrong)** | Qwen2.5-3B-Instruct, `n = 25` fit: the **shuffled-corpus lens scored ≥ the real lens on all six evaluations** (typo MRR 0.806 vs 0.318); concept-direction ablation had no systematic effect; on Gemma-4-E2B the shuffled lens failed likewise but ablation had a strong negative effect. The author self-reports severe underfitting (`n = 25` vs 1000; `skip_first` 4 vs 16). |
| arXiv 2608.25347 | Confirmed | Jacobian energy decays with depth, is highly sparse, and concentrates on diagonal pathways and key positions |

## How each row constrains this repository

| Row | Constraint |
|---|---|
| Upstream has no ablation/patching interface | `wda.intervene.*` is self-built and gated by the five Phase-0 fixtures (`configs/freeze1/fixtures.json`). SC0 fires if they cannot pass before the freeze. |
| Upstream has no GSM8K / runner | The primary item bank is generated (`wda.items.generator`); GSM8K is a secondary anchor loaded from a local file whose digest enters the seal. |
| Late-layer `J → I` is expected | `wda.lens.band.identity_energy` is reported, and late-layer identity is *not* treated as a pathology. This is the correction that F6′ makes to the predecessor's reading. |
| Logit lens captures most structure | The logit lens is a mandatory floor; Phase-B criterion 3 only requires the real lens to be no worse on the **first half** of the band, not superior everywhere. |
| `amaljithkuttamath` shuffled-lens failure | Fit-scale floors `n_prompts ≥ 200`, `skip_first ≥ 16` (`wda.lens.fit`), and a shuffled-corpus lens that the real lens must beat (Phase-B criterion 2) or the subject is out. |
| Nanda review | `rhyme_planning` and `mental_arithmetic` are `EXCLUDED` in `wda.assays.registry`; `probe_swap` is `EXPLORATORY` and requires the direct-substitution baseline; `flexible_generalization` is `SECONDARY_CONFIRMATORY`. |
| tao-hpu | `direct_substitution_baseline` is a mandatory control (§7) and is enforced by `wda.intervene.patch.require_direct_substitution_baseline`. |
| DeepSeek parent table | The 14B's parent is `Qwen2.5-14B` (general), **not** a Math checkpoint — which is why the 14B pair is the primary and the Math-parent 7B tier is a separate, never-pooled stratum. |
| arXiv 2608.25347 | Sparsity and depth-decay of Jacobian energy motivate the top-`k` direction selection and the participation-ratio effective-dimension statistic. |

## External sources (Appendix D)

- Paper — <https://transformer-circuits.pub/2026/workspace/index.html>
- Upstream code — <https://github.com/anthropics/jacobian-lens> (Apache-2.0; single-commit reference implementation, marked not maintained)
- Anthropic research page — <https://www.anthropic.com/research/global-workspace>
- Neel Nanda review — LessWrong `zFJ3ZdQwrTWE9jT5S`
- <https://github.com/tao-hpu/jspace-replication>
- <https://github.com/amaljithkuttamath/jlens-replication> (`RESULTS.md`)
- arXiv 2608.25347
- DeepSeek-R1 official model table — <https://github.com/deepseek-ai/DeepSeek-R1>
- Community lens for calibration — HF `Kameshr/jspace-lens-Qwen7B`
