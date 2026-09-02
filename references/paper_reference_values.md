# Paper reference values (Appendix B)

**These are conformance and effect-size calibration references. They are not targets.**

Matching them is not a success criterion for this project, and missing them is not a
failure of this project. They exist so that an implementation that is off by an order of
magnitude is visible as an *engineering* problem before it is mistaken for a *scientific*
result. No decision rule in §9 refers to any number on this page.

The paper's subjects are Claude-family models; this project's subjects are two open
14B checkpoints. Quantitative agreement is not expected and its absence is not evidence.

## Workspace structure

| Quantity | Value |
|---|---|
| Workspace band | ≈ L38–L92 on a 0–100 rescaled depth |
| Occupancy plateau | ≈ 25 |
| Excess explained variance | < 10% |
| Concept-vector J-space component, share of variance | median 6–7% |
| Probe J-space component | ≈ 10–15% |

## Swap results

| Quantity | Value |
|---|---|
| Verbal-report swap, top-5 | J-space component 59% · pure J-lens 88% · non-J 5% |
| Probe-swap (n = 90) | J-space component 61% · raw swap 60% · non-J 28% · non-J with clamp 6% |
| Multi-hop swap, top-1 | Haiku 4.5 54% · Sonnet 4.5 70% · Opus 4.5 70% |
| Intermediate vs answer swap depth | intermediate takes effect ≈ 17% of depth earlier |

## Editing and capacity

| Quantity | Value |
|---|---|
| Flexible generalization | 76/192 at `α = 1`; 101/192 at `α = 2` |
| Capacity, unrelated list | ≈ 6 words |
| Capacity, single layer | 1–2 words |

## Broadcast-head ablation

| Quantity | Value |
|---|---|
| recall@25 | 0.67 (ablated) vs 0.86 (clean) |
| top-1 changed | 5% vs 2% |
| Injected thought | 0.54 → 0.09 |

## Ablation battery

| Quantity | Value |
|---|---|
| `k` | 10 |
| Strength tiers | three, defined by layer band |
| Multi-hop | near ceiling → near zero |
| GSM8K | CoT markedly more robust than direct |

---

## Where each value is used in this repository

| Value | Use |
|---|---|
| `k = 10`, three layer-band tiers | starting point for the Phase-A search space (`configs/phase_a/search_space.json`); the operative value is fixed at Freeze-2 |
| Probe-swap 61 / 60 / 28 / 6 | printed alongside `wda.assays.probe_swap` output as `paper_reference`; never compared against by a rule |
| Flexible generalization 76/192, 101/192 | printed alongside `wda.assays.flexible_generalization` output |
| Intermediate-vs-answer depth ≈ 17% | reference in `wda.intervene.patch.swap_depth_separation`; the *sign* of the separation is what matters, not the magnitude |
| Band ≈ L38–L92 rescaled | sanity reference for the automatic band rule; the rule never consults it |
| GSM8K CoT more robust than direct | the phenomenon `G_gen` is comparable to — which is exactly why `G_gen` is secondary and `G_primary` (with its frozen rationale and matched window) is the endpoint |

## The one thing this page must not become

A reason to re-run. If a measured value here disagrees with the paper, that is a reportable
observation about two open checkpoints and a self-built instrument. It is not grounds to
change `k`, the band, `δ`, the seed, the sample size or the endpoint (§11, SC3).
