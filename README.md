# Fixed-Checkpoint Workspace-Dependence Audit

![status](https://img.shields.io/badge/status-FREEZE--1%20CANDIDATE-blue)
![phase](https://img.shields.io/badge/phase-0%20(conformance)-lightgrey)
![subject model calls](https://img.shields.io/badge/subject%20model%20calls-0-green)
![claim ceiling](https://img.shields.io/badge/claim%20ceiling-two%20checkpoints%20only-orange)

> Protocol of record: [`PROTOCOL_v1.1.md`](PROTOCOL_v1.1.md) — **read-only after Freeze-1.**
> Predecessor: [`J-space-observation`](https://github.com/Alanjiao1988/J-space-observation) @ `d87c1b9e4e9dca062cebad7b6eee981d9dba8c25` — `DISCONTINUED — RESEARCH QUESTION UNANSWERED`.

> ### ⛔ Execution state: nothing has been run
>
> **0 model calls · 0 checkpoints downloaded · 0 lenses fitted · 0 GPU-hours · 0 cloud resources.**
> This repository currently contains design, implementation and tests only. See
> [`STATUS.json`](STATUS.json) for the machine-readable state and the outstanding Freeze-1
> gates. `src/` contains no `from_pretrained`, no network client and no cloud SDK; nothing
> here can start a run on import.

---

## 1. One-sentence definition (§1)

This project measures, on one *named pair of fixed checkpoints* (`DeepSeek-R1-Distill-Qwen-14B` and
`Qwen2.5-14B-Instruct`, sharing the parent `Qwen2.5-14B`):

> **whether the effect of J-space ablation on final-answer accuracy differs according to whether an
> external rationale is present in the context, and whether that interaction differs materially
> between the two checkpoints.**

The only legitimate name for the main result is the **J-space ablation × response-format
interaction**. "Externalization" is a *candidate explanation*, not a conclusion.

## 2. Research questions and claim ceiling (§4)

| ID | Question |
|---|---|
| **RQ0** | Does the self-built J-lens fit + ablation implementation pass the conformance fixtures and the held-out instrument-qualification criteria? |
| **RQ1** (primary) | On the named 14B pair, is the difference `D` in the ablation × response-format interaction `G` materially different? |
| **RQ2** (conditional) | On the 7B (Math-parent) and 32B (general-parent) tiers, does `D` point the same way? Reported per tier, never pooled. |

**Estimands** (§5, Appendix E):

```
Δ(c)       = acc_unablated(c) − acc_ablated(c)          # ITT accuracy (§9.4)
G_primary  = Δ(C_direct) − Δ(C_frozen)
D          = G_primary(R1-Distill-14B) − G_primary(Qwen2.5-14B-Instruct)   # sole primary endpoint
```

**Permitted phrasing** — "Between checkpoints X and Y, the difference in G is D = … (CI …), judged
*material* / *equivalent* / *inconclusive*."

**Forbidden phrasing** — `reasoning-specific`, "reasoning-distilled model class", `scale trend`,
"distillation caused", "externalization confirmed", "J-space exists / does not exist", or any
generalization from two checkpoints to a training type.

## 3. Decision rule (§9.2) — three outcomes, none of which may be re-adjudicated

| Outcome | Rule |
|---|---|
| **Material difference** | 95% CI of `D` lies entirely above `+δ` or entirely below `−δ` |
| **Equivalent** | 90% CI of `D` lies entirely inside `[−δ, +δ]` |
| **Inconclusive** | otherwise — a legitimate third result |

`δ = 0.10` is given *a priori* at Freeze-1 and may never be revised from observed variance.

## 4. What this project explicitly does not claim (§13)

- Any `reasoning-specific`, training-type-level, or scale-trend generalization.
- That distillation **caused** any difference (that needs the controlled training design in Appendix C).
- That externalization is confirmed — it is one candidate explanation of `G_primary`.
- That J-space exists / does not exist, or equates to reasoning, understanding, or consciousness.
- Anything at all about the 1.5B checkpoint (it is not a subject — fact F1).
- "First reproduction of J-space in an open model."
- That any *instrument-level* failure of the predecessor project is evidence about J-space.

## 5. Phases and gates (§10)

```
Freeze-1  design + analysis plan + fixtures       <- before ANY model call
Phase 0   conformance   (5 fixtures, no inference)
Phase A   blinded calibration  (Qwen2.5-7B-Instruct only; zero evidential weight)
Phase B   instrument qualification (subjects, held-out) -> any FAIL = subject is out
Freeze-2  parameters + item seed + lens hashes + subject revisions
Phase C   confirmation run -> §9.2 decision
Phase D   stratified replication (only if Phase C = material difference)
```

Stopping conditions `SC0`–`SC5` (§12) each map to a *reportable* outcome. There is no
"try again" branch. Scientific re-runs are forbidden (§11); only registered infrastructure
failure codes may re-run a `run_id`.

## 6. Repository layout

| Path | Contents |
|---|---|
| `PROTOCOL_v1.1.md` | Full frozen protocol (read-only after Freeze-1) |
| `FREEZE-1.json` / `FREEZE-2.json` | Hash manifests produced by `wda.governance.seal` |
| `references/` | Predecessor index & facts, literature verification, paper reference values |
| `third_party/jacobian-lens/` | Upstream pin (`PINNED_COMMIT.txt`) — **never modified** |
| `src/wda/` | Implementation (see §14 of the protocol) |
| `configs/freeze1,phase_a,freeze2/` | Frozen constants, calibration search space, Phase-B outputs |
| `tests/conformance,governance,stats/` | Phase-0 fixtures, governance tests, synthetic stats tests |
| `runs/phase{0,A,B,C,D}/` | `<run_id>/{config.json, seal.json, log.jsonl, results.json}` |
| `reports/` | One report per phase, per the §17 template |

## 7. Quickstart

```bash
python -m pip install -e ".[dev]"        # numpy / scipy / pytest; torch is optional at test time
python -m pytest tests -q                # Phase-0 fixtures + governance + stats tests
python -m wda.governance.seal write  --stage freeze1     # regenerate FREEZE-1.json
python -m wda.governance.seal verify --stage freeze1     # must pass before every run
python -m wda.items.generator --split explore --n 40 --out runs/phase0/items_explore.jsonl
```

Every run entry point verifies the seal first and refuses to start on mismatch (§16).

## 8. Before anything is run

The repository is deliberately inert. Six gates stand between this state and the first
model call, and they are listed in [`STATUS.json`](STATUS.json):

1. Operator signs off `δ = 0.10` (§9.1).
2. Operator accepts or amends the **F6′ and F13 errata** in
   [`references/predecessor_facts.md`](references/predecessor_facts.md). Correcting an
   inherited fact is permitted *before* the freeze; afterwards it becomes an SC5 matter.
3. Subject and calibration revisions resolved and locked —
   `python -m wda.models.registry lock --key <key> --revision <sha>` (§8.1). Until then
   `registry.resolve` raises `UnlockedRevisionError` and no checkpoint can be loaded.
4. Upstream pinned commit vendored — `python tools/vendor_upstream.py` (read-only, untracked).
5. `L_p` re-verified against each subject's real tokenizer by `assert_cue_length` (fact F3).
6. Phase-0 fixtures 4 and 5 run in their model-facing form — currently `needs_model` and skipped.

## 9. Inherited constraints that shape the code

| Fact | Consequence in this repo |
|---|---|
| F1 | 1.5B is on the registry **denylist**; loading it raises. |
| F2, F4, F11 | Only within-model differences are estimands; no absolute-threshold gates on accuracy. |
| F7 | Ablation runs are fp32-enforced; a bit-exact no-op check is logged on every trial. |
| F6′, F8 | Instrument qualification and a causal positive control precede any scientific inference. |
| F6′ | The band rule carries an **absolute floor and a minimum length** — the predecessor rule had neither and fired on a negative control. |
| F13 | Seal precedes the first model call; scientific re-runs are refused by `rerun_policy`. |

See [`references/predecessor_facts.md`](references/predecessor_facts.md) for the verbatim sources and
[`references/predecessor_index.md`](references/predecessor_index.md) for the path index.

## 10. Licensing

Upstream `anthropics/jacobian-lens` is Apache-2.0 and is vendored unmodified under
`third_party/jacobian-lens/<commit>/`. This repository's own code is in `src/wda/`.
