# Predecessor repository index

`https://github.com/Alanjiao1988/J-space-observation` @ `d87c1b9e4e9dca062cebad7b6eee981d9dba8c25`

Terminal status: **`DISCONTINUED — RESEARCH QUESTION UNANSWERED`**
(`PROJECT_DISCONTINUATION_2026-08-29`). No Study 6, successor protocol, additional estimand
search, model execution or cloud operation is authorised *in that repository*. This
repository is a separate, newly frozen project; nothing here resumes that one.

> **The single most important inherited statement**, from its `README.md`:
> *"**This project never tested its hypothesis.** All six Study 5 phases failed at the level
> of the measurement apparatus, not at the level of the hypothesis. It must not be claimed
> that J-space exists or does not exist."*

---

## Top-level documents

| Path | What it is |
|---|---|
| `README.md` | Project overview; terminal status; the "never tested its hypothesis" boundary |
| `PROJECT_DISCONTINUATION.md` | Project-level stop. §3 = fact F12; §4 what the repo does/doesn't support; §5 per-study terminal records; §6 demotes prospective language to historical provenance; §8 the restart condition |
| `studies/README.md` | Terminal routing table, one row per study |

### Terminal state strings (from `studies/README.md`)

| Study | Terminal state |
|---|---|
| 1 | `INSUFFICIENT_BEHAVIORAL_SUPPORT_FOR_VALIDITY` |
| 2 | `STUDY2_PROTOCOL_V1_CLOSED_ON_DEVELOPMENT_FEASIBILITY` |
| 3 | `STUDY3_DRAFT_V0_7_REJECTED_TERMINAL_NO_EXECUTION` |
| 3R | `STUDY3R_TERMINAL_CLOSURE_COMPLETE_RESEARCH_QUESTION_UNANSWERED` |
| 4F-M1 | `STUDY4F_M1_NO_NATURAL_POSITIVE_REFERENCE_QUALIFIED_WITHIN_REGISTERED_LADDER` |
| 5 | J-lens route terminated |

Preserved assets it names: the constructed two-hop object (a selection-set engineering
asset) and the bit-exact patching harness — whose no-op guarantees *"must be re-proven on
any future object"*, which is exactly what Phase-0 fixture `noop_returns_zero` does here.

---

## Paths cited by this project

### Study 1 — behavioural eligibility collapse
- `studies/study1/README.md` — **F1**; 2/93, 2/55, 5/90; no lens was ever read
- `studies/study1/asset_index.csv`, `studies/study1/terminal_manifest.json`

### Study 2 — interface diagnosis
- `studies/study2/analysis/stage_bd_posthoc_interface_diagnostic.md` — **F2**; C = 0/384
- `studies/study2/decisions/study2_stage_bd_gate_a_decision.md` — **F11**; `X ≥ 43` gate; controls given zero authority
- `studies/study2/decisions/study2_stage_bd_interpretation_erratum.md`
- `studies/study2/protocol/reasoning_internalization_protocol.{json,md,schema.json}`
- `studies/study2/stage_bd/stage_bd_preinference_seal.json` — seal-before-inference precedent
- `studies/study2/STUDY2_PROTOCOL_V1_TERMINAL_HANDOFF.md`

### Study 3 — tokenisation gate and pilot infrastructure
- `studies/study3/README.md` — **F3** (P0-T, `" 0"`..`" 9"` = `[220, digit]`); also the hard-kill / CPU-recovery canary provenance re-attributed from **F13**
- `studies/study3/reviews/v0_7_operator_terminal_decision.md` — 12 BLOCKING / 3 MAJOR / 2 MINOR
- `studies/study3/pilot/p0_r1/container/p0_r1_hard_kill_canary_v3.py`, `…/p0_r1_recovery_v3.sh`, `…/p0_r1_recovery_job_v3.yaml`
- `studies/study3/pilot/p0_r2/p0_r2_hard_kill_canary_v2.py`
- `studies/study3/analysis/independent_methods_recalculation*.py` — independent-recalculation precedent

### Study 3R — ladder
- `studies/study3r/README.md`, `studies/study3r/STUDY3R_TERMINAL_CLOSURE.md`

### Study 4F — interfaces and the M1 execution
- `studies/study4f/analysis/study4f_interfaces.py` — **F5**; W1/C1 envelopes (see below)
- `studies/study4f/execution-m1/M1_FINAL_DISCLOSURE.md` — **F4**, **F13**; §11 cell table, §11 "Execution integrity", §13
- `studies/study4f/README.md` — the CoT / E0 absolute gates (**F11**)
- `studies/study4f/execution-m1/execution_seal.json`, `…/banks/bank_digests.json`
- `studies/study4f/protocol/study4f_protocol_v1.{json,schema.json}`

### Study 5 — the J-lens line
- `studies/study5/closure/STUDY5_CLOSURE.md` — **F6′**, **F7**, **F8**; §2 failure map, §5–6 bf16, §6–8 no positive control
- `studies/study5/qualification-eq2/TIMELINE.md` — the full-layer readrate sweep, the band-rule defect, HB-002 and its lifting by the matched-norm null
- `studies/study5/closure/STUDY5_HANDOFF.md` — the bit-exact harness and surviving assets
- `studies/study5/qualification-eq1/TIMELINE.md` — EQ1
- `studies/study5/closure/{audit_closure,verify_integrity,reorder_closure,restore_eol}.py`

### Vendored upstream (**F9**, **F10**)
- `third_party/jacobian-lens/581d398613e5602a5af361e1c34d3a92ea82ba8e/`
  - `LICENSE` · `PROVENANCE.json`
  - `data/evaluations/README.md` · `lens-eval-multihop.json` (93) · `lens-eval-order-ops.json` (55)
  - `data/experiments/README.md` · `probe-swap.json` (90)
  - **no upstream code of any kind**

---

## `study4f_interfaces.py` — the design this repo's `conditions/envelopes.py` inherits from

| Identifier | Role |
|---|---|
| `LABELS` | `("A","B","C","D")` — the only legal answer labels |
| `RAW_ENVELOPE_SEPARATOR`, `ANSWER_CUE` | `"\n\n"`, `"Answer:\n"` |
| `W1_PROVENANCE` | copied-bytes provenance + `uses_a_chat_template: False`, `forced_reasoning_closure: None` |
| `W1_SURFACE_FIXTURE_BODY` | a deterministic placeholder that freezes the rendered surface — *"surface fixture, never a scientific item"* |
| `render_w1_raw_direct(item_body)` | pure function of `item_body`, so no answer-derived field can leak |
| `w1_surface_sha256()` | binds the surface to a source hash |
| `E0_GENERATION_CONTRACT` | every decoding field frozen (`do_sample False`, `max_new_tokens 2`, …) |
| `E0_VERDICTS` | `CORRECT / INCORRECT_WRONG_LABEL / INCORRECT_MISSING_EOS / INCORRECT_EXTRA_TOKEN / UNPARSEABLE` |
| `parse_e0`, `score_e0`, `e0_outcome` | exact-shape parsing; nothing is dropped |
| `C1_PROVENANCE`, `C1_INSTRUCTION`, `C1_LEGAL_FINAL_LINES`, `C1_GENERATION_CONTRACT` | the CoT route, with the chat-template digest bound |
| `cot_seed(...)` | SHA-256-derived per-item seed, recordable before generation |
| `parse_cot`, `score_cot`, `cot_outcome` | the final non-empty line must match a legal line **exactly** |

The module docstring states the rationale this project adopts wholesale:

> *"Both parsers are written as **exact** comparisons rather than regular expressions, which
> makes the coordinated `adv_cot_parser_regex_unanchored` mutation structurally inapplicable
> rather than merely killed."*

Accordingly `wda.scoring.parse` uses explicit digit-run extraction and an exact
single-integer rule; two integers is `UNPARSEABLE_MULTIPLE_INTEGERS`, never "take the last
one".

---

## What this project does **not** inherit

- Any conclusion about J-space. The predecessor established none (F8, and its own §8).
- Any claim that the published lenses are defective — its closure explicitly forbids that.
- Any absolute-threshold accuracy gate (F2, F11).
- The 1.5B checkpoint as a subject (F1).

## Verification

Every path above was fetched from the GitHub API at the pinned commit while assembling
`predecessor_facts.md`. Three material errata in F6′ and one misattribution in F13 are
recorded there.
