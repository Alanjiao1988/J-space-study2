# Predecessor facts F1–F13 — verbatim sources and errata

Source repository: [`Alanjiao1988/J-space-observation`](https://github.com/Alanjiao1988/J-space-observation)
pinned at `d87c1b9e4e9dca062cebad7b6eee981d9dba8c25`.
Terminal status: `DISCONTINUED — RESEARCH QUESTION UNANSWERED` (operator decision
`PROJECT_DISCONTINUATION_2026-08-29`).

Every claimed path exists at that commit. **Twelve of thirteen facts verify; F6′ carries
three material errors and F13 one misattribution.** Those are recorded in the errata at the
bottom of this file and must be resolved before Freeze-1 is signed — pre-freeze correction
is permitted (§10 Phase 0: "冻结前允许"), post-freeze it would be an SC5 matter.

---

## F1 — 1.5B behavioural eligibility

**Path:** `studies/study1/README.md` § "Terminal E0 result" — **VERIFIED**

```
| Distribution | Official items | Mechanical eligible | Behavioral eligible | Confirmation |
|---|---:|---:|---:|---:|
| multihop | 93 | 79 | 2 | 0 |
| order-ops | 55 | 36 | 2 | 0 |
| causal-swap | 90 | 83 | 5 | 0 |

All nine behaviorally eligible rows entered development under the frozen
development-first split. Every confirmation floor was therefore zero. No
lens, E1/E2, intervention, ablation, patching, or RQ2 operation occurred.
```

**Scope corrections.** (a) 2/93 · 2/55 · 5/90 are **behavioural eligibility**, not accuracy;
confirmation was 0/0/0. (b) Only `multihop` (93) and `order-ops` (55) come from upstream
`data/evaluations/`; the 90-item `causal-swap` set is upstream
`data/experiments/probe-swap.json`, which the vendored `PROVENANCE.json` labels a
`"separate causal-swap benchmark"`. So "on the upstream evaluations … 5/90" is mis-scoped.

**Consequence in this repo:** `wda.models.registry.DENYLIST` refuses the 1.5B, quoting F1.

---

## F2 — four-option restricted-choice interface

**Path:** `studies/study2/analysis/stage_bd_posthoc_interface_diagnostic.md` §4.1, §4.3 — **VERIFIED**

```
**4.1 Option C was never selected in the no-trace arm.** Across all 384 target NT
rows, at every depth, in both families, the restricted prediction is C exactly zero
times. Under the balanced four-option design, C is the correct label in a quarter of
items, so the interface produced a correct response on those items zero times by
construction.
...
**4.3 The shift does not come with an accuracy gain.** Arm accuracies are 0.2396
(NT), 0.2617 (PT), 0.2656 (ST) and 0.2734 (WT). Every arm sits near the 0.25
restricted-choice chance level.
```

**Status caveat.** The document carries the banner
`POST_HOC_DESCRIPTIVE_ZERO_AUTHORITY_NOT_SCIENTIFIC_EVIDENCE` and §4.4 states: *"The cause
is not identified and cannot be identified from these data… Nothing here distinguishes an
incapable checkpoint from an inadequate instrument."* Cite F2 as **descriptive**, never as
a finding. The document's own term is `restricted_prediction` / "restricted-choice";
"restricted-logit" appears in `studies/study3/README.md`.

**Consequence:** absolute accuracy levels are never an estimand here; only within-model
differences are (§9, and `wda.stats.bootstrap`).

---

## F3 — two-token content surfaces

**Path:** `studies/study3/README.md` § "STUDY 3-P0 STAGE P0-T RAN AND STOPPED" — **VERIFIED, no discrepancy**

```
`S1`'s four label surfaces are distinct single tokens under both alphabets for all
three roles; but `S2` and `S3` are `INELIGIBLE_TOKEN_IDS` for all three roles
because each registered content surface `" 0"`..`" 9"` is **two** tokens
(`[220, digit]`), so the registered single-position restricted-logit rule is
not implementable as written.
```

**Consequence:** `wda.conditions.envelopes.assert_cue_length` verifies `L_p` against each
subject's real tokenizer *before the first model call*, rather than trusting the registered
constant.

---

## F4 — 7B / 14B CoT and raw-direct results

**Path:** `studies/study4f/execution-m1/M1_FINAL_DISCLOSURE.md` §11, §13 — **VERIFIED**

```
| `RP_B1|D2|COT` | 104 | 97  | 7 | 90 | **PASS** |
| `RP_B1|D3|COT` | 104 | 97  | 7 | 90 | **PASS** |
| `RP_B2|D2|COT` | 104 | 104 | 0 | 90 | **PASS** |
| `RP_B2|D3|COT` | 104 | 101 | 3 | 90 | **PASS** |
| `RP_B1|D2|E0`  |  60 |   0 | 60 | 41 | FAIL |
```

```
All 240 E0 continuations were `UNPARSEABLE`. ...
* RP_B1's continuations all began with token `151649` = `</think>`;
* RP_B2's all began with token `32313` = `Okay`;
```

**Count correction.** There are **four** CoT cells, not three: the 7B scores **97/104 at
both D2 and D3**. §13: *"the 7B checkpoint reaches 97/104 at both depths, the 14B reaches
104/104 and 101/104."* The 32B (`RP_B3`) is in the same table (94/104 D2 PASS, 85/104 D3
FAIL) and its E0 cells were never scheduled — which is why the E0 denominator is 240, not 360.

**Consequence:** the ITT composite is the only endpoint entering `D`; parseability is
reported per condition; a `<think>` opening is classified as
`UNPARSEABLE_REASONING_SPAN` rather than lumped in with "no integer found"
(`wda.scoring.parse`).

---

## F5 — the 4F E0 envelope

**Path:** `studies/study4f/analysis/study4f_interfaces.py` — **VERIFIED, no discrepancy**

```python
``W1_RAW_DIRECT`` (primary, E0)
    The raw direct-answer wrapper. No chat template and no forced ``</think>``
    closure. Legal answers are exactly the frozen one-token surfaces A/B/C/D.
```

```python
W1_PROVENANCE: Dict[str, object] = {
    ...
    "uses_a_chat_template": False,
    "forced_reasoning_closure": None,
    "forced_reasoning_closure_is_explicitly_absent": True,
}
```

**Refinement.** The contrast route `C1_LONG_GENERATED_COT_HEADROOM` **does** use a chat
template and also does not force closure. "No forced closure" is therefore true of *both*
routes; only "no template" distinguishes E0.

**Consequence:** §5 of the new protocol requires the **native chat template in every
condition**; `Envelope` refuses any other value.

---

## F6′ — Study 5 EQ2 layer sweep — **VERIFIED WITH THREE MATERIAL ERRORS**

**Paths:** `studies/study5/closure/STUDY5_CLOSURE.md`, `studies/study5/qualification-eq2/TIMELINE.md` — both exist.

What the sources actually say:

```
`Qwen2.5-7B-Instruct` @ `a09a3545` (positive control), `Qwen3-1.7B` @ `70d244cc`
(depth test), `gpt2` @ `607a30d7` (negative control), `trust_remote_code=false`.
```

```
**The positive control partially reproduces the published signature**, measured
with no kurtosis anywhere: readrate is *exactly* 0.0000 through layers 0–12,
first non-zero at 13, rises steeply from 20 → 21 → 22 → 23 → 24 → 25, then
**falls at the last layer** to 0.0319. ... The rise begins *later* than "a third of
the way through" — about layer 20 of 27
```

```
| band (registered rule) | **[21…26]** | **[20…26]** | **[]** |
```

```
**The band collapses from six layers to one.** ...
**exactly one layer, 21, clears both the null and the logit lens**.
```

```
| layer | 0 | 9 | 18 | 21 | 23 | 25 | 26 |
| identity share | 0.004 | 0.021 | 0.169 | 0.469 | 0.640 | 0.729 | **0.749** |
with the best scaled-identity α ≈ **1.0** at layers 23–26.
```

### Errata (must be applied before Freeze-1 is signed)

| # | Claim in the design document | What the source says |
|---|---|---|
| **E1** | swept on `Qwen2.5-Math-7B` | the swept model is **`Qwen2.5-7B-Instruct`**, the *external positive control*. That string appears nowhere in the file. The target `T` was **never touched** and `lens_A`/`lens_B` were **never read** (guard PASS, 0 records). |
| **E2** | 28 layers | the positive control has **27 source layers (0–26)**: *"about layer 20 of 27"*, *"layers 21–26 of source layers 0–26"*. The "28 layers" figure belongs to a different claim — the EQ1 shallowness hypothesis about the target, quoted as *"'28 layers is too shallow' is externally refuted."* |
| **E3** | "the negative control also produced band [9]" | `[9]` was the **pre-OA-004** result that fired blocker HB-002. After the matched-norm random-lens null was added, gpt2's band became **`[]`**: *"**The negative control now holds.** … HB-002 is lifted on the merits, not by moving a threshold."* Citing `[9]` as EQ2's standing result is incorrect. |
| E4 | "identity energy 0.749 … over L23–26" | 0.749 is at **L26 specifically**; `α ≈ 1.0` is what spans L23–26. `STUDY5_CLOSURE.md` §2 states this correctly. |
| E5 | "swept all layers" | supported, but the readrate is pass@1 over ~900 scored intermediates and the band's upper edge is explicitly *"right-censored by the end of the network"*. |

### Corrected statement of F6′

> Study 5 EQ2 swept all 27 layers of the external positive control **`Qwen2.5-7B-Instruct`**.
> Pass@1 readrate is exactly 0 at L0–12, first non-zero at L13, and rises from ≈L20; the
> registered-rule band is `[21…26]`, which condition (ii) collapses to the single layer
> `[21]`. Identity energy climbs monotonically to 0.749 at L26 with `α ≈ 1.0` over L23–26.
> The `gpt2` negative control produced band `[9]` under the pre-OA-004 rule; **after the
> matched-norm random-lens null was added it produces no band.** The target `T` was never
> touched and neither fitted lens was ever read.

**What survives regardless of the errata:** the terminal result *"no readrate signal in
early/middle layers"* holds; late-layer `J → I` is expected motor-layer behaviour and not an
instrument pathology; and the registered band rule genuinely lacked an **absolute floor** and
a **minimum length**.

**Consequence:** `configs/freeze1/band_rule.json` and `wda.lens.band.BandRule` add four
repairs — absolute floor, minimum length, matched-norm random-lens null margin, and
rejection of right-censored bands. E3 is the reason repair 3 exists: adding the null is what
actually made the negative control behave.

---

## F7 — bf16 reduction offset

**Path:** `studies/study5/closure/STUDY5_CLOSURE.md` §5–6 — **VERIFIED**

```
- **bfloat16 reduction order** produces a batch-width-dependent offset on the
  order of 0.476 / 0.110937 logits — the same order as the effects under
  measurement. **Any bf16 patching result that has not demonstrated bit-exact
  no-ops is uninterpretable.**
```

```
The full-donor signal is itself only +0.038606 against a bias of 0.507; after
re-centring, the signal-to-noise ratio still does not hold.
```

**Refinement.** The two statements live in different sections: `+0.038606` is compared in
§5(b) against a **no-op bias of 0.507**, while the 0.476 / 0.110937 bf16 figures are in §6.
The design document's single sentence merges them. Also §5(a): the attenuated transfer
`+0.043370` *exceeded* the full-donor `+0.038606`, and the record insists this must **never**
be written up as "non-monotonicity was established".

**Consequence:** `wda.intervene.precision` enforces fp32 and a bit-exact no-op, and
`noop_bitexact` is a per-trial field in `log.jsonl`.

---

## F8 — no positive control was ever obtained

**Path:** `studies/study5/closure/STUDY5_CLOSURE.md` §6–8 — **VERIFIED, no discrepancy**

```
- This study **never obtained a passing positive control on the J-lens line**.
  The negative conclusions are strong about *our execution* and weak about *the
  method itself*.
```

```
| cumulative actively used GPU-hours | **40.144672** |
| `T` touched | never |
| `lens_A` / `lens_B` read | never |
```

§8 adds two prohibitions this project inherits: *"It must not be claimed that the published
lenses are defective"*, and *"'The question remains open' is not this project's conclusion.
The correct statement is that **this project failed to test the question.**"*

**Consequence:** Phase-0 fixture 5 and Phase-B criterion 4 make a causal positive control a
precondition of any scientific run (`configs/freeze1/fixtures.json`).

---

## F9 — upstream metrics are all within-model

**Exact path:** `third_party/jacobian-lens/581d398613e5602a5af361e1c34d3a92ea82ba8e/data/experiments/README.md` — **VERIFIED**

```
## probe-swap ... Baseline: greedy next-token == `answer`. Swap: replace the
`intermediate` representation ... score next-token at the final position == `swap_answer`.
## verbal-introspection ... strength 0 is the control.
## directed-modulation ... The metric is hit rate, contrasted across `group_kind`.
## top-down-summoning ... Metric: Q2 − Q1 fraction of stimulus positions ...
## selectivity-language ... Metric: explicit − automatic label-hit rate.
## selectivity-linecount ... contrasted across conditions over the eleven passages.
## dual-task ... interference = single-task − dual-task reachability.
```

**Precision correction.** `ignition` is an **α-sweep over an embedding interpolation**
(`α·emb(A) + (1−α)·emb(B)`), not a two-condition difference; `capacity` reports a **raw
count** with no contrast. The safe and load-bearing phrasing is: *all upstream metrics are
computed **within a single model**; none compares two models.*

---

## F10 — what was vendored

**VERIFIED WITH CORRECTION.** The directory
`third_party/jacobian-lens/581d398613e5602a5af361e1c34d3a92ea82ba8e/` holds **7 files**:

| path | bytes | role |
|---|---:|---|
| `LICENSE` | 11,358 | Apache-2.0 |
| `PROVENANCE.json` | 2,353 | repo-authored vendoring manifest |
| `data/evaluations/README.md` | 3,815 | upstream |
| `data/evaluations/lens-eval-multihop.json` | 21,869 | 93 items |
| `data/evaluations/lens-eval-order-ops.json` | 9,589 | 55 items |
| `data/experiments/README.md` | 8,570 | upstream |
| `data/experiments/probe-swap.json` | 26,567 | **separate causal-swap benchmark**, 90 items |

"No code" is **correct** — zero `.py` / `.ipynb` / `.sh`. "Only three JSONs" is imprecise:
three *upstream data* JSONs, plus a fourth repo-authored `PROVENANCE.json`, two upstream
READMEs and the licence. `PROVENANCE.json` records `"modifications": "none"` and
`scope.excluded: ["association", "typo", "multilingual", "poetry"]`.

**Consequence:** this repo vendors upstream unmodified under
`third_party/jacobian-lens/<commit>/` and pins it in `PINNED_COMMIT.txt`; §2 of the protocol
already records that ablation / patching / gradient-pursuit must be self-built.

---

## F11 — absolute-threshold gates

**VERIFIED for Studies 2 and 4F; scope narrower than claimed.**

Study 2 — `studies/study2/decisions/study2_stage_bd_gate_a_decision.md` §2, §4:

```
- exact one-sided binomial upper tail under p₀ = 0.25, α = 0.025;
- a family passes only when X ≥ 43, equivalently p_exact ≤ 0.025;
...
There is no cross-family pooling, depth fallback, favorable subgroup, template
selection, multiplicity rescue, control-model rescue, or prior-result rescue.
```

The strongest single piece of evidence: the `lineage_base | affine_mod10` control cell
scored X = 44 — it *would* have passed — and was still given zero authority
(`controls_affect_decision: false`).

Study 4F — `studies/study4f/README.md`:

```
| CoT gate | `n = 104`, pass iff correct `≥ 90`; exact size `0.0029878`, exact power `0.9055` |
| E0 gate  | `n = 60`,  pass iff correct `≥ 41`; exact size `0.0031088`, exact power `0.9075` |
| depth pooling | prohibited — D2 and D3 are always separate cells |
```

**Two corrections.** (a) Study 3's *terminal* judgment is a **qualitative methods-review
rejection** (12 BLOCKING / 3 MAJOR / 2 MINOR), not a threshold gate; its registered *design*
did contain absolute thresholds but they were never exercised
(`formal_execution_authorized = false`). (b) A counter-example exists outside the claimed
scope: Study 5 EQ2 deliberately used a **relative** rule — *"`OD-015` defines the band as
the longest run at or above half of the maximum. That is scale-free by construction —
deliberately, to avoid inventing an absolute threshold."* So F11 should read
**"Studies 1 / 2 / 4F"**, not "all".

**Consequence:** this project's gate is a CI-vs-`δ` rule on a within-model difference
(§9.2), not an absolute accuracy threshold — and the band rule now carries *both* an
absolute floor and a relative bar, since each alone has a documented failure mode.

---

## F12 — what public checkpoints can identify

**Path:** `PROJECT_DISCONTINUATION.md` §3 — **VERIFIED verbatim**

```
3. Comparisons among public end-state checkpoints can establish, at most,
   checkpoint-level or distillation-associated differences; they cannot identify
   what the distillation training caused.
4. A stronger causal claim would require controlled training of matched model
   groups. ...
```

**Consequence:** §4 and §13 cap the claim at two named checkpoints; Appendix C is the only
authorised causal upgrade path and it is explicitly not authorised.

---

## F13 — execution discipline

**Path:** `studies/study4f/execution-m1/M1_FINAL_DISCLOSURE.md` — **VERIFIED (2 of 3 sub-claims)**

```
The seal was published at commit `1265638` **before** the first study-bank model
call. Seal sha256:
`5a59cc432ebb63dc23ea5ed07fe9ba207124b11f731a13d6422f203d4c3973c7`.
```

```
### Execution integrity
| Duplicate keys | **0** |
| Create-only violations | **0** |
| Cells repeated after seeing a result | 0 |
| Engineering fixes after the first study-bank call | 0 |
| Hardware switches after the first study-bank call | 0 |
| Reseals | 0 |
```

### Erratum — misattribution

The **hard-kill / CPU-recovery canaries are not a Study 4F-M1 artifact.** M1's canaries are
the §10 shakedown items (a four-device equivalence canary producing byte-identical token ids
`[151649, 271]` on all four A100s, *"recorded explicitly as an engineering qualification
only"*). The hard-kill and CPU-recovery canaries belong to **Study 3 P0-R1 / P0-R2** —
`studies/study3/README.md`: *"Exact ACR-task, non-root CLI injection, private-prefix, Blob
journal, recursive-manifest, hard-kill and CPU recovery canaries passed"* — with files
`studies/study3/pilot/p0_r1/container/p0_r1_hard_kill_canary_v3.py`,
`…/p0_r1_recovery_v3.sh`, `…/p0_r1_recovery_job_v3.yaml`, and
`studies/study3/pilot/p0_r2/p0_r2_hard_kill_canary_v2.py`.

**Qualifier.** M1 §10 discloses two whitelisted engineering repairs (two mis-authored
synthetic fixtures; `CUBLAS_WORKSPACE_CONFIG=:4096:8`), bounded as *"Neither touched a
scientific byte."* — so "zero re-run" is exactly right **at the scientific layer**, and
should always be stated with that qualifier.

**Consequence:** `wda.governance.seal` seals before the first call and re-verifies at the
end of every run; `wda.governance.rerun_policy` separates pre-registered infrastructure
codes (including `IF-CPU-FALLBACK` and `IF-HARD-KILL`) from scientific re-runs, which are
refused unconditionally.

---

## Summary of pre-Freeze-1 actions

| Erratum | Action |
|---|---|
| F6′ E1–E3 | Corrected statement above is the version this repository cites. `PROTOCOL_v1.1.md` retains the operator's original text; this file is the authoritative correction. |
| F13 misattribution | Hard-kill / CPU-recovery canary provenance re-attributed to Study 3 P0-R1/P0-R2. |
| F1, F4, F9, F10, F11 | Scope narrowed as noted; no design change follows. |

None of these errata changes the design of this study. F6′ E3 *strengthens* the case for
band-rule repair 3 (the matched-norm random-lens null), which was already adopted.
