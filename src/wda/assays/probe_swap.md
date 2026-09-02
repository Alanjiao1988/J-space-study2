# Estimand spec — `probe_swap`

**Tier:** EXPLORATORY (§8.5). Reported, never inferred from, not multiplicity-corrected.

## Estimand

Per item, with the recipient's lens coordinates on the selected components replaced by the
donor's (`wda.intervene.patch.swap_lens_coordinates`):

```
target_logit_delta            = logit(swap_answer | swapped) − logit(swap_answer | clean)
matched_norm_random_delta     = same, under an equal-norm random perturbation
direct_substitution_delta     = same, under direct final-token distribution substitution
```

## Mandatory paired control — this is the whole point

`direct_substitution_delta` is **required**; `analyze` raises without it. tao-hpu's
replication showed that simply substituting the donor's final-position token distribution
reproduces much of the apparent probe-swap effect. A swap result reported without that
baseline does not distinguish "the workspace carried the intermediate quantity" from
"the output distribution was overwritten".

`matched_norm_random_delta` is the null: the positive control holds only when the swap
moves the target logit and the equal-norm random perturbation does not.

## Depth separation control (§7)

`answer_vs_intermediate_swap_depth`: the intermediate-quantity swap should take effect at a
shallower depth than the answer swap (the paper reports ≈17% of network depth). A
separation at or below zero is consistent with the intermediate vector simply carrying the
answer, and must be reported as such.

## Item set

Upstream `data/experiments/probe-swap.json`, 90 items. Note this file is a **separate
causal-swap benchmark** under upstream `data/experiments/` — it is not part of
`data/evaluations/`. Fact F1's "5/90" refers to this set, not to an evaluation set.

## Paper reference values (Appendix B, n = 90)

J-space component 61% · raw swap 60% · non-J 28% · non-J with clamp 6%.
Descriptive only; not targets.

## Interpretation limit

Exploratory. No p-value from this assay is corrected, and no conclusion about `D`, about
J-space, or about either checkpoint may rest on it (§8.5, §9.6, §13).
