# Estimand spec — `flexible_generalization`

**Tier:** SECONDARY CONFIRMATORY (§8.5). Holm-corrected (§9.6), and interpreted **only if
the primary endpoint is material**.
**Why it is in the set:** of the paper's claims put through external review, multi-fact
editing / flexible generalization is the one that replicated cleanly. Rhyme-planning and
mental-arithmetic did not and are `EXCLUDED`; probe-swap was weak and is `EXPLORATORY`.

## Estimand

`rate(arm) = successes(arm) / n` over one common edit set, and

```
contrast = rate(jspace_edit) − rate(layer_matched_random)
```

evaluated at each registered intervention strength `α`.

## Mandatory arms

| Arm | Role |
|---|---|
| `jspace_edit` | the intervention |
| `layer_matched_random` | `k` random directions in the same layer band — the negative control |
| `matched_norm_perturbation` | equal-norm arbitrary perturbation |

All three run on the **same** edit set; `analyze` refuses arms of differing size, because
the contrast is only defined over a common set.

## Paper reference values (Appendix B)

76/192 at `α = 1`; 101/192 at `α = 2`.

These are **calibration references, not targets.** Matching them is not a success
criterion and missing them is not a failure of this study. They exist so that an
implementation that is off by an order of magnitude is visible as an engineering problem
rather than read as a scientific result.

## Reporting

Report the contrast with its bootstrap CI at every `α`, together with both control arms.
Interpretation is gated on the primary outcome; if the primary is `equivalent` (SC2) or
`inconclusive` (SC3), this assay is reported and not interpreted.
