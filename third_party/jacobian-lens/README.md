# `third_party/jacobian-lens/` — upstream pin

**Upstream:** <https://github.com/anthropics/jacobian-lens> — Apache-2.0, a single-commit
reference implementation marked *not maintained*.

**Pinned commit:** see [`PINNED_COMMIT.txt`](PINNED_COMMIT.txt) —
`581d398613e5602a5af361e1c34d3a92ea82ba8e`. This is the same commit the predecessor project
vendored, so the two are byte-comparable.

## Policy

1. **Never modified.** The clone lands in `third_party/jacobian-lens/<commit>/` and is
   read-only. Any change belongs in `src/wda/`, not here.
2. **Not tracked in git.** `.gitignore` excludes `third_party/jacobian-lens/*/`; only the
   pin and this policy file are committed. Vendor it locally with
   `python tools/vendor_upstream.py`.
3. **Provenance recorded.** The vendoring tool writes `PROVENANCE.json` next to the clone
   with the resolved commit, the licence, `"modifications": "none"`, and a digest per file.
4. **Digest enters the seal.** `PROVENANCE.json` is what Phase 0 checks; the fixtures that
   read upstream data resolve their paths through it.

## What upstream does and does not provide

| | |
|---|---|
| **Provides** | `jlens.fit`, `JacobianLens.from_pretrained`, `lens.apply`; prompt sets under `data/evaluations/` and `data/experiments/` |
| **Does not provide** | ablation, patching, or gradient-pursuit sparse decomposition; GSM8K; any direct/CoT runner or scoring script |

That second row is why §2 of the protocol records "cannot confirm" for the upstream
patch/ablate interface, and why `wda.intervene.*` is self-built and gated by the five
Phase-0 conformance fixtures. **Do not paper over the gap by assuming an upstream API
exists.**

## Files the predecessor vendored from this commit (7)

```
LICENSE                                   11,358 B   Apache-2.0
PROVENANCE.json                            2,353 B   repo-authored manifest
data/evaluations/README.md                 3,815 B
data/evaluations/lens-eval-multihop.json  21,869 B   93 items
data/evaluations/lens-eval-order-ops.json  9,589 B   55 items
data/experiments/README.md                 8,570 B
data/experiments/probe-swap.json          26,567 B   90 items — a separate causal-swap
                                                     benchmark, NOT an evaluation set
```

**No upstream code of any kind was vendored**, and none is required to be.
`data/experiments/probe-swap.json` supplies the item set for Phase-0 fixture 5
(`known_intermediate_positive`) and for the exploratory `probe_swap` assay.

The upstream `scope.excluded` set recorded by the predecessor was
`["association", "typo", "multilingual", "poetry"]`.

## All upstream metrics are within-model

Every metric defined in `data/experiments/README.md` is computed **within a single model**;
none compares two models (fact F9). Two refinements worth carrying: `ignition` is an
α-sweep over an embedding interpolation rather than a two-condition difference, and
`capacity` reports a raw count with no contrast at all.

Consequently the between-checkpoint comparison in this project — `D` — has **no upstream
precedent** and is this project's own construction. It is built as a difference of
*within-model* differences precisely so that no cross-model absolute comparison is ever
required (facts F2, F4, F11).
