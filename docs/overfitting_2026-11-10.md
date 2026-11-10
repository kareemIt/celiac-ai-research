# Tier-1 overfitting decomposition — result

Run 2026-11-10. Fitting time as the script measured it: 1,487 s on TG2, 3,761 s on CDK2 and
1,328 s on MAO-B, so about 1 h 50 m in total. Design pre-registered in `docs/proposal_overfitting_2026-11-09.md`
(proposal A, tier 1); measurement `scripts/run_overfitting.py`; per-fold output
`results/overfitting_tier1.csv` (572 folds); cell summary `results/overfitting_tier1_cells.csv`
via `scripts/summarise_overfitting.py`.

## What was measured

For each committed configuration, each outer fold is refit and scored four ways:

| Column | Meaning |
|---|---|
| `in_sample` | the training fold, scored on itself |
| `oof` | out-of-fold inside the training fold, grouped the same way as the outer split |
| `test` | the outer test fold, full training size |
| `test_small` | the outer test fold, refit at the inner folds' reduced training size |

and the three differences: `overfit_gap = in_sample − oof`, `shift_gap = oof − test_small`,
`size_effect = test − test_small`. `size_effect` is the learning-curve control the proposal named
as the single largest threat to the design.

498 of the 572 folds carry a committed AUROC. The other 74 have one class in the test fold, so
AUROC is undefined there and the fold contributes no control either; they are excluded rather than
imputed.

## Reproduction gate

**498 of 498 folds reproduced bit-exactly** — max |test − committed| = 0.00e+00. The tied-pair
allowance added in response to the discarded first run was used zero times.

That is worth stating carefully rather than as a clean pass. The fold that failed the gate on the
earlier run, TG2 / scaffold / rf / seed 15, reproduced here at 0.8619733925 — which is exactly the
*upper* of the two values `results/rf_determinism.csv` records for that fold. Last run it landed on
the lower one. So bit-exactness on a fold with tied pairs is a coin flip between two known values,
not a property of the pipeline, and a run of this sweep should be expected to fail an exactness
check occasionally for reasons that are not a defect. The §8 limitation is updated accordingly.

## Coverage gap, inherited not introduced

MAO-B's publication-held-out arm appears for the random forest only. The measurement refits what the
committed sweep selected, and `results/sota_maob.csv` holds `paper_lopo` for `rf` alone — logistic
regression and XGBoost were never run on that arm. Nothing was dropped here.

## Result 1 — the overfitting gap is large everywhere, and it is not informative

`overfit_gap` runs from 0.085 to 0.332 with tight intervals in all 25 model×target×arm cells. It
would be easy to read that as the models overfitting badly. It is not what it looks like.

Median in-sample AUROC is 0.975 for logistic regression, 0.993 for the random forest and 0.997 for
XGBoost. The proposal pre-registered that fully grown bagged trees reproduce their training labels
and that their in-sample score is therefore uninformative. The same holds for the other two families
here: 2048-bit fingerprints against a few hundred training molecules puts all three in the regime
where the training labels are separable, so `in_sample` is pinned near the ceiling by capacity and
`overfit_gap` is mostly a restatement of `1 − oof`. It measures the train-fit ceiling, not whether
any reported test score is inflated.

**This is the pre-registered null, and it is the result.** Nothing in the decomposition shows the
committed test scores to be optimistic.

## Result 2 — the shift term is readable in 5 cells of 27, and only under leave-one-publication-out

| Arm | `shift_gap` | control | readable |
|---|---|---|---|
| TG2 scaffold (3 models) | −0.004 … +0.015 | 0.008 … 0.015 | no — every interval crosses zero |
| TG2 scaffold_pub (3) | −0.017 … +0.004 | 0.008 … 0.015 | no |
| CDK2 scaffold_pub (3) | +0.011 … +0.018 | 0.006 … 0.009 | no |
| MAO-B scaffold_pub (3) | +0.002 … +0.009 | 0.006 … 0.009 | no |
| CDK2 paper_grouped (3) | +0.010 … +0.013 | 0.021 … 0.032 | no — control exceeds the effect |
| MAO-B paper_grouped (3) | +0.001 | 0.009 … 0.012 | no — control exceeds the effect |
| TG2 paper_lopo (3) | −0.002 … +0.077 | 0.002 … 0.043 | no — see result 3 |
| **CDK2 paper_lopo (3)** | **+0.127 … +0.168** | 0.002 … 0.013 | **yes** |
| **MAO-B paper_lopo (rf)** | **+0.174** | 0.002 | **yes** |

Pooled over CDK2 and MAO-B publication-held-out folds: `shift_gap` +0.154 [+0.127, +0.181],
learning-curve control +0.005 [−0.004, +0.015], 170 of 228 folds positive, Wilcoxon p ≈ 6e-20. The
control is a thirtieth of the effect, so on these arms the design does what it was built to do.

Everywhere else the term is at or below the control and every interval crosses zero. **The
publication penalty is not a general property of these models**; it appears only when the test fold
is a single held-out publication.

## Result 3 — the decomposition does not resolve on TG2, exactly as pre-registered

TG2's publication arm has about ten publications per training fold, and the proposal said in advance
that a grouped inner 5-fold on that would be too noisy for a per-model statement, with the fallback
being to pool across models. Pooled: `shift_gap` +0.032 [−0.045, +0.106], Wilcoxon p = 0.35,
against a control of −0.018 [−0.035, −0.001] — the same order as the effect. Per model the
intervals are wider still (logistic regression −0.002 [−0.124, +0.143]).

So on the paper's primary target the answer is: **the design cannot separate memorisation from
shift, and that is reported rather than worked around.** By the success criteria fixed in the
proposal, tier 1 succeeds on CDK2 and MAO-B and fails on TG2.

This also retracts the preview quoted in §9 of the paper. That preview — a near-zero shift term,
read as the publication penalty being visible inside the training fold — came from TG2 logistic
regression, which is one of the three cells that turn out not to resolve. It should not have been
given a directional reading.

## What the readable term is, and what it is not

Where `shift_gap` is large, it is tempting to call it publication shift. It cannot carry that name
without a caveat that weakens it considerably.

`oof` is pooled across the training fold's publications; `test` under leave-one-publication-out is a
single publication scored on its own. Those are different aggregations, and §5.14 already
demonstrated that the publication-held-out decline on this data is mostly novelty and aggregation
rather than publication identity. The arm that separates them here agrees: `paper_grouped` holds
publications out but scores them pooled, and its shift term is +0.001 to +0.013, below its own
control. The penalty appears when scoring is confined to one publication and disappears when the
same held-out publications are pooled.

**So the one readable term in this decomposition is consistent with §5.14's aggregation effect, and
this experiment does not separate the two.** Doing so would need an outer arm that holds out several
publications and scores them pooled while matching LOPO's training size — which `paper_grouped`
nearly is, and which is the cheap next measurement rather than a new intervention.

## Status

- Overfitting beyond the train-fit ceiling: **no evidence**, and the in-sample column is not
  capable of providing it for any of these three model families at this data size.
- A distribution-shift term separable from the learning curve: **demonstrated on CDK2 and MAO-B
  under leave-one-publication-out**, +0.154 pooled, and **not established on TG2**.
- The identity of that term: **not established** — confounded with the aggregation effect of §5.14.
