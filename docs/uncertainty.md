# Uncertainty: what works when calibration does not (run 2026-10-27)

Reproduce: `python scripts/run_uncertainty.py --seeds 10 --data combined`
Results: `results/uncertainty_{ensemble,conformal,selective}.csv`.

Temperature, Platt and isotonic scaling all failed under scaffold shift
([`calibration.md`](calibration.md)). These three alternatives do not assume the validation fold
resembles the test fold.

## 1. Ensembling: a small, real gain in probability quality

Mean probability over a random forest, XGBoost and logistic regression, all on the same fingerprints.

| | AUROC | Brier | log loss | ECE |
|---|---|---|---|---|
| random forest | 0.873 ± 0.066 | 0.126 | 0.423 | 0.153 |
| XGBoost | 0.880 ± 0.066 | 0.128 | 0.451 | 0.148 |
| logistic regression | 0.863 ± 0.082 | 0.135 | 0.479 | 0.157 |
| **ensemble** | 0.878 ± 0.063 | **0.123** | **0.411** | **0.148** |

Paired by seed against each member:

| vs | ΔAUROC | Δlog loss | ΔBrier |
|---|---|---|---|
| random forest | +0.004 (p=0.43) | −0.012 (p=0.49) | −0.002 (p=0.52) |
| XGBoost | −0.003 (p=0.63) | **−0.039 (p=0.036)** | −0.004 (p=0.31) |
| logistic regression | +0.015 (p=0.17) | **−0.068 (p=0.013)** | **−0.012 (p<0.001)** |

Ensembling does **not** improve ranking — no AUROC difference is significant. It does give
significantly better log loss than two of three members, i.e. better-behaved probabilities at no
cost. Worth keeping, not worth celebrating.

## 2. Conformal prediction: the one method that behaves correctly under shift

Inductive conformal prediction on the ensemble, calibrated on each split's validation fold.
Nonconformity is 1 − p(true label); the quantile uses the finite-sample correction
⌈(n+1)(1−α)⌉/n. The Mondrian variant takes the quantile separately within each class.

| split | target | **measured coverage** | mean set size | singleton sets |
|---|---|---|---|---|
| random, α=0.1 | 0.90 | 0.906 | 1.05 | 91% |
| **scaffold, α=0.1** | 0.90 | **0.922** | **1.32** | **66%** |
| random, α=0.2 | 0.80 | 0.827 | 0.90 | 90% |
| **scaffold, α=0.2** | 0.80 | **0.844** | 1.08 | 83% |

**Coverage holds under scaffold shift** — 0.922 against a 0.90 target, slightly conservative rather
than broken. The method pays for it in the right currency: the average prediction set grows from
1.05 labels to 1.32, and the share of decisive single-label answers falls from 91% to 66%. On a
third of unfamiliar molecules it returns {potent, not potent}, which is the correct answer to "what
do you know about this compound?"

This is the clearest methodological success in the project. Where temperature scaling tried to
express shift as a sharper or flatter probability and failed, conformal prediction expresses it as
*indecision*, and that transfers.

**The exchangeability caveat is real and must be stated.** Split conformal guarantees coverage when
calibration and test data are exchangeable. A scaffold split deliberately violates that: the
calibration fold holds different chemotypes from the test fold. The guarantee therefore does not
formally apply here, and the coverage reported above is an *empirical* finding over 10 splits, not a
theorem. It held in this setting; it is not promised to hold on a genuinely new chemotype, and the
Mondrian variant is slightly more conservative (0.926 at α=0.1) because class balance also shifts
between folds.

## 3. Selective prediction: abstaining on low confidence works, abstaining on low similarity does not

Error rate of the ensemble on the molecules it keeps, scaffold split, 10 seeds:

| coverage kept | by confidence | by model agreement | by similarity | random |
|---|---|---|---|---|
| 100% | 0.155 | 0.155 | 0.155 | 0.155 |
| 80% | 0.126 | 0.127 | 0.148 | 0.146 |
| 60% | 0.091 | 0.089 | 0.142 | 0.126 |
| 40% | 0.065 | 0.069 | 0.150 | 0.133 |
| 20% | **0.056** | 0.079 | 0.136 | 0.143 |
| 10% | 0.067 | 0.067 | 0.133 | 0.150 |

Keeping the 40% of molecules the ensemble is most confident about cuts the error rate from 15.5% to
6.5%. Ensemble disagreement works about as well. **Similarity to the training set is barely better
than abstaining at random** (0.142 against 0.126 at 60% coverage).

That looks contradictory next to [`generalization.md`](generalization.md), where AUROC collapsed on
dissimilar molecules, and the resolution matters: dissimilar molecules are mostly *negatives*
(18% positive below 0.5 Tanimoto against 73% above 0.7). A model that calls them "not potent" is
usually right, so its **error rate** stays low while its **ranking** is worthless. Error rate and
AUROC are measuring different failures.

The practical rule that follows: use **confidence** to decide whether to trust an individual
prediction, and use **similarity** to decide whether the model should be ranking that molecule at
all. They are not substitutes, and a screening pipeline needs both — which is how
[`candidates.md`](candidates.md) is built.

## Caveats
- 10 seeds, ~63 test molecules each; the error rates at 10–20% coverage rest on 6–13 molecules per
  seed and are correspondingly noisy.
- The ensemble is three model families on one representation. A fingerprint-only ensemble cannot
  express "this molecule is unlike anything I trained on" except through disagreement.
- Risk–coverage is reported as error rate at a 0.5 threshold. For a screening campaign, precision at
  a fixed budget would be the more decision-relevant curve.
