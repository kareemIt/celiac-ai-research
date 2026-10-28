# Fingerprint baselines (run 2026-10-26)

Reproduce: `python scripts/run_baselines.py tg2 --seeds 10` and `python scripts/run_baselines.py gdsc2`.
Per-seed rows: `results/baselines_{tg2,gdsc2}.csv`. Metrics: `src/evaluation/metrics.py`.

Protocol: fixed default hyperparameters, **no tuning**. Train on the train split, report test (valid is reserved for later tuning). Mean ± sd over seeds; each seed is a different split.
Features: 2048-bit Morgan (r=2). GDSC2 adds cell-line expression PCA-256, fit on training cell lines only. XGBoost 500 trees, depth 6 (GDSC2: 1000 trees, depth 8, GPU).

## TG2: scaffold split, 10 seeds

### Classification: potent if IC50 ≤ 1 µM (pIC50 ≥ 6)
Same 305 molecules as regression: **160 potent / 145 not (52.5%)**. Test folds have 24 ± 9 molecules, and two folds have only 6 and 10.

| model | AUROC | AUPRC | MCC | ECE | Brier |
|---|---|---|---|---|---|
| prior (always 52%) | 0.500 | 0.545 ± 0.161 | 0.000 | 0.140 ± 0.102 | 0.254 ± 0.010 |
| logistic regression | 0.834 ± 0.121 | 0.873 ± 0.113 | 0.503 ± 0.299 | 0.217 ± 0.105 | 0.173 ± 0.091 |
| random forest | **0.843 ± 0.088** | **0.891 ± 0.083** | **0.578 ± 0.240** | 0.196 ± 0.073 | **0.150 ± 0.067** |
| XGBoost | 0.836 ± 0.118 | 0.881 ± 0.100 | 0.538 ± 0.301 | **0.191 ± 0.093** | 0.162 ± 0.098 |

Fingerprints separate potent from weak TG2 inhibitors on unseen scaffolds (AUROC ≈ 0.84). Calibration is poor, though: ECE is about 0.2, worse than the constant prior, so the models are overconfident on new scaffolds. That gives a Jev-style calibrated model something concrete to improve on.

*Replaced (2026-10-26):* the repo's own Active/Inactive labels (10 µM cut) were 397/44 (90% active). Test folds had 0–15 inactives, usually 1–3, so AUROC 0.93–0.94 and MCC (sd ≈ 0.3) were not reliable. That definition is no longer used.

### Regression (pIC50, exact IC50 only)
305 molecules, pIC50 mean 5.98, sd 0.96. Test folds have 24 ± 9 molecules.

| model | RMSE | R² | Pearson | Spearman |
|---|---|---|---|---|
| mean predictor | 1.108 ± 0.232 | −0.189 ± 0.240 | n/a | n/a |
| random forest | **0.776 ± 0.163** | **0.353 ± 0.364** | **0.667 ± 0.283** | **0.644 ± 0.206** |
| XGBoost | 0.789 ± 0.225 | 0.283 ± 0.587 | 0.620 ± 0.329 | 0.608 ± 0.247 |

Fingerprints carry real signal for new scaffolds (RMSE about 0.78 log units vs 1.11 for the mean), but the variance between splits is large.

## GDSC2: 5 seeds

| split | model | RMSE | R² | Pearson | Spearman | per-drug Spearman |
|---|---|---|---|---|---|---|
| random | drug mean | 1.513 ± 0.014 | 0.690 ± 0.005 | 0.831 ± 0.003 | 0.762 ± 0.006 | n/a |
| random | XGBoost | **1.019 ± 0.012** | **0.860 ± 0.003** | **0.927 ± 0.002** | 0.897 ± 0.002 | 0.716 ± 0.009 |
| cold drug | global mean | 2.668 ± 0.529 | −0.115 ± 0.127 | n/a | n/a | n/a |
| cold drug | XGBoost | **2.351 ± 0.475** | **0.108 ± 0.286** | **0.464 ± 0.202** | 0.406 ± 0.160 | 0.646 ± 0.055 |
| cold cell | drug mean | 1.493 ± 0.048 | 0.696 ± 0.016 | 0.835 ± 0.010 | 0.771 ± 0.013 | n/a |
| cold cell | XGBoost | **1.254 ± 0.039** | **0.785 ± 0.011** | **0.887 ± 0.006** | 0.842 ± 0.007 | 0.522 ± 0.034 |

"Drug mean" predicts each test pair as that drug's average response in training. "Per-drug Spearman" is the average, over drugs, of how well cell lines are ranked within one drug.

## What this means
1. **GDSC2 random-split scores are mostly drug identity.** Knowing only the drug gives Pearson 0.83. Headline numbers like 0.93 overstate what a model learns about biology. Use **cold-drug** (new molecules, the setting that matters for drug discovery) and **per-drug Spearman** as the main metrics.
2. **New drugs are the hard problem.** Cold-drug R² is about 0.1, with high variance because there are only 137 drugs, so each test fold has about 14. This is where a better drug representation (e.g. Qwen features) has the most room to help, and also the highest risk of noise.
3. **TG2 targets to beat** (random forest, unseen scaffolds): regression **RMSE 0.78 / Spearman 0.64**, potency classification **AUROC 0.84 / MCC 0.58**, and calibration **ECE 0.20 / Brier 0.15**.
4. **The ≤ 1 µM threshold** is a modelling choice, not part of the source labels. Its units still need confirming against the paper (see `research_plan.md`). Because the labels are derived from pIC50, classification and regression are two views of the same 305 molecules, not independent evidence.
5. With 10 TG2 seeds the standard deviations are wide. Compare models **on the same seeds** (paired), not by overlapping mean ± sd.

Not done: hyperparameter tuning, other fingerprint types, and checking these numbers against the Wichka & Lai paper (which used different features, splits and labels, so it isn't directly comparable).


---

# Addendum, 2026-10-27: corrected data, expanded data, and tuned baselines

Everything above was computed on the 305-molecule PubChem-only set with the pre-audit preprocessing.
It is kept for comparison. The numbers below supersede it. See [`data.md`](data.md) for what changed
in the data and why.

Reproduce:
```bash
python scripts/run_baselines.py tg2 --seeds 10 --data combined
python scripts/run_tuned_baselines.py --seeds 10 --data combined
python scripts/compare_tuning.py
```

## Effect of each data change, TG2 potency classification, 10 scaffold seeds

| dataset | molecules | test fold | RF AUROC | RF ECE |
|---|---|---|---|---|
| PubChem, exact IC50 only (original) | 305 | 24 ± 9 | 0.843 ± 0.088 | 0.196 |
| \+ salts stripped, censored negatives recovered | 367 | 31 ± 6 | 0.815 ± 0.118 | 0.192 |
| \+ ChEMBL union (**current**) | **631** | **62 ± 1** | **0.872 ± 0.066** | 0.158 |

Reading this honestly: the middle row is *lower* than the original, and that is expected rather than
a regression. Recovering 62 censored negatives adds compounds that were measured only as "inactive at
10 µM", several of them structurally unlike the actives, which makes the task harder and the old 0.843
an overestimate obtained by quietly dropping every documented inactive.

Adding ChEMBL then raises AUROC to 0.872 **and halves the standard deviation** (±0.066 against
±0.088), because the test fold grows from 24 molecules to 62. The variance reduction matters more than
the mean: the earlier study could not resolve differences below ~0.07 AUROC, and this one can resolve
about half that.

## Tuned versus default, paired by seed

Each seed tunes on its own validation fold over a small fixed grid (`scripts/run_tuned_baselines.py`)
and reports test once.

| model | ΔAUROC | p | seeds better | ΔBrier | ΔECE |
|---|---|---|---|---|---|
| **random forest** | **+0.027** | **0.019** | 7/10 | −0.007 | +0.023 |
| XGBoost | +0.001 | 0.86 | 4/10 | −0.002 | −0.002 |
| logistic regression | +0.010 | 0.51 | 4/10 | +0.003 | +0.016 |

Tuning significantly helps the random forest and does nothing measurable for the other two.

## The bar the LLM has to clear

| model | AUROC | AUPRC | MCC | ECE | Brier |
|---|---|---|---|---|---|
| trivial prior | 0.500 | 0.569 | 0.000 | 0.155 | 0.262 |
| logistic regression | 0.863 | 0.888 | 0.620 | 0.157 | 0.135 |
| random forest (default) | 0.872 | 0.884 | 0.640 | 0.158 | 0.131 |
| XGBoost (default) | 0.880 | 0.900 | 0.639 | 0.148 | 0.128 |
| XGBoost (tuned) | 0.882 | 0.902 | 0.663 | **0.146** | 0.126 |
| **random forest (tuned)** | **0.898** | **0.910** | **0.685** | 0.180 | **0.125** |

**Tuned random forest on 2048-bit Morgan fingerprints, AUROC 0.898 ± 0.057, is the baseline any LLM
result must be compared against.** Note that it has the *worst* ECE in the table: accuracy and
calibration are not improving together, which is the gap [`calibration.md`](calibration.md) examines.
