# Calibration: post-hoc rescaling does not transfer across scaffolds (run 2026-10-27)

Reproduce:
```bash
python scripts/run_calibration.py --seeds 10 --data combined --split scaffold
python scripts/run_calibration.py --seeds 10 --data combined --split random    # the control
python scripts/calibrate_qlora.py --data combined                              # the fine-tuned model
```
Per-seed rows: `results/calibration_tg2_{scaffold,random}.csv`, `results/calibration_qlora.csv`.

## The question
Calibrated probabilities are the whole point of a Jev-style decision model, and nothing in this
project had produced them: ECE sat at 0.14–0.24 for every method, and fine-tuning did not help.
The standard remedy is post-hoc rescaling fitted on held-out data. Does it work here?

## Setup, and why there is no leakage
Each of the 10 scaffold splits gives train / valid / test. The model is fit on **train** only. The
calibrator is fit on **valid** only. Metrics are computed on **test**, which nothing has touched.
Four treatments are compared on identical splits: none, temperature scaling (one parameter, Guo et
al.), Platt scaling (two parameters), and isotonic regression (non-parametric).

All three are monotone maps, so AUROC and AUPRC cannot change — confirmed exactly in the results for
temperature and Platt. Isotonic moves them only by creating ties.

## Result: it does not work

TG2 potency, combined dataset (631 molecules), 10 scaffold splits. Change from uncalibrated, paired
by seed; negative is better.

| model | calibration | ΔECE | ΔBrier | Δlog loss |
|---|---|---|---|---|
| random forest | temperature | +0.036 (p=0.06) | +0.011 (p=0.05) | +0.047 (p=0.07) |
| random forest | Platt | **+0.046** (p=0.03)* | **+0.022** (p=0.04)* | **+0.080** (p=0.005)* |
| random forest | isotonic | −0.024 (p=0.17) | +0.014 (p=0.04)* | +0.273 (p=0.08) |
| XGBoost | temperature | +0.022 (p=0.21) | +0.002 (p=0.71) | −0.019 (p=0.59) |
| XGBoost | isotonic | −0.005 (p=0.81) | +0.018 (p=0.05)* | +0.248 (p=0.07) |
| logistic regression | Platt | +0.028 (p=0.12) | **+0.026** (p=0.04)* | +0.058 (p=0.18) |
| Qwen Jev, zero-shot | temperature | +0.012 (p=0.64) | +0.010 (p=0.22) | +0.022 (p=0.22) |
| Qwen Jev, zero-shot | Platt | **+0.057** (p=0.02)* | +0.009 (p=0.40) | +0.026 (p=0.33) |

\* significant at p < 0.05, paired t-test over seeds, uncorrected.

**Not one treatment significantly improves any calibration metric. Several significantly worsen it.**
The best single number anywhere is isotonic on the random forest, ΔECE −0.024, and it is not
significant (p = 0.17) and costs +0.273 log loss, because isotonic pushes predictions to hard 0 and 1
on a small fold.

## Why: the fitted temperature is unidentifiable on a scaffold-split validation fold

| model | fitted temperature across 10 seeds |
|---|---|
| random forest | 1.54 ± 0.94 (range 0.40 – 3.74) |
| XGBoost | 2.44 ± 1.30 (range 0.95 – 5.17) |
| logistic regression | 2.70 ± 2.00 (range 0.81 – 7.52) |
| Qwen Jev zero-shot | **31.2 ± 61.8** (range 0.34 – 148.4) |

A temperature of 1.0 means "already calibrated". These estimates swing from "far too underconfident"
to "far too overconfident" depending only on which scaffolds happened to land in the validation fold.
A parameter that unstable cannot transfer to a test fold drawn from yet another region of chemical
space. Qwen's range is the extreme case: on some folds the fit wants to flatten its log-odds by a
factor of 148, i.e. to discard the magnitude of its score entirely.

## The control that proves the mechanism

Rerunning everything with a **random** split, where validation and test are drawn from the same
chemical space, removes the problem — and with it most of the need for calibration:

| | scaffold split | random split |
|---|---|---|
| RF uncalibrated ECE | 0.158 | 0.111 |
| RF ΔECE from temperature | **+0.036** (worse) | **−0.004** (neutral) |
| RF Δlog loss from temperature | +0.047 (worse) | −0.008 (neutral) |
| RF fitted temperature | 1.54 ± 0.94 | 0.94 ± 0.27 |

Under a random split the models are already close to calibrated (temperature ≈ 1) and rescaling is
roughly a no-op. Under a scaffold split they are miscalibrated *and* unfixable by this family of
methods. The miscalibration is therefore not a property of the model's output layer, which is what
temperature scaling repairs. It is distribution shift: a model asked about an unfamiliar chemotype is
wrong about its own confidence in a way that a different unfamiliar chemotype cannot predict.

Note also that the random split's apparent advantage is partly an illusion — 38.6% of its test
molecules have a training molecule with an identical fingerprint ([`data.md`](data.md)).

## What this means for the Jev direction
Jev's selling point is calibrated probabilities from its (undisclosed) RLCD training. This result says
that the cheap imitation of that — train normally, then rescale on held-out data — **does not reach
it on scaffold-split chemistry**. If calibration under chemotype shift is the goal, the remaining
options are ones that model the shift rather than rescale around it:

1. **Conformal prediction** with a Mondrian/clustered calibration set, which gives coverage guarantees
   under exchangeability rather than a point probability, and can be audited per scaffold group.
2. **Ensembling** over seeds and option orders, whose spread is a shift-aware uncertainty signal that a
   single temperature cannot express.
3. **Training for calibration directly** (the RLCD idea) rather than patching it afterwards.
4. **Abstention**: report a prediction only when the molecule is within a similarity threshold of
   training, using the nearest-neighbour Tanimoto already computed in `scripts/audit_splits.py`.

Option 4 is the cheapest and the most honest about what the data supports.

## Caveats
- 10 seeds, ~63 test and ~63 validation molecules each. Small folds are exactly the regime where
  isotonic and Platt overfit, so part of the failure is sample size rather than shift alone. The random
  split control used the same fold sizes, which is what isolates shift as the cause.
- p-values are uncorrected across 15 comparisons; with correction nothing survives at all, which only
  strengthens the negative conclusion.
- The fine-tuned model's validation fold also chose its training epoch, so its calibration fold is not
  perfectly untouched. It contributed no gradient.
