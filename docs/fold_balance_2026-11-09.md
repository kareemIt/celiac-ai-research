# Experiment 3: does validation-fold class balance explain test-score variance? No (run 2026-11-09)

Reproduce: `python scripts/analyse_fold_balance.py`.
Outputs: `results/fold_balance.csv` (per model × arm × seed), `results/fold_balance_slopes.csv`,
`results/fold_balance_confound.csv`, `results/fold_balance_variance.csv`.
Tests: `tests/test_fold_balance.py`. Cost: seconds of CPU, no GPU, no retraining.
Plan and pre-registration: [`review_2026-11-09.md`](review_2026-11-09.md) §5, experiment 3.

**This is a negative result and it is reported as one.** The hypothesis is not supported, the one
place an association does appear has the wrong sign and is confounded, and a simpler explanation
accounts for the observation that prompted the experiment. The useful outcome is that a plausible
confound has been removed from the existing model comparison rather than left hanging.

---

## 1. The observation that prompted it

Scaffold splitting keeps whole analog series together, so the validation fold's class balance is
whatever the shuffle gives. Measured from `results/splits_tg2.csv`:

| arm | seeds | validation positive fraction |
|---|---|---|
| TG2 `scaffold` | 20 | **0.085** to 0.730 (median 0.504) |
| TG2 `scaffold_pub` | 20 | 0.182 to 0.846 (median 0.533) |
| CDK2 `scaffold_pub` | 10 | 0.420 to 0.781 (median 0.640) |
| MAO-B `scaffold_pub` | 10 | 0.277 to 0.476 (median 0.341) |

TG2 seed 0's validation fold is 59 molecules, 5 of them positive, in a dataset that is 47.9%
positive. Every entrant in the sweep selects its configuration on that fold, so the worry was that
an extreme fold means selecting on an unreliable signal — and `sota_comparison.md` §2 does record
that seeds 10–19 are harder for every model and that Chemprop's spread (± 0.123) is far wider than
the random forest's (± 0.077).

## 2. The design, and the deviation that forced a rewrite

The plan in the review assumed the fingerprint models ignore the validation fold, giving a clean
control group. **They do not.** `scripts/run_sota.py:fit_sklearn` tunes the random forest, logistic
regression and XGBoost on validation AUROC over 12, 4 and 8 configurations; every entrant in the
sweep uses between 3 and 12. There is no model here that ignores the fold.

The control group was therefore redefined by *what the fold decides*, before any fit:

| group | models | the validation fold decides |
|---|---|---|
| `uses_valid_for_training` | Chemprop, Chemprop + RDKit 2D, CheMeleon, ChemBERTa-2 fine-tuned | how much training happens — `EarlyStopping` and `ModelCheckpoint` on `val_loss` inside *every* configuration, and 3 epoch budgets selected on validation AUROC |
| `control` | random forest, logistic regression, XGBoost, tabular foundation model, ChemBERTa-2 frozen, MoLFormer-XL frozen | only hyperparameters of a converged fit |

Pre-registered: slope of test AUROC on |validation positive fraction − 0.5| **negative** for the
first group, **null** for the control. An effect in both means something other than selection. The
reporting rule, also fixed in advance, was that p > 0.05 is a null and not a trend, and that a
positive result must replicate on CDK2 and MAO-B before being written up.

## 3. Result: null on the primary target

TG2 `scaffold`, 20 seeds, one OLS slope per model:

| model | group | slope | 95% CI | r² | p |
|---|---|---|---|---|---|
| Tabular foundation model | control | −0.020 | [−0.428, +0.388] | 0.001 | 0.92 |
| Logistic regression | control | −0.007 | [−0.408, +0.394] | 0.000 | 0.97 |
| ChemBERTa-2 fine-tuned | uses valid | +0.078 | [−0.376, +0.533] | 0.006 | 0.74 |
| Chemprop | uses valid | +0.088 | [−0.451, +0.626] | 0.006 | 0.75 |
| XGBoost | control | +0.095 | [−0.269, +0.458] | 0.014 | 0.62 |
| Random forest | control | +0.099 | [−0.238, +0.437] | 0.018 | 0.57 |
| Chemprop + RDKit 2D | uses valid | +0.102 | [−0.354, +0.558] | 0.011 | 0.67 |
| ChemBERTa-2 frozen | control | +0.146 | [−0.278, +0.569] | 0.025 | 0.51 |
| CheMeleon | uses valid | +0.154 | [−0.271, +0.579] | 0.027 | 0.49 |
| MoLFormer-XL frozen | control | +0.239 | [−0.236, +0.714] | 0.051 | 0.34 |

**Every confidence interval spans zero, on both TG2 arms. r² never exceeds 0.051** — validation-fold
imbalance accounts for at most 5% of per-seed AUROC variance, and the point estimates are mostly
*positive*, the opposite of the pre-registered direction. The group contrast is absent too: across
all 32 model × arm slopes, 7 of 20 control slopes and 4 of 12 `uses_valid_for_training` slopes are
negative, with mean slopes of **+0.120 and +0.046** — if anything the control group moves more.

The sign is not even stable within one target: on TG2 `scaffold` the `uses_valid_for_training`
slopes are all positive, on TG2 `scaffold_pub` all three Chemprop-family and ChemBERTa-2 slopes turn
negative (−0.172 to −0.238), still with CIs spanning zero. That is what noise looks like.

## 4. The one significant set has the wrong sign and is confounded

MAO-B `scaffold_pub` produces three slopes with p < 0.05 — logistic regression +0.493 (p = 0.003),
CheMeleon +0.377 (p = 0.018), XGBoost +0.375 (p = 0.016) — all **positive**, i.e. a *more*
imbalanced validation fold going with a *higher* test AUROC, and spread across both groups. Two
reasons not to read anything into it:

- MAO-B's validation positive fraction runs 0.277 to 0.476 and **never crosses 0.5**, so on this arm
  |pos − 0.5| is just a relabelling of prevalence. The slope says "lower validation prevalence, higher
  test AUROC", which is a statement about class balance in general, not about selection.
- Validation imbalance is entangled with fold size here: r = 0.583 with the training-set size
  (p = 0.077) on 10 seeds. On CDK2 it is entangled with test composition instead, r = −0.647 with
  test positive fraction (p = 0.043). Under whole-group splitting the folds are not independent of
  each other, so a slope on one fold's composition is not evidence about that fold.

Full confound table in `results/fold_balance_confound.csv`.

## 5. What actually explains the spread: the models' own skill

The observation in §1 holds — Chemprop varies more than the random forest — but the explanation is
simpler than selection. Correlating each model's mean AUROC against its SD across seeds:

| target | arm | models | r (mean vs SD) | p |
|---|---|---|---|---|
| TG2 | `scaffold` | 10 | **−0.912** | 0.0003 |
| TG2 | `scaffold_pub` | 10 | **−0.841** | 0.002 |
| MAO-B | `scaffold_pub` | 6 | **−0.888** | 0.018 |
| CDK2 | `scaffold_pub` | 6 | −0.322 | 0.53 |

On TG2 `scaffold` the ordering is almost monotone: random forest 0.864 ± 0.077, XGBoost
0.853 ± 0.083, logistic regression 0.849 ± 0.091, ChemBERTa-2 fine-tuned 0.831 ± 0.104, CheMeleon
0.802 ± 0.098, MoLFormer frozen 0.760 ± 0.111, Chemprop 0.749 ± 0.123. **Weaker models vary more,
whatever they select on** — and the group-level SDs bear this out, with the control group's
0.098 against 0.110 on TG2 `scaffold` driven by MoLFormer frozen (0.111, a control model) as much as
by Chemprop.

This is a correlation over 6–10 models, not a mechanism, and it does not reach significance on CDK2.
It is reported because it is the obvious competing explanation and it fits the data far better than
the hypothesis did.

## 6. Consequences

- **The hypothesis is withdrawn.** Validation-fold class balance is not a mechanism to add to the
  paper's failure catalogue. On the evidence here it explains almost none of the per-seed variance.
- **A confound is cleared from the existing comparison.** The ten-entrant ranking in
  `sota_comparison.md` cannot be dismissed on the grounds that some models were tuned on degenerate
  validation folds — the fold's balance does not predict the test score. That is worth having
  stated, since the TG2 seed-0 fold at 8.5% positive is the kind of detail a reviewer would raise.
- **One genuine protocol note stands**, independent of the null: stratifying validation folds by
  class costs nothing and removes the question. It is not justified by a measured benefit here, so
  it should be described as hygiene rather than as a fix, and it would change committed numbers if
  applied, so it is not being applied retroactively.

## 7. Limits

20 seeds on one target, 10 on each replication target, and 6–10 models per arm — small for a slope
and smaller for a correlation over models. A null here is "no detectable effect at this power", not
"no effect"; an r² of 0.05 at n = 20 cannot be distinguished from zero. The groups were defined by
reading the training code, not by instrumenting it, so a model could depend on the validation fold
in a way not captured by the two-way split. CDK2 and MAO-B contribute only `scaffold_pub`, and
MAO-B's validation prevalence never crosses 0.5, so neither is a full replication of the TG2 design.
And the alternative explanation in §5 was found after looking at these data, so it is a hypothesis
for a future test, not a result.
