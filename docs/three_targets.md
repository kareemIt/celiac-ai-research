# TG2, CDK2 and MAO-B: what replicates about publication-held-out evaluation

Written 2026-11-05. Every number here comes from a committed result file; the script that produces
each is named beside it. Nothing in this document is a projection.

Reproduce the analyses: `python scripts/analyse_sota.py --target {tg2,cdk2,maob} --split <split>`,
`scripts/analyse_split_penalty.py`, `scripts/analyse_three_targets.py`,
`scripts/analyse_protocol_gap.py`, `scripts/validate_novelty_share.py`,
`scripts/analyse_confounders.py`, `scripts/audit_assay_confound.py`.

## 0. The protocols are not the same, and that matters more than expected

| | TG2 | CDK2 | MAO-B |
|---|---|---|---|
| Molecules (publications with ≥10 labelled) | 444 | 2,078 | 5,091 |
| Qualifying publications | 11 | 83 | 209 |
| Publication-held-out protocol | **leave-one-publication-out** | **grouped K-fold, K=10** | **grouped K-fold, K=10** |
| Folds | 11 (1 undefined) | 10 | 10 |
| Scaffold comparison arm | `scaffold_pub`, 20 seeds | `scaffold_pub`, 10 seeds | `scaffold_pub`, 10 seeds |
| Models | 10 | 6 | 6 |

**TG2 uses a different publication protocol from the two replication targets.** This is deliberate
— its published arm is preserved — but leave-one-publication-out and grouped K-fold do not produce
comparable macro-AUROCs, for the reason in §2. Any TG2-against-replication comparison in this
document is labelled with the protocol it used, and a pooled estimate that mixes them says so.

Molecule counts also differ between arms within a target: a publication arm tests every molecule
exactly once, while the seeded scaffold arm tests a molecule up to five times and covers fewer
unique molecules (CDK2: 2,078 against 1,278). Where the two are compared, a molecule's probability
is averaged over the folds that tested it first.

## 1. The decline reproduces in direction on every model and every target

`scripts/analyse_split_penalty.py`, `scripts/analyse_three_targets.py`. Matched molecules, matched
training-set size (drift 1.1–1.2%), Welch's test because the arms are not paired.

| model | TG2 (LOPO) | CDK2 (grouped) | MAO-B (grouped) |
|---|---|---|---|
| XGBoost | −0.282 | −0.131 | −0.046 |
| Random forest | −0.242 | −0.104 | −0.057 |
| Logistic regression | −0.239 | −0.101 | −0.036 |
| CheMeleon | −0.169 | −0.084 | −0.034 |
| ChemBERTa-2 fine-tuned | −0.148 | −0.081 | −0.051 |
| ChemBERTa-2 frozen | −0.107 | −0.077 | −0.031 |
| **mean** | **−0.198** | **−0.096** | **−0.042** |

6/6 models lose AUROC on every target. Significance after Holm across models: 2 of 10 on TG2, 2 of
6 on CDK2, **0 of 6 on MAO-B**. The effect is directionally universal; it is individually
significant only on the smaller targets, and its magnitude falls steeply with dataset size.

**The penalty ranking does not replicate.** TG2 and CDK2 agree perfectly (Spearman 1.00, Kendall
exact p = 0.0028) and MAO-B refutes it (Spearman 0.543, p = 0.27 against both). An earlier
reading of the two-target agreement as a finding is withdrawn.

## 2. Most of the apparent decline is the aggregation unit and the novelty of what is held out

`scripts/analyse_protocol_gap.py`. Both publication protocols test each molecule exactly once, so
pooled molecule-level AUROC is exactly comparable between them. On CDK2:

| model | LOPO macro | grouped macro | LOPO pooled | grouped pooled |
|---|---|---|---|---|
| logreg | 0.629 | 0.782 | 0.798 | 0.787 |
| random forest | 0.617 | 0.786 | **0.806** | **0.802** |
| XGBoost | 0.629 | 0.762 | 0.784 | 0.767 |

The 0.169 macro gap is **entirely** the aggregation unit. Re-aggregating the grouped arm's own
predictions by publication reproduces the leave-one-out numbers (0.633 / 0.604 / 0.628). The other
three candidates contribute nothing: size-weighting moves the gap from +0.152 to +0.141; restricting
the pooled score to the 1,637 of 2,078 molecules in scorable folds moves the random forest from
0.806 to 0.804; and leave-one-out trains on *more* data (1,856 against 1,685), so training size
works against it.

Verified on the molecules **both** protocols tested, which matters on MAO-B: its leave-one-out arm
is a random 50-publication sample covering 1,116 molecules against the grouped arm's 5,091, and an
unrestricted pooled comparison read +0.047 purely because the two arms were scored on different
sets. Restricted to the shared 1,116 it is **−0.025** — leave-one-out marginally ahead, matching
CDK2, where both arms already covered the same 2,078 molecules.

**Which published numbers this touches.** Only comparisons that put a pooled score against a
publication-macro one. `docs/second_target.md`'s CDK2 and MAO-B penalties (0.898 → 0.830,
0.865 → 0.789) are **not** affected: `run_second_target.py` uses an 80/10/10 `group_split`, so its
paper-arm test set holds many whole publications, and `evaluate` reports a pooled
one-row-per-molecule AUROC for both arms. Those are aggregation-matched and stand. TG2's
0.854 → 0.612 is the one that is not.

**The two metrics answer different questions.** Macro-over-publications is the score on a typical
*publication* — can the model rank compounds inside one paper's congeneric series, which is what a
campaign facing one new paper sees. Pooled molecule-level AUROC is the score on a typical
*molecule* — can it rank across the screening set. The within-series question is harder because the
easy between-series contrasts are gone. Neither is the right metric in general; comparing one
against the other is wrong, and TG2's headline 0.854 → 0.612 does exactly that.

### Novelty, as an association with wide intervals

`scripts/validate_novelty_share.py`. A publication split shifts the test set towards novel
chemistry, because a paper's whole series leaves at once.

| | mean max-Tanimoto shift | below 0.4 Tanimoto: publication arm vs scaffold arm | smallest shared band |
|---|---|---|---|
| TG2 | −0.331 | 64% vs **1.6%** | **7 molecules** |
| CDK2 | −0.243 | 54% vs 4.9% | 63 |
| MAO-B | −0.137 | 19% vs 6.4% | 207 |

Standardising to either arm's novelty composition removes the majority of the penalty on the two
replication targets. Publication-clustered bootstrap, 400 draws, published five bands:

| | raw penalty | residual after novelty | share |
|---|---|---|---|
| CDK2 random forest | −0.127 [−0.203, −0.068] | **−0.022 [−0.091, +0.038]** | 78% [33%, 98%] |
| CDK2 XGBoost | −0.166 [−0.248, −0.095] | −0.042 [−0.120, +0.025] | 73% [27%, 98%] |
| MAO-B random forest | −0.076 [−0.110, −0.046] | −0.030 [−0.077, +0.007] | 60% [28%, 98%] |
| MAO-B logreg | −0.067 [−0.097, −0.037] | −0.020 [−0.055, +0.011] | 71% [28%, 99%] |

The residual crosses zero for every model on both targets. Band definition is not the fragile part
(47–98% on CDK2, 36–81% on MAO-B across five schemes). **Common support is**, and TG2 does not have
it: only 33% of its publication arm and 47% of its scaffold arm sit in bands both arms can score,
against 100% for CDK2 and MAO-B, and its estimates run −992% to +51%.

**This is an association, not a causal decomposition.** It adjusts for measured similarity, not for
everything that differs between arms; molecules are not randomised; band standardisation removes
only between-band novelty, making it a lower bound on the novelty association; and it is undefined
without common support.

### Assay cannot be separated from publication identity

`scripts/audit_assay_confound.py`. **0 of 380 CDK2 and 0 of 516 MAO-B assays span more than one
publication.** ChEMBL defines an assay per paper, so an assay-held-out split *is* a
publication-held-out split. The within-publication contrast that could estimate an assay effect
with the laboratory fixed exists for 1 CDK2 publication (60 molecules) and 2 MAO-B publications
(109 molecules). Share of test molecules whose assay never appears in training: 100% under both
publication protocols on both targets, against 0.3% (CDK2) and 9.4% (MAO-B) under a scaffold split.

Any residual decline is therefore **publication identity and its measurement protocol, jointly and
inseparably**. No claim in this document assigns it to publication identity.

## 3. Under matched aggregation and matched publications, the targets disagree

`scripts/analyse_protocol_gap.py`, `matched_within_publication`. Each publication's own molecules
scored in both arms, restricted to the publications both arms can score, paired across publications.
This removes both confounds at once and asks only: can a model rank compounds within a paper's
series better having seen part of that series than not?

A publication arm tests a publication's whole set; the scaffold arm tests only the part whose
scaffold group landed on the test side. So the default comparison still scores different molecules
(CDK2: 810 against 1,052, the scaffold arm a strict subset). The `same mols` variant restricts both
arms to the molecules they both tested and keeps a publication only if that leaves the two sets
identical. Intervals are paired-t; a publication-clustered bootstrap (2,000 draws) agrees
throughout, so the parametric interval is adequate here.

| target | all molecules | same molecules | intervals excluding zero (same mols) | publications |
|---|---|---|---|---|
| CDK2 | −0.043 to −0.093 | **−0.034 to −0.110** | 2 of 6 (rf −0.110 [−0.192, −0.044], xgb −0.110 [−0.203, −0.023]) | 37 |
| MAO-B | +0.009 to +0.060 | **+0.007 to +0.062** | 3 of 6 (xgb +0.062 [+0.019, +0.104], logreg +0.051, ChemBERTa-ft +0.055) | 97 |
| TG2 (LOPO) | −0.096 to +0.073 | −0.097 to +0.061 | 0 of 10 | 10 |

The sign conflict **survives** the molecule-set restriction and the clustered bootstrap. XGBoost is
−0.110 on CDK2 and +0.062 on MAO-B, both significant — the same model, opposite directions.

CDK2 and MAO-B disagree in **sign**, both with intervals excluding zero for half their models. So
"holding out a publication reduces performance" is not supported as a general claim once the
aggregation unit is matched. TG2's 10 publications settle nothing either way.

## 4. One model-level finding replicates, and it is narrow

`scripts/analyse_three_targets.py`. Each entrant against the tuned random forest, paired per fold
within a target, combined across targets by inverse-variance weighting with Cochran's Q reported.

| arm | model | pooled Δ vs RF | 95% CI | p | p_holm | Q (df=2) |
|---|---|---|---|---|---|---|
| publication | **CheMeleon** | **+0.031** | [+0.007, +0.055] | 0.011 | **0.057** | **0.09** |
| publication | ChemBERTa-2 frozen | −0.020 | [−0.038, −0.003] | 0.024 | 0.096 | 3.55 |
| publication | XGBoost | +0.003 | [−0.009, +0.014] | 0.640 | 1.000 | 2.35 |
| publication | logreg | +0.002 | [−0.009, +0.013] | 0.719 | 1.000 | 0.24 |
| publication | ChemBERTa-2 fine-tuned | −0.002 | [−0.021, +0.017] | 0.841 | 1.000 | 1.31 |
| scaffold | CheMeleon | +0.005 | [−0.001, +0.012] | 0.116 | 0.173 | 9.71 |
| scaffold | ChemBERTa-2 frozen | −0.053 | [−0.066, −0.039] | <0.0001 | **<0.0001** | 1.46 |
| scaffold | ChemBERTa-2 fine-tuned | −0.019 | [−0.029, −0.010] | 0.0001 | **0.0003** | 4.29 |

CheMeleon beats the random forest on the publication arm of all three targets (+0.023, +0.023,
+0.033) with Q = 0.09 on 2 df — about as little heterogeneity as three targets can show — and the
same contrast on the matched scaffold arm is null. That is the shape the foundation-model
hypothesis predicts. Two reasons it is not confirmed: **p_holm = 0.057** across the five entrants,
and it is CheMeleon alone — ChemBERTa-2 fine-tuned pools to −0.002 and frozen to −0.020, so
pretrained models as a *class* do not generalize better. The pooled estimate also mixes TG2's
leave-one-out protocol with the replications' grouped K-fold.

**No ranking inversion.** Under publication hold-out on CDK2 nothing separates from the random
forest (every p_holm = 1.000); on MAO-B only CheMeleon comes close (p_holm = 0.200). The rankings
do not invert, they stop being distinguishable.

## 5. Controls that explain nothing

`scripts/analyse_confounders.py`, `scripts/audit_fold_sensitivity.py`.

- **Training-set size**: matched by construction, drift 1.1–1.2%. A design check that passed.
- **Class balance**: mean positive rate identical between arms by construction (every molecule
  tested once). Its spread is 1.8× larger in the publication arm, but restricting to the shared
  positive-rate band or adjusting with a quadratic leaves the penalty unchanged (−0.128 raw,
  −0.137 banded, −0.136 adjusted; CDK2 logreg).
- **Fold assignment**: over five deals, between-deal SD is 9–20× smaller than within-deal SD on
  both targets and both arms. The matched penalty is stable (CDK2 logreg −0.128, range −0.134 to
  −0.113). One deal is enough.
- **Publication size** and **single-class fold exclusion**: §2, both ~0.01 or less.

## 6. Conformal uncertainty

Measured on all three targets (`scripts/analyse_lopo_conformal.py --target {tg2,cdk2,maob}`).
CDK2 and MAO-B ran on `paper_grouped`, 10 folds each, 2,078 and 5,091 molecules per model.

### Publication-wise calibration restores marginal coverage, on every target

The i.i.d. regime draws calibration molecules at random from the training publications; the group
regime holds out whole training publications. At a 0.90 target, pooled over molecules:

| target | iid coverage | group coverage | paired group - iid | p |
|---|---|---|---|---|
| TG2 (`paper_lopo`) | 0.755 | 0.868 | +0.113 [+0.078, +0.148] | 2.5e-08 |
| CDK2 (`paper_grouped`) | 0.790-0.831 | 0.885-0.905 | +0.084 [+0.065, +0.102] | 1.6e-12 |
| MAO-B (`paper_grouped`) | 0.806-0.826 | 0.901-0.920 | +0.090 [+0.079, +0.102] | 1.6e-22 |

Ranges are across the six entrants. Pairing is per fold and model, so the interval is over 60
pairs on the replication targets and 66 on TG2; the sign test agrees (55/60, 60/60, 52/66).
**This is the single most robust finding in the whole analysis** -- same direction, similar
magnitude, and significant on three targets under two different publication protocols.

Exchangeability is what breaks, and holding out whole publications at calibration time is enough
to repair the marginal guarantee. At alpha=0.2 the i.i.d. regime does not merely under-cover, it
degenerates: it starts emitting *empty* prediction sets for 4.3-44.7% of CDK2 molecules and
2.1-13.5% of MAO-B molecules, so its nominal 0.80 coverage comes out as low as 0.446. Group
calibration emits none, on either target, at either alpha.

Mondrian (class-wise) quantiles change little -- at most 0.01 in coverage on either target, in
either regime. The shift that matters here is in the molecules, not in the class balance.

### Restored coverage is not the same as a useful answer

A prediction set of {inactive, active} is covered and says nothing. At alpha=0.1 under group
calibration:

| target | singleton fraction | error among singletons |
|---|---|---|
| TG2 | 0.15-0.32 | 0.25-0.39 |
| CDK2 | 0.41-0.48 | 0.22-0.27 |
| MAO-B | 0.46-0.56 | 0.16-0.18 |

So the replication targets are *less* bad than TG2 on both axes, but the qualitative conclusion
holds everywhere: roughly half the molecules get a decisive answer and between a sixth and a
quarter of those answers are wrong. Conformal prediction under a publication shift is honest
about its own ignorance rather than useful.

### Conditional coverage still fails, in the expected direction

Marginal coverage is restored; coverage *within* a novelty band is not. Random forest at
alpha=0.1, group calibration, by max Tanimoto to the fitting set:

| band | CDK2 | MAO-B |
|---|---|---|
| [0.0, 0.4) | 0.898 (n=1461) | 0.899 (n=1383) |
| [0.4, 0.5) | 0.874 (n=318) | 0.909 (n=1255) |
| [0.5, 0.6) | 0.874 (n=111) | 0.894 (n=1202) |
| [0.6, 0.7) | 0.940 (n=83) | 0.894 (n=700) |
| [0.7, 1.01) | 0.933 (n=105) | 0.920 (n=551) |

The spread is milder than TG2's but has the same shape: the unfamiliar bands sit at or below
target and the most familiar band overshoots, and the set sizes do the work -- CDK2 random forest
averages 1.652 in the lowest band against 1.048 in the highest. ChemBERTa-2 (frozen) is the worst
conditional offender on both targets (0.876 and 0.872 in the lowest band). A per-molecule
guarantee is not available from this procedure; only a population-level one is.

### Selective prediction separates the two targets sharply

Error at a 0.5 threshold among the most confident fraction kept (group calibration, random
forest):

| fraction kept | CDK2 | MAO-B |
|---|---|---|
| 100% | 0.309 | 0.279 |
| 60% | 0.256 | 0.187 |
| 20% | 0.215 | 0.077 |

On MAO-B, confidence ranking works: discarding the least confident 80% of molecules cuts decision
error by a factor of 3.6. On CDK2 it barely works at all -- 0.309 to 0.215 over the same range.
This is the same target asymmetry section 3 found under matched aggregation, arriving by a
completely independent route, and it is the practically useful form of it: on MAO-B a screening
campaign can triage on model confidence, and on CDK2 it cannot.

**Answering the question directly:** publication-aware calibration is a real and reliably
replicating improvement to *coverage*, and it is the only form of conformal prediction that should
be offered in this regime. It is not a fix for the underlying difficulty: it buys an honest
interval, not a usable prediction, and on CDK2 the confidence ordering it rests on is close to
uninformative.

## 7. Corrections this analysis forced

| claim | where | status |
|---|---|---|
| "Publication-level confounding" as the attribution of the decline | `paper.md` abstract, §5.9, two tables | **Corrected** in `paper.md` §5.14 — most of it is aggregation and novelty; the remainder is confounded with assay |
| Leave-one-publication-out folds overlap, "1.6× on CDK2" | `replication_feasibility.md` §2 | **Withdrawn** — measured mean 1.00, max 1 on all three targets. The 1.6× belongs to the seeded scaffold arm |
| An assay-held-out control is testable | `replication_feasibility.md` §6 | **Withdrawn** — assays are perfectly nested in publications |
| Novelty share "51–78%" | Stage 2 commit message | **Superseded** — a majority with wide intervals, roughly 30–95%, and not estimable on TG2 |
| Fold-assignment variance "not estimated" | `replication_feasibility.md` §7 | **Resolved** — bounded at 9–20× smaller than within-deal spread |
| MAO-B has 586 publications | `second_target.md`, `paper.md` | **Corrected** to 414 (first-attributed documents) |
| LOPO protocol check costs "under 4 h" | `replication_feasibility.md` §2 | **Corrected** to 3.4 h (CDK2) and 20.8 h (MAO-B) |

## 8. What is still uncertain

- Whether the publication-held-out decline, once aggregation and novelty are accounted for, is
  anything at all. CDK2 says −0.07, MAO-B says +0.04, both with half their models' intervals
  excluding zero. This is the central open question.
- Whether novelty explains the TG2 decline. Not estimable there for lack of common support, and
  the direction of its novelty shift is the largest of the three.
- Whether CheMeleon's publication-arm advantage is pretraining or the D-MPNN architecture.
  TG2's existing Chemprop run says pretraining (CheMeleon − Chemprop = +0.100 [+0.054, +0.146]
  on scaffolds), but a from-scratch D-MPNN on 359 training molecules is data-starved, so TG2
  handicaps the control. Unresolved on the targets where it could be answered fairly.
- Whether any of this generalizes past ChEMBL. All three targets come from one curation pipeline,
  and ChEMBL's one-assay-per-paper structure is what makes publication and assay inseparable.
