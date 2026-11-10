# Where the errors live: negative examples, stricter cliffs, and the missing-class problem

Kareem Itani · 2026-11-09 (third pass) · reproduce with `python scripts/analyse_error_sources.py`,
`python scripts/analyse_cliffs.py`, `python scripts/analyse_per_class.py`.
All three run on committed predictions: **no model was retrained and no GPU was used.** The
overfitting measurement and the model-improvement run, which do need training, are proposed but not
started in [`proposal_overfitting_2026-11-09.md`](proposal_overfitting_2026-11-09.md).
Prior write-ups are unaltered: [`per_class_2026-11-09.md`](per_class_2026-11-09.md),
[`cliffs_2026-11-09.md`](cliffs_2026-11-09.md),
[`validation_2026-11-09.md`](validation_2026-11-09.md).

**One mechanism accounts for most of what §5.17 and §5.18 described separately: the models predict
the class prior of the surrounding chemistry rather than the molecule.** It is measured three
different ways below, it replicates on all three targets, and it reframes the class asymmetry as a
symptom rather than a finding.

---

## 1. Negative examples: errors are concentrated, and the concentration is explainable

### 1.1 The same molecules fail for every model

Per molecule, counting how many of the (model × seed) scorings misclassify it at the 0.5 cut. The
null is that each model errs independently at the observed marginal rate:

| target | arm | models | molecules where ≥ 9/10 (or ≥ 5/6) models are wrong | null | molecules no model gets wrong | null |
|---|---|---|---|---|---|---|
| TG2 | `scaffold_pub` | 10 | **8.2%** | 0.0% | 51.2% | 8.9% |
| TG2 | `paper_lopo` | 10 | **8.1%** | 0.1% | 20.5% | 0.9% |
| CDK2 | `scaffold_pub` | 6 | **11.7%** | 0.1% | 68.1% | 30.2% |
| CDK2 | `paper_grouped` | 6 | **15.6%** | 0.8% | 47.2% | 14.2% |
| MAO-B | `scaffold_pub` | 6 | **10.7%** | 0.2% | 61.4% | 26.5% |
| MAO-B | `paper_grouped` | 6 | **14.7%** | 0.6% | 49.6% | 15.6% |

Difficulty is a property of the molecule, not of the model. Roughly one molecule in ten defeats
essentially every entrant, where independent errors predict almost none, and half to two-thirds are
got right by all of them. This is why the ten-entrant comparison in §5.10 is so tightly bunched:
the models disagree far less than their architectures suggest.

### 1.2 Errors concentrate in particular publications

Per-publication error rate, against a permutation null that reassigns molecules to publications at
random while holding publication sizes fixed — the null "the source carries no information about
error":

| target | arm | publications | observed spread | null mean | null 95th pct | p |
|---|---|---|---|---|---|---|
| TG2 | `scaffold_pub` | 11 | 0.112 | 0.048 | 0.066 | **< 0.001** |
| TG2 | `paper_lopo` | 11 | 0.139 | 0.044 | 0.060 | **< 0.001** |
| CDK2 | `paper_grouped` | 83 | 0.224 | 0.068 | 0.077 | **< 0.001** |
| MAO-B | `paper_grouped` | 209 | 0.208 | 0.069 | 0.074 | **< 0.001** |

Observed spread is two to three times the null on every target and arm, and the worst fifth of
publications carries **44% to 60% of all error mass** while holding 24% to 35% of the molecules.
This is the one result in this document that is unambiguously significant.

**On assay type.** It cannot be separated from publication. `results/confounders.csv` records that
no assay in either replication dataset spans two publications (0 of 380, 0 of 516), which §5.14
already reports as a structural non-identifiability. A per-assay analysis would be this analysis
with a different column name, so none was run.

### 1.3 The mechanism: a publication's class balance decides which error it attracts

The TG2 table makes it visible. Publications ranked by error rate, `scaffold_pub`:

| publication | molecules | actives | inactives | FN rate | FP rate |
|---|---|---|---|---|---|
| CHEMBL4138060 | 10 | 2 | 8 | **1.000** | 0.488 |
| CHEMBL1770135 | 51 | 8 | 43 | **0.981** | 0.060 |
| CHEMBL6078638 | 19 | 2 | 17 | **0.975** | 0.036 |
| CHEMBL3886499 | 35 | 22 | 13 | 0.156 | **0.691** |
| CHEMBL2150866 | 38 | 26 | 12 | 0.015 | **0.606** |
| CHEMBL5729393 | 60 | 51 | 9 | 0.037 | **0.476** |

Inactive-dominated papers have their few actives missed; active-dominated papers have their few
inactives called active. Tested as a rank correlation between a publication's active fraction and
each error rate:

| target | arm | FN rate vs active fraction | FP rate vs active fraction |
|---|---|---|---|
| TG2 | `scaffold_pub` | −0.633 (p = 0.067, 9 pubs) | **+0.733** (p = 0.025) |
| CDK2 | `scaffold_pub` | **−0.657** (p < 0.0001, 46) | **+0.757** (p < 0.0001, 42) |
| MAO-B | `scaffold_pub` | **−0.607** (p < 0.0001, 74) | **+0.702** (p < 0.0001, 133) |
| CDK2 | `paper_grouped` | **−0.524** (p < 0.0001, 64) | **+0.374** (p = 0.010, 47) |
| MAO-B | `paper_grouped` | **−0.358** (p = 0.0001, 120) | **+0.514** (p < 0.0001, 169) |

Every FN correlation is negative and every FP correlation positive, on three targets and two
protocols, with eight of ten significant at p < 0.05 and six at p < 0.0001. Under leave-one-
publication-out the correlations weaken (MAO-B FN −0.144, p = 0.47) which is expected: the held-out
publication's own balance is exactly what the model has not seen.

**The models are predicting the local class prior.** Not a universal prior — a *local* one, inherited
from whichever training molecules are nearest.

### 1.4 The same mechanism explains the sensitivity collapse

Error rate by maximum Tanimoto to the training fold, split by error type:

| target | arm | band | FN rate | FP rate |
|---|---|---|---|---|
| TG2 | `paper_lopo` | < 0.4 (283 mol) | **0.593** | 0.190 |
| TG2 | `paper_lopo` | ≥ 0.7 (13) | 0.283 | **0.943** |
| CDK2 | `paper_grouped` | < 0.4 (1,122) | **0.465** | 0.272 |
| CDK2 | `paper_grouped` | ≥ 0.7 (265) | 0.030 | **0.287** |
| MAO-B | `paper_lopo` | < 0.4 (300) | **0.879** | 0.055 |
| MAO-B | `paper_lopo` | ≥ 0.7 | 0.268 | 0.129 |

**Unfamiliar molecules are called inactive; familiar ones are called active.** On MAO-B under
publication shift, 88% of the actives below 0.4 Tanimoto are missed while only 5.5% of the
inactives are false alarms. This is nearest-neighbour behaviour, and it is the mechanism underneath
§5.17's sensitivity collapse: publication shift moves test molecules away from the training set, the
local prior goes to "inactive", and sensitivity follows.

It also explains why §5.17's asymmetry differs by target without needing a target-specific story.
The direction depends only on whether the shifted arm's molecules land far from training (actives
missed) or whether the arm retains familiar chemistry (inactives flagged).

### 1.5 Distance to the decision boundary, and what it says about the task

Error rate against |pIC50 − 6|, the distance to the 1 µM cut:

| target | arm | < 0.25 log | 0.25–0.5 | 0.5–1.0 | 1.0–2.0 |
|---|---|---|---|---|---|
| TG2 | `scaffold_pub` | 0.395 | 0.285 | 0.188 | 0.157 |
| CDK2 | `scaffold_pub` | 0.376 | 0.335 | 0.239 | 0.131 |
| MAO-B | `scaffold_pub` | 0.456 | 0.361 | 0.297 | 0.176 |
| TG2 | `paper_lopo` | 0.421 | 0.375 | 0.357 | **0.367** |

Within distribution the relationship is monotone on all three targets: a molecule within a factor of
~1.8 of the threshold is two to three times more likely to be misclassified than one a log unit
away. Those errors are largely an artefact of binarising a continuous measurement, and they are the
molecule-level counterpart of §5.18's near-threshold pairs.

**Under publication shift on TG2 the relationship flattens entirely** (0.421 / 0.375 / 0.357 /
0.367). That contrast is the useful part: within-distribution error is concentrated at the boundary,
where it is partly unavoidable; shifted error is spread across the whole potency range, where it is
not. Treating the two as one number, which AUROC does, hides the difference.

---

## 2. Activity cliffs under stricter similarity — the result survives

The §5.18 finding used MoleculeACE's rule, in which the generic-scaffold test admitted nearly every
pair (1,840 on TG2 against 23 for the fingerprint test). The obvious worry is that "cliff" there
means little more than "shares a skeleton". **A clearly labelled sensitivity analysis now scores the
same predictions under fingerprint-only similarity at three cuts, and the MoleculeACE results are
unchanged and still reported.** All five definitions live side by side in
`results/cliff_ranking_*.csv` under a `definition` column.

| definition | TG2 | CDK2 | MAO-B | median cliff pairs per fold |
|---|---|---|---|---|
| `two_test` (Morgan or scaffold) | 29/30 worse, 4 Holm-sig | 15/15, 8 | 13/13, 6 | 16 / 38 / 611 |
| `three_test` (MoleculeACE, full) | 30/30 worse, 2 Holm-sig | 15/15, 8 | 13/13, 6 | 15 / 36 / 634 |
| `morgan_only_0.9` (strictest) | 9/10 worse, 1 | 12/15, 5 | 13/13, 2 | **1 / 2 / 3** |
| `morgan_only_0.8` | 26/30 worse, 2 | 15/15, **7** | 13/13, 6 | 3 / 3 / 9 |
| `morgan_only_0.7` | 25/30 worse, 2 | 15/15, **10** | 13/13, **12** | 5 / 6 / 54 |

**The finding is not an artefact of the loose similarity test.** Under fingerprint-only similarity
the direction holds in 34 of 38 comparisons at the strictest cut, and on the two replication targets
it holds unanimously at every cut — with *more* Holm-significant comparisons at 0.7 and 0.8 (CDK2
10, MAO-B 12) than under MoleculeACE's own definition.

Two honest qualifications. At `morgan_only_0.9` the pair counts collapse to 1–3 per fold, so that
row is a direction check and nothing more; the 0.7 and 0.8 cuts are where the strict analysis has
power, and they are *looser* than MoleculeACE's threshold even though they are stricter in which
tests they use. And TG2 weakens slightly under every strict definition (25–26 of 30 against 29–30),
which is consistent with its having only 23 fingerprint-similar discordant pairs in the whole
inventory.

---

## 3. The missing-class problem: mostly manufactured by the split

`validation_2026-11-09.md` §1 reported that 20–52% of publication units yield no value for one rate
and flagged it as a limit on the corrected per-class analysis. Decomposing that number changes its
meaning.

| target | arm | single-class share of publication-**fold** units | single-class share of **publications** |
|---|---|---|---|
| TG2 | `scaffold_pub` | **50%** | **9%** |
| TG2 | `paper_lopo` | 9% | 9% |
| CDK2 | `scaffold_pub` | **71%** | 35% |
| CDK2 | `paper_grouped` | 24% | 24% |
| MAO-B | `scaffold_pub` | **72%** | 37% |
| MAO-B | `paper_grouped` | 26% | 26% |

In the publication-held-out arms the two columns coincide, as they must — whole publications go to
test. In the scaffold arms they diverge sharply: **a scaffold split cuts each publication across
train and test, and the test-side remnant is frequently all one class.** On TG2 only 9% of
publications are genuinely single-class, but half of the publication-fold units are. The missing
class is largely an artefact of the splitting, not a property of the literature.

Those publications are also small: on TG2 they hold 2–4% of molecules (median 3–30 molecules against
41–76 for both-class publications), on CDK2 and MAO-B 15–29%.

### An estimator that uses every publication

Summing confusion matrices across publications and dividing once afterwards keeps every unit: a
publication with no actives contributes nothing to the sensitivity denominator but still contributes
its inactives to specificity. `results/per_class_pooled_rates.csv` reports it with a cluster
bootstrap over publications. It is reported *beside* the equal-weight estimator, not instead of it,
because it restores the molecule-pooled weighting §5.14 warns about.

Three estimators of the same quantity, TG2 random forest, `scaffold_pub` → `paper_lopo` sensitivity:

| estimator | reference | shifted | Δ |
|---|---|---|---|
| fold average (original §5.17 run) | 0.823 | 0.320 | −0.503 |
| **pooled confusion, every publication** | 0.848 | 0.349 | **−0.499** |
| equal-weight publication (corrected) | 0.660 | 0.320 | −0.340 |

And the direction counts under the pooled estimator:

| target | sensitivity | specificity |
|---|---|---|
| TG2 | **9/10 fall, mean −0.284** | 2/10 fall, mean +0.068 |
| CDK2 | 6/6 fall, mean −0.091 | 6/6 fall, mean −0.074 |
| MAO-B | 5/6 fall, mean −0.041 | 6/6 fall, mean −0.041 |

**This changes the cross-target reading again, and in the direction of a simpler claim.** Under
pooled weighting MAO-B is symmetric (−0.041 against −0.041), not specificity-leaning as the
equal-weight estimator suggested; CDK2 is symmetric under both. The only statement robust to the
choice of estimator is:

> **TG2 loses sensitivity and not specificity. CDK2 and MAO-B lose both.**

No comparison is significant under any estimator — under pooling, 9 of 10 TG2 sensitivity
comparisons have overlapping 95% intervals, and all CDK2 and MAO-B comparisons do.

### Which estimator to prefer

Neither, alone. Equal-weighting publications is right in principle and discards half the units in
the scaffold arms for a reason that turns out to be a splitting artefact (above). Pooling keeps
every unit and lets the biggest publications dominate. **The defensible practice is to report both
and treat a conclusion that depends on the choice as unresolved** — which is what §5.17 now does.
The deeper fix is not an estimator: it is to stop splitting publications across folds, so that the
grouping unit used for aggregation matches the one used for splitting.

---

## 4. What changed, what did not

**Strengthened.** The activity-cliff result (§2) survives a stricter definition and gains
significance on the replication targets. The error-concentration result (§1.2) is the most clearly
significant finding in this pass, at p < 0.001 on every target and arm.

**Weakened further.** §5.17's cross-target class signature. Three estimators give three different
pictures for MAO-B, and nothing is significant under any of them.

**New, and mechanistic.** §1.3–§1.4: the models predict the local class prior. This is the first
account in the project that explains the class asymmetry, the novelty effect and the error
concentration with one idea rather than three.

**Still not measured.** Overfitting. Every number here is distribution shift or label noise. The
in-sample versus out-of-fold comparison that would measure memorisation needs training runs and is
proposed, not run.
