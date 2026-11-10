# The next three experiments: what the validated findings say to fix

Kareem Itani · 2026-11-09 · targets the three weaknesses established in
[`validation_2026-11-09.md`](validation_2026-11-09.md) and paper §5.17–§5.19.
Nothing here has been run. Each experiment names the committed baseline it must beat, the metric
fixed in advance, and the way it could fail or mislead.

Standing constraints carried from the rest of the project: local compute only, one RTX 5070 Ti, no
hosted APIs; committed results are never overwritten; every arm gets its seeds, splits and
configuration recorded; and a result that does not replicate on CDK2 and MAO-B is reported as
TG2-specific.

A note on what these can and cannot establish. All three address **reliability under distribution
shift**. None of them makes a prediction about a molecule trustworthy enough to act on without
assay confirmation, and none of them is a step toward a celiac treatment on its own. §7.2's position
is unchanged.

---

## Experiment 4 — Cut false negatives under publication shift

**The weakness.** §5.17: on TG2 the random forest's sensitivity falls from 0.660 to 0.320 under
publication shift while its specificity *rises* to 0.761, and its predicted-positive rate halves on
a test set that is 54% positive. CDK2 behaves the same way (6 of 6 models). A screening model that
answers "not potent" to two actives in three is useless for nominating compounds, whatever its
AUROC.

**The question, and the one thing that must be separated.** Is the collapse a *threshold* failure —
the probability scale drifting under shift — or a *ranking* failure? §5.17 already bounds this:
prevalence matching recovers almost nothing on TG2 (0.320 → 0.354) and a great deal on MAO-B
(0.518 → 0.728). So the two targets fail for different reasons and must be treated separately. The
experiment is built to tell them apart rather than to improve a headline.

**Arms.** Three thresholding rules on the *existing* saved predictions, no retraining:
(a) the 0.5 cut; (b) the training-prevalence cut; (c) a cut chosen on **held-out publications** —
an inner leave-publications-out loop inside the training fold, which is the same move that fixed
conformal coverage in §5.13 applied to the operating point. Then two refits that change the model
rather than the cut: (d) class-weighted fitting with the weight chosen on held-out publications, and
(e) group DRO over publications — minimise the worst publication's loss — implemented as iterative
reweighting on logistic regression, the random forest and XGBoost, which needs no GPU.

**Primary endpoint, fixed now: sensitivity at specificity ≥ 0.80**, on the publication unit, on the
shifted arm. A single number that cannot be improved by simply predicting "potent" more often —
which is the obvious way to fake success here and the reason balanced accuracy is *not* the primary
endpoint. Secondary: balanced accuracy, PR-AUC, and AUROC to confirm ranking is not damaged.

**Baselines to beat.** The committed `scaffold_pub` → shifted figures in
`results/per_class_shift_tests.csv`: TG2 random forest 0.320 sensitivity, XGBoost 0.347, logistic
regression 0.356; CDK2 and MAO-B as tabulated. Arms (a) and (b) are already computed, so (c), (d)
and (e) are the only new work.

**Cost.** Arm (c) is minutes of CPU. Arms (d) and (e) refit three fingerprint models across all arms
and seeds — hours of CPU, no GPU. The deep entrants are deliberately excluded: §5.10 shows they
lose to the random forest anyway and they would multiply the cost by an order of magnitude.

**How it could fail or mislead.**
- **The trade is real and may not be worth making.** Every sensitivity gain costs specificity. If
  arm (c) buys 0.15 sensitivity at 0.80 specificity, that is useful; if it needs specificity to fall
  to 0.50, it is just a relabelled threshold and should be reported as a null.
- **Inner publication loops are small.** TG2's training folds hold around 10 publications, so a
  threshold chosen on held-out publications is chosen on very few. That is the same weakness that
  limits §5.16(b), and it may make arm (c) unstable rather than better.
- **Group DRO optimises the worst publication**, which on this data may be one small, odd paper.
  Worst-case objectives on 11 groups are notoriously driven by the single hardest group. The
  pre-registered reading is that a DRO gain concentrated in one publication is not a gain.
- **Nothing here fixes ranking.** If TG2's collapse is mostly ranking, as the prevalence result
  suggests, the honest outcome of this experiment is "not fixable by reweighting or thresholding",
  which is worth establishing but is a negative result.

---

## Experiment 5 — Make cliff pairs predictable, or establish that they are not

**The weakness.** §5.18: cliff pairs rank below matched distant pairs in 58 of 58 comparisons, and
Chemprop on TG2 `scaffold_pub` ranks them at 0.468 — below chance. §5.18 also found that the
generic-scaffold test admits nearly every cliff pair (1,840 against 23 for the fingerprint test), so
our cliffs are *same-skeleton, different-substituent* pairs. That is a specific, actionable
diagnosis: the models are failing to resolve substituent effects within a fixed skeleton.

**The question.** Does a representation that distinguishes substituent *counts* and positions,
combined with an objective that scores pairs rather than molecules, improve cliff ranking without
costing overall performance?

**Arms.** (a) The committed binary Morgan random forest, as baseline. (b) **Count-based ECFP**
instead of binary folded bits — the one change `state_of_the_art_comparison.md` §1.0 records as
recommended by its most relevant source and never tested here, and the natural fix for a
substituent-count failure. (c) A **pairwise ranking objective**: train on discordant pairs within
the same generic scaffold, optimising the probability that the potent member scores higher, which
matches the evaluation metric to the training signal. (d) (b) and (c) combined. All four on the same
splits and seeds as §5.18.

**Primary endpoint, fixed now: cliff-pair ranking accuracy, with distant-pair accuracy reported
beside it in every table.** A method that lifts cliffs by flattening everything else has not helped,
and the matched control is the only thing that makes this visible. Both definitions (two-test and
three-test) are reported, as `results/cliff_ranking_*.csv` now carries both.

**Baselines to beat.** `results/cliff_ranking_*_tests.csv`, three-test definition: TG2
`scaffold_pub` random forest 0.736 cliff against 0.891 distant; CDK2 0.763 against 0.901; MAO-B
logistic regression 0.728 against 0.824. Target: narrow the gap, not raise the cliff number alone.

**Cost.** Count-ECFP is a featuriser change — hours of CPU across all arms and seeds. The pairwise
objective needs a small training loop but no GPU. Pair enumeration is already implemented and takes
seconds.

**How it could fail or mislead.**
- **Thin pair counts on TG2.** A median of 5.5 cliff pairs per `scaffold_pub` fold means per-fold
  accuracies are coarse. CDK2 (96–143) and MAO-B (732–1,166) must carry the conclusion; a TG2-only
  gain is noise until it replicates.
- **Pairwise training changes the task.** A model trained to rank pairs is no longer calibrated for
  single-molecule probability, so §5.13's and §5.16's uncertainty machinery would not apply to it.
  That is a real cost and it must be stated, not discovered later.
- **Training on within-scaffold pairs is training on the test construct.** The pairs used for
  training must come from training-fold molecules only, and the cliff pairs used for scoring must
  involve no training molecule. Getting this wrong would manufacture the result, and it is the single
  thing to verify before any number is believed.
- **The cliff definition may be doing the work.** If the generic-scaffold test is what makes these
  pairs "similar", a gain might reflect the definition's looseness rather than better chemistry. A
  pre-registered secondary analysis on fingerprint-similar pairs only (23 on TG2 — almost certainly
  too few) would be the check, and its likely verdict is that our data cannot answer it.

---

## Experiment 6 — Uncertainty that is both honest and usable

**The weakness.** §5.13 and §5.16 leave the project with a choice between two unusable options:
i.i.d. conformal calibration that under-covers under publication shift (0.784 against a nominal
0.90 on TG2), and hierarchical conformal prediction that honours its floor in all 54 runs but
commits to a single label on **7.7%** of TG2 molecules. A method that abstains on twelve molecules
in thirteen cannot prioritise anything. §5.19 adds that we cannot even predict which *folds* will be
hard from their composition.

**The question.** Does conditioning the nonconformity score on **chemical novelty** — maximum
Tanimoto to the training fold — give usable per-molecule reliability where conditioning on class
(Mondrian, withdrawn in §5.13) and on group (§5.16) did not? The motivation is internal and
specific: §5.6 shows accuracy collapsing below 0.4 Tanimoto on TG2, and §5.14 finds that what
remains of the publication penalty after the aggregation correction tracks novelty. If error depends
on novelty, calibrating within novelty bands should produce intervals that are tight where the
chemistry is familiar and wide only where it is not — which is the behaviour a screening cascade
needs.

**Arms.** All on committed predictions, no retraining: (a) i.i.d. split conformal, (b)
publication-grouped calibration, (c) hierarchical conformal prediction — the three already in
`results/` — against (d) **novelty-banded conformal**, calibrating separately within
maximum-Tanimoto bands, and (e) novelty-banded *and* publication-grouped.

**Primary endpoints, fixed now:** empirical coverage against nominal, **and** singleton rate,
reported as a pair. Neither alone is meaningful — the whole lesson of §5.16 is that coverage bought
with abstention is not progress. Secondary: risk–coverage curves, and coverage within each novelty
band, since a method can hit marginal coverage while failing badly on the novel molecules that
matter most.

**Baselines to beat.** `results/lopo_conformal_summary.csv`, the two
`conformal_*_paper_grouped_summary.csv` files and `results/hcp_comparison.csv`: 54 of 54 runs
holding the floor for HCP against 22 of 54 for publication grouping, with HCP's TG2 singleton rate
of 7.7% as the usability bar. An arm that holds the floor and raises the singleton rate above HCP's
is an improvement; one that does either alone is not.

**Cost.** Hours of CPU at most. This is a calibration experiment on saved predictions, which is why
it is third rather than first: the cheapest of the three, but also the one whose success depends most
on the other two producing a better-ranking model to calibrate.

**How it could fail or mislead.**
- **No guarantee survives band conditioning.** Splitting the calibration set by a covariate
  estimated from the training fold breaks the exchangeability that split conformal needs, exactly as
  §8 records for the publication-grouped method. Arm (d) would be **empirical, like §5.13's fix, not
  guaranteed like §5.16's**. The paper would have to say so in the same words.
- **Bands lack common support.** §8 already states this for §5.14's novelty analysis: the band
  populations differ systematically between arms. Calibrating within bands inherits that problem and
  may simply relocate it.
- **Fewer points per band.** TG2's 0.4 band holds 25 molecules. Conformal calibration on 25 points
  gives a quantile with no resolution, so the most interesting band is the one where the method is
  weakest — the same structural problem as §5.16(b).
- **Mondrian already failed this way.** Class-conditional calibration improved TG2 by 30% and did
  not replicate on either other target, and was withdrawn. Novelty conditioning is a different
  variable but the same family of move, and the pre-registered standard is that a TG2-only gain is
  withdrawn, not reported.

---

## Order, and why

**4, then 5, then 6.** Experiment 4 addresses the failure that matters most for the stated purpose —
missing actives — and most of it runs on data already committed. Experiment 5 has the most specific
mechanistic diagnosis behind it (substituent resolution within a fixed skeleton) and would, if it
worked, improve the ranking that experiment 6 then has to calibrate. Experiment 6 is cheapest but
depends on the other two: calibrating a model that ranks at 0.612 AUROC under shift cannot produce
useful intervals no matter how the calibration is arranged.

**What would count as a result worth writing up.** Any one of: sensitivity at 0.80 specificity
improved by more than 0.10 on two of three targets; the cliff-to-distant gap narrowed on CDK2 and
MAO-B together; or coverage held at nominal with a singleton rate above HCP's on TG2. Anything less
is a negative result, and — as §5.19 demonstrates — a negative result obtained against a written
prediction is kept in the repository rather than discarded.
