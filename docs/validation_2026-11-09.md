# Validation of experiments 1–3: three methodological corrections, and what they changed

Kareem Itani · 2026-11-09 (later the same day) · reproduce with
`python scripts/analyse_per_class.py` and `python scripts/analyse_cliffs.py`.
Originals, left in place and not rewritten: [`per_class_2026-11-09.md`](per_class_2026-11-09.md),
[`cliffs_2026-11-09.md`](cliffs_2026-11-09.md),
[`fold_balance_2026-11-09.md`](fold_balance_2026-11-09.md).

**Headline: one conclusion is materially weakened, one is unchanged, one is unaffected.** The
per-class asymmetry on TG2 survives in *direction* but no longer in *significance* once the
publication is used as the unit of independence. The activity-cliff result is unchanged. The
fold-class-balance null stands and is preserved as a negative result.

| correction | finding affected | outcome |
|---|---|---|
| 1. Publication as the aggregation unit | Experiment 1, class asymmetry | **Weakened.** TG2 sensitivity: 10/10 → 8/10 models fall, sign-test p 0.0020 → 0.109, Holm-significant 2 → 0 |
| 2. Oracle thresholds flagged, not reported as performance | Experiment 1, threshold analysis | **No change to any conclusion**; now enforced in the data rather than only in prose |
| 3. SMILES-Levenshtein similarity test added | Experiment 2, activity cliffs | **Not material.** Median change in cliff accuracy 0.005; direction becomes unanimous, 58/58; two TG2 significances lost |

Verification after all three: **433 tests pass, all 113 paper values still recompute.** No committed
result file was overwritten; every correction writes new files beside the originals.

---

## 1. Publication as the unit of aggregation — the correction that matters

### What was wrong

`audit_2026-11-05.md` correction #9 established that about 92% of CDK2's protocol gap is the
*aggregation unit* rather than generalisation: a publication-held-out arm is conventionally averaged
per publication while a scaffold arm is averaged per molecule. Experiment 1 averaged both arms per
**fold**, which is not the same thing, and the original write-up listed this under "not done".

Measured from `results/splits_*.csv`, the asymmetry is stark:

| arm | publications per test fold |
|---|---|
| `paper_lopo` | exactly **1** (median 1, min 1, max 1, all three targets) |
| TG2 `scaffold_pub` | median 8, range 4–11 |
| CDK2 `scaffold_pub` | median 47, range 40–55 |
| MAO-B `scaffold_pub` | median 101, range 85–113 |

So a leave-one-publication-out fold average **already is** a publication average — and indeed the
`paper_lopo` numbers are identical to three decimals under both units, which is a useful internal
check that the new code is doing what it claims. The error was entirely on the reference side: the
`scaffold_pub` arm pooled up to 113 publications per fold.

### What changed

Re-aggregating `scaffold_pub` per publication lowers *both* rates, because a molecule-pooled average
is dominated by the largest publications:

| TG2, t = 0.5 | sensitivity (fold → publication) | specificity (fold → publication) |
|---|---|---|
| Random forest | 0.823 → 0.789 | 0.679 → 0.595 |
| XGBoost | 0.848 → 0.767 | 0.691 → 0.626 |
| Chemprop | 0.796 → 0.755 | 0.479 → 0.404 |

Two publication averages are reported and they differ, which is itself informative. The summary
tables average every publication-fold row, so a large publication weighs more; the matched test in
`results/per_class_shift_tests.csv` averages each publication over the folds it appears in first, so
every publication weighs the same. On TG2 the random forest's `scaffold_pub` sensitivity is 0.789
the first way and **0.660** the second — equal-weighting publications costs 0.13, because the big
papers are the ones the model handles well.

The matched test uses the equal-weight figure throughout, and on it the asymmetry's point estimate
stays large: the random forest goes from **sensitivity 0.660 → 0.320 (−0.340)** while
**specificity rises 0.559 → 0.761 (+0.202)**, against the fold-unit figures of −0.503 and +0.081.

But the correction also changes the number of independent units, and that is what kills the
significance. The `scaffold_pub` arm's 20 seeds contain only **11 distinct publications** on TG2, and
variation between publications is far larger than between folds:

| unit | TG2 sensitivity | TG2 specificity |
|---|---|---|
| fold (original) | 10/10 fall, sign p = 0.0020, **2 Holm-significant** | 3/10 fall, p = 0.34, 0 significant |
| publication × fold | 10/10 fall, sign p = 0.0020, 1 Holm-significant | 3/10 fall, p = 0.34, 0 significant |
| **publication (corrected)** | **8/10 fall, sign p = 0.109, 0 Holm-significant** | 3/10 fall, p = 0.34, 0 significant |

Both intermediate units are reported in `results/per_class_shift_tests.csv` rather than only the one
that suits, because the conclusion depends on the choice. The publication unit is the defensible
default: a publication that lands in eight scaffold seeds is not eight independent observations, and
§4.5 of the paper already states that molecules from one publication form a correlated block.

### The corrected cross-target picture

| target | sensitivity | specificity | reading |
|---|---|---|---|
| TG2 | 8/10 fall, p = 0.109, mean −0.163 | 3/10 fall, p = 0.34, mean +0.054 | leans sensitivity, not significant |
| CDK2 | **6/6 fall, p = 0.031**, mean −0.101 | 4/6 fall, p = 0.69, mean −0.026 | leans sensitivity |
| MAO-B | 3/6 fall, p = 1.00, mean −0.001 | **6/6 fall, p = 0.031**, mean −0.035 | leans specificity |

This is a **different story from the original write-up**, which reported "three targets, three
different class signatures" (TG2 sensitivity, CDK2 symmetric, MAO-B specificity). On the corrected
unit it is **two targets leaning sensitivity and one leaning specificity**, with nothing surviving
Holm anywhere. The claim that now fits the evidence is narrower:

> The publication-shift penalty is not class-symmetric, and which class absorbs it differs by
> target. On no target is the asymmetry significant once publications are treated as the unit of
> independence, so this is a direction-consistent observation across three targets, not a
> demonstrated effect on any one of them.

### A selection effect that must travel with these numbers

A publication that reports only actives yields no specificity; one that reports only inactives
yields no sensitivity. Those units are skipped and **counted**, never read as zero:

| arm | units | no sensitivity | no specificity |
|---|---|---|---|
| TG2 `scaffold_pub` | 160 | 26% | 24% |
| TG2 `paper_lopo` | 11 | 0% | 9% |
| CDK2 `scaffold_pub` | 479 | 36% | 34% |
| CDK2 `paper_grouped` | 83 | 10% | 14% |
| MAO-B `scaffold_pub` | 990 | **52%** | 19% |
| MAO-B `paper_grouped` | 209 | 20% | 6% |

Between a fifth and a half of publication units are single-class. The publication-averaged rates
describe the publications that happen to report both classes, and those are the larger, more
balanced papers. This is a limitation of the corrected analysis, not of the original one, and it is
part of why the corrected numbers are lower.

## 2. Oracle thresholds, now separated in the data

No conclusion changes here — the original write-up already argued the oracle cut is selection noise,
showing its gain scales with fold size (+0.020 on 205-molecule folds against +0.183 on
19-molecule folds). The correction is mechanical: every row of `results/per_class_*.csv` and
`results/per_class_*_bypub.csv` now carries an **`achievable`** column, true for the 0.5 cut and the
training-prevalence cut, false for `oracle`. The summaries are keyed by it, so an oracle number
cannot be averaged into an achievable one by accident.

Restating the standing position, because it is the kind of number that escapes its caveat: **the
oracle row is an upper bound on what any threshold rule could reach, not a performance estimate.**
TG2 random forest balanced accuracy of 0.711 on `paper_lopo` at the oracle cut is not achievable; the
achievable figures are 0.506 at the 0.5 cut and 0.545 at the prevalence-matched cut.

## 3. The SMILES-Levenshtein test — implemented, and not material

### Done without touching the pinned environment

MoleculeACE ORs a third similarity test into its cliff definition: normalised SMILES Levenshtein
similarity ≥ 0.9. No edit-distance library is installed and installing one would alter the
environment every committed result was produced under. It is now implemented inside
`scripts/analyse_cliffs.py` with two vectorised lower bounds on the edit distance pruning first and
an abandoning DP deciding the survivors:

- **length** — similarity ≥ 0.9 requires |len(a) − len(b)| ≤ ⌊0.1 · max(len)⌋;
- **histogram** — one edit changes the character histogram by at most 2, so the distance is at
  least half the total histogram difference.

Both are lower bounds, so neither can discard a qualifying pair. `tests/test_cliffs.py` checks the
budgeted DP against a plain DP on 300 random string pairs at every budget from 0 to 7, and checks
that the prefiltered search returns **exactly** the same set as an exhaustive scan. Cost: the full
three-target run went from 31 to 57 seconds.

### What it adds

| target | cliffs (2 tests) | cliffs (3 tests) | pairs admitted by Levenshtein alone | of those, cliffs |
|---|---|---|---|---|
| TG2 | 705 | **728** (+3.3%) | 83 | 23 |
| CDK2 | 5,210 | **5,249** (+0.7%) | 101 | 39 |
| MAO-B | 51,124 | **51,448** (+0.6%) | 502 | 324 |

Effect on the result, over all 58 model × arm comparisons: median absolute change in cliff ranking
accuracy **0.005**, mean 0.009, maximum 0.041.

- **Direction: unchanged and slightly strengthened.** Cliffs are harder in **58 of 58** comparisons
  under the full definition, against 57 of 58 before — the single exception, ChemBERTa-2 frozen on
  TG2 `paper_lopo`, moves from +0.014 to −0.003.
- **Significance: two TG2 losses.** Holm-significant comparisons go 4 → 2 on TG2 (Chemprop 0.008 →
  0.004 and Chemprop + RDKit 0.010 → 0.036 survive; logistic regression 0.028 → 0.079 and
  MoLFormer 0.031 → 0.123 do not). CDK2 stays at 8 of 15, MAO-B at 6 of 13.
- **Rank correlations: unchanged.** MAO-B `scaffold_pub` 0.886 → 0.943 (p = 0.005), TG2 `scaffold`
  0.758 → 0.770, TG2 `scaffold_pub` and both `paper_grouped` arms identical. The §4 conclusion — that
  overall ranking predicts cliff ranking within-distribution but not under shift — is untouched.

**Verdict: the omission was not material.** Chemprop on TG2 `scaffold_pub` remains below chance
(0.490 → 0.468).

### An incidental finding about the definition itself

Counting which test admitted each similar discordant pair:

| target | Morgan ≥ 0.9 | generic scaffold ≥ 0.9 | Levenshtein ≥ 0.9 |
|---|---|---|---|
| TG2 | 23 | 1,840 | 270 |
| CDK2 | 34 | 9,404 | 636 |
| MAO-B | 214 | 72,054 | 2,097 |

**The generic-scaffold test does essentially all the work**, and the fingerprint test almost none —
23 of 1,863 pairs on TG2. In practice MoleculeACE's "structurally similar" resolves, on our data, to
"shares a generic Bemis–Murcko scaffold", which is a much looser relation than a 0.9 fingerprint
Tanimoto: the generic scaffold discards atom and bond identity, so a pyridine and a benzene ring
collapse together. That is faithful to the published rule and it is worth stating plainly, because it
means our cliff pairs are mostly *same-skeleton, different-substituent* pairs rather than
*near-identical-fingerprint* pairs. Whether that is the right notion of a cliff is a question for the
original authors, not something our data can settle, and no parameter sweep was run — the published
values were fixed first, as pre-registered.

## 4. The fold-class-balance null, preserved

Experiment 3 needed no correction and its conclusion is unchanged: validation-fold imbalance
explains at most 5% of per-seed AUROC variance (r² ≤ 0.051), every confidence interval spans zero on
both TG2 arms, and the point estimates mostly carry the opposite sign to the prediction. The
hypothesis is withdrawn and the result is kept in the repository as a negative finding, with its
pre-registration, its rewritten control group, and the competing explanation it turned up (weaker
models vary more; r = −0.912 on TG2 `scaffold`). `results/fold_balance*.csv` are unmodified.

## 5. Threats to validity, after correction

- **Power, not absence.** Nothing in experiment 1 is significant on the corrected unit. With 11
  publications on TG2 and 6 models on the replication targets, a null here means "not detectable at
  this power". It does not license the opposite claim either.
- **Single-class publications.** 20–52% of publication units give no value for one rate (§1). The
  corrected rates describe a non-random subset of publications.
- **Protocol is not matched across targets.** TG2's shifted arm is leave-one-publication-out (11
  folds, median 38 molecules); CDK2's and MAO-B's is grouped publication folds (10 folds, 210 and
  512). The cross-target comparison in §1 therefore varies protocol as well as target, and
  prevalence differs too (TG2 54% positive in the shifted arm, CDK2 56%, MAO-B 38%).
- **Cliff definition.** The generic-scaffold test dominates (§3), the censored negatives are excluded
  entirely (12–26% of similar discordant pairs, non-randomly — they are the weakest-potency
  molecules), and TG2's per-fold cliff counts are thin (median 5.5 per `scaffold_pub` fold).
- **Generalisation failure is not overfitting.** Every result here measures performance falling when
  the test distribution changes. None of it demonstrates that a model memorised its training set:
  no train-versus-test gap was measured, because training-fold predictions were never saved. That
  distinction is kept throughout and the word "overfitting" is not used for these findings.
