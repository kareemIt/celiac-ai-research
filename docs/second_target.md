# Do the TG2 evaluation failures reproduce on CDK2? (run 2026-10-29)

Reproduce: `python scripts/audit_candidate_targets.py`, then `python scripts/run_second_target.py`.
Tables: `results/candidate_targets.csv`, `results/second_target_splits.csv`,
`results/second_target_leakage.csv`, `results/second_target_novelty.csv`. CPU, a few minutes.

The TG2 work ([`paper.md`](paper.md)) rests on four evaluation failures observed on one target. A
single case study cannot distinguish "this is how small bioactivity datasets behave" from "this is
how the TG2 dataset behaves". This phase tests three of the four on an independent target.

## Phase 1 — choosing the target

Five human enzymes were fetched and scored on whether they could *test* the failures, not on size.
The binding criterion is publications contributing at least 10 labelled molecules: TG2's covalent
set had 6, so each leave-one-paper-out fold was a single laboratory. Thresholds were fixed in code
before the numbers were seen.

| target | molecules | % pos | scaffolds | papers ≥10 | largest paper's share of positives | dup. fp | Tanimoto to TG2 |
|---|---|---|---|---|---|---|---|
| BACE1 | 8,431 | 68% | 3,000 | 227 | 6% | 14.0% | 0.205 |
| CA2 | 983 | 71% | 337 | 40 | 9% | 5.2% | 0.220 |
| **CDK2** | **2,663** | **53%** | **1,192** | **84** | **9%** | **8.7%** | **0.220** |
| MAO-B | 5,859 | 36% | 1,863 | 209 | 3% | 10.2% | 0.235 |
| sEH | 2,791 | 80% | 1,137 | 85 | 9% | 7.1% | 0.272 |
| *TG2, same code* | *460* | *53%* | *213* | *11* | *24%* | *4.8%* | *—* |

**CDK2 selected.** Best class balance (53%, against TG2's 48% on the full set), highest scaffold
diversity per molecule (0.448 against TG2's 0.463), 84 publications with enough molecules for real
folds, a CPU-sized workload, and chemistry mechanistically unlike TG2 — reversible ATP-competitive
heterocycles against covalent acrylamides, median Tanimoto 0.220 with one shared molecule. BACE1
was disqualified on workload, CA2 because sulfonamide-dominated chemistry repeats TG2's
single-warhead problem, sEH because 80% positive leaves a thin negative class.

### A prediction registered before running anything

Every candidate spreads its positives across publications far more evenly than TG2's selectivity
set did — 3–9% in the largest paper against 38%. We therefore predicted that **publication-level
confounding would largely not reproduce**, and said so in the selecting commit.

**That prediction was wrong.** It is recorded here because a prediction that only appears when it
succeeds is worthless.

## Phase 2 — provenance, and two decisions it forced

CDK2 required two data decisions TG2 did not.

**Non-binding assays must be excluded.** CDK2's IC50 records mix assay types: 3,555 binding (B), 25
functional (F) and 17 ADMET (A). The F rows are MTT cell-viability readouts, with a median IC50 of
14,100 nM against 200 nM for the binding assays. Pooling them would label the same chemistry both
potent and not. `src/data/chembl.py` now restricts labels to `assay_type == "B"`. This is verified
to be a **no-op for the entire transglutaminase family**, whose files are 100% type B — every TG2
label is byte-identical before and after.

**Censored `<` rows are still discarded.** CDK2 has 65 rows with a `<` bound, which would prove
potency where the bound sits below the 1 µM cut; TG2 has 10. Handling them would change TG2's
published numbers, so the existing rule is kept for both targets. Consistency across targets is
worth more than 65 molecules, and the discarded counts are recorded here.

Other provenance notes: all 10,344 records are *Homo sapiens*; 120 rows carry ChEMBL's "outside
typical range" flag and one a "potential transcription error", all dropped; 52 IC50 rows in µg/mL
are excluded as unconvertible without molecular weight. Unlike TG2, **2,017 of 6,752 CDK2 molecules
carry stereochemistry**, so this dataset does not suffer the enantiomer collapse that caps TG2.

Final dataset: **2,646 molecules, 1,400 potent (53%), 1,184 scaffolds, 351 publications.**

## Phase 3 — what each split actually holds out

| split | shared scaffolds | shared papers | mean max Tanimoto | identical fingerprint | near-duplicate (≥0.7) |
|---|---|---|---|---|---|
| random | 110.4 | 104.4 | 0.752 | 7.3% | 74.4% |
| scaffold | 0.0 | 69.2 | 0.654 | 0.2% | 47.7% |
| paper | 11.2 | 0.0 | 0.436 | 1.0% | 10.4% |

**The `shared papers` column is the mechanism.** A scaffold split leaves **69 publications with
molecules on both sides of the fold**. It removes scaffold overlap completely and still lets the
model see most laboratories' chemistry during training, because one laboratory's series spans
several Murcko scaffolds. Only the paper split drives mean similarity down to 0.436 and
near-duplicates to 10%.

## Results — the same molecules, the same models, three protocols

| split | model | molecules | AUROC | 95% CI | per-seed | ECE | Brier |
|---|---|---|---|---|---|---|---|
| random | RF | 1,722 | 0.928 | [0.916, 0.939] | 0.923 ± 0.013 | 0.027 | 0.106 |
| random | logreg | 1,722 | 0.919 | [0.907, 0.931] | 0.916 ± 0.016 | 0.050 | 0.117 |
| scaffold | RF | 1,740 | 0.898 | [0.884, 0.913] | 0.891 ± 0.046 | 0.042 | 0.128 |
| scaffold | logreg | 1,740 | 0.876 | [0.858, 0.892] | 0.866 ± 0.056 | 0.079 | 0.150 |
| **paper** | **RF** | **1,544** | **0.830** | **[0.808, 0.850]** | 0.824 ± 0.055 | 0.092 | 0.173 |
| paper | logreg | 1,544 | 0.784 | [0.761, 0.806] | 0.778 ± 0.040 | 0.141 | 0.210 |

Random → scaffold costs **−0.030** AUROC; scaffold → paper costs a further **−0.068**. The random
and paper intervals do not overlap, and neither do the scaffold and paper intervals. Calibration
degrades monotonically along the same axis: ECE 0.027 → 0.042 → 0.092.

### 1. Random-split leakage — **reproduces, more mildly**

Random splitting inflates RF AUROC by 0.030 over scaffold splitting, with 7.3% of test molecules
fingerprint-identical to a training molecule and 74.4% having a near-duplicate. The direction and
mechanism match TG2 exactly. The *magnitude* is much smaller: TG2 had **38.6%** identical
fingerprints and an inflation of roughly 0.08 AUROC. TG2 is an unusually redundant dataset, and
quoting its 38.6% as typical would overstate the problem.

### 2. Pooled-prediction artefact — **reproduces, mildly, and the mechanism is confirmed**

| scoring | n | AUROC | 95% CI | width |
|---|---|---|---|---|
| pooled over fold rows | 2,470 | 0.829 | [0.812, 0.845] | 0.033 |
| one row per molecule | 1,544 | 0.830 | [0.810, 0.850] | 0.041 |

Each molecule appears 1.6 times on average and pooling narrows the interval by **18%**. The point
estimate barely moves; it is the interval, and therefore the significance, that is wrong.

The TG2 case was far worse because of a condition absent here: its zero-shot language model was
*not refitted per fold*, so one cached prediction was counted in every fold it appeared in, up to
ten times. Every model here is refitted, so only ordinary fold overlap contributes. **The artefact
is general; its severity depends on fold overlap and is much worse when comparing a fixed model
against a refitted one** — which is precisely the comparison any "LLM versus classical baseline"
study makes.

### 3. Publication-level confounding — **reproduces, and is the single largest effect**

This is the finding that contradicts our prediction. Despite CDK2 spreading its positives across
351 publications with no single paper holding more than 9%, **holding out whole publications costs
more than scaffold splitting does** (−0.068 against −0.030). The effect is not confined to small,
concentrated panels like TG2's 29-molecule selectivity set.

The mechanism is visible in the leakage table rather than inferred: a scaffold split still shares
69 publications between training and test. Whatever a laboratory's series has in common beyond its
Murcko scaffold — assay protocol, substitution patterns, compound provenance — remains available to
the model.

We regard this as the most useful result of the phase, precisely because it went against the
registered expectation and because CDK2 is a large, diverse, well-studied dataset rather than an
awkward corner case.

### 4. Chemical novelty and reliability — **reproduces directionally; the TG2 threshold does not**

| max Tanimoto to training set | molecules | % positive | AUROC |
|---|---|---|---|
| 0.0–0.4 | 799 | 39.7% | **0.739** |
| 0.4–0.5 | 296 | 48.3% | 0.865 |
| 0.5–0.6 | 190 | 57.9% | 0.834 |
| 0.6–0.7 | 132 | 73.5% | 0.894 |
| > 0.7 | 175 | 70.9% | 0.915 |

Novelty predicts reliability on CDK2 as it does on TG2: performance rises with similarity to the
training set, 0.739 to 0.915. **But the floor is completely different.** TG2 found every model at
*chance* below 0.4 Tanimoto; CDK2 retains AUROC 0.739 there, on 799 molecules with a 40% positive
rate — a large, balanced bin, not a statistical artefact.

**The paper's claim that models rank at chance below 0.4 Tanimoto is therefore TG2-specific and has
been qualified accordingly.** The transferable statement is the weaker one: predictive reliability
declines with chemical distance from the training set, and where it reaches chance is a property of
the individual dataset that must be measured, not assumed.

## What this changes

| Failure | TG2 | CDK2 | Status |
|---|---|---|---|
| Random-split leakage | 38.6% identical, ~0.08 AUROC | 7.3% identical, 0.030 AUROC | **Reproduces**, milder |
| Pooled predictions | severe (cached, up to 10×) | 18% interval narrowing (1.6×) | **Reproduces**, milder; severity condition identified |
| Training nondeterminism | −0.028 ± 0.089 noise floor | not retested (no GPU training) | Untested |
| Publication confounding | 0.974 → 0.574 | 0.898 → 0.830 | **Reproduces**, and larger than the scaffold effect |
| Novelty → reliability | chance below 0.4 | 0.739 below 0.4 | **Partly** — trend reproduces, threshold does not |

**Strengthened:** publication-level confounding is not an artefact of small concentrated panels. It
appeared on a 2,646-molecule, 351-publication dataset and exceeded the scaffold effect. The
recommendation to report a publication-held-out split is now supported on two independent targets.

**Weakened:** the 0.4 Tanimoto applicability floor is a TG2 property, not a general one. Any claim
that models fail below a fixed similarity threshold must be made per dataset.

**Unchanged:** random-split leakage and the pooling artefact reproduce in direction and mechanism
while differing in magnitude, which is what a case study should predict — the mechanism transfers,
the numbers do not.

## Limitations

- One additional target. Two targets establish that a failure is not unique to TG2; they do not
  establish a general law, and the magnitudes still differ substantially between them.
- Nondeterminism was not retested, since this phase ran no GPU training. It remains a single-target
  observation.
- Molecules are attributed to their first source publication. A molecule measured in several papers
  is assigned to one, so the paper split is slightly optimistic — the true penalty may be larger.
- The paper split scores 1,544 of 2,646 molecules, because folds that end up single-class are
  skipped. The three splits therefore score overlapping but not identical molecule sets.
- CDK2 is a mature, heavily-studied kinase target. Its data density and stereochemical annotation
  are better than TG2's; conclusions may not extend to sparsely-studied targets.
- No prospective validation, and no claim about CDK2 inhibitor discovery is made here. This phase
  used CDK2 as a methodological instrument, not as a drug-discovery objective.
