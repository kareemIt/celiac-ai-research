# Design proposal: measuring overfitting directly, and one intervention for unfamiliar chemistry

Kareem Itani · 2026-11-09 · **nothing in this document has been run.** It covers the two requests
that require new training runs. The three analyses that need no training — error sources, a stricter
cliff-similarity sensitivity analysis, and the single-class publication problem — run on committed
predictions and are reported separately.

Costs below are not guesses. The sweep recorded wall-clock seconds for every one of its 1,049 fits
in `results/sota_{tg2,cdk2,maob}.csv`, and every estimate here is derived from those columns.

## Measured cost of the existing sweep, as recorded

| model | fits | recorded total | median per fit (12-config grid) |
|---|---|---|---|
| CheMeleon | 91 | 9.75 h | 278 s |
| ChemBERTa-2 fine-tuned | 91 | 4.54 h | 77 s |
| ChemBERTa-2 frozen | 91 | 2.85 h | 3.4 s |
| Chemprop + RDKit 2D | 51 | 1.33 h | 82 s |
| Chemprop | 51 | 1.26 h | 83 s |
| Random forest | 224 | 0.97 h | 12.8 s |
| Logistic regression | 174 | 0.76 h | 15.0 s |
| XGBoost | 174 | 0.34 h | 6.3 s |
| Tabular foundation model | 51 | 0.29 h | 19.0 s |
| MoLFormer-XL frozen | 51 | 0.07 h | 4.2 s |
| **all three targets** | **1,049** | **22.15 h** | — |

Peak VRAM never exceeded 1,536 MiB (CheMeleon), so nothing here is memory-constrained on the
RTX 5070 Ti. Note that the 6 h 30 m quoted in `sota_comparison.md` is the TG2 nine-entrant sweep
alone; 22.15 h is the recorded total across all three targets.

**The key cost lever:** each recorded figure covers a *grid* of 3 to 12 configurations. An
experiment that reuses the already-selected configuration pays roughly `k/n_configs` of the original
time for `k` extra fits. That is what makes proposal A affordable.

---

## Proposal A — Measure overfitting, separated from distribution shift

### The question, stated so the two causes cannot be confused

Everything in §5.17–§5.19 measures performance falling when the test distribution changes. That is
distribution shift. **Overfitting is a different quantity**: the model performing better on data it
was fitted on than on data from the *same* distribution that it was not. The paper currently cannot
report it, because training-fold predictions were never saved (§8).

The design separates them by producing three numbers per fit, on one axis each:

| quantity | what it is | what a gap to it means |
|---|---|---|
| **in-sample** | the selected model scored on its own training fold | memorisation capacity, not generalisation; for a fully grown random forest this is near-perfect by construction and is reported only as a sanity check |
| **out-of-fold within training** | inner *K*-fold cross-validation inside the training fold, honouring the outer split's grouping, predictions assembled out-of-fold | the honest estimate of generalisation **within the training distribution** |
| **held-out test** | the committed figure, unchanged | generalisation to the outer split's distribution |

Then:

- **in-sample − out-of-fold** is overfitting, measured on one distribution. It cannot be confused
  with shift because both sides come from the same molecules.
- **out-of-fold − test** is distribution shift, measured with capacity held fixed. Under a random
  split it should be ≈ 0; under a scaffold split it isolates the chemical-novelty cost; under a
  publication split it isolates the protocol cost.

This decomposition is the whole point of the experiment, and it answers the professor's question in
the only form that distinguishes the two: *if out-of-fold ≈ test under the scaffold split while test
collapses under the publication split, the models are not overfitting — they are failing to
transfer.*

### Protocol

- **Inner folds grouped the same way as the outer split.** A scaffold-split arm gets inner folds
  grouped by Murcko scaffold; a publication-split arm gets inner folds grouped by publication. An
  ungrouped inner CV would leak analogues across inner folds and report an inflated out-of-fold
  score, manufacturing the appearance of no overfitting.
- **No hyperparameter search inside the inner loop.** The configuration already selected on the
  outer validation fold is reused verbatim. Re-searching per inner fold would measure a different
  estimator than the one the paper reports, and would multiply the cost twelvefold.
- **K = 5**, fixed in advance. `src/data/chem.py:group_split` already does grouped assignment and is
  reused rather than reimplemented.
- **Nothing is fitted on test.** The inner loop lives entirely inside the outer training fold; the
  outer validation fold is excluded from it so that the selected configuration stays untainted.
- Seeds, configurations and fold assignments recorded per row, as elsewhere.

### Cost, derived from the recorded timings

At `5/12` of the original grid cost per fit (5 inner fits reusing one configuration):

| tier | models | arms | estimated |
|---|---|---|---|
| **1** | random forest, logistic regression, XGBoost | all three targets, all arms | **≈ 0.9 h**, CPU only |
| **2** | add Chemprop, Chemprop + RDKit, ChemBERTa-2 fine-tuned | TG2 only | **≈ 3.9 h**, GPU |
| **3** | all ten entrants | all three targets | **≈ 9.2 h**, GPU, overnight |

**Recommendation: run tier 1 first and read it before deciding on tier 2.** Tier 1 covers the three
models that actually win in §5.10, costs under an hour, and is sufficient to answer the professor's
question. Tier 3 exists only if the answer turns out to differ by model family, which tier 1 and
tier 2 together would reveal.

### What could go wrong, and the pre-registered reading

- **Random-forest in-sample scores are uninformative.** Fully grown bagged trees reproduce their
  training labels almost perfectly, so in-sample ≈ 1.0 tells us nothing about whether the model
  generalises. The out-of-fold column is the one that matters, and the in-sample column is reported
  only so that nobody later mistakes its absence for a missing control.
- **An inner 5-fold on TG2 trains on about 400 molecules against the outer fold's 510.** The
  out-of-fold estimate is therefore pessimistic relative to test by a learning-curve amount that has
  nothing to do with shift. This must be bounded before the comparison is believed: the honest
  control is to also fit at the reduced training size on the *outer* split and report the difference.
  Without that, a learning-curve effect would be read as overfitting. **This is the single largest
  threat to the design.**
- **Publication-split arms have few groups.** TG2's training folds hold roughly 10 publications, so
  a grouped inner 5-fold leaves 2 publications per inner fold. The estimate will be noisy and may
  not support a per-model statement; the pre-registered fallback is to report the publication arms
  pooled across models rather than per model.
- **A null is a real outcome.** If in-sample − out-of-fold is small for every model, the answer is
  "these models are not overfitting, and the failures documented in §5.17–§5.19 are shift". That is
  a negative result and it is the one the current evidence predicts.

### Success criteria, fixed now

Not a performance target — this is a measurement, not an improvement. It succeeds if it produces,
for each model and arm, an in-sample/out-of-fold/test triple whose learning-curve confound is
bounded as above, and a statement of the form "overfitting accounts for X of the Y AUROC gap, shift
for the rest", with intervals. It fails if the learning-curve control is of the same size as the
effect, in which case the design cannot separate the two and that must be reported rather than
worked around.

---

## Proposal B — One intervention for generalisation to unfamiliar TG2 molecules

### The intervention: count-based extended-connectivity fingerprints

One change, chosen because three separate findings in this repository point at the same mechanism
and because it is the cheapest thing that addresses them:

- §5.6 and `generalization.md`: below 0.4 Tanimoto to training, every model ranks at chance **on
  TG2**.
- §5.18: cliff pairs are harder in 58 of 58 comparisons, and the pairs are overwhelmingly
  *same-skeleton, different-substituent* — the generic-scaffold test admits 1,840 TG2 pairs against
  23 for the fingerprint test. The failure is substituent resolution within a fixed skeleton.
- Binary folded Morgan bits discard substituent *multiplicity* and collide at 2,048 bits. Count-based
  ECFP keeps multiplicity.

`state_of_the_art_comparison.md` §1.0 records its most relevant source as holding that no such study
is complete without a count-ECFP and tree-model comparison, and this project has never run one. It is
the one intervention with published support, a mechanistic rationale from our own results, and a cost
measured in minutes.

### Fair baseline, and why it is fair

The incumbent is the **tuned random forest on binary Morgan, 2,048 bits**, the winner of §5.10. The
count variant gets **the same model class, the same hyperparameter grid, the same 12-configuration
budget, the same splits, the same seeds and the same 20-seed protocol**. The only thing that changes
is the featuriser. Paired per seed, Welch unpaired across arms, Holm across the comparisons, matching
§5.10.

### Predefined success criteria

Fixed before running, and all three must hold:

1. **Primary:** AUROC in the two lowest novelty bands on TG2 (< 0.4 and 0.4–0.5 maximum Tanimoto to
   training) improves by **≥ 0.05** with a 95% interval excluding zero.
2. **No regression:** overall scaffold-arm AUROC does not fall by more than **0.02**.
3. **Replication:** the direction of the band-level change repeats on CDK2 and MAO-B. A TG2-only
   gain is reported as TG2-specific and not adopted, which is the standard §5.13's withdrawn
   Mondrian recommendation set.

Secondary, reported but not decisive: cliff-pair ranking accuracy against the matched distant
control from §5.18, and sensitivity at specificity ≥ 0.80 under publication shift.

### Cost

Re-running the three fingerprint models with the new featuriser across all arms, seeds and targets is
the 2.065 h fingerprint subtotal — call it **≈ 2 h of CPU**, no GPU. Adding the count featuriser to
`src/data/chem.py` is a few lines beside the existing `morgan`.

### Drawbacks, stated in advance

- **The novelty bands lack common support.** §8 already records this for §5.14. The < 0.4 band holds
  25 TG2 molecules, so the primary endpoint rests on 25 molecules per fold at best, and its interval
  will be wide. This is the main reason criterion 3 exists.
- **Count fingerprints are higher-variance.** More distinct features on the same 631 molecules can
  hurt as easily as help, particularly for logistic regression.
- **It does not address the sensitivity collapse.** Proposal B targets unfamiliar chemistry, not the
  false-negative problem under publication shift; those are different failures and conflating them
  would be a mistake.
- **A negative result is likely and acceptable.** §5.10 found that nine specialised models failed to
  beat binary Morgan. The prior should be that representation changes do little on this dataset, and
  a clean null would still be worth recording, because the field's own benchmark literature
  recommends this comparison and nobody has published it on a TG2 dataset.

---

## Order and total cost

| step | cost | compute |
|---|---|---|
| Proposal A, tier 1 | ≈ 0.9 h | CPU |
| Proposal B | ≈ 2 h | CPU |
| Proposal A, tier 2 (only if tier 1 warrants it) | ≈ 3.9 h | GPU |
| Proposal A, tier 3 | ≈ 9.2 h | GPU, overnight |

**Recommended: A tier 1, then B, then decide about tier 2 — about 3 hours of CPU in total, no GPU.**
Proposal A comes first because it answers the question that was asked, and because its answer
changes how proposal B's result should be read: if the models are not overfitting, then a
representation change is the right kind of intervention; if they are, regularisation would come
first.

Nothing is started until this design is approved.
