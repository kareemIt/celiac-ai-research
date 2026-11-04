# Can the full TG2 program be replicated on CDK2 and MAO-B? (audit run 2026-11-04)

Reproduce: `python scripts/audit_replication_feasibility.py`.
Table: `results/replication_feasibility.csv`. CPU, about a minute. Fits nothing.

Phase 0 of the three-target replication. It decides what is affordable **before** any GPU time is
spent, because the two replication targets are 4x and 12x TG2's size and the planned program is a
nine-entrant sweep plus a conformal experiment on each.

## 1. What already exists, and what the replication actually adds

[`second_target.md`](second_target.md) has already evaluated CDK2 and MAO-B under random,
scaffold and publication-held-out splits. That work is **not** being redone. What it does not have
is the model field and the uncertainty arm:

| component | TG2 | CDK2 | MAO-B |
|---|---|---|---|
| Random / scaffold / paper splits | yes | yes | yes |
| Random forest, logistic regression | yes | yes | yes |
| XGBoost, Chemprop, Chemprop+RDKit, CheMeleon, ChemBERTa-2 (frozen + fine-tuned) | yes | — | — |
| Paired model-vs-model tests, Holm-corrected | yes | — | — |
| Publication-aware conformal prediction (iid vs group calibration) | yes | — | — |

So the open questions are precisely the ones that need the expensive entrants: **does the model
ranking change under publication shift, do foundation models generalize better than fingerprints,
and does publication-aware calibration transfer?** None can be answered from the two-model results
already in hand.

## 2. Leave-one-publication-out does not scale, and is also the weaker design

TG2's publication arm held out one publication at a time — 11 folds over 394 molecules. CDK2 has 83
qualifying publications and MAO-B 209. Projections scale this project's **measured** per-fold
seconds from `results/sota_tg2.csv` by training-set size, per model, on the same RTX 5070 Ti:

| target | protocol | folds | usable | single-class | mean train | test size | sweep hours |
|---|---|---|---|---|---|---|---|
| CDK2 | leave-one-publication-out | 83 | 63 | **20** | 2,053 | 10–135 | **52.3** |
| CDK2 | grouped, K=10 | 10 | 10 | 0 | 1,870 | 201–210 | 7.6 |
| CDK2 | scaffold, 10 seeds (matched molecules) | 10 | 10 | 0 | 1,879 | 182–207 | 7.1 |
| MAO-B | leave-one-publication-out | 209 | 154 | **55** | 5,067 | 10–224 | **315.4** |
| MAO-B | grouped, K=10 | 10 | 10 | 0 | 4,582 | 503–512 | 18.5 |
| MAO-B | scaffold, 10 seeds (matched molecules) | 10 | 10 | 0 | 4,585 | 497–509 | 17.3 |

**Leave-one-publication-out is out of reach on MAO-B** — 315 hours, 13 days serial — and marginal on
CDK2 at 52 hours.

The more interesting finding is that it is also the *worse* protocol here, for two reasons that have
nothing to do with compute:

- **A quarter of its folds are unusable.** 20 of 83 CDK2 folds and 55 of 209 MAO-B folds are
  single-class — one publication reporting only potent compounds, or only inactive ones — and AUROC
  is undefined on them. Discarding them does not merely cost data; it selects *which* chemistry gets
  tested, keeping the publications with mixed outcomes and dropping the focused medicinal-chemistry
  series. The grouped protocol has no single-class folds on either target.
- **Its folds overlap, so its intervals are wrong.** Each molecule appears in several
  leave-one-out folds (1.6x on CDK2, 1.4x on MAO-B), which is the pooling artefact
  [`second_target.md`](second_target.md) measured narrowing bootstrap intervals by 15–18%. Grouped
  K-fold tests every molecule exactly once with no overlap, so the artefact cannot arise.

**Recommendation: grouped publication K-fold, K = 10**, with whole publications dealt to folds
largest-first by size and never by label. It holds out 201–512 molecules and 19–21 publications per
fold, still shares zero publications between train and test — the property that produced the entire
effect — and costs 7.6 h and 18.5 h instead of 52 h and 315 h.

TG2's 11-fold leave-one-out arm stays as published. To keep the two protocols comparable rather than
merely both present, CDK2 and MAO-B should be run under **both** protocols for the cheap models
(random forest, XGBoost, logistic regression — 6.5% of the sweep budget), which measures directly
how much the protocol change moves the number. The expensive entrants then run on grouped folds
only, and the measured protocol offset is reported alongside them.

> **Correction (2026-11-04).** This paragraph first put that check at "under 4 h for both targets
> combined". That was a prose estimate, not one computed by the script, and it was wrong: the three
> cheap models over every usable leave-one-out fold cost **3.4 h on CDK2 and 20.8 h on MAO-B**.
> MAO-B's 154 usable folds are the reason. The check therefore runs as three cheap models over
> CDK2's full 83 folds, and random forest alone over a **random sample of 50 MAO-B folds** (1.9 h),
> which estimates the protocol offset without buying all 154. The sample must be drawn at random
> rather than taken as the first 50: folds are indexed by sorted ChEMBL document id, which tracks
> publication date, so a prefix would be a sample of the oldest papers.

## 3. Matched molecules, as the hypothesis requires

Both publication arms and the scaffold arm run on the **same** molecule set: the 2,078 CDK2 and
5,091 MAO-B molecules from publications contributing at least 10 labelled molecules. This is TG2's
`scaffold_pub` vs `paper_lopo` design, and it is what makes "scaffold evaluation overestimates
performance on unseen publications" a statement about the split rather than about the data.

The scaffold arm drops to **10 seeds** from TG2's 20. TG2's own incumbent baselines were published
at 10 seeds, and halving the seed count costs about 7 h of the 50 h program while widening the
per-seed SD by roughly sqrt(2).

## 4. The conformal arm fits, with one change

`run_lopo_conformal.py` carves whole training publications out as a calibration set until 25% of
the training molecules are held. On grouped folds there are 75 (CDK2) and 188 (MAO-B) training
publications per fold, so reaching 25% needs roughly 19 and 47 of them — ample. TG2 managed this
with 5.6 publications per fold, so the replication is in a *better* position than the original, and
the group-vs-iid contrast is cleanly estimable on both targets.

The arm refits every entrant per fold, so it costs roughly a second publication-arm pass. Restricted
to the six entrants [`lopo_uncertainty.md`](lopo_uncertainty.md) used, that is about 5 h on CDK2 and
12 h on MAO-B.

## 5. Total, and where it can be cut

| stage | CDK2 | MAO-B | total |
|---|---|---|---|
| Scaffold arm, 10 seeds, 8 entrants | 7.1 h | 17.3 h | 24.4 h |
| Grouped publication arm, 8 entrants | 7.6 h | 18.5 h | 26.1 h |
| Protocol-offset check, 3 cheap models, leave-one-out | 1.1 h | 2.7 h | 3.8 h |
| Conformal arm, 6 entrants | ~5 h | ~12 h | ~17 h |
| **Total** | **~21 h** | **~51 h** | **~71 h** |

About three days of serial GPU time. Two reductions are available if that is too much, in the order
they should be taken:

1. **Drop CheMeleon and Chemprop+RDKit** from the two biggest arms. They are 57% of the per-fold
   cost (CheMeleon alone is 38%) and on TG2 they placed 7th and 8th of eleven. Keeping plain
   Chemprop preserves a from-scratch graph entrant, but note this removes the only *pretrained*
   graph model, so the foundation-model hypothesis would rest on ChemBERTa-2 alone. Saves ~29 h.
2. **Subsample MAO-B** to CDK2's scale (~2,000 molecules, whole publications kept intact), with the
   cheap models run at full size to demonstrate the subsample does not change the split effect.
   Saves ~25 h but weakens the dataset-size control in §6.

Neither is needed if three days of GPU time is acceptable, and the full field is the stronger result.

## 6. Mechanism analysis is affordable, and needs no new fits

The question of *why* performance drops — chemical similarity, assay differences, training-set size,
class balance, chemical series — is answered from the per-molecule predictions the arms already
write, plus CPU-only controls:

- **Similarity**: similarity-banded AUROC, already implemented in `analyse_sota.py`.
- **Training-set size**: the scaffold arm re-run at the grouped arm's training size, cheap models
  only. This is the control that separates "unseen publication" from "less training data".
- **Class balance**: per-fold positive rate against per-fold AUROC, from existing outputs.
- **Assay differences**: see the correction below — this control cannot be built.
- **Chemical series**: publication-level scaffold entropy against the fold's AUROC drop.

A publication-held-out split confounds *who measured it* with *how it was measured*, and nothing in
the existing three-target work separates them.

> **Correction (2026-11-04).** This section called the assay-held-out control "the important one"
> and "testable with the cheap models". It is **not testable**, and
> `scripts/audit_assay_confound.py` now establishes why: **0 of 380 CDK2 assays and 0 of 516 MAO-B
> assays span more than one publication.** ChEMBL defines an assay per paper, so assays are
> perfectly nested inside publications and an assay-held-out split *is* a publication-held-out
> split. The within-publication contrast that could estimate an assay effect with the laboratory
> held fixed exists for only **1 CDK2 publication (60 molecules) and 2 MAO-B publications (109
> molecules)** — far too little. The 786 and 1,420 distinct *descriptions* quoted above are mostly
> free-text variants of per-paper assays and do not imply crossed grouping; citing them as evidence
> of a testable control was the error.
>
> What is measurable, and is reported instead: the share of test molecules whose assay never
> appears in training. On the **same molecules**, a scaffold split leaves that at **0.3% (CDK2) and
> 9.4% (MAO-B)**, while the grouped publication split leaves it at **100% on both**. So the
> publication penalty is inseparably an assay penalty, and the honest conclusion is that
> ChEMBL-derived data cannot decompose it. This strengthens rather than weakens the practical
> recommendation — a publication-held-out split is the only one that withholds the measurement
> protocol too — but it forbids the claim that the decline is attributable to publication identity
> alone.

## 7. Limitations of this audit

- Runtime projections are **linear in training-set size**, scaled per model from TG2's measured
  folds. That is right for the gradient-trained entrants, where cost is epochs x batches, and
  approximate for the tree models, whose fit is closer to *n* log *n*. Treat them as
  order-of-magnitude; a single measured CDK2 fold should be timed before the full sweep is launched.
- Projections assume serial execution on one GPU and no failed runs.
- The grouped-fold assignment is one draw with a fixed seed. Publications are dealt largest-first,
  so the seed changes the assignment only where sizes tie — but ties are the common case here (83
  CDK2 publications span 32 distinct sizes, 209 MAO-B publications span 47), and re-dealing with
  another seed moves **35% of CDK2 and 57% of MAO-B molecules** between folds. Fold-assignment
  variance is therefore real and estimable; it is not estimated in this audit and the cheap models
  should be re-dealt across several seeds during the run to bound it.
- This audit does not revisit the provenance decisions of [`second_target.md`](second_target.md)
  (binding assays only, censored rows discarded, first-publication attribution). Those stand, and
  the last of them still means the publication penalty measured on all three targets is slightly
  optimistic.
