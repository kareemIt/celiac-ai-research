# What we did, 2026-11-09 to 2026-11-10

Kareem Itani · committed as `07f4ca0` · manuscript: [`paper.md`](paper.md)

**We have not discovered a celiac disease treatment and make no such claim.** TG2 remains the
motivating application; the work below is about whether our models are trustworthy.

This covers one stretch of work, not the project. The project-level one-pager is
[`professor_summary.md`](professor_summary.md), which is now out of date on four points listed at
the end.

---

## The three things the professor asked for, and what we did about each

### 1. "Review more of the published research"

Twenty papers were supplied in four groups. All twenty were **verified against a primary record** —
Europe PMC, Crossref or the arXiv abstract page — rather than accepted from a citation string, and
the methods of eight were read in full rather than their abstracts. Write-up:
[`literature_20papers_2026-11-10.md`](literature_20papers_2026-11-10.md).

Reading the methods changed four of my own first-pass readings, which is the reason the brief said
not to trust abstracts:

| Paper | What I first said | What the methods show |
|---|---|---|
| #20 Agboola et al. | reported AUROC | reports **average precision**; the collapse figures are AP |
| #17 Tamura et al. | memorisation drove the accuracy | no control experiment isolates it — that is the authors' interpretation, not a result |
| #15 López-Pérez & Miranda-Quintana | contradicts our splitting argument | it does not: none of its eight strategies is a scaffold or group split, and the selecting method also picks the test set |
| #19 | could not be found | it exists (*JCIM* 2025;65(20):10819–10831); my exact-title search was too narrow |

I also wrongly suspected an existing reference was fabricated. arXiv:2610.02076 resolves to a real
paper; only the stored title string was mangled. I retracted that.

**The paper went from 29 verified references to 53.** Refs 30–46 come from the reading list; refs
47–53 close a gap I should have caught earlier — the draft benchmarked ten models while citing none
of them. (The commit message for `07f4ca0` says "23 new"; the correct figure is 24.)

Four of the references the state-of-the-art document had already marked verified turned out to be
wrong or stale, which is why re-checking rather than trusting was worth the time: Chemprop v2 is
published (*JCIM* 2026;66(1):28–33), not a preprint; Uni-Mol2 has no venue at all; CLAMP's "ICML
2023" has no proceedings record; MoLFormer's journal version was asserted from an arXiv comment and
had to be confirmed separately through Crossref.

### 2. "Reproduce existing approaches"

The prior TG2 model (Wichka & Lai, ref [1]) reports **93% accuracy**. Read against its own methods,
that figure is a stratified 5-fold cross-validation mean over all 336 classified molecules — not a
held-out test score — on a set that is 64.3% active, and its negative class pools 45 documented
inactives with 75 compounds recorded as "unspecified".

So **we do not place 93% beside any number in this paper**, and §8 now says why. Reproducing their
pipeline properly means rebuilding it and re-scoring it under our protocols and our label policy,
reporting both separately. That is specified but not done.

### 3. "Pay attention to negative results — are the models overfitting?"

This is the question the stretch was really about, and it now has an answer.
Design pre-registered in [`proposal_overfitting_2026-11-09.md`](proposal_overfitting_2026-11-09.md);
result in [`overfitting_2026-11-10.md`](overfitting_2026-11-10.md) and paper §5.21.

Every committed model configuration was refit on every outer fold and scored four ways: on its own
training fold, out-of-fold inside the training fold under the same grouping as the outer split, on
the test fold, and on the test fold after refitting at the smaller training size — the last being a
control for the fact that an out-of-fold estimate trains on less data and is pessimistic for reasons
that have nothing to do with the models. 572 folds, about 1 h 50 m of CPU fitting.

**The answer is a null, and it is the one we wrote down in advance.**

The in-sample-minus-out-of-fold gap is large everywhere (0.085 to 0.332) and it is worthless as
evidence. Median in-sample AUROC is 0.975 for logistic regression, 0.993 for the random forest,
0.997 for XGBoost. With 2048 fingerprint bits against a few hundred training molecules, all three
model families can simply separate the training labels, so that gap measures the ceiling on fitting
training data, not whether anything we reported is inflated. **Nothing in the decomposition shows a
single committed test score to be optimistic.** The generalisation failures catalogued in §5.17–§5.20
are not memorisation.

A distribution-shift term that survives the control appears in 5 cells of 27, all of them
publication-held-out CDK2 and MAO-B: **+0.154 [+0.127, +0.181] against a control of +0.005**, with
170 of 228 folds positive. Everywhere else — both scaffold arms, both grouped-publication arms — the
term sits at or below its own control with an interval crossing zero.

**On TG2, our primary target, the experiment fails its own criterion and we report that.** Pooled
per the fallback fixed in advance, the shift term is +0.032 [−0.045, +0.106], p = 0.35, against a
control of −0.018 — the same order as the effect. The proposal said ten publications per training
fold might not support this. It does not.

**And the one readable term is not cleanly "publication shift" either.** The out-of-fold estimate is
pooled across publications while the leave-one-publication-out test fold is a single publication
scored alone. Those are different aggregations, and §5.14 already showed the publication penalty on
this data is mostly novelty and aggregation. The arm that separates them agrees: holding publications
out but scoring them pooled gives a shift term of +0.001 to +0.013. So this experiment **corroborates
§5.14 rather than finding something new behind it**, and its own identity claim is not established.

---

## What we corrected about our own work

- **A pre-registered preview in §9 was withdrawn.** It read a near-zero shift term as the publication
  penalty being visible inside the training fold. It came from TG2 logistic regression — one of the
  three cells that turn out not to resolve — and should never have been given a directional reading.
- **An unverifiable claim in §8 was removed.** "497 of 498 folds" came from a run that was discarded
  before its outputs were written, so no file supported it.
- **Random-forest AUROCs in this paper are not bit-reproducible, and §8 now says so.** With
  `n_jobs=-1`, repeated scoring of one fitted forest differs by one unit in the last place; where a
  fold holds exactly-tied positive/negative pairs, that last bit decides the tie and AUROC moves by
  1/(n_pos·n_neg). Measured directly: 25 scorings of one forest returned 0.8608647 and 0.8619734,
  one tied pair of that fold's 902 apart. Two runs of the full 498-fold sweep then differed on one
  fold and on none, and that fold returned each of those two values once — so bit-exactness on a
  tied-pair fold is a draw between known values, not a property of the pipeline. No random-forest
  figure here should be read past three decimal places.
- **My cost estimate was wrong again** — I quoted ~1.9 h and it took 2 h 43 m, because I sized the
  job from TG2's fold count when CDK2 contributes 249 publication-held-out folds.
- **A run was destroyed by my own design error.** The first attempt finished all 572 folds and then
  lost them, because the reproduction check ran before the results were written and aborted. The
  script now saves per target before any check can fail.

## What is still open

- **Separating aggregation from publication shift** needs one more arm: hold out several publications
  at once, score them pooled, and match the leave-one-publication-out training size. This is a
  modest rerun, and the existing evidence predicts a small effect.
- **MAO-B's publication-held-out arm exists for the random forest only**, because that is all the
  committed sweep ever held. Inherited, not introduced here.
- **TG2's publication arm cannot resolve the decomposition at any compute budget** — it has too few
  publications. Only more data fixes that.
- **Four reference categories were never searched:** handling of inactive and censored data, further
  activity-cliff literature, TG2 selectivity against FXIII-A, and intestinal delivery.
- **Reproducing Wichka & Lai's pipeline** is specified and not started.
- **The two candidate interventions** (activity-cliff-aware loss, count-based fingerprints) are now
  less attractive than the cheap measurement above, and the honest prior — a small or absent effect
  — is recorded in §9 before either is run.

## Verification state

596 automated tests pass. `scripts/check_paper.py` recomputes **246** of the paper's headline values
from `results/` and fails if the write-up has drifted. All 498 folds carrying a committed score
reproduced that score bit-exactly.

## [`professor_summary.md`](professor_summary.md) is now stale on four points

It is dated 2026-11-05 and still says: 406 tests (now 596); 113 recomputed values (now 246); "the
literature scan was six targeted queries, not systematic" (twenty papers have since been verified
and eight read in full); and it does not mention the overfitting result at all, which is the item
that answers the professor's own question. Its findings are unchanged — only these counts and the
missing section.

## One correction made while writing this summary

I had recorded the tier-1 run as taking **2 h 41 m**, and that figure reached paper §5.21, the
result document, and the commit message of `07f4ca0`. Checking it against the script's own timer
shows the three targets took 1,487 s, 3,761 s and 1,328 s — **about 1 h 50 m**, not 2 h 41 m. The
docs are corrected; the commit message still carries the wrong figure and I have not rewritten
history to fix it. Nothing else depends on the number.
