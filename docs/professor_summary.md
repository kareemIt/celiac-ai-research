# Research summary for supervisor review — one page

Kareem Itani · 2026-11-05 · full manuscript: [`paper.md`](paper.md) · corrections ledger:
[`audit_2026-11-05.md`](audit_2026-11-05.md)

**We have not discovered a celiac disease treatment and make no such claim.** TG2 is the
motivating application; the contribution is about how molecular AI is evaluated.

## 1. The original goal

Can machine learning help find small-molecule inhibitors of **transglutaminase 2 (TG2)**, the
enzyme that chemically modifies gluten peptides into the forms that trigger the immune response in
celiac disease? We began by asking whether a locally fine-tuned ~9B language model could rank
candidate molecules better than standard cheminformatics.

**It cannot, at this data scale.** On 631 molecules with measured TG2 potency over ten identical
splits, a tuned random forest on molecular fingerprints reached AUROC 0.898 ± 0.057 against
0.850 ± 0.058 for the fine-tuned model (Δ −0.049, 95% CI [−0.076, −0.021], p = 0.007). That answer
was quick; what followed is the actual research.

## 2. Models and targets

**Models:** random forest and logistic regression on Morgan fingerprints, XGBoost, Chemprop v2,
CheMeleon, ChemBERTa-2 (frozen and fine-tuned), MoLFormer-XL, a tabular foundation model, and
Qwen3.5-9B under QLoRA. **Targets:** TG2 (631 molecules), and two chemically unrelated replication
targets chosen in advance — **CDK2**, a kinase (2,646 molecules, 351 publications) and **MAO-B**, an
oxidase (5,840 molecules, 414 publications).

## 3. The three strongest findings

1. **How you split the data decided the conclusion more often than which model you used.** Holding
   out whole *source publications* — each contributing its own chemistry, measured in its own
   laboratory — costs more accuracy than the community-standard scaffold split: −0.068 AUROC on
   CDK2 and −0.076 on MAO-B, replicating on both.
2. **Most of that apparent collapse is an accounting artifact, not a failure of generalisation.**
   Publication-held-out results are conventionally averaged per publication, while random splits are
   averaged per molecule. Re-averaging the same predictions on the same unit recovers about **92%**
   of CDK2's 0.169 protocol gap to within 0.013 AUROC. The rest tracks the chemical novelty of the
   molecules a publication split happens to isolate. Crucially, **no assay in either dataset spans
   two publications**, so "new laboratory" and "new chemistry" cannot be separated with this data at
   all — a structural limit, not a gap we can close.
3. **Uncertainty estimates calibrated the usual way break under this shift.** Conformal prediction
   promised 90% coverage and delivered 78.4% on TG2, 81.1% on CDK2, 81.9% on MAO-B. Calibrating on
   held-out publications restores it (0.914 / 0.896 / 0.910). Worse, at a nominal 80% target the
   standard method collapses into *empty* prediction sets for up to 44.7% of CDK2 molecules — it
   reports confidence by refusing to answer.

## 4. Hierarchical conformal prediction: why it matters, and its limits

Our publication-grouped fix is empirical: it worked, but we could prove nothing. **Hierarchical
conformal prediction** (Lee, Barber & Willett, arXiv:2306.06342) has a finite-sample guarantee for a
molecule from an *unseen group*, which is exactly the drug-discovery question. We implemented
Theorem 1's split construction, verified it against 15 synthetic tests before reading any result,
and compared all three methods on identical held-out predictions.

**HCP held its nominal floor in 54 of 54 runs; our method in 22 of 54** (5 of 18 in the
theorem-faithful design). Paired per publication, HCP was ahead in all 54 comparisons.

Three practical limits, which matter as much as the result. **(a) It costs decisiveness:** confident
single-label predictions fell from 61.5% to 30.8% at a small calibration budget — guaranteed
coverage bought by saying "either" more often. The penalty nearly vanishes with a large budget
(56.0% → 52.9%), but on TG2 itself — our smallest dataset — HCP commits to a single label on only
7.7% of molecules, one in thirteen. The guarantee is affordable on a publication-rich dataset and
close to useless on a small one. **(b) It needs publications to spend:** the theorem requires at least
⌈1/α⌉−1 calibration groups, so a leave-one-publication-out protocol *cannot* support a guaranteed
calibration at any error rate, and CDK2 cannot at 10%. **(c) The guarantee is conditional** on the
fold's publication set and on our own randomisation of which publications calibrate — not on the
corpus being exchangeable, which it is not. HCP does not solve chemical distribution shift.

## 5. Contribution, and how it differs from existing work

**The methodological novelty is nil, by our own assessment.** Group-aware conformal prediction
exists; so does OOD evaluation of molecular models. The closest prior work (Rakhshaninejad et al.,
arXiv:2505.18890, 2025) asks a very similar question on drug–target data with different groupings.

What we add is empirical and specific: the **publication** as the grouping unit, with the finding
that assay and publication are *non-identifiable* in ChEMBL; a decomposition showing how much of a
reported publication-shift penalty is the aggregation unit rather than generalisation; and the design
constraint in 4(b), which we have not seen stated and which rules out the most intuitive protocol.

## 6. Limitations, and what validation would require

CDK2 and MAO-B **disagree in sign** on one secondary finding and we cannot resolve it with two
targets. Only the split construction of the theorem is implemented, so our claim is "HCP holds its
floor where ours does not", *not* "HCP is the best group-aware method". Chemical novelty is reported
as an association, never a cause — the novelty bands lack common support. The literature scan was
six targeted queries, not systematic.

**Wet-lab validation** would need 16 untested compounds from the one chemical series whose
selectivity model survives a cluster split, spanning the predicted range rather than its top, with
predictions committed in writing beforehand (preregistered in `docs/experimental_plan.md`;
success ≥ 0.5 Spearman, failure < 0.3), plus solubility and permeability measured in one laboratory
under one protocol. No laboratory has been contacted and no compound ordered.

## 7. Proposed venue, and the feedback we need

***Journal of Cheminformatics*** first — it publishes evaluation methodology at length and would not
require cutting the failure catalogue. *JCIM* is the more selective alternative, needing the
manuscript roughly halved. A workshop short paper on the calibration thread alone would be the
fastest route to feedback.

Four questions:

1. **Is the decomposition finding (3.2) the paper, or is it too deflationary?** It reinterprets a
   result the field reports as a generalisation failure. That may read as a correction rather than a
   contribution.
2. **Is "no methodological novelty, but a useful empirical constraint" a publishable framing**, or
   should we build something new first?
3. **Does the unresolved CDK2/MAO-B disagreement need a third replication target** before
   submission, or is reporting it as a bound on the claim sufficient?
4. **Venue and length** — is halving for *JCIM* worth it over *J Cheminformatics* at full length?

**Status:** ready for your review; not submission-ready. ~21,900 words (too long for any venue),
no venue chosen, figures not at publication quality. 406 automated tests pass, 113 headline numbers
recomputed from source, 19 corrections logged with evidence.
