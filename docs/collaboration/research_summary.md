# Predicting TG2 / Factor XIII-A selectivity: a one-page summary

*Kareem Itani — MSc research project, 2026-10-29. Prepared for prospective collaborators.*
*Status: computational work complete; no laboratory work has been performed.*

## Context

Tissue transglutaminase 2 (TG2) deamidates gluten peptides into the forms that trigger the
autoimmune response in celiac disease. The target is clinically credible: **ZED1227**, a covalent
TG2 inhibitor, reduced gluten-induced mucosal damage against placebo in a randomized trial of 163
patients (Schuppan et al., *NEJM* 2021;385:35–45, doi:10.1056/NEJMoa2032441; the authors describe
it as a proof-of-concept trial). A separate programme compound,
GSK3915393, completed phase 1 (NCT04604795) and was removed from GSK's celiac pipeline without a
published reason.

Because any celiac therapy would be taken chronically by otherwise-healthy people, **selectivity
against Factor XIII-A matters as a safety gate** — FXIII-A crosslinks fibrin, and inhibiting it
causes bleeding. Most transglutaminases share a catalytic cysteine and most published TG2
inhibitors are covalent, so cross-reactivity is the expected case.

## What we did

An evaluation-methodology study of machine-learned bioactivity prediction, on TG2 and two
unrelated targets (CDK2, MAO-B). The work is deliberately conservative: its main output is a
catalogue of ways an evaluation protocol, rather than a model, can determine a published result.
Two of our own positive findings were withdrawn after auditing.

From that work we assembled **147 molecules with an exact IC50 against both human TG2 and human
FXIII-A measured within the same publication** — so every selectivity ratio is a within-study
comparison, not one spanning laboratories (we measure a 0.50-log inter-laboratory offset in
comparable data, which is larger than most selectivity differences of interest).

## Two findings

**1. Highly selective chemistry exists.** Across the 147 paired molecules the median
log₁₀(TG2/FXIII-A) is +0.15 — most known inhibitors are not selective — but the range extends to
+3.40, and **8 compounds exceed 100-fold selectivity for TG2**. The most selective measures 14 nM
against TG2 and 35,000 nM against FXIII-A.

**2. Selectivity is predictable within one chemical series, and only one.** A random forest on
Morgan fingerprints predicts the continuous log selectivity ratio with Spearman ρ = 0.42–0.57
within one 69-molecule series (CHEMBL2086255, Prime et al., *J Med Chem* 2012), robust across
scaffold splits, Butina cluster splits, and splits that remove near-analogs. A second 57-molecule
series looked equally good under a scaffold split (ρ = 0.52) but **collapsed to ρ = −0.01 under a
cluster split** — its apparent signal was near-identical analogs straddling the fold. Across the
two series, prediction fails entirely (ρ = 0.29 and −0.05).

## Limitations, stated plainly

- **One series is one observation.** The surviving result rests on 69 molecules from a single
  publication. It does not establish a general capability.
- **Chemotype mismatch.** That series comes from a Huntington's disease programme and consists of
  acrylamide Michael acceptors — not the dihydroisoxazole chemotype most relevant to celiac
  disease, which has only 8 paired FXIII-A measurements in public data.
- **Retrospective only.** Every number above is retrospective. No prediction has been tested
  against a measurement made afterwards.
- **Scaffold-held-out within one series still leaves analogs** on both sides of the fold, so even
  the surviving estimate is optimistic to an unknown degree.
- **Predicted ≠ demonstrated ≠ safe.** A predicted ratio is a model output; a measured ratio is
  biochemistry in a cuvette; neither is evidence of therapeutic safety.

## What we propose

A 16-compound prospective test of whether the model's predicted selectivity rank-orders measured
selectivity within that one series. **The model is the subject of the experiment, not the
compounds.** The design, inclusion criteria and success thresholds are fixed in advance
(see the accompanying proposal). We are not proposing drug candidates and have produced no
candidate shortlist.
