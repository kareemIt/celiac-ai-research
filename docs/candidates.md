# Candidate screening: approved drugs as TG2 hypotheses (run 2026-10-27)

> **Nothing in this document is a discovery, a hit, or a treatment.** Every row is a model
> prediction about a molecule that has never, as far as these datasets record, been assayed against
> TG2. No compound here has been synthesized, tested, or validated. None of this is medical advice
> or a basis for any clinical decision.

Reproduce:
```bash
python scripts/fetch_screening_library.py --max-phase 4
python scripts/screen_candidates.py --top 12
```
Full scored library: `results/candidates_tg2.csv`; shortlist: `results/candidates_tg2_shortlist.csv`.

## Library and exclusions
ChEMBL approved drugs (max_phase 4): 4,225 records → 3,475 small molecules → **2,415 after
standardizing** (largest fragment, canonical SMILES, deduplicated) → **2,414 after removing any
molecule with an existing TG2 measurement** in either source. Only one approved drug was already in
the TG2 data, so overlap is not what limits this screen.

Approved drugs were chosen deliberately over a billion-compound virtual library: every structure is
real and has human safety data, so a prediction is a repurposing hypothesis a laboratory could
actually act on, rather than a measurement of how far the model will extrapolate.

## The headline result: the model cannot speak to this library

Applying the applicability domain from [`generalization.md`](generalization.md):

| max Tanimoto to training set | approved drugs | share | model's measured skill there |
|---|---|---|---|
| ≥ 0.50 | 2 | 0.1% | AUROC ≈ 0.90, reliable |
| 0.40 – 0.50 | 17 | 0.7% | AUROC ≈ 0.80, marginal |
| **< 0.40** | **2,395** | **99.2%** | **AUROC ≈ chance** |

**99% of approved drugs sit outside the domain where this model has demonstrated any ability to
rank TG2 potency.** Scoring them anyway would produce a confident-looking ranking with no evidence
behind it. This is the central finding of Phase 4 and it is a negative one.

The reason is chemical, not statistical. The TG2 training set is dominated by covalent
peptidomimetic inhibitors — acrylamides, chloroacetamides, 3-halo-dihydroisoxazoles — a chemotype
almost absent from the approved pharmacopoeia. Known TG2 inhibitor development is also in that
space: ZED1227 met its primary endpoint against placebo in 163 patients (NEJM
2021;385:35-45), while GSK3915393 reached phase 1 and was dropped when GSK left celiac
(Wichka & Lai, *CSBJ* 2024).

For completeness, the unfiltered model output: 35 of 2,414 drugs score p ≥ 0.5, 7 have unanimous
agreement across the three models, and 42 receive a conformal set of {potent} alone. **These counts
are reported only to be transparent; they are not candidates**, because almost all of them fall in
the < 0.40 band.

## Shortlist

Criteria, applied in this order: inside the applicability domain (≥ 0.40) → at least 2 of 3 models
calling potent → ensemble p ≥ 0.4 → Butina clustering at 0.65 distance to keep the set diverse.
Tier 1 additionally requires a conformal set of {potent} alone and no PAINS or Brenk alert.

**Tier 1 (passes every criterion): 0 compounds.**

**Tier 2: 1 compound.**

| ID | Name | p(potent) | Models agree | Spread | Tanimoto to train | Tanimoto to nearest known inhibitor | Novelty | QED | Alerts | Warhead |
|---|---|---|---|---|---|---|---|---|---|---|
| CHEMBL231813 | Telaprevir | 0.856 | 3/3 | 0.073 | 0.449 | 0.449 | 0.551 | 0.23 | 1 (Brenk) | α-ketoamide |

**Why it was prioritized.** Telaprevir is a covalent inhibitor of the HCV NS3/4A **cysteine-adjacent
serine** protease, carrying an α-ketoamide warhead. TG2 is a cysteine transferase whose potent
inhibitors are overwhelmingly covalent, so an electrophilic peptidomimetic is a mechanistically
coherent hypothesis rather than a fingerprint coincidence. It is the only approved drug that is
simultaneously inside the applicability domain and predicted potent by all three models, with low
disagreement (spread 0.073).

**Why to be skeptical of it.** Its similarity to training (0.449) sits in the *marginal* band, where
measured AUROC is ~0.80, not the reliable one. A peptidomimetic covalent inhibitor is exactly what
the model has learned to recognize, so it may be pattern-matching the chemotype rather than TG2
affinity specifically — the same prediction would likely be made for many covalent protease
inhibitors. QED 0.23 and a Brenk alert reflect that it is a large, greasy peptidomimetic. And
telaprevir was withdrawn from several markets over severe skin reactions, which makes it a poor
repurposing destination even if the prediction were right. It is most useful as a **probe of the
model's chemotype bias**, not as a drug candidate.

Nirmatrelvir (CHEMBL4802135, p = 0.655, 3/3 models, nitrile warhead) is the nearest miss, excluded
at Tanimoto 0.333 — just outside the domain. It is noted because it is the same mechanistic story.

## Liability flags computed
RDKit `FilterCatalog` PAINS (A/B/C) and Brenk alerts; QED; molecular weight, cLogP, TPSA, heavy-atom
count; and a SMARTS panel for covalent warheads relevant to a cysteine enzyme — acrylamide,
chloroacetamide, epoxide, maleimide, vinyl sulfone, 3-halo-dihydroisoxazole, aldehyde, nitrile,
boronic acid, α-ketoamide. 106 of 2,414 approved drugs carry at least one warhead; 1,222 are free of
PAINS and Brenk alerts.

A warhead is **ambivalent evidence** here: it supports the mechanism for a cysteine enzyme and
simultaneously raises selectivity and reactivity concerns. It is reported, not scored.

## Honest assessment
This screen did not produce a credible novel candidate, and the applicability-domain analysis
explains why: **approved-drug repurposing is the wrong strategy for TG2 with this training set.**
Reporting the 35 nominal p ≥ 0.5 hits as candidates would have been the easy and wrong thing to do.

The useful next libraries, in order:
1. **Analog expansion around known TG2 chemotypes** — enumerate or retrieve ChEMBL compounds within
   0.5–0.7 Tanimoto of known actives. Squarely inside the applicability domain, and the only setting
   where the measured AUROC of ~0.90 actually applies.
2. **Covalent-fragment libraries** (Enamine's covalent screening sets, which publish SMILES) matched
   to cysteine warheads. Chemically appropriate, and small enough to screen honestly.
3. **A targeted ChEMBL slice**: compounds assayed against other transglutaminases or cysteine
   proteases, where a cross-target model could be built and validated rather than assumed.

Doing any of these is a change of direction from drug repurposing and should be agreed first.
