# Collaboration proposal: prospective test of within-series TG2/FXIII-A selectivity prediction

*Kareem Itani — MSc research project. Draft 2026-10-29. **No laboratory has been contacted.***
Companion to [`research_summary.md`](research_summary.md); full methods in
[`../experimental_plan.md`](../experimental_plan.md).

## 1. What this study is, and is not

| | |
|---|---|
| **It is** | a test of whether a computational model's predicted TG2/FXIII-A selectivity rank-orders measured selectivity across 16 untested analogs of one defined series |
| **It is not** | a search for drug candidates, a celiac disease therapeutic programme, or a test of any compound's clinical potential |
| **The subject of the experiment** | the model |
| **Expected outcome under the null** | failure, and that is a publishable result |

We have produced **no candidate shortlist** and are not asking anyone to evaluate one. If the study
succeeds, the deliverable is a validated *method* for ranking selectivity inside a series a
medicinal chemist is already working on — a tool for lead optimisation, not for discovery.

## 2. Scientific rationale

Within one 69-molecule series the model predicts the continuous log selectivity ratio at Spearman
ρ = 0.42 under a Butina cluster split (95% CI [0.171, 0.625]), stable at 0.56 under scaffold
splits and 0.563 when training molecules above 0.8 Tanimoto to the test fold are removed. The
protocol's own shuffled-label null is −0.230, so the observed values sit well clear of chance.

That is the sole basis for the study, and it is a retrospective result on one publication's
chemistry. A prospective test is the only way to learn whether it means anything.

## 3. Required experimental capabilities

| Capability | Why it is needed | Non-negotiable? |
|---|---|---|
| **Human TG2 transamidation inhibition**, full dose–response | the potency axis | yes |
| **Human FXIII-A inhibition**, full dose–response, **run in parallel** | the selectivity axis. Parallel rather than sequential so both share a baseline; a counter-screen run later against a different reference reintroduces the inter-laboratory offset the whole design avoids | **yes** |
| **Covalent kinetics**: k_inact and K_I fitted separately, k_inact/K_I reported **with units** | the series is covalent; IC50 alone is pre-incubation-dependent | yes for confirmed actives |
| **Electrophile stability**: glutathione half-life (e.g. DTNB assay) | separates genuine TG2 engagement from general electrophilicity — essential for an acrylamide series | **yes** |
| LC-MS compound QC | confirms identity and purity before any conclusion | yes |
| Cellular assay | *out of scope for this study* | no |

FXIII-A requires activation (thrombin/Ca²⁺) before assay, which is an extra step and a known source
of between-laboratory variation — one reason we prefer a group already running it.

**Assay formats** (any is acceptable; comparability with our training data is a bonus, not a
requirement): colorimetric Cbz-Gln-Gly with GDH coupling; fluorogenic Z-Glu(HMC)-Gly-OH;
biotin-cadaverine incorporation into N,N′-dimethylcasein with streptavidin-HRP; fluorescence
anisotropy.

## 4. Study design, preregistered

**Compounds.** 16 untested analogs of CHEMBL2086255, selected by criteria fixed before any
compound is chosen: Tanimoto ≥ 0.5 to a training molecule; never previously measured against TG2
or FXIII-A in any source we hold; commercially available or ≤ 3 synthetic steps; PAINS-clean with
no second electrophile; adequate predicted solubility. Sampling takes **4 compounds from each of 4
equal-width bins spanning the full predicted selectivity range**, with at most 2 per Butina
cluster.

Selecting only the top predictions would confound a good model with an easy series. A spanning
design measures rank correlation, which is the actual claim, and is the only design under which a
negative result is interpretable. **Predictions are committed in writing before any assay runs.**

**Controls.** Potency positive (characterised TG2 inhibitor from the training set); selectivity
positive (one of the 8 measured >100-fold selective compounds, confirming the counter-screen can
detect selectivity when present); negative (a measured non-selective compound); vehicle (DMSO) and
no-enzyme blanks on every plate.

**Endpoints and thresholds.**

| outcome | criterion |
|---|---|
| Success | Spearman ρ ≥ 0.5 between predicted and measured log selectivity, **and** ≥ 1 compound confirmed >10-fold TG2-selective |
| Partial | ρ ∈ [0.3, 0.5), or selectivity confirmed without rank correlation |
| Failure | ρ < 0.3 |
| Void | positive controls fail, or >25% of compounds show glutathione half-life below the assay window |

**Power, stated in advance.** n = 16 gives roughly ±0.4 on a Spearman confidence interval. It can
distinguish ρ ≈ 0.6 from ρ ≈ 0 and nothing finer. The retrospective estimate is ρ = 0.42 with a
wide interval, so a prospective result below 0.3 is possible even if the model is genuinely useful.
The study is sized to be informative either way, not precise.

## 5. Feasibility, cost drivers and timeline — **all estimates, clearly labelled**

We have run no laboratory work and have not requested quotations. Everything in this section is an
**estimate from published method descriptions**, not a costing. Figures should be replaced by the
collaborating laboratory's own.

**Likely cost drivers, in rough order of magnitude**

1. **Compound sourcing** — probably the largest and most uncertain item. Catalogue availability of
   analogs within 0.5–0.7 Tanimoto of a 2012 medicinal-chemistry series is genuinely unknown until
   searched; if few are purchasable, custom synthesis dominates both cost and schedule.
2. **Human FXIII-A** — plasma-derived or recombinant, plus thrombin for activation. Typically more
   expensive per assay than TG2 and the reason counter-screens are often skipped.
3. **Human recombinant TG2** — widely available; modest.
4. **Substrate and consumables** — modest.
5. **Personnel time** — the dominant real cost in an academic setting.

**Compound sourcing challenges, specifically**

- The series is from 2012; some analogs may be discontinued or available only in milligram
  quantities at long lead times.
- Acrylamides can have limited shelf stability; fresh material and DMSO stock QC matter.
- Stereochemistry must be confirmed — our dataset collapses some stereoisomers, which is a known
  limitation of the underlying data rather than of the supplier.
- **Risk:** if fewer than 16 eligible compounds are purchasable, the bins must be filled by
  synthesis or the design reduced. Shrinking n below ~12 would make the rank correlation
  uninformative, and we would rather postpone than run an underpowered study.

**Approximate timeline — estimate only**

| phase | estimate |
|---|---|
| Compound selection and written preregistration | 1–2 weeks (computational, our side) |
| Sourcing and QC | 4–12 weeks, dominated by availability |
| Assay development / adaptation, if the counter-screen is not already running | 2–4 weeks |
| Measurement (16 compounds + 3 controls, two enzymes, kinetics, GSH) | 2–4 weeks of bench time |
| Analysis and write-up | 2 weeks |

**Unknowns we cannot estimate:** compound prices and availability; enzyme lot-to-lot variability;
whether the laboratory's existing FXIII-A protocol is directly usable; institutional overheads;
whether any compound proves insoluble at assay concentrations.

## 6. What each side brings

**We provide:** the preregistered design and analysis plan; the model and its predictions,
committed before measurement; the 147-molecule audited paired dataset; an applicability-domain
analysis; a robustness audit that has already discarded half of our own positive result; and all
analysis code and documentation.

**We ask for:** paired TG2/FXIII-A measurement on 16 compounds plus controls, covalent kinetics on
confirmed actives, and a glutathione stability readout — plus a medicinal chemist's judgement on
compound selection, which a fingerprint model cannot supply.

**Authorship and data.** We propose joint authorship with the measuring laboratory, and that all
data and code be published regardless of outcome, including a negative one. We have no commercial
interest in any compound and seek none.

## 7. Candidate laboratories — **researched, not contacted**

Contact routes below are official institutional pages verified at the time of writing. **No
personal email addresses are listed where we could not confirm them from an official directory,
and none should be guessed.** Anyone approaching these groups should re-verify details first.

### Best capability match — Keillor group, University of Ottawa

Department of Chemistry and Biomolecular Sciences, D'Iorio Hall, 10 Marie Curie, Ottawa,
ON K1N 6N5, Canada. Group site: `https://mysite.science.uottawa.ca/jkeillor/`
Departmental contact (general): `chmchair@uOttawa.ca`

Why: this group publishes TG2 covalent inhibitors with k_inact/K_I measured under Kitz–Wilson
conditions, works on **acrylamide** chemistry (the same chemotype as our surviving series), has
published **Factor XIIIa** alongside TG2, and has used a **DTNB glutathione-reactivity assay**.
That is all three required capabilities in one laboratory. Several of their papers are already in
our training data (RSC Med Chem 2022 13:413; Eur J Med Chem 2022 232:114172; RSC Med Chem 2023
14:277), which also makes them a critical audience — their own measurements are part of what we
modelled.

### Best disease-area match — Khosla laboratory, Stanford University

Profile: `https://profiles.stanford.edu/chaitan-khosla`

Why: the central academic group on TG2 in celiac disease and the origin of the dihydroisoxazole
series (ERW1041E lineage) that is chemically most relevant to the indication. Their 2014 paper
(*J Med Chem* 57:9042) profiles TG1, TG3 and FXIIIa alongside TG2. Caveat: our surviving model
result is **not** in their chemotype, so a collaboration would more likely be about extending the
work to dihydroisoxazoles than about running this specific study.

### Commercial / reagent route — Zedira GmbH, Darmstadt, Germany

Website: `https://zedira.com`

Why: a transglutaminase specialist that developed ZED1227 and supplies transglutaminase enzymes,
substrates and assays. Relevant as a **reagent source**, and as the group with the deepest
commercial expertise in this exact enzyme family. Caveat: a commercial developer with active TG2
interests has obvious incentive considerations, and any approach should be about reagents or
fee-for-service measurement rather than shared unpublished predictions.

### Contract research organisations

Fee-for-service enzymology CROs can run dose–response and covalent kinetics. We have not
identified a specific CRO offering a validated **human FXIII-A** counter-screen, and that — rather
than the TG2 assay — is the capability to ask about first. Any CRO enquiry should request the
FXIII-A protocol and activation conditions before scoping.

## 8. Disclosures carried into every conversation

1. The model works in **one series**, which **collapsed to chance in a second**, and fails across
   series entirely.
2. That series is **acrylamides from a Huntington's programme**, not the celiac-relevant
   chemotype — a mismatch we chose knowingly, because it is where the model can be tested.
3. All supporting evidence is **retrospective**; nothing has been tested prospectively.
4. **n = 16** is small, and the confidence interval on the result will be wide.
5. Predicted selectivity is **not** demonstrated selectivity, and demonstrated biochemical
   selectivity is **not** therapeutic safety.
6. We are a single MSc project with no laboratory of our own and no funding attached to this
   proposal.
