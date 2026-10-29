# TG2 data discovery and screening feasibility (audited 2026-10-29)

Reproduce: `python scripts/audit_recovery.py`. Table: `results/unit_recovery.csv`. CPU, seconds.
Separate from the evaluation paper ([`paper.md`](paper.md)), which is unchanged by this work.

The evaluation phase ended with a specific, cheap recommendation: recover the kinact/Ki rows that
ChEMBL stores without units, because a 0.50-log inter-laboratory offset and six publications were
what defeated the covalent model ([`covalent.md`](covalent.md)). This phase acts on that, then asks
whether the result changes our ability to propose credible TG2 inhibitor candidates.

## 1. What the unitless rows actually are

118 kinact/Ki rows across seven publications carry no `standard_units`. Resolving the documents:

| ChEMBL doc | rows | publication |
|---|---|---|
| CHEMBL2086255 | 55 | Prime et al., *J Med Chem* 2012;55:1021 — GSK covalent TG2 inhibitors for Huntington's |
| CHEMBL3351998 | 48 | Klöck et al., *J Med Chem* 2014;57:9042 — dihydroisoxazole inhibitors of human TG2 |
| CHEMBL5230175 | 8 | RSC Med Chem 2023;14:277 — warhead optimisation |
| others (4 docs) | 7 | 1–2 rows each |

Two of these matter, and they matter for opposite reasons.

**Klöck 2014 is the prize.** Its 48 molecules are 3-bromo-4,5-dihydroisoxazoles descended from
ERW1041E — a *different warhead class* from the acrylamides that make up 169 of the 174 molecules
in our current kinetic set. Every one of the 48 is absent from our 631-molecule IC50 set, and every
one sits below 0.5 Tanimoto to it (median 0.275). The same paper also reports TG1, TG3 and FXIIIa
data for the same compounds on the same measure.

**Prime 2012 adds no chemistry.** Its 55 molecules all already carry an IC50 label in our set
(median max-Tanimoto 1.000) and 40 are acrylamides. Recovering it would add kinetic labels for
chemistry we already have.

## 2. Unit recovery, and what we refused to recover

Recorded in `src/data/kinetic_units.py`, which supplies missing unit strings and converts nothing;
the existing conversion then applies.

**Recovered — CHEMBL3351998 → `10^3/M/min`.** Klöck et al. report k_inh/K_i in **mM⁻¹ min⁻¹**,
stating ERW1041E's potency against human TG2 as 17 mM⁻¹ min⁻¹. Since 1 mM⁻¹ min⁻¹ = 10³ M⁻¹ min⁻¹,
this is exactly ChEMBL's existing `10^3/M/min` convention. The stored values for this document span
1.2–108 with a median of 10.4, matching that scale and the published figure, and matching the
13.2–130 range of rows filed under the explicit `10'3/M/min` convention elsewhere in the same file.

The assignment is **document-level, from the paper's stated unit plus a scale match — not verified
row by row**, because the paper's tables are embedded as images. That limitation is recorded in the
module and should be stated wherever these data are used.

**Refused — CHEMBL2086255.** The largest block, and we left it out. The article is not open access;
a later paper from the same group quotes k_inact/K_I in M⁻¹ min⁻¹, while the stored values here
(0.29–1460, median 80.5) would fit either M⁻¹ min⁻¹ or mM⁻¹ min⁻¹ depending on the series. A
three-order-of-magnitude error would be invisible in aggregate and would corrupt every model
trained on it. Since all 55 molecules already carry IC50 labels, the cost of refusing is kinetic
labels for known chemistry, not lost molecules. **This is the single highest-value manual step
remaining**: one person reading one table settles it.

## 3. What the recovery buys

| | before | after |
|---|---|---|
| Molecules with usable kinact/Ki | 174 | **219** |
| Murcko scaffolds | 70 | **94** |
| Publications | 6 | 7 |
| log₁₀(kinact/Ki) range | 1.60–5.00 | 1.30–5.00 |

All **45** new molecules are new chemistry, not duplicates: median max-Tanimoto **0.276** to the
IC50 set and **0.216** to the existing kinetic set, with all 45 below 0.5 against both. This is the
first addition in the project that materially extends the chemical space rather than the row count.

### A mechanism contrast now exists

| warhead | before | after |
|---|---|---|
| acrylamide | 169 | 169 |
| **dihydroisoxazole** | **0** | **45** |
| chloroacetamide | 3 | 3 |

Dominant-warhead share falls from **97% to 77%**. [`covalent.md`](covalent.md) could not attempt
mechanism classification because one electrophile accounted for essentially everything; two classes
of 169 and 45 is a real contrast.

*A correction this surfaced:* the dihydroisoxazole SMARTS in `scripts/audit_covalent.py` was
`C1CNO1`, a four-membered ring that cannot match a 4,5-dihydroisoxazole and matched 0 of the 48
Klöck compounds. The corrected five-membered pattern `C1=NOCC1` matches all 48. The earlier report
of "0 dihydroisoxazoles" was true of the old dataset but was being produced by a broken pattern.

### Paired selectivity, within one laboratory

| | before | after |
|---|---|---|
| TG2 & TGM1 | 2 | **47** |
| TG2 & FXIII-A | 2 | **10** |
| TG2 & TGM3 | 0 | 0 |

This is the structure [`selectivity.md`](selectivity.md) said was missing. The earlier selectivity
failure was driven by between-publication variation: AUROC 0.974 under a scaffold split collapsed
to 0.574 when publications were held out. These 47 TG2/TGM1 pairs come from **one laboratory, one
assay, one measure**, so a within-series selectivity analysis is now possible in a way it was not
before. The FXIII-A subset is 10 molecules — the paper notes FXIIIa was assayed on only a subset
because of limited enzyme availability — which is too few to model and enough only to describe.

## 4. Does this change what we can claim?

**Yes for mechanism, partly for selectivity, no for the IC50 novelty problem.**

- **Covalent efficiency modelling** becomes worth retrying: 219 molecules, 94 scaffolds, 7
  publications, two warhead classes. The paper-held-out protocol still applies, and the
  inter-laboratory offset has not gone away.
- **Within-series selectivity** becomes answerable for TG2 vs TGM1 at n=47. Note that TGM1 is the
  *wrong* antitarget for celiac safety — FXIII-A is the bleeding risk, and it still has only 10
  paired molecules. So this improves the science without improving the safety argument.
- **The IC50 classification set is unchanged at 631 molecules.** kinact/Ki is a rate constant and
  cannot be merged with a concentration. The 42-molecule, 6-positive novel stratum that limits every
  generalisation claim is exactly as it was.

**No candidate shortlist is produced in this phase, and none is justified yet.** The reasons are
unchanged from [`selectivity.md`](selectivity.md): we cannot predict selectivity in a way that
transfers across series, and ranking by predicted TG2 potency carries no selectivity information
(AUROC 0.454). A shortlist without a selectivity column would be misleading for a chronic
indication in otherwise-healthy people; a shortlist with one would be unsupported.

## 5. Screening libraries, surveyed not screened

For when the models justify it. Nothing here has been downloaded or scored.

| Source | What it offers | Fit for TG2 |
|---|---|---|
| **ChEMBL / PubChem known actives** | the 631 IC50 + 219 kinetic molecules | **measured, not candidates** — must be excluded from any "new hit" claim |
| **Enamine Covalent Fragment Library** (~8,480 cpds, incl. ~2,240 acrylamide and ~1,360 chloroacetamide sublibraries) | cysteine-reactive warheads with balanced reactivity | best mechanistic match: TG2 is a cysteine transferase and its known inhibitors are covalent. Counts are vendor-stated and must be re-checked against current SD files |
| **Enamine Covalent Screening Library** (~11,760) | larger, lead-like rather than fragment | second choice; more molecules outside our applicability domain |
| **ZINC-22** (Tingle et al. 2023, free, cartblanche22.docking.org) | multi-billion make-on-demand, substructure searchable | the practical route to *analogs of known actives*, which is where our models are valid |
| **Dihydroisoxazole analogs** | now characterised by 45 measured compounds | the Klöck series is celiac-relevant by lineage (ERW1041E) and newly represented in our data |

The applicability-domain result dictates the strategy. 99.2% of approved drugs fall below the 0.4
Tanimoto floor where TG2 models rank at chance, so repurposing is exhausted. The defensible search
space is **analogs within 0.5–0.7 Tanimoto of measured actives**, which is a ZINC-22 substructure
and similarity query, not a diversity screen.

## 6. A conservative screening workflow, for when it is justified

Specified now so it cannot be tuned after seeing results. Every stage is a filter, and the order
matters: cheap, hard exclusions first.

1. **Exclude everything measured.** Any molecule present in the 631 IC50 set, the 219 kinetic set
   or the off-target files is a known compound and can never be reported as a new hit.
2. **Applicability domain.** Keep only molecules with max Tanimoto ≥ 0.5 to a *training* active.
   Below 0.4 the TG2 models rank at chance; 0.4–0.5 is unvalidated. This is a hard gate, not a
   ranking term.
3. **Model agreement.** Require the tuned random forest and XGBoost to agree on the positive class.
   Disagreement is an abstention, per the selective-prediction result (error 15.5% → 6.5% at 40%
   coverage).
4. **Conformal set.** Require a singleton conformal prediction set at α = 0.1 — the one uncertainty
   method that held its coverage under scaffold shift (0.922 against 0.90).
5. **Chemical diversity.** Butina-cluster survivors and take at most one per cluster, so a shortlist
   is not ten analogs of one scaffold.
6. **Reactivity triage.** Flag warhead class explicitly. A promiscuous electrophile that inhibits
   TG2 is not a hit; PAINS and reactive-group filters apply, and every covalent compound carries a
   nonspecific-reactivity caveat by default.
7. **Selectivity.** Currently **cannot be predicted**. Until that changes, every compound carries
   "selectivity unknown — not counter-screened against FXIII-A" and no compound is described as
   selective.

Stages 1–6 are implementable today from existing code. Stage 7 is the blocker, and it is a data
problem rather than a modelling one.

## 7. Prospective validation plan

What a laboratory would need to measure, were a shortlist produced. This is a plan, not a request;
nothing here is ready to send to a collaborator.

**Primary assay.** Human recombinant TG2 transamidation, colorimetric or fluorimetric, with
Cbz-Gln-Gly (AL5) or Z-Glu(HMC)-Gly-OH as substrate — the substrates used by the publications in
our kinetic set, so results are comparable to the training data. Report IC50 at a stated enzyme and
substrate concentration.

**Covalent characterisation.** For any electrophilic compound, time-dependent inhibition to obtain
k_inact and K_I separately, reported as k_inact/K_I **with its unit** — this project lost 103
measurements to missing units, and that is an avoidable error.

**Mandatory counter-screen.** FXIII-A, in parallel, not as a follow-up. 103 of 132 TG2-potent
molecules with paired data also inhibit FXIII-A at 1 µM, so cross-reactivity is the expected
outcome and a compound without this measurement is uninterpretable for a chronic indication. TG1
and TG3 secondary.

**Nonspecific reactivity control.** Glutathione half-life, or an unrelated cysteine-enzyme
counter-screen, to separate genuine TG2 engagement from general electrophilicity.

**Scale.** 10–20 compounds, single-concentration triage then full curves on confirmed actives. A
plausible hit rate for a model operating inside its applicability domain is low, and the experiment
should be sized as a test of the model rather than as a discovery campaign.

**What would count as success.** Not a potent compound. A *calibrated* model: predicted-active
compounds confirming at a materially higher rate than predicted-inactive ones at matched chemical
similarity. That is a prospective test of everything in [`paper.md`](paper.md), and it is worth
doing even if no compound is interesting.

## 8. Limitations

- Unit recovery is document-level from a stated unit and a scale match, not row-by-row
  verification. If Klöck et al. used a different convention in a table than in their discussion,
  45 molecules would be wrong by a constant factor — which would shift that paper's intercept
  without changing within-paper rank order.
- The 47 TG2/TGM1 pairs are one laboratory's series. They support a within-series analysis and say
  nothing about transfer, which is the very failure mode this project documented.
- TGM1 is not the safety-relevant antitarget; FXIII-A is, and it has 10 paired molecules.
- No library has been downloaded, no molecule scored, no shortlist produced.
- Nothing here is evidence about TG2 inhibition in a cell or an organism.
