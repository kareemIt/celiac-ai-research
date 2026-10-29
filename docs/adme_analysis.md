# Intestinal exposure and ADME: what the data holds (analysis 2026-10-29)

Reproduce: `python scripts/fetch_adme.py` then `python scripts/analyse_adme.py`.
Tables: `results/adme_measurements.csv`, `results/adme_summary.csv`. Log: `results/adme_analysis.log`.
CPU, seconds. **No model is fitted.** The evaluation paper and every previous dataset are
unchanged by this work; the new data lives in `data/raw/adme/`.

**Evidence labelling**, as in [`zed1227_analysis.md`](zed1227_analysis.md). **[E]** published
measurement with a citation, **[I]** our inference from it, **[U]** unknown or not publicly
available. Nothing here is a claim about any compound's clinical potential.

---

## 0. A correction to the previous report

[`zed1227_analysis.md`](zed1227_analysis.md) §4.2 states that "intestinal exposure is absent from
every dataset we hold… we have no data on it at all for any molecule." **That is wrong, and this
report withdraws it.**

It was wrong for a mechanical reason. Our six activity files were pulled from ChEMBL *by target*,
so they contain enzyme inhibition and nothing else. ChEMBL files ADME under the **molecule**, as
`assay_type=A`. Asking molecule-by-molecule for the 869 transglutaminase-family molecules we
already hold returns **380 measured ADME values on 113 molecules from 27 publications**, including
measured permeability for 37 molecules that appear in TG2 medicinal-chemistry papers.

The axis was there the whole time. We had not queried for it. That is the third time in this
project that a negative result turned out to be a property of our retrieval rather than of the
literature, and it is worth recording as such.

---

## Phase 1 — What ADME data exists

### 1.1 The audit

From 544 `assay_type=A` rows covering 131 of our 869 family molecules, 115 rows are potency or
assay-control readouts filed as type A (`Activity`, `CC50`, `EC50`, `Drug recovery` …) and 49 are
`Not Determined` placeholders with no value. Both groups are excluded and reported, leaving **380
measured values**. Molecule counts by property **[E, ChEMBL]**:

| property | molecules, TG2 papers | molecules, incidental papers | also in our 631-molecule TG2 set |
|---|---|---|---|
| half-life | 65 | 5 | 53 |
| permeability | **37** | 2 | 14 |
| lipophilicity (CHI) | 19 | 0 | 5 |
| chemical/metabolic stability | 17 | 0 | 13 |
| clearance | 12 | 6 | 14 |
| efflux ratio (MDR1) | 9 | 0 | 3 |
| plasma protein binding | 9 | 1 | 7 |
| in vivo PK | 4 | 3 | 1 |
| **aqueous solubility** | **0** | **0** | **0** |

"Incidental" means the molecule reached our files through a transglutaminase measurement but its
ADME comes from an unrelated programme (an Alzheimer's series, an AstraZeneca DMPK reference set).
Those values are real measurements; they are not evidence about TG2 chemistry, so they are counted
separately and never pooled.

**[I] Two things stand out.** Solubility — the property that most directly governs whether a drug
dissolves in the gut lumen at all — has **zero** measurements in this data. And the largest block,
plasma half-life, is **86 of 113 values censored** (`>8 hr`, `>24 hr`): it is mostly the statement
"did not degrade during the experiment", not a number.

### 1.2 Provenance of the permeability data

64 permeability values on 39 molecules, in **four unit systems** (`nm/s`, `10^-6 cm/s`,
`10'-6 cm/s`, `10'-4 cm/s`) and four assay systems. 45 are absorptive-direction and convertible to
10⁻⁶ cm/s (1 nm/s = 0.1 × 10⁻⁶ cm/s; a factor-of-ten slip here would reverse the whole
ranking, so the conversion is unit-tested). Basolateral-to-apical values are kept but never counted
as absorption. **Assay systems are not pooled** — a PAMPA Peff is passive diffusion through a lipid
film, a Caco-2 Papp includes transporters, and an MDCK-MDR1 Papp is dominated by P-gp efflux.

| publication | what it is | molecules with permeability | system |
|---|---|---|---|
| CHEMBL4219231 | Nε-acryloyllysine piperazides, *J Med Chem* 2018, [10.1021/acs.jmedchem.8b00286](https://doi.org/10.1021/acs.jmedchem.8b00286) | **19** | PAMPA |
| CHEMBL2086255 | Prime 2012 covalent TG2 inhibitors, *J Med Chem*, [10.1021/jm201310y](https://doi.org/10.1021/jm201310y) | **5** | Caco-2 |
| CHEMBL2203174 | lysine-based irreversible inhibitors for Huntington's, *ACS Med Chem Lett* 2012, [10.1021/ml300241m](https://doi.org/10.1021/ml300241m) | 3 | Caco-2, MDCK |
| CHEMBL5154739 | N-terminal variants of peptidomimetics, *Eur J Med Chem* 2022, [10.1016/j.ejmech.2022.114172](https://doi.org/10.1016/j.ejmech.2022.114172) | 2 | MDCK-MDR1 |
| CHEMBL5230179 | novel irreversible peptidic inhibitors, *RSC Med Chem* 2023, [10.1039/d2md00417h](https://doi.org/10.1039/d2md00417h) | 2 | MDCK-MDR1 |
| CHEMBL6114950 | diamino piperazine variants, *Bioorg Med Chem Lett* 2025, [10.1016/j.bmcl.2024.130078](https://doi.org/10.1016/j.bmcl.2024.130078) | 2 | MDCK-MDR1 |
| CHEMBL2150866 | 4-aminopiperidines for Huntington's, *ACS Med Chem Lett* 2012, [10.1021/ml3001352](https://doi.org/10.1021/ml3001352) | 1 | MDCK |
| CHEMBL6078638 | 7-aminoquinoline-5,8-diones, *J Med Chem* 2024, [10.1021/acs.jmedchem.4c02081](https://doi.org/10.1021/acs.jmedchem.4c02081) | 1 | PAMPA |
| CHEMBL4130383 | *incidental* — Alzheimer's cinnamoyl series, *Eur J Med Chem* 2017 | 3 | PAMPA |

Measured absorptive permeability across the TG2 papers spans **0.0 to 66 × 10⁻⁶ cm/s** — four
orders of magnitude. All of it is experimental; nothing in `results/adme_measurements.csv` is
predicted, and every row carries its assay text, document and units.

### 1.3 ZED1227's exposure properties

**ZED1227 has zero ADME rows in ChEMBL.** Everything below is hand-curated from the primary
literature into `data/curated/literature_adme.csv`, with the table it came from recorded.

All from Büchold et al. 2022, *Cells* 11:1667, Table 1
([10.3390/cells11101667](https://doi.org/10.3390/cells11101667)) **[E]**:

| property | value | conditions |
|---|---|---|
| Caco-2 Papp (A→B) | **< 1 × 10⁻⁶ cm/s** | 200 µM apical, HBSS pH 6.5 + 1% DMSO, propranolol 20 µM control |
| logD | 2.0 | shake-flask, octanol/PBS pH 7.4 |
| solubility, pH 7.4 | **100 µM** (0.05 g/L) | HPLC in PBS; the authors call this "rather low" |
| solubility, pH 1.1 | **48,900 µM** (25.9 g/L) | HPLC in dilute HCl |
| stability, pH 7.4 | **10% parent remaining** | 100 µM, 24 h, 37 °C |
| stability, pH 5.0 | 89% remaining | 100 µM, 24 h, 37 °C |
| simulated gastric fluid, pH 1.2 | 95% remaining | 100 µM, 24 h, 37 °C |
| simulated intestinal fluid, pH 6.5 | 82% remaining | 100 µM, 24 h, 37 °C |
| glutathione reactivity | 98% parent recovered | 10 mM GSH (160× excess), 48 h, 25 °C |
| cytotoxicity | none up to 1000 µM | Caco-2 and Huh7, BrdU 24 h and EZ4U 48 h |

Plus intestinal target engagement **[E]**: a TG2-bound-ZED1227-specific antibody gives
dose-dependent signal in villous enterocytes, ~2.7× higher than lamina propria, absent in placebo
and pre-treatment biopsies (Isola et al. 2023, *Int J Mol Sci* 24:10815,
[10.3390/ijms241310815](https://doi.org/10.3390/ijms241310815)).

**Still [U] for the clinical compound:** plasma protein binding, microsomal or hepatocyte
stability, CYP data, and any quantitative human PK (Cmax, AUC, half-life, bioavailability). The
*Cells* paper reports none of these, and we found no other primary source that does.

**[I] The stability profile is the most informative thing in the table, and it is not what a naive
reading predicts.** ZED1227 is *stable* at the pH of the stomach (95%) and small intestine (82%)
and *unstable* at blood pH (10% remaining after 24 h, which the authors attribute to an
intramolecular aza-Michael cyclization that kills the warhead). Its solubility is 489× higher at
gastric pH than at pH 7.4. So the design is not simply "a molecule too large to absorb": it is a
molecule that **dissolves and survives where the target is and self-inactivates where it would
cause harm**. Low Caco-2 permeability is one of four properties pointing the same way, not the
mechanism on its own.

### 1.4 Other TG2 inhibitors described as gut-restricted

**[U] We found none with comparable published measurements.** The closest literature is a
proof-of-concept study contrasting a cell-impermeable TG2 inhibitor (R281) with a cell-permeable
one (R283) in Caco-2 monolayers; it reports biological effects, not a permeability coefficient, so
it cannot be placed on the same axis. A 2024 study on TG2 conformational inhibitors reports only
*predicted* (QikProp) Caco-2 values, which do not belong in an experimental table. **ZED1227 is
the only TG2 inhibitor we could find with a measured permeability value and a stated
gut-restriction design intent.**

**[I] This matters for the comparison that follows: n = 1.** Any statement of the form "clinically
studied compounds differ systematically from the rest" is untestable here, and this report does not
make one.

---

## Phase 2 — ZED1227 against our compounds

### 2.1 The most selective compound we hold is highly permeable

Our best measured compound by FXIII-A selectivity — 14 nM TG2 / 35,000 nM FXIII-A, 2,500-fold,
CHEMBL2203451 — turns out to have a full ADME package **[E, ChEMBL]**:

| property | value | assay | source |
|---|---|---|---|
| permeability | **14.2 × 10⁻⁶ cm/s** (A→B) | wild-type MDCK, 10 µM | CHEMBL2203174 |
| permeability | 8.8 × 10⁻⁶ cm/s | PAMPA | CHEMBL4219231 |
| **MDR1 efflux ratio** | **118** | MDCK-MDR1, 10 µM | CHEMBL2203174 |
| intrinsic clearance | > 1386 mL·min⁻¹·g⁻¹ | human liver microsomes, 1 µM | CHEMBL2203174 |
| plasma half-life | > 24 h (human and mouse) | 1 µM, LC-MS/MS | CHEMBL2203174 |
| metabolic stability | 62% remaining at 60 min | mouse liver microsomes, 100 µM | CHEMBL4219231 |
| plasma stability | 65% at 15 min, 29% at 45 min | Wistar rat plasma | CHEMBL4024703 — **a review, secondary** |
| CHI | 29.4 | immobilised artificial membrane, pH 7.4 | CHEMBL4219231 |

Rank 2 (CHEMBL2203469, 62 nM / 67,000 nM, 1,081-fold): MDCK A→B **30.3 × 10⁻⁶ cm/s**, MDR1 efflux
ratio 7.1, hepatic CLint > 1386, plasma half-life > 8 h. Ranks 3–5 have **no ADME measurement in
ChEMBL**.

**[I]** Both compounds with exposure data come from the lysine-based SAR paper for **Huntington's
disease**, where central and systemic exposure was the design goal — the opposite requirement.
Their absorptive permeability is one to two orders of magnitude above ZED1227's upper bound.

### 2.2 The whole paired set, not just the top five

Of the 147 molecules with an exact IC50 on both enzymes, **7 have an absorptive permeability
measurement** and exactly **one** falls below ZED1227's 1 × 10⁻⁶ cm/s bound:

| molecule | Papp (10⁻⁶ cm/s) | system | TG2 IC50 | FXIII-A IC50 | fold |
|---|---|---|---|---|---|
| CHEMBL2089387 | **0.00** | Caco-2 | 1,300 nM | 1,100 nM | 1× |
| CHEMBL2086540 | 1.10 | Caco-2 | 180 nM | 1,500 nM | 8× |
| CHEMBL2086522 | 6.50 | Caco-2 | 110 nM | 310 nM | 3× |
| CHEMBL2203451 | 8.77 / 14.20 | PAMPA / MDCK | 14 nM | 35,000 nM | **2,500×** |
| CHEMBL2086528 | 10.20 | Caco-2 | 88 nM | 180 nM | 2× |
| CHEMBL2086480 | 13.60 | Caco-2 | 196 nM | 280 nM | 1× |
| CHEMBL2203469 | 30.30 | MDCK | 62 nM | 67,000 nM | 1,081× |

The one low value is recorded as *exactly* 0.00 nm/s, which is a detection limit and not a
measurement — and that molecule is 1,300 nM on TG2 with no selectivity whatsoever.

**[I] No compound in our data combines TG2 potency, FXIII-A selectivity and measured permeability
consistent with gut restriction.** Not one is close. This is a statement about what the data
contains, not a ranking failure: the candidate does not exist in our data, so no model could have
found it.

### 2.3 Descriptors, and what they do and do not show

ZED1227 against the 37 molecules with absorptive permeability measured in a TG2 publication
(descriptors computed with RDKit, **[I]** throughout — these are not measurements):

| descriptor | ZED1227 | median | min | max | molecules above ZED1227 |
|---|---|---|---|---|---|
| MW | 528.6 | 494.6 | 225.2 | 703.8 | 7 of 37 |
| cLogP | 1.4 | 2.4 | 0.4 | 4.6 | **33 of 37** |
| TPSA | 153.4 | 94.6 | 50.2 | 137.8 | **0 of 37** |
| H-bond donors | 3 | 2 | 0 | 3 | 0 of 37 |
| H-bond acceptors | 7 | 5 | 3 | 7 | 0 of 37 |
| rotatable bonds | 14 | 11 | 3 | 15 | 1 of 37 |

ZED1227 is the **most polar** molecule in the comparison and has the **lowest measured
permeability**, and it is among the least lipophilic. But it is not the heaviest (7 are heavier) and
not the most flexible (1 is more flexible), so weight and flexibility do not separate it.

**[I] This is a consistent story, not a tested one.** With 37 molecules drawn from 9 publications,
a descriptor difference and a publication difference cannot be told apart: each series brings its
own chemotype, its own assay and its own laboratory at the same time. The project has been burned
by exactly this confound before ([`selectivity.md`](selectivity.md), and the collapse of the
cross-paper signal in [`experimental_plan.md`](experimental_plan.md) §1). A TPSA/permeability
correlation computed across these nine papers would be a between-paper comparison wearing a
descriptor's clothing.

### 2.4 Low permeability does not establish gut restriction

Stated plainly, because the temptation runs in both directions:

- **A low Papp does not demonstrate gut restriction.** It is consistent with a drug sitting in the
  lumen, and equally consistent with one that never dissolves, never reaches the epithelium, or is
  degraded before it acts. ZED1227's case is strong because of *four* measurements pointing the
  same way plus direct human target-engagement histology — not because of its Papp.
- **A high Papp does not demonstrate absorption either.** Our rank-1 compound has an MDR1 efflux
  ratio of **118**: high passive permeability and heavy P-gp efflux coexist in the same molecule.
  What crosses the gut wall in vivo is not settled by either number alone.
- **No safety claim follows from any of this.** Low systemic exposure reduces one class of risk and
  says nothing about local toxicity, chronic dosing, or the acrylamide and Michael-acceptor
  reactivity concerns already recorded in [`covalent.md`](covalent.md).

---

## Phase 3 — Is this enough to model?

### 3.1 The sufficiency audit

Counting only molecules measured in a publication that also reports a transglutaminase activity:

| property | molecules | publications | Murcko scaffolds | with an exact TG2 IC50 | censored values |
|---|---|---|---|---|---|
| permeability | 37 | 9 | 24 | 14 | 0 of 62 |
| half-life | 65 | 8 | 44 | 48 | **86 of 113** |
| stability | 17 | 3 | 10 | 13 | 1 of 36 |
| lipophilicity | 19 | **1** | 10 | 5 | 0 of 19 |
| clearance | 12 | 6 | 11 | 9 | 5 of 14 |
| efflux ratio | 9 | 5 | 7 | 3 | 0 of 15 |
| protein binding | 9 | 3 | 8 | 7 | 0 of 10 |
| in vivo PK | 4 | 3 | 3 | 1 | 0 of 19 |
| solubility | **0** | 0 | 0 | 0 | — |

### 3.2 The verdict: no model

**No model is fitted, and that is a decision, not an omission.**

The permeability set is 37 molecules spread over 9 publications: 19 + 5 + 3 + 3 + 2 + 2 + 2 + 1 + 1.
This project's own evaluation work is the reason that is disqualifying. [`paper.md`](paper.md) §5
and [`second_target.md`](second_target.md) establish that on small literature-derived sets, a
random split inflates performance and only a **cluster or scaffold split** gives an interpretable
number. The largest single-publication, single-assay block here is **19 molecules** (PAMPA,
CHEMBL4219231). A cluster split cannot be built inside 19 molecules, and a split *across*
publications would pool PAMPA with Caco-2 with MDCK-MDR1 and compare laboratories rather than
molecules.

A model trained on this would produce a number. The number would measure which paper a molecule
came from. We have written that paper already; the point of having written it is not to do this.

**Descriptors being available is not a reason to model.** All six descriptors in §2.3 are
computable for all 631 molecules. That is precisely the condition under which the project's
documented failure mode occurs.

### 3.3 The measurements that would actually be worth having

Ranked by how much each would change what we can say, and all of them obtainable in a standard
ADME laboratory:

1. **Kinetic aqueous solubility at pH 1.2, 6.5 and 7.4.** Zero measurements exist for any molecule
   in our data, and it is the first gate on whether a compound is present in the lumen at all.
   ZED1227's 489-fold pH swing is the clearest single feature of its profile and we cannot compare
   anything to it.
2. **Caco-2 Papp (A→B and B→A, so the efflux ratio comes out) in one laboratory, one protocol, on
   compounds from a single series.** 37 molecules exist across 9 protocols; 16 in one protocol
   would be worth more than all of them.
3. **Stability in simulated gastric and intestinal fluid, and at pH 7.4.** ZED1227's profile is
   defined by this and we have it for 17 molecules from 3 papers, none head-to-head.
4. **Plasma protein binding**, where 9 molecules have a value.
5. **For the clinical compound specifically [U]:** protein binding and quantitative human PK. Not
   obtainable in a laboratory — it needs the sponsor to publish, or the EU CTR record
   (2020-004612-97).

**[I]** Items 1–4 are a routine commercial ADME panel on 16 compounds. They are the kind of
measurement a collaboration could realistically produce, and unlike more model training they would
add an axis rather than re-describe one. *No laboratory has been contacted and no compound has been
ordered.*

---

## Phase 4 — Where this leaves the research direction

### 4.1 Can our computational approach contribute to gut-restricted TG2 inhibitors?

**Not as a predictor of gut restriction — the data forbids it, as §3.2 sets out.** Two narrower
contributions are defensible:

- **As a selectivity filter inside one series.** Our single validated capability is ranking FXIII-A
  selectivity within CHEMBL2086255 (Spearman 0.42 under a cluster split). That series now turns out
  to have **measured Caco-2 permeability for 5 molecules and plasma protein binding for 6, from the
  same publication**. It is the only series in our data where a model-relevant property and a
  measured exposure property coexist under one protocol.
- **As a triage step, not a decision step.** Selectivity against FXIII-A is a safety gate that
  matters most when systemic exposure is *not* zero. A gut-restricted design weakens the argument
  for our own best capability — worth saying out loud, since it is an argument against the thing we
  are good at.

### 4.2 The recommended next experiment

**Add a measured-exposure arm to the already-preregistered 16-compound study, and do not change
anything else about it.**

[`experimental_plan.md`](experimental_plan.md) specifies a prospective, preregistered study: 16
compounds from CHEMBL2086255, selected to **span** the predicted selectivity range rather than its
top, with predictions committed in writing before assay. The recommendation here is one addition:

> on the same 16 compounds, in one laboratory and one protocol, measure kinetic solubility at pH
> 1.2 / 6.5 / 7.4 and Caco-2 Papp in both directions.

Why this and not something else:

- **It is independent validation, not retrospective accuracy.** The prediction is recorded first and
  the measurement decides. No ADME model is fitted, so there is nothing to retro-fit.
- **It creates the comparability we lack.** 16 molecules, one protocol, one series, measured
  alongside 5 existing Caco-2 values in the same series — the first block of ADME data in this
  project that a split could ever be built inside.
- **It is cheap and additive.** The compounds are already being chosen and plated for the
  selectivity assay. The solubility and permeability panel is a standard commercial service on the
  same plate.
- **It can fail informatively.** If selectivity and permeability turn out uncorrelated in the series,
  that is a real finding about the chemotype. If the whole series is permeable — which §2.1 suggests
  it may be, since its sibling series was built for systemic exposure — then the series is
  **wrong for celiac disease** regardless of how well we rank selectivity in it, and we will know
  that from 16 measurements rather than from speculation.

**[I] What this does not do:** it does not produce a candidate, it does not test gut restriction
(which needs dissolution, stability, target engagement and a disease endpoint), and it does not make
our models predictive of exposure. It converts the project's blind axis into a measured one for a
single series, which is the largest honest step available from here.

---

## Limitations

- **n = 1 on the clinical side.** ZED1227 is the only TG2 inhibitor with measured permeability and
  a gut-restriction design intent, so every comparison in Phase 2 is a single-compound contrast. No
  systematic difference between clinically studied and other TG2 inhibitors is claimed or testable.
- **Cross-assay throughout.** Four permeability systems (Caco-2, MDCK, MDCK-MDR1, PAMPA), nine
  laboratories, 15 years. Values are reported with their system attached and never pooled; even so,
  the comparison in §2.1 puts a wild-type MDCK number next to a Caco-2 bound, which is indicative
  and not equivalent.
- **One value is a review, not a primary source** (the rat plasma stability for CHEMBL2203451, from
  CHEMBL4024703, a 2017 *J Med Chem* review). It is flagged in the table and in the CSV.
- **One value is a detection limit** (Caco-2 Papp recorded as exactly 0.00 nm/s for CHEMBL2089387)
  and is treated as such, not as a measurement.
- **ZED1227's ChEMBL record contains no ADME at all**; its entire profile here is hand-curated from
  two papers into `data/curated/literature_adme.csv`. Hand curation is a transcription risk, and
  the table records the source table for each value so it can be checked.
- **Descriptors are computed, not measured.** Only the permeability, solubility, logD, stability and
  binding values are experimental.
- **ChEMBL and the primary literature only.** PubChem holds no systematic experimental ADME for
  these molecules; supplementary files behind publisher paywalls were not retrieved, so this is a
  floor on what exists, not a ceiling.
- **Nothing here evaluates any compound as a treatment.** Permeability, intestinal exposure and
  clinical benefit are three different things, and ZED1227 — histology improved, symptoms did not —
  is itself the demonstration that the first two do not guarantee the third.

## Sources

1. **[E]** Büchold C, et al. Features of ZED1227. *Cells* 2022;11(10):1667.
   doi:10.3390/cells11101667. — Table 1 (all ZED1227 physicochemistry, permeability and stability).
2. **[E]** Isola J, et al. The oral transglutaminase 2 inhibitor ZED1227 accumulates in the villous
   enterocytes… *Int J Mol Sci* 2023;24(13):10815. doi:10.3390/ijms241310815.
3. **[E]** ChEMBL `assay_type=A` activity records, fetched 2026-10-29 by `scripts/fetch_adme.py`;
   27 source documents, each recorded per row in `results/adme_measurements.csv`. The nine
   permeability sources are cited inline in §1.2.
4. **[E]** Schuppan D, et al. A randomized trial of a transglutaminase 2 inhibitor for celiac
   disease. *N Engl J Med* 2021;385(1):35–45. doi:10.1056/NEJMoa2032441. — the clinical context
   only; it reports no ADME.
5. **[U]** EU CTR 2020-004612-97 (phase 2b CEC-004) — not retrieved; would be the source for human
   PK if it is public anywhere.
