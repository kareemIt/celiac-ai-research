# ZED1227 as a clinical reference compound (analysis 2026-10-29)

Reproduce: `python scripts/analyse_zed1227.py`. Table: `results/zed1227_analysis.csv`.
CPU, seconds. The evaluation paper and all datasets are unchanged by this work.

**Evidence labelling.** Lines marked **[E]** are published measurements with a citation. Lines
marked **[I]** are our inference from them. Lines marked **[U]** are unknown or not publicly
available. Nothing here is a claim about any compound's clinical potential.

---

## Phase 1 — Verified published evidence

### 1.1 Identity and structure

| | |
|---|---|
| Name | ZED1227, also TAK-227 **[E]** |
| CAS | 1542132-88-6 **[E]** |
| ChEMBL | CHEMBL4081588 **[E]** |
| Formula / MW | C₂₆H₃₆N₆O₆ / 528.6 **[E, PubChem]** |
| SMILES | `CCC(CC)CNC(=O)CN1C=CC=C(C1=O)NC(=O)[C@H](CC/C=C/C(=O)OC)NC(=O)C2=CN=CN2C` **[E, PubChem]** |

A peptidomimetic built to resemble the preferred TG2 substrate Cbz-QQPL-OMe, carrying an
**α,β-unsaturated methyl ester** as the electrophile; Cys277 adds across the β-carbon, giving
irreversible covalent inhibition. A methylimidazole improves solubility at low pH **[E, ref 1]**.

**[I]** The warhead is *not* an acrylamide. Our kinetic dataset is 169 acrylamides, 45
dihydroisoxazoles, 2 chloroacetamides and 3 other — ZED1227's ester Michael acceptor is a fourth
class we have essentially no data on.

### 1.2 Biochemical potency and selectivity — the ~53 nM / >50,000 nM figures

**Verified as original measurements**, not secondary claims (Büchold et al. 2022, ref 1):

| enzyme | IC50 | fold vs TG2 |
|---|---|---|
| **TG2** | **53 nM** (IC90 300 nM) | — |
| TG1 | 24,863 nM | 469× |
| TG3 | > 50,000 nM | > 900× |
| TG6 | 6,441 nM | 122× |
| **FXIII-A2** | **> 50,000 nM** | **> 900×** |

**Assay conditions [E]:** fluorescent transamidation (DCC) measuring dansylcadaverine
incorporation into N,N-dimethylcasein; 0.81 µg/mL recombinant hTG2 (Zedira T022); 47.5 mM Tris,
9.5 mM CaCl₂, 9.5 mM glutathione, 2.4% glycerol, 190 mM NaCl, pH 8.0; 15.9 µM dansylcadaverine and
3.8 µM N,N-dimethylcasein; read 30 min at 37 °C. No TG2 pre-incubation step is described.

**Two caveats that matter [E]:** the authors state these are *apparent* IC50 values from a
four-parameter fit, because ZED1227 is irreversible — for a covalent inhibitor, IC50 depends on
pre-incubation time and is not a time-independent constant. And **they did not measure k_inact/K_I
for ZED1227**; they assume it is "at least in the same range" as a reference compound. So the
definitive covalent-efficiency constant for the clinical compound is **[U]**.

### 1.3 The 163-patient proof-of-concept trial (2021)

**[E, ref 2]** Randomized, placebo-controlled, 163 patients with celiac disease in remission,
6-week daily 3 g gluten challenge, doses 10/50/100 mg. Primary endpoint: change in
villus-height-to-crypt-depth ratio. Result: 0.44 (95% CI 0.15–0.73) at 10 mg, 0.49 (0.20–0.77) at
50 mg, 0.48 (0.20–0.77) at 100 mg — all significantly better than placebo. Intraepithelial
lymphocyte density and symptom scores also improved. The authors describe it as a proof-of-concept
trial and do not assign a phase number.

**Limitations [E/I]:** a gluten-*challenge* model in patients already in remission tests
prevention of induced injury, not treatment of established disease — a different question from the
one a patient asks.

### 1.4 The phase 2b CEC-004/CEL trial

**[E]** Randomized, double-blind, placebo-controlled, in patients with persisting symptoms and
mucosal damage *despite* a gluten-free diet. Part 1 reported **histological improvement at 50 mg
once daily** but **no significant symptom improvement across dosing groups**.

**[U]** We could not retrieve the trial's exact primary endpoint definition, per-arm numbers,
effect sizes or confidence intervals from a primary source. The UEG Week abstract and the sponsor's
releases were not accessible to us. **We therefore cannot confirm the characterisation that the
primary endpoint "failed"** — what is documented is a split result, histology improving and
symptoms not. Anyone relying on this section should obtain the abstract or the EU CTR record
(2020-004612-97).

**[E]** Part 2 is reported to have resumed enrolment in 2026, testing multiple doses under
controlled minimal gluten exposure. A further trial in non-responsive celiac disease is registered
(NCT07298343).

### 1.5 Pharmacokinetics, exposure and safety

| | |
|---|---|
| Caco-2 permeability | Papp < 1 × 10⁻⁶ cm/s — very low **[E, ref 1]** |
| logD | 2.0 **[E, ref 1]** |
| Systemic exposure | described only as "some systemic exposure", unquantified **[E→U, ref 1]** |
| Oral bioavailability, Cmax, AUC, half-life | **[U]** — no public quantitative human PK found |
| Intestinal target engagement | **[E, ref 3]** demonstrated: a TG2-bound-ZED1227-specific antibody shows dose-dependent signal in villous enterocytes, ~2.7× higher than lamina propria, absent in placebo and pre-treatment biopsies |
| Long-term safety | **[U]** — beyond the 6-week trial's adverse-event summary, no public long-term data |

**[I]** The low permeability is the design intent, not a flaw: a gut-restricted drug that acts on
luminal/epithelial TG2 and minimises systemic exposure. Ref 3 supports this — the drug is found
where the disease is.

---

## Phase 2 — ZED1227 against our datasets

### 2.1 It is already in our training data

| our dataset | exact match | stereochemistry-blind match |
|---|---|---|
| **TG2 IC50 set (631 molecules)** | **yes** | yes |
| kinetic set (219) | no | no |
| TG2/FXIII-A paired (147) | no | no |
| TGM1 / TGM3 / FXIII-A / guinea-pig sets | no | no |

**ZED1227 is CHEMBL4081588 and sits in our 631-molecule TG2 set, labelled potent.** Our data gives
**IC50 = 45 nM** from two independent ChEMBL documents (CHEMBL4024703, CHEMBL5729393) against the
published **53 nM** from a different assay — close agreement, and an independent sanity check on
our ingestion pipeline **[I]**.

### 2.2 Why it is missing from the selectivity analysis — and a data-quality finding

ZED1227 does have FXIII-A records in our files, but the value is **censored at "> 10 nM"**. Our
pipeline requires an exact IC50 on both enzymes, so it is excluded.

**[I] That ChEMBL bound is nearly useless and far weaker than the published value of
> 50,000 nM.** A ">10 nM" bound excludes almost nothing. This is a concrete example of the
curation gap our evaluation work keeps encountering: the single most clinically important
selectivity measurement for this target is, in the public database, recorded in a form that
carries almost no information.

### 2.3 Chemical neighbourhood

| our set | max Tanimoto to ZED1227 | mean of top 5 | shares its Murcko scaffold |
|---|---|---|---|
| TG2 IC50 set (631) | **1.000** (itself) | 0.647 | yes |
| kinetic set (219) | 0.302 | 0.214 | no |
| TG2/FXIII-A paired (147) | 0.358 | 0.298 | no |
| FXIII-A labelled (196) | 0.418 | 0.368 | no |

Excluding itself, ZED1227's chemistry is **remote from everything we model quantitatively**. The
paired selectivity set it would need to belong to sits at 0.358.

### 2.4 Physicochemistry: a deliberately unusual molecule

| property | ZED1227 | our TG2 set median | our 5th–95th pct |
|---|---|---|---|
| MW | 528.6 | 425.5 | 237–880 |
| cLogP | **1.4** | 2.5 | 0–4 |
| TPSA | **153.4** | 95.6 | 46–287 |
| H-bond donors | 3 | 2 | 0–6 |
| Rotatable bonds | **14** | 6 | 1–19 |

**[I]** Large, polar and very flexible — a peptidomimetic engineered *not* to be absorbed. Every
property sits inside our distribution's tails, so nothing flags it as an outlier, yet the
combination is unusual and purposeful in a way fingerprints do not represent.

---

## Phase 3 — What our models can and cannot say

### 3.1 No prediction is reported, and that is the correct outcome

**ZED1227 is in our training set.** Any TG2 prediction we produced would be memorisation, not
validation. Presenting it as evidence would be precisely the error the evaluation paper documents,
so the script refuses to produce a number and records why.

It is also excluded from the selectivity model's training data (censored FXIII-A), but it is
**0.358 Tanimoto** from that set's chemistry — below the 0.4 applicability floor at which every
model we tested ranks at chance. So a selectivity prediction would be unreliable *and*
uninterpretable.

**[I] ZED1227 cannot serve as external validation for our models.** This is a genuine negative: the
one compound with human efficacy data cannot test us, because it is simultaneously inside our
potency training data and outside our selectivity domain.

### 3.2 It is not more selective than what we already have

| compound | TG2 IC50 | FXIII-A IC50 | fold |
|---|---|---|---|
| **ZED1227 (published)** | 53 nM | > 50,000 nM | **> 900×** |
| our measured rank 1 | 14 nM | 35,000 nM | **2,500×** |
| our measured rank 2 | 62 nM | 67,000 nM | 1,081× |
| our measured rank 3 | 54 nM | 21,000 nM | 389× |

**[E/I]** Our dataset already contains measured compounds with comparable or greater TG2 potency
*and* comparable or greater FXIII-A selectivity. **Neither potency nor selectivity is what
distinguishes the clinical compound.**

### 3.3 What actually distinguishes a clinical TG2 inhibitor

**[I], and this is the central conclusion of the analysis.** The properties that took ZED1227 into
humans are ones our datasets do not contain and our models therefore cannot optimise:

| property | in our data? | ZED1227 |
|---|---|---|
| TG2 potency | yes (631 molecules) | 53 nM — matched by our compounds |
| FXIII-A selectivity | partly (147 pairs) | > 900× — exceeded by our compounds |
| **Gut-restricted delivery** | **no** | Caco-2 Papp < 1 × 10⁻⁶ cm/s, by design |
| **Human intestinal target engagement** | **no** | demonstrated in enterocytes, ref 3 |
| **Oral dosing and tolerability** | **no** | 6 weeks at up to 100 mg |
| **Activity in disease, not just enzyme** | **no** | mucosal histology endpoint |

Our models rank molecules on one axis — enzyme inhibition — that the clinical compound was
*already competitive on without our help*. The differentiator was pharmaceutics and human biology.

### 3.4 Could our model optimise within a relevant series?

**[I]** Only in a narrow sense, and not in ZED1227's series. Our one validated capability is
ranking FXIII-A selectivity inside the CHEMBL2086255 acrylamide series (Spearman 0.42 under a
cluster split). ZED1227 belongs to CHEMBL5729393 — **the series whose apparent selectivity signal
collapsed to −0.013 under the same test**. We have no validated capability in the clinical
compound's chemotype.

---

## Phase 4 — Where this leaves the research direction

### 4.1 Is ZED1227 chemistry a better direction than our acrylamide series?

**More relevant, and currently less tractable.** More relevant because it is the chemotype with
human efficacy, gut-restriction by design, and a celiac indication rather than a Huntington's one.
Less tractable because our only validated model capability is in the other series, the ZED1227
series failed our robustness test, and its defining properties are not in our data.

**[I]** Switching chemotype on relevance alone would mean abandoning the one thing we can do for
the one thing we would like to do. The honest position is that neither series is currently a route
to a candidate, and the limiting factor is not the model.

### 4.2 The most important remaining gaps

1. **Intestinal exposure is absent from every dataset we hold.** It is the property that made
   ZED1227 a drug rather than a tool compound, and we have no data on it at all for any molecule.
2. **No quantitative human PK or long-term safety is public [U]**, so even a well-predicted
   molecule could not be assessed against the clinical benchmark.
3. **The FXIII-A record for the clinical compound is a ">10 nM" bound** where the literature says
   > 50,000 nM — the public data is far worse than the published knowledge.
4. **k_inact/K_I for ZED1227 was never measured [U]**, so covalent efficiency cannot be compared
   with the 219-molecule kinetic set.
5. **Histology improved while symptoms did not**, in patients on a gluten-free diet. **[I]** That
   gap is about disease biology, not enzyme chemistry, and no molecular model addresses it.

### 4.3 Recommended next experiment

**Build a gut-restriction axis into the data, starting with what is already measurable.**

Concretely: assemble permeability and related ADME data (Caco-2/PAMPA Papp, TPSA, logD, molecular
flexibility) for the TG2 molecules we already hold, from the primary literature where it exists,
and ask whether anything distinguishes compounds that progressed toward the clinic from those that
did not. This is a data-collection experiment, not a modelling one, and it targets the axis the
whole project has been blind to.

**[I]** It beats the alternatives: retraining on the same potency data cannot produce new
information; screening a new library would rank on the axis that is already saturated; and more
selectivity modelling optimises a property our existing compounds already exceed the clinical
benchmark on.

---

## References

1. **[E]** Büchold C, Hils M, Gerlach U, Weber J, Pelzer C, Heil A, Aeschlimann D, Pasternack R.
   Features of ZED1227: the first-in-class tissue transglutaminase inhibitor undergoing clinical
   evaluation for the treatment of celiac disease. *Cells* 2022;11(10):1667.
   doi:10.3390/cells11101667. PMID 35626704.
2. **[E]** Schuppan D, Mäki M, Lundin KEA, et al. A randomized trial of a transglutaminase 2
   inhibitor for celiac disease. *N Engl J Med* 2021;385(1):35–45. doi:10.1056/NEJMoa2032441.
   PMID 34192430.
3. **[E]** Isola J, Mäki M, Hils M, Pasternack R, Viiri K, Dotsenko V, Montonen T, Zimmermann T,
   Mohrbacher R, Greinwald R, Schuppan D. The oral transglutaminase 2 inhibitor ZED1227 accumulates
   in the villous enterocytes in celiac disease patients during gluten challenge and drug
   treatment. *Int J Mol Sci* 2023;24(13):10815. doi:10.3390/ijms241310815. PMID 37445994.
4. **[E]** PubChem compound record, name "ZED1227" (CAS 1542132-88-6) — structure, formula, mass.
5. **[E]** ChEMBL CHEMBL4081588; source documents CHEMBL4024703 and CHEMBL5729393.
6. **[E]** ClinicalTrials.gov NCT07298343 (ZED1227 in non-responsive celiac disease).
7. **[E, secondary]** Celiac Disease Foundation, "Future Therapies" — phase 2b Part 1 outcome
   summary and Part 2 status. *Secondary source; primary trial report not retrieved.*

## Limitations of this analysis

- Phase 2b outcomes rest on a **secondary source**; the primary abstract was not accessible, and we
  do not assert that the primary endpoint failed.
- All biochemical comparisons are **cross-assay**. ZED1227's 53 nM comes from a pH 8.0 DCC assay;
  our dataset's values come from several different assays, and apparent IC50 for an irreversible
  inhibitor depends on pre-incubation.
- **No prediction was made for ZED1227**, by design.
- Physicochemical descriptors are computed, not measured, apart from the published Caco-2 and logD.
- Nothing here evaluates any compound as a treatment. Enzyme inhibition, intestinal target
  engagement and clinical benefit are three different things, and ZED1227 itself demonstrates that
  the first two do not guarantee the third.
