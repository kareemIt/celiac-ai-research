# Can the paired TG2/FXIII-A data support selectivity modelling? (audited 2026-11-04)

**No — and two of the numbers this project has been quoting need qualifying.** The 147 pairs are
clean on units, species and assay type, but they are not clean on *assay format*, and the pooled
selectivity statistic is a mixture of two drug-discovery programmes pointing in opposite
directions. The one robust new finding is negative and useful: **TG2 and FXIII-A potency are
positively correlated, so ranking candidates by predicted TG2 potency enriches for
cross-reactivity.**

Reproduce: `python scripts/audit_selectivity_data.py`. CPU, seconds, fits nothing.
Tables: `results/selectivity_data_audit.csv`, log `results/selectivity_data_audit.log`.
Tests: `tests/test_selectivity_data_audit.py` (15 cases).

This is a feasibility audit, not a model. It extends [`selectivity.md`](selectivity.md) (binary
labels at 1 µM, answer "no") and `scripts/audit_fxiiia.py` (continuous within-paper ratios, answer
"no"). Neither asked whether the paired measurements are commensurable in the first place. They
partly are not. It reuses `audit_fxiiia.paired()`, so both analyses see exactly the same 147 rows.

## 1. Provenance: units and species are clean

| check | result |
|---|---|
| pairs | 147 molecules, 7 publications, each pair measured on both enzymes **inside one publication** |
| units | `nM` on every IC50 row of both enzymes — no conversion enters the ratio |
| assay type | `B` (binding) throughout |
| organism | *Homo sapiens* throughout |
| censoring | only `standard_relation = "="` is paired; `>` rows were already excluded upstream |
| ChEMBL "outside typical range" | **0 pairs** |
| ChEMBL potential-duplicate flag | 1 pair |

Nothing here needs fixing. The within-publication pairing rule also cancels the ~0.50-log
inter-laboratory offset measured in [`covalent.md`](covalent.md), which is why it was adopted.

## 2. Assay format: 68 of 147 pairs are not clean comparisons

The pairing rule is "same molecule, same publication". A publication can run the two enzymes in
different assay formats, and these formats do not measure the same thing: **amine incorporation**
reports transamidase turnover on a protein substrate, **isopeptidase** reports cleavage of a
quenched peptide, and a **plasma clot** assay reports a downstream coagulation endpoint in whole
plasma. An IC50 ratio across two of those is a selectivity ratio plus an assay-format difference,
and nothing in the record separates them.

| publication | pairs | TG2 assay | FXIII-A assay | usable as a ratio? |
|---|---|---|---|---|
| CHEMBL2086255 | **69** | amine incorporation | amine incorporation | **yes** |
| CHEMBL5729393 | **57** | isopeptidase (title) | isopeptidase (title) | **no — see below** |
| CHEMBL3421597 | 11 | unspecified | plasma clot, fibrin formation | **no — formats differ** |
| CHEMBL2413014 | 4 | amine incorporation | amine incorporation | yes |
| CHEMBL2203174 | 3 | unspecified | unspecified | unverifiable |
| CHEMBL2150866 | 2 | unspecified | unspecified | unverifiable |
| CHEMBL5233250 | 1 | unspecified | fluorometric, 55 min | formats differ |

**CHEMBL5729393, 57 pairs (39% of the set), carries one methods blob on both assays.** The
identical 1,643-character description is attached to the TG2-target assay and the FXIII-A-target
assay alike, and its title — *"Isopeptidase Assay for Estimating Inhibitor Potency: FXIIIa activity
has been determined using substrate A101…"* — describes the FXIII-A measurement. The same
publication also has a separate transamidation assay carrying 1 TG2 row and 2 FXIII-A rows. The
values themselves are distinct (paired pIC50s differ on 100% of rows, r = 0.38), so this is a
provenance defect in the annotation, not duplicated data. But **the record cannot state which
format produced the TG2 number**, and the paper's structure implies TG2 by transamidation against
FXIII-A by isopeptidase — a cross-format ratio.

**CHEMBL3421597, 11 pairs, is unambiguously cross-format**: purified-enzyme TG2 inhibition against
FXIII-A inhibition *in plasma*, read out as fibrin clot formation. A clot endpoint folds in plasma
protein binding and the coagulation cascade. It is a more clinically meaningful number and a worse
selectivity ratio.

**Format-matched, with self-consistent and distinct descriptions on both sides: 79 pairs across 5
publications**, of which 13 are >10× TG2-selective and 6 are >100×. That 79 — not 147 — is the
defensible denominator for any selectivity claim, and 69 of it is one publication.

*Method note.* Formats are assigned from the ChEMBL assay **title** (text before the first colon),
not the whole field. Matching the whole field labels CHEMBL5729393 `plasma_clot`, because "plasma"
appears only inside a cited paper's title and a reagent source several hundred characters below the
assay title. That was a real bug in the first version of this audit and it is pinned by a test.
A title naming two formats returns `ambiguous` rather than whichever pattern is checked first.

## 3. The pooled selectivity statistic is a mixture of two programmes

| publication | pairs | median log₁₀(TG2/FXIII-A) | sd | >10× TG2 | >10× FXIII-A | programme |
|---|---|---|---|---|---|---|
| CHEMBL2086255 | 69 | **+0.45** (2.8×) | 0.64 | 8 | 1 | TG2-directed |
| CHEMBL5729393 | 57 | **−0.36** (2.3× the wrong way) | 0.83 | 7 | 8 | FXIII-A-directed |
| CHEMBL3421597 | 11 | +0.00 | 0.39 | 0 | 1 | neither |
| CHEMBL2413014 | 4 | +0.30 | 0.04 | 0 | 0 | TG2-directed |
| CHEMBL2203174 | 3 | +3.03 | 0.40 | 3 | 0 | TG2-directed |
| CHEMBL2150866 | 2 | +1.44 | 0.46 | 2 | 0 | TG2-directed |
| CHEMBL5233250 | 1 | +0.26 | — | 0 | 0 | — |
| **pooled** | **147** | **+0.15** (1.4×) | 0.87 | 20 | 10 | **mixture** |

The two largest publications hold 126 of 147 pairs and have **opposite median selectivity**
(+0.45 against −0.36, Mann–Whitney p = 2.3 × 10⁻⁷). One is a TG2 programme; the other is a FXIII-A
programme in which TG2 is the off-target being screened out.

**So the pooled "median 1.4× selective" figure is a mixture of two opposing design objectives, not
an estimate of the selectivity of TG2 chemistry.** It appears in this project's earlier summaries
and should be replaced by the stratified rows above. The TG2-directed series is a genuine 2.8×
median; the 1.4× is an artefact of averaging it with a series optimised in the other direction.
The "10 molecules selective the wrong way" figure is likewise concentrated in the FXIII-A
programme, where selecting against TG2 was the point.

## 4. Potency does not buy selectivity — it buys cross-reactivity

This was the most tempting thing to get wrong. Selectivity is defined as
pIC50(TG2) − pIC50(FXIII-A), so it *contains* the TG2 potency it is being correlated against, and
any such correlation is positive by construction. The column that matters is the last one.

| publication | n | r(potency, selectivity) | independence null | **r(TG2, FXIII-A)** | p |
|---|---|---|---|---|---|
| pooled | 147 | +0.338 | +0.615 | **+0.407** | 3.1e-07 |
| CHEMBL2086255 | 69 | +0.637 | +0.778 | **+0.494** | 1.6e-05 |
| CHEMBL5729393 | 57 | +0.474 | +0.586 | **+0.380** | 0.0035 |

Every observed correlation with selectivity is **below** its independence null — the value two
unrelated enzymes with these sample variances would already produce. Potency therefore buys *less*
selectivity than chance, not more. The direct, assumption-free quantity explains why: TG2 and
FXIII-A potency are **positively correlated** in every stratum. The molecules that inhibit TG2
harder also inhibit FXIII-A harder, which is what a shared catalytic cysteine and a covalent
warhead would predict.

**Consequence for screening: a potency-ranked cascade enriches for cross-reactivity.** This is
consistent with, and mechanistically explains, [`selectivity.md`](selectivity.md)'s finding that
ranking by *predicted* TG2 potency carried no information about a binary selectivity label
(AUROC 0.454). Those are different quantities — predicted versus measured potency, binary versus
continuous selectivity — so they are separate results, but they agree.

## 5. Feasibility of the three proposed approaches

| approach | verdict | why |
|---|---|---|
| **Selectivity regression** | not supportable | 2 publications hold ≥ 20 pairs. Leave-one-publication-out is the only control that separates selectivity from series identity, and it yields **2 transfer observations** — a point estimate with no interval. `audit_fxiiia.py` already ran both directions and neither beat predicting the training mean (RMSE 0.835 vs 1.038, and 0.905 vs 0.892). |
| **Selectivity classification** | not supportable | 13 TG2-selective molecules at >10× in the format-matched subset, 8 of them in one publication. In [`selectivity.md`](selectivity.md) the binary version scored 0.974 under a scaffold split and **0.574 [0.250, 0.864]** with papers held out — a 0.40 collapse, because only 3 of 98 scaffolds appear in both classes. |
| **Multi-task learning** | not supportable, and for a structural reason | Multi-task shares a representation across *tasks*; the shortage here is **groups**, not molecules. Two informative publications and two enzymes cannot support a shared representation that generalises to a third series. The earlier attempt moved AUROC 0.901 → 0.911, and that comparison is an upper bound because the auxiliary feature was fitted on the same molecules. |

Power, for the within-series question only (Fisher-z, α = 0.05, 80%):

| series size | smallest \|Spearman\| detectable |
|---|---|
| n = 57 | 0.37 |
| n = 69 | 0.34 |
| n = 147 (pooled, not independent) | 0.24 |

A within-series correlation weaker than ≈0.35 is undetectable at this size. The cross-series
question has no power calculation because it has two observations.

**No new selectivity model was fitted.** Section 4's finding is a property of the measured data and
needs none; everything a model could add was already measured and reported as negative.

## 6. Comparison against fingerprint baselines

Already run, in `scripts/audit_fxiiia.py` — the relevant baselines are reproduced here rather than
re-fitted:

| test | n | RMSE (log units) | baseline: training mean | Spearman |
|---|---|---|---|---|
| within CHEMBL2086255, scaffold held out | 69 | 0.515 | 0.634 | 0.559 |
| within CHEMBL5729393, scaffold held out | 57 | 0.624 | 0.827 | 0.523 |
| **train A → predict B** | 57 | **0.835** | 1.038 | 0.289 |
| **train B → predict A** | 69 | **0.905** | 0.892 | **−0.053** |

Random forest on Morgan fingerprints. Within a series it beats the mean; across series it does not
beat it in the direction that matters, with a Spearman indistinguishable from zero. Section 2
adds the reason this comparison was never going to work: A and B are not only chemically disjoint
with opposite median selectivity, they are **also probably in different assay formats**.

## 7. Additional public data

| source | what it would add | verdict |
|---|---|---|
| **kinact/Ki, 286 TG2 molecules** | the standard efficiency measure for a covalent inhibitor, comparable in size to the 302-molecule potency set | **the largest unused asset in this project.** Different units from IC50 (a rate over an affinity), so it cannot be merged into the existing label and must be a separate task. FXIII-A has only 11, so it cannot support *selectivity* |
| **TGM6** (CHEMBL2079852) | not covered by the earlier family audit | **7 IC50 rows** (5 exact, 2 censored), **all 7 molecules already in the project TG2 set**, drawn from 4 publications already present. Values span 17 nM to 69 µM. A selectivity panel, not new chemistry, and far too few to model |
| TGM1, TGM3, guinea-pig TG2 | audited in [`selectivity.md`](selectivity.md) | 5, 1 and 0 potent molecules respectively; TGM1 and TGM3 unmodellable, guinea-pig TG2 chemically novel but entirely inactive |
| TGM4, TGM5, TGM7, FXIII-B | — | no human single-protein ChEMBL target with activity data |
| more FXIII-A from ChEMBL | 235 IC50 rows exist against CHEMBL4530 | already fetched; the paired subset is limited by the **TG2** side and by format compatibility, not by FXIII-A volume |

Nothing available enlarges the paired set without mixing formats. The honest way to grow it is new
experiments in one assay format, not more database mining.

**One point of context on TGM6, offered as context and not as a finding:** anti-TG6 antibodies are
associated with gluten ataxia, a neurological manifestation of gluten sensitivity. TG6 is therefore
not obviously an anti-target in the way FXIII-A is. Seven measurements cannot support any claim
either way, and this project makes none.

## 8. Can selectivity against other family members be evaluated?

**No, for any of them.** TGM1 (5 potent), TGM3 (1 potent), TGM6 (7 measurements total) and
guinea-pig TG2 (0 potent) are selectivity panels run on molecules already in the TG2 set. FXIII-A
is the only family member with enough paired data to attempt, and sections 2–5 are why that attempt
does not succeed either. All four remain **descriptive evidence**: they document that
cross-reactivity across the family is common, which is a real property of the known chemical space.

## Verdict

1. **The data cannot support reliable selectivity modelling.** Reported as such rather than forcing
   a model. The binding constraint is not sample size but **group** count: two informative
   publications, pointing in opposite directions, probably in different assay formats.
2. **Two existing figures need qualifying.** "147 paired molecules" should be **79** format-matched
   pairs for any ratio claim; "median 1.4× selective" should be **+2.8× in the TG2-directed series
   and 2.3× the wrong way in the FXIII-A-directed series**, never the pooled value.
3. **One new, robust, negative finding.** TG2 and FXIII-A potency correlate positively
   (r = +0.41 pooled, +0.49 and +0.38 within papers, all significant). Potency-first ranking
   enriches for cross-reactivity. This is a measured property of the data, not a model output, and
   it is the strongest argument in this project against potency-only prioritisation.
4. **Selective chemistry does exist**: 13 molecules at >10× and 6 at >100× in the format-matched
   subset. That is a real observation about those molecules and not a predictive capability.

## Limitations

- Assay formats are inferred from free-text titles by regex. The classifier is tested, but
  `unspecified` covers 17 pairs where ChEMBL's description is too terse to classify, and a format
  that agrees by title could still differ in substrate, buffer, calcium or preincubation time —
  none of which is in the record. A fully reliable audit needs the primary publications.
- The 56 same-description pairs are called unverifiable, not wrong. The underlying values may be
  perfectly good; the *annotation* cannot attribute them.
- Power figures use the Fisher-z approximation and assume independent observations. Molecules
  within a congeneric series are not independent, so the real detectable effect is larger than
  the table states.
- Section 4's correlations are computed on exact IC50s only. Censored values were excluded, which
  removes the least potent molecules from both enzymes and compresses the range.
- **Nothing here is a statement about clinical safety.** Measured FXIII-A selectivity in a cuvette
  is not evidence of bleeding risk in a person, and predicted selectivity is not evidence of
  anything clinical whatsoever. No compound is identified as a treatment candidate.
