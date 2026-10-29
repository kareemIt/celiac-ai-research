# What experiment would it take? FXIII-A evidence and a prospective study design (2026-10-29)

Reproduce: `python scripts/audit_fxiiia.py`. Table: `results/fxiiia_evidence.csv`. CPU, seconds.
The evaluation paper ([`paper.md`](paper.md)) and all previous datasets are unchanged by this work.

Three things must stay distinct throughout, and this document marks them explicitly:

| | what it is |
|---|---|
| **Computational prediction** | a number from a model. Evidence about the model, not about the molecule. |
| **Biochemical inhibition** | a measured IC50 or k_inact/K_I against purified enzyme in a cuvette. Real, and still far from a drug. |
| **A celiac treatment** | requires cellular activity, oral exposure, gut delivery, chronic safety and a clinical endpoint. **Nothing in this project approaches this.** |

## 0. Direction of the selectivity ratio — verified

Selectivity is defined as `pIC50(TG2) − pIC50(FXIII-A)`, so **positive means TG2-preferring**.
Verified against raw concentrations rather than the formula alone
(`python scripts/audit_within_series.py`, §1):

| selectivity | TG2 IC50 | FXIII-A IC50 | interpretation |
|---|---|---|---|
| +3.40 | 14 nM | 35,000 nM | 2,500× **TG2**-preferring |
| +3.03 | 62 nM | 67,000 nM | 1,081× TG2-preferring |
| +2.59 | 54 nM | 21,000 nM | 389× TG2-preferring |
| −1.46 | 610 nM | 21 nM | 34× FXIII-A-preferring |

All 8 compounds above 100-fold have a **lower** (more potent) TG2 IC50 than FXIII-A IC50. **No
inverted interpretation was found** in any script or document.

**One convention issue was found.** Replicates are aggregated as the median of pIC50 — log space,
which equals the geometric mean for two values and is correct for log-normal potency data. Taking
the median of raw IC50 would be an arithmetic mean of log-normal data. The two disagree for
molecules with an even replicate count: 19 of 147 molecules, by up to 0.65 log, of which 7 change
sign. Those 7 all sit near zero and none is among the selective compounds, so no conclusion
depends on the choice — but the convention is now stated explicitly wherever it matters.

## 1. The paired evidence, audited

Every TG2/FXIII-A comparison used here is **within a single publication**. Cross-publication ratios
are excluded by construction: [`covalent.md`](covalent.md) measured a 0.50-log inter-laboratory
offset, which is larger than most of the selectivity differences being studied.

Of 216 molecules appearing in both the TG2 and FXIII-A files, **all 216 were measured on both
enzymes within the same paper** — the best possible case for comparability. Restricting to exact
IC50 values on both enzymes leaves **147 molecules across 7 publications**:

| publication | molecules | median log₁₀(TG2/FXIII-A) | SD |
|---|---|---|---|
| CHEMBL2086255 (Prime 2012) | 69 | +0.45 | 0.64 |
| CHEMBL5729393 | 57 | −0.36 | 0.83 |
| CHEMBL3421597 | 11 | 0.00 | — |
| CHEMBL2413014 | 4 | +0.30 | — |
| CHEMBL2203174 | 3 | +3.03 | 0.40 |
| CHEMBL2150866 | 2 | +1.44 | 0.46 |
| CHEMBL5233250 | 1 | +0.26 | — |

The data is concentrated: the largest publication holds 47% of pairs and the top two hold 86%. In
practice there are **two informative series**, not seven.

*Note:* Prime 2012 appears here. Its **IC50** values are complete and usable; only its
k_inact/K_I rows lack units ([`screening_prep.md`](screening_prep.md) §4b). The two problems are
separate.

## 2. Selective chemistry exists — a stronger result than we previously had

| | |
|---|---|
| median log₁₀(TG2/FXIII-A) | **+0.15** (≈1.4×, i.e. typically non-selective) |
| range | −1.46 to **+3.40** |
| >10-fold TG2-selective | **20 of 147** |
| **>100-fold TG2-selective** | **8 of 147** |
| >10-fold selective the *wrong* way | 10 of 147 |

[`selectivity.md`](selectivity.md) concluded from binary 1 µM labels that cross-reactivity is the
norm. That remains true on average, but the continuous ratio shows the tail the binary cut hid:
**eight compounds exceed 100-fold selectivity for TG2 over FXIII-A**. Selective chemistry against
this antitarget is achievable. That is a fact about chemistry, established by measurement, and it
is the strongest encouraging result the project has produced.

## 3. Can selectivity be predicted? In one series, yes. Across series, no.

The two large publications are chemically disjoint — **zero shared scaffolds**, median max-Tanimoto
0.160 — carry different warheads (66 of 69 acrylamide against 55 of 57 non-acrylamide), and have
**opposite median selectivity** (+0.45 against −0.36). A model trained on one and tested on the
other must predict a sign flip it has never seen, which is exactly what prospective screening asks.

| test | n | RMSE (log) | baseline | Spearman |
|---|---|---|---|---|
| **within CHEMBL2086255** (scaffold held out) | 69 | **0.515** | 0.634 | **0.559** |
| **within CHEMBL5729393** (scaffold held out) | 57 | **0.624** | 0.827 | **0.523** |
| train A → predict B | 57 | 0.835 | 1.038 | 0.289 |
| train B → predict A | 69 | 0.905 | 0.892 | **−0.053** |

*Baseline predicts the training set's mean selectivity for every test molecule.*

**Within a series the selectivity ratio appeared predictable** in both, at Spearman ≈ 0.52–0.56.
§3b shows that only one of the two survives a stricter split.

**Across series it is not.** One direction is modestly better than its baseline with weak ranking
(ρ = 0.289); the reverse is *worse* than the baseline with ranking indistinguishable from zero
(ρ = −0.053). The asymmetry is itself informative — transfer worked slightly in the direction with
the more diverse training set and failed entirely in the other.

**Consequence for strategy.** We can help a medicinal chemist rank selectivity **inside one
specific series** (CHEMBL2086255; see §3b for why the other does not count). We cannot screen novel
chemotypes for selectivity. Any prospective study must therefore be built around analog ranking
within that series, not novel-chemotype discovery — the opposite of how a virtual screen is usually
framed, and a direct consequence of the measurement.

## 3b. Robustness: one series survives, the other does not

A scaffold split is not automatically a safe split — that is this project's own central finding, so
it was applied to this result. Each series was re-evaluated under a Butina cluster split
(cutoff 0.6), and under scaffold splits that additionally drop any training molecule above a
similarity cap to the test fold. A label-shuffled null gives the protocol's own baseline.

| series | split | n | Spearman | 95% CI | RMSE | baseline |
|---|---|---|---|---|---|---|
| **CHEMBL2086255** | scaffold-out | 69 | 0.559 | [0.349, 0.740] | 0.515 | 0.634 |
| | Butina-cluster-out | 69 | **0.423** | [0.171, 0.625] | 0.555 | 0.634 |
| | scaffold-out, cap 0.95 | 69 | 0.567 | [0.351, 0.726] | 0.511 | 0.634 |
| | scaffold-out, cap 0.8 | 69 | **0.563** | [0.322, 0.737] | 0.508 | 0.634 |
| | *shuffled-label null* | 69 | −0.230 | [−0.443, 0.004] | 0.754 | 0.634 |
| **CHEMBL5729393** | scaffold-out | 57 | 0.523 | [0.251, 0.738] | 0.624 | 0.827 |
| | Butina-cluster-out | 57 | **−0.013** | [−0.323, 0.292] | 0.858 | 0.827 |
| | scaffold-out, cap 0.95 | 57 | 0.523 | [0.262, 0.734] | 0.624 | 0.827 |
| | scaffold-out, cap 0.8 | 57 | **0.258** | [−0.056, 0.528] | 0.725 | 0.827 |
| | *shuffled-label null* | 57 | 0.036 | [−0.190, 0.345] | 0.923 | 0.827 |

**CHEMBL2086255 holds up.** Spearman stays between 0.42 and 0.57 across every split, the
cluster-out interval excludes zero, and removing training molecules above 0.8 Tanimoto to the test
fold changes nothing (0.563). That is what a real structure–selectivity relationship looks like.

**CHEMBL5729393 does not.** Its scaffold-split 0.523 falls to **−0.013** under a cluster split and
to 0.258 (interval spanning zero) once close analogs are removed. Its apparent signal was carried
by near-identical analogs sitting on opposite sides of a scaffold boundary — the same failure mode
the evaluation paper documents, found here in our own new result.

**The honest revision: selectivity is predictable in one series, not two.** §3's conclusion is
halved. Two observations became one, and a single series is a much weaker basis for a general
claim — though it is still an adequate basis for a prospective test *of that series*.

*On leakage specifically:* neither series contains exact duplicate SMILES. Each contains 2
stereoisomer pairs, and CHEMBL2086255 has 2 test molecules with a ≥0.95 analog in training under a
scaffold split. Those are small numbers and the similarity-capped runs show they are not driving
the result.

*On the null:* the shuffled-label null for CHEMBL2086255 is −0.230 rather than 0, a small negative
bias from leave-one-small-group-out. The observed 0.42–0.57 sits well clear of it either way.

## 4. Established assay methods

Surveyed from the literature and the assay descriptions attached to our own training data, so a
prospective result would be comparable to what the models were trained on.

**TG2 transamidation activity**
- *Colorimetric, Cbz-Gln-Gly with GDH coupling* — the substrate behind a large block of our
  training data. Plate-reader compatible, cheap, continuous.
- *Fluorogenic Z-Glu(HMC)-Gly-OH* — used by another block of our training data; more sensitive,
  continuous, suited to k_inact/K_I work.
- *Biotin-cadaverine incorporation into N,N′-dimethylcasein with streptavidin-HRP* — ELISA-style
  endpoint, widely used, robust but discontinuous.
- *Fluorescence anisotropy* with fluorescent cadaverine conjugates — a reported Z′ of 0.86
  indicates HTS suitability.

**FXIII-A activity.** The literature route is an analogous amine-incorporation assay; FXIII-A
requires activation (thrombin/Ca²⁺) before assay, which is an extra step and a common source of
inter-laboratory variation. Commercial microassays exist — a TG-CovTest kit (Covalab) is cited in
the patent literature for TG1/TG3/FXIIIa — but **availability and specifications must be confirmed
with the vendor**; we have not verified them.

**Covalent kinetics.** Time-dependent inhibition with pre-incubation, fitting k_inact and K_I
separately and reporting **k_inact/K_I with its units**. This project lost 103 measurements to
missing units; that is an avoidable error and the single most important reporting instruction here.

**Nonspecific reactivity.** Glutathione half-life, or an unrelated cysteine-enzyme counter-screen,
to distinguish genuine TG2 engagement from general electrophilicity.

## 5. Choosing the series, and why it is a compromise

Three candidate series, scored on what the study needs:

| | CHEMBL2086255 (Prime 2012) | CHEMBL5729393 | Klöck 2014 dihydroisoxazoles |
|---|---|---|---|
| Paired TG2/FXIII-A molecules | **69** | 57 | 8 |
| Selectivity predictable (cluster-split) | **yes, ρ = 0.42** | no, ρ = −0.01 | not testable at n = 8 |
| Assay comparability | one paper, one protocol | one paper | one paper, kinetic not IC50 |
| Chemical diversity | 39 scaffolds | 36 scaffolds | 25 scaffolds |
| Measured selectivity available | median +0.45, up to +2.25 | median −0.36 | only 8 pairs |
| Celiac relevance of the chemotype | indirect (Huntington's programme) | indirect | **direct** (ERW1041E lineage) |
| Safety liability | acrylamide Michael acceptor | mixed | 3-bromo-dihydroisoxazole |

**Recommended: CHEMBL2086255.** It is the only series where the model's selectivity predictions
survive a cluster split, which is the entire basis for running the study. 69 paired measurements
under one protocol, 39 scaffolds, and a usable spread of measured selectivity give the prospective
comparison something to correlate against.

**The compromise is honest and should be stated to any collaborator.** This series comes from a
Huntington's disease programme, not a celiac one, and its compounds are acrylamide Michael
acceptors — a chemotype with genuine promiscuity concerns for chronic dosing. The
celiac-relevant chemotype is the dihydroisoxazole series, and it has only 8 paired FXIII-A
measurements, far too few to support anything. **We are proposing to test the model where the model
can be tested, not where the chemistry is most attractive.** That is the right choice for a
methodological experiment and the wrong one for a discovery campaign, and the proposal says so.

## 6. Selecting 16 compounds without cherry-picking

The criteria are fixed here, before any compound is chosen, so they cannot be fitted to a result.
**No compounds are selected in this document and none should be ordered.**

### Eligibility (all must hold)

1. **Analog of the chosen series**: max Tanimoto ≥ 0.5 to a CHEMBL2086255 training molecule. Below
   0.4 every TG2 model ranks at chance; 0.4–0.5 is unvalidated territory.
2. **Not previously measured** against TG2 *or* FXIII-A in any source we hold — checked against the
   631-molecule IC50 set, the 219-molecule kinetic set and all off-target files.
3. **Commercially available**, or synthesizable in ≤ 3 steps from catalogue material.
4. **Passes reactivity filters**: PAINS-clean, and no second electrophile beyond the series warhead.
5. **Soluble enough to assay**: predicted logS above the assay's working concentration.

### Sampling: span the prediction, never the top of it

Rank eligible compounds by **predicted** selectivity, divide into **4 equal-width bins across the
full predicted range**, and take **4 compounds per bin**, choosing within each bin to maximise
Butina-cluster diversity (at most 2 per cluster across the whole set).

This is the design decision that makes the study informative. Selecting the 16 highest predicted
selectivities would confound a good model with an easy series: if all 16 confirm, we cannot tell
whether the model ranked them or whether every compound in the series is selective. A spanning
design measures **rank correlation**, which is the actual claim, and it is the only design under
which a negative result is interpretable.

Compounds must be committed to in writing, with their predictions, **before** any assay is run. A
prediction recorded after the measurement is not a prediction.

### What is predicted, and what is not

| | |
|---|---|
| **Predicted** | the log selectivity ratio, from a random forest on Morgan fingerprints, trained on 69 molecules from one publication |
| **Not predicted** | potency against either enzyme in absolute terms, cellular activity, promiscuity, solubility, stability, toxicity |
| **Explicitly unknown** | whether any compound engages TG2 in a cell, in gut tissue, or in an organism |

## 6b. The laboratory-facing proposal

### Objective

Test whether model-predicted TG2/FXIII-A selectivity rank-orders measured selectivity for 16
untested analogs of a defined series. **The model is on trial, not the compounds.**

### Measurements per compound

| measurement | purpose |
|---|---|
| TG2 IC50, full dose–response | the potency axis |
| **FXIII-A IC50, full dose–response, run in parallel** | the selectivity axis; parallel, not follow-up, so both share a baseline |
| k_inact/K_I for confirmed covalent actives, **units reported** | covalent efficiency; this project lost 103 measurements to missing units |
| Glutathione half-life | separates genuine engagement from general electrophilicity |

### Controls

- **Potency positive**: a characterised TG2 inhibitor from the training set.
- **Selectivity positive**: one of the 8 measured >100-fold selective compounds — confirms the
  counter-screen can detect selectivity when it exists.
- **Negative**: a measured non-selective compound.
- **Vehicle** (DMSO) and **no-enzyme** blanks on every plate.

### Predefined success and failure

| outcome | criterion |
|---|---|
| **Success** | Spearman ρ ≥ 0.5 between predicted and measured log selectivity across the 16, **and** ≥ 1 compound confirmed >10-fold TG2-selective |
| **Partial** | ρ ∈ [0.3, 0.5), or selectivity confirmed without rank correlation |
| **Failure** | ρ < 0.3 — the within-series capability does not survive prospectively; computational selectivity work stops |
| **Void** | positive controls fail, or > 25% of compounds show glutathione half-life below the assay window |

Failure is the expected outcome under the null and is a publishable result. The retrospective
cluster-split estimate is ρ = 0.42 with a 95% interval of [0.171, 0.625], so a prospective ρ
below 0.3 is entirely possible even if the model is genuinely useful.

### Power, stated in advance

Sixteen compounds gives roughly ±0.4 on a Spearman confidence interval. It can distinguish ρ ≈ 0.6
from ρ ≈ 0 and nothing finer. The study is sized to be informative either way, not to be precise.

### Resources

**Equipment:** plate reader with absorbance and fluorescence, standard liquid handling, −80 °C
storage, LC-MS for compound QC. No specialised instrumentation.

**Reagents:** human recombinant TG2; human FXIII-A with thrombin/Ca²⁺ for activation; substrate
(Cbz-Gln-Gly with GDH coupling, or Z-Glu(HMC)-Gly-OH); 16 compounds plus 3 controls.

**Expertise:** an enzymologist experienced with time-dependent covalent inhibition — the critical
requirement, since k_inact/K_I fitting is where this work goes wrong. A medicinal chemist to
sanity-check selection and flag liabilities a fingerprint cannot see.

**Scale:** a few weeks of bench time for one person. We give no cost figure deliberately: reagent
and compound pricing varies by source and region, and a number invented here would later be quoted
as if it had been researched.

**Collaboration:** an academic enzymology or chemical-biology group already running
transglutaminase assays — they have the enzymes, the substrates and the reference compounds. The
groups behind our training data are the natural approach.

**What we bring:** a validated evaluation methodology, an applicability-domain analysis, 147
audited paired measurements, a robustness audit that already discarded half our own positive
result, and a preregistered analysis plan. **Not a candidate list.**

## 7. Remaining gaps

- **One series is one observation.** The within-series result initially appeared in two series;
  §3b showed one of them was near-analog leakage that vanishes under a cluster split. What remains
  is 69 molecules from a single publication. That justifies a prospective test of that series; it
  establishes nothing general, and a second independent series would be worth more than a larger
  study of this one.
- **The surviving series is the wrong chemotype for celiac disease.** It comes from a Huntington's
  programme and its compounds are acrylamide Michael acceptors. The celiac-relevant dihydroisoxazole
  series has 8 paired FXIII-A measurements and cannot support the analysis. We are testing where
  the model can be tested, not where the chemistry is most attractive.
- **Predicted ≠ demonstrated ≠ safe.** A predicted selectivity ratio is a model output. A measured
  one is biochemistry in a cuvette. Neither is evidence of therapeutic safety: a compound 100-fold
  selective over FXIII-A in vitro may still inhibit other cysteine enzymes, fail in cells, or be
  unsafe chronically. No compound in this project has been proposed as a drug candidate.
- **FXIII-A kinetic data is thin**: 10 molecules with paired k_inact/K_I, against 147 with paired
  IC50. Covalent selectivity is therefore an IC50-level claim only.
- **Scaffold-held-out within one series still leaves analogs** on both sides; the within-series
  numbers are optimistic to an unknown degree.
- **No cellular data.** Enzyme inhibition in a cuvette does not establish target engagement in a
  cell, let alone in gut tissue.
- **Prime 2012 k_inact/K_I units remain unrecovered** and need library access.
- **No compound in this project has been proposed as a drug candidate**, and none should be on
  present evidence.
