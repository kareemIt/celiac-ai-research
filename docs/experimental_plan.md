# What experiment would it take? FXIII-A evidence and a prospective study design (2026-10-29)

Reproduce: `python scripts/audit_fxiiia.py`. Table: `results/fxiiia_evidence.csv`. CPU, seconds.
The evaluation paper ([`paper.md`](paper.md)) and all previous datasets are unchanged by this work.

Three things must stay distinct throughout, and this document marks them explicitly:

| | what it is |
|---|---|
| **Computational prediction** | a number from a model. Evidence about the model, not about the molecule. |
| **Biochemical inhibition** | a measured IC50 or k_inact/K_I against purified enzyme in a cuvette. Real, and still far from a drug. |
| **A celiac treatment** | requires cellular activity, oral exposure, gut delivery, chronic safety and a clinical endpoint. **Nothing in this project approaches this.** |

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

## 3. Can selectivity be predicted? Within a series, yes. Across series, no.

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

**Within a series the selectivity ratio is predictable**, beating the baseline in both series with
Spearman ≈ 0.52–0.56. This is new. The earlier negative conclusion came from binary labels pooled
across publications; the continuous within-study ratio recovers a real signal.

**Across series it is not.** One direction is modestly better than its baseline with weak ranking
(ρ = 0.289); the reverse is *worse* than the baseline with ranking indistinguishable from zero
(ρ = −0.053). The asymmetry is itself informative — transfer worked slightly in the direction with
the more diverse training set and failed entirely in the other.

**Consequence for strategy.** We can help a medicinal chemist rank selectivity **inside a series
they are already working on**. We cannot screen novel chemotypes for selectivity. Any prospective
study must therefore be built around analog ranking, not novel-chemotype discovery — which is the
opposite of how a virtual screen is usually framed, and follows directly from the measurement.

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

## 5. A small prospective study

Designed before any compound is selected, so the criteria cannot be fitted to a result.

### Hypothesis

Within a defined chemical series, model-predicted TG2/FXIII-A selectivity rank-orders measured
selectivity better than chance. **The model is on trial, not the compounds.**

### Inclusion criteria, fixed in advance

1. Analog of a series in the training data, **max Tanimoto ≥ 0.5** to a training active — inside
   the validated applicability domain (below 0.4, every model ranks at chance on TG2).
2. **Not previously measured** against TG2 or FXIII-A in any source we hold.
3. Commercially available or synthesizable in ≤ 3 steps.
4. Spanning the **full predicted selectivity range**, not the top of it — see sampling below.
5. Chemically diverse: at most 2 compounds per Butina cluster.

### Sampling: span the prediction, do not cream the top

**16 compounds**, stratified by *predicted* selectivity into 4 bins of 4 (predicted
non-selective → predicted highly selective). Picking only predicted winners cannot distinguish a
good model from an easy chemical series — a spanning design measures rank correlation, which is the
actual claim.

### Controls

- **Positive, potency:** a known potent TG2 inhibitor from the training set (e.g. a characterised
  acrylamide), to confirm assay performance.
- **Positive, selectivity:** one of the 8 measured >100-fold selective compounds, to confirm the
  FXIII-A counter-screen can detect selectivity when it exists.
- **Negative:** a measured non-selective compound.
- **Vehicle** (DMSO) and **no-enzyme** blanks.
- **Reactivity control:** glutathione half-life for every electrophilic compound.

### Measurements, per compound

TG2 IC50 (full curve); FXIII-A IC50 (full curve, **in parallel, not as a follow-up**); k_inact/K_I
for confirmed covalent actives; glutathione half-life.

### Success and failure criteria, predefined

| outcome | criterion |
|---|---|
| **Success** | Spearman ρ ≥ 0.5 between predicted and measured log selectivity across the 16, **and** ≥ 1 compound confirmed >10-fold TG2-selective |
| **Partial** | ρ in [0.3, 0.5), or selectivity confirmed without rank correlation |
| **Failure** | ρ < 0.3 — the within-series capability does not survive prospectively, and computational selectivity work should stop |
| **Void** | positive controls fail, or >25% of compounds show glutathione half-life below the assay window (nonspecific reactivity dominates) |

A failure result is a publishable outcome and is the expected one under the null. The study is
sized to be *informative either way*, not to produce a hit.

### Power, honestly

Sixteen compounds gives a 95% CI on a Spearman ρ of roughly ±0.4 — wide. It can distinguish ρ ≈ 0.6
from ρ ≈ 0 and cannot resolve finer differences. That is the right scale for a first test, and the
limitation should be stated in advance rather than discovered afterwards.

## 6. What it would take

**Equipment.** Plate reader with absorbance and fluorescence; standard liquid handling; −80 °C
storage; LC-MS for compound QC. No specialised instrumentation.

**Reagents.** Human recombinant TG2; human FXIII-A plus thrombin/Ca²⁺ for activation; substrate
(Cbz-Gln-Gly with GDH coupling, or Z-Glu(HMC)-Gly-OH); 16 test compounds plus controls.

**Expertise.** An enzymologist comfortable with time-dependent covalent inhibition — the critical
requirement, since k_inact/K_I fitting is where this goes wrong. A medicinal chemist to sanity-check
compound selection and flag liabilities a fingerprint model cannot see.

**Scale.** A small study for one person over a few weeks of bench time, not a screening campaign.
We deliberately give no cost figure: reagent and compound pricing varies widely by source and
region, and a number invented here would be quoted later as if it were researched.

**Collaboration.** An academic enzymology or chemical-biology laboratory already running
transglutaminase assays. The TG2 field is small, and the groups behind our training data
(dihydroisoxazole and acrylamide series) are the natural approach — they already have the assays,
the enzymes and the reference compounds.

**What we bring.** A validated evaluation methodology, an applicability-domain analysis, 147
audited paired measurements, and a predefined analysis plan. **Not** a candidate list.

## 7. Remaining gaps

- **Two series is two observations.** The within-series result rests on 69 and 57 molecules from
  two laboratories. It justifies a prospective test; it does not establish a capability.
- **FXIII-A kinetic data is thin**: 10 molecules with paired k_inact/K_I, against 147 with paired
  IC50. Covalent selectivity is therefore an IC50-level claim only.
- **Scaffold-held-out within one series still leaves analogs** on both sides; the within-series
  numbers are optimistic to an unknown degree.
- **No cellular data.** Enzyme inhibition in a cuvette does not establish target engagement in a
  cell, let alone in gut tissue.
- **Prime 2012 k_inact/K_I units remain unrecovered** and need library access.
- **No compound in this project has been proposed as a drug candidate**, and none should be on
  present evidence.
