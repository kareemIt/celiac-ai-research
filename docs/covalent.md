# Covalent inhibition: can we predict kinact/Ki? (run 2026-10-29)

Reproduce: `python scripts/audit_covalent.py`, then `python scripts/run_covalent.py`.
Tables: `results/covalent_audit.csv`, `results/covalent.csv`. CPU, about a minute.

Most published TG2 inhibitors are covalent, and the IC50 filter this project uses everywhere else
discards 990 of the 1,583 TG2 records. The selectivity audit flagged that some of what it throws
away is mechanistically the most informative data in the file. This phase checks that claim and
then tests it.

## Phase 1 — the kinetic audit

### Three quantities, not one

They are counted and kept separately throughout; none is merged with the others, and none is mixed
with an IC50 label.

| quantity | what it measures | unit | records | molecules |
|---|---|---|---|---|
| **Ki** | affinity before anything reacts | nM | 240 | — |
| **kinact** | rate of covalent bond formation once bound | /min | 155 | 148 |
| **kinact/Ki** | overall efficiency, the quantity covalent inhibitors are ranked by | /M·s | 310 | 288 |
| kon, k_off | association / dissociation | /M·s, s⁻¹ | 142 | — |

kinact/Ki is the useful one: it is a second-order rate constant, so unlike kinact alone it is
comparable across compounds whose binding affinities differ.

### The 288 shrinks to 174

ChEMBL stores kinact/Ki under three unit conventions, and 118 rows carry no unit at all:

| paper | /M·s | 10³/M·min | no units |
|---|---|---|---|
| CHEMBL2086255 | | | 55 |
| CHEMBL3351998 | | | 48 |
| CHEMBL4017513 | | 30 | 2 |
| CHEMBL4219231 | 52 | | |
| CHEMBL5104118 | | 14 | 2 |
| CHEMBL5154739 | | 45 | 1 |
| CHEMBL5230175 | | 22 | 8 |
| CHEMBL5230179 | | 29 | |
| CHEMBL5346966 | | | 2 |

A unitless rate constant cannot be placed on a common scale, and guessing the convention from the
magnitude would be inventing data, so those rows are dropped. After converting 10³/M·min to /M·s
and keeping exact (`=`) rows only:

**192 measurements → 174 molecules → 70 Murcko scaffolds → 6 papers.**
log₁₀(kinact/Ki) spans 1.60 to 5.00, SD 0.62. Two entire papers (103 rows) are lost to missing
units.

### The label is partly a property of the lab

| paper | n | min | median | max | SD |
|---|---|---|---|---|---|
| CHEMBL4017513 | 30 | 2.34 | 2.96 | 3.34 | 0.23 |
| CHEMBL4219231 | 52 | 1.90 | 3.37 | 4.02 | 0.57 |
| CHEMBL5104118 | 14 | 1.60 | 2.18 | 4.40 | 0.85 |
| CHEMBL5154739 | 45 | 2.07 | 2.97 | 3.93 | 0.53 |
| CHEMBL5230175 | 22 | 2.05 | 3.23 | 5.13 | 0.98 |
| CHEMBL5230179 | 29 | 2.97 | 3.62 | 4.10 | 0.27 |

**Between-paper SD of medians is 0.50 log; within-paper SD is 0.57 log.** Each paper uses a
different substrate — Z-Glu(HMC)-Gly-OH, AL5, an unnamed colorimetric assay — and kinact/Ki is
nominally substrate-independent but in practice is not. A model trained on five papers and tested
on the sixth must absorb a systematic offset about as large as the signal it is trying to rank.
This single number is the main reason the experiment below comes out the way it does.

### Only one warhead, which cuts both ways

| | molecules |
|---|---|
| acrylamide | **169 of 174** |
| chloroacetamide | 3 |
| epoxide, dihydroisoxazole, maleimide, vinyl sulfone, nitrile | 0 |

**Good news:** a model cannot succeed here by recognizing the reactive group, because essentially
every molecule has the same one. Any signal must come from the recognition element around it. This
is a cleaner structure–activity question than the usual covalent dataset, where warhead identity
dominates everything.

**Bad news:** there is no mechanism *contrast* to classify. Assay descriptions state irreversible
inhibition for 238 molecules and reversible for 14 — not a usable two-class problem. **Mechanism
classification is therefore not attempted**, and the phase narrows to one question: can covalent
efficiency be predicted within the acrylamide class?

### What it adds to the IC50 data

159 of the 174 molecules carry no IC50 label, so this is mostly new material. But it is not new
*chemistry*: median max-Tanimoto to the existing 631-molecule set is 0.662, and only 7 molecules
fall below 0.5. **It does not address the 6-positive novel stratum** that limits the generalization
claims elsewhere in this project.

## Phases 2 and 3 — the experiment

Justified by the audit: 174 molecules, 70 scaffolds, a 3.4-log continuous label and a constant
warhead. Fingerprint baselines only, no GPU, leave-one-paper-out as the primary split.

| model | n | RMSE (log) | 95% CI | mean within-paper Spearman |
|---|---|---|---|---|
| **mean** (predict the training papers' average) | 174 | 0.644 | [0.580, 0.707] | n/a (constant) |
| molecular size alone | 174 | 0.644 | [0.569, 0.728] | 0.029 |
| ridge on Morgan bits | 174 | 0.759 | [0.693, 0.826] | 0.110 |
| **random forest on Morgan bits** | 174 | **0.598** | [0.537, 0.653] | **0.120** |

Paired over the same molecules: **random forest − mean = −0.046 log RMSE, 95% CI
[−0.101, +0.011], P(no better) = 0.054.**

Two things follow.

**The model barely beats predicting a constant, and not significantly.** A 0.046-log improvement
on a label with 0.62 log of spread is about 7% of the standard deviation, with an interval that
includes zero. Ridge is *worse* than the constant, which is what a high-dimensional sparse
fingerprint does when the training signal is weak.

**It cannot rank analogs inside a new series, which is the thing a prioritization would need.**
Mean within-paper Spearman is 0.120. Given a fresh paper's worth of acrylamides, the model orders
them about as well as chance.

### The scaffold split shows the same optimism as the selectivity work

| model | split | RMSE | 95% CI |
|---|---|---|---|
| mean | scaffold | 0.660 | [0.574, 0.744] |
| random forest | scaffold | **0.489** | [0.404, 0.571] |
| mean | paper | 0.644 | [0.580, 0.707] |
| random forest | paper | **0.598** | [0.537, 0.653] |

Under a scaffold split the random forest looks respectable, cutting RMSE by a quarter against the
constant. Under a paper split the same model on the same molecules gains almost nothing. The gap is
the same artefact [`selectivity.md`](selectivity.md) documents: scaffolds do not nest inside
papers, so a scaffold split leaves close analogs from the same lab on both sides of the fold.
Reporting the 0.489 alone would be the second time this project could have published a leakage
number as a result.

## Verdict

**Does the kinetic data contain information beyond the IC50 set?** Yes, in principle — 159 new
molecules with a continuous, mechanistically meaningful label and a 3.4-log range.

**Can we predict covalent efficiency?** **Not usefully.** Across papers the model improves on a
constant by 0.046 log RMSE (95% CI [−0.101, +0.011]) and ranks within a new series at Spearman
0.120. The honest statement is that we cannot.

**Can we classify inhibition mechanism?** **No, and it was not attempted.** 169 of 174 molecules
share one warhead and the reversible class has 14 molecules. There is no contrast to learn.

**Why it fails is clear, and is itself the finding.** 174 molecules from 6 papers, each using a
different substrate, with a between-paper offset (0.50 log) nearly equal to the within-paper spread
(0.57 log). The limiting factor is not the model or the featurization — ridge, random forest and a
single size descriptor all land within noise of a constant — it is that six laboratories measuring
different series on different substrates do not constitute a dataset you can learn transferable
chemistry from.

## Limitations

- 6 papers. Leave-one-paper-out means six folds, two of them with 14 and 22 molecules.
- 103 rows across two papers were discarded for missing units; recovering them from the primary
  literature would grow the set by roughly half and is the single highest-value manual step here.
- Only exact values are used. Censored kinact/Ki rows (1 row) are dropped.
- kinact/Ki is treated as comparable across substrates after unit conversion. The per-paper table
  shows that assumption is imperfect, and no inter-laboratory correction was attempted.
- Median over replicates per molecule, as elsewhere in this project.
- Nothing here measures selectivity, cell activity or stability. A fast covalent inhibitor of TG2
  may be a fast covalent inhibitor of many other cysteine enzymes.
