# When the Split Decides the Answer: Evaluation Methodology, Distribution Shift and Experimental Provenance in Machine-Learned TG2 Inhibitor Prediction

**Draft for supervisor review — 2026-10-29 (rev. 2: CDK2 validation added)**

*Status: complete draft, not submission-ready. Venue requirements have not been checked; see
§10.3 before any external submission.*

---

## Abstract

Tissue transglutaminase 2 (TG2) deamidates gluten peptides into the immunogenic forms that drive
celiac disease, making it the principal small-molecule target for a non-dietary therapy. We set out
to ask whether a locally-run 9-billion-parameter language model, used as a constrained decision
model, could predict TG2 inhibitory potency better than standard cheminformatics. It could not: a
tuned fingerprint random forest reaches AUROC 0.898 ± 0.057 across ten Bemis–Murcko scaffold splits
of 631 molecules, against 0.850 ± 0.058 for a QLoRA-fine-tuned Qwen3.5-9B (ΔAUROC −0.049, 95% CI
[−0.076, −0.021], p = 0.007).

The more durable finding is methodological. Across eight experimental campaigns on this single
target we encountered six distinct ways in which an evaluation protocol, or our own reading of it,
rather than a model, determined the reported result, and in each case the corrected protocol reversed or dissolved the
conclusion. (i) **Random-split leakage**: 38.6% of random-split test molecules have a training
molecule with an *identical* Morgan fingerprint, inflating AUROC from 0.87 to 0.95. (ii)
**Prediction pooling across overlapping folds**: a headline claim that the un-fine-tuned model
generalised better to unfamiliar chemistry (AUROC 0.807 against 0.794) was an artefact of counting
one cached prediction ten times; scored once per molecule the effect reverses to −0.019, 95% CI
[−0.257, +0.192], and the hypothesis was withdrawn. (iii) **Training nondeterminism**: two runs of
an identical configuration differed by −0.028 ± 0.089 AUROC with per-seed swings to 0.18, as large
as the effect under study; the cause was nondeterministic backward-pass kernel selection rather
than the quantized optimizer, and after correction the putative catastrophic-forgetting effect did
not reproduce. (iv) **Publication-level confounding**: a selectivity classifier scoring AUROC 0.974
under scaffold splitting falls to 0.574, 95% CI [0.250, 0.864], when whole source publications are
held out, because scaffolds do not nest within papers.

We additionally report that cross-reactivity is the norm in this chemical space (103 of 132
TG2-potent molecules with paired data also inhibit Factor XIII-A at 1 µM), that the transglutaminase
family cannot expand the dataset where it is weakest, and that covalent inhibition efficiency
(kinact/Ki) does not predict across laboratories (−0.046 log RMSE against a constant predictor, 95%
CI [−0.101, +0.011]). We then test three of the first four failures on an independent target, CDK2 (2,646 molecules, 351
publications, chemically unrelated to TG2) and again on MAO-B (5,840 molecules, 586 publications).
All three reproduce in direction and mechanism while differing in magnitude, and both independent
targets agree that the publication split costs more than the scaffold split. Publication-level confounding reproduces most strongly, costing more than
scaffold splitting (AUROC 0.898 → 0.830) on a large, diverse dataset whose positives are spread
across hundreds of papers — contrary to our own pre-registered prediction that it would not appear
there. The second target also corrects us: TG2's finding that models rank at chance below 0.4
Tanimoto does **not** generalise, as CDK2 retains AUROC 0.739 in that regime, so the applicability
floor is a per-dataset quantity rather than a constant.

Three later campaigns extend the picture and add two failures of different kinds. **A nine-entrant
state-of-the-art comparison** — Chemprop, CheMeleon, ChemBERTa-2, MoLFormer-XL and a tabular
foundation model against the tuned baselines — finds that **no specialized molecular model beats the
random forest** under scaffold evaluation (best Δ −0.033, Holm p = 0.045), while on the same 444
molecules the publication-held-out arm costs every model 0.10–0.28 AUROC and **inverts the ranking
inconclusively** (ChemBERTa-2 frozen Δ +0.089, 95% CI [−0.027, +0.205], every Holm p = 1.000 at 11
folds). **A direct label-leakage probe** shows base Qwen3.5-9B holds none of these IC50 values:
631 of 631 abstentions from SMILES alone, six distinct answers when forced to estimate, and
re-rooting the SMILES — same molecule, different string — *improves* every metric, which recall
cannot do. And **conformal prediction's coverage guarantee, which holds under scaffold shift, fails
under publication shift with the same calibration recipe** (0.784 against a 0.90 target), while
calibrating on held-out publications restores it (0.914, Δ +0.113, p = 2.5 × 10⁻⁸) at the price of
committing on only 15–32% of molecules and erring on 25–39% of those. The fifth failure is thus a
method carried outside the shift it was validated under; the sixth is our own inference from a
correlation between two components to the behaviour of their difference, which direct measurement
refuted.

We distinguish throughout between statistically supported findings, inconclusive results, withdrawn
hypotheses and findings since scoped, and we claim a three-target case study rather than a general
law.

**Keywords:** transglutaminase 2, celiac disease, evaluation methodology, distribution shift, data
leakage, conformal prediction, covalent inhibitors, large language models

---

## 1. Introduction

### 1.1 The target and the therapeutic gap

Celiac disease affects roughly 1% of the population and has one approved management strategy: total
lifelong avoidance of dietary gluten. Tissue transglutaminase 2 sits at the mechanistic centre of
the disease. It deamidates specific glutamine residues in gluten-derived peptides, converting them
into higher-affinity ligands for HLA-DQ2/DQ8 and thereby triggering the gluten-specific T-cell
response. Inhibiting TG2 is therefore among the most direct pharmacological interventions available
in principle [1, 17].

**TG2 inhibition has clinical proof of concept.** ZED1227, a covalent TG2 inhibitor, was tested
against placebo in 163 patients with celiac disease undergoing a 6-week gluten challenge, and met
its primary endpoint: the change in villus-height-to-crypt-depth ratio relative to placebo was 0.44
(95% CI 0.15 to 0.73) at 10 mg and 0.48 (95% CI 0.20 to 0.77) at 100 mg [20]. The authors describe
it as a proof-of-concept trial. No TG2 inhibitor has yet been approved, and not every programme has
progressed — GSK3915393 completed a Phase 1 study in healthy volunteers (NCT04604795) and was
subsequently removed from GSK's celiac pipeline, with no reason published [16].

So the target is pharmacologically validated in humans while the chemical space remains thin, which
is the gap computational prioritisation might narrow. It also raises the stakes on selectivity: a
drug taken chronically by otherwise-healthy people must not inhibit Factor XIII-A, which crosslinks
fibrin (§6).

### 1.2 The question we set out to answer

Recent work by Wichka and Lai applied boosting ensembles to approximately 1,100 TG2 inhibition
assays and released both a curated dataset and a screening web application [1]. That dataset is the
starting point for this project. Our original question was narrow:

> Used as a constrained decision model in the style of the recently proposed LLM-as-Jev framework
> [3], does a locally-run ~9B language model predict TG2 inhibitory potency better than standard
> cheminformatics baselines on identical splits — and if not, does it contribute anything they
> cannot?

The answer to the first clause is no, and we report it in §5.2. The answer to the second clause
turned out to be the more interesting part of the project, and not in the way we expected.

### 1.3 What this paper actually contributes

Over five campaigns we repeatedly found that our own evaluation protocol, rather than any
modelling choice, was setting the answer. Two results were caught as artefacts only after they had
been written down as findings. Rather than discard them, we treat them as the subject matter.

The contribution of this paper is a documented case study of **four mechanistically distinct
evaluation failures on one small, literature-derived bioactivity dataset**, each with a measured
before-and-after effect size:

| Failure mode | Where it bit | Effect of correcting it |
|---|---|---|
| Random-split leakage | Baseline model selection | AUROC 0.95 → 0.87 |
| Pooling across overlapping folds | Novel-chemistry hypothesis | +0.013 → −0.019 (hypothesis withdrawn) |
| Training nondeterminism | KL-regularisation ablation | Noise floor −0.028 ± 0.089 → exactly 0 |
| Publication-level confounding | Selectivity classification | AUROC 0.974 → 0.574 |
| A guarantee carried outside its shift | Conformal calibration recipe | Coverage 0.784 → 0.914 against a 0.90 target |
| Inferring a difference from its components | "Potency enriches cross-reactivity" | 13.6% → 46.7% gate pass rate (claim withdrawn) |

These are not six instances of one problem. Leakage is about molecular redundancy; pooling is
about the independence assumption in a confidence interval; nondeterminism is about numerical
reproducibility on GPU hardware; publication-level confounding is about where the data came from;
the fifth is about the scope condition attached to a distribution-free guarantee; and the sixth is
not a protocol error at all but a reasoning error of ours, caught only by computing the quantity we
actually cared about instead of a correlate of it. Each requires a different control, and standard
practice in the field addresses only the first two consistently.

We state explicitly that these are findings about **this dataset and this target**. We did not run
the same analyses on other targets and we make no claim that the magnitudes transfer. What we do
claim is that each failure is mechanistically plausible wherever a dataset is small, assembled from
heterogeneous literature, and dominated by analog series — a description that fits a large fraction
of published bioactivity benchmarks.

---

## 2. Related work

### 2.1 Benchmark validity in ligand-based modelling

The observation that ligand-based benchmarks can reward memorisation rather than generalisation is
not new. Wallach and Heifets examined seven widely used virtual-screening benchmarks and introduced
the AVE bias measure, showing that the amount of train/validation redundancy correlated strongly
with reported model performance regardless of the fingerprint, similarity measure or unbiasing
technique applied, and argued that such models will likely fail prospectively [10]. Sheridan made a
related argument from the opposite direction, showing that time-split cross-validation yields an R²
closer to true prospective prediction than random selection does, and recommending it as a standard
complement to random splits [9].

Our work differs in scope and in kind. Those papers establish that redundancy inflates benchmark
scores in general. We document, on a single target, that four *separate* protocol decisions each
independently determined a conclusion we were about to report, including two that neither AVE bias
nor time-splitting would have detected: GPU-level training nondeterminism, and confounding between
the class label and the source publication.

### 2.2 Measurement uncertainty in public bioactivity data

The question of how much noise is intrinsic to pooled public data has a quantitative answer. Kramer
et al. analysed heterogeneous public Ki values and estimated a mean inter-laboratory error of 0.44
pKi units with a standard deviation of 0.54, implying a maximum achievable Pearson R² of about 0.81
on large datasets [11]. Kalliokoski et al. extended the analysis to IC50 data, finding that the
standard deviation of mixed public IC50 values is about 25% larger than for Ki, and advising that
original literature be inspected wherever assay comparability matters [12].

These numbers are directly relevant to our §5.7. We measure a between-publication standard
deviation of 0.50 log units in kinact/Ki against a within-publication spread of 0.57 — almost
exactly the regime Kramer et al. describe, arrived at independently on a different quantity. Our
contribution here is to show what that noise level *does* to a predictive model trained across
publications: it reduces it to the performance of a constant.

### 2.3 Calibration and distribution-free uncertainty

Temperature scaling is the standard post-hoc calibration method for modern classifiers [5].
Conformal prediction provides distribution-free coverage guarantees under exchangeability [6]. Both
are well characterised in-distribution; their behaviour under the chemical distribution shift
induced by scaffold splitting is less often reported. We find that temperature scaling fails for
fingerprint models under scaffold shift while conformal prediction holds its nominal coverage
(§5.5), which is consistent with theory but, to our knowledge, not widely demonstrated on a small
single-target medicinal-chemistry dataset.

### 2.4 Language models as constrained predictors

The LLM-as-Jev framework extracts calibrated categorical decisions from a causal language model by
reading next-token probabilities over numbered option identifiers, with an optional fine-tuning
recipe that adds KL anchors to a frozen reference model [3]. We implement that method faithfully,
including the three anchor terms, and use QLoRA [4] over LoRA adapters [13] for the fine-tuning
itself. Our use is off-label: the published method targets general decision tasks and its anchors
are motivated by preserving general chat behaviour, whereas we ask them to preserve a *chemical*
prior. We flag this extrapolation explicitly rather than presenting it as a replication.

---

## 3. Data

### 3.1 Sources and provenance

Two sources were combined after an explicit agreement check.

| | PubChem export | ChEMBL |
|---|---|---|
| Origin | `Lailabcode/celiac-informatics`, the data behind Wichka & Lai [1] | ChEMBL REST API, target CHEMBL2730 (human TGM2, UniProt P21980) [2] |
| Raw records | 1,130 rows, 672 distinct CIDs | 1,583 activities, 781 molecules |
| Usable potency labels | 367 molecules | 460 molecules |

All ChEMBL records retained are binding assays in *Homo sapiens*. Target identifiers were resolved
through the ChEMBL target endpoint from UniProt accessions rather than assumed.

### 3.2 Unit validation

The source CSV stores bare numeric IC50 values with no unit column. The originating paper states
the unit directly — it describes molecules with IC50 ≤ 1 µM as the most potent and uses a 0.6 µM
active cut — establishing µM, so pIC50 = 6 − log₁₀(value) [1]. This had been flagged as an open
question and is now closed. We adopt a single 1 µM threshold throughout rather than the source's
mixed scheme, which avoids incorporating molecules that were never assayed.

### 3.3 Three corrections that changed the dataset

**Censored measurements were being discarded.** Every `Inactive` row in the source carries a `>`
qualifier: 46 such rows against 327 exact `=` actives. A filter keeping only exact rows silently
removed every documented inactive compound. A bound of `> 10 µM` yields no pIC50 but does establish
"not potent at 1 µM". Incorporating censored molecules as confirmed negatives where the bound lies
at or above the threshold — and ignoring bounds below it, which prove nothing — grew the set from
**305 to 367 molecules** (160 potent / 207 not).

**Salts were not stripped.** Eighteen entries are salts or mixtures whose counterions set their own
fingerprint bits and can shift the Murcko scaffold that splits group on. Canonicalisation now keeps
the largest fragment. Regression metrics barely moved (RF RMSE 0.776 → 0.778), confirming this as a
correctness fix rather than a performance lever.

**Source agreement was verified before merging.** On the 173 molecules with an exact pIC50 in both
sources, Spearman correlation is 1.000 (Pearson 0.997), median absolute difference 0.000 log units,
and 196 comparable class labels agree 100%. The union therefore gives **631 molecules** (302
potent), with the PubChem value retained where both exist so that earlier PubChem-only results
remain a strict subset.

### 3.4 Known data limitations

No SMILES in the PubChem export carries stereochemistry, so enantiomers are collapsed; this is a
property of the source and caps achievable performance on stereochemically driven activity. Of 77
molecules with replicate exact IC50 values, the median max/min ratio is 2.8× but 14 disagree by more
than 10× and the worst spans 77×; we take the per-molecule median of pIC50. Ki, kinact, kon/koff and
percentage-inhibition records are excluded from the potency label as different quantities (§5.7
treats the kinetic records separately). Thirty ChEMBL rows flagged by its own curation as outside
the typical range are dropped.

---

## 4. Methods

### 4.1 Splitting protocol

Unless stated otherwise, all experiments use Bemis–Murcko scaffold splits [7] at 80/10/10, with
whole scaffold groups assigned to a single fold and folds filled by molecule count rather than
scaffold count (60% of the 229 scaffolds are singletons but the largest holds 48 molecules). Ten
seeds are used throughout. Section 5.1 reports why random splits are not used, and §5.6 and §5.7
report where scaffold splitting is itself insufficient.

### 4.2 Representations and baselines

Molecules are represented as 2048-bit Morgan fingerprints of radius 2 [14], computed with RDKit
[15]. Baselines are logistic regression, random forest and XGBoost, each tuned per seed on that
seed's own validation fold over a small fixed grid, plus a three-model ensemble. A prior-only
classifier is included as a floor.

### 4.3 Language model configuration

Qwen3.5-9B is loaded in 4-bit NF4 quantization with LoRA adapters (r = 16, α = 32, dropout 0.05, all
linear projections), trained with paged 8-bit AdamW at learning rate 1e-4 and gradient accumulation
8, following QLoRA [4]. The decision readout follows LLM-as-Jev [3]: options are numbered, the
assistant turn is prefilled with `Best answer: [`, and option *k* is scored by the summed
log-probability of the suffix `k]`. All training and inference ran locally on a single RTX 5070 Ti;
no hosted APIs or cloud GPUs were used at any point.

### 4.4 KL anchors

We implement the three anchor terms of the published objective against the frozen base model
(realised as the same network with the adapter disabled): `L_mass`, a binary KL on probability mass
over legal option tokens at the decision node; `L_out`, a KL over the renormalised distribution
across illegal tokens; and `L_pos`, a KL over full next-token distributions at the reply start and
after the closing bracket. Each term is truncated to the reference's top-64 tokens, and the total
objective is `L_tree + λ·L_anchor`. One deviation is forced by our setting and is recorded in the
source: the published method samples three off-path trie nodes beside the gold path, but a
two-option single-digit decision has exactly one node.

### 4.5 Statistical treatment

Because molecules are the independent unit and folds overlap across seeds, pooled predictions are
collapsed to one row per molecule before any interval is computed (§5.3 explains why). Confidence
intervals are bootstrap percentiles over molecules, 2,000 replicates unless stated. Paired
comparisons resample the same molecules on both sides. Per-seed comparisons use paired t-tests
across the ten seeds. Chemical novelty is the maximum Tanimoto similarity to any training molecule;
effective sample size on novel subsets is assessed by Butina clustering [8].

### 4.6 Determinism

After §5.4, all language-model training enables `torch.use_deterministic_algorithms(True,
warn_only=False)`, a fixed cuBLAS workspace (`:4096:8`), cuDNN determinism, disabled TF32 and seeded
Python/NumPy/Torch RNGs. Strict mode never raised, meaning every operation in this model's forward
and backward pass has a deterministic kernel available and nothing required exemption.

---

## 5. Experiments and results

Each subsection states its conclusion with an explicit status: **demonstrated** (a measured effect
whose interval excludes the null), **not established** (the experiment ran and did not support the
claim), or **withdrawn** (previously reported here and retracted).

### 5.1 Random splits leak catastrophically — *demonstrated*

| | Scaffold split | Random split |
|---|---|---|
| Scaffolds shared between train and test | **0** | 28.6 |
| Mean Tanimoto to nearest training molecule | 0.696 | 0.841 |
| Test molecules with a training analog ≥ 0.7 | 51% | 83% |
| Test molecules with an **identical** fingerprint to a training molecule | **0.3%** | **38.6%** |

Nearly two-fifths of random-split test molecules are fingerprint-identical to something the model
trained on. Random-split AUROC reaches 0.95 against 0.87 under scaffold splitting. Random-split
numbers on this dataset should not be reported as generalisation.

A second observation tempers the first: the scaffold split is sound by its own definition but is
not a hard novelty test, since half its test molecules still have a close analog in training.
Results under it are best read as "a new scaffold within familiar chemistry".

### 5.2 A tuned random forest beats the fine-tuned language model — *demonstrated*

Ten identical scaffold splits, 631 molecules.

| Model | AUROC | ECE | Brier |
|---|---|---|---|
| **Random forest, tuned per seed** | **0.898 ± 0.057** | 0.180 | 0.125 |
| XGBoost, tuned | 0.882 ± 0.065 | 0.146 | 0.126 |
| Three-model ensemble | 0.878 ± 0.063 | 0.148 | **0.123** |
| Random forest, default | 0.872 ± 0.066 | 0.158 | 0.131 |
| Qwen3.5-9B, QLoRA fine-tuned | 0.850 ± 0.058 | 0.206 | 0.182 |
| Qwen3.5-9B, zero-shot readout | 0.651 ± 0.144 | 0.158 | 0.229 |

The fine-tuned model is significantly worse than the tuned forest: **ΔAUROC −0.049, 95% CI [−0.076,
−0.021], p = 0.007**, better on only 2 of 10 seeds. Fine-tuning itself works emphatically — **+0.199
AUROC over the zero-shot readout, 10/10 seeds, p = 0.001** — it simply does not close the gap.

This is the project's original question, answered negatively and cleanly.

### 5.3 The novel-chemistry advantage was a pooling artefact — *withdrawn*

An earlier analysis reported that on the 42 molecules with no training analog above 0.5 Tanimoto,
the zero-shot model outperformed the random forest (AUROC 0.807 against 0.794) while the fine-tuned
model collapsed to 0.442. This motivated an entire research phase.

The number was wrong, for an instructive reason. Those were predictions pooled across ten
overlapping scaffold folds. A molecule appearing in six test folds contributes six rows — and for
the zero-shot model, which is not retrained per fold, those six rows are *the same cached
prediction*. The pooled sample is therefore neither independent nor equally weighted across models,
and the confidence interval computed from it is meaningless.

Scored once per molecule, the ordering reverses:

| | Pooled (incorrect) | One row per molecule |
|---|---|---|
| Random forest | 0.794 | **0.782** |
| Zero-shot Qwen | 0.807 | 0.764 |
| Difference | +0.013 | **−0.019, 95% CI [−0.257, +0.192]** |

The interval spans nearly half an AUROC unit. A cutoff sweep showed the apparent lead exists only at
thresholds where the subset contains one to three positive molecules, and a per-seed paired test
across the six usable seeds found nothing (p = 0.717). **The hypothesis is withdrawn.**

The underlying constraint is sample size in disguise: the 42-molecule novel subset contains **23
Butina clusters and 6 positives**. No amount of statistical care recovers power that is not there.

### 5.4 Training nondeterminism exceeded the effect under study — *demonstrated*, and its consequence *not established*

While measuring the KL-anchor effect we ran what should have been an identical configuration twice.
The two runs differed by **−0.028 ± 0.089 mean AUROC, with per-seed swings to 0.18** (seed 0:
0.834 → 0.666). The effect being measured was +0.031 ± 0.069. The experiment could not resolve its
own subject.

**Root cause.** A two-stage probe calling the real training code localised it:

| Configuration | Per-example loss Δ | LoRA weight Δ | First example differing |
|---|---|---|---|
| Default, PagedAdamW8bit | 1.14e-1 | 2.71e-2 | **8** |
| Default, AdamW8bit | 1.42e-1 | 2.48e-2 | **8** |
| Default, fp32 AdamW | 1.18e-1 | 8.15e-4 | **8** |
| `CUBLAS_WORKSPACE_CONFIG` only | 1.67e-1 | 2.13e-2 | **8** |
| **Deterministic algorithms + workspace** | **0** | **0** | — |

Inference alone is bit-exact with no flags, so 4-bit dequantization and the attention kernels are
not implicated. Training diverges at example 8 of 64 — the first forward pass after the first
optimizer step — in *every* optimizer including fp32 AdamW. The cause is therefore nondeterministic
reduction order in backward-pass kernels, not the quantized optimizer. The 8-bit optimizer is an
*amplifier*: it converts an 8.1e-4 weight disagreement into 2.5e-2, roughly thirtyfold, by pushing a
tiny difference across a quantization boundary. Setting the cuBLAS workspace alone changes nothing.

**Correction and cost.** With deterministic algorithms enabled, three complete pipeline runs on one
seed agree exactly — identical per-epoch curves, identical selected epoch, identical saved LoRA
weights compared tensor by tensor, identical metrics — at approximately 10% wall-clock cost. Under
determinism, replicates of a configuration carry no information, so the remaining seed-to-seed
spread is purely split and initialisation variance, which is shared between conditions by
construction.

We emphasise what this means for interpretation: it does not merely tighten an interval, it
*changes what the interval means*. Before the fix, training randomness and test-set sampling were
confounded. After it, all residual uncertainty is sampling over molecules.

### 5.5 KL anchors: a small, non-significant improvement — *not established*

With determinism in place, λ = 0 and λ = 1 were compared over 10 seeds on identical splits.

**At the validation-selected epoch (the model one would deploy):**

| | Overall AUROC | Novel AUROC | ECE | Brier |
|---|---|---|---|---|
| λ = 0 | 0.818 ± 0.072 | 0.722 | 0.094 | 0.183 |
| λ = 1 | **0.856 ± 0.093** | **0.757** | 0.094 | **0.173** |

**Δ overall = +0.038, paired t-test p = 0.127, better on 8 of 10 seeds. Δ novel = +0.035, 95% CI
[−0.107, +0.225].** Every metric moves in the anchors' favour and none clears significance. Against
the baseline that matters, λ = 1 remains behind the random forest (Δ −0.016 ± 0.073, better on 4 of
10 seeds).

**The forgetting effect did not reproduce.** The per-epoch curves are the reason this phase ended:

| Epoch | λ = 0 overall | λ = 0 novel | λ = 1 overall | λ = 1 novel |
|---|---|---|---|---|
| 0 (base model) | 0.660 | 0.690 | 0.660 | 0.690 |
| 1 | 0.814 | 0.715 | 0.841 | 0.567 |
| 2 | 0.851 | 0.736 | 0.857 | 0.812 |
| 3 | 0.819 | 0.553 | 0.872 | 0.708 |

An earlier non-deterministic run had shown λ = 0's novel AUROC falling monotonically from 0.690 to
0.470 across epochs — a clean, interpretable catastrophic-forgetting curve. Under deterministic
replication it rises for two epochs before falling. The monotonicity was a trend read out of
numerical noise, and it is withdrawn. Paired bootstrap on the novel subset makes the incoherence
explicit: λ = 1 − λ = 0 is **−0.148, 95% CI [−0.286, −0.038]** at epoch 1 (significantly *against*
the anchors), +0.076 at epoch 2, and +0.155 with a fourfold wider interval at epoch 3. An effect
whose sign depends on where training stops is not an effect.

**Conclusion: we have not established that fine-tuning erodes the base model's chemical prior, nor
that KL anchors prevent it.** There is currently no reproducible forgetting effect for the anchors
to prevent.

### 5.6 Calibration, uncertainty and the applicability limit — *demonstrated*

**Where prediction stops working.** Below 0.4 Tanimoto to the training set, every model ranks at
chance *on this dataset*. This is the single most practically consequential number for TG2 — and
§5.9 shows it does not transfer: CDK2 retains AUROC 0.739 in the same regime. The applicability
floor must be measured per dataset, not assumed from a published threshold.

**Post-hoc calibration fails under scaffold shift** for every fingerprint model, and the mechanism
is measurable: the fitted temperature swings from 0.4 to 3.7 across folds, i.e. it is
unidentifiable on a scaffold-split validation fold, and a random-split control confirms the
mechanism. The one exception is the fine-tuned language model, where temperature scaling does work
(ΔECE −0.029, p = 0.010), because it is overconfident in the same direction on every fold.

**Conformal prediction holds its coverage** under the same shift: 0.922 against a 0.90 target, with
mean set size 1.32 and 66% of predictions decisive. It absorbs the shift by widening its sets rather
than by breaking its guarantee. This is the clearest methodological success in the project.

**Selective prediction works on confidence but not on similarity.** Keeping the 40% of molecules the
ensemble is most confident about cuts the error rate from 15.5% to 6.5%; abstaining on low
similarity to the training set does not help at all — a counter-intuitive result given that
similarity governs where the model is valid in aggregate.

**Consequence for screening.** Scoring 2,414 approved drugs, **99.2% fall below the 0.4 Tanimoto
applicability floor** where the models rank at chance, so the nominal 35 hits are not candidates.
Approved-drug repurposing is the wrong strategy for TG2 with this training set.

### 5.7 Selectivity against Factor XIII-A — *not established*, with a *demonstrated* negative

Chronic dosing in otherwise-healthy people makes selectivity within the transglutaminase family a
safety gate rather than an optimisation. Factor XIII-A crosslinks fibrin; its inhibition causes
bleeding.

**The family cannot expand the dataset where it is weak.** Four human enzymes plus the guinea-pig
TG2 ortholog were fetched separately and never merged; an off-target measurement is never used as a
TG2 label.

| Source | Molecules new to the TG2 set | Median max-Tanimoto to it | Below 0.5 |
|---|---|---|---|
| TGM1 | 5 | 0.341 | 5 |
| TGM3 | 1 | 0.232 | 1 |
| FXIII-A | 31 | 0.315 | 24 |
| TG2, guinea pig | 41 | 0.379 | 32 |

These files are overwhelmingly selectivity panels run on molecules already present: 51 of TGM1's 53
scaffolds, 41 of TGM3's 42 and 99 of FXIII-A's 113 are already in the TG2 set. Guinea-pig TG2 is the
one genuinely distant source and **contains no actives** — exact IC50 values run 2.1 to 74 µM. TGM1
and TGM3 have 5 and 1 potent molecules respectively and are not modellable at all.

**Cross-reactivity is the norm — *demonstrated*.** Of 165 molecules measured on both enzymes, **103
of the 132 TG2-potent ones also inhibit FXIII-A at 1 µM**; only 29 are clean. This is a measured
property of the published chemical space and a caution about the covalent TG2 literature generally.

**Selectivity cannot be predicted.** This is where publication-level confounding appeared.

| Task | Molecules | Positives | AUROC | 95% CI |
|---|---|---|---|---|
| TG2 potency | 432 | 234 | 0.866 | [0.828, 0.900] |
| FXIII-A potency | 118 | 79 | 0.948 | [0.892, 0.989] |
| Selectivity, **scaffold split** | 71 | 17 | **0.974** | [0.917, 1.000] |
| Selectivity, **whole papers held out** | 15 | 9 | **0.574** | [0.250, 0.864] |
| Selectivity ranked by predicted TG2 potency | 134 | 20 | 0.454 | [0.329, 0.581] |

The 0.974 was caught before being reported as a result. Per-seed AUROC of 0.998 ± 0.006 on a task
with 17 positives is not a plausible measurement. Two diagnostics explain it: **only 3 of the 98
scaffolds in the paired set appear in both classes**, so the classes are nearly scaffold-disjoint;
and **11 of the 29 selective molecules come from a single publication**, with the two classes
largely comprising different series from different laboratories (Cbz-glycine peptidics against
phenylsulfonyl-piperazines).

Scaffold splitting cannot separate publications when one laboratory's series spans several
scaffolds. Holding out whole documents instead collapses performance by 0.40, onto an interval
straddling chance in both directions. Adding the FXIII-A model's prediction as a feature changed
nothing material (0.901 → 0.911, and that is an upper bound since the auxiliary feature was fitted
on the same molecules).

Because ranking by predicted TG2 potency carries no selectivity information (0.454), **no
selectivity-aware shortlist was produced**; attaching a selectivity column to a candidate list would
misstate what was measured.

**A later audit of the paired measurements qualifies two of the numbers above and adds one
*demonstrated* negative.** Details in [`selectivity_feasibility.md`](selectivity_feasibility.md).
Units, species, assay type and censoring are clean throughout, but *assay format* is not. 57 of the
147 within-publication pairs come from one publication that carries a single 1,643-character methods
blob on both the TG2-target and the FXIII-A-target assay, so the record cannot state which format
produced which number; 11 more pair purified-enzyme TG2 inhibition against FXIII-A inhibition *in
plasma*, read out as fibrin clot formation. **79 pairs across 5 publications are format-matched with
self-consistent descriptions, and 69 of those are one paper** — that, not 147, is the defensible
denominator for a ratio claim.

The pooled selectivity statistic is also a mixture. The two largest publications hold 126 of the 147
pairs and have **opposite median selectivity** — +0.45 log units in a TG2-directed programme against
−0.36 in an FXIII-A-directed programme where TG2 is the off-target being screened out (Mann–Whitney
p = 2.3 × 10⁻⁷). A pooled median of +0.15 therefore estimates nothing about TG2 chemistry; it
averages two opposing design objectives.

**Potency and off-target potency are positively correlated — *demonstrated*.** pIC50(TG2) and
pIC50(FXIII-A) correlate at r = +0.41 pooled and +0.49 and +0.38 within the two large publications,
all significant. The molecules that inhibit TG2 harder also inhibit FXIII-A harder, which a shared
catalytic cysteine and a covalent warhead predict. The tempting version of this analysis —
correlating potency against *selectivity* and finding +0.34 — is a component artefact, because
selectivity is defined as pIC50(TG2) − pIC50(FXIII-A) and contains the potency it is regressed on;
every observed value lies *below* the correlation two independent enzymes with these variances would
already produce.

**The practical reading needs care, and our first one was wrong.** That correlation does not mean a
potency-ranked cascade enriches for cross-reactivity. Because the correlation is well below 1, net
selectivity still rises with potency: among the paired molecules the top potency decile passes a
10-fold selectivity gate at **46.7% against a 13.6% base rate** (odds ratio 8.0, Fisher exact
p = 0.001). The two gates agree rather than conflict. What potency-first ranking does is raise
**absolute** off-target potency alongside the ratio, so a ratio-only gate selects compounds that are
both more selective and more potent on FXIII-A than those it rejects — which argues for capping
absolute off-target potency rather than for abandoning potency ranking. The enrichment is also
concentrated in one of the two series (+21.7% against +1.1%), so it is a measured association in
this data rather than a general rule.

Feasibility of the three approaches a reviewer would propose next, each declined on the data rather
than on preference: selectivity regression has 2 publications with ≥ 20 pairs, so its only valid
control yields **2 transfer observations** and no interval; selectivity classification has 13
molecules above 10-fold in the format-matched subset, 8 of them in one publication; and multi-task
learning shares a representation across *tasks* when the binding shortage is **groups**. Within a
single series of 69 molecules, the smallest correlation detectable at 80% power is 0.34.

### 5.8 Covalent inhibition efficiency — *not established*

Most published TG2 inhibitors are covalent, and the IC50 filter discards 990 of 1,583 TG2 records.
We audited that discarded data.

**Three quantities, kept separate.** Ki is an affinity (nM); kinact a rate of bond formation (/min);
kinact/Ki a second-order rate constant (/M·s). Only the last is comparable across compounds with
differing affinities, so only it is modelled, and none is mixed with the IC50 labels.

**288 molecules reduce to 174.** ChEMBL stores kinact/Ki under three unit conventions and 118 rows
carry none at all. A unitless rate constant cannot be placed on a common scale, and inferring the
convention from magnitude would be fabrication, so two entire publications (103 rows) are lost. What
survives: **174 molecules, 70 Murcko scaffolds, 6 publications**, spanning log₁₀(kinact/Ki) from 1.60
to 5.00.

**One warhead, which cuts both ways.** 169 of the 174 molecules are acrylamides. A model therefore
cannot succeed by recognising the electrophile — the concern one would normally raise about a
covalent dataset — because there is only one. Equally, there is no mechanism contrast to classify:
assay descriptions state irreversible inhibition for 238 molecules and reversible for 14.
**Mechanism classification was therefore not attempted.**

**The label is partly a property of the laboratory.** Between-publication SD of medians is **0.50
log** against a within-publication SD of **0.57 log**, with each paper using a different substrate.
This is the regime Kramer et al. describe for public Ki data [11] and Kalliokoski et al. for IC50
[12], reached independently here on a rate constant.

| Model | n | RMSE (log) | 95% CI | Mean within-paper Spearman |
|---|---|---|---|---|
| Mean of training papers | 174 | 0.644 | [0.580, 0.707] | n/a (constant) |
| Molecular size alone | 174 | 0.644 | [0.569, 0.728] | 0.029 |
| Ridge on Morgan bits | 174 | 0.759 | [0.693, 0.826] | 0.110 |
| Random forest on Morgan bits | 174 | **0.598** | [0.537, 0.653] | 0.120 |

Paired over the same molecules, **random forest − mean = −0.046 log RMSE, 95% CI [−0.101, +0.011],
P(no better) = 0.054**: about 7% of the label's standard deviation, on an interval including zero.
Mean within-paper Spearman of 0.120 means the model ranks analogs inside a new series at close to
chance — precisely the task a prioritisation would require. Ridge performs *worse* than a constant,
which is the expected behaviour of a sparse high-dimensional featurisation given a weak signal.

**And the scaffold split is optimistic here too.** Under scaffold splitting the same random forest
reaches RMSE 0.489 [0.404, 0.571] against the constant's 0.660 — a quarter reduction that evaporates
under the publication split, for the same reason as §5.7. This is the second occasion in this
project on which a split that is correct for one question would have produced a publishable-looking
number for another.

### 5.9 Independent validation on CDK2 and MAO-B — *three failures reproduce, one claim is corrected*

A single target cannot distinguish "this is how small bioactivity datasets behave" from "this is how
the TG2 dataset behaves". We therefore repeated the diagnostics on an independent target. Five
candidates were scored on whether they could *test* the failures — principally on publications
contributing ≥10 labelled molecules, the criterion TG2 failed — with disqualifying thresholds fixed
before the numbers were seen. CDK2 (CHEMBL301) was selected: 2,646 molecules, 53% positive, 1,184
scaffolds, 351 publications, and chemistry unrelated to TG2 (median Tanimoto 0.220, one shared
molecule). Full detail in [`second_target.md`](second_target.md).

CDK2 forced one provenance decision TG2 did not. Its IC50 records mix binding (3,555), functional
(25) and ADMET (17) assays, and the functional rows are MTT cell-viability readouts with a median
of 14,100 nM against 200 nM for binding assays. Labels are now restricted to binding assays; this is
verified byte-identical for every transglutaminase file, which are 100% type B.

**A prediction was registered before running.** Because CDK2 spreads its positives across many
publications (largest holding 9%, against 38% for TG2's selectivity set), we predicted that
publication-level confounding would largely *not* reproduce. **The prediction was wrong**, and it is
reported because a prediction that surfaces only when it succeeds is worthless.

| Split | Shared scaffolds | Shared papers | Mean max Tanimoto | Identical fingerprint | RF AUROC | 95% CI |
|---|---|---|---|---|---|---|
| Random | 110.4 | 104.4 | 0.752 | 7.3% | 0.928 | [0.916, 0.939] |
| Scaffold | 0.0 | 69.2 | 0.654 | 0.2% | 0.898 | [0.884, 0.913] |
| **Publication** | 11.2 | **0.0** | 0.436 | 1.0% | **0.830** | [0.808, 0.850] |

Random → scaffold costs −0.030 AUROC; scaffold → publication costs a further **−0.068**, and the
intervals do not overlap. Calibration degrades along the same axis (ECE 0.027 → 0.042 → 0.092).

The `shared papers` column gives the mechanism directly rather than by inference: a scaffold split
removes scaffold overlap entirely and still leaves **69 publications with molecules on both sides**,
because one laboratory's series spans several Murcko scaffolds.

**What reproduces, and how strongly:**

| Failure | TG2 | CDK2 | Verdict |
|---|---|---|---|
| Random-split leakage | 38.6% identical, ≈0.08 AUROC | 7.3% identical, 0.030 AUROC | Reproduces, milder |
| Pooled predictions | severe (cached, up to 10×) | 18% interval narrowing (1.6×) | Reproduces, milder |
| Publication confounding | 0.974 → 0.574 | 0.898 → 0.830 | **Reproduces, largest effect** |
| Nondeterminism | −0.028 ± 0.089 | not retested (no GPU training) | Untested |
| Chance below 0.4 Tanimoto | yes | **no — AUROC 0.739** | **Does not transfer** |

The pooling artefact reproduces but mildly, and the comparison identifies its severity condition:
on CDK2 every model is refitted per fold, so molecules recur only 1.6× and the interval narrows by
18%. TG2's severe case required a model that was *not* refitted — a cached zero-shot prediction
counted in every fold it appeared in. The artefact is general; it is dangerous specifically when a
fixed model is compared against a refitted one, which is the comparison every "LLM versus classical
baseline" study performs.

**A third target tests whether the ordering is stable.** MAO-B (CHEMBL2039) is larger again
(5,840 molecules, 586 publications, 36% positive) and a different enzyme class once more, a
flavin-containing oxidase. Its pattern matches: random → scaffold −0.059, scaffold → paper a
further −0.076 (RF AUROC 0.923 → 0.865 → 0.789, no interval overlapping the next), with **122.2 of
586 publications straddling a scaffold split on average**, and 8.9% of its random-split test
molecules fingerprint-identical to a training molecule. On both independent targets the publication penalty
exceeds the scaffold penalty — 2.3× on CDK2, 1.3× on MAO-B. Two datasets differing in size, enzyme
class, class balance and publication count agree that the split which matters most is the one
rarely reported.

One detail of the CDK2 result does not reproduce: calibration degrades monotonically with split
severity on CDK2 (ECE 0.027 → 0.042 → 0.092) but not on MAO-B, whose paper split (0.035) is better
calibrated than its scaffold split (0.050). We offer no mechanism and do not speculate.

The novelty result is the one that corrects us. Reliability declines with chemical distance on both
targets (CDK2: 0.739 → 0.915 across similarity bins), but TG2's floor does not transfer: in the
below-0.4 bin CDK2 scores 0.739 on 799 molecules at 39.7% positive — a large, balanced bin rather
than a small-sample artefact — and MAO-B scores 0.622 on 704 molecules. Three targets give three
different floors with the same qualitative trend, so no fixed similarity threshold should be
carried between datasets. §5.6 has been qualified accordingly.

### 5.10 Nine modern entrants do not beat the random forest — *demonstrated*

The obvious objection to §5.2 is that it compares a language model against fingerprints and
concludes nothing about *molecular* deep learning. We therefore ran the comparison the objection
asks for: Chemprop v2 (D-MPNN), Chemprop with RDKit 2D descriptors, CheMeleon (descriptor-pretrained
D-MPNN), ChemBERTa-2 frozen and fine-tuned, MoLFormer-XL frozen, and a tabular foundation model,
against the project's three tuned baselines and both Qwen configurations. Identical splits exported
once and re-read by every entrant, 20 seeds, a 12-configuration budget for each new entrant, and a
reproduction gate passed before any new model ran (random forest 0.898 ± 0.057 reproduced to 1e-16).
Full protocol and results in [`sota_comparison.md`](sota_comparison.md).

**Scaffold split, primary endpoint.** The tuned random forest leads at **0.864 ± 0.077** AUROC.
XGBoost (Δ −0.010) and logistic regression (Δ −0.015) are statistically equivalent to it, inside the
0.05 margin fixed in advance. Every other entrant is significantly worse after Holm correction
across eleven comparisons: ChemBERTa-2 fine-tuned −0.033 (p = 0.045), the tabular model −0.034,
Qwen QLoRA −0.049, CheMeleon −0.062, Chemprop+RDKit −0.068, ChemBERTa-2 frozen −0.071, MoLFormer-XL
−0.104, Chemprop −0.115, Qwen zero-shot −0.250. **No specialized molecular model beat the random
forest on any metric.**

**Why.** We offer three reasons, in decreasing confidence. (i) **Sample size governs.** 631
molecules is below the range in which learned representations overtake fingerprints in published
comparisons; the two entrants that arrive with a pretrained representation and fine-tune only a head
(ChemBERTa-2) or a readout (CheMeleon) beat the two that learn a representation from scratch
(Chemprop, −0.115), which is the ordering that hypothesis predicts. (ii) **The task is
scaffold-local.** 96% of the scaffold-split test molecules sit above 0.4 Tanimoto to their training
fold, where substructure matching is sufficient and a learned representation has little to add.
(iii) **Pretraining scale does not help here, and may hurt.** MoLFormer-XL is trained on 1.1 billion
molecules against ChemBERTa-2's 77 million, is 13× larger in parameters, and scores 0.033 *lower*
frozen. Two SMILES transformers an order of magnitude apart in corpus size, both below fingerprints
and the larger one lower, make this a statement about the representation class on small datasets
rather than about one checkpoint.

One result deserves its own line because it qualifies the headline. **Qwen QLoRA has the best AUPRC
in the study, 0.896 ± 0.041**, while being significantly worse on AUROC. It ranks the top of the
list well and the remainder poorly. Prioritising compounds is a top-of-list task, so this is a real
qualification on "the random forest wins", not a footnote.

### 5.11 The ranking inverts under publication held-out evaluation, inconclusively — *not established*

Both publication arms cover **the same 444 molecules**, so splitting strategy is the only variable.

| | scaffold (same 444) | publication held out | Δ |
|---|---|---|---|
| Random forest | 0.854 | **0.612 ± 0.178** | **−0.242** |
| Logistic regression | 0.863 | 0.624 | −0.239 |
| XGBoost | 0.855 | 0.573 | −0.282 |
| ChemBERTa-2 frozen | 0.808 | **0.701 ± 0.172** | −0.107 |
| ChemBERTa-2 fine-tuned | 0.823 | 0.675 | −0.148 |
| Chemprop + RDKit 2D | 0.778 | 0.681 | −0.097 |

**The decline has a measured mechanism rather than a conjectured one.** Under leave-one-publication-
out, **64% of molecules fall below 0.4 Tanimoto to their training fold, against 4% under a scaffold
split.** A publication contributes a congeneric series; holding one out removes a whole region of
chemical space, whereas holding out a scaffold leaves the same laboratory's neighbouring scaffolds
in the training fold. The applicability floor of §5.6 is not a different finding from the
publication penalty — it *is* the publication penalty, seen from the molecule's side. Within the
below-0.4 band the two protocols agree (random forest 0.729 on 283 LOPO molecules), so no model
degrades on unfamiliar chemistry *because* of the protocol; the protocols differ in how many
unfamiliar molecules they present.

**The inversion is real and not statistically conclusive, and both halves of that sentence matter.**
The fingerprint models fall furthest and the frozen transformer least, moving ChemBERTa-2 frozen
from seventh to first and the random forest from first to seventh. But **every Holm-corrected p-value
on this arm is 1.000.** Eleven folds with per-fold standard deviations of 0.15–0.26 cannot resolve a
0.09 difference: the largest effect, ChemBERTa-2 frozen at Δ +0.089, carries a 95% interval of
[−0.027, +0.205]. The design is the limit, not the analysis — leave-one-publication-out over 11
publications yields 11 observations, and there is no seed to average over because every molecule is
tested exactly once. We therefore report the inversion as a hypothesis with a stated effect size and
decline to rank the models on this arm. Reading it as "frozen chemical language models generalize
better across laboratories" is the interesting conclusion and is not supported by these data; it is
testable on a target with hundreds of publications, which §5.9's CDK2 and MAO-B sets provide.

### 5.12 Base Qwen does not hold the TG2 labels — *demonstrated*, with bounded scope

§5.2's QLoRA result would mean something quite different if the base model had memorised these
IC50s from the literature. We probed it directly: 631 molecules, base weights with no adapter at the
revision the fine-tuning used, greedy decoding, five prompt arms. Full write-up in
[`qwen_label_probe.md`](qwen_label_probe.md).

| arm | answered | Spearman vs truth | distinct answers | AUROC @1 µM |
|---|---|---|---|---|
| SMILES only | **0 / 631** | — | 0 | — |
| SMILES + compound name | 8 / 631 | — | **1** | 0.500 |
| re-rooted SMILES | **0 / 631** | — | 0 | — |
| SMILES, number forced | 631 / 631 | 0.199 | **6** | 0.581 |
| re-rooted SMILES, number forced | 631 / 631 | **0.278** | **6** | **0.647** |

Permitted to abstain the model abstains on every molecule. The eight answers in the named arm are
all the example value printed in the prompt itself, two of them on Isatin and 5-Bromoisatin — the
trivially-named molecules most likely to be recalled. Forced to answer it emits six distinct values
for 631 compounds, 276 of which receive the identical `1.2 µM`.

**The decisive control is re-rooting.** A re-rooted SMILES is the same molecule written from a
different starting atom, asserted in code to round-trip to the same canonical form. Memorised text
is keyed on the string and must degrade; chemical inference is keyed on the structure and must not.
Re-rooting *improves* Spearman by 0.080 and AUROC by 0.067 and moves the median prediction from
pIC50 5.96 to 8.82 — a thousandfold shift in central tendency, placing the median prediction above
the most potent compound in the dataset. A lookup table cannot do that. The weak positive
correlation is a two-level nanomolar/micromolar guess whose prior depends on surface form; near-exact
agreement is 0.056 against a permutation chance rate of 0.049.

**What this establishes.** Experimental-activity leakage does not explain the QLoRA result; it is
learned from the fine-tuning data. **What it cannot establish.** One model, one revision, five
prompts, greedy decoding. A negative probe bounds what these prompts extract and is not proof of
absence from the pretraining corpus, which stays unauditable — **0 of 631 molecules can be checked
against the actual corpora**, while 398 are provably in PubChem and 460 in ChEMBL's TG2 file. It
speaks to labels, not structures. Structure familiarity points the same way by a separate argument:
the entrants that most plausibly saw these structures during pretraining are the ones that lost, so
structural leakage would bias §5.10 *against* its own conclusion.

### 5.13 Abstention is honest under publication shift but not useful — *demonstrated*

§5.6 reported conformal prediction holding 0.922 coverage against a 0.90 target under scaffold
shift, and called it the project's clearest methodological success. That claim now has a scope
condition. Details in [`lopo_uncertainty.md`](lopo_uncertainty.md).

Eleven leave-one-publication-out folds, six entrants, with the calibration set carved from the
**training** publications and the held-out publication's labels used only to score. Two calibration
regimes: random training molecules, which is what §5.6's recipe implies, against whole held-out
training publications.

| | target 0.90 | target 0.80 |
|---|---|---|
| i.i.d. calibration | **0.784** | **0.601** |
| publication-wise calibration | **0.914** | **0.810** |

**The recipe that works under scaffold shift under-covers under publication shift**, by 0.11–0.14,
and issues empty prediction sets — the method announcing that a molecule is unlike its calibration
data — for up to 10% of molecules. Calibrating on whole held-out publications restores coverage at
every α tested and emits no empty sets: **+0.113 coverage [+0.078, +0.148] paired over 66
fold-and-model pairs, t-test p = 2.5 × 10⁻⁸, sign test p = 2.8 × 10⁻⁶**. The three fingerprint
models fail i.i.d. calibration worst (0.691–0.806) and the two ChemBERTa-2 variants are nearly
unaffected (0.874, 0.894), the same ordering as the AUROC decline in §5.11; we note the agreement
and claim no mechanism. Mondrian quantiles give the same coverage with 30% more decisive answers
and are the variant to use here.

**The practical result is negative, and it is the one a screening campaign cares about.** At
α = 0.1, publication-wise calibration answers decisively on **15–32% of molecules and is wrong on
25–39% of those**. Marginal 90% coverage is achieved by returning both labels on two-thirds of the
set, which concentrates the error in the decisive third. Confidence-based selective prediction,
which cut error from 15.5% to 6.5% under scaffold shift, moves it only from **45.9% to 35.3%** here
— at full coverage the fingerprint models are wrong on 45–47% of molecules against a 48% positive
rate, i.e. coin-flip accuracy on an unseen publication.

Two further observations. **Coverage survives where ranking does not**: in the below-0.4 Tanimoto
band, which holds 80% of molecules on this arm, coverage stays within 0.03 of target while AUROC is
near chance. Honesty transfers to unfamiliar chemistry and discrimination does not. And **conditional
coverage visibly fails** — logistic regression covers 0.375 in the [0.7, 1.01] band — which is the
guarantee conformal prediction does not make, printed rather than elided. ChemBERTa-2 frozen has the
only usable risk–coverage curve (0.358 → 0.227) but the paired per-fold difference from the random
forest is not significant (p = 0.63–0.91, random forest worse in 2–5 of 11 folds): the same
inconclusive shape as §5.11, reached from an independent metric.

---

## 6. Synthesis: six failures, six different controls

The failures are worth separating because they are not variations on one theme, and no single
methodological habit catches all of them. The first four were found during the original work; the
fifth and sixth were found later, by the state-of-the-art comparison and the selectivity audit, and
they are of different kinds again.

**Molecular redundancy (§5.1)** is the familiar one, addressed by scaffold or time splitting [7, 9]
and quantified by bias measures such as AVE [10]. It is detected by asking how similar test
molecules are to training molecules.

**Non-independence of pooled predictions (§5.3)** is a statistical rather than chemical failure.
The split was correct; the aggregation was not. It is detected by asking what the independent unit
of observation is, and it is aggravated when models differ in whether they are refitted per fold —
a cached zero-shot prediction repeated across folds is not ten observations. We suspect this is
under-appreciated precisely because it arises when comparing a fixed model against a refitted one,
which is exactly the comparison any "LLM versus classical baseline" study performs.

**Numerical nondeterminism (§5.4)** is invisible to any amount of chemical or statistical care. It
is detected only by running the identical configuration twice and comparing. Our experience
suggests this should be a routine first step whenever a GPU-trained model is used to measure an
effect smaller than about 0.05 AUROC. The specific lesson — that the divergence originated in
backward-pass kernel selection rather than in 4-bit quantization or the paged optimizer, both of
which were our initial suspects — may save others a misdirected investigation.

**Publication-level confounding (§5.7, §5.8)** is detected by asking what else correlates with the
label besides chemistry. Here, the label correlated with the source publication, and the source
publication correlated with the chemical series. A scaffold split, the field's standard answer to
leakage, is powerless against this because scaffolds do not nest within publications: one
laboratory's series spans several scaffolds, so holding out a scaffold leaves the same laboratory on
both sides of the fold. We initially expected this to be a hazard of datasets assembled from a handful of papers, which
describes most selectivity panels and most kinetic datasets. **The CDK2 and MAO-B validations (§5.9) refuted
that:** the effect appeared on 2,646 molecules from 351 publications and on 5,840 from 586, in
neither case with a dominant paper, and on both it exceeded the scaffold effect (−0.068 against
−0.030, and −0.076 against −0.059). On the evidence of three targets spanning three enzyme classes
we suggest a publication-held-out split should be reported alongside the scaffold split as a matter
of course, not only for small panels.

A general caution follows from the pattern rather than from any single case. In three of the four
instances, the *erroneous* protocol produced the more publishable number: 0.95 instead of 0.87,
0.807 instead of 0.764, 0.974 instead of 0.574. Error in protocol design is not symmetric in its
consequences, and a result that looks unusually good on a small dataset is better treated as a
diagnostic signal than as a finding.

**A fifth failure, added later, is of a different kind: a method validated under one shift and
carried to another.** §5.13 found that conformal calibration on random training molecules — the
recipe that holds 0.922 coverage under scaffold shift — under-covers by 0.11–0.14 under publication
shift, and that calibrating on held-out *publications* restores it. Nothing was wrong with the
original experiment or its conclusion; what was wrong was the implicit scope. The detecting question
is not "is this method principled?" but "what exactly was it exchangeable with when it was
validated?". We note that this failure mode is specific to distribution-free methods whose guarantee
is conditional on an assumption the protocol itself violates, which makes the guarantee easy to
quote and easy to over-extend.

**A sixth, also added later, is a reasoning error of ours rather than a protocol error.** §5.7
measured r(TG2, FXIII-A) = +0.41 and we first read it as "potency-first ranking enriches for
cross-reactivity". Testing the claim directly refuted it: the top potency decile passes a 10-fold
selectivity gate at 46.7% against a 13.6% base rate (odds ratio 8.0, p = 0.001), because a
correlation of 0.41 leaves ample room for the ratio to improve. A correlation between two components
does not determine the behaviour of their difference, and the only reliable check was to compute the
quantity we actually cared about. This one is recorded because the erroneous version was the more
cautious-sounding claim, which is a direction of error the three-of-four pattern above does not
cover.

---

## 7. Discussion

### 7.1 What we would now say about language models on this task

The fine-tuned model is worse than a tuned random forest on a 631-molecule dataset, and no
configuration we tried changed that. KL anchoring moves every metric in the right direction and
clears no significance threshold. We see no evidence that a 9B language model contributes something
fingerprints cannot at this data scale, and the one hypothesis that suggested otherwise did not
survive correct scoring.

This is not a claim about language models in chemistry generally. It is a claim about one model
family, one readout method, one target and 631 molecules. A fair reading is that the LLM approach
was not given a regime where it could plausibly win: these methods are motivated by data scarcity
and transfer, and our novel-chemistry stratum — the one place transfer could have shown itself —
contains six positive molecules.

### 7.2 What the negative results say about the target

Three findings are about TG2 rather than about methodology, and they are mutually reinforcing. The
chemical space of known TG2 inhibitors is dominated by covalent acrylamides (§5.8); most known
inhibitors cross-react with Factor XIII-A (§5.7); and models are valid only within 0.4 Tanimoto of
that space (§5.6). Together these describe a target whose published chemistry is narrow, largely
non-selective, and poorly suited to the repurposing strategy that first motivated screening.

That is a useful negative conclusion for a medicinal-chemistry programme, and it was reached
computationally at zero experimental cost.

**Why potency prediction alone cannot identify a celiac treatment candidate.** The question that
motivated this project was whether machine learning could help find a compound to treat celiac
disease by inhibiting TG2. Our results support a sharper answer than "not yet", with four
independent reasons that do not depend on which model is used.

*First, the potency model is only valid where screening is not needed.* AUROC ≈0.86 holds above 0.4
Tanimoto to the training set, which is analogue ranking inside known series. 99.2% of approved drugs
and 80% of molecules on the publication-held-out arm fall below that floor, where every one of
eleven models ranks near chance (§5.6, §5.10, §5.11). A model that works on familiar chemistry and
fails on unfamiliar chemistry is not a discovery tool; it is a structure–activity interpolator.

*Second, abstention does not repair this.* Conformal prediction remains honest in that regime but
commits on 15–32% of molecules and errs on a quarter to a third of those (§5.13). There is no
subset of trustworthy predictions to act on.

*Third, potency is not the limiting property.* A TG2 inhibitor for celiac disease would be taken
chronically by otherwise-healthy people, which makes selectivity within the transglutaminase family
a safety gate. Of 147 paired molecules only 20 are 10-fold TG2-selective and 10 are selective the
wrong way; the defensible, format-matched subset is 79 pairs with 13 selective molecules, 8 of them
in one publication (§5.7). Selectivity cannot be predicted across series, and the five-gate cascade
that would combine potency with selectivity, applicability domain, uncertainty and drug-likeness
**cannot be assembled at all**: 0 molecules carry all five inputs, because measured selectivity
exists only for molecules already inside the training set. Potency prediction is therefore not one
component of a working pipeline that is missing the others — it is the only component that can
currently be evaluated.

*Fourth, even where potency ranking helps selectivity, it does not deliver clean compounds.* The
corrected finding in §5.7 is that potency and selectivity gates agree (46.7% against a 13.6% base
rate), but absolute FXIII-A potency rises along with the ratio. A cascade gating only on the ratio
selects compounds that are simultaneously more selective and more potent on the anti-target.

None of this says a TG2 inhibitor cannot treat celiac disease — ZED1227 met its primary endpoint in
a 163-patient trial. It says that the computational contribution available from this dataset is
ranking analogues of known chemotypes and quantifying when to refuse, and that the binding
constraint on progress is experimental selectivity data rather than model capacity. **No compound in
this work is a treatment candidate, and no shortlist is offered.**

### 7.3 Reproducibility as a prerequisite rather than a virtue

The determinism work (§5.4) was initially undertaken as housekeeping and turned out to be
load-bearing. Fixing it did not confirm the forgetting effect; it *removed* it. Had we not
investigated, we would have reported a clean monotone catastrophic-forgetting curve supported by a
plausible mechanism and a published remedy. The result would have been wrong, and nothing about the
chemistry or the statistics would have revealed it.

---

## 8. Limitations

**Three targets, not a survey.** The core findings come from one protein, one disease and 631
molecules, with three of the first four failures retested on CDK2 and MAO-B (§5.9). Three targets show a
failure is not unique to TG2; they do not establish a general law, and all three are human enzymes
drawn from ChEMBL through one curation pipeline. The magnitudes differ substantially between
them — 38.6% against 7.3% identical fingerprints, chance against AUROC 0.739 below 0.4 Tanimoto —
so the mechanisms appear transferable while the numbers plainly are not. Nondeterminism remains a
single-target observation, as the CDK2 phase ran no GPU training.

**Small subsets behind the strongest claims.** The novel-chemistry stratum has 42 molecules, 23
Butina clusters and 6 positives. The selectivity publication-held-out folds average 15 molecules.
The covalent dataset has 174 molecules in 6 publications, two folds of which hold 14 and 22. A
negative result at that scale shows we cannot demonstrate an effect; it does not show that none
exists. Readers should treat §5.5, §5.7 and §5.8 as failures to demonstrate rather than as evidence
of absence.

**No prospective validation.** No compound was tested in any assay. Every prediction here is a
retrospective ranking of published values.

**Stereochemistry absent** from the source SMILES, capping achievable performance (§3.4).

**One hyperparameter regime for the language model.** A single λ beyond zero, three epochs, one
LoRA rank, no learning-rate sweep — by design, to avoid a search that the evaluation could not
resolve. Validation selected the final epoch in 5 of 10 seeds, so 0.850 is a lower bound.

**The KL anchors are used off-label.** The published method motivates them by preserving general
chat behaviour; we asked them to preserve a chemical prior and never measured the former [3].

**Assay heterogeneity is handled by exclusion, not correction.** We drop incomparable records rather
than attempting inter-laboratory correction, which the literature suggests is possible in principle
for Ki/IC50 mixtures [12]. For kinact/Ki no such correction factor is established.

**Publication grouping is approximate.** Document identifiers come from the off-target side of each
pair in §5.7; a molecule's TG2 measurement may originate elsewhere.

**Assay format in the paired set is inferred from free-text titles.** §5.7's later audit classifies
each assay by regex over the ChEMBL description title. 17 of 147 pairs are too terse to classify,
and a format that agrees by title could still differ in substrate, buffer, calcium or preincubation
time, none of which the record contains. A reliable audit needs the primary publications. The 56
pairs flagged because one methods blob is attached to both assays are called *unverifiable*, not
wrong.

**Eleven folds on the publication arm.** §5.11 and §5.13 both rest on leave-one-publication-out over
11 publications, with per-fold standard deviations of 0.15–0.26 and fold sizes from 10 to 106
molecules. Every per-fold model comparison there is underpowered, and we report the two headline
differences as inconclusive rather than ranking the models. Only §5.13's coverage comparison, with
66 paired observations, is significant.

**The two conformal calibration regimes are not matched on training data.** Group calibration holds
out whole publications and therefore fits on 16 fewer molecules and fewer publications than the
i.i.d. regime. §5.13 argues the coverage gain is not an artefact of a weaker model — set sizes grow
far less than coverage, and a uniformly weaker model would not preferentially repair the fingerprint
entrants — but a cleanly matched experiment fixing the fitting set was not run.

**The leakage probe bounds prompts, not corpora.** §5.12 is one model at one revision through five
prompts under greedy decoding. 0 of 631 molecules can be audited against the actual pretraining
corpora, and the probe speaks to labels rather than structures.

**Five entrants were omitted from the uncertainty experiment.** §5.13 covers six of the eleven
models in §5.10; Chemprop+RDKit, which placed third on the publication arm by AUROC, is the most
arguable omission.

---

## 9. Future work

**Recover the missing kinetic units.** Two publications (103 rows) were dropped from §5.8 because
ChEMBL stores their rate constants without units. Reading the units from the primary literature
would grow the kinetic set by roughly half, to approximately 260 molecules across 8 publications.
This is manual work measured in hours, requires no computation, and is the single step most likely
to change a conclusion in this paper.

**Test within a single laboratory's series.** Every negative result in §5.7 and §5.8 is driven by
between-publication variation. One publication alone contributes 52 molecules on a single substrate.
A within-series model there would test chemistry rather than inter-laboratory offset — a smaller
question answered honestly instead of a larger one answered badly. We state in advance that a
52-molecule fold may also fail, and that this should not be reported as a surprise if it does.

**Independent datasets — begun, not finished.** Three of the four checks have now been repeated on
CDK2 (§5.9) and all three reproduced. MAO-B has since been added as a third target and agrees
with CDK2 on the ordering. What remains is a target from a *different data source*: all three are
ChEMBL records shaped by one curation pipeline, so a dataset curated elsewhere would test whether
publication confounding is a property of the literature or partly of ChEMBL's document attribution.
And the
nondeterminism check has still been run only once, on one GPU and one model family — the cheapest
way to generalise it is to repeat the identical-configuration comparison on any other quantized
fine-tune.

**A chemically appropriate screening library.** The one direction not yet ruled out. Approved-drug
repurposing is exhausted (§5.6), but analogs within 0.5–0.7 Tanimoto of known actives sit inside the
validated applicability domain. Covalent fragment collections would require the mechanism modelling
that §5.8 found infeasible, so they should follow the kinetic-unit recovery rather than precede it.

**Mondrian conformal prediction** stratified by scaffold cluster, to give per-chemotype rather than
marginal coverage [6]. Now partly done and partly re-prioritised: §5.13 runs the class-wise variant
on the publication arm and finds it strictly better (same coverage, 30% more decisive answers), and
also finds that conditional coverage fails in the high-similarity bands — which is the gap
scaffold-cluster stratification exists to close. It is no longer "incremental": it is the specific
repair the measured failure points at.

**Measure FXIII-A on molecules outside the TG2 training set.** This is now the step the selectivity
and screening work converges on, and it is experimental rather than computational. A five-gate
cascade combining activity, selectivity, applicability domain, uncertainty and drug-likeness cannot
be assembled from existing public data at any sample size, because the evidence sets are
structurally disjoint: **0 molecules carry all five inputs**, since a molecule with measured FXIII-A
selectivity is by construction one already in our training set, where applicability domain and
out-of-sample uncertainty are undefined. Roughly 30–50 FXIII-A measurements in one assay format on
molecules outside that set would convert the selectivity gate from unbuildable to testable, give
leave-one-publication-out a third and fourth informative group, and allow the 46.7%-against-13.6%
enrichment of §5.7 to be checked on chemistry it was not derived from. No further modelling of the
existing 147 pairs substitutes for it.

**Test the publication-split ranking inversion where it can be resolved.** §5.11's inversion is
unresolvable at 11 folds. CDK2 (351 publications) and MAO-B (586) already exist in this project and
would give a leave-one-publication-out design with enough groups to carry an interval. That is the
cheapest route from a hypothesis with an effect size to a finding, and it needs no new data.

**Prospective laboratory validation.** No computational result here justifies an assay yet. The
realistic path, in order: recover the kinetic units; build a within-series model on the largest
single publication; if it ranks held-out analogs from that series at Spearman > 0.5, propose a small
set of untested analogs for a TG2 colorimetric assay with an FXIII-A counter-screen run in parallel,
given §5.7. A counter-screen is not optional for this target. Any such proposal should be framed to
a collaborator as a test of the model, not as a set of candidate compounds.

---

## 10. Conclusion

### 10.1 Findings by status

**Statistically supported:** random splits leak on this dataset (38.6% identical fingerprints);
censored measurements are informative negatives (305 → 367 molecules); a tuned random forest beats
the QLoRA-fine-tuned 9B language model (−0.049 AUROC, 95% CI [−0.076, −0.021]); fine-tuning
substantially improves on zero-shot (+0.199, 10/10 seeds); below 0.4 Tanimoto all models rank at chance *on TG2*, while CDK2 retains
0.739 in the same regime, so the floor is dataset-specific; temperature scaling fails under scaffold shift while conformal prediction holds coverage
(0.922 against 0.90); confidence-based abstention cuts error 15.5% → 6.5%; 103 of 132 TG2-potent
molecules with paired data also inhibit FXIII-A; between-publication offset in kinact/Ki (0.50 log)
rivals within-publication spread (0.57 log); QLoRA training can be made bit-exact, and the cause of
its nondeterminism is backward-pass kernel selection; on CDK2 and MAO-B, holding out whole publications costs
more AUROC than scaffold splitting (0.898 → 0.830 and 0.865 → 0.789, non-overlapping intervals);
**no specialized molecular model beats the tuned random forest under scaffold evaluation** (nine
entrants, best Δ −0.033 at Holm p = 0.045, §5.10); **base Qwen holds none of the TG2 labels these
prompts can retrieve** — 631 of 631 abstentions, six distinct answers when forced, and re-rooting
the SMILES *improves* every metric (§5.12); **i.i.d. conformal calibration under-covers under
publication shift** (0.784 against 0.90) while publication-wise calibration holds it (0.914,
Δ +0.113, p = 2.5 × 10⁻⁸, §5.13); **TG2 and FXIII-A potency correlate positively** (r = +0.41) yet
the top potency decile still passes a 10-fold selectivity gate at 46.7% against 13.6% (odds ratio
8.0, p = 0.001, §5.7).

**Inconclusive:** KL anchors improve overall AUROC by 0.038 (p = 0.127); they change novel-chemistry
AUROC by +0.035 (95% CI [−0.107, +0.225]); selectivity is predictable across publications (0.574,
95% CI [0.250, 0.864]); covalent efficiency is predictable across publications (−0.046 log RMSE,
95% CI [−0.101, +0.011]); **the model ranking inverts under publication-held-out evaluation** —
ChemBERTa-2 frozen Δ +0.089 over the random forest, 95% CI [−0.027, +0.205], every Holm-corrected
p = 1.000 on 11 folds (§5.11); **frozen chemical language models abstain more usefully on an unseen
publication** (risk–coverage 0.358 → 0.227 against 0.459 → 0.353, paired p = 0.63–0.91, §5.13).

**Withdrawn:** the zero-shot model generalises better to unfamiliar chemistry (a pooling artefact);
QLoRA fine-tuning monotonically erodes the base model's chemical prior (did not reproduce under
deterministic training); **a potency-ranked cascade enriches for cross-reactivity** (our own
inference from r = +0.41, refuted by direct measurement — §5.7, §6).

**Scoped rather than overturned:** conformal prediction holds coverage under distribution shift —
true under scaffold shift with i.i.d. calibration, false under publication shift with the same
recipe (§5.13).

### 10.2 The contribution

On literature-derived bioactivity datasets, the evaluation protocol determined the conclusion more
often than the model did. We document four mechanistically distinct instances on TG2, each with a
measured before-and-after effect, two of which were caught only after being written down as
findings; three of the four erroneous protocols produced the more attractive number. Three of the
four then reproduced on two independent targets of different size, chemistry and provenance
structure, which also agree on which failure is largest.

The second target also constrains the claim in both directions. Publication-level confounding proved
more general than we predicted, appearing on a 351-publication dataset and there exceeding the
scaffold effect. The 0.4 Tanimoto applicability floor proved less general than TG2 suggested, and is
withdrawn as a cross-dataset claim.

The practical recommendation is modest and specific: alongside the scaffold split that is already
standard, small-dataset bioactivity studies should report (i) the independent unit behind every
confidence interval, (ii) a repeat run of the identical configuration where a GPU-trained model
measures an effect below ~0.05 AUROC, and (iii) a publication-held-out split wherever the dataset is
assembled from a small number of papers.

### 10.3 Status of this draft

This is a complete draft for supervisor review. It is **not** submission-ready: no target venue has
been selected, and formatting, length limits, author/affiliation details, data-availability and
competing-interests statements, and any venue-specific reporting checklist remain to be addressed.
Reference details below were checked against primary records where indicated and should be
re-verified at submission.

---

## References

Verification status recorded as of 2026-10-29. Items marked ✓ were checked against publisher,
PubMed or arXiv records during preparation of this draft.

1. ✓ Wichka I, Lai P-K. Rapid discovery of Transglutaminase 2 inhibitors for celiac disease with
   boosting ensemble machine learning. *Computational and Structural Biotechnology Journal*
   2024;23:3669–3679. doi:10.1016/j.csbj.2024.10.019
2. ✓ Zdrazil B, Felix E, Hunter F, et al. The ChEMBL Database in 2023: a drug discovery platform
   spanning multiple bioactivity data types and time periods. *Nucleic Acids Research*
   2024;52(D1):D1180–D1192. doi:10.1093/nar/gkad1004. Targets used: CHEMBL2730 (TGM2, P21980),
   CHEMBL2810 (TGM1, P22735), CHEMBL3363 (TGM3, Q08188), CHEMBL4530 (F13A1, P00488),
   CHEMBL3988613 (guinea-pig TG2).
3. ✓ Li Y, Wagle J. Fine-tuning LLMs as Jev models. arXiv:2610.02076. *(Preprint; not peer
   reviewed at time of writing.)*
4. ✓ Dettmers T, Pagnoni A, Holtzman A, Zettlemoyer L. QLoRA: Efficient Finetuning of Quantized
   LLMs. *Advances in Neural Information Processing Systems (NeurIPS)* 2023. arXiv:2305.14314
5. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks.
   *Proceedings of the 34th International Conference on Machine Learning (ICML)* 2017;70:1321–1330.
6. Vovk V, Gammerman A, Shafer G. *Algorithmic Learning in a Random World.* Springer, 2005.
   (Inductive and Mondrian conformal prediction.)
7. Bemis GW, Murcko MA. The properties of known drugs. 1. Molecular frameworks.
   *Journal of Medicinal Chemistry* 1996;39(15):2887–2893.
8. Butina D. Unsupervised data base clustering based on Daylight's fingerprint and Tanimoto
   similarity. *Journal of Chemical Information and Computer Sciences* 1999;39(4):747–750.
9. ✓ Sheridan RP. Time-split cross-validation as a method for estimating the goodness of
   prospective prediction. *Journal of Chemical Information and Modeling* 2013;53(4):783–790.
10. ✓ Wallach I, Heifets A. Most ligand-based classification benchmarks reward memorization rather
    than generalization. *Journal of Chemical Information and Modeling* 2018;58(5):916–932.
    doi:10.1021/acs.jcim.7b00403
11. ✓ Kramer C, Kalliokoski T, Gedeck P, Vulpetti A. The experimental uncertainty of heterogeneous
    public Ki data. *Journal of Medicinal Chemistry* 2012;55(11):5165–5173.
12. ✓ Kalliokoski T, Kramer C, Vulpetti A, Gedeck P. Comparability of mixed IC50 data — a
    statistical analysis. *PLoS ONE* 2013;8(4):e61007. doi:10.1371/journal.pone.0061007
13. Hu EJ, Shen Y, Wallis P, et al. LoRA: Low-rank adaptation of large language models.
    *International Conference on Learning Representations (ICLR)* 2022. arXiv:2106.09685
14. ✓ Rogers D, Hahn M. Extended-connectivity fingerprints. *Journal of Chemical Information and
    Modeling* 2010;50(5):742–754. doi:10.1021/ci100050t
15. RDKit: Open-source cheminformatics. https://www.rdkit.org
16. ✓ ClinicalTrials.gov, NCT04604795: Study to evaluate the safety, tolerability and
    pharmacokinetics of GSK3915393 in healthy participants. *(Phase 1, completed, 65 participants.
    GSK3915393 was subsequently removed from GSK's celiac pipeline; no reason has been published.)*
17. ✓ Endrizzi F, Grunst M, Rudloff S, De Laffolie J, Zimmer K-P, Stricker S. Targeting
    transglutaminase 2: pathways to celiac disease therapies. *Gastroenterology Report*
    2025;13:goaf086. doi:10.1093/gastro/goaf086
18. Baell JB, Holloway GA. New substructure filters for removal of pan assay interference compounds
    (PAINS) from screening libraries. *Journal of Medicinal Chemistry* 2010;53(7):2719–2740.
19. Brenk R, Schipani A, James D, et al. Lessons learnt from assembling screening libraries for drug
    discovery for neglected diseases. *ChemMedChem* 2008;3(3):435–444.
20. ✓ Schuppan D, Mäki M, Lundin KEA, et al. A randomized trial of a transglutaminase 2 inhibitor
    for celiac disease. *New England Journal of Medicine* 2021;385(1):35–45.
    doi:10.1056/NEJMoa2032441. PMID 34192430.

---

## Appendix A — Reproducibility

All computation ran locally on one RTX 5070 Ti under WSL2 Ubuntu 22.04; no hosted model APIs or
cloud GPUs were used at any stage. Environment: Python 3.12.16, PyTorch 2.14.1+cu130 (CUDA 13.0,
cuDNN 9.24), transformers 5.21.0, peft 0.21.2, bitsandbytes 0.50.2, NumPy 2.5.4.

| Experiment | Command | Cost |
|---|---|---|
| Data audit and leakage | `python scripts/check_sources.py`, `scripts/audit_splits.py` | CPU, seconds |
| Baselines | `python scripts/run_baselines.py`, `scripts/run_tuned_baselines.py` | CPU, minutes |
| Zero-shot readout | `python scripts/run_jev_probe.py` | GPU, minutes |
| QLoRA fine-tune | `python scripts/run_jev_qlora.py --seeds 10 --epochs 3 --data combined --kl {0,1} --deterministic --tag det_kl{0,1}` | GPU, ~6 h |
| Determinism probe | `python scripts/check_determinism.py --steps 64 --repeats 3` | GPU, minutes |
| End-to-end reproducibility | `python scripts/compare_repro.py` | CPU, seconds |
| Novelty statistics | `python scripts/run_novelty_stats.py` | CPU, seconds |
| KL comparison | `python scripts/run_kl_compare.py --tag det_kl` | CPU, ~1 min |
| Calibration / uncertainty | `python scripts/run_calibration.py`, `scripts/run_uncertainty.py` | CPU, minutes |
| Transglutaminase family | `python scripts/fetch_chembl_tg2.py --target all`, `scripts/audit_tg_family.py` | network + CPU |
| Selectivity | `python scripts/run_selectivity.py` | CPU, ~1 min |
| Covalent audit and model | `python scripts/audit_covalent.py`, `scripts/run_covalent.py` | CPU, ~1 min |
| Second-target selection | `python scripts/audit_candidate_targets.py` | network + CPU, ~25 min |
| Second-target validation | `python scripts/run_second_target.py --target cdk2` | CPU, ~3 min |
| Third-target validation | `python scripts/run_second_target.py --target maob` | CPU, ~5 min |

Supporting documentation: `data.md` (provenance and validation), `baselines.md`,
`calibration.md`, `generalization.md`, `uncertainty.md`, `forgetting.md` (KL anchors),
`reproducibility.md` (determinism audit), `selectivity.md`, `covalent.md`, `second_target.md` (CDK2 validation), `candidates.md`,
`report.md` (consolidated internal report). Thirty automated tests cover the data-handling and
statistical logic, including unit conversion, censored-label rules and split-grouping invariants.

## Appendix B — Figures

| Figure | File | Shows |
|---|---|---|
| 1 | `results/figures/similarity_vs_auroc.png` | Model performance against maximum Tanimoto to training set; the 0.4 applicability floor |
| 2 | `results/figures/model_comparison.png` | All models across ten scaffold splits |
| 3 | `results/figures/conformal.png` | Conformal coverage and set size under random and scaffold splits |
| 4 | `results/figures/risk_coverage.png` | Selective prediction: error against coverage for confidence- and similarity-based abstention |
| 5 | `results/figures/forgetting.png` | Overall and novel-subset AUROC per epoch for λ = 0 and λ = 1, deterministic runs |
