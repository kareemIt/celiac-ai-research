# When the Split Decides the Answer: Reliability of Molecular Property Prediction under Publication and Chemical Distribution Shift, with Transglutaminase 2 and Celiac Disease as the Motivating Application

**Draft for supervisor review — rev. 4, 2026-11-05 (hierarchical conformal comparison added;
references verified; full numerical and statistical audit)**

*Status: complete draft, ready for supervisor review, **not** submission-ready. See §10.3 for the
blocking items and [`audit_2026-11-05.md`](audit_2026-11-05.md) for the correction ledger.*

**What this paper is about.** Whether a machine-learned model of molecular bioactivity can be
trusted when the molecules it is asked about come from a publication, and a region of chemical
space, that it has never seen. Transglutaminase 2 and celiac disease are the motivating
application and the reason the dataset looks the way it does — small, literature-derived, dominated
by analog series — and that shape is what makes the reliability question sharp. **No compound in
this work is a treatment candidate and none was synthesised or assayed** (§1.3, §7.2).

**Structure.** §1 introduction and motivation · §2 related work · §3–4 dataset and methodology ·
§5 experimental results · §6–7 discussion · §8 limitations · §9 future work · §10 conclusion ·
§11 verified references · appendices A–B. Section numbers are stable across revisions, which is why
§3 and §4 remain separately numbered rather than merged.

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
not reproduce. (iv) **Publication-held-out collapse**: a selectivity classifier scoring AUROC 0.974
under scaffold splitting falls to 0.574, 95% CI [0.250, 0.864], when whole source publications are
held out, because scaffolds do not nest within papers. We originally attributed this to publication
identity; §5.14 corrects that attribution, and the mechanism is unresolved rather than established.

We additionally report that cross-reactivity is the norm in this chemical space (103 of 132
TG2-potent molecules with paired data also inhibit Factor XIII-A at 1 µM), that the transglutaminase
family cannot expand the dataset where it is weakest, and that covalent inhibition efficiency
(kinact/Ki) does not predict across laboratories (−0.046 log RMSE against a constant predictor, 95%
CI [−0.101, +0.011]). We then test three of the first four failures on an independent target, CDK2 (2,646 molecules, 351
publications, chemically unrelated to TG2) and again on MAO-B (5,840 molecules, 414 publications).
All three reproduce in direction and mechanism while differing in magnitude, and both independent
targets agree that the publication split costs more than the scaffold split, on a large, diverse
dataset whose positives are spread across hundreds of papers — contrary to our own pre-registered
prediction that it would not appear there. **Its attribution is corrected in §5.14**: most of that
difference is the unit the metric is averaged over plus the novelty of the molecules a publication
split places in the test set, and any remainder is confounded between publication identity and
assay protocol, so we no longer describe it as publication-level confounding. The second target also corrects us: TG2's finding that models rank at chance below 0.4
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
calibrating on held-out publications brings measured coverage back to target (0.914, Δ +0.113,
p = 2.5 × 10⁻⁸). **This is the one result that replicates cleanly on all three targets**
(Δ +0.084 on CDK2, +0.090 on MAO-B, every p < 10⁻⁷), and it is also the one we are most careful
about: grouped calibration does **not** restore exchangeability and carries no distribution-free
guarantee under publication shift, so we report it as an empirical regularity. **We then run the
method that does have a theorem for a test point from an unseen group** — split hierarchical
conformal prediction [21, 22] — on the same saved predictions: it holds its nominal floor in all 54
runs where ours holds it in 22, the advantage shrinks with the number of calibration publications at
exactly the rate its own upper bound predicts, and it pays for the floor by roughly halving the
molecules given a decisive answer. Leave-one-publication-out cannot support it at all, needing nine
unseen calibration publications where that protocol supplies one. Its practical
value is limited in the same way on every target: it commits on only 15–32% (TG2), 41–48% (CDK2)
and 46–56% (MAO-B) of molecules and errs on 16–39% of those, so marginal coverage is restored and
decision quality is not. The fifth failure is thus a
method carried outside the shift it was validated under; the sixth is our own inference from a
correlation between two components to the behaviour of their difference, which direct measurement
refuted.

We distinguish throughout between statistically supported findings, inconclusive results, withdrawn
hypotheses and findings since scoped, and we claim a three-target case study rather than a general
law.

**Keywords:** transglutaminase 2, celiac disease, evaluation methodology, distribution shift, data
leakage, conformal prediction, covalent inhibitors, large language models

---

## 1. Introduction and motivation

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
it as a proof-of-concept trial. ZED1227's design is described by its originators: it binds the
active state of TG2 selectively and forms a covalent bond with the catalytic-site cysteine [30].
A transcriptomic analysis of 116 duodenal biopsies from 58 patients in the same programme reports
that oral TG2 inhibition kept expression profiles for mucosal morphology, inflammation,
differentiation and nutrient absorption close to gluten-free-diet levels [31].
No TG2 inhibitor has yet been approved, and not every programme has progressed — GSK3915393
completed a Phase 1 study in healthy volunteers (NCT04604795) and was subsequently removed from
GSK's celiac pipeline, with no reason published [16].

**TG2 inhibition is also not the only route being pursued**, which bounds how much any
TG2-focused result can claim. Current reviews of the celiac pipeline cover luminal gluten-degrading
enzymes, barrier modulation and oral tolerance induction alongside TG2 inhibition [32, 34], a
systematic review treats TG2 inhibitors and regulatory-T-cell approaches as parallel strategies with
an evidence base that is still mostly preclinical or early-phase [33], and the diagnostic and
clinical-trial landscape is reviewed by Tye-Din [35]. We cite these to place the target, not to
compare strategies: this paper evaluates prediction for one target and says nothing about which
therapeutic route is most likely to succeed.

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

Across eight experimental campaigns we repeatedly found that our own evaluation protocol, rather than any
modelling choice, was setting the answer. Two results were caught as artefacts only after they had
been written down as findings. Rather than discard them, we treat them as the subject matter.

The contribution of this paper is a documented case study of **six mechanistically distinct
evaluation failures on one small, literature-derived bioactivity dataset**, each with a measured
before-and-after effect size, three of them then retested on two independent targets:

| Failure mode | Where it bit | Effect of correcting it |
|---|---|---|
| Random-split leakage | Baseline model selection | AUROC 0.95 → 0.87 |
| Pooling across overlapping folds | Novel-chemistry hypothesis | +0.013 → −0.019 (hypothesis withdrawn) |
| Training nondeterminism | KL-regularisation ablation | Noise floor −0.028 ± 0.089 → exactly 0 |
| Publication-held-out evaluation (attribution corrected, §5.14) | Selectivity classification | AUROC 0.974 → 0.574 |
| A guarantee carried outside its shift | Conformal calibration recipe | Coverage 0.784 → 0.914 against a 0.90 target (replicates on both other targets, §5.13) |
| Inferring a difference from its components | "Potency enriches cross-reactivity" | 13.6% → 46.7% gate pass rate (claim withdrawn) |

These are not six instances of one problem. Leakage is about molecular redundancy; pooling is
about the independence assumption in a confidence interval; nondeterminism is about numerical
reproducibility on GPU hardware; the publication-held-out decline is about the novelty and the measurement protocol of what is
held out (§5.14);
the fifth is about the scope condition attached to a distribution-free guarantee; and the sixth is
not a protocol error at all but a reasoning error of ours, caught only by computing the quantity we
actually cared about instead of a correlate of it. Each requires a different control, and standard
practice in the field addresses only the first two consistently.

We state explicitly that the magnitudes are findings about **this dataset and this target**. Three
of the six failures were retested on CDK2 and MAO-B (§5.9, §5.13, §5.14) and reproduce in direction
while differing substantially in size; the remaining three — nondeterminism, pooling and our own
reasoning error — remain single-target observations, the first because the replication phase ran no
GPU training. What we claim is that each failure is mechanistically plausible wherever a dataset is
assembled from heterogeneous literature and dominated by analog series — a description that fits a
large fraction of published bioactivity benchmarks — and that one of them, the conformal coverage
failure, now has three-target evidence behind it.

A note on what this paper does **not** claim, since the subject matter invites the inference.
TG2 is a validated target in celiac disease [20] and the motivation for the dataset, but no
compound here was synthesised or assayed, every number is a retrospective ranking of published
measurements, and nothing in this work identifies a candidate therapy. The therapeutic framing
explains why the dataset is small and literature-derived — which is precisely what makes it a good
subject for evaluation methodology — and that is the whole of its role.

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
complement to random splits [9]. More recently, Guo, Hernandez-Hernandez and Ballester have argued
that the scaffold split itself is optimistic, reporting substantially worse performance under
UMAP-based splits across 60 NCI-60 datasets [27], and the broader literature has converged on the
view that no single split is uniformly correct. MoleculeNet established the benchmark-suite
convention these arguments operate within [36].

Two independent analyses published since report the same collapse we measure, on different data and
a different task, which is the strongest external support our grouped-evaluation argument has.
Agboola et al. built 186,663 matched molecular pairs from 16,897 Ki and Kd measurements across 20
human kinases and compared five within-kinase partitioning schemes of increasing stringency against
leave-one-kinase-out validation: within-kinase average precision fell from 0.821 to 0.309 under
Murcko-scaffold grouping and to 0.211 under connected-molecular-component grouping, at which point
within-target and cross-target performance were no longer distinguishable (p = 0.31 and p = 0.17)
[46]. Tamura, Miyao and Bajorath report the same direction across 100 ChEMBL activity classes:
median balanced accuracy near 0.7 and MCC near 0.5 when pairs are split randomly, falling to about
0.6 and below 0.25 when 20% of *compounds* are held out so that training and test sets are
structurally distinct, with accuracy not scaling with model complexity [43].

Three qualifications matter for how much we may lean on them. Both evaluate *pair* classification,
where we classify single compounds, so the collapse is analogous rather than identical. The
non-significant p values in [46] show indistinguishability at that sample size, not equality. And
the memorisation mechanism asserted in [43]'s abstract is the authors' interpretation of the
leakage comparison — the paper contains no control experiment isolating memorisation — so we cite
the collapse as observed and not its explanation, which is the same distinction we hold ourselves to
in §8.

Our work differs in scope and in kind. Those papers establish that redundancy inflates benchmark
scores in general. We document, on a single target, that six *separate* protocol decisions each
independently determined a conclusion we were about to report, including three that neither AVE
bias nor time-splitting would have detected: GPU-level training nondeterminism, the unit an AUROC
is averaged over, and the inseparability of source publication from assay protocol. Where that
literature asks *which split is hardest*, our §5.14 asks a prior question — whether two protocols
that report different numbers are even measuring the same quantity — and finds on CDK2 that most of
the reported difference is the aggregation unit rather than the split.

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
(§5.6), which is consistent with theory but, to our knowledge, not widely demonstrated on a small
single-target medicinal-chemistry dataset.

**Conformal prediction is well established in this field, and so is group-aware calibration.** We
state this explicitly because our §5.13 result is easy to mistake for a new method, and it is not.
Conformal prediction has been applied to bioactivity prediction at scale across 550 ChEMBL targets
[25] and proposed as a transparent, consistent treatment of the applicability domain [24]; the
latter also supplies the framing this paper leans on throughout, that a classification prediction
set may contain no label, one label or several, so set size is part of the result rather than a
detail. Mondrian (class-conditional) calibration is standard and long-established [6]. Calibrating
within groups rather than pooling is likewise known: clustered conformal prediction groups classes
with similar score distributions and calibrates per cluster [26], and *hierarchical* conformal
prediction treats the two-level sampling structure directly and gives distribution-free sets for an
observation from a **previously unseen group** under hierarchical exchangeability [21, 22]. Barber
et al. [23] supply the general account of what happens when exchangeability fails: coverage degrades
by a bounded gap rather than vanishing.

**Closest prior art, and it is close.** Rakhshaninejad et al. [28] evaluate conformal prediction for
drug–target interaction on the KIBA dataset under four splits including drug-based, protein-based and
fully unseen drug–protein pairs, and compare marginal, group-conditioned and three cluster-conditioned
variants, reporting subgroup coverage. That is the same question we ask — does conformal coverage
survive a group-structured split of bioactivity data, and does group-aware calibration repair it —
on a different grouping. We found it only during the final audit, and it substantially narrows what
we can claim.

What remains, stated at the size it actually is. **The methodological novelty is nil**: every method
we run is published and we propose no estimator. The empirical contribution is three things. First,
the group is the *source publication* of a measurement — not the drug or the protein — which in
literature-derived single-target data is the grouping a screening campaign actually faces, and which
in ChEMBL coincides exactly with the assay, so the two are not separately identifiable (§5.14). We
have not found that identifiability result stated elsewhere. Second, the comparison includes a
method with a finite-sample guarantee for an unseen group (§5.16), which [28] does not test. Third,
and most useful practically, **leave-one-publication-out evaluation structurally cannot support a
guaranteed calibration at the conventional α = 0.1**, because it leaves one unseen publication per
fold against the nine such a statement requires — a constraint on experimental design that follows
from the theorem and, as far as we can tell, has not been pointed out. We claim a three-target case
study, not a method.

### 2.4 Language models as constrained predictors

The LLM-as-Jev framework extracts calibrated categorical decisions from a causal language model by
reading next-token probabilities over numbered option identifiers, with an optional fine-tuning
recipe that adds KL anchors to a frozen reference model [3]. We implement that method faithfully,
including the three anchor terms, and use QLoRA [4] over LoRA adapters [13] for the fine-tuning
itself. Our use is off-label: the published method targets general decision tasks and its anchors
are motivated by preserving general chat behaviour, whereas we ask them to preserve a *chemical*
prior. We flag this extrapolation explicitly rather than presenting it as a replication.

---

## 3. Dataset

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

## 4. Methodology

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

**The independent unit is not the same in every experiment, and getting it wrong was one of the six
failures (§5.3).** We therefore state it per regime.

*Under scaffold and random splitting*, the molecule is the independent unit. Because folds overlap
across seeds, pooled predictions are collapsed to one row per molecule before any interval is
computed (§5.3 explains why). Confidence intervals are bootstrap percentiles over molecules, 2,000
replicates unless stated, and paired comparisons resample the same molecules on both sides.

*Under publication-held-out evaluation, the molecule is not independent.* Molecules from one paper
are a congeneric series measured in one assay by one laboratory: they share chemistry, label noise
and protocol, so a bootstrap over molecules treats a correlated block as n independent draws and
reports an interval that is too narrow. Every interval in §5.14 and §5.15 is therefore a **cluster
bootstrap over publications** — publications are resampled with replacement and all of a sampled
publication's molecules travel together — which is why those intervals are wide and why several of
them cross zero where a molecule bootstrap would not have.

*Comparisons across arms are paired only where pairing is legitimate.* A publication is the same
molecule set in both arms of §5.14, so those comparisons are paired across publications (37 on
CDK2, 97 on MAO-B). The split-penalty arms of §5.11 and §5.14 are **not** paired: the scaffold arm
has 10–20 seeds and the publication arm has 10–11 folds, with no correspondence between a seed and
a fold, so those use Welch's unpaired t-test. §5.13's coverage comparison is paired per fold *and*
model, because both regimes score the identical test molecules in the identical fold.

*Multiple comparisons.* Where a family of models is compared within one experiment we report
Holm-corrected p-values alongside the raw ones and defer to the corrected value: this is what
leaves §5.10's best entrant significant (Holm p = 0.045), §5.11's ranking inversion inconclusive
(every Holm p = 1.000), and no MAO-B per-model penalty significant in §5.14. The three-target
pooled estimates in §5.14 use inverse-variance fixed-effect pooling with Cochran's Q reported, and
we do not pool across targets where Q indicates heterogeneity.

Chemical novelty is the maximum Tanimoto similarity to any training molecule; effective sample size
on novel subsets is assessed by Butina clustering [8].

### 4.6 Determinism

After §5.4, all language-model training enables `torch.use_deterministic_algorithms(True,
warn_only=False)`, a fixed cuBLAS workspace (`:4096:8`), cuDNN determinism, disabled TF32 and seeded
Python/NumPy/Torch RNGs. Strict mode never raised, meaning every operation in this model's forward
and backward pass has a deterministic kernel available and nothing required exemption.

---

## 5. Experimental results

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

**Selectivity cannot be predicted.** This is where the publication-held-out collapse first
appeared, and where we first (and wrongly) named it publication-level confounding; §5.14 corrects
the attribution without disturbing the measurement below.

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
the publication-held-out decline would largely *not* reproduce. **The prediction was wrong**, and it is
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
| Publication-held-out decline | 0.974 → 0.574 | 0.898 → 0.830 | Reproduces; **attribution corrected in §5.14** |
| Nondeterminism | −0.028 ± 0.089 | not retested (no GPU training) | Untested |
| Chance below 0.4 Tanimoto | yes | **no — AUROC 0.739** | **Does not transfer** |

The pooling artefact reproduces but mildly, and the comparison identifies its severity condition:
on CDK2 every model is refitted per fold, so molecules recur only 1.6× and the interval narrows by
18%. TG2's severe case required a model that was *not* refitted — a cached zero-shot prediction
counted in every fold it appeared in. The artefact is general; it is dangerous specifically when a
fixed model is compared against a refitted one, which is the comparison every "LLM versus classical
baseline" study performs.

**A third target tests whether the ordering is stable.** MAO-B (CHEMBL2039) is larger again
(5,840 molecules, 414 publications, 36% positive) and a different enzyme class once more, a
flavin-containing oxidase. Its pattern matches: random → scaffold −0.059, scaffold → paper a
further −0.076 (RF AUROC 0.923 → 0.865 → 0.789, no interval overlapping the next), with **122.2 of
414 publications straddling a scaffold split on average**, and 8.9% of its random-split test
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
asks for: Chemprop v2 [39] — the directed message-passing architecture of Yang et al. [37], as
packaged by Heid et al. [38] and rewritten for v2 — Chemprop with RDKit 2D descriptors, CheMeleon
(descriptor-pretrained D-MPNN) [47], ChemBERTa-2 frozen and fine-tuned [48, 49], MoLFormer-XL
frozen [50], and a tabular foundation model [51],
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

**This is the expected result, not a surprising one, which bounds what it contributes.** Two
independent benchmarks report the same ordering on far more data than we have: Praski, Adamczyk and
Czech find that pretrained molecular embedding models do not generally beat classical fingerprint
baselines on molecular representation benchmarks [52], and Deng et al. attribute much of the
apparent progress in molecular property prediction to elements other than the learned
representation itself [53]. Our contribution here is therefore not the finding that fingerprints
win — that is established — but that the margin survives on this target under scaffold- and
publication-grouped evaluation with a reproduction gate, and that the ranking inverts
inconclusively under publication held-out splitting (§5.11). Readers should treat §5.10 as a
local replication of a known result, not as evidence against learned representations in general.

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

**This replicates on both independent targets** (added 2026-11-05; 10 grouped publication folds
each, six entrants, 2,078 and 5,091 molecules). Coverage pooled over molecules:

| | TG2 (LOPO) | CDK2 (grouped) | MAO-B (grouped) |
|---|---|---|---|
| i.i.d. calibration, target 0.90 | 0.784 | **0.811** | **0.819** |
| publication-wise, target 0.90 | 0.914 | **0.896** | **0.910** |
| i.i.d. calibration, target 0.80 | 0.601 | **0.571** | **0.679** |
| publication-wise, target 0.80 | 0.810 | **0.791** | **0.814** |
| paired gain at target 0.90 | +0.113 [+0.078, +0.148] | +0.084 [+0.065, +0.102] | +0.090 [+0.079, +0.102] |
| t-test p | 2.5 × 10⁻⁸ | 1.6 × 10⁻¹² | 1.6 × 10⁻²² |
| sign test | 52/66 | 55/60 | 60/60 |

Three targets, two publication protocols, same direction and similar magnitude. The i.i.d. regime's
failure at α = 0.2 is not a mild under-coverage but a collapse: it returns **empty** prediction sets
for up to 44.7% of CDK2 molecules and 13.5% of MAO-B molecules, which is how a nominal 0.80 becomes
an observed 0.446 for the CDK2 random forest. Publication-wise calibration emits no empty set on
any target at any α tested.

**The recipe that works under scaffold shift under-covers under publication shift**, by 0.11–0.14,
and issues empty prediction sets — the method announcing that a molecule is unlike its calibration
data — for up to 10% of molecules. Calibrating on whole held-out publications brings measured
coverage back to target at every α tested and emits no empty sets: **+0.113 coverage [+0.078,
+0.148] paired over 66
fold-and-model pairs, t-test p = 2.5 × 10⁻⁸, sign test p = 2.8 × 10⁻⁶**. The three fingerprint
models fail i.i.d. calibration worst (0.691–0.806) and the two ChemBERTa-2 variants are nearly
unaffected (0.874, 0.894), the same ordering as the AUROC decline in §5.11; we note the agreement
and claim no mechanism. Mondrian quantiles give the same coverage with 30% more decisive answers on
TG2 (singleton fraction 0.267 → 0.348); **that advantage does not replicate** — on CDK2 the gain is
0.438 → 0.458 and on MAO-B it is 0.523 → 0.499, i.e. nothing and slightly negative. Mondrian is
worth having for the class-imbalance protection it is designed for [6], but we withdraw "the
variant to use here" as a recommendation.

**This is an empirical result and not a guarantee, which matters more than it may appear.**
Split conformal prediction attains its 1 − α coverage when the calibration and test points are
exchangeable [6]. Our publication-wise regime does **not** restore that condition, and nothing in
the construction entitles it to the guarantee:

- A calibration molecule and a test molecule come from different publications, and molecules within
  a publication are strongly dependent — one congeneric series, one assay, one laboratory. The
  exchangeability that fails for the i.i.d. regime fails for this one too.
- A formal guarantee for a prediction on a molecule from a **previously unseen group** needs
  *hierarchical* exchangeability — publications exchangeable with each other, molecules exchangeable
  within a publication — together with an estimator built for it, such as the hierarchical conformal
  methods of Dunn, Wasserman and Ramdas [21] and Lee, Barber and Willett [22]. Those use subsampling
  or a quantile-of-quantiles; we run plain split conformal with a grouped calibration set, which is
  not the same procedure and does not inherit their theorem.
- Group-level exchangeability is itself violated **by our own design**. `group_calibration` fills
  the calibration set with whole publications *smallest-first*, to keep the fitting set large. The
  consequence is measurable: calibration publications average 11.9 molecules on both replication
  targets, against a mean test-publication size of roughly 26 on CDK2, and on TG2 19.6 against a
  38-molecule median test publication. The calibration groups are systematically about half the
  size of the group being predicted.

What publication-wise calibration actually does is make the calibration distribution *resemble* the
test distribution — calibration molecules are also scored by a model that never saw their
publication — which shrinks the coverage gap in the sense of Barber et al. [23] without closing it.
We therefore report coverage as **measured on three targets** and claim no distribution-free
guarantee under publication shift. The honest summary is that the fix is empirical, reproducible,
and theoretically unjustified as stated.

**§5.16 settles what that costs.** We have since run the published method that does carry a theorem
for a test point from an unseen group, on these same saved predictions. It holds its nominal floor
in all 54 runs where ours holds it in 22, and the gap between the two is governed by the number of
calibration publications exactly as the theory predicts. Our method is a good approximation when
that number is large — which is the regime this section was run in, and why its numbers look
correct — and has no safety margin when it is small.

**Marginal coverage and a usable answer are different things, and only the first one works.**
A prediction set containing both labels is covered and says nothing; the quantity a screening
campaign cares about is the fraction of molecules given a *singleton* set and the error rate among
those. At α = 0.1 under publication-wise calibration, across six entrants:

| | TG2 | CDK2 | MAO-B |
|---|---|---|---|
| marginal coverage (target 0.90) | 0.914 | 0.896 | 0.910 |
| **singleton fraction** | 0.15–0.32 | 0.41–0.48 | 0.46–0.56 |
| **error among singletons** | 0.25–0.39 | 0.22–0.27 | 0.16–0.18 |

The replication targets are less bad than TG2 on both axes — unsurprisingly, since they are 5× and
11× larger — but the shape is identical everywhere: **roughly half the molecules get a decisive
answer at best, and between a sixth and a quarter of those answers are wrong.** Nominal 90%
coverage is purchased by returning both labels on the remainder, which concentrates all of the
error in the decisive fraction. A method that is honest about its own ignorance is not thereby
useful, and reporting coverage alone would have hidden this on all three targets.

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

### 5.14 The publication-held-out decline is mostly novelty and aggregation, not publication identity — *correction to §5.9*

Added 2026-11-05, after extending the model field to six entrants on both replication targets
(`docs/replication_feasibility.md`, `scripts/analyse_protocol_gap.py`,
`scripts/validate_novelty_share.py`). This section corrects the *attribution* of the decline
reported in §5.9 and in the abstract. The decline itself is not withdrawn: it reproduces in
direction on every entrant of all three targets.

**What reproduces.** With matched molecules and matched training-set size, every one of six models
loses AUROC under publication hold-out on both replication targets. The magnitude shrinks sharply
with dataset size: mean penalty −0.198 (TG2, 428 molecules), −0.096 (CDK2, 2,078), −0.042 (MAO-B,
5,091). On MAO-B no per-model penalty survives Holm correction.

**Three reasons the decline cannot be attributed to publication identity.**

*Aggregation.* Leave-one-publication-out macro-averages one AUROC per publication; a scaffold arm's
pooled score averages over molecules. These answer different questions, and the difference is
large. On CDK2 the random forest scores 0.617 under leave-one-out macro and 0.786 under grouped
K-fold macro, yet **pooled molecule-level AUROC is the same under both protocols** (0.806 against
0.802). Re-aggregating the grouped arm's own predictions by publication closes almost all of the
gap: 0.604, 0.633 and 0.628 for the random forest, logistic regression and XGBoost, against
leave-one-out's 0.617, 0.629 and 0.629. The residual disagreement is at most 0.013 AUROC, against
a protocol gap of 0.169 before re-aggregation — so the aggregation unit accounts for roughly 92% of
it and the two protocols barely disagree about the models at all. Ranking compounds inside one
paper's congeneric series is simply a harder task
than ranking across a whole screening set, because the easy between-series contrasts are gone.
TG2's headline 0.854 → 0.612 compares a molecule-pooled scaffold score against a
publication-macro leave-one-out score, so part of that gap is the change of question.

**Which claims in this paper this does and does not touch.** The aggregation confound applies only
where a pooled score is compared against a publication-macro one. §5.9's CDK2 and MAO-B numbers
(0.898 → 0.830 and 0.865 → 0.789) are **not** affected: `scripts/run_second_target.py` builds its
paper arm with `chem.group_split` at 80/10/10, so its test set holds many whole publications, and
`evaluate` reports a pooled one-row-per-molecule AUROC for *both* arms. Those comparisons are
aggregation-matched and stand as reported. Verified directly: on the matched molecule set the
pooled score shows no protocol gap at all (CDK2 −0.004 to −0.016, MAO-B −0.025 on the 1,116
molecules both protocols tested), while the macro score differs by +0.13 to +0.18. The novelty
qualification below applies to all of these numbers; the aggregation one applies only to TG2's
leave-one-out arm.

*Novelty.* A publication split also shifts the test set towards novel chemistry, because a paper's
whole series leaves at once: 54% of CDK2's publication-arm test molecules sit below 0.4
max-Tanimoto to training, against 5% of its scaffold-arm molecules. Standardising to either arm's
novelty composition removes the majority of the penalty. Under the five published bands, the share
runs **60–80% across the six CDK2 models and 48–71% across the six MAO-B models**; widening to five
different band definitions gives 47–98% (CDK2) and 36–81% (MAO-B), so the conclusion "a majority"
does not depend on the banding. A publication-clustered bootstrap gives, for the
random forest, 78% [33%, 98%] on CDK2 and 60% [28%, 98%] on MAO-B. **The residual crosses zero for
every model on both targets** (CDK2 random forest −0.022 [−0.091, +0.038]). This is an
*association*, not a causal decomposition: see the limitations below.

*Assay.* Whatever residual remains cannot be assigned to publication identity, because
`scripts/audit_assay_confound.py` establishes that **0 of 380 CDK2 and 0 of 516 MAO-B assays span
more than one publication**. ChEMBL defines an assay per paper, so an assay-held-out split *is* a
publication-held-out split. The within-publication contrast that could estimate an assay effect
with the laboratory fixed covers 1 CDK2 publication and 2 MAO-B publications. The residual is
therefore "publication identity **and** its measurement protocol", jointly and inseparably.

**The matched comparison, and where the targets disagree.** Three things have to be held fixed for
this comparison to mean anything, and the third is easy to miss. The aggregation unit is matched
(publication-level macro in both arms). The publication set is matched (paired across the
publications both arms can score: 37 on CDK2, 97 on MAO-B). And **the molecule sets must be
matched too**: a publication arm tests a paper's whole series, whereas the scaffold arm tests only
the molecules whose scaffold group happened to land on the test side, so the unrestricted version
scores 1,052 CDK2 molecules in one arm against 810 in the other. Restricting both arms to the
molecules they share, with a bootstrap clustered on publications rather than molecules:

| | CDK2 (37 publications) | MAO-B (97 publications) |
|---|---|---|
| penalty range over six models | **−0.034 to −0.110** | **+0.007 to +0.062** |
| intervals excluding zero | 2 of 6 (rf, XGBoost) | 3 of 6 (XGBoost, logreg, ChemBERTa-2 ft) |
| largest single effect | XGBoost −0.110 [−0.203, −0.023] | XGBoost +0.062 [+0.019, +0.104] |

**The sign differs between targets, and XGBoost is significant in both directions.** TG2's 10
publications give intervals spanning ±0.2 and settle nothing. So "holding out a publication reduces
performance" is not supported as a general claim once the aggregation unit is matched. §5.15
discusses what the disagreement might be and why we do not resolve it.

*Correction (2026-11-05).* An earlier version of this paragraph reported −0.043 to −0.093 and
+0.009 to +0.060 and described them as matched. Those are the **unrestricted** rows of the same
result table, in which the two arms score different molecule sets; the molecule-matched rows are
the ones above. The qualitative conclusion — small penalties of opposite sign — is unchanged, and
the largest effects are slightly larger under proper matching, not smaller.

**Limitations of the standardisation.** Three, and they bound what the novelty number means.
(i) It is an adjustment for measured similarity, not an identified causal effect; molecules are not
randomised to arms and novelty may proxy for other differences. (ii) Band standardisation removes
only *between*-band novelty, so it is a lower bound on the novelty association; a continuous
adjustment gives 13–80% depending on model and target, with no tighter conclusion. (iii) It
requires common support, and TG2 does not have it: its arms are nearly disjoint in novelty (7
scaffold-arm molecules below 0.4 Tanimoto against 283 in the publication arm), the smallest shared
band holds 7 molecules against 63 for CDK2 and 207 for MAO-B, and the resulting estimates range
from −992% to +51% with bootstrap intervals spanning several hundred percent. **The decomposition
is not estimable on TG2.** Note the direction, though: TG2 has the *largest* novelty shift of the
three (mean −0.33 against −0.24 and −0.14), so the evidence is consistent with novelty driving its
decline too — it simply cannot be quantified there, and we claim neither way.

**What §5.9 and the abstract should have said.** Not "publication-level confounding reproduces
most strongly", but: *publication-held-out evaluation consistently reports lower performance than
scaffold evaluation, and most of that difference is attributable to two measurable things — the
unit the metric is averaged over, and the novelty of the molecules a publication split puts in the
test set — with any remainder confounded between publication identity and assay protocol.* The
practical recommendation is unchanged and arguably strengthened: a publication-held-out split is
worth running because it generates a realistic novelty distribution automatically, which a
scaffold split does not.

---

### 5.15 Why CDK2 and MAO-B disagree, and why we do not resolve it — *not established*

§5.14's matched comparison leaves the two replication targets with penalties of opposite sign, and
XGBoost significant in both directions. A single universal "publication penalty" is therefore not
available, and the honest position is to characterise the disagreement rather than to average it
away. Three observations constrain it; none settles it.

**It is not an artefact of the matched comparison.** The conformal arm reaches the same split by a
route that shares none of its machinery — no scaffold arm, no publication pairing, no AUROC.
Selective prediction, ranking molecules by |p − 0.5| and keeping the most confident fraction
(random forest, publication-wise calibration):

| fraction kept | CDK2 decision error | MAO-B decision error |
|---|---|---|
| 100% | 0.309 | 0.279 |
| 60% | 0.256 | 0.187 |
| 20% | 0.215 | **0.077** |

On MAO-B, confidence ranking works: discarding the least confident 80% of molecules cuts decision
error 3.6-fold. On CDK2 it is close to uninformative (1.4-fold over the same range). Two independent
measurements agreeing on which target is harder is weak evidence that the disagreement is real and
no evidence at all about its cause.

**Dataset size is the obvious candidate and does not fit cleanly.** The unmatched penalty does fall
monotonically with size (−0.198 at 428 molecules, −0.096 at 2,078, −0.042 at 5,091), which would
predict MAO-B's penalty to be small — but it predicts *small*, not *positive*. A size explanation
accounts for the magnitude ordering and not for the sign change.

**Three structural differences are candidates we cannot separate with two targets.** CDK2 is a
kinase with a deep, well-conserved ATP pocket and a literature dominated by tight congeneric series;
MAO-B is a flavoenzyme with a smaller, more rigid cavity and a more fragmented literature (209
qualifying publications against 83, and 97 against 37 scoreable in both arms). Publication *size*
is not one of the differences: the median leave-one-out fold holds 19 molecules on both targets.
Any of these could plausibly make a held-out MAO-B publication
*easier* than a held-out scaffold group — for instance if MAO-B's scaffold arm is the harder task
because its scaffold groups cut across series that a publication split keeps intact. We did not
design an experiment to discriminate these, and with n = 2 targets we could not fit one.

**Status: not established.** We report the disagreement as a constraint on the claim, not as a
finding with a mechanism. It is the single strongest argument in this paper against treating any of
these penalties as a transferable constant, and the main reason §10 recommends reporting the split
rather than a correction factor derived from it. Resolving it needs more targets, which is the one
additional experiment §9 identifies as scientifically load-bearing.

---

### 5.16 Hierarchical conformal prediction holds its floor where our method does not — *demonstrated*

Added 2026-11-05 (`scripts/run_hcp.py`, `scripts/analyse_hcp.py`, `tests/test_hcp.py`). §5.13
reported an empirical fix with no theorem behind it. This section runs the published method that
does have one, on the same saved predictions, with no model refitted.

**The algorithm.** Split hierarchical conformal prediction (HCP), Theorem 1 of Lee, Barber and
Willett [22], which builds on the three constructions of Dunn, Wasserman and Ramdas [21]. Given
K₁ calibration groups with sizes N_k, the threshold T is the (1 − α)-quantile of the distribution
placing mass 1/((K₁+1)·N_k) on each calibration score and 1/(K₁+1) on +∞, and the set is
{y : s(x, y) ≤ T}. The guarantee is P{Y ∈ Ĉ(X)} ≥ 1 − α, with an upper bound of
1 − α + 2/(K₁+1) when scores are a.s. distinct.

**What it is a guarantee *about*, which is the substantive point.** The probability is marginal over
the draw of the calibration groups *and* of a fresh test group with an observation inside it. In
our terms the event is "a randomly chosen **publication**, then a randomly chosen molecule from
it". Ordinary split conformal — and therefore §5.13's method — targets "a randomly chosen
**molecule**", under which a 224-molecule paper counts 22 times a 10-molecule one. The two targets
coincide only if every publication is the same size, and ours run 10 to 224. Each publication
carrying total mass 1/(K₁+1) regardless of size is exactly how HCP tolerates that; with all
N_k = 1 it reduces to standard split conformal.

**Do our data satisfy the assumptions?** Hierarchical exchangeability asks that groups be
exchangeable with one another and observations exchangeable within a group given its size.
Within-publication exchangeability is reasonable and, importantly, does not require independence —
a congeneric series in one assay is the dependent-within-group case the theorem is built for.
Across publications it is *not* true of the corpus: publications are ordered in time and later
series build on earlier ones. Nor does it come from the fold assignment, which we checked during
the final audit: `deal_papers` deals publications largest-first to the lightest fold to equalise
fold sizes, so a fold's publications are a size-balanced selection rather than a random one.

**The exchangeability we rely on is one we impose, not one we hope for.** Theorem 1 needs only the
K₁ calibration groups and the test group to be exchangeable, and within a fold we partition its
publications *uniformly at random* into calibration and test. Conditional on the fold's publication
set, that randomisation makes them exchangeable by construction, whatever the corpus looks like.
Two consequences follow. The statement is conditional on that set, and since `deal_papers` balances
fold molecule counts the size mix inside a fold is not the corpus mix — so these figures are not a
claim about a publication drawn at random from the literature. And the smallest-first and
largest-first runs below deliberately remove the randomisation, so the theorem does not cover them;
they are a sensitivity analysis, not a second estimate. We therefore run two designs and are
explicit about which carries the theorem:

| design | construction | theorem applies? |
|---|---|---|
| **within-fold** | calibration and test publications are both drawn from one fold's held-out block, all unseen by that fold's model | **yes** — Theorem 1 as stated |
| cross-fold | calibration publications come from other folds, scored by their own fold's model | **no** — cross-conformal, and the scoring model for a calibration publication was fitted on data including the test publication |

The within-fold design needs K₁ ≥ ⌈1/α⌉ − 1 calibration publications or the quantile lands on the
infinite atom. That makes it **feasible on MAO-B at both α (17–22 publications per fold), feasible
on CDK2 only at α = 0.2 (6–9 per fold), and infeasible on TG2 at any α** — leave-one-publication-out
leaves exactly one unseen publication per fold, so TG2 can only be run cross-fold. We report this
rather than work around it: the protocol that produced our headline result cannot support the
method that would justify it.

**Implementation verified against the theorem before any result was read.** `tests/test_hcp.py`
(15 tests) builds synthetic worlds where hierarchical exchangeability holds by construction — group
means drawn i.i.d., members drawn i.i.d. within a group — and confirms over 3,000 trials that
coverage sits at or above the floor, including with group sizes 2 to 200. It also confirms that
the molecule-weighted quantile *under-covers* in the same world when one group dominates, that HCP
reduces to split conformal at N_k = 1, and that too few groups yield ∞ rather than a silently
invalid threshold. One real bug surfaced this way: at the smallest feasible K₁ the finite mass
equals 1 − α exactly, and floating-point summation fell a few ulps short, turning the smallest
usable calibration set into a trivial predictor.

#### The comparison

Random forest, random selection of calibration publications, publication-clustered intervals.

| target | design | K₁ | α | i.i.d.-pooled | publication-grouped (ours) | HCP |
|---|---|---|---|---|---|---|
| MAO-B | within | 13 | 0.10 | 0.869 | **0.841** [0.788, 0.897] | **0.948** [0.912, 0.982] |
| MAO-B | within | 14 | 0.20 | 0.771 | **0.765** [0.698, 0.821] | **0.845** [0.799, 0.888] |
| CDK2 | within | 6 | 0.20 | 0.800 | **0.769** [0.661, 0.867] | **0.908** [0.827, 0.955] |
| MAO-B | cross | 187 | 0.10 | 0.900 | 0.901 [0.876, 0.923] | 0.911 [0.888, 0.933] |
| CDK2 | cross | 74 | 0.10 | 0.902 | 0.899 [0.860, 0.938] | 0.914 [0.873, 0.947] |
| TG2 | cross | 10 | 0.10 | 0.910 | 0.890 [0.778, 0.960] | 0.982 [0.935, 0.998] |

**HCP holds its nominal floor in 54 of 54 runs across both designs, all six entrants and both α.
Our publication-grouped method holds it in 22 of 54**, and in the theorem-faithful within-fold
design it holds in only 5 of 18, with 2 runs whose entire interval lies below target. Paired on
publication with a cluster bootstrap, HCP − ours is positive in **all 54** model-by-target-by-α
comparisons, and the interval excludes zero in 52 of them; the two exceptions are XGBoost on CDK2
cross-fold at α = 0.2 (+0.002 [−0.001, +0.006]) and fine-tuned ChemBERTa-2 on TG2 cross-fold at
α = 0.1 (+0.134 [+0.000, +0.301]). The six entrants share folds and molecules, so these are 54
valid per-model statements, not 54 independent observations.

**The size of the advantage is governed by K₁, exactly as the theorem says it should be.** Sweeping
the calibration budget on MAO-B:

| K₁ | ours | HCP | HCP excess over 0.90 | 2/(K₁+1) |
|---|---|---|---|---|
| 9 | 0.852 | 0.995 | +0.095 | 0.200 |
| 13 | 0.871 | 0.966 | +0.066 | 0.143 |
| 25 | 0.892 | 0.940 | +0.040 | 0.077 |
| 50 | 0.895 | 0.930 | +0.030 | 0.039 |
| 100 | 0.899 | 0.916 | +0.016 | 0.020 |
| 187 | 0.901 | 0.911 | +0.011 | 0.011 |

HCP's conservatism decays at the rate its own upper bound predicts and converges onto it; in the
within-fold design, where the theorem applies, no run's interval lies above the bound. Our method
climbs from 0.852 to 0.901 over the same sweep: **it reaches nominal coverage only when the
calibration set spans many publications, and has no safety margin when it does not.** That both
explains and bounds §5.13 — its calibration sets were large, which is why its coverage looked
correct, and nothing about the construction guaranteed it would.

**What HCP costs.** Decisiveness, and a lot of it where K₁ is small. At α = 0.1 on MAO-B within-fold
the singleton fraction falls from 0.615 (ours) to 0.308 (HCP), i.e. **HCP halves the molecules given
a definite answer** to buy its floor. Its singletons are more reliable in exchange (error 0.258 →
0.169). At K₁ = 187 the trade almost vanishes (0.560 → 0.529). On TG2 cross-fold, where K₁ = 10, the
singleton fraction collapses to 0.077 — HCP commits on one molecule in thirteen. So the guarantee is
affordable on a large, publication-rich dataset and close to useless on a small one.

**Conditional coverage is still not fixed, by either method.** In the MAO-B within-fold design
12.8% of publications fall below target under HCP against 43.6% under ours — much better, and still
not a per-publication guarantee, which neither method claims. The worst publication covers 0.474
under HCP and 0.133 under ours; on MAO-B cross-fold the worst covers 0.000 under *both*. Across
novelty bands HCP is uniformly higher but the shape is unchanged. Marginal coverage over
publications is what Theorem 1 delivers, and per-publication reliability remains out of reach.

**Sensitivity to how calibration publications are chosen.** Under the three selection rules
(random, smallest-first as in our production code, largest-first) HCP's coverage spans 0.948–0.966
and never drops below target, while ours spans 0.841–0.918 and falls below it under random
selection. Changing the rule also changes which publications are *test*, so these rows do not share
an evaluation set and only each method's spread is comparable; the finding is that HCP is
insensitive to the choice and ours is not. Notably our production smallest-first bias (§8) reads
*better* here (0.895) than random selection (0.841), because taking the small publications for
calibration leaves the large ones as test and coverage is molecule-weighted — a second illustration
that the molecule-weighted target is the one sensitive to group sizes.

**One arm could not be reconstructed.** §5.13's i.i.d. regime calibrates on random molecules from
the *training* publications, which requires nonconformity scores for molecules the model was fitted
on; only out-of-fold predictions were saved. The "i.i.d.-pooled" column above is therefore a
different and milder thing — molecule-pooled calibration with a quarter of each test publication
mixed in — and it behaves accordingly, landing near our grouped method rather than failing as
§5.13's i.i.d. arm does. §5.13's measurement of that arm stands on its own run; this section
compares the two group-aware methods, which is what it was built to do.

**Status: demonstrated, with its scope stated.** HCP is the method with the guarantee and it
empirically honours it on every run we could construct. It does not solve distribution shift: it
widens prediction sets until the stated error rate is met, and under publication shift that means
refusing to commit on most molecules. Nothing here makes a prediction on an unseen publication
reliable enough to act on without experimental confirmation (§7.2).

### 5.17 The publication-shift penalty is not class-symmetric — *direction-consistent on three targets, not established on any one*

Every classification result above this point is threshold-free: AUROC, AUPRC, and MCC at a fixed
0.5 cut. None of them says whether a model that loses accuracy under publication shift is losing it
on the actives or on the inactives. For a screening cascade meant to nominate compounds for assay
that is the question that matters, because missing an active and raising a false alarm are not the
same failure and they have the same AUROC. This section fills that gap from predictions already
saved — `results/sota_preds*.csv` hold per-molecule labels and probabilities for every model, arm
and fold, so no model was retrained and no new data was used.

Three operating points are reported, all fixed before any output was read: the 0.5 cut used
elsewhere in the paper, the training fold's own prevalence, and the cut maximising balanced accuracy
on the test fold. **The third is an oracle and is not an achievable operating point.** It is
reported only as an upper bound, flagged in the data by an `achievable` column, and it is not quoted
as performance anywhere: on TG2 `paper_lopo` the random forest reaches 0.711 balanced accuracy at
the oracle cut against **0.506** at the 0.5 cut and **0.545** at the prevalence-matched cut, and the
oracle's apparent gain scales with how few molecules it was chosen on (+0.020 on 205-molecule folds,
+0.183 on 19-molecule folds), which is the signature of selection on noise rather than of a
recoverable threshold.

Aggregation follows §5.14. A leave-one-publication-out fold contains exactly one publication, so its
fold average already is a publication average; a `scaffold_pub` fold contains 4 to 113 publications.
Comparing the two arms on the fold therefore repeats precisely the unit error §5.14 identifies, and
all comparisons below use the **publication** as the unit, each publication averaged over the folds
it appears in. The fold-unit figures are retained in `results/per_class_*_summary.csv` and the
contrast between the two is reported rather than suppressed, because the conclusion depends on it.

On the matched arms, at the 0.5 cut, equal-weighting publications:

| target | sensitivity | specificity |
|---|---|---|
| TG2 (`scaffold_pub` → `paper_lopo`) | 8 of 10 models fall, mean **−0.163**, sign-test p = 0.109 | 3 of 10 fall, mean +0.054, p = 0.344 |
| CDK2 (`scaffold_pub` → `paper_grouped`) | **6 of 6 fall**, mean −0.101, p = 0.031 | 4 of 6 fall, mean −0.026, p = 0.688 |
| MAO-B (`scaffold_pub` → `paper_grouped`) | 3 of 6 fall, mean −0.001, p = 1.000 | **6 of 6 fall**, mean −0.035, p = 0.031 |

**That cross-target pattern is not robust to the choice of estimator, and the reason is instructive.**
Equal-weighting discards any publication that reports only one class, and in the *scaffold* arms
that is half to three-quarters of the units — not because the literature is one-sided, but because a
scaffold split cuts each publication across train and test and the test-side remnant is often all
one class. On TG2, 9% of publications are genuinely single-class while **50%** of publication-fold
units are; on CDK2 35% against 71%, on MAO-B 37% against 72%. In the publication-held-out arms,
where whole publications go to test, the two figures coincide exactly.

An estimator that keeps every unit — summing confusion matrices across publications and dividing
once — gives a third answer, reported with a cluster bootstrap in
`results/per_class_pooled_rates.csv`. For the TG2 random forest's sensitivity:

| estimator | reference arm | shifted arm | Δ |
|---|---|---|---|
| fold average | 0.823 | 0.320 | −0.503 |
| pooled confusion, every publication | 0.848 | 0.349 | **−0.499** |
| equal-weight publication | 0.660 | 0.320 | −0.340 |

Under pooling, TG2 sensitivity falls for 9 of 10 models (mean −0.284) while specificity falls for
only 2 of 10 (mean +0.068); CDK2 falls on both (−0.091, −0.074) and **MAO-B becomes symmetric**
(−0.041, −0.041) rather than specificity-leaning. Three estimators therefore agree on TG2 and
disagree on MAO-B. The only statement robust to the choice is the narrow one:

> **TG2 loses sensitivity and not specificity. CDK2 and MAO-B lose both.**

Neither estimator is preferable alone — equal-weighting is right in principle but throws away units
for an artefactual reason, pooling keeps them and lets the largest publications dominate, which is
the weighting §5.14 warns about. Both are reported, and a conclusion that depends on the choice is
treated as unresolved. The real fix is not an estimator but a protocol: do not split publications
across folds, so that the unit used for aggregation matches the unit used for splitting.

The largest single effect is the TG2 random forest, whose sensitivity falls from 0.660 to **0.320**
while its specificity *rises* from 0.559 to 0.761. Under publication shift it stops predicting
"potent" on a test set that is 54% positive. XGBoost behaves the same way, −0.294 sensitivity against
+0.231 specificity. Their AUROC drops — 0.854 to 0.612 and 0.855 to 0.573 — report the net of these
two movements and are silent about the direction.

**Not one comparison survives Holm correction on this unit** (best p_holm 0.939 on TG2, 0.216 on
CDK2, 0.421 on MAO-B). On the fold unit the TG2 sensitivity result had been significant for two
models; the correction removes that, because the 20 `scaffold_pub` seeds contain only 11 distinct
publications and variation between publications is much larger than between folds. Both units are
written to `results/per_class_shift_tests.csv` so the dependence is visible.

Two further limits travel with these numbers. A publication reporting only actives yields no
specificity and one reporting only inactives yields no sensitivity; those units are skipped and
counted, and they are **20% to 52%** of units depending on arm (MAO-B `scaffold_pub`: 52% give no
sensitivity). And the shifted arm is not the same protocol across targets — leave-one-publication-out
on TG2, grouped folds on CDK2 and MAO-B — so the cross-target comparison varies protocol as well as
target.

**Status: direction-consistent on TG2, not established on any target.** The claim this supports is
that the penalty is not class-symmetric on TG2, where sensitivity falls under all three estimators
and specificity does not under any of them, and that the replication targets lose both rates. It is
not a demonstrated effect at any conventional threshold — no comparison survives Holm under any
estimator, and under pooling 9 of 10 TG2 sensitivity comparisons have overlapping 95% intervals. A
null at 11 publications is a statement about power. §5.20 supplies a mechanism that makes the
cross-target variation unsurprising and removes the need for a target-specific explanation.

One corollary is firm regardless, and it bears on how the field reports this task. **Accuracy is
uninformative on these folds.** Chemprop on TG2 `paper_lopo` posts the highest accuracy of any
entrant in that arm, **0.704**, at AUROC **0.542** and balanced accuracy **0.515** — a model at
chance discrimination looking like the arm's best, because it predicts "potent" for 69% of a fold
that is 54% positive. The published TG2 study's headline is accuracy (§1.1); so is the 64% baseline
it improves on. Neither number should carry weight on data of this shape, and balanced accuracy with
the confusion matrix beside it should be quoted instead.

### 5.18 Activity cliffs are harder for every model, on every target — *demonstrated*

§5.6's applicability analysis bins molecules by distance to the training set. An activity cliff is a
different object: a pair of *mutually similar* molecules with divergent potency. A model can score
well in a similarity band and still invert every cliff inside it, and van Tilborg et al. [29] report
exactly that across 24 methods and 30 ChEMBL targets. That measurement had not been made on our
data.

The definition is theirs, with their published parameters rather than any of ours: a pair is a cliff
when it is similar by at least one of Morgan Tanimoto ≥ 0.9 (radius 2, 1024 bits), generic
Bemis–Murcko scaffold Tanimoto ≥ 0.9, or normalised SMILES Levenshtein similarity ≥ 0.9, **and** its
potency differs by more than tenfold. Scoring is pairwise and therefore threshold-free: for a
discordant pair whose two molecules fall in the same test fold, the model is right if it assigns the
potent one the higher probability. The design adds a control the original lacks — `distant` pairs,
with the same tenfold potency gap but *below* the similarity cut — because without it "cliffs are
hard" cannot be separated from "large potency gaps are hard".

Cliff accuracy is below distant accuracy in **58 of 58** model × arm comparisons across the three
targets, 16 of them significant after Holm within arm. On `scaffold_pub`:

| target | model | cliff | distant | Δ | p_holm |
|---|---|---|---|---|---|
| TG2 | Chemprop | **0.468** | 0.808 | −0.340 | 0.004 |
| TG2 | Random forest | 0.736 | 0.891 | −0.155 | 0.083 |
| CDK2 | Random forest | 0.763 | 0.901 | −0.138 | 0.019 |
| MAO-B | Logistic regression | 0.728 | 0.824 | −0.097 | 0.010 |

Chemprop on TG2 `scaffold_pub` ranks cliff pairs at 0.468, *below chance*: it does not merely weaken
on cliffs, it inverts them. Tests are paired per fold, since one molecule sits in many pairs and
pairs are not independent while each fold contributes one number. The distant pairs are the easy
ones, so the penalty is not an artefact of large potency gaps.

Where our data **does not** reproduce [29] is its framing that overall performance fails to predict
cliff performance. Correlating the two rankings across models, one Spearman per arm: TG2 `scaffold`
and `scaffold_pub` both **0.770** (p = 0.009), MAO-B `scaffold_pub` **0.943** (p = 0.005), against
0.657 (p = 0.156) on CDK2 `scaffold_pub` and 0.143 to 0.515 with no significance on the three shifted
arms. Within distribution, choosing a model by its headline score is not misleading; under
publication shift the correlation collapses. Each correlation is over 6 to 10 models, so this is a
qualification on [29] rather than a refutation, and the shifted arms have less power as well as
fewer entrants. Individual models still reorder even where the aggregate holds: on TG2 `scaffold`
CheMeleon is sixth of ten on distant pairs and first on cliffs.

An incidental observation about the definition deserves recording, because it changes what
"structurally similar" means in practice. Of the similar discordant pairs found, the generic-scaffold
test admits 1,840 on TG2 against 23 for the fingerprint test and 270 for Levenshtein; the pattern
holds on CDK2 (9,404 / 34 / 636) and MAO-B (72,054 / 214 / 2,097). The generic scaffold discards atom
and bond identity, so our cliff pairs are overwhelmingly *same-skeleton, different-substituent*
pairs, not near-identical-fingerprint pairs.

**That observation invites the obvious objection — that "cliff" here means little more than "shares
a skeleton" — so a labelled sensitivity analysis scores the same predictions under fingerprint-only
similarity.** All five definitions are retained side by side in `results/cliff_ranking_*.csv`; the
MoleculeACE rows above are unchanged.

| definition | TG2 | CDK2 | MAO-B | median cliff pairs per fold |
|---|---|---|---|---|
| `three_test` (MoleculeACE) | 30/30 worse, 2 Holm-significant | 15/15, 8 | 13/13, 6 | 15 / 36 / 634 |
| `morgan_only_0.9` | 9/10 worse, 1 | 12/15, 5 | 13/13, 2 | 1 / 2 / 3 |
| `morgan_only_0.8` | 26/30 worse, 2 | 15/15, 7 | 13/13, 6 | 3 / 3 / 9 |
| `morgan_only_0.7` | 25/30 worse, 2 | 15/15, **10** | 13/13, **12** | 5 / 6 / 54 |

The result is not an artefact of the loose test. Under fingerprint-only similarity the direction
holds in 34 of 38 comparisons at the strictest cut, unanimously on both replication targets at every
cut, and with *more* Holm-significant comparisons at the 0.7 and 0.8 cuts than under MoleculeACE's
own rule. Two qualifications: at 0.9 the pair counts fall to 1–3 per fold, so that row is a direction
check and no more; and TG2 weakens slightly under every strict definition, consistent with its having
only 23 fingerprint-similar discordant pairs in the entire inventory. No sweep over the tenfold
potency cut was run — the published value was fixed first.

**Status: demonstrated.** The direction is unanimous across 58 comparisons, three targets and four
splitting arms, with a matched-potency control and a definition taken from the source. The
qualification in the third paragraph is reported as exploratory.

**External context, and what it implies for whether this is fixable.** Our result is that cliffs are
harder, not that nothing helps. Cui et al. report that an activity-cliff-aware loss — a soft-margin
triplet term whose margin is set by the label gap itself, mined online per batch — improved all 30
targets under a Bemis–Murcko scaffold-held-out split and 28 of 30 under a split that removes test
molecules above 0.90 Tanimoto to training [42]. Those two supplementary protocols are the ones
comparable to our arms; their headline results use an easier split (80/20 within spectral ECFP
clusters) and should not be read as evidence in this regime. Set against that, Tamura et al. found
accuracy not scaling with model complexity on cliff *pair* classification [43] — but a loss function
on single-compound regression and architecture scaling on pair classification are different
interventions on different tasks, so the two are not in conflict and neither settles the question
for us. One finding in [42] bears directly on how we would adapt it: gating candidate triplets by
fingerprint, scaffold or SMILES thresholds discarded useful triplets and did worse than unrestricted
latent-space mining, which is a caution against the publication- or scaffold-gated variant we would
reach for first. Separately, López-Pérez and Miranda-Quintana's eSIM and eSALI give a
linear-time measure of activity-landscape roughness over a whole set [41], which is a cheaper
summary than our pairwise inventory; we have not used it, and we do not cite their splitting
comparison, whose selecting method also chooses the test set.

### 5.19 Validation-fold class balance does not explain per-seed variance — *not established*

Scaffold splitting keeps analog series intact, so a validation fold's class balance is whatever the
shuffle gives: on TG2 it ranges from **8.5%** positive (seed 0: 59 molecules, 5 of them active) to
73%, in a dataset that is 47.9% positive. Every entrant selects its configuration on that fold, and
§5.10 records that seeds 10–19 are harder for every model and that Chemprop's spread (± 0.123) is far
wider than the random forest's (± 0.077). The hypothesis was that selecting on a near-single-class
fold degrades the choice, and that the damage should concentrate in models where the validation fold
also decides *how much* training happens.

It was pre-registered before fitting, including the control group, the predicted sign and the rule
that p > 0.05 is a null rather than a trend. One design assumption had to be corrected first: the
fingerprint models do **not** ignore the validation fold — `fit_sklearn` tunes the random forest,
logistic regression and XGBoost on it over 12, 4 and 8 configurations — so no model here ignores it,
and the control group was redefined by what the fold decides. Early-stopping group: Chemprop,
Chemprop + RDKit 2D, CheMeleon, ChemBERTa-2 fine-tuned. Control: the other six.

**The hypothesis is not supported.** Regressing per-seed test AUROC on |validation positive fraction
− 0.5|, every confidence interval spans zero on both TG2 arms and the largest r² is **0.051**. The
point estimates mostly carry the *opposite* sign to the prediction, and the group contrast is absent:
mean slope **+0.120** for the control against **+0.046** for the early-stopping group. MAO-B produces
three slopes at p < 0.05, all positive and spread across both groups, on an arm whose validation
prevalence never crosses 0.5 — so there |pos − 0.5| is a relabelling of prevalence — and where
validation imbalance correlates with training-set size at r = 0.583. On CDK2 it correlates with test
positive fraction at r = −0.647 (p = 0.043). Under whole-group splitting the folds are not
independent of one another, so a slope on one fold's composition is not evidence about that fold.

The observation that prompted the experiment has a simpler explanation. Correlating each model's mean
AUROC with its SD across seeds gives r = **−0.912** (p = 0.0002) on TG2 `scaffold`, −0.841 (p = 0.002)
on TG2 `scaffold_pub` and −0.888 (p = 0.018) on MAO-B `scaffold_pub`: weaker models vary more,
whatever they select on. The group-level SD gap is driven as much by frozen MoLFormer, a control
model, as by Chemprop. This explanation was found after looking at these data and is a hypothesis for
a future test, not a result, and it does not reach significance on CDK2.

**Status: not established, and withdrawn as a mechanism.** Validation-fold class balance is not added
to §6's catalogue. The result is kept because it removes a live confound from §5.10 — the
ten-entrant ranking cannot be dismissed on the grounds that some models were tuned on degenerate
validation folds — and because a null obtained against a written prediction is worth more than one
never tested. Stratifying validation folds by class costs nothing and would remove the question
entirely; it is hygiene rather than a fix, it is not justified by any measured benefit here, and it
has not been applied retroactively because doing so would change committed numbers.

### 5.20 Errors concentrate by publication, and the models predict the local class prior — *concentration demonstrated, mechanism direction-consistent*

§5.17 reports *that* the class asymmetry exists without saying *why*, and its cross-target
variation needed a target-specific story. This section locates the errors and offers one mechanism
for all of it. Everything runs on the committed predictions at the 0.5 cut.

**Difficulty is a property of the molecule, not of the model.** Counting how many of the
(model × seed) scorings misclassify each molecule, against a null in which each model errs
independently at its observed marginal rate: 8.2% of TG2 `scaffold_pub` molecules are misclassified
by at least nine of ten entrants where the null predicts 0.0%, and 51.2% are misclassified by none
where the null predicts 8.9%. The pattern repeats on CDK2 (11.7% against 0.1%) and MAO-B (10.7%
against 0.2%). About one molecule in ten defeats essentially every architecture, which is part of why
§5.10's ten-entrant ranking is so tightly bunched.

**Errors concentrate in particular publications, and this is the clearest result in the section.**
Against a permutation null that reassigns molecules to publications at random while holding
publication sizes fixed, the observed between-publication spread in error rate is two to three times
the null mean on every target and arm, at **p < 0.001** throughout: TG2 0.112 against a null mean of
0.048, CDK2 `paper_grouped` 0.224 against 0.068, MAO-B `paper_grouped` 0.208 against 0.069. The worst
fifth of publications carries 44% to 60% of all error mass while holding 24% to 35% of the molecules.
Assay type cannot be separated from publication here — §5.14 records that no assay in either
replication dataset spans two publications (0 of 380, 0 of 516) — so a per-assay analysis would be
this one relabelled, and none was run.

**The mechanism: a publication's own class balance predicts which error it attracts.** Rank
correlation between a publication's active fraction and each of its error rates:

| target | arm | FN rate against active fraction | FP rate against active fraction |
|---|---|---|---|
| TG2 | `scaffold_pub` | −0.633 (p = 0.067, 9 publications) | **+0.733** (p = 0.025) |
| CDK2 | `scaffold_pub` | **−0.657** (p < 0.0001, 46) | **+0.757** (p < 0.0001, 42) |
| MAO-B | `scaffold_pub` | **−0.607** (p < 0.0001, 74) | **+0.702** (p < 0.0001, 133) |
| CDK2 | `paper_grouped` | **−0.524** (p < 0.0001, 64) | +0.374 (p = 0.010, 47) |
| MAO-B | `paper_grouped` | **−0.358** (p = 0.0001, 120) | **+0.514** (p < 0.0001, 169) |

Every false-negative correlation is negative and every false-positive correlation positive, across
three targets and two protocols. Inactive-dominated publications have their few actives missed — on
TG2, CHEMBL1770135 (8 actives, 43 inactives) has a false-negative rate of 0.981 against a
false-positive rate of 0.060 — while active-dominated ones have their few inactives called active:
CHEMBL3886499 (22 actives, 13 inactives) reverses it at 0.156 and 0.691.

**The same mechanism accounts for §5.17's sensitivity collapse.** Error rate by maximum Tanimoto to
the training fold, split by type: on MAO-B under publication shift, 87.9% of actives below 0.4
Tanimoto are missed while 5.5% of inactives are false alarms; on TG2 the figures are 0.593 and 0.190,
on CDK2 0.465 and 0.272. Above 0.7 Tanimoto the asymmetry reverses — TG2 0.283 false negative against
0.943 false positive. **Unfamiliar molecules are called inactive and familiar ones active**, which is
nearest-neighbour behaviour and makes the cross-target variation in §5.17 unsurprising: the direction
of the asymmetry follows how far the shifted arm's molecules land from training, not anything about
the protein.

**One further contrast bears on the task definition.** Error rate against |pIC50 − 6|, the distance
to the 1 µM cut, is monotone within distribution on all three targets — TG2 0.395 within 0.25 log
falling to 0.157 beyond 1 log, CDK2 0.376 to 0.131, MAO-B 0.456 to 0.176 — so a molecule within a
factor of about 1.8 of the threshold is two to three times more likely to be misclassified. Those
errors are largely a consequence of binarising a continuous measurement, and they are the
molecule-level counterpart of §5.18's near-threshold pairs. **Under publication shift on TG2 the
relationship flattens entirely** (0.421 / 0.375 / 0.357 / 0.367): within-distribution error sits at
the boundary where it is partly unavoidable, shifted error is spread across the whole potency range
where it is not. A single threshold-free number cannot distinguish the two.

**Status: the concentration is demonstrated; the mechanism is direction-consistent.** The
permutation test on publication identity is significant at p < 0.001 on every target and arm, and
the concordance result holds far outside its null everywhere. The class-prior mechanism has the same
sign in all ten correlations with eight significant at p < 0.05, but it is a correlational account
of an association, not an intervention: nothing here manipulates the local prior and observes the
error rate change. It should be read as the most economical explanation available for §5.17, §5.18
and §5.20 together, and as a hypothesis that a training intervention could test.

---

### 5.21 Overfitting does not explain the generalisation failures, and the one separable shift term is confounded — *null demonstrated, shift term demonstrated on two targets, its identity not established*

The experiment designed to answer the overfitting question directly has now run. For each committed
configuration it refits every outer fold and scores it four ways: on the training fold itself
(`in_sample`), out-of-fold inside the training fold under the same grouping as the outer split
(`oof`), on the outer test fold (`test`), and on the outer test fold after refitting at the inner
folds' reduced training size (`test_small`). The three differences are `overfit_gap = in_sample −
oof`, `shift_gap = oof − test_small`, and `size_effect = test − test_small`, the last being the
learning-curve control the design named in advance as its single largest threat. 572 folds across
three targets, three model families and three splitting arms; 2 h 41 m on CPU; 498 folds carry a
committed AUROC and the remaining 74 have one class in the test fold, where AUROC is undefined and
no control exists either, so they are excluded rather than imputed. All 498 reproduced the committed
score bit-exactly, with the tied-pair allowance of §8 used zero times.

**The overfitting gap is large everywhere and it is not informative.** It runs from 0.085 to 0.332
with tight intervals in all 25 cells. Median in-sample AUROC is 0.975 for logistic regression, 0.993
for the random forest and 0.997 for XGBoost. The design pre-registered that fully grown bagged trees
reproduce their training labels and that their in-sample score therefore says nothing about
generalisation; the measurement shows the same holds for the other two families, because 2048-bit
fingerprints against a few hundred training molecules leaves all three in the regime where the
training labels are separable. `in_sample` is pinned near the ceiling by capacity, so `overfit_gap`
is close to a restatement of `1 − oof`: it measures the train-fit ceiling, not whether any test score
in this paper is inflated. **This is the null the proposal predicted, and it is the answer to the
question that prompted the experiment** — nothing in the decomposition shows the committed scores to
be optimistic, and the failures of §5.17 through §5.20 are not explained by memorisation.

**A shift term separable from the learning curve exists on two targets and only under one arm.** Of
27 cells including the two pre-registered pooled fallbacks, five have a shift interval excluding
zero with the control below a third of the effect, and all five are publication-held-out arms on
CDK2 and MAO-B: `shift_gap` +0.127 to +0.174 against controls of 0.002 to 0.013. Pooled over those
228 folds it is +0.154 [+0.127, +0.181] with a control of +0.005 [−0.004, +0.015], 170 of 228 folds
positive, Wilcoxon p ≈ 6 × 10⁻²⁰ — the control is a thirtieth of the effect. In every other cell the
term sits at or below its own control with an interval crossing zero: TG2 scaffold −0.004 to +0.015,
TG2 scaffold_pub −0.017 to +0.004, CDK2 scaffold_pub +0.011 to +0.018, MAO-B scaffold_pub +0.002 to
+0.009, and on both `paper_grouped` arms the control is the larger of the two. The publication
penalty is not a general property of these models.

**On TG2 the decomposition does not resolve, exactly as the design said it might.** The proposal
fixed in advance that TG2's publication arm holds about ten publications per training fold, that a
grouped inner 5-fold on that would be too noisy for a per-model statement, and that the fallback was
to pool across models. Pooled, `shift_gap` is +0.032 [−0.045, +0.106] with a Wilcoxon p of 0.35
against a control of −0.018 [−0.035, −0.001] — the same order as the effect. Per model it is wider
still, logistic regression giving −0.002 [−0.124, +0.143]. By the criteria fixed before the run,
tier 1 succeeds on CDK2 and MAO-B and **fails on the paper's primary target**, and the failure is
reported here rather than worked around. This also withdraws the single-model preview previously
quoted in §9: it came from TG2 logistic regression, which is one of the cells that turn out not to
resolve, and it should not have been given a directional reading.

**What the readable term is not.** Calling +0.154 publication shift requires a caveat that weakens
it considerably. `oof` is pooled across the training fold's publications, while `test` under
leave-one-publication-out is a single publication scored on its own; those are different
aggregations, and §5.14 already demonstrated that the publication-held-out decline on this data is
mostly novelty and aggregation rather than publication identity. The arm that separates the two
agrees with §5.14 here: `paper_grouped` holds publications out but scores them pooled, and its shift
term is +0.001 to +0.013, below its own control. The penalty appears when scoring is confined to one
publication and vanishes when the same held-out publications are pooled. **So the one readable term
in this decomposition is consistent with the aggregation effect of §5.14, and this experiment does
not separate the two.** Separating them needs an outer arm that holds out several publications,
scores them pooled, and matches leave-one-publication-out's training size — close to what
`paper_grouped` already is, and a cheap measurement rather than a new method.

One coverage gap is inherited rather than introduced: MAO-B's publication-held-out arm appears for
the random forest only, because the measurement refits what the committed sweep selected and
`results/sota_maob.csv` holds that arm for `rf` alone.

**Status: the null on overfitting is demonstrated**, in the weak but adequate sense that the
in-sample column cannot support an overfitting claim for any of these families at this data size and
the shift decomposition finds nothing hidden behind it. **The shift term is demonstrated on CDK2 and
MAO-B** under leave-one-publication-out with its learning-curve confound bounded two orders below
it, **not established on TG2**, and **its identity is not established on any target**. Full table in
`docs/overfitting_2026-11-10.md`; per-fold output in `results/overfitting_tier1.csv`.

---

## 6. Discussion I: six failures, six different controls

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

**The publication-held-out decline (§5.7, §5.8, re-attributed in §5.14)** is detected by asking
what else correlates with the label besides chemistry. A scaffold split, the field's standard answer
to leakage, is powerless against it because scaffolds do not nest within publications: one
laboratory's series spans several scaffolds, so holding out a scaffold leaves the same laboratory on
both sides of the fold. We initially expected this to be a hazard of datasets assembled from a
handful of papers, which describes most selectivity panels and most kinetic datasets. **The CDK2 and
MAO-B validations (§5.9) refuted that:** the effect appeared on 2,646 molecules from 351
publications and on 5,840 from 414, in neither case with a dominant paper, and on both it exceeded
the scaffold effect (−0.068 against −0.030, and −0.076 against −0.059). Those two comparisons are
aggregation-matched and stand as reported (§5.14).

**What we got wrong here was the mechanism, not the measurement.** We called this failure
"publication-level confounding" and wrote that the label correlated with the source publication and
the publication with the chemical series. §5.14 shows that most of the reported difference is
instead (i) the unit the AUROC is averaged over — about 92% of CDK2's 0.169 protocol gap, since
re-aggregating the grouped arm's *own* predictions by publication reproduces the leave-one-out
number to within 0.013 — and (ii) the novelty of the molecules a publication split places in the
test set, which accounts for a majority of the remainder as an association with wide intervals.
Whatever is left cannot be assigned to publication identity at all, because **no assay in either
replication dataset spans more than one publication** (0 of 380 and 0 of 516), so "who published it"
and "how it was measured" are the same variable in ChEMBL. The detecting question is therefore
sharper than we first thought: not only "what else correlates with the label?" but "are these two
protocols reporting the same quantity?" and "is the thing I want to blame identified in this data at
all?". Neither has a standard answer in the field.

The practical recommendation survives the re-attribution and is arguably strengthened. A
publication-held-out split is worth reporting alongside the scaffold split because it generates a
realistic novelty distribution automatically, which a scaffold split does not — but it must be
reported with its aggregation unit stated, and the resulting penalty must not be read as a
publication effect or carried to another dataset as a constant (§5.15).

A general caution follows from the pattern rather than from any single case. In three of the six
instances, the *erroneous* protocol produced the more publishable number: 0.95 instead of 0.87,
0.807 instead of 0.764, 0.974 instead of 0.574. Error in protocol design is not symmetric in its
consequences, and a result that looks unusually good on a small dataset is better treated as a
diagnostic signal than as a finding.

**A fifth failure, added later, is of a different kind: a method validated under one shift and
carried to another.** §5.13 found that conformal calibration on random training molecules — the
recipe that holds 0.922 coverage under scaffold shift — under-covers by 0.11–0.14 under publication
shift, and that calibrating on held-out *publications* brings measured coverage back to target on
all three targets. Nothing was wrong with the
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

## 7. Discussion II: what the results mean

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
inhibitors cross-react with Factor XIII-A (§5.7); and on TG2, models are valid only within 0.4
Tanimoto of that space (§5.6 — a per-dataset floor, since CDK2 retains AUROC 0.739 below it, §5.9).
Together these describe a target whose published chemistry is narrow, largely
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
commits on 15–32% of TG2 molecules and errs on a quarter to a third of those (§5.13). The two
replication targets improve the proportions without changing the conclusion — 41–48% and 46–56%
decisive, erring on 16–27% — so even at 5× and 11× the data there is no subset of predictions
clean enough to act on without experimental confirmation.

*Third, potency is not the limiting property.* A TG2 inhibitor for celiac disease would be taken
chronically by otherwise-healthy people, which makes selectivity within the transglutaminase family
a safety gate. Of 147 paired molecules only 20 are 10-fold TG2-selective and 10 are selective the
wrong way; the defensible, format-matched subset is 79 pairs with 13 selective molecules, 8 of them
in one publication (§5.7). Selectivity cannot be predicted across series, and the five-gate cascade
that would combine potency with selectivity, applicability domain, uncertainty and drug-likeness —
the last implemented as PAINS [18] and Brenk [19] structural alerts plus covalent-warhead flags in
`scripts/screen_candidates.py` —
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

**Eleven folds on the TG2 publication arm.** §5.11 and §5.13 both rest on leave-one-publication-out
over 11 publications, with per-fold standard deviations of 0.15–0.26 and fold sizes from 10 to 106
molecules. Every per-fold model comparison there is underpowered, and we report the two headline
differences as inconclusive rather than ranking the models. Only §5.13's coverage comparison, with
66 paired observations, is significant — and that one is now supported by 60 further paired
observations on each replication target, where the same comparison gives p = 1.6 × 10⁻¹² and
1.6 × 10⁻²². The underpowering is specific to TG2's model *rankings*, not to the coverage result.

**TG2 and the replication targets use different publication protocols.** TG2's published arm is
leave-one-publication-out; CDK2 and MAO-B use grouped 10-fold, because leave-one-out over 83 and
209 publications was not affordable (3.4 h and 20.8 h respectively for a single protocol check).
§5.14 establishes that the two protocols agree on pooled molecule-level AUROC but not on
publication-macro AUROC, so **every TG2-against-replication comparison in this paper is labelled
with the protocol it used**, and we do not pool a leave-one-out macro score with a grouped one.
A reader should treat any cross-protocol magnitude comparison here as indicative only.

**Publication-wise conformal calibration has no formal guarantee, and we do not claim one.** §5.13
sets this out: a calibration molecule from one publication and a test molecule from another are not
exchangeable, our grouped scheme is plain split conformal rather than a hierarchical conformal
estimator [21, 22], and `group_calibration` selects publications smallest-first, so the calibration
groups average roughly half the size of the group being predicted (11.9 molecules against about 26
on the replication targets). Group-level exchangeability is thus violated by construction. §5.16
measures the consequence rather than leaving it hypothetical: our method misses its nominal floor in
32 of 54 runs, and in the theorem-faithful design in 13 of 18.

**The guaranteed alternative is feasible only on some protocols, and buys its floor with
abstention.** §5.16's HCP arm needs K₁ ≥ ⌈1/α⌉ − 1 calibration publications, which TG2's
leave-one-publication-out protocol cannot supply at any α and CDK2's folds cannot supply at
α = 0.1. Where it is feasible it halves the singleton fraction at small K₁. So the choice between
the two methods is a real trade, not a strict improvement, and §10.2's recommendation states the
condition rather than naming one method unconditionally.

**Only two of the three calibration regimes could be reconstructed without refitting.** §5.13's
i.i.d. arm calibrates on random molecules from the training publications, which needs nonconformity
scores for molecules the model was fitted on; only out-of-fold predictions were saved. §5.16's
"i.i.d.-pooled" column is therefore a milder surrogate and should not be read as a re-measurement
of §5.13's i.i.d. failure.

**The novelty share is an association, not a decomposition.** §5.14's 48–80% is a standardisation
over measured similarity bands, not an identified causal effect: molecules are not randomised to
arms, novelty may proxy for other between-arm differences, and the residual crosses zero for every
model on both targets. Band standardisation also removes only between-band novelty, making the
figure a lower bound on the association; a continuous adjustment spans 13–80%. On TG2 it is not
estimable at all for want of common support. Any reading of these numbers as "novelty causes *x*%
of the decline" is unsupported by this analysis.

**The two replication targets disagree in sign and we cannot explain it.** §5.15 reports the
disagreement, rules out the matched comparison as its source, and shows that dataset size explains
the magnitude ordering but not the sign change. With two replication targets we could not fit an
experiment to discriminate the structural candidates. This is the clearest limit on how far any
penalty reported here can be generalised.

**The claim checker is a presence test, not an attribution test.** `scripts/check_paper.py`
recomputes 98 values from the result tables and confirms each appears in the paper; for claims added
after the replication campaign it also confirms the number appears *in the section that should carry
it*. Earlier claims are checked against the whole document, so a value could in principle match
coincidentally elsewhere. This is not hypothetical — before section anchoring was added, all six of
§5.14's matched penalties "passed" against an unrelated ridge RMSE, a Holm delta and a p-value while
the section itself quoted the wrong rows of its own table (corrected in §5.14).

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

**Observed generalisation failure is not demonstrated overfitting, and we do not call it that.**
Every result in this paper measures performance falling when the test distribution changes — a new
scaffold, a new publication, a cliff pair. None of it establishes that a model memorised its
training set, because the quantity that would show it, the gap between training and held-out
performance, was never measured: training-fold predictions were not saved by the sweep and
recovering them would require refitting every entrant. §5.1's leakage finding is the closest thing
in the paper to a memorisation result, and it is a property of the *split*, not of the models. The
word "overfitting" therefore appears nowhere as a conclusion, and §5.17–§5.19 in particular should
be read as evidence about evaluation protocols rather than about model capacity.

**Per-class rates rest on the publications that report both classes, and the splitting creates the
problem.** §5.17 aggregates on the publication, which is the right unit, but a publication reporting
only actives yields no specificity and one reporting only inactives yields no sensitivity. Such
units are skipped and counted. Decomposing the count shows the shortfall is mostly an artefact: in
the scaffold arms 50% to 72% of publication-**fold** units are single-class while only 9% to 37% of
**publications** are, because a scaffold split cuts each publication across train and test and the
test-side remnant is often all one class. In the publication-held-out arms the two figures coincide.
So the equal-weight estimator discards half to three-quarters of its units in the reference arm for
a reason that has nothing to do with the literature, and computes the remainder on a
class-filtered half of each publication. §5.17 therefore reports a pooled estimator beside it and
treats conclusions that depend on the choice as unresolved. This limits §5.17 more tightly than any
other analysis in the paper.

**Two publication averages exist and they differ.** Averaging every publication-fold row weights
large publications more heavily; averaging each publication over the folds it appears in weights
them equally. On TG2 the random forest's `scaffold_pub` sensitivity is 0.789 the first way and 0.660
the second. §5.17 uses the equal-weight figure throughout, because a publication appearing in eight
scaffold seeds is not eight independent observations, but the 0.13 difference is a reminder that
"aggregated per publication" is not a single well-defined operation.

**The activity-cliff definition is dominated by one of its three tests.** §5.18 adopts
MoleculeACE's rule unchanged [29], and on our data the generic-scaffold test admits nearly every
pair — 1,840 on TG2 against 23 for the fingerprint test. Our cliff pairs are therefore mostly
same-skeleton, different-substituent pairs. Censored negatives are excluded entirely, since a ">10
µM" record has no pIC50 and the tenfold criterion cannot be applied to it; that removes 12–26% of
similar discordant pairs, non-randomly, as they are the weakest-potency molecules. No sweep over the
0.9 and tenfold parameters was run.

**One pre-registered design assumption was wrong and had to be rewritten mid-experiment.** §5.19's
control group was specified as the models that ignore the validation fold. No such model exists in
the sweep — every entrant tunes on it. The group was redefined by what the fold decides before any
fit was run, and the original specification is recorded in the experiment's write-up, but a
pre-registration amended after inspecting the code is weaker than one that held.

**Random-forest AUROCs in this paper are not bit-reproducible, and are quoted to more digits than
that supports.** The forests are fitted with `n_jobs=-1`, so scikit-learn accumulates per-tree
probabilities in thread-completion order and repeated `predict_proba` calls on a single fitted model
differ by about one unit in the last place. Where a fold contains positive/negative pairs at exactly
equal predicted probability, that last bit decides the tie and AUROC moves by 1/(n_pos·n_neg).
Measured directly on the TG2 scaffold arm, seed 15: 25 calls on one fitted forest returned
0.8608647 and 0.8619734, one tied pair of that fold's 902 apart, on a fold with five exactly-tied
pairs. Only folds carrying exactly tied pairs can move at all, which makes the effect occasional
rather than pervasive — on this fold just 5 of 902 pairs are tied — but it is real and it is not
removed by fixing `random_state`. §5.21's per-fold refit now bounds the rate from two runs of the
same 498-fold sweep: the first run differed on exactly one fold, the second on none. The informative
observation is not the rate but which value appeared. That one disputed fold reproduced at
0.8619734 the second time and at 0.8608647 the first, the two values the determinism probe records
for it, so bit-exactness on a fold carrying tied pairs is a draw between known values rather than a
property of the pipeline. Consequently
no random-forest figure here should be read as stable beyond three decimal places, and any
reproduction check that demands bit-exactness should be expected to fail occasionally for reasons
that are not a defect. The effect is an order of magnitude below every reported difference it could
affect. XGBoost's float32 leaf sums introduce a separate, smaller discrepancy (3.8e-8) already
noted in §4.6.

**The reported accuracy of the prior TG2 model is not comparable to the figures in this paper, and
we do not compare them.** Wichka and Lai report 93% [1]; read against its own methods, that figure
is a stratified 5-fold cross-validation mean over all 336 classified molecules rather than a
held-out test score, on a set that is 64.3% active, against a majority-class baseline whose AUROC is
0.50. Their negative class pools 45 documented inactives with 75 compounds recorded as
"unspecified", and their split is stratified random with no scaffold, publication or temporal
holdout. Every quantity we report is threshold-free AUROC under scaffold- or publication-grouped
splits with censored records handled explicitly, so the two numbers measure different things on
differently constructed labels. A faithful reproduction is feasible — their winning model is fully
specified — but it must report their protocol and ours separately rather than placing 93% beside any
AUROC in this paper. `docs/literature_20papers_2026-11-10.md` records the full reading.

---

## 9. Future work

**Separate the aggregation effect from publication shift.** §5.21 has run, and it leaves one
question that its own design cannot answer: its readable shift term compares an out-of-fold estimate
pooled across publications against a test fold that is a single publication, so the term is
confounded with the aggregation effect of §5.14. The measurement that would separate them is an
outer arm holding out several publications at once and scoring them pooled while matching
leave-one-publication-out's training size. The existing `paper_grouped` arm is close to this and its
shift term is already near zero, which is why we expect the separated effect to be small; what is
missing is the matched training size, and adding it is a modest rerun rather than a new design.
Two cheaper gaps sit alongside it: MAO-B's publication-held-out arm was only ever committed for the
random forest, and TG2's publication arm has too few publications for the decomposition to resolve
at all, which no amount of compute fixes.

**Candidate interventions, with the prediction stated first.** Two directions are better motivated
by the external literature than by anything we have run. An activity-cliff-aware loss [42] reports
gains under scaffold-held-out and similarity-capped splits, which are the protocols comparable to
ours; conformation-aware pretraining [40] reports improvements on structure-activity-cliff
benchmarks. Against both, accuracy did not scale with model complexity on cliff pair classification
[43]. We record now, before running either, that the honest prior is a small or absent effect on
publication-held-out TG2, that a fair comparison needs the tuned random forest at the same budget,
and that a null must be reported as a null. On the uncertainty side, combining data-based and
model-based signals through an error model [44] and the robust estimator of [45] are alternatives to
the hierarchical conformal route of §5.16; both are cited from abstracts only, so their designs
would have to be read before adoption.

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

**Independent datasets — begun, not finished.** Three of the six checks have now been repeated on
CDK2 (§5.9) and all three reproduced. MAO-B has since been added as a third target and agrees
with CDK2 on the ordering. What remains is a target from a *different data source*: all three are
ChEMBL records shaped by one curation pipeline, so a dataset curated elsewhere would test whether
the publication-held-out decline is a property of the literature or partly of ChEMBL's document
attribution. It would also be the only way to separate publication identity from assay protocol,
which ChEMBL's one-assay-per-paper structure makes impossible here (§5.14).
And the
nondeterminism check has still been run only once, on one GPU and one model family — the cheapest
way to generalise it is to repeat the identical-configuration comparison on any other quantized
fine-tune.

**A chemically appropriate screening library.** The one direction not yet ruled out. Approved-drug
repurposing is exhausted (§5.6), but analogs within 0.5–0.7 Tanimoto of known actives sit inside the
validated applicability domain. Covalent fragment collections would require the mechanism modelling
that §5.8 found infeasible, so they should follow the kinetic-unit recovery rather than precede it.

**Compare against hierarchical conformal prediction — done, §5.16.** This was the one experiment
that would convert §5.13's strongest result from an empirical regularity into a supported method
claim, and it has now been run on the saved predictions with no model refitted. HCP holds its floor
in all 54 runs against 22 for our method, at the cost of roughly halving the decisive answers where
calibration publications are few. Two things it leaves open. First, **the leave-one-publication-out
protocol cannot support it at all** — one unseen publication per fold against the K₁ ≥ 9 that
α = 0.1 needs — so a TG2-style protocol must either be re-run as grouped K-fold or accept that no
guaranteed calibration is available to it. Second, we implemented the split construction of
Theorem 1 only; the repeated-subsampling and CDF-pooling variants [21, 22] and clustered conformal
prediction [26] trade conservatism against assumptions differently, and a comparison among them
would say whether HCP's set sizes can be reduced without giving up the floor. That is a
methods-paper question rather than a blocker for this one.

**Mondrian conformal prediction** stratified by scaffold cluster, to give per-chemotype rather than
marginal coverage [6]. Partly done, and the motivation has changed: §5.13 runs the class-wise
variant on the publication arm, where it buys 30% more decisive answers on TG2 but **nothing on
either replication target** (singleton fraction 0.438 → 0.458 on CDK2 and 0.523 → 0.499 on MAO-B),
so we no longer recommend it on the strength of the TG2 result. What does still motivate
stratification is that **conditional coverage fails on all three targets, though not in the same
direction**: on CDK2 and MAO-B the unfamiliar bands sit at or below target while the most familiar
band overshoots (ChemBERTa-2 frozen, 0.876 → 0.933 and 0.872 → 0.973), whereas on TG2 the failure is
in the high-similarity band itself (logistic regression covers 0.375 in [0.7, 1.01]). Marginal
coverage hides all of this, and per-chemotype stratification is the gap's natural repair. That the
direction is target-dependent is a further reason not to ship a single recipe.

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
unresolvable at 11 folds. CDK2 (351 publications) and MAO-B (414) already exist in this project and
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
publication shift while publication-wise calibration empirically does not, on all three targets**
(0.784 → 0.914 on TG2 at Δ +0.113, p = 2.5 × 10⁻⁸; 0.811 → 0.896 on CDK2 at Δ +0.084,
p = 1.6 × 10⁻¹²; 0.819 → 0.910 on MAO-B at Δ +0.090, p = 1.6 × 10⁻²², §5.13 — measured coverage,
no distribution-free guarantee claimed); **most of CDK2's publication-held-out macro decline is the
aggregation unit**, since re-aggregating the grouped arm's own predictions by publication closes
about 92% of a 0.169 gap (§5.14); **no assay spans two publications** in either replication dataset
(0 of 380, 0 of 516), so publication identity and assay protocol are not separately identifiable
(§5.14); **split hierarchical conformal prediction holds its nominal coverage floor in all 54 runs
where our publication-grouped calibration holds it in 22**, with the advantage decaying in the
number of calibration publications as its own 2/(K₁+1) bound predicts, and costing roughly half the
decisive answers at small K₁ (§5.16); **TG2 and FXIII-A potency correlate positively** (r = +0.41) yet
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

**Inconclusive (continued):** the matched publication penalty has opposite signs on CDK2 and MAO-B
(−0.034 to −0.110 against +0.007 to +0.062, XGBoost significant in both directions), and we offer
no mechanism (§5.15); **CheMeleon beats the random forest on the publication arm of all three
targets** (+0.023, +0.023, +0.033; inverse-variance pooled +0.031 [+0.007, +0.055], Cochran's
Q = 0.09 on 2 df) but at Holm p = 0.057 it misses correction, and the effect is specific to
CheMeleon rather than to pretrained models, since frozen ChemBERTa-2 pools negative on the same arm.
**The publication-shift penalty is not class-symmetric on TG2** — sensitivity falls under all three
aggregation estimators (fold −0.503, pooled −0.499, equal-weight publication −0.340 for the random
forest) while specificity falls under none; the replication targets lose both rates, and MAO-B's
apparent specificity asymmetry does not survive a change of estimator. Nothing survives Holm on any
target under any estimator, and the cross-target comparison is confounded with protocol and
prevalence (§5.17).

**Statistically supported (continued):** **activity cliffs are harder for every model on every
target** — cliff pairs rank below distant pairs at a matched tenfold potency gap in 58 of 58
model × arm comparisons, 16 significant after Holm, with Chemprop below chance at 0.468 on TG2
`scaffold_pub` (§5.18), and the direction survives a fingerprint-only similarity definition in 34 of
38 comparisons, unanimously on both replication targets (§5.18); **accuracy is uninformative on
these folds**, since Chemprop posts the highest accuracy in the TG2 publication arm (0.704) at
chance discrimination (AUROC 0.542, balanced accuracy 0.515, §5.17); **errors concentrate by source
publication** — observed between-publication spread two to three times a size-matched permutation
null at p < 0.001 on every target and arm, with the worst fifth of publications carrying 44–60% of
all error mass, and about one molecule in ten misclassified by nine of ten entrants where
independent errors predict almost none (§5.20).

**Direction-consistent, offered as the most economical explanation rather than as a demonstration:**
**the models predict the local class prior rather than the molecule** — a publication's active
fraction correlates negatively with its false-negative rate and positively with its false-positive
rate in all ten target × arm × rate combinations, eight at p < 0.05; molecules below 0.4 Tanimoto to
training are called inactive (87.9% of MAO-B actives missed under shift) and those above 0.7 are
called active (94.3% false-positive rate on TG2). This accounts for §5.17's asymmetry and its
cross-target variation without a target-specific story, but nothing here manipulates the prior and
observes the error change (§5.20).

**A demonstrated null, pre-registered as the expected outcome:** **the generalisation failures are
not memorisation** — across 498 refit folds the in-sample-minus-out-of-fold gap is 0.085–0.332
everywhere, but median in-sample AUROC is 0.975–0.997 for all three model families, so that gap
measures the train-fit ceiling of 2048-bit fingerprints against a few hundred molecules and cannot
support an overfitting claim either way. The shift term that survives the learning-curve control is
confined to publication-held-out CDK2 and MAO-B, +0.154 [+0.127, +0.181] against a control of
+0.005, does not resolve on TG2 at all, and is confounded with §5.14's aggregation effect, so its
identity is not established (§5.21).

**Not established, with the prediction written first:** **validation-fold class balance does not
explain per-seed variance** — every confidence interval spans zero on both TG2 arms, largest
r² = 0.051, point estimates mostly of the opposite sign, and no contrast between models that select
a training length on the fold (+0.046 mean slope) and those that do not (+0.120); the spread is
better explained by model skill (mean AUROC against SD, r = −0.912 on TG2 `scaffold`), an
explanation found after the fact and therefore untested (§5.19).

**Scoped rather than overturned:** conformal prediction holds coverage under distribution shift —
true under scaffold shift with i.i.d. calibration, false under publication shift with the same
recipe (§5.13). **Novelty explains most of the publication-held-out decline** — supported as an
association with wide intervals (48–80% across models and targets, residual crossing zero
everywhere), not as a causal decomposition, and not estimable on TG2 (§5.14). **Mondrian
calibration is the variant to use** — true on TG2 (30% more decisive answers), null on both
replication targets, so withdrawn as a recommendation (§5.13, §9).

### 10.2 The contribution

On literature-derived bioactivity datasets, the evaluation protocol determined the conclusion more
often than the model did. We document six mechanistically distinct instances on TG2, each with a
measured before-and-after effect, three of which were caught only after being written down as
findings; three of the six erroneous protocols produced the more attractive number. Three of the
six then reproduced on two independent targets of different size, chemistry and provenance
structure. A seventh error, caught during the final audit and not counted among the six because it
was ours rather than the protocol's, is recorded for the same reason the others are: §5.14's matched
comparison initially quoted the unmatched rows of its own result table, and the claim checker passed
it because the digit strings happened to occur elsewhere in the paper. It now verifies a number
against the section that must carry it (§8).

The replication targets constrain the claim in both directions. The publication-held-out *decline*
proved more general than we predicted, appearing on a 351-publication dataset and there exceeding
the scaffold effect; its *attribution* proved much weaker than we claimed, and §5.14 re-assigns most
of it to the aggregation unit and to chemical novelty, with the remainder unidentifiable because
assay and publication are the same variable in ChEMBL. The 0.4 Tanimoto applicability floor proved
less general than TG2 suggested, and is withdrawn as a cross-dataset claim. The matched penalty
itself turns out to have **opposite signs on the two replication targets** (§5.15), which we report
as an unexplained constraint rather than averaging it away.

If one result here is worth carrying forward, it is not a penalty magnitude but the finding that
**i.i.d. conformal calibration under-covers badly whenever whole publications are held out, and
group-aware calibration does not** — three targets, two protocols, every p < 10⁻⁷ — together with
§5.16's demonstration that the group-aware method with a theorem behind it honours its floor where
our empirical one misses it in 32 of 54 runs.

**Which method we recommend reporting.** Split hierarchical conformal prediction [22] is the
strongest baseline and should be the primary report wherever the protocol can supply
⌈1/α⌉ − 1 calibration publications — grouped publication K-fold on a few hundred papers does, and
we recommend that protocol partly for this reason. Where it cannot, which includes any
leave-one-publication-out design and any dataset with few papers, there is no guaranteed option and
publication-grouped pooled calibration is the best available approximation, to be reported with
its measured coverage and without a guarantee. In both cases the singleton fraction and the error
among singletons must be reported next to coverage, because every method here achieves its coverage
largely by declining to answer.

The practical recommendation is modest and specific: alongside the scaffold split that is already
standard, small-dataset bioactivity studies should report (i) the independent unit behind every
confidence interval, (ii) a repeat run of the identical configuration where a GPU-trained model
measures an effect below ~0.05 AUROC, and (iii) a publication-held-out split wherever the dataset is
assembled from a small number of papers.

### 10.3 Status of this draft

Revision 4, 2026-11-05, after a full numerical and statistical audit, a reference verification pass
and the hierarchical conformal comparison. **Ready for supervisor review. Not ready for
submission.** The correction ledger is [`audit_2026-11-05.md`](audit_2026-11-05.md); the state is
summarised here.

*Verified.* Every headline number is recomputed from a committed result table by
`scripts/check_paper.py` (113 values, all passing); for sections written after the replication
campaign the check is anchored to the section that must carry the number, because a document-wide
substring test had already passed six values the paper did not contain. The statistical unit is
stated per regime in §4.5 and publication-clustered intervals are used wherever the molecule is not
independent. 406 automated tests pass. §5.16's implementation was checked against its own theorem by
15 synthetic-data tests before any result from it was read, which is how a boundary bug that turned
the smallest usable calibration set into a trivial predictor was caught. All 28 references have been
checked against publisher, PubMed, arXiv or institutional-repository records.

*Nineteen corrections are logged.* Four changed substantive claims in this paper: §5.14's matched
comparison quoted the unmatched rows of its own table; TG2's matched molecule count was 394 and is
428; reference 24 was cited for Mondrian conformal prediction, which it does not discuss; and the
within-fold design's exchangeability was justified by a property the fold assignment does not have
(§5.16 now rests on our own randomisation instead, which is a stronger argument).

*Outstanding and substantive.*

1. **The CDK2/MAO-B sign disagreement (§5.15) is unexplained** and cannot be resolved with two
   replication targets. Reported as a constraint on the claim, not papered over.
2. **Novelty is narrow and the closest prior art is close.** Reference 28 asks our question on a
   different grouping, and we found it only during the final audit. §2.3 claims what survives: no
   methodological novelty, and an empirical contribution resting on the publication as the group,
   the assay/publication non-identifiability, and the design constraint on leave-one-out protocols.
   A reviewer who knows this literature better may find closer work still.
3. **Only the split construction of Theorem 1 is implemented.** The repeated-subsampling and
   CDF-pooling variants [21, 22] and clustered conformal prediction [26] are not compared, so §5.16
   supports "HCP holds its floor and ours does not", not "HCP is the best group-aware method".
4. **Chemprop on the replication targets** is the control that would separate pretraining from
   architecture for the CheMeleon observation, which is why that observation is reported as
   exploratory rather than as a finding.
5. **The assay confound cannot be closed within ChEMBL at all** (§5.14). A limitation, not a gap.

*Outstanding and editorial.* No target venue selected. At roughly 21,900 words the manuscript is
long for every venue we would consider and the six-failure structure needs compressing for anything
but a methods journal. Author and affiliation details, data-availability and competing-interests
statements, figure preparation (Appendix B lists figures not yet drawn to publication quality) and
any venue-specific reporting checklist remain.

*Venues, in the order we would try them.* **Journal of Cheminformatics** is the best fit: it
publishes evaluation-methodology work at length, has published much of the conformal-prediction
literature cited here [25], and would not require the failure catalogue to be cut. **Journal of
Chemical Information and Modeling** is the strongest alternative and the more selective one; it
would want the manuscript roughly halved and the TG2 application compressed to its motivating role.
**Machine Learning for Health / ML4Molecules-style workshops** would take the §5.13–§5.16
calibration thread alone as a short paper, which is the most self-contained and most defensible
piece of the work, and would be the fastest route to feedback. A general ML venue is a poor fit:
the methodological novelty is nil by our own assessment, and the contribution is empirical and
domain-specific.

*What would change the verdict to submission-ready.* Items 2 and 3 above are the only ones that
affect whether the central claim is correctly bounded, and both are now stated rather than open.
What remains is editorial: choose a venue, cut to its length, and prepare the figures. We estimate
that as days of work, not weeks, and no further experiment is required for it.
---

## 11. Verified references

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
3. ✓ Li Y, Wagle J. LLM-as-Jev: LLMs Are Already Jev-Style Decision Models — When and How to
   Fine-Tune Them. arXiv:2610.02076, submitted 2026-10-01. *(Preprint; not peer reviewed at time
   of writing.)*
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
21. ✓ Dunn R, Wasserman L, Ramdas A. Distribution-free prediction sets for two-layer hierarchical
    models. *Journal of the American Statistical Association* 2023;118(544):2491–2502.
    arXiv:1809.07441. *(Conformal prediction for a new observation from a previously unseen group.)*
22. ✓ Lee Y, Barber RF, Willett R. Distribution-free inference with hierarchical data.
    arXiv:2306.06342, 2023 (revised 2024). *(Hierarchical conformal prediction; strengthens the
    earlier 1 − 2α guarantee to 1 − α by repeated subsampling. Preprint; no journal version located
    at time of writing.)*
23. ✓ Barber RF, Candès EJ, Ramdas A, Tibshirani RJ. Conformal prediction beyond exchangeability.
    *Annals of Statistics* 2023;51(2):816–845. doi:10.1214/23-AOS2276. arXiv:2202.13415.
    *(Coverage degrades by a bounded gap when exchangeability fails.)*
24. ✓ Alvarsson J, Arvidsson McShane S, Norinder U, Spjuth O. Predicting With Confidence: Using
    Conformal Prediction in Drug Discovery. *Journal of Pharmaceutical Sciences*
    2021;110(1):42–49. doi:10.1016/j.xphs.2020.09.055 *(Minireview. Cited for conformal prediction
    as a consistent treatment of the applicability domain, and for the framing that a
    classification prediction set may hold no label, one label or several.)*
25. ✓ Bosc N, Atkinson F, Felix E, Gaulton A, Hersey A, Leach AR. Large scale comparison of QSAR
    and conformal prediction methods and their applications in drug discovery. *Journal of
    Cheminformatics* 2019;11:4. doi:10.1186/s13321-018-0325-4. PMID 30631996; PMCID PMC6690068.
    *(550 human protein targets drawn from ChEMBL. A published comment by Krstajic,
    doi:10.1186/s13321-019-0387-y, disputes its treatment of random-forest variability; the use
    made of it here — that conformal prediction has been applied at this scale — is not affected.)*
26. ✓ Ding T, Angelopoulos AN, Bates S, Jordan MI, Tibshirani RJ. Class-conditional conformal
    prediction with many classes. *Advances in Neural Information Processing Systems (NeurIPS)*
    2023. arXiv:2306.09335. *(Clustered conformal prediction: calibrate per cluster of groups; the
    guarantee is cluster-conditional, not group-conditional.)*
27. ✓ Guo Q, Hernandez-Hernandez S, Ballester PJ. Scaffold splits overestimate virtual screening
    performance. arXiv:2406.00873, 2024. *(Preprint; 60 NCI-60 datasets, scaffold against Butina
    and UMAP splits.)*

28. ✓ Rakhshaninejad M, Jurgens M, Dewolf N, Waegeman W. Conformal Prediction for Uncertainty
    Estimation in Drug-Target Interaction Prediction. arXiv:2505.18890, 2025. *(Preprint, not peer
    reviewed at time of writing. Conformal coverage on KIBA under four splits including
    drug-based, protein-based and fully unseen drug–protein pairs, comparing marginal,
    group-conditioned and three cluster-conditioned variants. The closest prior art to §5.13 and
    §5.16; see §2.3.)*

29. ✓ van Tilborg D, Alenicheva A, Grisoni F. Exposing the Limitations of Molecular Machine
    Learning with Activity Cliffs. *Journal of Chemical Information and Modeling*
    2022;62(23):5938–5951. doi:10.1021/acs.jcim.2c01073. PMID 36456532; PMCID PMC9749029.
    *(24 methods across 30 ChEMBL bioactivity targets; descriptor-based models outperform deep
    ones and every method struggles on activity cliffs. Benchmark and code at
    `github.com/molML/MoleculeACE`, MIT. The cliff definition and its parameters in §5.18 are
    taken from this paper and its implementation. Verified 2026-11-09 against the PMC record;
    the publisher page returned HTTP 403 and PMC was used instead. Note that MEDLINE gives the
    issue date as 12 December 2022 and PMC as 1 December 2022; volume, issue and pages agree.)*

References 30–46 were supplied as a reading list on 2026-11-10 and verified that day against
indexed primary records (Europe PMC `resultType=core`, PubMed or arXiv). Methods were read in full
for 36–40 where cited for software behaviour, and for 41, 42, 43 and 46; where only the abstract
was obtainable this is stated in the entry, and no methodological claim is made about such a paper
anywhere in this draft. `docs/literature_20papers_2026-11-10.md` records the reading in full,
including four first-pass readings that the methods contradicted and that were corrected.

30. ✓ Büchold C, Hils M, Gerlach U, Weber J, Pelzer C, Heil A, Aeschlimann D, Pasternack R.
    Features of ZED1227: The First-In-Class Tissue Transglutaminase Inhibitor Undergoing Clinical
    Evaluation for the Treatment of Celiac Disease. *Cells* 2022;11(10):1667.
    doi:10.3390/cells11101667. PMID 35626704; PMCID PMC9139979.
31. ✓ Dotsenko V, Tewes B, Hils M, et al. (CEC-3 Investigators). Transcriptomic analysis of
    intestine following administration of a transglutaminase 2 inhibitor to prevent gluten-induced
    intestinal damage in celiac disease. *Nature Immunology* 2024;25(7):1218–1230.
    doi:10.1038/s41590-024-01867-0. PMID 38914866; PMCID PMC11224021.
32. ✓ Taavela J, Elli L, Bouma G, Tye-Din JA, Schuppan D, Lundin KEA, Schumann M. Upcoming
    Treatments in Celiac Disease: From Luminal Enzymes to Oral Immune Tolerance. *United European
    Gastroenterology Journal* 2026;14(3):e70222. doi:10.1002/ueg2.70222. PMID 42033586;
    PMCID PMC13109976.
33. ✓ Javed A, Saeed A, Akhtar A, Khalid A, Khan AS, Rabba W. Role of Regulatory T Cells and
    Transglutaminase 2 Inhibitors in Celiac Disease: A Systematic Review. *Cureus*
    2025;17(5):e84901. doi:10.7759/cureus.84901. PMID 40575215; PMCID PMC12199851.
34. ✓ Kounatidis D, Pavlou A, Evangelopoulos A, et al. Rethinking Celiac Disease Management:
    Treatment Approaches Beyond the Gluten-Free Diet. *Biomedicines* 2025;14(1):29.
    doi:10.3390/biomedicines14010029. PMID 41595566; PMCID PMC12837603. *(Published 22 December
    2025 into the 2026 volume; the reading list gave the year as 2026.)*
35. ✓ Tye-Din JA. Evolution in coeliac disease diagnosis and management. *JGH Open*
    2024;8(7):e13107. doi:10.1002/jgh3.13107. PMCID PMC11217771.
36. ✓ Wu Z, Ramsundar B, Feinberg EN, Gomes J, Geniesse C, Pappu AS, Leswing K, Pande V.
    MoleculeNet: a benchmark for molecular machine learning. *Chemical Science* 2018;9(2):513–530.
    doi:10.1039/c7sc02664a. PMID 29629118; PMCID PMC5868307.
37. ✓ Yang K, Swanson K, Jin W, et al. Analyzing Learned Molecular Representations for Property
    Prediction. *Journal of Chemical Information and Modeling* 2019;59(8):3370–3388.
    doi:10.1021/acs.jcim.9b00237. PMID 31361484; PMCID PMC6727618. *(See also the published
    correction, PMID 31814400.)*
38. ✓ Heid E, Greenman KP, Chung Y, Li SC, Graff DE, Vermeire FH, Wu H, Green WH, McGill CJ.
    Chemprop: A Machine Learning Package for Chemical Property Prediction. *Journal of Chemical
    Information and Modeling* 2024;64(1):9–17. doi:10.1021/acs.jcim.3c01250. PMID 38147829;
    PMCID PMC10777403.
39. ✓ Graff DE, Morgan NK, Burns JW, et al. Chemprop v2: An Efficient, Modular Machine Learning
    Package for Chemical Property Prediction. *Journal of Chemical Information and Modeling*
    2026;66(1):28–33. doi:10.1021/acs.jcim.5c02332. PMID 41453060. *(Primary-verified 2026-11-10;
    this draft previously cited Chemprop v2 without a verified record.)*
40. ✓ Qiao J, Jin J, Wang D, et al. A self-conformation-aware pre-training framework for molecular
    property prediction with substructure interpretability. *Nature Communications*
    2025;16(1):4382. doi:10.1038/s41467-025-59634-0. PMID 40355450; PMCID PMC12069555. *(SCAGE.)*
41. ✓ López-Pérez K, Miranda-Quintana RA. Extended Activity Cliffs-Driven Approaches on Data
    Splitting for the Study of Bioactivity Machine Learning Predictions. *Molecular Informatics*
    2025;44(1):e202400054. doi:10.1002/minf.202400054. PMID 39558572; PMCID PMC12143937.
    *(Methods read in full. Eight selection strategies — random, medoid, uniform, diverse,
    Kennard-Stone, eSALI, anti-eSALI, bSALI — on the 30 van Tilborg [29] targets, regression scored
    by RMSE. None is a scaffold or group split, and the selecting method also chooses the test set,
    so strategies are not scored on a common held-out set. Cited here only for eSIM and eSALI as
    activity-landscape descriptors, and not on the splitting question in either direction.)*
42. ✓ Cui C, Su X, Zhang Z, Velez-Arce A, Wang J, Shi X, Zhang Y, Wu J, Chen YZ, Zitnik M, Shen W.
    Activity-cliff awareness enables robust graph learning for molecular property prediction.
    *Nature Communications* 2026;17(1):9376. doi:10.1038/s41467-026-75713-2. PMID 42680733;
    PMCID PMC13534492. *(ACANet. Methods read in full. The headline protocol splits 80/20 within
    spectral ECFP clusters, which is weaker than our arms; the results that bear on our setting are
    the supplementary Bemis–Murcko scaffold-held-out split, improved on all 30 targets, and the
    cliff-rich split removing test molecules above 0.90 similarity to training, 28 of 30.)*
43. ✓ Tamura S, Miyao T, Bajorath J. Large-scale prediction of activity cliffs using machine and
    deep learning methods of increasing complexity. *Journal of Cheminformatics* 2023;15:4.
    doi:10.1186/s13321-022-00676-7. PMCID PMC9825040. *(Methods read in full. Pair classification,
    not single-compound prediction; class-specific cliff threshold of mean plus two standard
    deviations. The leakage-excluded arm covers 42 of the 100 classes. The abstract's claim that
    accuracy was driven by memorisation is the authors' interpretation of the leakage comparison;
    no control experiment in the paper isolates memorisation, and we do not repeat the claim as
    established.)*
44. Parrondo-Pizarro R, Lanini J, Rodríguez-Pérez R. Uncertainty Quantification in Molecular
    Machine Learning for Property Predictions under Data Shifts. *Journal of Chemical Information
    and Modeling* 2026;66(2):923–935. doi:10.1021/acs.jcim.5c02381. PMID 41533656. *(Citation
    verified; full text paywalled and not obtained, so no methodological claim is made.)*
45. Kötter A, Singh K, Matter H, Hessler G, Grebner C. Upgrading Reliability in Molecular Property
    Prediction by Robust Quantification of Uncertainty from Machine Learning Models. *Journal of
    Chemical Information and Modeling* 2025;65(20):10819–10831. doi:10.1021/acs.jcim.5c00464.
    PMID 41060699. *(Citation verified; full text paywalled and not obtained. The methods,
    datasets and splitting strategies are not stated in the indexed record, so this draft cites its
    conclusion only and makes no comparison of designs.)*
46. ✓ Agboola SS, Agboola OE, Olasehinde OR, et al. Activity cliffs resist prediction within and
    across protein kinases: a leakage-controlled machine-learning analysis of structure-potency
    discontinuities. *Bioorganic Chemistry* 2026;181:110406. doi:10.1016/j.bioorg.2026.110406.
    PMID 42612264. *(Abstract and deposited code and derived results at doi:10.5281/zenodo.21522395
    read; publisher full text paywalled. Reported figures are average precision, not AUROC.)*

References 47–53 are the entrants benchmarked in §5.10 and the two benchmark studies cited beside
them. Until 2026-11-10 this draft benchmarked ten models while citing none of them; each entry below
was verified that day against the arXiv or publisher record named in it. Four are arXiv preprints
with no journal reference listed, which is marked in the entry rather than left to the reader.

47. ✓ Burns JW, Zalte AS, Abreu CRA, Sieg J, Feldmann C, Mathea M, Green WH. Deep Learning
    Foundation Models from Classical Molecular Descriptors. arXiv:2506.15792, v1 18 June 2025,
    v2 9 February 2026. *(CheMeleon. Preprint; no journal reference listed.)*
48. ✓ Chithrananda S, Grand G, Ramsundar B. ChemBERTa: Large-Scale Self-Supervised Pretraining for
    Molecular Property Prediction. arXiv:2010.09885, 19 October 2020. *(Preprint; no journal
    reference listed.)*
49. ✓ Ahmad W, Simon E, Chithrananda S, Grand G, Ramsundar B. ChemBERTa-2: Towards Chemical
    Foundation Models. arXiv:2209.01712, 5 September 2022. *(Preprint; no journal reference listed.
    The checkpoint used here is `DeepChem/ChemBERTa-77M-MTR`. Third-party reproductions of this
    paper's MoleculeNet table are not quoted anywhere in this draft.)*
50. ✓ Ross J, Belgodere B, Chenthamarakshan V, Padhi I, Mroueh Y, Das P. Large-scale chemical
    language representations capture molecular structure and properties. *Nature Machine
    Intelligence* 2022;4:1256–1264. doi:10.1038/s42256-022-00580-7. Preprint arXiv:2106.09553.
    *(MoLFormer-XL; checkpoint `ibm-research/MoLFormer-XL-both-10pct`. The arXiv page names no
    journal; the journal version was confirmed separately against the Crossref record.)*
51. ✓ Ben Hicham KK, Rittig JG, Grohe M, Mitsos A. Tabular foundation models for in-context
    prediction of molecular properties. arXiv:2604.16123, v1 17 April 2026, v2 20 April 2026.
    *(The tabular entrant in §5.10. Preprint; no journal reference listed, and recent.)*
52. ✓ Praski M, Adamczyk J, Czech W. Benchmarking Pretrained Molecular Embedding Models For
    Molecular Representation Learning. arXiv:2508.06199, v1 8 August 2025, v5 3 November 2026.
    *(Preprint; no journal reference listed. Version drift noted: an earlier internal review of
    ours cited v4.)*
53. ✓ Deng J, Yang Z, Wang H, Ojima I, Samaras D, Wang F. A systematic study of key elements
    underlying molecular property prediction. *Nature Communications* 2023;14(1):6395.
    doi:10.1038/s41467-023-41948-6. PMID 37833262; PMCID PMC10575948.

References 21–28 were added during the final audits of 2026-11-05, and reference 29 on 2026-11-09
for §5.18. **All bibliographic details
above marked ✓ have now been checked against publisher, PubMed, arXiv or institutional-repository
records.** Two corrections arose from that check and are logged in
[`audit_2026-11-05.md`](audit_2026-11-05.md): reference 24 does not discuss Mondrian conformal
prediction and had been cited for it, so Mondrian is now attributed to [6] alone; and reference 28
was located during the check, is closer prior art than anything previously cited, and required
§2.3's novelty claim to be narrowed. One further candidate reference — a 2026 *Journal of Chemical
Information and Modeling* paper on leave-assay-out validation of bioactivity models — could not be
verified against a publisher record and is therefore **not cited**, although it would be relevant
to §2.1 if confirmed.

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
