# Can a local 9B language model help find TG2 inhibitors for celiac disease?

**Month 1 research report — 2026-10-27**
All computation performed locally on one RTX 5070 Ti. No hosted model APIs, no cloud GPUs.

---

## 1. Research question

Tissue transglutaminase 2 (TG2) deamidates gluten peptides into the forms that trigger the
autoimmune response in celiac disease, which makes it the central small-molecule target for a
non-dietary therapy. No TG2 inhibitor has reached the clinic: GSK3915393 was discontinued in phase 1
when GSK ended its celiac programme.

The question this project asks is narrow and answerable:

> Used as a decision model in the style the professor suggested (Jev), does a locally-run ~9B
> language model predict TG2 inhibitory potency better than standard cheminformatics baselines on
> identical splits — and if not, does it contribute anything they cannot?

A secondary question emerged from the data and turned out to matter more: **when should any of these
models be trusted at all?**

## 2. Dataset: 305 → 631 molecules

| | molecules | source |
|---|---|---|
| Starting point | 305 | PubChem export from Wichka & Lai (2024), exact IC50 only |
| After recovering censored measurements and stripping salts | 367 | same source, corrected |
| After merging ChEMBL target CHEMBL2730 | **631** | + 1,587 human TG2 activities |

Merging was only done after checking the sources agree, which they do almost exactly:

| Cross-source check | Result |
|---|---|
| Molecules with an exact pIC50 in both | 173 |
| Spearman correlation | **1.000** (Pearson 0.997) |
| Median absolute difference | **0.000 log units** |
| Class labels compared | 196 molecules, **100% identical** |

Ki, Kinact, kon/koff and %-inhibition records were **excluded rather than converted** — an IC50 and a
Ki are not interchangeable without the substrate Km, which neither source reports.

The practical gain is statistical power: test folds grew from 24 to 62 molecules and the standard
deviation of test AUROC nearly halved. Detail: [`data.md`](data.md).

## 3. Data validation and leakage corrections

**Three errors found, all fixed:**

1. **Units confirmed as µM**, stated directly in the source paper. This closes the open question
   flagged last week, and confirms `pIC50 = 6 − log10(IC50)` was correct.
2. **Every documented inactive was being discarded.** All `Inactive` rows carry a `>` qualifier, and
   the pipeline kept only exact `=` values. A bound of `>10 µM` yields no pIC50 but does prove "not
   potent at 1 µM"; these are now kept as negatives (+62 molecules).
3. **18 salts were not stripped**, so counterions contributed fingerprint bits and could move the
   Murcko scaffold used for splitting.

**Leakage audit** (`scripts/audit_splits.py`):

| | scaffold split | random split |
|---|---|---|
| Murcko scaffolds shared train↔test | **0** | 28.6 |
| Test molecules with an *identical* training fingerprint | **0.3%** | **38.6%** |
| Test molecules with a training analog ≥ 0.7 Tanimoto | 51% | 83% |

**Random splits on this dataset are invalid**: nearly 40% of test molecules have a structurally
identical training molecule, which is why random-split AUROC reaches 0.95 against 0.87 under
scaffold splits. Scaffold splits are sound but are not a novelty test — half their test molecules
still have a close analog in training.

## 4. Qwen versus traditional ML

All models on identical scaffold splits of the same 631 molecules.

![model comparison](../results/figures/model_comparison.png)

| model | AUROC | AUPRC | MCC | ECE | Brier | log loss |
|---|---|---|---|---|---|---|
| trivial prior | 0.500 | 0.569 | 0.000 | 0.155 | 0.262 | — |
| logistic regression | 0.863 ± 0.082 | 0.888 | 0.620 | 0.157 | 0.135 | 0.479 |
| random forest (default) | 0.872 ± 0.066 | 0.884 | 0.640 | 0.158 | 0.131 | 0.423 |
| XGBoost | 0.880 ± 0.066 | 0.900 | 0.639 | 0.148 | 0.128 | 0.451 |
| 3-model ensemble | 0.878 ± 0.063 | 0.896 | 0.665 | 0.148 | **0.123** | **0.411** |
| **random forest (tuned per seed)** | **0.898 ± 0.057** | **0.910** | **0.685** | 0.180 | 0.125 | 0.407 |
| Qwen zero-shot Jev readout | 0.651 ± 0.144 | 0.718 | 0.192 | 0.158 | 0.229 | 0.650 |
| Qwen QLoRA fine-tune | 0.850 ± 0.058 | 0.896 | 0.493 | 0.206 | 0.182 | 0.569 |

**Tuned random forest on 2048-bit Morgan fingerprints is the best model, at AUROC 0.898.**
Tuning significantly improved RF (+0.027, p = 0.019) and did nothing for the other two.

### Did the fine-tuned LLM beat it? No — it is significantly worse.

Paired over the same 10 scaffold seeds (`scripts/compare.py`):

| comparison | ΔAUROC | 95% CI | seeds better | p |
|---|---|---|---|---|
| **QLoRA vs tuned RF** | **−0.049** | [−0.076, −0.021] | 2/10 | **0.007** |
| QLoRA vs tuned RF, MCC | −0.193 | [−0.299, −0.086] | 2/10 | 0.006 |
| QLoRA vs tuned RF, Brier | −0.058 | [−0.083, −0.032] | 0/10 | 0.002 |
| **QLoRA vs Qwen zero-shot** | **+0.199** | [+0.122, +0.277] | **10/10** | **0.001** |

So the direction is unambiguous in both places. Fine-tuning works — it adds 0.199 AUROC over the
zero-shot readout, on every single seed — and the fine-tuned model still loses to a tuned random
forest on fingerprints, on 8 of 10 seeds. On 631 molecules, the 9B LLM does not reach the baseline.

Other established results:
- Ensembling gives significantly better log loss than XGBoost (p = 0.036) and logistic regression
  (p = 0.013), but **no** significant AUROC gain over any member.
- Qwen's zero-shot readout is clearly worse than every fingerprint model overall.

**One caveat on the fine-tune.** Validation selected the final epoch in 5 of 10 seeds, so the
3-epoch schedule truncated training for half the runs and 0.850 is a *lower bound*. A longer
schedule is the first item in §9; it would have to close a 0.049 gap to change the conclusion.

## 5. Calibration and distribution shift

Post-hoc calibration was expected to be the easy win. It is not available.

**No calibration method significantly improved ECE, Brier or log loss on any model. Several made
them significantly worse** (Platt on RF: ΔECE +0.046, p = 0.026).

The cause is measurable — the fitted temperature is unidentifiable:

| model | fitted temperature across seeds (1.0 = already calibrated) |
|---|---|
| random forest | 1.54 ± 0.94 (range 0.40 – 3.74) |
| Qwen zero-shot | **31.2 ± 61.8** (range 0.34 – **148**) |

A **random-split control** confirms the mechanism: there, temperature scaling is neutral
(ΔECE −0.004) and the fitted temperature sits near 1.0. So the miscalibration is not a broken output
layer, which is what temperature scaling repairs — it is **chemotype shift**. A model facing an
unfamiliar scaffold is wrong about its own confidence in a way a *different* unfamiliar scaffold
cannot predict.

### The one exception: the fine-tuned LLM

The QLoRA model is the single case where post-hoc calibration works, and it works clearly:

| calibration | ECE | Brier | log loss | paired change |
|---|---|---|---|---|
| none | 0.206 | 0.182 | 0.569 | — |
| **temperature** | 0.177 | 0.174 | 0.527 | ΔECE **−0.029 (p=0.010)**, Δlog loss **−0.042 (p=0.007)** |
| isotonic | **0.139** | 0.168 | 0.871 | ΔECE **−0.067 (p<0.001)**, but log loss worsens |

The fine-tuned model starts as the *worst-calibrated* model in the study (ECE 0.206) and is
**systematically overconfident in the same direction on every fold** — fitted temperature 1.57 ±
0.77, never wildly unstable like the zero-shot readout's 31 ± 62. That consistency is what makes the
correction transfer. The fingerprint models, by contrast, are miscalibrated in *different directions*
on different folds, so a single scalar fitted on one fold actively harms another.

Even after calibration, QLoRA's ECE (0.177) remains worse than the uncalibrated random forest's
(0.158). Detail: [`calibration.md`](calibration.md).

## 6. Generalization and uncertainty

### Where prediction stops working

![similarity vs AUROC](../results/figures/similarity_vs_auroc.png)

| max Tanimoto to training set | n | RF AUROC |
|---|---|---|
| 0.0 – 0.4 | 21 | **0.450** (chance) |
| 0.4 – 0.5 | 30 | 0.801 |
| 0.5 – 0.6 | 101 | 0.903 |
| 0.6 – 0.7 | 151 | 0.931 |
| 0.7 – 1.0 | 319 | 0.832 |

**Below 0.4 Tanimoto the fingerprint models rank at chance.** The headline 0.898 is earned on
molecules with a reasonably close training analog.

On the **strict novelty subset** (42 molecules with no training analog above 0.5 Tanimoto), scored
one row per molecule with a cluster bootstrap that resamples molecules rather than predictions:

| model | novel only | all test | 95% CI on the novel column |
|---|---|---|---|
| **random forest** | **0.782** | 0.892 | [0.569, 0.959] |
| Qwen zero-shot | 0.764 | 0.663 | [0.556, 0.931] |
| XGBoost | 0.694 | 0.894 | [0.430, 0.925] |
| Qwen QLoRA | **0.396** | 0.808 | [0.062, 0.745] |

An earlier version of this report put zero-shot Qwen first here at 0.807, ahead of the forest. That
was an artefact: those numbers pooled 51 predictions over 10 *overlapping* test folds, and the
zero-shot score for a molecule is cached and identical in every fold, so pooling reweighted its
molecules by fold membership. Molecule for molecule the zero-shot readout is **0.019 behind** the
random forest (95% CI [−0.257, +0.192]), and it leads only at novelty cutoffs where the subset holds
one or three positive molecules. **The claim that the base LLM generalizes better to unfamiliar
chemistry than fingerprints do is withdrawn.**

What survives is the other half: **the fine-tuned model is far below the base model it started
from** (−0.368, 95% CI [+0.013, +0.658] on the difference), which motivated the next experiment.

### Does fine-tuning overwrite the base model's chemical prior?

![forgetting](../results/figures/forgetting.png)

The LLM-as-Jev objective includes KL anchors to a frozen reference model, omitted from our first
implementation. We implemented all three (`src/models/jev_anchors.py`, paper §3.4) and reran 10
seeds at λ = 0 and λ = 1, logging predictions after every epoch including epoch 0 — the adapter is
reset to B = 0 per seed, so each run carries its own matched base-model control on its own molecules.

**Training is now reproducible.** The first version of this comparison was not measurable: rerunning
the identical λ = 0 configuration moved the headline AUROC by −0.028 ± 0.089, as much as the effect
being measured. The cause was nondeterministic backward-pass kernel selection, not the quantized
optimizer ([`reproducibility.md`](reproducibility.md)). With deterministic algorithms enabled, three
full pipeline runs on one seed are **bit-exact** — identical epoch curves, selected epoch, saved LoRA
weights and metrics — at about 10% wall-clock cost. The numbers below are the rerun under that
regime and supersede the 2026-10-28 ones.

| epochs | overall AUROC (λ=0) | novel AUROC (λ=0) | overall AUROC (λ=1) | novel AUROC (λ=1) |
|---|---|---|---|---|
| 0 (base model) | 0.660 | 0.690 | 0.660 | 0.690 |
| 1 | 0.814 | 0.715 | 0.841 | 0.567 |
| 2 | 0.851 | 0.736 | 0.857 | 0.812 |
| 3 | 0.819 | 0.553 | 0.872 | 0.708 |
| validation-selected | 0.818 | 0.722 | **0.856** | **0.757** |

**KL regularization improved overall AUROC by 0.038, and the improvement is not statistically
significant.** At the validation-selected epoch λ = 1 scores 0.856 ± 0.093 against λ = 0's
0.818 ± 0.072, better on 8 of 10 seeds, paired t-test p = 0.127. Brier improves 0.183 → 0.173; ECE is
unchanged at 0.094.

**The novel-chemistry improvement is inconclusive.** +0.035 at the selected epoch with a 95%
bootstrap interval of [−0.107, +0.225]. Per epoch the sign flips: −0.148 at epoch 1 (interval
[−0.286, −0.038], excluding zero *against* the anchors), +0.076 at epoch 2, +0.155 at epoch 3 with an
interval four times wider. An effect whose direction depends on where training stops is not an
effect.

**We have not established that KL anchors prevent catastrophic forgetting.** We have not established
the forgetting either. The monotone epoch-by-epoch decline reported on 2026-10-28 did not reproduce
under deterministic training — λ = 0's novel AUROC rises for two epochs before falling (0.690 → 0.715
→ 0.736 → 0.553), and the endpoint loss of −0.137 has a 95% interval of [−0.522, +0.176]. That earlier
monotonicity was a trend read out of nondeterministic noise, and it is withdrawn.

**Random forest remains the stronger baseline.** 0.872 ± 0.066 against λ = 1's 0.856 ± 0.093
(λ = 1 better on 4 of 10 seeds, Δ −0.016 ± 0.073). The anchors close roughly two thirds of the gap
without crossing it, and both QLoRA conditions remain worse on Brier.

KL experiments are **stopped**. The limiting factor is no longer measurement noise but the
evaluation set: 42 novel molecules in 23 chemical clusters containing 6 positives cannot resolve a
0.035 difference, and no additional training run changes that. Detail in
[`forgetting.md`](forgetting.md).

### Conformal prediction is the method that survives shift

![conformal](../results/figures/conformal.png)

| split | target coverage | measured | mean set size | decisive answers |
|---|---|---|---|---|
| random, α=0.1 | 0.90 | 0.906 | 1.05 | 91% |
| **scaffold, α=0.1** | 0.90 | **0.922** | **1.32** | **66%** |

Coverage holds under scaffold shift, and the method pays for it by returning wider sets: on a third
of unfamiliar molecules it answers {potent, not potent}, which is honest. **Caveat:** split conformal
guarantees coverage under exchangeability, which a scaffold split deliberately violates. The result
above is empirical over 10 splits, not a theorem.

### Selective prediction

![risk coverage](../results/figures/risk_coverage.png)

| coverage kept | by confidence | by similarity | random |
|---|---|---|---|
| 100% | 0.155 | 0.155 | 0.155 |
| 40% | **0.065** | 0.150 | 0.133 |
| 20% | **0.056** | 0.136 | 0.143 |

Abstaining on low confidence cuts error from 15.5% to 6.5%. **Abstaining on low similarity barely
beats abstaining at random** — because dissimilar molecules are mostly negatives, so the model gets
them right while ranking them badly. Error rate and AUROC fail differently. The operational rule:
**confidence decides whether to trust a prediction; similarity decides whether to make one.**
Detail: [`uncertainty.md`](uncertainty.md).

## 7. Candidate prioritization

> **Everything below is a computational hypothesis about an untested molecule. Nothing has been
> synthesized, assayed, or validated, and nothing here is a clinical recommendation.**

**Library.** ChEMBL approved drugs (max_phase 4): 4,225 records → 2,415 standardized unique small
molecules → 2,414 after removing any with an existing TG2 measurement. Approved drugs were chosen so
that a prediction would be an actionable repurposing hypothesis rather than a measure of
extrapolation.

**Method.** Three-model ensemble trained on 569 TG2 molecules, conformal-calibrated on a held-out 62;
each candidate annotated with ensemble probability, model agreement, prediction spread, conformal
label set, nearest-neighbour Tanimoto to the training set and to known actives, QED and physical
properties, PAINS/Brenk alerts, and a covalent-warhead SMARTS panel; shortlist diversified by Butina
clustering.

**Result — the applicability domain disqualifies the library:**

| max Tanimoto to training set | approved drugs | share |
|---|---|---|
| ≥ 0.50 (reliable) | 2 | 0.1% |
| 0.40 – 0.50 (marginal) | 17 | 0.7% |
| **< 0.40 (chance-level)** | **2,395** | **99.2%** |

Nominally 35 drugs score p ≥ 0.5, but almost all are in the chance-level band and are **not**
candidates. The TG2 training set is dominated by covalent peptidomimetics — acrylamides,
chloroacetamides, 3-halo-dihydroisoxazoles — a chemotype essentially absent from approved drugs.

**Tier 1 (passes every criterion): 0 compounds. Tier 2: 1 compound.**

| ID | Name | p(potent) | Agreement | Tanimoto to train | QED | Alerts | Warhead | Why prioritized / why doubt it |
|---|---|---|---|---|---|---|---|---|
| CHEMBL231813 | Telaprevir | 0.856 | 3/3 | 0.449 | 0.23 | 1 (Brenk) | α-ketoamide | Covalent peptidomimetic protease inhibitor; TG2 is a cysteine transferase whose potent inhibitors are covalent, so the mechanism is coherent. **But** 0.449 is only the marginal band, the model may be recognizing the peptidomimetic chemotype rather than TG2 affinity, and telaprevir was withdrawn in several markets for severe skin reactions. Best read as a probe of model bias, not a drug candidate. |

Nirmatrelvir (p = 0.655, 3/3 models, nitrile warhead) is the nearest miss, excluded at Tanimoto
0.333. Detail and full table: [`candidates.md`](candidates.md).

**The honest conclusion is that approved-drug repurposing is the wrong strategy for TG2 with this
training set.** Reporting the 35 nominal hits as candidates would have been easy and wrong.

## 7b. Selectivity against Factor XIII-A

Celiac disease would mean chronic dosing in otherwise healthy people, so selectivity within the
transglutaminase family is a safety gate rather than an optimization. FXIII-A crosslinks fibrin and
its inhibition causes bleeding. Full method and tables in [`selectivity.md`](selectivity.md).

**The family data cannot expand our chemical coverage.** TGM1, TGM3 and FXIII-A add 5, 1 and 31
molecules respectively that the TG2 set does not already contain — they are selectivity panels run
on compounds we have, and 51 of TGM1's 53 Murcko scaffolds are already present. The guinea-pig TG2
ortholog is the only genuinely distant chemistry available (41 molecules, 32 below 0.5 Tanimoto,
only 2 of 18 scaffolds shared), and every one of its compounds is inactive (IC50 2.1–74 uM). The
six-positive novel stratum that limits §6 therefore cannot be grown from these sources.

**Most known TG2 inhibitors are not selective.** 165 molecules carry both a TG2 and an FXIII-A
measurement; 103 of them inhibit both at 1 uM and only 29 are TG2-potent and FXIII-A-clean. That is
a measured property of the published chemical space, and a caution about the whole covalent TG2
literature.

**We cannot predict which ones are selective.**

| task | molecules | positives | AUROC | 95% CI |
|---|---|---|---|---|
| TG2 potency | 432 | 234 | 0.866 | [0.828, 0.900] |
| FXIII-A potency | 118 | 79 | 0.948 | [0.892, 0.989] |
| selectivity, scaffold split | 71 | 17 | 0.974 | [0.917, 1.000] |
| **selectivity, whole papers held out** | 15 | 9 | **0.574** | **[0.250, 0.864]** |
| selectivity ranked by predicted TG2 potency | 134 | 20 | 0.454 | [0.329, 0.581] |

The 0.974 is an artefact and is reported here only because catching it is the result. Per-seed
AUROC of 0.998 ± 0.006 on 17 positives is not a plausible measurement; 11 of the 29 selective
molecules come from a single paper, and only 3 of the 98 scaffolds in the paired set appear in both
classes. Scaffold splitting — correct everywhere else in this report — cannot separate papers when
one lab's series spans several scaffolds, so the model was free to classify provenance. Holding out
whole documents collapses it by 0.40, onto an interval that straddles chance in both directions.
Adding the FXIII-A model's own prediction as a feature changed nothing (0.901 to 0.911, and that is
an upper bound, since the feature was fitted on the same molecules).

**Consequence: no selectivity-aware shortlist.** Phase 4 of this work was conditional on the models
supporting it, and they do not. A ranking by predicted TG2 potency carries no selectivity
information at all (0.454), so attaching a selectivity column to a candidate list would misstate
what was measured.

**One asset remains unused.** 286 TG2 molecules carry a kinact/Ki value — comparable in size to the
302 potent molecules the project trains on — and 238 come from assays explicitly describing
irreversible inhibition. kinact/Ki cannot be merged into an IC50 label, but it is the natural basis
for a covalent-mechanism model and is the largest piece of unexploited data in this dataset.

## 8. Limitations

1. **631 molecules** remains small for a classification task with a 57% positive rate.
2. **Scaffold splits are not novelty splits** — half of test molecules have a close training analog.
3. **No stereochemistry** in the source SMILES; enantiomers are collapsed.
4. **The novel-chemistry Qwen advantage rests on 42 molecules**, pooled across overlapping folds.
5. **Conformal exchangeability is violated by design**; coverage held empirically but is not
   guaranteed.
6. **Replicate disagreement**: 14 molecules have IC50s differing by more than 10× between reports.
7. **One target, one assay family.** No selectivity data against FXIII-A or other transglutaminases,
   so nothing here speaks to selectivity.
8. **Potency ≠ a drug.** No ADME, no permeability, no toxicity modelling, no gut-restriction
   analysis — all decisive for a celiac therapeutic, which should ideally act in the intestinal lumen.
9. p-values are uncorrected across many comparisons.

## 9. Next experiments

The KL-anchor line is closed (§6): reproducible, small, non-significant, and limited by the
evaluation set rather than by GPU time. Priorities below are ordered by what would most change our
ability to identify genuinely promising TG2 inhibitors.

1. **Grow the low-similarity stratum — the binding constraint on everything.** Six positive molecules
   below 0.5 Tanimoto is why every generalization interval in this report is uninformative. Related
   transglutaminases in ChEMBL (TGM1, TGM3, FXIII-A) would add chemically distant actives at the cost
   of mixing targets; a multi-task model with a target indicator is the principled version. CPU-cheap
   and it makes every later experiment testable.
2. **Screen a chemically appropriate library instead of approved drugs.** 99.2% of approved drugs fall
   below the applicability floor (§7), so that avenue is exhausted. Analogs within 0.5–0.7 Tanimoto of
   known actives, and covalent warhead fragment libraries — TG2 is a cysteine transglutaminase and its
   established inhibitors are largely covalent — are where a hit could plausibly come from. This is a
   change of direction and needs agreement first.
3. **Model the mechanism, not just the label.** Separate reversible from irreversible inhibitors, and
   warhead chemotype from scaffold. A model that cannot tell a Michael acceptor from a benzimidazole
   is ranking the wrong feature.
4. **Selectivity modelling** against FXIII-A and TG1/TG3, where ChEMBL has data. A TG2 inhibitor that
   hits FXIII-A is a bleeding risk, so selectivity is a gating criterion for any shortlist, not a
   refinement.
5. **Extend the QLoRA schedule** if the LLM line continues. Validation chose the final epoch in 5 of
   10 seeds, so 0.856 is a lower bound. Six epochs with early stopping would settle whether the
   remaining gap to the random forest is real. Lower priority: the LLM has not beaten fingerprints at
   this data scale in any configuration tried.
6. **Conformal with a Mondrian split by scaffold cluster**, to give per-chemotype coverage.

## 10. What would need laboratory collaboration

Nothing in this project can be validated computationally. Specifically:

- **A TG2 inhibition assay** on a handful of predicted compounds — the standard colorimetric or
  fluorescent transamidation assay with a gluten-derived peptide substrate. Even 10–20 compounds
  would provide the first genuinely external validation.
- **A prospective test**, where predictions are recorded before assay. Everything here is
  retrospective.
- **Counter-screens** against FXIII-A for selectivity, and a cysteine-reactivity control to
  distinguish genuine TG2 affinity from nonspecific covalent reactivity — essential given that the
  model's strongest signal is warhead chemistry.
- **Access to unpublished or in-house IC50 data**, which would be the fastest way past the
  631-molecule ceiling.

A realistic ask: a small assay of compounds chosen by the model from an appropriate library, with a
negative control set matched on warhead chemistry.

## References

1. Wichka I, Lai P-K. Rapid discovery of transglutaminase 2 inhibitors for celiac disease with
   boosting ensemble machine learning. *Comput Struct Biotechnol J* 2024;23:3669–3679.
   doi:10.1016/j.csbj.2024.10.019
2. Zheng S, et al. ChEMBL: a drug discovery database. Target CHEMBL2730 (human TGM2, UniProt P21980).
3. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. *ICML* 2017.
4. Vovk V, Gammerman A, Shafer G. *Algorithmic Learning in a Random World.* Springer, 2005.
   (inductive and Mondrian conformal prediction)
5. Bemis GW, Murcko MA. The properties of known drugs. 1. Molecular frameworks. *J Med Chem*
   1996;39:2887–2893. (scaffold splitting)
6. Sheridan RP. Time-split cross-validation as a method for estimating the goodness of prospective
   prediction. *J Chem Inf Model* 2013;53:783–790. (why random splits mislead)
7. Baell JB, Holloway GA. New substructure filters for removal of pan assay interference compounds.
   *J Med Chem* 2010;53:2719–2740. (PAINS)
8. Brenk R, et al. Lessons learnt from assembling screening libraries. *ChemMedChem* 2008;3:435–444.
9. Li & Wagle. LLM-as-Jev. arXiv:2610.02076. (option-logit readout and LoRA objective used here)
10. Butina D. Unsupervised data base clustering based on Daylight's fingerprint and Tanimoto
    similarity. *J Chem Inf Comput Sci* 1999;39:747–750.

*Reference details should be verified against the originals before any external submission; items
2, 9 and the Jev literature in [`jev_research.md`](jev_research.md) are preprints or database
records rather than peer-reviewed papers.*
