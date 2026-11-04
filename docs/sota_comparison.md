# Do specialized molecular AI models beat our baselines on TG2? (run 2026-11-03)

Reproduce: `bash scripts/setup_sota_envs.sh`, then `python scripts/export_splits.py`,
`bash scripts/run_sota_all.sh all`, then `python scripts/analyse_sota.py --split <arm> --figures`
for each of `scaffold`, `scaffold_pub`, `paper_lopo`.
Raw folds: `results/sota_tg2.csv` (510 rows). Tables: `results/sota_{summary,paired,bands}_<arm>.csv`.
Figures: `results/figures/sota_{comparison,calibration,generalization}_<arm>.png`.
Protocol and literature review: [`state_of_the_art_comparison.md`](state_of_the_art_comparison.md),
including the addendum recording every deviation from the approved plan.
Total compute: 6 h 30 m for the nine-entrant sweep plus 6 m for MoLFormer, one RTX 5070 Ti, serial.

**The answer is no.** Ten entrants, three splitting protocols, 510 model fits. On the primary
endpoint not one specialized molecular model beat the tuned random forest on Morgan fingerprints,
and the three that came closest are a 4-configuration logistic regression, XGBoost, and a
3.4M-parameter transformer. The pretrained graph and SMILES models — the ones the field has moved
to — were **significantly worse**. The 9B Qwen QLoRA is also worse, confirming the paper's finding
against a much wider field than it tested.

---

## 1. The headline table

Scaffold split, 631 molecules, 20 seeds, mean ± SD across folds. Incumbents at 10 seeds, as
published.

| model | AUROC | AUPRC | MCC | ECE | Brier |
|---|---|---|---|---|---|
| **Random forest (Morgan, tuned)** | **0.864 ± 0.077** | 0.874 ± 0.083 | 0.593 ± 0.219 | 0.187 | **0.144** |
| XGBoost (Morgan) | 0.853 ± 0.083 | 0.871 ± 0.090 | **0.609 ± 0.199** | **0.158** | 0.140 |
| Logistic regression (Morgan) | 0.849 ± 0.091 | 0.863 ± 0.103 | 0.545 ± 0.228 | 0.193 | 0.156 |
| Qwen3.5-9B QLoRA *(10 seeds)* | 0.850 ± 0.058 | 0.896 ± 0.041 | 0.493 ± 0.114 | 0.206 | 0.182 |
| ChemBERTa-2 (fine-tuned) | 0.831 ± 0.104 | 0.850 ± 0.112 | 0.558 ± 0.202 | 0.166 | 0.158 |
| Tabular foundation model | 0.829 ± 0.093 | 0.848 ± 0.114 | 0.550 ± 0.231 | 0.164 | 0.154 |
| CheMeleon (pretrained D-MPNN) | 0.802 ± 0.098 | 0.829 ± 0.086 | 0.454 ± 0.254 | 0.193 | 0.180 |
| Chemprop v2 + RDKit 2D | 0.796 ± 0.104 | 0.806 ± 0.107 | 0.488 ± 0.194 | 0.177 | 0.171 |
| ChemBERTa-2 (frozen) | 0.793 ± 0.097 | 0.814 ± 0.102 | 0.489 ± 0.182 | 0.180 | 0.176 |
| MoLFormer-XL (frozen) | 0.760 ± 0.111 | 0.775 ± 0.145 | 0.425 ± 0.195 | 0.214 | 0.206 |
| Chemprop v2 D-MPNN | 0.749 ± 0.123 | 0.752 ± 0.124 | 0.357 ± 0.240 | 0.224 | 0.212 |
| Qwen3.5-9B zero-shot *(10 seeds)* | 0.648 ± 0.123 | 0.728 ± 0.123 | 0.091 ± 0.129 | 0.148 | 0.236 |

Paired per-seed against the tuned random forest, positive meaning the entrant is better. `p_holm` is
Holm-corrected across the ten comparisons on the primary endpoint; the equivalence margin set in
advance was 0.05 AUROC.

| model | ΔAUROC | 95% CI | wins | p_holm | verdict |
|---|---|---|---|---|---|
| XGBoost | −0.010 | [−0.023, +0.002] | 8/20 | 0.202 | equivalent |
| Logistic regression | −0.015 | [−0.032, +0.002] | 6/20 | 0.202 | equivalent |
| ChemBERTa-2 fine-tuned | −0.033 | [−0.056, −0.010] | 4/20 | **0.045** | worse |
| Tabular foundation model | −0.034 | [−0.061, −0.007] | 6/20 | 0.069 | worse, uncorrected |
| Qwen3.5-9B QLoRA | −0.049 | [−0.076, −0.021] | 2/10 | **0.034** | worse |
| CheMeleon | −0.062 | [−0.093, −0.030] | 3/20 | **0.008** | worse |
| Chemprop + RDKit 2D | −0.068 | [−0.106, −0.030] | 4/20 | **0.014** | worse |
| ChemBERTa-2 frozen | −0.071 | [−0.104, −0.037] | 4/20 | **0.005** | worse |
| MoLFormer-XL frozen | −0.104 | [−0.138, −0.070] | 1/20 | **0.000** | worse |
| Chemprop D-MPNN | −0.115 | [−0.154, −0.075] | 1/20 | **0.000** | worse |
| Qwen3.5-9B zero-shot | −0.250 | [−0.321, −0.179] | 0/10 | **0.001** | worse |

Two entrants are statistically indistinguishable from the random forest and both are
fingerprint models. Every learned-representation model is below it, and six of the seven are below
it by more than the pre-registered 0.05 equivalence margin with the CI excluding zero.

**On this dataset, the specialized molecular AI stack does not beat a 2001-vintage random forest on
a 2010-vintage fingerprint.** The ranking is not close and it is not within noise.

---

## 2. The reproduction gate, and a finding about the published variance

The baselines reproduce **exactly** on the ten seeds the paper published: random forest
0.8983 ± 0.0565 against the committed 0.898 ± 0.057, maximum absolute per-seed difference 0 across
AUROC, AUPRC, MCC, ECE, Brier and log loss. Logistic regression is likewise exact; XGBoost agrees to
3.8e-8, from float32 leaf sums. Pinned in `tests/test_sota_reproduction.py`.

Extending to 20 seeds moved the random forest from 0.898 ± 0.057 to **0.864 ± 0.077**, and this is
worth stating plainly because it affects a published number:

| | seeds 0–9 | seeds 10–19 |
|---|---|---|
| Random forest | 0.898 ± 0.057 | 0.829 ± 0.082 |
| XGBoost | 0.882 ± 0.065 | 0.825 ± 0.093 |
| Logistic regression | 0.873 ± 0.081 | 0.825 ± 0.098 |
| ChemBERTa-2 fine-tuned | 0.854 ± 0.081 | 0.808 ± 0.122 |
| Chemprop | 0.797 ± 0.100 | 0.701 ± 0.129 |

Seeds 10–19 are harder for **every** model, by 0.046 to 0.096. This is not a bug and not a
discrepancy: the first ten scaffold draws happened to be an easy half, and the published
± 0.057 understates the true spread across scaffold splits. The ranking is unchanged, so no
conclusion in the paper is overturned — but a 10-seed interval on this dataset is optimistic, and
future work should use 20.

---

## 3. The splitting effect, isolated

The three arms differ only in how molecules are divided. `scaffold_pub` and `paper_lopo` cover the
**same 444 molecules**, so between those two the splitting strategy is the only variable.

| | scaffold (631) | scaffold_pub (444) | paper_lopo (444) |
|---|---|---|---|
| Random forest | 0.864 ± 0.077 | 0.854 ± 0.094 | **0.612 ± 0.178** |
| Logistic regression | 0.849 ± 0.091 | 0.863 ± 0.084 | 0.624 ± 0.225 |
| XGBoost | 0.853 ± 0.083 | 0.855 ± 0.076 | 0.573 ± 0.213 |
| Tabular foundation model | 0.829 ± 0.093 | 0.854 ± 0.092 | 0.653 ± 0.204 |
| ChemBERTa-2 fine-tuned | 0.831 ± 0.104 | 0.823 ± 0.116 | 0.675 ± 0.228 |
| ChemBERTa-2 frozen | 0.793 ± 0.097 | 0.808 ± 0.090 | **0.701 ± 0.172** |
| MoLFormer-XL frozen | 0.760 ± 0.111 | 0.788 ± 0.131 | 0.604 ± 0.250 |
| CheMeleon | 0.802 ± 0.098 | 0.805 ± 0.104 | 0.635 ± 0.236 |
| Chemprop + RDKit 2D | 0.796 ± 0.104 | 0.778 ± 0.116 | 0.681 ± 0.150 |
| Chemprop | 0.749 ± 0.123 | 0.705 ± 0.135 | 0.542 ± 0.263 |

**This is the most important result in the study.** Holding the molecules fixed and changing only the
split, every model loses 0.10 to 0.25 AUROC, and the random forest loses the most (−0.242). At
0.612 ± 0.178 it is barely better than guessing and its interval includes 0.5.

And the model ranking **inverts**. On `paper_lopo` the leader is frozen ChemBERTa-2 at 0.701 ± 0.172
and the random forest is seventh of ten. But not one of the ten paired comparisons survives
correction — every `p_holm` is 1.000, and the best CI, ChemBERTa-2 frozen at +0.089, runs
[−0.027, +0.205]. With 11 folds and SDs of 0.15 to 0.26 this arm cannot resolve a 0.09 difference.

The honest reading: **when a model must predict a new laboratory's chemistry, nothing here works
reliably and the study cannot say which approach is least bad.** The scaffold-split numbers — 0.86
for the random forest, and the 0.898 in the paper — describe interpolation within a chemical series
already represented in training. They are not estimates of what any of these models would do on a
genuinely new compound class.

`docs/generalization.md` reached the same conclusion for the random forest alone. This extends it to
the entire modern stack, including the three models whose published selling point is generalization.

---

## 4. Generalization to unfamiliar chemistry

AUROC by maximum Tanimoto similarity to the training fold, one row per molecule (pooling per
prediction was the error documented in `docs/forgetting.md`).

Scaffold arm:

| model | <0.4 (n=25) | 0.4–0.5 (33) | 0.5–0.6 (105) | 0.6–0.7 (155) | ≥0.7 (298) |
|---|---|---|---|---|---|
| Random forest | 0.370 | 0.720 | 0.873 | 0.886 | 0.861 |
| XGBoost | 0.478 | 0.525 | 0.793 | 0.915 | 0.861 |
| Logistic regression | 0.239 | 0.455 | 0.778 | 0.897 | 0.845 |
| Tabular foundation model | **0.804** | 0.670 | 0.805 | 0.923 | 0.852 |
| ChemBERTa-2 fine-tuned | 0.370 | 0.650 | 0.620 | 0.905 | 0.874 |
| ChemBERTa-2 frozen | 0.174 | 0.750 | 0.598 | 0.834 | 0.841 |
| MoLFormer-XL frozen | 0.196 | 0.625 | 0.787 | 0.844 | 0.794 |
| CheMeleon | 0.478 | 0.685 | 0.686 | 0.858 | 0.837 |
| Chemprop + RDKit 2D | 0.543 | 0.645 | 0.625 | 0.871 | 0.813 |
| Chemprop | 0.304 | 0.525 | 0.753 | 0.738 | 0.709 |

Below Tanimoto 0.4, **eight of ten models score below 0.5** — worse than a coin flip. The random forest
manages 0.370, logistic regression 0.239, frozen ChemBERTa-2 0.174. Only the tabular foundation
model stays up at 0.804, and on 25 molecules that is one well-placed fold, not a finding; its
`paper_lopo` ≥0.7 band of 0.238 argues it is noise rather than a real advantage.

The `paper_lopo` arm puts **283 of 444 molecules (64%)** in the <0.4 band, against 25 of 631 (4%) on
the scaffold split. That is the whole explanation for the collapse in §3: leave-one-publication-out
is not a harder version of the same task, it is a task where most test molecules are chemically
unlike anything in training. In that band the ten models span 0.448 to 0.753 with no separation.

**This is the regime that matters for drug discovery.** Screening a new chemical series is by
definition prediction below Tanimoto 0.4. Every model here is unreliable there, the pretrained ones
included.

---

## 5. Calibration

No entrant is usefully calibrated. On the scaffold arm ECE runs 0.158 to 0.224 — a model saying 80%
is right 60% to 95% of the time. The random forest's Brier of 0.144 is the best in the study and
still only modestly better than the 0.254 of a constant predictor at the class prior.

Paired Brier against the random forest (exploratory, uncorrected): **nothing improves on it.**
XGBoost ties (+0.003, p=0.571); everything else is worse, from −0.010 for the tabular model to
−0.068 for Chemprop and −0.111 for Qwen zero-shot. ECE moves the other way for a few entrants
(XGBoost +0.030, ChemBERTa-2 fine-tuned +0.022), which is the usual tension: the models that rank
slightly worse sometimes bin slightly better. Neither effect survives as a reason to prefer them.

On `paper_lopo` calibration degrades to ECE 0.246–0.338 and Brier 0.214–0.313, with log losses up to
1.74. Several models are then *worse than the class prior*. Predicted probabilities from any of
these models on out-of-series chemistry should not be read as probabilities.

---

## 6. Cost

Per fold on one RTX 5070 Ti, scaffold arm, including the full tuning grid.

| model | s/fold | peak VRAM | parameters | configs |
|---|---|---|---|---|
| ChemBERTa-2 frozen | **2.6** | 195 MiB | 3,427,440 | 12 |
| MoLFormer-XL frozen | 5.8 | 576 MiB | 44,375,040 | 12 |
| XGBoost | 9.3 | — | — | 8 |
| Random forest | 14.6 | — | — | 12 |
| Logistic regression | 16.8 | — | — | 4 |
| Tabular foundation model | 25.0 | 1,026 MiB | — | 3 |
| ChemBERTa-2 fine-tuned | 47.9 | 482 MiB | 3,428,210 | 12 |
| Chemprop + RDKit 2D | 84.2 | 336 MiB | 1,061,101 | 12 |
| Chemprop | 95.0 | 335 MiB | 996,001 | 12 |
| CheMeleon | 169.3 | 1,382 MiB | 11,174,241 | 12 |
| *Qwen3.5-9B QLoRA (incumbent)* | *1,001* | *8.03 GiB* | *9B, LoRA-adapted* | *1* |

Every entrant fits the 16 GB GPU with room to spare; peak usage was 1.4 GiB, under 9% of the card.
Hardware was never the constraint — dataset size was. The Qwen3.5-9B QLoRA incumbent is the outlier:
1,001 s and 8.03 GiB per fold (mean over its 10 seeds, from `results/jev_qlora_tg2_combined.csv`),
a 19 GB checkpoint on disk. Frozen ChemBERTa-2 is **385× cheaper per fold** and scores 0.057 AUROC
lower; the tuned random forest is **69× cheaper**, needs no GPU at all, and scores 0.014 *higher*.

---

## 7. Leakage and provenance

`scripts/audit_sota_leakage.py`, output `results/sota_leakage.csv`. The audit separates three
distinct things, which the literature routinely conflates:

| | | |
|---|---|---|
| molecules provably in PubChem (carry a CID) | 398 / 631 | so in ChemBERTa-2's and MoLFormer's corpora **by construction** |
| in ChEMBL's TG2 target file | 460 / 631 | ChEMBL is not a pretraining corpus for any entrant here |
| auditable against the actual pretraining corpora | **0** | the 77M and 1.1B corpora are not downloadable; membership is inferred, not measured |

| model | corpus | saw structures | saw computed properties | saw experimental activity |
|---|---|---|---|---|
| logreg, rf, xgb, chemprop, chemprop_rdkit | none | no | no | no |
| tabular foundation model | synthetic tabular data | no | no | no |
| CheMeleon | computed molecular descriptors | likely | **yes** | no |
| ChemBERTa-2 (both) | 77M PubChem (MTR) | likely | **yes** | no |
| MoLFormer-XL | 1.1B PubChem + ZINC (MLM) | likely | no | no |
| Qwen3.5-9B (both) | general web text | likely, as text | n/a | **possible: the papers** |

**No entrant saw TG2 activity labels.** Structure membership is near-certain for the pretrained
models and property-prediction pretraining is documented for two of them, but neither is label
leakage: an MTR or descriptor objective teaches computed physicochemistry, not measured IC50.

The useful consequence is that leakage cannot be propping up this comparison, because **it points the
wrong way**. The models that almost certainly saw these 398 structures during pretraining are the
models that lost. Any structure-membership advantage they hold is already included in the numbers
above, and they still trail a random forest that saw nothing.

The one channel that *could* carry label information is the Qwen prompt, and the protocol's
assumption about it was wrong. `scripts/run_jev_qlora.py` appends `Compound name:` for **398 of 631
molecules (63.1%)**, not SMILES alone as [`state_of_the_art_comparison.md`](state_of_the_art_comparison.md)
asserted. The audit bounds the exposure: **384 of those names (96.5%) are systematic IUPAC strings**,
which redescribe the structure and carry nothing a model could not read off the SMILES; only **14
molecules (2.2% of the set)** carry a trivial or code-like name (Isatin, Indirubin, 5-Bromoisatin,
two bare `CID` strings, one registry hash), and **none is a recognised TG2 clinical compound
identifier**. Independently, `results/jev_probe_combined.log` records that adding names to the
zero-shot prompt moved AUROC 0.648 → 0.651. The channel is real, documented, and too small to
qualify the Qwen results.

**Now done, reported separately: the direct label-leakage probe.** Base Qwen3.5-9B was asked for
each molecule's IC50 as a number, in five arms. It abstains on 631 of 631 molecules when allowed to;
forced to answer it emits six stock values for 631 compounds, and rewriting the same molecules as
re-rooted SMILES *improves* every metric — which recall cannot do. No label leakage is detectable.
Full write-up and controls in [`qwen_label_probe.md`](qwen_label_probe.md).

---

## 8. How this compares with the published literature

| claim in the literature | what we found |
|---|---|
| Chemprop's D-MPNN beats fingerprint baselines on molecular property prediction (Yang 2019; Heid 2024) | **Not reproduced at n=631.** Chemprop is the weakest non-LLM entrant, −0.115 AUROC. Yang's own paper reports the advantage concentrating on datasets of thousands to hundreds of thousands of molecules, and notes fingerprint+descriptor baselines remain competitive below that. Our dataset is 631. |
| RDKit descriptors rescue D-MPNN on small data (Yang 2019) | **Partly.** +0.047 AUROC over graph-only Chemprop, consistent in direction and size with the published effect — but still −0.068 against the random forest. |
| CheMeleon's descriptor pretraining gives large gains on small datasets (2025) | **Not reproduced.** +0.053 over graph-only Chemprop, so the pretrained block does help the architecture, but at 0.802 it remains −0.062 against the random forest. |
| ChemBERTa-2's MTR pretraining transfers to downstream property tasks (Ahmad 2022) | **Partly.** Fine-tuning beats frozen embeddings by +0.038, the best learned-representation result at 0.831, and the closest any such model came. Still −0.033, Holm p=0.045. |
| MoLFormer-XL's 1.1B-molecule corpus yields superior representations (Ross 2022) | **Not reproduced.** At 0.760 it is the *worst* pretrained entrant, 0.033 below ChemBERTa-2 frozen despite 14× the pretraining corpus and 13× the parameters. |
| Tabular foundation models are strong on small tabular problems (TabPFN line) | **Partly.** Competitive at 0.829 and tied with the random forest on `scaffold_pub` (Δ −0.000), from 3 configurations and no task-specific training. The most interesting near-miss in the study. |
| Fingerprints plus gradient boosting remain hard to beat on small, single-target activity data | **Reproduced emphatically.** The top three on the primary endpoint are a random forest, XGBoost and logistic regression, all on the same 2048-bit Morgan fingerprint. |

The pattern — representation learning needing more data than a single-target ChEMBL extraction
provides, and fingerprint baselines holding up when properly tuned — is a known and repeatedly
reported result. This study is a clean, well-controlled instance of it on a therapeutically motivated
target, with the splitting protocol made an explicit variable rather than a footnote.

MoLFormer deserves its own line because it was the entrant added last, specifically to test whether
the frozen-SMILES-transformer deficit was a ChemBERTa checkpoint quirk. It was not: **two
independently pretrained SMILES transformers, 14× apart in corpus size, both land well below
fingerprints, and the larger one lands lower.** That is a statement about the representation class
on this task, not about one release.

---

## 9. What this means for the celiac disease goal

The research goal is identifying compounds that could help treat celiac disease by inhibiting TG2.
**No cure or validated treatment has been discovered here, and nothing in this comparison moves us
closer to one.** What it does is sharpen the measurement of how far away we are.

**Which approach could realistically support compound prioritization?** The tuned random forest on
Morgan fingerprints, which is what the project already uses. It is the most accurate entrant on the
primary endpoint, the best calibrated, costs 15 seconds per fold on a CPU, and nothing in the modern
stack displaced it. Adding Chemprop, CheMeleon, ChemBERTa-2 or MoLFormer to this pipeline would cost
compute and complexity and buy accuracy that is equal at best and significantly worse at worst. The
one entrant worth keeping in view is the tabular foundation model: equal to the random forest on the
publication subset from three configurations and no training, which is a different cost/accuracy
trade rather than a better one.

**Are predictions reliable enough to prioritize compounds for experimental testing?** For
*interpolation* within a series already in the training data, cautiously yes — 0.86 AUROC ranks
usefully, though calibration is too poor for the probabilities to be read as probabilities. For
anything genuinely new, **no**, and this comparison strengthens that answer in two independent ways:

1. **Leave-one-publication-out collapses every model to 0.54–0.70 AUROC**, the random forest's
   interval including 0.5. A new laboratory's chemistry is the operational definition of a screening
   campaign, and no model here handles it.
2. **Below Tanimoto 0.4, eight of ten models score worse than chance.** 64% of the publication-held-out
   molecules sit in that band.

So a virtual screen over a novel library would be ranking molecules in exactly the regime where
these models are measurably unreliable. That constrains the project to prioritizing analogues of
known series — useful, but a different and much more modest claim than screening for new chemistry.

**TG2 selectivity over FXIII-A remains the binding constraint, and no model here addresses it.**
Every entrant in this study predicts TG2 potency only. From `docs/selectivity.md` and
`results/fxiiia_evidence.csv`: of 147 molecules with paired TG2 and FXIII-A measurements, the median
selectivity is **1.4×** (median log selectivity 0.151), only **8 reach 100×**, and **10 are selective
the wrong way** — more potent against the anti-target. FXIII-A crosslinks fibrin and inhibiting it
causes bleeding, so for a drug healthy people would take chronically this is a safety gate, not a
refinement. A better TG2 potency model does not help here; the data needed is paired selectivity
data, and 147 molecules across 7 publications is not enough to train on.

**Limitations of the available data.** 631 molecules at a 1 µM threshold, 460 attributable to 17
publications with the largest seven holding 394 of them. Publication-level confounding is therefore
severe, which is precisely what §3 measures. Potency is not exposure: `docs/adme_analysis.md` records
that the project's selective compounds fail the intestinal-exposure axis that distinguishes ZED1227,
and nothing in this study predicts gut-restricted delivery, permeability or stability. Activity
labels are heterogeneous assays pooled across two decades.

**No compound is a viable treatment on the basis of anything in this document.** Predicted potency —
and for that matter predicted selectivity, which we cannot do reliably — is not efficacy, not
safety, and not delivery.

---

## 10. What to investigate next

Ranked by expected value for the original research goal, not by novelty.

1. ~~**The direct Qwen label-leakage probe.**~~ **Done** — no leakage detectable; the QLoRA result is
   not explained by retrieval. See [`qwen_label_probe.md`](qwen_label_probe.md).
2. **Stop benchmarking potency models and get paired selectivity data.** The limiting factor for a
   celiac TG2 inhibitor is TG2-over-FXIII-A, and 147 paired molecules is the entire public basis for
   it. Nothing in this comparison changes that, and no architecture will.
3. **Predict with an abstention rule instead of a point score.** Since every model fails below
   Tanimoto 0.4 and that is where screening happens, the useful engineering move is a model that
   declines to predict there. `docs/uncertainty.md` already has conformal machinery; measuring
   risk–coverage on the `paper_lopo` arm would turn "unreliable on new chemistry" into a quantified
   coverage budget.
4. **20 seeds as the project standard.** §2 shows the published 10-seed intervals are optimistic by
   roughly 0.03 AUROC in the mean and understate the SD. Cheap to adopt.
5. **A multi-task model across the transglutaminase family.** The one structural reason a graph or
   transformer model might eventually beat fingerprints here is sharing signal across TG2, TGM1,
   TGM3 and FXIII-A rather than fitting 631 molecules alone — and it would predict the selectivity
   ratio directly, which is the quantity that matters. Speculative, and the family data is thin
   (TGM1 92 molecules, TGM3 63), but it is the only route by which the models tested here would
   become the right tool.
6. **Not worth repeating:** Uni-Mol, left optional because the dataset has no reliable
   stereochemistry, and more SMILES transformers. Two of them, 14× apart in pretraining scale, both
   lost to fingerprints; a third would not change the conclusion.

---

## Verification

| test file | what it pins |
|---|---|
| `tests/test_sota_splits.py` (12 tests) | every exported fold re-derived from `src.data.chem` and compared row for row; no scaffold or publication crosses a fold; LOPO tests each molecule exactly once; both publication arms cover the same molecules; the export is deterministic |
| `tests/test_sota_reproduction.py` (7 tests) | the harness reproduces `results/tuned_baselines_tg2.csv` per seed; every entrant's `n_configs` recorded and capped; predictions cover the exported test fold and nothing else |

`scripts/analyse_sota.py` independently reproduces the paper's QLoRA-against-random-forest
comparison (ΔAUROC −0.049, 95% CI [−0.076, −0.021]), which validates the paired statistics against a
published result before any new model is compared with them.

No existing experiment, result file, dataset or document was modified. The new entrants write only
to `results/sota_*`; `requirements.txt` is unchanged, and the dependency conflicts
(scikit-learn 1.6.1 for the tabular model, a separate torch for Chemprop) are isolated in
`~/sota-envs/` so they cannot move a published number. Environment of record, model versions, seeds
and hardware are in the addendum to
[`state_of_the_art_comparison.md`](state_of_the_art_comparison.md).
