# Modern molecular property prediction, against what this project built

*Literature review and proposed protocol, written 2026-11-03. **Nothing here has been run.** No model
has been trained, no result in this repository has been changed, and no number below is a
measurement from this project unless it cites an existing document.*

The question this answers, as posed: how does our approach — a QLoRA-fine-tuned Qwen3.5-9B with an
LLM-as-Jev decision readout, and a tuned random forest on Morgan fingerprints — compare against the
methods the molecular-property-prediction field actually uses? And can we test that fairly on the
631-molecule TG2 set we already have?

Short version, before the detail. The literature says our headline finding is the field's finding:
on datasets of this size, pretrained molecular models mostly do not beat a fingerprint baseline. The
most recent large benchmark (25 models, 25 datasets, scaffold splits) found exactly one of 25
pretrained embedders significantly better than count-ECFP, and it is itself fingerprint-based. That
makes this comparison less of a horse race and more of a replication, which changes what we should
claim from it — and it means the project's likely contribution is not "which model wins" but
**"does the winner survive a publication-held-out split"**, a question the benchmarking literature
does not ask and our own data says matters more than the model choice.

---

## 1. Literature review

### 1.0 Where the field stands, and why that matters for us first

Two results should be read before any individual model, because they set what a positive result here
would even mean.

**Praski, Adamczyk & Czech (2025/2026), *Benchmarking Pretrained Molecular Embedding Models for
Molecular Representation Learning*, arXiv:2508.06199 (v4, Feb 2026).** 25 pretrained models against
25 classification datasets (7 MoleculeNet, 18 TDC; 475 to ~93,000 molecules), scaffold splits,
AUROC, frozen embeddings with random forest / logistic regression / kNN heads tuned identically for
every embedder, compared with a hierarchical Bayesian Bradley–Terry model. Conclusion: *nearly all
neural models show negligible or no improvement over the baseline ECFP fingerprint.* Only CLAMP was
significantly better (mean AUROC 82.55% against 79.89% for ECFP), and CLAMP is itself
fingerprint-based. R-MAT, MolBERT, ChemBERTa, CDDD, Atom Pair and MAT came out practically
equivalent to ECFP; everything else was worse, with GNN embeddings generally poor (GraphFP lowest at
59.67%). On 5 of 25 datasets nothing beat ECFP at all. The authors recommend that no molecular ML
study be considered complete without a count-ECFP + tree-model comparison.

This is the most relevant paper to our situation, and it is also the paper that most limits what we
can claim. Two caveats on using it: it tests **frozen embeddings only**, so it does not cover
supervised D-MPNN training or a fine-tuned transformer, and it is a preprint.

**van Tilborg, Alenicheva & Grisoni (2022), *Exposing the Limitations of Molecular Machine Learning
with Activity Cliffs*, J. Chem. Inf. Model. 62(23):5938–5951, doi 10.1021/acs.jcim.2c01073.** 24
methods across 30 ChEMBL bioactivity targets — the same kind of data as ours, potency on a single
protein. Descriptor-based machine learning outperformed the more complex deep models, and every
method struggled on activity cliffs. Benchmark and code: `github.com/molML/MoleculeACE` (MIT).
Relevance: our task *is* a MoleculeACE-shaped task. The paper's framing — that overall accuracy does
not predict cliff accuracy — is the same shape as our own similarity-band collapse in
[`generalization.md`](generalization.md).

**Li et al. (2026), *A Systematic Survey and Benchmark of Deep Learning for Molecular Property
Prediction in the Foundation Model Era*, arXiv:2604.16586, accepted *J. Chem. Theory Comput.* 2026.**
Surveys >100 methods across four paradigms (quantum, descriptor ML, geometric DL, foundation
models). Of direct use to us: it reviews data curation, splitting and evaluation protocols and flags
inconsistent stereochemistry, heterogeneous assay sources and unreliable random splits as field-wide
problems, and calls for **time- and scaffold-aware** benchmark design. Those are three of the four
failures in [`paper.md`](paper.md), independently identified. Preprint, with an acceptance claim we
have not verified.

### 1.1 Chemprop (D-MPNN) — the supervised reference implementation

| | |
|---|---|
| Original paper | Yang K, Swanson K, Jin W, Coley C, et al., *Analyzing Learned Molecular Representations for Property Prediction*, J. Chem. Inf. Model. 2019, 59(8):3370–3388 |
| Software paper | Heid E, Greenman KP, Chung Y, Li S-C, Graff DE, Vermeire FH, Wu H, Green WH, McGill CJ, *Chemprop: A Machine Learning Package for Chemical Property Prediction*, J. Chem. Inf. Model. 2024, 64:9–17 |
| v2 rewrite | Graff et al., *Chemprop v2: An Efficient, Modular Machine Learning Package for Chemical Property Prediction* (Cambridge Engage preprint, Sep 2025); ~2× faster, ~3× less memory, algorithmic parity with v1 |
| Implementation | `github.com/chemprop/chemprop`, `pip install chemprop` (2.3.1). Python 3.11–3.12. GPU optional |
| Representation | Molecular graph, **directed** message passing over bonds; optional concatenated RDKit 2D descriptors (the "Chemprop-RDKit" variant) |
| Binary classification | Yes, native (`--task-type classification`), sigmoid output, per-molecule probability |

**Architecture.** Messages live on directed bonds rather than atoms, which avoids the
message-bouncing of plain atom-based MPNNs. Default depth 3, hidden 300, then a feed-forward
readout: ~1–2M parameters, three orders of magnitude below our Qwen adapter stack and five below the
base model.

**Published benchmarks and limitations.** Strong on MoleculeNet and on large ADMET collections; it is
the model behind ADMET-AI (Swanson et al., *Bioinformatics* 2024). Its known limitation is the one
that matters most to us: with a few hundred training molecules a D-MPNN must learn a representation
from scratch, and the CheMeleon paper below reports plain Chemprop at a **32% win rate on Polaris
against random forest's 68%** — losing to a random forest on small practical datasets. Chemprop also
has no uncertainty quantification by default (the 2024 paper adds ensembles, evidential and MVE
options).

**Difference from our approaches.** Against the random forest: a learned graph representation
instead of a fixed 2048-bit hash, trained end-to-end, so it can represent features Morgan bits
cannot — and in practice has far more capacity to overfit 500 molecules. Against Qwen QLoRA: no
pretraining at all (unless combined with CheMeleon weights), no text, no world knowledge, and
seconds instead of ~7 minutes per seed.

### 1.2 CheMeleon — pretrained D-MPNN weights, and the strongest small-data candidate

| | |
|---|---|
| Paper | Burns JW, Zalte AS, Abreu CRA, Sieg J, Feldmann C, Mathea M, Green WH, *Deep Learning Foundation Models from Classical Molecular Descriptors*, arXiv:2506.15792 (v1 Jun 2025, v2 Feb 2026) |
| Representation | The same D-MPNN graph as Chemprop; pretrained to predict **low-noise computed molecular descriptors** rather than noisy experimental labels or biased QM simulations |
| Size | ~10M parameters |
| Binary classification | Yes, through the Chemprop fine-tuning path |

**Published benchmarks.** 58 datasets from Polaris and MoleculeACE. Win rates: **Polaris 75%**
against random forest 68%, fastprop 36%, Chemprop 32%. **MoleculeACE 97%** against random forest 50%
and other foundation models. MoleculeACE is ChEMBL single-target potency data — the closest published
benchmark to our task that exists — and this is the only model we found that clearly beats a random
forest there.

**Limitations.** A preprint from the group that maintains Chemprop, so it is both well-engineered and
self-reported. Win rate is a coarse statistic: it says how often a model is ahead, not by how much.
The published comparisons do not use publication-held-out splits.

**Why it is our top recommendation.** It is the one candidate with published evidence of beating a
random forest on exactly our data type; it rides on an installation we need anyway (Chemprop); and
its pretraining signal is **computed descriptors**, which cannot leak an experimental IC50 — the
cleanest provenance of any pretrained model in this review.

### 1.3 ChemBERTa / ChemBERTa-2 — SMILES masked language model

| | |
|---|---|
| Papers | Chithrananda S, Grand G, Ramsundar B, *ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction*, arXiv:2010.09885 (2020). Ahmad W, Simon E, Chithrananda S, Grand G, Ramsundar B, *ChemBERTa-2: Towards Chemical Foundation Models*, arXiv:2209.01712 (Sep 2022) |
| Implementation | HuggingFace `DeepChem/ChemBERTa-77M-MLM` and `DeepChem/ChemBERTa-77M-MTR`; DeepChem ecosystem |
| Representation | SMILES string, RoBERTa encoder (~83M params at the standard config) |
| Pretraining | Up to **77M PubChem compounds**; masked language modelling (MLM) compared against multi-task regression (MTR) on computed properties, with MTR generally better |
| Binary classification | Yes — `AutoModelForSequenceClassification` with 2 labels, or a frozen pooled embedding plus a head |

**Published benchmarks and limitations.** The paper claims competitiveness with the state of the art
on MoleculeNet; third-party reproductions of its MoleculeNet table are marked unverified and should
not be quoted. In the Praski benchmark ChemBERTa lands in the "practically equivalent to ECFP" group
— which, in that paper's framing, is one of the better outcomes. Known limitations: a SMILES LM is
sensitive to canonicalization, and it has no notion of 3D structure.

**Difference from our approaches.** This is the direct, domain-matched counterpart to our Qwen
experiment: both read SMILES as text, both are transformers. The variables are pretraining corpus
(77M molecules against a general text corpus), scale (83M against 9B) and objective (MLM/MTR on
chemistry against next-token on everything). If ChemBERTa-2 beats our QLoRA Qwen on the same splits,
the honest conclusion is that domain pretraining matters more than scale at this data size; if it
does not, that is evidence about the data rather than about either model.

### 1.4 MoLFormer / MoLFormer-XL — large-scale chemical language model

| | |
|---|---|
| Paper | Ross J, Belgodere B, Chenthamarakshan V, Padhi I, Mroueh Y, Das P, *Large-Scale Chemical Language Representations Capture Molecular Structure and Properties*, arXiv:2106.09553; published in *Nature Machine Intelligence* (2022) |
| Implementation | `huggingface.co/ibm-research/MoLFormer-XL-both-10pct` — the public checkpoint is pretrained on 10% of both corpora; needs `trust_remote_code=True` and `deterministic_eval=True` |
| Representation | SMILES, linear-attention transformer with rotary position embeddings, ~47M params |
| Pretraining | SMILES of **1.1 billion** molecules from PubChem and ZINC, masked language modelling |
| Binary classification | Yes — frozen embeddings plus a head, or fine-tuned with a classification head |

**Published benchmarks and limitations.** Reported to match or beat GNN baselines across MoleculeNet,
including quantum-chemical targets from SMILES alone; the paper itself notes that 3D models (SchNet,
DimeNet) beat it on QM9 atomization energies. Two caveats for us: the public checkpoint is the
10%-data variant, so a result here is not a result for the published model; and in the Praski
benchmark MoLFormer was **not** in the ECFP-equivalent group.

**Difference from our approaches.** The same shape as ChemBERTa-2 but a far larger pretraining corpus
and linear attention — of particular interest because our Qwen3.5-9B also uses linear attention (24
Gated DeltaNet layers of 32), so the architectural family is shared and the corpus is the variable.

### 1.5 Uni-Mol / Uni-Mol2 — 3D geometric pretraining

| | |
|---|---|
| Papers | Zhou G, Gao Z, Ding Q, Zheng H, Xu H, Wei Z, Zhang L, Ke G, *Uni-Mol: A Universal 3D Molecular Representation Learning Framework*, ICLR 2023. Ji X, Wang Z, Gao Z, Zheng H, Zhang L, Ke G, E W, *Uni-Mol2: Exploring Molecular Pretraining Model at Scale* (2024) |
| Implementation | `github.com/deepmodeling/Uni-Mol`; the practical path is `pip install unimol_tools` (`MolTrain`, `MolPredict`, `UniMolRepr`), weights from HF `dptech/Uni-Mol-Models` |
| Representation | **3D conformers.** SE(3) transformer over atom coordinates plus pair-distance representations; two pretrained models, one for molecules and one for protein pockets |
| Pretraining | 209M molecular conformations (≈19M molecules); 3M candidate pockets for the pocket model. Uni-Mol2 scales to 1.1B parameters over ~800M conformations, with power-law scaling in model size, data and compute |
| Binary classification | Yes, `MolTrain(task='classification')` |

**Published benchmarks and limitations.** Uni-Mol reports beating prior methods on 14 of 15 molecular
property prediction tasks, and additionally handles binding-pose prediction and conformation
generation. The limitations for us are severe and specific:

1. It needs **conformers**, generated by RDKit at inference time. Our TG2 SMILES carry **no
   stereochemistry at all** ([`data.md`](data.md) §4 — not one `@` or `/` in the PubChem export). A
   3D model asked to embed a molecule whose stereocentres were discarded is being fed a guess. This
   is the fairness problem of the whole comparison: Uni-Mol's advantage lies in a modality our
   dataset does not contain, so a poor Uni-Mol result here would be a statement about our data and
   not about Uni-Mol, and must be reported that way.
2. In the Praski benchmark both Uni-Mol and Uni-Mol2 were in the group **worse than ECFP**.
3. `unimol_tools` documents `numpy<2.0.0` for RDKit; our environment runs numpy 2.5.4. It needs its
   own virtualenv (§4.1).
4. Uni-Mol2 at 1.1B parameters is out of scope for full fine-tuning on 16 GB and is not recommended.

### 1.6 Tabular foundation models — the cheap surprise

| | |
|---|---|
| Paper | Ben Hicham KK, Rittig JG, Grohe M, Mitsos A, *Tabular foundation models for in-context prediction of molecular properties*, arXiv:2604.16123 (Apr 2026) |
| Representation | A tabular foundation model (TabPFN-class) doing **in-context** prediction over a feature table: CheMeleon embeddings, RDKit 2D descriptors, Mordred descriptors, or fingerprints |
| Binary classification | Yes, natively, and this is its home setting |

**Published results.** Aimed explicitly at low- and medium-data regimes. Reports up to **100% win
rates across the 30 MoleculeACE tasks** with CheMeleon embeddings, and strong results from compact
RDKit2d / Mordred descriptor tables, at lower computational cost than fine-tuning — there is no
task-specific training at all, because the training set is passed as context.

**Why this belongs in our comparison.** 631 molecules with a fingerprint or descriptor table is a
small table, which is the regime these models are built for; it costs seconds on CPU; and it has a
property no other entrant has. A TabPFN-class model is pretrained on **synthetic tabular data**, so
its pretraining cannot contain any molecule or any assay value — a true zero-chemical-leakage
control (§4.6). It is a 2026 preprint and should be presented as exploratory.

### 1.7 Considered and not recommended

| Method | Reference | Why not |
|---|---|---|
| GROVER | Rong et al., NeurIPS 2020 | Worse than ECFP in the Praski benchmark; heavy, and with known reproduction difficulties |
| MolCLR | Wang et al., *Nat. Mach. Intell.* 2022 | Contrastive GNN; GNN embeddings were the weakest family in Praski |
| MAT / R-MAT | Maziarka et al. | ECFP-equivalent in Praski, so no expected gain over a baseline we already have |
| Uni-Mol2 1.1B, ChemGPT, ChemFM | — | Out of hardware scope, or worse than ECFP, or both |
| CLAMP | Seidl et al., ICML 2023 | The *only* significant winner in Praski — but it is a bioactivity-text contrastive model trained on **PubChem assay data**, which very likely includes TG2 assays. Direct label leakage. Worth a leakage audit and a paragraph of discussion; **not** a fair entrant here |

CLAMP deserves its own line: the one model that beat ECFP in the largest fair benchmark is the one we
must exclude on provenance grounds. That is itself worth saying in any write-up.

---

## 2. Comparison of current research methods

| Method | Representation | Pretraining corpus | Params | Binary cls. | Published standing vs fingerprints on small data | Feasible on RTX 5070 Ti 16 GB |
|---|---|---|---|---|---|---|
| **RF / Morgan (ours)** | 2048-bit ECFP4 | none | — | yes | the bar: **0.898 ± 0.057** AUROC on our data | trivial, ~1 s |
| **Qwen3.5-9B QLoRA (ours)** | SMILES text | general web/text | 9B base, 43M LoRA | yes, Jev readout | loses to RF by 0.049 on our data | yes — 7.98 GiB peak, ~7 min/seed |
| Chemprop v2 D-MPNN | 2D graph | none | ~1–2M | yes | 32% Polaris win rate against RF's 68% | yes, seconds–minutes |
| **CheMeleon** | 2D graph | computed descriptors | ~10M | yes | **75% Polaris, 97% MoleculeACE** | yes, minutes |
| ChemBERTa-2 | SMILES | 77M PubChem | ~83M | yes | ≈ ECFP (Praski) | yes, a full fine-tune fits easily |
| MoLFormer-XL (10pct) | SMILES | 1.1B PubChem+ZINC | ~47M | yes | worse than ECFP (Praski) | yes, frozen or fine-tuned |
| Uni-Mol | 3D conformer | 209M conformations | ~48M | yes | worse than ECFP (Praski) | yes, but see §4.1 |
| Tabular FM (TabPFN-class) | descriptor table | synthetic tables | ~10M | yes | up to 100% MoleculeACE win rate | yes, CPU, seconds |
| Uni-Mol2 1.1B | 3D conformer | ~800M conformations | 1.1B | yes | worse than ECFP (Praski) | **no** — not at full fine-tune |
| CLAMP | fingerprint + assay text | PubChem **assays** | — | yes | the only significant winner in Praski | yes, but **excluded on leakage** |

### How these differ from what we have, by axis

**Representation.** We have tested exactly two: a fixed hash of substructures (Morgan) and SMILES as
character text (Qwen, both frozen and fine-tuned). We have never tested a *learned graph*
representation, and we have never tested a representation pretrained on chemistry. Those are the two
real gaps, and they map onto Chemprop/CheMeleon and ChemBERTa-2/MoLFormer respectively.

**Pretraining provenance.** Ours is the odd one out, and not in a good way. Every chemical model in
the table was pretrained on molecular structures or computed properties. Qwen3.5-9B was pretrained on
text, which can include the *papers* our labels come from — and we have already documented one
instance ([`zed1227_analysis.md`](zed1227_analysis.md): the clinical compound is in our training data
and the model can plausibly have read about it). This asymmetry is the single most important thing
the experiment must control for, and §4.6 is about it.

**Scale and cost.** Our QLoRA run is roughly 6 GPU-hours for the full 10-seed set. Every alternative
in the recommended set is between 100× and 10,000× cheaper. If one of them ties the random forest,
the LLM's cost has no remaining justification on this task; if none of them does, the fingerprint
baseline's standing gets stronger rather than weaker.

---

## 3. Recommended models to implement

Ranked by expected information per hour of work. **Tiers 1 and 2 are the proposal; tier 3 is listed
so the write-up can say why it was skipped.**

### Tier 1 — implement first

1. **Chemprop v2 D-MPNN**, two variants: graph only, and graph + RDKit 2D descriptors. The
   supervised-GNN reference the field expects to see, and the installation CheMeleon needs anyway.
2. **CheMeleon** (Chemprop fine-tuned from the pretrained descriptor weights). The highest prior
   probability of actually beating our random forest, on the most relevant published benchmark, with
   the cleanest pretraining provenance.
3. **ChemBERTa-2** fine-tuned (`77M-MTR`), plus a frozen-embedding variant. The domain-matched
   counterpart to our Qwen experiment; makes "domain pretraining against scale" a measured quantity
   rather than a speculation.
4. **Tabular foundation model** (TabPFN-class) on Morgan plus RDKit 2D descriptors. Near-zero cost,
   built for exactly this sample size, and the only zero-chemical-leakage pretrained entrant.

### Tier 2 — implement if tier 1 runs clean

5. **MoLFormer-XL-both-10pct**, frozen embeddings with the same heads, and a LoRA fine-tune. The
   frozen variant is directly comparable to our existing frozen-Qwen probe
   ([`qwen_probe.md`](qwen_probe.md)) — same head, same splits, same seeds — which turns "is a
   chemical LM's embedding better than a general LLM's embedding?" into a paired test costing
   minutes of GPU time.
6. **Uni-Mol** via `unimol_tools`, in its own virtualenv, reported **with** the stereochemistry
   caveat stated up front rather than in a footnote.

### Tier 3 — recommended against, with reasons recorded

GROVER, MolCLR, MAT/R-MAT (no expected gain over a baseline we already have); Uni-Mol2 1.1B (out of
hardware scope); CLAMP (assay-label leakage — discussed and audited, not entered).

### Why this set and not a bigger one

The study can resolve about ±0.07 AUROC at 10 seeds ([`jev_qlora.md`](jev_qlora.md)), so entering
twelve models would produce a ranking that is mostly noise plus a multiple-comparison problem that
erases whatever signal survives. Six entrants against two incumbents, with a pre-registered primary
comparison, is what this dataset can support.

---

## 4. Detailed experimental protocol

Pre-registered before any model is run: every number below is fixed in code and committed *before*
the first result is seen, following [`experimental_plan.md`](experimental_plan.md).

### 4.1 Environments, and how splits are kept provably identical

Chemprop pins Lightning, `unimol_tools` documents `numpy<2.0.0`, and our main `.venv` runs torch
2.14.1+cu130 / numpy 2.5.4 with a 19 GB Qwen checkpoint cached against it. Installing all of this
into one environment would risk the existing results.

**Design: one exported split file, several read-only virtualenvs.** New code, kept deliberately
small:

```
scripts/export_splits.py     # NEW, ~30 lines. Writes results/splits_tg2.csv once:
                             #   smiles, y, source, seed, fold(train|valid|test), split_kind
scripts/run_sota.py          # NEW. One file, one --model flag per entrant. Reads that CSV,
                             #   never calls scaffold_split itself, and writes
                             #   results/sota_tg2.csv in the SAME per-seed schema the other
                             #   result CSVs already use
```

`export_splits.py` calls the existing `src.data.chem.scaffold_split` with the existing seeds, so the
folds are the same objects every prior result used — verifiable by asserting that the exported
train/valid/test index sets equal those of a fresh `scaffold_split` call, which is what
`tests/test_sota_splits.py` will do. Metrics come from `src/evaluation/metrics.py` unchanged, so
AUROC/AUPRC/MCC/ECE/Brier/log-loss are computed by the same code for every model in the project.
Significance comes from the existing `scripts/compare.py`, which already does paired-by-seed tests.
Per-model environments live in `envs/<model>/`, built by `scripts/setup_sota_envs.sh`; `run_sota.py`
shells out to the right interpreter per entrant.

No new metrics module, no new split logic, no new statistics code, no orchestration framework.

### 4.2 Splits

| Split | Definition | Molecules | Status |
|---|---|---|---|
| **Scaffold (primary)** | `chem.scaffold_split`, Bemis–Murcko, 80/10/10, seeds 0–9 (0–19 where affordable) | 631 | identical to every existing TG2 result |
| **Publication held out (secondary)** | `chem.group_split` on `document_chembl_id` | **~460, the ChEMBL-attributable subset only** | see the caveat below |
| Random (control only) | `group_split` on molecule index | 631 | reported only to quantify leakage, never as generalization |

Scaffold is the primary split because it is what the comparison literature uses, so our numbers stay
commensurable with published ones.

The publication split is where this study can contribute something the literature does not have.
[`second_target.md`](second_target.md) measured that a scaffold split still leaves 69 publications
with molecules on both sides of the fold, and that holding out whole papers costs more AUROC than
holding out scaffolds (CDK2: 0.898 → 0.830). The open question is whether the *ranking* of models
changes under it — whether a pretrained model's apparent advantage is partly an advantage at
recognising a laboratory's series.

**Feasibility caveat, stated plainly.** Only the ChEMBL half of the combined set carries
`document_chembl_id`; the PubChem export records no provenance ([`data.md`](data.md)). The
publication split therefore runs on the ~460 ChEMBL-attributable molecules, not on all 631, and its
results are **not** paired with the scaffold-split results on the same molecules. It must be reported
as a different dataset, not as a different protocol on the same one. Recovering provenance for the
171 PubChem-only molecules is a separate task and is not assumed here.

### 4.3 Hyperparameter tuning, with an equal budget

The fairness failure mode is obvious. Our random forest was tuned per seed on its validation fold
([`baselines.md`](baselines.md)), gaining +0.027 AUROC at p = 0.019. Giving a deep model its paper's
default configuration against our tuned baseline would rig the comparison in our favour; hand-tuning
a deep model until it wins would rig it the other way.

**Rule: every entrant gets exactly 12 configurations, no more and no fewer.** For each seed all 12
are trained on `train`, selected on `valid` by **Brier score** — not AUROC, because calibration is
the project's standing concern and selecting on the headline metric is the classic optimistic-bias
mistake — and the winner is scored **once** on `test`. Each model's 12-point grid is written into
`run_sota.py` and committed before the first run. The incumbents are re-run under the same rule so
the comparison is symmetric; where their existing grid was smaller, that is recorded.

Early stopping for the neural entrants uses `valid` only, with a fixed patience, and the chosen epoch
is logged per seed. No configuration, epoch count or preprocessing decision may be changed after a
test number is seen; if one must be, the run is relabelled exploratory and reported as such.

### 4.4 Metrics

Everything from `src/evaluation/metrics.py`, unchanged: **AUROC, AUPRC, MCC, ECE (15 equal-width
bins), Brier, log loss (clipped)**. Plus:

- **Generalization:** each test molecule's max Tanimoto to that seed's training set, and AUROC inside
  the bands of [`generalization.md`](generalization.md) (0–0.4 / 0.4–0.5 / 0.5–0.6 / 0.6–0.7 /
  0.7–1.0). Pooled **one row per molecule**, never one per prediction — the pooling error of
  [`forgetting.md`](forgetting.md) must not be repeated.
- **Calibration under shift:** the temperature fitted on `valid` per seed, reported with its spread,
  to test whether a chemically pretrained model is less unidentifiable than the 1.54 ± 0.94 (RF) and
  31.2 ± 61.8 (Qwen) of [`calibration.md`](calibration.md).
- **Abstention curve:** AUROC against coverage when predictions below a nearest-neighbour Tanimoto
  threshold are withheld. This is the metric that matters for actually using a model, and
  `scripts/audit_splits.py` already computes the similarity.
- **Cost:** wall-clock seconds per seed, peak VRAM (`nvidia-ml-py`, already a dependency), parameters
  trained, and each model's one-off pretraining cost as published by its authors.

### 4.5 Statistics

- **Primary endpoint, pre-registered:** ΔAUROC against the **tuned random forest** on the scaffold
  split, paired by seed. One primary test per entrant.
- **Secondary:** ΔBrier and ΔECE against the same baseline; ΔAUROC against **Qwen QLoRA**; the
  low-similarity bands; the publication split.
- Paired t-test and sign test over seeds (which `compare.py` already does), plus a BCa bootstrap CI
  over molecules for the pooled estimate.
- **Multiplicity:** Holm correction across the six entrants on the primary endpoint. Secondary
  endpoints are reported uncorrected and labelled exploratory, as in
  [`calibration.md`](calibration.md).
- **Power, stated before the fact:** per-seed AUROC spread on this dataset runs ±0.057 to ±0.128, so
  10 seeds resolve roughly ±0.07. **A difference smaller than that cannot be claimed, and the
  likeliest outcome of this study is a set of ties.** Because the tier-1 models cost seconds to
  minutes rather than 7 minutes per seed, the seed count rises to **20** for everything that can
  afford it (Chemprop, CheMeleon, ChemBERTa-2, tabular FM, RF, logreg, XGBoost), cutting the
  resolvable difference to about ±0.05. Qwen QLoRA stays at its existing 10 seeds, and comparisons
  involving it use the paired 10.
- **Equivalence, not only difference.** With ties expected, the primary result is reported as a
  two-sided CI against a pre-stated margin of 0.05 AUROC, so "no difference" is a measurement rather
  than a failure to measure.

### 4.6 Pretraining-data leakage: the audit, and what it can and cannot establish

This is the part of the design needing the most honesty, because the naive audit gives a meaningless
answer.

Our 631 molecules come from PubChem and ChEMBL. ChemBERTa-2 was pretrained on 77M PubChem compounds
and MoLFormer on 1.1B PubChem+ZINC molecules. **Essentially every one of our molecules is in those
corpora as a structure.** Reporting "99% overlap" would be true and uninformative. The distinction
that matters is between *structure* membership and *label* membership:

| Model | Saw our structures? | Saw a computed property? | Saw an experimental TG2 IC50? |
|---|---|---|---|
| RF / Morgan | no pretraining | — | — |
| Tabular FM (TabPFN-class) | **no** — synthetic tables | — | — |
| Chemprop (from scratch) | no pretraining | — | — |
| CheMeleon | almost certainly | **yes, computed descriptors** | no |
| ChemBERTa-2 MTR | almost certainly | **yes, computed properties** | no |
| MoLFormer-XL | almost certainly | no (MLM only) | no |
| Uni-Mol | likely | no (conformations) | no |
| **Qwen3.5-9B** | **likely, as text** | — | **possible — the papers themselves** |
| CLAMP | yes | — | **yes, PubChem assay data** → excluded |

Three concrete audit steps, all cheap:

1. **Canonical-structure overlap**, reported per model where a corpus is downloadable, with the
   explicit note that near-100% is expected and is not by itself a confound for an unsupervised
   objective. What it does undermine is a narrower claim: that these models are being tested on
   "novel chemotypes". They are not. Our low-similarity bands are novel relative to *our training
   fold*, not relative to their pretraining.
2. **A label-leakage probe for Qwen specifically.** Zero-shot, ask the base model for each molecule's
   TG2 IC50 as a number, and correlate with the true value on molecules excluded from that seed's
   training fold. A correlation materially above chance is direct evidence of memorised assay values
   and would qualify every Qwen result in [`paper.md`](paper.md).
   [`zed1227_analysis.md`](zed1227_analysis.md) gives a reason to expect a non-null result for at
   least the named clinical compounds. This probe is worth running whatever happens to the rest of
   the study.
3. **A name-stripping control for Qwen.** [`qwen_probe.md`](qwen_probe.md) already found that adding
   drug names helps on GDSC2, and argued the gain cannot transfer to novel molecules. The TG2 prompts
   carry SMILES only; that should be confirmed in code and asserted in a test rather than assumed.

**What this audit cannot do.** It cannot certify that a pretrained model has not memorised an IC50
from a paper that is not in any corpus we can inspect. The honest form of the limitation: pretrained
models are tested here under an assumption of no label leakage that is supported by their published
objectives and, for Qwen, probed directly — and is not proven for any of them.

### 4.7 Assay provenance

Carried over unchanged from [`data.md`](data.md), and re-asserted in the tests rather than restated
in prose: human TG2 only (CHEMBL2730 / P21980); `assay_type == "B"`; exact `=` IC50 values plus
censored `>` bounds at or above the 1 µM threshold as confirmed negatives; Ki and kinact excluded;
ChEMBL `data_validity_comment` rows dropped; median pIC50 per molecule; salts stripped to the largest
fragment. Every model in this study reads the **same** exported CSV, so none of these decisions can
drift between entrants.

One provenance item is new and model-specific: **Uni-Mol's conformers are generated by us**, so the
conformer settings — RDKit ETKDG version, random seed, number of conformers, force field, and whether
unspecified stereocentres are enumerated or arbitrarily assigned — become part of the data pipeline
and must be recorded in the results CSV alongside the metrics.

### 4.8 Order of work

| | Step | Cost | Gate |
|---|---|---|---|
| 1 | `export_splits.py` plus the split-identity test | ~1 h | the test passes, or stop |
| 2 | Re-run RF / logreg / XGBoost at 20 seeds under the 12-config rule | minutes | reproduces 0.898 ± 0.057 within noise, or stop and find out why |
| 3 | Chemprop v2, both variants | ~2 h incl. env | — |
| 4 | CheMeleon | ~1 h | — |
| 5 | Tabular FM | ~1 h | — |
| 6 | ChemBERTa-2, fine-tuned and frozen | ~3 h | — |
| 7 | Leakage audit, all three probes | ~2 h | — |
| 8 | Publication split, ChEMBL subset, all entrants | ~2 h | — |
| 9 | MoLFormer (tier 2) | ~3 h | only if 1–8 are clean |
| 10 | Uni-Mol (tier 2) | ~4 h, env risk | only if 1–9 are clean |
| 11 | Extend `scripts/check_paper.py` so the new numbers are recomputed from `results/` | ~1 h | the project's existing standard |

Step 2 is a gate and not a formality: if the existing baseline does not reproduce under the new
harness, nothing downstream of it means anything.

---

## 5. Expected scientific contribution, and limitations

### What this can contribute

**1. A leakage-audited, equal-budget comparison of six model families on one small target-specific
bioactivity dataset.** The field's benchmarks are broad and shallow — Praski et al. cover 25 datasets
with frozen embeddings and a fixed head; MoleculeACE covers 30 targets with published defaults.
Neither audits its splits for fingerprint-identical duplicates, publication confounding or
pretraining membership, and neither gives every entrant the same tuning budget. This study would be
narrow and deep: one target, 631 molecules, splits already audited in [`data.md`](data.md), 12
configurations each, one test evaluation per seed, paired statistics and a pre-stated equivalence
margin.

**2. The first placement of a general-purpose LLM on the same footing as chemical foundation
models.** Our Qwen3.5-9B QLoRA result exists; what it has never had is a domain-pretrained
transformer of similar shape on identical folds. The frozen comparison is the sharpest version:
Qwen-9B mean-pooled hidden states against MoLFormer and ChemBERTa-2 embeddings, same heads, same
seeds, same molecules. "Does chemical pretraining beat 100× the parameters at this data size?" is a
question the field gestures at without measuring.

**3. Whether a model ranking survives a publication-held-out split.** This is the most likely real
contribution, and it is ours to make because the methodology work is already done. Every benchmark in
§1 ranks models under scaffold splits. We have measured, on two independent targets, that a scaffold
split leaves most laboratories' chemistry on both sides of the fold and that holding out whole
publications costs more. If the ranking reorders under a publication split — if a pretrained model's
advantage is partly an advantage at recognising a series — that is a finding about *how the field
evaluates*, which is this project's established line of work.

**4. Calibration under chemotype shift, across model families.** [`calibration.md`](calibration.md)
established that post-hoc rescaling fails here and diagnosed why: the fitted temperature is
unidentifiable on a scaffold-split validation fold. Whether that is a property of fingerprint models
or of the shift itself is open, and measuring the same quantity on six architectures answers it.

**5. A negative result worth reporting either way.** If the recommended models tie the tuned random
forest, that replicates Praski et al. on a target-specific bioactivity dataset under an independent
protocol, strengthening the existing conclusion rather than weakening it. If CheMeleon wins, the
project gains a better model and a clean story about why descriptor pretraining is what helps at
n = 631.

### Limitations, stated before the results exist

1. **631 molecules, one target.** Nothing here generalises to other targets by itself. The CDK2 and
   MAO-B replications in [`second_target.md`](second_target.md) are the template for what a second
   target would cost, and it is not in this scope.
2. **The study is powered for ties.** ±0.07 AUROC at 10 seeds, ±0.05 at 20. A model that is genuinely
   0.03 better will not be distinguishable. Reporting that as "no difference" without the interval
   would be a fifth protocol failure in a project about protocol failures.
3. **No stereochemistry** ([`data.md`](data.md) §4). This caps every structure-based model and
   **specifically handicaps the 3D entrant**. A poor Uni-Mol result must not be reported as evidence
   against 3D pretraining.
4. **Pretraining overlap is unavoidable and only partly auditable** (§4.6). The no-label-leakage
   assumption rests on published objectives rather than on inspection for every model but the tabular
   one.
5. **One public checkpoint is not the published model.** The available MoLFormer is the 10%-data
   variant, so a result for it is not a result for MoLFormer-XL as published.
6. **The publication split runs on a different, smaller dataset** (~460 ChEMBL-attributable
   molecules), so it is not paired with the primary result.
7. **Hardware excludes the largest models.** Uni-Mol2 1.1B and anything of that scale is out of reach
   at 16 GB, so "state of the art" here means "state of the art that runs on one consumer GPU".
8. **Computational prediction is not biochemical inhibition and is not a celiac treatment.** The
   standing constraint of the whole project. Nothing in this study moves a compound closer to a
   patient, and no candidate shortlist follows from it.

### What this study will not do

No model is trained until this protocol is approved. [`paper.md`](paper.md), `results/` and every
existing number are untouched by this document. If the study runs, its numbers go into new
`results/sota_*.csv` files and a new section, and `scripts/check_paper.py` is extended rather than
edited around.

---

## Sources

1. Praski M, Adamczyk J, Czech W. *Benchmarking Pretrained Molecular Embedding Models For Molecular Representation Learning.* arXiv:2508.06199 (2025; v4 Feb 2026). Code: `github.com/scikit-fingerprints/benchmarking_molecular_models`
2. van Tilborg D, Alenicheva A, Grisoni F. *Exposing the Limitations of Molecular Machine Learning with Activity Cliffs.* J. Chem. Inf. Model. 2022, 62(23):5938–5951. doi:10.1021/acs.jcim.2c01073. Code: `github.com/molML/MoleculeACE`
3. Li Z, Chen X, Wen H, Zhang RQ, Li M, Zhang X, Yin H, Yang Q, Lam K-Y, Lio P, Yiu S-M. *A Systematic Survey and Benchmark of Deep Learning for Molecular Property Prediction in the Foundation Model Era.* arXiv:2604.16586 (2026); accepted, J. Chem. Theory Comput.
4. Yang K, Swanson K, Jin W, Coley C, et al. *Analyzing Learned Molecular Representations for Property Prediction.* J. Chem. Inf. Model. 2019, 59(8):3370–3388
5. Heid E, Greenman KP, Chung Y, Li S-C, Graff DE, Vermeire FH, Wu H, Green WH, McGill CJ. *Chemprop: A Machine Learning Package for Chemical Property Prediction.* J. Chem. Inf. Model. 2024, 64:9–17
6. Graff DE, et al. *Chemprop v2: An Efficient, Modular Machine Learning Package for Chemical Property Prediction.* Cambridge Engage preprint, Sep 2025
7. Burns JW, Zalte AS, Abreu CRA, Sieg J, Feldmann C, Mathea M, Green WH. *Deep Learning Foundation Models from Classical Molecular Descriptors* (CheMeleon). arXiv:2506.15792 (2025; v2 Feb 2026)
8. Chithrananda S, Grand G, Ramsundar B. *ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction.* arXiv:2010.09885 (2020)
9. Ahmad W, Simon E, Chithrananda S, Grand G, Ramsundar B. *ChemBERTa-2: Towards Chemical Foundation Models.* arXiv:2209.01712 (2022)
10. Ross J, Belgodere B, Chenthamarakshan V, Padhi I, Mroueh Y, Das P. *Large-Scale Chemical Language Representations Capture Molecular Structure and Properties.* arXiv:2106.09553; Nature Machine Intelligence (2022). Checkpoint: `huggingface.co/ibm-research/MoLFormer-XL-both-10pct`
11. Zhou G, Gao Z, Ding Q, Zheng H, Xu H, Wei Z, Zhang L, Ke G. *Uni-Mol: A Universal 3D Molecular Representation Learning Framework.* ICLR 2023. Code: `github.com/deepmodeling/Uni-Mol`; `pip install unimol_tools`
12. Ji X, Wang Z, Gao Z, Zheng H, Zhang L, Ke G, E W. *Uni-Mol2: Exploring Molecular Pretraining Model at Scale.* (2024)
13. Ben Hicham KK, Rittig JG, Grohe M, Mitsos A. *Tabular foundation models for in-context prediction of molecular properties.* arXiv:2604.16123 (2026)
14. Dettmers T, Pagnoni A, Holtzman A, Zettlemoyer L. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023 — the method already in use here; listed for continuity with `paper.md` ref 4
15. Wu Z, Ramsundar B, Feinberg EN, Gomes J, Geniesse C, Pappu AS, Leswing K, Pande V. *MoleculeNet: a benchmark for molecular machine learning.* Chem. Sci. 2018, 9:513–530
16. Swanson K, et al. *ADMET-AI: a machine learning ADMET platform for evaluation of large-scale chemical libraries.* Bioinformatics 2024 — Chemprop-RDKit in production use
17. Deng J, Yang Z, Wang H, Ojima I, et al. *A systematic study of key elements underlying molecular property prediction.* Nature Communications 2023, 14:6395
18. Seidl P, et al. *Enhancing Activity Prediction Models in Drug Discovery with the Ability to Understand Human Language* (CLAMP). ICML 2023

### A note on citation confidence

References 1, 2, 4, 5, 7, 8, 9, 10, 11, 13 and 17 were verified against the publisher or arXiv page
during this review. References 3 and 13 are 2026 preprints: real and fetched, but recent and not
peer-reviewed at the time of writing, and marked as such wherever they are used above. References 6,
12, 16 and 18 are cited from secondary confirmation and should be checked against the primary source
before they appear in a manuscript. Reported figures that came only from third-party aggregator
pages — notably the ChemBERTa-2 MoleculeNet table — are deliberately **not** quoted anywhere in this
document.
