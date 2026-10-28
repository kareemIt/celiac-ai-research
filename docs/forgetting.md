# Does QLoRA make Qwen forget unfamiliar chemistry? (run 2026-10-28)

Reproduce: `python scripts/run_novelty_stats.py` (CPU, seconds) and
`python scripts/run_jev_qlora.py --seeds 10 --epochs 3 --data combined --kl {0,1}` (GPU, hours),
then `python scripts/run_kl_compare.py`.
Tables: `results/novelty_stats.csv`, `results/kl_compare.csv`.

## The hypothesis

[`generalization.md`](generalization.md) reported that on test molecules with no training analog
above 0.5 Tanimoto, the **zero-shot** Jev readout scored AUROC 0.807 — best of any model, having
been the worst overall — while the **fine-tuned** model scored 0.442, worse than chance. The reading
offered there was that LoRA on 569 molecules teaches the model these chemotypes and overwrites the
general chemical prior that made the zero-shot readout robust, and that the KL anchors of
LLM-as-Jev, omitted from our implementation, exist to prevent exactly that.

Two claims are bundled together there, and they turn out to have very different support.

## 1. Those numbers were not what they looked like

The 0.807 and 0.442 came from 51 predictions pooled over 10 overlapping scaffold test folds. Those
51 rows are 42 distinct molecules, so a molecule appearing in several folds was counted several
times, and — the real problem — **the zero-shot score for a molecule is literally the same number in
every fold**, because it is read from a cache and never refit. Pooling therefore reweights the
zero-shot model's molecules by how often they happened to land in a test fold, which has nothing to
do with chemistry.

Recomputed with one row per molecule (`scripts/run_novelty_stats.py`), and a cluster bootstrap that
resamples **molecules**, not predictions, so the two models always see the same resampled set:

| model | novel AUROC | AUPRC | 95% CI |
|---|---|---|---|
| random forest | **0.782** | 0.411 | [0.569, 0.959] |
| Qwen zero-shot | 0.764 | 0.460 | [0.556, 0.931] |
| XGBoost | 0.694 | 0.325 | [0.430, 0.925] |
| logistic regression | 0.676 | 0.320 | [0.414, 0.921] |
| Qwen QLoRA | **0.396** | 0.173 | [0.062, 0.745] |

| paired difference | ΔAUROC | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| zero-shot − QLoRA | **+0.368** | [+0.013, +0.658] | 0.023 |
| zero-shot − RF | −0.019 | [−0.257, +0.192] | 0.566 |
| RF − QLoRA | **+0.387** | [+0.085, +0.699] | 0.002 |

**The first claim dies.** "The zero-shot LLM is the best model on unfamiliar chemistry" was an
artefact of pooling: molecule for molecule it is 0.019 behind the random forest, and the interval
around that difference is a quarter of an AUROC wide. Taking the first seed's prediction for each
molecule instead of the average gives the same picture (−0.009, CI [−0.255, +0.206]).

**The second claim survives this test.** The fine-tuned model really is far below both the base
model it started from and the random forest, and both gaps clear zero.

## 2. But the subset is much smaller than 42

Those 42 molecules form only **23 Butina clusters** at 0.5 Tanimoto, and just **6 of them are
positive**. An AUROC on 6 positives is a statement about the rank of 6 molecules. The bootstrap
intervals above already show this — every one of them is 0.3 to 0.7 AUROC wide.

The cutoff was fixed at 0.5 before any of these numbers existed, so the conclusion should not depend
on it. It does:

| cutoff | molecules | positives | clusters | zero-shot | QLoRA | RF | zs − RF | zs − QLoRA |
|---|---|---|---|---|---|---|---|---|
| 0.35 | 12 | 1 | 7 | 0.636 | 0.000 | 0.273 | +0.364 | +0.636 |
| 0.40 | 19 | 1 | 9 | 0.722 | 0.000 | 0.389 | +0.333 | +0.722 |
| 0.45 | 25 | 3 | 11 | 0.742 | 0.159 | 0.667 | +0.076 | +0.583 |
| **0.50** | 42 | 6 | 23 | 0.764 | 0.396 | **0.782** | −0.019 | +0.368 |
| 0.55 | 73 | 19 | 29 | 0.616 | 0.580 | **0.909** | −0.293 | +0.036 |
| 0.60 | 115 | 23 | 36 | 0.621 | 0.588 | **0.871** | −0.250 | +0.033 |

The zero-shot model only beats the random forest at cutoffs 0.45 and below, where the subset
contains **one or three positive molecules**. From 0.55 up the random forest wins by a quarter of an
AUROC. The "zero-shot generalizes better" story lives entirely in a corner of the data too small to
support it.

The QLoRA deficit behaves differently: it is large and consistent from 0.35 to 0.50 and then
disappears at 0.55. That is at least the right shape for a real effect — damage concentrated in the
deep-novelty tail — but it is measured on the same 1–6 positives.

## 3. The split-respecting test finds nothing

Pooling across folds, however carefully reweighted, still treats correlated folds as one sample. The
only test that fully respects the experimental design is per-seed: score each seed's novel subset on
its own, then compare models seed by seed. Six of nine seeds have both classes in their novel
subset:

| paired difference | mean Δ | paired t-test | sign test | seeds better |
|---|---|---|---|---|
| zero-shot − QLoRA | +0.053 | 0.717 | 0.688 | 2/6 |
| zero-shot − RF | −0.125 | 0.562 | 0.219 | 1/6 |
| RF − QLoRA | +0.178 | 0.392 | 0.688 | 2/6 |

**Nothing is significant, and the forgetting effect nearly vanishes** (+0.053 rather than +0.368).
Six seeds with about five molecules each cannot detect anything short of an enormous effect, so this
is not evidence of absence. It is, however, the honest ceiling on what the existing data can show:
the molecule-level analysis says the effect is probably there, the design-respecting analysis says
we cannot demonstrate it.

## 4. One leak we cannot close

The TG2 molecules here come from PubChem and ChEMBL, both long predating Qwen3.5. The base model may
simply have **memorised** some of these published inhibitors and their activities. That would not
affect the fine-tuned-versus-base comparison, which is the one we care about, but it does mean the
zero-shot readout's score on "unfamiliar" molecules is not a clean measure of chemical reasoning —
unfamiliar to the random forest is not unfamiliar to a model trained on the internet. Settling this
needs molecules published after the pretraining cutoff, which we do not have.

## Where this leaves the question

| claim | verdict |
|---|---|
| Zero-shot Qwen generalizes to unfamiliar chemistry better than fingerprints | **Not supported.** Artefact of pooling; holds only where there is one positive molecule |
| QLoRA fine-tuning costs novel-chemistry performance | **Suggestive, not established.** Survives the molecule-level bootstrap, disappears per seed |
| The random forest is the best model on unfamiliar chemistry too | **Supported** at every cutoff from 0.5 up |

The surviving question is narrow but still worth answering, and it is the one the
[KL-anchor experiment](#5-kl-anchors) tests directly, with far more data than this endpoint
comparison: every epoch of every seed on the same molecules, rather than one final number.

## 5. KL anchors

*(results pending — the λ = 0 and λ = 1 runs are in progress)*

`src/models/jev_anchors.py` implements the three anchors of LLM-as-Jev §3.4 against the frozen base
model, which here is the same network with the LoRA adapter switched off:

- **L_mass** — binary KL on how much probability mass sits on the legal option tokens at the
  decision node.
- **L_out** — KL over the renormalized distribution across *illegal* tokens at that node, which
  preserves their relative ranking.
- **L_pos** — KL over the full next-token distribution at two context positions, the reply start and
  the position after the closing bracket, weighted 0.5 each.

Every term is KL(p₀ ‖ p_θ) truncated to the reference's top-64 tokens, and the total objective is
`L_tree + λ·L_anchor` with the paper's default λ = 1. Two deviations are forced by our setting and
recorded in the module: the paper samples three off-path trie nodes alongside the gold path, but a
two-option single-digit decision has exactly one node; and the paper's stated purpose for the
anchors is preserving **general chat behaviour**, whereas we are asking them to preserve a **chemical
prior**. The second is an extrapolation, not something the paper claims.

Both runs log valid and test log-odds after every epoch, including **epoch 0** — the adapter is
reset to B = 0 before each seed, so epoch 0 is the untouched base model evaluated on exactly the
molecules of that seed's folds. Each run therefore carries its own matched zero-shot control, which
removes the cache-reuse problem described in §1.
