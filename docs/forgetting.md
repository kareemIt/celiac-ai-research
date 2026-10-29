# Does QLoRA make Qwen forget unfamiliar chemistry? (run 2026-10-29)

Reproduce: `python scripts/run_novelty_stats.py` (CPU, seconds) and
`python scripts/run_jev_qlora.py --seeds 10 --epochs 3 --data combined --kl {0,1} --deterministic --tag det_kl{0,1}` (GPU, hours),
then `python scripts/run_kl_compare.py --tag det_kl`.
Tables: `results/novelty_stats.csv`, `results/kl_compare_det_kl.csv` (deterministic, current) and `results/kl_compare.csv` (2026-10-28, superseded).

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
removes the cache-reuse problem of §1.

### The first attempt was not measurable, and we said so

The 2026-10-28 version of this comparison ran without deterministic kernels. Rerunning the
*identical* λ = 0 configuration shifted the headline AUROC by −0.028 ± 0.089 with per-seed swings to
0.18 — as large as the anchor effect it was trying to measure. That experiment was underpowered
against its own noise floor, and its numbers are superseded by everything below.
[`reproducibility.md`](reproducibility.md) has the root cause: nondeterministic backward-pass kernel
selection, not the quantized optimizer.

**Training is now bit-exact.** With `torch.use_deterministic_algorithms(True, warn_only=False)` and
a fixed cuBLAS workspace, three full pipeline runs on one seed agreed exactly — epoch curve,
selected epoch, saved LoRA weights tensor by tensor, every metric — at about 10% wall-clock cost.
Everything in this section is the rerun under that regime: 10 seeds, 3 epochs, identical splits,
λ = 0 at 985 s/seed and λ = 1 at 1223 s/seed, peak 8.05 GiB.

**Replicates are therefore unnecessary and were not run.** Under determinism two runs of a
configuration are the same run, so the replicate null is exactly zero. The seed-to-seed spread that
remains is split and initialization variance, and it is shared between λ = 0 and λ = 1 by
construction, which is what makes the paired test below a test of the objective alone.

### The forgetting curve

![forgetting](../results/figures/forgetting.png)

| | | λ = 0 | | λ = 1 | |
|---|---|---|---|---|---|
| **epoch** | | **overall AUROC** | **novel AUROC** | **overall AUROC** | **novel AUROC** |
| 0 (base model) | | 0.660 ± 0.146 | 0.690 | 0.660 ± 0.146 | 0.690 |
| 1 | | 0.814 ± 0.093 | 0.715 | 0.841 ± 0.093 | 0.567 |
| 2 | | 0.851 ± 0.084 | 0.736 | 0.857 ± 0.093 | 0.812 |
| 3 | | 0.819 ± 0.088 | 0.553 | 0.872 ± 0.059 | 0.708 |
| validation-selected | | 0.818 ± 0.072 | 0.722 | **0.856 ± 0.093** | **0.757** |

**The monotone decline reported on 2026-10-28 did not reproduce.** In the deterministic run λ = 0's
novel AUROC goes 0.690 → 0.715 → 0.736 → 0.553: up for two epochs, then down. The earlier claim of a
clean fall at every epoch was reading a trend out of nondeterministic noise, and it is withdrawn.

**λ = 1 moves around just as much, in the opposite order** — 0.567, then 0.812, then 0.708. Two curves
that cross twice over three epochs on 42 molecules are not two different behaviours; they are one
noisy quantity measured twice.

### What the statistics allow

Paired bootstrap over molecules, both sides always scored on the same resampled set:

| comparison | ΔAUROC on the novel subset | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| λ = 0, epoch 3 − base model | −0.137 | [−0.522, +0.176] | 0.779 |
| λ = 1, epoch 3 − base model | +0.019 | [−0.319, +0.338] | 0.446 |
| epoch 1, λ = 1 − λ = 0 | **−0.148** | **[−0.286, −0.038]** | 0.997 |
| epoch 2, λ = 1 − λ = 0 | +0.076 | [−0.013, +0.178] | 0.055 |
| epoch 3, λ = 1 − λ = 0 | +0.155 | [−0.256, +0.568] | 0.224 |
| selected epoch, λ = 1 − λ = 0 | +0.035 | [−0.107, +0.225] | 0.368 |

**The one interval that excludes zero says the anchors made novel-chemistry performance worse**, at
epoch 1. By epoch 2 the sign has flipped and by epoch 3 the interval is four times wider. A quantity
whose sign depends on where you stop training is not an effect.

On the full test fold, paired per seed at the validation-selected epoch: λ = 1 − λ = 0 is
**+0.038 AUROC** (paired t-test p = 0.127, 8 of 10 seeds better), with Brier 0.173 against 0.183 and
ECE identical at 0.094. Per epoch the overall difference is +0.027 (p = 0.174), +0.006 (p = 0.588)
and +0.053 (p = 0.054).

### Against the baseline that matters

| | overall AUROC | seeds better than RF |
|---|---|---|
| random forest (default) | 0.872 ± 0.066 | — |
| QLoRA λ = 1 | 0.856 ± 0.093 | 4/10 (Δ −0.016 ± 0.073) |
| QLoRA λ = 0 | 0.818 ± 0.072 | 2/10 (Δ −0.054 ± 0.080) |
| Qwen zero-shot | 0.651 ± 0.144 | — |

The anchors close roughly two thirds of the gap to the random forest and do not cross it. Both
QLoRA conditions remain worse calibrated than the forest on Brier.

## Verdict

**Is training reproducible?** Yes. Bit-exact, verified end to end on the full pipeline including
saved weights. This is the one unambiguous result of the exercise.

**Do KL anchors improve overall accuracy?** By +0.038 AUROC, on 8 of 10 seeds, p = 0.127.
**Directionally consistent and not statistically significant.** With training noise eliminated this
is now a clean measurement of a small effect, and the remaining uncertainty is test-set sampling: 10
scaffold splits of 631 molecules are not enough to resolve 0.038.

**Do they protect unfamiliar chemistry?** **Inconclusive.** +0.035 at the selected epoch with a 95%
interval of [−0.107, +0.225] that comfortably includes zero, and a per-epoch sign that flips from
significantly negative to positive. Nothing here supports the claim.

**Have we established that KL anchors prevent catastrophic forgetting?** **No.** We have not even
established the forgetting. The monotone decline that motivated the hypothesis vanished under
deterministic replication, so there is currently no reproducible effect for the anchors to prevent.

**Is the fine-tuned LLM competitive?** No. Random forest remains the stronger baseline at
0.872 ± 0.066 against λ = 1's 0.856 ± 0.093.

**Decision: stop here.** The anchors are a small, plausible, unproven improvement, and the
bottleneck is no longer GPU time or measurement noise — it is 42 novel molecules containing 6
positives. More training runs cannot fix that. Further KL work is not justified until the evaluation
set can resolve the effect size, and the project's next phase should go after data and chemistry
rather than the objective.

## Limitations

- 42 novel molecules, 23 chemical clusters, 6 positives. Every interval here is wide for that reason,
  and this is now the binding constraint on the whole question.
- Three epochs. λ = 1 was still improving at epoch 3 on overall AUROC, so its curve is truncated. We
  are not extending it, because the overall gain is not what the hypothesis was about.
- One λ per condition beyond zero; no learning-rate or LoRA-rank sweep, by design.
- The anchors are applied to decision prompts only, as in the paper. We never measured whether they
  preserve general chat behaviour, which is the paper's actual claim — only whether they preserve a
  chemical prior, which is ours.
- Determinism removes measurement noise, not sample limitation. Every interval above is still a
  statement about 631 molecules and 10 splits.
- Nothing here is a statement about TG2 inhibition in a cell or an animal. These are retrospective
  rankings of published assay values.
