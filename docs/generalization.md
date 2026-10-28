# Chemical generalization: where TG2 prediction stops working (run 2026-10-27)

Reproduce: `python scripts/run_generalization.py --seeds 10 --data combined`
Per-molecule predictions: `results/generalization_preds.csv`; per-bin scores: `results/generalization_bins.csv`.

## The question
A scaffold split guarantees no Murcko core is shared between train and test, but
[`data.md`](data.md) showed that 51% of scaffold-split test molecules still have a training molecule
within 0.7 Tanimoto. "Generalizes to new scaffolds" and "generalizes to new chemistry" are not the
same claim. This experiment measures the second one.

Every test molecule is labelled with its **maximum Tanimoto similarity to any training molecule** on
2048-bit Morgan fingerprints, and scored inside similarity bands. Bands are pooled across the 10
scaffold seeds because a single seed puts only a handful of molecules in the far bands.

## Performance collapses as molecules get less familiar

Random forest (fixed tuned configuration), pooled over 10 seeds:

| max Tanimoto to train | n | positive rate | AUROC | MCC | Brier |
|---|---|---|---|---|---|
| 0.0 – 0.4 | 21 | 0.05 | **0.450** | 0.000 | 0.066 |
| 0.4 – 0.5 | 30 | 0.27 | 0.801 | 0.489 | 0.155 |
| 0.5 – 0.6 | 101 | 0.26 | 0.903 | 0.634 | 0.134 |
| 0.6 – 0.7 | 151 | 0.59 | **0.931** | 0.753 | 0.104 |
| 0.7 – 1.0 | 319 | 0.72 | 0.832 | 0.591 | 0.134 |

**Below 0.4 Tanimoto the model ranks at chance.** The headline AUROC of ~0.89 is earned almost
entirely on molecules that have a reasonably close training analog. XGBoost and logistic regression
show the same shape (0.450 and 0.550 in the lowest band).

Two cautions on reading this table. The lowest band holds 21 predictions with a single positive, so
its AUROC is a weak estimate — the honest statement is "no evidence of skill there", not "exactly
0.45". And the positive rate climbs steeply with similarity, because the known potent inhibitors
cluster into a few analog series; that is why per-band **accuracy** is misleading and AUROC and MCC
are reported instead.

## Strict novelty subset

Restricting to test molecules with **no training analog above 0.5 Tanimoto** — 42 distinct molecules,
51 predictions across seeds, 18% positive:

| model | AUROC, novel only | AUROC, all test | drop |
|---|---|---|---|
| random forest | 0.794 | 0.892 | −0.098 |
| XGBoost | 0.696 | 0.894 | −0.198 |
| logistic regression | 0.672 | 0.880 | −0.208 |
| Qwen zero-shot | **0.807** | 0.663 | **+0.144** |

Qwen's zero-shot readout is the only model that does not degrade — it is the *best* model on the
novel subset, having been the worst overall. That is the first sign in this project of an LLM
advantage that fingerprints do not have, and it is consistent with the mechanism you would expect:
a fingerprint model has nothing to match against for an unfamiliar scaffold, while a language model
can draw on chemical knowledge from pretraining.

**It is not yet a finding.** 42 molecules, pooled across overlapping test folds, is far too little to
support the claim, and the comparison was not pre-registered. It is listed in
[`summary.md`](summary.md) as the single most promising hypothesis to test properly.

## Does the model know when it is out of its depth?

Spearman correlation between a test molecule's similarity to training and the model's confidence
(|p − 0.5|, so higher means more decisive):

| model | ρ(similarity, confidence) | p |
|---|---|---|
| random forest | **0.238** | <0.0001 |
| XGBoost | 0.144 | 0.0003 |
| Qwen zero-shot | 0.115 | 0.004 |
| logistic regression | 0.070 | 0.08 |

The correlation is positive and significant for three of four models: they *are* less confident on
unfamiliar molecules. The effect is real but weak — ρ ≈ 0.24 means similarity explains under 6% of
the variance in confidence, so confidence alone is not a usable novelty detector. Pairing a
prediction with its nearest-neighbour similarity, which costs nothing, is much more informative than
the probability by itself.

## Consequence for screening
An applicability domain of **max Tanimoto ≥ 0.4** to the training set is the floor for taking a
prediction seriously, and ≥ 0.5 is where the model is genuinely reliable. [`candidates.md`](candidates.md)
applies that rule, and it is the reason 99% of approved drugs are excluded from consideration.
