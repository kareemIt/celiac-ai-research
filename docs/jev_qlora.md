# Approach D: LLM-as-Jev QLoRA fine-tune, TG2 potency (run 2026-10-27)

Reproduce: `python scripts/run_jev_qlora.py` (~72 min: 10 seeds × ~7 min). Per-seed rows: `results/jev_qlora_tg2.csv`. Log: `results/jev_qlora_tg2.log`.

## Method
- Same prompt and readout as the zero-shot probe (`jev_probe.md`), SMILES + name variant: two numbered options, prefill `Best answer: [`, P(potent) from the "1"/"2" logits, averaged over both option orders.
- **Training objective:** the LLM-as-Jev tree loss. For two single-digit options this is exactly cross-entropy over the logits of "1" vs "2" at the answer position. Option order is shuffled per example. KL anchors are omitted (the paper's λ=0 ablation), since we don't need general chat ability preserved.
- 4-bit NF4 Qwen3.5-9B, frozen, + LoRA r=16, α=32, dropout 0.05 on all linear layers (43.3M trainable). PagedAdamW8bit, lr 1e-4 (as in the paper), batch 1 × accumulation 8, gradient checkpointing.
- Per seed: fresh adapter, train on that seed's train fold (~240 molecules) for up to 4 epochs, keep the epoch with the best **valid** Brier, score the **test** fold once. Same 10 scaffold splits as every other TG2 result.
- Cost: **428 s/seed, peak VRAM 7.98 GiB.**

## Results (10 scaffold seeds, mean ± sd)
| model | AUROC | AUPRC | MCC | ECE | Brier |
|---|---|---|---|---|---|
| prior | 0.500 | 0.545 ± 0.161 | 0.000 | **0.140 ± 0.102** | 0.254 ± 0.010 |
| fingerprint RF | 0.843 ± 0.088 | 0.891 ± 0.083 | **0.578 ± 0.240** | 0.196 ± 0.073 | **0.150 ± 0.067** |
| Jev zero-shot (Qwen, no training) | 0.694 ± 0.191 | 0.724 ± 0.168 | 0.379 ± 0.321 | 0.225 ± 0.089 | 0.225 ± 0.019 |
| **Jev QLoRA (Qwen fine-tuned)** | **0.852 ± 0.128** | **0.898 ± 0.093** | 0.570 ± 0.256 | 0.241 ± 0.092 | 0.178 ± 0.119 |

Paired (same test molecules):
- **QLoRA vs RF:** ΔAUROC +0.009 ± 0.104 (better on 4/10), ΔMCC −0.008 ± 0.223 (3/10), ΔECE +0.045 (worse; 4/10 better), ΔBrier +0.028 (worse; 4/10 better).
- **QLoRA vs zero-shot:** ΔAUROC **+0.158 ± 0.119, better on 10/10**. ΔBrier −0.047 (8/10).

Per seed (test AUROC): 1.000, 0.881, 0.729, 0.943, 0.862, 0.975, 1.000, 0.733, 0.655, 0.742. Seeds 0 and 6 have 6 and 10 test molecules, so their perfect scores carry little weight. Seed 8 is the worst case: AUROC 0.655, Brier 0.468, meaning confidently wrong on that scaffold family.

Best epoch: 1 (4 seeds), 2 (3), 3 (1), 4 (2). The model fits ~240 molecules within about one epoch, and longer training only helps on some splits.

## Interpretation
1. **Fine-tuning works and matters.** It lifts Qwen from AUROC 0.69 (zero-shot) to 0.85, better on every seed. The Jev-style decision readout is a valid way to train an LLM as a classifier on this data.
2. **It ties the fingerprint random forest; it doesn't beat it.** +0.009 AUROC with ±0.10 per-seed spread is noise. With 10 seeds and ~24 test molecules each, this experiment can't separate a difference smaller than about ±0.07.
3. **Calibration is worse than RF**, and seed 8 shows the failure mode: an overconfident fine-tuned LLM on an unfamiliar scaffold. Jev's selling point, calibrated decisions, does **not** come from this objective alone. It would need calibration-aware training (Jev's RLCD is undisclosed), temperature scaling on a larger held-out set, or ensembling.
4. **Cost:** ~7 min/seed on the GPU vs ~1 s for the random forest. With no accuracy gain on 305 molecules, RF stays the practical choice for TG2 screening today.

## Where this leaves the project
The four approaches in `research_plan.md` have now all been measured on identical splits:

| approach | TG2 AUROC | verdict |
|---|---|---|
| fingerprint RF (baseline) | 0.843 | the bar |
| A/C: frozen Qwen features + head | 0.839 | ties |
| Jev-style zero-shot readout | 0.694 | real prior knowledge, not competitive |
| D: LLM-as-Jev QLoRA | 0.852 | ties the bar, beats zero-shot on 10/10 |

**The honest summary: on 305 molecules, a 9B LLM does not beat Morgan fingerprints at TG2 potency, however it is used.** That is a legitimate, publishable negative result for a small-data cheminformatics task, and it matches the general finding that fingerprint baselines are hard to beat below ~10³ molecules.

Options that could still change the picture, in rough order of expected value:
1. **More data.** 305 molecules is the binding constraint. ChEMBL has additional TG2 bioactivity data; merging it (carefully, with assay-type filtering) could move all methods and would test whether the LLM scales better than RF with data size.
2. **Calibration-focused work.** The clearest unmet need, and closest to the professor's Jev motivation: no method here achieves good ECE. Temperature scaling, ensembling over option orders and seeds, or conformal prediction would be a focused contribution.
3. **Multi-task / transfer.** Train one adapter on GDSC2 + TG2 together; test whether cancer drug-response data transfers to TG2 potency. This is where an LLM's shared representation could plausibly beat per-task RF.
4. **More seeds** on any comparison before claiming a difference: the current design cannot resolve gaps smaller than ~0.07 AUROC.
4. **Cost:** ~7 min/seed on the GPU vs ~1 s for RF. With no accuracy gain on this small dataset, RF remains the practical choice for TG2 screening today