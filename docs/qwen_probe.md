# Approach C: frozen Qwen3.5-9B features + simple heads (run 2026-10-27)

Reproduce:
```bash
python scripts/embed_molecules.py        # caches data/processed/qwen_{tg2,gdsc2_drugs}.npz (~1 min)
python scripts/run_qwen_probe.py         # TG2 (10 seeds) + GDSC2 cold-drug (5 seeds), ~8 min
```
Per-seed rows: `results/qwen_probe_{tg2,gdsc2}.csv`.

## Setup
- **Input text:** `"Molecule SMILES: <smiles>"` into the frozen 4-bit Qwen3.5-9B. Mean-pooled hidden states from layers 8 / 16 / 24 / 32 (4096-d each).
- **TG2 heads:** standardized features → logistic regression (classification) or ridge (regression). Layer and regularization are chosen on the **valid** split per seed (Brier / MSE). Variants: `qwen` alone, and `qwen+fp` (concatenated with 2048-bit Morgan).
- **GDSC2 cold-drug:** the drug's Qwen embedding is reduced to 32-d by PCA fit on training drugs, then combined with expression PCA-256 and fed to the same XGBoost as the baseline. Layer chosen on valid.
- **Identical splits and seeds as `docs/baselines.md`**, so every comparison is **paired** (same test molecules).

## Results

### TG2 classification (potent ≤ 1 µM), 10 scaffold splits
| model | AUROC | MCC | ECE | Brier |
|---|---|---|---|---|
| fingerprint RF (baseline) | **0.843 ± 0.088** | **0.578 ± 0.240** | **0.196 ± 0.073** | **0.150 ± 0.067** |
| Qwen + linear | 0.839 ± 0.139 | 0.540 ± 0.351 | 0.222 ± 0.066 | 0.154 ± 0.082 |
| Qwen + FP + linear | 0.832 ± 0.143 | 0.540 ± 0.324 | 0.239 ± 0.076 | 0.161 ± 0.085 |

Paired vs RF: Qwen ΔAUROC −0.004 ± 0.120 (better on 4/10 seeds), ΔECE +0.027 (worse).

### TG2 regression (pIC50), 10 scaffold splits
| model | RMSE | R² | Spearman |
|---|---|---|---|
| fingerprint RF (baseline) | **0.776 ± 0.163** | 0.353 ± 0.364 | **0.644 ± 0.206** |
| Qwen + ridge | 0.816 ± 0.152 | 0.303 ± 0.300 | 0.627 ± 0.232 |
| Qwen + FP + ridge | 0.778 ± 0.153 | **0.373 ± 0.265** | 0.620 ± 0.208 |

Paired vs RF: Qwen ΔSpearman −0.017 ± 0.124 (4/10), Qwen+FP ΔRMSE +0.002 ± 0.082 (5/10).

### GDSC2 cold-drug, 5 seeds
| drug features | RMSE | Pearson | per-drug Spearman |
|---|---|---|---|
| Morgan fingerprint (baseline) | **2.351 ± 0.475** | **0.464 ± 0.202** | **0.646 ± 0.055** |
| Qwen embedding (PCA-32) | 2.415 ± 0.499 | 0.432 ± 0.175 | 0.622 ± 0.059 |

Paired: ΔPearson −0.032 ± 0.075 (better on 1/5), Δper-drug Spearman −0.025 ± 0.017 (1/5).

### GDSC2 cold-drug with drug names in the prompt (rerun 2026-10-27), 5 seeds
Three prompts for the same 137 drugs, re-embedded in one pass (`scripts/embed_molecules.py`):
`"Molecule SMILES: …"`, `"Drug: <name>. SMILES: …"`, `"Drug: <name>"`. About half the names are lab codes (e.g. AZD7762, MK-2206), the rest are common names (e.g. Gefitinib, Docetaxel).

| drug features | RMSE | R² | Pearson | per-drug Spearman |
|---|---|---|---|---|
| Morgan fingerprint (baseline) | 2.351 ± 0.475 | 0.108 ± 0.286 | 0.464 ± 0.202 | **0.646 ± 0.055** |
| Qwen: SMILES | 2.359 ± 0.316 | 0.085 ± 0.272 | 0.460 ± 0.144 | 0.618 ± 0.056 |
| **Qwen: name + SMILES** | **2.250 ± 0.393** | **0.183 ± 0.210** | **0.502 ± 0.153** | 0.635 ± 0.066 |
| Qwen: name only | 2.309 ± 0.432 | 0.145 ± 0.217 | 0.472 ± 0.131 | 0.621 ± 0.068 |

Paired vs fingerprint, name + SMILES: ΔPearson **+0.038 ± 0.066** (better on 3/5), ΔRMSE **−0.101 ± 0.199** (better on 4/5), Δper-drug Spearman −0.012 ± 0.016 (2/5).

**Read with caution:** the SMILES-only embeddings, re-extracted in a different batch, moved from Pearson 0.432 to 0.460 with no change in method. So **run-to-run noise is about ±0.03 Pearson**, the same size as the name + SMILES gain. The direction is consistent (the name helps, and the name plus structure helps most), but 5 seeds can't establish it. Rerun at 20 seeds below.

**Where that instability comes from** (measured 2026-10-27): the embeddings themselves are near-identical across batch sizes — median |Δ| ≈ 1e-3 and cosine ≥ 0.99998 — with one or two outlier dimensions differing by up to 0.25. Padding is not the cause (mixed-length batches are no worse than equal-length ones); it is GPU reduction order. The amplification happens **downstream**: PCA → XGBoost → picking the best layer on a valid fold of only ~14 drugs. `scripts/embed_molecules.py` now defaults to `--batch 1`, which is bit-identical on repeat (verified, max |Δ| = 0.0) at 97 s for all 442 molecules.

**Scope caveat:** names help only for *known* drugs whose literature Qwen saw in pretraining. A genuinely new TG2 candidate has no name, so this gain wouldn't transfer to novel-molecule discovery. It's useful for repurposing known drugs.

### Settled at 20 seeds with deterministic embeddings (2026-10-27)
Re-embedded at `--batch 1`, baselines and probe both rerun at 20 cold-drug seeds.

| drug features | RMSE | R² | Pearson | per-drug Spearman |
|---|---|---|---|---|
| Morgan fingerprint | 2.382 ± 0.450 | 0.124 ± 0.254 | 0.447 ± 0.195 | **0.647 ± 0.034** |
| Qwen: SMILES | 2.407 ± 0.486 | 0.088 ± 0.330 | 0.442 ± 0.213 | 0.629 ± 0.038 |
| **Qwen: name + SMILES** | **2.296 ± 0.426** | **0.183 ± 0.231** | **0.499 ± 0.177** | 0.633 ± 0.042 |
| Qwen: name only | 2.312 ± 0.319 | 0.171 ± 0.246 | 0.484 ± 0.137 | 0.628 ± 0.040 |

Paired tests, name + SMILES vs fingerprint (`python scripts/compare.py results/qwen_probe_gdsc2.csv 'qwen_name+smiles' results/baselines_gdsc2.csv xgb --split cold_drug`):

| metric | mean Δ | 95% CI | wins | t-test p | sign p |
|---|---|---|---|---|---|
| RMSE (inverted) | +0.086 | [−0.071, +0.242] | 14/20 | 0.296 | 0.115 |
| R² | +0.059 | [−0.061, +0.178] | 14/20 | 0.349 | 0.115 |
| Pearson | +0.053 | [−0.024, +0.130] | 12/20 | 0.197 | 0.503 |
| per-drug Spearman | **−0.014** | [−0.020, −0.008] | 4/20 | **0.0002** | **0.012** |

**Verdict: the drug-name advantage is NOT established.** The direction held across both runs and all four overall metrics, but every confidence interval includes zero and nothing reaches p < 0.05. Quadrupling the seeds did not shrink the per-seed spread (±0.18 Pearson), because the variance comes from *which 14 drugs land in the test fold*, not from the random seed. With only 137 drugs this design cannot resolve a +0.05 Pearson effect; it would need leave-one-drug-out or nested cross-validation over all 137.

**One effect is significant, and it goes against Qwen:** every Qwen variant is slightly but reliably worse at ranking cell lines *within* a drug (−0.014 to −0.019, p ≈ 0.0002). Plausible reason: the 32-d PCA of a Qwen embedding is a coarser drug descriptor than a 2048-bit fingerprint, so XGBoost leans harder on the expression features and loses some drug-specific interaction. The effect is small enough to be of little practical consequence.

## Interpretation
1. **Frozen Qwen embeddings of SMILES are about as informative as Morgan fingerprints, but not better.** Every paired difference is within noise, and most lean slightly toward fingerprints. Combining both doesn't help either.
2. **Middle layers work best:** layer 16 was chosen in 8/10 TG2 classification seeds. On GDSC2 the choice is unstable, since its valid fold has only ~14 drugs.
3. **Calibration is not better.** Qwen heads are slightly worse on ECE, so frozen features alone don't deliver the "calibrated decisions" Jev advertises.
4. **Why this is plausible:** a general LLM reads SMILES as a string of characters. Its real advantage is **world knowledge** (drug names, mechanisms, targets), which this prompt doesn't use.

This is a negative result for approach C as configured, and a useful one. It doesn't rule out Qwen. It says SMILES-only frozen embeddings aren't the way in.

## Most promising next experiments (cheap, in order)
1. ~~Add drug names to the prompt~~: **done and settled at 20 seeds — not significant.** See above. Don't claim this gain without a leave-one-drug-out design.
2. **Jev-style option-logit probe on TG2** (no training): ask Qwen "potent (IC50 ≤ 1 µM) or not?" and read P(option) from the logits, per LLM-as-Jev. This tests the professor's Jev idea directly, and its ECE/Brier compare against the table above.
3. Only if (1) or (2) shows a gain: QLoRA fine-tuning (approach D).

Not tried: other pooling (last token), other prompts, nonlinear heads on embeddings, or a 4-bit vs bf16 feature comparison (bf16 doesn't fit in 16 GB for 9B).
