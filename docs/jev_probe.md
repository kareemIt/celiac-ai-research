# Jev-style option-logit probe: TG2 potency (run 2026-10-27)

Reproduce: `python scripts/run_jev_probe.py` (about 3.5 min of scoring, cached in `results/jev_tg2_scores.csv`. Use `--rescore` to redo).
Readout: `src/models/jev_readout.py`, following LLM-as-Jev (arXiv 2610.02076). See `jev_research.md`.

## Method
Frozen 4-bit Qwen3.5-9B, thinking off, **no training**. Prompt:
```
Compound SMILES: <smiles>
[Compound name: <name>]            # "smiles+name" variant only
Question: Is this compound a potent inhibitor of human tissue transglutaminase 2 (TG2)?
Options:
1. Potent TG2 inhibitor (IC50 <= 1 uM)
2. Not potent (IC50 > 1 uM)
Answer with the number of the best option.
```
- The assistant turn is prefilled with `Best answer: [`. Each option's score is log P("k]"), and a softmax over the two gives P(potent). Sanity check: Qwen's own greedy continuation is `1]`, which agrees with the scores.
- **Each molecule is scored with both option orders and the log-odds averaged**, to cancel position bias (measured below).
- `+platt`: a one-feature logistic calibration of that score, fit on each seed's train fold. It can't change AUROC; it's meant to fix calibration.
- Same 305 molecules and 10 scaffold splits as `baselines.md`, compared seed by seed.

## Results

| model | AUROC | AUPRC | MCC | ECE | Brier |
|---|---|---|---|---|---|
| prior (always 52%) | 0.500 | 0.545 ± 0.161 | 0.000 | **0.140 ± 0.102** | 0.254 ± 0.010 |
| **fingerprint RF** | **0.843 ± 0.088** | **0.891 ± 0.083** | **0.578 ± 0.240** | 0.196 ± 0.073 | **0.150 ± 0.067** |
| Jev readout, SMILES | 0.674 ± 0.204 | 0.728 ± 0.173 | 0.173 ± 0.192 | 0.192 ± 0.121 | 0.236 ± 0.023 |
| ↳ + Platt | 0.674 ± 0.204 | 0.728 ± 0.173 | 0.217 ± 0.374 | 0.235 ± 0.054 | 0.216 ± 0.048 |
| Jev readout, SMILES + name | 0.694 ± 0.191 | 0.724 ± 0.168 | 0.379 ± 0.321 | 0.225 ± 0.089 | 0.225 ± 0.019 |
| ↳ + Platt | 0.694 ± 0.191 | 0.724 ± 0.168 | 0.299 ± 0.305 | 0.242 ± 0.062 | 0.213 ± 0.045 |

Paired vs RF: ΔAUROC −0.15 to −0.17 (better on 1/10 seeds), ΔBrier +0.06 to +0.09 (worse on 9–10/10).

### Position bias (all 305 molecules, one fixed order)
| prompt | potent listed | mean P(potent) | AUROC |
|---|---|---|---|
| SMILES | first | 0.641 | 0.603 |
| SMILES | second | 0.535 | 0.725 |
| SMILES + name | first | 0.546 | 0.598 |
| SMILES + name | second | 0.530 | 0.666 |

The true potent rate is 0.525. Listing "potent" first inflates P(potent) and *also* changes the ranking (AUROC 0.60 vs 0.73), so the bias isn't just a constant offset. The order-averaged score is reported above. Picking the better order after seeing these numbers would be test-set tuning.

## Interpretation
1. **Qwen has some real prior knowledge of TG2 potency.** With zero training it ranks potent above weak inhibitors at AUROC ≈ 0.67–0.69 on unseen scaffolds, clearly above chance. The compound name adds a little.
2. **Zero-shot is not competitive.** A fingerprint random forest trained on about 240 molecules is far better (0.84) and better calibrated by Brier score.
3. **"Calibrated decisions" don't come for free.** Raw Jev-style probabilities have ECE ≈ 0.19–0.23 (no better than RF), and Platt scaling on small train folds doesn't fix it. Jev's calibration claims come from its RLCD *training*, which this zero-shot readout lacks. That matches the LLM-as-Jev paper, where fine-tuning provides the targeted gains.
4. **Option order matters**, as `jev_research.md` warned (results "sensitive to option names"). Any Jev-style evaluation should average or shuffle orders.

## Next step this points to
**Approach D, LLM-as-Jev fine-tuning:** QLoRA on each seed's train fold using the option-suffix objective (with shuffled option order), evaluated on the same test folds. The QLoRA infrastructure is validated (7.66 GiB peak). Estimated cost is roughly 5–10 min per seed. This tests whether Qwen's prior (0.67–0.69 AUROC zero-shot) plus TG2 training can beat RF's 0.84.
