# Project summary — state as of 2026-10-27

Start here. Each section links to the doc with the full numbers and the command to reproduce them.

## What was built
A fully local research environment on the user's PC: Windows 11 → WSL2 Ubuntu 22.04 → Python 3.12 → PyTorch 2.14.1 (CUDA 13.0) → Qwen3.5-9B in 4-bit. **No hosted model APIs are used anywhere.** Details and measured numbers: [`setup.md`](setup.md).

| | measured |
|---|---|
| GPU | RTX 5070 Ti, 15.89 GiB usable, compute capability 12.0 |
| Qwen3.5-9B inference | 7.21 GiB peak, 21.1 tok/s |
| QLoRA fine-tuning | 7.98 GiB peak, 428 s per TG2 training run |
| Headroom | ~8 GiB spare, so longer sequences and bigger batches are available |

## The research question
Can a ~9B local LLM, used in the "decision model" style the professor suggested (Jev), predict drug activity better than standard cheminformatics baselines?

## The answer so far: no, not on this data
All four approaches measured on **identical scaffold splits** of the same 305 TG2 molecules:

| approach | TG2 AUROC | vs fingerprint RF (paired) |
|---|---|---|
| **Morgan fingerprint + random forest** | **0.843 ± 0.088** | — the bar |
| Frozen Qwen embeddings + linear head | 0.839 ± 0.139 | −0.004, not significant |
| Jev-style option-logit readout, no training | 0.694 ± 0.191 | −0.150, clearly worse |
| LLM-as-Jev QLoRA fine-tune | 0.852 ± 0.128 | +0.009, **p = 0.80, not significant** |

On GDSC2 cancer drug response (unseen drugs, 20 seeds), adding drug names to the prompt looked promising at +0.053 Pearson, but at 20 seeds **p = 0.20 — not established** ([`qwen_probe.md`](qwen_probe.md)).

**Headline: on a few hundred molecules, a 9B LLM matches but does not beat Morgan fingerprints, however it is used.** This is a legitimate negative result and consistent with the wider literature on small-data cheminformatics.

## What *is* statistically established
1. **Fine-tuning works.** QLoRA lifts Qwen from AUROC 0.694 to 0.852, better on **10/10 seeds**, p = 0.002 ([`jev_qlora.md`](jev_qlora.md)). The Jev-style training objective is sound; the limit is the data.
2. **Qwen has genuine prior knowledge of TG2 chemistry.** With zero training it ranks potent above weak inhibitors at AUROC 0.69, well above chance.
3. **Qwen features are slightly worse at within-drug ranking** on GDSC2 (−0.014 per-drug Spearman, p = 0.0002). Small, but real.
4. **No method is well calibrated.** ECE is 0.14–0.24 everywhere, and fine-tuning does not improve it (p = 0.68). The worst split shows a fine-tuned model confidently wrong on an unfamiliar scaffold family.
5. **GDSC2 random splits are misleading.** Predicting each drug's average response already gives Pearson 0.83; the honest setting is unseen drugs, where the best model reaches 0.50.

## The clearest opening for new work
**Calibration.** It is the one unmet need across every method, and it is exactly what the professor's Jev reference is about: Jev's selling point is calibrated probabilities from its (undisclosed) RLCD training. Nothing tried here reproduces that. Concrete options: temperature scaling on a proper held-out set, ensembling over option orders and seeds, or conformal prediction for guaranteed coverage.

Other directions, in rough order of expected value:
1. **More data** — 305 molecules is the binding constraint. Merging ChEMBL TG2 bioactivity data (with careful assay filtering) would test whether the LLM scales better with data than the random forest does.
2. **Multi-task transfer** — one adapter trained on GDSC2 + TG2 together, testing whether cancer drug-response data transfers to TG2 potency. This is where a shared LLM representation could plausibly beat per-task models.
3. **Better experiment design** — leave-one-drug-out on GDSC2 (137 drugs) instead of random cold-drug folds; the current design cannot resolve effects smaller than ~0.05 Pearson or ~0.07 AUROC.

## Open questions for the professor
1. **TG2 IC50 units.** The source file gives bare numbers; µM is assumed, which sets the "potent" threshold at IC50 ≤ 1 µM. This changes label definitions but not any model comparison. Confirm against Wichka & Lai (CSBJ 2024).
2. **Which "Jev"?** TypeSafe AI's Jev is closed-weights with no paper ([`jev_research.md`](jev_research.md)). The reproducible substitute used here is the published LLM-as-Jev method on Qwen. Confirm this is the intended direction.
3. **Is a negative result acceptable** as a first-month outcome, framed as "fingerprint baselines remain hard to beat at this data scale, and here is what an LLM does and does not add"?

## Caveats on everything above
- Test folds are small (TG2: 24 ± 9 molecules; GDSC2 cold-drug: ~14 drugs). Per-seed spread is wide.
- The TG2 classification and regression tasks are two views of the same 305 molecules, not independent evidence.
- No hyperparameter tuning was done for any model; all use fixed sensible defaults.
- The repo's own TG2 Active/Inactive labels were replaced with a potency threshold because they were 90% positive ([`baselines.md`](baselines.md)).

## Reproducing everything
```bash
cd ~/projects/celiac-ai-research && source .venv/bin/activate
bash scripts/download_data.sh                      # GDSC2 + TG2
pytest                                             # 10 tests
python scripts/run_baselines.py                    # baselines.md
python scripts/embed_molecules.py                  # deterministic Qwen embeddings
python scripts/run_qwen_probe.py                   # qwen_probe.md
python scripts/run_jev_probe.py                    # jev_probe.md
python scripts/run_jev_qlora.py                    # jev_qlora.md  (~72 min)
```
Git is initialized but **nothing is committed yet** — the user deferred the first commit.
