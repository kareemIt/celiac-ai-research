# Project summary — state as of 2026-10-29

Start here. [`report.md`](report.md) is the full write-up for the professor; this page is the index
and the short version.

## The question
Can a locally-run ~9B language model, used as a Jev-style decision model, predict TG2 inhibitory
potency better than standard cheminformatics — and if not, does it add anything they cannot?

## The answer

**No, and probably yes — but not in the way expected.**

On 631 TG2 molecules over 10 identical scaffold splits:

| model | AUROC | ECE | Brier |
|---|---|---|---|
| **random forest, tuned per seed** | **0.898 ± 0.057** | 0.180 | 0.125 |
| XGBoost | 0.880 ± 0.066 | 0.148 | 0.128 |
| 3-model ensemble | 0.878 ± 0.063 | 0.148 | **0.123** |
| random forest, default | 0.872 ± 0.066 | 0.158 | 0.131 |
| Qwen QLoRA fine-tune | 0.850 ± 0.058 | 0.206 | 0.182 |
| Qwen zero-shot Jev readout | 0.651 ± 0.144 | 0.158 | 0.229 |

**The fine-tuned LLM is significantly worse than the tuned random forest**: ΔAUROC −0.049,
95% CI [−0.076, −0.021], p = 0.007, better on only 2 of 10 seeds. Fine-tuning itself works
emphatically (+0.199 AUROC over zero-shot, 10/10 seeds, p = 0.001) — it just does not get there.

## What is statistically established

1. **Tuned fingerprint random forest is the best model** at AUROC 0.898, and the LLM does not reach it.
2. **Fine-tuning works**: +0.199 AUROC over the zero-shot readout, every seed, p = 0.001.
3. **Random splits on this dataset are invalid**: 38.6% of random-split test molecules have a
   training molecule with an *identical* fingerprint ([`data.md`](data.md)).
4. **Below 0.4 Tanimoto to the training set, fingerprint models rank at chance**
   ([`generalization.md`](generalization.md)).
5. **Post-hoc calibration fails under scaffold shift** for every fingerprint model, and the cause is
   measurable — the fitted temperature swings from 0.4 to 3.7 across folds. A random-split control
   confirms the mechanism ([`calibration.md`](calibration.md)).
6. **Except for the fine-tuned LLM**, where temperature scaling does work (ΔECE −0.029, p = 0.010),
   because it is overconfident in the same direction on every fold.
7. **Conformal prediction holds its coverage under scaffold shift** (0.922 against a 0.90 target) by
   widening its sets instead ([`uncertainty.md`](uncertainty.md)).
8. **Abstaining on low confidence cuts error 15.5% → 6.5%; abstaining on low similarity does not**
   help at all.

## The hypothesis worth chasing

On the 42 molecules with no training analog above 0.5 Tanimoto:

| | novel only | all test |
|---|---|---|
| random forest | 0.794 | 0.892 |
| **Qwen zero-shot** | **0.807** | 0.663 |
| Qwen QLoRA | **0.442** | 0.808 |

**Both halves of that table were re-examined on 2026-10-28 and only one survived**
([`forgetting.md`](forgetting.md)). Those are pooled predictions over overlapping folds; scored one
row per molecule, the zero-shot model is *behind* the random forest (0.764 vs 0.782, difference
−0.019, 95% CI [−0.257, +0.192]), and it leads only at cutoffs where the subset holds one or three
positive molecules. The QLoRA collapse does survive the molecule-level bootstrap (−0.368 against the
base model, CI [+0.013, +0.658] on the difference) but not a per-seed test (p = 0.72, 2/6 seeds).
The live question is therefore narrower: does fine-tuning cost deep-novelty performance, and do the
LLM-as-Jev KL anchors prevent it? Both were tested with the anchors implemented, a matched
epoch-0 control inside every run, and — after a first attempt proved unmeasurable — **bit-exact
deterministic training** ([`reproducibility.md`](reproducibility.md)):

| epochs of QLoRA | overall AUROC (lambda 0 / 1) | novel AUROC (lambda 0) | novel AUROC (lambda 1) |
|---|---|---|---|
| 0 (base model) | 0.660 | 0.690 | 0.690 |
| 3 | 0.819 / 0.872 | 0.553 | 0.708 |
| validation-selected | 0.818 / **0.856** | 0.722 | **0.757** |

**Training is now reproducible** — three full runs on a seed agree exactly, including saved weights.
**KL regularization improved overall AUROC by 0.038 and the improvement is not significant**
(8/10 seeds, p = 0.127). **The novel-chemistry improvement is inconclusive**: +0.035, 95% CI
[-0.107, +0.225], with a per-epoch sign that flips from significantly negative at epoch 1 to positive
at epoch 3. **We have not established that KL anchors prevent catastrophic forgetting**, and the
monotone decline that motivated the hypothesis did not survive deterministic replication.
**Random forest remains the stronger baseline** (0.872 +/- 0.066 vs 0.856 +/- 0.093).
KL experiments are stopped: the limit is 42 novel molecules with 6 positives, not GPU time.

## Candidate screening

2,414 approved drugs scored. **99.2% fall below the 0.4 Tanimoto applicability floor where the model
ranks at chance**, so the nominal 35 hits are not candidates. One compound (telaprevir) survives
every filter but the conformal one, and is documented as a probe of the model's chemotype bias
rather than a drug candidate. Approved-drug repurposing is the wrong strategy for TG2 with this
training set ([`candidates.md`](candidates.md)). Everything there is a computational hypothesis
about an untested molecule.

## Where things stand

| | |
|---|---|
| TG2 molecules | 631 (from 305), PubChem + ChEMBL, sources verified to agree |
| Experiments | baselines, tuned baselines, frozen-embedding probe, zero-shot Jev readout, QLoRA fine-tune, calibration, generalization, uncertainty, screening |
| Artifacts | 30 LoRA adapters (5.1 GB), per-seed and per-epoch predictions, 31 result tables, 5 figures |
| Tests | 19 passing |
| Git | 8 commits; nothing pushed |

## Documents
- [`report.md`](report.md) — the professor-facing write-up, with figures and references
- [`data.md`](data.md) — provenance, unit validation, leakage audit. **Read before trusting a number**
- [`baselines.md`](baselines.md) — fingerprint baselines, tuned and default
- [`calibration.md`](calibration.md) — why post-hoc calibration fails, and where it doesn't
- [`generalization.md`](generalization.md) — performance against chemical similarity
- [`forgetting.md`](forgetting.md) — does fine-tuning destroy the base model's chemical prior?
- [`reproducibility.md`](reproducibility.md) — why two identical runs disagreed, and what fixed it
- [`uncertainty.md`](uncertainty.md) — ensembles, conformal, selective prediction
- [`candidates.md`](candidates.md) — screening methodology and the shortlist
- [`setup.md`](setup.md), [`jev_research.md`](jev_research.md), [`jev_probe.md`](jev_probe.md),
  [`jev_qlora.md`](jev_qlora.md), [`qwen_probe.md`](qwen_probe.md), [`research_plan.md`](research_plan.md)

## Open questions for the professor
1. ~~TG2 IC50 units~~ — **settled**: µM, confirmed against the source paper.
2. **Which "Jev"?** TypeSafe AI's model is closed-weights with no paper; the reproducible substitute
   used here is the published LLM-as-Jev method on Qwen.
3. **Is a negative result acceptable** as the month-1 outcome? The evidence is clean and the
   methodology is sound; the headline is that fingerprints remain the better model at this data scale.
4. **Which direction next?** The novel-chemistry hypothesis was chased to the end and came back
   negative: with training made bit-exact, neither the forgetting nor the anchors' prevention of it
   is statistically established, and the binding constraint is 6 positive molecules below 0.5
   Tanimoto ([`forgetting.md`](forgetting.md)). The recommended next phase goes after data and
   chemistry rather than the objective — low-similarity actives from related transglutaminases, a
   chemically appropriate screening library (covalent fragments rather than approved drugs), and
   selectivity against FXIII-A (§9 of [`report.md`](report.md)).
