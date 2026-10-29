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

## Selectivity, and why there is still no shortlist

A TG2 inhibitor for celiac disease is taken chronically by otherwise healthy people, and Factor
XIII-A — which crosslinks fibrin — is the bleeding-risk antitarget. Four human transglutaminases
plus the guinea-pig TG2 ortholog were pulled from ChEMBL and audited without being merged
([`selectivity.md`](selectivity.md)):

- **The data cannot be expanded where it matters.** TGM1 contributes 5 molecules the TG2 set does
  not already have, TGM3 contributes 1, FXIII-A 31. They are selectivity panels on the same
  compounds, not new chemistry: 51 of TGM1's 53 scaffolds are already present. Guinea-pig TG2 is the
  one real source of distant chemistry (41 molecules, 32 below 0.5 Tanimoto) and **contains no
  actives** — IC50 2.1 to 74 uM. The previous phase's recommendation to grow the novel stratum this
  way is withdrawn.
- **TGM1 and TGM3 are not modellable**: 5 and 1 potent molecules respectively.
- **Cross-reactivity is the norm.** Of the 132 TG2-potent molecules with paired FXIII-A data, 103
  also inhibit FXIII-A at 1 uM and only 29 are clean.
- **Selectivity cannot be predicted.** A scaffold-split random forest scored 0.974 on the selective
  class, which was caught as an artefact rather than reported: 11 of the 29 selective molecules come
  from one paper and only 3 of 98 scaffolds appear in both classes. Holding out whole papers instead
  of whole scaffolds, AUROC falls to **0.574, 95% CI [0.250, 0.864]** — chance.
- **Ranking by predicted TG2 potency says nothing about selectivity** (AUROC 0.454).

No selectivity-aware shortlist was produced, because producing one would imply a safety signal the
models do not have.

## Candidate screening

2,414 approved drugs scored. **99.2% fall below the 0.4 Tanimoto applicability floor where the model
ranks at chance**, so the nominal 35 hits are not candidates. One compound (telaprevir) survives
every filter but the conformal one, and is documented as a probe of the model's chemotype bias
rather than a drug candidate. Approved-drug repurposing is the wrong strategy for TG2 with this
training set ([`candidates.md`](candidates.md)). Everything there is a computational hypothesis
about an untested molecule.

## Covalent mechanism: the last experiment

Most published TG2 inhibitors are covalent, and the IC50 filter discards 990 of 1,583 TG2 records.
That discarded data was audited and tested ([`covalent.md`](covalent.md)):

- **kinact, Ki and kinact/Ki are three different quantities** and are kept separate from each other
  and from the IC50 labels. Only kinact/Ki is comparable across compounds.
- **288 molecules shrink to 174** across 6 papers and 70 scaffolds: ChEMBL stores the rate constant
  under three unit conventions and 118 rows carry none, losing two entire papers.
- **169 of 174 molecules share one warhead** (acrylamide). A model cannot be keying on the reactive
  group — but there is also no mechanism contrast to classify, so mechanism classification was not
  attempted.
- **Covalent efficiency does not predict across laboratories.** Random forest beats predicting a
  constant by 0.046 log RMSE, 95% CI [-0.101, +0.011], and ranks within a new paper at Spearman
  0.120. The between-paper offset (0.50 log) rivals the within-paper spread (0.57 log).
- **The scaffold split was optimistic here too** (RMSE 0.489 against 0.598 across papers) — the
  second time in this project that a split correct for one question would have produced a
  publishable-looking number for another.

## Where things stand

| | |
|---|---|
| TG2 molecules | 631 (from 305), PubChem + ChEMBL, sources verified to agree |
| Experiments | baselines, tuned baselines, frozen-embedding probe, zero-shot Jev readout, QLoRA fine-tune, calibration, generalization, uncertainty, screening |
| Artifacts | 30 LoRA adapters (5.1 GB), per-seed and per-epoch predictions, 31 result tables, 5 figures |
| Tests | 27 passing |
| Git | 10 commits; nothing pushed |

## Documents
- [`report.md`](report.md) — the professor-facing write-up, with figures and references
- [`data.md`](data.md) — provenance, unit validation, leakage audit. **Read before trusting a number**
- [`baselines.md`](baselines.md) — fingerprint baselines, tuned and default
- [`calibration.md`](calibration.md) — why post-hoc calibration fails, and where it doesn't
- [`generalization.md`](generalization.md) — performance against chemical similarity
- [`forgetting.md`](forgetting.md) — does fine-tuning destroy the base model's chemical prior?
- [`selectivity.md`](selectivity.md) — TG2 against FXIII-A: can we predict selectivity? **No**
- [`covalent.md`](covalent.md) — kinact/Ki: can we predict covalent efficiency? **Not across labs**
- [`second_target.md`](second_target.md) — do the evaluation failures reproduce on CDK2? **Three of four**
- [`screening_prep.md`](screening_prep.md) — TG2 data discovery, unit recovery and screening feasibility
- [`experimental_plan.md`](experimental_plan.md) — FXIII-A selectivity evidence and a prospective study design
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
