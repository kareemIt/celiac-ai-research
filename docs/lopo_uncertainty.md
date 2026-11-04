# Abstention under publication shift: coverage holds, usefulness does not (run 2026-11-04)

**Conformal prediction can be made honest in this regime, but only by declaring most predictions
unreliable — and the predictions it keeps are still wrong a quarter to a third of the time.** The
method that works is not the one [`uncertainty.md`](uncertainty.md) validated: calibrating on random
training molecules **under-covers badly** when a whole publication is held out (0.784 against a 0.90
target, and 0.601 against 0.80). Calibrating on whole held-out training *publications* restores it
(0.914 and 0.810), a difference of **+0.113 coverage [+0.078, +0.148], p = 2.5 × 10⁻⁸**.

Reproduce: `python scripts/run_lopo_conformal.py` then `python scripts/analyse_lopo_conformal.py`.
One RTX 5070 Ti, about 25 minutes. Results: `results/lopo_conformal{,_summary,_stats}.csv`,
logs `results/lopo_conformal{,_analysis}.log`. Tests: `tests/test_lopo_conformal.py` (13 cases).

## What was run, and what the held-out labels never touched

[`uncertainty.md`](uncertainty.md) showed conformal prediction holding 0.922 coverage against a 0.90
target under a **scaffold** split. [`sota_comparison.md`](sota_comparison.md) then showed the
publication-held-out arm to be a far larger shift — the random forest drops 0.854 → 0.612 AUROC —
and that is the regime a screening campaign operates in. Coverage under a scaffold split says
nothing about it.

Eleven leave-one-publication-out folds over the 444-molecule set, each molecule tested exactly once.
Six entrants: the three tuned baselines, their ensemble (the subject of `uncertainty.md`), and the
two strongest molecular AI approaches on this arm, ChemBERTa-2 frozen and fine-tuned. Chemprop,
CheMeleon, MoLFormer and the tabular model were left out — they need separate environments or
scored below these on this arm — and that is a scope choice, not a result.

**The calibration set always comes out of the training publications.** Within each fold it is carved
from the training molecules; the validation slice keeps its existing job of hyperparameter selection,
so no model is tuned on its own calibration data, and **the held-out publication's labels are used
only to score**. Two regimes are compared:

| regime | calibration set | mean fitting molecules | mean calibration molecules |
|---|---|---|---|
| **iid** | random 25% of the training molecules | 274 | 91 |
| **group** | whole training publications, smallest first, until 25% is reached | 258 | 107 (5.6 publications) |

Entrants are fitted through `run_sota.py`'s own `ENTRANTS`, unchanged, by handing each fitter a test
index that is the calibration set followed by the real test set and splitting the returned vector.
One fit yields both, and the models are identical to the ones in `sota_comparison.md`.

## 1. i.i.d. calibration breaks; publication-wise calibration holds

Pooled over molecules (folds hold 10 to 106, so a fold mean would weight a 10-molecule publication
like a 106-molecule one). Marginal, non-Mondrian sets.

**α = 0.1, target coverage 0.90**

| model | iid coverage | **group coverage** | iid set size | group set size |
|---|---|---|---|---|
| XGBoost | **0.691** | 0.944 | 1.27 | 1.85 |
| logistic regression | **0.712** | 0.910 | 1.32 | 1.75 |
| ensemble (rf+xgb+logreg) | **0.725** | 0.901 | 1.34 | 1.73 |
| random forest | **0.806** | 0.887 | 1.47 | 1.68 |
| ChemBERTa-2 (frozen) | 0.874 | 0.917 | 1.48 | 1.71 |
| ChemBERTa-2 (fine-tuned) | 0.894 | 0.923 | 1.60 | 1.69 |
| **pooled** | **0.784** | **0.914** | 1.41 | 1.73 |

**α = 0.2, target coverage 0.80**

| | iid | **group** |
|---|---|---|
| pooled coverage | **0.601** | **0.810** |
| pooled set size | 1.01 | 1.48 |
| empty sets | 1.6–10.1% per model | **0%** |

Paired per fold and model, 66 pairs: **group calibration gains +0.113 coverage [+0.078, +0.148] at
α = 0.1 (t-test p = 2.5 × 10⁻⁸, sign test p = 2.8 × 10⁻⁶, higher in 52 of 66) and +0.142
[+0.090, +0.193] at α = 0.2.**

Two details are worth stating plainly. First, the i.i.d. failure is **severe and asymmetric**: at
α = 0.2 the fingerprint models cover 53–55% against an 80% target, and they issue *empty* prediction
sets for up to 10% of molecules — a conformal set that excludes both labels is the method announcing
that the test molecule is unlike anything in its calibration data, and under i.i.d. calibration it
happens routinely. Group calibration produces none at any α.

Second, **the models that fail i.i.d. calibration worst are the fingerprint models**, and the two
ChemBERTa-2 variants are nearly unaffected (0.874 and 0.894 against 0.90). The same ordering appears
in the AUROC decline in `sota_comparison.md`. Fingerprint scores on a new publication's chemistry are
not exchangeable with scores on random training molecules; the frozen transformer's are closer to it.
We note the agreement and do not claim a mechanism.

### The confound in this comparison, stated

Group calibration holds out whole publications, so it fits on fewer publications and 16 fewer
molecules on average (258 against 274). Part of the coverage gain could in principle come from a
weaker model producing less confident scores rather than from better-matched calibration. Two things
argue against that reading — the set sizes grow only 1.41 → 1.73 while coverage grows 0.784 → 0.914,
and a uniformly weaker model would not preferentially fix the *fingerprint* entrants — but the
regimes are not matched on training data and a cleanly matched version would fix the fitting set and
vary only the calibration source. That experiment was not run.

## 2. Mondrian calibration is strictly better here

Class-wise quantiles, pooled, group regime:

| α | variant | coverage | set size | **decisive (singleton) share** |
|---|---|---|---|---|
| 0.1 | marginal | 0.914 | 1.73 | 0.267 |
| 0.1 | **Mondrian** | 0.908 | 1.65 | **0.348** |
| 0.2 | marginal | 0.810 | 1.48 | 0.520 |
| 0.2 | **Mondrian** | 0.815 | 1.40 | **0.596** |

Same coverage, materially more decisive answers. Class balance shifts between publications — some
papers report only potent molecules — and a class-wise quantile absorbs that instead of paying for
it in set size. **Mondrian is the variant to use on this arm.** No fold produced an infinite
quantile, so no calibration class was ever too small to support the target.

## 3. How many molecules get a usable answer, and is it right?

A set of size 2 (`{potent, not potent}`) is honest and useless for ranking. The decision-relevant
quantity is the share of molecules receiving a **singleton** set, and the error rate among those.
Group calibration, pooled:

**α = 0.1**

| model | decisive share | **error on decisive predictions** |
|---|---|---|
| random forest | 0.318 | 0.355 |
| ChemBERTa-2 (fine-tuned) | 0.311 | **0.246** |
| ChemBERTa-2 (frozen) | 0.295 | 0.282 |
| ensemble | 0.275 | 0.361 |
| logistic regression | 0.255 | 0.354 |
| XGBoost | 0.146 | 0.385 |

**α = 0.2**

| model | decisive share | error on decisive predictions |
|---|---|---|
| ChemBERTa-2 (frozen) | 0.640 | 0.342 |
| ChemBERTa-2 (fine-tuned) | 0.631 | 0.321 |
| random forest | 0.523 | 0.379 |
| logistic regression | 0.455 | 0.416 |
| XGBoost | 0.435 | 0.378 |

**This is the practically important result and it is negative.** At α = 0.1 the method is honest —
coverage 0.91 against a 0.90 target — and it answers decisively on **15–32% of molecules**, of which
it gets **25–39% wrong**. The guarantee is marginal over all molecules, not conditional on the set
being a singleton: 90% coverage is achieved by returning both labels on two-thirds of the molecules,
which leaves the decisive third carrying far more than 10% error. Loosening to α = 0.2 doubles the
decisive share and the error rate stays at 32–42%.

So conformal prediction **does** identify unreliable predictions here, in the only sense it promises:
it declines to commit, and it declines on most molecules. What it does not deliver is a usable subset
of trustworthy predictions. A screening cascade cannot act on a 65%-accurate decisive answer covering
a third of its library.

## 4. Chemically unfamiliar molecules

Band sizes by max Tanimoto to the fitting set, group regime: **357 / 58 / 12 / 9 / 8** across
[0, 0.4), [0.4, 0.5), [0.5, 0.6), [0.6, 0.7), [0.7, 1.01]. **80% of molecules fall below 0.4** —
higher than the 64% reported in `sota_comparison.md` because group calibration removes publications
from the fitting set, which is the cost of the calibration regime made visible.

In the below-0.4 band, n = 357:

| model | coverage | decisive share | error on decisive |
|---|---|---|---|
| random forest | 0.871 | 0.367 | 0.351 |
| ChemBERTa-2 (frozen) | 0.902 | 0.342 | 0.287 |
| ChemBERTa-2 (fine-tuned) | 0.919 | 0.308 | 0.264 |
| ensemble | 0.882 | 0.328 | 0.359 |
| logistic regression | 0.908 | 0.289 | 0.320 |
| XGBoost | 0.936 | 0.174 | 0.371 |

**Coverage survives where AUROC does not.** This is the central methodological point: the band where
every model ranks near chance is the band where conformal coverage stays within 0.03 of target.
Honesty transfers to unfamiliar chemistry; discrimination does not.

**Conditional coverage does not hold, and the high-similarity bands show it.** Conformal prediction
guarantees coverage marginally, not within arbitrary subgroups, and logistic regression covers
**0.375** in the [0.7, 1.01] band. Those bands hold 8 to 12 molecules, so they are anecdotes rather
than estimates — but they are the right anecdotes to print, because they are what a per-molecule
guarantee would have to rule out and does not. A reader should not read "coverage 0.91" as a promise
about any particular molecule.

## 5. Selective prediction barely works under publication shift

Error rate at a 0.5 threshold among the most confident fraction kept, group regime, pooled:

| model | 100% | 80% | 60% | 40% | 20% |
|---|---|---|---|---|---|
| random forest | 0.459 | 0.460 | 0.416 | 0.389 | 0.353 |
| XGBoost | 0.450 | 0.429 | 0.410 | 0.394 | 0.368 |
| logistic regression | 0.459 | 0.437 | 0.428 | 0.372 | 0.353 |
| ensemble | 0.468 | 0.452 | 0.421 | 0.377 | 0.366 |
| **ChemBERTa-2 (frozen)** | **0.358** | **0.331** | **0.272** | **0.245** | **0.227** |
| ChemBERTa-2 (fine-tuned) | 0.403 | 0.390 | 0.372 | 0.343 | 0.320 |

Against the scaffold-split numbers in [`uncertainty.md`](uncertainty.md) — 0.155 falling to 0.056 —
this is a different world. **At full coverage the fingerprint models are wrong on 45–47% of
molecules**, against a 48% positive rate: at a 0.5 threshold they are at coin-flip accuracy on an
unseen publication. Abstaining on the least confident 80% improves the random forest from 0.459 to
0.353 and no further. The confidence-based abstention that worked under scaffold shift **largely
fails here**, because the probabilities themselves are no longer informative about correctness.

**ChemBERTa-2 frozen is the only entrant with a usable risk–coverage curve** (0.358 → 0.227), and it
has the lowest pooled error at every coverage level. **Paired per fold the difference is not
significant**: rf − chemberta_frozen is −0.009 at full coverage and +0.026 to +0.044 at lower
coverage, with t-test p between 0.63 and 0.91 and the random forest worse in only 2–5 of 11 folds.
The pooled advantage comes from the large folds. This is the same shape as the inconclusive AUROC
inversion in `sota_comparison.md` §4, from an independent metric, and it should be read the same way:
a hypothesis with an effect size, not a ranking.

## Verdict

1. **Calibrate on held-out publications, not on random molecules.** The i.i.d. regime under-covers by
   0.11–0.14 and emits empty prediction sets; the group regime holds coverage at every α tested. This
   is the one unambiguous, significant, actionable result here.
2. **Use the Mondrian variant.** Same coverage, 30% more decisive answers.
3. **Do not expect abstention to rescue screening on new chemistry.** Honest coverage is achievable;
   a trustworthy subset is not. 15–32% of molecules get a decisive answer at α = 0.1 and a quarter to
   a third of those are wrong.
4. **`uncertainty.md`'s conclusions are now scoped, not overturned.** Conformal coverage holding at
   0.922 under scaffold shift stands. The implicit extrapolation — that it would therefore hold under
   any shift with the same calibration recipe — is false, and this is where it fails.
5. **Confidence-based selective prediction is scaffold-split-specific.** The 15.5% → 6.5% improvement
   reported there becomes 45.9% → 35.3% here.

## Limitations

- **Eleven folds.** Every per-fold comparison in section 5 is underpowered; only the pooled
  coverage comparison, which has 66 paired observations, is significant. The model comparison is
  not, and is not claimed.
- **The two calibration regimes are not matched on fitting data** (258 against 274 molecules, and
  group calibration removes whole publications). Section 1 argues the gain is not an artefact of a
  weaker model; it does not prove it.
- **Exchangeability is violated by construction.** The theorem does not apply to either regime; group
  calibration makes calibration and test data *more* similar in kind — both are unseen publications —
  but it does not make them exchangeable. Every coverage figure here is empirical over 11 folds.
- **Conditional coverage is not guaranteed and visibly fails** in bands holding 8–12 molecules.
- Four entrants from `sota_comparison.md` were not run: Chemprop, Chemprop+RDKit, CheMeleon,
  MoLFormer-XL and the tabular model. Chemprop+RDKit scored third on this arm by AUROC (0.681) and
  its omission is the most arguable one.
- Error rates are at a fixed 0.5 threshold. For a screening campaign, precision at a fixed budget
  would be more decision-relevant.
- **Nothing here is evidence about any compound.** A calibrated abstention rate is a statement about
  a model, not about chemistry, and certainly not about clinical safety.
