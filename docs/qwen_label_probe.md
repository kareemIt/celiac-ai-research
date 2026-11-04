# Direct label-leakage probe: does base Qwen already know these IC50s?

**Answer: no.** On 631 TG2 molecules, base Qwen3.5-9B recalls no published IC50 value. The
measurement, its controls, and the one result that needs explaining are below.

This probe audits an existing result rather than adding a model. It was the outstanding item in
[`state_of_the_art_comparison.md`](state_of_the_art_comparison.md) (addendum, "Still not done") and
the last open question in [`sota_comparison.md`](sota_comparison.md) §7.

| | |
|---|---|
| script | `scripts/probe_qwen_labels.py` |
| tests | `tests/test_qwen_label_probe.py` (27 cases) |
| raw generations | `results/qwen_label_probe_raw.csv` (3,155 rows) |
| summary | `results/qwen_label_probe.csv` |
| log | `results/qwen_label_probe.log` |
| model | `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, 4-bit NF4, **base weights, no adapter** |
| decoding | greedy, `do_sample=False`, `max_new_tokens=24`, seed 0 |
| truth | `tg2.combined("regression")`, 557 of 631 molecules with an exact pIC50 (median 6.119, IQR 1.313, max 8.22) |

## What the probe asks

Each molecule is shown to the **base** model, one molecule per prompt, with no fine-tuning and no
examples, and the model is asked for that compound's IC50 against human TG2 as a number. The answer
is parsed to nM, converted to pIC50, and compared with the measured value.

Five arms. The first three permit `UNKNOWN`; the last two forbid it and demand an estimate:

| arm | SMILES shown | name shown | number forced |
|---|---|---|---|
| `smiles` | canonical | no | no |
| `smiles+name` | canonical | yes (398 of 631 have one) | no |
| `smiles_shuffled` | **re-rooted, same molecule** | no | no |
| `smiles_forced` | canonical | no | **yes** |
| `smiles_shuffled_forced` | **re-rooted, same molecule** | no | **yes** |

**The discriminator is canonical versus shuffled.** A shuffled SMILES is the same molecule written
from a different starting atom — a different string, an identical structure, asserted in code to
round-trip to the same canonical form. Memorised text is keyed on the string, so recall must fall
when the string changes. Chemical inference is keyed on the structure, so it must not. That
comparison is what separates leakage from guessing, and it is why the probe is not just "ask and see".

The two forced arms exist because of a pilot result: the model answered `UNKNOWN` to all 12 pilot
prompts. A model told that `UNKNOWN` is permitted may abstain reflexively whether or not it knows,
which would have made an all-abstain result uninformative. Forcing a number removes that escape and
makes the canonical-versus-shuffled comparison testable.

## Results

| arm | answered | Spearman vs truth | 95% CI | distinct answers | near-exact | chance | AUROC @1 µM |
|---|---|---|---|---|---|---|---|
| `smiles` | **0 / 631** | — | — | 0 | — | — | — |
| `smiles+name` | 8 / 631 (1.3%) | — | — | **1** | — | — | 0.500 |
| `smiles_shuffled` | **0 / 631** | — | — | 0 | — | — | — |
| `smiles_forced` | 631 / 631 | 0.199 | [0.120, 0.273] | **6** | 0.056 | 0.049 | 0.581 |
| `smiles_shuffled_forced` | 631 / 631 | **0.278** | [0.193, 0.351] | **6** | 0.048 | 0.035 | **0.647** |

Four readings, in order of how much work they do.

**1. Permitted to abstain, the model abstains on every single molecule.** 631 of 631 `UNKNOWN` on
SMILES alone, canonical and shuffled alike. No recall to detect.

**2. The eight answers in the named arm are the number from the prompt's own example.** All eight are
the string `IC50 = 250 nM`, which is the example value in the instruction
(*"answer with the number and its unit, for example 'IC50 = 250 nM'"*). That is prompt echo, not
retrieval. Two of the eight are Isatin and 5-Bromoisatin — among the 14 trivially-named molecules the
earlier audit flagged as the most retrievable in the set. They were the best candidates for genuine
recall and they returned the echoed example.

**3. Forced to answer, the model emits six stock numbers for 631 molecules.** The entire output
vocabulary is `{1.1, 1.2, 1.5} × {nM, µM}` in both forced arms. 276 of 631 molecules receive the
identical answer `IC50 = 1.2 µM`. A model recalling 631 published values does not have six of them.
The weak positive Spearman is carried almost entirely by the binary nM-versus-µM choice; there is no
within-unit resolution to carry anything else.

**4. Rewriting the molecule makes the answers better, which recall cannot do.** Every metric moves
the wrong way for the leakage hypothesis: Spearman +0.080 in the shuffled arm, AUROC +0.067, and the
median prediction moves from pIC50 5.96 to **8.82** — a thousandfold shift in central tendency from
rewriting the same 631 molecules, landing the *median* prediction above the most potent compound in
the dataset (8.22). A lookup table keyed on structure cannot shift three orders of magnitude when the
string is re-rooted. The forced output is a function of SMILES surface form, not of the compound.

### What the weak correlation is

It is real (p = 2e-6 canonical, 2e-11 shuffled) and it is not recall. The forced arms are a two-level
predictor — "nanomolar" or "micromolar" — applied with a prior that shifts when the input string is
rewritten. Near-exact agreement is 0.056 against a permutation chance rate of 0.049: indistinguishable.
AUROC of 0.581 is in the same region as the zero-shot option-logit readout's 0.648 reported in
[`sota_comparison.md`](sota_comparison.md), and both are far below the random forest's 0.864. A model
holding these labels would not score there.

## What this does and does not establish

**Establishes.** Base Qwen3.5-9B does not hold the TG2 IC50 values for these 631 molecules in a form
any of these five prompts can retrieve. The 0.850 AUROC of the QLoRA result is therefore not
explained by the base model already knowing the answers — it is learned from the fine-tuning data.
That was the one remaining way the headline Qwen number could have been an artifact, and it is now
measured rather than assumed.

**Does not establish.** This is a probe of one model at one revision, through five prompts, under
greedy decoding. A negative probe bounds what these prompts can extract; it is not proof of absence
from the pretraining corpus, which remains unauditable (0 of 631 molecules can be checked against
the actual corpora — see [`sota_comparison.md`](sota_comparison.md) §7). A different prompt, a larger
model, or sampling could in principle do better. It also says nothing about the *structures* being
familiar, only the labels; structure familiarity is covered separately by the PubChem and ChEMBL
membership audit, and it points the same way — the models that most plausibly saw these structures
are the ones that lost.

**Not a claim about any compound.** Nothing here concerns whether any molecule is a viable celiac
treatment, and nothing in this probe should be read that way.

## Verification

The detector is itself tested, because a probe that cannot detect leakage would also return "no
leakage". `tests/test_qwen_label_probe.py` includes a **positive control**: a synthetic arm answering
with each molecule's true IC50 must yield Spearman ≈ 1.0, near-exact fraction > 0.99, a chance rate
below 0.2, and AUROC 1.0. It does. A constant-answer arm must yield one distinct answer and a NaN
Spearman. It does.

Two bugs were found and fixed before the real run, both of which would have produced a false
negative:

- **The number parser read `IC50 = 1.413e+04 nM` as 4 nM**, matching its own mantissa tail. Any
  scientific-notation recall would have been silently mis-scaled. Fixed with a lookbehind and full
  handling of scientific notation, `× 10^k` forms and thousands separators.
- **The analysis crashed on an arm that abstains on everything** — an all-`None` column that
  `log10` rejects. That is the most likely real outcome, so the analysis used to die precisely on the
  expected result. It is now the subject of its own test.

One deliberate limit: `causal_conv1d` is not installed, so the model falls back to reference PyTorch
convolutions. Slower, numerically equivalent, and it keeps the environment identical to the one the
published QLoRA runs used.
