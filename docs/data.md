# TG2 data: provenance, validation and preprocessing (audited 2026-10-27)

Reproduce: `python scripts/check_sources.py` (cross-source agreement), `python scripts/audit_splits.py` (leakage).

## Sources

| | PubChem export | ChEMBL |
|---|---|---|
| Origin | `Lailabcode/celiac-informatics`, the data behind Wichka & Lai, *CSBJ* 23:3669–3679 (2024) | ChEMBL REST API, target **CHEMBL2730** (human TG2, UniProt P21980) |
| Fetched | with the repo CSVs | 2026-10-27, `scripts/fetch_chembl_tg2.py` |
| Raw records | 1,130 rows, 672 distinct PubChem CIDs | 1,587 activities, 786 molecules, all *Homo sapiens*, all binding assays |
| Usable | 367 molecules with a potency label | 460 molecules with a potency label |

## Question 1, now settled: the IC50 values are in µM

The source CSV stores bare numbers. The paper states it directly — "Molecules with an IC50 ≤ 1 µM are typically the most potent", and its own active set used "an IC50 threshold of less than 0.6 µM". So `pIC50 = 6 − log10(value)` is correct, and the 1 µM threshold used here matches the paper's own description of potency. **This closes the open question that was flagged for the professor.**

The paper's other choices, for reference: it built classification from a 0.6 µM active cut plus 45 documented inactives and a slice of unspecified molecules at 300 µM, and restricted its regression set to IC50 ≤ 10 µM. This project instead uses one consistent 1 µM threshold on measured values, which is simpler to state and does not mix in molecules that were never assayed.

## What the audit found, and what changed

### 1. Censored measurements were being thrown away (fixed)
Every `Inactive` row in the source carries a `>` qualifier: 46 of them, against 327 exact `=` Active rows. The old filter kept only `=` rows, so **every documented inactive compound was silently discarded**.

A bound of `> 10 µM` gives no pIC50, but it does prove "not potent at 1 µM". `tg2.classification` now adds such molecules as confirmed negatives when the bound is at or above the threshold, and still ignores bounds below it (`> 0.5 µM` against a 1 µM cut proves nothing). Regression is unchanged — a censored value genuinely cannot be regressed on.

Effect: **305 → 367 molecules**, 160 potent / 207 not potent.

### 2. Salts were not stripped (fixed)
18 entries are salts or mixtures (SMILES containing `.`). Their counterions set their own Morgan bits and can change the Murcko scaffold the splits group on. `chem.canonical` now keeps the largest fragment. Regression results barely moved (RF RMSE 0.776 → 0.778), confirming this was a correctness fix rather than a performance lever.

### 3. Replicate disagreement is real but modest
77 molecules have more than one exact IC50. Their max/min ratio has median 2.8×, but 14 molecules disagree by more than 10× and the worst spans 77×. The median of pIC50 is used per molecule, which is robust to a single bad replicate. In ChEMBL, 17 of 425 molecules have replicates spanning more than one log unit.

### 4. Stereochemistry is absent
No SMILES in the PubChem export carries stereochemistry (`@` or `/`). Enantiomers are therefore collapsed into one record. This is a property of the source and cannot be fixed downstream; it caps how well any structure-based model can do on stereochemically-driven activity.

### 5. The two sources agree, so merging them is defensible
This was checked before combining anything:

| Check | Result |
|---|---|
| Molecules with an exact pIC50 in both | 173 |
| Spearman between the two sources | **1.000** (Pearson 0.997) |
| Median absolute difference | **0.000 log units** |
| Agreement within 1 log unit | 100% |
| Class labels compared | 196 molecules, **100% identical** |

`tg2.combined()` therefore takes the union, keeping the PubChem value where both have one so that the earlier PubChem-only results remain a strict subset. **367 → 631 molecules** (264 new: 142 potent, 122 not).

### 6. A data-quality note on the published regression file
`ic50_full_data.csv` in the source repo contains negative values (minimum −0.115) in a column that should hold IC50 potencies. This project never reads that file — it rebuilds everything from `TG2_compound_smiles.csv` — but it is worth knowing if the published model is ever used as a reference.

## What was deliberately excluded

| Excluded | Count | Why |
|---|---|---|
| Ki measurements | 131 (PubChem) + 240 (ChEMBL) | A Ki and an IC50 are not interchangeable without the substrate Km, which neither source records here |
| Kinact, kon, koff, % inhibition, ratios | 32 (PubChem) + ~700 (ChEMBL) | Different quantities entirely |
| `Inconclusive` rows | 45 | The source's own judgement |
| ChEMBL rows flagged `data_validity_comment` | 30 | ChEMBL's own curation flagged them as outside the typical range |
| Non-human TG2 targets | all | Mouse and guinea-pig TG2 are separate ChEMBL targets and were not requested |

## Split integrity (`scripts/audit_splits.py`)

631 molecules fall into 229 Murcko scaffolds; 60% of scaffolds are singletons but the largest group holds 48 molecules, which is why the split fills folds by molecule count rather than scaffold count.

| | scaffold split | random split |
|---|---|---|
| Scaffolds shared between train and test | **0** | 28.6 |
| Mean Tanimoto to the nearest training molecule | 0.696 | 0.841 |
| Test molecules with a training analog ≥ 0.7 Tanimoto | **51%** | 83% |
| Test molecules whose fingerprint is **identical** to a training molecule | **0.3%** | **38.6%** |

Two conclusions. First, the random split leaks badly: nearly 40% of its test molecules have a training molecule with an identical Morgan fingerprint, which is why random-split AUROC reaches 0.95 against 0.87 under scaffold splits. Random-split numbers for this dataset should not be reported as generalization. Second, the scaffold split is sound by its own definition but is not a hard test of novelty: half its test molecules still have a close analog in training. Numbers here are best read as "a new scaffold within familiar chemistry", not "a genuinely new chemotype".
