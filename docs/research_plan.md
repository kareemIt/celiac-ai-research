# Research plan and architecture

## Datasets (profiled 2026-10-26, see `src/data/`)

### GDSC2 (TDC `DrugRes(name='GDSC2')`)
- 92,703 rows, deduplicated to **92,029** drug–cell-line pairs (674 repeated pairs averaged). **137 drugs × 805 cancer cell lines.**
- Drug: SMILES. Cell line: **17,737-gene expression vector** (RMA-normalized, range 2.1–13.9, no NaNs).
- Target `Y` = log-normalized IC50 (mean 2.64, sd 2.72, range −8.6 to 10.0). No missing values.
- License: **CC BY-NC-ND 2.5** (non-commercial).
- Splits (`gdsc2.split`): `random` (TDC default, optimistic), `cold_drug` (unseen drugs), `cold_cell` (unseen cell lines). With only 137 drugs, report cold-drug results with variance across seeds.
- Note: these are cancer cell lines. GDSC2 is for method development (pretraining or validating the drug-encoder pipeline), not celiac biology.

### TG2 / celiac-informatics (Wichka & Lai, CSBJ 2024)
- Source file `TG2_compound_smiles.csv`: 1,130 rows, **656 unique molecules** (canonical SMILES), all SMILES valid.
- Measurement types: IC50 724, Ki 131, k_inact 32, missing 243. Qualifier: `=` 524, `>` 94 (censored).
- Activity labels: Active 621, Unspecified 361, Inactive 103, Inconclusive 45. In this data, actives have IC50 ≤ 10 and inactives ≥ 10, which implies a 10 µM threshold. **Units are assumed to be µM; confirm against the paper.**
- Processed:
  - **Regression:** 305 molecules, pIC50 from exact IC50 only (mean 5.98, sd 0.96).
  - **Classification:** same 305 molecules, potent if IC50 ≤ 1 µM (160 potent / 145 not). The repo's Active/Inactive labels (10 µM cut, 397/44) were too imbalanced and are not used (see `baselines.md`).
- **Use scaffold splits** (`chem.scaffold_split`). The data are analog series (e.g. thienopyrimidine hydrazides), and random splits would leak near-duplicates.
- The repo's pre-featurized CSVs have no SMILES/IDs, and `ic50_full_data.csv` column `0` does not match any obvious transform of the labels (corr 0.90 with raw IC50). Those files are **not used**. The repo's `.pkl`/`.joblib` models are pickles and were **not downloaded or loaded**.
- The two datasets share **no common labels or features**, only the idea of "molecule → activity".

## Approaches compared

| | Approach | Pros | Cons | Month-1 fit |
|---|---|---|---|---|
| A | Biological features as structured text into Qwen | Simple, uses LLM knowledge | 17k-gene vectors can't be text. SMILES-as-text is weak for LLMs | Only for drug-only TG2, as a probe |
| B | Numerical/molecular encoders projected into Qwen | Most expressive | New architecture, alignment training, lots of tuning | ❌ too complex now |
| C | **Frozen Qwen backbone + trainable head** | Cheap (embed once, train heads in seconds), reproducible, easy to combine with fingerprints/expression | Frozen features may not encode chemistry well | ✅ **recommended** |
| D | QLoRA fine-tune Qwen | Adapts the model to the task, infrastructure already validated | Overfitting risk on 300–440 molecules, slower, harder to evaluate | Month 2, only if C shows signal |

## Recommendation for the first month
1. **Baselines first:** Morgan fingerprints (2048-bit) + XGBoost / logistic regression on TG2 (scaffold split, 5 seeds). On GDSC2, use fingerprint + expression (e.g. PCA-256) + XGBoost on the cold-drug and cold-cell splits. Every LLM result must beat these.
2. **Approach C:** extract mean-pooled Qwen3.5-9B hidden states for each molecule's prompt (SMILES + optionally name), cache them to `data/processed/`, and train a linear/MLP head. Also try `[Qwen embedding ‖ fingerprint]`.
3. **Jev-style probe (zero training):** LLM-as-Jev option-logit readout for TG2 "potent (IC50 ≤ 1 µM) vs not". Measure AUROC, ECE and Brier. This is the cheapest test of the professor's Jev idea (see `jev_research.md`).
4. Evaluation: fixed seeds, scaffold/cold splits, mean ± sd over 5 seeds. Report calibration for classifiers.

Planned modules: `src/models/heads.py` (sklearn/torch heads), `src/training/embed.py` (cache Qwen features), `src/evaluation/metrics.py`. These are not written yet, on purpose.

**Honest prior:** on a few hundred molecules, fingerprint baselines often match or beat LLM features. A negative result against strong baselines is still a valid finding.
