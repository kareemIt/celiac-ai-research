#!/usr/bin/env bash
# Fetch raw data. Run from project root. TG2: CSVs only (the repo's .pkl/.joblib are pickles and are never loaded).
set -euo pipefail
TG2=data/raw/tg2
REPO=https://raw.githubusercontent.com/Lailabcode/celiac-informatics/main
mkdir -p "$TG2"
for f in README.md models/data/ic50_full_data.csv models/data/reg_fing_desc_clean.csv \
         models/data/clf_celiac_desc_fing_data.csv colab/TG2_compound_smiles.csv \
         colab/TG2_cleaned_descriptors_fingerprints_classification.csv utils/features.py; do
  curl -fsSL "$REPO/$f" -o "$TG2/$(basename "$f")"
done

# GDSC2 via PyTDC (downloads to data/raw/tdc).
.venv/bin/python -c "from tdc.multi_pred import DrugRes; DrugRes(name='GDSC2', path='data/raw/tdc')"
