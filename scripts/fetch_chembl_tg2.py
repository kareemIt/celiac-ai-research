"""Fetch every human TG2 (TGM2, CHEMBL2730 / UniProt P21980) activity from the ChEMBL REST API.

Writes the raw records unchanged to data/raw/chembl/tg2_activities.csv; all filtering and unit
handling happens later in src/data/chembl.py, so the provenance stays auditable.

Usage: python scripts/fetch_chembl_tg2.py
"""
import json
import os
import time
import urllib.request

import pandas as pd

TARGET = "CHEMBL2730"  # Homo sapiens protein-glutamine gamma-glutamyltransferase 2
BASE = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
OUT = "data/raw/chembl/tg2_activities.csv"
FIELDS = ["molecule_chembl_id", "canonical_smiles", "standard_type", "standard_relation",
          "standard_value", "standard_units", "pchembl_value", "assay_chembl_id",
          "assay_type", "assay_description", "document_chembl_id", "target_organism",
          "data_validity_comment", "activity_comment", "potential_duplicate"]


def fetch_all(target=TARGET, page=1000):
    rows, offset = [], 0
    while True:
        url = f"{BASE}?target_chembl_id={target}&limit={page}&offset={offset}"
        with urllib.request.urlopen(url, timeout=120) as r:
            page_data = json.load(r)
        acts = page_data["activities"]
        rows += [{k: a.get(k) for k in FIELDS} for a in acts]
        total = page_data["page_meta"]["total_count"]
        print(f"  {len(rows)}/{total}", flush=True)
        offset += page
        if offset >= total or not acts:
            return pd.DataFrame(rows)
        time.sleep(0.3)  # be polite to the EBI API


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = fetch_all()
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}: {len(df)} activity records, {df.molecule_chembl_id.nunique()} molecules")
    print("\nstandard_type:\n", df.standard_type.value_counts().head(12).to_string())
    print("\nstandard_units:\n", df.standard_units.value_counts().head(8).to_string())
    print("\nassay_type:\n", df.assay_type.value_counts().to_string())
    print("\ntarget_organism:\n", df.target_organism.value_counts().to_string())
    print("\ndata_validity_comment:\n", df.data_validity_comment.value_counts(dropna=False).head(8).to_string())
