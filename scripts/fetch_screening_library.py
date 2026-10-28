"""Approved drugs from ChEMBL as a screening library for TG2 repurposing hypotheses.

Approved drugs are the defensible library for a project this size: every structure is real, has a
name, has human safety data, and a hit would be a repurposing hypothesis a lab could actually test.
A virtual library of billions would mostly measure how far the model extrapolates.

Usage: python scripts/fetch_screening_library.py [--max-phase 4]
"""
import argparse
import json
import os
import time
import urllib.request

import pandas as pd

BASE = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
OUT = "data/raw/screening/chembl_approved.csv"


def fetch(max_phase, page=1000):
    rows, offset = [], 0
    while True:
        url = f"{BASE}?max_phase={max_phase}&limit={page}&offset={offset}"
        with urllib.request.urlopen(url, timeout=180) as r:
            data = json.load(r)
        for m in data["molecules"]:
            st = m.get("molecule_structures") or {}
            props = m.get("molecule_properties") or {}
            rows.append({
                "chembl_id": m.get("molecule_chembl_id"),
                "name": m.get("pref_name"),
                "smiles": st.get("canonical_smiles"),
                "max_phase": m.get("max_phase"),
                "molecule_type": m.get("molecule_type"),
                "withdrawn": m.get("withdrawn_flag"),
                "oral": m.get("oral"),
                "mw": props.get("full_mwt"),
            })
        total = data["page_meta"]["total_count"]
        print(f"  {len(rows)}/{total}", flush=True)
        offset += page
        if offset >= total or not data["molecules"]:
            return pd.DataFrame(rows)
        time.sleep(0.3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-phase", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = fetch(a.max_phase)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}: {len(df)} records, {df.smiles.notna().sum()} with a SMILES")
    print(df.molecule_type.value_counts(dropna=False).head(8).to_string())
