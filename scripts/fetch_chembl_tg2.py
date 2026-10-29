"""Fetch human transglutaminase activities from the ChEMBL REST API, one target at a time.

Writes the raw records unchanged to data/raw/chembl/<target>_activities.csv; all filtering and unit
handling happens later in src/data/chembl.py, so the provenance stays auditable. Each target is kept
in its own file and never merged here: a measurement against TGM1, TGM3 or FXIII-A is evidence about
that enzyme, not a TG2 label.

Target IDs were resolved from UniProt accessions through the ChEMBL target endpoint, not guessed;
all four are SINGLE PROTEIN, Homo sapiens.

Usage: python scripts/fetch_chembl_tg2.py [--target tgm2|tgm1|tgm3|f13a1|all]
"""
import argparse
import json
import os
import time
import urllib.request

import pandas as pd

# name -> (ChEMBL target, UniProt accession, what it is)
TARGETS = {
    "tgm2": ("CHEMBL2730", "P21980", "TG2, the celiac autoantigen -- the target of interest"),
    "tgm1": ("CHEMBL2810", "P22735", "TGase K, epidermal cornification"),
    "tgm3": ("CHEMBL3363", "Q08188", "TGase E, epidermal/hair follicle"),
    "f13a1": ("CHEMBL4530", "P00488", "Factor XIII-A, fibrin crosslinking -- the bleeding-risk antitarget"),
    # Not a human target. Guinea-pig liver TG2 is the classical assay enzyme of the older
    # literature, so it reaches chemistry the human set never saw -- as a separate ortholog label,
    # never as a human TG2 measurement.
    "tgm2_gp": ("CHEMBL3988613", None, "guinea-pig TG2 ortholog, ~80% identical to human"),
}
TARGET = TARGETS["tgm2"][0]  # kept so the original TG2 invocation reproduces byte for byte
BASE = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
OUT = "data/raw/chembl/{name}_activities.csv"
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


def summarize(df, name):
    print(f"\nwrote {OUT.format(name=name)}: {len(df)} activity records, "
          f"{df.molecule_chembl_id.nunique()} molecules")
    for col in ("standard_type", "standard_units", "assay_type", "target_organism"):
        print(f"\n{col}:\n", df[col].value_counts(dropna=False).head(10).to_string())
    print("\ndata_validity_comment:\n",
          df.data_validity_comment.value_counts(dropna=False).head(8).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="tgm2", choices=sorted(TARGETS) + ["all"])
    a = ap.parse_args()
    names = sorted(TARGETS) if a.target == "all" else [a.target]
    os.makedirs("data/raw/chembl", exist_ok=True)
    for name in names:
        tid, acc, what = TARGETS[name]
        print(f"\n=== {name} ({tid}, UniProt {acc or 'n/a'}) -- {what}", flush=True)
        df = fetch_all(tid)
        df.insert(0, "target_name", name)
        df.insert(1, "target_chembl_id", tid)
        df.to_csv(OUT.format(name=name), index=False)
        summarize(df, name)
