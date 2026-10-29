"""Fetch ChEMBL ADMET-type activities for every molecule already in our transglutaminase files.

Our six activity files hold enzyme inhibition only, because they were pulled by target. ADME
measurements in ChEMBL are attached to the molecule instead, under assay_type 'A', so they are
reached by asking for each molecule rather than each target.

Nothing is filtered or converted here. Units, relations, assay text and the source document are
written through unchanged, so that the audit in scripts/analyse_adme.py can judge provenance for
itself; 'Papp' alone appears in three different unit systems in this data.

Writes data/raw/adme/chembl_adme.csv (regenerable, so untracked, like every other download);
the hand-curated literature values live in data/curated/literature_adme.csv instead. Kept apart
from data/raw/chembl/ because these are not activities against any of our targets.

Usage: python scripts/fetch_adme.py
"""
import glob
import json
import os
import time
import urllib.request

import pandas as pd

ACTIVITIES = "data/raw/chembl/*_activities.csv"
BASE = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
OUT = "data/raw/adme/chembl_adme.csv"
BATCH = 40  # keeps the molecule_chembl_id__in query string inside the server's URL limit

# The transglutaminase family plus the FXIII-A antitarget. The second-target files (CDK2, MAO-B and
# the rest) are controls for the evaluation work and have no bearing on celiac pharmaceutics.
FAMILY = ("tg2", "tgm2", "tgm2_gp", "tgm1", "tgm3", "f13a1")

# Both the standardized and the author-reported value are kept: ChEMBL rewrites some of them (a
# fraction unbound published as "2 %" is standardized to 0.02), and the audit needs to see both.
FIELDS = ["molecule_chembl_id", "canonical_smiles", "standard_type", "standard_relation",
          "standard_value", "standard_units", "value", "units", "assay_chembl_id", "assay_type",
          "assay_description", "document_chembl_id", "target_pref_name", "target_organism",
          "data_validity_comment", "activity_comment", "potential_duplicate"]


def molecule_ids(pattern=ACTIVITIES):
    ids = set()
    for path in sorted(glob.glob(pattern)):
        if os.path.basename(path).rsplit("_", 1)[0] in FAMILY:
            ids |= set(pd.read_csv(path)["molecule_chembl_id"].dropna())
    return sorted(ids)


def fetch(ids, batch=BATCH):
    rows = []
    for i in range(0, len(ids), batch):
        url = f"{BASE}?assay_type=A&limit=1000&molecule_chembl_id__in={','.join(ids[i:i + batch])}"
        with urllib.request.urlopen(url, timeout=120) as r:
            page = json.load(r)
        got = page["activities"]
        if page["page_meta"]["total_count"] > len(got):  # 1000 rows is far above any batch seen
            raise RuntimeError(f"batch at {i} truncated: {page['page_meta']['total_count']} rows")
        rows += got
        print(f"   {i + len(ids[i:i + batch]):>4}/{len(ids)} molecules  {len(rows):>5} rows",
              flush=True)
        time.sleep(0.2)
    return rows


def main():
    ids = molecule_ids()
    print(f"{len(ids)} distinct molecules across {', '.join(FAMILY)}\n")
    rows = fetch(ids)
    df = pd.DataFrame(rows).reindex(columns=FIELDS)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.sort_values(["molecule_chembl_id", "standard_type"]).to_csv(OUT, index=False)
    print(f"\n{len(df)} rows for {df.molecule_chembl_id.nunique()} molecules from "
          f"{df.document_chembl_id.nunique()} documents -> {OUT}")


if __name__ == "__main__":
    main()
