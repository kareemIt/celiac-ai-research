"""Can the transglutaminase family support selectivity modelling, or is there not enough data?

Phase 1 of the selectivity work. Four human enzymes were fetched separately
(scripts/fetch_chembl_tg2.py --target all) and are audited here without being merged: a measurement
against TGM1, TGM3 or FXIII-A is evidence about that enzyme, never a TG2 label.

The questions, in the order they decide whether Phase 2 is possible at all:

  1. inventory   what exists per target, by measurement type, unit, qualifier, species, provenance
  2. usable      how much survives the same IC50/nM rule the TG2 set already uses
  3. pairs       how many molecules carry BOTH a TG2 and an off-target measurement -- the only
                 molecules that can teach or test selectivity
  4. novelty     do the off-target molecules reach chemistry the TG2 training set does not have
  5. leakage     scaffold overlap between targets, which would make a selectivity split optimistic
  6. covalent    how much irreversible-inhibition evidence the IC50-only filter is discarding

Writes results/tg_family_audit.csv. CPU, seconds.

Usage: python scripts/audit_tg_family.py [--cutoff-um 1.0]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from rdkit.Chem.Scaffolds import MurckoScaffold

from scripts.audit_splits import tanimoto_to_train
from src.data import chem, chembl, tg2

TARGETS = ["tgm2", "tgm1", "tgm3", "f13a1"]
ORTHOLOGS = ["tgm2_gp"]  # same enzyme, different species: separate labels, never pooled with human
ALL = TARGETS + ORTHOLOGS
LABEL = {"tgm2": "TG2 (target)", "tgm1": "TGM1", "tgm3": "TGM3", "f13a1": "FXIII-A (antitarget)",
         "tgm2_gp": "TG2 guinea pig"}
RAW = "data/raw/chembl/{name}_activities.csv"
OUT = "results/tg_family_audit.csv"


def load(name):
    return chembl.load_raw(RAW.format(name=name))


def inventory(raw):
    """What is in the file before any filtering, so the exclusions below are auditable."""
    u = chembl._usable(raw)
    exact = u[u.standard_relation == "="]
    return {
        "records": len(raw),
        "molecules": raw.smiles.nunique(),
        "documents": raw.document_chembl_id.nunique(),
        "non_human": int((raw.target_organism != "Homo sapiens").sum()),
        "assay_types": "/".join(sorted(raw.assay_type.dropna().unique())),
        "ic50_nM_records": len(u),
        "ic50_nM_molecules": u.smiles.nunique(),
        "exact_molecules": exact.smiles.nunique(),
        "censored_molecules": u[u.standard_relation != "="].smiles.nunique(),
        "flagged_rows": int(raw.data_validity_comment.notna().sum()),
    }


def scaffolds(smiles):
    return {MurckoScaffold.MurckoScaffoldSmiles(s) for s in smiles}


def main(cutoff_um):
    rows = []
    raws = {t: load(t) for t in ALL}

    print("## 1. Inventory: what ChEMBL actually has, per enzyme\n")
    inv = pd.DataFrame({LABEL[t]: inventory(raws[t]) for t in ALL}).T
    print(inv.to_string())
    for t in ALL:
        rows.append({"section": "inventory", "target": t, **inventory(raws[t])})

    print("\n## 2. Measurement types excluded, and what that costs\n")
    for t in ALL:
        vc = raws[t].standard_type.value_counts()
        kept = int(vc.get("IC50", 0))
        print(f"  {LABEL[t]:<22} IC50 {kept:>5}   other {len(raws[t]) - kept:>5}   "
              f"({', '.join(f'{k} {v}' for k, v in vc.drop('IC50', errors='ignore').head(5).items())})")

    print(f"\n## 3. Usable labels at the project's {cutoff_um:g} uM cut\n")
    cls = {}
    for t in ALL:
        c = chembl.classification(raws[t], ic50_um=cutoff_um)
        cls[t] = c
        n_pos = int(c.y.sum())
        print(f"  {LABEL[t]:<22} {len(c):>4} molecules   {n_pos:>4} potent / {len(c) - n_pos:>4} not"
              f"   ({n_pos / len(c):.0%} positive)")
        rows.append({"section": "labels", "target": t, "molecules": len(c), "positive": n_pos,
                     "frac_positive": n_pos / len(c)})

    # The TG2 set actually used by every model in this project (PubChem + ChEMBL union).
    tg2_tab = tg2.combined("classification")
    print(f"\n  project TG2 set (PubChem + ChEMBL union): {len(tg2_tab)} molecules, "
          f"{int(tg2_tab.y.sum())} potent")

    print("\n## 4. Selectivity pairs: molecules measured on TG2 AND an off-target\n")
    print("  These are the only molecules that can teach or test selectivity.\n")
    tg2_smiles = set(tg2_tab.smiles)
    for t in TARGETS[1:]:
        both = set(cls[t].smiles) & tg2_smiles
        sub = cls[t][cls[t].smiles.isin(both)].merge(
            tg2_tab[["smiles", "y"]], on="smiles", suffixes=("_off", "_tg2"))
        sel = sub[(sub.y_tg2 == 1) & (sub.y_off == 0)]
        dual = sub[(sub.y_tg2 == 1) & (sub.y_off == 1)]
        print(f"  TG2 & {LABEL[t]:<22} {len(both):>4} shared molecules   "
              f"{len(sel):>3} TG2-potent-but-not-{t}   {len(dual):>3} potent on both")
        rows.append({"section": "pairs", "target": t, "shared": len(both),
                     "selective": len(sel), "dual": len(dual)})

    print("\n## 5. Do the off-targets add unfamiliar chemistry to the TG2 set?\n")
    tg2_fp = chem.morgan(list(tg2_tab.smiles))
    for t in TARGETS[1:] + ORTHOLOGS:
        new = [s for s in cls[t].smiles if s not in tg2_smiles]
        if not new:
            print(f"  {LABEL[t]:<22} no molecules outside the TG2 set")
            continue
        sims = tanimoto_to_train(chem.morgan(new), tg2_fp)
        far = int((sims < 0.5).sum())
        print(f"  {LABEL[t]:<22} {len(new):>4} molecules not in the TG2 set   "
              f"median max-Tanimoto {np.median(sims):.3f}   {far:>3} below 0.5")
        rows.append({"section": "novelty", "target": t, "new_molecules": len(new),
                     "median_max_tanimoto": float(np.median(sims)), "below_0.5": far})

    print("\n## 6. Scaffold overlap between targets (a selectivity split must respect this)\n")
    scaf = {t: scaffolds(cls[t].smiles) for t in ALL}
    scaf["project_tg2"] = scaffolds(tg2_tab.smiles)
    print(f"  {'':<24}" + "".join(f"{LABEL[t][:12]:>14}" for t in TARGETS))
    for a in ["project_tg2"] + ALL:
        name = "project TG2 set" if a == "project_tg2" else LABEL[a]
        line = f"  {name:<24}"
        for b in TARGETS:
            line += f"{len(scaf[a] & scaf[b]):>14}"
        print(line + f"   ({len(scaf[a])} scaffolds)")
        rows.append({"section": "scaffolds", "target": a, "n_scaffolds": len(scaf[a]),
                     **{f"shared_{b}": len(scaf[a] & scaf[b]) for b in TARGETS}})

    print("\n## 7. Duplicate and replicate quality within each target\n")
    for t in ALL:
        r = chembl.regression(raws[t])
        if r.empty:
            continue
        multi = r[r.n_meas > 1]
        wide = int((r.spread_log > 1).sum())
        print(f"  {LABEL[t]:<22} {len(r):>4} with an exact pIC50   {len(multi):>3} replicated   "
              f"{wide:>3} spanning >1 log   median papers/molecule {r.n_docs.median():.0f}")
        rows.append({"section": "replicates", "target": t, "exact": len(r), "replicated": len(multi),
                     "spread_gt_1log": wide})

    print("\n## 8. kinact/Ki: the covalent signal the IC50 filter discards\n")
    for t in ALL:
        r = raws[t]
        ratio = r[(r.standard_type == "Ratio") & r.assay_description.str.contains(
            "kinact|Kinact|Kinh", case=False, na=False)]
        kin = r[r.standard_type == "Kinact"]
        irrev = r[r.assay_description.str.contains("rreversible", na=False)].smiles.nunique()
        print(f"  {LABEL[t]:<22} kinact/Ki on {ratio.smiles.nunique():>4} molecules   "
              f"kinact on {kin.smiles.nunique():>4}   "
              f"assays naming irreversible inhibition: {irrev:>4} molecules")
        rows.append({"section": "covalent", "target": t, "kinact_over_ki": ratio.smiles.nunique(),
                     "kinact": kin.smiles.nunique(), "irreversible_assay": irrev})

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff-um", type=float, default=1.0)
    main(ap.parse_args().cutoff_um)
