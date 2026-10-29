"""Does recovering the missing kinetic units actually buy us anything?

Phase 1 of the screening-preparation work: a feasibility audit, not a model.

docs/covalent.md concluded that covalent efficiency could not be predicted across laboratories, and
named one step that might change that: recover the kinact/Ki rows ChEMBL stores without units. This
measures what that recovery is worth, under the rule that counting molecules is not the point --
new *chemistry*, new *warheads* and new *paired selectivity* are.

Writes results/unit_recovery.csv. CPU, seconds.

Usage: python scripts/audit_recovery.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from scripts.audit_covalent import WARHEADS
from scripts.audit_splits import tanimoto_to_train
from scripts.run_covalent import TO_PER_M_S, kinetic_table
from src.data import chem, chembl, kinetic_units, tg2

OUT = "results/unit_recovery.csv"
RAW = "data/raw/chembl/{name}_activities.csv"


def kinetic(name, recover):
    """Molecules with a usable kinact/Ki for one target, with or without unit recovery."""
    r = chembl.load_raw(RAW.format(name=name))
    if recover:
        r = kinetic_units.apply(r)
    d = r[(r.standard_type == "Ratio") & (r.standard_relation == "=")
          & r.standard_units.isin(TO_PER_M_S) & r.standard_value.notna()].copy()
    if d.empty:  # several off-targets have no unit-bearing kinetic rows before recovery
        return pd.DataFrame(columns=["smiles", "logk", "paper"])
    d["logk"] = np.log10(d.standard_value.astype(float) * d.standard_units.map(TO_PER_M_S))
    return d.groupby("smiles").agg(logk=("logk", "median"),
                                   paper=("document_chembl_id", "first")).reset_index()


def warheads(smiles):
    out = {}
    for n, sm in WARHEADS.items():
        q = Chem.MolFromSmarts(sm)
        out[n] = sum(1 for s in smiles
                     if (m := Chem.MolFromSmiles(s)) is not None and m.HasSubstructMatch(q))
    return {k: v for k, v in out.items() if v}


def main():
    rows = []
    print("## 1. The decisions, and the evidence for each\n")
    for p in kinetic_units.provenance():
        print(f"  [{p['status'].upper():<9}] {p['document']}"
              + (f"  -> {p['unit']}" if p["unit"] else ""))
        print(f"      {p['citation']}")
        print(f"      {p['evidence']}\n")
        rows.append({"section": "provenance", **p})

    before, after = kinetic(0 or "tgm2", False), kinetic("tgm2", True)
    print("## 2. What the recovery adds to the TG2 kinetic set\n")
    print(f"  molecules {len(before)} -> {len(after)}   "
          f"papers {before.paper.nunique()} -> {after.paper.nunique()}")
    new = sorted(set(after.smiles) - set(before.smiles))
    sb = {MurckoScaffold.MurckoScaffoldSmiles(s) for s in before.smiles}
    sa = {MurckoScaffold.MurckoScaffoldSmiles(s) for s in after.smiles}
    print(f"  scaffolds {len(sb)} -> {len(sa)}   new molecules {len(new)}")
    print(f"  log10(kinact/Ki) range {before.logk.min():.2f}-{before.logk.max():.2f} -> "
          f"{after.logk.min():.2f}-{after.logk.max():.2f}")
    rows.append({"section": "kinetic_set", "before_molecules": len(before),
                 "after_molecules": len(after), "before_scaffolds": len(sb),
                 "after_scaffolds": len(sa), "new_molecules": len(new),
                 "before_papers": int(before.paper.nunique()),
                 "after_papers": int(after.paper.nunique())})

    print("\n## 3. Is it new chemistry, or more of the same?\n")
    t = tg2.combined("classification")
    tfp = chem.morgan(list(t.smiles))
    sims_ic50 = tanimoto_to_train(chem.morgan(new), tfp)
    sims_kin = tanimoto_to_train(chem.morgan(new), chem.morgan(list(before.smiles)))
    print(f"  {len(new)} new molecules")
    print(f"    vs the 631-molecule IC50 set:   median max-Tanimoto {np.median(sims_ic50):.3f}, "
          f"{int((sims_ic50 < 0.5).sum())} below 0.5")
    print(f"    vs the existing kinetic set:    median max-Tanimoto {np.median(sims_kin):.3f}, "
          f"{int((sims_kin < 0.5).sum())} below 0.5")
    rows.append({"section": "novelty", "new_molecules": len(new),
                 "median_tanimoto_to_ic50": float(np.median(sims_ic50)),
                 "below_0.5_vs_ic50": int((sims_ic50 < 0.5).sum()),
                 "median_tanimoto_to_kinetic": float(np.median(sims_kin)),
                 "below_0.5_vs_kinetic": int((sims_kin < 0.5).sum())})

    print("\n## 4. Warheads: does a mechanism contrast now exist?\n")
    wb, wa = warheads(before.smiles), warheads(after.smiles)
    print(f"  before: {wb}")
    print(f"  after:  {wa}")
    top_before = max(wb.values()) / len(before) if wb else 0
    top_after = max(wa.values()) / len(after) if wa else 0
    print(f"\n  dominant warhead share {top_before:.0%} -> {top_after:.0%}")
    print("  docs/covalent.md could not attempt mechanism classification because 169 of 174 "
          "molecules\n  shared one warhead. Two classes of reasonable size would change that.")
    rows.append({"section": "warheads", "before": str(wb), "after": str(wa),
                 "dominant_share_before": top_before, "dominant_share_after": top_after})

    print("\n## 5. Paired selectivity on a single consistent measure\n")
    tg2k = kinetic("tgm2", True)
    for off in ("tgm1", "tgm3", "f13a1"):
        ok = kinetic(off, True)
        shared = set(tg2k.smiles) & set(ok.smiles)
        was = set(kinetic("tgm2", False).smiles) & set(kinetic(off, False).smiles)
        print(f"  TG2 & {off:<7} paired molecules {len(was)} -> {len(shared)}")
        rows.append({"section": "pairs", "off_target": off, "before": len(was),
                     "after": len(shared)})
    print("\n  These pairs come from ONE laboratory on ONE assay, which is the within-series "
          "comparison\n  docs/selectivity.md identified as the only honest way to ask the "
          "selectivity question.")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
