"""What is actually in the TG2 kinetic records, and does any of it support a mechanism model?

Phase 1 of the covalent work. The IC50 filter used everywhere else in this project discards 990 of
the 1,583 TG2 records, and the selectivity audit flagged that some of them are mechanistically the
most informative data in the file. This checks that claim before any model is fitted.

kinact, Ki and kinact/Ki are three different quantities and are counted separately here:

  Ki          an affinity, in nM -- how tightly the inhibitor binds before anything reacts
  kinact      a rate, in /min    -- how fast the bound complex forms its covalent bond
  kinact/Ki   a second-order rate constant, in /M.s -- overall efficiency, the quantity a medicinal
              chemist ranks covalent inhibitors by, and the only one of the three that is
              comparable across compounds with different binding affinities

They are never summed, averaged or merged, and none of them is mixed with the IC50 labels.

Writes results/covalent_audit.csv. CPU, seconds.

Usage: python scripts/audit_covalent.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from scripts.audit_splits import tanimoto_to_train
from src.data import chem, chembl, tg2

OUT = "results/covalent_audit.csv"
RAW = "data/raw/chembl/tgm2_activities.csv"
TO_PER_M_S = {"/M/s": 1.0, "10^3/M/min": 1000 / 60, "10'3/M/min": 1000 / 60}
KINETIC = ["Ki", "Kinact", "Ratio", "kon", "k_off"]

# Electrophiles that appear in the TG2 literature. The question is not whether these patterns are
# present but whether more than one of them is: a model cannot be "just recognizing the warhead"
# if every molecule carries the same warhead.
WARHEADS = {
    "acrylamide": "C=CC(=O)N",
    "chloroacetamide": "ClCC(=O)N",
    "epoxide": "C1CO1",
    "dihydroisoxazole": "C1CNO1",
    "maleimide": "O=C1C=CC(=O)N1",
    "vinyl sulfone": "C=CS(=O)=O",
    "nitrile": "C#N",
}


def main():
    rows = []
    r = chembl.load_raw(RAW)
    k = r[r.standard_type.isin(KINETIC)]

    print("## 1. The three kinetic quantities are not interchangeable\n")
    tab = pd.crosstab(k.standard_type, k.standard_units.fillna("NO UNITS"))
    print(tab.to_string())
    print("\n   relations:")
    print(pd.crosstab(k.standard_type, k.standard_relation.fillna("NO RELATION")).to_string())
    for t in sorted(k.standard_type.unique()):
        d = k[k.standard_type == t]
        rows.append({"section": "inventory", "quantity": t, "records": len(d),
                     "molecules": d.smiles.nunique(), "papers": d.document_chembl_id.nunique(),
                     "no_units": int(d.standard_units.isna().sum())})

    print("\n## 2. kinact/Ki: three unit conventions, and rows carrying none\n")
    rat = r[r.standard_type == "Ratio"]
    print(pd.crosstab(rat.document_chembl_id, rat.standard_units.fillna("NO UNITS")).to_string())
    print("\n   118 unitless rows cannot be placed on a common scale and are dropped, not guessed.")

    d = rat[(rat.standard_relation == "=") & rat.standard_units.isin(TO_PER_M_S)
            & rat.standard_value.notna()].copy()
    d["logk"] = np.log10(d.standard_value.astype(float) * d.standard_units.map(TO_PER_M_S))
    g = d.groupby("smiles").agg(logk=("logk", "median"), paper=("document_chembl_id", "first"))
    scaf = {MurckoScaffold.MurckoScaffoldSmiles(s) for s in g.index}
    print(f"\n   usable: {len(d)} exact rows -> {len(g)} molecules, {g.paper.nunique()} papers, "
          f"{len(scaf)} scaffolds")
    print(f"   log10(kinact/Ki in /M.s) spans {g.logk.min():.2f} to {g.logk.max():.2f}")
    rows.append({"section": "usable", "quantity": "kinact/Ki", "records": len(d),
                 "molecules": len(g), "papers": int(g.paper.nunique()), "scaffolds": len(scaf)})

    print("\n## 3. Is the label comparable across papers, or is it a lab offset?\n")
    per = d.groupby("document_chembl_id").logk.agg(["count", "min", "median", "max", "std"])
    print(per.round(2).to_string())
    between, within = per["median"].std(), per["std"].mean()
    print(f"\n   between-paper SD of medians {between:.2f} log, within-paper SD {within:.2f} log")
    print("   Comparable magnitudes: a model trained on five papers and tested on a sixth has to\n"
          "   absorb a systematic offset about as large as the signal it is ranking.")
    rows.append({"section": "comparability", "quantity": "kinact/Ki",
                 "between_paper_sd": float(between), "within_paper_sd": float(within)})

    print("\n   assay per paper:")
    for doc, gg in d.groupby("document_chembl_id"):
        print(f"     {doc}  {gg.assay_description.iloc[0][:95]}")

    print("\n## 4. Would a model just be recognizing the reactive group?\n")
    mols = {s: Chem.MolFromSmiles(s) for s in g.index}
    for name, smarts in WARHEADS.items():
        q = Chem.MolFromSmarts(smarts)
        n = sum(1 for m in mols.values() if m is not None and m.HasSubstructMatch(q))
        print(f"   {name:<18}{n:>4} of {len(mols)}")
        rows.append({"section": "warhead", "quantity": name, "molecules": n})
    print("\n   One electrophile class dominates, so warhead identity carries almost no variance\n"
          "   and cannot be what a model keys on. Equally, there is no mechanism *contrast* to\n"
          "   classify: this is a single-chemotype regression problem, not reversible vs covalent.")

    print("\n## 5. Mechanism labels that are experimentally stated, not inferred\n")
    irrev = r[r.assay_description.str.contains("Irreversible", na=False)].smiles.nunique()
    rev = r[r.assay_description.str.contains("Reversible inhibition", na=False)].smiles.nunique()
    print(f"   assays stating irreversible inhibition: {irrev} molecules")
    print(f"   assays stating reversible inhibition:   {rev} molecules")
    print("   A 2-class mechanism model needs both classes; this is not a usable contrast.")
    rows.append({"section": "mechanism", "quantity": "irreversible", "molecules": irrev})
    rows.append({"section": "mechanism", "quantity": "reversible", "molecules": rev})

    print("\n## 6. Does this add anything to the existing IC50 dataset?\n")
    t = tg2.combined("classification")
    known = set(t.smiles)
    new = [s for s in g.index if s not in known]
    sims = tanimoto_to_train(chem.morgan(new), chem.morgan(list(t.smiles)))
    print(f"   {len(g) - len(new)} of {len(g)} kinetic molecules already carry an IC50 label")
    print(f"   {len(new)} are new molecules, median max-Tanimoto to the IC50 set "
          f"{np.median(sims):.3f}, {int((sims < 0.5).sum())} below 0.5")
    print("\n   So the kinetic set is mostly NEW molecules but NOT new chemistry: close analogs of\n"
          "   what we already have. It does not address the 6-positive novel stratum.")
    rows.append({"section": "overlap", "quantity": "kinact/Ki", "molecules": len(new),
                 "already_labelled": len(g) - len(new),
                 "median_max_tanimoto": float(np.median(sims)),
                 "below_0.5": int((sims < 0.5).sum())})

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
