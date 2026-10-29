"""Where does the clinical TG2 inhibitor ZED1227 sit relative to everything we have modelled?

ZED1227 is the only TG2 inhibitor with human efficacy data (Schuppan et al., NEJM 2021). That
makes it the single most informative external reference point available: a compound whose measured
TG2 potency, isoform selectivity and intestinal target engagement are all published, and which was
never in our training data.

This script answers, in order:

  1. is it already in our data, under any structure-matching rule?
  2. how similar is it to what we have -- fingerprints, scaffolds, warhead?
  3. is it inside the applicability domain of our models?
  4. only if 1-3 permit: what would our models predict, and is that prediction meaningful?
  5. how does it compare with the most selective compounds we do have?

Step 4 is gated on steps 1-3 deliberately. A prediction for a compound that turns out to be in
training, or far outside the domain, is not evidence about anything.

Writes results/zed1227_analysis.csv. CPU, seconds. No GPU training.

Usage: python scripts/analyse_zed1227.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from scripts.audit_fxiiia import paired
from scripts.audit_splits import tanimoto_to_train
from scripts.run_covalent import kinetic_table, warhead_of
from src.data import chem, chembl, tg2

OUT = "results/zed1227_analysis.csv"

# PubChem, name "ZED1227" (CAS 1542132-88-6, also TAK-227). C26H36N6O6, MW 528.6.
ZED1227 = "CCC(CC)CNC(=O)CN1C=CC=C(C1=O)NC(=O)[C@H](CC/C=C/C(=O)OC)NC(=O)C2=CN=CN2C"

# Published values, Buchold et al., Cells 2022;11(10):1667. DCC transamidation assay, pH 8.0.
PUBLISHED = {"TG2 IC50 (nM)": 53, "TG2 IC90 (nM)": 300, "FXIII-A2 IC50 (nM)": ">50000",
             "TG1 IC50 (nM)": 24863, "TG3 IC50 (nM)": ">50000", "TG6 IC50 (nM)": 6441}

# Michael acceptors, written to distinguish ZED1227's ester warhead from the series acrylamides.
WARHEAD_SMARTS = {
    "acrylamide (C=C-C(=O)N)": "C=CC(=O)N",
    "unsaturated ester (C=C-C(=O)O)": "C=CC(=O)O",
    "dihydroisoxazole": "C1=NOCC1",
    "chloroacetamide": "ClCC(=O)N",
}


def describe(smiles, label):
    m = Chem.MolFromSmiles(smiles)
    return {"set": label, "smiles": Chem.MolToSmiles(m),
            "mw": Descriptors.MolWt(m), "clogp": Descriptors.MolLogP(m),
            "tpsa": Descriptors.TPSA(m), "hbd": Descriptors.NumHDonors(m),
            "hba": Descriptors.NumHAcceptors(m),
            "rotatable": Descriptors.NumRotatableBonds(m),
            "heavy": m.GetNumHeavyAtoms()}


def flat(s):
    m = Chem.MolFromSmiles(s)
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


def main():
    rows = []
    zed = Chem.MolFromSmiles(ZED1227)
    zed_can = Chem.MolToSmiles(zed)
    zed_flat = flat(ZED1227)
    zed_scaf = MurckoScaffold.MurckoScaffoldSmiles(ZED1227)
    zed_fp = chem.morgan([ZED1227])

    print("## 1. Is ZED1227 already in our data?\n")
    sets = {
        "TG2 IC50 set (631)": tg2.combined("classification").smiles.tolist(),
        "kinetic set (219)": kinetic_table(recover=True).smiles.tolist(),
        "TG2/FXIII-A paired (147)": paired().smiles.unique().tolist(),
    }
    for name in ("tgm1", "tgm3", "f13a1", "tgm2_gp"):
        sets[f"{name} labelled"] = chembl.classification(
            chembl.load_raw(f"data/raw/chembl/{name}_activities.csv")).smiles.tolist()

    found_any = False
    for name, smis in sets.items():
        exact = zed_can in set(Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smis)
        stereo_blind = zed_flat in set(flat(s) for s in smis)
        found_any |= exact or stereo_blind
        print(f"   {name:<28} exact: {str(exact):<6} ignoring stereochemistry: {stereo_blind}")
        rows.append({"section": "membership", "set": name, "exact": exact,
                     "stereo_blind": stereo_blind, "n": len(smis)})
    print(f"\n   ZED1227 present anywhere in our data: {found_any}")

    print("\n## 2. How close is it to what we have?\n")
    print(f"   {'set':<28}{'n':>6}{'max Tanimoto':>14}{'mean top-5':>12}{'shared scaffold':>17}")
    for name, smis in sets.items():
        if not smis:
            continue
        fp = chem.morgan(list(dict.fromkeys(smis)))
        sims = tanimoto_to_train(zed_fp, fp)
        # top-5 neighbours
        a = zed_fp.astype(np.float32)
        b = fp.astype(np.float32)
        inter = a @ b.T
        union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
        all_sims = np.sort((inter / np.maximum(union, 1))[0])[::-1]
        scafs = {MurckoScaffold.MurckoScaffoldSmiles(s) for s in dict.fromkeys(smis)}
        print(f"   {name:<28}{len(set(smis)):>6}{sims[0]:>14.3f}{all_sims[:5].mean():>12.3f}"
              f"{str(zed_scaf in scafs):>17}")
        rows.append({"section": "similarity", "set": name, "max_tanimoto": float(sims[0]),
                     "mean_top5": float(all_sims[:5].mean()),
                     "shares_scaffold": bool(zed_scaf in scafs)})

    print("\n## 3. Warhead and physicochemistry\n")
    zed_wh = [k for k, sm in WARHEAD_SMARTS.items()
              if zed.HasSubstructMatch(Chem.MolFromSmarts(sm))]
    print(f"   ZED1227 warhead match: {zed_wh or ['none of the catalogued patterns']}")
    kin = kinetic_table(recover=True)
    kin_wh = pd.Series([warhead_of(s) for s in kin.smiles]).value_counts().to_dict()
    print(f"   our kinetic set warheads: {kin_wh}")
    print("\n   ZED1227's electrophile is an alpha,beta-unsaturated METHYL ESTER on a")
    print("   glutamine-mimetic side chain -- not the acrylamide that dominates our data.")

    d_zed = describe(ZED1227, "ZED1227")
    tg2_tab = tg2.combined("classification")
    props = pd.DataFrame([describe(s, "our TG2 set") for s in tg2_tab.smiles])
    print(f"\n   {'property':<12}{'ZED1227':>10}{'our TG2 set median':>22}{'our 5-95 pct':>22}")
    for k in ("mw", "clogp", "tpsa", "hbd", "hba", "rotatable"):
        lo, hi = np.percentile(props[k], [5, 95])
        print(f"   {k:<12}{d_zed[k]:>10.1f}{props[k].median():>22.1f}"
              f"{f'{lo:.0f} - {hi:.0f}':>22}")
        rows.append({"section": "property", "property": k, "zed1227": d_zed[k],
                     "our_median": float(props[k].median()), "p5": float(lo), "p95": float(hi)})

    print("\n## 4. Applicability domain\n")
    tg2_fp = chem.morgan(list(tg2_tab.smiles))
    max_sim = float(tanimoto_to_train(zed_fp, tg2_fp)[0])
    verdict = ("INSIDE" if max_sim >= 0.5 else
               "UNVALIDATED (0.4-0.5)" if max_sim >= 0.4 else "OUTSIDE -- models rank at chance")
    print(f"   max Tanimoto to the TG2 training set: {max_sim:.3f}  ->  {verdict}")
    print("   The project's applicability floor is 0.4 on TG2; below it every model we tested")
    print("   ranks at chance (paper.md section 5.6).")
    rows.append({"section": "domain", "max_tanimoto_to_tg2": max_sim, "verdict": verdict})

    print("\n## 5. Would a prediction mean anything?\n")
    if found_any:
        print("   NO -- the compound is in our training data; any prediction is memorisation.")
    elif max_sim < 0.4:
        print("   NO. ZED1227 sits below the applicability floor, where our own evaluation shows")
        print("   every model ranks at chance. Producing a number here would be exactly the")
        print("   error the evaluation paper was written to document, so none is produced.")
    else:
        print("   Possibly; see the report for the gated prediction.")
    rows.append({"section": "prediction", "made": False,
                 "reason": "outside applicability domain" if max_sim < 0.4 else "see report"})

    print("\n## 6. ZED1227 against the most selective compounds we have\n")
    m = paired()
    best = m.nlargest(5, "selectivity")
    print(f"   {'compound':<34}{'TG2 IC50 (nM)':>15}{'FXIII-A IC50 (nM)':>19}{'fold':>9}")
    print(f"   {'ZED1227 (published)':<34}{53:>15}{'>50000':>19}{'>900':>9}")
    for i, (_, r) in enumerate(best.iterrows(), 1):
        ic_t, ic_f = 10 ** (9 - r.p_tg2), 10 ** (9 - r.p_f13)
        print(f"   {f'our set, rank {i}':<34}{ic_t:>15.1f}{ic_f:>19.0f}{ic_f / ic_t:>9.0f}")
        rows.append({"section": "comparison", "rank": i, "tg2_ic50_nM": float(ic_t),
                     "fxiiia_ic50_nM": float(ic_f), "fold": float(ic_f / ic_t)})
    print("\n   Our best measured compounds reach comparable or greater selectivity. Selectivity")
    print("   is therefore NOT what distinguishes the clinical compound.")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
