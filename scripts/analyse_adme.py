"""Is there an intestinal-exposure axis in the data we already hold, and is it enough to model?

zed1227_analysis.md ended on a claim that turns out to be too pessimistic: that intestinal exposure
"appears in no dataset we hold". It appears in none of our *target* files, because those were pulled
by target and hold enzyme inhibition only. ChEMBL keeps ADME under the molecule instead. Asking
molecule-by-molecule (scripts/fetch_adme.py) returns measured permeability, clearance, half-life,
plasma binding and stability for molecules that are already in our sets.

This script audits that data and asks three questions in order:

  1. what is actually measured, in which assay system, from which publication, for how many
     molecules that we hold a TG2 measurement for?
  2. how do our most selective compounds compare with ZED1227 on the exposure axis, rather than
     on potency and selectivity where they already win?
  3. is there enough of it to model, or only enough to describe?

Question 3 is asked last and answered honestly. Writes results/adme_measurements.csv (one
harmonised row per measurement, provenance kept) and results/adme_summary.csv. CPU, seconds, no
GPU training and no model is fitted.

Usage: python scripts/analyse_adme.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from scripts.analyse_zed1227 import ZED1227, describe
from scripts.audit_fxiiia import paired
from src.data import chembl, tg2

RAW = "data/raw/adme/chembl_adme.csv"
LIT = "data/curated/literature_adme.csv"
TG2_ACT = "data/raw/chembl/tgm2_activities.csv"
OUT_MEAS = "results/adme_measurements.csv"
OUT_SUM = "results/adme_summary.csv"

# ChEMBL's standard_type -> the property we are actually interested in. Types left out of this map
# (IC50, Potency, CC50, NADH oxidation, tR, FC, TI ...) are potency or assay-control readouts that
# happen to be filed as assay_type A; they are not ADME and are reported as excluded.
PROPERTY = {
    "Papp": "permeability", "Peff": "permeability", "permeability": "permeability",
    "Solubility": "solubility", "logD": "lipophilicity", "logP": "lipophilicity",
    "CHI": "lipophilicity", "pKa": "ionisation",
    "Stability": "stability", "T1/2": "half-life", "CL": "clearance", "CLH": "clearance",
    "PPB": "protein binding", "Fu": "protein binding",
    "AUC": "in vivo PK", "Cmax": "in vivo PK", "Tmax": "in vivo PK", "F": "in vivo PK",
    "Vd": "in vivo PK", "MRT": "in vivo PK", "MRT(t)": "in vivo PK", "Cp": "in vivo PK",
    "Biodistribution": "in vivo PK",
}

# Permeability units -> 10^-6 cm/s. 1 nm/s = 1e-7 cm/s = 0.1 x 10^-6 cm/s.
PERM_SCALE = {"nm/s": 0.1, "10^-6 cm/s": 1.0, "10'-6 cm/s": 1.0}


def assay_system(desc):
    """Which membrane the permeability was measured across. These are not interchangeable: a PAMPA
    Peff is passive diffusion through a lipid film, a Caco-2 Papp includes transporters, and an
    MDCK-MDR1 Papp is dominated by P-gp efflux."""
    d = (desc or "").lower()
    for pat, name in (("caco", "Caco-2"), ("mdck", "MDCK"), ("pampa", "PAMPA"),
                      ("jejun", "human jejunum")):
        if pat in d:
            return name
    return "unspecified"


def direction(desc):
    d = (desc or "").lower()
    if "apical to basolateral" in d or "apical side to basolateral" in d:
        return "A-B"
    if "basolateral" in d and "apical" in d:
        return "B-A"
    return "n/a"  # PAMPA and jejunum are single-direction by construction


def harmonise(df):
    """One row per measurement, with the value converted only where the conversion is exact."""
    d = df.copy()
    d["property"] = d.standard_type.map(PROPERTY)
    # 'Ratio' covers both MDCK-MDR1 efflux ratios and unrelated CYP-induction ratios, so it is
    # read from the assay text rather than the type. Efflux matters here: it decides how much of a
    # permeable molecule actually crosses the gut wall.
    efflux = d.standard_type.eq("Ratio") & d.assay_description.str.contains(
        "efflux", case=False, na=False)
    d.loc[efflux, "property"] = "efflux ratio"
    d["assay_system"] = d.assay_description.map(assay_system)
    d["direction"] = d.assay_description.map(direction)
    d["value_10e6_cm_s"] = np.where(
        d.property.eq("permeability") & d.standard_units.isin(PERM_SCALE),
        d.standard_value * d.standard_units.map(PERM_SCALE).fillna(np.nan), np.nan)
    return d


def main():
    raw = pd.read_csv(RAW)
    tg2_docs = set(pd.read_csv(TG2_ACT).document_chembl_id.dropna())
    tg2_tab = tg2.combined("classification")
    ic50 = chembl.regression(chembl.load_raw(TG2_ACT))
    sel = paired()

    # molecule_chembl_id -> standardized SMILES, from the files the IDs were drawn from
    fam = pd.concat([chembl.load_raw(f"data/raw/chembl/{n}_activities.csv")
                     for n in ("tgm2", "f13a1", "tgm1", "tgm3", "tgm2_gp")])
    id2smi = dict(zip(fam.molecule_chembl_id, fam.smiles))
    smi2id = {}
    for i, s in id2smi.items():
        smi2id.setdefault(s, i)

    d = harmonise(raw)
    d["smiles"] = d.molecule_chembl_id.map(id2smi)
    d["in_tg2_set"] = d.smiles.isin(set(tg2_tab.smiles))
    d["tg2_paper"] = d.document_chembl_id.isin(tg2_docs)
    d["measured"] = d.standard_value.notna()

    print("## 1. What ADME data exists for molecules we already hold?\n")
    print(f"   {len(raw)} ChEMBL assay_type-A rows for {raw.molecule_chembl_id.nunique()} of our "
          f"{len(set(id2smi))} transglutaminase-family molecules")
    nonadme = d[d.property.isna()]
    print(f"   {len(nonadme)} rows are potency or assay-control readouts filed as type A "
          f"({', '.join(sorted(set(nonadme.standard_type))[:6])} ...) -- excluded")
    nd = d[d.property.notna() & ~d.measured]
    print(f"   {len(nd)} rows are 'Not Determined' placeholders with no value -- excluded")
    a = d[d.property.notna() & d.measured].copy()
    print(f"   leaves {len(a)} measured ADME values on {a.molecule_chembl_id.nunique()} molecules "
          f"from {a.document_chembl_id.nunique()} publications\n")

    tab = pd.DataFrame({
        "TG2 paper": a[a.tg2_paper].groupby("property").molecule_chembl_id.nunique(),
        "incidental paper": a[~a.tg2_paper].groupby("property").molecule_chembl_id.nunique(),
        "in our TG2 set": a[a.in_tg2_set].groupby("property").molecule_chembl_id.nunique(),
    })
    print(tab.fillna(0).astype(int).to_string())
    print("\n   'incidental paper' means the molecule reached our files through a "
          "transglutaminase\n   measurement but the ADME comes from an unrelated programme; "
          "those values are real,\n   but they are not evidence about TG2 chemistry.")

    print("\n## 2. Permeability, the axis that matters for a gut-restricted drug\n")
    perm = a[a.property.eq("permeability")].copy()
    print(f"   {len(perm)} measured permeability values on {perm.molecule_chembl_id.nunique()} "
          f"molecules, in {perm.standard_units.nunique()} different unit systems:")
    print(perm.groupby(["assay_system", "direction", "standard_units"]).size()
          .rename("rows").to_frame().to_string())
    absorptive = perm[perm.direction.isin(["A-B", "n/a"]) & perm.value_10e6_cm_s.notna()]
    print(f"\n   {len(absorptive)} of them are absorptive-direction values convertible to "
          f"10^-6 cm/s, on {absorptive.molecule_chembl_id.nunique()} molecules "
          f"from {absorptive.document_chembl_id.nunique()} publications.")
    print("   Values are NOT pooled across assay systems; the system is carried on every row.\n")
    show = absorptive[absorptive.tg2_paper].sort_values("value_10e6_cm_s", ascending=False)
    print(f"   {'molecule':<16}{'system':<10}{'Papp/Peff (1e-6 cm/s)':>23}{'in TG2 set':>12}"
          f"  document")
    for _, r in show.iterrows():
        print(f"   {r.molecule_chembl_id:<16}{r.assay_system:<10}{r.value_10e6_cm_s:>23.2f}"
              f"{str(r.in_tg2_set):>12}  {r.document_chembl_id}")

    print("\n## 3. ZED1227 against our most selective compounds, on exposure\n")
    lit = pd.read_csv(LIT)
    zed_perm = lit[lit.property.eq("permeability_AB")].iloc[0]
    print(f"   ZED1227 has {(raw.molecule_chembl_id == 'CHEMBL4081588').sum()} ADME rows in "
          f"ChEMBL. Its only ADME source is the literature table:")
    print(f"     Caco-2 Papp {zed_perm.relation} {zed_perm.value} {zed_perm.units} "
          f"[{zed_perm.source}, {zed_perm.locator}]\n")

    cmp_rows = []
    for i, (_, r) in enumerate(sel.nlargest(5, "selectivity").iterrows(), 1):
        cid = smi2id.get(r.smiles)
        ic_t, ic_f = 10 ** (9 - r.p_tg2), 10 ** (9 - r.p_f13)
        print(f"   rank {i}: {cid}   TG2 {ic_t:.0f} nM / FXIII-A {ic_f:.0f} nM = "
              f"{ic_f / ic_t:.0f}x   (selectivity pair from {r.document_chembl_id})")
        mine = a[a.molecule_chembl_id.eq(cid)]
        if mine.empty:
            print("      no ADME measurement in ChEMBL")
        for _, v in mine.iterrows():
            val = (f"{v.value_10e6_cm_s:.1f} x 10^-6 cm/s" if pd.notna(v.value_10e6_cm_s)
                   else f"{v.standard_relation if pd.notna(v.standard_relation) else ''}"
                        f"{v.standard_value:g} "
                        f"{v.standard_units if pd.notna(v.standard_units) else ''}".strip())
            print(f"      {v.property:<16}{val:<22}{v.assay_description[:62]:<63}"
                  f"{v.document_chembl_id}")
        perm_me = absorptive[absorptive.molecule_chembl_id.eq(cid)]
        cmp_rows.append({"section": "comparison", "rank": i, "molecule_chembl_id": cid,
                         "tg2_ic50_nM": float(ic_t), "fxiiia_ic50_nM": float(ic_f),
                         "fold": float(ic_f / ic_t),
                         "papp_10e6_cm_s": float(perm_me.value_10e6_cm_s.max())
                         if len(perm_me) else np.nan,
                         "assay_system": "; ".join(sorted(set(perm_me.assay_system))) or "",
                         "n_adme_rows": int(len(mine)),
                         "document_chembl_id": r.document_chembl_id})

    print("\n   Both compounds that have exposure data were measured in the lysine-based SAR paper")
    print("   for HUNTINGTON'S disease, where the design goal was systemic and central exposure.")
    print("   Their absorptive permeability is one to two orders of magnitude above ZED1227's")
    print("   upper bound. Their MDR1 efflux ratios are high too: high passive permeability and")
    print("   heavy P-gp efflux coexist, so neither number alone settles what crosses the gut")
    print("   wall in vivo. That cuts both ways, and is the reason the permeability numbers here")
    print("   are reported as permeability and not as 'gut restriction'.")

    print("\n   Across the whole paired selectivity set, not just the top five:")
    pid = {smi2id[s] for s in sel.smiles.unique() if s in smi2id}
    pperm = absorptive[absorptive.molecule_chembl_id.isin(pid)]
    print(f"   {len(pid)} molecules have an exact IC50 on both enzymes; "
          f"{pperm.molecule_chembl_id.nunique()} of them have an absorptive")
    print(f"   permeability measurement. Values below 1 x 10^-6 cm/s, the bound reported for "
          f"ZED1227: {(pperm.value_10e6_cm_s < 1).sum()}")
    id2sel = {smi2id[r.smiles]: r for _, r in sel.iterrows() if r.smiles in smi2id}
    for _, v in pperm.sort_values("value_10e6_cm_s").iterrows():
        r = id2sel[v.molecule_chembl_id]
        ic_t, ic_f = 10 ** (9 - r.p_tg2), 10 ** (9 - r.p_f13)
        print(f"      {v.molecule_chembl_id:<16}{v.value_10e6_cm_s:>7.2f} "
              f"{'(' + v.assay_system + ')':<10}TG2 {ic_t:>6.0f} nM / "
              f"FXIII-A {ic_f:>6.0f} nM = {ic_f / ic_t:>5.0f}x")
    print("   The lowest of those is recorded as exactly 0 nm/s, which is a detection limit and")
    print("   not a measurement -- and that molecule is 1300 nM on TG2 with no selectivity at all.")
    print("   So no compound in our data combines TG2 potency, FXIII-A selectivity and measured")
    print("   permeability consistent with gut restriction. The candidate does not exist in our")
    print("   data; it is not that we have failed to rank it.")
    sum_rows_extra = [{"section": "combined_criteria", "paired_molecules": len(pid),
                       "with_permeability": int(pperm.molecule_chembl_id.nunique()),
                       "below_1e-6_cm_s": int((pperm.value_10e6_cm_s < 1).sum())}]

    print("\n## 4. Descriptors: ZED1227 against the molecules with measured permeability\n")
    zed = describe(ZED1227, "ZED1227")
    have = sorted(set(absorptive[absorptive.tg2_paper].smiles.dropna()))
    props = pd.DataFrame([describe(s, "measured") for s in have])
    print(f"   {'property':<12}{'ZED1227':>10}{'median':>10}{'min':>8}{'max':>8}"
          f"{'molecules above ZED1227':>26}")
    prop_rows = []
    for k in ("mw", "clogp", "tpsa", "hbd", "hba", "rotatable"):
        above = int((props[k] > zed[k]).sum())
        print(f"   {k:<12}{zed[k]:>10.1f}{props[k].median():>10.1f}{props[k].min():>8.1f}"
              f"{props[k].max():>8.1f}{f'{above} of {len(props)}':>26}")
        prop_rows.append({"section": "descriptor", "property": k, "zed1227": zed[k],
                          "measured_median": float(props[k].median()),
                          "measured_min": float(props[k].min()),
                          "measured_max": float(props[k].max()),
                          "n_above_zed1227": above, "n": len(props)})
    ndoc = absorptive[absorptive.tg2_paper].document_chembl_id.nunique()
    print(f"\n   n = {len(props)} molecules with an absorptive permeability measurement in a TG2")
    print("   publication. ZED1227 is the most polar molecule in the comparison and its measured")
    print("   permeability is the lowest, but on weight and flexibility it sits inside the range,")
    print(f"   so the descriptors do not separate it. With {len(props)} molecules from {ndoc} "
          f"publications, a\n   descriptor difference and a publication difference cannot be "
          f"told apart.")

    print("\n## 5. Is this enough to model?\n")
    sum_rows = []
    for prop in ("permeability", "efflux ratio", "solubility", "stability", "half-life",
                 "clearance", "protein binding", "lipophilicity", "in vivo PK"):
        p = a[a.property.eq(prop)]
        pt = p[p.tg2_paper]
        with_ic50 = pt[pt.smiles.isin(set(ic50.smiles))]
        scaf = {MurckoScaffold.MurckoScaffoldSmiles(s) for s in set(pt.smiles.dropna())}
        censored = (pt.standard_relation.fillna("=") != "=").sum()
        print(f"   {prop:<16} {pt.molecule_chembl_id.nunique():>3} molecules from TG2 papers, "
              f"{pt.document_chembl_id.nunique():>2} publications, {len(scaf):>2} scaffolds, "
              f"{with_ic50.molecule_chembl_id.nunique():>3} with an exact TG2 IC50, "
              f"{censored:>2} of {len(pt):>3} values censored")
        sum_rows.append({"section": "sufficiency", "property": prop,
                         "molecules_tg2_papers": int(pt.molecule_chembl_id.nunique()),
                         "publications": int(pt.document_chembl_id.nunique()),
                         "scaffolds": len(scaf),
                         "with_exact_tg2_ic50": int(with_ic50.molecule_chembl_id.nunique()),
                         "values": int(len(pt)), "values_censored": int(censored),
                         "molecules_all_sources": int(p.molecule_chembl_id.nunique())})

    pt = absorptive[absorptive.tg2_paper]
    per_doc = pt.groupby("document_chembl_id").molecule_chembl_id.nunique()
    print(f"\n   The absorptive permeability set is {pt.molecule_chembl_id.nunique()} molecules "
          f"spread over {len(per_doc)} publications:")
    print("   " + ", ".join(f"{k} n={v}" for k, v in per_doc.sort_values(ascending=False).items()))
    print("\n   The project's own evaluation work requires a cluster or scaffold split to mean")
    print("   anything, and the largest single-publication, single-assay block here is "
          f"{per_doc.max()}\n   molecules. A split cannot be built inside that. No model is fitted.")

    os.makedirs("results", exist_ok=True)
    keep = ["molecule_chembl_id", "smiles", "property", "standard_type", "standard_relation",
            "standard_value", "standard_units", "value", "units", "value_10e6_cm_s",
            "assay_system", "direction", "assay_description", "document_chembl_id", "tg2_paper",
            "in_tg2_set", "data_validity_comment", "activity_comment"]
    out = a.reindex(columns=keep).assign(source="ChEMBL assay_type=A", evidence="experimental")
    lit_out = lit.assign(source=lit.source, evidence=lit.evidence, assay_system=lit.assay,
                         molecule_chembl_id=lit.chembl_id, standard_relation=lit.relation,
                         standard_value=lit.value, standard_units=lit.units,
                         assay_description=lit.conditions)
    pd.concat([out, lit_out.reindex(columns=out.columns.tolist() + ["compound", "locator"])],
              ignore_index=True).to_csv(OUT_MEAS, index=False)
    pd.DataFrame(sum_rows + sum_rows_extra + cmp_rows + prop_rows).to_csv(OUT_SUM, index=False)
    print(f"\nwrote {OUT_MEAS} and {OUT_SUM}")


if __name__ == "__main__":
    main()
