"""TG2 inhibitor data from Lailabcode/celiac-informatics (Wichka & Lai, CSBJ 2024).

Source of truth is TG2_compound_smiles.csv (PubChem CID, Type, Value, Activity, qualifier, SMILES).
The repo's pre-featurized CSVs drop SMILES/IDs, so we rebuild features from SMILES instead.
"""
import numpy as np
import pandas as pd

from src.data.chem import canonical

RAW = "data/raw/tg2/TG2_compound_smiles.csv"


def load_raw(path=RAW):
    df = pd.read_csv(path)
    df["smiles"] = df["Smiles"].map(canonical)
    return df.dropna(subset=["smiles"])


def classification(df, ic50_um=1.0):
    """Potent=1 if median exact IC50 <= ic50_um (pIC50 >= 6 for 1 uM), else 0.
    Molecules with no exact IC50 but a censored ">X" bound at or above the threshold are added as
    confirmed negatives: ">10 uM" proves "not potent at 1 uM" even though it gives no pIC50. A bound
    below the threshold (">0.5 uM" against a 1 uM cut) proves nothing, so it stays out.
    The repo's own Active/Inactive labels (10 uM cut) are 90% active, too imbalanced to evaluate on."""
    pot = regression(df)
    pot["y"] = (pot["y"] >= 6 - np.log10(ic50_um)).astype(int)
    cens = df[(df["Type"] == "IC50") & (df["Acqual"] == ">") & df["Value"].ge(ic50_um)]
    cens = cens[~cens["smiles"].isin(pot["smiles"])].drop_duplicates("smiles")
    return pd.concat([pot, cens[["smiles"]].assign(y=0)], ignore_index=True)


def regression(df):
    """pIC50 = 6 - log10(IC50 in uM), exact ('=') IC50 measurements only, median per molecule.
    Censored values ('>') are excluded rather than imputed."""
    d = df[(df["Type"] == "IC50") & (df["Acqual"] == "=") & df["Value"].gt(0)]
    y = (6 - np.log10(d["Value"])).groupby(d["smiles"]).median()
    return y.rename("y").reset_index()


def combined(task="classification", ic50_um=1.0):
    """Union of the PubChem export and ChEMBL's human TG2 target (CHEMBL2730), one row per molecule.

    Merging is safe because the two sources agree where they overlap: Spearman 1.000 on the 173
    molecules both give an exact pIC50 for, and identical class labels on all 196 they share
    (scripts/check_sources.py). PubChem's value is kept for a shared molecule so the earlier
    PubChem-only results stay a strict subset. `source` records where each row came from.
    """
    from src.data import chembl

    raw_p, raw_c = load_raw(), chembl.load_raw()
    if task == "classification":
        a, b = classification(raw_p, ic50_um), chembl.classification(raw_c, ic50_um)
    else:
        a, b = regression(raw_p), chembl.regression(raw_c)
    a = a[["smiles", "y"]].assign(source="pubchem")
    b = b.loc[~b["smiles"].isin(a["smiles"]), ["smiles", "y"]].assign(source="chembl")
    return pd.concat([a, b], ignore_index=True)
