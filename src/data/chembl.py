"""Human TG2 (CHEMBL2730) activities from ChEMBL, standardized the same way as the PubChem set.

Only IC50 in nM is kept. Ki, Kinact, kon/koff and %-inhibition measure different quantities and are
excluded rather than converted: a Ki and an IC50 are not interchangeable without the substrate Km,
which ChEMBL does not record here.
"""
import numpy as np
import pandas as pd

from src.data.chem import canonical

RAW = "data/raw/chembl/tg2_activities.csv"


def load_raw(path=RAW):
    df = pd.read_csv(path).dropna(subset=["canonical_smiles"])
    df["smiles"] = df["canonical_smiles"].map(canonical)
    return df.dropna(subset=["smiles"])


def _usable(df):
    """IC50 in nM from a binding assay, positive, not flagged by ChEMBL's own curation.

    The assay_type filter matters only outside the transglutaminase family, whose files are 100%
    type B. CDK2 mixes in functional (F) and ADMET (A) rows whose IC50 is a cell-viability readout
    rather than enzyme inhibition -- its F rows have a median of 14,100 nM against 200 nM for the
    binding assays -- so pooling them would label the same chemistry inconsistently.

    Censored rows with a '<' bound are still ignored rather than read as positives. Keeping one
    rule across targets is worth more than the 65 CDK2 and 10 TG2 rows it costs, since the whole
    point of the second target is a like-for-like comparison.
    """
    return df[(df["standard_type"] == "IC50") & (df["standard_units"] == "nM")
              & df["standard_value"].gt(0) & df["data_validity_comment"].isna()
              & (df["assay_type"] == "B")]


def regression(df):
    """pIC50 = 9 - log10(IC50 in nM), exact ('=') measurements only, median per molecule.
    Carries provenance: how many measurements and how many separate papers back each value."""
    d = _usable(df)
    d = d[d["standard_relation"] == "="]
    p = 9 - np.log10(d["standard_value"])
    g = p.groupby(d["smiles"])
    return pd.DataFrame({
        "y": g.median(),
        "n_meas": g.size(),
        "n_docs": d.groupby("smiles")["document_chembl_id"].nunique(),
        "spread_log": g.max() - g.min(),
    }).reset_index()


def classification(df, ic50_um=1.0):
    """Same rule as the PubChem set: potent if median exact pIC50 clears the cut, plus censored
    '>X' molecules whose bound already proves they are not potent."""
    pot = regression(df)
    pot["y"] = (pot["y"] >= 9 - np.log10(ic50_um * 1000)).astype(int)
    d = _usable(df)
    cens = d[(d["standard_relation"] == ">") & d["standard_value"].ge(ic50_um * 1000)]
    cens = cens[~cens["smiles"].isin(pot["smiles"])].drop_duplicates("smiles")
    add = cens[["smiles"]].assign(y=0, n_meas=1, n_docs=1, spread_log=0.0)
    return pd.concat([pot, add], ignore_index=True)
