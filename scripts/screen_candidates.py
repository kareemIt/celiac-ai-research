"""Rank approved drugs as computational TG2-inhibition hypotheses.

NOTHING HERE IS A DISCOVERY. Every output is a model prediction on a molecule that has, as far as
these datasets show, never been assayed against TG2. They are hypotheses to test, not hits.

Pipeline:
  1. standardize the library and drop anything with a known TG2 measurement in either source
  2. train the ensemble on 90% of the TG2 data (scaffold groups), hold 10% for conformal calibration
  3. predict, with ensemble agreement, conformal label sets, and distance to the training set
  4. flag structural liabilities (PAINS, Brenk) and covalent warheads
  5. shortlist on prediction AND applicability AND cleanliness, then pick a diverse subset

Usage: python scripts/screen_candidates.py [--top 12]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, FilterCatalog
from rdkit.ML.Cluster import Butina

from scripts.audit_splits import tanimoto_to_train
from scripts.run_baselines import XGB
from scripts.run_generalization import RF_CFG
from scripts.run_uncertainty import conformal
from src.data import chem, chembl, tg2

LIB = "data/raw/screening/chembl_approved.csv"
OUT = "results/candidates_tg2.csv"
ALPHA = 0.2          # conformal level; 0.1 is too conservative to separate anything here
DOMAIN_LO = 0.40     # below 0.4 Tanimoto the model ranks at chance (docs/generalization.md)
CLUSTER_CUT = 0.65   # Butina distance cutoff for the diversity pick

# TG2 is a cysteine transferase and most potent published inhibitors are covalent. A warhead is
# therefore evidence the model may be right, and simultaneously a selectivity/toxicity liability.
WARHEADS = {
    "acrylamide": "C=CC(=O)[NX3]",
    "chloroacetamide": "[Cl,Br,I]C[CX3](=O)[NX3]",
    "epoxide": "C1OC1",
    "maleimide": "O=C1C=CC(=O)N1",
    "vinyl_sulfone": "C=C[SX4](=O)(=O)",
    "dihydroisoxazole": "C1[CX4]C(=NO1)[Cl,Br]",  # the classic TG2 warhead (e.g. ERW1041E)
    "aldehyde": "[CX3H1](=O)[#6]",
    # nitriles react reversibly with catalytic cysteines; this is nirmatrelvir's warhead and the
    # reason a covalent cysteine-protease drug can plausibly touch a cysteine transferase
    "nitrile": "[NX1]#[CX2][CX4,CX3]",
    "boronic_acid": "[BX3]([OX2H])[OX2H]",
    "alpha_ketoamide": "[CX3](=O)[CX3](=O)[NX3]",  # telaprevir/boceprevir class
}


def filter_catalog():
    p = FilterCatalog.FilterCatalogParams()
    for c in ("PAINS", "BRENK"):
        p.AddCatalog(getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, c))
    return FilterCatalog.FilterCatalog(p)


def known_tg2_smiles():
    """Every molecule with any TG2 measurement in either source, not just the labelled 631."""
    return set(tg2.load_raw()["smiles"]) | set(chembl.load_raw()["smiles"])


def load_library(path=LIB):
    df = pd.read_csv(path)
    df = df[(df.molecule_type == "Small molecule") & df.smiles.notna()].copy()
    df["smiles"] = df["smiles"].map(chem.canonical)          # largest fragment, same as training
    df = df.dropna(subset=["smiles"]).drop_duplicates("smiles")
    known = known_tg2_smiles()
    overlap = df["smiles"].isin(known)
    print(f"library: {len(df)} standardized unique small molecules, "
          f"{overlap.sum()} already have a TG2 measurement and are removed")
    return df[~overlap].reset_index(drop=True)


def annotate(df):
    cat = filter_catalog()
    warheads = {k: Chem.MolFromSmarts(v) for k, v in WARHEADS.items()}
    rows = []
    for s in df["smiles"]:
        m = Chem.MolFromSmiles(s)
        hits = [e.GetDescription() for e in cat.GetMatches(m)]
        wh = [k for k, patt in warheads.items() if patt is not None and m.HasSubstructMatch(patt)]
        rows.append({
            "qed": QED.qed(m), "mw": Descriptors.MolWt(m), "logp": Descriptors.MolLogP(m),
            "tpsa": Descriptors.TPSA(m), "heavy_atoms": m.GetNumHeavyAtoms(),
            "alerts": len(hits), "alert_names": "; ".join(sorted(set(hits))[:3]),
            "warheads": ",".join(wh),
        })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def fit_and_predict(lib_smiles, seed=0):
    """Ensemble trained on 90% of the TG2 data; the held-out 10% is the conformal calibration set.
    That fold was never used to choose anything here, so it is a valid calibration set."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    table = tg2.combined("classification")
    smiles, y = table["smiles"].tolist(), table["y"].to_numpy()
    X = chem.morgan(smiles)
    tr, va, cal = chem.scaffold_split(smiles, seed=seed)
    fit = np.concatenate([tr, va])
    Xlib = chem.morgan(lib_smiles)

    models = {"rf": RandomForestClassifier(random_state=seed, **RF_CFG),
              "xgb": XGBClassifier(**XGB, random_state=seed),
              "logreg": LogisticRegression(C=1.0, max_iter=5000)}
    P_cal, P_lib = {}, {}
    for name, m in models.items():
        m.fit(X[fit], y[fit])
        P_cal[name] = m.predict_proba(X[cal])[:, 1]
        P_lib[name] = m.predict_proba(Xlib)[:, 1]

    p_cal = np.mean(list(P_cal.values()), axis=0)
    p_lib = np.mean(list(P_lib.values()), axis=0)
    votes = np.sum([(p >= 0.5).astype(int) for p in P_lib.values()], axis=0)
    sets = conformal(p_cal, y[cal], p_lib, ALPHA, mondrian=True)

    potent_idx = [i for i in fit if y[i] == 1]
    return pd.DataFrame({
        "p_potent": p_lib,
        "models_agree": votes,                       # 0-3 models calling it potent
        "spread": np.std(list(P_lib.values()), axis=0),
        "conformal_potent_only": sets[1] & ~sets[0],
        "conformal_both": sets[1] & sets[0],
        "conformal_empty": ~(sets[0] | sets[1]),
        "max_tanimoto_train": tanimoto_to_train(Xlib, X[fit]),
        "max_tanimoto_potent": tanimoto_to_train(Xlib, X[potent_idx]),
    }), len(fit), len(cal)


def diverse_top(df, n, cutoff=CLUSTER_CUT):
    """Butina-cluster the shortlist and take the best-scoring member of each cluster, so the table is
    not n analogs of one scaffold."""
    fps = chem.morgan(df["smiles"].tolist()).astype(np.float32)
    inter = fps @ fps.T
    union = fps.sum(1)[:, None] + fps.sum(1)[None, :] - inter
    dist = 1 - inter / np.maximum(union, 1)
    flat = [dist[i, j] for i in range(1, len(df)) for j in range(i)]
    clusters = Butina.ClusterData(flat, len(df), cutoff, isDistData=True)
    pick = [max(c, key=lambda i: df["p_potent"].iloc[i]) for c in clusters]
    out = df.iloc[sorted(pick, key=lambda i: -df["p_potent"].iloc[i])]
    return out.head(n), len(clusters)


def reason(r):
    bits = [f"ensemble p={r.p_potent:.2f}", f"{int(r.models_agree)}/3 models agree"]
    bits.append("conformal set = {potent}" if r.conformal_potent_only else "conformal set ambiguous")
    bits.append(f"nearest known inhibitor {r.max_tanimoto_potent:.2f} Tanimoto")
    if r.warheads:
        bits.append(f"covalent warhead ({r.warheads}) consistent with TG2 cysteine chemistry")
    if r.alerts == 0:
        bits.append("no PAINS/Brenk alert")
    return "; ".join(bits)


def main(top):
    lib = annotate(load_library())
    preds, n_fit, n_cal = fit_and_predict(lib["smiles"].tolist())
    df = pd.concat([lib, preds], axis=1)
    df["novelty"] = 1 - df["max_tanimoto_potent"]
    df.sort_values("p_potent", ascending=False).to_csv(OUT, index=False)
    print(f"model trained on {n_fit} TG2 molecules, conformal calibrated on {n_cal}")

    print(f"\n## Library screened: {len(df)} approved drugs")
    print(f"  predicted potent (p>=0.5):            {(df.p_potent >= 0.5).sum()}")
    print(f"  all three models agree potent:        {(df.models_agree == 3).sum()}")
    print(f"  conformal set = {{potent}} alone:       {df.conformal_potent_only.sum()}")
    print(f"  conformal set ambiguous {{both}}:       {df.conformal_both.sum()}")
    print(f"  inside applicability domain (>={DOMAIN_LO}):  {(df.max_tanimoto_train >= DOMAIN_LO).sum()}")
    print(f"  carrying a covalent warhead:          {(df.warheads != '').sum()}")
    print(f"  free of PAINS/Brenk alerts:           {(df.alerts == 0).sum()}")

    print("\n## Applicability domain: how much of the library can the model speak to at all?")
    for lo, hi, label in [(0.5, 1.01, "0.50+  reliable (AUROC ~0.90 in this band)"),
                          (0.4, 0.50, "0.40-0.50  marginal (AUROC ~0.80)"),
                          (0.0, 0.40, "below 0.40  extrapolation (AUROC ~chance)")]:
        n = ((df.max_tanimoto_train >= lo) & (df.max_tanimoto_train < hi)).sum()
        print(f"  {label:<48} {n:5d}  ({n / len(df):.0%})")

    strict = df[(df.models_agree == 3) & df.conformal_potent_only
                & (df.max_tanimoto_train >= DOMAIN_LO) & (df.alerts == 0)]
    print(f"\n## Tier 1 — passes every criterion (unanimous, conformal {{potent}}, in domain, no alert): "
          f"{len(strict)}")

    # Tier 2 keeps the applicability requirement, which is the one that actually protects the
    # conclusion, and relaxes unanimity and the structural alerts.
    pool = df[(df.models_agree >= 2) & (df.max_tanimoto_train >= DOMAIN_LO) & (df.p_potent >= 0.4)]
    print(f"## Tier 2 — in domain, >=2/3 models, p>=0.4: {len(pool)}")
    if pool.empty:
        print("  Nothing qualifies. That is a result, not a failure: see docs/candidates.md.")
        return
    picks, n_clusters = diverse_top(pool, top)
    picks = picks.assign(tier=np.where(picks.index.isin(strict.index), 1, 2),
                         reason=[reason(r) for r in picks.itertuples()])
    picks.to_csv(OUT.replace(".csv", "_shortlist.csv"), index=False)
    print(f"  {n_clusters} Butina clusters at {CLUSTER_CUT}; best of each, {len(picks)} shown\n")
    cols = ["chembl_id", "name", "tier", "p_potent", "models_agree", "spread", "max_tanimoto_train",
            "max_tanimoto_potent", "novelty", "qed", "alerts", "warheads"]
    print(picks[cols].to_string(index=False, float_format="%.3f"))
    print("\nEvery row is a COMPUTATIONAL HYPOTHESIS about an untested molecule, not a discovery.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    main(ap.parse_args().top)
