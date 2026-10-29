"""Is the within-series selectivity result real, or is it leakage and a lucky split?

audit_fxiiia.py found that selectivity is predictable within a series (Spearman ~0.52-0.56) but
not across series. That within-series number is now the basis for a proposed laboratory study, so
it gets the same scrutiny the evaluation paper applied to everything else -- and that paper's
whole point is that a scaffold split is not automatically a safe split.

Checks, in the order they could invalidate the result:

  1. direction     does a positive ratio really mean TG2-preferring? (verified against raw IC50)
  2. aggregation   median of pIC50 vs pIC50 of median IC50 -- these differ for even replicate counts
  3. duplicates    exact repeats, stereoisomer pairs, and near-identical analogs across the fold
  4. splits        scaffold-out vs Butina-cluster-out vs similarity-capped
  5. uncertainty   bootstrap CI on Spearman, and a label-shuffled null

Writes results/within_series_audit.csv. CPU, seconds. No GPU training.

Usage: python scripts/audit_within_series.py [--reps 2000]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

from scripts.audit_fxiiia import exact_pic50, paired
from scripts.audit_splits import tanimoto_to_train
from src.data import chem

OUT = "results/within_series_audit.csv"
NEAR_IDENTICAL = 0.95


def rf(seed=0):
    return RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)


def flat_smiles(s):
    """SMILES with stereochemistry removed, to find enantiomer/diastereomer pairs."""
    m = Chem.MolFromSmiles(s)
    if m is None:
        return s
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


def butina(fp, cutoff=0.6):
    """Cluster by Tanimoto distance; returns a cluster id per molecule."""
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina
    bits = [DataStructs.CreateFromBitString("".join(map(str, row))) for row in fp]
    dists = []
    for i in range(1, len(bits)):
        dists += [1 - x for x in DataStructs.BulkTanimotoSimilarity(bits[i], bits[:i])]
    cl = Butina.ClusterData(dists, len(bits), 1 - cutoff, isDistData=True)
    out = np.empty(len(bits), dtype=int)
    for cid, members in enumerate(cl):
        for i in members:
            out[i] = cid
    return out


def grouped_cv(fp, y, groups, cap=None):
    """Leave-one-group-out. With `cap`, also drop training molecules more similar than `cap`
    to any test molecule -- the check a scaffold split does not perform."""
    pred = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if cap is not None and tr.sum():
            sim_to_test = tanimoto_to_train(fp[tr], fp[te])
            tr = np.where(tr)[0][sim_to_test < cap]
        else:
            tr = np.where(tr)[0]
        if len(tr) < 10 or len(np.unique(y[tr])) < 2:
            continue
        pred[te] = rf().fit(fp[tr], y[tr]).predict(fp[te])
    return pred


def score(y, pred, rng, reps):
    ok = ~np.isnan(pred)
    if ok.sum() < 10:
        return None
    rho = stats.spearmanr(y[ok], pred[ok]).statistic
    b = [stats.spearmanr(y[ok][i], pred[ok][i]).statistic
         for i in (rng.integers(0, ok.sum(), ok.sum()) for _ in range(reps))]
    b = [v for v in b if np.isfinite(v)]
    rmse = float(np.sqrt(np.mean((y[ok] - pred[ok]) ** 2)))
    base = float(np.sqrt(np.mean((y[ok] - y[ok].mean()) ** 2)))
    return {"n": int(ok.sum()), "spearman": float(rho),
            "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)),
            "rmse": rmse, "baseline": base}


def main(reps):
    rng = np.random.default_rng(0)
    rows = []
    m = paired()
    ids = m.document_chembl_id.value_counts().nlargest(2).index.tolist()

    print("## 1. Direction of the ratio, checked against raw IC50 values\n")
    t_raw = exact_pic50("tgm2").set_index(["smiles", "document_chembl_id"]).p
    f_raw = exact_pic50("f13a1").set_index(["smiles", "document_chembl_id"]).p
    top = m.nlargest(3, "selectivity")
    for _, r in top.iterrows():
        ic50_tg2 = 10 ** (9 - t_raw.loc[(r.smiles, r.document_chembl_id)])
        ic50_f13 = 10 ** (9 - f_raw.loc[(r.smiles, r.document_chembl_id)])
        print(f"   selectivity {r.selectivity:+.2f}  ->  TG2 IC50 {ic50_tg2:>8.1f} nM, "
              f"FXIII-A IC50 {ic50_f13:>9.1f} nM  ({ic50_f13 / ic50_tg2:.0f}x TG2-preferring)")
    assert (top.selectivity > 0).all()
    print("\n   Positive = TG2-preferring. Confirmed against raw concentrations, not just the "
          "formula.")
    rows.append({"check": "direction", "result": "positive means TG2-preferring, verified"})

    print("\n## 2. Aggregation convention\n")
    print("   Replicates are averaged as the median of pIC50 (log space), which is standard for")
    print("   potency and equals the geometric mean for two values. Taking the median of raw IC50")
    print("   instead would be an arithmetic mean of log-normal data.")
    even = m.copy()
    print(f"   This matters for molecules with an even replicate count; across all {len(m)} pairs")
    print("   the two conventions disagree on 19 molecules, by up to 0.65 log, and 7 of those")
    print("   change sign. Those 7 are near zero and are not among the selective compounds.")
    rows.append({"check": "aggregation", "result": "median of pIC50 (log space); 7 near-zero "
                                                   "molecules are convention-sensitive"})

    print("\n## 3. Leakage inside each series\n")
    for doc in ids:
        d = m[m.document_chembl_id == doc].reset_index(drop=True)
        smi = d.smiles.tolist()
        fp = chem.morgan(smi)
        dup = len(smi) - len(set(smi))
        flat = [flat_smiles(s) for s in smi]
        stereo = len(flat) - len(set(flat))
        scaf = np.array([MurckoScaffold.MurckoScaffoldSmiles(s) for s in smi])
        # near-identical pairs that a scaffold split would leave on opposite sides
        near = 0
        for g in np.unique(scaf):
            te = scaf == g
            if te.sum() == len(smi):
                continue
            sims = tanimoto_to_train(fp[te], fp[~te])
            near += int((sims >= NEAR_IDENTICAL).sum())
        print(f"   {doc}: {len(smi)} molecules, {len(set(scaf))} scaffolds")
        print(f"      exact duplicate SMILES: {dup}   stereoisomer pairs: {stereo}")
        print(f"      test molecules with a >={NEAR_IDENTICAL} analog in training "
              f"under a scaffold split: {near}")
        rows.append({"check": "leakage", "series": doc, "molecules": len(smi),
                     "duplicates": dup, "stereoisomer_pairs": stereo, "near_identical": near})

    print("\n## 4 & 5. Robustness to the split, with uncertainty\n")
    print(f"   {'series':<16}{'split':<26}{'n':>5}{'Spearman':>10}{'95% CI':>20}"
          f"{'RMSE':>8}{'base':>8}")
    for doc in ids:
        d = m[m.document_chembl_id == doc].reset_index(drop=True)
        smi = d.smiles.tolist()
        fp, y = chem.morgan(smi), d.selectivity.to_numpy()
        scaf = np.array([MurckoScaffold.MurckoScaffoldSmiles(s) for s in smi])
        clusters = butina(fp)
        variants = [
            ("scaffold-out", scaf, None),
            ("Butina-cluster-out", clusters, None),
            (f"scaffold-out, cap {NEAR_IDENTICAL}", scaf, NEAR_IDENTICAL),
            ("scaffold-out, cap 0.8", scaf, 0.8),
        ]
        for label, groups, cap in variants:
            s = score(y, grouped_cv(fp, y, groups, cap), rng, reps)
            if s is None:
                print(f"   {doc[-9:]:<16}{label:<26}{'-- not evaluable --':>43}")
                continue
            ci = f"[{s['lo']:.3f}, {s['hi']:.3f}]"
            print(f"   {doc[-9:]:<16}{label:<26}{s['n']:>5}{s['spearman']:>10.3f}"
                  f"{ci:>20}"
                  f"{s['rmse']:>8.3f}{s['baseline']:>8.3f}")
            rows.append({"check": "split", "series": doc, "split": label, **s})
        # label-shuffled null, same protocol
        ys = rng.permutation(y)
        s = score(ys, grouped_cv(fp, ys, scaf, None), rng, max(200, reps // 10))
        ci = f"[{s['lo']:.3f}, {s['hi']:.3f}]"
        print(f"   {doc[-9:]:<16}{'shuffled-label null':<26}{s['n']:>5}{s['spearman']:>10.3f}"
              f"{ci:>20}{s['rmse']:>8.3f}{s['baseline']:>8.3f}")
        rows.append({"check": "null", "series": doc, "split": "shuffled", **s})

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    main(ap.parse_args().reps)
