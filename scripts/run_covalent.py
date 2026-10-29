"""Can we predict covalent inhibition efficiency (kinact/Ki) for a TG2 acrylamide?

Phase 3 of the covalent work, run because the audit justified it and no further.

What the audit established (scripts/audit_covalent.py):
  - 174 molecules carry an exact, unit-harmonizable kinact/Ki, spanning 40 to 1.3e5 /M.s
  - they come from only 6 papers, and 169 of the 174 carry the same warhead (an acrylamide)
  - between-paper SD of medians is 0.50 log against a within-paper SD of 0.57 log

The constant warhead is what makes this worth doing: with one electrophile class, a model cannot
succeed by spotting the reactive group, so any signal has to come from the recognition element.
The six papers are what makes it fragile, which is why leave-one-paper-out is the primary split and
a scaffold split is reported beside it only to show the optimism gap.

Kinetic labels are never mixed with the IC50 labels: a second-order rate constant and a
concentration are different quantities.

Baselines, in increasing order of what they would prove:
  mean        predict the training papers' mean for everything -- what a model must beat
  n_heavy     predict from molecular size alone -- the cheapest real descriptor
  ridge / rf  Morgan fingerprints

Writes results/covalent.csv. CPU, under a minute.

Usage: python scripts/run_covalent.py [--reps 2000]
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
from sklearn.linear_model import Ridge

from rdkit import Chem

from src.data import chem, chembl, kinetic_units

OUT = "results/covalent.csv"
OUT_RECOVERED = "results/covalent_recovered.csv"


def MODELS_RF():
    return RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1)
RAW = "data/raw/chembl/tgm2_activities.csv"
# kinact/Ki is a second-order rate constant; ChEMBL stores it in three different units.
TO_PER_M_S = {"/M/s": 1.0, "10^3/M/min": 1000 / 60, "10'3/M/min": 1000 / 60}


def warhead_of(smiles):
    """Which electrophile class a molecule carries. Used as a split, never as a feature."""
    pats = {"dihydroisoxazole": "C1=NOCC1", "acrylamide": "C=CC(=O)N",
            "chloroacetamide": "ClCC(=O)N"}
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return "other"
    for name, sm in pats.items():  # dihydroisoxazole first: it never co-occurs with acrylamide
        if m.HasSubstructMatch(Chem.MolFromSmarts(sm)):
            return name
    return "other"


def kinetic_table(path=RAW, recover=False):
    """One row per molecule: median log10(kinact/Ki in /M.s), with its source paper.

    Only exact ('=') rows in a convertible unit survive. Rows with no unit at all cannot be placed
    on a common scale and are dropped rather than guessed at."""
    r = chembl.load_raw(path)
    if recover:
        r = kinetic_units.apply(r)
    d = r[(r.standard_type == "Ratio") & (r.standard_relation == "=")
          & r.standard_units.isin(TO_PER_M_S) & r.standard_value.notna()].copy()
    if d.empty:  # every row censored or unitless; the multiply below would raise on empty columns
        return pd.DataFrame(columns=["smiles", "logk", "paper", "n_meas"])
    d["logk"] = np.log10(d.standard_value.astype(float) * d.standard_units.map(TO_PER_M_S))
    g = d.groupby("smiles").agg(logk=("logk", "median"),
                                paper=("document_chembl_id", "first"),
                                n_meas=("logk", "size"))
    return g.reset_index()


def evaluate(name, X, y, groups, reps, rng):
    """Leave-one-group-out. Returns pooled RMSE and the Spearman correlation inside each held-out
    group -- ranking analogs within a new paper is the question a prioritization would ask."""
    preds, rhos = np.full(len(y), np.nan), []
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if te.sum() < 3 or tr.sum() < 10:
            continue
        if name == "mean":
            p = np.full(te.sum(), y[tr].mean())
        else:
            model = {"n_heavy": Ridge(alpha=1.0), "ridge": Ridge(alpha=1.0),
                     "rf": RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1)}[name]
            p = model.fit(X[tr], y[tr]).predict(X[te])
        preds[te] = p
        # A constant prediction has no ranking to score, which is exactly the mean baseline.
        if np.std(p) > 0 and te.sum() > 4:
            rhos.append(stats.spearmanr(y[te], p).statistic)
    ok = ~np.isnan(preds)
    rmse = float(np.sqrt(np.mean((y[ok] - preds[ok]) ** 2)))
    b = [np.sqrt(np.mean((y[ok][i] - preds[ok][i]) ** 2))
         for i in (rng.integers(0, ok.sum(), ok.sum()) for _ in range(reps))]
    rho = float(np.mean(rhos)) if rhos else np.nan
    return {"model": name, "n": int(ok.sum()), "rmse": rmse,
            "rmse_lo": float(np.percentile(b, 2.5)), "rmse_hi": float(np.percentile(b, 97.5)),
            "mean_within_group_spearman": rho, "n_groups_scored": len(rhos),
            "_pred": preds, "_y": y}


def main(reps, recover):
    rng = np.random.default_rng(0)
    t = kinetic_table(recover=recover)
    y = t.logk.to_numpy()
    smiles = t.smiles.tolist()
    papers = t.paper.to_numpy()
    scaf = np.array([MurckoScaffold.MurckoScaffoldSmiles(s) for s in smiles])
    warhead = np.array([warhead_of(s) for s in smiles])
    fp = chem.morgan(smiles)
    heavy = np.array([[Chem.MolFromSmiles(s).GetNumHeavyAtoms()] for s in smiles], dtype=float)

    print(f"## The data: {len(t)} molecules, {len(set(papers))} papers, "
          f"{len(set(scaf))} Murcko scaffolds"
          + (" (with recovered units)" if recover else ""))
    print("   warheads: " + ", ".join(f"{w}={int((warhead == w).sum())}"
                                      for w in sorted(set(warhead))))
    print(f"   log10(kinact/Ki) spans {y.min():.2f} to {y.max():.2f}, SD {y.std(ddof=1):.2f}\n")
    print("   molecules per paper: " + ", ".join(
        f"{p[-5:]}={int((papers == p).sum())}" for p in np.unique(papers)))

    rows = []
    print("\n## Leave-one-paper-out: predicting a series this model has never seen\n")
    print(f"   {'model':<10}{'n':>5}{'RMSE (log)':>13}{'95% CI':>20}"
          f"{'mean within-paper Spearman':>30}")
    for name, X in [("mean", fp), ("n_heavy", heavy), ("ridge", fp), ("rf", fp)]:
        r = evaluate(name, X, y, papers, reps, rng)
        r["split"] = "paper"
        rows.append(r)
        rho = r["mean_within_group_spearman"]
        shown = "n/a (constant)" if np.isnan(rho) else f"{rho:.3f}"
        print(f"   {name:<10}{r['n']:>5}{r['rmse']:>13.3f}"
              f"{f'[{r['rmse_lo']:.3f}, {r['rmse_hi']:.3f}]':>20}{shown:>30}")

    base = next(r for r in rows if r["model"] == "mean")
    best = next(r for r in rows if r["model"] == "rf")
    ok = ~np.isnan(base["_pred"]) & ~np.isnan(best["_pred"])
    eb = (base["_y"][ok] - base["_pred"][ok]) ** 2
    em = (best["_y"][ok] - best["_pred"][ok]) ** 2
    diffs = []
    for _ in range(reps):
        i = rng.integers(0, ok.sum(), ok.sum())
        diffs.append(np.sqrt(em[i].mean()) - np.sqrt(eb[i].mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    pw = float(np.mean(np.array(diffs) >= 0))
    print(f"\n   paired: rf - mean = {np.sqrt(em.mean()) - np.sqrt(eb.mean()):+.3f} log RMSE, "
          f"95% CI [{lo:+.3f}, {hi:+.3f}], P(rf no better) = {pw:.3f}")
    print("   Negative favours the model. The same molecules are resampled on both sides, so this\n"
          "   is the comparison the two overlapping intervals above cannot settle.")
    rows.append({"model": "rf - mean (paired)", "split": "paper",
                 "rmse": float(np.sqrt(em.mean()) - np.sqrt(eb.mean())),
                 "rmse_lo": float(lo), "rmse_hi": float(hi), "p_not_better": pw})

    classes = [w for w in set(warhead) if (warhead == w).sum() >= 20]
    if len(classes) > 1:
        print("\n## Across warhead classes: train on one electrophile, predict the other\n")
        print("   This is the question the recovered data makes askable. A model that transfers\n"
              "   here has learned the recognition element; one that does not has learned a series.\n")
        print(f"   {'held-out class':<20}{'n':>5}{'RMSE (log)':>13}{'baseline':>11}{'Spearman':>11}")
        for w in sorted(classes):
            te = warhead == w
            tr = ~te
            m = MODELS_RF()
            m.fit(fp[tr], y[tr])
            pr = m.predict(fp[te])
            rmse = float(np.sqrt(np.mean((y[te] - pr) ** 2)))
            base = float(np.sqrt(np.mean((y[te] - y[tr].mean()) ** 2)))
            rho = stats.spearmanr(y[te], pr).statistic if np.std(pr) > 0 else np.nan
            print(f"   {w:<20}{int(te.sum()):>5}{rmse:>13.3f}{base:>11.3f}{rho:>11.3f}")
            rows.append({"model": "rf", "split": f"held-out warhead: {w}", "n": int(te.sum()),
                         "rmse": rmse, "rmse_lo": np.nan, "rmse_hi": np.nan,
                         "mean_within_group_spearman": rho, "baseline_rmse": base})
        print("\n   'baseline' predicts the training classes' mean for everything.")

    print("\n## The same models under a scaffold split, for contrast only\n")
    print(f"   {'model':<10}{'n':>5}{'RMSE (log)':>13}{'95% CI':>20}")
    for name, X in [("mean", fp), ("rf", fp)]:
        r = evaluate(name, X, y, scaf, reps, rng)
        r["split"] = "scaffold"
        rows.append(r)
        print(f"   {name:<10}{r['n']:>5}{r['rmse']:>13.3f}"
              f"{f'[{r['rmse_lo']:.3f}, {r['rmse_hi']:.3f}]':>20}")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]).to_csv(
        OUT_RECOVERED if recover else OUT, index=False)
    print(f"\nwrote {OUT_RECOVERED if recover else OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--recover", action="store_true",
                    help="use units recovered from primary publications (src/data/kinetic_units.py)")
    a = ap.parse_args()
    main(a.reps, a.recover)
