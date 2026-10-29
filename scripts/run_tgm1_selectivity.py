"""TG2 against TGM1 within one laboratory's series: is selectivity predictable when provenance
is held constant?

The earlier selectivity work failed for a specific reason: AUROC 0.974 under a scaffold split
collapsed to 0.574 when whole publications were held out, because the classes were different
series from different labs (docs/selectivity.md). Recovering the Klock 2014 units gives 47
molecules with TG2 *and* TGM1 kinact/Ki from one laboratory, one assay, one measure -- the
within-series comparison that failure said was the only honest way to ask the question.

IMPORTANT, and stated here so it cannot be lost downstream: **TGM1 is not the safety-relevant
antitarget.** Factor XIII-A crosslinks fibrin and its inhibition causes bleeding, which is the risk
that matters for a drug taken chronically by otherwise-healthy people. TGM1 is epidermal. Only 10
molecules carry paired TG2/FXIII-A kinetic data, which is too few to model. Nothing in this script
is evidence about FXIII-A safety.

Selectivity here is a continuous log ratio, not a class: log10(kinact/Ki for TG2) minus the same
for TGM1. That avoids inventing a threshold, and a ratio of two measurements from one lab on one
day cancels much of the assay offset that defeated the cross-paper analysis.

Writes results/tgm1_selectivity.csv. CPU, seconds.

Usage: python scripts/run_tgm1_selectivity.py [--reps 2000]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

from scripts.run_covalent import TO_PER_M_S, warhead_of
from src.data import chem, chembl, kinetic_units

OUT = "results/tgm1_selectivity.csv"
RAW = "data/raw/chembl/{name}_activities.csv"


def kinetic(name):
    r = kinetic_units.apply(chembl.load_raw(RAW.format(name=name)))
    d = r[(r.standard_type == "Ratio") & (r.standard_relation == "=")
          & r.standard_units.isin(TO_PER_M_S) & r.standard_value.notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["smiles", "logk", "paper"])
    d["logk"] = np.log10(d.standard_value.astype(float) * d.standard_units.map(TO_PER_M_S))
    return d.groupby("smiles").agg(logk=("logk", "median"),
                                   paper=("document_chembl_id", "first")).reset_index()


def main(reps):
    rng = np.random.default_rng(0)
    tg2, tgm1 = kinetic("tgm2"), kinetic("tgm1")
    m = tg2.merge(tgm1, on="smiles", suffixes=("_tg2", "_tgm1"))
    m["selectivity"] = m.logk_tg2 - m.logk_tgm1
    print(f"## {len(m)} molecules with paired TG2 and TGM1 kinact/Ki")
    print(f"   source publications: {sorted(m.paper_tg2.unique())}")
    print(f"   warheads: {pd.Series([warhead_of(s) for s in m.smiles]).value_counts().to_dict()}\n")

    s = m.selectivity
    print("## 1. How selective are these compounds, as measured?\n")
    print(f"   log10(TG2/TGM1) ranges {s.min():+.2f} to {s.max():+.2f}, median {s.median():+.2f}")
    print(f"   TG2-preferring (>0): {int((s > 0).sum())} of {len(s)};  "
          f">10-fold selective: {int((s > 1).sum())};  >10-fold AGAINST TG2: {int((s < -1).sum())}")
    r, p = stats.pearsonr(m.logk_tg2, m.logk_tgm1)
    print(f"\n   TG2 and TGM1 potency correlate at r={r:.3f} (p={p:.3g}).")
    if abs(r) >= 0.7:
        print("   That is high: potency and selectivity are nearly the same axis here, so a potent\n"
              "   compound is mostly just a potent compound on both enzymes.")
    else:
        print("   That is moderate, so selectivity is not simply a restatement of TG2 potency --\n"
              "   there is in principle a separate signal to learn. Section 2 asks whether we can.")

    print("\n## 2. Can the selectivity ratio be predicted from structure?\n")
    smiles = m.smiles.tolist()
    fp = chem.morgan(smiles)
    scaf = np.array([MurckoScaffold.MurckoScaffoldSmiles(x) for x in smiles])
    y = s.to_numpy()

    # Leave-one-scaffold-out: the only split available, since all 47 come from one paper.
    pred = np.full(len(y), np.nan)
    for g in np.unique(scaf):
        te = scaf == g
        if (~te).sum() < 10:
            continue
        pred[te] = RandomForestRegressor(n_estimators=500, random_state=0,
                                         n_jobs=-1).fit(fp[~te], y[~te]).predict(fp[te])
    ok = ~np.isnan(pred)
    rmse = float(np.sqrt(np.mean((y[ok] - pred[ok]) ** 2)))
    base = float(np.sqrt(np.mean((y[ok] - y[ok].mean()) ** 2)))
    rho = stats.spearmanr(y[ok], pred[ok]).statistic
    b = [np.sqrt(np.mean((y[ok][i] - pred[ok][i]) ** 2))
         for i in (rng.integers(0, ok.sum(), ok.sum()) for _ in range(reps))]
    print(f"   {'model':<28}{'n':>5}{'RMSE (log ratio)':>19}{'95% CI':>20}{'Spearman':>11}")
    print(f"   {'predict the mean':<28}{int(ok.sum()):>5}{base:>19.3f}{'':>20}{'n/a':>11}")
    print(f"   {'random forest':<28}{int(ok.sum()):>5}{rmse:>19.3f}"
          f"{f'[{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]':>20}{rho:>11.3f}")

    rows = [{"n_pairs": len(m), "n_scored": int(ok.sum()), "median_selectivity": float(s.median()),
             "frac_tg2_preferring": float((s > 0).mean()), "n_10x_selective": int((s > 1).sum()),
             "tg2_tgm1_pearson_r": float(r), "tg2_tgm1_p": float(p),
             "rmse": rmse, "rmse_baseline": base, "spearman": float(rho),
             "rmse_lo": float(np.percentile(b, 2.5)), "rmse_hi": float(np.percentile(b, 97.5))}]

    print("\n## 3. Caveats that travel with this number\n")
    print("   - One laboratory, one paper, one warhead class. No cross-series claim is possible.")
    print("   - Leave-one-scaffold-out within a single series is a weak test: analogs remain.")
    print("   - TGM1 IS NOT FXIII-A. This says nothing about bleeding risk, which is the safety")
    print("     question for a chronic celiac therapy. Only 10 molecules carry paired FXIII-A data.")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    main(ap.parse_args().reps)
