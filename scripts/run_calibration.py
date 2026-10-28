"""Does post-hoc calibration fix TG2 potency probabilities? Fingerprint models and, when its cached
scores exist, the zero-shot Qwen Jev readout.

Every calibrator is fit on the validation fold of the same scaffold split and applied to test, so no
test label is ever seen during fitting. The uncalibrated model is fit on train only; the calibrated
ones reuse that same fit, so the comparison isolates the calibration map.

Usage: python scripts/run_calibration.py [--seeds 10] [--data combined]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from scripts.run_baselines import XGB, summarize
from src.data import chem, tg2
from src.evaluation import calibration, metrics

OUT = "results/calibration_tg2.csv"
CHANGE_METRICS = ["ece", "brier", "logloss"]
JEV_SCORES = "results/jev_tg2_scores.csv"


def fingerprint_scores(table, tr, va, te, seed):
    """Validation and test logits for each fingerprint model, trained on the train fold alone."""
    X, y = chem.morgan(table["smiles"].tolist()), table["y"].to_numpy()
    models = {
        "logreg": LogisticRegression(C=1.0, max_iter=5000),
        "rf": RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed),
        "xgb": XGBClassifier(**XGB, random_state=seed),
    }
    for name, m in models.items():
        m.fit(X[tr], y[tr])
        yield name, calibration.logit(m.predict_proba(X[va])[:, 1]), calibration.logit(m.predict_proba(X[te])[:, 1])


def jev_scores(table, tr, va, te):
    """Cached zero-shot Qwen log-odds, already the natural logit of P(potent). Needs no training,
    so its 'valid' fold is used only to fit the calibrator."""
    if not os.path.exists(JEV_SCORES):
        return
    sc = pd.read_csv(JEV_SCORES)
    have = set(sc["smiles"])
    if not set(table["smiles"]) <= have:
        print(f"  (skipping Qwen: cached scores cover {len(have & set(table['smiles']))}"
              f"/{len(table)} molecules — rerun run_jev_probe.py --rescore)")
        return
    for kind, g in sc.groupby("prompt", sort=False):
        x = g.pivot(index="smiles", columns="order", values="logodds").loc[table["smiles"]].mean(axis=1).to_numpy()
        yield f"qwen_jev_{kind}", x[va], x[te]


def main(seeds, data, split):
    table = tg2.combined("classification") if data == "combined" else tg2.classification(tg2.load_raw())
    y, smiles = table["y"].to_numpy(), table["smiles"].tolist()
    rows = []
    for seed in range(seeds):
        # random is the control for the scaffold split: valid and test are then drawn from the same
        # chemical space, which is the assumption post-hoc calibration transfer rests on.
        tr, va, te = (chem.scaffold_split(smiles, seed=seed) if split == "scaffold"
                      else chem.group_split(np.arange(len(smiles)), seed=seed))
        sources = list(fingerprint_scores(table, tr, va, te, seed)) + list(jev_scores(table, tr, va, te))
        for name, z_va, z_te in sources:
            temp = calibration.fit_temperature(z_va, y[va])
            for kind in calibration.KINDS:
                p = calibration.fit(kind, z_va, y[va])(z_te)
                rows.append({"model": name, "calibration": kind, "seed": seed, "split": split,
                             "temperature": temp, "n_valid": len(va), "n_test": len(te),
                             **metrics.classification(y[te], p)})
        print(f"  seed {seed}: {len(va)} valid / {len(te)} test molecules", flush=True)

    out = pd.DataFrame(rows)
    path = OUT.replace(".csv", f"_{split}.csv")
    out.to_csv(path, index=False)
    print(f"\n## TG2 potency calibration ({data}, {split} split, {seeds} seeds, mean ± sd)\n"
          f"{summarize(out.drop(columns=['n_valid', 'n_test', 'split']), ['model', 'calibration']).to_string()}")
    print("\nFitted temperature per model (1.0 = already calibrated, >1 = overconfident):")
    print(out[out.calibration == 'none'].groupby('model')['temperature']
          .agg(['mean', 'std', 'min', 'max']).round(2).to_string())

    print("\n## Change from uncalibrated, paired by seed (negative = better for ECE/Brier/log loss)")
    print(f"{'model':<22}{'calibration':<13}" + "".join(f"{m:>22}" for m in CHANGE_METRICS))
    for name, g in out.groupby("model", sort=False):
        base = g[g.calibration == "none"].set_index("seed")
        for kind in calibration.KINDS[1:]:
            m = g[g.calibration == kind].set_index("seed")
            parts = []
            for metric in CHANGE_METRICS:
                d = (m[metric] - base[metric]).dropna()
                p_val = ttest_rel(d, np.zeros(len(d))).pvalue if d.std() > 0 else 1.0
                star = "*" if p_val < 0.05 else " "
                parts.append(f"{d.mean():+.3f} (p={p_val:.3f}){star}".rjust(22))
            print(f"  {name:<20}{kind:<13}" + "".join(parts))
    print("\n* = p < 0.05, paired t-test over seeds, uncorrected. AUROC is unchanged by construction "
          "for monotone maps.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--data", default="combined", choices=["pubchem", "combined"])
    ap.add_argument("--split", default="scaffold", choices=["scaffold", "random"])
    a = ap.parse_args()
    main(a.seeds, a.data, a.split)
