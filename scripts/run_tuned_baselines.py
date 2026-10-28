"""Hyperparameter-tuned fingerprint baselines for TG2 potency, so the LLM is compared against
traditional ML at its best rather than at its defaults.

Each seed tunes on that seed's own validation fold and reports test once. The grid is small and
fixed in this file; the point is a fair baseline, not an exhaustive search.

Usage: python scripts/run_tuned_baselines.py [--seeds 10] [--data combined]
"""
import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from scripts.run_baselines import summarize
from src.data import chem, tg2
from src.evaluation import metrics

OUT = "results/tuned_baselines_tg2.csv"

GRIDS = {
    "logreg": (LogisticRegression, {"C": [0.01, 0.1, 1.0, 10.0], "max_iter": [5000]}),
    "rf": (RandomForestClassifier, {"n_estimators": [500], "max_depth": [None, 8], "n_jobs": [-1],
                                    "min_samples_leaf": [1, 2, 5], "max_features": ["sqrt", 0.1]}),
    "xgb": (XGBClassifier, {"n_estimators": [300, 800], "learning_rate": [0.05, 0.1],
                            "max_depth": [3, 6], "subsample": [0.8], "colsample_bytree": [0.5],
                            "tree_method": ["hist"]}),
}


def grid(params):
    keys = list(params)
    return [dict(zip(keys, v)) for v in itertools.product(*(params[k] for k in keys))]


def tune(cls, params, X, y, tr, va, seed):
    """Pick the setting with the best validation AUROC. Ties go to the first, so the grid is ordered
    simplest-first and tuning cannot quietly prefer a more complex model for no gain."""
    best = (-np.inf, None, None)
    for cfg in grid(params):
        m = cls(random_state=seed, **cfg) if cls is not LogisticRegression else cls(**cfg)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        auc = roc_auc_score(y[va], p) if len(np.unique(y[va])) == 2 else 0.5
        if auc > best[0]:
            best = (auc, cfg, m)
    return best


def main(seeds, data):
    table = tg2.combined("classification") if data == "combined" else tg2.classification(tg2.load_raw())
    smiles, y = table["smiles"].tolist(), table["y"].to_numpy()
    X = chem.morgan(smiles)

    rows, chosen = [], []
    for seed in range(seeds):
        t0 = time.perf_counter()
        tr, va, te = chem.scaffold_split(smiles, seed=seed)
        for name, (cls, params) in GRIDS.items():
            val_auc, cfg, m = tune(cls, params, X, y, tr, va, seed)
            rows.append({"model": f"{name}_tuned", "seed": seed, "valid_auroc": val_auc,
                         **metrics.classification(y[te], m.predict_proba(X[te])[:, 1])})
            chosen.append({"model": name, "seed": seed, **{k: str(v) for k, v in cfg.items() if k not in
                                                           ("n_jobs", "max_iter", "tree_method")}})
        print(f"  seed {seed}: {time.perf_counter() - t0:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\n## TG2 potency, tuned on each seed's validation fold ({data}, {seeds} seeds)\n"
          f"{summarize(out.drop(columns='valid_auroc'), ['model']).to_string()}")

    picks = pd.DataFrame(chosen)
    picks.to_csv(OUT.replace("tuned_baselines", "tuned_hparams"), index=False)
    print("\n## Hyperparameters chosen (how often)")
    for name, g in picks.groupby("model"):
        # each model has its own grid keys, so the others come back as NaN for its rows
        cols = [c for c in g.columns if c not in ("model", "seed") and g[c].notna().any()]
        combo = g[cols].astype(str).agg(", ".join, axis=1).value_counts()
        print(f"  {name}: " + " | ".join(f"{k} x{v}" for k, v in combo.items()))

    base_path = f"results/baselines_tg2{'' if data == 'pubchem' else '_' + data}.csv"
    if os.path.exists(base_path):
        base = pd.read_csv(base_path)
        base = base[(base.task == "classification") & (base.seed < seeds)]
        print("\n## Tuned vs default, paired by seed")
        for name in GRIDS:
            a = out[out.model == f"{name}_tuned"].set_index("seed")
            b = base[base.model == name].set_index("seed")
            if b.empty:
                continue
            parts = []
            for metric in ["auroc", "auprc", "brier", "ece"]:
                d = (a[metric] - b[metric]).dropna()
                p = ttest_rel(d, np.zeros(len(d))).pvalue if d.std() > 0 else 1.0
                parts.append(f"Δ{metric} {d.mean():+.3f} (p={p:.2f})")
            print(f"  {name:8s} " + "  ".join(parts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--data", default="combined", choices=["pubchem", "combined"])
    a = ap.parse_args()
    main(a.seeds, a.data)
