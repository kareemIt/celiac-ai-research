"""Fingerprint baselines for TG2 and GDSC2. Fixed default hyperparameters (no tuning):
models train on the train split only, valid is reserved for later tuning, test is reported.

Usage: python scripts/run_baselines.py [tg2|gdsc2|all] [--seeds N]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from src.data import chem, gdsc2, tg2
from src.evaluation import metrics

XGB = dict(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.5, tree_method="hist")
GDSC_XGB = {**XGB, "n_estimators": 1000, "max_depth": 8, "device": "cuda"}


def summarize(df, keys):
    agg = df.drop(columns="seed").groupby(keys, sort=False).agg(["mean", "std"])
    out = pd.DataFrame(index=agg.index)
    for m in agg.columns.levels[0]:
        out[m] = agg[(m, "mean")].map("{:.3f}".format) + " ± " + agg[(m, "std")].map("{:.3f}".format)
    return out


def run_tg2(seeds, data="pubchem"):
    d = tg2.load_raw()
    cls_models = {
        "prior": lambda s: DummyClassifier(strategy="prior"),
        "logreg": lambda s: LogisticRegression(C=1.0, max_iter=5000),
        "rf": lambda s: RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=s),
        "xgb": lambda s: XGBClassifier(**XGB, random_state=s),
    }
    reg_models = {
        "mean": lambda s: DummyRegressor(),
        "rf": lambda s: RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=s),
        "xgb": lambda s: XGBRegressor(**XGB, random_state=s),
    }
    def table_for(task):
        if data == "combined":
            return tg2.combined(task)
        return tg2.classification(d) if task == "classification" else tg2.regression(d)

    rows = []
    for task, table, models in [("classification", table_for("classification"), cls_models),
                                ("regression", table_for("regression"), reg_models)]:
        X, y = chem.morgan(table["smiles"].tolist()), table["y"].to_numpy()
        for seed in range(seeds):
            tr, _, te = chem.scaffold_split(table["smiles"].tolist(), seed=seed)
            for name, make in models.items():
                m = make(seed).fit(X[tr], y[tr])
                if task == "classification":
                    res = metrics.classification(y[te], m.predict_proba(X[te])[:, 1])
                else:
                    res = metrics.regression(y[te], m.predict(X[te]))
                rows.append({"task": task, "model": name, "seed": seed, "n_test": len(te),
                             "test_pos_frac": y[te].mean() if task == "classification" else np.nan, **res})
    out = pd.DataFrame(rows)
    suffix = "" if data == "pubchem" else f"_{data}"
    out.to_csv(f"results/baselines_tg2{suffix}.csv", index=False)
    for task in ["classification", "regression"]:
        t = out[out.task == task].dropna(axis=1, how="all").drop(columns="task")
        print(f"\n## TG2 {task} ({data}, scaffold split, {seeds} seeds, mean ± sd)\n"
              f"{summarize(t, ['model']).to_string()}")


def per_drug_spearman(drug_ids, y, pred):
    """Mean within-drug Spearman: does the model rank cell lines correctly for a given drug?"""
    df = pd.DataFrame({"d": drug_ids, "y": y, "p": pred})
    rs = [spearmanr(g.y, g.p)[0] for _, g in df.groupby("d") if len(g) >= 10 and np.ptp(g.p) > 0]
    return np.mean(rs) if rs else np.nan


def run_gdsc2(seeds):
    df = gdsc2.load()
    y = df["Y"].to_numpy()
    drug_ids = df["Drug_ID"].to_numpy()
    drugs = df.drop_duplicates("Drug_ID")
    fp = dict(zip(drugs["Drug_ID"], chem.morgan(drugs["Drug"].tolist())))

    rows = []
    for kind in ["random", "cold_drug", "cold_cell"]:
        for seed in range(seeds):
            t0 = time.perf_counter()
            tr, _, te = gdsc2.split(df, kind, seed=seed)
            X = gdsc2.features(df, tr, fp, seed=seed)

            drug_mean = pd.Series(y[tr]).groupby(drug_ids[tr]).mean()
            preds = {
                "drug_mean": pd.Series(drug_ids[te]).map(drug_mean).fillna(y[tr].mean()).to_numpy(),
                "xgb": XGBRegressor(**GDSC_XGB, random_state=seed).fit(X[tr], y[tr]).predict(X[te]),
            }
            for name, p in preds.items():
                rows.append({"split": kind, "model": name, "seed": seed, "n_test": len(te),
                             **metrics.regression(y[te], p), "per_drug_spearman": per_drug_spearman(drug_ids[te], y[te], p)})
            print(f"  {kind} seed {seed}: {time.perf_counter() - t0:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("results/baselines_gdsc2.csv", index=False)
    print(f"\n## GDSC2 regression ({seeds} seeds, mean ± sd)\n{summarize(out, ['split', 'model']).to_string()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="all", choices=["tg2", "gdsc2", "all"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--data", default="pubchem", choices=["pubchem", "combined"],
                    help="TG2 source: the PubChem export alone, or its union with ChEMBL")
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    if a.dataset in ("tg2", "all"):
        run_tg2(a.seeds, a.data)
    if a.dataset in ("gdsc2", "all"):
        run_gdsc2(a.seeds)
