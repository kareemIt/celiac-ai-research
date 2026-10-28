"""Approach C: frozen Qwen3.5-9B embeddings + simple trained heads, on the SAME splits/seeds as
run_baselines.py so results pair up seed-by-seed. Layer and regularization are chosen on the valid split.

Needs: scripts/embed_molecules.py (cache) and scripts/run_baselines.py (fingerprint results to compare).
Usage: python scripts/run_qwen_probe.py [tg2|gdsc2|all] [--seeds N]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "4")  # small matrices: 16 BLAS threads thrash instead of helping

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from scripts.run_baselines import GDSC_XGB, per_drug_spearman, summarize
from src.data import chem, gdsc2, tg2
from src.evaluation import metrics
from src.training import embed

C_GRID = [1e-3, 1e-2, 1e-1, 1.0]  # logistic regression inverse regularization
ALPHA_GRID = [1e1, 1e2, 1e3, 1e4, 1e5]  # ridge


def make_head(task, g):
    # n_train (~240) << dim (4096+): an L2-regularized linear solution lies in the span of the training
    # points, so rotating onto ALL train principal components is exact and makes each fit ~25x smaller.
    head = LogisticRegression(C=g, max_iter=5000) if task == "classification" else Ridge(alpha=g)
    return make_pipeline(StandardScaler(), PCA(), head)


def predict(task, m, X):
    return m.predict_proba(X)[:, 1] if task == "classification" else m.predict(X)


def val_loss(task, y, p):
    # Brier / RMSE: defined even when a small valid fold has one class.
    return np.mean((y - p) ** 2)


def paired(new, base, keys, metric, higher_better=True):
    m = new.merge(base, on=keys, suffixes=("", "_base"))
    d = (m[metric] - m[f"{metric}_base"]).dropna()
    wins = (d > 0).sum() if higher_better else (d < 0).sum()
    return f"Δ{metric} = {d.mean():+.3f} ± {d.std():.3f}, better on {wins}/{len(d)} seeds"


def run_tg2(seeds):
    keys, embs = embed.load("data/processed/qwen_tg2.npz")
    d = tg2.load_raw()
    rows = []
    for task, table in [("classification", tg2.classification(d)), ("regression", tg2.regression(d))]:
        smiles = table["smiles"].tolist()
        assert list(keys) == smiles, "embedding cache out of date: rerun scripts/embed_molecules.py"
        y, fp = table["y"].to_numpy(), chem.morgan(smiles).astype(np.float32)
        grid = C_GRID if task == "classification" else ALPHA_GRID
        for seed in range(seeds):
            tr, va, te = chem.scaffold_split(smiles, seed=seed)
            for variant in ["qwen", "qwen+fp"]:
                feats = {l: (e if variant == "qwen" else np.hstack([e, fp])) for l, e in embs.items()}
                if len(va) >= 5:
                    _, layer, g = min((val_loss(task, y[va], predict(task, make_head(task, g).fit(X[tr], y[tr]), X[va])), l, g)
                                      for l, X in feats.items() for g in grid)
                else:  # valid fold too small to select on: fixed middle-of-grid default
                    layer, g = 16, grid[len(grid) // 2]
                p = predict(task, make_head(task, g).fit(feats[layer][tr], y[tr]), feats[layer][te])
                res = metrics.classification(y[te], p) if task == "classification" else metrics.regression(y[te], p)
                rows.append({"task": task, "model": variant, "seed": seed, "layer": layer, "reg": g, **res})
            print(f"  {task} seed {seed} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("results/qwen_probe_tg2.csv", index=False)

    base = pd.read_csv("results/baselines_tg2.csv")
    base = base[(base.model == "rf") & (base.seed < seeds)]
    for task, metric, hb in [("classification", "auroc", True), ("classification", "ece", False),
                             ("regression", "spearman", True), ("regression", "rmse", False)]:
        o = out[out.task == task]
        if metric in ("auroc", "spearman"):
            cols = ["auroc", "auprc", "mcc", "ece", "brier"] if task == "classification" else ["rmse", "r2", "pearson", "spearman"]
            print(f"\n## TG2 {task}: frozen Qwen + linear head ({seeds} seeds)\n"
                  f"{summarize(o[['model', 'seed'] + cols], ['model']).to_string()}")
            print("layers chosen:", o.groupby("model")["layer"].agg(lambda s: s.value_counts().to_dict()).to_dict())
        for v in ["qwen", "qwen+fp"]:
            print(f"  {v} vs fingerprint RF: {paired(o[o.model == v], base[base.task == task], ['seed'], metric, hb)}")


GDSC_PROMPTS = {  # model name -> embedding cache (prompts defined in scripts/embed_molecules.py)
    "qwen_smiles": "qwen_gdsc2_drugs",
    "qwen_name+smiles": "qwen_gdsc2_drugs_name",
    "qwen_name": "qwen_gdsc2_drugs_nameonly",
}


def run_gdsc2(seeds):
    df = gdsc2.load()
    y, drug_ids = df["Y"].to_numpy(), df["Drug_ID"].to_numpy()
    rows = []
    for model_name, cache in GDSC_PROMPTS.items():
        keys, embs = embed.load(f"data/processed/{cache}.npz")
        for seed in range(seeds):
            tr, va, te = gdsc2.split(df, "cold_drug", seed=seed)
            train_drugs = np.isin(keys, np.unique(drug_ids[tr]))
            best = None
            for layer, e in embs.items():
                # 32-dim PCA fit on training drugs only (~110 drugs, so the raw 4096-d vector is far too wide).
                z = PCA(n_components=32, random_state=seed).fit(e[train_drugs]).transform(e)
                X = gdsc2.features(df, tr, dict(zip(keys, z)), seed=seed)
                m = XGBRegressor(**GDSC_XGB, random_state=seed).fit(X[tr], y[tr])
                loss = val_loss("regression", y[va], m.predict(X[va]))
                if best is None or loss < best[0]:
                    best = (loss, layer, m.predict(X[te]))
            _, layer, p = best
            rows.append({"split": "cold_drug", "model": model_name, "seed": seed, "layer": layer,
                         **metrics.regression(y[te], p), "per_drug_spearman": per_drug_spearman(drug_ids[te], y[te], p)})
            print(f"  {model_name} seed {seed}: layer {layer}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("results/qwen_probe_gdsc2.csv", index=False)
    base = pd.read_csv("results/baselines_gdsc2.csv")
    base = base[(base.split == "cold_drug") & (base.model == "xgb") & (base.seed < seeds)]
    both = pd.concat([base.assign(model="fingerprint_xgb"), out])[out.columns.drop("layer")]
    print(f"\n## GDSC2 cold-drug: drug features + expression, XGBoost ({seeds} seeds)\n"
          f"{summarize(both, ['split', 'model']).to_string()}")
    for model_name in GDSC_PROMPTS:
        o = out[out.model == model_name]
        for metric, hb in [("pearson", True), ("rmse", False), ("per_drug_spearman", True)]:
            print(f"  {model_name} vs fingerprint XGB: {paired(o, base, ['seed'], metric, hb)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="all", choices=["tg2", "gdsc2", "all"])
    ap.add_argument("--seeds", type=int, default=None, help="default: 10 for TG2, 5 for GDSC2 (as in baselines)")
    a = ap.parse_args()
    if a.dataset in ("tg2", "all"):
        run_tg2(a.seeds or 10)
    if a.dataset in ("gdsc2", "all"):
        run_gdsc2(a.seeds or 5)
