"""How far does TG2 potency prediction generalize as test molecules get less like the training set?

Three questions, all on the same 10 scaffold splits used everywhere else:
  1. Accuracy vs chemical similarity: bin each test molecule by its maximum Tanimoto to any training
     molecule and score each bin separately.
  2. Strict novelty: rescore on the subset of test molecules that have no training analog at all
     (max Tanimoto below a cutoff), which is the evaluation a scaffold split only approximates.
  3. Does confidence track familiarity? If a model knew what it did not know, its confidence would
     fall as similarity falls.

Writes per-molecule predictions so later analyses (conformal, selective prediction) need no refit.

Usage: python scripts/run_generalization.py [--seeds 10] [--data combined]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from scripts.audit_splits import tanimoto_to_train
from scripts.run_baselines import XGB
from src.data import chem, tg2
from src.evaluation import metrics

PRED_OUT = "results/generalization_preds.csv"
BIN_OUT = "results/generalization_bins.csv"
BINS = [0.0, 0.4, 0.5, 0.6, 0.7, 1.01]
NOVEL_CUTOFF = 0.5
JEV_SCORES = "results/jev_tg2_scores.csv"
QLORA_PREDS = "results/jev_qlora_preds_combined.csv"

# One good setting taken from the tuning grid, held fixed across seeds so these analyses compare
# like with like. This is NOT the per-seed tuned RF of run_tuned_baselines.py (AUROC 0.898); holding
# the config fixed costs about 0.02 AUROC and keeps the uncertainty results interpretable.
RF_CFG = dict(n_estimators=500, max_depth=None, min_samples_leaf=1, max_features=0.1, n_jobs=-1)


def qwen_zeroshot(smiles):
    """Cached zero-shot Jev log-odds as a probability, or None if the cache does not cover the set."""
    if not os.path.exists(JEV_SCORES):
        return None
    sc = pd.read_csv(JEV_SCORES)
    sc = sc[sc["prompt"] == "smiles"]
    piv = sc.pivot(index="smiles", columns="order", values="logodds")
    if not set(smiles) <= set(piv.index):
        return None
    return 1 / (1 + np.exp(-piv.loc[smiles].mean(axis=1).to_numpy()))


def qlora_preds():
    """Per-seed test probabilities from the fine-tuned model, for whatever seeds have finished."""
    if not os.path.exists(QLORA_PREDS):
        return {}
    d = pd.read_csv(QLORA_PREDS)
    d = d[d.fold == "test"]
    return {int(s): dict(zip(g.smiles, 1 / (1 + np.exp(-g.logodds)))) for s, g in d.groupby("seed")}


def main(seeds, data):
    table = tg2.combined("classification") if data == "combined" else tg2.classification(tg2.load_raw())
    smiles, y = table["smiles"].tolist(), table["y"].to_numpy()
    X = chem.morgan(smiles)
    zs = qwen_zeroshot(smiles)
    qlora = qlora_preds()
    if qlora:
        print(f"including fine-tuned Qwen for {len(qlora)} finished seeds: {sorted(qlora)}")

    rows = []
    for seed in range(seeds):
        tr, va, te = chem.scaffold_split(smiles, seed=seed)
        sim = tanimoto_to_train(X[te], X[tr])
        preds = {
            "rf": RandomForestClassifier(random_state=seed, **RF_CFG).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1],
            "xgb": XGBClassifier(**XGB, random_state=seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1],
            "logreg": LogisticRegression(C=1.0, max_iter=5000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1],
        }
        if zs is not None:
            preds["qwen_zeroshot"] = zs[te]
        if seed in qlora:
            preds["qwen_qlora"] = np.array([qlora[seed][smiles[i]] for i in te])
        for name, p in preds.items():
            rows.append(pd.DataFrame({"model": name, "seed": seed, "smiles": [smiles[i] for i in te],
                                      "y": y[te], "p": p, "max_tanimoto": sim}))
        print(f"  seed {seed}: {len(te)} test molecules, similarity {sim.min():.2f}-{sim.max():.2f}", flush=True)

    pred = pd.concat(rows, ignore_index=True)
    pred.to_csv(PRED_OUT, index=False)

    # 1. Pooled over seeds: per-seed bins hold too few molecules to score on their own.
    pred["bin"] = pd.cut(pred.max_tanimoto, BINS, right=False,
                         labels=[f"{a:.1f}-{b:.1f}" for a, b in zip(BINS, BINS[1:])])
    out = []
    for (name, b), g in pred.groupby(["model", "bin"], observed=True):
        if len(g) < 20:
            continue
        out.append({"model": name, "bin": str(b), "n": len(g), "pos_frac": g.y.mean(),
                    "confidence": np.abs(g.p - 0.5).mean() * 2, **metrics.classification(g.y, g.p)})
    bins = pd.DataFrame(out)
    bins.to_csv(BIN_OUT, index=False)
    print("\n## Accuracy by maximum Tanimoto similarity to the training set (pooled over seeds)")
    for name, g in bins.groupby("model"):
        print(f"\n  {name}")
        print(g.drop(columns="model").to_string(index=False,
              formatters={c: "{:.3f}".format for c in ["pos_frac", "confidence", "auroc", "auprc", "mcc", "ece", "brier", "logloss"]}))

    # 2. Strict novelty subset.
    print(f"\n## Strict novelty: test molecules with max Tanimoto < {NOVEL_CUTOFF}")
    nov = pred[pred.max_tanimoto < NOVEL_CUTOFF]
    ref = nov[nov.model == "rf_tuned"]
    print(f"  {len(ref)} of {len(pred[pred.model == 'rf'])} rf test predictions "
          f"({len(ref) / len(pred[pred.model == 'rf']):.0%}), covering {ref.smiles.nunique()} distinct "
          f"molecules, positive rate {ref.y.mean():.3f}")
    print("  Pooled across seeds, so a molecule in several test folds is counted several times and the "
          "errors are correlated: read these as indicative, not as 51 independent observations.")
    strict = []
    for name, g in nov.groupby("model"):
        full = pred[pred.model == name]
        strict.append({"model": name, "n": len(g), "n_mols": g.smiles.nunique(),
                       **{f"novel_{k}": v for k, v in metrics.classification(g.y, g.p).items()},
                       "all_auroc": metrics.classification(full.y, full.p)["auroc"]})
    print(pd.DataFrame(strict).to_string(index=False, float_format="%.3f"))

    # 3. Does confidence fall as molecules get less familiar?
    print("\n## Does the model know when it is out of its depth?")
    print(f"{'model':<16}{'rho(sim, confidence)':>22}{'p':>10}{'rho(sim, correct)':>20}{'p':>10}")
    for name, g in pred.groupby("model"):
        conf = np.abs(g.p - 0.5) * 2
        correct = ((g.p >= 0.5).astype(int) == g.y).astype(int)
        r1, p1 = spearmanr(g.max_tanimoto, conf)
        r2, p2 = spearmanr(g.max_tanimoto, correct)
        print(f"{name:<16}{r1:>22.3f}{p1:>10.4f}{r2:>20.3f}{p2:>10.4f}")
    print("\nA model that knew its own limits would show a positive rho for confidence: less similar, "
          "less confident.")
    print("rho(sim, correct) is confounded: the least-similar molecules are overwhelmingly negatives "
          f"(positive rate {pred[pred.max_tanimoto < 0.5].y.mean():.2f} below 0.5 Tanimoto, "
          f"{pred[pred.max_tanimoto >= 0.7].y.mean():.2f} above 0.7), so predicting 'not potent' is "
          "right most of the time there. Judge the bins by AUROC and MCC, not accuracy.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--data", default="combined", choices=["pubchem", "combined"])
    a = ap.parse_args()
    main(a.seeds, a.data)
