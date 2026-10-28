"""Knowing when not to trust a TG2 prediction: ensembles, conformal sets, selective prediction.

Post-hoc rescaling failed under scaffold shift (docs/calibration.md). These are the alternatives that
do not assume the validation fold looks like the test fold:

  ensemble      mean probability over model families; their disagreement is a shift-aware signal
  conformal     inductive conformal prediction calibrated on the valid fold, giving label SETS with a
                coverage target. Exchangeability is violated by a scaffold split on purpose, so the
                measured coverage is the experiment, not an assumption.
  selective     abstain on the least trustworthy molecules and measure error against coverage, for
                three different notions of "least trustworthy"

Usage: python scripts/run_uncertainty.py [--seeds 10] [--data combined]
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

from scripts.audit_splits import tanimoto_to_train
from scripts.run_baselines import XGB
from scripts.run_generalization import RF_CFG
from src.data import chem, tg2
from src.evaluation import metrics

CONF_OUT = "results/uncertainty_conformal.csv"
SEL_OUT = "results/uncertainty_selective.csv"
ALPHAS = (0.1, 0.2)
COVERAGES = np.arange(0.1, 1.01, 0.1)


def members(seed):
    return {"rf": RandomForestClassifier(random_state=seed, **RF_CFG),
            "xgb": XGBClassifier(**XGB, random_state=seed),
            "logreg": LogisticRegression(C=1.0, max_iter=5000)}


def conformal(p_va, y_va, p_te, alpha, mondrian):
    """Inductive conformal prediction sets for binary labels.

    Nonconformity of a labelled example is 1 - p(its own label). The calibration quantile is taken at
    ceil((n+1)(1-alpha))/n, the finite-sample correction that makes coverage valid under
    exchangeability. `mondrian` takes the quantile separately within each class, which protects the
    minority class when the base rate shifts between folds.
    """
    s_va = np.where(y_va == 1, 1 - p_va, p_va)
    s_te = np.stack([1 - (1 - p_te), 1 - p_te])  # nonconformity of label 0, label 1

    def q_of(scores):
        n = len(scores)
        if n == 0:
            return 1.0
        k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
        return np.sort(scores)[k - 1]

    qs = [q_of(s_va[y_va == k]) for k in (0, 1)] if mondrian else [q_of(s_va)] * 2
    return np.stack([s_te[k] <= qs[k] for k in (0, 1)])  # [2, n_test] membership


def main(seeds, data):
    table = tg2.combined("classification") if data == "combined" else tg2.classification(tg2.load_raw())
    smiles, y = table["smiles"].tolist(), table["y"].to_numpy()
    X = chem.morgan(smiles)

    conf_rows, sel_rows, ens_rows = [], [], []
    for split_kind in ("scaffold", "random"):
        for seed in range(seeds):
            tr, va, te = (chem.scaffold_split(smiles, seed=seed) if split_kind == "scaffold"
                          else chem.group_split(np.arange(len(smiles)), seed=seed))
            P_va, P_te = {}, {}
            for name, m in members(seed).items():
                m.fit(X[tr], y[tr])
                P_va[name], P_te[name] = m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1]
            ens_va = np.mean(list(P_va.values()), axis=0)
            ens_te = np.mean(list(P_te.values()), axis=0)
            spread = np.std(list(P_te.values()), axis=0)
            sim = tanimoto_to_train(X[te], X[tr])

            ens_rows.append({"split": split_kind, "seed": seed, "model": "ensemble",
                             **metrics.classification(y[te], ens_te)})
            for name, p in P_te.items():
                ens_rows.append({"split": split_kind, "seed": seed, "model": name,
                                 **metrics.classification(y[te], p)})

            for alpha in ALPHAS:
                for mondrian in (False, True):
                    sets = conformal(ens_va, y[va], ens_te, alpha, mondrian)
                    size = sets.sum(0)
                    covered = sets[y[te], np.arange(len(te))]
                    conf_rows.append({"split": split_kind, "seed": seed, "alpha": alpha,
                                      "mondrian": mondrian, "target_coverage": 1 - alpha,
                                      "coverage": covered.mean(), "mean_set_size": size.mean(),
                                      "singleton": (size == 1).mean(), "empty": (size == 0).mean(),
                                      "both": (size == 2).mean()})

            # Selective prediction: keep the most trustworthy fraction, measure error on what is kept.
            criteria = {"confidence": np.abs(ens_te - 0.5), "similarity": sim,
                        "agreement": -spread, "random": np.random.default_rng(seed).random(len(te))}
            err = (ens_te >= 0.5).astype(int) != y[te]
            for crit, score in criteria.items():
                order = np.argsort(-score)  # most trustworthy first
                for c in COVERAGES:
                    k = max(int(round(c * len(te))), 5)
                    keep = order[:k]
                    sel_rows.append({"split": split_kind, "seed": seed, "criterion": crit,
                                     "coverage": c, "n_kept": k, "error": err[keep].mean(),
                                     "brier": np.mean((ens_te[keep] - y[te][keep]) ** 2)})
        print(f"  {split_kind}: {seeds} seeds done", flush=True)

    conf = pd.DataFrame(conf_rows)
    sel = pd.DataFrame(sel_rows)
    conf.to_csv(CONF_OUT, index=False)
    sel.to_csv(SEL_OUT, index=False)
    ens = pd.DataFrame(ens_rows)
    ens.to_csv("results/uncertainty_ensemble.csv", index=False)

    print("\n## Ensemble vs its members (scaffold split, mean ± sd over seeds)")
    e = ens[ens.split == "scaffold"].drop(columns=["split", "seed"]).groupby("model").agg(["mean", "std"])
    for m in ["auroc", "auprc", "mcc", "ece", "brier", "logloss"]:
        print(f"  {m:8s} " + "   ".join(f"{k}: {e.loc[k, (m, 'mean')]:.3f}±{e.loc[k, (m, 'std')]:.3f}"
                                        for k in ["rf", "xgb", "logreg", "ensemble"]))

    print("\n## Conformal prediction: does the coverage guarantee hold under scaffold shift?")
    g = conf.groupby(["split", "alpha", "mondrian"])[["target_coverage", "coverage", "mean_set_size",
                                                      "singleton", "empty"]].mean()
    print(g.round(3).to_string())

    print("\n## Ensemble vs its best single member, paired by seed (scaffold split)")
    base = ens[(ens.split == "scaffold") & (ens.model == "ensemble")].set_index("seed")
    for k in ["rf", "xgb", "logreg"]:
        other = ens[(ens.split == "scaffold") & (ens.model == k)].set_index("seed")
        parts = []
        for m, higher_better in [("auroc", True), ("brier", False), ("logloss", False), ("ece", False)]:
            d = (base[m] - other[m]).dropna()
            pv = ttest_rel(d, np.zeros(len(d))).pvalue if d.std() > 0 else 1.0
            wins = int((d > 0).sum()) if higher_better else int((d < 0).sum())
            parts.append(f"Δ{m} {d.mean():+.3f} (p={pv:.3f}, {wins}/{len(d)})")
        print(f"  vs {k:8s} " + "  ".join(parts))

    print("\n## Selective prediction: error rate when abstaining (scaffold split)")
    s = sel[sel.split == "scaffold"].groupby(["criterion", "coverage"])["error"].mean().unstack(0)
    print(s.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--data", default="combined", choices=["pubchem", "combined"])
    a = ap.parse_args()
    main(a.seeds, a.data)
