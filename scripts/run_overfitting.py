"""Measure overfitting directly, separated from distribution shift. Tier 1: the fingerprint models.

Design and pre-registration: docs/proposal_overfitting_2026-11-09.md.

For every (model, arm, seed) the sweep already scored, this refits the *same selected
configuration* and produces four numbers on one axis:

  in_sample   the selected model scored on its own training fold. Memorisation capacity, not
              generalisation; for a fully grown random forest it is near-perfect by construction
              and is reported so that its absence cannot be mistaken for a missing control.
  oof         inner K-fold cross-validation inside the training fold, predictions assembled
              out-of-fold. The honest estimate of generalisation *within* the training
              distribution.
  test        the outer held-out score. Must reproduce the committed value, or the refit is not
              the model the paper reports -- this is the gate.
  test_small  the outer test scored by each inner model and averaged. Same held-out molecules as
              `test` but fitted on K-1/K of the training fold, which is the learning-curve control:
              without it, the smaller training set behind `oof` would masquerade as overfitting.

Then:
  in_sample - oof   is overfitting, both sides from the same distribution
  oof - test_small  is distribution shift, with training size held fixed as well as capacity

Inner folds are grouped the same way the outer split is -- scaffold for the scaffold arms,
publication for the publication arms -- because an ungrouped inner CV would leak analogues between
inner folds and report an inflated out-of-fold score, manufacturing the appearance of no
overfitting. No hyperparameter search runs inside the inner loop: re-searching would measure a
different estimator than the one the paper reports.

Nothing is fitted on test, and the outer validation fold is excluded from the inner loop so the
selected configuration stays untainted.

Usage: python scripts/run_overfitting.py [--target tg2|cdk2|maob] [--model rf|logreg|xgb]
Output: results/overfitting_tier1.csv
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold  # noqa: E402

from run_sota import FEATURES, _fmt, expand, load_folds, paths, sklearn_grids  # noqa: E402
from src.data.chem import MurckoScaffold  # noqa: E402
from src.evaluation.metrics import classification  # noqa: E402

RESULTS = ROOT / "results"
MODELS = ("rf", "logreg", "xgb")
TARGETS = ("tg2", "cdk2", "maob")
K = 5  # fixed in advance
TOL = 1e-6  # xgboost's float32 leaf sums moved the original reproduction by 3.8e-8


def tie_band(y, p):
    """How far AUROC can move when last-bit float noise breaks tied pairs the other way.

    sklearn's forest accumulates per-tree probabilities through joblib, so with n_jobs=-1 the
    thread order varies and repeated predict_proba calls on one fitted model differ by about one
    ULP. Positive/negative pairs sitting at exactly equal probability then resolve either way, and
    each flip moves AUROC by 1/(n_pos*n_neg). Measured on tg2/rf/scaffold/seed 15: 25 calls on a
    single fitted model returned 0.8608647 and 0.8619734, one pair of that fold's 902 apart.

    This returns the worst case, every tied pair flipping, so the reproduction gate can allow
    exactly that much and no more rather than a tolerance picked to fit.
    """
    y, p = np.asarray(y), np.asarray(p)
    n_pos, n_neg = int(np.sum(y == 1)), int(np.sum(y == 0))
    if not n_pos or not n_neg:
        return np.nan  # AUROC is undefined, so there is nothing to reproduce
    pos = pd.Series(p[y == 1]).value_counts()
    neg = pd.Series(p[y == 0]).value_counts()
    shared = pos.index.intersection(neg.index)
    return float((pos[shared] * neg[shared]).sum()) / (n_pos * n_neg)


def config_for(params, stored):
    """The grid entry whose printed form is the stored string.

    Looked up rather than parsed: run_sota._fmt drops n_jobs, max_iter and tree_method when it
    prints, so parsing the string back would silently lose them. Enumerating the grid and matching
    on the same formatter cannot.
    """
    for cfg in expand(params):
        if _fmt(cfg) == stored:
            return cfg
    raise SystemExit(f"no configuration in the grid prints as {stored!r}")


def inner_groups(fold, split, tr):
    """Group labels for the inner folds, matching how the outer split was grouped."""
    smiles = [fold["smiles"][i] for i in tr]
    if split.startswith("paper"):
        if fold["papers"] is None:
            return None
        return [fold["papers"][i] for i in tr]
    if split.startswith("scaffold"):
        return [MurckoScaffold.MurckoScaffoldSmiles(s) for s in smiles]
    return None  # random arm: no grouping to honour


def fit(cls, cfg, seed, X, y):
    m = cls(**cfg) if cls is LogisticRegression else cls(random_state=seed, **cfg)
    m.fit(X, y)
    return m


def auroc(y, p):
    return classification(np.asarray(y), np.asarray(p))["auroc"]


def one_fold(model, fold, split, seed, stored_config):
    cls, feat, params = sklearn_grids()[model]
    cfg = config_for(params, stored_config)
    X = FEATURES[feat](fold["smiles"])
    y = fold["y"]
    tr, _va, te = fold["idx"]

    outer = fit(cls, cfg, seed, X[tr], y[tr])
    p_test = outer.predict_proba(X[te])[:, 1]
    row = {"model": model, "split": split, "seed": seed, "config": stored_config,
           "n_train": len(tr), "n_test": len(te),
           "in_sample": auroc(y[tr], outer.predict_proba(X[tr])[:, 1]),
           "test": auroc(y[te], p_test), "tie_band": tie_band(y[te], p_test)}

    groups = inner_groups(fold, split, tr)
    n_groups = len(set(groups)) if groups is not None else len(tr)
    k = min(K, n_groups)
    if k < 2:
        row.update({"oof": np.nan, "test_small": np.nan, "inner_folds": 0, "inner_groups": n_groups})
        return row
    splitter = (GroupKFold(n_splits=k).split(np.asarray(tr), y[tr], groups)
                if groups is not None else KFold(n_splits=k, shuffle=True,
                                                 random_state=seed).split(np.asarray(tr)))
    oof = np.full(len(tr), np.nan)
    small_scores, sizes = [], []
    for inner_tr, inner_te in splitter:
        m = fit(cls, cfg, seed, X[tr][inner_tr], y[tr][inner_tr])
        oof[inner_te] = m.predict_proba(X[tr][inner_te])[:, 1]
        small_scores.append(auroc(y[te], m.predict_proba(X[te])[:, 1]))
        sizes.append(len(inner_tr))
    row.update({"oof": auroc(y[tr], oof), "test_small": float(np.nanmean(small_scores)),
                "inner_folds": k, "inner_groups": n_groups,
                "n_train_inner": int(np.mean(sizes))})
    return row


def run(target, models):
    splits_path, metrics_path, _ = paths(target)
    committed = pd.read_csv(RESULTS.parent / metrics_path)
    want = committed[committed.model.isin(models)]
    rows, t0 = [], time.perf_counter()
    for split in sorted(want.split.unique()):
        by_seed = {int(s): g for s, g in want[want.split == split].groupby("seed")}
        for fold in load_folds(split, None, str(RESULTS.parent / splits_path),
                               only=sorted(by_seed)):
            for _, ref in by_seed[fold["seed"]].iterrows():
                row = one_fold(ref.model, fold, split, fold["seed"], ref.config)
                row["target"] = target
                row["test_committed"] = ref.auroc
                rows.append(row)
        print(f"  {target} {split}: {len(rows)} rows, {time.perf_counter() - t0:.0f}s elapsed",
              flush=True)
    return pd.DataFrame(rows)


def gate(rows):
    """The refit must reproduce the committed test AUROC, or it is not the published model.

    Bit-exact equality is not available to us: the committed value was itself produced under the
    tie-breaking noise `tie_band` measures, so a fold passes if its difference is zero or no larger
    than its own tied pairs can account for. Anything above that is a genuine failure to reproduce.
    Returns the worst difference, the number of folds checked, and how many needed the allowance --
    that last count is reported rather than hidden, because a gate that quietly relaxes is no gate.
    """
    d = rows.dropna(subset=["test", "test_committed"]).copy()
    d["diff"] = (d.test - d.test_committed).abs()
    allowed = np.maximum(TOL, d.get("tie_band", pd.Series(0.0, index=d.index)).fillna(0.0))
    bad = d.loc[d["diff"] > allowed,
                ["target", "model", "split", "seed", "test", "test_committed", "diff", "tie_band"]]
    if len(bad):
        raise SystemExit(f"refit does not reproduce committed test AUROC on {len(bad)} of "
                         f"{len(d)} folds, beyond what tied pairs explain:\n{bad.head(10)}")
    return d["diff"].max(), len(d), int((d["diff"] > TOL).sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="append", choices=TARGETS, default=None)
    ap.add_argument("--model", action="append", choices=MODELS, default=None)
    args = ap.parse_args()
    models = args.model or list(MODELS)
    targets = args.target or list(TARGETS)
    # A restricted run must not clobber a complete one.
    out = RESULTS / ("overfitting_tier1.csv" if (len(targets), len(models)) == (3, 3)
                     else "overfitting_tier1_partial.csv")
    done = []
    for t in targets:
        done.append(run(t, models))
        rows = pd.concat(done, ignore_index=True)
        rows["overfit_gap"] = rows.in_sample - rows.oof
        rows["shift_gap"] = rows.oof - rows.test_small
        rows["size_effect"] = rows.test - rows.test_small
        # Saved before the gate can abort: hours of fitting must not depend on a check passing.
        rows.to_csv(out, index=False)
        print(f"  saved {len(rows)} rows to {out.name} after {t}", flush=True)
    worst, n, relaxed = gate(rows)
    print(f"\nreproduction gate: max |test - committed| = {worst:.2e} over {n} folds "
          f"({relaxed} within tied-pair allowance, {n - relaxed} bit-exact)")
    agg = rows.groupby(["target", "split", "model"])[
        ["in_sample", "oof", "test", "test_small", "overfit_gap", "shift_gap", "size_effect"]].mean()
    print(agg.round(3).to_string())
