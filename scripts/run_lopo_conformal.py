"""Can conformal prediction flag unreliable predictions when a whole publication is held out?

Phase 2. docs/uncertainty.md showed conformal prediction holds coverage under a SCAFFOLD split
(0.922 against a 0.90 target). The publication-held-out arm is a far larger shift -- the random
forest drops 0.854 to 0.612 AUROC there -- and it is the regime a screening campaign actually
operates in. Coverage under a scaffold split says nothing about it.

The experiment hinges on one choice. Split conformal guarantees coverage when the calibration and
test data are exchangeable, and under leave-one-publication-out they are not: the test fold is an
unseen publication. So two calibration regimes are run and compared directly:

  iid    calibration molecules drawn at random from the training publications
  group  calibration molecules are WHOLE training publications, held out from fitting

If the i.i.d. regime under-covers and the group regime does not, the shift is what breaks it and
group-wise calibration is the fix. If both under-cover, conformal prediction does not transfer to
this regime and should not be offered as one.

Calibration never touches the held-out publication. Within each fold the calibration set comes out
of the TRAINING publications only; the validation slice keeps its existing job of hyperparameter
selection, so no model is tuned on its own calibration data.

Entrants are fitted through scripts/run_sota.py's own ENTRANTS, unchanged, by handing each fitter a
test index that is the calibration set followed by the real test set and splitting the returned
vector. One fit produces both, and the models are identical to the ones in docs/sota_comparison.md.

Writes results/lopo_conformal.csv (per-molecule prediction sets) and
results/lopo_conformal_summary.csv.

Usage: python scripts/run_lopo_conformal.py [--models rf,xgb,logreg,chemberta_frozen,chemberta_ft]
                                            [--alphas 0.1,0.2] [--cal-frac 0.25]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from scripts.audit_splits import tanimoto_to_train
from scripts.run_sota import ENTRANTS, load_folds, morgan

PRED = "results/lopo_conformal.csv"
OUT = "results/lopo_conformal_summary.csv"
SPLIT = "paper_lopo"


def conformal_paths(target, split):
    """TG2's published filenames are kept exactly; a replication target gets its own pair."""
    if target == "tg2" and split == SPLIT:
        return PRED, OUT
    stem = f"results/conformal_{target}_{split}"
    return f"{stem}.csv", f"{stem}_summary.csv"
BAND_EDGES = (0.0, 0.4, 0.5, 0.6, 0.7, 1.01)        # identical to scripts/analyse_sota.py
ENSEMBLE = ("rf", "xgb", "logreg")                  # the ensemble docs/uncertainty.md calibrates


def group_calibration(papers, train_idx, frac, rng):
    """Whole training publications, smallest-first, until `frac` of the training molecules are held
    out. Smallest-first keeps the fitting set as large as possible for a given calibration size."""
    p = np.asarray(papers)[train_idx]
    sizes = pd.Series(p).value_counts().sort_values()
    target, taken, held = frac * len(train_idx), 0, []
    for name, n in sizes.items():
        if taken >= target:
            break
        held.append(name)
        taken += n
    mask = np.isin(p, held)
    return train_idx[mask], train_idx[~mask], held


def iid_calibration(train_idx, frac, rng):
    shuffled = rng.permutation(train_idx)
    cut = int(round(frac * len(train_idx)))
    return shuffled[:cut], shuffled[cut:]


def conformal_sets(p_cal, y_cal, p_test, alpha, mondrian):
    """Inductive conformal prediction sets for a binary classifier.

    Nonconformity is 1 - p(true label). The quantile uses the finite-sample correction
    ceil((n+1)(1-alpha))/n, as in scripts/run_uncertainty.py. Mondrian takes the quantile within
    each calibration class, which protects coverage when class balance shifts between folds.
    """
    scores = 1 - np.where(y_cal == 1, p_cal, 1 - p_cal)
    p_mat = np.column_stack([1 - p_test, p_test])

    def q_of(s):
        n = len(s)
        if n == 0:
            return np.inf
        k = int(np.ceil((n + 1) * (1 - alpha)))
        return np.inf if k > n else np.sort(s)[k - 1]

    if mondrian:
        q = [q_of(scores[y_cal == k]) for k in (0, 1)]
    else:
        q = [q_of(scores)] * 2
    return np.column_stack([(1 - p_mat[:, k]) <= q[k] for k in (0, 1)]), q


def risk_coverage(y, p, levels=(1.0, 0.8, 0.6, 0.4, 0.2)):
    """Error rate at a 0.5 threshold among the most confident fraction kept."""
    order = np.argsort(-np.abs(p - 0.5))
    wrong = ((p > 0.5).astype(int) != y)[order]
    return {c: float(wrong[:max(1, int(round(c * len(y))))].mean()) for c in levels}


def main(models, alphas, cal_frac, split, splits_path, pred_path, out_path):
    rng = np.random.default_rng(0)
    rows = []

    for fold in load_folds(split, None, splits_path):
        tr, va, te = fold["idx"]
        y, smiles = fold["y"], fold["smiles"]
        cal_g, fit_g, held = group_calibration(fold["papers"], tr, cal_frac, rng)
        cal_i, fit_i = iid_calibration(tr, cal_frac, rng)
        # paper_lopo names the one held-out publication; paper_grouped holds out a block of them
        # and carries no single name, so printing fold["paper"] there showed "held out nan".
        held_label = fold["paper"] if isinstance(fold["paper"], str) else "a block of publications"
        print(f"\nfold {fold['seed']:>2} (held out {held_label}): test {len(te)}, "
              f"group-cal {len(cal_g)} from {len(held)} papers, iid-cal {len(cal_i)}")
        if len(cal_g) == 0 or len(fit_g) == 0:
            print("   skipped: no group calibration set available")
            continue

        regimes = {"group": (fit_g, cal_g), "iid": (fit_i, cal_i)}
        preds = {}                                   # (regime, model) -> (p_cal, p_test)
        for regime, (fit_idx, cal_idx) in regimes.items():
            probe = np.concatenate([cal_idx, te])
            for model in models:
                sub = dict(fold, idx=(fit_idx, va, probe))
                p, _ = ENTRANTS[model](model, sub, fold["seed"])
                preds[(regime, model)] = (p[:len(cal_idx)], p[len(cal_idx):])
            if all(m in models for m in ENSEMBLE):
                stack = [preds[(regime, m)] for m in ENSEMBLE]
                preds[(regime, "ensemble")] = (np.mean([s[0] for s in stack], axis=0),
                                               np.mean([s[1] for s in stack], axis=0))

        # similarity of each test molecule to the molecules its model actually saw
        fp = morgan(smiles)
        sim = {r: tanimoto_to_train(fp[te], fp[regimes[r][0]]) for r in regimes}

        for (regime, model), (p_cal, p_test) in preds.items():
            cal_idx = regimes[regime][1]
            for alpha in alphas:
                for mondrian in (False, True):
                    sets, q = conformal_sets(p_cal, y[cal_idx], p_test, alpha, mondrian)
                    size = sets.sum(1)
                    covered = sets[np.arange(len(te)), y[te]]
                    single = size == 1
                    rows.append({
                        "section": "conformal",
                        "fold": fold["seed"], "held_out_paper": fold["paper"],
                        "regime": regime, "model": model, "alpha": alpha,
                        "mondrian": mondrian, "n_test": len(te),
                        "n_cal": len(cal_idx), "n_fit": len(regimes[regime][0]),
                        "cal_papers": len(held) if regime == "group" else np.nan,
                        "coverage": float(covered.mean()),
                        "mean_set_size": float(size.mean()),
                        "singleton_frac": float(single.mean()),
                        "empty_frac": float((size == 0).mean()),
                        "singleton_error": float((~covered[single]).mean()) if single.any()
                        else np.nan,
                        "q_lo": float(q[0]), "q_hi": float(q[1]),
                    })
        # per-molecule output, for banding and for any later re-analysis
        for (regime, model), (_, p_test) in preds.items():
            for alpha in alphas:
                sets, _ = conformal_sets(preds[(regime, model)][0], y[regimes[regime][1]],
                                         p_test, alpha, mondrian=False)
                pd.DataFrame({
                    "fold": fold["seed"], "regime": regime, "model": model, "alpha": alpha,
                    "smiles": [smiles[i] for i in te], "y": y[te], "p": p_test,
                    "max_tanimoto": sim[regime],
                    "in_set_0": sets[:, 0], "in_set_1": sets[:, 1],
                }).to_csv(pred_path, mode="a", header=not os.path.exists(pred_path), index=False)

        for model in sorted({m for _, m in preds}):
            rc = risk_coverage(y[te], preds[("group", model)][1])
            rows.append({"section": "risk_coverage", "fold": fold["seed"],
                         "held_out_paper": fold["paper"], "regime": "group", "model": model,
                         "n_test": len(te),
                         **{f"error_at_{int(c * 100)}pct_coverage": v for c, v in rc.items()}})

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nwrote {out_path} and {pred_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="rf,xgb,logreg,chemberta_frozen,chemberta_ft")
    ap.add_argument("--alphas", default="0.1,0.2")
    ap.add_argument("--cal-frac", type=float, default=0.25)
    ap.add_argument("--target", default="tg2", choices=["tg2", "cdk2", "maob"])
    ap.add_argument("--split", default=SPLIT, choices=["paper_lopo", "paper_grouped"])
    a = ap.parse_args()

    from scripts.run_sota import paths
    splits_path, _, _ = paths(a.target)
    pred_path, out_path = conformal_paths(a.target, a.split)
    if os.path.exists(pred_path):
        os.remove(pred_path)                         # the per-molecule file is appended to
    main(a.models.split(","), [float(x) for x in a.alphas.split(",")], a.cal_frac,
         a.split, splits_path, pred_path, out_path)
