"""Experiment 3: does the class balance of the validation fold explain test-score variance?

Scaffold splitting leaves whole analog series intact, so the validation fold's class balance is
whatever the shuffle gives. On TG2 it ranges from 8.5% positive (seed 0: 54 negatives, 5 positives)
to 73% (seed 1) while the dataset as a whole is 47.9% positive. Every entrant in the sweep selects
its configuration on that fold, so an extreme fold means selecting on an unreliable signal.

Pre-registered before any fit, and recorded here rather than in a commit message:

  H1  For models where the validation fold also decides *how much training happens*, test AUROC
      falls as the fold gets more imbalanced: slope on |valid positive fraction - 0.5| < 0.
  H0  For models where it only picks hyperparameters of a converged fit, the slope is null.
      This group is the control. An effect in both groups means something other than selection.

  Groups, fixed by reading scripts/run_sota.py and scripts/_run_chemprop.py:
    uses_valid_for_training  chemprop, chemprop_rdkit, chemeleon (EarlyStopping and
                             ModelCheckpoint on val_loss, inside every configuration) and
                             chemberta_ft (3 epoch budgets selected on validation AUROC)
    control                  rf, logreg, xgb, tabpfn, chemberta_frozen, molformer_frozen
                             (grid over a converged fit; no training-length decision)

  Reporting rule: p > 0.05 is a null, not a trend. A positive result must replicate on CDK2 and
  MAO-B before it is written up as a finding.

The design in docs/review_2026-11-09.md assumed the fingerprint models ignore the validation fold.
They do not -- every entrant tunes on it, over 3 to 12 configurations -- so the control group is
defined by what the fold decides rather than by whether it is read at all. Recorded as a deviation.

Usage: python scripts/analyse_fold_balance.py
Outputs: results/fold_balance.csv (per model x arm x seed), results/fold_balance_slopes.csv,
         results/fold_balance_confound.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
USES_VALID_FOR_TRAINING = ("chemprop", "chemprop_rdkit", "chemeleon", "chemberta_ft")
ARMS = {"tg2": ("scaffold", "scaffold_pub"), "cdk2": ("scaffold_pub",), "maob": ("scaffold_pub",)}
MIN_SEEDS = 8  # a slope over fewer points is not worth reporting


def fold_composition(target):
    """Positive fraction and size of each fold, per arm and seed."""
    s = pd.read_csv(RESULTS / f"splits_{target}.csv", low_memory=False)
    g = s.groupby(["split_kind", "fold", "seed"]).y.agg(["mean", "size"])
    return g.rename(columns={"mean": "pos_frac", "size": "n"}).reset_index()


def build(target):
    comp = fold_composition(target)
    wide = comp.pivot_table(index=["split_kind", "seed"], columns="fold", values=["pos_frac", "n"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index().rename(columns={"split_kind": "split"})

    scores = pd.read_csv(RESULTS / f"sota_{target}.csv")
    keep = [c for c in ("model", "split", "seed", "auroc", "valid_auroc", "n_configs") if c in scores]
    d = scores[keep].merge(wide, on=["split", "seed"], how="inner")
    d = d[d.split.isin(ARMS[target])].copy()
    d["target"] = target
    d["valid_imbalance"] = (d.pos_frac_valid - 0.5).abs()
    d["test_imbalance"] = (d.pos_frac_test - 0.5).abs()
    d["group"] = np.where(d.model.isin(USES_VALID_FOR_TRAINING), "uses_valid_for_training", "control")
    return d


def slopes(d):
    """One OLS slope per target x arm x model, test AUROC on validation-fold imbalance."""
    out = []
    for (target, split, model), g in d.groupby(["target", "split", "model"]):
        g = g.dropna(subset=["auroc", "valid_imbalance"])
        if len(g) < MIN_SEEDS or g.valid_imbalance.nunique() < 3:
            continue
        r = linregress(g.valid_imbalance, g.auroc)
        half = 1.96 * r.stderr
        out.append({"target": target, "split": split, "model": model,
                    "group": g.group.iloc[0], "seeds": len(g),
                    "auroc_sd": g.auroc.std(), "slope": r.slope,
                    "ci_lo": r.slope - half, "ci_hi": r.slope + half,
                    "r2": r.rvalue ** 2, "p": r.pvalue})
    return pd.DataFrame(out)


def confounds(d):
    """Is validation imbalance entangled with test-fold composition? If it is, a slope on it is
    not evidence about the validation fold."""
    out = []
    for (target, split), g in d.groupby(["target", "split"]):
        one = g.drop_duplicates("seed")
        if len(one) < MIN_SEEDS:
            continue
        row = {"target": target, "split": split, "seeds": len(one)}
        for name, col in (("test_imbalance", "test_imbalance"), ("test_pos_frac", "pos_frac_test"),
                          ("n_train", "n_train")):
            if col not in one or one[col].nunique() < 3:
                continue
            r = pearsonr(one.valid_imbalance, one[col])
            row[f"r_{name}"], row[f"p_{name}"] = r.statistic, r.pvalue
        out.append(row)
    return pd.DataFrame(out)


def variance_vs_skill(d):
    """The alternative explanation for per-seed spread: weaker models are less stable, whatever
    they select on. One correlation per arm, over models."""
    out = []
    for (target, split), g in d.groupby(["target", "split"]):
        per = g.groupby("model").auroc.agg(["mean", "std"]).dropna()
        if len(per) < 4:
            continue
        r = pearsonr(per["mean"], per["std"])
        out.append({"target": target, "split": split, "models": len(per),
                    "r_mean_vs_sd": r.statistic, "p": r.pvalue})
    return pd.DataFrame(out)


def main():
    d = pd.concat([build(t) for t in ARMS], ignore_index=True)
    d.to_csv(RESULTS / "fold_balance.csv", index=False)

    sl = slopes(d)
    sl.to_csv(RESULTS / "fold_balance_slopes.csv", index=False)
    cf = confounds(d)
    cf.to_csv(RESULTS / "fold_balance_confound.csv", index=False)

    print(f"validation positive fraction, per arm:")
    for (target, split), g in d.groupby(["target", "split"]):
        one = g.drop_duplicates("seed")
        print(f"  {target:5s} {split:13s} {len(one)} seeds, "
              f"{one.pos_frac_valid.min():.3f} to {one.pos_frac_valid.max():.3f} "
              f"(median {one.pos_frac_valid.median():.3f})")

    print("\nslopes of test AUROC on |validation positive fraction - 0.5|:")
    for group, g in sl.groupby("group"):
        neg = int((g.slope < 0).sum())
        print(f"  {group:24s} {neg}/{len(g)} negative, "
              f"{int((g.p < 0.05).sum())} with p < 0.05, mean slope {g.slope.mean():+.3f}")

    print("\nSD of test AUROC across seeds, by group:")
    print(d.groupby(["target", "split", "group"]).auroc.std().round(3).to_string())

    vs = variance_vs_skill(d)
    vs.to_csv(RESULTS / "fold_balance_variance.csv", index=False)
    print("\ndoes a weaker model simply vary more? (mean AUROC against SD, over models)")
    print(vs.round(3).to_string(index=False))
    return d, sl, cf, vs


if __name__ == "__main__":
    main()
