"""How much AUROC does each model lose when publications are held out instead of scaffolds?

This is the replication's primary endpoint, and `analyse_sota.py` cannot produce it: that script
compares models *within* one split, while the hypothesis is about the same model *across* two
splits. Both arms run on the same molecules at the same training-set size, so the difference is
attributable to the grouping variable rather than to the data or to how much of it was seen.

The two arms are NOT paired. `scaffold_pub` seed 3 and `paper_grouped` fold 3 hold out different
molecules, so a paired t-test over fold index would be pairing unrelated quantities. Welch's
two-sample test is used instead -- it also drops the equal-variance assumption, which matters here
because the publication arm's fold-to-fold spread is roughly twice the scaffold arm's.

Reads only committed result files; fits nothing.

Reproduce: python scripts/analyse_split_penalty.py --target cdk2
  -> results/split_penalty_<target>.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from scripts.analyse_sota import LABELS, ORDER, TARGETS

OUT = "results/split_penalty_{target}.csv"
DEFAULT_ARMS = {"tg2": ("scaffold_pub", "paper_lopo")}      # TG2 published leave-one-out
FALLBACK_ARMS = ("scaffold_pub", "paper_grouped")


def penalty(d, baseline_arm, shift_arm):
    """One row per model: the AUROC difference between the two arms, with a Welch interval."""
    rows = []
    for model in [m for m in ORDER if m in set(d["model"])]:
        g = d[d["model"] == model]
        # AUROC is NaN on single-class folds, which leave-one-publication-out produces in
        # quantity (20 of 83 on CDK2). They are dropped, and counted, because dropping them is
        # itself a selection effect: see docs/replication_feasibility.md section 2.
        a = g.loc[g["split"] == baseline_arm, "auroc"]
        b = g.loc[g["split"] == shift_arm, "auroc"]
        undefined = int(a.isna().sum() + b.isna().sum())
        a, b = a.dropna().to_numpy(), b.dropna().to_numpy()
        if len(a) < 2 or len(b) < 2:
            continue
        res = ttest_ind(b, a, equal_var=False)
        ci = res.confidence_interval(0.95)
        # Cohen's d on the pooled SD of the two arms; the arms are independent samples of folds.
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        rows.append({
            "model": model, "label": LABELS.get(model, model),
            "folds_baseline": len(a), "folds_shift": len(b), "folds_undefined": undefined,
            "auroc_baseline": a.mean(), "auroc_shift": b.mean(),
            "penalty": b.mean() - a.mean(), "ci_lo": ci.low, "ci_hi": ci.high,
            "cohens_d": (b.mean() - a.mean()) / pooled if pooled else np.nan,
            "sd_baseline": a.std(ddof=1), "sd_shift": b.std(ddof=1),
            "p_welch": res.pvalue,
        })
    return holm(pd.DataFrame(rows))


def holm(out):
    """Holm across models on the one endpoint, matching analyse_sota.py's correction."""
    if out.empty:
        return out
    out = out.sort_values("p_welch").reset_index(drop=True)
    running = 0.0
    for i, row in out.iterrows():
        running = max(running, min(1.0, (len(out) - i) * row["p_welch"]))
        out.loc[i, "p_holm"] = running
    return out.sort_values("penalty").reset_index(drop=True)


def main():
    from scripts.run_sota import paths

    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="cdk2", choices=list(TARGETS))
    ap.add_argument("--arms", default=None,
                    help="baseline,shift split names; defaults per target")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base, shift = (tuple(args.arms.split(",")) if args.arms
                   else DEFAULT_ARMS.get(args.target, FALLBACK_ARMS))
    _, sota_path, _ = paths(args.target)
    d = pd.read_csv(sota_path)
    missing = {base, shift} - set(d["split"])
    if missing:
        raise SystemExit(f"{sota_path} has no folds for {sorted(missing)}; "
                         f"it has {sorted(set(d['split']))}")

    out = penalty(d, base, shift)
    print(f"\n## {TARGETS[args.target]}: {base} -> {shift}, AUROC penalty per model")
    print("   Welch's two-sample test (the arms are not paired), Holm over "
          f"{len(out)} models\n")
    print(f"{'model':<18}{'baseline':>10}{'shift':>8}{'penalty':>10}{'95% CI':>20}"
          f"{'d':>7}{'p':>8}{'p_holm':>9}")
    for _, r in out.iterrows():
        print(f"{r['model']:<18}{r['auroc_baseline']:>10.3f}{r['auroc_shift']:>8.3f}"
              f"{r['penalty']:>+10.3f}"
              f"{f'[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]':>20}"
              f"{r['cohens_d']:>7.2f}{r['p_welch']:>8.3f}{r['p_holm']:>9.3f}")

    worse = int((out["penalty"] < 0).sum())
    print(f"\n  {worse}/{len(out)} models worse under {shift}; "
          f"{int((out['p_holm'] < 0.05).sum())} survive Holm at 0.05")
    dropped = int(out["folds_undefined"].max())
    if dropped:
        print(f"  {dropped} folds per model had an undefined AUROC (single-class) and were "
              f"dropped; {int(out['folds_shift'].min())} remain on the {shift} arm")

    path = args.out or OUT.format(target=args.target)
    out.assign(target=args.target, baseline_arm=base, shift_arm=shift).to_csv(path, index=False)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
