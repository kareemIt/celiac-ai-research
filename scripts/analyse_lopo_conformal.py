"""Read results/lopo_conformal*.csv and answer the five Phase-2 questions.

Coverage, set size, how many molecules get a useful (singleton) answer, what the error rate is on
those, and all of it again inside similarity bands. Pooled over molecules rather than averaged over
folds, because LOPO folds range from 10 to 106 molecules and a fold mean would weight a 10-molecule
publication like a 106-molecule one.

Usage: python scripts/analyse_lopo_conformal.py [--target cdk2] [--split paper_grouped]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy import stats

from scripts.run_lopo_conformal import BAND_EDGES, OUT, PRED, SPLIT, conformal_paths

STATS = "results/lopo_conformal_stats.csv"


def stats_path(target, split):
    """TG2's published filename is kept exactly; a replication target gets its own."""
    if target == "tg2" and split == SPLIT:
        return STATS
    return f"results/conformal_{target}_{split}_stats.csv"

ORDER = ["rf", "xgb", "logreg", "ensemble", "chemberta_frozen", "chemberta_ft"]
LABELS = {"rf": "random forest", "xgb": "XGBoost", "logreg": "logistic regression",
          "ensemble": "ensemble (rf+xgb+logreg)", "chemberta_frozen": "ChemBERTa-2 (frozen)",
          "chemberta_ft": "ChemBERTa-2 (fine-tuned)"}


def pooled(preds, alpha):
    """Per-molecule metrics pooled over folds, for the marginal (non-Mondrian) sets."""
    d = preds[preds.alpha == alpha].copy()
    d["size"] = d.in_set_0.astype(int) + d.in_set_1.astype(int)
    d["covered"] = np.where(d.y == 1, d.in_set_1, d.in_set_0)
    d["single"] = d["size"] == 1
    d["wrong_singleton"] = d.single & ~d.covered
    return d


def table(d, by):
    g = d.groupby(by)
    out = pd.DataFrame({
        "n": g.size(),
        "coverage": g.covered.mean(),
        "mean_set_size": g["size"].mean(),
        "useful_frac": g.single.mean(),
        "error_on_useful": g.apply(
            lambda x: (~x.covered[x.single]).mean() if x.single.any() else np.nan,
            include_groups=False),
        "empty_frac": g.apply(lambda x: (x["size"] == 0).mean(), include_groups=False),
    })
    return out.reset_index()


def main(target="tg2", split=SPLIT):
    pred_path, out_path = conformal_paths(target, split)
    stats_out = stats_path(target, split)
    if not (os.path.exists(pred_path) and os.path.exists(out_path)):
        raise SystemExit(f"run scripts/run_lopo_conformal.py --target {target} "
                         f"--split {split} first ({pred_path} missing)")
    rows_out = []
    preds = pd.read_csv(pred_path)
    summary = pd.read_csv(out_path)
    conf = summary[summary.section == "conformal"]
    alphas = sorted(preds.alpha.unique())
    models = [m for m in ORDER if m in set(preds.model)]
    print(f"{preds.fold.nunique()} folds, {preds.model.nunique()} models, "
          f"{len(preds[(preds.model == models[0]) & (preds.regime == 'group') & (preds.alpha == alphas[0])])}"
          f" molecules scored per model\n")

    print("## 1. Does coverage hold when a whole publication is held out?\n")
    print("   Target is 1 - alpha. 'iid' calibrates on random training molecules, 'group' on whole")
    print("   training publications. Pooled over molecules.\n")
    for alpha in alphas:
        d = pooled(preds, alpha)
        print(f"   alpha={alpha} (target coverage {1 - alpha:.2f})")
        t = table(d, ["regime", "model"])
        t["model"] = t.model.map(LABELS).fillna(t.model)
        print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print()

    print("## 2. Mondrian (class-wise) calibration\n")
    m = conf.groupby(["regime", "mondrian", "alpha"]).apply(
        lambda x: pd.Series({
            "coverage": np.average(x.coverage, weights=x.n_test),
            "mean_set_size": np.average(x.mean_set_size, weights=x.n_test),
            "singleton_frac": np.average(x.singleton_frac, weights=x.n_test),
            "folds_with_infinite_q": int(np.isinf(x.q_lo).sum() + np.isinf(x.q_hi).sum()),
        }), include_groups=False).reset_index()
    print(m.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n## 3. How many molecules get a usable answer, and is it right?\n")
    for alpha in alphas:
        d = pooled(preds, alpha)
        d = d[d.regime == "group"]
        print(f"   alpha={alpha}, group calibration")
        t = table(d, ["model"])
        t["model"] = t.model.map(LABELS).fillna(t.model)
        print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print()

    print("## 4. Chemically unfamiliar molecules\n")
    alpha = alphas[0]
    d = pooled(preds, alpha)
    d = d[d.regime == "group"].copy()
    d["band"] = pd.cut(d.max_tanimoto, list(BAND_EDGES), right=False)
    sizes = d[d.model == models[0]].band.value_counts().sort_index()
    print(f"   band sizes (max Tanimoto to the fitting set), alpha={alpha}: "
          f"{ {str(k): int(v) for k, v in sizes.items()} }\n")
    t = table(d, ["band", "model"])
    t["model"] = t.model.map(LABELS).fillna(t.model)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n## 5. Selective prediction: error rate among the most confident molecules\n")
    rc = summary[summary.section == "risk_coverage"]
    cols = [c for c in rc.columns if c.startswith("error_at_")]
    cols.sort(key=lambda c: -int(c.split("_")[2].rstrip("pct")))
    agg = rc.groupby("model").apply(
        lambda x: pd.Series({c: np.average(x[c].astype(float), weights=x.n_test) for c in cols}),
        include_groups=False)
    agg.index = [LABELS.get(i, i) for i in agg.index]
    print(agg.reindex([LABELS.get(m, m) for m in models]).to_string(
        float_format=lambda v: f"{v:.3f}"))
    print("\n   Columns are the fraction of test molecules kept, ranked by |p - 0.5|. Error is at a"
          "\n   0.5 threshold, so this is decision error, not AUROC.")

    print("\n## 6. Are the two headline differences significant?\n")
    # Measured, not hardcoded: these counts were written for TG2's 11 leave-one-out folds and
    # silently mislabelled the replication targets, whose grouped folds hold far more molecules.
    sizes = conf.groupby("fold")["n_test"].first()
    print(f"   Paired per fold over the {len(sizes)} folds. Folds hold {sizes.min()} to "
          f"{sizes.max()} molecules, so the"
          "\n   sign test is reported alongside: it ignores fold size entirely.\n")
    for alpha in alphas:
        w = conf[(conf.alpha == alpha) & (~conf.mondrian)]
        p = w.pivot_table(index=["fold", "model"], columns="regime", values="coverage")
        p = p.dropna()
        d = p["group"] - p["iid"]
        t = stats.ttest_rel(p["group"], p["iid"])
        sign = stats.binomtest(int((d > 0).sum()), len(d)).pvalue
        print(f"   group vs iid coverage, alpha={alpha}: mean +{d.mean():.3f} "
              f"[{d.mean() - 1.96 * d.sem():+.3f}, {d.mean() + 1.96 * d.sem():+.3f}]  "
              f"higher in {int((d > 0).sum())}/{len(d)}  t-test p={t.pvalue:.2g}  "
              f"sign p={sign:.2g}")
        rows_out.append({"section": "group_vs_iid", "alpha": alpha, "n_pairs": len(d),
                         "mean_delta_coverage": float(d.mean()), "ttest_p": float(t.pvalue),
                         "sign_p": float(sign), "n_higher": int((d > 0).sum())})

    print()
    rc_f = summary[summary.section == "risk_coverage"]
    for col in cols:
        w = rc_f.pivot_table(index="fold", columns="model", values=col)
        if not {"chemberta_frozen", "rf"} <= set(w.columns):
            continue
        d = (w["rf"] - w["chemberta_frozen"]).dropna()
        t = stats.ttest_rel(w.loc[d.index, "rf"], w.loc[d.index, "chemberta_frozen"])
        sign = stats.binomtest(int((d > 0).sum()), len(d)).pvalue
        print(f"   {col:<28} rf - chemberta_frozen = {d.mean():+.3f}  "
              f"rf worse in {int((d > 0).sum())}/{len(d)} folds  "
              f"t-test p={t.pvalue:.2g}  sign p={sign:.2g}")
        rows_out.append({"section": "rf_vs_chemberta_frozen", "metric": col, "n_pairs": len(d),
                         "mean_delta": float(d.mean()), "ttest_p": float(t.pvalue),
                         "sign_p": float(sign), "n_rf_worse": int((d > 0).sum())})

    pd.DataFrame(rows_out).to_csv(stats_out, index=False)
    print(f"\nwrote {stats_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="tg2", choices=["tg2", "cdk2", "maob"])
    ap.add_argument("--split", default=SPLIT, choices=["paper_lopo", "paper_grouped"])
    a = ap.parse_args()
    main(a.target, a.split)
