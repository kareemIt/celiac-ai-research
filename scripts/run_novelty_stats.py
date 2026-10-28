"""Is the zero-shot Qwen advantage on unfamiliar chemistry real, or an artefact of pooling?

docs/generalization.md reported novel-subset AUROCs (zero-shot 0.807, QLoRA 0.442, RF 0.794) over 51
predictions pooled from 10 overlapping scaffold test folds. Those 51 rows cover 42 distinct
molecules, so they are neither independent nor equally weighted, and the zero-shot score for a given
molecule is literally the same number in every fold it appears in. This script redoes the comparison
without that assumption:

  1. one row per molecule (seed-averaged, and first-seed-only as a control for the averaging)
  2. cluster bootstrap over molecules for every AUROC and every paired difference
  3. per-seed paired comparison, which is the only fully split-respecting test available
  4. an effective-sample-size check: how many chemically distinct clusters those molecules form

Reads results/generalization_preds.csv; refits nothing. CPU only.

Usage: python scripts/run_novelty_stats.py [--cutoff 0.5] [--reps 5000]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from rdkit.ML.Cluster import Butina
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import chem

PREDS = "results/generalization_preds.csv"
OUT = "results/novelty_stats.csv"
PAIRS = [("qwen_zeroshot", "qwen_qlora"), ("qwen_zeroshot", "rf"), ("rf", "qwen_qlora")]


def auroc(y, p):
    """AUROC, or NaN when a resample lands on a single class."""
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def boot_ci(y, cols, reps, rng):
    """Percentile CIs for each model's AUROC and for every paired difference, resampling molecules
    (not predictions) so the two models always see the same resampled set."""
    n = len(y)
    stat = {m: [] for m in cols} | {f"{a}-{b}": [] for a, b in PAIRS}
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if not 0 < ys.sum() < n:
            continue
        a = {m: auroc(ys, cols[m][idx]) for m in cols}
        for m, v in a.items():
            stat[m].append(v)
        for x, z in PAIRS:
            stat[f"{x}-{z}"].append(a[x] - a[z])
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5),
                float(np.mean(np.array(v) <= 0))) for k, v in stat.items() if v}


def n_clusters(smiles, cutoff=0.5):
    """Butina clusters among the novel molecules: 42 near-duplicates are not 42 observations."""
    X = chem.morgan(smiles).astype(float)
    pop = X.sum(1)
    dist = []  # lower triangle, the order Butina expects
    for i in range(1, len(smiles)):
        inter = X[:i] @ X[i]
        dist.extend(1 - inter / (pop[:i] + pop[i] - inter))
    return len(Butina.ClusterData([float(v) for v in dist], len(smiles), 1 - cutoff, isDistData=True))


def main(cutoff, reps):
    pred = pd.read_csv(PREDS)
    nov = pred[pred.max_tanimoto < cutoff]
    models = sorted(nov.model.unique())
    rng = np.random.default_rng(0)

    print(f"## Novel subset: test molecules with max Tanimoto < {cutoff}")
    ref = nov[nov.model == "rf"]
    print(f"{len(ref)} pooled predictions over {ref.smiles.nunique()} distinct molecules; "
          f"{ref.drop_duplicates('smiles').y.sum():.0f} of them positive")
    mols = sorted(ref.smiles.unique())
    k = n_clusters(mols, cutoff=0.5)
    print(f"those molecules form {k} Butina clusters at 0.5 Tanimoto -> effective n is nearer {k} than "
          f"{len(mols)}, and the positives are rarer still\n")

    rows = []
    for how in ["seed-averaged", "first-seed"]:
        g = (nov.groupby(["model", "smiles"]).agg(y=("y", "first"), p=("p", "mean")).reset_index()
             if how == "seed-averaged" else
             nov.sort_values("seed").drop_duplicates(["model", "smiles"])[["model", "smiles", "y", "p"]])
        piv = g.pivot(index="smiles", columns="model", values="p").dropna()
        y = g.drop_duplicates("smiles").set_index("smiles").loc[piv.index, "y"].to_numpy()
        cols = {m: piv[m].to_numpy() for m in models}
        ci = boot_ci(y, cols, reps, rng)

        print(f"### One row per molecule, {how} (n={len(y)}, {y.sum()} positive)")
        print(f"{'model':<16}{'AUROC':>8}{'AUPRC':>8}{'95% CI (bootstrap over molecules)':>38}")
        for m in models:
            a = auroc(y, cols[m])
            lo, hi, _ = ci[m]
            print(f"{m:<16}{a:>8.3f}{average_precision_score(y, cols[m]):>8.3f}{f'[{lo:.3f}, {hi:.3f}]':>38}")
            rows.append({"how": how, "comparison": m, "n": len(y), "auroc": a, "lo": lo, "hi": hi})
        print(f"\n{'paired difference':<28}{'ΔAUROC':>9}{'95% CI':>22}{'P(Δ<=0)':>10}")
        for a, b in PAIRS:
            dd = auroc(y, cols[a]) - auroc(y, cols[b])
            lo, hi, pneg = ci[f"{a}-{b}"]
            flag = "" if lo > 0 or hi < 0 else "   (CI spans 0)"
            print(f"{a + ' - ' + b:<28}{dd:>+9.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>22}{pneg:>10.3f}{flag}")
            rows.append({"how": how, "comparison": f"{a}-{b}", "n": len(y), "auroc": dd,
                         "lo": lo, "hi": hi, "p_le_0": pneg})
        print()

    # Per-seed: the only test that respects the splits, at the cost of tiny per-seed subsets.
    print("### Per-seed paired comparison (seeds whose novel subset has both classes)")
    per = {}
    for (m, s), g in nov.groupby(["model", "seed"]):
        per.setdefault(s, {})[m] = auroc(g.y.to_numpy(), g.p.to_numpy())
    usable = [s for s, v in per.items() if all(np.isfinite(v.get(m, np.nan)) for m in models)]
    print(f"{len(usable)} of {nov.seed.nunique()} seeds usable: {usable}")
    if len(usable) >= 3:
        for m in models:
            v = np.array([per[s][m] for s in usable])
            print(f"  {m:<16} AUROC {v.mean():.3f} ± {v.std(ddof=1):.3f}")
        print(f"\n{'paired difference':<28}{'mean Δ':>9}{'t-test p':>10}{'sign test p':>13}{'better':>9}")
        for a, b in PAIRS:
            dd = np.array([per[s][a] - per[s][b] for s in usable])
            t = stats.ttest_rel([per[s][a] for s in usable], [per[s][b] for s in usable]).pvalue
            w = int((dd > 0).sum())
            sgn = stats.binomtest(w, len(dd), 0.5).pvalue
            print(f"{a + ' - ' + b:<28}{dd.mean():>+9.3f}{t:>10.3f}{sgn:>13.3f}{f'{w}/{len(dd)}':>9}")
            rows.append({"how": "per-seed", "comparison": f"{a}-{b}", "n": len(dd),
                         "auroc": dd.mean(), "p_ttest": t, "p_sign": sgn})
    else:
        print("  too few usable seeds for a paired test; this is itself the headline finding")

    # 0.5 was chosen before any of these numbers were seen, so check the conclusion is not a
    # property of that one threshold.
    print("\n### Sensitivity to the novelty cutoff (molecule-level, seed-averaged)")
    print(f"{'cutoff':>7}{'n mols':>8}{'pos':>5}{'clusters':>10}" + "".join(f"{m:>15}" for m in models)
          + f"{'zs - rf':>10}{'zs - qlora':>12}")
    for c in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        sub = pred[pred.max_tanimoto < c]
        g = sub.groupby(["model", "smiles"]).agg(y=("y", "first"), p=("p", "mean")).reset_index()
        piv = g.pivot(index="smiles", columns="model", values="p").dropna()
        if len(piv) < 5 or not set(models) <= set(piv.columns):
            continue
        yv = g.drop_duplicates("smiles").set_index("smiles").loc[piv.index, "y"].to_numpy()
        a = {m: auroc(yv, piv[m].to_numpy()) for m in models}
        line = (f"{c:>7.2f}{len(yv):>8}{int(yv.sum()):>5}{n_clusters(list(piv.index), 0.5):>10}"
                + "".join(f"{a[m]:>15.3f}" for m in models)
                + f"{a['qwen_zeroshot'] - a['rf']:>+10.3f}{a['qwen_zeroshot'] - a['qwen_qlora']:>+12.3f}")
        print(line)
        rows.append({"how": "cutoff-sweep", "comparison": "zs-rf", "cutoff": c, "n": len(yv),
                     "n_pos": int(yv.sum()), "auroc": a["qwen_zeroshot"] - a["rf"]})
        rows.append({"how": "cutoff-sweep", "comparison": "zs-qlora", "cutoff": c, "n": len(yv),
                     "n_pos": int(yv.sum()), "auroc": a["qwen_zeroshot"] - a["qwen_qlora"]})
    print("Read the leftmost rows with care: at cutoff 0.40 the subset holds a single positive "
          "molecule, so its AUROC is one molecule's rank, not a performance estimate.")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, default=0.5)
    ap.add_argument("--reps", type=int, default=5000)
    a = ap.parse_args()
    main(a.cutoff, a.reps)
