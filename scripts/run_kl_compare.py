"""Does KL anchoring preserve the base model's chemical prior while QLoRA learns TG2 potency?

Reads the per-epoch log-odds written by run_jev_qlora.py for each lambda (epoch 0 is the untouched
base model, so each run contains its own zero-shot control on exactly the same molecules) and reports:

  1. the forgetting curve: overall test AUROC and unfamiliar-chemistry AUROC after each epoch
  2. lambda = 0 against lambda = 1 at matched seeds and epochs, paired
  3. what it costs: AUROC, AUPRC, ECE, Brier on the full test fold

Novel-subset numbers are computed one row per molecule, as run_novelty_stats.py argued they must be,
with a cluster bootstrap over molecules. Refits nothing; CPU only.

Usage: python scripts/run_kl_compare.py [--lambdas 0 1] [--data combined] [--reps 5000]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy import stats

from scripts.run_novelty_stats import auroc
from src.evaluation import metrics

SIMS = "results/generalization_preds.csv"
OUT = "results/kl_compare.csv"
NOVEL_CUTOFF = 0.5


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def load(lam, data):
    f = f"results/jev_qlora_epochs_{data}_kl{lam:g}.csv"
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    d = d[d.fold == "test"].assign(lam=lam, p=lambda t: sigmoid(t.logodds))
    sims = pd.read_csv(SIMS).query("model == 'rf'")[["seed", "smiles", "max_tanimoto"]]
    return d.merge(sims, on=["seed", "smiles"], how="left")


def novel_auroc(g, reps, rng):
    """Molecule-level AUROC on the unfamiliar subset, with a bootstrap CI over molecules."""
    n = g[g.max_tanimoto < NOVEL_CUTOFF].groupby("smiles").agg(y=("y", "first"), p=("p", "mean"))
    y, p = n.y.to_numpy(), n.p.to_numpy()
    a = auroc(y, p)
    if not np.isfinite(a) or reps == 0:
        return a, np.nan, np.nan, len(y)
    b = [auroc(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(reps))]
    b = [v for v in b if np.isfinite(v)]
    return a, np.percentile(b, 2.5), np.percentile(b, 97.5), len(y)


def main(lambdas, data, reps):
    rng = np.random.default_rng(0)
    runs = {lam: load(lam, data) for lam in lambdas}
    missing = [lam for lam, d in runs.items() if d is None]
    if missing:
        print(f"no per-epoch file yet for lambda {missing}; reporting what exists")
    runs = {lam: d for lam, d in runs.items() if d is not None}
    if not runs:
        sys.exit("nothing to compare")

    rows = []
    print("## Forgetting curve: epoch 0 is the base model, before any gradient step\n")
    for lam, d in runs.items():
        print(f"### lambda = {lam:g}  ({d.seed.nunique()} seeds)")
        print(f"{'epoch':>6}{'overall AUROC':>24}{'novel AUROC':>14}{'95% CI':>22}{'n mols':>8}{'ECE':>8}{'Brier':>8}")
        for ep, g in d.groupby("epoch"):
            per_seed = np.array([auroc(s.y.to_numpy(), s.p.to_numpy()) for _, s in g.groupby("seed")])
            per_seed = per_seed[np.isfinite(per_seed)]
            a, lo, hi, nm = novel_auroc(g, reps, rng)
            m = metrics.classification(g.y.to_numpy(), g.p.to_numpy())
            print(f"{ep:>6}{f'{per_seed.mean():.3f} ± {per_seed.std(ddof=1):.3f}':>24}"
                  f"{a:>14.3f}{f'[{lo:.3f}, {hi:.3f}]':>22}{nm:>8}{m['ece']:>8.3f}{m['brier']:>8.3f}")
            rows.append({"lam": lam, "epoch": ep, "overall_auroc": per_seed.mean(),
                         "overall_sd": per_seed.std(ddof=1), "novel_auroc": a, "novel_lo": lo,
                         "novel_hi": hi, "n_novel_mols": nm, **m})
        print()

    if len(runs) > 1:
        print("## Paired at matched seed and epoch: does the anchor change anything?\n")
        print(f"{'comparison':<22}{'epoch':>6}{'Δ overall AUROC':>18}{'t-test p':>10}{'better':>9}"
              f"{'Δ novel AUROC':>16}")
        base = min(runs)
        for lam in sorted(runs):
            if lam == base:
                continue
            for ep in sorted(set(runs[base].epoch) & set(runs[lam].epoch)):
                if ep == 0:
                    continue
                pa, pb = [], []
                for seed in sorted(set(runs[base].seed) & set(runs[lam].seed)):
                    x = runs[base].query("seed == @seed and epoch == @ep")
                    z = runs[lam].query("seed == @seed and epoch == @ep")
                    u, v = auroc(x.y.to_numpy(), x.p.to_numpy()), auroc(z.y.to_numpy(), z.p.to_numpy())
                    if np.isfinite(u) and np.isfinite(v):
                        pa.append(u); pb.append(v)
                if len(pa) < 3:
                    continue
                dd = np.array(pb) - np.array(pa)
                t = stats.ttest_rel(pb, pa).pvalue
                dn = (novel_auroc(runs[lam].query("epoch == @ep"), 0, rng)[0]
                      - novel_auroc(runs[base].query("epoch == @ep"), 0, rng)[0])
                print(f"{f'lambda {lam:g} - {base:g}':<22}{ep:>6}{dd.mean():>+18.3f}{t:>10.3f}"
                      f"{f'{(dd > 0).sum()}/{len(dd)}':>9}{dn:>+16.3f}")
                rows.append({"lam": lam, "epoch": ep, "comparison": f"{lam:g}-{base:g}",
                             "d_overall": dd.mean(), "p_ttest": t, "n_seeds": len(dd), "d_novel": dn})

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--data", default="combined")
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()
    main(a.lambdas, a.data, a.reps)
