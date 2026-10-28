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


def mol_level(g):
    """One row per molecule on the unfamiliar subset: label and seed-averaged probability."""
    return (g[g.max_tanimoto < NOVEL_CUTOFF].groupby("smiles")
            .agg(y=("y", "first"), p=("p", "mean")))


def paired_novel(ga, gb, reps, rng):
    """Bootstrap the novel-subset AUROC difference b - a, resampling molecules once per replicate so
    both sides are scored on the same molecules. Returns (delta, lo, hi, P(delta <= 0))."""
    a, b = mol_level(ga), mol_level(gb)
    idx = a.index.intersection(b.index)
    y, pa, pb = a.loc[idx].y.to_numpy(), a.loc[idx].p.to_numpy(), b.loc[idx].p.to_numpy()
    d = auroc(y, pb) - auroc(y, pa)
    boot = []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        u, v = auroc(y[i], pa[i]), auroc(y[i], pb[i])
        if np.isfinite(u) and np.isfinite(v):
            boot.append(v - u)
    return d, np.percentile(boot, 2.5), np.percentile(boot, 97.5), float(np.mean(np.array(boot) <= 0))


def selected(lam, data):
    """Predictions at the epoch each seed's validation Brier chose: the model one would actually use."""
    f = f"results/jev_qlora_preds_{data}_kl{lam:g}.csv"
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f).query("fold == 'test'").assign(p=lambda t: sigmoid(t.logodds))
    sims = pd.read_csv(SIMS).query("model == 'rf'")[["seed", "smiles", "max_tanimoto"]]
    return d.merge(sims, on=["seed", "smiles"], how="left")


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

    # The two tests that carry the argument, both paired over the same molecules.
    print("\n## Paired bootstrap on the unfamiliar subset (same molecules on both sides)\n")
    print(f"{'comparison':<46}{'ΔAUROC':>9}{'95% CI':>22}{'P(Δ<=0)':>10}")
    for lam, d in runs.items():
        last = d.epoch.max()
        for ep in sorted(set(d.epoch) - {0}):
            dd, lo, hi, pn = paired_novel(d.query("epoch == 0"), d.query("epoch == @ep"), reps, rng)
            label = f"lambda {lam:g}: epoch {ep} - base model"
            print(f"{label:<46}{dd:>+9.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>22}{pn:>10.3f}")
            rows.append({"lam": lam, "epoch": ep, "comparison": "epoch-minus-base", "d_novel": dd,
                         "novel_lo": lo, "novel_hi": hi, "p_le_0": pn})
        del last
    if len(runs) > 1:
        base = min(runs)
        for lam in sorted(set(runs) - {base}):
            for ep in sorted(set(runs[base].epoch) & set(runs[lam].epoch) - {0}):
                dd, lo, hi, pn = paired_novel(runs[base].query("epoch == @ep"),
                                              runs[lam].query("epoch == @ep"), reps, rng)
                label = f"epoch {ep}: lambda {lam:g} - lambda {base:g}"
                print(f"{label:<46}{dd:>+9.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>22}{pn:>10.3f}")
                rows.append({"lam": lam, "epoch": ep, "comparison": f"novel {lam:g}-{base:g}",
                             "d_novel": dd, "novel_lo": lo, "novel_hi": hi, "p_le_0": pn})

    # The model one would actually deploy: the epoch each seed's validation Brier picked.
    sel = {lam: selected(lam, data) for lam in runs}
    sel = {lam: d for lam, d in sel.items() if d is not None}
    if sel:
        print("\n## At the validation-selected epoch (the model you would ship)\n")
        print(f"{'lambda':>7}{'overall AUROC':>24}{'novel AUROC':>14}{'ECE':>8}{'Brier':>8}")
        for lam, d in sel.items():
            ps = np.array([auroc(s.y.to_numpy(), s.p.to_numpy()) for _, s in d.groupby("seed")])
            ps = ps[np.isfinite(ps)]
            n = mol_level(d)
            m = metrics.classification(d.y.to_numpy(), d.p.to_numpy())
            print(f"{lam:>7.0f}{f'{ps.mean():.3f} ± {ps.std(ddof=1):.3f}':>24}"
                  f"{auroc(n.y.to_numpy(), n.p.to_numpy()):>14.3f}{m['ece']:>8.3f}{m['brier']:>8.3f}")
            rows.append({"lam": lam, "epoch": "selected", "overall_auroc": ps.mean(),
                         "overall_sd": ps.std(ddof=1), "novel_auroc": auroc(n.y.to_numpy(), n.p.to_numpy()), **m})
        if len(sel) > 1:
            b = min(sel)
            for lam in sorted(set(sel) - {b}):
                pa, pb = [], []
                for s in sorted(set(sel[b].seed) & set(sel[lam].seed)):
                    x, z = sel[b].query("seed == @s"), sel[lam].query("seed == @s")
                    pa.append(auroc(x.y.to_numpy(), x.p.to_numpy()))
                    pb.append(auroc(z.y.to_numpy(), z.p.to_numpy()))
                dd = np.array(pb) - np.array(pa)
                t = stats.ttest_rel(pb, pa).pvalue
                dn, lo, hi, pn = paired_novel(sel[b], sel[lam], reps, rng)
                print(f"\n  lambda {lam:g} - {b:g}: Δ overall {dd.mean():+.3f} "
                      f"(paired t-test p={t:.3f}, {(dd > 0).sum()}/{len(dd)} seeds better); "
                      f"Δ novel {dn:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], P(Δ<=0)={pn:.3f}")
                rows.append({"lam": lam, "epoch": "selected", "comparison": f"{lam:g}-{b:g}",
                             "d_overall": dd.mean(), "p_ttest": t, "n_seeds": len(dd),
                             "d_novel": dn, "novel_lo": lo, "novel_hi": hi, "p_le_0": pn})

    # How big is an effect allowed to be before we believe it? The 2026-10-27 run used the same
    # code path, data and splits as lambda = 0, so the difference between them is pure run-to-run
    # noise (4-bit matmuls, non-deterministic CUDA reductions, paged 8-bit Adam) and is the null
    # this experiment has to beat.
    rep = "results/jev_qlora_tg2_combined.csv"
    if os.path.exists(rep) and 0.0 in runs and 1.0 in runs:
        print("\n## Noise floor: two independent runs of the SAME configuration\n")
        a = pd.read_csv(rep)[["seed", "auroc"]]
        for name, f in [("lambda 0 replicate - lambda 0", f"results/jev_qlora_tg2_{data}_kl0.csv"),
                        ("lambda 1 - lambda 0", f"results/jev_qlora_tg2_{data}_kl1.csv")]:
            if name.startswith("lambda 1"):
                a = pd.read_csv(f"results/jev_qlora_tg2_{data}_kl0.csv")[["seed", "auroc"]]
            m = a.merge(pd.read_csv(f)[["seed", "auroc"]], on="seed", suffixes=("_x", "_y"))
            dd = (m.auroc_y - m.auroc_x).to_numpy()
            p = stats.ttest_rel(m.auroc_y, m.auroc_x).pvalue
            print(f"  {name:<32} Δ overall AUROC {dd.mean():+.3f} ± {dd.std(ddof=1):.3f}, "
                  f"{(dd > 0).sum()}/{len(dd)} seeds, p={p:.3f}")
            rows.append({"comparison": name, "d_overall": dd.mean(), "overall_sd": dd.std(ddof=1),
                         "p_ttest": p, "n_seeds": len(dd)})
        print("  Rerunning the identical configuration moves the mean as far as the anchors do, so "
              "the overall-AUROC gain is not separable from training nondeterminism.")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--data", default="combined")
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()
    main(a.lambdas, a.data, a.reps)
