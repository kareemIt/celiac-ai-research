"""Figures for the write-up. Reads only saved result CSVs, so it refits nothing.

Usage: python scripts/make_figures.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "results/figures"
PALETTE = {"rf": "#2d6a9f", "xgb": "#c2693a", "logreg": "#7a8b99",
           "ensemble": "#1b4332", "qwen_zeroshot": "#9b5fa8", "qwen_qlora": "#b03a5b"}


def fig_similarity():
    """AUROC against how close the nearest training molecule is."""
    d = pd.read_csv("results/generalization_bins.csv")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    order = sorted(d["bin"].unique())
    for name, g in d.groupby("model"):
        g = g.set_index("bin").reindex(order).dropna(subset=["auroc"])
        ax.plot(g.index, g.auroc, marker="o", label=name, color=PALETTE.get(name), lw=2)
    ax.axhline(0.5, ls="--", c="#999", lw=1)
    ax.text(0.02, 0.505, "chance", color="#777", fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_xlabel("maximum Tanimoto similarity to any training molecule")
    ax.set_ylabel("test AUROC (pooled over 10 scaffold seeds)")
    ax.set_title("TG2 potency ranking degrades on unfamiliar chemistry")
    ax.set_ylim(0.35, 1.0)
    ax.legend(frameon=False, fontsize=8)
    counts = d[d.model == "rf"].set_index("bin").reindex(order)["n"]
    for x, n in enumerate(counts):
        if not np.isnan(n):
            ax.annotate(f"n={int(n)}", (x, 0.37), ha="center", fontsize=7, color="#777")
    fig.tight_layout()
    fig.savefig(f"{OUT}/similarity_vs_auroc.png", dpi=160)


def fig_selective():
    """Risk-coverage: error rate of what you keep, by abstention criterion."""
    d = pd.read_csv("results/uncertainty_selective.csv")
    d = d[d.split == "scaffold"].groupby(["criterion", "coverage"])["error"].agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = {"confidence": "#1b4332", "agreement": "#2d6a9f", "similarity": "#c2693a", "random": "#999999"}
    for crit, g in d.groupby("criterion"):
        ax.errorbar(g.coverage, g["mean"], yerr=g["sem"], marker="o", capsize=2,
                    label=crit, color=colors.get(crit), lw=2)
    ax.set_xlabel("fraction of molecules kept (coverage)")
    ax.set_ylabel("error rate on kept molecules")
    ax.set_title("Abstaining on low confidence works; abstaining on low similarity does not")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/risk_coverage.png", dpi=160)


def fig_conformal():
    """Conformal sets get wider under scaffold shift instead of losing coverage."""
    d = pd.read_csv("results/uncertainty_conformal.csv")
    d = d[~d.mondrian].groupby(["split", "alpha"])[["target_coverage", "coverage", "mean_set_size"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    labels = [f"{s}\nalpha={a}" for s, a in d.index]
    x = np.arange(len(d))
    axes[0].bar(x, d.coverage, color=["#2d6a9f", "#2d6a9f", "#c2693a", "#c2693a"])
    for i, t in enumerate(d.target_coverage):
        axes[0].hlines(t, i - 0.4, i + 0.4, color="k", ls="--", lw=1.2)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("empirical coverage"); axes[0].set_ylim(0.7, 1.0)
    axes[0].set_title("Coverage holds (dashed = target)", fontsize=10)
    axes[1].bar(x, d.mean_set_size, color=["#2d6a9f", "#2d6a9f", "#c2693a", "#c2693a"])
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("mean prediction-set size")
    axes[1].set_title("...by becoming less decisive", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/conformal.png", dpi=160)


def fig_models():
    """Headline model comparison with 95% CIs over seeds."""
    rows = []
    base = pd.read_csv("results/baselines_tg2_combined.csv")
    base = base[base.task == "classification"]
    for m in ["prior", "logreg", "rf", "xgb"]:
        rows.append((m, base[base.model == m].auroc.values))
    tuned = pd.read_csv("results/tuned_baselines_tg2.csv")
    rows.append(("rf_tuned", tuned[tuned.model == "rf_tuned"].auroc.values))
    ens = pd.read_csv("results/uncertainty_ensemble.csv")
    rows.append(("ensemble", ens[(ens.split == "scaffold") & (ens.model == "ensemble")].auroc.values))
    zs = pd.read_csv("results/jev_probe_tg2.csv")
    rows.append(("qwen_zeroshot", zs[zs.model == "jev_smiles+name"].auroc.values))
    q = "results/jev_qlora_tg2_combined.csv"
    if os.path.exists(q):
        v = pd.read_csv(q).auroc.values
        rows.append((f"qwen_qlora (n={len(v)})", v))

    rows = [(n, v[~np.isnan(v)]) for n, v in rows if len(v)]
    rows.sort(key=lambda r: np.mean(r[1]))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = np.arange(len(rows))
    means = [np.mean(v) for _, v in rows]
    errs = [1.96 * np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0 for _, v in rows]
    cols = ["#b03a5b" if "qwen" in n else "#2d6a9f" for n, _ in rows]
    ax.barh(y, means, xerr=errs, color=cols, capsize=3)
    ax.set_yticks(y); ax.set_yticklabels([n for n, _ in rows], fontsize=9)
    ax.axvline(0.5, ls="--", c="#999", lw=1)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("test AUROC, mean with 95% CI over scaffold seeds")
    ax.set_title("TG2 potency, 631 molecules, identical scaffold splits")
    fig.tight_layout()
    fig.savefig(f"{OUT}/model_comparison.png", dpi=160)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (fig_similarity, fig_selective, fig_conformal, fig_models):
        f()
        print("wrote", f.__name__)
    print(f"figures in {OUT}/")
