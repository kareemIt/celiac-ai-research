"""Results table, paired statistics, generalization bands and figures for the SOTA comparison.

Reads only committed result files; computes nothing new from a model. The incumbents (tuned random
forest, Qwen QLoRA, Qwen zero-shot) come from the files the paper already cites, so this script
never re-derives a number the paper depends on.

Reproduce: python scripts/analyse_sota.py [--split scaffold] [--figures]
  -> results/sota_{summary,paired,bands}_<split>.csv
  -> results/figures/sota_{comparison,calibration,generalization}_<split>.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_rel

from src.data import chem

SOTA = "results/sota_tg2.csv"
PREDS = "results/sota_preds.csv"
SPLITS = "results/splits_tg2.csv"
FIGDIR = "results/figures"
TARGETS = {"tg2": "TG2", "cdk2": "CDK2", "maob": "MAO-B"}

METRICS = ["auroc", "auprc", "mcc", "ece", "brier", "logloss"]
HIGHER_BETTER = {"auroc", "auprc", "mcc"}
BASELINE = "rf"

# Incumbents, from the files the paper cites. Each used the same 10 scaffold seeds.
INCUMBENTS = [
    ("results/jev_qlora_tg2_combined.csv", "jev_qlora", "qwen_qlora"),
    ("results/jev_probe_tg2.csv", "jev_smiles", "qwen_zeroshot"),
]

LABELS = {
    "logreg": "Logistic regression (Morgan)", "rf": "Random forest (Morgan, tuned)",
    "xgb": "XGBoost (Morgan)", "tabpfn": "Tabular foundation model",
    "chemprop": "Chemprop v2 D-MPNN", "chemprop_rdkit": "Chemprop v2 + RDKit 2D",
    "chemeleon": "CheMeleon (pretrained D-MPNN)", "chemberta_frozen": "ChemBERTa-2 (frozen)",
    "chemberta_ft": "ChemBERTa-2 (fine-tuned)", "molformer_frozen": "MoLFormer-XL (frozen)",
    "qwen_qlora": "Qwen3.5-9B QLoRA", "qwen_zeroshot": "Qwen3.5-9B zero-shot",
}
ORDER = list(LABELS)


def load(split, sota=SOTA, incumbents=True):
    """New entrants for this split, plus the incumbents where the split matches theirs.

    Incumbents are TG2-only: the Qwen runs exist for no other target, so a replication target
    loads its entrants alone rather than silently comparing against another target's numbers.
    """
    d = pd.read_csv(sota)
    d = d[d.split == split].copy()
    if split == "scaffold" and incumbents:
        for path, model, name in INCUMBENTS:
            if not os.path.exists(path):
                continue
            inc = pd.read_csv(path)
            inc = inc[inc.model == model].copy()
            if inc.empty:
                continue
            inc["model"], inc["split"] = name, split
            keep = ["model", "split", "seed"] + [m for m in METRICS if m in inc]
            d = pd.concat([d, inc[keep]], ignore_index=True)
    d["model"] = pd.Categorical(d["model"], [m for m in ORDER if m in set(d["model"])], ordered=True)
    return d.sort_values(["model", "seed"])


def summarise(d):
    rows = []
    for model, g in d.groupby("model", observed=True, sort=False):
        row = {"model": model, "label": LABELS.get(str(model), str(model)), "folds": len(g)}
        for m in METRICS:
            if m in g and g[m].notna().any():
                row[m] = f"{g[m].mean():.3f} ± {g[m].std():.3f}"
                row[f"{m}_mean"] = g[m].mean()
        for c in ("seconds", "peak_vram_mib", "n_params", "n_configs"):
            if c in g and g[c].notna().any():
                row[c] = g[c].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def paired(d, baseline=BASELINE):
    """Paired per-seed differences against the baseline, on the seeds both models cover."""
    base = d[d.model == baseline].set_index("seed")
    if base.empty:
        return pd.DataFrame()
    rows = []
    for model, g in d.groupby("model", observed=True, sort=False):
        if str(model) == baseline:
            continue
        g = g.set_index("seed")
        shared = sorted(set(g.index) & set(base.index))
        if len(shared) < 3:
            continue
        for m in METRICS:
            if m not in g or m not in base:
                continue
            delta = (g.loc[shared, m] - base.loc[shared, m]).dropna()
            if len(delta) < 3 or delta.std(ddof=1) == 0:
                continue
            signed = delta if m in HIGHER_BETTER else -delta  # positive = entrant better
            wins = int((signed > 0).sum())
            se = signed.std(ddof=1) / np.sqrt(len(signed))
            rows.append({
                "model": str(model), "vs": baseline, "metric": m, "n_seeds": len(signed),
                "mean_delta": signed.mean(), "ci_lo": signed.mean() - 1.96 * se,
                "ci_hi": signed.mean() + 1.96 * se, "wins": wins,
                "p_ttest": ttest_rel(signed, np.zeros(len(signed))).pvalue,
                "p_sign": binomtest(wins, len(signed)).pvalue,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = holm(out)
    return out


def holm(out, metric="auroc"):
    """Holm correction across entrants on the primary endpoint only, as pre-registered.
    Secondary metrics stay uncorrected and are labelled exploratory."""
    out["p_holm"] = np.nan
    primary = out[out.metric == metric].sort_values("p_ttest")
    n = len(primary)
    running = 0.0
    for rank, (i, row) in enumerate(primary.iterrows()):
        running = max(running, min(1.0, (n - rank) * row["p_ttest"]))
        out.loc[i, "p_holm"] = running
    return out


def bands(split, preds_path=PREDS, splits_path=SPLITS,
          edges=(0.0, 0.4, 0.5, 0.6, 0.7, 1.01)):
    """AUROC inside maximum-Tanimoto-to-training bands, one row per molecule per model.

    Pooled per molecule, never per prediction: docs/forgetting.md records how pooling across
    overlapping folds inflated a model whose score was read from a cache.
    """
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit import Chem
    from sklearn.metrics import roc_auc_score

    preds = pd.read_csv(preds_path)
    preds = preds[preds.split == split]
    if preds.empty:
        return pd.DataFrame()
    sp = pd.read_csv(splits_path, low_memory=False)
    sp = sp[sp.split_kind == split]

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = {s: gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in sp["smiles"].unique()}

    sim = {}
    for seed, g in sp.groupby("seed"):
        train = [fps[s] for s in g[g.fold == "train"]["smiles"]]
        for s in g[g.fold == "test"]["smiles"]:
            sim[(seed, s)] = max(DataStructs.BulkTanimotoSimilarity(fps[s], train))
    preds["max_tanimoto"] = [sim[(r.seed, r.smiles)] for r in preds.itertuples()]
    preds["band"] = pd.cut(preds["max_tanimoto"], list(edges), right=False)

    rows = []
    for (model, band), g in preds.groupby(["model", "band"], observed=True):
        # one row per molecule: average the probability where a molecule is tested in several folds
        per_mol = g.groupby("smiles").agg(y=("y", "first"), p=("p", "mean"))
        if len(per_mol) < 5 or per_mol["y"].nunique() < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(per_mol["y"], per_mol["p"])
        rows.append({"model": model, "band": str(band), "n_molecules": len(per_mol),
                     "positive_rate": per_mol["y"].mean(), "auroc": auc})
    return pd.DataFrame(rows)


def figures(d, pair, band_table, split, stem="sota"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGDIR, exist_ok=True)
    present = [m for m in ORDER if m in set(d["model"].astype(str))]

    # 1. AUROC and Brier per model, with the baseline marked
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, metric, better in ((axes[0], "auroc", "higher"), (axes[1], "brier", "lower")):
        vals = [d[d.model.astype(str) == m][metric].dropna() for m in present]
        ax.boxplot(vals, tick_labels=[LABELS[m].replace(" (", "\n(") for m in present],
                   showmeans=True)
        ref = d[d.model.astype(str) == BASELINE][metric].mean()
        ax.axhline(ref, color="crimson", ls="--", lw=1,
                   label=f"tuned random forest ({ref:.3f})")
        ax.set_ylabel(f"{metric.upper()} ({better} is better)")
        ax.tick_params(axis="x", rotation=90, labelsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"TG2 potency, {split} split — per-fold distribution")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/{stem}_comparison_{split}.png", dpi=150)
    plt.close(fig)

    # 2. accuracy against calibration: the two do not improve together
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for m in present:
        g = d[d.model.astype(str) == m]
        if g["auroc"].isna().all() or g["ece"].isna().all():
            continue
        ax.scatter(g["auroc"].mean(), g["ece"].mean(), s=70)
        ax.annotate(LABELS[m], (g["auroc"].mean(), g["ece"].mean()), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("AUROC (higher is better)")
    ax.set_ylabel("Expected calibration error (lower is better)")
    ax.set_title(f"Ranking skill against calibration, {split} split")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/{stem}_calibration_{split}.png", dpi=150)
    plt.close(fig)

    # 3. where each model stops working
    if not band_table.empty:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        order = sorted(band_table["band"].unique())
        for m in present:
            g = band_table[band_table.model == m].set_index("band").reindex(order)
            if g["auroc"].isna().all():
                continue
            ax.plot(range(len(order)), g["auroc"], marker="o", label=LABELS[m])
        ax.axhline(0.5, color="grey", ls=":", lw=1, label="chance")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=8)
        ax.set_xlabel("maximum Tanimoto similarity to the training fold")
        ax.set_ylabel("AUROC")
        ax.set_title(f"Generalization to unfamiliar chemistry, {split} split")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{FIGDIR}/{stem}_generalization_{split}.png", dpi=150)
        plt.close(fig)
    print(f"-> {FIGDIR}/{stem}_{{comparison,calibration,generalization}}_{split}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="scaffold",
                    choices=["scaffold", "random", "scaffold_pub", "paper_lopo",
                             "paper_grouped"])
    ap.add_argument("--target", default="tg2", choices=sorted(TARGETS))
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    from scripts.run_sota import paths
    splits_path, sota_path, preds_path = paths(args.target)
    stem = "sota" if args.target == "tg2" else f"sota_{args.target}"

    d = load(args.split, sota_path, incumbents=args.target == "tg2")
    if d.empty:
        raise SystemExit(f"no results for split {args.split!r} in {sota_path}")

    summary = summarise(d)
    print(f"## {TARGETS[args.target]} potency, {args.split} split\n")
    cols = ["label", "folds"] + [m for m in METRICS if m in summary]
    print(summary[cols].to_string(index=False))

    pair = paired(d)
    if not pair.empty:
        print(f"\n## Paired against the tuned random forest (positive = entrant better)\n")
        prim = pair[pair.metric == "auroc"].sort_values("mean_delta", ascending=False)
        print(f"{'model':<18}{'ΔAUROC':>9}{'95% CI':>20}{'wins':>8}{'p':>9}{'p_holm':>9}")
        for _, r in prim.iterrows():
            print(f"{r['model']:<18}{r['mean_delta']:>+9.3f}"
                  f"{f'[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]':>20}"
                  f"{f'{int(r.wins)}/{int(r.n_seeds)}':>8}{r['p_ttest']:>9.3f}{r['p_holm']:>9.3f}")
        print("\n  secondary metrics (uncorrected, exploratory):")
        for metric in ("brier", "ece"):
            sub = pair[pair.metric == metric].sort_values("mean_delta", ascending=False)
            for _, r in sub.iterrows():
                star = " *" if r["p_ttest"] < 0.05 else ""
                print(f"    {r['model']:<18} Δ{metric} {r['mean_delta']:>+7.3f} "
                      f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]  p={r['p_ttest']:.3f}{star}")

    band_table = bands(args.split, preds_path, splits_path)
    if not band_table.empty:
        print(f"\n## AUROC by similarity to the training fold (one row per molecule)\n")
        piv = band_table.pivot(index="model", columns="band", values="auroc")
        n = band_table.groupby("band")["n_molecules"].max()
        print(piv.reindex([m for m in ORDER if m in piv.index]).round(3).to_string())
        print(f"\n  molecules per band: {n.to_dict()}")

    if "seconds" in summary:
        print(f"\n## Cost per fold\n")
        cost = summary[["label", "seconds", "peak_vram_mib", "n_params", "n_configs"]].dropna(
            subset=["seconds"])
        print(cost.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    # one file set per protocol arm: the three arms answer different questions and a shared
    # filename would leave only whichever ran last
    for name, frame in (("summary", summary), ("paired", pair), ("bands", band_table)):
        frame.assign(split=args.split).to_csv(
            f"results/{stem}_{name}_{args.split}.csv", index=False)
    print(f"\n-> results/{stem}_{{summary,paired,bands}}_{args.split}.csv")

    if args.figures:
        figures(d, pair, band_table, args.split, stem)


if __name__ == "__main__":
    main()
