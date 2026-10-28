"""Paired significance tests between two models' per-seed results on the same splits.

Usage: python scripts/compare.py results/qwen_probe_gdsc2.csv qwen_name+smiles \
                                 results/baselines_gdsc2.csv xgb --split cold_drug
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_rel

HIGHER_BETTER = {"auroc", "auprc", "mcc", "r2", "pearson", "spearman", "per_drug_spearman"}


def load(path, model, split=None, task=None):
    d = pd.read_csv(path)
    d = d[d.model == model]
    if split and "split" in d:
        d = d[d.split == split]
    if task and "task" in d:
        d = d[d.task == task]
    return d.set_index("seed")


def compare(a, b, label_a, label_b):
    shared = sorted(set(a.index) & set(b.index))
    print(f"{label_a}  vs  {label_b}   ({len(shared)} paired seeds)\n")
    print(f"{'metric':<20}{'mean Δ':>10}{'95% CI':>20}{'wins':>8}{'t-test p':>11}{'sign p':>9}")
    for m in [c for c in a.columns if c in HIGHER_BETTER or c in ("rmse", "ece", "brier")]:
        d = (a.loc[shared, m] - b.loc[shared, m]).dropna()
        if len(d) < 3 or d.std() == 0:
            continue
        if m not in HIGHER_BETTER:
            d = -d  # express every metric as "positive = better"
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        wins = int((d > 0).sum())
        p_t = ttest_rel(d, np.zeros(len(d))).pvalue
        p_s = binomtest(wins, len(d)).pvalue
        star = " *" if min(p_t, p_s) < 0.05 else ""
        print(f"{m + (' (inv)' if m not in HIGHER_BETTER else ''):<20}{d.mean():>+10.3f}"
              f"{f'[{d.mean() - ci:+.3f}, {d.mean() + ci:+.3f}]':>20}{f'{wins}/{len(d)}':>8}"
              f"{p_t:>11.3f}{p_s:>9.3f}{star}")
    print("\n'positive = better' for every row. * = p < 0.05 on either test (uncorrected).")
    print(f"Detectable at 80% power, n={len(shared)}: roughly 0.66 sd of the per-seed difference.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a"); ap.add_argument("model_a")
    ap.add_argument("file_b"); ap.add_argument("model_b")
    ap.add_argument("--split"); ap.add_argument("--task")
    a = ap.parse_args()
    compare(load(a.file_a, a.model_a, a.split, a.task), load(a.file_b, a.model_b, a.split, a.task),
            a.model_a, a.model_b)
