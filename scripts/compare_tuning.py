"""Paired tuned-vs-default comparison from the saved CSVs (no refitting)."""
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

tuned = pd.read_csv("results/tuned_baselines_tg2.csv")
base = pd.read_csv("results/baselines_tg2_combined.csv")
base = base[base.task == "classification"]

print("## Tuned vs default fingerprint baselines, combined TG2, 10 scaffold seeds (paired)")
for name in ["logreg", "rf", "xgb"]:
    a = tuned[tuned.model == f"{name}_tuned"].set_index("seed")
    b = base[base.model == name].set_index("seed")
    parts = []
    for metric in ["auroc", "auprc", "mcc", "brier", "ece"]:
        d = (a[metric] - b[metric]).dropna()
        p = ttest_rel(d, np.zeros(len(d))).pvalue if d.std() > 0 else 1.0
        wins = int((d > 0).sum()) if metric not in ("brier", "ece") else int((d < 0).sum())
        parts.append(f"Δ{metric} {d.mean():+.3f} (p={p:.3f}, {wins}/{len(d)})")
    print(f"  {name:8s} " + "  ".join(parts))

print("\n## Best model per metric (all fingerprint variants, combined, 10 seeds)")
allm = pd.concat([base.assign(src="default"), tuned.assign(src="tuned")])
print(allm.groupby("model")[["auroc", "auprc", "mcc", "ece", "brier"]].mean().round(3).sort_values("auroc").to_string())
