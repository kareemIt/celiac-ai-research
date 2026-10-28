"""Post-hoc calibration of the fine-tuned Jev QLoRA model, from the log-odds it saved per seed.

Uses the same validation fold the training run selected its epoch on. That fold is therefore not
fully untouched — it chose the epoch — but it never contributed a gradient, and the alternative
(a fourth fold) would cost another 60 molecules from an already small set. Noted as a caveat.

Usage: python scripts/calibrate_qlora.py [--data combined]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from scripts.run_baselines import summarize
from src.evaluation import calibration, metrics

OUT = "results/calibration_qlora.csv"


def main(data):
    preds = pd.read_csv(f"results/jev_qlora_preds_{data}.csv")
    rows = []
    for seed, g in preds.groupby("seed"):
        va, te = g[g.fold == "valid"], g[g.fold == "test"]
        for kind in calibration.KINDS:
            p = calibration.fit(kind, va.logodds.to_numpy(), va.y.to_numpy())(te.logodds.to_numpy())
            rows.append({"model": "jev_qlora", "calibration": kind, "seed": int(seed),
                         "temperature": calibration.fit_temperature(va.logodds.to_numpy(), va.y.to_numpy()),
                         **metrics.classification(te.y.to_numpy(), p)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"## Jev QLoRA calibration ({data}, {out.seed.nunique()} seeds)\n"
          f"{summarize(out.drop(columns=['temperature', 'model']), ['calibration']).to_string()}")
    t = out[out.calibration == 'none']['temperature']
    print(f"\nfitted temperature: {t.mean():.2f} ± {t.std():.2f} (range {t.min():.2f}–{t.max():.2f})")

    base = out[out.calibration == "none"].set_index("seed")
    print("\n## Change from uncalibrated, paired by seed")
    for kind in calibration.KINDS[1:]:
        m = out[out.calibration == kind].set_index("seed")
        parts = []
        for metric in ["ece", "brier", "logloss"]:
            d = (m[metric] - base[metric]).dropna()
            p_val = ttest_rel(d, np.zeros(len(d))).pvalue if d.std() > 0 else 1.0
            parts.append(f"Δ{metric} {d.mean():+.3f} (p={p_val:.3f})")
        print(f"  {kind:12s} " + "  ".join(parts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="combined")
    main(ap.parse_args().data)
