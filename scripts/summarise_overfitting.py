"""Summarise the tier-1 overfitting decomposition into the cells §5.21 reports.

run_overfitting.py writes one row per outer fold. This reduces those to one row per
target/split/model, plus the two pooled rows the design pre-registered as fallbacks, with
percentile bootstrap intervals over folds. The bootstrap is seeded so the intervals quoted in the
paper are recomputable; the means are exact and need no seed.

`decides` is the pre-registered control check: an effect is only readable when the learning-curve
term is small beside it. It holds when the shift interval excludes zero and |size_effect| is below
a third of shift_gap -- a third rather than a half so that a control one-third the size of the
effect is still reported as undecided rather than quietly accepted.

Usage: python scripts/summarise_overfitting.py
Output: results/overfitting_tier1_cells.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GAPS = ["overfit_gap", "shift_gap", "size_effect"]
BOOT, SEED = 4000, 0


def summarise(g, rng):
    row = {"folds": len(g)}
    for c in GAPS:
        x = g[c].to_numpy(float)
        row[c] = float(x.mean())
        lo, hi = np.percentile(rng.choice(x, (BOOT, len(x)), replace=True).mean(1), [2.5, 97.5])
        row[c + "_lo"], row[c + "_hi"] = float(lo), float(hi)
    shift, ctl = row["shift_gap"], abs(row["size_effect"])
    row["decides_shift"] = bool(row["shift_gap_lo"] > 0 and ctl < shift / 3)
    return row


def main():
    # Folds with one class in the test set have no AUROC, so no committed score and no control.
    d = pd.read_csv(ROOT / "results" / "overfitting_tier1.csv").dropna(subset=["test_small"])
    rng = np.random.default_rng(SEED)

    rows = []
    for (target, split, model), g in d.groupby(["target", "split", "model"], sort=True):
        rows.append({"target": target, "split": split, "model": model, **summarise(g, rng)})
    # The two pooled rows the proposal fixed in advance: TG2's publication arm has too few
    # publications for a per-model statement, and CDK2+MAO-B are the arms that resolve.
    pooled = {"tg2 paper_lopo pooled": (d.target == "tg2") & (d.split == "paper_lopo"),
              "cdk2+maob paper_lopo pooled": d.target.isin(["cdk2", "maob"]) & (d.split == "paper_lopo")}
    for name, mask in pooled.items():
        rows.append({"target": name, "split": "pooled", "model": "all", **summarise(d[mask], rng)})

    out = pd.DataFrame(rows)
    path = ROOT / "results" / "overfitting_tier1_cells.csv"
    out.to_csv(path, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(out.round(3).to_string(index=False))
    print(f"\nwrote {path}")
    print(f"cells where shift is readable against its control: {int(out.decides_shift.sum())}"
          f" of {len(out)}")


if __name__ == "__main__":
    main()
