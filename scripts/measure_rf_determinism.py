"""Measure how far a random-forest AUROC moves between identical scorings of one fitted model.

The §8 limitation on bit-reproducibility rests on four numbers. This makes them recomputable on the
same footing as every other figure in the paper, rather than resting on a note of a probe I ran once.

With `n_jobs=-1` scikit-learn accumulates per-tree probabilities through joblib, so the thread
completion order varies and repeated `predict_proba` calls on a single fitted forest differ by about
one unit in the last place. Where a fold holds positive/negative pairs at exactly equal predicted
probability, that last bit decides the tie, and AUROC moves by 1/(n_pos*n_neg) per flipped pair.

The fold probed is the one the full reproduction gate flagged: TG2, scaffold arm, seed 15, the
configuration the committed sweep selected. Nothing here is fitted on test beyond scoring it, and no
committed result file is written.

Usage: python scripts/measure_rf_determinism.py
Output: results/rf_determinism.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_overfitting import config_for, tie_band  # noqa: E402
from run_sota import FEATURES, load_folds, paths, sklearn_grids  # noqa: E402
from src.evaluation.metrics import classification  # noqa: E402

TARGET, SPLIT, SEED, MODEL = "tg2", "scaffold", 15, "rf"
N_CALLS = 25  # fixed in advance


def main():
    splits_path, metrics_path, _ = paths(TARGET)
    committed = pd.read_csv(ROOT / metrics_path)
    ref = committed[(committed.model == MODEL) & (committed.split == SPLIT)
                    & (committed.seed == SEED)].iloc[0]

    fold = next(iter(load_folds(SPLIT, None, str(ROOT / splits_path), only=[SEED])))
    cls, feat, params = sklearn_grids()[MODEL]
    cfg = config_for(params, ref.config)
    X, y = FEATURES[feat](fold["smiles"]), fold["y"]
    tr, _va, te = fold["idx"]

    model = cls(random_state=SEED, **cfg).fit(X[tr], y[tr])
    scores, probs = [], []
    for _ in range(N_CALLS):
        p = model.predict_proba(X[te])[:, 1]
        probs.append(p)
        scores.append(classification(y[te], p)["auroc"])

    distinct = sorted(set(scores))
    n_pos, n_neg = int(np.sum(y[te] == 1)), int(np.sum(y[te] == 0))
    pairs = n_pos * n_neg
    ulp = max(float(np.abs(probs[i] - probs[0]).max()) for i in range(1, N_CALLS))
    tied = int(round(tie_band(y[te], probs[0]) * pairs))

    row = {"target": TARGET, "split": SPLIT, "seed": SEED, "model": MODEL, "calls": N_CALLS,
           "distinct_aurocs": len(distinct), "auroc_low": distinct[0], "auroc_high": distinct[-1],
           "spread": distinct[-1] - distinct[0], "n_pos": n_pos, "n_neg": n_neg,
           "pairs": pairs, "one_pair": 1 / pairs, "tied_pairs": tied,
           "max_prob_difference": ulp, "committed": float(ref.auroc)}
    out = ROOT / "results" / "rf_determinism.csv"
    pd.DataFrame([row]).to_csv(out, index=False)
    for k, v in row.items():
        print(f"  {k:22s} {v}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
