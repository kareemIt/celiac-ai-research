"""Tabular foundation model entrant, run inside the tabpfn virtualenv by scripts/run_sota.py.

It consumes the features the parent computed with src.data.chem, so the fingerprints and
descriptors here are byte-identical to the ones the fingerprint baselines were scored on.

A TabPFN-class model does no task-specific training: the training fold is passed as context and
prediction is in-context inference. Its only real knob here is which feature block to condition on,
so the grid is 3 configurations rather than the 12 the other entrants get. That is recorded in the
results rather than padded out with duplicates, and it favours the other entrants if anything.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def valid_score(y, p):
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else 0.5
    return float(auc), float(brier_score_loss(y, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, default=12)
    args = ap.parse_args()

    import torch
    from tabpfn import TabPFNClassifier

    f = np.load(f"{args.work}/features.npz")
    tr, va, te = f["train"], f["valid"], f["test"]
    # each CSV holds that fold's rows; the index arrays say where they sit in the feature matrix
    y = np.empty(len(tr) + len(va) + len(te), dtype=int)
    for part, idx in (("train", tr), ("valid", va), ("test", te)):
        y[idx] = pd.read_csv(f"{args.work}/{part}.csv")["y"].to_numpy()

    # TabPFN is documented to support at most 500 features, and the 2048-bit Morgan fingerprint
    # exceeds that. Rather than override the limit, the fingerprint is reduced by PCA fitted on the
    # training fold only -- the same treatment this project already applies to GDSC2 expression
    # features. The compact RDKit descriptor block is used raw, as in arXiv:2604.16123.
    from sklearn.decomposition import PCA
    fp, desc = np.asarray(f["morgan"], dtype=np.float32), np.asarray(f["rdkit2d"], np.float32)
    n_comp = min(256, len(tr) - 1, fp.shape[1])
    pca = PCA(n_components=n_comp, random_state=args.seed).fit(fp[tr])
    fp_small = pca.transform(fp).astype(np.float32)
    blocks = {"rdkit2d": desc, f"morgan_pca{n_comp}": fp_small,
              f"rdkit2d+morgan_pca{n_comp}": np.hstack([desc, fp_small])}
    assert all(b.shape[1] <= 500 for b in blocks.values()), "a block exceeds TabPFN's limit"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best = None
    for feat, X in blocks.items():
        X = np.asarray(X, dtype=np.float32)
        m = TabPFNClassifier(random_state=args.seed, device=dev)
        m.fit(X[tr], y[tr])
        score = valid_score(y[va], m.predict_proba(X[va])[:, 1])
        if best is None or score[0] > best[0][0]:
            best = (score, feat, m.predict_proba(X[te])[:, 1])

    (va_auc, va_brier), feat, p_test = best
    pd.DataFrame({"p": p_test}).to_csv(f"{args.work}/pred.csv", index=False)
    pd.DataFrame([{
        "valid_auroc": va_auc, "valid_brier": va_brier, "config": f"features={feat}",
        "n_configs": len(blocks), "features": feat, "n_params": np.nan,
        "peak_vram_mib": (torch.cuda.max_memory_allocated() / 2**20) if dev == "cuda" else np.nan,
    }]).to_csv(f"{args.work}/info.csv", index=False)


if __name__ == "__main__":
    main()
