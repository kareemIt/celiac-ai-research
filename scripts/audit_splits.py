"""Leakage audit of the TG2 scaffold splits: exact duplicates, scaffold overlap, and how close the
nearest training analog of each test molecule is.

A scaffold split stops the same Murcko core crossing folds, but two molecules can share almost all
their structure and still have different cores. Tanimoto similarity to the nearest training molecule
is the honest measure of how novel the test set really is.

Usage: python scripts/audit_splits.py [--seeds 10] [--data combined]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.data import chem, tg2

OUT = "results/split_audit.csv"
NEAR_DUPLICATE = 0.7  # Tanimoto above which two molecules are usually the same analog series


def tanimoto_to_train(fp_test, fp_train):
    """Max Tanimoto from each test molecule to any training molecule, on the Morgan bit vectors."""
    a = fp_test.astype(np.float32)
    b = fp_train.astype(np.float32)
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return np.max(inter / np.maximum(union, 1), axis=1)


def main(seeds, data):
    table = tg2.combined("classification") if data == "combined" else tg2.classification(tg2.load_raw())
    smiles = table["smiles"].tolist()
    y = table["y"].to_numpy()
    fp = chem.morgan(smiles)
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(s) for s in smiles]

    print(f"{len(smiles)} molecules, {len(set(smiles))} unique SMILES, {len(set(scaf))} Murcko scaffolds")
    counts = pd.Series(scaf).value_counts()
    print(f"scaffolds appearing once: {(counts == 1).sum()} ({(counts == 1).mean():.0%} of scaffolds, "
          f"{(counts == 1).sum() / len(smiles):.0%} of molecules)")
    print(f"largest scaffold group: {counts.iloc[0]} molecules")
    assert len(set(smiles)) == len(smiles), "duplicate SMILES in the table"

    rows = []
    for kind in ["scaffold", "random"]:
        for seed in range(seeds):
            tr, va, te = (chem.scaffold_split(smiles, seed=seed) if kind == "scaffold"
                          else chem.group_split(np.arange(len(smiles)), seed=seed))
            assert not (set(tr) & set(te) or set(tr) & set(va) or set(va) & set(te)), "index overlap"
            s_tr, s_te = {scaf[i] for i in tr}, {scaf[i] for i in te}
            sim = tanimoto_to_train(fp[te], fp[tr])
            rows.append({
                "split": kind, "seed": seed, "n_train": len(tr), "n_valid": len(va), "n_test": len(te),
                "test_pos_frac": y[te].mean(),
                "shared_scaffolds": len(s_tr & s_te),
                "max_tanimoto_mean": sim.mean(),
                "max_tanimoto_median": np.median(sim),
                "frac_test_near_duplicate": (sim >= NEAR_DUPLICATE).mean(),
                "frac_test_identical_fp": (sim >= 0.999).mean(),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    print(f"\n## Split audit ({data}, {seeds} seeds, mean ± sd)")
    agg = out.drop(columns="seed").groupby("split").agg(["mean", "std"])
    for col in ["n_train", "n_test", "test_pos_frac", "shared_scaffolds", "max_tanimoto_mean",
                "frac_test_near_duplicate", "frac_test_identical_fp"]:
        line = "  ".join(f"{k}: {agg.loc[k, (col, 'mean')]:.3f} ± {agg.loc[k, (col, 'std')]:.3f}"
                         for k in ["scaffold", "random"])
        print(f"  {col:26s} {line}")

    sc = out[out.split == "scaffold"]
    assert sc.shared_scaffolds.eq(0).all(), "scaffold split leaked a scaffold into test"
    print(f"\nNo Murcko scaffold is shared between train and test in any of the {seeds} scaffold seeds.")
    print(f"Still, {sc.frac_test_near_duplicate.mean():.1%} of scaffold-split test molecules have a "
          f"training molecule within Tanimoto {NEAR_DUPLICATE} "
          f"(random split: {out[out.split == 'random'].frac_test_near_duplicate.mean():.1%}).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--data", default="combined", choices=["pubchem", "combined"])
    a = ap.parse_args()
    main(a.seeds, a.data)
