"""Write every train/valid/test fold used by the state-of-the-art comparison to one CSV.

The comparison runs models in separate virtualenvs (chemprop pins lightning, unimol_tools wants
numpy<2), so they cannot import src.data.chem. They read this file instead, which makes the folds
provably identical to the ones every existing TG2 result used rather than merely re-derived.

Four protocols:
  scaffold      631 molecules, 20 seeds. The primary evaluation, identical to every existing result.
  random        631 molecules, 20 seeds. A leakage control, never reported as generalization.
  scaffold_pub  394 molecules, 20 seeds. Arm A of the splitting-strategy comparison.
  paper_lopo    394 molecules, 11 folds. Arm B: leave one source publication out.

Arms A and B cover exactly the same molecules, so the splitting strategy is the only variable
between them.

Reproduce: python scripts/export_splits.py   ->  results/splits_tg2.csv
Verified by tests/test_sota_splits.py, which re-derives each fold and compares it row for row.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from scripts.audit_replication_feasibility import deal_papers
from src.data import chem, chembl, tg2

OUT = "results/splits_{target}.csv"
FOLDS = ("train", "valid", "test")
MIN_PER_PAPER = 10  # the criterion scripts/run_second_target.py used to call a paper fold usable
GROUPED_K = 10      # folds for paper_grouped; docs/replication_feasibility.md sets the value
RAW = "data/raw/chembl/{target}_activities.csv"


def chembl_dataset(target):
    """A replication target's labelled molecules, loaded exactly as run_second_target.py does.

    TG2 draws on two sources and needs tg2.combined; CDK2 and MAO-B are ChEMBL-only, so the loader
    is the simpler one already used for them -- not a second curation path.
    """
    raw = chembl.load_raw(RAW.format(target=target))
    lab = chembl.classification(raw)
    doc = chembl._usable(raw).groupby("smiles")["document_chembl_id"].first()
    return lab.assign(paper=lab.smiles.map(doc)).dropna(subset=["paper"]).reset_index(drop=True)


def dataset():
    """The 631-molecule combined set, with the source publication where ChEMBL records one.

    Provenance is taken from ChEMBL for every molecule ChEMBL knows, including the 173 the PubChem
    export also carries: the label still comes from tg2.combined (PubChem wins where both have a
    value), only the paper attribution is added. The PubChem export records no provenance of its
    own, so molecules ChEMBL has never seen get no paper and stay out of the publication arms.
    """
    df = tg2.combined("classification")
    raw = chembl.load_raw()
    doc = chembl._usable(raw).groupby("smiles")["document_chembl_id"].first()
    return df.assign(paper=df["smiles"].map(doc))


def publication_subset(df):
    """Molecules from publications contributing at least MIN_PER_PAPER labelled molecules.

    TG2 has only 17 attributable publications and the largest seven hold 394 of 460 molecules, so a
    random 80/10/10 group split over papers produces test folds of 1 to 45 molecules and leaves 4
    of 20 seeds with a single class. Leave-one-publication-out instead gives one fold per paper,
    tests every molecule exactly once, and needs no seed.
    """
    big = df["paper"].value_counts()
    keep = big[big >= MIN_PER_PAPER].index
    return df[df["paper"].isin(keep)].reset_index(drop=True)


def lopo(sub):
    """Leave one publication out. Valid is a scaffold slice of the remaining papers, so the held-out
    paper contributes nothing to tuning or early stopping. `seed` indexes the held-out paper."""
    papers = sorted(sub["paper"].unique())
    for i, paper in enumerate(papers):
        test = sub.index[sub["paper"] == paper].to_numpy()
        rest = sub.index[sub["paper"] != paper].to_numpy()
        inner = chem.scaffold_split(sub.loc[rest, "smiles"].tolist(), frac=(0.9, 0.1, 0.0), seed=i)
        yield i, paper, (rest[inner[0]], rest[inner[1]], test)


def grouped(sub, k, seed=0):
    """Leave K disjoint blocks of whole publications out, one block per fold.

    Valid is a scaffold slice of the remaining papers, exactly as in `lopo`, so the held-out
    publications contribute nothing to tuning or early stopping. `seed` indexes the fold, and the
    assignment draw is fixed at `seed=0` -- re-dealing is a separate sensitivity run, not a fold.
    """
    fold_of = deal_papers(sub, k, seed=seed).to_numpy()
    for f in range(k):
        test = sub.index[fold_of == f].to_numpy()
        rest = sub.index[fold_of != f].to_numpy()
        inner = chem.scaffold_split(sub.loc[rest, "smiles"].tolist(), frac=(0.9, 0.1, 0.0), seed=f)
        yield f, (rest[inner[0]], rest[inner[1]], test)


def splits(df, seeds, grouped_k=None):
    """One row per molecule per fold per protocol.

    `grouped_k` adds the paper_grouped protocol. It is off for TG2, whose 17 publications put 394
    of 460 molecules in the largest seven, so grouped folds there would be wildly uneven and
    leave-one-publication-out is both affordable and better.
    """
    sub = publication_subset(df)
    rows = []
    for kind, frame, split in (
        ("scaffold", df, lambda s: chem.scaffold_split(df["smiles"].tolist(), seed=s)),
        ("random", df, lambda s: chem.group_split(np.arange(len(df)), seed=s)),
        ("scaffold_pub", sub, lambda s: chem.scaffold_split(sub["smiles"].tolist(), seed=s)),
    ):
        for seed in seeds:
            for fold, idx in zip(FOLDS, split(seed)):
                rows.append(frame.iloc[idx].assign(split_kind=kind, seed=seed, fold=fold))

    for i, paper, idx in lopo(sub):
        for fold, part in zip(FOLDS, idx):
            rows.append(sub.loc[part].assign(split_kind="paper_lopo", seed=i, fold=fold,
                                             held_out_paper=paper))

    if grouped_k:
        for i, idx in grouped(sub, grouped_k):
            for fold, part in zip(FOLDS, idx):
                rows.append(sub.loc[part].assign(split_kind="paper_grouped", seed=i, fold=fold))

    cols = ["split_kind", "seed", "fold", "smiles", "y", "source", "paper", "held_out_paper"]
    return pd.concat(rows, ignore_index=True).reindex(columns=cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="tg2", choices=["tg2", "cdk2", "maob"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or OUT.format(target=args.target)
    df = dataset() if args.target == "tg2" else chembl_dataset(args.target)
    sub = publication_subset(df)
    out = splits(df, range(args.seeds), grouped_k=None if args.target == "tg2" else GROUPED_K)
    out.to_csv(out_path, index=False)

    print(f"{len(df)} molecules, {int(df.y.sum())} potent ({df.y.mean():.1%}); "
          f"{df['paper'].notna().sum()} attributable to {df['paper'].nunique()} publications")
    print(f"publication subset (>= {MIN_PER_PAPER} molecules/paper): {len(sub)} molecules, "
          f"{sub['paper'].nunique()} papers, {sub.y.mean():.1%} potent")
    print(f"{len(out)} rows -> {out_path}\n")
    print(f"{'protocol':<15}{'folds':>6}{'test n':>14}{'% potent':>12}{'1-class':>9}")
    for kind, g in out.groupby("split_kind", sort=False):
        t = g[g.fold == "test"].groupby("seed")
        sizes, pos, cls = t.size(), t["y"].mean(), t["y"].nunique()
        print(f"{kind:<15}{len(sizes):>6}{f'{sizes.min()}-{sizes.max()}':>14}"
              f"{f'{pos.mean():.1%}':>12}{int((cls == 1).sum()):>9}")


if __name__ == "__main__":
    main()
