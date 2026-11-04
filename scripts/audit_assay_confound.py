"""Can an assay-held-out split separate "who measured it" from "how it was measured"?

Stage 2 control for the CDK2/MAO-B replication. docs/second_target.md shows that holding out whole
publications costs more AUROC than holding out scaffolds, and attributes that to publication-level
confounding. But a publication-held-out split holds out two things at once: the laboratory and its
assay protocol. If the decline were really an assay effect, the recommendation would change -- you
would report an assay-held-out split, not a publication-held-out one.

This script tests whether that control can be built at all, before any model is fitted. It asks
three questions:

  1. Are assays nested inside publications? If every assay belongs to exactly one publication, an
     assay-held-out split IS a publication-held-out split and the two cannot be told apart.
  2. Is there a within-publication assay contrast? Publications running two or more assays would
     let the assay effect be estimated with the laboratory held fixed.
  3. How much assay overlap does each split actually leave? This is measurable whatever the answer
     to 1 and 2, and it says what the publication split is doing that the scaffold split is not.

Writes results/assay_confound.csv. CPU, under a minute. Fits nothing.

Usage: python scripts/audit_assay_confound.py [--targets cdk2 maob] [--min-per-assay 5]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.data import chembl

OUT = "results/assay_confound.csv"
RAW = "data/raw/chembl/{target}_activities.csv"
SPLITS = "results/splits_{target}.csv"


def load(target):
    """Labelled molecules with both the publication and the assay each is attributed to.

    Attribution is first-record, exactly as the publication attribution elsewhere in the project,
    so the two groupings are built the same way and their overlap is not an artefact of method.
    """
    raw = chembl.load_raw(RAW.format(target=target))
    usable = chembl._usable(raw)
    lab = chembl.classification(raw)
    assay = usable.groupby("smiles")["assay_chembl_id"].first()
    paper = usable.groupby("smiles")["document_chembl_id"].first()
    desc = usable.groupby("smiles")["assay_description"].first()
    return lab.assign(assay=lab.smiles.map(assay), paper=lab.smiles.map(paper),
                      description=lab.smiles.map(desc)).dropna(
                          subset=["assay", "paper"]).reset_index(drop=True)


def nesting(m):
    """Is the assay grouping a refinement of the publication grouping, or crossed with it?"""
    papers_per_assay = m.groupby("assay")["paper"].nunique()
    assays_per_paper = m.groupby("paper")["assay"].nunique()
    return {
        "molecules": len(m),
        "assays": m["assay"].nunique(),
        "papers": m["paper"].nunique(),
        "assays_spanning_multiple_papers": int((papers_per_assay > 1).sum()),
        "papers_with_multiple_assays": int((assays_per_paper > 1).sum()),
        "molecules_in_multi_assay_papers": int(m["paper"].isin(
            assays_per_paper[assays_per_paper > 1].index).sum()),
    }


def within_paper_contrast(m, min_per_assay):
    """Publications with two or more assays each carrying `min_per_assay` molecules.

    This is the only design that estimates an assay effect free of the laboratory, so its size is
    the ceiling on what any assay control here could ever measure.
    """
    counts = m.groupby(["paper", "assay"]).size()
    big = counts[counts >= min_per_assay]
    per_paper = big.groupby(level=0).size()
    usable_papers = per_paper[per_paper >= 2].index
    sub = m[m["paper"].isin(usable_papers)]
    return {
        "papers_with_2plus_assays_of_min_size": int(len(usable_papers)),
        "molecules_in_those_papers": int(len(sub)),
        "min_per_assay": min_per_assay,
    }


def assay_overlap(target, m):
    """Per split, the share of test molecules whose assay appears nowhere in training.

    Measurable regardless of identifiability, and it is the quantity that says what a publication
    split removes beyond scaffolds: if a scaffold split leaves most test assays represented in
    training and a publication split leaves none, the publication split is also an assay split.
    """
    path = SPLITS.format(target=target)
    if not os.path.exists(path):
        return pd.DataFrame()
    sp = pd.read_csv(path, low_memory=False)
    by_smiles = m.set_index("smiles")["assay"]
    rows = []
    for (kind, seed), g in sp.groupby(["split_kind", "seed"]):
        tr = g[g.fold.isin(["train", "valid"])]["smiles"].map(by_smiles).dropna()
        te = g[g.fold == "test"]["smiles"].map(by_smiles).dropna()
        if te.empty:
            continue
        unseen = ~te.isin(set(tr))
        rows.append({"split_kind": kind, "seed": seed, "n_test": len(te),
                     "frac_test_assay_unseen": unseen.mean(),
                     "test_assays": te.nunique(),
                     "test_assays_unseen": te[unseen.to_numpy()].nunique()})
    d = pd.DataFrame(rows)
    return d.groupby("split_kind").agg(
        folds=("seed", "size"), mean_n_test=("n_test", "mean"),
        frac_test_assay_unseen=("frac_test_assay_unseen", "mean"),
        mean_test_assays=("test_assays", "mean"),
        mean_test_assays_unseen=("test_assays_unseen", "mean")).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["cdk2", "maob"])
    ap.add_argument("--min-per-assay", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    rows = []
    for target in args.targets:
        m = load(target)
        facts = {"target": target, **nesting(m),
                 **within_paper_contrast(m, args.min_per_assay)}
        rows.append(facts)

        print(f"\n=== {target}")
        print(f"  {facts['molecules']} molecules, {facts['assays']} assays, "
              f"{facts['papers']} publications")
        print(f"  assays spanning more than one publication: "
              f"{facts['assays_spanning_multiple_papers']}")
        print(f"  publications running more than one assay : "
              f"{facts['papers_with_multiple_assays']} "
              f"({facts['molecules_in_multi_assay_papers']} molecules)")
        print(f"  publications with 2+ assays of >= {args.min_per_assay} molecules: "
              f"{facts['papers_with_2plus_assays_of_min_size']} "
              f"({facts['molecules_in_those_papers']} molecules)")

        overlap = assay_overlap(target, m)
        if not overlap.empty:
            print("\n  share of test molecules whose assay is absent from training:")
            print(overlap.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
