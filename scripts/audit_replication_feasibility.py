"""Can the full TG2 model sweep and conformal experiment actually be run on CDK2 and MAO-B?

Phase 0 of the three-target replication. docs/second_target.md already evaluated CDK2 and MAO-B
under random / scaffold / paper splits, but with two models only (random forest and logistic
regression). Extending that to the nine-entrant field of docs/sota_comparison.md and to the
publication-aware conformal experiment of docs/lopo_uncertainty.md needs GPU time, and the two
replication targets are 4x and 12x TG2's size. This script decides whether that is affordable
before any of it is launched.

It answers three questions and nothing else:

  1. How many publication folds does each target have, and how many are usable? A leave-one-
     publication-out fold is wasted if it is single-class, and a grouped K-fold alternative
     changes both the fold count and the per-fold training size.
  2. What would the sweep cost? Measured per-fold seconds from results/sota_tg2.csv are scaled by
     the ratio of training-set sizes, per model, so the projection comes from this project's own
     timings on this project's own GPU rather than from a guess.
  3. Does the publication-aware conformal design still fit? Group calibration carves whole
     training publications out until 25% of the training molecules are held; with few papers per
     fold that is not possible, which is a feasibility limit rather than a result.

Writes results/replication_feasibility.csv (one row per target per protocol) and prints the
projection table. CPU, about a minute. Fits nothing and trains nothing.

Usage: python scripts/audit_replication_feasibility.py [--targets cdk2 maob] [--grouped-folds 10]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.data import chem, chembl

OUT = "results/replication_feasibility.csv"
RAW = "data/raw/chembl/{name}_activities.csv"
TIMINGS = "results/sota_tg2.csv"
MIN_PER_PAPER = 10          # the criterion export_splits.py uses to call a paper fold usable
CAL_FRAC = 0.25             # run_lopo_conformal.py's calibration fraction
SWEEP = ["logreg", "rf", "xgb", "chemprop", "chemprop_rdkit", "chemeleon",
         "chemberta_frozen", "chemberta_ft"]


def load(name):
    """Labelled molecules with the publication each is attributed to, as run_second_target.py does."""
    raw = chembl.load_raw(RAW.format(name=name))
    lab = chembl.classification(raw)
    doc = chembl._usable(raw).groupby("smiles")["document_chembl_id"].first()
    lab = lab.assign(paper=lab.smiles.map(doc))
    return lab.dropna(subset=["paper"]).reset_index(drop=True)


def publication_subset(df):
    big = df["paper"].value_counts()
    return df[df["paper"].isin(big[big >= MIN_PER_PAPER].index)].reset_index(drop=True)


def lopo_folds(sub):
    """One fold per publication: (n_train, n_test, test is single-class, n training papers)."""
    for paper in sorted(sub["paper"].unique()):
        test = sub[sub["paper"] == paper]
        rest = sub[sub["paper"] != paper]
        yield (len(rest), len(test), test.y.nunique() == 1, rest["paper"].nunique())


def deal_papers(sub, k, seed=0):
    """Assign each molecule a fold by dealing whole publications largest-first to the lightest fold.

    Returns one fold index per row of `sub`. Labels are never consulted, so the assignment cannot
    be tuned to flatter a model. Dealing largest-first keeps folds close in size, which matters
    because a fold holding most of the molecules would dominate a pooled estimate.
    """
    sizes = sub["paper"].value_counts()
    rng = np.random.default_rng(seed)
    order = sizes.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).sort_values(
        ascending=False)
    assign, load_ = {}, np.zeros(k)
    for paper, n in order.items():
        f = int(np.argmin(load_))
        assign[paper] = f
        load_[f] += n
    return sub["paper"].map(assign)


def grouped_folds(sub, k, seed=0):
    """K folds, each holding out a disjoint block of whole publications.

    Every molecule is tested exactly once and no two folds share a test molecule, so unlike
    leave-one-publication-out there is no fold overlap to narrow a bootstrap interval -- the
    pooling artefact of docs/second_target.md cannot arise.
    """
    fold = deal_papers(sub, k, seed)
    for f in range(k):
        test, rest = sub[fold == f], sub[fold != f]
        yield (len(rest), len(test), test.y.nunique() == 1, rest["paper"].nunique())


def per_fold_seconds(arm="paper_lopo"):
    """Measured TG2 seconds per fold per model, with that arm's training size.

    Each arm is scaled from its own TG2 timings rather than from one global number, because the
    publication and scaffold arms differ in training size and in how often early stopping fires.
    """
    d = pd.read_csv(TIMINGS)
    d = d[d.split == arm]
    return d.groupby("model").seconds.mean(), d.n_train.mean()


def project(n_train, tg2_seconds, tg2_train, folds):
    """Hours for one protocol, scaling each model's measured per-fold cost by training size.

    Linear in training molecules. That is right for the gradient-trained entrants, where cost is
    epochs x batches, and conservative-to-optimistic for the tree models, whose fit is closer to
    n log n. It is an order-of-magnitude projection and is reported as one.
    """
    scale = n_train / tg2_train
    per_fold = sum(tg2_seconds.get(m, np.nan) for m in SWEEP) * scale
    return per_fold * folds / 3600.0


def scaffold_folds(sub, seeds):
    """Arm A: a scaffold split of the SAME molecules the publication arms use, one fold per seed.

    Matching the molecule set is what makes the two arms comparable -- the split is then the only
    variable, which is the design docs/sota_comparison.md used for TG2 (scaffold_pub vs paper_lopo).
    """
    smiles = sub["smiles"].tolist()
    for seed in range(seeds):
        tr, va, te = chem.scaffold_split(smiles, seed=seed)
        test, rest = sub.iloc[te], sub.iloc[np.concatenate([tr, va])]
        yield (len(rest), len(test), test.y.nunique() == 1, rest["paper"].nunique())


def audit(name, k, seeds, timings):
    df = load(name)
    sub = publication_subset(df)
    rows = []

    protocols = [
        ("paper_lopo", list(lopo_folds(sub)), "paper_lopo"),
        (f"paper_grouped_k{k}", list(grouped_folds(sub, k)), "paper_lopo"),
        (f"scaffold_pub_{seeds}seeds", list(scaffold_folds(sub, seeds)), "scaffold_pub"),
    ]
    for protocol, folds, arm in protocols:
        tg2_seconds, tg2_train = timings[arm]
        n_train = np.mean([f[0] for f in folds])
        test_n = [f[1] for f in folds]
        single = sum(f[2] for f in folds)
        cal_papers = np.mean([f[3] for f in folds])
        rows.append(dict(
            target=name, protocol=protocol, molecules=len(sub), papers=sub["paper"].nunique(),
            frac_positive=sub.y.mean(), folds=len(folds), usable_folds=len(folds) - single,
            single_class_folds=single, mean_n_train=round(n_train),
            test_min=min(test_n), test_max=max(test_n), mean_train_papers=round(cal_papers, 1),
            # group calibration needs enough whole training papers to reach CAL_FRAC
            cal_papers_needed=round(cal_papers * CAL_FRAC, 1),
            sweep_hours=round(project(n_train, tg2_seconds, tg2_train, len(folds) - single), 1),
        ))
    return pd.DataFrame(rows), sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["cdk2", "maob"])
    ap.add_argument("--grouped-folds", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=10,
                    help="seeds for the matched scaffold arm (TG2 published 20)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    timings = {arm: per_fold_seconds(arm) for arm in ("paper_lopo", "scaffold_pub")}
    for arm, (sec, train) in timings.items():
        print(f"TG2 reference ({arm}): {train:.0f} training molecules, "
              f"{sum(sec.get(m, np.nan) for m in SWEEP):.0f} s per fold for the "
              f"{len(SWEEP)}-entrant sweep")
    print()

    out = []
    for name in args.targets:
        frame, sub = audit(name, args.grouped_folds, args.seeds, timings)
        out.append(frame)
        print(f"{name}: {len(sub)} molecules in papers with >= {MIN_PER_PAPER}, "
              f"{sub['paper'].nunique()} papers, {sub.y.mean():.1%} potent")

    out = pd.concat(out, ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)

    cols = ["target", "protocol", "folds", "usable_folds", "single_class_folds", "mean_n_train",
            "test_min", "test_max", "mean_train_papers", "sweep_hours"]
    print()
    print(out[cols].to_string(index=False))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
