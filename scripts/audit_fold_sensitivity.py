"""How much of the grouped publication result is the luck of one fold assignment?

Stage 2 control for the CDK2/MAO-B replication. The grouped protocol deals whole publications to
K folds largest-first, which fixes the order wherever publication sizes differ but leaves ties to
the seed -- and ties dominate, so re-dealing moves 35% of CDK2 and 57% of MAO-B molecules between
folds (docs/replication_feasibility.md). A result that moved with the deal would be an artefact of
one arbitrary draw.

Two things are measured per target, over several deals:

  between-deal spread   the SD of the mean AUROC across deals. This is the quantity the main
                        result's confidence interval does NOT contain, because that interval is
                        computed within a single deal.
  within-deal spread    the SD across the K folds of one deal, for comparison. If between-deal
                        spread is small against within-deal spread, one deal is enough.

Only the cheap models run. The question is about the protocol, not the model, and answering it for
random forest and logistic regression costs minutes where the pretrained entrants would cost hours.

The matched scaffold arm is run at the same training-set size for the same deals, so the
publication penalty is read off as a difference rather than against the separately published
scaffold number.

Writes results/fold_sensitivity_{target}.csv. CPU, about 20 min for CDK2 and an hour for MAO-B.

Usage: python scripts/audit_fold_sensitivity.py [--target cdk2] [--deals 5] [--k 10]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from scripts.audit_replication_feasibility import deal_papers
from scripts.export_splits import chembl_dataset, publication_subset
from src.data import chem

OUT = "results/fold_sensitivity_{target}.csv"

MODELS = {
    "rf": lambda seed: RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                              random_state=seed, n_jobs=-1),
    "logreg": lambda seed: LogisticRegression(C=1.0, max_iter=5000),
}


def auroc_over_folds(X, y, fold_of, k, model_name):
    """One AUROC per fold, for one assignment. Never pooled: folds are scored separately and the
    spread across them is the thing being measured."""
    out = []
    for f in range(k):
        te = fold_of == f
        if y[te].min() == y[te].max():
            continue
        clf = MODELS[model_name](f)
        clf.fit(X[~te], y[~te])
        p = clf.predict_proba(X[te])[:, 1]
        out.append({"fold": f, "n_test": int(te.sum()), "frac_positive": float(y[te].mean()),
                    "n_train": int((~te).sum()), "auroc": roc_auc_score(y[te], p)})
    return out


def scaffold_assignment(sub, k, seed):
    """A scaffold split of the same molecules into k test blocks, as the matched comparison.

    Murcko scaffolds are dealt to folds exactly as publications are, so the two arms differ only in
    the grouping variable -- not in fold count, fold size or training-set size.
    """
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(s) for s in sub["smiles"]]
    frame = sub.assign(paper=scaf)  # deal_papers groups on "paper"; here the group IS the scaffold
    return deal_papers(frame, k, seed=seed).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="cdk2", choices=["cdk2", "maob"])
    ap.add_argument("--deals", type=int, default=5)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--models", default="rf,logreg")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sub = publication_subset(chembl_dataset(args.target))
    X = chem.morgan(sub["smiles"].tolist())
    y = sub["y"].to_numpy()
    print(f"{args.target}: {len(sub)} molecules, {sub['paper'].nunique()} publications, "
          f"{y.mean():.1%} potent\n", flush=True)

    rows = []
    for model in args.models.split(","):
        for deal in range(args.deals):
            for arm, fold_of in (
                ("paper_grouped", deal_papers(sub, args.k, seed=deal).to_numpy()),
                ("scaffold_matched", scaffold_assignment(sub, args.k, deal)),
            ):
                got = auroc_over_folds(X, y, fold_of, args.k, model)
                for r in got:
                    rows.append({"target": args.target, "model": model, "arm": arm,
                                 "deal": deal, **r})
                m = np.mean([r["auroc"] for r in got])
                print(f"  {model:<7} {arm:<17} deal {deal}: AUROC {m:.3f} "
                      f"over {len(got)} folds", flush=True)

    out = pd.DataFrame(rows)
    path = args.out or OUT.format(target=args.target)
    out.to_csv(path, index=False)

    print("\n## between-deal vs within-deal spread\n")
    print(f"{'model':<8}{'arm':<18}{'mean':>8}{'between-deal SD':>18}{'within-deal SD':>17}")
    for (model, arm), g in out.groupby(["model", "arm"]):
        per_deal = g.groupby("deal")["auroc"].mean()
        within = g.groupby("deal")["auroc"].std().mean()
        print(f"{model:<8}{arm:<18}{per_deal.mean():>8.3f}{per_deal.std():>18.4f}"
              f"{within:>17.4f}")

    print("\n## publication penalty, matched within each deal\n")
    wide = out.groupby(["model", "arm", "deal"])["auroc"].mean().unstack("arm")
    wide["penalty"] = wide["paper_grouped"] - wide["scaffold_matched"]
    for model, g in wide.groupby(level=0):
        print(f"  {model:<8} mean {g['penalty'].mean():+.3f}  "
              f"range [{g['penalty'].min():+.3f}, {g['penalty'].max():+.3f}]  "
              f"over {len(g)} deals")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
