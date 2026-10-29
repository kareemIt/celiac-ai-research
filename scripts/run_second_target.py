"""Do the TG2 evaluation failures reproduce on an independent target?

Phase 2 of the second-target work. CDK2 (CHEMBL301) was selected in
scripts/audit_candidate_targets.py: 53% positive, 0.448 scaffolds per molecule, 84 publications
with at least 10 labelled molecules, and chemistry unlike TG2 (median Tanimoto 0.220).

The same molecules and the same models are evaluated under three splits, so any difference is the
split and nothing else:

  random       the protocol TG2 showed to leak (38.6% of its test molecules were fingerprint
               -identical to a training molecule)
  scaffold     Bemis-Murcko, this project's standard
  paper        whole source publications held out -- the control that collapsed TG2 selectivity
               from AUROC 0.974 to 0.574

Predicted in advance (docs/second_target.md, and the commit that selected CDK2): the paper split
should cost LITTLE here, because CDK2 spreads its positives across many publications (largest holds
9% against 38% for the TG2 selectivity set). A small gap is a real finding about how specific the
TG2 result was, not a failed replication.

Everything is scored one row per molecule with a bootstrap CI over molecules, because folds overlap
across seeds and pooled rows are not independent -- the artefact that withdrew a TG2 hypothesis.

Writes results/second_target_splits.csv and results/second_target_leakage.csv. CPU, a few minutes.

Usage: python scripts/run_second_target.py [--target cdk2] [--seeds 10] [--reps 2000]
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

from scripts.audit_splits import tanimoto_to_train
from src.data import chem, chembl
from src.evaluation import metrics

SPLITS_OUT = "results/second_target_splits.csv"
LEAK_OUT = "results/second_target_leakage.csv"
NOVEL_OUT = "results/second_target_novelty.csv"
RAW = "data/raw/chembl/{name}_activities.csv"
NEAR_DUPLICATE = 0.7

MODELS = {
    "rf": lambda seed: RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                              random_state=seed, n_jobs=-1),
    "logreg": lambda seed: LogisticRegression(C=1.0, max_iter=5000),
}


def load(name):
    """Labelled molecules with the publication each one is attributed to."""
    raw = chembl.load_raw(RAW.format(name=name))
    lab = chembl.classification(raw)
    doc = chembl._usable(raw).groupby("smiles")["document_chembl_id"].first()
    lab = lab.assign(paper=lab.smiles.map(doc))
    return lab.dropna(subset=["paper"]).reset_index(drop=True)


def splits_for(kind, smiles, papers, seed):
    if kind == "random":
        return chem.group_split(np.arange(len(smiles)), seed=seed)  # every molecule its own group
    if kind == "scaffold":
        return chem.scaffold_split(smiles, seed=seed)
    return chem.group_split(papers, seed=seed)


def auroc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan


def leakage(kind, smiles, fp, papers, scaf, seeds):
    """What a test fold actually looks like under each protocol."""
    rows = []
    for seed in range(seeds):
        tr, va, te = splits_for(kind, smiles, papers, seed)
        if not len(te):
            continue
        sim = tanimoto_to_train(fp[te], fp[tr])
        rows.append({
            "shared_scaffolds": len(set(scaf[te]) & set(scaf[tr])),
            "shared_papers": len(set(papers[te]) & set(papers[tr])),
            "mean_max_tanimoto": sim.mean(),
            "frac_identical": float((sim >= 0.999).mean()),
            "frac_near_duplicate": float((sim >= NEAR_DUPLICATE).mean()),
        })
    d = pd.DataFrame(rows).mean()
    d["split"] = kind
    return d


def evaluate(kind, name, smiles, fp, y, papers, seeds, reps, rng):
    """Pooled one-row-per-molecule AUROC with a bootstrap CI, plus per-seed spread."""
    preds = []
    for seed in range(seeds):
        tr, va, te = splits_for(kind, smiles, papers, seed)
        tr = np.concatenate([tr, va])
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = MODELS[name](seed).fit(fp[tr], y[tr])
        p = m.predict_proba(fp[te])[:, 1]
        preds.append(pd.DataFrame({"seed": seed, "smiles": [smiles[i] for i in te],
                                   "y": y[te], "p": p}))
    pred = pd.concat(preds, ignore_index=True)
    per_seed = np.array([auroc(g.y.to_numpy(), g.p.to_numpy()) for _, g in pred.groupby("seed")])
    per_seed = per_seed[np.isfinite(per_seed)]

    mol = pred.groupby("smiles").agg(y=("y", "first"), p=("p", "mean"))
    yy, pp = mol.y.to_numpy(), mol.p.to_numpy()
    a = auroc(yy, pp)
    b = [auroc(yy[i], pp[i]) for i in (rng.integers(0, len(yy), len(yy)) for _ in range(reps))]
    b = [v for v in b if np.isfinite(v)]
    m = metrics.classification(yy, pp)
    return {"split": kind, "model": name, "n_mols": len(mol), "auroc": a,
            "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5),
            "per_seed_mean": per_seed.mean(), "per_seed_sd": per_seed.std(ddof=1),
            "auprc": m["auprc"], "ece": m["ece"], "brier": m["brier"],
            "pooled_rows": len(pred)}


BINS = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]


def novelty_curve(smiles, fp, y, papers, seeds):
    """AUROC against distance from the training set, under the paper split -- the honest protocol.

    Each test molecule is binned by its own max Tanimoto to that fold's training set, so a bin
    means the same thing across folds. Scored one row per molecule within each bin."""
    rows = []
    for seed in range(seeds):
        tr, va, te = splits_for("paper", smiles, papers, seed)
        tr = np.concatenate([tr, va])
        if len(np.unique(y[tr])) < 2:
            continue
        m = MODELS["rf"](seed).fit(fp[tr], y[tr])
        rows.append(pd.DataFrame({
            "smiles": [smiles[i] for i in te], "y": y[te],
            "p": m.predict_proba(fp[te])[:, 1],
            "sim": tanimoto_to_train(fp[te], fp[tr])}))
    pred = pd.concat(rows, ignore_index=True)
    out = []
    for lo, hi in BINS:
        g = pred[(pred.sim >= lo) & (pred.sim < hi)]
        if len(g) < 20:
            continue
        mol = g.groupby("smiles").agg(y=("y", "first"), p=("p", "mean"))
        out.append({"tanimoto_bin": f"{lo:.1f}-{hi:.1f}" if hi <= 1 else f">{lo:.1f}",
                    "molecules": len(mol), "frac_positive": float(mol.y.mean()),
                    "auroc": auroc(mol.y.to_numpy(), mol.p.to_numpy())})
    return pd.DataFrame(out)


def pooled_vs_molecule(smiles, fp, y, papers, seeds, reps, rng):
    """TG2's withdrawn hypothesis came from bootstrapping pooled fold rows instead of molecules.
    This measures how much that understates the interval here."""
    rows = []
    for seed in range(seeds):
        tr, va, te = splits_for("paper", smiles, papers, seed)
        tr = np.concatenate([tr, va])
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = MODELS["rf"](seed).fit(fp[tr], y[tr])
        rows.append(pd.DataFrame({"smiles": [smiles[i] for i in te], "y": y[te],
                                  "p": m.predict_proba(fp[te])[:, 1]}))
    pred = pd.concat(rows, ignore_index=True)
    mol = pred.groupby("smiles").agg(y=("y", "first"), p=("p", "mean"))

    def ci(yy, pp):
        b = [auroc(yy[i], pp[i]) for i in (rng.integers(0, len(yy), len(yy)) for _ in range(reps))]
        b = [v for v in b if np.isfinite(v)]
        return np.percentile(b, 2.5), np.percentile(b, 97.5)

    plo, phi = ci(pred.y.to_numpy(), pred.p.to_numpy())
    mlo, mhi = ci(mol.y.to_numpy(), mol.p.to_numpy())
    dup = len(pred) / len(mol)
    print(f"  pooled over {len(pred)} fold rows:      AUROC {auroc(pred.y.to_numpy(), pred.p.to_numpy()):.3f}"
          f"  95% CI [{plo:.3f}, {phi:.3f}]  width {phi - plo:.3f}")
    print(f"  one row per molecule ({len(mol)}):     AUROC {auroc(mol.y.to_numpy(), mol.p.to_numpy()):.3f}"
          f"  95% CI [{mlo:.3f}, {mhi:.3f}]  width {mhi - mlo:.3f}")
    print(f"\n  Each molecule appears {dup:.1f}x on average. Pooling narrows the interval by "
          f"{1 - (phi - plo) / (mhi - mlo):.0%},")
    print("  which is the mechanism that made a TG2 difference look real. The point estimate "
          "barely moves;\n  it is the interval, and therefore the significance, that is wrong.")


def main(target, seeds, reps):
    rng = np.random.default_rng(0)
    lab = load(target)
    smiles = lab.smiles.tolist()
    y = lab.y.to_numpy()
    papers = lab.paper.to_numpy()
    scaf = np.array([MurckoScaffold.MurckoScaffoldSmiles(s) for s in smiles])
    fp = chem.morgan(smiles)

    print(f"## {target}: {len(lab)} molecules, {int(y.sum())} potent ({y.mean():.0%}), "
          f"{len(set(scaf))} scaffolds, {lab.paper.nunique()} publications\n")

    print("## 1. What each split actually holds out\n")
    leak = pd.DataFrame([leakage(k, smiles, fp, papers, scaf, seeds)
                         for k in ("random", "scaffold", "paper")])
    leak = leak[["split", "shared_scaffolds", "shared_papers", "mean_max_tanimoto",
                 "frac_identical", "frac_near_duplicate"]]
    print(leak.to_string(index=False, float_format="%.3f"))
    leak.to_csv(LEAK_OUT, index=False)

    print("\n## 2. The same molecules and models under each split\n")
    rows = [evaluate(k, m, smiles, fp, y, papers, seeds, reps, rng)
            for k in ("random", "scaffold", "paper") for m in ("rf", "logreg")]
    d = pd.DataFrame(rows)
    print(f"{'split':<10}{'model':<8}{'mols':>6}{'AUROC':>8}{'95% CI':>18}"
          f"{'per-seed':>18}{'ECE':>7}{'Brier':>7}")
    for r in rows:
        print(f"{r['split']:<10}{r['model']:<8}{r['n_mols']:>6}{r['auroc']:>8.3f}"
              f"{f'[{r['lo']:.3f}, {r['hi']:.3f}]':>18}"
              f"{f'{r['per_seed_mean']:.3f} +/- {r['per_seed_sd']:.3f}':>18}"
              f"{r['ece']:>7.3f}{r['brier']:>7.3f}")

    print("\n## 3. What the protocol costs\n")
    base = {r["model"]: r for r in rows if r["split"] == "random"}
    for m in ("rf", "logreg"):
        for k in ("scaffold", "paper"):
            r = next(x for x in rows if x["split"] == k and x["model"] == m)
            print(f"  {m:<8}{k:<10} vs random: {r['auroc'] - base[m]['auroc']:+.3f} AUROC")
    rf = {r["split"]: r for r in rows if r["model"] == "rf"}
    print(f"\n  random -> scaffold {rf['scaffold']['auroc'] - rf['random']['auroc']:+.3f}"
          f"   scaffold -> paper {rf['paper']['auroc'] - rf['scaffold']['auroc']:+.3f}")

    print("\n## 4. Does chemical novelty predict where the model is reliable?\n")
    nov = novelty_curve(smiles, fp, y, papers, seeds)
    print(nov.to_string(index=False, float_format="%.3f"))
    nov.to_csv(NOVEL_OUT, index=False)
    print("\n  TG2 found every model at chance below 0.4 Tanimoto. The same bins are used here.")

    print("\n## 5. Does the pooled-prediction artefact reproduce?\n")
    pooled_vs_molecule(smiles, fp, y, papers, seeds, reps, rng)

    os.makedirs("results", exist_ok=True)
    d.to_csv(SPLITS_OUT, index=False)
    print(f"\nwrote {SPLITS_OUT} and {LEAK_OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="cdk2")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()
    main(a.target, a.seeds, a.reps)
