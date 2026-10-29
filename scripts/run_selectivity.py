"""Can a fingerprint model tell a TG2-selective inhibitor from a pan-transglutaminase one?

Phase 2. The audit (scripts/audit_tg_family.py) found that only FXIII-A has enough paired data to
try: 165 molecules measured on both TG2 and FXIII-A, of which 29 are TG2-potent but not FXIII-A
potent and 103 hit both. TGM1 and TGM3 have 5 and 1 potent molecules in total and are not modelled.

Three questions, each on scaffold splits with a bootstrap CI over molecules:

  A. TG2 potency          the existing baseline, refit here so every task shares one protocol
  B. FXIII-A potency      can the antitarget be predicted at all?
  C. selectivity          among molecules measured on both, TG2-potent-and-FXIII-A-clean vs the rest

C is the question that matters: a model that scores (A) well but (C) at chance can rank TG2 actives
and cannot tell which of them would also cripple fibrin crosslinking.

Honest-baseline note: task C is scored against the majority-class rate AND against predicted TG2
potency alone, because a selectivity model that merely rediscovers potency has added nothing.

Writes results/selectivity.csv. CPU, about a minute.

Usage: python scripts/run_selectivity.py [--seeds 10] [--reps 2000]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.data import chem, chembl, tg2
from src.evaluation import metrics

OUT = "results/selectivity.csv"
RAW = "data/raw/chembl/{name}_activities.csv"
# The tuned RF configuration this project already settled on for TG2.
RF = dict(n_estimators=500, min_samples_leaf=1, max_features="sqrt", n_jobs=-1)


def auroc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan


def boot_ci(y, p, reps, rng):
    """Bootstrap over molecules, which are the independent units here."""
    b = [auroc(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(reps))]
    b = [v for v in b if np.isfinite(v)]
    return (np.percentile(b, 2.5), np.percentile(b, 97.5)) if b else (np.nan, np.nan)


def run_task(name, smiles, y, seeds, reps, rng, extra=None, groups=None):
    """RF on Morgan bits, pooled out-of-fold predictions, bootstrap CI over molecules.

    `groups` chooses what is held out: None splits by Bemis-Murcko scaffold, as everywhere else in
    this project; passing the source document splits by paper instead, which is the only way to
    tell a real structure-activity signal from one series measured in one lab."""
    fp = chem.morgan(smiles)
    X = np.hstack([fp, extra]) if extra is not None else fp
    rows = []
    for seed in range(seeds):
        tr, va, te = (chem.scaffold_split(smiles, seed=seed) if groups is None
                      else chem.group_split(groups, seed=seed))
        tr = np.concatenate([tr, va])  # no tuning here, so the validation fold is training data
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = RandomForestClassifier(random_state=seed, **RF).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        rows.append(pd.DataFrame({"seed": seed, "smiles": [smiles[i] for i in te],
                                  "y": y[te], "p": p}))
    if not rows:
        print(f"  {name:<34} not evaluable: a fold had one class only")
        return None
    pred = pd.concat(rows, ignore_index=True)
    # One row per molecule: folds overlap across seeds, so pooling raw rows would double-count.
    mol = pred.groupby("smiles").agg(y=("y", "first"), p=("p", "mean"))
    a = auroc(mol.y.to_numpy(), mol.p.to_numpy())
    lo, hi = boot_ci(mol.y.to_numpy(), mol.p.to_numpy(), reps, rng)
    per_seed = np.array([auroc(g.y.to_numpy(), g.p.to_numpy()) for _, g in pred.groupby("seed")])
    per_seed = per_seed[np.isfinite(per_seed)]
    print(f"  {name:<34} {len(mol):>4} mols  {int(mol.y.sum()):>3} pos ({mol.y.mean():.0%})   "
          f"AUROC {a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]   per-seed {per_seed.mean():.3f} "
          f"+/- {per_seed.std(ddof=1):.3f}")
    return {"task": name, "n_mols": len(mol), "n_pos": int(mol.y.sum()), "auroc": a,
            "lo": lo, "hi": hi, "per_seed_mean": per_seed.mean(),
            "per_seed_sd": per_seed.std(ddof=1), "pred": pred, "mol": mol}


def main(seeds, reps):
    rng = np.random.default_rng(0)
    rows = []

    tg2_tab = tg2.combined("classification")
    f13 = chembl.classification(chembl.load_raw(RAW.format(name="f13a1")))

    print("## A/B. Can each enzyme be predicted on its own?\n")
    a = run_task("TG2 potency", tg2_tab.smiles.tolist(), tg2_tab.y.to_numpy(), seeds, reps, rng)
    b = run_task("FXIII-A potency", f13.smiles.tolist(), f13.y.to_numpy(), seeds, reps, rng)

    print("\n## C. Selectivity: TG2-potent AND FXIII-A-clean, among molecules measured on both\n")
    both = tg2_tab[["smiles", "y"]].merge(f13[["smiles", "y"]], on="smiles",
                                          suffixes=("_tg2", "_f13"))
    sel_y = ((both.y_tg2 == 1) & (both.y_f13 == 0)).astype(int).to_numpy()
    smi = both.smiles.tolist()
    print(f"  {len(both)} molecules measured on both; {sel_y.sum()} are TG2-selective, "
          f"{int(((both.y_tg2 == 1) & (both.y_f13 == 1)).sum())} hit both, "
          f"{int((both.y_tg2 == 0).sum())} are not TG2-potent\n")
    c = run_task("selectivity (fingerprint)", smi, sel_y, seeds, reps, rng)

    # Does a selectivity model beat simply ranking by predicted TG2 potency? If not, it is only
    # rediscovering potency and tells us nothing new about the antitarget.
    if a is not None:
        tg2_p = a["mol"].reindex(smi).p
        covered = tg2_p.notna().to_numpy()
        if covered.sum() > 10 and len(np.unique(sel_y[covered])) > 1:
            base = auroc(sel_y[covered], tg2_p.to_numpy()[covered])
            lo, hi = boot_ci(sel_y[covered], tg2_p.to_numpy()[covered], reps, rng)
            print(f"  {'ranking by predicted TG2 potency':<34} {int(covered.sum()):>4} mols  "
                  f"{int(sel_y[covered].sum()):>3} pos          AUROC {base:.3f}  "
                  f"95% CI [{lo:.3f}, {hi:.3f}]")
            rows.append({"task": "selectivity via TG2 potency", "n_mols": int(covered.sum()),
                         "n_pos": int(sel_y[covered].sum()), "auroc": base, "lo": lo, "hi": hi})

    print("\n## C2. The same task, holding out whole papers instead of whole scaffolds\n")
    docs = (chembl.load_raw(RAW.format(name="f13a1"))
            .groupby("smiles")["document_chembl_id"].first())
    g = docs.reindex(smi)
    print(f"  {g.nunique()} source documents; the selective class draws "
          f"{g[sel_y == 1].nunique()} of them, and its largest single paper supplies "
          f"{int(g[sel_y == 1].value_counts().iloc[0])} of {int(sel_y.sum())} selective molecules.\n")
    c2 = run_task("selectivity (paper held out)", smi, sel_y, seeds, reps, rng,
                  groups=g.fillna("unknown").to_numpy())
    if c2 is not None:
        rows.append({k: v for k, v in c2.items() if k not in ("pred", "mol")})
    if c is not None and c2 is not None:
        print(f"\n  scaffold-split {c['auroc']:.3f} -> paper-split {c2['auroc']:.3f} "
              f"(delta {c2['auroc'] - c['auroc']:+.3f})")
        print("  A large drop means the scaffold number was reading series provenance, not "
              "selectivity:\n  Murcko scaffolds do not separate papers when one lab's series "
              "spans several scaffolds.")

    print("\n## D. Multi-task: does knowing the FXIII-A prediction help the selectivity model?\n")
    if b is not None and c is not None:
        f13_p = b["mol"].reindex(smi).p.to_numpy()
        keep = np.isfinite(f13_p)
        print(f"  the FXIII-A model covers {int(keep.sum())} of {len(smi)} paired molecules "
              f"({int(sel_y[keep].sum())} selective); scoring on those\n")
        if keep.sum() > 20 and len(np.unique(sel_y[keep])) > 1:
            smi_k = [s for s, k in zip(smi, keep) if k]
            single = run_task("selectivity, covered subset", smi_k, sel_y[keep], seeds, reps, rng)
            dd = run_task("selectivity + FXIII-A feature", smi_k, sel_y[keep], seeds, reps, rng,
                          extra=f13_p[keep][:, None])
            for r in (single, dd):
                if r is not None:
                    rows.append({k: v for k, v in r.items() if k not in ("pred", "mol")})
            if single is not None and dd is not None:
                print(f"\n  single-task {single['auroc']:.3f} -> multi-task {dd['auroc']:.3f} "
                      f"(delta {dd['auroc'] - single['auroc']:+.3f})")
                print("  The FXIII-A feature is itself a prediction fitted on these molecules, so "
                      "this is an\n  upper bound on a two-head model, not a fair estimate.")
        else:
            print("  not evaluable: too few covered molecules")

    for r in (a, b, c):
        if r is not None:
            rows.append({k: v for k, v in r.items() if k not in ("pred", "mol")})
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()
    main(a.seeds, a.reps)
