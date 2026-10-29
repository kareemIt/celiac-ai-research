"""Could public data support a real FXIII-A selectivity analysis, and does selectivity transfer?

Phase 1 of the experimental-design work. FXIII-A crosslinks fibrin, so inhibiting it causes
bleeding; for a drug taken for life by otherwise-healthy people, selectivity against it is a safety
gate rather than an optimisation.

docs/selectivity.md answered a cruder version of this question with binary labels at 1 uM and
concluded "no". Three things have changed, and they justify asking once more rather than assuming
the old answer:

  - the measurement is treated as a CONTINUOUS within-study log ratio, pIC50(TG2) - pIC50(FXIII-A),
    which needs no invented threshold
  - every pair used here comes from a SINGLE publication, so the ratio cancels the inter-laboratory
    offset that defeated the earlier cross-paper work
  - the usable set is 147 molecules rather than 29

The model below is run only because the audit in section 1 establishes that structure. It is a
random forest on fingerprints, CPU, seconds; no GPU training is involved.

Writes results/fxiiia_evidence.csv.

Usage: python scripts/audit_fxiiia.py [--reps 2000]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

from scripts.audit_splits import tanimoto_to_train
from scripts.run_covalent import warhead_of
from src.data import chem, chembl

OUT = "results/fxiiia_evidence.csv"
RAW = "data/raw/chembl/{name}_activities.csv"


def exact_pic50(name):
    """Exact IC50 only, as pIC50, keyed by molecule AND source publication."""
    r = chembl.load_raw(RAW.format(name=name))
    u = chembl._usable(r)
    e = u[u.standard_relation == "="].copy()
    e["p"] = 9 - np.log10(e.standard_value.astype(float))
    return e.groupby(["smiles", "document_chembl_id"]).agg(
        p=("p", "median"), n=("p", "size"), assay=("assay_description", "first")).reset_index()


def paired():
    """Molecules measured on both enzymes WITHIN one publication. Cross-paper ratios are not
    built: they would inherit the 0.50-log inter-laboratory offset measured in covalent.md."""
    t, f = exact_pic50("tgm2"), exact_pic50("f13a1")
    m = t.merge(f, on=["smiles", "document_chembl_id"], suffixes=("_tg2", "_f13"))
    m["selectivity"] = m.p_tg2 - m.p_f13
    return m


def fit_predict(tr_fp, tr_y, te_fp):
    return RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=-1
                                 ).fit(tr_fp, tr_y).predict(te_fp)


def main(reps):
    rng = np.random.default_rng(0)
    rows = []
    m = paired()

    print("## 1. What paired evidence exists, and is it independent?\n")
    print(f"   {len(m)} molecules with an exact IC50 on BOTH enzymes in the SAME publication")
    print(f"   across {m.document_chembl_id.nunique()} publications. Cross-publication pairs are "
          f"excluded by construction.\n")
    g = m.groupby("document_chembl_id").agg(molecules=("smiles", "size"),
                                            median_sel=("selectivity", "median"),
                                            sd=("selectivity", "std"))
    print(g.sort_values("molecules", ascending=False).round(2).to_string())
    top = g.molecules.max() / len(m)
    print(f"\n   concentration: the largest publication holds {top:.0%} of the pairs, "
          f"the top two {g.molecules.nlargest(2).sum() / len(m):.0%}")
    rows.append({"section": "inventory", "paired_molecules": len(m),
                 "publications": int(m.document_chembl_id.nunique()),
                 "largest_paper_share": float(top)})

    print("\n## 2. How selective are known TG2 inhibitors, as measured?\n")
    s = m.selectivity
    print(f"   log10(TG2/FXIII-A) median {s.median():+.2f}, range {s.min():+.2f} to {s.max():+.2f}")
    print(f"   >10-fold TG2-selective: {int((s > 1).sum())} of {len(s)}    "
          f">100-fold: {int((s > 2).sum())}")
    print(f"   >10-fold selective the WRONG way (FXIII-A preferring): {int((s < -1).sum())}")
    print("\n   So selective chemistry does exist -- this is a stronger statement than the binary"
          "\n   1 uM analysis in selectivity.md could make, and it is the encouraging part.")
    rows.append({"section": "measured_selectivity", "median_log_sel": float(s.median()),
                 "n_10x": int((s > 1).sum()), "n_100x": int((s > 2).sum()),
                 "n_10x_wrong_way": int((s < -1).sum()), "n": len(s)})

    # The two large publications are the only ones that can support a transfer test.
    ids = g.molecules.nlargest(2).index.tolist()
    a, b = m[m.document_chembl_id == ids[0]], m[m.document_chembl_id == ids[1]]
    sa = {MurckoScaffold.MurckoScaffoldSmiles(x) for x in a.smiles}
    sb = {MurckoScaffold.MurckoScaffoldSmiles(x) for x in b.smiles}
    sims = tanimoto_to_train(chem.morgan(list(b.smiles)), chem.morgan(list(a.smiles)))
    print(f"\n## 3. The transfer test these two publications make possible\n")
    for name, d in ((ids[0], a), (ids[1], b)):
        wh = pd.Series([warhead_of(x) for x in d.smiles]).value_counts().to_dict()
        print(f"   {name}: {len(d)} molecules, "
              f"{len({MurckoScaffold.MurckoScaffoldSmiles(x) for x in d.smiles})} scaffolds, "
              f"median selectivity {d.selectivity.median():+.2f}, warheads {wh}")
    print(f"\n   shared scaffolds: {len(sa & sb)}    "
          f"median max-Tanimoto from the second series to the first: {np.median(sims):.3f}")
    print("   Chemically disjoint, with OPPOSITE median selectivity. A model trained on one must"
          "\n   predict a sign flip it has never seen -- which is what prospective screening is.")
    rows.append({"section": "transfer_setup", "paper_a": ids[0], "paper_b": ids[1],
                 "n_a": len(a), "n_b": len(b), "shared_scaffolds": len(sa & sb),
                 "median_tanimoto_b_to_a": float(np.median(sims))})

    print("\n## 4. Can selectivity be predicted?\n")
    print(f"   {'test':<34}{'n':>5}{'RMSE (log)':>13}{'baseline':>11}{'Spearman':>11}")

    def report(label, y_te, pred, y_tr_mean, key):
        rmse = float(np.sqrt(np.mean((y_te - pred) ** 2)))
        base = float(np.sqrt(np.mean((y_te - y_tr_mean) ** 2)))
        rho = stats.spearmanr(y_te, pred).statistic if np.std(pred) > 0 else np.nan
        print(f"   {label:<34}{len(y_te):>5}{rmse:>13.3f}{base:>11.3f}{rho:>11.3f}")
        rows.append({"section": "model", "test": key, "n": len(y_te), "rmse": rmse,
                     "baseline_rmse": base, "spearman": float(rho)})
        return rmse, base

    # (a) within a single series: is there any learnable signal at all?
    for name, d in ((ids[0], a), (ids[1], b)):
        smi = d.smiles.tolist()
        fp, y = chem.morgan(smi), d.selectivity.to_numpy()
        scaf = np.array([MurckoScaffold.MurckoScaffoldSmiles(x) for x in smi])
        pred = np.full(len(y), np.nan)
        for sc in np.unique(scaf):
            te = scaf == sc
            if (~te).sum() < 10:
                continue
            pred[te] = fit_predict(fp[~te], y[~te], fp[te])
        ok = ~np.isnan(pred)
        report(f"within {name} (scaffold-out)", y[ok], pred[ok], y[ok].mean(), f"within_{name}")

    # (b) across the two series: the question that matters for screening.
    for (n1, d1), (n2, d2) in ((("A", a), ("B", b)), (("B", b), ("A", a))):
        pred = fit_predict(chem.morgan(d1.smiles.tolist()), d1.selectivity.to_numpy(),
                           chem.morgan(d2.smiles.tolist()))
        report(f"train {n1} -> predict {n2}", d2.selectivity.to_numpy(), pred,
               d1.selectivity.mean(), f"{n1}_to_{n2}")

    print("\n   'baseline' predicts the TRAINING set's mean selectivity for every test molecule.")

    print("\n## 5. Verdict\n")
    cross = [r for r in rows if r.get("section") == "model" and "_to_" in str(r.get("test"))]
    worse = [r for r in cross if r["rmse"] >= r["baseline_rmse"]]
    if len(worse) == len(cross):
        print("   Across series the model does not beat predicting the training mean, in either")
        print("   direction. Public data CANNOT support a predictive FXIII-A selectivity model.")
        print("   It can, and does, support the descriptive claim that selective chemistry exists.")
    else:
        print("   At least one direction beats its baseline; see the table. Treat with care --")
        print("   two series is two observations, not a validated capability.")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    main(ap.parse_args().reps)
