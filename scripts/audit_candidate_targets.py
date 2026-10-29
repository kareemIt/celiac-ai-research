"""Which second target can actually test the evaluation failures we found on TG2?

Phase 1 of the second-target work. This selects a validation target; it does not model anything.

The TG2 study (docs/paper.md) found four evaluation failures. A second target is only useful if it
can *test* them, which constrains the choice more than raw dataset size does:

  random-split leakage         needs enough near-duplicate chemistry for leakage to be possible
  pooled-prediction artefact   needs nothing special; arises from the protocol
  training nondeterminism      not retested here (no GPU training planned)
  publication confounding      needs MANY publications with enough molecules each -- the single
                               binding constraint, and the one TG2 failed (6 usable papers)

So the headline criterion is `papers_ge10`: publications contributing at least 10 labelled
molecules. A paper-held-out split needs enough of those to form folds; with 6, as in TG2's covalent
set, each fold is one laboratory and the estimate is worthless.

We also require the chemistry to be distinct from TG2 -- otherwise a "second target" is a
restatement of the first -- and the class balance to be usable at the project's 1 uM threshold.

Writes results/candidate_targets.csv. Network + CPU, a few minutes.

Usage: python scripts/audit_candidate_targets.py [--targets cdk2 seh maob bace1 ca2]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold

from scripts.audit_splits import tanimoto_to_train
from scripts.fetch_chembl_tg2 import fetch_all
from src.data import chem, chembl, tg2

OUT = "results/candidate_targets.csv"
RAW = "data/raw/chembl/{name}_activities.csv"

# Resolved from UniProt through the ChEMBL target endpoint, all SINGLE PROTEIN / Homo sapiens.
CANDIDATES = {
    "cdk2": ("CHEMBL301", "P24941", "Cyclin-dependent kinase 2 -- ATP-competitive heterocycles"),
    "seh": ("CHEMBL2409", "P34913", "Soluble epoxide hydrolase -- ureas and amides"),
    "maob": ("CHEMBL2039", "P27338", "Monoamine oxidase B -- small aromatics, propargylamines"),
    "bace1": ("CHEMBL4822", "P56817", "Beta-secretase 1 -- aspartyl protease, amidines"),
    "ca2": ("CHEMBL205", "P00918", "Carbonic anhydrase 2 -- sulfonamide zinc binders"),
}
MIN_PER_PAPER = 10


def fetch_if_needed(name):
    path = RAW.format(name=name)
    if os.path.exists(path):
        return path
    tid, acc, what = CANDIDATES[name]
    print(f"\nfetching {name} ({tid}, UniProt {acc}) -- {what}", flush=True)
    df = fetch_all(tid)
    df.insert(0, "target_name", name)
    df.insert(1, "target_chembl_id", tid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


def audit(name, tg2_fp, tg2_smiles):
    raw = chembl.load_raw(fetch_if_needed(name))
    lab = chembl.classification(raw)  # same 1 uM rule as the TG2 work
    usable = chembl._usable(raw)

    # Papers are attributed per molecule, so one molecule counts once toward its first source.
    doc = usable.groupby("smiles")["document_chembl_id"].first()
    lab = lab.assign(paper=lab.smiles.map(doc))
    per_paper = lab.dropna(subset=["paper"]).groupby("paper").size()
    big = per_paper[per_paper >= MIN_PER_PAPER]

    smiles = lab.smiles.tolist()
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(s) for s in smiles]
    fp = chem.morgan(smiles)

    # Exact fingerprint collisions: the raw material random-split leakage is made of.
    _, inv, counts = np.unique(fp, axis=0, return_inverse=True, return_counts=True)
    dup_frac = float((counts[inv] > 1).mean())

    # How much of the positive class sits in its single largest paper? A high share is the
    # confounding risk that sank the TG2 selectivity model.
    pos = lab[lab.y == 1].dropna(subset=["paper"])
    top_pos_share = float(pos.paper.value_counts().iloc[0] / len(pos)) if len(pos) else np.nan

    sims = tanimoto_to_train(fp, tg2_fp)
    overlap = len(set(smiles) & tg2_smiles)

    return {
        "target": name,
        "chembl_id": CANDIDATES[name][0],
        "records": len(raw),
        "molecules_labelled": len(lab),
        "positive": int(lab.y.sum()),
        "frac_positive": float(lab.y.mean()),
        "censored_negatives": int((lab.n_meas == 1).sum() & 0) or int(len(lab) - len(lab.dropna())),
        "scaffolds": len(set(scaf)),
        "scaffolds_per_mol": len(set(scaf)) / len(lab),
        "papers": int(lab.paper.nunique()),
        "papers_ge10": int(len(big)),
        "mols_in_papers_ge10": int(big.sum()),
        "largest_paper_pos_share": top_pos_share,
        "dup_fingerprint_frac": dup_frac,
        "assay_descriptions": int(raw.assay_description.nunique()),
        "frac_human": float((raw.target_organism == "Homo sapiens").mean()),
        "median_tanimoto_to_tg2": float(np.median(sims)),
        "molecules_shared_with_tg2": overlap,
    }


def verdict(r):
    """Plain-language suitability, with the binding constraint named."""
    fails = []
    if r["papers_ge10"] < 15:
        fails.append(f"only {r['papers_ge10']} papers with >={MIN_PER_PAPER} molecules")
    if not 0.15 <= r["frac_positive"] <= 0.85:
        fails.append(f"class balance {r['frac_positive']:.0%} positive")
    if r["molecules_labelled"] < 300:
        fails.append(f"only {r['molecules_labelled']} labelled molecules")
    if r["median_tanimoto_to_tg2"] > 0.5:
        fails.append(f"chemistry too close to TG2 ({r['median_tanimoto_to_tg2']:.2f})")
    if r["molecules_labelled"] > 6000:
        fails.append(f"{r['molecules_labelled']} molecules is a heavy workload")
    return "SUITABLE" if not fails else "; ".join(fails)


def main(names):
    t = tg2.combined("classification")
    tg2_fp, tg2_smiles = chem.morgan(list(t.smiles)), set(t.smiles)
    print(f"reference: TG2 set has {len(t)} molecules, {int(t.y.sum())} potent\n")

    rows = [audit(n, tg2_fp, tg2_smiles) for n in names]
    d = pd.DataFrame(rows)

    print("\n## Candidate inventory\n")
    cols = ["target", "molecules_labelled", "positive", "frac_positive", "scaffolds",
            "papers", "papers_ge10", "mols_in_papers_ge10"]
    print(d[cols].to_string(index=False, float_format="%.2f"))

    print("\n## Can it test what TG2 could not?\n")
    print(f"{'target':<8}{'papers>=10':>12}{'mols in them':>14}{'largest paper':>15}"
          f"{'dup fp':>9}{'Tanimoto to TG2':>18}{'shared mols':>13}")
    for r in rows:
        print(f"{r['target']:<8}{r['papers_ge10']:>12}{r['mols_in_papers_ge10']:>14}"
              f"{r['largest_paper_pos_share']:>14.0%}{r['dup_fingerprint_frac']:>9.1%}"
              f"{r['median_tanimoto_to_tg2']:>18.3f}{r['molecules_shared_with_tg2']:>13}")

    print("\n## Metadata consistency\n")
    for r in rows:
        print(f"  {r['target']:<8}{r['records']:>7} records  "
              f"{r['assay_descriptions']:>5} distinct assay descriptions  "
              f"{r['frac_human']:>6.1%} human")

    print("\n## Verdict\n")
    for r in rows:
        v = verdict(r)
        print(f"  {r['target']:<8}{v}")
        r["verdict"] = v

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=sorted(CANDIDATES), choices=sorted(CANDIDATES))
    main(ap.parse_args().targets)
