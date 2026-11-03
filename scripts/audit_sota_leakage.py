"""Pretraining-data leakage audit for the state-of-the-art comparison.

The naive audit is meaningless here. Our 631 molecules come from PubChem and ChEMBL; ChemBERTa-2 was
pretrained on 77M PubChem compounds and MoLFormer on 1.1B PubChem+ZINC molecules, so essentially
every molecule we test is in their pretraining corpora as a structure. Reporting "~100% overlap"
would be true and useless.

What matters is whether a model could have seen an experimental TG2 potency, not whether it has seen
the structure. This script separates the two and audits the one channel that is actually local to us:
what the Qwen prompts contain.

Reproduce: python scripts/audit_sota_leakage.py   ->  results/sota_leakage.csv
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.data import chembl, tg2

OUT = "results/sota_leakage.csv"

# What each entrant's pretraining objective could have exposed it to. "structures" means the
# molecules themselves; "computed" means properties a program calculated from the structure;
# "experimental" means a measured bioactivity, which is the only channel that leaks our label.
PROVENANCE = [
    # model, pretraining corpus, structures, computed properties, experimental activity
    ("logreg", "none", "no", "no", "no"),
    ("rf", "none", "no", "no", "no"),
    ("xgb", "none", "no", "no", "no"),
    ("chemprop", "none", "no", "no", "no"),
    ("chemprop_rdkit", "none", "no", "no", "no"),
    ("tabpfn", "synthetic tabular data", "no", "no", "no"),
    ("chemeleon", "computed molecular descriptors", "likely", "yes", "no"),
    ("chemberta_frozen", "77M PubChem compounds (MTR)", "likely", "yes", "no"),
    ("chemberta_ft", "77M PubChem compounds (MTR)", "likely", "yes", "no"),
    ("molformer_frozen", "1.1B PubChem+ZINC (MLM)", "likely", "no", "no"),
    ("jev_qlora", "general web/text corpus", "likely, as text", "n/a", "possible: the papers"),
    ("jev_zeroshot", "general web/text corpus", "likely, as text", "n/a", "possible: the papers"),
    # the only entrant excluded on provenance: CLAMP is trained on PubChem bioassay readouts
    ("clamp (excluded)", "PubChem assay data", "yes", "n/a", "yes"),
]

# A systematic IUPAC name redescribes the structure, so a model reading it learns nothing it could
# not read off the SMILES. A trivial or code name is a retrieval key into the literature, which is
# the channel that could leak a measured potency.
SYSTEMATIC = re.compile(r"[\[\(]|yl\b|amino|oxy|ene\b", re.IGNORECASE)


def corpus_membership():
    """What we can and cannot establish locally about pretraining-corpus overlap."""
    raw = tg2.load_raw()
    comb = tg2.combined("classification")
    cid = raw.groupby("smiles")["CID"].first()
    in_pubchem = comb["smiles"].map(cid).notna()
    in_chembl = comb["smiles"].isin(chembl._usable(chembl.load_raw())["smiles"])
    return [
        ("molecules evaluated", len(comb), ""),
        ("provably in PubChem (carry a CID)", int(in_pubchem.sum()),
         "so in ChemBERTa-2's and MoLFormer's corpora by construction"),
        ("present in ChEMBL's TG2 target file", int(in_chembl.sum()),
         "ChEMBL is not a pretraining corpus for any entrant here"),
        ("auditable against the actual pretraining corpora", 0,
         "77M/1.1B corpora are not downloadable locally; membership is inferred, not measured"),
    ]


def prompt_audit():
    """What the Qwen prompts actually contain.

    scripts/run_jev_qlora.py builds each prompt as "Compound SMILES: ..." plus "Compound name: ..."
    whenever the PubChem export has a name. The protocol assumed SMILES only, so this is the one
    place a general LLM could retrieve a literature potency rather than infer it from structure.
    """
    raw = tg2.load_raw()
    names = raw.groupby("smiles")["Compound_Name"].first().to_dict()
    comb = tg2.combined("classification")
    comb = comb.assign(name=comb["smiles"].map(names))
    named = comb[comb["name"].apply(lambda v: isinstance(v, str))]
    systematic = named["name"].str.contains(SYSTEMATIC)
    return comb, named, systematic


def main():
    print("## What is auditable locally\n")
    rows = []
    for label, n, note in corpus_membership():
        print(f"  {label:<48}{n:>6}   {note}")
        rows.append({"section": "corpus", "item": label, "value": n, "note": note})

    comb, named, systematic = prompt_audit()
    risky = named[~systematic]
    print(f"\n## Qwen prompt content ({len(comb)} molecules)\n")
    facts = [
        ("molecules whose prompt carries a compound name", len(named),
         f"{len(named) / len(comb):.1%} of the set"),
        ("names that are systematic (IUPAC-style)", int(systematic.sum()),
         f"{systematic.mean():.1%} of named; a redescription of the structure, not a retrieval key"),
        ("names that are trivial or code-like", len(risky),
         f"{len(risky) / len(comb):.1%} of the set; the only name-based retrieval channel"),
    ]
    for label, n, note in facts:
        print(f"  {label:<48}{n:>6}   {note}")
        rows.append({"section": "prompt", "item": label, "value": n, "note": note})

    print(f"\n  the {len(risky)} non-systematic names, with their label:")
    for _, r in risky.iterrows():
        print(f"    {r['name']:<40} potent={r['y']}")

    print("\n## Pretraining provenance by entrant\n")
    prov = pd.DataFrame(PROVENANCE, columns=["model", "corpus", "saw_structures",
                                             "saw_computed_properties", "saw_experimental_activity"])
    print(prov.to_string(index=False))

    pd.concat([pd.DataFrame(rows), prov.assign(section="provenance")], ignore_index=True) \
        .to_csv(OUT, index=False)
    risky[["smiles", "name", "y"]].to_csv(OUT.replace(".csv", "_named.csv"), index=False)
    print(f"\n-> {OUT}")

    print("\n## What this audit cannot establish\n"
          "  It cannot certify that any pretrained model has not memorised an IC50 from a paper\n"
          "  outside the corpora we can inspect. The claim it supports is narrower: no entrant's\n"
          "  published pretraining objective consumed experimental TG2 activities, and the one\n"
          "  model that is known to have (CLAMP) is excluded. For Qwen, the name channel is real\n"
          "  but narrow, and no name in the set is a recognised TG2 clinical compound identifier.")


if __name__ == "__main__":
    main()
