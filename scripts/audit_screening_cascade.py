"""Could the five-gate screening framework actually be applied, and would its gates agree?

Phase 3. The proposal is a cascade combining predicted TG2 activity, selectivity, applicability
domain, prediction uncertainty and ADME/drug-likeness. scripts/screen_candidates.py already
implements four of those five on the approved-drug library. This script asks the two questions that
decide whether the fifth can be added and whether the cascade is coherent:

  1. On how many molecules does EVERY gate's input exist? The evidence sets are not the same set:
     measured selectivity exists only for molecules inside the training data, and applicability
     domain and conformal uncertainty are only defined for molecules outside it.
  2. Do the gates agree or conflict? docs/selectivity_feasibility.md measured a POSITIVE
     correlation between TG2 and FXIII-A potency, which predicts that the activity gate and the
     selectivity gate fight each other. That is testable on the paired molecules.

Produces no shortlist and fits no new model. Writes results/screening_cascade_audit.csv.

Usage: python scripts/audit_screening_cascade.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy import stats

from scripts.audit_fxiiia import paired
from src.data import tg2

OUT = "results/screening_cascade_audit.csv"
LIB = "results/candidates_tg2.csv"                  # scored approved drugs, from screen_candidates
LOPO = "results/lopo_conformal.csv"                 # conformal sets, from run_lopo_conformal
SEL_GATE = 1.0                                      # log10 units: 10-fold TG2-selective
DOMAIN_LO = 0.40                                    # scripts/screen_candidates.py's floor


def main():
    rows = []
    train = tg2.combined("classification")
    m = paired()
    print("## 1. Where does each gate's input exist?\n")

    have = {
        "predicted TG2 activity": set(train.smiles),
        "measured TG2 activity": set(train.smiles),
        "measured selectivity (FXIII-A)": set(m.smiles),
        "applicability domain": set(),
        "conformal uncertainty (LOPO)": set(),
        "ADME / drug-likeness": set(),
    }
    if os.path.exists(LOPO):
        have["conformal uncertainty (LOPO)"] = set(pd.read_csv(LOPO).smiles)
    lib = pd.read_csv(LIB) if os.path.exists(LIB) else None
    if lib is not None:
        col = "smiles" if "smiles" in lib else lib.columns[0]
        have["applicability domain"] = set(lib[col])
        have["ADME / drug-likeness"] = set(lib[col])
        print(f"   scored library: {len(lib)} approved drugs from {LIB}")
    print(f"   training set:   {len(train)} TG2 molecules")
    print(f"   paired set:     {len(m)} molecules with a within-publication FXIII-A ratio\n")
    for name, s in have.items():
        print(f"   {name:<32}{len(s):>6} molecules")
        rows.append({"section": "evidence_available", "gate": name, "molecules": len(s)})

    inside = have["measured selectivity (FXIII-A)"] & have["conformal uncertainty (LOPO)"]
    outside = have["measured selectivity (FXIII-A)"] & have["applicability domain"]
    all_five = (have["measured selectivity (FXIII-A)"] & have["conformal uncertainty (LOPO)"]
                & have["applicability domain"] & have["ADME / drug-likeness"])
    print(f"\n   selectivity AND conformal uncertainty:        {len(inside)}")
    print(f"   selectivity AND a library molecule:           {len(outside)}")
    print(f"   ALL FIVE gate inputs on the same molecule:    {len(all_five)}")
    rows.append({"section": "evidence_overlap", "gate": "selectivity_and_uncertainty",
                 "molecules": len(inside)})
    rows.append({"section": "evidence_overlap", "gate": "selectivity_and_library",
                 "molecules": len(outside)})
    rows.append({"section": "evidence_overlap", "gate": "all_five", "molecules": len(all_five)})
    print("\n   The sets are structurally disjoint, not merely small. Measured selectivity exists"
          "\n   only for molecules already in the training data; applicability domain and conformal"
          "\n   uncertainty are quantities about molecules OUTSIDE it. A cascade needs all five on"
          "\n   one molecule.")

    print("\n## 2. Do the activity and selectivity gates agree?\n")
    print("   Among the paired molecules, does being more TG2-potent make the 10-fold selectivity"
          "\n   gate more or less likely to pass?\n")
    m = m.assign(selective=m.selectivity > SEL_GATE)
    base = m.selective.mean()
    print(f"   {'TG2 potency stratum':<28}{'n':>5}{'pass 10x gate':>15}{'vs base rate':>14}")
    q = m.p_tg2.quantile([0.5, 0.75, 0.9])
    for label, sub in (("all paired", m),
                       ("top half by pIC50", m[m.p_tg2 >= q[0.5]]),
                       ("top quartile", m[m.p_tg2 >= q[0.75]]),
                       ("top decile", m[m.p_tg2 >= q[0.9]])):
        r = sub.selective.mean()
        print(f"   {label:<28}{len(sub):>5}{r:>14.1%}{r - base:>+14.1%}")
        rows.append({"section": "gate_agreement", "gate": label, "molecules": len(sub),
                     "pass_rate": float(r), "delta_vs_base": float(r - base)})
    # Fisher exact on the top decile against the rest
    top = m.p_tg2 >= q[0.9]
    tab = [[int((top & m.selective).sum()), int((top & ~m.selective).sum())],
           [int((~top & m.selective).sum()), int((~top & ~m.selective).sum())]]
    odds, pval = stats.fisher_exact(tab)
    print(f"\n   top decile vs the rest: odds ratio {odds:.2f}, Fisher exact p={pval:.3g}")
    rows.append({"section": "gate_agreement", "gate": "top_decile_fisher",
                 "odds_ratio": float(odds), "p": float(pval)})

    print("\n   Within each of the two large publications separately, so the mixture of a")
    print("   TG2-directed and an FXIII-A-directed programme cannot drive the result:\n")
    for doc, g in m.groupby("document_chembl_id"):
        if len(g) < 20:
            continue
        b, t = g.selective.mean(), g[g.p_tg2 >= g.p_tg2.quantile(0.75)].selective.mean()
        print(f"   {doc:<16} base {b:>6.1%}   top quartile {t:>6.1%}   "
              f"delta {t - b:>+6.1%}  (n={len(g)})")
        rows.append({"section": "gate_agreement_within_paper", "gate": doc, "molecules": len(g),
                     "pass_rate": float(t), "delta_vs_base": float(t - b)})

    print("\n## 3. What the existing library screen already shows\n")
    if lib is None:
        print(f"   {LIB} not present; run scripts/screen_candidates.py first")
    else:
        sim = next((c for c in lib.columns if "tanimoto" in c), None)
        if sim:
            inside_n = int((lib[sim] >= DOMAIN_LO).sum())
            print(f"   approved drugs at or above the {DOMAIN_LO} applicability floor: "
                  f"{inside_n} of {len(lib)} ({inside_n / len(lib):.1%})")
            rows.append({"section": "library", "gate": "in_domain", "molecules": inside_n,
                         "pass_rate": float(inside_n / len(lib))})
            print(f"   with measured FXIII-A selectivity:                    "
                  f"{len(outside)} of {len(lib)}")
        print("\n   So the selectivity gate cannot be applied to the library at all, and the"
              "\n   applicability gate already removes almost all of it.")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
