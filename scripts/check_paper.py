"""Does every headline number in docs/paper.md still match the result tables it came from?

A paper is a cache of numbers computed elsewhere, and caches go stale -- one already did in this
project, when a figure was rebuilt from a superseded table. This recomputes the claims that carry
the argument straight from results/ and fails loudly on any disagreement.

Usage: python scripts/check_paper.py
"""
import io
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

PAPER = "docs/paper.md"


def claims():
    """(label, computed value, decimals) for every number recomputed from a result table."""
    out = []

    t = pd.read_csv("results/tuned_baselines_tg2.csv")
    rf = t[t.model == "rf_tuned"].auroc
    out.append(("tuned RF AUROC", rf.mean(), 3))
    out.append(("tuned RF AUROC sd", rf.std(ddof=1), 3))
    out.append(("tuned XGB AUROC", t[t.model == "xgb_tuned"].auroc.mean(), 3))

    k = pd.read_csv("results/kl_compare_det_kl.csv")
    sel = k[k.epoch.astype(str) == "selected"]
    out.append(("KL lambda=0 selected AUROC", sel[sel.lam == 0].overall_auroc.iloc[0], 3))
    out.append(("KL lambda=1 selected AUROC", sel[sel.lam == 1].overall_auroc.iloc[0], 3))
    out.append(("KL lambda=0 selected novel", sel[sel.lam == 0].novel_auroc.iloc[0], 3))
    out.append(("KL lambda=1 selected novel", sel[sel.lam == 1].novel_auroc.iloc[0], 3))
    cmp_ = sel[sel.comparison.notna()].iloc[0]
    out.append(("KL d_overall", cmp_.d_overall, 3))
    out.append(("KL p_ttest", cmp_.p_ttest, 3))
    out.append(("KL d_novel", cmp_.d_novel, 3))
    out.append(("KL d_novel CI lo", cmp_.novel_lo, 3))
    out.append(("KL d_novel CI hi", cmp_.novel_hi, 3))

    s = pd.read_csv("results/selectivity.csv").set_index("task")
    for task, label in [("TG2 potency", "selectivity: TG2 potency AUROC"),
                        ("FXIII-A potency", "selectivity: FXIII-A AUROC"),
                        ("selectivity (fingerprint)", "selectivity scaffold-split AUROC"),
                        ("selectivity (paper held out)", "selectivity paper-split AUROC"),
                        ("selectivity via TG2 potency", "selectivity via TG2 potency AUROC")]:
        out.append((label, s.loc[task].auroc, 3))
    out.append(("selectivity paper-split CI lo", s.loc["selectivity (paper held out)"].lo, 3))
    out.append(("selectivity paper-split CI hi", s.loc["selectivity (paper held out)"].hi, 3))

    c = pd.read_csv("results/covalent.csv")
    pap = c[c.split == "paper"].set_index("model")
    for m, label in [("mean", "covalent mean RMSE"), ("n_heavy", "covalent size RMSE"),
                     ("ridge", "covalent ridge RMSE"), ("rf", "covalent rf RMSE")]:
        out.append((label, pap.loc[m].rmse, 3))
    out.append(("covalent rf-mean paired", pap.loc["rf - mean (paired)"].rmse, 3))
    out.append(("covalent paired CI lo", pap.loc["rf - mean (paired)"].rmse_lo, 3))
    out.append(("covalent paired CI hi", pap.loc["rf - mean (paired)"].rmse_hi, 3))
    out.append(("covalent rf within-paper rho", pap.loc["rf"].mean_within_group_spearman, 3))
    scaf = c[c.split == "scaffold"].set_index("model")
    out.append(("covalent rf scaffold RMSE", scaf.loc["rf"].rmse, 3))

    a = pd.read_csv("results/covalent_audit.csv")
    u = a[(a.section == "usable")].iloc[0]
    out.append(("covalent molecules", u.molecules, 0))
    out.append(("covalent papers", u.papers, 0))
    out.append(("covalent scaffolds", u.scaffolds, 0))
    comp = a[a.section == "comparability"].iloc[0]
    out.append(("covalent between-paper SD", comp.between_paper_sd, 2))
    out.append(("covalent within-paper SD", comp.within_paper_sd, 2))
    out.append(("covalent acrylamides",
                a[(a.section == "warhead") & (a.quantity == "acrylamide")].molecules.iloc[0], 0))

    s2 = pd.read_csv("results/second_target_splits.csv")
    rf2 = s2[s2.model == "rf"].set_index("split")
    for k in ("random", "scaffold", "paper"):
        out.append((f"CDK2 {k}-split RF AUROC", rf2.loc[k].auroc, 3))
        out.append((f"CDK2 {k}-split ECE", rf2.loc[k].ece, 3))
    lk = pd.read_csv("results/second_target_leakage.csv").set_index("split")
    out.append(("CDK2 scaffold shared papers", lk.loc["scaffold"].shared_papers, 1))
    out.append(("CDK2 random identical frac", lk.loc["random"].frac_identical * 100, 1))
    nv = pd.read_csv("results/second_target_novelty.csv").set_index("tanimoto_bin")
    out.append(("CDK2 AUROC below 0.4 Tanimoto", nv.loc["0.0-0.4"].auroc, 3))
    out.append(("CDK2 molecules below 0.4", nv.loc["0.0-0.4"].molecules, 0))

    f = pd.read_csv("results/tg_family_audit.csv")
    pairs = f[f.section == "pairs"].set_index("target")
    out.append(("FXIII-A shared molecules", pairs.loc["f13a1"].shared, 0))
    out.append(("FXIII-A dual-active", pairs.loc["f13a1"].dual, 0))
    out.append(("FXIII-A TG2-selective", pairs.loc["f13a1"].selective, 0))
    return out


def main():
    text = io.open(PAPER, encoding="utf-8").read()
    # Normalize unicode minus so "-0.046" in the paper matches a computed negative.
    hay = text.replace("−", "-").replace(" ", "")
    bad = []
    print(f"{'claim':<38}{'computed':>12}   in paper?")
    for label, value, dp in claims():
        s = f"{float(value):.{dp}f}"
        # A value may appear with or without its sign/zero padding; match the digit string.
        needle = s.lstrip("-")
        found = needle in hay
        print(f"{label:<38}{s:>12}   {'yes' if found else 'MISSING'}")
        if not found:
            bad.append((label, s))
    if bad:
        print("\nFAIL: these computed values do not appear in the paper:")
        for label, s in bad:
            print(f"  {label} = {s}")
        sys.exit(1)
    print(f"\nOK: all {len(claims())} recomputed values appear in {PAPER}")


if __name__ == "__main__":
    main()
