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

    for tgt, label in (("cdk2", "CDK2"), ("maob", "MAO-B")):
        s2 = pd.read_csv(f"results/second_target_{tgt}_splits.csv")
        rf2 = s2[s2.model == "rf"].set_index("split")
        for k in ("random", "scaffold", "paper"):
            out.append((f"{label} {k}-split RF AUROC", rf2.loc[k].auroc, 3))
            out.append((f"{label} {k}-split ECE", rf2.loc[k].ece, 3))
        lk = pd.read_csv(f"results/second_target_{tgt}_leakage.csv").set_index("split")
        out.append((f"{label} scaffold shared papers", lk.loc["scaffold"].shared_papers, 1))
        out.append((f"{label} random identical frac", lk.loc["random"].frac_identical * 100, 1))
        nv = pd.read_csv(f"results/second_target_{tgt}_novelty.csv").set_index("tanimoto_bin")
        out.append((f"{label} AUROC below 0.4 Tanimoto", nv.loc["0.0-0.4"].auroc, 3))
        out.append((f"{label} molecules below 0.4", nv.loc["0.0-0.4"].molecules, 0))
        # The split penalties are what the argument rests on, so check them, not just the levels.
        out.append((f"{label} random->scaffold penalty",
                    rf2.loc["scaffold"].auroc - rf2.loc["random"].auroc, 3))
        out.append((f"{label} scaffold->paper penalty",
                    rf2.loc["paper"].auroc - rf2.loc["scaffold"].auroc, 3))

    f = pd.read_csv("results/tg_family_audit.csv")
    pairs = f[f.section == "pairs"].set_index("target")
    out.append(("FXIII-A shared molecules", pairs.loc["f13a1"].shared, 0))
    out.append(("FXIII-A dual-active", pairs.loc["f13a1"].dual, 0))
    out.append(("FXIII-A TG2-selective", pairs.loc["f13a1"].selective, 0))

    # --- §5.13/§5.14: the aggregation, novelty and conformal claims -------------------------
    # These were added 2026-11-05. Everything above was already covered; the sections written
    # after the replication campaign were not, which is how §5.14 came to quote the
    # unmatched-molecule rows of its own result table while calling them matched.
    g = pd.read_csv("results/protocol_gap.csv")
    arms = g[g.penalty.isna() & g.macro_unweighted.notna()].set_index(["target", "split", "model"])
    for split, lbl in (("paper_lopo", "LOPO"), ("paper_grouped", "grouped")):
        r = arms.loc[("cdk2", split, "rf")]
        out.append((f"CDK2 rf {lbl} macro", r.macro_unweighted, 3, "5.14"))
        out.append((f"CDK2 rf {lbl} pooled", r.pooled_molecule, 3, "5.14"))
    # the re-aggregation check: the grouped arm's own predictions, averaged by publication
    reagg = g[g.within_pub_macro.notna() & g.penalty.isna()].set_index(["target", "split", "model"])
    for m in ("rf", "logreg", "xgb"):
        out.append((f"CDK2 {m} grouped re-aggregated by publication",
                    reagg.loc[("cdk2", "paper_grouped", m)].within_pub_macro, 3, "5.14"))
        out.append((f"CDK2 {m} LOPO macro",
                    reagg.loc[("cdk2", "paper_lopo", m)].within_pub_macro, 3, "5.14"))
    out.append(("MAO-B rf pooled protocol gap on shared molecules",
                arms.loc[("maob", "paper_grouped", "rf")].pooled_molecule_shared_set
                - arms.loc[("maob", "paper_lopo", "rf")].pooled_molecule_shared_set, 3, "5.14"))

    # The matched comparison. same_molecules=True is the one that is actually matched: the other
    # scores a publication's whole set in the publication arm against only the part of it the
    # scaffold arm tested (CDK2: 1,052 molecules against 810).
    mt = g.dropna(subset=["penalty"])
    mt = mt[mt.same_molecules == True]                                             # noqa: E712
    for tgt, label in (("cdk2", "CDK2"), ("maob", "MAO-B")):
        d = mt[mt.target == tgt]
        out.append((f"{label} matched penalty min", d.penalty.min(), 3, "5.14"))
        out.append((f"{label} matched penalty max", d.penalty.max(), 3, "5.14"))
        out.append((f"{label} matched publications", d.publications.iloc[0], 0, "5.14"))
        # the claim that the molecule sets are identical is the point of this variant
        assert (d.molecules_scaffold == d.molecules_publication).all(), \
            f"{tgt}: same_molecules rows are not molecule-matched"

    # The six-model mean penalty per target, and the molecule count each arm actually covers.
    # "TG2, 394 molecules" stood in §5.14 until 2026-11-05; the matched set is 428 (the whole
    # scaffold_pub arm, which is a subset of the 444-molecule publication arm).
    six = ["rf", "xgb", "logreg", "chemeleon", "chemberta_ft", "chemberta_frozen"]
    for tgt, label in (("tg2", "TG2"), ("cdk2", "CDK2"), ("maob", "MAO-B")):
        sp = pd.read_csv(f"results/split_penalty_{tgt}.csv")
        out.append((f"{label} six-model mean penalty",
                    sp[sp.model.isin(six)].penalty.mean(), 3, "5.14"))
    pr = pd.read_csv("results/sota_preds.csv")
    arms_mol = {s: set(g.smiles) for s, g in pr.groupby("split")}
    out.append(("TG2 matched molecules (scaffold_pub n paper_lopo)",
                len(arms_mol["scaffold_pub"] & arms_mol["paper_lopo"]), 0, "5.14"))

    nv = pd.read_csv("results/novelty_share.csv")
    nv = nv[(nv.band_set == "published_5") & nv.share.notna()]
    for tgt, label in (("cdk2", "CDK2"), ("maob", "MAO-B")):
        d = nv[nv.target == tgt]
        out.append((f"{label} novelty share min pct", d.share.min() * 100, 0, "5.14"))
        out.append((f"{label} novelty share max pct", d.share.max() * 100, 0, "5.14"))

    # Conformal coverage, pooled over molecules exactly as §5.13's table reports it.
    def coverage(path, regime, alpha, mondrian=False):
        c = pd.read_csv(path)
        c = c[(c.section == "conformal") & (c.regime == regime) & (c.alpha == alpha)
              & (c.mondrian == mondrian)]
        return (c.coverage * c.n_test).sum() / c.n_test.sum()

    conf = {"TG2": "results/lopo_conformal_summary.csv",
            "CDK2": "results/conformal_cdk2_paper_grouped_summary.csv",
            "MAO-B": "results/conformal_maob_paper_grouped_summary.csv"}
    for label, path in conf.items():
        for alpha in (0.1, 0.2):
            for regime in ("iid", "group"):
                out.append((f"{label} {regime} coverage a={alpha}",
                            coverage(path, regime, alpha), 3, "5.13"))
    return out


def locate(claim, text):
    """(needle, hits, scope_label) for one claim against the paper text.

    Searching the whole paper for a digit string is too weak for a number that also occurs
    elsewhere: every one of 5.14's matched penalties "passed" against an unrelated ridge RMSE, a
    Holm delta and a p-value while the section itself quoted different numbers. A claim may
    therefore name, as its fourth element, the section that must carry it, and then only that
    section is searched. Shared with tests/test_paper.py so both apply the same rule.
    """
    value, dp = claim[1], claim[2]
    anchor = claim[3] if len(claim) > 3 else None
    flat = text.replace("−", "-").replace(" ", "")
    if anchor is None:
        scope, scope_label = flat, ""
    else:
        heads = re.split(r"\n(?=#{2,3} )", text)
        hit = [q for q in heads
               if q.split("\n", 1)[0].lstrip("# ").strip().startswith(anchor)]
        if not hit:
            raise LookupError(f"no section of {PAPER} starts with {anchor!r}")
        scope = hit[0].replace("−", "-").replace(" ", "")
        scope_label = anchor.split()[0]
    # A value may appear with or without its sign/zero padding; match the digit string.
    needle = f"{float(value):.{dp}f}".lstrip("-")
    return needle, scope.count(needle), scope_label


def main():
    text = io.open(PAPER, encoding="utf-8").read()
    bad = []
    print(f"{'claim':<46}{'computed':>10}  hits  verdict  scope")
    for claim in claims():
        label, value, dp = claim[:3]
        anchor = claim[3] if len(claim) > 3 else None
        needle, hits, scope_label = locate(claim, text)
        s = f"{float(value):.{dp}f}"
        # This is a presence test, not an attribution test: it says the number is somewhere in the
        # paper, not that it is in the sentence that should carry it. A short or repeated digit
        # string can match coincidentally, so the hit count is printed and anything matching in
        # several places is flagged WEAK rather than reported as a clean pass.
        soft = (hits > 2 or len(needle) <= 3) and anchor is None
        verdict = "MISSING" if hits == 0 else ("weak" if soft else "ok")
        print(f"{label:<46}{s:>10}  {hits:>4}  {verdict:<7}  {scope_label}")
        if hits == 0:
            bad.append((label, s))
    if bad:
        print("\nFAIL: these computed values do not appear in the paper:")
        for label, s in bad:
            print(f"  {label} = {s}")
        sys.exit(1)
    print(f"\nOK: all {len(claims())} recomputed values appear in {PAPER}")
    print("'weak' means the digit string matches in several places or is very short, so presence\n"
          "there is not evidence the right sentence carries it. Those need reading, not trusting.")


if __name__ == "__main__":
    main()
