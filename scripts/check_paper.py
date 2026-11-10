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

    # --- 5.16: the three-method conformal comparison -------------------------------------
    if os.path.exists("results/hcp_comparison.csv"):
        h = pd.read_csv("results/hcp_comparison.csv")
        h["tc"] = 1 - h.alpha
        for m, lbl in (("hcp", "HCP"), ("grouped_pooled", "ours")):
            d = h[h.method == m]
            out.append((f"{lbl} runs holding the coverage floor",
                        int((d.coverage >= d.tc).sum()), 0, "5.16"))
        w = h[(h.design == "within") & (h.method == "grouped_pooled")]
        out.append(("ours within-design runs holding the floor",
                    int((w.coverage >= w.tc).sum()), 0, "5.16"))
        # the headline row of the comparison table: MAO-B within-fold, random forest, alpha=0.1
        for m, lbl in (("grouped_pooled", "ours"), ("hcp", "HCP")):
            r = h[(h.target == "maob") & (h.design == "within") & (h.alpha == 0.1)
                  & (h.model == "rf") & (h.method == m)].iloc[0]
            out.append((f"MAO-B within rf a=0.1 {lbl} coverage", r.coverage, 3, "5.16"))
            out.append((f"MAO-B within rf a=0.1 {lbl} singleton", r.singleton_frac, 3, "5.16"))
        pr = pd.read_csv("results/hcp_paired.csv")
        out.append(("paired comparisons", len(pr), 0, "5.16"))
        out.append(("paired comparisons excluding zero", int((pr.ci_lo > 0).sum()), 0, "5.16"))
        sw = pd.read_csv("results/hcp_k1_sweep.csv")
        hs = sw[sw.method.str.startswith("HCP")].sort_values("K1")
        for _, r in hs.iterrows():
            out.append((f"K1={int(r.K1)} HCP coverage", r.coverage, 3, "5.16"))

    conf = {"TG2": "results/lopo_conformal_summary.csv",
            "CDK2": "results/conformal_cdk2_paper_grouped_summary.csv",
            "MAO-B": "results/conformal_maob_paper_grouped_summary.csv"}
    for label, path in conf.items():
        for alpha in (0.1, 0.2):
            for regime in ("iid", "group"):
                out.append((f"{label} {regime} coverage a={alpha}",
                            coverage(path, regime, alpha), 3, "5.13"))

    # --- 5.17: per-class rates under publication shift, on the publication unit -----------
    if os.path.exists("results/per_class_shift_tests.csv"):
        st = pd.read_csv("results/per_class_shift_tests.csv")
        st = st[st.unit == "publication"]
        for tgt, label in (("tg2", "TG2"), ("cdk2", "CDK2"), ("maob", "MAO-B")):
            for rate in ("sensitivity", "specificity"):
                d = st[(st.target == tgt) & (st.rate == rate)]
                if d.empty:
                    continue
                out.append((f"{label} {rate} mean shift delta", d.delta.mean(), 3, "5.17"))
                out.append((f"{label} {rate} sign-test p", d.sign_test_p.iloc[0], 3, "5.17"))
            # the paper quotes one best-case p per target, across both rates
            both = st[st.target == tgt]
            if not both.empty:
                out.append((f"{label} best p_holm, either rate", both.p_holm.min(), 3, "5.17"))
        for model, rate in (("rf", "sensitivity"), ("rf", "specificity")):
            r = st[(st.target == "tg2") & (st.model == model) & (st.rate == rate)].iloc[0]
            out.append((f"TG2 {model} {rate} reference arm", r.mean_ref, 3, "5.17"))
            out.append((f"TG2 {model} {rate} shifted arm", r.mean_shifted, 3, "5.17"))
        for rate in ("sensitivity", "specificity"):
            r = st[(st.target == "tg2") & (st.model == "xgb") & (st.rate == rate)].iloc[0]
            out.append((f"TG2 xgb {rate} shift delta", r.delta, 3, "5.17"))

        s = pd.read_csv("results/per_class_tg2_summary.csv")
        cp = s[(s.model == "chemprop") & (s.split == "paper_lopo")
               & (s.threshold_kind == "t050")].iloc[0]
        out.append(("Chemprop TG2 LOPO accuracy", cp.accuracy_mean, 3, "5.17"))
        out.append(("Chemprop TG2 LOPO AUROC", cp.auroc_mean, 3, "5.17"))
        out.append(("Chemprop TG2 LOPO balanced accuracy", cp.balanced_accuracy_mean, 3, "5.17"))
        rf_lopo = s[(s.model == "rf") & (s.split == "paper_lopo")].set_index("threshold_kind")
        for kind, lbl in (("oracle", "oracle"), ("t050", "0.5 cut"), ("prior", "prevalence cut")):
            out.append((f"TG2 rf LOPO balanced accuracy, {lbl}",
                        rf_lopo.loc[kind].balanced_accuracy_mean, 3, "5.17"))

    # --- 5.18: activity cliffs, MoleculeACE's full three-test definition ------------------
    if os.path.exists("results/cliff_ranking_tg2_tests.csv"):
        ct = pd.concat([pd.read_csv(f"results/cliff_ranking_{t}_tests.csv")
                        for t in ("tg2", "cdk2", "maob")], ignore_index=True)
        ct = ct[ct.definition == "three_test"]
        out.append(("cliff comparisons", len(ct), 0, "5.18"))
        out.append(("cliff comparisons where cliffs are worse", int((ct.delta < 0).sum()), 0, "5.18"))
        out.append(("cliff comparisons significant after Holm",
                    int((ct.p_holm < 0.05).sum()), 0, "5.18"))
        for tgt, model in (("tg2", "chemprop"), ("tg2", "rf"), ("cdk2", "rf"), ("maob", "logreg")):
            r = ct[(ct.target == tgt) & (ct.split == "scaffold_pub") & (ct.model == model)].iloc[0]
            lbl = f"{tgt} {model} scaffold_pub cliff"
            out.append((f"{lbl} accuracy", r.acc_cliff, 3, "5.18"))
            out.append((f"{lbl} distant accuracy", r.acc_distant, 3, "5.18"))
            out.append((f"{lbl} delta", r.delta, 3, "5.18"))
            out.append((f"{lbl} p_holm", r.p_holm, 3, "5.18"))
        for tgt, arm in (("tg2", "scaffold"), ("cdk2", "scaffold_pub"), ("maob", "scaffold_pub")):
            r = ct[(ct.target == tgt) & (ct.split == arm)].iloc[0]
            out.append((f"{tgt} {arm} cliff rank rho", r.rank_rho_models, 3, "5.18"))
            out.append((f"{tgt} {arm} cliff rank rho p", r.rank_rho_p, 3, "5.18"))
        pairs = pd.read_csv("results/cliff_pairs_tg2.csv")
        out.append(("TG2 pairs admitted by the scaffold test", int(pairs.by_scaffold.sum()), 0, "5.18"))
        out.append(("TG2 pairs admitted by the morgan test", int(pairs.by_morgan.sum()), 0, "5.18"))

    # --- 5.17 addenda: the pooled estimator and the single-class decomposition ------------
    if os.path.exists("results/per_class_pooled_rates.csv"):
        po = pd.read_csv("results/per_class_pooled_rates.csv")
        shift = {"tg2": "paper_lopo", "cdk2": "paper_grouped", "maob": "paper_grouped"}
        rf = po[po.model == "rf"].set_index(["target", "split"])
        out.append(("pooled TG2 rf sensitivity, reference arm",
                    rf.loc[("tg2", "scaffold_pub")].sensitivity, 3, "5.17"))
        out.append(("pooled TG2 rf sensitivity, shifted arm",
                    rf.loc[("tg2", "paper_lopo")].sensitivity, 3, "5.17"))
        for tgt, arm in shift.items():
            a = po[(po.target == tgt) & (po.split == "scaffold_pub")].set_index("model")
            b = po[(po.target == tgt) & (po.split == arm)].set_index("model")
            both = a.index.intersection(b.index)
            for rate in ("sensitivity", "specificity"):
                d = b.loc[both, rate] - a.loc[both, rate]
                out.append((f"pooled {tgt} {rate} models falling", int((d < 0).sum()), 0, "5.17"))
                out.append((f"pooled {tgt} {rate} mean delta", d.mean(), 3, "5.17"))
        au = pd.read_csv("results/per_class_single_class_audit.csv").set_index(["target", "split"])
        for tgt in ("tg2", "cdk2", "maob"):
            out.append((f"{tgt} scaffold_pub single-class publications pct",
                        au.loc[(tgt, "scaffold_pub")].pub_single_class_frac * 100, 0, "5.17"))

    # --- 5.18 addendum: the fingerprint-only sensitivity analysis -------------------------
    if os.path.exists("results/cliff_ranking_tg2_tests.csv"):
        ct = pd.concat([pd.read_csv(f"results/cliff_ranking_{t}_tests.csv")
                        for t in ("tg2", "cdk2", "maob")], ignore_index=True)
        strict = ct[ct.definition == "morgan_only_0.9"]
        out.append(("strict-similarity comparisons", len(strict), 0, "5.18"))
        out.append(("strict-similarity comparisons where cliffs are worse",
                    int((strict.delta < 0).sum()), 0, "5.18"))
        for defn in ("morgan_only_0.7", "morgan_only_0.8"):
            d = ct[(ct.definition == defn) & (ct.target == "cdk2")]
            out.append((f"CDK2 {defn} Holm-significant", int((d.p_holm < 0.05).sum()), 0, "5.18"))
        d = ct[(ct.definition == "morgan_only_0.7") & (ct.target == "maob")]
        out.append(("MAO-B morgan_only_0.7 Holm-significant",
                    int((d.p_holm < 0.05).sum()), 0, "5.18"))

    # --- 5.20: where the errors live ------------------------------------------------------
    if os.path.exists("results/error_concordance.csv"):
        ec = pd.read_csv("results/error_concordance.csv")
        conc = ec[ec.kind == "model_concordance"].set_index(["target", "split"])
        perm = ec[ec.kind == "publication_permutation"].set_index(["target", "split"])
        for tgt, arm in (("tg2", "scaffold_pub"), ("cdk2", "scaffold_pub"), ("maob", "scaffold_pub")):
            r = conc.loc[(tgt, arm)]
            out.append((f"{tgt} molecules almost all models miss, pct",
                        r.frac_molecules_almost_all_models_wrong * 100, 1, "5.20"))
            out.append((f"{tgt} that null expectation, pct", r.null_expectation_all * 100, 1, "5.20"))
        r = conc.loc[("tg2", "scaffold_pub")]
        out.append(("TG2 molecules no model misses, pct",
                    r.frac_molecules_no_model_wrong * 100, 1, "5.20"))
        out.append(("TG2 that null expectation, pct", r.null_expectation_none * 100, 1, "5.20"))
        for tgt, arm in (("tg2", "scaffold_pub"), ("cdk2", "paper_grouped"), ("maob", "paper_grouped")):
            r = perm.loc[(tgt, arm)]
            out.append((f"{tgt} publication error spread", r.observed_spread, 3, "5.20"))
            out.append((f"{tgt} publication error spread null", r.null_spread_mean, 3, "5.20"))
        bal = pd.concat([pd.read_csv(f"results/error_balance_{t}.csv")
                         for t in ("tg2", "cdk2", "maob")], ignore_index=True)
        b = bal.set_index(["target", "split", "rate"])
        for tgt, arm in (("tg2", "scaffold_pub"), ("cdk2", "scaffold_pub"),
                         ("maob", "scaffold_pub"), ("cdk2", "paper_grouped"),
                         ("maob", "paper_grouped")):
            for rate in ("fn_rate", "fp_rate"):
                out.append((f"{tgt} {arm} {rate} vs active fraction",
                            b.loc[(tgt, arm, rate)].spearman_vs_active_frac, 3, "5.20"))
        ch = pd.concat([pd.read_csv(f"results/error_chemistry_{t}.csv")
                        for t in ("tg2", "cdk2", "maob")], ignore_index=True)
        novel = ch[(ch.variable == "max Tanimoto to training") & (ch.band == "[0.0, 0.4)")]
        # MAO-B is quoted as a percentage in §5.20, TG2 and CDK2 as decimals; the claim has to
        # match the form the sentence uses, since locate() searches for the digit string.
        for tgt, arm, as_pct in (("tg2", "paper_lopo", False), ("cdk2", "paper_grouped", False),
                                 ("maob", "paper_lopo", True)):
            r = novel[(novel.target == tgt) & (novel.split == arm)].iloc[0]
            scale, dp = (100, 1) if as_pct else (1, 3)
            out.append((f"{tgt} novel-band FN rate", r.fn_rate * scale, dp, "5.20"))
            out.append((f"{tgt} novel-band FP rate", r.fp_rate * scale, dp, "5.20"))
        cut = ch[ch.variable == "distance to 1 uM cut"]
        for tgt, arm in (("tg2", "scaffold_pub"), ("cdk2", "scaffold_pub"), ("maob", "scaffold_pub")):
            d = cut[(cut.target == tgt) & (cut.split == arm)].set_index("band")
            out.append((f"{tgt} error rate nearest the cut", d.loc["[0.0, 0.25)"].error_rate, 3, "5.20"))
            out.append((f"{tgt} error rate a log from the cut",
                        d.loc["[1.0, 2.0)"].error_rate, 3, "5.20"))

    # --- 5.19: the validation-fold class balance null -------------------------------------
    if os.path.exists("results/fold_balance_slopes.csv"):
        sl = pd.read_csv("results/fold_balance_slopes.csv")
        out.append(("fold balance largest TG2 r2", sl[sl.target == "tg2"].r2.max(), 3, "5.19"))
        out.append(("fold balance control mean slope",
                    sl[sl.group == "control"].slope.mean(), 3, "5.19"))
        out.append(("fold balance early-stopping mean slope",
                    sl[sl.group == "uses_valid_for_training"].slope.mean(), 3, "5.19"))
        v = pd.read_csv("results/fold_balance_variance.csv").set_index(["target", "split"])
        for (tgt, arm) in (("tg2", "scaffold"), ("tg2", "scaffold_pub"), ("maob", "scaffold_pub")):
            out.append((f"{tgt} {arm} skill-vs-variance r",
                        v.loc[(tgt, arm)].r_mean_vs_sd, 3, "5.19"))

    # --- 8: the random-forest bit-reproducibility limitation ------------------------------
    # The two AUROCs are quoted to seven decimals deliberately: fewer digits would not show
    # that they differ, which is the whole point of the limitation.
    if os.path.exists("results/rf_determinism.csv"):
        d = pd.read_csv("results/rf_determinism.csv").iloc[0]
        out.append(("rf determinism lower auroc", d.auroc_low, 7, "8"))
        out.append(("rf determinism upper auroc", d.auroc_high, 7, "8"))
        out.append(("rf determinism pos-neg pairs", d.pairs, 0, "8"))

    # --- 5.21: the overfitting decomposition -----------------------------------------------
    # The cell file carries bootstrap intervals; the fold counts and means come from the raw
    # per-fold output so that a resampling seed cannot move a number the paper states.
    if os.path.exists("results/overfitting_tier1.csv"):
        f = pd.read_csv("results/overfitting_tier1.csv")
        scored = f.dropna(subset=["test_small"])
        lopo = scored[scored.target.isin(["cdk2", "maob"]) & (scored.split == "paper_lopo")]
        out.append(("5.21 folds measured", len(f), 0, "5.21"))
        out.append(("5.21 folds with a committed score", len(scored), 0, "5.21"))
        out.append(("5.21 folds with one test class", len(f) - len(scored), 0, "5.21"))
        out.append(("5.21 cdk2+maob lopo folds", len(lopo), 0, "5.21"))
        out.append(("5.21 cdk2+maob lopo shift_gap", lopo.shift_gap.mean(), 3, "5.21"))
        out.append(("5.21 cdk2+maob lopo size_effect", lopo.size_effect.mean(), 3, "5.21"))
        out.append(("5.21 cdk2+maob lopo folds positive", int((lopo.shift_gap > 0).sum()), 0, "5.21"))
        for m, med in scored.groupby("model").in_sample.median().items():
            out.append((f"5.21 {m} median in-sample", med, 3, "5.21"))
    if os.path.exists("results/overfitting_tier1_cells.csv"):
        c = pd.read_csv("results/overfitting_tier1_cells.csv").set_index("target")
        out.append(("5.21 tg2 lopo pooled shift_gap", c.loc["tg2 paper_lopo pooled"].shift_gap, 3, "5.21"))
        out.append(("5.21 tg2 lopo pooled size_effect", c.loc["tg2 paper_lopo pooled"].size_effect, 3, "5.21"))
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
    # Thousands separators are stripped as well as spaces and the Unicode minus, so a claim of
    # 1840 matches a paper that writes "1,840". Stripping can only add matches, never remove one.
    flat = text.replace("−", "-").replace(" ", "").replace(",", "")
    if anchor is None:
        scope, scope_label = flat, ""
    else:
        heads = re.split(r"\n(?=#{2,3} )", text)
        hit = [q for q in heads
               if q.split("\n", 1)[0].lstrip("# ").strip().startswith(anchor)]
        if not hit:
            raise LookupError(f"no section of {PAPER} starts with {anchor!r}")
        scope = hit[0].replace("−", "-").replace(" ", "").replace(",", "")
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
