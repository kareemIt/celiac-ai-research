"""Why do leave-one-publication-out and grouped publication K-fold disagree by so much?

On CDK2 the random forest scores 0.617 under leave-one-publication-out and 0.786 under grouped
K-fold -- a gap of 0.169, larger than the publication penalty the two protocols were built to
measure. Both hold out whole publications, both run on the same 2,078 molecules, and each tests
every molecule exactly once. So the gap is not about what is hidden; it is about how the folds are
cut and how the per-fold scores are combined. Four candidates, separated here:

  metric aggregation    Leave-one-out macro-averages 63 publication-level AUROCs; grouped averages
                        10. A macro average over folds answers "how well does this do on a typical
                        PUBLICATION", a pooled molecule-level AUROC answers "how well does this
                        rank a typical MOLECULE". They are different questions and the second is
                        comparable across protocols, because both protocols test each molecule once.
  publication size      Leave-one-out folds hold 10 to 135 molecules; grouped folds hold ~208. An
                        unweighted macro average lets a 10-molecule publication count as much as a
                        135-molecule one, and a small fold's AUROC is both noisier and -- if small
                        publications are focused series -- genuinely harder.
  single-class folds    20 of CDK2's 83 publications report only potent or only inactive compounds,
                        so AUROC is undefined and those folds drop out. That is a selection on
                        chemistry, not a random loss.
  training-set size     Leave-one-out trains on more data (median 1,856 against 1,685), so this
                        works AGAINST leave-one-out scoring lower and cannot explain the gap. It is
                        measured to confirm the direction.

Molecule-level pooled AUROC is computed by pooling every out-of-fold prediction and scoring once.
This is only legitimate because each molecule is tested exactly once under both protocols; where a
molecule appears in several folds (the scaffold arm, with its repeated seeds) pooling would mix
correlated predictions and is not reported.

Reads only committed result files; fits nothing.

Reproduce: python scripts/analyse_protocol_gap.py --targets cdk2 maob
  -> results/protocol_gap.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT = "results/protocol_gap.csv"
PROTOCOLS = ("paper_lopo", "paper_grouped")


def fold_table(target, split):
    """Per-fold metrics as the sweep recorded them, with the undefined folds still visible."""
    from scripts.run_sota import paths
    _, sota, _ = paths(target)
    d = pd.read_csv(sota)
    return d[d["split"] == split]


def predictions(target, split):
    """Out-of-fold predictions, one row per molecule per model, with the fold it came from."""
    from scripts.run_sota import paths
    splits_path, _, preds_path = paths(target)
    p = pd.read_csv(preds_path)
    p = p[p["split"] == split]
    if p.empty:
        return p
    sp = pd.read_csv(splits_path, low_memory=False)
    sp = sp[sp["split_kind"] == split]
    paper = sp.drop_duplicates("smiles").set_index("smiles")["paper"]
    return p.assign(paper=p["smiles"].map(paper))


def tested_once(p):
    """True when every molecule appears in exactly one fold, so pooling is legitimate."""
    if p.empty:
        return False
    per_model = p.groupby(["model", "smiles"]).size()
    return bool((per_model == 1).all())


def pooled_auroc(p, keep=None):
    """One AUROC per model over all pooled out-of-fold predictions."""
    out = {}
    for model, g in p.groupby("model"):
        if keep is not None:
            g = g[g["smiles"].isin(keep)]
        if len(g) < 10 or g["y"].nunique() < 2:
            continue
        out[model] = float(roc_auc_score(g["y"], g["p"]))
    return out


def usable_molecules(p, by="paper"):
    """Molecules whose own fold had both classes -- the set leave-one-out can actually score.

    Under leave-one-publication-out a fold IS a publication, so a single-class publication's
    molecules are exactly those that drop out of the macro average.
    """
    keep = set()
    for model, g in p.groupby("model"):
        for _, gg in g.groupby(by):
            if gg["y"].nunique() > 1:
                keep |= set(gg["smiles"])
        break                              # the grouping is the same for every model
    return keep


def summarise(target):
    rows = []
    cache = {}
    for split in PROTOCOLS:
        folds = fold_table(target, split)
        if folds.empty:
            continue
        p = predictions(target, split)
        cache[split] = (folds, p)
        once = tested_once(p)
        keep = usable_molecules(p) if not p.empty else set()
        pooled_all = pooled_auroc(p) if once else {}
        pooled_keep = pooled_auroc(p, keep) if once else {}

        for model, g in folds.groupby("model"):
            ok = g.dropna(subset=["auroc"])
            if ok.empty:
                continue
            rows.append({
                "target": target, "split": split, "model": model,
                "folds_total": len(g), "folds_scored": len(ok),
                "folds_undefined": int(g["auroc"].isna().sum()),
                "macro_unweighted": float(ok["auroc"].mean()),
                "macro_size_weighted": float(np.average(ok["auroc"], weights=ok["n_test"])),
                "macro_sd": float(ok["auroc"].std(ddof=1)),
                "pooled_molecule": pooled_all.get(model, np.nan),
                "pooled_molecule_scored_folds_only": pooled_keep.get(model, np.nan),
                "n_test_median": float(g["n_test"].median()),
                "n_test_min": int(g["n_test"].min()), "n_test_max": int(g["n_test"].max()),
                "n_train_median": float(g["n_train"].median()),
                "molecules_scored": len(keep) if once else np.nan,
                "molecules_total": int(p["smiles"].nunique()) if not p.empty else np.nan,
                "tested_once": once,
            })
    return pd.DataFrame(rows), cache


def within_publication_macro(p, min_per_paper=10):
    """Macro-average of per-PUBLICATION AUROCs, computed inside whichever protocol's folds.

    This is the confirmatory test for the aggregation explanation. Leave-one-out's macro average
    is unavoidably a mean of within-publication AUROCs, because each of its folds IS one
    publication. Grouped K-fold's folds each contain about twenty publications, so its macro
    average is over mixed folds -- but the same predictions can be re-aggregated by publication.

    If re-aggregating the GROUPED arm by publication reproduces leave-one-out's number, then the
    two protocols never disagreed about the model: they disagreed about whether the question is
    "rank molecules within one paper's series" or "rank molecules across the whole set". Ranking
    inside a congeneric series is the harder question, because the easy between-series contrasts
    that inflate a global AUROC are gone.
    """
    out = {}
    for model, g in p.groupby("model"):
        per = []
        for _, gg in g.groupby("paper"):
            if len(gg) < min_per_paper or gg["y"].nunique() < 2:
                continue
            per.append((roc_auc_score(gg["y"], gg["p"]), len(gg)))
        if len(per) < 3:
            continue
        a = np.array([x[0] for x in per])
        w = np.array([x[1] for x in per], float)
        out[model] = {"within_pub_macro": float(a.mean()),
                      "within_pub_macro_weighted": float(np.average(a, weights=w)),
                      "publications_scored": len(per)}
    return out


def per_publication_auroc(p, min_per_paper=10):
    """{model: {publication: (auroc, n)}} -- one AUROC per publication, per model."""
    out = {}
    for model, g in p.groupby("model"):
        d = {}
        for pub, gg in g.groupby("paper"):
            if len(gg) >= min_per_paper and gg["y"].nunique() > 1:
                d[pub] = (roc_auc_score(gg["y"], gg["p"]), len(gg))
        out[model] = d
    return out


def one_row_per_molecule(p):
    """Average a molecule's probability over the folds that tested it.

    The scaffold arm runs repeated seeds, so a molecule can be tested up to five times. Averaging
    first is what analyse_sota.bands does and is what makes the scaffold arm comparable to a
    publication arm that tests each molecule once.
    """
    return p.groupby(["model", "smiles"]).agg(
        y=("y", "first"), p=("p", "mean"), paper=("paper", "first")).reset_index()


def matched_within_publication(target, scaffold_arm="scaffold_pub", pub_arm=None,
                               min_per_paper=10):
    """The publication penalty with the aggregation unit AND the publications held fixed.

    The headline penalty compares a molecule-pooled scaffold score against a publication-macro
    publication-arm score, so it mixes two changes: whether the publication was seen, and whether
    the question is "rank molecules globally" or "rank molecules inside one series". This function
    removes both confounds. It scores each publication's own molecules in both arms, keeps only
    the publications both arms can score, and pairs them.

    The contrast is then exactly: can a model rank compounds within a paper's series better when
    it has seen part of that series than when it has not?

    Paired across publications, which is legitimate here because a publication is the same set of
    molecules in both arms -- unlike fold index, which means different molecules in each.
    """
    from scipy.stats import ttest_rel
    pub_arm = pub_arm or ("paper_lopo" if target == "tg2" else "paper_grouped")
    a_raw, b_raw = predictions(target, scaffold_arm), predictions(target, pub_arm)
    if a_raw.empty or b_raw.empty:
        return None
    A = per_publication_auroc(one_row_per_molecule(a_raw), min_per_paper)
    B = per_publication_auroc(one_row_per_molecule(b_raw), min_per_paper)

    rows = []
    for model in sorted(set(A) & set(B)):
        shared = sorted(set(A[model]) & set(B[model]))
        if len(shared) < 5:
            continue
        a = np.array([A[model][k][0] for k in shared])
        b = np.array([B[model][k][0] for k in shared])
        tt = ttest_rel(b, a)
        ci = tt.confidence_interval(0.95)
        rows.append({"target": target, "model": model, "scaffold_arm": scaffold_arm,
                     "pub_arm": pub_arm, "publications": len(shared),
                     "within_pub_scaffold": float(a.mean()),
                     "within_pub_publication": float(b.mean()),
                     "penalty": float(b.mean() - a.mean()),
                     "ci_lo": float(ci.low), "ci_hi": float(ci.high),
                     "p_paired": float(tt.pvalue)})
    return pd.DataFrame(rows) if rows else None


def size_effect(cache, split="paper_lopo"):
    """Does a fold's AUROC depend on how many molecules the publication contributed?"""
    from scipy.stats import spearmanr
    if split not in cache:
        return None
    folds = cache[split][0].dropna(subset=["auroc"])
    out = []
    for model, g in folds.groupby("model"):
        if len(g) < 5:
            continue
        r = spearmanr(g["n_test"], g["auroc"])
        small = g[g["n_test"] <= g["n_test"].median()]["auroc"].mean()
        large = g[g["n_test"] > g["n_test"].median()]["auroc"].mean()
        out.append({"model": model, "rho_size_auroc": r.statistic, "p": r.pvalue,
                    "auroc_small_folds": small, "auroc_large_folds": large,
                    "n_folds": len(g)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["cdk2", "maob"])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    allrows = []
    for target in args.targets:
        tab, cache = summarise(target)
        if tab.empty:
            print(f"\n=== {target}: no fold metrics for {PROTOCOLS}")
            continue
        allrows.append(tab)
        print(f"\n=== {target}")
        for split in PROTOCOLS:
            s = tab[tab["split"] == split]
            if s.empty:
                print(f"\n  {split}: not run")
                continue
            r0 = s.iloc[0]
            print(f"\n  {split}: {int(r0['folds_total'])} folds, "
                  f"{int(r0['folds_undefined'])} undefined, test size "
                  f"{r0['n_test_min']}-{r0['n_test_max']} (median {r0['n_test_median']:.0f}), "
                  f"train median {r0['n_train_median']:.0f}")
            print(f"   every molecule tested exactly once: {bool(r0['tested_once'])}"
                  + (f"; {int(r0['molecules_scored'])} of {int(r0['molecules_total'])} molecules "
                     f"sit in a scorable fold" if r0["tested_once"] else ""))
            print(f"   {'model':<9}{'macro':>9}{'macro wtd':>11}{'pooled':>9}"
                  f"{'pooled scored':>15}")
            for _, r in s.iterrows():
                print(f"   {r['model']:<9}{r['macro_unweighted']:>9.3f}"
                      f"{r['macro_size_weighted']:>11.3f}{r['pooled_molecule']:>9.3f}"
                      f"{r['pooled_molecule_scored_folds_only']:>15.3f}")

        wide = tab.pivot_table(index="model", columns="split",
                               values=["macro_unweighted", "macro_size_weighted",
                                       "pooled_molecule"])
        if all(s in tab["split"].values for s in PROTOCOLS):
            print(f"\n  the gap, by how the score is formed (grouped minus leave-one-out)")
            print(f"   {'model':<9}{'macro':>9}{'macro wtd':>11}{'pooled':>9}")
            for model in wide.index:
                vals = []
                for key in ("macro_unweighted", "macro_size_weighted", "pooled_molecule"):
                    try:
                        vals.append(wide.loc[model, (key, "paper_grouped")]
                                    - wide.loc[model, (key, "paper_lopo")])
                    except KeyError:
                        vals.append(np.nan)
                print(f"   {model:<9}" + "".join(f"{v:>+9.3f}" if i == 0 else
                                                 (f"{v:>+11.3f}" if i == 1 else f"{v:>+9.3f}")
                                                 for i, v in enumerate(vals)))

        print("\n  the same predictions re-aggregated BY PUBLICATION, within each protocol")
        print(f"   {'split':<15}{'model':<9}{'within-pub macro':>18}{'pooled':>9}"
              f"{'publications':>14}")
        for split in PROTOCOLS:
            if split not in cache:
                continue
            p = cache[split][1]
            if p.empty:
                continue
            wp = within_publication_macro(p)
            pooled = pooled_auroc(p) if tested_once(p) else {}
            for model, v in sorted(wp.items()):
                print(f"   {split:<15}{model:<9}{v['within_pub_macro']:>18.3f}"
                      f"{pooled.get(model, np.nan):>9.3f}{v['publications_scored']:>14}")
                allrows.append(pd.DataFrame([{"target": target, "section": "within_publication",
                                              "split": split, "model": model,
                                              "pooled_molecule": pooled.get(model, np.nan), **v}]))

        mw = matched_within_publication(target)
        if mw is not None:
            print(f"\n  the publication penalty with BOTH the aggregation unit and the"
                  f"\n  publications held fixed ({int(mw['publications'].iloc[0])} publications, "
                  f"paired)")
            print(f"   {'model':<18}{'scaffold':>10}{'publication':>13}{'penalty':>10}"
                  f"{'95% CI':>20}{'p':>8}")
            for _, r in mw.iterrows():
                print(f"   {r['model']:<18}{r['within_pub_scaffold']:>10.3f}"
                      f"{r['within_pub_publication']:>13.3f}{r['penalty']:>+10.3f}"
                      f"{f'[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]':>20}{r['p_paired']:>8.3f}")
            allrows.append(mw.assign(section="matched_within_publication"))

        se = size_effect(cache)
        if se is not None and not se.empty:
            print(f"\n  publication size against fold AUROC, leave-one-out")
            print(f"   {'model':<9}{'rho':>8}{'p':>8}{'small folds':>13}{'large folds':>13}")
            for _, r in se.iterrows():
                print(f"   {r['model']:<9}{r['rho_size_auroc']:>+8.3f}{r['p']:>8.3f}"
                      f"{r['auroc_small_folds']:>13.3f}{r['auroc_large_folds']:>13.3f}")
                allrows.append(pd.DataFrame([{"target": target, "section": "size_effect",
                                              **r.to_dict()}]))

    if allrows:
        os.makedirs("results", exist_ok=True)
        pd.concat(allrows, ignore_index=True).to_csv(args.out, index=False)
        print(f"\n-> {args.out}")
    print("\n  Macro and pooled AUROC answer different questions. Macro over folds is the score on"
          "\n  a typical PUBLICATION and is what a screening campaign facing one new paper would"
          "\n  see; pooled molecule-level AUROC is the score on a typical MOLECULE and is the only"
          "\n  one comparable across protocols here. Neither is the right one in general.")


if __name__ == "__main__":
    main()
