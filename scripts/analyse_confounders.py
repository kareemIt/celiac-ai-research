"""What, other than the held-out publication, could explain the penalty?

Stage 2's accounting. The publication penalty is only interesting if it is not just one of the
things a publication split changes incidentally. Four candidates are checked here, each against
committed result files -- nothing is fitted:

  training-set size   Both arms are built to the same size, so this is a check that the design
                      held, not a result. If the sizes had drifted, every penalty below would be
                      partly a learning-curve effect.
  class balance       Every molecule is tested exactly once in both arms, so the MEAN positive
                      rate is identical by construction. What differs is its spread: publications
                      are enriched for one outcome, so publication folds vary more. Two
                      adjustments test whether that spread carries the penalty -- restricting both
                      arms to the positive-rate band they share, and a quadratic adjustment for
                      positive rate with the arm as a dummy.
  fold assignment     Between-deal against within-deal spread, from audit_fold_sensitivity.
  chemical novelty    Penalty by max-Tanimoto band. If the penalty were only a similarity effect,
                      it would vanish once novelty is held fixed.

The assay confound is NOT here: scripts/audit_assay_confound.py establishes that assays are
perfectly nested inside publications, so it cannot be separated at all. That is a limitation of
ChEMBL-derived data, not something this script can adjust away.

Reproduce: python scripts/analyse_confounders.py
  -> results/confounders.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT = "results/confounders.csv"
SENS = "results/fold_sensitivity_{target}.csv"
BANDS = "results/sota_{target}_bands_{split}.csv"
ARMS = ("paper_grouped", "scaffold_matched")


def training_size(target):
    """Mean training and test size per arm, from the Stage 1 sweep."""
    from scripts.run_sota import paths
    _, sota, _ = paths(target)
    if not os.path.exists(sota):
        return None
    d = pd.read_csv(sota)
    d = d[d["split"].isin(["scaffold_pub", "paper_grouped"])]
    return d.groupby("split").agg(n_train=("n_train", "mean"), n_test=("n_test", "mean"))


def balance_adjusted(g):
    """The arm effect on AUROC, raw and with the fold's positive rate taken out two ways.

    `g` is one target and model, both arms, one row per fold per deal.
    """
    pg, sm = (g[g["arm"] == a] for a in ARMS)
    if pg.empty or sm.empty:
        return None
    raw = pg["auroc"].mean() - sm["auroc"].mean()

    # 1. the positive-rate band both arms cover, so no fold is compared against an unmatched one
    lo = max(pg["frac_positive"].min(), sm["frac_positive"].min())
    hi = min(pg["frac_positive"].max(), sm["frac_positive"].max())
    a = pg[pg["frac_positive"].between(lo, hi)]
    b = sm[sm["frac_positive"].between(lo, hi)]
    band = a["auroc"].mean() - b["auroc"].mean() if len(a) and len(b) else np.nan

    # 2. a quadratic in positive rate, with the arm as a dummy; the dummy is the adjusted penalty
    pos = g["frac_positive"].to_numpy()
    X = np.column_stack([np.ones(len(g)), (g["arm"] == ARMS[0]).to_numpy(float), pos, pos ** 2])
    beta = np.linalg.lstsq(X, g["auroc"].to_numpy(), rcond=None)[0]

    r = spearmanr(pg["frac_positive"], pg["auroc"])
    return {
        "penalty_raw": raw, "penalty_shared_band": band, "penalty_balance_adjusted": beta[1],
        "band_lo": lo, "band_hi": hi, "n_band_pub": len(a), "n_band_scaffold": len(b),
        "pos_rate_mean_pub": pg["frac_positive"].mean(),
        "pos_rate_mean_scaffold": sm["frac_positive"].mean(),
        "pos_rate_sd_pub": pg["frac_positive"].std(ddof=1),
        "pos_rate_sd_scaffold": sm["frac_positive"].std(ddof=1),
        "rho_pos_auroc_pub": r.statistic, "p_rho_pub": r.pvalue,
    }


def fold_assignment(g):
    """Between-deal against within-deal spread, per arm."""
    out = {}
    for arm in ARMS:
        a = g[g["arm"] == arm]
        if a.empty:
            continue
        per_deal = a.groupby("deal")["auroc"].mean()
        out[arm] = {"mean": per_deal.mean(), "between_deal_sd": per_deal.std(ddof=1),
                    "within_deal_sd": a.groupby("deal")["auroc"].std(ddof=1).mean(),
                    "deals": len(per_deal)}
    return out


def novelty(target):
    """Penalty by max-Tanimoto band: is the penalty only a chemical-similarity effect?"""
    pub = BANDS.format(target=target, split="paper_grouped")
    sca = BANDS.format(target=target, split="scaffold_pub")
    if not (os.path.exists(pub) and os.path.exists(sca)):
        return None
    a = pd.read_csv(pub).groupby("band").agg(auroc_pub=("auroc", "mean"),
                                             n_pub=("n_molecules", "first"))
    b = pd.read_csv(sca).groupby("band").agg(auroc_scaffold=("auroc", "mean"),
                                             n_scaffold=("n_molecules", "first"))
    out = a.join(b, how="inner")
    out["penalty"] = out["auroc_pub"] - out["auroc_scaffold"]
    return out


def standardise(target):
    """How much of the penalty is just a different mix of chemical novelty?

    A publication split does not only hide the publication: it also shifts the test set towards
    novel chemistry, because a whole paper's series goes out at once. Direct standardisation
    separates the two. The within-band penalty is averaged under each arm's own band weights, so
    the composition difference is removed and only the within-band gap is left. If the raw penalty
    is large and both standardised penalties are near zero, the decline is a novelty effect that a
    publication split produces incidentally, not an effect of publication identity.

    Crude on two counts: five bands, and molecules within one band may still differ in novelty.
    It bounds the composition share rather than measuring it exactly.
    """
    pub = BANDS.format(target=target, split="paper_grouped")
    sca = BANDS.format(target=target, split="scaffold_pub")
    if not (os.path.exists(pub) and os.path.exists(sca)):
        return None, None
    a, b = pd.read_csv(pub), pd.read_csv(sca)
    n_pub = a.groupby("band")["n_molecules"].first()
    n_sca = b.groupby("band")["n_molecules"].first()
    comp = pd.DataFrame({"pub_n": n_pub, "scaffold_n": n_sca})
    comp["pub_frac"] = n_pub / n_pub.sum()
    comp["scaffold_frac"] = n_sca / n_sca.sum()

    rows = []
    for model in sorted(set(a["model"]) & set(b["model"])):
        aa = a[a["model"] == model].set_index("band")["auroc"]
        bb = b[b["model"] == model].set_index("band")["auroc"]
        bands = aa.index.intersection(bb.index)
        if bands.empty:
            continue
        gap = aa.reindex(bands) - bb.reindex(bands)
        w_pub = (comp["pub_frac"].reindex(bands) / comp["pub_frac"].reindex(bands).sum())
        w_sca = (comp["scaffold_frac"].reindex(bands) / comp["scaffold_frac"].reindex(bands).sum())
        raw = float((aa.reindex(bands) * w_pub).sum() - (bb.reindex(bands) * w_sca).sum())
        rows.append({"model": model, "penalty_raw": raw,
                     "penalty_std_scaffold_mix": float((gap * w_sca).sum()),
                     "penalty_std_pub_mix": float((gap * w_pub).sum()), "bands": len(bands)})
    return comp, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["cdk2", "maob"])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    rows = []
    for target in args.targets:
        print(f"\n=== {target}")

        size = training_size(target)
        if size is not None:
            print("\n  training-set size per arm (matched by construction):")
            print("   " + "   ".join(f"{s} train {r.n_train:.0f} test {r.n_test:.0f}"
                                     for s, r in size.iterrows()))
            drift = size["n_train"].max() - size["n_train"].min()
            print(f"    drift {drift:.0f} molecules "
                  f"({drift / size['n_train'].mean():.1%} of the training set)")

        path = SENS.format(target=target)
        if not os.path.exists(path):
            print(f"  no {path}; run scripts/audit_fold_sensitivity.py --target {target}")
            continue
        sens = pd.read_csv(path)

        print("\n  class balance: does the arm effect survive adjusting for the positive rate?")
        print(f"   {'model':<9}{'raw':>8}{'shared band':>14}{'adjusted':>11}"
              f"{'SD pos pub':>13}{'SD pos scaf':>13}")
        for model, g in sens.groupby("model"):
            b = balance_adjusted(g)
            if b is None:
                continue
            print(f"   {model:<9}{b['penalty_raw']:>+8.3f}{b['penalty_shared_band']:>+14.3f}"
                  f"{b['penalty_balance_adjusted']:>+11.3f}"
                  f"{b['pos_rate_sd_pub']:>13.3f}{b['pos_rate_sd_scaffold']:>13.3f}")
            rows.append({"target": target, "model": model, "section": "balance", **b})

        print("\n  fold assignment: between-deal against within-deal spread")
        for model, g in sens.groupby("model"):
            for arm, s in fold_assignment(g).items():
                ratio = s["within_deal_sd"] / s["between_deal_sd"] if s["between_deal_sd"] else np.nan
                print(f"   {model:<9}{arm:<18} mean {s['mean']:.3f}  between {s['between_deal_sd']:.4f}"
                      f"  within {s['within_deal_sd']:.4f}  ratio {ratio:.1f}x")
                rows.append({"target": target, "model": model, "section": "fold_assignment",
                             "arm": arm, **s, "within_over_between": ratio})

        nov = novelty(target)
        if nov is not None:
            print("\n  chemical novelty: penalty within each max-Tanimoto band")
            print(nov.round(3).to_string())
            for band, r in nov.iterrows():
                rows.append({"target": target, "section": "novelty", "band": band, **r.to_dict()})

        comp, std = standardise(target)
        if comp is not None and not std.empty:
            print("\n  novelty composition of each arm's test set (share of molecules)")
            print("   " + "  ".join(f"{b} pub {r.pub_frac:.0%} scaffold {r.scaffold_frac:.0%}"
                                    for b, r in comp.iterrows()))
            print("\n  penalty, raw against standardised to each arm's novelty mix")
            print(f"   {'model':<18}{'raw':>8}{'std scaffold mix':>19}{'std pub mix':>14}"
                  f"{'composition share':>20}")
            for _, r in std.iterrows():
                share = (1 - abs(r["penalty_std_scaffold_mix"]) / abs(r["penalty_raw"])
                         if r["penalty_raw"] else np.nan)
                print(f"   {r['model']:<18}{r['penalty_raw']:>+8.3f}"
                      f"{r['penalty_std_scaffold_mix']:>+19.3f}"
                      f"{r['penalty_std_pub_mix']:>+14.3f}{share:>19.0%}")
                rows.append({"target": target, "section": "standardised", **r.to_dict(),
                             "composition_share": share})
            for b, r in comp.iterrows():
                rows.append({"target": target, "section": "composition", "band": b,
                             **r.to_dict()})

    if rows:
        os.makedirs("results", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\n-> {args.out}")
    print("\n  The assay confound is not adjustable: assays are perfectly nested inside"
          "\n  publications (scripts/audit_assay_confound.py), so the publication penalty is"
          "\n  inseparably also an assay penalty on ChEMBL-derived data.")


if __name__ == "__main__":
    main()
