"""Experiment 1: per-class error structure from saved predictions. No retraining, no GPU.

Reads results/sota_preds*.csv (per-molecule y and p for every model x split arm x seed) and
reports sensitivity, specificity, FPR, FNR, PPV, NPV, accuracy, balanced accuracy and F1 at three
operating points fixed before any output was read:

  t050      the 0.5 cut the project reports today (via MCC)
  prior     the training fold's own prevalence, read from results/splits_<target>.csv; uses no
            test information
  oracle    the cut maximising balanced accuracy on the test fold itself: an upper bound on what
            any threshold rule could reach, not an achievable operating point

A balanced-accuracy cut fitted on the training fold was the third point in the plan. Training-fold
predictions were never saved, so it would need a retrain; `oracle` replaces it and brackets it from
above.

Usage: python scripts/analyse_per_class.py [--target tg2|cdk2|maob] ...
Outputs: results/per_class_<target>.csv, results/per_class_<target>_summary.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metrics import (_ratio, best_balanced_threshold,  # noqa: E402
                                    classification, per_class)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TARGETS = {
    "tg2": ("sota_preds.csv", "sota_tg2.csv", "splits_tg2.csv"),
    "cdk2": ("sota_preds_cdk2.csv", "sota_cdk2.csv", "splits_cdk2.csv"),
    "maob": ("sota_preds_maob.csv", "sota_maob.csv", "splits_maob.csv"),
}
TOL = 1e-9


def train_prevalence(splits):
    train = splits[splits.fold == "train"]
    return train.groupby(["split_kind", "seed"]).y.mean()


def verify(rows, committed, target):
    """Gate: AUROC and MCC@0.5 recomputed from the predictions must match the committed tables,
    or the loader is joining the wrong rows. Folds the committed run dropped are skipped."""
    ours = rows[rows.threshold_kind == "t050"].set_index(["model", "split", "seed"])
    theirs = committed.set_index(["model", "split", "seed"])
    shared = ours.index.intersection(theirs.index)
    if not len(shared):
        raise SystemExit(f"{target}: no folds in common with {TARGETS[target][1]}")
    bad = []
    for col in ("auroc", "mcc"):
        a, b = ours.loc[shared, col], theirs.loc[shared, col]
        diff = (a - b).abs()
        diff = diff[~(a.isna() & b.isna())]
        worst = diff.max()
        if worst > TOL:
            bad.append(f"{col}: max |diff| {worst:.2e} over {len(diff)} folds")
    if bad:
        raise SystemExit(f"{target}: recomputation does not match committed results -- " + "; ".join(bad))
    return len(shared)


def by_publication(target, preds, splits, thresholds):
    """The same rates, re-aggregated on the publication rather than the fold.

    docs/audit_2026-11-05.md correction #9: a publication-held-out arm is conventionally averaged
    per publication while a scaffold arm is averaged per molecule, and about 92% of CDK2's
    protocol gap is that difference rather than generalisation. A leave-one-publication-out fold
    holds exactly one publication, so its fold average already *is* a publication average; a
    scaffold_pub fold holds 4 to 113. Without this re-aggregation the two arms are not comparable.

    A publication that reports only actives gives no specificity and one that reports only
    inactives gives no sensitivity. Those are skipped and counted, never read as zero.
    """
    pub = (splits[splits.fold == "test"]
           .drop_duplicates(["split_kind", "seed", "smiles"])
           .set_index(["split_kind", "seed", "smiles"]).paper)
    rows = []
    for (model, split, seed), g in preds.groupby(["model", "split", "seed"], sort=True):
        g = g.drop_duplicates("smiles")
        papers = pub.reindex(pd.MultiIndex.from_product([[split], [seed], g.smiles])).to_numpy()
        if pd.isna(papers).all():
            continue
        t = thresholds.get((model, split, seed))
        if t is None:
            continue
        for kind, cut in t.items():
            if np.isnan(cut):
                continue
            for paper, idx in pd.Series(range(len(g))).groupby(papers):
                y = g.y.to_numpy()[idx.to_numpy()]
                p = g.p.to_numpy()[idx.to_numpy()]
                rows.append({"target": target, "model": model, "split": split, "seed": seed,
                             "paper": paper, "threshold_kind": kind, "threshold": cut,
                             "n": len(y), "pos_frac": float(y.mean()), **per_class(y, p, cut)})
    return pd.DataFrame(rows)


def run(target):
    preds_f, committed_f, splits_f = TARGETS[target]
    preds = pd.read_csv(RESULTS / preds_f)
    splits = pd.read_csv(RESULTS / splits_f, low_memory=False)
    prev = train_prevalence(splits)

    chosen = {}
    rows = []
    for (model, split, seed), g in preds.groupby(["model", "split", "seed"], sort=True):
        y, p = g.y.to_numpy(), g.p.to_numpy()
        # AUROC is recomputed here only as the verification gate; the committed tables stay the
        # source of truth for threshold-free metrics.
        auroc = classification(y, p)["auroc"]
        thresholds = {
            "t050": 0.5,
            "prior": float(prev.get((split, seed), np.nan)),
            "oracle": best_balanced_threshold(y, p),
        }
        chosen[(model, split, seed)] = thresholds
        for kind, t in thresholds.items():
            if np.isnan(t):
                continue
            rows.append({"target": target, "model": model, "split": split, "seed": seed,
                         "threshold_kind": kind, "threshold": t, "n": len(y),
                         "pos_frac": float(y.mean()), "auroc": auroc, **per_class(y, p, t)})
    rows = pd.DataFrame(rows)

    n_checked = verify(rows, pd.read_csv(RESULTS / committed_f), target)
    rows["achievable"] = rows.threshold_kind != "oracle"
    rows.to_csv(RESULTS / f"per_class_{target}.csv", index=False)

    cols = ["sensitivity", "specificity", "fpr", "fnr", "ppv", "npv", "accuracy",
            "balanced_accuracy", "f1", "pred_pos_rate", "auroc"]
    summary = summarise(rows, cols).assign(agg_unit="fold")
    summary.to_csv(RESULTS / f"per_class_{target}_summary.csv", index=False)

    pubs = by_publication(target, preds, splits, chosen)
    if not pubs.empty:
        pubs["achievable"] = pubs.threshold_kind != "oracle"
        pubs.to_csv(RESULTS / f"per_class_{target}_bypub.csv", index=False)
        pub_sum = summarise(pubs, [c for c in cols if c != "auroc"]).assign(agg_unit="publication")
        pub_sum.to_csv(RESULTS / f"per_class_{target}_bypub_summary.csv", index=False)
        print(f"{target}: {len(rows)} fold rows ({n_checked} verified against {committed_f}), "
              f"{len(pubs)} publication rows")
    else:
        print(f"{target}: {len(rows)} fold rows, {n_checked} verified; no publication labels")
    return rows


SHIFTED = {"tg2": "paper_lopo", "cdk2": "paper_grouped", "maob": "paper_grouped"}


def shift_tests(bypub):
    """Does the class asymmetry survive the matched comparison on the publication unit?

    Reported under both plausible units, because the conclusion turns out to depend on the choice:
      publication       each publication once, averaged over the scaffold folds it appears in.
                        Conservative and the defensible default -- a publication that lands in
                        eight scaffold seeds is not eight independent observations.
      publication_fold  every appearance counted. Anti-conservative; reported so the dependence
                        is visible rather than hidden behind one choice.
    Welch's unpaired t-test, Holm across models within each rate, as elsewhere in the project.
    """
    from scipy.stats import binomtest, ttest_ind
    units = {
        "publication": (bypub.groupby(["target", "model", "split", "paper"])
                        [["sensitivity", "specificity"]].mean().reset_index()),
        "publication_fold": bypub,
    }
    out = []
    for unit, d in units.items():
        for target, shifted in SHIFTED.items():
            t = d[d.target == target]
            for rate in ("sensitivity", "specificity"):
                res = []
                for model in sorted(t.model.unique()):
                    a = t[(t.model == model) & (t.split == "scaffold_pub")][rate].dropna()
                    b = t[(t.model == model) & (t.split == shifted)][rate].dropna()
                    if len(a) < 5 or len(b) < 5:
                        continue
                    res.append({"unit": unit, "target": target, "shifted_arm": shifted,
                                "rate": rate, "model": model, "n_ref": len(a), "n_shifted": len(b),
                                "mean_ref": a.mean(), "mean_shifted": b.mean(),
                                "delta": b.mean() - a.mean(),
                                "p_ttest": ttest_ind(b, a, equal_var=False).pvalue})
                if not res:
                    continue
                res = pd.DataFrame(res)
                res["p_holm"] = _holm(res.p_ttest.to_numpy())
                fall = int((res.delta < 0).sum())
                res["n_fall"] = fall
                res["n_models"] = len(res)
                res["sign_test_p"] = binomtest(fall, len(res)).pvalue
                out.append(res)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def pooled_rates(bypub, n_boot=2000, seed=0):
    """Sensitivity and specificity from confusion matrices **summed** over publications.

    The skip-and-average estimator in shift_tests() can only use publications that report both
    classes, and 20% to 52% of them do not (one-class papers). Summing tp/fn/fp/tn across every
    publication first and dividing once afterwards uses all of them: a publication with no actives
    contributes nothing to the sensitivity denominator but still contributes its inactives to
    specificity. The cost is that large publications dominate, which is the molecule-pooled
    weighting §5.14 warns about -- so this is reported *beside* the equal-weight estimator, not
    instead of it.

    Intervals are a cluster bootstrap over publications, matching scripts/analyse_protocol_gap.py.
    """
    rng = np.random.default_rng(seed)
    out = []
    for (target, model, split), g in bypub.groupby(["target", "model", "split"]):
        g = g.groupby("paper")[["tp", "fn", "fp", "tn"]].sum()
        if g.empty:
            continue
        counts = g.to_numpy()
        papers = len(counts)

        def rates(block):
            tp, fn, fp, tn = block.sum(axis=0)
            return (_ratio(tp, tp + fn), _ratio(tn, tn + fp))

        sens, spec = rates(counts)
        draws = np.array([rates(counts[rng.integers(0, papers, papers)]) for _ in range(n_boot)])
        lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
        out.append({"target": target, "model": model, "split": split, "publications": papers,
                    "molecules": int(counts.sum()),
                    "sensitivity": sens, "sensitivity_lo": lo[0], "sensitivity_hi": hi[0],
                    "specificity": spec, "specificity_lo": lo[1], "specificity_hi": hi[1]})
    return pd.DataFrame(out)


def single_class_audit(bypub):
    """How much of the data sits in publications that report only one class, and how big they are.
    The equal-weight estimator discards these units; the pooled estimator keeps them."""
    out = []
    for (target, split), g in bypub.groupby(["target", "split"]):
        one = g[g.model == g.model.iloc[0]].groupby("paper")[["tp", "fn", "fp", "tn"]].sum()
        pos, neg = one.tp + one.fn, one.fp + one.tn
        only_pos, only_neg = (neg == 0), (pos == 0)
        mols = (pos + neg)
        out.append({
            "target": target, "split": split, "publications": len(one),
            "molecules": int(mols.sum()),
            "pub_only_actives": int(only_pos.sum()), "pub_only_inactives": int(only_neg.sum()),
            "pub_single_class_frac": float((only_pos | only_neg).mean()),
            "mol_in_single_class_frac": float(mols[only_pos | only_neg].sum() / mols.sum()),
            "median_size_single_class": float(mols[only_pos | only_neg].median()),
            "median_size_both_classes": float(mols[~(only_pos | only_neg)].median()),
        })
    return pd.DataFrame(out)


def _holm(p):
    out, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        out[i] = running
    return out


def summarise(rows, cols):
    """Mean, SD and the number of units each rate was defined on. `*_undefined` counts the units
    where it had no denominator -- a single-class publication -- because dropping them is itself a
    selection effect and has to be visible."""
    keys = ["target", "model", "split", "threshold_kind", "achievable"]
    keys = [k for k in keys if k in rows]
    out = rows.groupby(keys)[cols].agg(["mean", "std", "count"])
    out.columns = [f"{c}_{s}" for c, s in out.columns]
    total = rows.groupby(keys).size().rename("units")
    out = out.join(total)
    for c in cols:
        out[f"{c}_undefined"] = out.units - out[f"{c}_count"]
    if {"fnr_mean", "fpr_mean"} <= set(out.columns):
        out["fnr_minus_fpr"] = out.fnr_mean - out.fpr_mean
    return out.reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="append", choices=sorted(TARGETS), default=None)
    args = ap.parse_args()
    chosen_targets = args.target or sorted(TARGETS)
    for t in chosen_targets:
        run(t)
    if set(chosen_targets) == set(TARGETS):
        bypub = pd.concat([pd.read_csv(RESULTS / f"per_class_{t}_bypub.csv")
                           for t in chosen_targets], ignore_index=True)
        at_half = bypub[bypub.threshold_kind == "t050"]
        tests = shift_tests(at_half)
        tests.to_csv(RESULTS / "per_class_shift_tests.csv", index=False)

        audit = single_class_audit(at_half)
        audit.to_csv(RESULTS / "per_class_single_class_audit.csv", index=False)
        pooled = pooled_rates(at_half)
        pooled.to_csv(RESULTS / "per_class_pooled_rates.csv", index=False)
        print("\nsingle-class publications, and how much data they hold:")
        for _, r in audit.iterrows():
            print(f"  {r.target:5s} {r.split:14s} {r.publications:4.0f} pubs, "
                  f"{r.pub_single_class_frac:.0%} single-class holding "
                  f"{r.mol_in_single_class_frac:.0%} of molecules "
                  f"(median {r.median_size_single_class:.0f} molecules against "
                  f"{r.median_size_both_classes:.0f})")
        print("\npooled-confusion rates, which use every publication (rf):")
        for _, r in pooled[pooled.model == "rf"].iterrows():
            print(f"  {r.target:5s} {r.split:14s} sens {r.sensitivity:.3f} "
                  f"[{r.sensitivity_lo:.3f}, {r.sensitivity_hi:.3f}]  "
                  f"spec {r.specificity:.3f} [{r.specificity_lo:.3f}, {r.specificity_hi:.3f}]")
        head = tests.drop_duplicates(["unit", "target", "rate"])
        print("\nmatched-arm asymmetry, scaffold_pub against the shifted arm, t=0.5:")
        for _, r in head.iterrows():
            sig = int((tests[(tests.unit == r.unit) & (tests.target == r.target)
                             & (tests.rate == r.rate)].p_holm < 0.05).sum())
            print(f"  {r.unit:17s} {r.target:5s} {r.rate:12s} "
                  f"{r.n_fall}/{r.n_models} fall, sign p={r.sign_test_p:.3f}, {sig} Holm-significant")
