"""Hierarchical conformal prediction against our publication-grouped calibration.

docs/paper.md 5.13 reports that i.i.d. conformal calibration under-covers badly when a whole
publication is held out, and that calibrating on whole held-out publications brings measured
coverage back to target on three targets. That fix is empirical: it carries no distribution-free
guarantee, and 8 records that our own `group_calibration` violates group exchangeability by taking
publications smallest-first. This script runs the published method that does have a theorem for a
test point from a previously unseen group, on the same saved predictions, and asks whether the
guaranteed method matches the empirical one.

THE ALGORITHM (implemented in `hcp_threshold`)
----------------------------------------------
Hierarchical conformal prediction, split version, Theorem 1 of

    Lee Y, Barber RF, Willett R. Distribution-free inference with hierarchical data.
    arXiv:2306.06342 (2023, rev. 2024).

building on the three constructions (pooling CDFs, subsampling once, repeated subsampling) of

    Dunn R, Wasserman L, Ramdas A. Distribution-free prediction sets for two-layer hierarchical
    models. J. Am. Stat. Assoc. 2023;118(544):2491-2502. arXiv:1809.07441.

Given K1 calibration groups with sizes N_k and nonconformity scores s(Z_{k,i}), the threshold T is
the (1-alpha)-quantile of the discrete distribution that puts mass

    1 / ((K1 + 1) * N_k)   on each calibration score s(Z_{k,i}),
    1 / (K1 + 1)           on +infinity,

and the prediction set is {y : s(X_test, y) <= T}. Theorem 1 gives

    P{ Y_test in C(X_test) } >= 1 - alpha,

and, when scores are almost surely distinct, an upper bound of 1 - alpha + 2/(K1 + 1).

WHAT THE GUARANTEE IS OVER, AND WHAT IT IS NOT
----------------------------------------------
The probability is marginal over the draw of the calibration groups AND the draw of a fresh test
group together with an observation inside it. In our terms the covered event is "a randomly chosen
publication, then a randomly chosen molecule from it" -- publication-weighted. That is NOT the
target of ordinary split conformal, which is "a randomly chosen molecule" and therefore lets a
224-molecule paper count 22 times a 10-molecule one. The two targets coincide only when every
publication is the same size. This is the substantive difference between HCP and our method, and
`publication_coverage` versus `coverage` in the output measures exactly it.

Unequal publication sizes do not threaten validity: each group carries total mass 1/(K1+1)
whatever its size, so large publications cannot drag the threshold. With all N_k = 1 the
construction reduces to standard split conformal.

ASSUMPTIONS, AND WHETHER OUR DATA PLAUSIBLY MEET THEM
-----------------------------------------------------
Hierarchical exchangeability (Definition 1 of the Lee-Barber-Willett paper): the groups are
exchangeable with one another, and the observations within each group are exchangeable given its
size. Data within a group need not be independent, which is the point -- a congeneric series
measured in one assay is exactly the dependent-within-group case the theorem is built for.

For the within-fold design below, the honest reading is "plausible but not established":

  - Within a publication, molecules are a series the authors chose to report together. Treating
    them as exchangeable given the publication is reasonable; they are certainly not independent,
    and the theorem does not ask them to be.
  - Across publications, exchangeability is the doubtful half. Publications are ordered in time and
    later series build on earlier ones, so the K publications of a target are not an exchangeable
    sample in any strict sense. Our grouped K-fold assigns publications to folds at random, which
    makes the calibration and test publications *within one fold* exchangeable with each other by
    construction -- and that, not exchangeability of the whole corpus, is what Theorem 1 needs.
    This is why the within-fold design is the one we call theorem-faithful.
  - Publication sizes vary 10 to 224. Theorem 1 tolerates that for the 1-alpha bound. The sharper
    1-alpha result for repeated subsampling additionally wants exchangeable group sizes, which we
    do not claim; we do not implement that variant.

DEVIATIONS FROM THE PUBLISHED ALGORITHM
---------------------------------------
Within-fold design (`--design within`): none in the threshold construction. The score function is
fitted on the other folds' publications, which play the role of the K0 fitting groups, and the
calibration and test groups are the fold's held-out publications, none of which the model saw. This
is Theorem 1 as stated.

Cross-fold design (`--design cross`): one real deviation, and it is why both designs are run.
Calibration publications are taken from other folds and scored by their own fold's model, so the
score function is not a single function fitted independently of the calibration and test groups --
and worse, the model that scores a calibration publication was fitted on data including the test
publication. This is cross-conformal rather than split conformal, and Theorem 1 does not cover it.
We run it only because TG2's leave-one-publication-out arm leaves exactly one unseen publication
per fold, so the within-fold design is impossible there, and because CDK2 has at most 9 publications
per fold against the K1 >= 9 that alpha = 0.1 requires. Running it on MAO-B, where the faithful
design also works, measures what the deviation costs.

No model is refitted. Predictions come from the saved per-molecule conformal files, so every method
compared here scores the identical molecules with the identical fitted model, and only the
calibration set and the quantile rule differ.

Usage: python scripts/run_hcp.py --target maob [--design within|cross] [--alphas 0.1,0.2]
Writes results/hcp_{target}_{design}.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

PREDS = {"tg2": "results/lopo_conformal.csv",
         "cdk2": "results/conformal_cdk2_paper_grouped.csv",
         "maob": "results/conformal_maob_paper_grouped.csv"}
SPLITS = {"tg2": ("results/splits_tg2.csv", "paper_lopo"),
          "cdk2": ("results/splits_cdk2.csv", "paper_grouped"),
          "maob": ("results/splits_maob.csv", "paper_grouped")}
MODELS = ("rf", "xgb", "logreg", "chemberta_frozen", "chemberta_ft", "ensemble")
BANDS = ((0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01))


# ------------------------------------------------------------------ thresholds

def hcp_threshold(scores, groups, alpha):
    """Theorem 1 of Lee, Barber & Willett (arXiv:2306.06342), split HCP.

    Each calibration GROUP carries total mass 1/(K1+1) spread evenly over its members, and
    +infinity carries 1/(K1+1). Returns the (1-alpha)-quantile of that distribution.

    The finite mass sums to K1/(K1+1), so whenever alpha < 1/(K1+1) the quantile lands on the
    atom at infinity and the method must return the trivial set. That is a property of the
    theorem, not a defect of this code: fewer than ceil(1/alpha)-1 calibration groups cannot
    support a 1-alpha statement, and `feasible_k1` is what the caller checks beforehand.
    """
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    if len(scores) == 0:
        return np.inf
    uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    k1 = len(uniq)
    w = 1.0 / ((k1 + 1) * counts[inv])
    order = np.argsort(scores, kind="mergesort")
    cum = np.cumsum(w[order])
    # The boundary is exact, not approximate: with K1 = ceil(1/alpha)-1 the finite mass is
    # K1/(K1+1) = 1-alpha to the last bit, so the quantile is the largest finite score and the
    # method is usable. Summing 1/((K1+1)N_k) in floating point lands a few ulps under that, which
    # without this tolerance pushes the quantile onto the infinite atom and silently turns the
    # smallest feasible calibration set into a trivial predictor. tests/test_hcp.py pins it.
    hit = int(np.searchsorted(cum, 1.0 - alpha - 1e-9, side="left"))
    if hit >= len(scores):                      # mass ran out before 1-alpha: the infinite atom
        return np.inf
    return float(scores[order][hit])


def split_threshold(scores, alpha):
    """Standard split-conformal quantile, ceil((n+1)(1-alpha))/n.

    Identical to the rule in scripts/run_lopo_conformal.py, so the pooled arms here use the same
    estimator 5.13 reports. Every calibration MOLECULE carries equal mass, which is what makes
    this target the molecule-marginal rather than the publication-marginal.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else float(np.sort(scores)[k - 1])


def feasible_k1(alpha):
    """Smallest number of calibration groups for which HCP can return a non-trivial set."""
    return int(np.ceil(1.0 / alpha)) - 1


def scores_of(p, y):
    """Nonconformity 1 - P(true label), as in run_lopo_conformal.conformal_sets."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    return 1.0 - np.where(y == 1, p, 1.0 - p)


def sets_of(p, t):
    """{label k : 1 - P(k) <= t} as an (n, 2) boolean array."""
    p = np.asarray(p, dtype=float)
    mat = np.column_stack([1.0 - p, p])          # P(0), P(1)
    return np.column_stack([(1.0 - mat[:, k]) <= t for k in (0, 1)])


# ------------------------------------------------------------------ data

def load_arm(target):
    """One row per (model, molecule) on the publication arm, with publication, label and novelty.

    Uses the group regime's predictions so a single fitted model per fold scores every method
    compared here; the regime only determines which model produced `p`, and all three methods are
    handed the same `p`.
    """
    p = pd.read_csv(PREDS[target])
    p = p[(p.regime == "group") & (p.alpha == p.alpha.min())].copy()
    spl, kind = SPLITS[target]
    s = pd.read_csv(spl, low_memory=False)
    s = s[s.split_kind == kind].drop_duplicates("smiles")
    p["paper"] = p.smiles.map(s.set_index("smiles")["paper"])
    if p.paper.isna().any():
        raise SystemExit(f"{target}: {int(p.paper.isna().sum())} molecules lack a publication id")
    return p[["fold", "model", "smiles", "y", "p", "max_tanimoto", "paper"]]


def pick_calibration(papers, k1, selection, rng):
    """Which publications become calibration groups.

    'random' is the unbiased choice the theorem wants. 'smallest' reproduces the bias in
    run_lopo_conformal.group_calibration, which takes publications smallest-first to keep the
    fitting set large; 'largest' is its mirror. The sensitivity analysis runs all three.
    """
    sizes = papers.value_counts()
    if selection == "smallest":
        return set(sizes.sort_values(ascending=True).index[:k1])
    if selection == "largest":
        return set(sizes.sort_values(ascending=False).index[:k1])
    return set(rng.permutation(sizes.index)[:k1])


def record(name, fold, model, alpha, k1, n_cal, t, ev):
    """Per-molecule outcome rows for one (method, fold, model, alpha)."""
    s = sets_of(ev.p, t)
    y = ev.y.to_numpy().astype(int)
    return pd.DataFrame({
        "method": name, "fold": fold, "model": model, "alpha": alpha,
        "k1": k1, "n_cal": n_cal, "threshold": t,
        "smiles": ev.smiles.to_numpy(), "y": y, "paper": ev.paper.to_numpy(),
        "novelty": ev.max_tanimoto.to_numpy(),
        "covered": s[np.arange(len(ev)), y], "set_size": s.sum(1)})


# ------------------------------------------------------------------ designs

def within_fold(d, alpha, cal_frac, selection, leak_frac, rng):
    """Theorem-faithful design: calibration and test publications both come from one fold.

    The model for fold k was fitted on the other folds, so it has seen neither the calibration nor
    the test publications, and the grouped K-fold assigned the fold's publications at random --
    which is what makes them exchangeable with one another.
    """
    need = feasible_k1(alpha)
    rows, skipped = [], 0
    for (fold, model), g in d.groupby(["fold", "model"]):
        m = g.paper.nunique()
        k1 = max(int(np.ceil(cal_frac * m)), need)
        if m - k1 < 1:
            skipped += 1
            continue
        cal_pubs = pick_calibration(g.paper, k1, selection, rng)
        cal = g[g.paper.isin(cal_pubs)]
        test = g[~g.paper.isin(cal_pubs)]
        # Hold aside a slice of each test publication for the i.i.d. arm's calibration pool, so
        # that arm genuinely shares publications with the evaluation set -- its defining optimism
        # -- while every arm is scored on the identical remaining molecules.
        leak_idx = []
        for _, tg in test.groupby("paper"):
            n_leak = int(round(leak_frac * len(tg)))
            if n_leak:
                leak_idx += list(rng.permutation(tg.index)[:n_leak])
        ev = test.drop(index=leak_idx)
        if not len(ev) or not len(cal):
            skipped += 1
            continue
        cal_s = scores_of(cal.p, cal.y)
        pool = pd.concat([cal, test.loc[leak_idx]])
        pool = pool.iloc[rng.permutation(len(pool))[:len(cal)]]
        for name, t in (("iid_pooled", split_threshold(scores_of(pool.p, pool.y), alpha)),
                        ("grouped_pooled", split_threshold(cal_s, alpha)),
                        ("hcp", hcp_threshold(cal_s, cal.paper.to_numpy(), alpha))):
            rows.append(record(name, fold, model, alpha, k1, len(cal), t, ev))
    if skipped:
        print(f"   alpha={alpha}: {skipped} (fold, model) pairs infeasible, need K1>={need}")
    return rows


def cross_fold(d, alpha, selection, cal_pubs, rng):
    """Cross-conformal design: calibration publications come from the OTHER folds.

    Theorem 1 does not cover this -- see the module docstring. It exists because TG2's
    leave-one-publication-out arm leaves one unseen publication per fold, and because CDK2 has too
    few publications per fold for alpha=0.1.
    """
    need = feasible_k1(alpha)
    rows = []
    for (fold, model), g in d.groupby(["fold", "model"]):
        other = d[(d.fold != fold) & (d.model == model)]
        avail = other.paper.nunique()
        k1 = avail if cal_pubs is None else min(cal_pubs, avail)
        if k1 < need:
            continue
        cal = other[other.paper.isin(pick_calibration(other.paper, k1, selection, rng))]
        cal_s = scores_of(cal.p, cal.y)
        # the i.i.d. arm ignores publication structure and may draw from any fold, this one
        # included, at the same molecule budget
        allm = d[d.model == model]
        pool = allm.iloc[rng.permutation(len(allm))[:len(cal)]]
        for name, t in (("iid_pooled", split_threshold(scores_of(pool.p, pool.y), alpha)),
                        ("grouped_pooled", split_threshold(cal_s, alpha)),
                        ("hcp", hcp_threshold(cal_s, cal.paper.to_numpy(), alpha))):
            rows.append(record(name, fold, model, alpha, k1, len(cal), t, g))
    return rows


# ------------------------------------------------------------------ summary

def publication_bootstrap(per, draws, rng):
    """Cluster bootstrap over publications: a sampled publication brings all its molecules.

    Molecules in one paper are a congeneric series measured in one assay, so resampling molecules
    would treat a correlated block as independent draws and understate the interval.
    """
    arr = per.reset_index(drop=True)
    blocks = [idx for _, idx in arr.groupby("paper").indices.items()]
    cov = arr.covered.to_numpy().astype(float)
    keys = np.arange(len(blocks))
    vals = np.empty(draws)
    for b in range(draws):
        rows = np.concatenate([blocks[k] for k in keys[rng.integers(0, len(keys), len(keys))]])
        vals[b] = cov[rows].mean()
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarise(rec, draws, rng):
    out = []
    for (method, model, alpha), g in rec.groupby(["method", "model", "alpha"]):
        covered = g.covered.to_numpy().astype(bool)
        size = g.set_size.to_numpy()
        single = size == 1
        pub_size = g.groupby("paper").smiles.transform("size").to_numpy()
        pc = g.groupby("paper").covered.mean()
        med = np.median(pub_size)
        row = {"method": method, "model": model, "alpha": alpha,
               "n": len(g), "publications": g.paper.nunique(),
               "k1_median": float(g.k1.median()), "n_cal_median": float(g.n_cal.median()),
               "infinite_threshold_frac": float(np.isinf(g.threshold.to_numpy()).mean()),
               "coverage": float(covered.mean()),
               "publication_coverage": float(pc.mean()),
               "publication_coverage_min": float(pc.min()),
               "frac_publications_below_target": float((pc < 1 - alpha).mean()),
               "coverage_small_pubs": float(covered[pub_size <= med].mean()),
               "coverage_large_pubs": float(covered[pub_size > med].mean()),
               "mean_set_size": float(size.mean()),
               "singleton_frac": float(single.mean()),
               "singleton_error": float((~covered[single]).mean()) if single.any() else np.nan,
               "empty_frac": float((size == 0).mean())}
        for lo, hi in BANDS:
            sel = ((g.novelty >= lo) & (g.novelty < hi)).to_numpy()
            row[f"coverage_{lo}_{hi}"] = float(covered[sel].mean()) if sel.any() else np.nan
            row[f"n_{lo}_{hi}"] = int(sel.sum())
        row["coverage_ci_lo"], row["coverage_ci_hi"] = publication_bootstrap(g, draws, rng)
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="maob", choices=list(PREDS))
    ap.add_argument("--design", default="within", choices=["within", "cross"])
    ap.add_argument("--alphas", default="0.1,0.2")
    ap.add_argument("--cal-frac", type=float, default=0.6,
                    help="within design: fraction of a fold's publications used to calibrate")
    ap.add_argument("--cal-pubs", type=int, default=None,
                    help="cross design: cap on calibration publications, for budget matching")
    ap.add_argument("--selection", default="random", choices=["random", "smallest", "largest"])
    ap.add_argument("--leak-frac", type=float, default=0.25)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    d = load_arm(a.target)
    d = d[d.model.isin(MODELS)]
    print(f"{a.target}: {d.smiles.nunique()} molecules, {d.paper.nunique()} publications, "
          f"{d.fold.nunique()} folds, {d.model.nunique()} models; design={a.design} "
          f"selection={a.selection}")

    recs = []
    for alpha in [float(x) for x in a.alphas.split(",")]:
        if a.design == "within":
            recs += within_fold(d, alpha, a.cal_frac, a.selection, a.leak_frac, rng)
        else:
            recs += cross_fold(d, alpha, a.selection, a.cal_pubs, rng)
    if not recs:
        raise SystemExit(f"{a.target}: no feasible (fold, model, alpha) combination for design "
                         f"{a.design!r}; the census is in docs/audit_2026-11-05.md")
    rec = pd.concat(recs, ignore_index=True)
    summary = summarise(rec, a.draws, rng)
    summary["target"], summary["design"], summary["selection"] = a.target, a.design, a.selection
    out = a.out or f"results/hcp_{a.target}_{a.design}.csv"
    os.makedirs("results", exist_ok=True)
    summary.to_csv(out, index=False)

    # Per-publication coverage, so a paired method comparison can be clustered on the publication
    # -- the same publication is scored by every method, which is what makes pairing legitimate.
    pubs = (rec.groupby(["method", "model", "alpha", "paper"])
            .agg(n=("smiles", "size"), covered=("covered", "mean"),
                 singleton=("set_size", lambda s: float((s == 1).mean())))
            .reset_index())
    pubs["target"], pubs["design"], pubs["selection"] = a.target, a.design, a.selection
    pubs.to_csv(out.replace(".csv", "_pubs.csv"), index=False)
    cols = ["method", "model", "alpha", "n", "publications", "coverage", "coverage_ci_lo",
            "coverage_ci_hi", "publication_coverage", "singleton_frac", "singleton_error",
            "empty_frac", "k1_median"]
    print(summary.sort_values(["alpha", "model", "method"])[cols].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
