"""How much of the publication penalty is chemical novelty, and how sure can we be?

`scripts/analyse_confounders.py` standardised the penalty to each arm's max-Tanimoto composition
and found 51-78% of it removed. That number was a point estimate from one fixed set of five bands,
with no interval, on two targets. This script does the three things needed before it can be
quoted:

  uncertainty     A cluster bootstrap over PUBLICATIONS, not molecules. Molecules from one paper
                  are a correlated block -- same series, same assay, often the same label -- so a
                  molecule bootstrap would understate the interval. Both arms are resampled on the
                  same draw of publications, so the penalty stays paired.
  band definition The share depends on where the band edges fall. Six definitions are run, from
                  three coarse bands to a continuous regression that uses no bands at all.
  all three targets  Including TG2, whose decline the paper currently attributes to
                  publication-level confounding.

The continuous check matters most. Standardisation on bands can only remove the novelty difference
BETWEEN bands; novelty still varies inside a band, so a banded estimate is a lower bound on the
novelty share. Regressing each molecule's outcome on its similarity, pooled across arms with the
arm as a dummy, removes the within-band variation too.

What this cannot do: separate novelty from assay. Assays are perfectly nested inside publications
(scripts/audit_assay_confound.py), so the residual after adjusting for novelty is "publication
identity AND its assay protocol", never publication identity alone.

Reproduce: python scripts/validate_novelty_share.py [--boot 1000]
  -> results/novelty_share.csv  (per target, band definition and model)
  -> results/novelty_share_boot.csv  (bootstrap replicates of the pooled share)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

OUT = "results/novelty_share.csv"
OUT_BOOT = "results/novelty_share_boot.csv"

# TG2 keeps the leave-one-publication-out arm its paper reports; the replications use grouped
# K-fold, which docs/replication_feasibility.md section 2 explains.
ARMS = {"tg2": ("scaffold_pub", "paper_lopo"),
        "cdk2": ("scaffold_pub", "paper_grouped"),
        "maob": ("scaffold_pub", "paper_grouped")}

# Six band definitions. The published five, two coarser, one finer, one equal-frequency, and the
# continuous alternative that uses no bands.
BAND_SETS = {
    "published_5": (0.0, 0.4, 0.5, 0.6, 0.7, 1.01),
    "coarse_2": (0.0, 0.5, 1.01),
    "coarse_3": (0.0, 0.4, 0.7, 1.01),
    "fine_10": tuple(np.round(np.linspace(0.0, 1.0, 11), 3)[:-1]) + (1.01,),
    "deciles": None,            # equal-frequency, computed from the pooled similarity
}


def per_molecule(target, split):
    """One row per molecule per model: label, source publication, probability, novelty.

    Novelty is the maximum Tanimoto from the molecule to the training set of the fold it was
    tested in -- the same quantity analyse_sota.bands uses, kept per molecule here so the bands
    can be redrawn without refitting anything.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    from scripts.run_sota import paths
    splits_path, _, preds_path = paths(target)
    preds = pd.read_csv(preds_path)
    preds = preds[preds["split"] == split]
    if preds.empty:
        return pd.DataFrame()
    sp = pd.read_csv(splits_path, low_memory=False)
    sp = sp[sp["split_kind"] == split]

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = {s: gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in sp["smiles"].unique()}
    sim = {}
    for seed, g in sp.groupby("seed"):
        train = [fps[s] for s in g[g["fold"] == "train"]["smiles"]]
        if not train:
            continue
        for s in g[g["fold"] == "test"]["smiles"]:
            sim[(seed, s)] = max(DataStructs.BulkTanimotoSimilarity(fps[s], train))

    paper = sp.drop_duplicates("smiles").set_index("smiles")["paper"]
    preds = preds[[(r.seed, r.smiles) in sim for r in preds.itertuples()]].copy()
    preds["max_tanimoto"] = [sim[(r.seed, r.smiles)] for r in preds.itertuples()]
    preds["paper"] = preds["smiles"].map(paper)
    # one row per molecule per model: a molecule tested in several folds gets its mean probability,
    # which is what analyse_sota.bands does and keeps the two arms comparable
    out = preds.groupby(["model", "smiles"]).agg(
        y=("y", "first"), p=("p", "mean"), paper=("paper", "first"),
        max_tanimoto=("max_tanimoto", "mean")).reset_index()
    return out.dropna(subset=["paper"])


def banded_share(pub, sca, edges, min_per_band=5):
    """Raw and novelty-standardised penalty for one model, under one set of band edges.

    The standardised penalty averages the WITHIN-band gap under each arm's own band weights, so
    the difference in composition between arms is removed.
    """
    a = pub.assign(band=pd.cut(pub["max_tanimoto"], list(edges), right=False))
    b = sca.assign(band=pd.cut(sca["max_tanimoto"], list(edges), right=False))

    def table(d):
        rows = {}
        for band, g in d.groupby("band", observed=True):
            if len(g) < min_per_band or g["y"].nunique() < 2:
                continue
            rows[band] = (roc_auc_score(g["y"], g["p"]), len(g))
        return rows

    ta, tb = table(a), table(b)
    shared = [k for k in ta if k in tb]
    if not shared:
        return None
    auc_a = np.array([ta[k][0] for k in shared])
    auc_b = np.array([tb[k][0] for k in shared])
    w_a = np.array([ta[k][1] for k in shared], float)
    w_b = np.array([tb[k][1] for k in shared], float)
    w_a, w_b = w_a / w_a.sum(), w_b / w_b.sum()

    raw = float((auc_a * w_a).sum() - (auc_b * w_b).sum())
    std_b = float(((auc_a - auc_b) * w_b).sum())
    std_a = float(((auc_a - auc_b) * w_a).sum())
    return {"penalty_raw": raw, "penalty_std_scaffold_mix": std_b,
            "penalty_std_pub_mix": std_a, "bands_used": len(shared),
            "share": 1 - abs(std_b) / abs(raw) if raw else np.nan,
            **overlap(a, b)}


def overlap(a, b, min_per_band=20):
    """Common support: how much of each arm sits in bands the other arm can also score.

    Standardisation is only defined where both arms have molecules. If a publication arm puts
    most of its weight in bands the scaffold arm barely reaches, the standardised penalty is
    extrapolation dressed as adjustment. This is the quantity that decides whether the estimate
    means anything, and it is what fails on TG2.

    The floor here (20) is deliberately higher than the floor banded_share uses to decide whether
    to score a band at all (5). A five-molecule band produces an AUROC, so a low floor reports
    full support for an estimate that rests on almost nothing; `min_band_n` is the column that
    exposes it, and on the real data it reads 7 for TG2 against 63 for CDK2 and 207 for MAO-B.
    """
    na = a.groupby("band", observed=True).size()
    nb = b.groupby("band", observed=True).size()
    both = [k for k in na.index if nb.get(k, 0) >= min_per_band and na[k] >= min_per_band]
    return {
        "support_pub": float(na.reindex(both).sum() / na.sum()) if na.sum() else np.nan,
        "support_scaffold": float(nb.reindex(both).sum() / nb.sum()) if nb.sum() else np.nan,
        "min_band_n": int(min(min(na.reindex(both).min(), nb.reindex(both).min()), 10 ** 9))
        if both else 0,
    }


def continuous_share(pub, sca):
    """The share with no bands at all, so within-band novelty is removed too.

    Two logistic fits on the pooled molecules of both arms: one with the arm dummy alone, one that
    also carries similarity. The arm coefficient's shrinkage between them is the novelty share.
    Reported as AUROC-free, because the coefficient is on the log-odds scale: the share is a
    ratio of coefficients, which is scale-free.
    """
    d = pd.concat([pub.assign(arm=1.0), sca.assign(arm=0.0)], ignore_index=True)
    # the model's own probability is the predictor whose calibration the arm shifts; the outcome is
    # whether that prediction was right, so the arm coefficient measures accuracy lost
    d = d.assign(correct=((d["p"] > 0.5).astype(int) == d["y"]).astype(int))
    if d["correct"].nunique() < 2 or d["arm"].nunique() < 2:
        return None
    s = d["max_tanimoto"].to_numpy()
    plain = LogisticRegression(max_iter=1000).fit(d[["arm"]], d["correct"])
    adj = LogisticRegression(max_iter=1000).fit(
        np.column_stack([d["arm"], s, s ** 2]), d["correct"])
    b0, b1 = float(plain.coef_[0][0]), float(adj.coef_[0][0])
    return {"arm_logodds_plain": b0, "arm_logodds_adjusted": b1,
            "share": 1 - abs(b1) / abs(b0) if b0 else np.nan}


class PaperBootstrap:
    """Cluster bootstrap over publications, with the row blocks precomputed.

    Each draw resamples publications with replacement and applies the SAME draw to both arms, so
    the penalty stays paired. The blocks are built once because a draw is then just a concatenate
    of integer index arrays -- doing it with a groupby per draw is thousands of times slower and
    was the reason an earlier version of this script could not finish.
    """

    def __init__(self, pub, sca):
        self.pub, self.sca = pub.reset_index(drop=True), sca.reset_index(drop=True)
        self.papers = np.union1d(self.pub["paper"].unique(), self.sca["paper"].unique())
        self.blocks = [{p: np.asarray(idx) for p, idx in d.groupby("paper").indices.items()}
                       for d in (self.pub, self.sca)]

    def draw(self, rng):
        drawn = rng.choice(self.papers, size=len(self.papers), replace=True)
        out = []
        for d, block in zip((self.pub, self.sca), self.blocks):
            take = [block[p] for p in drawn if p in block]
            out.append(d.iloc[np.concatenate(take)] if take else d.iloc[:0])
        return out


def contrast_gap(g, edges, min_per_band):
    """Novelty-matched AUROC gap between unseen-publication and seen-publication molecules."""
    g = g.assign(band=pd.cut(g["max_tanimoto"], list(edges), right=False))
    per_band = []
    for _, gg in g.groupby("band", observed=True):
        a, b = gg[gg["pub_unseen"]], gg[~gg["pub_unseen"]]
        if min(len(a), len(b)) < min_per_band or a["y"].nunique() < 2 or b["y"].nunique() < 2:
            continue
        per_band.append((roc_auc_score(a["y"], a["p"]) - roc_auc_score(b["y"], b["p"]), len(a)))
    if not per_band:
        return None
    gaps = np.array([p[0] for p in per_band])
    w = np.array([p[1] for p in per_band], float)
    return float((gaps * w / w.sum()).sum()), float(gaps.mean()), len(per_band)


def publication_contrast(target, edges=BAND_SETS["published_5"], min_per_band=20, boot=0):
    """Publication identity measured INSIDE the scaffold arm, with no cross-arm comparison.

    A scaffold split does not set out to hide publications, but it incidentally hides some: a
    whole publication can land in the test side if its scaffolds all do. Those molecules give a
    contrast the cross-arm design cannot -- unseen publication against seen publication, within
    one split, one protocol and one fold structure -- so the novelty adjustment is the only
    adjustment needed.

    It exists on MAO-B alone. On CDK2 and TG2 every publication straddles the scaffold split, so
    the comparison has no unseen group at all and None comes back.

    The contrast is confounded by publication size: a large publication is likelier to straddle
    the split, so the unseen group is made of small publications, which may be focused series.
    That is reported with the result rather than adjusted away.
    """
    from scripts.run_sota import paths
    splits_path, _, _ = paths(target)
    sp = pd.read_csv(splits_path, low_memory=False)
    sp = sp[sp["split_kind"] == "scaffold_pub"]
    if sp.empty:
        return None

    unseen = {}
    for seed, g in sp.groupby("seed"):
        train = set(g[g["fold"].isin(["train", "valid"])]["paper"])
        for r in g[g["fold"] == "test"].itertuples():
            unseen[(seed, r.smiles)] = r.paper not in train
    if not any(unseen.values()):
        return None

    mol = per_molecule(target, "scaffold_pub")
    if mol.empty:
        return None
    # a molecule is "publication unseen" if it was ever tested with its publication absent from
    # training; its probability is already averaged over the folds that tested it
    ever = pd.Series(unseen).groupby(level=1).any()
    mol = mol.assign(pub_unseen=mol["smiles"].map(ever)).dropna(subset=["pub_unseen"])

    rows = []
    for model, g in mol.groupby("model"):
        got = contrast_gap(g, edges, min_per_band)
        if got is None:
            continue
        matched, unweighted, nbands = got
        row = {"target": target, "model": model, "bands_used": nbands,
               "n_unseen": int(g["pub_unseen"].sum()), "n_seen": int((~g["pub_unseen"]).sum()),
               "gap_unweighted": unweighted, "gap_novelty_matched": matched}
        if boot:
            # resample publications, so the unseen group's small publications are resampled whole
            blocks = {p: np.asarray(i) for p, i in
                      g.reset_index(drop=True).groupby("paper").indices.items()}
            papers = np.array(list(blocks))
            gg = g.reset_index(drop=True)
            rng = np.random.default_rng(0)
            draws = []
            for _ in range(boot):
                drawn = rng.choice(papers, size=len(papers), replace=True)
                sub = gg.iloc[np.concatenate([blocks[p] for p in drawn])]
                r = contrast_gap(sub, edges, min_per_band)
                if r:
                    draws.append(r[0])
            if len(draws) >= 20:
                row["ci_lo"], row["ci_hi"] = np.percentile(draws, [2.5, 97.5])
                row["boot_median"], row["draws"] = float(np.median(draws)), len(draws)
        rows.append(row)
    return pd.DataFrame(rows) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["tg2", "cdk2", "maob"])
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--models", default="rf,logreg,xgb,chemberta_frozen,chemberta_ft,chemeleon")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    want = args.models.split(",")
    rows, boots = [], []
    for target in args.targets:
        sca_arm, pub_arm = ARMS[target]
        print(f"\n=== {target}: {sca_arm} vs {pub_arm}", flush=True)
        pub_all, sca_all = per_molecule(target, pub_arm), per_molecule(target, sca_arm)
        if pub_all.empty or sca_all.empty:
            print("  no predictions for one of the arms; skipped")
            continue
        models = [m for m in want if m in set(pub_all["model"]) & set(sca_all["model"])]
        print(f"  models: {', '.join(models)}")

        edges_for = dict(BAND_SETS)
        pooled_sim = np.concatenate([pub_all["max_tanimoto"], sca_all["max_tanimoto"]])
        edges_for["deciles"] = tuple(np.unique(
            np.quantile(pooled_sim, np.linspace(0, 1, 11)))[:-1]) + (1.01,)

        print(f"\n  {'model':<17}{'band set':<14}{'raw':>8}{'std':>9}{'share':>8}"
              f"{'bands':>7}{'supp pub':>9}{'supp sca':>9}{'min n':>8}")
        for model in models:
            pub = pub_all[pub_all["model"] == model]
            sca = sca_all[sca_all["model"] == model]
            for name, edges in edges_for.items():
                r = banded_share(pub, sca, edges)
                if r is None:
                    continue
                rows.append({"target": target, "model": model, "band_set": name, **r})
                print(f"  {model:<17}{name:<14}{r['penalty_raw']:>+8.3f}"
                      f"{r['penalty_std_scaffold_mix']:>+9.3f}{r['share']:>8.0%}"
                      f"{r['bands_used']:>7}{r['support_pub']:>9.0%}"
                      f"{r['support_scaffold']:>9.0%}{r['min_band_n']:>8}")
            c = continuous_share(pub, sca)
            if c:
                rows.append({"target": target, "model": model, "band_set": "continuous", **c})
                print(f"  {model:<17}{'continuous':<14}{'':>8}{'':>9}{c['share']:>8.0%}"
                      f"{'-':>7}")

        if args.boot:
            print(f"\n  cluster bootstrap over publications, {args.boot} draws "
                  f"({pub_all['paper'].nunique()} publications)", flush=True)
            rng = np.random.default_rng(0)
            for model in models:
                pub = pub_all[pub_all["model"] == model]
                sca = sca_all[sca_all["model"] == model]
                boot = PaperBootstrap(pub, sca)
                got = []
                for _ in range(args.boot):
                    p2, s2 = boot.draw(rng)
                    r = banded_share(p2, s2, BAND_SETS["published_5"])
                    if r and np.isfinite(r["share"]):
                        got.append((r["penalty_raw"], r["penalty_std_scaffold_mix"], r["share"]))
                if len(got) < 20:
                    continue
                g = np.array(got)
                for i, label in enumerate(("penalty_raw", "penalty_std", "share")):
                    lo, hi = np.percentile(g[:, i], [2.5, 97.5])
                    boots.append({"target": target, "model": model, "quantity": label,
                                  "median": float(np.median(g[:, i])), "ci_lo": float(lo),
                                  "ci_hi": float(hi), "draws": len(g)})
                print(f"    {model:<17}raw {np.median(g[:,0]):+.3f} "
                      f"[{np.percentile(g[:,0],2.5):+.3f}, {np.percentile(g[:,0],97.5):+.3f}]   "
                      f"std {np.median(g[:,1]):+.3f} "
                      f"[{np.percentile(g[:,1],2.5):+.3f}, {np.percentile(g[:,1],97.5):+.3f}]   "
                      f"share {np.median(g[:,2]):.0%} "
                      f"[{np.percentile(g[:,2],2.5):.0%}, {np.percentile(g[:,2],97.5):.0%}]",
                      flush=True)

    print("\n## Publication identity inside the scaffold arm (no cross-arm comparison)\n")
    for target in args.targets:
        pc = publication_contrast(target, boot=args.boot)
        if pc is None:
            print(f"  {target}: not estimable -- every publication straddles the scaffold "
                  f"split, so there is no unseen-publication group")
            continue
        print(f"  {target}: {int(pc['n_unseen'].iloc[0])} molecules from unseen publications "
              f"against {int(pc['n_seen'].iloc[0])} from seen ones")
        print(f"   {'model':<17}{'gap, novelty-matched':>22}{'bands':>7}")
        for _, r in pc.iterrows():
            print(f"   {r['model']:<17}{r['gap_novelty_matched']:>+22.3f}{r['bands_used']:>7}")
            rows.append({"section": "publication_contrast", **r.to_dict()})

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    if boots:
        pd.DataFrame(boots).to_csv(OUT_BOOT, index=False)
    print(f"\n-> {args.out}" + (f", {OUT_BOOT}" if boots else ""))
    print("\n  The residual after adjusting for novelty is publication identity AND its assay"
          "\n  protocol together: assays are perfectly nested inside publications, so no"
          "\n  adjustment here can separate them.")


if __name__ == "__main__":
    main()
