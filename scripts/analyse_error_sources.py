"""Where do the errors live? False positives and false negatives by molecule, publication and
chemistry, from saved predictions. No retraining.

The professor's question in its sharpest form: are the mistakes spread evenly, or do particular
structures, assays or source publications produce them repeatedly? Four analyses, all at the 0.5
cut on `results/sota_preds*.csv`:

  molecule     how often each molecule is misclassified across models and seeds, and whether the
               same molecules fail for every model (concordance) or different ones for each
  publication  false-negative and false-positive rate per source publication, with a permutation
               test: if publication identity carried no information about error, reassigning
               molecules to publications at random would reproduce the observed spread
  chemistry    error rate against distance to the 1 uM decision boundary, nearest-training-set
               similarity, and molecular size
  assay        not separable from publication. results/confounders.csv records that no assay in
               either replication dataset spans two publications (0 of 380, 0 of 516), so a
               per-assay analysis would be the per-publication one with a different label

Usage: python scripts/analyse_error_sources.py [--target tg2|cdk2|maob] ...
Outputs: results/error_molecules_<target>.csv, results/error_publications_<target>.csv,
         results/error_chemistry_<target>.csv, results/error_concordance.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.chem import morgan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TARGETS = ("tg2", "cdk2", "maob")
CUT = 0.5
PIC50_THRESHOLD = 6.0  # 1 uM, the classification boundary used throughout
N_PERM = 1000


def preds_file(target):
    return "sota_preds.csv" if target == "tg2" else f"sota_preds_{target}.csv"


def potency(target):
    if target == "tg2":
        from src.data import tg2
        return tg2.combined("regression").set_index("smiles").y
    from src.data import chembl
    raw = chembl.load_raw(f"data/raw/chembl/{target}_activities.csv")
    return chembl.regression(raw).set_index("smiles").y


def nearest_training_similarity(splits, arm):
    """Max Tanimoto from each test molecule to the training fold it was scored against, averaged
    over the folds the molecule appears in. Computed per fold because the training set differs."""
    d = splits[splits.split_kind == arm]
    out = {}
    for seed, g in d.groupby("seed"):
        tr = g[g.fold == "train"].smiles.tolist()
        te = g[g.fold == "test"].smiles.tolist()
        if not tr or not te:
            continue
        a, b = morgan(te).astype(np.float32), morgan(tr).astype(np.float32)
        inter = a @ b.T
        union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0).max(axis=1)
        for s, v in zip(te, sim):
            out.setdefault(s, []).append(float(v))
    return pd.Series({k: float(np.mean(v)) for k, v in out.items()})


def molecule_errors(preds, arm):
    """One row per molecule: how many (model, seed) scorings got it wrong, and of what kind."""
    d = preds[preds.split == arm].copy()
    d["wrong"] = (d.p >= CUT) != (d.y == 1)
    g = d.groupby("smiles")
    out = pd.DataFrame({
        "y": g.y.first(),
        "scorings": g.size(),
        "models": g.model.nunique(),
        "errors": g.wrong.sum(),
        "error_rate": g.wrong.mean(),
        "mean_p": g.p.mean(),
    })
    out["kind"] = np.where(out.y == 1, "false_negative_when_wrong", "false_positive_when_wrong")
    # how many distinct models get it wrong at least once, as a share of models that scored it
    per_model = d.groupby(["smiles", "model"]).wrong.mean()
    out["models_ever_wrong"] = per_model.gt(0).groupby("smiles").sum()
    out["models_mostly_wrong"] = per_model.gt(0.5).groupby("smiles").sum()
    return out.reset_index()


def concordance(mol, target, arm):
    """Do the same molecules fail for every model, or different ones for each?

    The null is that each model errs independently at its own rate. Under it, the number of models
    mostly-wrong on a molecule is a sum of independent Bernoullis, so the share of molecules where
    almost every model is wrong would be small. Comparing the observed share to that null says
    whether difficulty is a property of the molecule or of the model.
    """
    n_models = int(mol.models.max())
    hard = mol[mol.models == n_models]
    if hard.empty or n_models < 3:
        return None
    rate = hard.models_mostly_wrong.sum() / (len(hard) * n_models)  # marginal per-model error rate
    rng = np.random.default_rng(0)
    draws = rng.binomial(n_models, rate, size=(N_PERM, len(hard)))
    obs_all = float((hard.models_mostly_wrong >= n_models - 1).mean())
    null_all = float((draws >= n_models - 1).mean(axis=1).mean())
    obs_none = float((hard.models_mostly_wrong == 0).mean())
    null_none = float((draws == 0).mean(axis=1).mean())
    return {"target": target, "split": arm, "molecules": len(hard), "models": n_models,
            "per_model_error_rate": rate,
            "frac_molecules_almost_all_models_wrong": obs_all, "null_expectation_all": null_all,
            "frac_molecules_no_model_wrong": obs_none, "null_expectation_none": null_none}


def publication_errors(mol, paper_of, target, arm):
    """Error rates per publication, and a permutation test on whether publication identity
    explains any of the spread. Molecules are reassigned to publications at random, keeping the
    publication sizes fixed, which is the null 'errors are not a property of the source'."""
    m = mol.assign(paper=mol.smiles.map(paper_of)).dropna(subset=["paper"])
    if m.empty:
        return pd.DataFrame(), None
    pos, neg = m[m.y == 1], m[m.y == 0]
    per_pub = pd.DataFrame({
        "molecules": m.groupby("paper").size(),
        "actives": pos.groupby("paper").size().reindex(m.paper.unique()).fillna(0),
        "inactives": neg.groupby("paper").size().reindex(m.paper.unique()).fillna(0),
        "fn_rate": pos.groupby("paper").error_rate.mean(),
        "fp_rate": neg.groupby("paper").error_rate.mean(),
        "error_rate": m.groupby("paper").error_rate.mean(),
    }).reset_index().rename(columns={"index": "paper"})
    per_pub.insert(0, "split", arm)
    per_pub.insert(0, "target", target)

    rng = np.random.default_rng(0)
    sizes = m.groupby("paper").size().to_numpy()
    rates = m.error_rate.to_numpy()

    def spread(values, sizes):
        """Molecule-count-weighted SD of per-publication mean error rate."""
        means, start = [], 0
        for n in sizes:
            means.append(values[start:start + n].mean())
            start += n
        return float(np.average((np.array(means) - values.mean()) ** 2, weights=sizes) ** 0.5)

    observed = spread(np.concatenate([m[m.paper == p].error_rate.to_numpy()
                                      for p in m.groupby("paper").size().index]), sizes)
    null = np.array([spread(rng.permutation(rates), sizes) for _ in range(N_PERM)])
    test = {"target": target, "split": arm, "publications": len(sizes),
            "observed_spread": observed, "null_spread_mean": float(null.mean()),
            "null_spread_p95": float(np.percentile(null, 95)),
            "p_permutation": float((null >= observed).mean())}
    return per_pub, test


def balance_vs_error_type(per_pub, target, arm):
    """Does a publication's own class balance decide which kind of error it attracts?

    The TG2 table suggests it does: inactive-dominated papers have their few actives missed,
    active-dominated papers have their few inactives called active. If so, the model is predicting
    the prior of the surrounding chemistry rather than the molecule, and that single mechanism
    would account for the class asymmetry, the novelty effect and the error concentration at once.
    Spearman, because the relationship need not be linear; publications with fewer than 5 molecules
    of a class give unstable rates and are excluded from that rate's correlation.
    """
    from scipy.stats import spearmanr
    d = per_pub.copy()
    d["active_frac"] = d.actives / d.molecules
    out = []
    for rate, need in (("fn_rate", "actives"), ("fp_rate", "inactives")):
        usable = d[(d[need] >= 5) & d[rate].notna()]
        # a correlation against a constant is undefined; emitting a nan row would read as
        # "no relationship" rather than "not computable"
        if len(usable) < 5 or usable.active_frac.nunique() < 2 or usable[rate].nunique() < 2:
            continue
        r = spearmanr(usable.active_frac, usable[rate])
        out.append({"target": target, "split": arm, "rate": rate, "publications": len(usable),
                    "spearman_vs_active_frac": r.statistic, "p": r.pvalue})
    return pd.DataFrame(out, columns=["target", "split", "rate", "publications",
                                      "spearman_vs_active_frac", "p"])


def chemistry_errors(mol, pic50, near_sim, target, arm):
    """Error rate against distance to the decision boundary, nearest-training similarity and size."""
    m = mol.copy()
    m["pic50"] = m.smiles.map(pic50)
    m["abs_log_from_cut"] = (m.pic50 - PIC50_THRESHOLD).abs()
    m["max_tanimoto_to_train"] = m.smiles.map(near_sim)
    m["heavy_atoms"] = [int(np.asarray(morgan([s])).sum()) for s in m.smiles]  # bit count ~ size
    bands = {
        "distance to 1 uM cut": ("abs_log_from_cut", [0, 0.25, 0.5, 1.0, 2.0, np.inf]),
        "max Tanimoto to training": ("max_tanimoto_to_train", [0, 0.4, 0.5, 0.6, 0.7, 1.01]),
    }
    out = []
    for label, (col, edges) in bands.items():
        if m[col].isna().all():
            continue
        cut = pd.cut(m[col], edges, right=False)
        for band, g in m.groupby(cut, observed=True):
            out.append({"target": target, "split": arm, "variable": label, "band": str(band),
                        "molecules": len(g), "actives": int((g.y == 1).sum()),
                        "error_rate": float(g.error_rate.mean()),
                        "fn_rate": float(g[g.y == 1].error_rate.mean()) if (g.y == 1).any() else np.nan,
                        "fp_rate": float(g[g.y == 0].error_rate.mean()) if (g.y == 0).any() else np.nan})
    return pd.DataFrame(out)


def run(target):
    preds = pd.read_csv(RESULTS / preds_file(target))
    splits = pd.read_csv(RESULTS / f"splits_{target}.csv", low_memory=False)
    paper_of = (splits[splits.fold == "test"].drop_duplicates("smiles")
                .set_index("smiles").paper)
    pic50 = potency(target)

    arms = [a for a in ("scaffold_pub", "paper_lopo", "paper_grouped") if a in set(preds.split)]
    mols, pubs, chem, conc, bal = [], [], [], [], []
    for arm in arms:
        mol = molecule_errors(preds, arm)
        mol.insert(0, "split", arm)
        mol.insert(0, "target", target)
        mols.append(mol)
        per_pub, test = publication_errors(mol, paper_of, target, arm)
        if not per_pub.empty:
            pubs.append(per_pub)
            bal.append(balance_vs_error_type(per_pub, target, arm))
            if test:
                conc.append({**test, "kind": "publication_permutation"})
        c = concordance(mol, target, arm)
        if c:
            conc.append({**c, "kind": "model_concordance"})
        chem.append(chemistry_errors(mol, pic50, nearest_training_similarity(splits, arm),
                                     target, arm))

    pd.concat(mols, ignore_index=True).to_csv(RESULTS / f"error_molecules_{target}.csv", index=False)
    if pubs:
        pd.concat(pubs, ignore_index=True).to_csv(RESULTS / f"error_publications_{target}.csv",
                                                  index=False)
    pd.concat(chem, ignore_index=True).to_csv(RESULTS / f"error_chemistry_{target}.csv", index=False)
    if bal:
        pd.concat(bal, ignore_index=True).to_csv(RESULTS / f"error_balance_{target}.csv", index=False)
    print(f"{target}: {sum(len(m) for m in mols)} molecule-arm rows over arms {arms}")
    return pd.DataFrame(conc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="append", choices=TARGETS, default=None)
    chosen = ap.parse_args().target or list(TARGETS)
    allc = pd.concat([run(t) for t in chosen], ignore_index=True)
    allc.to_csv(RESULTS / "error_concordance.csv", index=False)
    for kind, g in allc.groupby("kind"):
        print(f"\n{kind}:")
        cols = [c for c in g.columns if g[c].notna().any() and c != "kind"]
        print(g[cols].round(4).to_string(index=False))
