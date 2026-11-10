"""Phase 1: reproduce the published AdaBoost + MLP classifier of Wichka & Lai (CSBJ 2024).

Source of truth is their repository `Lailabcode/celiac-informatics`, read on 2026-11-11:
`colab/Final_Model_rdkit_fingerprints_descriptors_classification.ipynb` is the notebook that
produced their saved model, and `models/MLP_adaboost.py` is a later, broken transcription of its
cell 45. Where the two disagree the notebook wins, because the notebook is what ran.

Nothing here touches a committed result file. Output is `results/wichka_reproduction.csv`
(one row per evaluation) and `results/wichka_features.csv` (the rebuild check).

Three sections, deliberately separate so that nothing assumed contaminates what is exact:

  rebuild   their 336-molecule dataset and 667-column feature matrix, re-derived from
            `TG2_compound_smiles.csv` by their label rules and their featurisation, compared
            cell-by-cell against the matrix they committed. This tests whether their published
            pipeline actually regenerates their published data.
  holdout   their single stratified 75:25 split (random_state=1), scaler fit on train only, the
            model fitted exactly as their notebook cell 45 fits it. This is the procedure behind
            the `celiac_clf.joblib` they ship.
  cv        their reported protocol: StandardScaler and model inside a pipeline, StratifiedKFold
            (5 splits, shuffle, no seed), repeated, the mean over folds recorded per repeat. Their
            0.93 +/- 0.02 is the mean and SD across 50 such repeats.

CORRECTIONS required to run their code at all, each one forced and each one recorded:

  1. `MLP_Base_For_AdaBoost` is undefined in `models/MLP_adaboost.py`. Notebook cell 45 shows the
     name that ran: `CustomMLPClassifier`, the class that file defines and never uses. This is not
     a guess -- the notebook settles it -- and it is the only class in either file that can serve
     as an AdaBoost base estimator, because MLPClassifier does not accept sample_weight and
     CustomMLPClassifier exists solely to convert those weights into a resample.
  2. `CustomMLPClassifier.fit` calls `self._fit(X, y, incremental=...)`. In scikit-learn 1.9
     `MLPClassifier._fit` takes `sample_weight` before `incremental`, so the published call raises
     TypeError. Passing `sample_weight=None` restores the published behaviour: the weights have
     already been consumed by the resample, and the inner fit was always unweighted.
  3. `models/MLP_adaboost.py` passes `base_estimator=`, removed in scikit-learn 1.4. The notebook
     passes `estimator=`, so their own run already used the modern name. No behavioural change.

DIVERGENCE that cannot be corrected, only reported:

  4. They did not set `algorithm`, so their AdaBoost ran SAMME.R, the default up to scikit-learn
     1.5. SAMME.R was removed in 1.6 and this environment is 1.9, which offers SAMME only. SAMME.R
     reweights with predicted probabilities and SAMME with hard predictions, so the fitted ensemble
     is not the same object. Nothing in this script can close that gap; only a pinned old
     environment could. If the reproduction lands on their reported accuracy anyway, the gap did
     not matter for the headline number, and that is the finding rather than an assumption.

Usage: python scripts/reproduce_wichka.py [--repeats N] [--section rebuild|holdout|cv|all]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import classification, per_class  # noqa: E402

RAW = ROOT / "data" / "raw" / "tg2" / "TG2_compound_smiles.csv"
THEIR_MATRIX = ROOT / "data" / "raw" / "tg2" / "clf_celiac_desc_fing_data.csv"

# Their published hyperparameters, notebook cell 45, repeated in models/MLP_adaboost.py.
MLP_KW = dict(hidden_layer_sizes=(50,), activation="relu", learning_rate_init=0.001, max_iter=1000)
ADA_KW = dict(n_estimators=279, learning_rate=0.022977463078217686, random_state=42)
# The 55 fingerprint bits their features.py drops: correlated (r >= 0.9) plus variance <= 0.1,
# selected on all 336 molecules before any split. Hard-coded by them, so hard-coded here.
THEIR_DROPS = [330, 342, 2, 3, 16, 36, 40, 51, 64, 81, 87, 103, 112, 117, 141, 142, 161, 166, 186,
               194, 209, 233, 236, 238, 248, 265, 280, 284, 290, 294, 297, 312, 341, 351, 369, 371,
               372, 379, 386, 402, 403, 410, 416, 417, 428, 439, 443, 447, 460, 479, 480, 491, 504,
               506, 509]


class CustomMLPClassifier(MLPClassifier):
    """Their class, verbatim except for correction 2 (see module docstring).

    AdaBoost needs a base estimator that honours sample_weight; MLPClassifier does not, so this
    draws a weighted bootstrap of the training fold instead and fits on that. `np.random.choice`
    uses global NumPy state, so the ensemble is not reproducible from `random_state` alone -- their
    design, not ours, and the reason this script seeds NumPy before every fit.
    """

    def resample(self, X_train, y_train, sample_weight):
        sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)
        X_new = np.zeros((len(X_train), len(X_train[0])), dtype=np.float32)
        y_new = np.zeros((len(y_train)), dtype=int)
        for i in range(len(X_train)):
            x = np.random.choice(np.arange(len(X_train)), p=sample_weight)
            X_new[i] = X_train[x]
            y_new[i] = y_train[x]
        return X_new, y_new

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            X, y = self.resample(X, y, sample_weight)
        return self._fit(X, y, sample_weight=None,
                         incremental=(self.warm_start and hasattr(self, "classes_")))


def their_model():
    return AdaBoostClassifier(estimator=CustomMLPClassifier(**MLP_KW), **ADA_KW)


# --- their dataset, re-derived ----------------------------------------------------------------

def their_labels(raw):
    """Their notebook cells 4-16, in order, with their canonicalisation (no salt stripping).

    actives    Activity == "Active" and IC50 Value < 0.6
    inactives  Activity == "Inactive", with no value filter at all
    negatives  also Activity == "Unspecified" with 10 < Value < 301, which is applied as
               > 10, then < 350, then < 301 across their cells 11 and 14

    Note what this does and does not say about the "unspecified" compounds: they are not molecules
    without evidence, they are molecules with a measured value between 10 and 301 whose PubChem
    activity outcome field was left blank. Note also that neither the inactive nor the unspecified
    branch filters Type == "IC50", so a Ki or k_inact row can enter on its Value alone.
    """
    df = raw.copy()
    df["Smiles"] = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in df.Smiles]
    df = df.drop_duplicates(subset="CID").reset_index(drop=True)  # by CID, not by structure
    active = df[(df.Activity == "Active") & (df.Value < 0.6)]
    inactive = df[df.Activity == "Inactive"]
    unspec = df[(df.Activity == "Unspecified") & (df.Value > 10) & (df.Value < 301)]
    out = pd.concat([active, inactive, unspec], ignore_index=True)
    out["y"] = (out.Activity == "Active").astype(int)
    return out, {"active": len(active), "inactive": len(inactive), "unspecified": len(unspec),
                 "non_ic50_negatives": int((pd.concat([inactive, unspec]).Type != "IC50").sum())}


def their_features(smiles, keep_bits, want_descriptors):
    """Their features.py: 512-bit RDKFingerprint at maxPath=5 less 55 bits, plus 210 descriptors.

    `keep_bits` and `want_descriptors` both come from their committed matrix's own header, and both
    have to, for different reasons.

    Descriptors: RDKit has gained 7 since they ran (NumAmideBonds, NumAtomStereoCenters,
    NumBridgeheadAtoms, NumHeterocycles, NumSpiroAtoms, NumUnspecifiedAtomStereoCenters, Phi), so
    `Descriptors._descList` as it stands today gives 217 columns and not their 210.

    Fingerprint bits: the list in their `features.py` is not the selection their training matrix
    uses. Their notebook drops correlated columns by label and then low-variance columns by
    POSITION within the already-reduced frame, and concatenates the two into one `all_drops`. The
    positional half is therefore wrong once applied to the original 512 bits, and 21 of the 55
    entries land on the wrong bit. The matrix they committed keeps the original labels and is
    correct; `features.py` is the broken one, so the retained indices are read from the header.
    """
    fn_by_name = dict(Descriptors._descList)
    missing = [n for n in want_descriptors if n not in fn_by_name]
    if missing:
        raise SystemExit(f"this RDKit no longer provides {len(missing)} of their descriptors: "
                         f"{missing[:5]} -- the feature block cannot be rebuilt faithfully")
    fps, descs = [], []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        bits = np.array(AllChem.RDKFingerprint(mol, maxPath=5, fpSize=512))
        fps.append(bits[keep_bits])
        descs.append([fn_by_name[n](mol) for n in want_descriptors])
    return np.hstack([np.array(fps), np.array(descs, dtype=float)])


def rebuild():
    RDLogger.DisableLog("rdApp.*")
    raw = pd.read_csv(RAW)
    built, counts = their_labels(raw)
    theirs = pd.read_csv(THEIR_MATRIX)
    feature_cols = list(theirs.columns[:-1])
    keep_bits = [int(c) for c in feature_cols if c.isdigit()]
    X = their_features(built.Smiles, keep_bits, [c for c in feature_cols if not c.isdigit()])
    features_py_keeps = sorted(set(range(512)) - set(THEIR_DROPS))

    rows = [{"check": "their committed matrix rows", "value": len(theirs), "expected": 336},
            {"check": "bits their features.py selects but their matrix does not",
             "value": len(set(features_py_keeps) - set(keep_bits)), "expected": 0},
            {"check": "our rebuild rows", "value": len(built), "expected": 336},
            {"check": "their committed matrix columns", "value": theirs.shape[1] - 1,
             "expected": 667},
            {"check": "our rebuild columns", "value": X.shape[1], "expected": 667},
            {"check": "their actives", "value": int(theirs.iloc[:, -1].sum()), "expected": 216},
            {"check": "our rebuild actives", "value": int(built.y.sum()), "expected": 216}]
    for k, v in counts.items():
        rows.append({"check": f"rebuild {k}", "value": v, "expected": np.nan})

    # Their matrix carries no SMILES or CID, so rows have to be matched on content, and the only
    # usable key is the fingerprint block. 17 of their molecules share a fingerprint with another
    # (their dedup is by CID, so salts, stereoisomers and close analogues survive as separate
    # rows), and for those a key match cannot say which twin is which. Cell-level comparison is
    # therefore restricted to keys unique on both sides; the ambiguous rows are counted, not
    # guessed, because pairing them arbitrarily would invent descriptor disagreements.
    if X.shape == (theirs.shape[0], theirs.shape[1] - 1):
        n_fp = sum(c.isdigit() for c in feature_cols)
        B = theirs[feature_cols].to_numpy(dtype=float)
        key = lambda M: [tuple(r) for r in M[:, :n_fp].astype(int)]  # noqa: E731
        ours, thrs = key(X), key(B)
        once = {k for k in set(thrs) if thrs.count(k) == 1 and ours.count(k) == 1}
        paired = [(i, thrs.index(k)) for i, k in enumerate(ours) if k in once]
        rows += [{"check": "their fingerprint rows matched by ours", "value":
                  sum(1 for k in ours if k in set(thrs)), "expected": 336},
                 {"check": "fingerprint keys unique on both sides", "value": len(once),
                  "expected": np.nan},
                 {"check": "rows compared cell by cell", "value": len(paired), "expected": np.nan}]
        if paired:
            a = X[[i for i, _ in paired]]
            b = B[[j for _, j in paired]]
            rows += [{"check": "paired fingerprint cells identical",
                      "value": float(np.mean(a[:, :n_fp] == b[:, :n_fp])), "expected": 1.0},
                     {"check": "paired descriptor cells within 1e-4 relative",
                      "value": float(np.mean(np.isclose(a[:, n_fp:], b[:, n_fp:],
                                                        rtol=1e-4, atol=1e-6))), "expected": 1.0},
                     {"check": "descriptor columns reproducing on every paired row",
                      "value": int(np.sum(np.isclose(a[:, n_fp:], b[:, n_fp:], rtol=1e-4,
                                                     atol=1e-6).all(axis=0))),
                      "expected": len(feature_cols) - n_fp},
                     {"check": "paired labels identical",
                      "value": float(np.mean(built.y.to_numpy()[[i for i, _ in paired]]
                                             == theirs.iloc[[j for _, j in paired], -1].to_numpy())),
                      "expected": 1.0}]
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "wichka_features.csv", index=False)
    print(out.to_string(index=False))
    return built, X


def scored(y, p, label, **extra):
    """Their reported metrics plus the ones they did not report.

    Their CV used make_scorer(roc_auc_score) on predicted LABELS, not probabilities, which is not
    an AUROC at all -- it is the mean of sensitivity and specificity. Both are recorded so the
    published figure can be matched and the real one read.
    """
    hard = (p >= 0.5).astype(int)
    m = classification(y, p)
    pc = per_class(y, p, 0.5)
    return {"evaluation": label, "n": len(y), "accuracy": pc["accuracy"],
            "auroc": m["auroc"], "auroc_on_labels": float(np.mean([pc["sensitivity"],
                                                                   pc["specificity"]])),
            "average_precision": m.get("average_precision", np.nan),
            "balanced_accuracy": pc["balanced_accuracy"], "sensitivity": pc["sensitivity"],
            "specificity": pc["specificity"], "fnr": pc["fnr"], "fpr": pc["fpr"],
            "mcc": pc["mcc"], "tp": pc["tp"], "fn": pc["fn"], "fp": pc["fp"], "tn": pc["tn"],
            "hard_label_accuracy": float(np.mean(hard == y)), **extra}


def holdout(X, y):
    """Their notebook cells 26-27 and 45: one stratified 75:25 split, scaler fit on train."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)
    sc = StandardScaler().fit(Xtr)
    np.random.seed(42)  # their resample draws on global NumPy state; see CustomMLPClassifier
    t = time.perf_counter()
    model = their_model().fit(sc.transform(Xtr), ytr)
    p = model.predict_proba(sc.transform(Xte))[:, 1]
    return scored(yte, p, "their 75:25 holdout", fit_seconds=round(time.perf_counter() - t, 1),
                  estimators_fitted=len(model.estimators_))


def cv(X, y, repeats):
    """Their commented cell 46: pipeline(scaler, model), StratifiedKFold(5, shuffle, no seed)."""
    per_repeat = []
    for r in range(repeats):
        folds, t = [], time.perf_counter()
        # No random_state, exactly as published: each repeat sees a different partition, which is
        # what makes their +/- 0.02 a spread over partitions rather than over model seeds.
        for tr, te in StratifiedKFold(n_splits=5, shuffle=True).split(X, y):
            np.random.seed(1000 + r)
            pipe = make_pipeline(StandardScaler(), their_model()).fit(X[tr], y[tr])
            folds.append(scored(y[te], pipe.predict_proba(X[te])[:, 1], "fold"))
        mean = {k: float(np.mean([f[k] for f in folds])) for k in folds[0]
                if isinstance(folds[0][k], (int, float, np.floating))}
        per_repeat.append(mean)
        print(f"  repeat {r + 1}/{repeats}: accuracy {mean['accuracy']:.4f} "
              f"auroc {mean['auroc']:.4f} ({time.perf_counter() - t:.0f}s)", flush=True)
    agg = {"evaluation": f"their 5-fold CV, mean over {repeats} repeats", "repeats": repeats}
    for k in per_repeat[0]:
        agg[k] = float(np.mean([p[k] for p in per_repeat]))
        agg[k + "_sd_over_repeats"] = float(np.std([p[k] for p in per_repeat]))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5, help="their paper used 50")
    ap.add_argument("--section", default="all", choices=["rebuild", "holdout", "cv", "all"])
    a = ap.parse_args()

    built, X_rebuilt = rebuild()
    if a.section == "rebuild":
        return
    # Reproduction runs on the matrix they committed, not on our rebuild: their saved model was
    # trained on that file, so it is the only input that makes the comparison theirs.
    theirs = pd.read_csv(THEIR_MATRIX)
    X, y = theirs.iloc[:, :-1].to_numpy(dtype=float), theirs.iloc[:, -1].to_numpy()

    rows = []
    if a.section in ("holdout", "all"):
        rows.append(holdout(X, y))
    if a.section in ("cv", "all"):
        rows.append(cv(X, y, a.repeats))
    out = ROOT / "results" / "wichka_reproduction.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print()
    print(df[[c for c in ["evaluation", "n", "accuracy", "accuracy_sd_over_repeats", "auroc",
                          "auroc_on_labels", "balanced_accuracy", "sensitivity", "specificity"]
              if c in df]].to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
