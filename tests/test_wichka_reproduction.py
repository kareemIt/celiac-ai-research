"""The Wichka & Lai reproduction: their label rules, their corrected base estimator, their metrics.

These guard the three things that would silently invalidate the reproduction: a label rule drifting
away from theirs, the corrected `CustomMLPClassifier` ceasing to work as an AdaBoost base estimator,
and the reproduction's own committed numbers moving without anyone noticing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reproduce_wichka import (ADA_KW, MLP_KW, THEIR_DROPS, CustomMLPClassifier,  # noqa: E402
                             their_labels, their_model)

RAW = ROOT / "data" / "raw" / "tg2" / "TG2_compound_smiles.csv"
MATRIX = ROOT / "data" / "raw" / "tg2" / "clf_celiac_desc_fing_data.csv"


def test_published_hyperparameters_are_the_ones_in_their_notebook():
    """Transcribed from notebook cell 45. A typo here invalidates everything downstream, and the
    learning rate in particular is quoted to 18 digits in their source."""
    assert MLP_KW == {"hidden_layer_sizes": (50,), "activation": "relu",
                      "learning_rate_init": 0.001, "max_iter": 1000}
    assert ADA_KW == {"n_estimators": 279, "learning_rate": 0.022977463078217686,
                      "random_state": 42}


def test_the_corrected_base_estimator_actually_boosts():
    """Correction 1 (CustomMLPClassifier for the undefined MLP_Base_For_AdaBoost) and correction 2
    (the _fit signature) are only right if AdaBoost can fit a real ensemble with them. Their class
    exists to consume sample_weight, which is exactly what AdaBoost requires of a base estimator."""
    from sklearn.utils.validation import has_fit_parameter
    assert has_fit_parameter(CustomMLPClassifier(), "sample_weight")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = their_model().set_params(n_estimators=3).fit(X, y)
    assert len(model.estimators_) == 3
    assert all(isinstance(e, CustomMLPClassifier) for e in model.estimators_)
    assert model.predict_proba(X).shape == (60, 2)


def test_their_label_rules_recover_their_published_class_counts():
    """216 active / 45 inactive / 75 unspecified, totalling the 336 molecules of their matrix. This
    is the strongest available check that the label reconstruction is theirs and not ours."""
    if not RAW.exists():
        pytest.skip("their TG2_compound_smiles.csv is not present")
    built, counts = their_labels(pd.read_csv(RAW))
    assert counts == {"active": 216, "inactive": 45, "unspecified": 75, "non_ic50_negatives": 44}
    assert len(built) == 336
    assert int(built.y.sum()) == 216


def test_the_unspecified_negatives_carry_measured_values_not_missing_ones():
    """Phase 3 turns on this: their 'unspecified' compounds are not molecules without evidence,
    they are molecules with a measured value between 10 and 301 whose PubChem outcome field is
    blank. Anything that silently admitted a missing value would change what the class means."""
    if not RAW.exists():
        pytest.skip("their TG2_compound_smiles.csv is not present")
    built, _ = their_labels(pd.read_csv(RAW))
    unspec = built[built.Activity == "Unspecified"]
    assert len(unspec) == 75
    assert unspec.Value.notna().all()
    assert unspec.Value.between(10, 301, inclusive="neither").all()
    # Their inactive branch applies no value filter at all, so it may hold missing values.
    assert (built.Activity == "Inactive").sum() == 45


def test_their_features_py_drop_list_disagrees_with_their_training_matrix():
    """A finding, pinned as a test so it cannot quietly stop being true: 21 of the 55 bits their
    deployed features.py drops are not the bits their training matrix dropped, because their
    notebook mixes column labels with positions when it builds the list."""
    if not MATRIX.exists():
        pytest.skip("their feature matrix is not present")
    header_bits = {int(c) for c in pd.read_csv(MATRIX, nrows=0).columns[:-1] if c.isdigit()}
    features_py_bits = set(range(512)) - set(THEIR_DROPS)
    assert len(header_bits) == len(features_py_bits) == 457
    assert len(features_py_bits - header_bits) == 21


def test_the_committed_reproduction_still_says_what_phase_1_reports():
    path = ROOT / "results" / "wichka_reproduction.csv"
    if not path.exists():
        pytest.skip("run scripts/reproduce_wichka.py first")
    d = pd.read_csv(path).set_index("evaluation")
    hold = d.loc["their 75:25 holdout"]
    assert hold.n == 84                                  # 25% of 336
    assert hold.tp + hold.fn + hold.fp + hold.tn == 84   # confusion matrix closes
    assert 0.85 < hold.accuracy < 1.0
    # Their CV scored AUC on hard labels, which is the mean of sensitivity and specificity, not an
    # AUROC. The two must not be allowed to drift into being reported as the same thing.
    assert hold.auroc_on_labels == pytest.approx((hold.sensitivity + hold.specificity) / 2)
    assert hold.auroc != pytest.approx(hold.auroc_on_labels, abs=1e-6)


def test_the_feature_rebuild_still_matches_their_matrix():
    path = ROOT / "results" / "wichka_features.csv"
    if not path.exists():
        pytest.skip("run scripts/reproduce_wichka.py --section rebuild first")
    d = pd.read_csv(path).set_index("check")
    assert d.loc["our rebuild rows", "value"] == 336
    assert d.loc["our rebuild columns", "value"] == 667
    assert d.loc["paired fingerprint cells identical", "value"] == 1.0
    assert d.loc["paired labels identical", "value"] == 1.0
    # 207 of 210 descriptors reproduce exactly; the three that do not are RDKit definition changes.
    assert d.loc["descriptor columns reproducing on every paired row", "value"] >= 207
