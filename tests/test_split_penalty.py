"""The cross-split penalty in scripts/analyse_split_penalty.py.

The sign of `penalty` is the replication's whole claim: negative means the publication protocol
scores lower. A swapped subtraction would invert the finding while leaving every interval and
p-value looking plausible, so the sign, the Holm correction and the ordering are pinned here.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.analyse_split_penalty import holm, penalty


def folds(model, split, values):
    return pd.DataFrame({"model": model, "split": split, "auroc": values})


def test_a_model_scoring_lower_on_the_shift_arm_gets_a_negative_penalty():
    d = pd.concat([folds("rf", "scaffold_pub", [0.90, 0.91, 0.89, 0.90]),
                   folds("rf", "paper_grouped", [0.80, 0.81, 0.79, 0.80])])
    r = penalty(d, "scaffold_pub", "paper_grouped").iloc[0]
    assert r["penalty"] < 0
    assert np.isclose(r["penalty"], -0.10)
    assert r["ci_hi"] < 0                     # the interval excludes "no penalty"
    assert np.isclose(r["auroc_baseline"], 0.90)


def test_the_arms_are_not_treated_as_paired():
    """Reordering one arm's folds must not change the result; a paired test would."""
    a, b = [0.90, 0.80, 0.85, 0.95], [0.70, 0.60, 0.75, 0.65]
    d = pd.concat([folds("rf", "scaffold_pub", a), folds("rf", "paper_grouped", b)])
    shuffled = pd.concat([folds("rf", "scaffold_pub", a),
                          folds("rf", "paper_grouped", list(reversed(b)))])
    one, two = penalty(d, "scaffold_pub", "paper_grouped"), penalty(
        shuffled, "scaffold_pub", "paper_grouped")
    assert np.isclose(one.iloc[0]["p_welch"], two.iloc[0]["p_welch"])


def test_a_model_present_in_only_one_arm_is_dropped():
    d = pd.concat([folds("rf", "scaffold_pub", [0.90, 0.91, 0.89]),
                   folds("rf", "paper_grouped", [0.80, 0.82, 0.78]),
                   folds("xgb", "scaffold_pub", [0.90, 0.91, 0.89])])
    assert set(penalty(d, "scaffold_pub", "paper_grouped")["model"]) == {"rf"}


def test_holm_is_monotone_and_scales_the_smallest_p_by_the_model_count():
    out = holm(pd.DataFrame({"model": list("abcd"), "penalty": [-0.4, -0.3, -0.2, -0.1],
                             "p_welch": [0.001, 0.02, 0.04, 0.5]}))
    byp = out.sort_values("p_welch")
    assert np.isclose(byp["p_holm"].iloc[0], 0.004)          # 4 x 0.001
    assert list(byp["p_holm"]) == sorted(byp["p_holm"])      # never decreases
    assert list(out["penalty"]) == sorted(out["penalty"])    # reported worst-first


def test_rows_are_ordered_by_penalty_not_by_p_value():
    d = pd.concat([folds("rf", "scaffold_pub", [0.90, 0.901, 0.899, 0.90]),
                   folds("rf", "paper_grouped", [0.85, 0.851, 0.849, 0.85]),
                   folds("xgb", "scaffold_pub", [0.90, 0.80, 0.95, 0.85]),
                   folds("xgb", "paper_grouped", [0.60, 0.50, 0.70, 0.55])])
    out = penalty(d, "scaffold_pub", "paper_grouped")
    assert list(out["model"]) == ["xgb", "rf"]               # xgb loses more
    assert out["p_welch"].iloc[0] > out["p_welch"].iloc[1]   # but is the less certain of the two
