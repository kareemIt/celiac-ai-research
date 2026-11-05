"""The cross-target pooling in scripts/analyse_three_targets.py.

This script produces the replication's headline numbers, so the two ways it could lie quietly are
pinned: an inverse-variance mean that ignores the weights (which would let one noisy target
dominate), and a heterogeneity statistic that stays near zero when the targets in fact disagree.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.analyse_three_targets import holm, paired_deltas, pool, rank_agreement


def test_pooling_weights_the_precise_target_more():
    """Same two estimates, but one target measured ten times more precisely."""
    p = pool({"a": (0.10, 0.001, 10), "b": (0.00, 0.010, 10)})
    assert p["pooled"] > 0.09                      # pulled almost entirely to the precise one
    unweighted = pool({"a": (0.10, 0.005, 10), "b": (0.00, 0.005, 10)})
    assert np.isclose(unweighted["pooled"], 0.05)  # equal precision -> plain mean


def test_agreeing_targets_give_near_zero_heterogeneity():
    p = pool({"a": (0.030, 0.01, 10), "b": (0.031, 0.01, 10), "c": (0.029, 0.01, 10)})
    assert p["q"] < 0.1
    assert p["df"] == 2


def test_disagreeing_targets_give_large_heterogeneity():
    """A pooled mean near zero must not hide targets that point opposite ways."""
    p = pool({"a": (+0.10, 0.01, 10), "b": (-0.10, 0.01, 10)})
    assert abs(p["pooled"]) < 0.001
    assert p["q"] > 100                            # the number the reader needs to distrust it


def test_a_target_with_no_spread_is_dropped_not_given_infinite_weight():
    p = pool({"a": (0.5, 0.0, 10), "b": (0.0, 0.01, 10), "c": (0.0, 0.01, 10)})
    assert p["targets"] == "b,c"
    assert np.isclose(p["pooled"], 0.0)


def test_pooling_needs_at_least_two_usable_targets():
    assert pool({"a": (0.1, 0.01, 10)}) is None
    assert pool({"a": (0.1, 0.0, 10), "b": (0.2, 0.0, 10)}) is None


def test_paired_deltas_pair_on_fold_index_within_one_split():
    sota = pd.DataFrame({
        "split": ["s"] * 6, "seed": [0, 1, 2] * 2,
        "model": ["rf"] * 3 + ["chemeleon"] * 3,
        "auroc": [0.80, 0.70, 0.90, 0.85, 0.75, 0.95],
    })
    v = paired_deltas(sota, "s", "chemeleon")
    assert np.allclose(v, [0.05, 0.05, 0.05])      # fold-wise, not mean-minus-mean


def test_paired_deltas_ignore_folds_a_model_is_missing_from():
    sota = pd.DataFrame({
        "split": ["s"] * 7, "seed": [0, 1, 2, 3] + [0, 1, 2],
        "model": ["rf"] * 4 + ["chemeleon"] * 3,
        "auroc": [0.80, 0.70, 0.90, 0.60, 0.85, 0.75, 0.95],
    })
    assert len(paired_deltas(sota, "s", "chemeleon")) == 3


def test_paired_deltas_refuse_too_few_folds():
    sota = pd.DataFrame({"split": ["s"] * 4, "seed": [0, 1] * 2,
                         "model": ["rf"] * 2 + ["chemeleon"] * 2,
                         "auroc": [0.8, 0.7, 0.85, 0.75]})
    assert paired_deltas(sota, "s", "chemeleon") is None


def test_rank_agreement_reports_every_pair():
    tab = pd.DataFrame({"tg2": [-0.3, -0.2, -0.1], "cdk2": [-0.3, -0.2, -0.1],
                        "maob": [-0.1, -0.2, -0.3]}, index=list("abc"))
    a = rank_agreement(tab)
    assert len(a) == 3
    assert np.isclose(a.set_index(["target_a", "target_b"]).loc[("tg2", "cdk2"), "spearman"], 1.0)
    assert np.isclose(a.set_index(["target_a", "target_b"]).loc[("tg2", "maob"), "spearman"], -1.0)


def test_holm_scales_the_smallest_p_by_the_contrast_count():
    out = holm(pd.DataFrame({"model": list("abc"), "p": [0.01, 0.2, 0.9]}))
    assert np.isclose(out["p_holm"].iloc[0], 0.03)
    assert list(out["p_holm"]) == sorted(out["p_holm"])
