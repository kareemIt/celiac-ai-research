"""Does our hierarchical conformal implementation actually attain its guarantee?

scripts/run_hcp.py implements Theorem 1 of Lee, Barber & Willett (arXiv:2306.06342). The theorem
promises coverage at least 1-alpha for a test point from a previously unseen group, under
hierarchical exchangeability. That is a checkable promise: build synthetic data where hierarchical
exchangeability genuinely holds, run many trials, and confirm empirical coverage sits at or above
the target -- and confirm that the molecule-weighted alternative does NOT, when group sizes vary.

The headline result in the paper rests on the arithmetic in `hcp_threshold`, so these tests drive
the real function rather than a reimplementation of it.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_hcp import (feasible_k1, hcp_threshold, pick_calibration, scores_of, sets_of,
                             split_threshold)


# ------------------------------------------------------------------ the quantile itself

def test_hcp_weights_each_group_equally_not_each_point():
    """One huge group and several small ones. A molecule-weighted quantile is dragged to the huge
    group's scores; the group-weighted one is not."""
    groups = ["big"] * 100 + [f"s{i}" for i in range(9)]
    scores = [0.0] * 100 + [0.9] * 9
    # 10 groups: the big one carries 1/11 in total, the nine small ones 1/11 each.
    t_hcp = hcp_threshold(scores, groups, alpha=0.2)
    t_pooled = split_threshold(scores, alpha=0.2)
    assert t_pooled == 0.0, "pooled quantile follows the 100 cheap points"
    assert t_hcp == 0.9, "group-weighted quantile must reach the small groups"


def test_hcp_reduces_to_split_conformal_when_every_group_has_one_member():
    rng = np.random.default_rng(0)
    s = rng.random(50)
    g = [f"g{i}" for i in range(50)]
    # with N_k == 1 the HCP distribution is uniform on the 50 scores plus a 1/51 atom at infinity,
    # which is exactly the split-conformal ceil((n+1)(1-alpha)) rule
    for alpha in (0.1, 0.2, 0.3):
        assert hcp_threshold(s, g, alpha) == pytest.approx(split_threshold(s, alpha))


def test_hcp_returns_infinity_when_there_are_too_few_groups():
    """alpha < 1/(K1+1) cannot be supported: the quantile lands on the infinite atom."""
    s = [0.1, 0.2, 0.3, 0.4]
    g = ["a", "b", "c", "d"]
    assert feasible_k1(0.1) == 9
    assert np.isinf(hcp_threshold(s, g, alpha=0.1))        # needs 9 groups, has 4
    assert np.isfinite(hcp_threshold(s, g, alpha=0.25))    # needs 3, has 4


def test_feasible_k1_matches_the_atom_arithmetic():
    for alpha in (0.05, 0.1, 0.2, 0.25, 0.5):
        k = feasible_k1(alpha)
        g = [f"g{i}" for i in range(k)]
        s = list(np.linspace(0, 1, k))
        assert np.isfinite(hcp_threshold(s, g, alpha)), alpha
        if k > 1:
            assert np.isinf(hcp_threshold(s[:k - 1], g[:k - 1], alpha)), alpha


def test_empty_calibration_set_is_infinite_not_an_error():
    assert np.isinf(hcp_threshold([], [], 0.1))
    assert np.isinf(split_threshold([], 0.1))


# ------------------------------------------------------------------ coverage under the assumption

def hierarchical_trial(rng, n_groups, alpha, sizes=None, method="hcp"):
    """One trial of a world where hierarchical exchangeability holds by construction.

    Each group draws its own latent difficulty mu_k ~ N(0,1) (groups i.i.d., hence exchangeable),
    then its members draw scores N(mu_k, 0.5) (i.i.d. within a group, hence exchangeable). A fresh
    group is drawn the same way and one of its members is the test point. Returns whether the test
    score fell under the threshold, which is exactly the covered event.
    """
    sizes = sizes or [10] * n_groups
    mus = rng.normal(0, 1, n_groups)
    scores, groups = [], []
    for k, (mu, n) in enumerate(zip(mus, sizes)):
        scores += list(rng.normal(mu, 0.5, n))
        groups += [f"g{k}"] * n
    t = (hcp_threshold(scores, groups, alpha) if method == "hcp"
         else split_threshold(scores, alpha))
    test_mu = rng.normal(0, 1)
    return rng.normal(test_mu, 0.5) <= t


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_hcp_attains_its_guarantee_on_equal_sized_groups(alpha):
    rng = np.random.default_rng(1)
    n = 3000
    hits = sum(hierarchical_trial(rng, 20, alpha) for _ in range(n))
    cov = hits / n
    se = (alpha * (1 - alpha) / n) ** 0.5
    # Theorem 1: coverage >= 1 - alpha. Allow 4 standard errors of slack below target for
    # Monte-Carlo noise; the upper bound 1-alpha+2/(K1+1) is loose, so only the floor is asserted.
    assert cov >= 1 - alpha - 4 * se, f"coverage {cov:.3f} below the {1 - alpha} floor"


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_hcp_still_attains_it_when_group_sizes_are_wildly_unequal(alpha):
    """The case our real data is in: publications run 10 to 224 molecules. Theorem 1 tolerates
    unequal sizes because each group carries mass 1/(K1+1) regardless of its size."""
    rng = np.random.default_rng(2)
    sizes = [2, 3, 5, 8, 10, 15, 20, 40, 80, 200] * 2
    n = 3000
    hits = sum(hierarchical_trial(rng, len(sizes), alpha, sizes=sizes) for _ in range(n))
    cov = hits / n
    se = (alpha * (1 - alpha) / n) ** 0.5
    assert cov >= 1 - alpha - 4 * se, f"coverage {cov:.3f} below the {1 - alpha} floor"


def test_molecule_weighted_calibration_undercovers_when_sizes_are_unequal():
    """Why the method matters. With one dominant group, the pooled quantile tracks that group and
    a test point from a fresh group is under-covered, while HCP is not."""
    rng = np.random.default_rng(3)
    sizes = [400] + [5] * 19
    n = 2000
    alpha = 0.1
    pooled = sum(hierarchical_trial(rng, len(sizes), alpha, sizes, "pooled") for _ in range(n)) / n
    rng = np.random.default_rng(3)
    hcp = sum(hierarchical_trial(rng, len(sizes), alpha, sizes, "hcp") for _ in range(n)) / n
    assert pooled < 1 - alpha - 0.01, f"pooled coverage {pooled:.3f} should fall short"
    assert hcp > pooled, f"hcp {hcp:.3f} should beat pooled {pooled:.3f}"


# ------------------------------------------------------------------ set construction

def test_sets_and_scores_agree_with_each_other():
    """The true label is in the set exactly when its nonconformity score is under the threshold."""
    rng = np.random.default_rng(4)
    p = rng.random(200)
    y = rng.integers(0, 2, 200)
    for t in (0.0, 0.3, 0.5, 0.8, 1.0):
        covered = sets_of(p, t)[np.arange(len(y)), y]
        assert (covered == (scores_of(p, y) <= t)).all(), t


def test_an_infinite_threshold_gives_the_trivial_set_and_a_negative_one_gives_empty():
    p = np.array([0.01, 0.5, 0.99])
    assert sets_of(p, np.inf).all()
    assert not sets_of(p, -1.0).any()


def test_single_class_groups_are_handled():
    """20 of 83 CDK2 publications are single-class. Coverage is still defined for them, unlike
    AUROC, so they must not be dropped."""
    p = np.array([0.2, 0.3, 0.25, 0.4])
    y = np.zeros(4, dtype=int)                       # all inactive
    s = scores_of(p, y)
    t = hcp_threshold(np.tile(s, 3), ["a"] * 4 + ["b"] * 4 + ["c"] * 4, alpha=0.25)
    assert np.isfinite(t)
    assert sets_of(p, t)[np.arange(4), y].all()


# ------------------------------------------------------------------ selection bias

def test_group_selection_strategies_pick_what_they_say():
    import pandas as pd
    papers = pd.Series(["big"] * 50 + ["mid"] * 20 + ["small"] * 5 + ["tiny"] * 2)
    rng = np.random.default_rng(5)
    assert pick_calibration(papers, 2, "smallest", rng) == {"tiny", "small"}
    assert pick_calibration(papers, 2, "largest", rng) == {"big", "mid"}
    assert len(pick_calibration(papers, 2, "random", rng)) == 2


def test_smallest_first_selection_biases_the_calibration_group_sizes():
    """The bias docs/paper.md 8 records in our own group_calibration: picking publications
    smallest-first makes the calibration groups systematically smaller than the test group."""
    import pandas as pd
    sizes = {f"p{i}": n for i, n in enumerate([5, 8, 10, 20, 40, 60, 100])}
    papers = pd.Series([k for k, n in sizes.items() for _ in range(n)])
    rng = np.random.default_rng(6)
    small = pick_calibration(papers, 3, "smallest", rng)
    mean_small = np.mean([sizes[k] for k in small])
    mean_all = np.mean(list(sizes.values()))
    assert mean_small < mean_all / 2
