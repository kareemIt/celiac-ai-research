"""Activity-cliff enumeration: the similarity maths and the MoleculeACE criteria."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyse_cliffs import (LOG_FOLD_CUT, SIM_CUT, _edit_distance, discordant_pairs,  # noqa: E402
                            fingerprints, generic_scaffold, holm, levenshtein_similar,
                            tanimoto_rows)


def brute_edit_distance(a, b):
    """Full Levenshtein DP with no early exit, to check the budgeted version against."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def test_edit_distance_matches_a_plain_dp_whenever_it_is_within_budget():
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(300):
        a = "".join(rng.choice(list("CNOc1()=@"), size=int(rng.integers(3, 25))))
        b = "".join(rng.choice(list("CNOc1()=@"), size=int(rng.integers(3, 25))))
        truth = brute_edit_distance(a, b)
        for k in range(0, 8):
            got = _edit_distance(a, b, k)
            if truth <= k:
                assert got == truth, (a, b, k, got, truth)
                checked += 1
            else:
                assert got > k, (a, b, k, got, truth)  # may be the sentinel, never a wrong value
    assert checked > 50  # the budgeted branch was actually exercised


def test_edit_distance_known_cases():
    assert _edit_distance("", "", 0) == 0
    assert _edit_distance("CCO", "CCO", 0) == 0
    assert _edit_distance("CCO", "CCN", 1) == 1
    assert _edit_distance("CCO", "CC", 1) == 1
    assert _edit_distance("CCO", "", 3) == 3
    assert _edit_distance("CCO", "", 1) > 1  # abandoned, not silently wrong


def test_levenshtein_similar_needs_a_near_identical_string():
    long_a = "CCCCCCCCCCCCCCCCCCCCO"      # 21 chars, budget floor(0.1*21) = 2
    long_b = "CCCCCCCCCCCCCCCCCCCCN"      # one substitution away
    far = "c1ccc2ccccc2c1"
    got = levenshtein_similar([long_a], [long_b, far])
    assert (0, 0) in got and (0, 1) not in got


def test_levenshtein_prefilters_never_discard_a_qualifying_pair():
    """The length and histogram bounds are lower bounds on the distance, so the fast path must
    agree with an exhaustive check on the same strings."""
    rng = np.random.default_rng(1)
    pos = ["".join(rng.choice(list("CNOc1"), size=20)) for _ in range(12)]
    neg = ["".join(rng.choice(list("CNOc1"), size=20)) for _ in range(12)]
    fast = levenshtein_similar(pos, neg, cut=0.7)  # loose cut so some pairs actually qualify
    slow = {(i, j) for i, a in enumerate(pos) for j, b in enumerate(neg)
            if 1 - brute_edit_distance(a, b) / max(len(a), len(b)) >= 0.7}
    assert fast == slow


def test_tanimoto_against_hand_computed_bits():
    a = np.array([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=np.uint8)
    b = np.array([[1, 0, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    # row0 vs b: {1,1,0,0}&{1,0,0,0} -> 1/2 ; {1,1,0,0}&{0,0,1,1} -> 0/4
    # row1 vs b: 1/4 ; 2/4
    assert np.allclose(tanimoto_rows(a, b), [[0.5, 0.0], [0.25, 0.5]])


def test_tanimoto_of_identical_is_one_and_empty_is_zero():
    a = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    assert np.allclose(tanimoto_rows(a, a), [[1.0]])
    zero = np.zeros((1, 4), dtype=np.uint8)
    assert np.allclose(tanimoto_rows(zero, zero), [[0.0]])  # 0/0 must not be nan


def test_generic_scaffold_collapses_atom_identity():
    """Pyridine and benzene share a generic scaffold: MoleculeACE's scaffold test is meant to be
    blind to which atoms sit in the ring."""
    assert generic_scaffold("c1ccccc1") == generic_scaffold("c1ccncc1")
    assert generic_scaffold("not a molecule") == ""


def test_fingerprints_tolerate_an_empty_scaffold_string():
    fp = fingerprints(["", "CCO"])
    assert fp.shape == (2, 1024) and fp[0].sum() == 0 and fp[1].sum() > 0


def test_discordant_pairs_uses_the_scaffold_test_not_only_fingerprints():
    """Benzanilide against its methyl analog: Morgan Tanimoto is only 0.55, but the two share a
    generic scaffold, so MoleculeACE's OR of the two tests must still call them similar.
    Naphthalene shares neither and must be left out."""
    analog, methyl, distant = "O=C(Nc1ccccc1)c1ccccc1", "O=C(Nc1ccccc1)c1ccccc1C", "c1ccc2ccccc2c1"
    pairs = discordant_pairs(pd.DataFrame({"smiles": [analog, methyl, distant], "y": [1, 0, 0]}))
    got = set(zip(pairs.smiles_pos, pairs.smiles_neg))
    assert (analog, methyl) in got
    assert (analog, distant) not in got
    assert (pairs.similarity >= SIM_CUT).all()
    assert tanimoto_rows(fingerprints([analog]), fingerprints([methyl]))[0, 0] < SIM_CUT


def test_only_discordant_pairs_are_enumerated():
    mols = pd.DataFrame({"smiles": ["CCO", "CCO.O"], "y": [1, 1]})
    assert discordant_pairs(mols).empty


def test_cliff_cut_is_one_log_unit():
    assert LOG_FOLD_CUT == 1.0  # fold change > 10, MoleculeACE's potency_fold default


def test_holm_matches_the_projects_step_down():
    """Step-down with the running maximum: 0.04 would get x1 = 0.04 on its own, but cannot fall
    below the 0.06 already assigned to the smaller 0.03."""
    assert holm(np.array([0.01, 0.04, 0.03])).tolist() == pytest.approx([0.03, 0.06, 0.06])
    assert holm(np.array([0.5, 0.9])).tolist() == pytest.approx([1.0, 1.0])  # clipped at 1
