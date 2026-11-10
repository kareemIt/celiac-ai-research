"""The overfitting measurement: configuration lookup, inner grouping, and the reproduction gate."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_overfitting import config_for, gate, inner_groups, one_fold, tie_band  # noqa: E402
from run_sota import _fmt, expand, sklearn_grids  # noqa: E402


def test_config_lookup_recovers_the_keys_the_formatter_drops():
    """run_sota._fmt hides n_jobs, max_iter and tree_method. Parsing the printed string back would
    lose them silently; looking the grid entry up cannot."""
    _, _, params = sklearn_grids()["rf"]
    cfg = expand(params)[0]
    found = config_for(params, _fmt(cfg))
    assert found == cfg
    assert found["n_jobs"] == -1  # dropped by _fmt, still present here

    _, _, lr_params = sklearn_grids()["logreg"]
    lr = config_for(lr_params, _fmt(expand(lr_params)[0]))
    assert lr["max_iter"] == 5000


def test_every_grid_entry_prints_uniquely_so_the_lookup_is_well_defined():
    for name, (_, _, params) in sklearn_grids().items():
        printed = [_fmt(c) for c in expand(params)]
        assert len(set(printed)) == len(printed), name


def test_unknown_configuration_is_refused_rather_than_guessed():
    _, _, params = sklearn_grids()["rf"]
    with pytest.raises(SystemExit):
        config_for(params, "max_depth=999")


def _fold(smiles, papers=None):
    n = len(smiles)
    return {"smiles": smiles, "papers": papers,
            "y": np.array([i % 2 for i in range(n)]),
            "idx": (np.arange(n), np.array([], dtype=int), np.array([], dtype=int))}


def test_inner_groups_follow_the_outer_grouping():
    smiles = ["c1ccccc1C", "c1ccccc1CC", "C1CCCCC1C", "CCO"]
    papers = ["P1", "P1", "P2", "P2"]
    fold = _fold(smiles, papers)
    tr = np.arange(4)

    assert inner_groups(fold, "paper_lopo", tr) == papers
    assert inner_groups(fold, "paper_grouped", tr) == papers
    scaf = inner_groups(fold, "scaffold_pub", tr)
    assert scaf[0] == scaf[1]          # benzene analogues share a scaffold
    assert scaf[0] != scaf[2]          # cyclohexane does not
    assert inner_groups(fold, "random", tr) is None   # nothing to honour


def test_paper_grouping_falls_back_when_the_split_file_has_no_publications():
    fold = _fold(["CCO", "CCN"], papers=None)
    assert inner_groups(fold, "paper_lopo", np.arange(2)) is None


def test_gate_accepts_an_exact_refit_and_rejects_a_drifted_one():
    ok = pd.DataFrame({"test": [0.8, 0.9], "test_committed": [0.8, 0.9 + 1e-9], "tie_band": 0.0,
                       "target": "tg2", "model": "rf", "split": "scaffold", "seed": [0, 1]})
    worst, n, relaxed = gate(ok)
    assert n == 2 and worst < 1e-6 and relaxed == 0

    bad = ok.copy()
    bad.loc[0, "test"] = 0.75
    with pytest.raises(SystemExit):
        gate(bad)


def test_gate_ignores_folds_with_no_committed_score():
    d = pd.DataFrame({"test": [0.8, np.nan], "test_committed": [0.8, 0.9], "tie_band": 0.0,
                      "target": "tg2", "model": "rf", "split": "scaffold", "seed": [0, 1]})
    _, n, _ = gate(d)
    assert n == 1


def test_gate_allows_a_tied_pair_but_not_a_pair_more():
    """The committed AUROC was produced under the same last-bit noise, so a difference one tied
    pair wide is reproduction. Two pairs wide, on a fold with one tie, is not."""
    one_pair = 1 / 902
    d = pd.DataFrame({"test": [0.8 + one_pair], "test_committed": [0.8], "tie_band": [one_pair],
                      "target": "tg2", "model": "rf", "split": "scaffold", "seed": [15]})
    worst, n, relaxed = gate(d)
    assert n == 1 and relaxed == 1 and worst == pytest.approx(one_pair)

    with pytest.raises(SystemExit):
        gate(d.assign(test=0.8 + 2 * one_pair))


def test_measured_determinism_spread_is_exactly_what_tied_pairs_allow():
    """The §8 limitation claims the observed AUROC spread IS tie-breaking and nothing else. That is
    falsifiable: the spread must be a whole number of pairs, and no more pairs than are tied."""
    path = ROOT / "results" / "rf_determinism.csv"
    if not path.exists():
        pytest.skip("run scripts/measure_rf_determinism.py first")
    d = pd.read_csv(path).iloc[0]
    assert d.one_pair == pytest.approx(1 / d.pairs)
    assert d.pairs == d.n_pos * d.n_neg
    flipped = d.spread / d.one_pair
    assert flipped == pytest.approx(round(flipped))          # a whole number of pairs
    assert 0 < flipped <= d.tied_pairs                       # and only tied ones could flip
    assert d.max_prob_difference < 1e-15                     # caused by last-bit noise, not drift


def test_tie_band_counts_every_tied_pair_and_is_undefined_for_one_class():
    # Two actives and one inactive all share p=0.5; the fourth molecule is unambiguous.
    y = np.array([1, 1, 0, 0])
    p = np.array([0.5, 0.5, 0.5, 0.1])
    assert tie_band(y, p) == pytest.approx(2 / 4)   # 2 tied pairs out of 2 pos x 2 neg
    assert np.isnan(tie_band(np.array([1, 1]), np.array([0.5, 0.9])))   # no negatives: undefined
    assert tie_band(np.array([1, 0]), np.array([0.9, 0.1])) == 0.0      # no ties at all


def test_one_fold_reports_in_sample_above_out_of_fold_on_memorisable_data():
    """Labels that are pure noise given the features: a fully grown forest memorises the training
    fold, so in_sample must be far above oof. That is overfitting, measured on one distribution --
    the shape the experiment is built to detect."""
    rng = np.random.default_rng(0)
    # Substituted benzenes, not alkyl chains: radius-2 Morgan bits saturate on a long chain, so
    # "CCCCO" and "CCCCCCO" share a fingerprint and nothing could be memorised. Two varying
    # substituents give 60 genuinely distinct feature vectors.
    subs = ["C", "CC", "CCC", "N", "O", "F", "Cl", "Br", "C(=O)O", "CO"]
    smiles = [f"{a}c1ccc({b})cc1" for a in subs for b in subs][:60]
    y = rng.integers(0, 2, len(smiles))
    _, _, params = sklearn_grids()["rf"]
    cfg = next(c for c in expand(params) if c["max_depth"] is None and c["min_samples_leaf"] == 1)
    fold = {"smiles": smiles, "papers": None, "y": y,
            "idx": (np.arange(40), np.array([], dtype=int), np.arange(40, 60))}
    row = one_fold("rf", fold, "random", 0, _fmt(cfg))
    assert row["in_sample"] > 0.95
    assert row["oof"] < row["in_sample"]
    assert row["inner_folds"] == 5


def test_summary_calls_shift_readable_only_when_the_control_is_far_below_it():
    """The pre-registered control check: a shift term is readable when its interval clears zero and
    the learning-curve term is well under it. A control a third the size is not good enough."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from summarise_overfitting import summarise

    rng = np.random.default_rng(0)
    big = pd.DataFrame({"overfit_gap": np.full(40, 0.2), "shift_gap": np.full(40, 0.15),
                        "size_effect": np.full(40, 0.005)})
    assert summarise(big, rng)["decides_shift"]
    # Same shift, control a third of it: undecided rather than accepted.
    assert not summarise(big.assign(size_effect=0.05), rng)["decides_shift"]
    # A shift term whose interval includes zero is not readable however small the control.
    noisy = big.assign(shift_gap=np.linspace(-0.2, 0.25, 40), size_effect=1e-6)
    assert not summarise(noisy, rng)["decides_shift"]


def test_the_committed_decomposition_agrees_with_what_5_21_claims():
    """Guards the three statements §5.21 rests on against a silent change to the results file."""
    path = ROOT / "results" / "overfitting_tier1.csv"
    if not path.exists():
        pytest.skip("run scripts/run_overfitting.py first")
    d = pd.read_csv(path).dropna(subset=["test_small"])
    # 1. in-sample is at the ceiling for every family, so overfit_gap cannot carry the claim.
    assert d.groupby("model").in_sample.median().min() > 0.95
    # 2. the shift term is readable only on cdk2/maob leave-one-publication-out.
    lopo = d[d.target.isin(["cdk2", "maob"]) & (d.split == "paper_lopo")]
    assert lopo.shift_gap.mean() > 20 * abs(lopo.size_effect.mean())
    # 3. and not on tg2, where the control is the same order as the effect.
    tg2 = d[(d.target == "tg2") & (d.split == "paper_lopo")]
    assert tg2.shift_gap.mean() < 4 * abs(tg2.size_effect.mean())
