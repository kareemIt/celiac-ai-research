"""The confounder adjustments in scripts/analyse_confounders.py.

The script's job is to say whether the publication penalty survives adjustment. Two failures would
be invisible in the output: an adjustment that cannot remove a penalty even when the penalty IS
entirely the confounder, and one that removes a penalty that has nothing to do with it. Both
directions are pinned.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.analyse_confounders import balance_adjusted, fold_assignment


def folds(arm, pos, auroc, deal=0):
    return pd.DataFrame({"arm": arm, "deal": deal, "frac_positive": pos, "auroc": auroc})


def test_a_penalty_that_is_purely_class_balance_adjusts_away():
    """AUROC depends only on the positive rate; the arms differ only in which rates they drew."""
    pos_a = np.linspace(0.20, 0.50, 20)        # publication arm: low positive rates
    pos_b = np.linspace(0.50, 0.80, 20)        # scaffold arm: high ones
    f = lambda p: 0.5 + 0.5 * p                # one shared relationship, no arm effect
    g = pd.concat([folds("paper_grouped", pos_a, f(pos_a)),
                   folds("scaffold_matched", pos_b, f(pos_b))])
    b = balance_adjusted(g)
    assert b["penalty_raw"] < -0.1                       # looks like a large penalty
    assert abs(b["penalty_balance_adjusted"]) < 0.01     # and is entirely explained


def test_a_real_penalty_survives_the_adjustment():
    """Identical positive rates in both arms, but the publication arm scores lower."""
    pos = np.linspace(0.3, 0.7, 20)
    g = pd.concat([folds("paper_grouped", pos, 0.5 + 0.5 * pos - 0.10),
                   folds("scaffold_matched", pos, 0.5 + 0.5 * pos)])
    b = balance_adjusted(g)
    assert np.isclose(b["penalty_raw"], -0.10)
    assert np.isclose(b["penalty_balance_adjusted"], -0.10, atol=0.01)
    assert np.isclose(b["penalty_shared_band"], -0.10, atol=0.01)


def test_the_shared_band_drops_folds_outside_the_overlap():
    g = pd.concat([folds("paper_grouped", [0.1, 0.4, 0.5, 0.9], [0.7, 0.7, 0.7, 0.7]),
                   folds("scaffold_matched", [0.4, 0.5], [0.8, 0.8])])
    b = balance_adjusted(g)
    assert (b["band_lo"], b["band_hi"]) == (0.4, 0.5)
    assert b["n_band_pub"] == 2 and b["n_band_scaffold"] == 2


def test_an_arm_with_no_folds_returns_nothing():
    assert balance_adjusted(folds("paper_grouped", [0.4, 0.5], [0.7, 0.8])) is None


def test_balance_reports_the_spread_that_differs_not_the_mean_that_does_not():
    """The real data's signature: identical means, different spreads."""
    g = pd.concat([folds("paper_grouped", [0.2, 0.5, 0.8], [0.7, 0.7, 0.7]),
                   folds("scaffold_matched", [0.45, 0.5, 0.55], [0.8, 0.8, 0.8])])
    b = balance_adjusted(g)
    assert np.isclose(b["pos_rate_mean_pub"], b["pos_rate_mean_scaffold"])
    assert b["pos_rate_sd_pub"] > 5 * b["pos_rate_sd_scaffold"]


def test_fold_assignment_separates_between_deal_from_within_deal_spread():
    """Deals agree with each other; folds inside a deal do not. The real finding's shape."""
    g = pd.concat([folds("paper_grouped", [0.5] * 4, [0.6, 0.8, 0.7, 0.9], deal=d)
                   for d in range(3)])
    s = fold_assignment(g)["paper_grouped"]
    assert s["deals"] == 3
    assert s["between_deal_sd"] < 1e-9        # every deal has the same mean
    assert s["within_deal_sd"] > 0.1


def test_fold_assignment_skips_an_absent_arm():
    g = folds("paper_grouped", [0.5] * 3, [0.7, 0.8, 0.9], deal=0)
    assert set(fold_assignment(g)) == {"paper_grouped"}


def bands_frame(model, aurocs, counts, bands=("lo", "mid", "hi")):
    return pd.DataFrame({"model": model, "band": list(bands),
                         "auroc": list(aurocs), "n_molecules": list(counts)})


def test_standardisation_removes_a_penalty_that_is_purely_composition(tmp_path, monkeypatch):
    """Identical per-band AUROC in both arms; only the test-set novelty mix differs."""
    import scripts.analyse_confounders as mod
    per_band = (0.60, 0.80, 0.95)
    monkeypatch.setattr(mod, "BANDS", str(tmp_path) + "/{split}.csv")
    bands_frame("rf", per_band, (800, 100, 100)).to_csv(      # mostly novel molecules
        tmp_path / "paper_grouped.csv", index=False)
    bands_frame("rf", per_band, (100, 100, 800)).to_csv(      # mostly familiar ones
        tmp_path / "scaffold_pub.csv", index=False)

    _, std = mod.standardise("ignored")
    r = std.iloc[0]
    assert r["penalty_raw"] < -0.1                          # a large apparent penalty
    assert abs(r["penalty_std_scaffold_mix"]) < 1e-9        # entirely composition
    assert abs(r["penalty_std_pub_mix"]) < 1e-9


def test_standardisation_keeps_a_penalty_present_inside_every_band(tmp_path, monkeypatch):
    import scripts.analyse_confounders as mod
    pub = bands_frame("rf", (0.55, 0.75, 0.90), (300, 300, 300))
    sca = bands_frame("rf", (0.60, 0.80, 0.95), (300, 300, 300))
    monkeypatch.setattr(mod, "BANDS", str(tmp_path) + "/{split}.csv")
    pub.to_csv(tmp_path / "paper_grouped.csv", index=False)
    sca.to_csv(tmp_path / "scaffold_pub.csv", index=False)

    _, std = mod.standardise("ignored")
    r = std.iloc[0]
    assert np.isclose(r["penalty_raw"], -0.05, atol=1e-9)
    assert np.isclose(r["penalty_std_scaffold_mix"], -0.05, atol=1e-9)
    assert np.isclose(r["penalty_std_pub_mix"], -0.05, atol=1e-9)


def test_composition_fractions_sum_to_one_per_arm(tmp_path, monkeypatch):
    import scripts.analyse_confounders as mod
    monkeypatch.setattr(mod, "BANDS", str(tmp_path) + "/{split}.csv")
    bands_frame("rf", (0.6, 0.8, 0.9), (800, 100, 100)).to_csv(
        tmp_path / "paper_grouped.csv", index=False)
    bands_frame("rf", (0.6, 0.8, 0.9), (100, 100, 800)).to_csv(
        tmp_path / "scaffold_pub.csv", index=False)
    comp, _ = mod.standardise("ignored")
    assert np.isclose(comp["pub_frac"].sum(), 1.0)
    assert np.isclose(comp["scaffold_frac"].sum(), 1.0)
