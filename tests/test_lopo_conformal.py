"""The conformal machinery and the two calibration regimes in scripts/run_lopo_conformal.py.

The whole experiment is a comparison of measured coverage against a target, so an error in the
quantile would produce a plausible-looking wrong answer rather than a crash. These cases pin the
finite-sample correction, the behaviour when the calibration set is too small to support the
target at all, and that calibration and fitting sets never overlap.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_lopo_conformal import (conformal_sets, group_calibration, iid_calibration,
                                        risk_coverage)


def _exchangeable(n, seed=0):
    """A calibrated classifier's outputs on exchangeable data: p is the true P(y=1)."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    y = (rng.uniform(size=n) < p).astype(int)
    return p, y


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_coverage_holds_on_exchangeable_data(alpha):
    p_cal, y_cal = _exchangeable(2000, seed=1)
    p_te, y_te = _exchangeable(4000, seed=2)
    sets, _ = conformal_sets(p_cal, y_cal, p_te, alpha, mondrian=False)
    covered = sets[np.arange(len(y_te)), y_te].mean()
    assert covered >= 1 - alpha - 0.02, f"under-covered: {covered:.3f} vs {1 - alpha}"
    assert covered <= 1 - alpha + 0.05, f"far too conservative: {covered:.3f}"


def test_quantile_uses_the_finite_sample_correction():
    # n=9, alpha=0.1 -> k = ceil(10 * 0.9) = 9, so the quantile is the LARGEST score, not the 90th
    # percentile of nine points
    p_cal = np.linspace(0.05, 0.95, 9)
    y_cal = np.ones(9, dtype=int)
    _, q = conformal_sets(p_cal, y_cal, np.array([0.5]), 0.1, mondrian=False)
    assert np.isclose(q[1], 1 - p_cal.min())


def test_too_small_a_calibration_set_returns_the_full_set():
    # n=5, alpha=0.1 -> k = ceil(6 * 0.9) = 6 > 5. There is no finite quantile that can promise
    # 90% coverage from five points, so every label must stay in the set.
    sets, q = conformal_sets(np.linspace(0.1, 0.9, 5), np.ones(5, dtype=int),
                             np.array([0.01, 0.5, 0.99]), 0.1, mondrian=False)
    assert np.isinf(q).all()
    assert sets.all(), "a calibration set too small for the target must abstain, not guess"


def test_an_empty_calibration_set_does_not_crash():
    sets, q = conformal_sets(np.array([]), np.array([], dtype=int),
                             np.array([0.5]), 0.1, mondrian=False)
    assert np.isinf(q).all() and sets.all()


def test_mondrian_quantiles_differ_by_class_when_the_classes_differ():
    # class 1 is predicted well, class 0 badly: the class-0 quantile must be the looser one
    p_cal = np.concatenate([np.full(50, 0.8), np.full(50, 0.9)])
    y_cal = np.concatenate([np.zeros(50, dtype=int), np.ones(50, dtype=int)])
    _, q = conformal_sets(p_cal, y_cal, np.array([0.5]), 0.1, mondrian=True)
    assert q[0] > q[1]
    _, qm = conformal_sets(p_cal, y_cal, np.array([0.5]), 0.1, mondrian=False)
    assert qm[0] == qm[1]


def test_a_confident_correct_classifier_gives_singleton_sets():
    p_cal, y_cal = np.array([0.99] * 40 + [0.01] * 40), np.array([1] * 40 + [0] * 40)
    sets, _ = conformal_sets(p_cal, y_cal, np.array([0.99, 0.01]), 0.1, mondrian=False)
    assert sets.sum(1).tolist() == [1, 1]
    assert sets[0].tolist() == [False, True] and sets[1].tolist() == [True, False]


# ---------------------------------------------------------------- calibration regimes

PAPERS = ["A"] * 50 + ["B"] * 30 + ["C"] * 12 + ["D"] * 8


def test_group_calibration_takes_whole_publications():
    train = np.arange(len(PAPERS))
    cal, fit, held = group_calibration(PAPERS, train, 0.25, np.random.default_rng(0))
    p = np.asarray(PAPERS)
    assert set(p[cal]) == set(held)
    assert not set(p[cal]) & set(p[fit]), "a publication must not straddle the split"
    assert len(cal) + len(fit) == len(train)


def test_group_calibration_reaches_the_requested_fraction():
    train = np.arange(len(PAPERS))
    cal, _, held = group_calibration(PAPERS, train, 0.25, np.random.default_rng(0))
    assert len(cal) >= 0.25 * len(train)
    # smallest-first: D(8) + C(12) = 20, then B(30) crosses the 25 target -> 50 molecules
    assert held[:2] == ["D", "C"]


def test_group_calibration_stops_as_soon_as_the_target_is_met():
    train = np.arange(len(PAPERS))
    cal, fit, _ = group_calibration(PAPERS, train, 0.01, np.random.default_rng(0))
    assert len(cal) == 8 and len(fit) == len(train) - 8


def test_iid_calibration_is_disjoint_and_the_right_size():
    train = np.arange(100, 200)
    cal, fit = iid_calibration(train, 0.25, np.random.default_rng(0))
    assert len(cal) == 25 and len(fit) == 75
    assert not set(cal) & set(fit)
    assert set(cal) | set(fit) == set(train)


def test_risk_coverage_falls_when_confidence_is_informative():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 500)
    y = (rng.uniform(size=500) < p).astype(int)          # calibrated, so confidence is meaningful
    rc = risk_coverage(y, p)
    assert rc[0.2] < rc[1.0], "keeping only confident molecules must cut the error rate"


def test_risk_coverage_keeps_at_least_one_molecule():
    assert risk_coverage(np.array([1]), np.array([0.9]))[0.2] == 0.0


def _fake_fold(seed, n=40):
    """One synthetic fold shaped like load_folds' output: train, valid, test index arrays."""
    y = np.array([0, 1] * (n // 2))
    idx = np.arange(n)
    tr, va, te = idx[: n - 12], idx[n - 12 : n - 6], idx[n - 6 :]
    return {"split": "paper_grouped", "seed": seed,
            "smiles": [f"C{seed}_{i}" for i in range(n)], "y": y, "idx": (tr, va, te),
            "paper": np.nan,
            "papers": [f"P{seed}_{i % 6}" for i in range(n)]}


def _run(tmp_path, monkeypatch, folds, resume=False):
    """Drive main() with fake folds and a trivial entrant, writing into tmp_path."""
    import scripts.run_lopo_conformal as mod
    monkeypatch.setattr(mod, "load_folds", lambda *a, **k: iter(folds))
    monkeypatch.setattr(mod, "morgan", lambda smiles: np.eye(len(smiles), dtype=np.uint8))
    monkeypatch.setattr(mod, "tanimoto_to_train", lambda a, b: np.zeros(len(a)))
    monkeypatch.setattr(mod, "ENTRANTS", {
        "logreg": lambda name, fold, seed: (np.full(len(fold["idx"][2]), 0.6), {})})
    pred = str(tmp_path / "pred.csv")
    out = str(tmp_path / "summary.csv")
    mod.main(["logreg"], [0.1], 0.25, "paper_grouped", "unused", pred, out, resume)
    return pred, out


def test_each_fold_is_checkpointed_as_it_finishes(tmp_path, monkeypatch):
    """The sidecar must exist with every completed fold, so a kill loses nothing."""
    import scripts.run_lopo_conformal as mod
    folds = [_fake_fold(s) for s in range(3)]
    kept = {}

    real_to_csv = pd.DataFrame.to_csv

    def spy(self, path_or_buf=None, *a, **k):
        if isinstance(path_or_buf, str) and path_or_buf.endswith(".partial"):
            kept.setdefault("writes", 0)
            kept["writes"] += 1
        return real_to_csv(self, path_or_buf, *a, **k)

    monkeypatch.setattr(pd.DataFrame, "to_csv", spy)
    _run(tmp_path, monkeypatch, folds)
    assert kept.get("writes") == 3                     # one checkpoint append per fold


def test_the_sidecar_is_removed_once_the_summary_is_written(tmp_path, monkeypatch):
    _, out = _run(tmp_path, monkeypatch, [_fake_fold(s) for s in range(2)])
    assert os.path.exists(out)
    assert not os.path.exists(f"{out}.partial")


def test_resume_skips_the_folds_already_in_the_sidecar(tmp_path, monkeypatch):
    """A resumed run must not recompute, and must not lose, the folds already done."""
    import scripts.run_lopo_conformal as mod
    folds = [_fake_fold(s) for s in range(3)]
    _, out = _run(tmp_path, monkeypatch, folds[:2])
    # rebuild the sidecar from the completed summary, as a killed run would have left it
    pd.read_csv(out).to_csv(f"{out}.partial", index=False)

    seen = []
    monkeypatch.setattr(mod, "load_folds", lambda *a, **k: iter(folds))
    pred2 = str(tmp_path / "pred2.csv")
    monkeypatch.setattr(mod, "morgan", lambda smiles: np.eye(len(smiles), dtype=np.uint8))
    monkeypatch.setattr(mod, "tanimoto_to_train", lambda a, b: np.zeros(len(a)))
    monkeypatch.setattr(mod, "ENTRANTS", {"logreg": lambda name, fold, seed: (
        seen.append(fold["seed"]) or np.full(len(fold["idx"][2]), 0.6), {})})
    mod.main(["logreg"], [0.1], 0.25, "paper_grouped", "unused", pred2, out, True)

    assert set(seen) == {2}                            # only the unfinished fold was computed
    final = pd.read_csv(out)
    assert set(final["fold"]) == {0, 1, 2}             # and all three are in the summary
