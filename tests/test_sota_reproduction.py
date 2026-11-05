"""The new harness must reproduce the committed baselines exactly, or nothing it reports is comparable.

results/tuned_baselines_tg2.csv was produced by scripts/run_tuned_baselines.py, which derives its
own scaffold splits. scripts/run_sota.py reads the exported folds instead and tunes over the same
grid. The two paths must agree per seed, not merely on average: a mean can match while individual
folds differ, which would mean the comparison and the baseline are scored on different molecules.

This caught one real discrepancy. Casting the Morgan fingerprint to float32 (it is uint8 in
src.data.chem) left AUROC and MCC identical but moved logistic regression's Brier by 3e-5, because
lbfgs stopped at a slightly different point. The cast was removed rather than the tolerance raised.
"""
import os

import numpy as np
import pandas as pd
import pytest

BASELINE = "results/tuned_baselines_tg2.csv"
SOTA = "results/sota_tg2.csv"
METRICS = ["auroc", "auprc", "mcc", "ece", "brier", "logloss"]
SEEDS = 10  # the span the committed baseline covers

# xgboost sums float32 leaf values, so its probabilities are reproducible only to ~1e-7
TOL = {"xgb": 1e-6}


def _load():
    try:
        old = pd.read_csv(BASELINE)
        new = pd.read_csv(SOTA)
    except FileNotFoundError as e:
        pytest.skip(f"{e.filename} not present; run scripts/run_sota.py --model logreg,rf,xgb")
    return old, new[(new.split == "scaffold") & (new.seed < SEEDS)]


@pytest.mark.parametrize("model", ["logreg", "rf", "xgb"])
def test_harness_reproduces_the_committed_baseline(model):
    old, new = _load()
    a = old[old.model == f"{model}_tuned"].set_index("seed").sort_index()
    b = new[new.model == model].set_index("seed").sort_index()
    if b.empty:
        pytest.skip(f"{model} has not been run on the scaffold split yet")
    assert a.index.tolist() == b.index.tolist(), "different seeds"
    for metric in METRICS:
        assert np.allclose(a[metric], b[metric], atol=TOL.get(model, 1e-12), rtol=0), (
            f"{model} {metric} drifted: max |Δ| = {(a[metric] - b[metric]).abs().max():.2e}")


def test_tuning_budget_is_recorded_and_capped():
    """Every entrant must declare how many configurations it was given, so an unequal budget is
    visible in the results rather than buried in a script."""
    _, new = _load()
    assert new["n_configs"].notna().all()
    assert new["n_configs"].max() <= 12


# Every target's prediction and split files. The leakage check below ran on TG2 alone until
# 2026-11-05, so the replication targets -- where all the new claims live -- were never covered.
TARGETS = {
    "tg2": ("results/sota_preds.csv", "results/splits_tg2.csv"),
    "cdk2": ("results/sota_preds_cdk2.csv", "results/splits_cdk2.csv"),
    "maob": ("results/sota_preds_maob.csv", "results/splits_maob.csv"),
}


def test_every_expected_result_file_is_present():
    """Fail loudly for a missing file instead of letting the guarded tests skip.

    The other tests here skip when their input is absent, which is right for a fresh checkout but
    means a DELETED or renamed result file turns a real check into a silent pass. This test is the
    one place that notices, so `324 passed` cannot hide a vanished arm.
    """
    missing = [f for pair in TARGETS.values() for f in pair if not os.path.exists(f)]
    missing += [f for f in (BASELINE, SOTA) if not os.path.exists(f)]
    assert not missing, f"expected result files absent: {missing}"


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_no_entrant_was_scored_on_a_fold_it_trained_on(target):
    """Cheap end-to-end leakage check: predictions must cover the exported test fold and nothing else."""
    preds_path, splits_path = TARGETS[target]
    try:
        preds = pd.read_csv(preds_path)
        splits = pd.read_csv(splits_path, low_memory=False)
    except FileNotFoundError as e:
        pytest.skip(f"{e.filename} not present")
    test = splits[splits.fold == "test"]
    checked = 0
    for (model, split, seed), g in preds.groupby(["model", "split", "seed"]):
        want = test[(test.split_kind == split) & (test.seed == seed)]
        assert sorted(g["smiles"]) == sorted(want["smiles"]), f"{model}/{split}/{seed} wrong fold"
        assert g["p"].between(0, 1).all(), f"{model}/{split}/{seed} has a probability out of range"
        checked += 1
    assert checked > 0, f"{preds_path} held no prediction groups -- a vacuous pass"


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_no_split_puts_a_molecule_in_both_training_and_test(target):
    """The split files themselves, independent of any model's predictions."""
    _, splits_path = TARGETS[target]
    try:
        splits = pd.read_csv(splits_path, low_memory=False)
    except FileNotFoundError as e:
        pytest.skip(f"{e.filename} not present")
    checked = 0
    for (split, seed), g in splits.groupby(["split_kind", "seed"]):
        train = set(g[g.fold.isin(["train", "valid"])]["smiles"])
        held = set(g[g.fold == "test"]["smiles"])
        assert not (train & held), f"{target}/{split}/{seed} shares {len(train & held)} molecules"
        checked += 1
    assert checked > 0, f"{splits_path} held no folds -- a vacuous pass"
