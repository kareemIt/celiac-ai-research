"""Error-source analysis: the per-molecule tally, the concordance null and the balance correlation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyse_error_sources import (balance_vs_error_type, concordance,  # noqa: E402
                                   molecule_errors, publication_errors)


def preds(rows):
    """rows: (model, smiles, y, p). One seed, one arm."""
    return pd.DataFrame([{"model": m, "split": "scaffold_pub", "seed": 0,
                          "smiles": s, "y": y, "p": p} for m, s, y, p in rows])


def test_molecule_errors_counts_each_scoring_and_keeps_the_label():
    # "A" is an active every model calls inactive; "B" an inactive every model gets right
    d = preds([(m, "A", 1, 0.2) for m in ("rf", "xgb", "logreg")]
              + [(m, "B", 0, 0.1) for m in ("rf", "xgb", "logreg")])
    out = molecule_errors(d, "scaffold_pub").set_index("smiles")
    assert out.loc["A", "scorings"] == 3 and out.loc["A", "errors"] == 3
    assert out.loc["A", "error_rate"] == 1.0
    assert out.loc["A", "kind"] == "false_negative_when_wrong"
    assert out.loc["B", "errors"] == 0 and out.loc["B", "kind"] == "false_positive_when_wrong"
    assert out.loc["A", "models_mostly_wrong"] == 3
    assert out.loc["B", "models_mostly_wrong"] == 0


def test_error_at_exactly_the_cut_counts_as_a_positive_prediction():
    """p >= 0.5 is the rule used everywhere else in the project, so a 0.5 on an inactive is a
    false positive, not a free pass."""
    out = molecule_errors(preds([("rf", "A", 0, 0.5)]), "scaffold_pub")
    assert out.errors.iloc[0] == 1


def test_concordance_separates_shared_difficulty_from_independent_error():
    """Same marginal error rate, two very different structures: errors all on one molecule, or
    spread one per molecule. The shared case must sit far above the independent null."""
    models = [f"m{i}" for i in range(6)]
    shared = preds([(m, "hard", 1, 0.1) for m in models]
                   + [(m, f"easy{k}", 1, 0.9) for m in models for k in range(5)])
    spread = []
    for i, m in enumerate(models):
        for k in range(6):
            wrong = (k == i)  # each model errs on a different molecule
            spread.append((m, f"mol{k}", 1, 0.1 if wrong else 0.9))
    c_shared = concordance(molecule_errors(shared, "scaffold_pub"), "tg2", "scaffold_pub")
    c_spread = concordance(molecule_errors(preds(spread), "scaffold_pub"), "tg2", "scaffold_pub")
    assert c_shared["frac_molecules_almost_all_models_wrong"] > c_shared["null_expectation_all"]
    assert c_spread["frac_molecules_almost_all_models_wrong"] == 0.0


def test_concordance_needs_at_least_three_models():
    d = preds([("rf", "A", 1, 0.2), ("xgb", "A", 1, 0.2)])
    assert concordance(molecule_errors(d, "scaffold_pub"), "tg2", "scaffold_pub") is None


def test_publication_permutation_sees_a_planted_bad_publication():
    """One publication whose molecules are all wrong, the rest all right: the observed spread must
    exceed the null, which reassigns the same error values to the same publication sizes."""
    rows, paper = [], {}
    for k in range(10):
        bad = k < 3
        for m in ("rf", "xgb", "logreg"):
            rows.append((m, f"mol{k}", 1, 0.1 if bad else 0.9))
        paper[f"mol{k}"] = "BAD" if bad else "GOOD"
    mol = molecule_errors(preds(rows), "scaffold_pub")
    per_pub, test = publication_errors(mol, pd.Series(paper), "tg2", "scaffold_pub")
    assert set(per_pub.paper) == {"BAD", "GOOD"}
    assert per_pub.set_index("paper").loc["BAD", "fn_rate"] == 1.0
    assert test["observed_spread"] > test["null_spread_p95"]
    assert test["p_permutation"] < 0.05


def test_balance_correlation_has_the_signs_the_mechanism_predicts():
    """Inactive-heavy publications miss their actives, active-heavy ones over-call: so FN rate
    should fall with active fraction and FP rate should rise with it."""
    per_pub = pd.DataFrame({
        "paper": [f"P{i}" for i in range(8)],
        "molecules": [40] * 8,
        "actives": [5, 10, 15, 20, 25, 30, 33, 35],
        "inactives": [35, 30, 25, 20, 15, 10, 7, 5],
        "fn_rate": [0.9, 0.8, 0.6, 0.5, 0.4, 0.2, 0.1, 0.05],
        "fp_rate": [0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9],
        "error_rate": [0.5] * 8,
    })
    out = balance_vs_error_type(per_pub, "tg2", "scaffold_pub").set_index("rate")
    assert out.loc["fn_rate", "spearman_vs_active_frac"] == pytest.approx(-1.0)
    assert out.loc["fp_rate", "spearman_vs_active_frac"] == pytest.approx(1.0)
    assert (out.p < 0.05).all()


def test_balance_correlation_skips_classes_with_too_few_molecules():
    """A publication with 2 actives gives an unstable false-negative rate; those rows must not
    enter the correlation for that rate."""
    per_pub = pd.DataFrame({
        "paper": [f"P{i}" for i in range(6)],
        "molecules": [30] * 6, "actives": [2] * 6, "inactives": [28] * 6,
        "fn_rate": np.linspace(0.1, 0.9, 6), "fp_rate": np.linspace(0.1, 0.9, 6),
        "error_rate": [0.5] * 6,
    })
    out = balance_vs_error_type(per_pub, "tg2", "scaffold_pub")
    assert "fn_rate" not in set(out.rate)  # 2 actives each, below the floor of 5
    assert "fp_rate" not in set(out.rate)  # active fraction is constant, so no correlation exists
