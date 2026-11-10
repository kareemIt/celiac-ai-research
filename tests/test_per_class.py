"""Per-class metrics, checked against a confusion matrix small enough to count by hand."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import best_balanced_threshold, per_class

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyse_per_class import _holm, shift_tests, summarise  # noqa: E402

# y        : 1 1 1 1 0 0 0 0
# p >= 0.5 : 1 1 1 0 1 0 0 0   -> tp 3, fn 1, fp 1, tn 3
Y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
P = np.array([0.9, 0.8, 0.6, 0.4, 0.7, 0.3, 0.2, 0.1])


def test_confusion_counts_and_ratios():
    m = per_class(Y, P, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 3)
    assert m["sensitivity"] == 0.75 and m["fnr"] == 0.25
    assert m["specificity"] == 0.75 and m["fpr"] == 0.25
    assert m["ppv"] == 0.75 and m["npv"] == 0.75
    assert m["accuracy"] == 0.75 and m["balanced_accuracy"] == 0.75
    assert m["f1"] == 0.75 and m["pred_pos_rate"] == 0.5


def test_undefined_ratios_are_nan_not_zero():
    """Threshold above every score: nothing predicted positive, so PPV has no denominator.
    0 would read as a measured failure."""
    m = per_class(Y, P, 1.1)
    assert (m["tp"], m["fp"]) == (0, 0)
    assert math.isnan(m["ppv"])
    assert m["sensitivity"] == 0.0 and m["specificity"] == 1.0


def test_single_class_fold_gives_nan_mcc_and_no_oracle():
    y = np.ones(4, dtype=int)
    assert math.isnan(per_class(y, np.array([0.1, 0.4, 0.6, 0.9]), 0.5)["mcc"])
    assert math.isnan(best_balanced_threshold(y, np.array([0.1, 0.4, 0.6, 0.9])))


def test_oracle_threshold_is_at_least_as_good_as_half():
    t = best_balanced_threshold(Y, P)
    assert per_class(Y, P, t)["balanced_accuracy"] >= per_class(Y, P, 0.5)["balanced_accuracy"]


def _rows(sens, spec, **extra):
    """One row per unit, with whatever keys summarise/shift_tests group on."""
    base = {"target": "tg2", "model": "rf", "split": "scaffold_pub", "threshold_kind": "t050",
            "achievable": True, "fnr": 0.0, "fpr": 0.0}
    base.update(extra)
    return pd.DataFrame([{**base, "sensitivity": a, "specificity": b}
                         for a, b in zip(sens, spec)])


def test_summarise_counts_undefined_units_instead_of_hiding_them():
    """A single-class publication gives one rate and not the other. Skipping it silently would
    make a mean look better supported than it is."""
    rows = _rows([0.8, 0.6, np.nan], [0.5, np.nan, np.nan])
    out = summarise(rows, ["sensitivity", "specificity"])
    assert out.units.iloc[0] == 3
    assert out.sensitivity_count.iloc[0] == 2 and out.sensitivity_undefined.iloc[0] == 1
    assert out.specificity_count.iloc[0] == 1 and out.specificity_undefined.iloc[0] == 2
    assert out.sensitivity_mean.iloc[0] == pytest.approx(0.7)


def test_holm_is_monotone_and_clipped():
    assert _holm(np.array([0.01, 0.04, 0.03])).tolist() == pytest.approx([0.03, 0.06, 0.06])
    assert _holm(np.array([0.6, 0.9])).tolist() == pytest.approx([1.0, 1.0])


def test_shift_tests_reports_both_units_and_the_sign_test():
    """The publication unit averages each publication over its folds first, so a publication
    appearing in many folds cannot count many times. The two units must both be reported."""
    ref = pd.concat([_rows([0.9], [0.5], paper=f"P{i}", seed=s)
                     for i in range(6) for s in range(2)], ignore_index=True)
    shifted = pd.concat([_rows([0.3], [0.6], paper=f"P{i}", seed=0, split="paper_lopo")
                         for i in range(6)], ignore_index=True)
    out = shift_tests(pd.concat([ref, shifted], ignore_index=True))
    assert set(out.unit) == {"publication", "publication_fold"}
    pub = out[(out.unit == "publication") & (out.rate == "sensitivity")]
    assert len(pub) == 1 and pub.n_ref.iloc[0] == 6  # 6 publications, not 12 publication-folds
    assert pub.delta.iloc[0] == pytest.approx(-0.6)
    both = out[(out.unit == "publication_fold") & (out.rate == "sensitivity")]
    assert both.n_ref.iloc[0] == 12
    spec = out[(out.unit == "publication") & (out.rate == "specificity")]
    assert spec.delta.iloc[0] == pytest.approx(0.1)  # specificity rises while sensitivity falls
