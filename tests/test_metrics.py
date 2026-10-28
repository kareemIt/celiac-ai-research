import numpy as np
import pytest

from src.evaluation import metrics


def test_ece_zero_when_calibrated_and_large_when_not():
    y = np.array([0, 1] * 50)
    assert metrics.ece(y, np.full(100, 0.5)) == pytest.approx(0.0)
    assert metrics.ece(y, np.where(y == 1, 0.0, 1.0)) == pytest.approx(1.0)


def test_classification_perfect_and_single_class():
    m = metrics.classification([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["auroc"] == 1.0 and m["mcc"] == 1.0
    assert np.isnan(metrics.classification([1, 1], [0.6, 0.7])["auroc"])


def test_regression():
    m = metrics.regression([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert m["rmse"] == 0 and m["r2"] == 1 and m["spearman"] == pytest.approx(1)
    assert np.isnan(metrics.regression([1.0, 2.0], [5.0, 5.0])["pearson"])
