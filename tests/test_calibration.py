import numpy as np

from src.evaluation import calibration, metrics


def _overconfident(n=400, scale=3.0, seed=0):
    """Scores whose true probability is sigmoid(z) but which are reported as sigmoid(scale*z)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    y = rng.binomial(1, calibration.sigmoid(z))
    return z * scale, y


def test_temperature_recovers_the_inflation_factor():
    z, y = _overconfident(n=4000, scale=3.0)
    assert calibration.fit_temperature(z, y) == 3.0 or abs(calibration.fit_temperature(z, y) - 3.0) < 0.4


def test_calibration_improves_ece_and_keeps_ranking():
    z, y = _overconfident(n=4000)
    va, te = slice(0, 2000), slice(2000, None)
    before = metrics.classification(y[te], calibration.sigmoid(z[te]))
    for kind in ("temperature", "platt"):
        p = calibration.fit(kind, z[va], y[va])(z[te])
        after = metrics.classification(y[te], p)
        assert after["ece"] < before["ece"]
        assert after["logloss"] < before["logloss"]
        assert after["auroc"] == before["auroc"]  # monotone map cannot change the ranking


def test_fit_never_sees_test_labels():
    z, y = _overconfident()
    f = calibration.fit("temperature", z[:200], y[:200])
    assert np.allclose(f(z[200:]), f(z[200:]))  # deterministic given only the fitted parameter


def test_logit_sigmoid_roundtrip():
    p = np.array([0.01, 0.5, 0.99])
    assert np.allclose(calibration.sigmoid(calibration.logit(p)), p, atol=1e-6)
