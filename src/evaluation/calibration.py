"""Post-hoc calibrators, each fit on the validation fold only and applied unchanged to test.

Every calibrator here is a monotone map of the score, so AUROC and AUPRC are unchanged by
construction (isotonic can move them slightly by creating ties). That is the point: calibration
is about whether the probabilities mean anything, not about ranking.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def _nll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_temperature(z_valid, y_valid):
    """One scalar T > 0 dividing the logits, chosen by validation NLL. The classic Guo et al. recipe.
    Optimized over log T so the search is unconstrained and scale-free."""
    f = lambda log_t: _nll(y_valid, sigmoid(z_valid / np.exp(log_t)))
    best = minimize_scalar(f, bounds=(-5, 5), method="bounded")
    return float(np.exp(best.x))


def fit(kind, z_valid, y_valid):
    """Return a function mapping test logits to calibrated probabilities. `kind` is one of
    none / temperature / platt / isotonic. Fit on validation scores only — never on test."""
    if kind == "none":
        return sigmoid
    if kind == "temperature":
        t = fit_temperature(z_valid, y_valid)
        return lambda z: sigmoid(z / t)
    if kind == "platt":
        # Two parameters (slope and intercept): can also shift the base rate, unlike temperature.
        lr = LogisticRegression(C=1e4).fit(np.asarray(z_valid)[:, None], y_valid)
        return lambda z: lr.predict_proba(np.asarray(z)[:, None])[:, 1]
    if kind == "isotonic":
        # Non-parametric, so it needs more validation points than the other two and can overfit a small fold.
        iso = IsotonicRegression(out_of_bounds="clip").fit(z_valid, y_valid)
        return lambda z: np.clip(iso.predict(z), EPS, 1 - EPS)
    raise ValueError(kind)


KINDS = ("none", "temperature", "platt", "isotonic")
