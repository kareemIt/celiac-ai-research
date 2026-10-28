"""Shared metrics so every model (baseline or Qwen-based) is scored identically."""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, matthews_corrcoef,
                             r2_score, roc_auc_score)


def ece(y, p, bins=15):
    """Expected calibration error, equal-width bins (same as LLM-as-Jev)."""
    y, p = np.asarray(y), np.asarray(p)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def classification(y, p):
    y, p = np.asarray(y), np.asarray(p)
    two = len(np.unique(y)) == 2  # AUROC/AUPRC undefined if the test fold has one class
    return {
        "auroc": roc_auc_score(y, p) if two else np.nan,
        "auprc": average_precision_score(y, p) if two else np.nan,
        "mcc": matthews_corrcoef(y, p >= 0.5),
        "ece": ece(y, p),
        "brier": brier_score_loss(y, p),
        # clipped: one confident mistake would otherwise send log loss to infinity
        "logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]),
    }


def regression(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    const = np.ptp(pred) == 0  # correlations undefined for a constant predictor
    return {
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "r2": r2_score(y, pred),
        "pearson": np.nan if const else pearsonr(y, pred)[0],
        "spearman": np.nan if const else spearmanr(y, pred)[0],
    }
