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


def _ratio(num, den):
    return float(num) / den if den else np.nan


def per_class(y, p, threshold):
    """Threshold-dependent metrics, kept separate from classification() so that function's
    committed outputs do not change. Undefined ratios (an empty row or column of the confusion
    matrix) are nan rather than 0, because 0 would read as a measured failure."""
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= threshold
    tp = int(np.sum((y == 1) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    fp = int(np.sum((y == 0) & pred))
    tn = int(np.sum((y == 0) & ~pred))
    sens, spec = _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    ppv = _ratio(tp, tp + fp)
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sensitivity": sens,
        "specificity": spec,
        "fnr": np.nan if np.isnan(sens) else 1 - sens,
        "fpr": np.nan if np.isnan(spec) else 1 - spec,
        "ppv": ppv,
        "npv": _ratio(tn, tn + fn),
        "accuracy": _ratio(tp + tn, tp + tn + fp + fn),
        "balanced_accuracy": np.nan if np.isnan(sens) or np.isnan(spec) else (sens + spec) / 2,
        "f1": np.nan if np.isnan(ppv) or np.isnan(sens) or ppv + sens == 0 else 2 * ppv * sens / (ppv + sens),
        "mcc": matthews_corrcoef(y, pred) if len(np.unique(y)) == 2 else np.nan,
        "pred_pos_rate": _ratio(tp + fp, tp + tn + fp + fn),
    }


def best_balanced_threshold(y, p):
    """The threshold maximising balanced accuracy on these same labels: an oracle upper bound,
    not an achievable operating point. Ties go to the lowest threshold."""
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2:
        return np.nan
    cands = np.unique(np.concatenate([p, [0.0, 1.0]]))
    scores = [per_class(y, p, t)["balanced_accuracy"] for t in cands]
    return float(cands[int(np.nanargmax(scores))])


def regression(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    const = np.ptp(pred) == 0  # correlations undefined for a constant predictor
    return {
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "r2": r2_score(y, pred),
        "pearson": np.nan if const else pearsonr(y, pred)[0],
        "spearman": np.nan if const else spearmanr(y, pred)[0],
    }
