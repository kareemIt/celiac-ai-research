"""Fold-balance analysis: the imbalance derivation, the pre-declared groups, and the slope fit."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyse_fold_balance import (MIN_SEEDS, USES_VALID_FOR_TRAINING, slopes,  # noqa: E402
                                  variance_vs_skill)


def frame(n_seeds, slope=0.0, model="rf", noise=0.0):
    """A synthetic arm: validation imbalance spread over [0, 0.4], AUROC on a known line."""
    imb = np.linspace(0.0, 0.4, n_seeds)
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "target": "tg2", "split": "scaffold", "model": model, "seed": range(n_seeds),
        "group": "uses_valid_for_training" if model in USES_VALID_FOR_TRAINING else "control",
        "valid_imbalance": imb, "test_imbalance": 0.1,
        "auroc": 0.8 + slope * imb + noise * rng.standard_normal(n_seeds),
    })


def test_slope_recovers_a_known_line():
    out = slopes(frame(20, slope=-0.25))
    assert len(out) == 1
    assert out.slope.iloc[0] == pytest.approx(-0.25)
    assert out.ci_lo.iloc[0] <= -0.25 <= out.ci_hi.iloc[0]
    assert out.r2.iloc[0] == pytest.approx(1.0)


def test_a_flat_relationship_gives_a_ci_spanning_zero():
    out = slopes(frame(20, slope=0.0, noise=0.05))
    assert out.ci_lo.iloc[0] < 0 < out.ci_hi.iloc[0]
    assert out.p.iloc[0] > 0.05


def test_short_series_are_dropped_rather_than_fitted():
    assert slopes(frame(MIN_SEEDS - 1)).empty
    assert len(slopes(frame(MIN_SEEDS))) == 1


def test_constant_imbalance_is_dropped():
    """A slope needs spread on the x axis; without it linregress returns nonsense."""
    d = frame(20)
    d["valid_imbalance"] = 0.2
    assert slopes(d).empty


def test_groups_match_the_pre_declared_list():
    assert set(USES_VALID_FOR_TRAINING) == {"chemprop", "chemprop_rdkit", "chemeleon", "chemberta_ft"}
    assert frame(10, model="chemprop").group.iloc[0] == "uses_valid_for_training"
    assert frame(10, model="rf").group.iloc[0] == "control"


def test_variance_vs_skill_sees_a_weaker_model_varying_more():
    parts = []
    for i, model in enumerate(["rf", "xgb", "logreg", "tabpfn", "chemprop"]):
        d = frame(20, model=model)
        d["auroc"] = 0.9 - 0.05 * i + (0.01 + 0.02 * i) * np.linspace(-1, 1, 20)
        parts.append(d)
    out = variance_vs_skill(pd.concat(parts, ignore_index=True))
    assert out.r_mean_vs_sd.iloc[0] < -0.9 and out.p.iloc[0] < 0.05
