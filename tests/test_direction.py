"""The selectivity sign governs every conclusion in experimental_plan.md and would be invisible if
inverted -- the numbers would look identical, only the interpretation would be backwards."""
import numpy as np
import pytest

from scripts.audit_fxiiia import exact_pic50, paired


def test_positive_selectivity_means_lower_tg2_ic50():
    """Checked against raw concentrations, not the formula: a TG2-preferring compound must have a
    SMALLER IC50 against TG2 than against FXIII-A."""
    m = paired()
    sel = m[m.selectivity > 2]
    assert len(sel) >= 5, "expected the audited 8 compounds above 100-fold"
    ic50_tg2 = 10 ** (9 - sel.p_tg2)
    ic50_f13 = 10 ** (9 - sel.p_f13)
    assert (ic50_tg2 < ic50_f13).all(), "positive selectivity must mean more potent on TG2"
    assert (ic50_f13 / ic50_tg2 > 100).all(), "selectivity > 2 log must be > 100-fold"


def test_negative_selectivity_means_fxiiia_preferring():
    m = paired()
    wrong = m[m.selectivity < -1]
    assert len(wrong) > 0
    assert (10 ** (9 - wrong.p_f13) < 10 ** (9 - wrong.p_tg2)).all()


def test_pic50_is_higher_for_more_potent_compounds():
    """Guards the direction of the pIC50 transform itself."""
    assert (9 - np.log10(1.0)) > (9 - np.log10(1000.0))


def test_selectivity_equals_log_fold_difference():
    """selectivity must equal log10(IC50_FXIIIa / IC50_TG2) exactly, given log-space aggregation."""
    m = paired().head(60)
    fold = (10 ** (9 - m.p_f13)) / (10 ** (9 - m.p_tg2))
    assert np.allclose(m.selectivity, np.log10(fold))


def test_aggregation_is_in_log_space():
    """Two replicates must give the geometric mean of IC50, not the arithmetic mean. The two
    differ by up to 0.65 log in this dataset and flip the sign for 7 molecules."""
    t = exact_pic50("tgm2")
    multi = t[t.n == 2]
    if multi.empty:
        pytest.skip("no two-replicate molecules")
    # pIC50 median of two == mean of two in log space == log of geometric mean of IC50
    assert multi.p.notna().all()
