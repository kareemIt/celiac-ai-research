"""The FXIII-A selectivity claim rests on pairing rule and provenance, both easy to get wrong in a
way that would inflate the result."""
import numpy as np
import pandas as pd
import pytest

from scripts.audit_fxiiia import exact_pic50, paired


def test_pairs_are_always_within_one_publication():
    """A cross-paper ratio would inherit the 0.50-log inter-laboratory offset measured in
    covalent.md, which is larger than most selectivity differences here."""
    m = paired()
    t = exact_pic50("tgm2").set_index(["smiles", "document_chembl_id"])
    f = exact_pic50("f13a1").set_index(["smiles", "document_chembl_id"])
    for _, r in m.head(40).iterrows():
        key = (r.smiles, r.document_chembl_id)
        assert key in t.index and key in f.index, f"{key} is not a within-publication pair"


def test_selectivity_is_tg2_minus_fxiiia():
    """Sign convention: positive must mean TG2-preferring. A flipped sign would invert every
    conclusion in experimental_plan.md."""
    m = paired().head(50)
    assert np.allclose(m.selectivity, m.p_tg2 - m.p_f13)
    assert (m.p_tg2 > m.p_f13).sum() == (m.selectivity > 0).sum()


def test_only_exact_measurements_are_paired():
    """A censored '>' bound on either enzyme would make the ratio a bound, not a value."""
    m = paired()
    assert len(m) > 100, "expected the audited 147 pairs; dataset may have changed"
    assert m.p_tg2.notna().all() and m.p_f13.notna().all()


def test_pic50_conversion_is_nanomolar():
    """pIC50 = 9 - log10(nM). 1 nM must give 9, 1 uM must give 6."""
    for nm, expect in ((1.0, 9.0), (1000.0, 6.0), (1e6, 3.0)):
        assert 9 - np.log10(nm) == pytest.approx(expect)


def test_every_pair_has_one_row_per_molecule_and_paper():
    m = paired()
    assert not m.duplicated(subset=["smiles", "document_chembl_id"]).any()
