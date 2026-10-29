"""The ADME work has one failure mode worth testing for: a permeability number read out of
context. Values arrive in three unit systems and four assay systems, one of them recorded as
exactly zero, and the conclusion the project must never reach is that a low number by itself
demonstrates gut restriction. These tests pin the conversion, the provenance and the caveat.
"""
import io
import os

import pandas as pd
import pytest

from scripts.analyse_adme import PERM_SCALE, assay_system, direction, harmonise

DOC = "docs/adme_analysis.md"
RAW = "data/raw/adme/chembl_adme.csv"
LIT = "data/curated/literature_adme.csv"
SUM = "results/adme_summary.csv"


def test_nm_per_s_converts_to_1e6_cm_s_by_a_factor_of_ten():
    """1 nm/s = 1e-7 cm/s = 0.1 x 10^-6 cm/s. A factor-of-ten slip here would turn the most
    permeable molecule in the set into the least."""
    assert PERM_SCALE["nm/s"] == 0.1
    d = harmonise(pd.DataFrame({
        "standard_type": ["Papp"], "standard_value": [142.0], "standard_units": ["nm/s"],
        "assay_description": ["Apparent permeability of the compound across apical to "
                              "basolateral side in wild type MDCK cells at 10 uM"]}))
    assert d.value_10e6_cm_s.iloc[0] == pytest.approx(14.2)
    assert d.assay_system.iloc[0] == "MDCK"
    assert d.direction.iloc[0] == "A-B"


def test_assay_systems_are_never_pooled():
    """Caco-2, MDCK-MDR1 and PAMPA measure different things; each row must name its own."""
    assert assay_system("Apparent permeability ... in human Caco2 cells after 1 hr") == "Caco-2"
    assert assay_system("... in MDCK-MDR1 cells") == "MDCK"
    assert assay_system("Effective permeability of the compound by PAMPA assay") == "PAMPA"
    assert assay_system("Effective permeability across human jejunum") == "human jejunum"
    assert assay_system(None) == "unspecified"


def test_basolateral_to_apical_is_not_counted_as_absorption():
    assert direction("Apparent permeability from basolateral side to apical side in Caco2") == "B-A"
    assert direction("Effective permeability of the compound by PAMPA assay") == "n/a"


def test_raw_adme_keeps_both_the_standardized_and_the_reported_value():
    if not os.path.exists(RAW):
        pytest.skip("run scripts/fetch_adme.py")
    d = pd.read_csv(RAW)
    for col in ("standard_value", "standard_units", "value", "units", "assay_description",
                "document_chembl_id"):
        assert col in d.columns, col
    assert (d.assay_type == "A").all()


def test_zed1227_caco2_value_is_a_bound_not_a_measurement():
    """Papp '< 1 x 10^-6 cm/s' is an upper bound. Treating it as 1.0 would invent precision."""
    lit = pd.read_csv(LIT)
    row = lit[lit.property.eq("permeability_AB")].iloc[0]
    assert row.relation == "<"
    assert row.compound == "ZED1227"
    assert "Caco-2" in row.assay


def test_zed1227_unknowns_are_recorded_as_unknown():
    """Protein binding, metabolic stability and human PK are not published. The table must say so
    rather than leave them absent, so a later reader does not fill them in from a prediction."""
    lit = pd.read_csv(LIT)
    unknown = set(lit[lit.evidence.eq("unknown")].property)
    assert {"plasma_protein_binding", "metabolic_stability", "human_PK"} <= unknown


def test_no_compound_combines_selectivity_with_gut_restricted_permeability():
    """The headline negative. If this ever becomes false the recommendation section is wrong."""
    if not os.path.exists(SUM):
        pytest.skip("run scripts/analyse_adme.py")
    s = pd.read_csv(SUM)
    comb = s[s.section.eq("combined_criteria")].iloc[0]
    assert comb.with_permeability < comb.paired_molecules
    sel = s[s.section.eq("comparison")]
    gut_restricted = sel[sel.papp_10e6_cm_s < 1]
    assert gut_restricted.empty, "a selective compound now has sub-1e-6 permeability"


def test_report_refuses_to_read_gut_restriction_off_a_permeability_number():
    if not os.path.exists(DOC):
        pytest.skip("report not present")
    t = io.open(DOC, encoding="utf-8").read()
    assert "efflux" in t.lower()
    assert "does not establish" in t or "does not demonstrate" in t
    for tag in ("**[E]**", "**[I]**", "**[U]**"):
        assert tag in t, f"missing evidence tag {tag}"


def test_report_does_not_claim_a_model_was_trained():
    if not os.path.exists(DOC):
        pytest.skip("report not present")
    t = io.open(DOC, encoding="utf-8").read()
    assert "No model is fitted" in t or "no model was fitted" in t.lower()
